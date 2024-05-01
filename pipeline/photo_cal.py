import numpy as np

import astropy.units as u

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
import pipeline.catalog_tools

from models.base import _logger
from models.zero_point import ZeroPoint

from util.exceptions import BadMatchException

# TODO: Make max_catalog_mag and mag_range_catalog defaults be supplied
#  by the instrument, since there are going to be different sane defaults
#  for different instruments.
#  (E.g., LS4 can probably see stars down to 11th magnitude without
#  saturating, whereas for DECam we don't want to go brighter than 15th
#  or 16th magnitude.)


class ParsPhotCalibrator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_match_catalog = self.add_par(
            'cross_match_catalog',
            'gaia_dr3',
            str,
            'Which catalog should be used for cross matching for photometric calibration. '
        )
        self.add_alias('catalog', 'cross_match_catalog')

        self.max_catalog_mag = self.add_par(
            'max_catalog_mag',
            [22.],
            list,
            ( 'Maximum (dimmest) magnitudes to try requesting for the matching catalog (list of float).  It will '
              'try these in order until it gets a catalog excerpt with at least catalog_min_stars. (Cached '
              'catalog excerpts will be considered a match if their max mag is within 0.1 mag of the one '
              'specified here.) ' ),
            critical=True
        )
        self.add_alias( 'max_mag', 'max_catalog_mag' )

        self.mag_range_catalog = self.add_par(
            'mag_range_catalog',
            6.,
            ( float, None ),
            ( 'Range between maximum and minimum magnitudes to request for the catalog. '
              'Make this None to have no lower (bright) limit.' ),
            critical=True
        )
        self.add_alias( 'mag_range', 'mag_range_catalog' )

        self.min_catalog_stars = self.add_par(
            'min_catalog_stars',
            50,
            int,
            'Minimum number of stars the catalog must have',
            critical=True
        )
        self.add_alias( 'min_stars', 'min_catalog_stars' )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'photo_cal'


class PhotCalibrator:
    def __init__(self, **kwargs):
        self.pars = ParsPhotCalibrator(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

    def _solve_zp( self, image, sources, wcs, catexp, min_matches=10, match_radius=1. ):
        """Get the instrument zeropoint using a catalog excerpt.

        Assumes that a single zeropoint is good enough, and that there's
        need for color terms.  (Using color terms is complicated anyway
        if you're going to process a single image at a time!)

        Parameters
        ---------
          image: Image
            The image we're finding the zeropoint for.

          sources: SourceList
            Image sources. The sources must have been extracted from an
            image with a good WCS, as the ra and dec of the sources will
            be matched directly to the X_WORLD and Y_WORLD fields of the
            gaia_dr3 catalog excerpt.

          wcs: WorldCoordinates
            A WorldCoordinates object that can be used to find ra and
            dec from x and y for the objects in sources.

          catexp: CatalogExcerpt
            A catalog excerpt overlapping the image.

          min_matches: int, default 20
            Minimum number of matches between the two catalogs to return
            a zeropoint.

          match_radius : float, default 1.
            Maximum arcseconds away an image source and a catalog object
            may be from each other in both RA and Dec to be considered a
            match.

        Returns
        -------
          zp, zperr

          zp: float
            The zeropoint, defined so that -2.5*log(ADU) + zp = Mag

          zperr: float
           The uncertainty on the zeropoint

        """
        catname = self.pars.cross_match_catalog
        prune_func = getattr(image.instrument_object, f'{catname}_prune_star_cat')
        coord_func = getattr(image.instrument_object, f'{catname}_get_skycoords')
        trans_func = getattr(image.instrument_object, f'{catname}_to_instrument_mag')

        # Extract from the catalog the range of Gaia colors that we want
        # to use (motivated by looking at Gaia H-R diagrams).
        catdata = prune_func(catexp.data)
        _logger.debug( f"{len(catdata)} catalog stars passed the pruning process." )

        # transform the coordinates columns into astropy.coordinates.SkyCoord objects
        catcoords = coord_func(catdata, image.mjd)

        # Pull out the source information. Use the aperture that the
        # source list has designated as the "infinite" aperture, so that
        # there's no need to add an aperture correction into the
        # calculated zeropoint.
        sourceflux, sourcefluxerr = sources.apfluxadu( apnum=sources.inf_aper_num )
        skycoords = wcs.wcs.pixel_to_world( sources.x, sources.y )

        # Only use stars that are "good" and that have flux > 3*fluxerr
        wgood = ( sources.good
                  & ( ~np.isnan( sourceflux) ) & ( ~np.isnan( sourcefluxerr ) )
                  & ( sourceflux > 3.*sourcefluxerr ) )
        skycoords = skycoords[wgood]
        sourceflux = sourceflux[wgood]
        sourcefluxerr = sourcefluxerr[wgood]
        _logger.debug( f"{len(skycoords)} of {sources.num_sources} image sources "
                       f"have flux in 'infinite' aperture >3σ" )

        # Match catalog excerpt RA/Dec to source RA/Dec
        # ref https://docs.astropy.org/en/stable/coordinates/matchsep.html#matching-catalogs
        max_sep = match_radius * u.arcsec
        idx, d2d, d3d = skycoords.match_to_catalog_sky(catcoords)
        sep_constraint = d2d < max_sep
        skycoords = skycoords[sep_constraint]
        sourceflux = sourceflux[sep_constraint]
        sourcefluxerr = sourcefluxerr[sep_constraint]

        # make sure to use idx to index into the catalog
        # which will now match the order of the sources
        # catcoords = catcoords[idx[sep_constraint]]  # do we need this?
        catdata = catdata[idx[sep_constraint]]

        _logger.debug( f"Matched {len(skycoords)} stars between catalog and the image source list" )

        if len(skycoords) < min_matches:
            raise BadMatchException( f"Only matched {len(skycoords)} stars between catalog and the image source list, "
                                     f"which is less than the minimum of {min_matches}" )

        # Save this in the object for testing/evaluation purposes
        self.sourcera = skycoords.ra
        self.sourcedec = skycoords.dec
        self.sourceflux = sourceflux
        self.sourcefluxerr = sourcefluxerr
        self.catdata = catdata

        # At this point we have an Astropy Table in catdata with the
        # catalog information, index matched to source information
        # arrays (skycoords, sourceflux, sourcefluxerr)

        # Pull the catalog to instrument filter transformation from the Instrument object.
        transformed_mag, transformed_magerr = trans_func(image.filter, catdata)

        # instrumental_mag = -2.5*log10(flux)
        # transformed_mag(catalog_mag) = instrumental_mag + zp
        # ...so...
        # zp = transformed_mag(catalog_mag) + 2.5*log10( flux )
        zps = transformed_mag + 2.5*np.log10( sourceflux )
        zpvars = transformed_magerr**2 + ( 1.0857362 * sourcefluxerr / sourceflux )**2

        # Save these values so that tests outside can pull them and interrogate them
        self.individual_mags = transformed_mag
        self.individual_zps = zps
        self.individual_zpvars = zpvars

        wgood = ( ~np.isnan( zps ) ) & ( ~np.isnan( zpvars ) )
        zps = zps[wgood]
        zpvars = zpvars[wgood]
        _logger.debug( f"{len(zps)} stars survived the not-NaN zeropoint check" )

        zpval = np.sum( zps / zpvars ) / np.sum( 1. / zpvars )
        dzpval = 1. / np.sum( 1. / (zpvars ) )
        # TODO : right now, this dzpval is way too low, looking at the scatter in the plots
        #  produced in test_photo_cal.py:test_decam_photo_cal.  Make the estimate more
        #  reasonable by implementing some sort of outlier rejection, and then expanding
        #  errorbars to make the reduced chisq 1.  However, the systematic error on the
        #  zeropoint is probably bigger than this statistical uncertainty anyway, so
        #  even after we do that this estimate is likely to be too small.

        return zpval, dzpval

    def run(self, *args, **kwargs):
        """Perform the photometric calibration.

        Gets the image sources, finds a catalog Excerpt (using code in
        astro_cal.py), matches the two lists on RA/Dec,

        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        In addition to potentially loading previous products, this one
        will add a ZeroPoint object to the .zp field of the DataStore.

        """
        self.has_recalculated = False
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find the world coordinates in memory or in the database:
        zp = ds.get_zp(prov, session=session)

        if zp is None:  # must create a new ZeroPoint object
            self.has_recalculated = True
            if self.pars.cross_match_catalog != 'gaia_dr3':
                raise NotImplementedError( f"Currently only know how to calibrate to gaia_dr3, not "
                                           f"{self.pars.cross_match_catalog}" )

            image = ds.get_image(session=session)

            sources = ds.get_sources(session=session)
            if sources is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')

            wcs = ds.get_wcs( session=session )
            if wcs is None:
                raise ValueError( f'Cannot find a wcs for image {image.filepath}' )

            catname = self.pars.cross_match_catalog
            fetch_func = getattr(pipeline.catalog_tools, f'fetch_{catname}_excerpt')
            catexp = fetch_func(
                image=image,
                minstars=self.pars.min_catalog_stars,
                maxmags=self.pars.max_catalog_mag,
                magrange=self.pars.mag_range_catalog,
                session=session,
            )

            # Save for testing/evaluation purposes
            self.catexp = catexp

            zpval, dzpval = self._solve_zp( image, sources, wcs, catexp )

            # Add the aperture corrections
            apercors = []
            for i, rad in enumerate( sources.aper_rads ):
                if i == sources.inf_aper_num:
                    apercors.append( 0. )
                else:
                    apercors.append( sources.calc_aper_cor( aper_num=i ) )

            # Make the ZeroPoint object
            ds.zp = ZeroPoint( sources=ds.sources, provenance=prov, zp=zpval, dzp=dzpval,
                               aper_cor_radii=sources.aper_rads, aper_cors=apercors )
            
            if ds.zp._upstream_bitflag is None:
                ds.zp._upstream_bitflag = 0
            ds.zp._upstream_bitflag |= sources.bitflag
            ds.zp._upstream_bitflag |= wcs.bitflag

            ds.image.zero_point_estimate = ds.zp.zp  # TODO: should we only write if the property is None?
            # TODO: we should also add a limiting magnitude calculation here.

        # make sure the DataStore is returned to be used in the next step
        return ds
