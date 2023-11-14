import numpy as np

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.astro_cal import AstroCalibrator

from models.base import _logger
from models.zero_point import ZeroPoint

from util.exceptions import BadMatchException

# TODO: Make max_catalog_mag and mag_range_catalog defaults be supplied
# by the instrument, since there are going to be different sane defaults
# for different instruments.
# (E.g., LS4 can probably see stars down to 11th magnitude without
# saturating, whereas for DECam we don't want to go brighter than 15th
# or 16th magnitude.)

class ParsPhotCalibrator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_match_catalog = self.add_par(
            'cross_match_catalog',
            'GaiaDR3',
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

    def _solve_zp_GaiaDR3( self, image, sources, wcs, catexp, min_matches=10, match_radius=1. ):
        """Get the instrument zeropoint using a GaiaDR3 reference catalog.

        Assumes that a single zeropoint is good enough, and that there's
        need for color terms.  (Using color terms is complicated anyway
        if you're going to process a single image at a time!)

        Parmeters
        ---------
          image: Image
            The image we're finding the zeropoint for.

          sources: SourceList
            Image sources.  The sources must have been extracted from an
            image with a good WCS, as the ra and dec of the sources will
            be matched directly to the X_WORLD and Y_WORLD fields of the
            GaiaDR3 catalog excerpt.

          wcs: WoorldCoordinates
            A WorldCoordinates object that can be used to find ra and
            dec from x and y for the objects in sources.

          catexp: CatalogExcerpt
            Gaia DR3 catalog excerpt overlapping image

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

        # For GaiaDR3, catexp.data has fields:
        #   X_WORLD
        #   Y_WORLD
        #   MAG_G
        #   MAG_BP
        #   MAG_RP
        #   MAGERR*

        # Extract from the catalog the range of Gaia colors that we want
        # to use (motivated by looking at Gaia H-R diagrams).

        gaiaminbp_rp = 0.5
        gaiamaxbp_rp = 3.0
        col = catexp.data[ 'MAG_BP' ] - catexp.data[ 'MAG_RP' ]
        catdata = catexp.data[ ( col >= gaiaminbp_rp ) & ( col <= gaiamaxbp_rp ) &
                               ( catexp.data['STARPROB'] > 0.95 ) ]
        _logger.debug( f"{len(catdata)} Gaia stars with {gaiaminbp_rp}<BP-RP<{gaiamaxbp_rp}" )

        # Pull out the source information. Use the aperture that the
        # source list has designated as the "infinite" aperture, so that
        # there's no need to add an aperture correction into the
        # calculated zeropoint.

        sourceflux, sourcefluxerr = sources.apfluxadu( apnum=sources.inf_aper_num )
        skycoords = wcs.wcs.pixel_to_world( sources.x, sources.y )
        sourcera = skycoords.ra.deg
        sourcedec = skycoords.dec.deg

        # Only use stars that are "good" and that have flux > 3*fluxerr

        wgood = ( sources.good
                  & ( ~np.isnan( sourceflux) ) & ( ~np.isnan( sourcefluxerr ) )
                  & ( sourceflux > 3.*sourcefluxerr ) )
        sourcera = sourcera[wgood]
        sourcedec = sourcedec[wgood]
        sourceflux = sourceflux[wgood]
        sourcefluxerr = sourcefluxerr[wgood]
        _logger.debug( f"{len(sourcera)} of {sources.num_sources} image sources "
                       f"have flux in 'infinite' aperture >3σ" )

        # Match catalog excerpt RA/Dec to source RA/Dec

        catdex = []
        sourcedex = []
        # TODO : get rid of for loop and use array function
        for i in range( len(catdata) ):
            # TODO : the 360 to 0° issue that we have to deal with in a huge number of places
            # Astropy.Longitude could come to our rescue?
            dra = sourcera - catdata[i]['X_WORLD']
            ddec = sourcedec - catdata[i]['Y_WORLD']
            w = np.where( ( np.fabs( dra ) < match_radius/3600./np.cos( sourcedec ) )
                          & ( np.fabs( ddec ) < match_radius/3600. ) )[0]
            if len(w) == 0:
                continue
            else:
                # If there are more than one, just pick the first one in the list
                # that matches.
                # TODO : think about whether we can do better.
                catdex.append( i )
                sourcedex.append( w[0] )

        _logger.debug( f"Matched {len(catdex)} stars between Gaia and the image source list" )

        if len(catdex) < min_matches:
            raise BadMatchException( f"Only matched {lencatdex} stars between Gaia and the image source list, "
                                     f"which is less than the minimum of {min_matches}" )

        sourcera = sourcera[ sourcedex ]
        sourcedec = sourcedec[ sourcedex ]
        sourceflux = sourceflux[ sourcedex ]
        sourcefluxerr = sourcefluxerr[ sourcedex ]
        catdata = catdata[ catdex ]

        # Save this in the object for testing/evaluation purposes

        self.sourcera = sourcera
        self.sourcedec = sourcedec
        self.sourceflux = sourceflux
        self.sourcefluxerr = sourcefluxerr
        self.catdata = catdata

        # At this point we have an Astropy Table in catdata with the
        # catalog information, and we have the image source information
        # arrays index-matched to catdata (sourcera, sourcedec,
        # sourceflux, sourcefluxerr)

        # Pull the GaiaDR3 MAG_G to instrument filter transformation
        # from the Instrument object.

        trns = image.instrument_object.get_GaiaDR3_transformation( image.filter_short )

        # mag = -2.5*log10(flux) + zp
        # mag = GaiaGmag - sum( trns[i] * ( GaiaBP - GaiaRP )**i )
        # ...so...
        # zp = GaiaGmag - sum( trns[i] * ( GaiaBP - GaiaRP )**i ) + 2.5*log10( flux )

        fitorder = len(trns) - 1

        cols = catdata[ 'MAG_BP' ] - catdata[ 'MAG_RP' ]
        colerrs = np.sqrt( catdata[ 'MAGERR_BP' ]**2 + catdata[ 'MAGERR_RP' ]**2 )
        colton = cols[ :, np.newaxis ] ** np.arange( 0, fitorder+1, 1 )
        coltonminus1 = np.zeros( colton.shape )
        coltonminus1[ :, 1: ] = cols[ :, np.newaxis ] ** np.arange( 0, fitorder, 1 )
        coltonerr = np.zeros( colton.shape )
        coltonerr[ :, 1: ] = np.arange( 1, fitorder+1, 1 ) * coltonminus1[ :, 1: ] * colerrs.value[ :, np.newaxis ]

        zps = ( catdata['MAG_G'] - ( trns[ np.newaxis, : ] * colton ).sum( axis=1 )
                + 2.5*np.log10( sourceflux ) )
        zpvars = ( catdata[ 'MAGERR_G' ]**2 +
                   ( trns[np.newaxis, : ] * coltonerr ).sum( axis=1 )**2 +
                   ( 1.0857362 * sourcefluxerr / sourceflux )**2
                  )

        # Save these values so that tests outside can pull them and interrogate them
        self.individual_zps = zps
        self.individual_zpvars = zpvars
        self.individual_cols = cols
        self.individual_mags = catdata[ 'MAG_G' ]

        wgood = ( ~np.isnan( zps ) ) & ( ~np.isnan( zpvars ) )
        zps = zps[wgood]
        zpvars = zpvars[wgood]
        _logger.debug( f"{len(zps)} stars survived the not-NaN zeropoint check" )

        zpval = np.sum( zps / zpvars ) / np.sum( 1. / zpvars )
        dzpval = 1. / np.sum( 1. / (zpvars ) )
        # TODO : right now, this dzpval is way too low, looking at the scatter in the plots
        # produced in test_photo_cal.py:test_decam_photo_cal.  Make the estimate more
        # reasonable by implementing some sort of outlier rejection, and then expanding
        # errorbars to make the reduced chisq 1.  However, the systematic error on the
        # zeropoint is probably bigger than this statistical uncertainty anyway, so
        # even after we do that this estimate is likely to be too small.

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
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find the world coordinates in memory or in the database:
        zp = ds.get_zp(prov, session=session)

        if zp is None:  # must create a new ZeroPoint object

            if self.pars.cross_match_catalog != 'GaiaDR3':
                raise NotImplementedError( f"Currently only know how to calibrate to GaiaDR3, not "
                                           f"{self.pars.cross_match_catalog}" )

            image = ds.get_image(session=session)

            sources = ds.get_sources(session=session)
            if sources is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')

            wcs = ds.get_wcs( session=session )
            if wcs is None:
                raise ValueError( f'Cannot find a wcs for image {image.filepath}' )

            # Need an AstroCalibrator to use its ability to fetch catalogs
            self.astrometor = AstroCalibrator( cross_match_catalog=self.pars.cross_match_catalog,
                                               max_catalog_mag=self.pars.max_catalog_mag,
                                               mag_range_catalog=self.pars.mag_range_catalog,
                                               min_catalog_stars=self.pars.min_catalog_stars )
            catexp = self.astrometor.fetch_GaiaDR3_excerpt( image, session )

            # Save for testing/evaluation purposes
            self.catexp = catexp

            zpval, dzpval = self._solve_zp_GaiaDR3( image, sources, wcs, catexp )

            # Add the aperture corrections
            apercors = []
            for i, rad in enumerate( sources.aper_rads ):
                if i == sources.inf_aper_num:
                    apercors.append( 0. )
                else:
                    apercors.append( sources.calc_aper_cor( aper_num=i ) )

            # Make the ZeroPoint object
            ds.zp = ZeroPoint( source_list=ds.sources, provenance=prov, zp=zpval, dzp=dzpval,
                               aper_cor_radii=sources.aper_rads, aper_cors=apercors )

        # make sure the DataStore is returned to be used in the next step
        return ds
