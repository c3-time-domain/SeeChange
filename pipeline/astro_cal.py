import os
import time
import pathlib

import improc.scamp

from util.exceptions import CatalogNotFoundError, SubprocessFailure, BadMatchException
from util.logger import SCLogger
from util.util import parse_bool

from models.catalog_excerpt import CatalogExcerpt
from models.world_coordinates import WorldCoordinates

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.catalog_tools import fetch_gaia_dr3_excerpt


class ParsAstroCalibrator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_match_catalog = self.add_par(
            'cross_match_catalog',
            'gaia_dr3',
            str,
            'Which catalog should be used for cross matching for astrometry. ',
            critical=True
        )
        self.add_alias('catalog', 'cross_match_catalog')

        self.solution_method = self.add_par(
            'solution_method',
            'scamp',
            str,
            'Method/algorithm to use to match the catalog to the image source list. ',
            critical=True
        )
        self.add_alias( 'method', 'solution_method' )

        self.max_catalog_mag = self.add_par(
            'max_catalog_mag',
            [22.],
            list,
            ( 'Maximum (dimmest) magnitudes to try requesting for the matching catalog (list of float).  It will '
              'try these in order until it gets a catalog excerpt with at least catalog_min_stars, '
              'and until it gets a succesful WCS solution.  (Cached catalog excerpts will be considered a match '
              'if their max mag is within 0.1 mag of the one specified here.) ' ),
            critical=True
        )
        self.add_alias( 'max_mag', 'max_catalog_mag' )

        self.mag_range_catalog = self.add_par(
            'mag_range_catalog',
            4.,
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

        self.max_arcsec_residual = self.add_par(
            'max_arcsec_residual',
            0.15,
            float,
            ( 'Maximum residual in arcseconds for a WCS solution to be considered succesful.  The exact '
              'meaning of this depends on the method, but it should be something reasonable.'
             ),
            critical=True
        )
        self.add_alias( 'max_resid', 'max_arcsec_residual' )

        self.crossid_radii = self.add_par(
            'crossid_radii',
            [2.0],
            list,
            'List of initial radius in arcsec for cross-identifications to match; this is a scamp-specific parameter, '
            'passed to scamp via -CROSSID_RADIUS.  Pass the ones to try in order; the algorithm will try '
            'these (inside the mag_range_catalog loop) until it gets a successful WCS solution.',
            critical=True
        )

        self.min_frac_matched = self.add_par(
            'min_frac_matched',
            0.1,
            float,
            ( 'At least this fraction of the smaller of (image source list length, catalog excerpt lenght) '
              'must have been matched between the two for a WCS solution to be considered successful.' ),
            critical=True
        )
        self.add_alias( 'min_frac', 'min_frac_matched' )

        self.min_matched_stars = self.add_par(
            'min_matched_stars',
            10,
            int,
            ( 'At least this many stars must be matched between the source list and the catalog excerpt. '
              'Set this to 0 to not use this criterion.  (Both this and min_frac_matched are checked.) ' ),
            critical=True
        )
        self.add_alias( 'min_matches', 'min_matched_stars' )

        self.max_sources_to_use = self.add_par(
            'max_sources_to_use',
            2000,
            ( int, list ),
            ( 'If there are more than this many sources on the source list, crop it down this many, '
              'keeping the brightest sources.' ),
            critical=True
        )

        self.scamp_timeout = self.add_par(
            'scamp_timeout',
            300,
            int,
            'Timeout in seconds for scamp to run',
            critical=True
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'astro_cal'


class AstroCalibrator:
    def __init__(self, **kwargs):
        self.pars = ParsAstroCalibrator(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

    # ----------------------------------------------------------------------

    def _solve_wcs_scamp( self, image, sources, catexp, crossid_radius=2. ):
        """Solve for the WCS of image, updating image.header.

        If scamp does not succeed, will raise a SubprocessFailure
        exception (see utils/exceptions.py).

        Parameters
        ----------
          image: Image
            The image to solve the WCS for.  If the WCS solution
            succeeds, then the header field of the image will be
            updated with the keywords that define the new WCS.

          sources: SourceList
            Sources extracted from image

          catexp: CatalogExcerpt
            Astrometric calibration catalog excerpt that overlaps image.

          crossid_radius: float
            The radius in arcseconds for the initial scamp match (not the final solution).

        Returns
        -------
          astropy.wcs.WCS

        """

        if catexp.format != 'fitsldac':
            raise ValueError( f'_solve_wcs_scamp requires a fitsldac catalog excerpt, not {catexp.format}' )
        if sources.format != 'sextrfits':
            raise ValueError( f'_solve_wcs_scamp requires a sextrffits source list, not {sources.format}' )
        if catexp.origin != 'gaia_dr3':
            raise NotImplementedError( f"Don't know what magnitude key to choose for astrometric reference "
                                       f"{catexp.origin}; only gaia_dr3 is implemented." )

        if sources.filepath is None:
            sources.save()

        sourcefile = pathlib.Path( sources.get_fullpath() )
        catfile = pathlib.Path( catexp.get_fullpath() )

        wcs = improc.scamp.solve_wcs_scamp(
            sourcefile,
            catfile,
            crossid_radius=crossid_radius,
            max_sources_to_use=self.pars.max_sources_to_use,
            min_frac_matched=self.pars.min_frac_matched,
            min_matched=self.pars.min_matched_stars,
            max_arcsec_residual=self.pars.max_arcsec_residual,
            magkey='MAG_G', magerrkey='MAGERR_G',
            timeout=self.pars.scamp_timeout,
        )

        # Update image.header with the new wcs.  Process this
        # through astropy.wcs.WCS to make sure everything is copacetic.
        image.header.extend( wcs.to_header(), update=True )

        return wcs

    # ----------------------------------------------------------------------

    def _run_scamp( self, ds, prov, session=None ):
        """Do the work of run for the scamp matching method."""

        image = ds.get_image( session=session )

        # use the latest source list in the data store,
        # or load using the provenance given in the
        # data store's upstream_provs, or just use
        # the most recent provenance for "extraction"
        sources = ds.get_sources( session=session )
        if sources is None:
            raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')

        success = False
        exceptions = []
        for maxmag in self.pars.max_catalog_mag:
            try:
                catexp = fetch_gaia_dr3_excerpt(
                    image=image,
                    minstars=self.pars.min_catalog_stars,
                    maxmags=maxmag,
                    magrange=self.pars.mag_range_catalog,
                    session=session,
                )
            except CatalogNotFoundError as ex:
                SCLogger.info( f"Failed to get a catalog excerpt with enough stars with maxmag {maxmag}, "
                               f"trying the next one." )
                exceptions.append(ex)
                continue

            for radius in self.pars.crossid_radii:
                try:
                    wcs = self._solve_wcs_scamp( image, sources, catexp, crossid_radius=radius )
                    success = True
                    break
                except SubprocessFailure as ex:
                    SCLogger.info( f"Scamp failed for maxmag {maxmag} and crossid_rad {radius}, "
                                   f"trying the next crossid_rad" )
                    exceptions.append(ex)
                    continue
                except BadMatchException as ex:
                    SCLogger.info( f"Scamp didn't produce a successful match for maxmag {maxmag} "
                                   f"and crossid_rad {radius}; trying the next crossid_rad" )
                    exceptions.append(ex)
                    continue

            if success:
                break
            else:
                SCLogger.info( f"Failed to solve for WCS with maxmag {maxmag}, trying the next one." )

        if not success:
            raise RuntimeError( f"_run_scamp failed to find a match. Exceptions that were raised: {exceptions}" )

        # Save these in case something outside wants to
        # probe them (e.g. tests)
        self.maxmag = maxmag
        self.crossid_radius = radius
        self.catexp = catexp

        ds.wcs = WorldCoordinates( sources=sources, provenance=prov )
        ds.wcs.wcs = wcs
        if session is not None:
            ds.wcs = session.merge( ds.wcs )

    # ----------------------------------------------------------------------

    def run(self, *args, **kwargs):
        """Extract sources and use their positions to calculate the astrometric solution.

        Arguments are parsed by the DataStore.parse_args() method.
        Returns a DataStore object with the products of the processing.
        """
        self.has_recalculated = False
        try:  # first make sure we get back a datastore, even an empty one
            ds, session = DataStore.from_args(*args, **kwargs)
        except Exception as e:
            return DataStore.catch_failure_to_parse(e, *args)

        try:
            t_start = time.perf_counter()
            if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

            # try to find the world coordinates in memory or in the database:
            wcs = ds.get_wcs(prov, session=session)

            if wcs is None:  # must create a new WorldCoordinate object
                self.has_recalculated = True
                image = ds.get_image(session=session)
                if image.astro_cal_done:
                    SCLogger.warning(
                        f"Failed to find a wcs for image {pathlib.Path( image.filepath ).name}, "
                        f"but it has astro_cal_done=True"
                    )

                if self.pars.solution_method == 'scamp':
                    self._run_scamp( ds, prov, session=session )
                else:
                    raise ValueError( f'Unknown solution method {self.pars.solution_method}' )

                # update the upstream bitflag
                sources = ds.get_sources( session=session )
                if sources is None:
                    raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')
                if ds.wcs._upstream_bitflag is None:
                    ds.wcs._upstream_bitflag = 0
                ds.wcs._upstream_bitflag |= sources.bitflag

                # If an astro cal wasn't previously run on this image,
                # update the image's ra/dec and corners attributes based on this new wcs
                if not image.astro_cal_done:
                    image.set_corners_from_header_wcs(wcs=ds.wcs.wcs, setradec=True)
                    image.astro_cal_done = True

                ds.runtimes['astro_cal'] = time.perf_counter() - t_start
                if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                    import tracemalloc
                    ds.memory_usages['astro_cal'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:
            # make sure datastore is returned to be used in the next step
            return ds

