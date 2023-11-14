import os
import io
import pathlib
import math
import time
import random
import collections
import subprocess

import astropy.table
import astropy.io
from astropy.wcs import WCS

import healpy

from util import ldac
from util.exceptions import CatalogNotFoundError, SubprocessFailure, BadMatchException
from models.base import SmartSession, FileOnDiskMixin, _logger
from models.catalog_excerpt import CatalogExcerpt
from models.world_coordinates import WorldCoordinates
from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

# Here's a dysfunctionality in queryClient.  It wants ~/.datalab to
# exist.  So, it checks that if exists, and if it doesn't, it makes it.
# Only when you run under MPI, all processes do this at once, and you
# have a race condition where some processes will check, not find it,
# then fail to make it as another process has just made it.  Packages
# should *not* be doing stuff like this in imports.  Work around it
# by premaking the directory with exist_ok=True.
datalibdir = pathlib.Path( os.getenv("HOME") ) / ".datalab"
datalibdir.mkdir( exist_ok=True )
from dl import queryClient
import dl.helpers.utils


class ParsAstroCalibrator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_match_catalog = self.add_par(
            'cross_match_catalog',
            'GaiaDR3',
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

        self.crossid_radius = self.add_par(
            'crossid_radius',
            [2.0],
            list,
            ( 'Initial radius in arcsec for cross-identifications to match; this is a scamp-specific parameter, '
              'passed to scamp via -CROSSID_RADIUS.  Pass the ones to try in order; the algorithm will try '
              'these (inside the mag_range_catalog loop) until it gets a succesful WCS solution.'
             ),
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
            int,
            ( 'If there are more than this many sources on the source list, crop it down this many, '
              'keeping the brightest sources.' ),
            critical=True
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'astro_cal'


class AstroCalibrator:
    def __init__(self, **kwargs):
        self.pars = ParsAstroCalibrator(**kwargs)

    # ----------------------------------------------------------------------

    def download_GaiaDR3( self, minra, maxra, mindec, maxdec, padding=0.1, minmag=18., maxmag=22. ):
        """Download objects from GaiaDR3 as served by Noirlab Astro Data Lab

        Will Get a square on the sky given the limits, with the limits
        fractionally expanded at each edge by padding.

        minra: float
           Minimum ra of the image we're trying to match.
        maxra: float
           Maximum ra of the image we're trying to match.
        mindec: float
           Minimum dec of the image we're trying to match.
        maxdec: float
           Maximum dec of the image we're tryhing to match.
        padding: float, default 0.1
           A fraction of the width/height by which we expand the area searched on each side.
        minmag: float, default 18.
           Only get stars dimmer than this g magnitude.  (Useful for images that are
           deep enough that brighter stars will have saturated.)  Make it None to
           turn off the limit.
        maxmag: float, default 22.
           Only get stars brigther than this g magnitude.  (Useful for images that aren't
           as deep as Gaia, or for galacitc fields that have huge numbers of stars.)  Make
           it None to turn off the limit.

        Returns
        -------
        CatalogExcerpt
          A new CatalogExceprt object
        str
          The absolute path where the catalog was saved
        str
          The absolute path to where the catalog should be saved if it is to be saved to the local file store

        Filename is the path where the file was stored; pass this to save() to
        properly save things to the database.

        """

        # Sanity check
        if maxra < minra:
            minra, maxra = maxra, minra
        if maxdec < mindec:
            mindec, maxdec = maxdec, mindec

        ra = ( maxra + minra ) / 2.
        dec = ( maxdec + mindec ) / 2.
        dra = maxra - minra
        ddec = maxdec - mindec
        ralow = minra - padding * dra
        rahigh = maxra + padding * dra
        declow = mindec - padding * ddec
        dechigh = maxdec + padding * ddec

        _logger.info( f'Querying NOIRLab Astro Data Archive for Gaia DR3 stars' )

        gaia_query = (
            f"SELECT ra, dec, ra_error, dec_error, pm, pmra, pmdec, "
            f"       phot_g_mean_mag, phot_g_mean_flux_over_error, "
            f"       phot_bp_mean_mag, phot_bp_mean_flux_over_error, "
            f"       phot_rp_mean_mag, phot_rp_mean_flux_over_error, "
            f"       classprob_dsc_combmod_star "
            f"FROM gaia_dr3.gaia_source "
            f"WHERE ra>={ralow} AND ra<={rahigh} AND dec>={declow} AND dec<={dechigh} "
        )
        if minmag is not None:
            gaia_query += f"AND phot_g_mean_mag>={minmag} "
        if maxmag is not None:
            gaia_query += f"AND phot_g_mean_mag<={maxmag} "
        _logger.debug( f'gaia_query is "{gaia_query}"' )

        for i in range(5):
            try:
                qresult = queryClient.query( sql=gaia_query )
                break
            except Exception as e:
                _logger.info( f"Failed Gaia download: {str(e)}" )
                if i < 4:
                    _logger.info( "Sleeping 5s and retrying gaia query after failed attempt." )
                    time.sleep(5)
        else:
            errstr = f"Gaia query failed after {countdown} repeated failures."
            _logger.error( errstr )
            raise RuntimeError( errstr )

        df = dl.helpers.utils.convert( qresult, "pandas" )

        # Convert this into a FITS file format that scamp would recognize.
        # In particular, FITS header keywords have to be all upper case
        # The _WORLD variables are standard.
        # The others are not global standards, so we'll have to do things
        # consistent with what we'll do in scamp.
        # NOTE : the mag error columns are wrong by a factor of 1.09 = 2.5/ln(10)

        df.rename(
            columns={
                "ra": "X_WORLD",
                "dec": "Y_WORLD",
                "ra_error": "ERRA_WORLD",
                "dec_error": "ERRB_WORLD",
                "phot_g_mean_mag": "MAG_G",
                "phot_g_mean_flux_over_error": "MAGERR_G",
                "phot_bp_mean_mag": "MAG_BP",
                "phot_bp_mean_flux_over_error": "MAGERR_BP",
                "phot_rp_mean_mag": "MAG_RP",
                "phot_rp_mean_flux_over_error": "MAGERR_RP",
                "pm": "PM",
                "pmra": "PMRA",
                "pmdec": "PMDEC",
                "classprob_dsc_combmod_star": "STARPROB" },
            inplace=True
        )
        # Make the errors actual magnitude errors.  (1.0857 = 2.5/ln(10))
        for band in [ 'G', 'BP', 'RP' ]:
            df[ f'MAGERR_{band}' ] = 1.0857 / df[ f'MAGERR_{band}' ]
        # Put in two more fields that scamp really wants
        df[ 'OBSDATE' ] = 2015.5
        df[ 'FLAGS' ] = 0

        # To avoid saving too many files in one directory, use a healpix subdirectory
        hpix = healpy.ang2pix( 4, ra, dec, lonlat=True )
        minmagstr = 'None' if minmag is None else f'{minmag:.1f}'
        maxmagstr = 'None' if maxmag is None else f'{maxmag:.1f}'
        relpath = ( pathlib.Path( "GaiaDR3_excerpt" ) / str(hpix) /
                    f"Gaia_DR3_{ra:.4f}_{dec:.4f}_{minmagstr}_{maxmagstr}.fits" )
        ofpath = pathlib.Path( FileOnDiskMixin.temp_path ) / relpath
        dbpath = pathlib.Path( FileOnDiskMixin.local_path ) / relpath
        ofpath.parent.mkdir( parents=True, exist_ok=True )

        fitstab = astropy.table.Table.from_pandas( df )
        _logger.debug( f"Writing {len(fitstab)} gaia stars to {ofpath}" )
        ldac.save_table_as_ldac( fitstab, ofpath, overwrite=True )

        catexp = CatalogExcerpt( format='fitsldac', origin='GaiaDR3', num_items=len(fitstab),
                                 minmag=minmag, maxmag=maxmag, ra=ra, dec=dec,
                                 ra_corner_00=ralow, ra_corner_01=ralow, ra_corner_10=rahigh, ra_corner_11=rahigh,
                                 dec_corner_00=declow, dec_corner_10=declow,
                                 dec_corner_01=dechigh, dec_corner_11=dechigh )
        catexp.calculate_coordinates()

        return catexp, str( ofpath ), str( dbpath )

    # ----------------------------------------------------------------------

    def fetch_GaiaDR3_excerpt( self, image, maxmags=None, session=None, onlycached=False ):
        """Search catalog exertps for a compatible GaiaDR3 excerpt; if not found, make one.

        If multiple matching catalogs are found, will return the first
        one that the database happens to return.  (TODO: perhaps return
        the smallest one?  That would be useful if there are cameras
        with significantly different chip sizes.)

        NOTE : there is a race condition built in here.  If two
        processes ask for the same catalog at the same time (or close
        enough that the one that got there first hasn't made it and
        saved it yet), then the catalog will be grabbed twice.
        Hopefully this will be infrequent enough that the inefficiency
        of redundant catalogs won't be a big deal.  Perhaps we could put
        in some sort of locking, but that would be very touchy to do
        well without grinding everything to a halt, given that creating
        a new catalog excerpt isn't instant.

        Parameters
        ----------
          image : Image
            The Image we're searching for.  Uses the four corners to
            determine the range of RA/Dec needed for the excerpt.  Will
            make a footprint on the sky that starts with the RA/Dec
            aligned bounding square containing the image, which is then
            expanded by 5% on all sides.  Any catalog excerpt that fully
            includes that footprint is a potential match.

          maxmags: sequence of float, optional
            The maximum magnitudes to try pulling, using them in order
            until we get a catalog excerpt with at least
            self.pars.min_catalog_stars stars.  If None, will use
            self.pars.max_catalog_mag

          session : sqlalchemy.orm.session.Session, optional
            If not None, use this session for communication with the
            database; otherwise, will create and close a new
            SmartSession.

          onlycached : bool, default False
            If True, only search cached excerpts, don't make a new one
            if a matching one isn't found.

        Returns
        -------
          CatalogExcerpt

        """

        if maxmags is None:
            maxmags = self.pars.max_catalog_mag
        magrange = self.pars.mag_range_catalog
        numstars = self.pars.min_catalog_stars

        minra = min( image.ra_corner_00, image.ra_corner_01, image.ra_corner_10, image.ra_corner_11 )
        maxra = max( image.ra_corner_00, image.ra_corner_01, image.ra_corner_10, image.ra_corner_11 )
        mindec = min( image.dec_corner_00, image.dec_corner_01, image.dec_corner_10, image.dec_corner_11 )
        maxdec = max( image.dec_corner_00, image.dec_corner_01, image.dec_corner_10, image.dec_corner_11 )
        dra = ( maxra - minra ) * math.cos( ( maxdec + mindec ) / 2. * math.pi / 180. )
        ddec = maxdec - mindec
        # Limits we'll use when searching cached CatalogExcerpts.
        # Put in a 5% padding, assuming that the initial corners
        # on the image are at least that good.
        ralow = minra - 0.05 * dra
        rahigh = maxra + 0.05 * dra
        declow = mindec - 0.05 * ddec
        dechigh = maxdec + 0.05 * ddec

        if not isinstance( maxmags, collections.abc.Sequence ):
            maxmags = [ maxmags ]

        catexp = None
        with SmartSession(session) as session:
            # Cycle through the given limiting magnitudes to see if we
            #  can get a catalog with enough stars
            for maxmag in maxmags:
                # See if there's a cached catalog we can use.
                q = ( session.query( CatalogExcerpt )
                      .filter( CatalogExcerpt.origin == 'GaiaDR3' )
                      .filter( CatalogExcerpt.ra_corner_00 <= ralow )
                      .filter( CatalogExcerpt.ra_corner_01 <= ralow )
                      .filter( CatalogExcerpt.ra_corner_10 >= rahigh )
                      .filter( CatalogExcerpt.ra_corner_11 >= rahigh )
                      .filter( CatalogExcerpt.dec_corner_00 <= declow )
                      .filter( CatalogExcerpt.dec_corner_10 <= declow )
                      .filter( CatalogExcerpt.dec_corner_01 >= dechigh )
                      .filter( CatalogExcerpt.dec_corner_11 >= dechigh )
                      .filter( CatalogExcerpt.maxmag >= maxmag-0.1 )
                      .filter( CatalogExcerpt.maxmag <= maxmag+0.1 )
                      .filter( CatalogExcerpt.num_items >= numstars )
                     )
                if magrange is not None:
                    minmag = maxmag - magrange
                    q = ( q.filter( CatalogExcerpt.minmag >= minmag-0.1 )
                          .filter( CatalogExcerpt.minmag <= minmag+0.1 ) )
                if q.count() > 0:
                    catexp = q.first()
                    break
                elif not onlycached:
                    # No cached catalog excerpt, so query the NOIRLab server
                    catexp, localfile, dbfile = self.download_GaiaDR3( minra, maxra, mindec, maxdec,
                                                                       minmag=minmag, maxmag=maxmag )
                    if catexp.num_items >= numstars:
                        catexp.filepath = dbfile
                        catexp.save( localfile )
                        session.add( catexp )
                        session.commit()
                        break
                    else:
                        catexp = None
                        pathlib.Path( localfile ).unlink( missing_ok=True )

            if catexp is None:
                s = f"Failed to fetch Gaia DR3 stars at ( {(minra+maxra)/2.:.04f},{(mindec+maxdec)/2.:.04f} )"
                _logger.error( s )
                raise CatalogNotFoundError( s )

        return catexp

    # ----------------------------------------------------------------------

    def _solve_wcs_scamp( self, image, sources, catexp, crossid_rad=2. ):
        """Solve for the WCS of image, updating image.raw_header.

        If scamp does not succeed, will raise a SubprocessFailure
        exception (see utils/exceptions.py).

        Parameters
        ----------
          image: Image
            The image to solve the WCS for.  If the WCS solution
            succeeds, then the raw_header field of the image will be
            updated with the keywords that define the new WCS.

          sources: SourceList
            Sources extracted from image

          catexp: CatalogExcerpt
            Astrometric calibration catalog excerpt that overlaps image.

          crossid_rad: float
            The radius in arcseconds for the initial scamp match (not the final solution).

        Returns
        -------
          astropy.wcs.WCS

        """

        if catexp.format != 'fitsldac':
            raise ValueError( f'_solve_wcs_scamp requires a fitsldac catalog excerpt, not {catexp.format}' )
        if sources.format != 'sextrfits':
            raise ValueError( f'_solve_wcs_scamp requires a sextrffits source list, not {sources.format}' )

        sourcefile = pathlib.Path( sources.get_fullpath() )
        catfile = pathlib.Path( catexp.get_fullpath() )
        barf = ''.join( random.choices( 'abcdefghijlkmnopqrstuvwxyz', k=10 ) )
        xmlfile = pathlib.Path( FileOnDiskMixin.temp_path ) / f'scamp_{barf}.xml'
        # Scamp will have written a file stripping .fits from sourcefile and adding .head
        # I'm sad that there's not an option to scamp to explicitly specify this output filename.
        headfile = sourcefile.parent / f"{sourcefile.name[:-5]}.head"

        max_nmatch = 0
        if sources.num_sources > self.pars.max_sources_to_use:
            max_nmatch = self.pars.max_sources_to_use

        try:
            # Something I don't know : does scamp only use MATCH_NMAX
            # stars for the whole astrometric solution?  Back in the day
            # (we're talking ca. 1999 or 2000) when I wrote my own
            # transformation software, I had a parameter that allowed it
            # to use a subset of the list to get an initial match
            # (because that's an N² process), but then once I had that
            # initial match I used it to match all of the stars on the
            # list, and then used all of those stars on the list in the
            # solution.  I don't know if scamp works that way, or if it
            # just does the entire astrometric solution on MATCH_NMAX
            # stars.  The documentation is silent on this....
            command = [ 'scamp', sourcefile,
                        '-ASTREF_CATALOG', 'FILE',
                        '-ASTREFCAT_NAME', catfile,
                        '-MATCH', 'Y',
                        '-MATCH_NMAX', str( max_nmatch ),
                        '-SOLVE_PHOTOM', 'N',
                        '-CHECKPLOT_DEV', 'NULL',
                        '-CHECKPLOT_TYPE', 'NONE',
                        '-CHECKIMAGE_TYPE', 'NONE',
                        '-SOLVE_ASTROM', 'Y',
                        '-PROJECTION_TYPE', 'TPV',
                        '-WRITE_XML', 'Y',
                        '-XML_NAME', xmlfile,
                        '-CROSSID_RADIUS', str( crossid_rad )
                       ]

            # TODO : use a different gaia magnitude for different image
            # filters (see Issue #107)

            t0 = time.perf_counter()
            if catexp.origin == 'GaiaDR3':
                command.extend( [ '-ASTREFMAG_KEY', 'MAG_G', '-ASTREFMAGERR_KEY', 'MAGERR_G' ] )
            else:
                raise NotImplementedError( f"Don't know what magnitude key to choose for astrometric reference "
                                           f"{catexp.origin}; only GaiaDR3 is implemented." )

            res = subprocess.run( command, capture_output=True )
            t1 = time.perf_counter()
            _logger.debug( f"Scamp with {sources.num_sources} sources and {catexp.num_items} catalog stars "
                           f"(with match_nmax={max_nmatch}) took {t1-t0:.2f} seconds" )
            if res.returncode != 0:
                raise SubprocessFailure( res )

            scampstat = astropy.io.votable.parse( xmlfile ).get_table_by_index( 1 )
            nmatch = scampstat.array["AstromNDets_Reference"][0]
            sig0 = scampstat.array["AstromSigma_Reference"][0][0]
            sig1 = scampstat.array["AstromSigma_Reference"][0][1]
            infostr = ( f"Scamp on {pathlib.Path(catfile).name} to catalog with nominal magnitude "
                        f"range {catexp.minmag}-{catexp.maxmag} yielded {nmatch} matches out of "
                        f"{len(sources.data)} sources and {len(catexp.data)} catalog objects, "
                        f"with position sigmas of ({sig0:.2f}\", {sig1:.2f}\")" )
            if not ( ( nmatch > self.pars.min_frac_matched * min( len(sources.data), len(catexp.data ) ) )
                     and ( nmatch > self.pars.min_matched_stars )
                     and ( ( sig0 + sig1 ) / 2. <= self.pars.max_arcsec_residual )
                    ):
                infostr += ( f", which isn't good enough.\n"
                             f"Scamp command: {res.args}\n"
                             f"-------------\nScamp stderr:\n{res.stderr.decode('utf-8')}\n"
                             f"-------------\nScamp stdout:\n{res.stdout.decode('utf-8')}\n" )
                # A warning not an error in case something outside is iterating
                _logger.warning( infostr )
                raise BadMatchException( infostr )

            _logger.info( infostr )

            # Move the header information written in the ".head" file
            # scamp created to the image header, and to a WCS object
            # we're going to return.
            with open( headfile ) as ifp:
                hdrtext = ifp.read()
            # The FITS spec says ASCII, but Emmanuel put a non-ASCII latin-1
            # character in his comments... and astropy.io.fits.Header is
            # anal about only ASCII.  Sigh.
            hdrtext = hdrtext.replace( 'é', 'e' )
            strio = io.StringIO( hdrtext )
            hdr = astropy.io.fits.Header.fromfile( strio, sep='\n', padding=False, endcard=False )

            # Update image.raw_header with the new wcs.  Process this
            # through astropy.wcs.WCS to make sure everything is copacetic.
            wcs = WCS( hdr )
            image.raw_header.extend( wcs.to_header(), update=True )

            return wcs

        finally:
            xmlfile.unlink( missing_ok=True )
            headfile.unlink( missing_ok=True )

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
        for maxmag in self.pars.max_catalog_mag:
            try:
                catexp = self.fetch_GaiaDR3_excerpt( image, maxmags=(maxmag,), session=session )
            except CatalogNotFoundError as ex:
                _logger.info( f"Failed to get a catalog excerpt with enough stars with maxmag {maxmag}, "
                              f"trying the next one." )
                continue

            for crossid_rad in self.pars.crossid_radius:
                try:
                    wcs = self._solve_wcs_scamp( image, sources, catexp, crossid_rad=crossid_rad )
                    success = True
                    break
                except SubprocessFailure as ex:
                    _logger.info( f"Scamp failed for maxmag {maxmag} and crossid_rad {crossid_rad}, "
                                  f"trying the next crossid_rad" )
                    continue
                except BadMatchException as ex:
                    _logger.info( f"Scamp didn't produce a successful match for maxmag {maxmag} "
                                  f"and crossid_rad {crossid_rad}; trying the next crossid_rad" )
                    continue

            if success:
                break
            else:
                _logger.info( f"Failed to solve for WCS with maxmag {maxmag}, trying the next one." )

        if not success:
            raise RuntimeError( "_run_scamp failed to find a match." )

        # Save these in case something outside wants to
        # probe them (e.g. tests)
        self.maxmag = maxmag
        self.crossid_rad = crossid_rad
        self.catexp = catexp

        ds.wcs = WorldCoordinates( source_list=sources, provenance=prov )
        ds.wcs.wcs = wcs
        if session is not None:
            ds.wcs = ds.wcs.recursive_merge( session )

    # ----------------------------------------------------------------------

    def run(self, *args, **kwargs):
        """Extract sources and use their positions to calculate the astrometric solution.

        Arguments are parsed by the DataStore.parse_args() method.
        Returns a DataStore object with the products of the processing.
        """
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find the world coordinates in memory or in the database:
        wcs = ds.get_wcs(prov, session=session)

        if wcs is None:  # must create a new WorldCoordinate object
            image = ds.get_image()
            if image.astro_cal_done:
                _logger.warning( f"Failed to find a wcs for image {pathlib.Path( image.filepath ).name}, "
                                 f"but it has astro_cal_done=True" )

            if self.pars.solution_method == 'scamp':
                self._run_scamp( ds, prov, session=session )
            else:
                raise ValueError( f'Unknown solution method {self.pars.solution_method}' )

        # make sure this is returned to be used in the next step
        return ds
