# an assortment of tools that relate to getting and manipulating source catalogs

import os
import time
import pathlib
import requests

import numpy as np
import pandas
import sqlalchemy as sa

import astropy.table
import healpy
from util import ldac

from models.base import SmartSession, FileOnDiskMixin
from models.catalog_excerpt import CatalogExcerpt, GaiaDR3DownloadLock

from util.exceptions import CatalogNotFoundError
from util.util import listify
from util.logger import SCLogger
from util.config import Config


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


class Bandpass:
    """A helper class to keep track of the bandpass of filters.

    This is currently only used for determining which filter in the instrument
    is most similar to a filter in a reference catalog.
    For example, is a V filter closer to Gaia G or to Gaia B_P or R_P?
    """
    def __init__(self, *args):
        """Create a new Bandpass object.

        Currently, initialize using lower/upper wavelength in nm.
        This assumes all filters have a simple, top-hat bandpass.
        This is good enough for comparing a discrete list of filters
        (e.g., the Gaia filters vs. the DECam filters) but not for
        more complicated things like finding the flux from a spectrum.

        Additional initializations could be possible in the future.

        """
        self.lower_wavelength = args[0]
        self.upper_wavelength = args[1]

    def __getitem__(self, idx):
        """A shortcut for getting the lower/upper wavelength.
        band[0] gives the lower wavelength, band[1] gives the upper wavelength.
        """
        if idx == 0:
            return self.lower_wavelength
        elif idx == 1:
            return self.upper_wavelength
        else:
            raise IndexError("Bandpass index must be 0 or 1. ")

    def get_overlap(self, other):
        """Find the overlap between two bandpasses.

        Will accept another Bandpass object, or a tuple of (lower, upper) wavelengths.
        By definition, all wavelengths are given in nm, but this works just as well
        if all the bandpasses are defined in other units (as long as they are consistent!).

        """

        return max(0, min(self.upper_wavelength, other[1]) - max(self.lower_wavelength, other[0]))

    def get_score(self, other):
        """Find the score (goodness) of the overlap between two bandpasses.

        The input can be another Bandpass object, or a tuple of (lower, upper) wavelengths.
        The score is calculated by the overlap between the two bandpasses,
        divided by the sqrt of the width of the other bandpass.
        This means that if two filters have similar overlap with this filter,
        the one with a much wider total bandpass gets a lower score.

        This can happen when one potentially matching filter is very wide
        and another potential filter covers only part of it (as in Gaia G,
        which is very broad) but the narrower filter has good overlap,
        so it is a better match.
        """
        width = other[1] - other[0]
        return self.get_overlap(other) / np.sqrt(width)

    def find_best_match(self, bandpass_dict):
        """Find the best match for this bandpass in a dictionary of bandpasses.

        If dict is empty or all bandpasses have a zero match to this bandpass,
        returns None.

        # TODO: should we have another way to match based on the "nearest central wavelength"
           or somthing like that? Seems like a corner case and not a very interesting one.
        """
        best_match = None
        best_score = 0
        for k, v in bandpass_dict.items():
            score = self.get_score(v)
            if score > best_score:
                best_match = k
                best_score = score

        return best_match


# Gaia related tools


def get_bandpasses_Gaia():
    """Get a dictionary of Bandpass objects for each filter in Gaia."""
    return dict(G=Bandpass(400, 850), BP=Bandpass(380, 650), RP=Bandpass(620, 900))


def _download_gaia_dr3_custom_server( minra, maxra, mindec, maxdec, minmag, maxmag ):
    cfg = Config.get()
    # Don't have to worry about minra/maxra being backwards in the case of
    #  ra spanning 0, because the server will detect that and correct for it.
    # However make sure that all ras are in the range 0..360
    minra = minra if minra >= 0. else minra + 360.
    minra = minra if minra < 360. else minra - 360.
    maxra = maxra if maxra >= 0. else maxra + 360.
    maxra = maxra if maxra < 360. else maxra - 360.

    url = f"{cfg.value( 'catalog_gaiadr3.server_url' )}/gaiarect/{minra}/{maxra}/{mindec}/{maxdec}"
    if maxmag is not None:
        url += f"/{maxmag}"
    if minmag is not None:
        url += f"/{minmag}"

    res = requests.post( url, timeout=cfg.value( 'catalog_gaiadr3.server_timeout_sec' ) )
    if res.status_code != 200:
        raise RuntimeError( f"Requests to custom gaia dr3 server returned status {res.status_code}" )

    return pandas.DataFrame( res.json() )


def download_gaia_dr3( minra, maxra, mindec, maxdec, padding=0.1, minmag=18., maxmag=22. ):
    """Download objects from gaia_dr3 as served by Noirlab Astro Data Lab

    Will Get a square on the sky given the limits, with the limits
    fractionally expanded at each edge by padding.

    NOTE: the code assumes that the RA range is not more than 180° (which
    is not realistic anyway, it should always be less than a few
    degrees).

    minra: float
       Minimum ra (degrees) of the image we're trying to match.
    maxra: float
       Maximum ra (degrees) of the image we're trying to match.
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
      A new CatalogExcerpt object
    str
      The absolute path where the catalog was saved
    str
      The absolute path to where the catalog should be saved if it is to be saved to the local file store

    Filename is the path where the file was stored; pass this to save() to
    properly save things to the database.

    """
    # TODO: should add another parameter to say which filter to use?
    #  Should the choosing of filters happen in this function or
    #  in the calling function?

    # Sanity check
    if maxra < minra:
        minra, maxra = maxra, minra
    if maxdec < mindec:
        mindec, maxdec = maxdec, mindec

    # Handle ra that spans 0°: make
    #   minra negative, fix that later
    #   (The call to _download_gaia_dr3_custom_server can handle negative RAs)
    if maxra - minra > 180.:
        maxra, minra = minra, maxra
        minra -= 360.

    ra = ( maxra + minra ) / 2.
    dra = maxra - minra
    ra = ra if ra >= 0. else ra + 360.    # Make sure ra is non-negative
    dec = ( maxdec + mindec ) / 2.
    ddec = maxdec - mindec
    ralow = minra - padding * dra
    rahigh = maxra + padding * dra
    declow = mindec - padding * ddec
    dechigh = maxdec + padding * ddec

    cfg = Config.get()

    df = None

    if cfg.value( 'catalog_gaiadr3.use_server' ):
        for i in range(5):
            try:
                df = _download_gaia_dr3_custom_server( ralow, rahigh, declow, dechigh, minmag, maxmag )
                break
            except Exception as ex:
                SCLogger.debug( f"Exception trying to download ra=({ralow}:{rahigh}), dec=({declow}:{dechigh}), "
                                f"mag=({minmag}:{maxmag}) from custom gaia server: {ex}" )
                if i < 4:
                    SCLogger.debug( "Sleeping 1s and retrying gaia query" )
                    time.sleep( 1 )
        else:
            SCLogger.error( f"Repeated failures trying to download  ra=({ralow}:{rahigh}), dec=({declow}:{dechigh}), "
                            f"mag=({minmag}:{maxmag}) from custom gaia server" )

    if ( ( ( df is None ) and cfg.value( 'catalog_gaiadr3.fallback_datalab' ) )
         or ( cfg.value( 'catalog_gaiadr3.use_datalab' ) ) ):
        # SCLogger.error( 'Skipping quering NOIRLab Astro Data Archive for Gaia DR3 stars; '
        #                 'need to handle RA spanning 0, or verify that noirlab does it right.' )
        # # Leave this code here for the unspecified future time when we deal with
        # # ra spanning 0.  For now, we'll hope that the custom gaia dr3 server
        # # is just working....
        # if False:
        if True:
            SCLogger.warning( "NOIRLab Astro Data Archive querying right now doesn't handle RA spanning 0" )
            SCLogger.info( 'Querying NOIRLab Astro Data Archive for Gaia DR3 stars' )

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
            SCLogger.debug( f'gaia_query is "{gaia_query}"' )

            for i in range(5):
                try:
                    qresult = queryClient.query( sql=gaia_query )
                    break
                except Exception as e:
                    SCLogger.info( f"Failed NOIRLab data lab Gaia download: {str(e)}" )
                    if i < 4:
                        SCLogger.info( "Sleeping 5s and retrying gaia query after failed attempt." )
                        time.sleep(5)
            else:
                errstr = f"NOIRLab data lab Gaia query giving up after {i} repeated failures."
                SCLogger.error( errstr )
                raise RuntimeError( errstr )

            df = dl.helpers.utils.convert( qresult, "pandas" )

    if df is None:
        raise RuntimeError( "Failed to download Gaia DR3 sources" )

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
    relpath = ( pathlib.Path( "gaia_dr3_excerpt" ) / str(hpix) /
                f"Gaia_DR3_{ra:.4f}_{dec:.4f}_{minmagstr}_{maxmagstr}.fits" )
    ofpath = pathlib.Path( FileOnDiskMixin.temp_path ) / relpath
    dbpath = pathlib.Path( FileOnDiskMixin.local_path ) / relpath
    ofpath.parent.mkdir( parents=True, exist_ok=True )

    fitstab = astropy.table.Table.from_pandas( df )
    SCLogger.debug( f"Writing {len(fitstab)} gaia stars to {ofpath}" )
    ldac.save_table_as_ldac( fitstab, ofpath, overwrite=True )

    # Fix ralow in the case of ra range spanning 0
    ralow = ralow if ralow >= 0 else ralow + 360

    catexp = CatalogExcerpt( format='fitsldac', origin='gaia_dr3', num_items=len(fitstab),
                             minmag=minmag, maxmag=maxmag, ra=ra, dec=dec,
                             ra_corner_00=ralow, ra_corner_01=ralow, ra_corner_10=rahigh, ra_corner_11=rahigh,
                             dec_corner_00=declow, dec_corner_10=declow, dec_corner_01=dechigh, dec_corner_11=dechigh,
                             minra=ralow, maxra=rahigh, mindec=declow, maxdec=dechigh )
    catexp.calculate_coordinates()

    return catexp, str( ofpath ), str( dbpath )


# ----------------------------------------------------------------------

def fetch_gaia_dr3_excerpt( image, minstars=50, maxmags=22, magrange=None, session=None, onlycached=False ):
    """Search catalog excerpts for a compatible gaia_dr3 excerpt; if not found, make one.

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
      minstars: int, default 50
        The minimum number of stars to require in the catalog excerpt.
      maxmags: sequence of float, default 22
        The maximum magnitudes to try pulling, using them in order
        until we get a catalog excerpt with at least minstars.
      magrange: float, default None
        If not None, then the minimum magnitude of stars fetched
        from the catalog would be this many magnitudes lower
        than the maximum magnitude.
        If None (default) there is no lower limit (the brightest
        stars will be included).
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

    ras = np.array( [ [ image.ra_corner_00, image.ra_corner_01], [ image.ra_corner_10, image.ra_corner_11 ] ] )
    decs = np.array( [ [ image.dec_corner_00, image.dec_corner_01], [ image.dec_corner_10, image.dec_corner_11 ] ] )

    # Detect ra spanning 0
    if ( ras.max() - ras.min() > 180. ):
        ras[ ras > 180. ] -= 360.

    minra = ras.min()
    maxra = ras.max()
    mindec = decs.min()
    maxdec = decs.max()
    dra = ( maxra - minra ) * np.cos( ( maxdec + mindec ) / 2. * np.pi / 180. )
    ddec = maxdec - mindec

    # Limits we'll use when searching cached CatalogExcerpts.
    # Put in a 5% padding, assuming that the initial corners
    # on the image are at least that good.
    ralow = minra - 0.05 * dra
    # Handle ra spanning 0
    if ralow < 0:
        ralow += 360.
    rahigh = maxra + 0.05 * dra
    declow = mindec - 0.05 * ddec
    dechigh = maxdec + 0.05 * ddec

    maxmags = listify( maxmags )

    catexp = None
    # Cycle through the given limiting magnitudes to see if we
    #  can get a catalog with enough stars
    for maxmag in maxmags:
        minmag = None if magrange is None else maxmag - magrange

        # See if there's a cached catalog we can use.

        # NOTE: there's a pathalogical case here where ralow is slightly
        #   above 0; it won't detect catalogs with ra_corner_00 below 0
        #   as overlapping!  For now, accept that, as that will be
        #   uncommon, and will just lead to occasionally downloading a
        #   new exceprt that we didn't strictly need to download.
        # See tests/pipeline/test_catalog_tools.py::test_gaia_dr3_excerpt_ra_span_zero

        # There's also kind of a nasty race condition here, where
        #   multiple processes ask for the same catalog at the same
        #   time.  The race condition formally comes in where two of
        #   them find it not existing, and then both of them try to
        #   download insert it; one of them will fail, becuase the
        #   database row will already be there from the other one.  We
        #   could solve this with some cleverness around the insert.
        #   However, this also' means that multiple processes might be
        #   querying NOIRLab at the same time for exactly the same
        #   catalog, which is just bad citizenship.  We don't want to
        #   just ham-handedly lock the catalog excerpts table, because
        #   the time to download the catalog might be non-trivial, and
        #   we could end up with a whole bunch of processes holding open
        #   database connections while they wait for just the one
        #   process to finish.  (Even worse, this stops them from
        #   parallelizing when they're asking for *different* catalogs.)
        #   So, to handle this, we created the gaiadr3_downloadlock
        #   table.

        with GaiaDR3DownloadLock.lock( ralow, rahigh, declow, dechigh, minmag, maxmag ):
            with SmartSession(session) as dbsess:
                q = ( dbsess.query( CatalogExcerpt )
                      .filter( CatalogExcerpt.origin == 'gaia_dr3' )
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
                      .filter( CatalogExcerpt.num_items >= minstars )
                     )
                if minmag is not None:
                    q = ( q.filter( CatalogExcerpt.minmag >= minmag-0.1 )
                          .filter( CatalogExcerpt.minmag <= minmag+0.1 ) )
                if q.count() > 0:
                    catexp = q.first()

                    if not os.path.isfile( catexp.get_fullpath() ):
                        SCLogger.warning( f"CatalogExcerpt {catexp.id} has no file at {catexp.filepath}, "
                                          f"deleting the database row!" )
                        dbsess.delete( catexp )
                        dbsess.commit()
                        catexp = None
                    else:
                        break

            if catexp is None and not onlycached:
                # No cached catalog excerpt, so query the NOIRLab server
                catexp, localfile, dbfile = download_gaia_dr3(
                    minra,
                    maxra,
                    mindec,
                    maxdec,
                    minmag=minmag,
                    maxmag=maxmag,
                )
                if catexp.num_items >= minstars:
                    catexp.filepath = dbfile
                    catexp.save( localfile )
                    with SmartSession( session ) as dbsess:
                        existing_catexp = dbsess.scalars(
                            sa.select(CatalogExcerpt).where(CatalogExcerpt.filepath == dbfile)
                        ).first()
                        if existing_catexp is None:
                            dbsess.add( catexp )  # add if it doesn't exist
                        else:
                            raise RuntimeError(f'CatalogExcerpt {dbfile} already exists in the database!')
                        dbsess.commit()
                        break
                else:
                    catexp = None
                    pathlib.Path( localfile ).unlink( missing_ok=True )

    if catexp is None:
        s = f"Failed to fetch Gaia DR3 stars at ( {(minra+maxra)/2.:.04f},{(mindec+maxdec)/2.:.04f} )"
        SCLogger.error( s )
        raise CatalogNotFoundError( s )

    return catexp

# ----------------------------------------------------------------------
