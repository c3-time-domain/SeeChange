import pathlib
import random
import time
import io
import subprocess

import astropy.io
from astropy.wcs import WCS

from models.base import FileOnDiskMixin
from util import ldac
from util.logger import SCLogger
from util.exceptions import SubprocessFailure, BadMatchException


def solve_wcs_scamp( sources, catalog, crossid_radius=2.,
                      max_sources_to_use=2000, min_frac_matched=0.1,
                      min_matched=10, max_arcsec_residual=0.15,
                      magkey='MAG', magerrkey='MAGERR', timeout=60 ):
    """Solve for the WCS of image with sourcelist sources, based on catalog.

    If scamp does not succeed, will raise a SubprocessFailure
    exception (see utils/exceptions.py).

    Parameters
    ----------
      sources: Path or str
        A file with a FITS LDAC table of sources from the image whose
        WCS we want.

      catalog: Path or str
        A file with a FITS LDAC table of the catalog.  Must include
        'X_WORLD', 'Y_WORLD', 'MAG', and 'MAGERR' (where the latter two
        may be overridden with the magkey and magerrkey keywords).

      crossid_radius: float, default 2.0
        The radius in arcseconds for the initial scamp match (not the final solution).

      max_sources_to_use: int or list of int
        If specified, if the number of objects in sources is larger than
        this number, tell scamp to only use this many sources for the
        initial match.  (This makes the initial match go faster.)  If a
        list, it will iterate on this.

      min_frac_matched: float, default 0.1
        scamp must be able to match at least this fraction of
        min(number of sources, number of catalog objects) for the
        match to be considered good.

      min_matched: int, default 10
        scamp must be able to match at least this many objects
        for the match to be considered good.

      max_arcsec_residual: float, default 0.15
        maximum residual in arcseconds, along both RA and Dec
        (i.e. not a radial residual), for the WCS solution to be
        considered successful.

      magkey: str, default MAG
        The keyword to use for magnitudes in the catalog file.

      magerrkey: str, default MAGERR
        The keyword to use for the magnitude errors in the catalog file.

    Returns
    -------
      astropy.wcs.WCS
        The WCS for image

    """

    sourcefile = pathlib.Path( sources )
    catfile = pathlib.Path( catalog )
    barf = ''.join( random.choices( 'abcdefghijlkmnopqrstuvwxyz', k=10 ) )
    xmlfile = pathlib.Path( FileOnDiskMixin.temp_path ) / f'scamp_{barf}.xml'
    # Scamp will have written a file stripping .fits from sourcefile and adding .head
    # I'm sad that there's not an option to scamp to explicitly specify this output filename.
    headfile = sourcefile.parent / f"{sourcefile.name[:-5]}.head"

    sourceshdr, sources = ldac.get_table_from_ldac( sourcefile, imghdr_as_header=True )
    cathdr, cat = ldac.get_table_from_ldac( catfile, imghdr_as_header=True )

    try:
        max_nmatches = [ 0 ]
        if ( max_sources_to_use is not None ):
            if isinstance( max_sources_to_use, int ):
                max_nmatches = [ max_sources_to_use ]
            else:
                max_nmatches = max_sources_to_use

        success = False
        for max_nmatch in max_nmatches:
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
            SCLogger.debug( f"Trying scamp with MATCH_NMAX {max_nmatch}" )
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
                        '-CROSSID_RADIUS', str( crossid_radius ),
                        '-ASTREFMAG_KEY', magkey,
                        '-ASTREFMAGERR_KEY', magerrkey
                       ]

            t0 = time.perf_counter()
            res = subprocess.run( command, capture_output=True, timeout=timeout )
            t1 = time.perf_counter()
            SCLogger.debug( f"Scamp with {len(sources)} sources and {len(cat)} catalog stars "
                           f"(with match_nmax={max_nmatch}) took {t1-t0:.2f} seconds" )

            if res.returncode != 0:
                raise SubprocessFailure( res )

            scampstat = astropy.io.votable.parse( xmlfile ).get_table_by_index( 1 )
            nmatch = scampstat.array["AstromNDets_Reference"][0]
            sig0 = scampstat.array["AstromSigma_Reference"][0][0]
            sig1 = scampstat.array["AstromSigma_Reference"][0][1]
            infostr = ( f"Scamp on sources {pathlib.Path(sourcefile).name} with catalog {pathlib.Path(catfile).name} "
                        f"yielded {nmatch} matches out of {len(sources)} sources and {len(cat)} catalog objects "
                        f"(max_nmatch={max_nmatch}), with position sigmas of ({sig0:.2f}\", {sig1:.2f}\")" )
            if not ( ( nmatch > min_frac_matched * min( len(sources), len(cat), max_nmatch ) )
                     and ( nmatch > min_matched )
                     and ( ( sig0 + sig1 ) / 2. <= max_arcsec_residual )
                    ):
                infostr += ( f", which isn't good enough.\n" )
                # A warning not an error in case something outside is iterating
                SCLogger.warning( infostr )
            else:
                success = True
                break

        if not success:
            SCLogger.warning( f"Last scamp command: {res.args}\n"
                             f"-------------\nScamp stderr:\n{res.stderr.decode('utf-8')}\n"
                             f"-------------\nScamp stdout:\n{res.stdout.decode('utf-8')}\n" )
            raise BadMatchException( infostr )

        SCLogger.debug( infostr )

        # Create a WCS object based on the information
        # written in the ".head" file scamp created.
        with open( headfile ) as ifp:
            hdrtext = ifp.read()
        # The FITS spec says ASCII, but Emmanuel put a non-ASCII latin-1
        # character in his comments... and astropy.io.fits.Header is
        # anal about only ASCII.  Sigh.
        hdrtext = hdrtext.replace( 'é', 'e' )
        strio = io.StringIO( hdrtext )
        hdr = astropy.io.fits.Header.fromfile( strio, sep='\n', padding=False, endcard=False )
        wcs = WCS( hdr )

        return wcs

    finally:
        xmlfile.unlink( missing_ok=True )
        headfile.unlink( missing_ok=True )
