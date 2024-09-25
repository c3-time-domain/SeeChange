import sys
import os
import io
import argparse
import hashlib
import uuid
import multiprocessing
import multiprocessing.pool
import traceback

import numpy as np

from astropy.io import fits

from models.base import SmartSession
from models.instrument import get_instrument_instance
from models.provenance import Provenance
from models.decam import DECam
from models.image import Image
from models.exposure import Exposure

from pipeline.data_store import DataStore
from pipeline.top_level import Pipeline
from pipeline.preprocessing import Preprocessor

import util.radec
from util.logger import SCLogger


class DECamRefFetcher:
    def __init__( self, ra, dec, filter, chips=None, numprocs=10,
                  min_exptime=None, max_seeing=1.2, min_depth=None,
                  min_mjd=None, max_mjd=None, min_per_chip=9,
                  only_local=False, no_fallback=False, full_report=False ):
        self.numprocs = numprocs
        self.ra = ra
        self.dec = dec
        self.filter = filter
        self.min_exptime = min_exptime
        self.max_seeing = max_seeing
        self.min_depth = min_depth
        self.min_mjd = min_mjd
        self.max_mjd = max_mjd
        self.min_per_chip = min_per_chip
        self.only_local = only_local
        self.no_fallback = no_fallback
        self.full_report = full_report

        # NOTE -- empirically, the NOIRLab pipeline seems to be more optimistic
        #   about both seeing and depth that what our pipeline comes up with.
        #   when searcing noirlab, reduce the seeing cut by seeingfac,
        #   and add limmagoff to the limiting magnitude
        self.seeingfac = 1.06
        self.limmagoff = 0.5

        self.decam = get_instrument_instance( 'DECam' )
        knownchips = self.decam.get_section_ids()
        if chips is not None:
            if not isinstance( chips, list ):
                raise TypeError( f"DECamRefFetcher: chips must be a list, not a {type(chips)}" )
            if not set(chips).issubset( set(knownchips ) ):
                raise ValueError( f"DECamRefFetcher: passed chips {chips}, which includes unknown section ids" )
            self.chips = chips
        else:
            self.chips = knownchips
        self.chippos = { c: self.decam.get_ra_dec_for_section( self.ra, self.dec, c ) for c in self.chips }

        # We're going to demand at least min_per_chip images overlapping four points
        #  on the image, each point being "near" one of the corners.
        # Put those ra/dec points in the self.corners dictionary.
        self.corners = {}
        self.cornercount = {}
        self.chipimgs = {}
        cornerfrac = 0.8
        for chip in self.chips:
            ra = self.chippos[chip][0]
            dec = self.chippos[chip][1]
            cd = np.cos( dec * np.pi / 180. )
            dra = cornerfrac * 2048. * self.decam.pixel_scale / 3600. / cd
            ddec = cornerfrac * 1024. * self.decam.pixel_scale / 3600.
            self.corners[chip] = [ ( ra - dra, dec - ddec ),
                                   ( ra - dra, dec + ddec ),
                                   ( ra + dra, dec - ddec ),
                                   ( ra + dra, dec + ddec ) ]
            self.cornercount[chip] = [ 0, 0, 0, 0 ]
            self.chipimgs[chip] = []

    def log_corner_counts( self, prefix="" ):
        strio = io.StringIO()
        strio.write( f"{prefix}corner count for all chips:\n" )
        for chip, counts in self.cornercount.items():
            strio.write( f"    {chip:4s} : {counts}\n" )
        SCLogger.info( strio.getvalue() )


    def do_i_have_enough( self ):
        yes = True
        for chip in self.chips:
            if chip not in self.cornercount:
                yes = False
                break
            if not all( [ self.cornercount[chip][c] >= self.min_per_chip for c in [ 0, 1, 2, 3 ] ] ):
                yes = False
                break
        return yes

    def identify_existing_images( self, reset=True ):
        """Find DECam instcal images already in the database we can use.

        These will probably be here from a previous run of this class.
        Looks for images for each chip whose center includes the chip's
        center (based on ra, dec, and the chip's known offset), and that
        overlap the chip by at least 10%.

        Updates self.chipimgs and self.cornercount

        Parameters
        ----------
          reset : bool, default True
            reset all counts to 0 before starting.  Not doing this could
            easily lead to redundancies.

        """
        if reset:
            self.cornercount = { c: [ 0, 0, 0, 0 ] for c in self.chips }
            self.chipimgs = { c: [] for c in self.chips }

        imgidsfound = set()
        imgs = []

        # OK, this is a little scary.  I'm trying to reconstruct the
        #   provenance that the exposure would have so I can reconstruct
        #   the upstreams of the image provenance we want to search on.
        #   However, the code_version business means that if we ever
        #   change the code_version, things will not match any more. We
        #   really need to do Issue #317 to solve this.

        # This code is based on decam.py::DECam._commit_exposure
        expprov = Provenance( process='download',
                              parameters={ 'proc_type': 'instcal', 'Instrument': 'DECam' },
                              code_version_id=Provenance.get_code_version().id )
        # Make a pipeline using the same arguments we'll use later just so we can get the provenance
        # (This is ugly.)
        p = Pipeline( preprocessing={ 'preprocessing': 'noirlab_instcal',
                                      'calibset': 'externally_supplied',
                                      'flattype': 'externally_supplied' } )
        imgprov = Provenance( process='preprocessing',
                              parameters=p.preprocessor.pars.get_critical_pars(),
                              upstreams=[expprov],
                              code_version_id=Provenance.get_code_version().id )

        for chip in self.chips:
            for corner in [ 0, 1, 2, 3 ]:
                kwargs = { 'ra': self.corners[chip][corner][0],
                           'dec': self.corners[chip][corner][1],
                           'instrument': 'DECam',
                           'filter': self.filter,
                           'provenance_ids': [imgprov.id],
                           'order_by': 'quality'
                          }
                if self.min_exptime is not None:
                    kwargs['min_exp_time'] = self.min_exptime
                if self.min_mjd is not None:
                    kwargs['min_mjd'] = self.min_mjd
                if self.max_mjd is not None:
                    kwargs['max_mjd'] = self.max_mjd
                if self.max_seeing is not None:
                    kwargs['max_seeing'] = self.max_seeing
                if self.min_depth is not None:
                    kwargs['min_lim_mag'] = self.min_depth

                found = Image.find_images( **kwargs )
                newlyfound = [ i for i in found if i.id not in imgidsfound ]
                imgs.extend( newlyfound )
                imgidsfound = set( i.id for i in imgs )

        if len( imgs ) > 0:
            for chip in self.chips:
                for img in imgs:
                    # Totally punting on ra/dec crossing 0°....
                    for corner in range(4):
                        if ( ( img.minra < self.corners[chip][corner][0] ) and
                             ( img.maxra > self.corners[chip][corner][0] ) and
                             ( img.mindec < self.corners[chip][corner][1] ) and
                             ( img.maxdec > self.corners[chip][corner][1] )
                            ):
                            self.cornercount[chip][corner] += 1
                            if img.id not in [ i.id for i in self.chipimgs[chip] ]:
                                self.chipimgs[chip].append( img )
                self.chipimgs[chip].sort( key=lambda i: i.mjd )


    def identify_useful_remote_exposures( self ):
        """Get a DECamOriginExposures objects and a set of indexes of the ones we want.

        Starts with whats in self.cornercount.  Look for exposures
        within 2° of our exposure, then go through all chips and find
        things that overlap.

        """

        # Get all exposures that might be useful

        kwargs = { 'skip_exposures_in_database': True,
                   'ctr_ra': self.ra,
                   'ctr_dec': self.dec,
                   'radius': 2.2,          # DECam has 2.2° field of view, so this gets to edge overlap
                   'proc_type': 'instcal'
                  }
        if self.min_mjd is not None:
            kwargs['minmjd'] = self.min_mjd
        if self.max_mjd is not None:
            kwargs['maxmjd'] = self.max_mjd

        origexps = self.decam.find_origin_exposures( **kwargs )
        # Omit onces that we already have in the database.  If they are there,
        #   and we did things right before, then their images will have already
        #   been found with identify_existing_images.  (This does assume that
        #   when the exposure was loaded, all the images were loaded too.  That's
        #   how things in here work, so that should be OK... I hope.)
        omitdexen = set()
        with SmartSession() as sess:
            for i in range( len(origexps) ):
                # I know that sometimes this fails, so try it up
                try:
                    identifier = origexps.exposure_origin_identifier( i )
                except Exception as ex:
                    omitdexen.add( i )
                else:
                    existing = sess.query( Exposure ).filter( Exposure.origin_identifier==identifier ).all()
                    if len(existing) > 0:
                        omitdexen.add( i )

        # Things that we need to download will go into usefuldexen
        usefuldexen = set()

        # For each chip we're looking for.  We're going to go *backwards* in MJD
        #   because we want to favor more recent images.  (Detector more likely to
        #   be the same as what we're using now, yadda yadda.)
        dexen = list( range(len(origexps)) )
        dexen.reverse()
        for chip in self.chips:
            for expdex in dexen:
                if expdex in omitdexen:
                    continue
                filter = origexps.exposure_filter( expdex )
                expra, expdec = origexps.exposure_coords( expdex )

                # Basic cuts.  The "not" cuts are there, rather than
                #   just using "<=" where we currently have "not >="
                #   (etc.), because if the values we got from the
                #   archive are NaN, the inequality will always be
                #   False.  We want it to not use the image when the
                #   value is NaN (if the appropriate limit isn't itself
                #   None).
                if self.decam.get_short_filter_name( filter ) != self.decam.get_short_filter_name( self.filter ):
                    continue
                if ( ( ( self.max_seeing is not None )
                       and
                       ( not origexps.exposure_seeing(expdex) <= ( self.max_seeing / self.seeingfac ) ) )
                     or
                     ( ( self.min_depth is not None )
                       and
                       ( not origexps.exposure_depth(expdex) >= ( self.min_depth + self.limmagoff ) ) )
                     or
                     ( ( self.min_exptime is not None )
                       and
                       ( not origexps.exposure_exptime(expdex) >= self.min_exptime ) )
                    ):
                    continue

                expra, expdec = origexps.exposure_coords( expdex )

                # Go through all chips of the exposure to figure out which chips
                #   overlap which chips of our target.  Add exposures that
                #   help to usefuldexen, and increment self.cornercount
                for expchip in self.decam.get_section_ids():
                    ra, dec = self.decam.get_ra_dec_for_section( expra, expdec, expchip )
                    cd = np.cos( dec * np.pi / 180. )
                    dra = 2048. * self.decam.pixel_scale / 3600. / cd
                    ddec = 1024. * self.decam.pixel_scale / 3600.
                    minra = ra - dra
                    maxra = ra + dra
                    mindec = dec - ddec
                    maxdec = dec + ddec

                    for corner in range(4):
                        if ( ( minra < self.corners[chip][corner][0] ) and
                             ( maxra > self.corners[chip][corner][0] ) and
                             ( mindec < self.corners[chip][corner][1] ) and
                             ( maxdec > self.corners[chip][corner][1] )
                            ):
                            self.cornercount[chip][corner] += 1
                            usefuldexen.add( expdex )

                # Are we done?  If so, break out of exposure loop, go on to next chip of target
                if all( [ self.cornercount[chip][c] >= self.min_per_chip for c in [ 0, 1, 2, 3 ] ] ):
                    break

        return origexps, usefuldexen


    def extract_image_and_do_things( self, exposure, section_id ):
        """Pull the image out of the exposure, run backgrounding, extraction, psf, wcs, and zp.

        Returns
        -------
          exposure_id, section_id, success(bool), status message

        """

        success = False
        try:
            SCLogger.multiprocessing_replace()
            SCLogger.info( f"Running section {section_id} of exposure {os.path.basename(exposure.filepath)}" )

            # Make a pipeline to do all the things.  Image will be
            #   extracted from the preprocessor in the preprocessing
            #   step.
            save_on_exception = True # False     # Only set to true for developing/debugging
            ds = DataStore( exposure, section_id )
            pipeline = Pipeline( pipeline={ 'through_step': 'zp',
                                            'save_before_subtraction': True,
                                            'save_on_exception': save_on_exception },
                                 preprocessing={ 'preprocessing': 'noirlab_instcal',
                                                 'calibset': 'externally_supplied',
                                                 'flattype': 'externally_supplied' } )
            pipeline.run( ds, no_provtag=True, ok_no_ref_provs=True )
            ds.reraise()

            success = True
            return ( ds.exposure.id, ds.section_id, True, "OK" )

        except Exception as ex:
            strio = io.StringIO()
            traceback.print_exc( file=strio )
            SCLogger.exception( f"Exception processing section {ds.section_id}:\n{strio.getvalue()}" )
            return ( ds.exposure.id, ds.section_id, False, str(ex) )

        finally:
            # THINK about what happens in multiprocessing when a process
            #   is reused.  exposure was passed in; will it be the same
            #   object when this process is used again by the
            #   multiprocessing pool?  I don't know.  Just in case it
            #   is, clear out the memory we used when reading in the
            #   image, so that memory usage doesn't build up.
            # (If exposure is somehow shared memory, then this
            #   may be creating a disaster.)
            exposure.data.clear_cache()
            exposure.weight.clear_cache()
            exposure.flags.clear_cache()

            # Just in case this was run in the master process, restore the
            #   logger to defaults.
            SCLogger.info( f"Finished {'successfully?' if success else 'unsuccessfully'} with section {section_id} "
                           f"of exposure {os.path.basename(exposure.filepath)}" )
            SCLogger.replace()


    def download_and_extract( self, origexps, usefuldexen ):
        """Download identified exposures; load them and their images into the database."""

        SCLogger.info( f"============ Downloading {len(usefuldexen)} reduced exposures." )
        # exposures = origexps.download_and_commit_exposures( list(usefuldexen) )
        exposures = origexps.download_and_commit_exposures( list(usefuldexen),
                                                            delete_downloads=False,
                                                            existing_ok=True )

        for exposure_n, exposure in enumerate( exposures ):
            SCLogger.info( f"Downloading and extracting an exposure\n"
                           "------------------------------------------------------------\n"
                           f"Doing {exposure_n+1} of {len(exposures)} exposures: {exposure.filepath} ({exposure.id})\n"
                           f"------------------------------------------------------------" )
            submitted = set()
            success = {}
            message = {}
            def collate( res ):
                SCLogger.info( f"Got {res[2]} for chip {res[1]} ({res[3]})" )
                success[ res[1] ] = res[2]
                message[ res[1] ] = res[3]

            SCLogger.info( f"Creating pool of {self.numprocs} processes to do "
                           f"{len(self.decam.get_section_ids())} chips." )
            with multiprocessing.pool.Pool( self.numprocs ) as pool:
                for section_id in self.decam.get_section_ids():
                    pool.apply_async( self.extract_image_and_do_things,  (exposure, section_id), {}, collate )
                    submitted.add( section_id )
                SCLogger.info( "Submitted all worker jobs, waiting for them to finish." )
                pool.close()
                pool.join()

            if len(success) != len(submitted):
                strio = io.StringIO()
                strio.write( f"Exposure failure, not enough responses\n"
                             f"************************************************************\n" )
                strio.write( f"   FAILED for exposure {os.path.basename(exposure.filepath)}: "
                             f"only {len(success)} responses "
                             f"from subprocesses, expected {len(submitted)}\n" )
                strio.write( f"************************************************************" )
                SCLogger.error( strio.getvalue() )

            succeeded = set( k for k, v in success.items() if v )
            failed = set( k for k, v in success.items() if not v )
            if len(failed) > 0:
                strio = io.StringIO()
                strio.write( f"Exposure failure, error returns\n"
                             f"************************************************************\n" )
                strio.write( f"   FAILED for exposure {os.path.basename(exposure.filepath)}: "
                             f"the following sensor sections reported failure: {failed}\n" )
                for failure in failed:
                    strio.write( f"   {failure}: {message[failure]}\n" )
                strio.write( f"************************************************************" )
                SCLogger.error( strio.getvalue() )

            SCLogger.info( f"Finished with exposure\n"
                           f"------------------------------------------------------------\n"
                           f"  FINISHED with exposure {exposure.filepath}; "
                           f"{len(succeeded)} of {len(submitted)} passed\n"
                           f"------------------------------------------------------------\n" )


    def __call__( self ):
        # NOIRLab archives have seeing and depth information for reduced images... only not for
        #   all of them, sadly.  Ideally, we'd be able to get enough just by looking at the ones
        #   with that information, but in practice, there are so many that don't have it that we
        #   need to work around that.  So, work as follows:
        #
        # 1. Identify existing images already in our database, build up counts of what we have
        # 2. If we have enough, be done!  Go to step 7.
        # 3. Go to NOIRLab and download enough exposures with good enough seeing/depth
        #    to meet the number of images we want for all exposures, or all of them that
        #    meeting the seeing/depth criteria if there aren't enough.  Some of the reductions
        #    may fail; if so, just shrug and work with the ones that succeed.
        # 4. Look back in our database and identify what's there, building up counts of what we have.
        # 5. If we have enough, be done!  Go to step 7.
        # 6. ITERATE:
        #    A. Go to NOIRLab and download exposures with a high enough exptime.  Assume that
        #       all of them have good enough seeing/depth, and download enough to fill out the
        #       counts we want.  If there are no exposures left we haven't grabbed, stop iteration.
        #    B. Look back in our database and identify what's there, building up counts of what we have.
        #       If it's enough, stop iteration.  Otherwise, continue.
        # 7. Look back our database and identify what's there.  Report on how well we did.

        exptimecache = self.min_exptime
        maxseeingcache = self.max_seeing
        mindepthcache = self.min_depth

        SCLogger.info( f"============ Initial identification of existing images ============" )
        self.min_exptime = None
        self.identify_existing_images( reset=True )
        self.log_corner_counts( ">>>>> After initial database search, " )

        if self.do_i_have_enough():
            SCLogger.info( f"============ We're done! ============" )
        elif not self.only_local:
            done = False
            first = True
            while not done:
                if first:
                    self.min_exptime = None
                else:
                    if self.no_fallback:
                        done = True
                        break
                    self.min_exptime = exptimecache
                    self.max_seeing = None
                    self.min_depth = None

                if ( self.min_depth is not None ) or ( self.max_seeing is not None ):
                    SCLogger.info( f"============ Pull NOIRLab exposures with known quality ============" )
                else:
                    SCLogger.info( f"============ Pull whatever NOIRLab exposures and hope ============" )
                origexps, usefuldexen = self.identify_useful_remote_exposures()
                if ( ( not first ) or ( self.no_fallback ) ) and ( len( usefuldexen ) == 0 ):
                    SCLogger.error( f"============ Ran out of NOIRLab exposures before we were done ============" )
                    done = True
                self.log_corner_counts( f">>>>> After identifying NOIRLab exposures, expect " )
                self.download_and_extract( origexps, usefuldexen )

                self.min_exptime = None
                self.max_seeing = maxseeingcache
                self.min_depth = mindepthcache
                self.identify_existing_images( reset=True )
                self.log_corner_counts( f">>>>> After latest iteration pulling from NOIRlab: " )
                if self.do_i_have_enough():
                    SCLogger.info( f"============ We're done! ============" )
                    done = True

                first = False
        else:
            SCLogger.info( "only_local is set, not searching NOIRLab archive for more references." )

        # Report
        self.min_exptime = None
        self.max_seeing = maxseeingcache
        self.min_depth = mindepthcache
        self.identify_existing_images( reset=True )
        self.log_corner_counts( "Final " )

        if self.full_report:
            strio = io.StringIO()
            for chip in self.chips:
                strio.write( f"================ Chip {chip}\n" )
                for corner in [ 0, 1, 2, 3 ]:
                    strio.write( f"    ============ Corner {corner}: {self.cornercount[chip][corner]:2d} "
                                 f" (RA={self.corners[chip][corner][0]:.5f}, "
                                 f"Dec={self.corners[chip][corner][1]:.5f})\n" )
                strio.write( f"    ============ Ref Images:\n" )
                strio.write( f"        Seeing  Lim_Mag  Filepath\n" )
                strio.write( f"        ------  -------  --------\n" )
                for img in self.chipimgs[chip]:
                    strio.write( f"        {img.fwhm_estimate:6.2f}  {img.lim_mag_estimate:7.2f}  {img.filepath}\n" )

            SCLogger.info( f"Full report:\n{strio.getvalue()}\n" )



class MyArgFormatter( argparse.ArgumentDefaultsHelpFormatter ): # , argparse.RawDescriptionHelpFormatter ):
    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )

def main():
    SCLogger.info( f"sys.argv: {sys.argv}" )

    parser = argparse.ArgumentParser( 'acquire_decam_refs',
                                      description="Download NOIRlab-procssed DECam exposures for use as references.",
                                      epilog=( "WARNING: Does not properly handle RA crossing 0°. "
                                               "If you are looking for an exposure with RA < 3° or RA > 357°, "
                                               "fix the code before runnig this!" ),
                                      formatter_class=MyArgFormatter )
    parser.add_argument( '-r', '--ra', type=float, required=True, help="Center RA (degrees) of target exposure" )
    parser.add_argument( '-d', '--dec', type=float, required=True, help="Center Dec (degrees) of target exposure" )
    parser.add_argument( '-f', '--filter', required=True, help="Filter (g, r, i, z) to search for" )
    parser.add_argument( '-t', '--min-exptime', type=float, default=None, help="Min exposure time in seconds to pull" )
    parser.add_argument( '-s', '--max-seeing', type=float, default=None, required=True,
                         help="Maximum seeing of exposures to pull" )
    parser.add_argument( '-l', '--min-depth', type=float, default=None, required=True,
                         help=( "Minimum magnitude limit.  This will be filter-dependent" ) )
    parser.add_argument( '-n', '--min-mjd', type=float, default=None, help="Minimum mjd; default=no limit" )
    parser.add_argument( '-x', '--max-mjd', type=float, default=None, help="Maximum mjd; default=no limit" )
    parser.add_argument( '-c', '--min-num-per-chip', type=int, default=9,
                         help=( "Make sure to get exposure so that each chip will have at least this many "
                                "images overlapping it." ) )
    parser.add_argument( '--only-local', action='store_true', default=False,
                         help=( "Don't actually try to get references, just look in the database to see "
                                "what we have already.  Implicitly includes --no-fallback." ) )
    parser.add_argument( '--no-fallback', action='store_true', default=False,
                         help=( "If not enough exposures that match the seeing/depth criterion are found initiially, "
                                "don't fall back to the slow process of downloading lots of exposures until "
                                "we have enough." ) )
    parser.add_argument( '--full-report', action='store_true', default=False,
                         help=( "At the end, show all of the images found for all of the corners. "
                                "This will be long." ) )
    parser.add_argument( '-p', '--numprocs', type=int, default=10,
                         help="Number of extraction/wcs/zp processes to run" )
    args = parser.parse_args()

    # I'm not really happy about how we handle the whole filter and filter_short thing
    #   right now.  This is kind of a hack, needed for the database searches to work.
    # It'd be nice if in the database we stored filter_short, and then the class had a
    #   lookup from full filter descriptions in headers to filter_short.
    filtertranslation = { 'g': 'g DECam SDSS c0001 4720.0 1520.0',
                          'r': 'r DECam SDSS c0002 6415.0 1480.0',
                          'i': 'i DECam SDSS c0003 7835.0 1470.0',
                          'z': 'z DECam SDSS c0004 9260.0 1520.0' }
    if args.filter not in filtertranslation:
        raise ValueError( f"Unknown filter {args.filter}" )
    band = filtertranslation[ args.filter ]

    fetcher = DECamRefFetcher( args.ra, args.dec, band,
                               min_exptime=args.min_exptime, max_seeing=args.max_seeing, min_depth=args.min_depth,
                               min_mjd=args.min_mjd, max_mjd=args.max_mjd, min_per_chip=args.min_num_per_chip,
                               numprocs=args.numprocs,
                               only_local=args.only_local, no_fallback=args.no_fallback, full_report=args.full_report )
    fetcher()


# **********************************************************************
if __name__ == "__main__":
    main()
