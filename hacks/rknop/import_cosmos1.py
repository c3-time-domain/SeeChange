import os
import sys
import io
import re
import time
import argparse
import logging
import multiprocessing
import multiprocessing.pool

from util.logger import SCLogger

from models.base import Session
from models.reference import Reference
from models.instrument import get_instrument_instance
from models.decam import DECam

from import_decam_reference import import_decam_reference

class Importer:
    def __init__( self, filts, ccdnums ):
        self.filts = filts
        self.ccdnums = ccdnums
        self.success = { f: { c: None for c in self.ccdnums } for f in self.filts }

    def importref( self, fbase, target, filt, ccdnum ):
        try:
            me = multiprocessing.current_process()
            # (I know that the process names are going to be something like ForkPoolWorker-{number}
            match = re.search( '(\d+)', me.name )
            if match is not None:
                me.name = f'{int(match.group(1)):3d}'
            else:
                me.name = str( me.pid )
            SCLogger.replace( me.name )
            SCLogger.info( f"importer starting for filter {filt} ccd {ccdnum}; process {me.name} PID {me.pid} " )

            decam = get_instrument_instance( 'DECam' )

            chipid = None
            for k, v in decam._chip_radec_off.items():
                if v['ccdnum'] == ccdnum:
                    chipid = k
                    break

            if chipid is None:
                raise ValueError( f"{me.name} couldn't find chipid for ccd {ccdnum}" )

            SCLogger.info( f"Got chipid {chipid} for ccdnum {ccdnum}" )

            # See if there's one already, don't import if so
            with Session() as sess:
                them = ( sess.query( Reference )
                         .filter( Reference.target == target )
                         .filter( Reference.filter.startswith(filt) )
                         .filter( Reference.section_id == chipid ) ).all()
            if len(them) > 0:
                if len(them) > 1:
                    s = ( f"{me.name}: >1 Reference entry for {target} "
                          f"section {chipid} filter {filt} !!!" )
                    SCLogger.warning( s )
                else:
                    s = ( f"{me.name}: Reference already present for {target} "
                          f"section {chipid} filter {filt}" )
                    SCLogger.info( s )
                return ( filt, ccdnum, True )

            else:
                image = f'{fbase}/{target}-{filt}-templ/{target}-{filt}-templ.{ccdnum:02d}.fits.fz'
                weight = f'{fbase}/{target}-{filt}-templ/{target}-{filt}-templ.{ccdnum:02d}.weight.fits.fz'
                mask = f'{fbase}/{target}-{filt}-templ/{target}-{filt}-templ.{ccdnum:02d}.bpm.fits.fz'
                try:
                    import_decam_reference( image, weight, mask, target, 1, chipid )
                    return ( filt, ccdnum, True )
                except Exception as ex:
                    SCLogger.exception( f"Exception importing filter {filt} chip {chipid}: {ex}" )
                    return ( filt, ccdnum, False )

        except Exception as ex:
            SCLogger.exception( f"{me.name} exception: {ex}" )
            return ( filt, ccdnum, False )

    def report( self, tup ):
        filt, ccdnum, result = tup
        self.success[filt][ccdnum] = result

    def count( self ):
        ntot = 0
        nsucc = 0
        nfail = 0
        nunknown = 0
        for filt in self.filts:
            for ccd in self.ccdnums:
                ntot += 1
                if self.success[filt][ccd] is None:
                    nunknown += 1
                elif self.success[filt][ccd]:
                    nsucc += 1
                else:
                    nfail += 1

        return ntot, nsucc, nfail, nunknown

# ======================================================================

def main():
    parser = argparse.ArgumentParser( "Import DECam refs",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "-t", "--target", default="COSMOS-1", help="target / field name" )
    parser.add_argument( "-n", "--numprocs", type=int, default=10, help="Number of importer processes" )
    parser.add_argument( "-b", "--basedir", default="/refs", help="Base directory; rest of path is assumed standard" )
    parser.add_argument( "-f", "--filters", nargs='+', default=['g','r','i'], help="Filters" )
    parser.add_argument( "-c", "--ccdnums", nargs='+', type=int, default=[],
                         help="CCD numbers to do.  Default: 1-62, omitting 31 and 61." )
    args = parser.parse_args()

    SCLogger.setLevel( logging.INFO )

    ncpus = multiprocessing.cpu_count()
    ompnthreads = int( ncpus / args.numprocs )
    SCLogger.info( f"Setting OMP_NUM_THREADS={ompnthreads} for {ncpus} cpus and {args.numprocs} processes" )
    os.environ[ "OMP_NUM_THREADS" ] = str(ompnthreads)

    ccds = args.ccdnums
    if len(ccds) == 0:
        ccds = list( range(1, 31) )
        ccds.extend( range(32, 61) )
        ccds.append( 62 )

    importer = Importer( args.filters, ccds )

    SCLogger.info( f"Creating Pool of {args.numprocs} processes" )
    with multiprocessing.pool.Pool( args.numprocs, maxtasksperchild=1 ) as pool:
        for filt in args.filters:
            for ccdnum in ccds:
                SCLogger.info( f"Launching filter {filt} ccd {ccdnum}" )
                pool.apply_async( importer.importref,
                                  ( args.basedir, args.target, filt, ccdnum ),
                                  {},
                                  importer.report )

        SCLogger.info( f"Submitted all worker jobs, waiting for them to finish." )
        pool.close()
        pool.join()

    tot, succ, fail, unk = importer.count()

    SCLogger.info( f"All done.  {tot} jobs : {succ} succeeded (maybe), {fail} failed, {unk} not reported" )

# ======================================================================

if __name__ == "__main__":
    main()
