import sys
import os
import re
import logging
import argparse
import pathlib
import multiprocessing
from mpi4py import MPI

import sqlalchemy as sa
from astropy.io import fits

from util.config import Config
from util.logger import SCLogger
from util.runner import Runner

from models.base import Session
from models.exposure import Exposure
from models.calibratorfile import CalibratorFile
from models.datafile import DataFile
from models.instrument import get_instrument_instance
import models.decam

from pipeline.top_level import Pipeline

_config = Config.get()

_results = {}

# ======================================================================

def load_exposure( decam, exposurefile ):
    # Make sure we can read the input exposure file

    dataroot = pathlib.Path( _config.value( 'path.data_root' ) )
    exposurefile = pathlib.Path( exposurefile )
    if not exposurefile.is_relative_to( dataroot ):
        raise ValueError( f"Exposure needs to be under {dataroot}, but {exposurefile} is not" )
    relpath = str( exposurefile.relative_to( dataroot ) )

    if not exposurefile.is_file():
        raise FileNotFoundError( f"Can't find file {exposurefile}" )
    with fits.open( exposurefile, memmap=True ) as ifp:
        hdr = ifp[0].header
    exphdrinfo = decam.extract_header_info( hdr, ['mjd', 'exp_time', 'filter', 'project', 'target'] )

    with Session() as sess:
        exposure = sess.scalars( sa.select(Exposure).where( Exposure.filepath == relpath ) ).first()
        if exposure is None:
            SCLogger.info( f"Loading exposure {relpath} into database" )
            exposure = Exposure( filepath=relpath, instrument='DECam', **exphdrinfo )
            xposure.save()

            exposure.provenance = sess.merge( exposure.provenance )
            sess.merge( exposure )
            sess.commit()
        else:
            SCLogger.info( f"Exposure {relpath} is already in the database" )

    SCLogger.info( f"Exposure id is {exposure.id}" )
    return exposure.id


def processchip( tup ):
    exposure_id, chip = tup
    try:
        SCLogger.info( f"Processing chip {chip}" )

        with Session() as sess:
            exposure = sess.query( Exposure).filter( Exposure.id==exposure_id ).first()

        pipeline = Pipeline()
        ds = pipeline.run( exposure, chip )
        ds.save_and_commit()
        return ( chip, True )
    except Exception as ex:
        SCLogger.exception( f"Exception processing chip {chip}: {ex}" )
        return ( chip, False )


def collate( self, res ):
    chip, succ = res
    _results[ chip ] = res

# ======================================================================

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    SCLogger.replace( f'{rank:03d}' )
    SCLogger.set_level( logging.DEBUG )

    parser = argparse.ArgumentParser( 'Run a DECam exposure through the pipeline',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "exposure", help="Path to exposure file" )
    parser.add_argument( "-c", "--chips", nargs='+', default=[], help="Chips to process (default: all good)" );
    args = parser.parse_args()

    ncpus = multiprocessing.cpu_count()
    ompnumthreads = int( ncpus / size )
    if rank == 0:
        SCLogger.info( f"Setting OMP_NUM_THREADS={ompnumthreads} for {ncpus} cpus and {size} processes" )
    os.environ[ "OMP_NUM_THREADS" ] = str( ompnumthreads )

    decam = get_instrument_instance( 'DECam' )


    # There are several things that have multiprocessing problems, not
    # just linearity.  sqlalchemy.merge() doesn't handle multiple
    # processes trying to merge the same thing at the same time very
    # well.  This happens with provenances, and exposures.  So, for now,
    # the hackaround is going to be to run a single chip first, to
    # "prime the pump" and get stuff loaded into the database, and only
    # the run all the chips.

    # # Before I even begin, I know that I'm going to have problems with
    # # the decam linearity file.  There's only one... but all processes
    # # are going to try to import it into the database at once.  This
    # # ends up confusing the archive as a whole bunch of processes try to
    # # write exactly the same file at exactly the same time.  This
    # # behavior should be investigated -- why did the archive fail?  On
    # # the other hand, it's highly dysfunctional to have a whole bunch of
    # # processes trying to upload the same file at the same time; very
    # # wasteful to have them all repeating each other's effort of
    # # aquiring the file.  So, just pre-import it for current purposes.

    # SCLogger.info( "Ensuring presence of DECam linearity calibrator file" )

    # with Session() as session:
    #     df = ( session.query( DataFile )
    #            .filter( DataFile.filepath=='DECam_default_calibrators/linearity/linearity_table_v0.4.fits' ) )
    #     if df.count() == 0:
    #         cf = decam._get_default_calibrator( 60000, 'N1', calibtype='linearity', session=session )
    #         df = cf.datafile
    #     else:
    #         df = df.first()

    #     decam = get_instrument_instance( 'DECam' )
    #     secs = decam.get_section_ids()
    #     for sec in secs:
    #         cf = ( session.query( CalibratorFile )
    #                .filter( CalibratorFile.type == 'linearity' )
    #                .filter( CalibratorFile.calibrator_set == 'externally_supplied' )
    #                .filter( CalibratorFile.instrument == 'DECam' )
    #                .filter( CalibratorFile.sensor_section == sec )
    #                .filter( CalibratorFile.datafile == df ) )
    #         if cf.count() == 0:
    #             cf = CalibratorFile( type='linearity',
    #                                  calibrator_set='externally_supplied',
    #                                  flat_type=None,
    #                                  instrument='DECam',
    #                                  sensor_section=ssec,
    #                                  datafile=df )
    #             cf = session.merge( cf )
    #     session.commit()

    # SCLogger.info( "DECam linearity calibrator file is accounted for" )

    # Now on to the real work

    if rank == 0:
        SCLogger.info( f"Ensuring exposure {args.exposure} is in database" )
        exposure_id = load_exposure( decam, args.exposure )
        SCLogger.info( f"Done loading exposure." )

        chips = args.chips
        if len(chips) == 0:
            decam_bad_chips = [ 'S7', 'N30' ]
            chips = [ ( exposure_id, i ) for i in decam.get_section_ids() if i not in decam_bad_chips ]
    else:
        chips = None

    chips = comm.bcast( chips, root=0 )

    if rank == 0:
        SCLogger.info( f'Running' )
    runner = Runner( comm, collate, processchip )
    succ = runner.go( chips )
    if rank == 0:
        SCLogger.info( f"runner.go returned {succ}" )

        succeeded = { k for k, v in _results.items() if v }
        failed = { k for k, v in _results.items() if not v }
        SCLogger.info( f"{len(succeeded)+len(failed)} chips processed; "
                       f"{len(succeeded)} succeeded (maybe), {len(failed)} failed (definitely)" )
        SCLogger.info( f"Succeeded (maybe): {succeeded}" )
        SCLogger.info( f"Failed (definitely): {failed}" )

    MPI.Finalize()

# ======================================================================

if __name__ == "__main__":
    main()
