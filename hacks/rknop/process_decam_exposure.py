import sys
import os
import logging
import argparse
import pathlib
import multiprocessing
import multiprocessing.pool

import sqlalchemy as sa
from astropy.io import fits

from util.config import Config

from models.base import Session
from models.exposure import Exposure
from models.calibratorfile import CalibratorFile
from models.datafile import DataFile
from models.instrument import get_instrument_instance
import models.decam

from pipeline.top_level import Pipeline

_config = Config.get()

_logger = logging.getLogger(__name__)
_logout = logging.StreamHandler( sys.stderr )
_logger.addHandler( _logout )
_formatter = logging.Formatter( f'[%(asctime)s - %(levelname)s] - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S' )
_logout.setFormatter( _formatter )
_logger.setLevel( logging.INFO )

# ======================================================================

class ExposureProcessor:
    def __init__( self, exposurefile, decam ):
        self.decam = decam

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

        # Load this exposure into the database if it's not there already
        # (And fill the self.exposure property)
                
        with Session() as sess:
            self.exposure = sess.scalars( sa.select(Exposure).where( Exposure.filepath == relpath ) ).first()
            if self.exposure is None:
                _logger.info( f"Loading exposure {relpath} into database" )
                self.exposure = Exposure( filepath=relpath, instrument='DECam', **exphdrinfo )
                self.exposure.save()

                self.exposure.provenance = sess.merge( self.exposure.provenance )
                sess.merge( self.exposure )
                sess.commit()
            else:
                _logger.info( f"Exposure {relpath} is already in the database" )

        _logger.info( f"Exposure id is {self.exposure.id}" )
        self.results = {}
                

    def processchip( self, chip ):
        try:
            me = multiprocessing.current_process()
            _logger.info( f"Processing chip {chip} in process {me.name} PID {me.pid}" )
            pipeline = Pipeline()
            ds = pipeline.run( self.exposure, chip )
            ds.save_and_commit()
            return ( chip, True )
        except Exception as ex:
            _logger.exception( f"Exception processing chip {chip}: {ex}" )
            return ( chip, False )

    def collate( self, res ):
        chip, succ = res
        self.results[ chip ] = res
                
# ======================================================================
        
def main():
    parser = argparse.ArgumentParser( 'Run a DECam exposure through the pipeline',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "exposure", help="Path to exposure file" )
    parser.add_argument( "-n", "--numprocs", default=60, type=int, help="Number of processes to run at once" )
    parser.add_argument( "-c", "--chips", nargs='+', default=[], help="Chips to process (default: all good)" );
    args = parser.parse_args()

    ncpus = multiprocessing.cpu_count()
    ompnumthreads = int( ncpus / args.numprocs )
    _logger.info( f"Setting OMP_NUM_THREADS={ompnumthreads} for {ncpus} cpus and {args.numprocs} processes" )
    os.environ[ "OMP_NUM_THREADS" ] = str( ompnumthreads )

    decam = get_instrument_instance( 'DECam' )


    # Before I even begin, I know that I'm going to have problems with
    # the decam linearity file.  There's only one... but all processes
    # are going to try to import it into the database at once.  This
    # ends up confusing the archive as a whole bunch of processes try to
    # write exactly the same file at exactly the same time.  This
    # behavior should be investigated -- why did the archive fail?  On
    # the other hand, it's highly dysfunctional to have a whole bunch of
    # processes trying to upload the same file at the same time; very
    # wasteful to have them all repeating each other's effort of
    # aquiring the file.  So, just pre-import it for current purposes.

    _logger.info( "Ensuring presence of DECam linearity calibrator file" )
    
    with Session() as session:
        df = ( session.query( DataFile )
               .filter( DataFile.filepath=='DECam_default_calibrators/linearity/linearity_table_v0.4.fits' ) )
        if df.count() == 0:
            cf = decam._get_default_calibrator( 60000, 'N1', calibtype='linearity', session=session )
            df = cf.datafile
        else:
            df = df.first()

        decam = get_instrument_instance( 'DECam' )
        secs = decam.get_section_ids()
        for sec in secs:
            cf = ( session.query( CalibratorFile )
                   .filter( CalibratorFile.type == 'linearity' )
                   .filter( CalibratorFile.calibrator_set == 'externally_supplied' )
                   .filter( CalibratorFile.instrument == 'DECam' )
                   .filter( CalibratorFile.sensor_section == sec )
                   .filter( CalibratorFile.datafile == df ) )
            if cf.count() == 0:
                cf = CalibratorFile( type='linearity',
                                     calibrator_set='externally_supplied',
                                     flat_type=None,
                                     instrument='DECam',
                                     sensor_section=ssec,
                                     datafile=df )
                cf = session.merge( cf )
        session.commit()
                
    _logger.info( "DECam linearity calibrator file is accounted for" )


    # Now on to the real work
    
    exproc = ExposureProcessor( args.exposure, decam )

    chips = args.chips
    if len(chips) == 0:
        decam_bad_chips = [ 'S7', 'N30' ]
        chips = [ i for i in decam.get_section_ids() if i not in decam_bad_chips ]
    
    if args.numprocs > 1:
        _logger.info( f"Creating Pool of {args.numprocs} processes to do {len(chips)} chips" )
        with multiprocessing.pool.Pool( args.numprocs, maxtasksperchild=1 ) as pool:
            for chip in chips:
                pool.apply_async( exproc.processchip, ( chip, ), {}, exproc.collate )

            _logger.info( f"Submitted all worker jobs, waiting for them to finish." )
            pool.close()
            pool.join()
    else:
        # This is useful for some debugging (though it can't catch
        # process interaction issues (like database locks)).
        _logger.info( f"Running {len(chips)} chips serially" )
        for chip in chips:
            exproc.collate( exproc.processchip( chip ) )

    succeeded = { k for k, v in exproc.results.items() if v }
    failed = { k for k, v in exproc.results.items() if not v }
    _logger.info( f"{len(succeeded)+len(failed)} chips processed; "
                  f"{len(succeeded)} succeeded (maybe), {len(failed)} failed (definitely)" )
    _logger.info( f"Succeeded (maybe): {succeeded}" )
    _logger.info( f"Failed (definitely): {failed}" )
    
            
# ======================================================================

if __name__ == "__main__":
    main()
