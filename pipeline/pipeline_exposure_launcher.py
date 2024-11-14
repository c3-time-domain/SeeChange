import re
import time
import requests
import binascii
import multiprocessing
import multiprocessing.pool
import psutil
import logging
import argparse

from util.config import Config
from util.conductor_connector import ConductorConnector
from util.logger import SCLogger

from models.base import SmartSession
from models.knownexposure import KnownExposure
from models.instrument import get_instrument_instance

# Importing this because otherwise when I try to do something completly
# unrelated to Object or Measurements, sqlalchemy starts objecting about
# relationships between those two that aren't defined.
# import models.object

# Gotta import the instruments we might use before instrument fills up
# its cache of known instrument instances
import models.decam

from pipeline.top_level import Pipeline

class ExposureProcessor:
    def __init__( self, instrument, identifier, params, numprocs, onlychips=None,
                  through_step=None, worker_log_level=logging.WARNING ):
        """A class that processes all images in a single exposure, potentially using multiprocessing.

        This is used internally by ExposureLauncher; normally, you would not use it directly.

        Parameters
        ----------
        instrument : str
          The name of the instrument

        identifier : str
          The identifier of the exposure (as defined in the KnownExposures model)

        params : str
          Parameters necessary to get this exposure (as defined in the KnownExposures model)

        numprocs: int
          Number of worker processes (not including the manager process)
          to run at once.  0 or 1 = do all work in the main manager process.

        onlychips : list, default None
          If not None, will only process the sensor sections whose names
          match something in this list.  If None, will process all
          sensor sections returned by the instrument's get_section_ids()
          class method.

        through_step : str or None
          Passed on to top_level.py::Pipeline

        worker_log_level : log level, default logging.WARNING
          The log level for the worker processes.  Here so that you can
          have a different log level for the overall control process
          than in the individual processes that run the actual pipeline.

        """
        self.instrument = get_instrument_instance( instrument )
        self.identifier = identifier
        self.params = params
        self.numprocs = numprocs
        self.onlychips = onlychips
        self.through_step = through_step
        self.worker_log_level = worker_log_level

    def cleanup( self ):
        """Do our best to free memory."""

        self.exposure = None   # Praying to the garbage collection gods

    def download_and_load_exposure( self ):
        """Download the exposure and load it into the database (and archive)."""

        SCLogger.info( f"Downloading exposure {self.identifier}..." )
        self.exposure = self.instrument.acquire_and_commit_origin_exposure( self.identifier, self.params )
        SCLogger.info( f"...downloaded." )
        # TODO : this Exposure object is going to be copied into every processor subprocess
        #   *Ideally* no data was loaded, only headers, so the amount of memory used is
        #   not significant, but we should investigate/verify this, and deal with it if
        #   that is not the case.

    def processchip( self, chip ):
        """Process a single chip of the exposure through the top level pipeline.

        Parameters
        ----------
          chip : str
            The SensorSection identifier

        """
        origloglevel = SCLogger.getEffectiveLevel()
        try:
            me = multiprocessing.current_process()
            # (I know that the process names are going to be something like ForkPoolWorker-{number}
            match = re.search( '([0-9]+)', me.name )
            if match is not None:
                me.name = f'{int(match.group(1)):3d}'
            else:
                me.name = str( me.pid )
            SCLogger.replace( midformat=me.name, level=self.worker_log_level )
            SCLogger.info( f"Processing chip {chip} in process {me.name} PID {me.pid}..." )
            SCLogger.setLevel( self.worker_log_level )
            pipeline = Pipeline()
            if ( self.through_step is not None ) and ( self.through_step != 'exposure' ):
                pipeline.pars.through_step = self.through_step
            ds = pipeline.run( self.exposure, chip, save_intermediate_products=False )
            ds.save_and_commit()
            SCLogger.setLevel( origloglevel )
            SCLogger.info( f"...done processing chip {chip} in process {me.name} PID {me.pid}." )
            return ( chip, True )
        except Exception as ex:
            SCLogger.exception( f"Exception processing chip {chip}: {ex}" )
            return ( chip, False )
        finally:
            # Just in case this was run in the master process, we want to reset
            #   the log format and level to what it was before.
            SCLogger.replace()
            SCLogger.setLevel( origloglevel )

    def collate( self, res ):
        """Collect responses from the processchip() parameters (for multiprocessing)."""
        chip, succ = res
        self.results[ chip ] = res

    def __call__( self ):
        """Run all the pipelines for the chips in the exposure."""

        if self.through_step == 'exposure':
            SCLogger.info( f"Only running through exposure, not launching any image processes" )
            return

        chips = self.instrument.get_section_ids()
        if self.onlychips is not None:
            chips = [ c for c in chips if c in self.onlychips ]
        self.results = {}

        if self.numprocs > 1:
            SCLogger.info( f"Creating pool of {self.numprocs} processes to do {len(chips)} chips" )
            with multiprocessing.pool.Pool( self.numprocs, maxtasksperchild=1 ) as pool:
                for chip in chips:
                    pool.apply_async( self.processchip, ( chip, ), {}, self.collate )

                SCLogger.info( f"Submitted all worker jobs, waiting for them to finish." )
                pool.close()
                pool.join()
        else:
            # This is useful for some debugging (though it can't catch
            # process interaction issues (like database locks)).
            SCLogger.info( f"Running {len(chips)} chips serially" )
            for chip in chips:
                self.collate( self.processchip( chip ) )

        succeeded = { k for k, v in self.results.items() if v }
        failed = { k for k, v in self.results.items() if not v }
        SCLogger.info( f"{len(succeeded)+len(failed)} chips processed; "
                       f"{len(succeeded)} succeeded (maybe), {len(failed)} failed (definitely)" )
        SCLogger.info( f"Succeeded (maybe): {succeeded}" )
        SCLogger.info( f"Failed (definitely): {failed}" )


class ExposureLauncher:
    """A class that polls the conductor asking for things to do, launching a pipeline when one is found.

    Instantiate it with cluster_id, node_id, and numprocs, and then call the instance as a function.

    """

    def __init__( self, cluster_id, node_id, numprocs=None, verify=True, onlychips=None,
                  through_step=None, worker_log_level=logging.WARNING ):
        """Make an ExposureLauncher.

        Parameters
        ----------
        cluster_id : str
           The id of the cluster that this ExposureLauncher is running on

        node_id : str
           The id of the node within the cluster that this ExposureLauncher is running on

        numprocs : int or None
           The number of worker processes to run (in addition to a
           single manager process).  Make this 0 if you want to run
           this single-threaded, with the actual work happening
           serially in the manager process.  Normally, you want to
           make this the number of CPUs you have minus one (for the
           manager process), but you might make it less if you have
           e.g. memory limitations.  if this is None, it will ask the
           system for the number of physical (not logical) CPUs and
           set this value to that minus one.

        verify : bool, default True
           Make this False if the conductor doesn't have a properly
           signed SSL certificate and you really know what you're
           doing.  (Normally, this is only False within self-contained
           test environments, and should never be False in
           production.)

        onlychips : list, default None
          If not None, will only process the sensor sections whose names
          match something in this list.  If None, will process all
          sensor sections returned by the instrument's get_section_ids()
          class method.

        through_step : str or None
          Parameter passed on to top_level.py::Pipeline, unless it is "exposure"
          in which case all we do is download the exposure and load it into the
          database.

        worker_log_level : log level, default logging.WARNING
          The log level for the worker processes.  Here so that you can
          have a different log level for the overall control process
          than in the individual processes that run the actual pipeline.

        """
        self.sleeptime = 120
        # Subtract 1 from numprocs because this process is running... though this process will mostly
        #  be waiting, so perhaps that's not really necessary.
        self.numprocs = numprocs if numprocs is not None else ( psutil.cpu_count(logical=False) - 1 )
        self.cluster_id = cluster_id
        self.node_id = node_id
        self.onlychips = onlychips
        self.through_step = through_step
        self.worker_log_level = worker_log_level
        self.conductor = ConductorConnector( verify=verify )

    def register_worker( self, replace=False ):
        url = f'registerworker/cluster_id={self.cluster_id}/node_id={self.node_id}/nexps=1/replace={int(replace)}'
        data = self.conductor.send( url )
        self.pipelineworker_id = data['id']

    def unregister_worker( self, replace=False ):
        url = f'unregisterworker/pipelineworker_id={self.pipelineworker_id}'
        try:
            data = self.conductor.send( url )
            if data['status'] != 'worker deleted':
                SClogger.error( "Surprising response from conductor unregistering worker: {data}" )
        except Exception as e:
            SCLogger.exception( "Exception unregistering worker, continuing" )

    def send_heartbeat( self ):
        url = f'workerheartbeat/{self.pipelineworker_id}'
        self.conductor.send( url )

    def __call__( self, max_n_exposures=None, die_on_exception=False ):
        """Run the pipeline launcher.

        Will run until it's processed max_n_exposures exposures
        (default: runs indefinitely).  Regularly queries the conductor
        for an exposure to do.  Sleeps 120s if nothing was found,
        otherwise, runs an ExposureProcessor to process the exposure.

        Parameters
        ----------
        max_n_exposures : int, default None
           Succesfully process at most this many exposures before
           exiting.  Primarily useful for testing.  If None, there is no
           limit.  If you set this, you probably also want to set
           die_on_exception.  (Otherwise, if it fails each time it tries
           an exposure, it will never exit.)

        die_on_exception : bool, default False
           The exposure processing loop is run inside a try block that
           catches exceptions.  Normally, a log message is printed about
           the exception and the loop continues.  If this is true, the
           exception is re-raised.

        """

        done = False
        req = None
        n_processed = 0
        while not done:
            try:
                data = self.conductor.send( f'requestexposure/cluster_id={self.cluster_id}' )

                if data['status'] == 'not available':
                    SCLogger.info( f'No exposures available, sleeping {self.sleeptime} s' )
                    self.send_heartbeat()
                    time.sleep( self.sleeptime )
                    continue

                if data['status'] != 'available':
                    raise ValueError( f"Unexpected value of data['status']: {data['status']}" )

                with SmartSession() as session:
                    knownexp = ( session.query( KnownExposure )
                                 .filter( KnownExposure._id==data['knownexposure_id'] ) ).all()
                    if len( knownexp ) == 0:
                        raise RuntimeError( f"The conductor gave me KnownExposure id {data['knownexposure_id']}, "
                                            f"but I can't find it in the knownexposures table" )
                    if len( knownexp ) > 1:
                        raise RuntimeError( f"More than one KnownExposure with id {data['knownexposure_id']}; "
                                            f"you should never see this error." )
                knownexp = knownexp[0]

                exposure_processor = ExposureProcessor( knownexp.instrument,
                                                        knownexp.identifier,
                                                        knownexp.params,
                                                        self.numprocs,
                                                        onlychips=self.onlychips,
                                                        through_step=self.through_step,
                                                        worker_log_level=self.worker_log_level )
                SCLogger.info( f'Downloading and loading exposure {knownexp.identifier}...' )
                exposure_processor.download_and_load_exposure()
                SCLogger.info( f'...downloaded.  Launching process to handle all chips.' )

                with SmartSession() as session:
                    knownexp = ( session.query( KnownExposure )
                                 .filter( KnownExposure._id==data['knownexposure_id'] ) ).first()
                    knownexp.exposure_id = exposure_processor.exposure.id
                    session.commit()

                exposure_processor()
                SCLogger.info( f"Done processing exposure {exposure_processor.exposure.origin_identifier}" )

                n_processed += 1
                if ( max_n_exposures is not None ) and ( n_processed >= max_n_exposures ):
                    SCLogger.info( f"Hit max {n_processed} exposures, existing" )
                    done = True

            except Exception as ex:
                if die_on_exception:
                    raise
                else:
                    SCLogger.exception( "Exception in ExposureLauncher loop" )
                    SCLogger.info( f"Sleeping {self.sleeptime} s and continuing" )
                    time.sleep( self.sleeptime )

# ======================================================================

class ArgFormatter( argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter ):
    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )

def main():
    parser = argparse.ArgumentParser( 'pipeline_exposure_launcher.py',
                                      description='Ask the conductor for exposures to do, launch piplines to run them',
                                      formatter_class=ArgFormatter,
                                      epilog=

        """pipeline_exposure_launcher.py

Runs a process that regularly (by default, every 2 minutes) polls the
SeeChange conductor to see if there are any exposures that need
processing.  If given one by the conductor, will download the exposure
and load it into the database, and then launch multiple processes with
pipelines to process each of the chips in the exposure.
"""
                                      )
    parser.add_argument( "-c", "--cluster-id", required=True, help="Name of the cluster where this is running" )
    parser.add_argument( "-n", "--node-id", default=None,
                         help="Name of the node (if applicable) where this is running" )
    parser.add_argument( "--numprocs", default=None, type=int,
                         help="Number of worker processes to run at once.  (Default: # of CPUS - 1.)" )
    parser.add_argument( "--noverify", default=False, action='store_true',
                         help="Don't verify the conductor's SSL certificate" )
    parser.add_argument( "-l", "--log-level", default="info",
                         help="Log level for the main process (error, warning, info, or debug)" )
    parser.add_argument( "-w", "--worker-log-level", default="warning",
                         help="Log level for worker processes (error, warning, info, or debug)" )
    parser.add_argument( "--chips", default=None, nargs="+",
                         help="Only do these sensor sections (for debugging purposese)" )
    parser.add_argument( "-t", "--through-step", default=None,
                         help=( "Only run through this step; default=run everything.  Step can be "
                                "exposure, preprocessing, backgrounding, extraction, wcs, zp, "
                                "subtraction, detection, cutting, measuring" ) )
    args = parser.parse_args()

    loglookup = { 'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG }
    if args.log_level.lower() not in loglookup.keys():
        raise ValueError( f"Unknown log level {args.log_level}" )
    SCLogger.setLevel( loglookup[ args.log_level.lower() ] )
    if args.worker_log_level.lower() not in loglookup.keys():
        raise ValueError( f"Unknown worker log level {args.worker_log_level}" )
    worker_log_level = loglookup[ args.worker_log_level.lower() ]

    elaunch = ExposureLauncher( args.cluster_id, args.node_id, numprocs=args.numprocs, onlychips=args.chips,
                                verify=not args.noverify, through_step=args.through_step,
                                worker_log_level=worker_log_level )
    elaunch.register_worker()
    try:
        elaunch()
    finally:
        elaunch.unregister_worker()

# ======================================================================

if __name__ == "__main__":
    main()
