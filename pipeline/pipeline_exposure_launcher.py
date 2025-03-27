import sys
import time
import psutil
import logging
import argparse
import signal

from util.conductor_connector import ConductorConnector
from util.logger import SCLogger

from models.base import SmartSession
from models.knownexposure import KnownExposure

from pipeline.exposure_processor import ExposureProcessor
from pipeline.top_level import Pipeline


class ExposureLauncher:
    """A class that polls the conductor asking for things to do, launching a pipeline when one is found.

    Instantiate it with cluster_id, node_id, and numprocs, and then call the instance as a function.

    """

    def __init__( self, cluster_id, node_id, numprocs=None, verify=True, onlychips=None,
                  through_step=None, max_run_time=None, worker_log_level=logging.WARNING ):
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
          database.  The conductor may also give us a through step.  The pipeline
          will run through the *earlier* of this parameter, or the parameter
          passed by the conductor, if both are present.

        max_run_time : float, default None
          Normally, when you call an ExposureLauncher it will loop
          forever, asking the conductor for things to do.  If you set
          this, it will run at most this many seconds before exiting.
          Before asking the conductor for something to do, it will check
          to see that it hasn't been running this long, and if it has,
          it will exit.

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
        self.max_run_time = max_run_time
        self.worker_log_level = worker_log_level
        self.verify = verify
        self.conductor = ConductorConnector( verify=verify )

    def register_worker( self, replace=False ):
        url = ( f'conductor/registerworker/cluster_id={self.cluster_id}/'
                f'node_id={self.node_id}/nexps=1/replace={int(replace)}' )
        data = self.conductor.send( url )
        self.pipelineworker_id = data['id']

    def unregister_worker( self ):
        url = f'conductor/unregisterworker/{str(self.pipelineworker_id)}'
        try:
            data = self.conductor.send( url )
            if data['status'] != 'worker deleted':
                SCLogger.error( "Surprising response from conductor unregistering worker: {data}" )
        except Exception:
            SCLogger.exception( f"Exception unregistering worker {self.pipelineworker_id}, continuing" )

    def send_heartbeat( self ):
        url = f'conductor/workerheartbeat/{self.pipelineworker_id}'
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

        start_time = time.perf_counter()
        done = False
        n_processed = 0
        while not done:
            try:
                run_time = time.perf_counter() - start_time
                if ( self.max_run_time is not None ) and ( run_time > self.max_run_time ):
                    SCLogger.info( f"ExposureLauncher has been running for {run_time:.0f} seconds, returning." )
                    done = True
                    continue

                self.send_heartbeat()
                data = self.conductor.send( f'conductor/requestexposure/cluster_id={self.cluster_id}' )

                if data['status'] == 'not available':
                    SCLogger.info( f'No exposures available, sleeping {self.sleeptime} s' )
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

                # Figure out what step we're supposed to run through
                through_step = None
                conddex = None
                mydex = None
                if self.through_step is not None:
                    try:
                        mydex = Pipeline.ALL_STEPS.index( self.through_step )
                    except ValueError:
                        SCLogger.error( f"Unknown step {self.through_step}" )
                        raise
                if 'through_step' in data:
                    try:
                        conddex = Pipeline.ALL_STEPS.index( data['through_step'] )
                    except ValueError:
                        SCLogger.error( f"Unknown step {data['through_step']} given by conductor!" )
                        raise
                if conddex is not None:
                    if mydex is not None:
                        through_step = Pipeline.ALL_STEPS[ min( conddex, mydex ) ]
                    else:
                        through_step = data['through_step']
                elif mydex is not None:
                    through_step = self.through_step

                # Run
                exposure_processor = ExposureProcessor( knownexp.instrument,
                                                        knownexp.identifier,
                                                        knownexp.params,
                                                        self.numprocs,
                                                        self.cluster_id,
                                                        self.node_id,
                                                        onlychips=self.onlychips,
                                                        through_step=through_step,
                                                        verify=self.verify,
                                                        worker_log_level=self.worker_log_level )
                exposure_processor.start_work()
                SCLogger.info( f'Downloading and loading exposure {knownexp.identifier}...' )
                exposure_processor.download_and_load_exposure()
                SCLogger.info( '...downloaded.  Launching process to handle all chips.' )

                with SmartSession() as session:
                    knownexp = ( session.query( KnownExposure )
                                 .filter( KnownExposure._id==data['knownexposure_id'] ) ).first()
                    knownexp.exposure_id = exposure_processor.exposure.id
                    session.commit()

                exposure_processor()
                exposure_processor.finish_work()
                SCLogger.info( f"Done processing exposure {exposure_processor.exposure.origin_identifier}" )

                n_processed += 1
                if ( max_n_exposures is not None ) and ( n_processed >= max_n_exposures ):
                    SCLogger.info( f"Hit max {n_processed} exposures, returning." )
                    done = True

            except Exception:
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
    parser.add_argument( "-m", "--max-run-time", default=None, type=float,
                         help=( "Maximum time to run before exiting.  If this is on a job that will get cancelled "
                                "(e.g. one launched on a slurm queue), make sure this is less than the runtime "
                                "of the job by an amount conservatively equal to what you'd need to process a "
                                "single exposure." ) )
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
                                "exposure, preprocessing, extraction, astrocal, photocal, "
                                "subtraction, detection, cutting, measuring, scoring.  Will run "
                                "through the earlier of this step or the through step given by the conductor." ) )
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
                                max_run_time=args.max_run_time, worker_log_level=worker_log_level )
    elaunch.register_worker()

    def goodbye( signum, frame ):
        SCLogger.warning( "Got INT/TERM signal, unregistering worker and exiting." )
        sys.exit()

    signal.signal( signal.SIGINT, goodbye )
    signal.signal( signal.SIGTERM, goodbye )

    try:
        elaunch()
    finally:
        elaunch.unregister_worker()


# ======================================================================
if __name__ == "__main__":
    main()
