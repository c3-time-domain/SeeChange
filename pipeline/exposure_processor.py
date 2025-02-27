import sys
import argparse
import re
import datetime
import multiprocessing
import multiprocessing.pool
import logging
import psutil

import psycopg2
import psycopg2.extras

from util.logger import SCLogger
from util.util import asUUID

from models.base import Psycopg2Connection
from models.instrument import get_instrument_instance
from models.exposure import Exposure

# Importing this because otherwise when I try to do something completly
# unrelated to Object or Measurements, sqlalchemy starts objecting about
# relationships between those two that aren't defined.
# import models.object

# Gotta import the instruments we might use before instrument fills up
# its cache of known instrument instances
import models.decam  # noqa: F401

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

        SCLogger.info( f"Securing exposure {self.identifier}..." )
        self.exposure = self.instrument.acquire_and_commit_origin_exposure( self.identifier, self.params )
        SCLogger.info( "...secured." )
        # TODO : this Exposure object is going to be copied into every processor subprocess
        #   *Ideally* no data was loaded, only headers, so the amount of memory used is
        #   not significant, but we should investigate/verify this, and deal with it if
        #   that is not the case.

    def set_existing_exposure( self, exposure_id ):
        self.exposure = Exposure.get_by_id( exposure_id )
        if self.exposure is None:
            raise ValueError( f"Unknown exposure {exposure_id}" )


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
            ds = pipeline.run( self.exposure, chip )
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
        self.results[ chip ] = succ

    def __call__( self ):
        """Run all the pipelines for the chips in the exposure."""

        if self.through_step == 'exposure':
            SCLogger.info( "Only running through exposure, not launching any image processes" )
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

                SCLogger.info( "Submitted all worker jobs, waiting for them to finish." )
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


# ======================================================================

def main():
    sys.stderr.write( f"exposure_processor starting at {datetime.datetime.now(tz=datetime.UTC).isoformat()}\n" )

    parser = argparse.ArgumentParser( 'exposure_processor', 'process a single known exposure' )
    parser.add_argument( 'instrument', help='Name of the instrument of the known exposure' )
    parser.add_argument( 'identifier', help='Identifier of the known exposure' )
    parser.add_argument( '-c', '--cluster-id', default="manual",
                         help="Cluster ID to mark as claiming the exposure in the knownexposures table" )
    parser.add_argument( '-n', '--numprocs', default=None, type=int,
                         help=( "Number of chip processors to run (defaults to number of physical "
                                "system CPUs minus 1" ) )
    parser.add_argument( '-t', '--through-step', default=None, help="Process through this step" )
    parser.add_argument( '--chips', default=None, nargs='+', help="Only do these sensor sections (defaults to all)" )
    parser.add_argument( '--cont', '--continue', default=False, action='store_true',
                         help="If exposure already exists, try continuing it." )
    parser.add_argument( '-d', '--delete', default=False, action='store_true',
                         help="Delete exposure from disk and database before starting if it exists." )
    parser.add_argument( '--really-delete', default=False, action='store_true',
                         help="Must be specified if -d or --delete is specified for it to do its dirty work." )
    parser.add_argument( '-l', '--log-level', default='info',
                         help="Log level (error, warning, info, or debug) (defaults to info)" )
    parser.add_argument( '-w', '--worker-log-level', default='warning',
                         help="Log level for the chip worker subprocesses (defaults to warning)" )
    parser.add_argument( '--assume-claimed', default=False, action='store_true',
                         help=( "Normally, will object if somebody else has claimed this exposure. Set "
                                "this flag to True to ignore claims in the knownexposures table." ) )

    args = parser.parse_args()

    loglookup = { 'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG }
    if args.log_level.lower() not in loglookup.keys():
        raise ValueError( f"Unknown log level {args.log_level}" )
    SCLogger.setLevel( loglookup[ args.log_level.lower() ] )

    # Try to get the known exposure

    instrument = None
    identifier = None
    exposureid = None
    exposure_to_delete = None
    with Psycopg2Connection() as conn:
        cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
        cursor.execute( "LOCK TABLE knownexposures" )
        cursor.execute( "SELECT * FROM knownexposures WHERE instrument=%(inst)s AND identifier=%(iden)s",
                        { 'inst': args.instrument, 'iden': args.identifier } )
        rows = cursor.fetchall()
        if len(rows) == 0:
            raise ValueError( f"Unknown known exposure with instrument {args.instrument} "
                              f"and identifier {args.identifier}" )
        if len(rows) > 1:
            raise ValueError( f"Multiple known exposures with instrument {args.instrument} "
                              f"and identifier {args.identifier}" )
        row = rows[0]

        if ( not args.assume_claimed) and ( ( row['cluster_id'] is not None ) or ( row['claim_time'] is not None ) ):
            raise RuntimeError( f"Known exposure with instrument {args.instrument} "
                                f"and identifier {args.identifier} is claimed by "
                                f"{row['cluster_id']} at {row['claim_time']}" )
        instrument = row['instrument']
        identifier = row['identifier']
        params = {} if row['params'] is None else row['params']

        # Check to see if the exposure already exists
        cursor.execute( "SELECT _id FROM exposures WHERE instrument=%(inst)s AND origin_identifier=%(iden)s",
                        { 'inst': args.instrument, 'iden': args.identifier } )
        rows = cursor.fetchall()
        if len(rows) > 1:
            raise RuntimeError( f"Database corruption, multiple exposures with instrument {args.instrument} "
                                f"and identifier {args.identifier}" )
        if len(rows) == 1:
            exposureid = asUUID( rows[0]['_id'] )

        if exposureid is not None:
            if args.delete and args.really_delete:
                SCLogger.warning( f"There's already an exposure associated with instrument {args.instrument} "
                                  f"and identifier {args.identifier}.  You specified --delete and --really-delete, "
                                  f"so we will try to delete it...." )
                exposure_to_delete = exposureid
                exposureid = None
            elif args.cont:
                SCLogger.warning( f"There's already an exposure associated with instrument {args.instrument} "
                                  f"and identifier {args.identifier}.  You specified --continue, so running "
                                  f"the pipeline in hopes that picking up partway through really works." )
            else:
                raise ValueError( f"There's already an exposure associated with instrument {args.instrument} "
                                  f"and identifier {args.identifier}, but you specified neither "
                                  f"--continue nor --delete." )

        if not args.assume_claimed:
            cursor.execute( "UPDATE knownexposures SET cluster_id=%(clust)s, claim_time=%(t)s, exposure_id=%(exp)s "
                            "WHERE instrument=%(inst)s AND identifier=%(iden)s",
                            { 'inst': args.instrument,
                              'iden': args.identifier,
                              'clust': args.cluster_id,
                              't': datetime.datetime.now( tz=datetime.UTC ),
                              'exp': str(exposureid)
                             } )
        conn.commit()

    # If we got this far, we have a known exposure to process, so process it

    # Blow away existing exposure if user was foolish enough to request that
    if exposure_to_delete is not None:
        SCLogger.warning( f"Deleting exposure {exposure_to_delete} from disk and database!..." )
        exposure = Exposure.get_by_id( exposure_to_delete )
        exposure.delete_from_disk_and_database()
        SCLogger.warning( "...done deleting exposure from disk and database." )

    # Process the exposure
    numprocs = args.numprocs if args.numprocs is not None else ( psutil.cpu_count( logical=False ) -1  )
    SCLogger.info( f"Running with {numprocs} chip processors" )

    processor = ExposureProcessor( instrument, identifier, params, numprocs,
                                   onlychips=args.chips,
                                   through_step=args.through_step,
                                   worker_log_level=loglookup[args.worker_log_level.lower()] )
    if exposureid is None:
        processor.download_and_load_exposure()
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "UPDATE knownexposures SET exposure_id=%(expid)s "
                            "WHERE instrument=%(inst)s AND identifier=%(iden)s",
                            { 'inst': args.instrument,
                              'iden': args.identifier,
                              'expid': str(processor.exposure.id) }
                           )
            conn.commit()
    else:
        processor.set_existing_exposure( exposureid )

    processor()


# ======================================================================
if __name__ == "__main__":
    main()
