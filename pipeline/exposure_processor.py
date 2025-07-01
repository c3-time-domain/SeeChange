import sys
import argparse
import re
import datetime
import multiprocessing
import logging
import psutil

import psycopg2
import psycopg2.extras

from util.logger import SCLogger
from util.util import asUUID
from util.config import Config

from models.base import Psycopg2Connection
from models.enums_and_bitflags import KnownExposureStateConverter
from models.instrument import get_instrument_instance
from models.exposure import Exposure

# Importing this because otherwise when I try to do something completly
# unrelated to Object or Measurements, sqlalchemy starts objecting about
# relationships between those two that aren't defined.
# import models.object

# Gotta import the instruments we might use before instrument fills up
# its cache of known instrument instances
import models.decam  # noqa: F401

from pipeline.data_store import DataStore
from pipeline.top_level import Pipeline
from pipeline.configchooser import ConfigChooser


class ExposureProcessor:
    def __init__( self, instrument, identifier, numprocs, cluster_id,
                  node_id, machine_name=None, onlychips=None,
                  through_step=None, worker_log_level=logging.WARNING ):
        """A class that processes all images in a single exposure, potentially using multiprocessing.

        It's sort of a wrapper around top_level.py::Pipeline, but
        handles downloading exposures, some updates of the
        knownexposures table, and running lots of processes each
        of which run a single Pipeline (on one chip).

        This is used internally by ExposureLauncher; normally, you would not use it directly.

        Parameters
        ----------
        instrument : str
          The name of the instrument

        identifier : str
          The identifier of the exposure (as defined in the KnownExposures model)

        numprocs: int
          Number of worker processes (not including the manager process)
          to run at once.  0 or 1 = do all work in the main manager process.

        cluster_id: str

        node_id: str

        machine_name: str or None

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
        self.cluster_id = cluster_id
        self.node_id = node_id
        self.machine_name = machine_name
        self.numprocs = numprocs
        self.onlychips = onlychips
        self.through_step = through_step
        self.worker_log_level = worker_log_level
        self.provtag = Config.get().value( 'pipeline.provenance_tag' )

    def cleanup( self ):
        """Do our best to free memory."""

        self.exposure = None   # Praying to the garbage collection gods


    def finish_work( self ):
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "LOCK TABLE knownexposures" )
            cursor.execute( "SELECT cluster_id, start_time FROM knownexposures "
                            "WHERE instrument=%(inst)s AND identifier=%(ident)s",
                            { 'inst': self.instrument.name, 'ident': self.identifier } )
            rows = cursor.fetchall()
            if len(rows) == 0:
                raise ValueError ( f"Error, unknown known exposure with instrument {self.instrument} "
                                   f"and identifier {self.identifier}" )
            if len(rows) > 1:
                raise RuntimeError ( f"Error, multiple known exposures with instrument {self.instrument} "
                                     f"and identifier {self.identifier}; this should not happen" )
            if rows[0][0] != self.cluster_id:
                raise ValueError( f"Error, known exposure with instrument {self.instrument.name} and identifier "
                                  f"{self.identifier} is claimed by cluster {rows[0][0]}, not {self.cluster_id}" )
            if rows[0][1] is None:
                raise ValueError( f"Error, known exposure with instrument {self.instrument.name} and identifier "
                                  f"{self.identifier} has a null start time" )

            cursor.execute( "UPDATE knownexposures SET release_time=%(t)s, _state=%(state)s "
                            "WHERE instrument=%(inst)s AND identifier=%(ident)s",
                            { 'inst': self.instrument.name, 'ident': self.identifier,
                              't': datetime.datetime.now( tz=datetime.UTC ),
                              'state': KnownExposureStateConverter.to_int('done') } )
            conn.commit()


    def secure_exposure( self, assume_claimed=False, cont=True, delete=False ):
        """Make sure the exposure is downloaded and in the database and archive.

        Also updates the knownexposures table to indicate that this
        exposure is running on this cluster, node, and machine.

        If it's not yet in the database, use the Instrument subclass to download and load it.

        If it is in the database, raise an exception unless cont is True.

        Parmaeters
        ----------
          assume_claimed : bool, default False
             If True, assume that this process has claimed the exposure.
             Totally ignore the state field of the knownexposures table
             in the database.  This is scary, but, sometimes things
             crash, so we need this abilty.  If assume_claimed is False,
             and the knownexposures state is "running", raise an
             exception.  If assume_claimed is False and the
             knownexposures state is "claimed", raise an exception
             unless the cluster_id, node_id, and machine_name in the
             knownexposures table all match this object's values.

          cont : bool, default False
             If this is False, and an exposure already exists in the
             database, raise an exception.  If this is True, then don't
             raise an exception, make sure that the known exposure table
             has the right exposure id, and proceed, trusting the
             pipeline to pick up where it left off.

          delete : bool, default False
             OMG, don't use this.  Unless you're testing and developing,
             and really know what you're doing.  If the exposure already
             exists in the database, *blow it away* (and any derived
             data products) and start it over.  The design of the
             pipeline is such that this isn't supposed to happen, but,
             well, development has needs.

        """

        SCLogger.info( f"Securing exposure {self.identifier}..." )
        exposureid = None
        exposure_to_delete = None
        ke = None
        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( "LOCK TABLE knownexposures" )
            cursor.execute( "SELECT * FROM knownexposures WHERE instrument=%(inst)s AND identifier=%(iden)s",
                            { 'inst': self.instrument.name, 'iden': self.identifier } )
            rows = cursor.fetchall()
            if len(rows) == 0:
                raise ValueError( f"Unknown known exposure with instrument {self.instrument.name} "
                                  f"and identifier {self.identifier}" )
            if len(rows) > 1:
                raise ValueError( f"Multiple known exposures with instrument {self.instrument.name} "
                                  f"and identifier {self.identifier}" )
            ke = rows[0]
            state = KnownExposureStateConverter.to_string( ke['_state'] )

            claim_time = ke['claim_time'] if ke['claim_time'] is not None else datetime.datetime.now( tz=datetime.UTC )
            if any( [ ( ke['cluster_id'] != self.cluster_id ),
                      ( ke['node_id'] != self.node_id ),
                      ( ke['machine_name'] != self.machine_name ) ] ):
                claim_time = datetime.datetime.now( tz=datetime.UTC )
                if ( not assume_claimed ) and ( state not in ( 'ready', 'held', 'done' ) ):
                    raise RuntimeError( f"Exposure is in state {state}, isn't claimed by me, "
                                        f"and assume_claimed is False." )
            if state == 'running':
                SCLogger.warning( "Running an exposure even though it's state is \"running\"! "
                                  "Trusting that you know what you're doing." )

            # Check to see if the exposure already exists
            cursor.execute( "SELECT _id FROM exposures WHERE instrument=%(inst)s AND origin_identifier=%(iden)s",
                            { 'inst': self.instrument.name, 'iden': self.identifier } )
            exps = cursor.fetchall()
            if len(exps) > 1:
                raise RuntimeError( f"Database corruption, multiple exposures with instrument {self.instrument.name} "
                                    f"and identifier {self.identifier}" )
            if len(exps) == 1:
                exposureid = asUUID( exps[0]['_id'] )

            if exposureid is not None:
                if delete:
                    SCLogger.warning( f"There's already an exposure associated with instrument {self.instrument.name} "
                                      f"and identifier {self.identifier}.  delete is True, so, deleting it... scary " )
                    exposure_to_delete = exposureid
                    exposureid = None
                elif cont:
                    SCLogger.info( f"There's already an exposure associated with instrument {self.instrument.name} "
                                      f"and identifier {self.identifier}.  cont is True, so running "
                                      f"the pipeline allow it to pick the exposure up partway through." )
                else:
                    raise ValueError( f"There's already an exposure associated with instrument {self.instrument.name} "
                                      f"and identifier {self.identifier}, but cont is False" )

            q = ( "UPDATE knownexposures SET cluster_id=%(clust)s, node_id=%(node)s, machine_name=%(mach)s, "
                  "claim_time=%(t)s, start_time=NULL, release_time=NULL, _state=%(state)s, exposure_id=%(expid)s "
                  "WHERE instrument=%(inst)s AND identifier=%(iden)s" )
            cursor.execute( q, { 'inst': self.instrument.name,
                                 'iden': self.identifier,
                                 'clust': self.cluster_id,
                                 'node': self.node_id,
                                 'mach': self.machine_name,
                                 't': claim_time,
                                 'state': KnownExposureStateConverter.to_int( 'running' ),
                                 'expid': None if exposureid is None else str(exposureid)
                                } )
            conn.commit()

        # If we're supposed to delete an exposure, do that...
        if exposure_to_delete is not None:
            SCLogger.warning( f"Deleting exposure from disk and database! : {exposure_to_delete}" )
            exposure = Exposure.get_by_id( exposure_to_delete )
            exposure.delete_from_disk_and_database()
            SCLogger.warning ( "...done deleting exposure from disk and database." )

        # If there wasn't an existing exposure, we have to download it:
        if exposureid is None:
            params = {} if ke['params'] is None else ke['params']
            SCLogger.info( f"Downloading exposure {self.identifier}..." )
            self.exposure = self.instrument.acquire_and_commit_origin_exposure( self.identifier, params )
            SCLogger.info( "...downloaded." )
            with Psycopg2Connection() as conn:
                cursor = conn.cursor()
                cursor.execute( "UPDATE knownexposures SET exposure_id=%(expid)s "
                                "WHERE instrument=%(inst)s AND identifier=%(iden)s",
                                { "inst": self.instrument.name, "iden": self.identifier,
                                  "expid": str(self.exposure.id) } )
                conn.commit()
        else:
            self.exposure = Exposure.get_by_id( exposureid )
            if self.exposure is None:
                raise ValueError( f"Unknown exposure {exposureid}.  This should never happen." )


        SCLogger.info( "...secured." )
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
            ds = DataStore.from_args( self.exposure, chip )
            ds.cluster_id = self.cluster_id
            ds.node_id = self.node_id
            ds = pipeline.run( ds )
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

        origconfig = Config._default
        try:
            with Psycopg2Connection() as conn:
                cursor = conn.cursor()
                cursor.execute( "UPDATE knownexposures SET start_time=%(t)s "
                                "WHERE instrument=%(inst)s AND identifier=%(iden)s",
                                { 'inst': self.instrument.name, 'iden': self.identifier,
                                  't': datetime.datetime.now( tz=datetime.UTC ) } )
                conn.commit()

            # Update the config if necessary.  This changes the global
            #   config cache, which is scary, because we're changing it
            #   based on the needs of the current exposure.  So, in
            #   the finally block below, we try to restore the original
            #   config.
            config_chooser = ConfigChooser()
            config_chooser.run( self.exposure )

            chips = self.instrument.get_section_ids()
            if self.onlychips is not None:
                chips = [ c for c in chips if c in self.onlychips ]
            self.results = {}

            if self.numprocs > 1:
                SCLogger.info( f"Creating pool of {self.numprocs} processes to do {len(chips)} chips" )
                with multiprocessing.Pool( self.numprocs, maxtasksperchild=1 ) as pool:
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

            self.finish_work()
        finally:
            # Restore the global config we had before we ran this
            Config.init( origconfig, setdefault=True )


# ======================================================================

def main():
    sys.stderr.write( f"exposure_processor starting at {datetime.datetime.now(tz=datetime.UTC).isoformat()}\n" )

    parser = argparse.ArgumentParser( 'exposure_processor', 'Process a single known exposure',
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      epilog="""Process a single known exposure.

This will process an exposure through the step specified by
--through-step (or all steps, if --through-step isn't given).  It will
pick up exposures that have already been started partway, and do the
rest of the processing on them.  The exposure must be in the
knownexposures table.

This is what you use to manually force the pipeline to run a specific
exposure.  If you want to bulk process exposures specified by the
conductor, instead use pipeline_exposure_launcher.

TODO: provide a mechanism to run an exposure that's *not* in the
knownexpsoures table.

Warning: be careful.  exposure_processor does not contact the conductor,
nor does it verify that another process isn't already working on this
exposure.  It will just update the state in the knownexposures table
without regard to what was there before.  If two processes work on the
same exposure at the same time, you may make a tremendous mess.  To be
safe, make sure that this exposure is in a state other than "ready"
(i.e. one of "held", "running", or "done") in the conductor.  If the
current state is "running", make sure that it's not really running
(i.e. the process that said it was running crashed).  If the current
state is "claimed", be *very* careful, as another process might be about
to start it.

"""
                                     )
    parser.add_argument( 'instrument', help='Name of the instrument of the known exposure' )
    parser.add_argument( 'identifier', help='Identifier of the known exposure' )
    parser.add_argument( '-c', '--cluster-id', default="manual",
                         help="Cluster ID to mark as claiming the exposure in the knownexposures table" )
    parser.add_argument( '--node', '--node-id', default="manual",
                         help="Node ID to mark as claiming the exposure in the knownexposures table" )
    parser.add_argument( '-m', '--machine', default="manual",
                         help="Machine name to mark as claiming the exposure in the knownexposures table" )
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

    reallydelete = args.delete and args.really_delete
    numprocs = args.numprocs if args.numprocs is not None else ( psutil.cpu_count( logical=False ) -1  )
    SCLogger.info( f"Running with {numprocs} chip processors" )

    processor = ExposureProcessor( args.instrument, args.identifier, numprocs,
                                   args.cluster_id, args.node, machine_name=args.machine,
                                   onlychips=args.chips, through_step=args.through_step,
                                   worker_log_level=loglookup[args.worker_log_level.lower()] )

    processor.secure_exposure( assume_claimed=args.assume_claimed, cont=args.cont, delete=reallydelete )
    processor()

    SCLogger.info( "All done" )


# ======================================================================
if __name__ == "__main__":
    main()
