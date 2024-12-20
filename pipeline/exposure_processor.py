import re
import multiprocessing
import multiprocessing.pool
import logging

from util.logger import SCLogger

from models.instrument import get_instrument_instance

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
