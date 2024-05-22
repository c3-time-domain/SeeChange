import logging
from mpi4py import MPI

from util.logger import SCLogger

class Runner:
    def __init__( self, comm, controller_callback, worker_callback, dieonerror=False ):
        """A framework to run a series of tasks in parallel.

        Of the MPI ranks, the first one is a controller rank, and just
        sends out jobs to the other ranks.  The other ranks actually do
        the work.

        comm : MPI communicator, or None of if you're using this entirely gratuitously
        worker_callback : Callback called once for each element of datalist passed to go.  This one is
                          run in parallel, and should have the majority of the meat of the work.
        controller_callback : Callback called for each return value of worker_callback.  This one is run
                              in the controller thread every time, so it should be lightweight.
        dieonerror : If True, and any process throws an exception or otherwise has an error, throw a RuntimeError
                     otherwise, just move on.  If True, you probably want to run with python -m mpi4py

        """
        self.comm = comm
        if self.comm is None:
            self.size = 1
            self.rank = 0
        else:
            self.size = comm.Get_size()
            self.rank = comm.Get_rank()
        self._controller_callback = controller_callback
        self._worker_callback = worker_callback
        self.dieonerror = dieonerror
        if self.size == 2 and self.rank == 0:
            SCLogger.warning( 'Runner won\'t give any improvement over single-threaded for 2 processes.' )

    def go( self, datalist ):
        """Run the processes for a list of data

        datalist : a list of objects that will be passed to the worker callback

        Returns an array of booleans corresponding to datalist.  Where
        True, Runner thinks the process worked.

        """
        if self.rank == 0:
            if self.size == 1:
                success = self._justdoit( datalist )
            else:
                success = self._controller( datalist )
        else:
            self._worker()
            success = []
        if self.size > 1:
            success = self.comm.bcast( success, root=0 )
        return success

    def _justdoit( self, datalist ):
        success = [ False ] * len(datalist)
        for i, data in enumerate(datalist):
            try:
                SCLogger.info( f'Runner running task {i}' )
                result = self._worker_callback( data )
                self._controller_callback( result )
                success[i] = True
            except Exception as e:
                if self.dieonerror:
                    raise(e)
                else:
                    SCLogger.exception( f'Error running task {i}; results are missing or incomplete!' )
        return success

    def _controller( self, datalist ):
        workers = [ False ] * self.size
        success = [ False ] * len(datalist)
        avail = []
        ncheckedin = 0
        sentdex = 0
        ndone = 0

        # Wait for all workers to check in
        status = MPI.Status()
        while ncheckedin < self.size-1:
            SCLogger.debug( f'Waiting for a worker to check in' )
            msg = self.comm.recv( source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status )
            rank = status.Get_source()
            SCLogger.debug( f'Rank {rank} just checked in' )

            if msg["message"] == "Hello":
                if workers[rank]:
                    raise RuntimeError( f'Rank {rank} checked in more than once!' )
                workers[rank] = True
                avail.append( rank )
                ncheckedin += 1
            else:
                raise RuntimeError( f'Got message {msg["message"]} while trying to initialize' )

        # Run the things
        while ndone < len(datalist):
            # Send out jobs to all available workers
            while ( sentdex < len(datalist) ) and ( len( avail ) > 0 ):
                rank = avail.pop()
                msg = { "message": "do", "dex": sentdex, "data": datalist[sentdex] }
                SCLogger.info( f'Sending job {sentdex} of {len(datalist)} to rank {rank}' )
                self.comm.send( msg, dest=rank )
                sentdex += 1

            # Wait for responses
            SCLogger.debug( f'Waiting for responses from workers' )
            msg = self.comm.recv( source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status )
            rank = status.Get_source()

            if msg["message"] == "Finished":
                SCLogger.debug( f'Got complete from rank {rank}' )
                self._controller_callback( msg["data"] )
                success[ msg["dex"] ] = True
                ndone += 1
                avail.append( rank )
            elif msg["message"] == "Error":
                if self.dieonerror:
                    raise RuntimeError( f'Error from rank {rank}: {msg["error"]}' )
                else:
                    SCLogger.error( f'ERROR: rank {rank} returned error {msg["error"]}. '
                                       'Task is incomplete or only partially complete!' )
                    ndone += 1
                    avail.append( rank )
            else:
                raise RuntimeError( f'Unexpected message {msg["message"]}' )
            SCLogger.info( f'{ndone} of {len(datalist)} jobs finished' )

        # Tell everybody to die
        SCLogger.info( f'All jobs complete, closing down' )
        for rank in range(1, self.size):
            SCLogger.debug( f'Sending "die" to rank {rank}' )
            self.comm.send( { "message": "die" }, dest=rank )
        SCLogger.debug( f'Done telling workers to die.' )

        return success

    def _worker( self ):
        # Check in with home base
        SCLogger.debug( f'Checking in with controller' )
        self.comm.send( { "message": "Hello" }, dest=0 )

        # Wait to be commanded
        while True:
            msg = self.comm.recv( source=0, tag=MPI.ANY_TAG )
            if msg["message"] == "die":
                break
            elif msg["message"] == "do":
                result = None
                try:
                    SCLogger.debug( f'Got data: {msg["data"]}' )
                    result = self._worker_callback( msg["data"] )
                    SCLogger.debug( f'Worker callback got result: {result}' )
                except Exception as e:
                    SCLogger.exception( f'Rank {self.rank} got an exception' )
                    self.comm.send( { "message": "Error",
                                      "dex": msg["dex"],
                                      "error": str(e),
                                      "exception": e }, dest=0 )
                else:
                    self.comm.send( { "message": "Finished", "dex": msg["dex"], "data": result }, dest=0 )
            else:
                self.comm.send( { "message": "Error",
                                  "dex": msg["dex"],
                                  "error": f'Rank {self.rank} got unexpected command {msg["message"]}' } )
        SCLogger.debug( "Runner worker closing down." )
