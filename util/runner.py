import sys
import time
import multiprocessing
import logging

class Runner:
    """Python class for running a bunch of multprocessing tasks at once.

    Usage: Create a Runner object, passing it two functions.  The first
    function does the heavy lifting, the second function should be fast
    and is called at the end of the first function.  (See __init__ actual
    calling sequence and details.)  Then call the go() method of the
    Runner object.,

    """

    def __init__( self, numprocs, executor, aggregator=None, persistent_processes=True, logger=None,
                  end_sleeptime=0.1 ):
        """Initialize a runner object.

        Parameters
        ----------
        numprocs : int
            Number of worker processes.  Make this no more than the
            number of CPUs you have minus one (as there will also be a
            supervisor process running-- the process you call go()
            from).

        executor : function
            A function that takes one argument and returns one argument.
            This is the function that the worker processes run.  It will
            be called once for each element of the list passed to go().

        aggregator : function, optional, default None
            A function that takes two arguments.  This will be called
            after each worker process finishes running the function
            passed as executor.  The aggregator will receive two
            arguments; an int, the index into the list passed to
            Runner.go(), and the return value from the executor.  The
            aggregator is called from the supervisor process, which is
            the process that called Runner.go().  Use this to collect
            the results from the worker processes.  (See
            tests/util/test_runner.py for examples.)

        persistent_processes : bool, default True
            If True, then numprocs worker processes will be created when
            Runner.go() is called, and supervisor process will farm the
            work out to the worker processes.  If False, then a new
            process is created each time the supervisor has more to do.

            True has less overhead.  False should lead to cleanup (but,
            TODO, think about whether the Process.join calls are really
            in the right place for this).

        logger : logging.Logger, default <internal>

        end_sleeptime : float, default 0.1
            After all of the work has been submitted to worker
            processes, at the end the supervisor process will wait until
            it gets a response from all the worker processes.  It will
            loop through busy processes and see if the process is done.
            This is the time to sleep between loops.  Set it to 0. for
            no delay, which is fastest, but does mean that the
            supervisor process will be rapidly polling MPI-style.  (This
            is probably not a big deal.)

        """
        self.numprocs = numprocs
        self.executor = executor
        self.aggregator = aggregator
        self.persistproc = persistent_processes
        self.end_sleeptime = end_sleeptime

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger( "Runner" )
            logout = logging.StreamHandler( sys.stderr )
            self.logger.addHandler( logout )
            formatter = logging.Formatter( f'[%(asctime)s - %(levelname)s] - %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S' )
            logout.setFormatter( formatter )
            # self.logger.setLevel( logging.INFO )
            self.logger.setLevel( logging.DEBUG )

    def _subproc( self, pipe, logger ):
        me = multiprocessing.current_process()
        self.logger.info( "Process starting: {me.name} PID {me.pid}" )

        try:
            done = False
            while not done:
                command = pipe.recv()
                if command['command'] == 'die':
                    self.logger.debug( f"{me.name} : got die command" )
                    done = True

                elif command['command'] == 'do':
                    try:
                        if 'info' in command:
                            self.logger.info( f'{me.name} running: {command["info"]}' )
                        res = self.executor( command['arg'] )
                        pipe.send( { 'status': 'done', 'result': res } )
                    except Exception as ex:
                        s = f"{me.name} exception: {ex}"
                        self.logger.exception( s )
                        pipe.send( { 'status': 'error', 'error': s } )

                    if not self.persistproc:
                        done = True

        except Exception as ex:
            self.logger.exception( f"Process {me.name} PID {me.pid} returning after exception" )
            return

        self.logger.debug( f"Process {me.name} PID {me.pid} exiting " )
        return


    def _makeproc( self, name ):
        mypipe, theirpipe = multiprocessing.Pipe( True )
        proc = multiprocessing.Process( target=lambda: self._subproc( theirpipe, self.logger ), name=name )
        proc.start()
        self.allprocs.append( proc )
        return proc, mypipe

    def _poll_busy( self ):
        busydex = 0
        while busydex < len( self.busyprocs ):
            pipe = self.busypipes[ busydex ]
            if pipe.poll():
                proc = self.busyprocs[ busydex ]
                argdex = self.busyargdex[ busydex ]
                self.logger.debug( f"Getting response from {proc.name}" )
                rval = pipe.recv()
                fail = False
                if 'status' not in rval:
                    self.logger.error( f"Unexpected response from {proc.name}: {rval}" )
                    self.nfail += 1
                    fail = True
                elif rval['status'] == 'error':
                    self.logger.error( f"Error return from {proc.name}: {rval}" )
                    self.nfail += 1
                    fail = True
                else:
                    self.nsuccess += 1
                if self.aggregator is not None:
                    self.aggregator( argdex, None if fail else rval['result'] )
                if self.persistproc:
                    self.idleprocs.append( self.busyprocs[ busydex ] )
                    self.idlepipes.append( self.busypipes[ busydex ] )

                del self.busyprocs[ busydex ]
                del self.busypipes[ busydex ]
                del self.busyargdex[ busydex ]
            else:
                busydex += 1

    def go( self, args ):
        """Run worker processes.

        Parameters
        ----------
        args : list
            List of arguments to pass to executor.

        This process will create subprocesses, and in each subprocess
        run the function passed as the executor to the Runner
        constructor with one element of the args list.  When the
        subprocess finishes, this process will call the aggregator
        function (if any was provided to the Runner constructor) with
        two arguments: the index into args, and the return value from
        executor.

        It will run at most numprocs (passed to the Runner constructor)
        worker processes at once.  If persistent_processes was set to
        True, then it will create numprocs processes at the beginning,
        and farm out the work from args to those processes.  If
        persistent_processes was set to False, it will create a new
        process for each element of args, but will not have more than
        numprocs worker processes running at one time.

        """


        self.idleprocs = []
        self.idlepipes = []
        self.busyprocs = []
        self.busypipes = []
        self.busyargdex = []
        self.allprocs = []
        self.nargs = len(args)
        self.nsent = 0
        self.nsuccess = 0
        self.nfail = 0

        if self.persistproc:
            self.logger.info( f"Creating {self.numprocs} persistent worker processes"  )
            for i in range( self.numprocs ):
                proc, pipe = self._makeproc( f"proc {i}" )
                self.idleprocs.append( proc )
                self.idlepipes.append( pipe )

        self.logger.info( f"Beinning to work through {len(args)} jobs" )

        argdex = 0
        while argdex < len(args):

            # Launch jobs until we have max processes running

            while ( argdex < len(args) ) and ( len(self.busyprocs) < self.numprocs ):
                proc = None
                pipe = None
                if self.persistproc:
                    proc = self.idleprocs.pop()
                    pipe = self.idlepipes.pop()
                else:
                    proc, pipe = self._makeproc( f"proc {self.nsent}" )
                self.busyprocs.append( proc )
                self.busypipes.append( pipe )
                self.busyargdex.append( argdex )
                self.logger.info( f"Sending job {self.nsent} / {self.nargs} to {proc.name}" )
                pipe.send( { 'command': 'do', 'arg': args[argdex] } )
                argdex += 1
                self.nsent += 1

            # Collect results from any finished proces

            self._poll_busy()

        self.logger.info( f"Have sent {self.nsent} / {self.nargs} jobs, "
                          f"{self.nfail+self.nsuccess} are done ({self.nfail} failures), "
                     f"waiting for {len(self.busyprocs)} processes to finish" )

        while len(self.busyprocs) > 0:
            self._poll_busy()
            if ( len(self.busyprocs) > 0 ) and ( self.end_sleeptime > 0. ):
                time.sleep( self.end_sleeptime )

        s = f"Runner all done; {self.nsuccess}/{self.nargs} successes, {self.nfail}/{self.nargs} failures."
        if self.persistproc:
            self.logger.info( f"{s}  Telling worker processes to die." )
            for pipe in self.idlepipes:
                pipe.send( {'command': 'die'} )
        else:
            self.logger.info( s )

        # Join all subprocesses to make sure that they actually end.
        # Since self.subproc doesn't have much to do after receiving
        # a die command, I'm going to assume a timeout of 1 second
        # is enough.  (Should be way more than enough.  The downside
        # is, if there aren't persistent processes, and many hang,
        # this will wait a long time.)

        for proc in self.allprocs:
            proc.join( 1 )

        # TODO: Check to make sure all the processes are really finished, raise an exception if not

        return self.nargs, self.nsuccess, self.nfail
