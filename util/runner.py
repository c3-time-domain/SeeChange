import sys
import time
import multiprocessing
import logging

class Runner:
    """Python class for running a bunch of multprocessing tasks at once."""

    def __init__( self, numprocs, executor, aggregator=None, persistent_processes=True, logger=None,
                  end_sleeptime=0.1 ):
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

    def subproc( self, pipe, logger ):
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


    def makeproc( self, name ):
        mypipe, theirpipe = multiprocessing.Pipe( True )
        proc = multiprocessing.Process( target=lambda: self.subproc( theirpipe, self.logger ), name=name )
        proc.start()
        self.allprocs.append( proc )
        return proc, mypipe

    def poll_busy( self ):
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
                proc, pipe = self.makeproc( f"proc {i}" )
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
                    proc, pipe = self.makeproc( f"proc {self.nsent}" )
                self.busyprocs.append( proc )
                self.busypipes.append( pipe )
                self.busyargdex.append( argdex )
                self.logger.info( f"Sending job {self.nsent} / {self.nargs} to {proc.name}" )
                pipe.send( { 'command': 'do', 'arg': args[argdex] } )
                argdex += 1
                self.nsent += 1

            # Collect results from any finished proces

            self.poll_busy()

        self.logger.info( f"Have sent {self.nsent} / {self.nargs} jobs, "
                          f"{self.nfail+self.nsuccess} are done ({self.nfail} failures), "
                     f"waiting for {len(self.busyprocs)} processes to finish" )

        while len(self.busyprocs) > 0:
            self.poll_busy()
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
