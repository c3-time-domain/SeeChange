import sys
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
            self.logger.setLevel( logging.INFO )

    def subproc( self, pipe, logger ):
        me = multiprocessing.current_process()
        logger.info( "Process starting: {me.name} PID {me.pid}" )

        try:
            done = False
            while not done:
                command = pipe.recv()
                if command['command'] == 'die':
                    logger.debug( f"{me.name} : got die command" )
                    done = True

                elif command['command'] == 'do':
                    try:
                        if 'info' in command:
                            logger.info( f'{me.name} running: {command["info"]}' )
                        res = self.executor( command['arg'] )
                        pipe.send( { 'status': 'done', 'result': res } )
                    except Exception as ex:
                        s = f"{me.name} exception: {ex}"
                        logger.Exception( s )
                        pipe.send( { 'status': 'error', 'error': s } )

                    if not self.persistproc:
                        done = True

        except Exception as ex:
            logger.exception( f"Process {me.name} PID {me.pid} returning after exception" )
            return

        logger.debug( f"Process {me.name} PID {me.pid} " )
        return

    def go( self, args ):

        idleprocs = []
        idlepipes = []
        busyprocs = []
        busypipes = []
        nargs = len(args)
        nsent = 0
        nsuccess = 0
        nfail = 0

        def makeproc( i ):
            mypipe, theirpipe = multiprocessing.Pipe( True )
            proc = multiprocessing.Process( target=lambda: self.subprocs( theirpipe, self.logger ),
                                            name=f"proc {i}" )
            proc.start()
            return proc, mypipe

        def collect( proc, pipe ):
            logger.debug( f"Getting response from {proc.name}" )
            rval = pipe.recv()
            fail = False
            if 'status' not in rval:
                logger.error( f"Unexpected response from {proc.name}: {rval}" )
                nfail += 1
                fail = True
            elif rval['status'] == 'error':
                logger.error( f"Error return from {proc.name}: {rval}" )
                nfail += 1
                fail = True
            else:
                nsuccess += 1
            if self.aggregator is not None:
                self.aggregator( None if fail else rval )
            if self.persistproc:
                idleprocs.append( busyprocs[busydex] )
                idlepipes.append( busypipes[busydex] )
                
            
        if self.persistproc:
            self.loggerr.info( f"Creating {self.numprocs} persistent worker processes"  )
            for i in range(numprocs):
                proc, pipe = makeproc( i )
                idleprocs.append( proc )
                idlepipes.append( pipe )

        logger.info( f"Beinning to work through {len(args)} jobs" )

        args = args.copy()
        while len(args) > 0:

            # Launch jobs until we have max processes running
            
            while ( len(args) > 0 ) and ( len(busyprocs) < self.numprocs ):
                proc = None
                pipe = None
                if self.persistproc:
                    proc = idleprocs.pop()
                    pipe = idlepipes.pop()
                else:
                    proc, pipe = makeproc( nsent )
                busyprocs.append( proc )
                busypipes.append( pipe )
                arg = args.pop()
                logger.info( f"Sending job {nsent} / {nargs} to {proc.name}" )
                pipe.send( arg )

            # Collect results from any finished proces

            busydex = 0
            while busydex < len( busyprocs ):
                if busypipes[busydex].poll():
                    collect( busyprocs[busydex], busypipes[busydex] )
                    del busyprocs[ busydex ]
                    del busypipes[ busydex ]
                else:
                    busydex += 1

        logger.info( f"Have sent {sent} / {nargs} jobs, {nfail+nsuccess} are done ({nfail} failures), "
                     f"waiting for {len(busyprocs)} processes to finish" )

        while len(busyprocs) > 0:
            busydex = 0
            while busydex < len( busyprocs ):
                if busypipes[busydex].poll():
                    collect( busyprocs[busydex], busypipes[busydex] )
                    del busyprocs[ busydex ]
                    del busypipes[ busydex ]
                else:
                    busydex += 1
            if ( len(busyprocs) > 0 ) and ( self.end_sleeptime > 0. ):
                time.sleep( self.end_sleeptime )

        s = f"Runner all done; {nsuccess}/{nargs} successes, {nfail}/{nargs} failures."
        if self.persistproc:
            logger.info( f"{s}  Telling worker processes to die." )
            for pipe in idlepipes:
                pipe.send( {'command': 'die'} )
        else:
            logger.info( s )

`        return nargs, nsuccess, nfail
