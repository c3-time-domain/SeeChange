import sys
import os.path
import socket
import select
import time
import datetime
import pathlib
import logging
import json
import multiprocessing

# Have to manually import any instrument modules
#  we want to be able to find.  (Otherwise, they
#  won't be found when models.instrument  is
#  initialized.)
import models.decam
from models.instrument import get_instrument_instance

_logger = logging.getLogger("main")
if not _logger.hasHandlers():
    _logout = logging.StreamHandler( sys.stderr )
    _logger.addHandler( _logout )
    _formatter = logging.Formatter( f'[%(asctime)s - UPDATER - %(levelname)s] - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S' )
    _logout.setFormatter( _formatter )
_logger.propagate = False
# _logger.setLevel( logging.INFO )
_logger.setLevel( logging.DEBUG )

_max_message_size = 16384     # This should be way more than enough, unless updateargs gets out of hand

def now():
    return datetime.datetime.now( tz=datetime.timezone.utc ).strftime( '%Y-%m-%d %H:%M:%S %Z' )

class Updater():
    def __init__( self ):
        self.instrument_name = None
        self.instrument = None
        self.updateargs = None
        self.timeout = 120
        self.pause = False
        self.hold = False

        self.lasttimeout = None
        self.lastupdate = None
        self.configchangetime = None

    def run_update( self ):
        if self.instrument is not None:
            self.lastupdate = now()
            _logger.info( "Updating known exposures" )
            _logger.debug( f"updateargs = {self.updateargs}" )
            exps = self.instrument.find_origin_exposures( **self.updateargs )
            if ( exps is not None ) and ( len(exps) > 0 ):
                exps.add_to_known_exposures( hold=self.hold )
                _logger.info( f"Got {len(exps)} exposures to possibly add" )
            else:
                _logger.info( f"No exposures found." )
        else:
            _logger.warning( "No instrument defined, not updating" )


    def parse_bool_arg( self, arg ):
        try:
            iarg = int( arg )
        except ValueError:
            iarg = None
        if ( ( isinstance( arg, str ) and ( arg.lower().strip() == 'true' ) ) or
             ( ( iarg is not None ) and bool(iarg) ) ):
            return True
        else:
            return False

    def __call__( self ):
        # Open up a socket that we'll listen on to be told things to do
        # We'll have a timeout (default 120s).  Every timeout, *if* we
        # have an instrument defined, run instrument.find_origin_exposures()
        # and add_to_known_exposures() on the return value.  Meanwhile,
        # listen for connections that tell us to change our state (e.g.
        # change parameters, change timeout, change instrument).

        sock = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM, 0 )
        sockpath = "/tmp/updater_socket"
        if os.path.exists( sockpath ):
            os.remove( sockpath )
        sock.bind( sockpath )
        sock.listen()
        poller = select.poll()
        poller.register( sock )

        self.lasttimeout = time.perf_counter() - self.timeout
        done = False
        while not done:
            try:
                _logger.debug( f"self.timeout={self.timeout}, time.perf_counter={time.perf_counter()}, "
                               f"self.lasttimeout={self.lasttimeout}" )
                waittime = max( self.timeout - ( time.perf_counter() - self.lasttimeout ), 0.1 )
                _logger.debug( f"Waiting {waittime} sec" )
                res = poller.poll( 1000 * waittime )
                if len(res) == 0:
                    # Didn't get a message, must have timed out
                    self.lasttimeout = time.perf_counter();
                    if self.pause:
                        _logger.warning( "Paused, not updating." )
                    else:
                        self.run_update()
                else:
                    # Got a message, parse it
                    conn, address = sock.accept()
                    bdata = conn.recv( _max_message_size )
                    try:
                        msg = json.loads( bdata )
                    except Exception as ex:
                        _logger.error( f"Failed to parse json: {bdata}" )
                        conn.send( json.dumps( {'status': 'error',
                                                'error': 'Error parsing message as json' } ) )
                        continue

                    if ( not isinstance( msg, dict ) ) or ( 'command' not in msg.keys() ):
                        _logger.error( f"Don't understand message {msg}" )
                        conn.send( json.dumps( { 'status': 'error',
                                                 'error': "Don't understand message {msg}" } ).encode( 'utf-8' ) )

                    elif msg['command'] == 'die':
                        _logger.info( f"Got die, dying." )
                        conn.send( json.dumps( { 'status': 'dying' } ).encode( 'utf-8' ) )
                        done = True

                    elif msg['command'] == 'forceupdate':
                        # Forced update resets the timeout clock
                        self.lasttimeout = time.perf_counter()
                        self.run_update()
                        conn.send( json.dumps( { 'status': 'forced update' } ).encode( 'utf-8' ) )

                    elif msg['command'] == 'updateparameters':
                        _logger.info( f"Updating poll parameters" )
                        if 'timeout' in msg.keys():
                            self.timeout = float( msg['timeout'] )

                        if 'instrument' in msg.keys():
                            self.instrument_name = msg['instrument']
                            self.instrument = None
                            self.updateargs = None

                        if 'updateargs' in msg.keys():
                            self.updateargs = msg['updateargs']

                        if 'hold' in msg.keys():
                            self.hold = self.parse_bool_arg( msg['hold'] )

                        if 'pause' in msg.keys():
                            self.pause = self.parse_bool_arg( msg['pause'] )

                        if ( self.instrument_name is None ) != ( self.updateargs is None ):
                            errmsg = ( f'Either both or neither of instrument and updateargs must be None; '
                                       f'instrument={self.instrument_name}, updateargs={self.updateargs}' )
                            self.instrument_name = None
                            self.instrument = None
                            self.updateargs = None
                            conn.send( json.dumps( { 'status': 'error', 'error': errmsg } ).encode( 'utf-8' ) )
                        else:
                            try:
                                self.configchangetime = now()
                                if self.instrument_name is not None:
                                    self.instrument = get_instrument_instance( self.instrument_name )
                                    if self.instrument is None:
                                        raise RuntimeError( "Unknown instrument" )
                            except Exception as ex:
                                conn.send( json.dumps( { 'status': 'error',
                                                         'error': f'Failed to find instrument {self.instrument_name}' }
                                                      ).encode( 'utf-8' ) )
                                self.instrument_name = None
                                self.instrument = None
                                self.updateargs = None
                            else:
                                conn.send( json.dumps( { 'status': 'updated',
                                                         'instrument': self.instrument_name,
                                                         'updateargs': self.updateargs,
                                                         'hold': int(self.hold),
                                                         'pause': int(self.pause),
                                                         'timeout': self.timeout,
                                                         'lastupdate': self.lastupdate,
                                                         'configchangetime': self.configchangetime }
                                                      ).encode( 'utf-8' ) )

                    elif msg['command'] == 'status':
                        conn.send( json.dumps( { 'status': 'status',
                                                 'timeout': self.timeout,
                                                 'instrument': self.instrument_name,
                                                 'updateargs': self.updateargs,
                                                 'hold': int(self.hold),
                                                 'pause': int(self.pause),
                                                 'lastupdate': self.lastupdate,
                                                 'configchangetime': self.configchangetime } ).encode( 'utf-8' ) )
                    else:
                        conn.send( json.dumps( { 'status': 'error',
                                                 'error': f"Unrecognized command {msg['command']}" }
                                              ).encode( 'utf-8' ) )
            except Exception as ex:
                _logger.exception( "Exception in poll loop; continuing" )

# ======================================================================

if __name__ == "__main__":
    updater = Updater()
    updater()
