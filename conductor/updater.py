import sys
import os.path
import socket
import select
import time
import pathlib
import logging
import json
import multiprocessing

from models.instrument import get_instrument_instance

logger = logging.getLogger("main")
if not logger.hasHandlers():
    logout = logging.StreamHandler( sys.stderr )
    logger.addHandler( logout )
    formatter = logging.Formatter( f'[%(asctime)s - UPDATER - %(levelname)s] - %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S' )
    logout.setFormatter( formatter )
logger.propagate = False
# logger.setLevel( logging.INFO )
logger.setLevel( logging.DEBUG )

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

instrument_name = None
instrument = None
updateargs = None
timeout = 120

lasttimeout = time.perf_counter() - timeout
done = False
while not done:
    try:
        waittime = max( timeout - ( time.perf_counter() - lasttimeout ), 0.1 )
        logger.debug( f"Waiting {waittime} sec" )
        res = poller.poll( 1000 * waittime )
        if len(res) == 0:
            if instrument is not None:
                logger.info( "Updating known exposures" )
                exps = instrument.find_origin_exposures( **updateargs )
                logger.info( f"Got {len(exps)} exposures to possibly add" )
                if len(exps) > 0:
                    exps.add_to_known_exposures()
            else:
                logger.warning( "No instrument defined, not updating" )
            lasttimeout = time.perf_counter()

        else:
            conn, address = sock.accept()
            bdata = conn.recv( 16384 )
            try:
                msg = json.loads( bdata )
            except Exception as ex:
                conn.send( json.dumps( {'status': 'error',
                                        'error': 'Error parsing message as json' } ) )
                continue

            if ( not isinstance( msg, dict ) ) or ( 'command' not in msg.keys() ):
                logger.error( f"Don't understand message {msg}" )
                conn.send( json.dumps( { 'status': 'error',
                                         'error': "Don't understand message {msg}" } ).encode( 'utf-8' ) )

            elif msg['command'] == 'die':
                logger.info( f"Got die, dying." )
                conn.send( json.dumps( { 'status': 'dying' } ).encode( 'utf-8' ) )
                done = True

            elif msg['command'] == 'update':
                logger.info( f"Updating poll parameters" )
                if 'timeout' in msg.keys():
                    timeout = msg['timeout']

                if 'instrument' in msg.keys():
                    instrument_name = msg['instrument']
                    instrument = None
                    updateargs = None

                if 'updateargs' in msg.keys():
                    updateargs = msg['updateargs']

                if ( instrument_name is None ) != ( updateargs is None ):
                    instrument_name = None
                    instrument = None
                    updateargs = None
                    conn.send( json.dumps( { 'status': 'error',
                                             'error': ( 'Either both or neither of instrument and updateargs '
                                                        'must be None' ) } ).encode( 'utf-8' ) )
                else:
                    try:
                        instrument = get_instrument_instance( instrument_name )
                    except Exception as ex:
                        conn.send( json.dumps( { 'status': 'error',
                                                 'error': 'Failed to find instrument {instrument_name}' }
                                              ).encode( 'utf-8' ) )
                        instrument_name = None
                        instrument = None
                        updateargs = None
                    else:
                        conn.send( json.dumps( { 'status': 'updated',
                                                 'instrument_name': instrument_name,
                                                 'updateargs': updateargs,
                                                 'timeout': timeout } ).encode( 'utf-8' ) )

            elif msg['command'] == 'status':
                conn.send( json.dumps( { 'status': 'status',
                                         'timeout': timeout,
                                         'instrument_name': instrument_name,
                                         'updateargs': updateargs } ).encode( 'utf-8' ) )
            else:
                conn.send( json.dumps( { 'status': 'error',
                                         'error': f"Unrecognized command {msg['command']}" }
                                      ).encode( 'utf-8' ) )
    except Exception as ex:
        logger.exception( "Exception in poll loop; continuing" )
