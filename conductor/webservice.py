import sys
import pathlib
import os
import re
import time
import datetime
import logging
import subprocess
import multiprocessing
import socket
import json

import flask
import flask_session
import flask.views

from models.instrument import get_instrument_instance
from util.config import Config

class BadUpdaterReturnError(Exception):
    pass

# ======================================================================

class BaseView( flask.views.View ):
    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self.updater_socket_file = "/tmp/updater_socket"

    def check_auth( self ):
        self.username = flask.session['username'] if 'username' in flask.session else '(None)'
        self.displayname = flask.session['userdisplayname'] if 'userdisplayname' in flask.session else '(None)'
        self.authenticated = ( 'authenticated' in flask.session ) and flask.session['authenticated']
        return self.authenticated
        
    def argstr_to_args( self, argstr ):
        """Parse argstr as a bunch of /kw=val to a dictionary, update with request body if it's json."""

        kwargs = {}
        if argstr is not None:
            for arg in argstr.split("/"):
                match = re.search( '^(?P<k>[^=]+)=(?P<v>.*)$', arg )
                if match is None:
                    app.logger.error( f"error parsing url argument {arg}, must be key=value" )
                    raise Exception( f'error parsing url argument {arg}, must be key=value' )
                kwargs[ match.group('k') ] = match.group('v')
        if flask.request.is_json:
            kwargs.update( flask.request.json )
        return kwargs

    def talk_to_updater( self, req, bsize=16384, timeout0=1, timeoutmax=16 ):
        sock = None
        try:
            sock = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM, 0 )
            sock.connect( self.updater_socket_file )
            sock.send( json.dumps( req ).encode( "utf-8" ) )
            timeout = timeout0
            while True:
                try:
                    sock.settimeout( timeout )
                    bdata = sock.recv( bsize )
                    msg = json.loads( bdata )
                    if 'status' not in msg:
                        raise BadUpdaterReturnError( f"Unexpected response from updater: {msg}" )
                    if msg['status'] == 'error':
                        if 'error' in msg:
                            raise BadUpdaterReturnError( f"Error return from updater: {msg['error']}" )
                        else:
                            raise BadUpdaterReturnError( "Unknown error return from updater" )
                    return msg
                except TimeoutError:
                    timeout *= 2
                    if timeout > timeoutmax:
                        app.logger.exception( f"Timed out trying to talk to updater, "
                                              f"last delay was {timeout/2} sec" )
                        raise BadUpdaterReturnError( "Connection to updater timed out" )
        except Exception as ex:
            app.logger.exception( ex )
            raise BadUpdaterReturnError( str(ex) )
        finally:
            if sock is not None:
                sock.close()

    def get_updater_status( self ):
        return self.talk_to_updater( { 'command': 'status' } )
                
    def dispatch_request( self, *args, **kwargs ):
        if not self.check_auth():
            return f"Not logged in", 500
        try:
            return self.do_the_things( *args, **kwargs )
        except BadUpdaterReturnError as ex:
            return str(ex), 500
        except Exception as ex:
            app.logger.exception()
            return f"Exception handling request: {ex}", 500
        
# ======================================================================
# /
#
# This is the only view that doesn't require authentication (Hence it
# has its own dispatch_request method rather than calling the
# do_the_things method in BaseView's dispatch_request.)
 
class MainPage( BaseView ):
    def dispatch_request( self ):
        return flask.render_template( "conductor_root.html" )

# ======================================================================
# /status

class GetStatus( BaseView ):
    def do_the_things( self ):
        return self.get_updater_status()

# ======================================================================
# /forceupdate

class ForceUpdate( BaseView ):
    def do_the_things( self ):
        return self.talk_to_updater( { 'command': 'forceupdate' } )

# ======================================================================
# /updateparameters

class UpdateParameters( BaseView ):
    def do_the_things( self, argstr=None ):
        curstatus = self.get_updater_status()
        args = self.argstr_to_args( argstr )
        if args == {}:
            curstatus['status'] == 'unchanged'
            return curstatus

        knownkw = [ 'instrument', 'timeout', 'updateargs' ]
        unknown = set()
        for arg, val in args.items():
            if arg not in knownkw:
                unknown.add( arg )
        if len(unknown) != 0:
            return f"Unknown arguments to UpdateParameters: {unknown}", 500

        args['command'] = 'updateparameters'
        res = self.talk_to_updater( args )
        del( curstatus['status'] )
        res['oldsconfig'] = curstatus

        return res
        
        
        
    
# ======================================================================
# Create and configure the web app

cfg = Config.get()

app = flask.Flask( __name__, instance_relative_config=True )
# app.logger.setLevel( logging.INFO )
app.logger.setLevel( logging.DEBUG )
app.config.from_mapping(
    SECRET_KEY='szca2ukaz4l33v13yx7asrwqudigau46n0bjcc9yc9bau1sn709c5or44rmg2ybb',
    SESSION_COOKIE_PATH='/',
    SESSION_TYPE='filesystem',
    SESSION_PERMANENT=True,
    SESSION_FILE_DIR='/sessions',
    SESSION_FILE_THRESHOLD=1000,
)
server_session = flask_session.Session( app )

# Import and configure the auth subapp

sys.path.insert( 0, pathlib.Path(__name__).parent )
import flaskauth
for attr in [ 'email_from', 'email_subject', 'email_system_name',
              'smtp_server', 'smtp_port', 'smtp_use_ssl', 'smtp_username', 'smtp_password' ]:
    setattr( flaskauth.RKAuthConfig, attr, cfg.value( f'conductor.{attr}' ) )
flaskauth.RKAuthConfig.webap_url = cfg.value('conductor.conductor_url')
if flaskauth.RKAuthConfig.webap_url[-1] != '/':
    flaskauth.RKAuthConfig.webap_url += '/'
flaskauth.RKAuthConfig.webap_url += "auth"
# app.logger.debug( f'webap_url is {flaskauth.RKAuthConfig.webap_url}' )
app.register_blueprint( flaskauth.bp )
             
# Configure urls

urls = {
    "/": MainPage,
    "/status": GetStatus,
    "/updateparameters": UpdateParameters,
    "/updateparameters/<path:argstr>": UpdateParameters,
    "/forceupdate": ForceUpdate,
}

usedurls = {}
for url, cls in urls.items():
    if url not in usedurls.keys():
        usedurls[ url ] = 0
        name = url
    else:
        usedurls[ url ] += 1
        name = f"url.{usedurls[usr]}"
        
    app.add_url_rule( url, view_func=cls.as_view(name), methods=["GET", "POST"], strict_slashes=False )
