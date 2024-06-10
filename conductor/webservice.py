import sys
import pathlib
import os
import time
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

# ======================================================================

class BaseView( flask.views.View ):
    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self.updater_socket_file = "/tmp/updater_socket"

    def check_auth( self ):
        self.username = flask.session['username'] if 'username' in flask.session else '(None)'
        self.displayname = flask.session['userdisplayname'] if 'userdisplayname' in flask.session else '(None)'
        self.authenticated = ( 'authenticated' in flask.session ) and flask.sesison['authenticated']
        return self.authenticated
        
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
                    return msg
                except TimeoutError:
                    timeout *= 2
                    if timeout > timeoutmax:
                        return { "status": "error", "error": "Connection to updater timed out" }
        except Exception as ex:
            app.logger.exception( ex )
            return { "status": "error", "error": f"Error talking to updater: {ex}" }
        finally:
            if sock is not None:
                sock.close()

    def dispatch_request( self ):
        if not self.check_auth():
            return f"Not logged in", 500
        return self.do_the_things()
        
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
        res = self.talk_to_updater( { 'command': 'status' } )
        if "status" not in res:
            return f"Unexpected response from udpater: {res}", 500
        if res['status'] == 'error':
            if 'error' in res:
                return f"Error getting status: {res['error']}", 500
            else:
                return f"Unknown error getting status", 500
        return res
        

# ======================================================================
# /update


    
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
flaskauth.RKAuthConfig.webap_url = f"{cfg.value('conductor.conductor_url')}/auth"
app.logger.debug( f'webap_url is {flaskauth.RKAuthConfig.webap_url}' )
app.register_blueprint( flaskauth.bp )
             
# Configure urls

urls = {
    "/": MainPage,
    "/status": GetStatus
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
