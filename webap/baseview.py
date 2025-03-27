import re
import copy
import uuid
import datetime
import simplejson

import numpy as np

import flask
import flask.views

from models.base import SmartSession
from models.user import AuthUser


# This is just like util/util.py:NumpyAndUUIDJsonEncoder,
#   only here it derives from simplejson.JSONEncoder
# Perhaps we should go to simplejson everywhere???
class MyParticularJSONEncoder( simplejson.JSONEncoder ):
    def default( self, obj ):
        if isinstance( obj, np.integer ):
            return int( obj )
        if isinstance( obj, np.floating ):
            return float( obj )
        if isinstance( obj, np.bool_ ):
            return bool( obj )
        if isinstance( obj, np.ndarray ):
            return obj.tolist()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, datetime.datetime ):
            return obj.isoformat()
        return super().default(self, obj)


class BadUpdaterReturnError(Exception):
    pass


# ======================================================================

class BaseView( flask.views.View ):
    _admin_required = False
    _any_group_required = None

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )

    def check_auth( self ):
        self.username = flask.session['username'] if 'username' in flask.session else '(None)'
        self.displayname = flask.session['userdisplayname'] if 'userdisplayname' in flask.session else '(None)'
        self.authenticated = ( 'authenticated' in flask.session ) and flask.session['authenticated']
        self.user = None
        if self.authenticated:
            self.user = self.session.query( AuthUser ).filter( AuthUser.username==self.username ).first()
            if self.user is None:
                self.authenticated = False
                raise ValueError( f"Error, failed to find user {self.username} in database" )
        return self.authenticated


    def argstr_to_args( self, argstr, initargs={} ):
        """Parse argstr as a bunch of /kw=val to a dictionary, update with request body if it's json."""

        args = copy.deepcopy( initargs )
        if argstr is not None:
            for arg in argstr.split("/"):
                match = re.search( '^(?P<k>[^=]+)=(?P<v>.*)$', arg )
                if match is None:
                    flask.current_app.logger.error( f"error parsing url argument {arg}, must be key=value" )
                    raise Exception( f'error parsing url argument {arg}, must be key=value' )
                args[ match.group('k') ] = match.group('v')
        if flask.request.is_json:
            args.update( flask.request.json )
        return args


    def dispatch_request( self, *args, **kwargs ):
        # Webaps, where you expect the runtime to be short (ideally at
        #  most seconds, or less!) is the use case where holding open a
        #  database connection for the whole runtime actually might make
        #  sense.

        with SmartSession() as session:
            self.session = session
            # Also get the raw psycopg2 connection, because we need it
            #   to be able to avoid dealing with SA where possible.
            self.conn = session.bind.raw_connection()

            if not self.check_auth():
                return "Not logged in", 500
            if ( self._admin_required ) and ( 'root' not in self.user.groups ):
                return "Action requires root", 500
            if ( ( self._any_group_required is not None ) and
                 ( not any( [ g in self.user.groups for g in self._any_group_required ] ) )
                ):
                return ( f"Action requires user to be in one of the groups "
                         f"{', '.join([str(g) for g in self._any_group_required])}", 500 )

            try:
                retval = self.do_the_things( *args, **kwargs )
                # Can't just use the default JSON handling, because it
                #   writes out NaN which is not standard JSON and which
                #   the javascript JSON parser chokes on.  Sigh.
                if isinstance( retval, dict ) or isinstance( retval, list ):
                    return ( simplejson.dumps( retval, ignore_nan=True, cls=MyParticularJSONEncoder ),
                             200, { 'Content-Type': 'application/json' } )
                elif isinstance( retval, str ):
                    return retval, 200, { 'Content-Type': 'text/plain; charset=utf-8' }
                elif isinstance( retval, tuple ):
                    return retval
                else:
                    return retval, 200, { 'Content-Type': 'application/octet-stream' }
            except BadUpdaterReturnError as ex:
                return str(ex), 500
            except Exception as ex:
                flask.current_app.logger.exception( str(ex) )
                return f"Exception handling request: {ex}", 500
