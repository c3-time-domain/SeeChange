import sys
import pathlib
import socket
import json
import datetime

import psycopg2
import psycopg2.extras

import flask

from util.util import asUUID
from models.base import SmartSession
from models.knownexposure import PipelineWorker, KnownExposure
# NOTE: for get_instrument_instrance to work, must manually import all
#  known instrument classes we might want to use here.
# If models.instrument gets imported somewhere else before this file
#  is imported, then even this won't work.  There must be a better way....
import models.decam  # noqa: F401
from models.instrument import get_instrument_instance

sys.path.insert( 0, pathlib.Path(__name__).resolve().parent )
from baseview import BaseView, BadUpdaterReturnError


class ConductorBaseView( BaseView ):
    _any_group_required = [ 'root', 'admin' ]

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self.updater_socket_file = "/tmp/updater_socket"


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
                        flask.current_app.logger.exception( f"Timed out trying to talk to updater, "
                                                            f"last delay was {timeout/2} sec" )
                        raise BadUpdaterReturnError( "Connection to updater timed out" )
        except Exception as ex:
            flask.current_app.logger.exception( ex )
            raise BadUpdaterReturnError( str(ex) )
        finally:
            if sock is not None:
                sock.close()


    def get_updater_status( self ):
        return self.talk_to_updater( { 'command': 'status' } )


# ======================================================================
# /status


class GetStatus( ConductorBaseView ):
    def do_the_things( self ):
        return self.get_updater_status()

# ======================================================================
# /forceupdate


class ForceUpdate( ConductorBaseView ):
    def do_the_things( self ):
        return self.talk_to_updater( { 'command': 'forceupdate' } )

# ======================================================================
# /updateparameters


class UpdateParameters( ConductorBaseView ):
    def do_the_things( self, argstr=None ):
        curstatus = self.get_updater_status()
        args = self.argstr_to_args( argstr )
        if args == {}:
            curstatus['status'] == 'unchanged'
            return curstatus

        flask.current_app.logger.debug( f"In UpdateParameters, argstr='{argstr}', args={args}" )

        knownkw = [ 'instrument', 'timeout', 'updateargs', 'hold', 'pause' ]
        unknown = set()
        for arg, val in args.items():
            if arg not in knownkw:
                unknown.add( arg )
        if len(unknown) != 0:
            return f"Unknown arguments to UpdateParameters: {unknown}", 500

        args['command'] = 'updateparameters'
        res = self.talk_to_updater( args )
        del curstatus['status']
        res['oldsconfig'] = curstatus

        return res

# ======================================================================
# /registerworker
#
# Register a Pipeline Worker.  This is really just for informational
# purposes; the conductor won't push jobs to workers, but it maintains
# a list of workers that have checked in so the user can see what's
# out there.
#
# parameters:
#   cluster_id str,
#   node_id str, optional
#   replace int, optional -- if non-zero, will replace an existing entry with this cluster/node
#   nexps int, optional number of exposures this pipeline worker can do at once (default 1)


class RegisterWorker( ConductorBaseView ):
    def do_the_things( self, argstr=None ):
        args = self.argstr_to_args( argstr, { 'node_id': None, 'replace': 0, 'nexps': 1 } )
        args['replace'] = int( args['replace'] )
        args['nexps'] = int( args['nexps'] )
        if 'cluster_id' not in args.keys():
            return "cluster_id is required for registerworker", 500
        with SmartSession() as session:
            existing = ( session.query( PipelineWorker )
                         .filter( PipelineWorker.cluster_id==args['cluster_id'] )
                         .filter( PipelineWorker.node_id==args['node_id'] )
                        ).all()
            newworker = None
            status = None
            if len( existing ) > 0:
                if len( existing ) > 1:
                    return ( f"cluster_id {args['cluster_id']} node_id{args['node_id']} multiply defined, "
                             f"database needs to be cleaned up" ), 500
                if args['replace']:
                    newworker = existing[0]
                    newworker.nexps = args['nexps']
                    newworker.lastheartbeat = datetime.datetime.now()
                    status = 'updated'
                else:
                    return f"cluster_id {args['cluster_id']} node_id {args['node_id']} already exists", 500

            else:
                newworker = PipelineWorker( cluster_id=args['cluster_id'],
                                            node_id=args['node_id'],
                                            nexps=args['nexps'],
                                            lastheartbeat=datetime.datetime.now() )
                status = 'added'
            session.add( newworker )
            session.commit()
            # Make sure that newworker has the id field loaded
            # session.merge( newworker )
        return { 'status': status,
                 'id': newworker.id,
                 'cluster_id': newworker.cluster_id,
                 'node_id': newworker.node_id,
                 'nexps': newworker.nexps }


# ======================================================================
# /unregisterworker
#
# Remove a Pipeline Worker registration.  Call with /unregsiterworker/n
# where n is the integer ID of the pipeline worker.

class UnregisterWorker( ConductorBaseView ):
    def do_the_things( self, pipelineworker_id ):
        with SmartSession() as session:
            pipelineworker_id = asUUID( pipelineworker_id )
            existing = session.query( PipelineWorker ).filter( PipelineWorker._id==pipelineworker_id ).all()
            if len(existing) == 0:
                return f"Unknown pipeline worker {pipelineworker_id}", 500
            else:
                session.delete( existing[0] )
                session.commit()
        return { "status": "worker deleted" }


# ======================================================================
# /workerheartbeat
#
# Call at /workerheartbeat/n where n is the uuid of the pipeline worker

class WorkerHeartbeat( ConductorBaseView ):
    def do_the_things( self, pipelineworker_id ):
        pipelineworker_id = asUUID( pipelineworker_id )
        with SmartSession() as session:
            existing = session.query( PipelineWorker ).filter( PipelineWorker._id==pipelineworker_id ).all()
            if len( existing ) == 0:
                return f"Unknown pipelineworker {pipelineworker_id}"
            existing = existing[0]
            existing.lastheartbeat = datetime.datetime.now()
            session.merge( existing )
            session.commit()
            return { 'status': 'updated' }

# ======================================================================
# /getworkers


class GetWorkers( ConductorBaseView ):
    def do_the_things( self ):
        with SmartSession() as session:
            workers = session.query( PipelineWorker ).all()
            return { 'status': 'ok',
                     'workers': [ w.to_dict() for w in workers ] }

# ======================================================================
# /requestexposure


class RequestExposure( ConductorBaseView ):
    def do_the_things( self, argstr=None ):
        args = self.argstr_to_args( argstr )
        if 'cluster_id' not in args.keys():
            return "cluster_id is required for RequestExposure", 500
        # Using direct postgres here since I don't really know how to
        #  lock tables with sqlalchemy.  There is with_for_udpate(), but
        #  then the documentation has this red-backgrounded warning
        #  that using this is not recommended when there are
        #  relationships.  Since I can't really be sure what
        #  sqlalchemy is actually going to do, just communicate
        #  with the database the way the database was meant to
        #  be communicated with.
        knownexp_id = None
        with SmartSession() as session:
            dbcon = None
            cursor = None
            try:
                dbcon = session.bind.raw_connection()
                cursor = dbcon.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
                cursor.execute( "LOCK TABLE knownexposures" )
                cursor.execute( "SELECT _id, cluster_id FROM knownexposures "
                                "WHERE cluster_id IS NULL AND NOT hold "
                                "ORDER BY mjd LIMIT 1" )
                rows = cursor.fetchall()
                if len(rows) > 0:
                    knownexp_id = rows[0]['_id']
                    cursor.execute( "UPDATE knownexposures "
                                    "SET cluster_id=%(cluster_id)s, claim_time=NOW() "
                                    "WHERE _id=%(id)s",
                                    { 'id': knownexp_id, 'cluster_id': args['cluster_id'] } )
                    dbcon.commit()
            except Exception:
                raise
            finally:
                if cursor is not None:
                    cursor.close()
                if dbcon is not None:
                    dbcon.rollback()

        if knownexp_id is not None:
            return { 'status': 'available', 'knownexposure_id': knownexp_id }
        else:
            return { 'status': 'not available' }


# ======================================================================


class GetKnownExposures( ConductorBaseView ):
    def do_the_things( self, argstr=None ):
        args = self.argstr_to_args( argstr, { "minmjd": None, "maxmjd": None } )
        args['minmjd'] = float( args['minmjd'] ) if args['minmjd'] is not None else None
        args['maxmjd'] = float( args['maxmjd'] ) if args['maxmjd'] is not None else None
        with SmartSession() as session:
            q = session.query( KnownExposure )
            if args['minmjd'] is not None:
                q = q.filter( KnownExposure.mjd >= args['minmjd'] )
            if args['maxmjd'] is not None:
                q = q.filter( KnownExposure.mjd <= args['maxmjd'] )
            q = q.order_by( KnownExposure.instrument, KnownExposure.mjd )
            kes = q.all()
        retval= { 'status': 'ok',
                  'knownexposures': [ ke.to_dict() for ke in kes ] }
        # Add the "id" field that's the same as "_id" for convenience,
        #   and make the filter the short name
        for ke in retval['knownexposures']:
            ke['id'] = ke['_id']
            ke['filter'] = get_instrument_instance( ke['instrument'] ).get_short_filter_name( ke['filter'] )
        return retval

# ======================================================================


class HoldReleaseExposures( ConductorBaseView ):
    def hold_or_release( self, keids, hold ):
        # flask.current_app.logger.info( f"HoldOrReleaseExposures with hold={hold} and keids={keids}" )
        if len( keids ) == 0:
            return { 'status': 'ok', 'held': [], 'missing': [] }
        held = []
        with SmartSession() as session:
            q = session.query( KnownExposure ).filter( KnownExposure._id.in_( keids ) )
            todo = q.all()
            # flask.current_app.logger.info( f"HoldOrRelease got {len(todo)} things "
            #                                f"to {'hold' if hold else 'release'}" )
            kes = { str(i._id) : i for i in todo }
            notfound = []
            for keid in keids:
                if keid not in kes.keys():
                    notfound.append( keid )
                else:
                    kes[keid].hold = hold
                    held.append( keid )
            session.commit()
        return { 'status': 'ok', 'held': held, 'missing': notfound }


class HoldExposures( HoldReleaseExposures ):
    def do_the_things( self ):
        args = self.argstr_to_args( None, { 'knownexposure_ids': [] } )
        return self.hold_or_release( args['knownexposure_ids'], True )


class ReleaseExposures( HoldReleaseExposures ):
    def do_the_things( self ):
        args = self.argstr_to_args( None, { 'knownexposure_ids': [] } )
        retval = self.hold_or_release( args['knownexposure_ids'], False )
        retval['released'] = retval['held']
        del retval['held']
        return retval


# ======================================================================
# Create and configure the sub web ap (i.e. flask blueprint)

bp = flask.Blueprint( 'conductor', __name__, url_prefix='/conductor' )

urls = {
    "/status": GetStatus,
    "/updateparameters": UpdateParameters,
    "/updateparameters/<path:argstr>": UpdateParameters,
    "/forceupdate": ForceUpdate,
    "/requestexposure": RequestExposure,
    "/requestexposure/<path:argstr>": RequestExposure,
    "/registerworker": RegisterWorker,
    "/registerworker/<path:argstr>": RegisterWorker,
    "/workerheartbeat/<pipelineworker_id>": WorkerHeartbeat,
    "/unregisterworker/<pipelineworker_id>": UnregisterWorker,
    "/getworkers": GetWorkers,
    "/getknownexposures": GetKnownExposures,
    "/getknownexposures/<path:argstr>": GetKnownExposures,
    "/holdexposures": HoldExposures,
    "/releaseexposures": ReleaseExposures,
}

usedurls = {}
for url, cls in urls.items():
    if url not in usedurls.keys():
        usedurls[ url ] = 0
        name = url
    else:
        usedurls[ url ] += 1
        name = f"url.{usedurls[url]}"

    bp.add_url_rule( url, view_func=cls.as_view(name), methods=["GET", "POST"], strict_slashes=False )
