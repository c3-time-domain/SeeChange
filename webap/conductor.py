import sys
import pathlib
import socket
import json
import datetime
import pytz
import traceback

import psycopg2
import psycopg2.extras

import flask

from util.util import asUUID
from models.base import SmartSession, Psycopg2Connection
from models.enums_and_bitflags import KnownExposureStateConverter
from models.knownexposure import PipelineWorker
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

    updater_socket_file = "/tmp/updater_socket"

    instrument_name = None
    updateargs = None
    update_timeout = 120
    pause_updates = False
    hold_new_exposures = False
    configchangetime = None
    throughstep = "alerting"
    pickuppartial = False

    @classmethod
    def restore_conductor_state( cls ):
        """This class method is called once upon module init."""

        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "LOCK TABLE conductor_config" )
            cursor.execute( "SELECT * FROM conductor_config" )
            columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
            rows = cursor.fetchall()
            if len( rows ) > 1:
                raise RuntimeError( "Multiple rows in conductor config!" )
            if len( rows ) == 0:
                cursor.execute( "INSERT INTO conductor_config(instrument_name, updateargs, update_timeout, "
                                "                             pause_updates, hold_new_exposures, configchangetime, "
                                "                             throughstep, pickuppartial) "
                                "VALUES( %(inst)s, %(upda)s, %(updt)s, %(pause)s, %(hold)s, "
                                "        %(t)s, %(through)s, %(partial)s )",
                                { 'inst': cls.instrument_name,
                                  'upda': cls.updateargs,
                                  'updt': cls.update_timeout,
                                  'pause': cls.pause_updates,
                                  'hold': cls.hold_new_exposures,
                                  't': datetime.datetime.now( tz=datetime.UTC ),
                                  'through': cls.throughstep,
                                  'partial': cls.pickuppartial } )
                conn.commit()
            else:
                row = rows[0]
                cls.instrument_name = row[ columns[ 'instrument_name' ] ]
                cls.updateargs = row[ columns[ 'updateargs' ] ]
                cls.update_timeout = row[ columns[ 'update_timeout' ] ]
                cls.pause_updates = row[ columns[ 'pause_updates' ] ]
                cls.hold_new_exposures = row[ columns[ 'hold_new_exposures' ] ]
                cls.configchangetime = row[ columns[ 'configchangetime' ] ]
                cls.throughstep = row[ columns[ 'throughstep' ] ]
                cls.pickuppartial = row[ columns[ 'pickuppartial' ] ]
                msg = cls.talk_to_updater( { 'command': 'updateparameters',
                                             'instrument': cls.instrument_name,
                                             'updateargs': cls.updateargs,
                                             'hold': cls.hold_new_exposures,
                                             'pause': cls.pause_updates,
                                             'timeout': cls.update_timeout } )
                cls.confighcangetime = msg[ 'configchangetime' ]


    @classmethod
    def talk_to_updater( cls, req, bsize=16384, timeout0=1, timeoutmax=16 ):
        sock = None
        try:
            sock = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM, 0 )
            sock.connect( cls.updater_socket_file )
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
            # Need this next try because we call restore_conductor_state, which in turn
            #   calls talk_to_updater, before the flask application is initialized,
            #   so flask.current_app doesn't work yet.  (But we also call this a lot
            #   once the flask app is started, and we want to use the logger then.)
            try:
                flask.current_app.logger.exception( ex )
            except Exception:
                sys.stderr.write( "Exception talking to updater during init\n" )
                traceback.print_exception( ex, file=sys.stderr )
            raise BadUpdaterReturnError( str(ex) )
        finally:
            if sock is not None:
                sock.close()


    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )


    def get_updater_status( self ):
        return self.talk_to_updater( { 'command': 'status' } )


# ======================================================================
# /status


class GetStatus( ConductorBaseView ):
    def do_the_things( self ):
        status = self.get_updater_status()
        status[ 'throughstep' ] = self.__class__.throughstep
        status[ 'pickuppartial' ] = self.__class__.pickuppartial
        return status

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

        updaterkw = [ 'instrument', 'timeout', 'updateargs', 'hold', 'pause' ]
        clsatt = { 'instrument': 'instrument_name',
                   'timeout': 'update_timeout',
                   'updateargs': 'updateargs',
                   'hold': 'hold_new_exposures',
                   'pause': 'pause_updates',
                   'throughstep': 'throughstep',
                   'pickuppartial': 'pickuppartial' }
        unknown = set()
        updaterargs = {}
        clsatttoset = {}
        for arg, val in args.items():
            if ( arg not in updaterkw ) and ( arg not in clsatt ):
                unknown.add( arg )
            else:
                if arg in updaterkw:
                    updaterargs[arg] = val
                if arg in clsatt.keys():
                    clsatttoset[arg] = val

        if len(unknown) != 0:
            return f"Unknown arguments to UpdateParameters: {unknown}", 500

        for att, val in clsatttoset.items():
            setattr( self.__class__, att, val )
        # Bools will have been passed as ints through the web interface, so make
        #   sure they're really bools.  (This matters when passing to Postgres.)
        self.__class__.pause_updates = bool( self.__class__.pause_updates )
        self.__class__.hold_new_exposures = bool( self.__class__.hold_new_exposures )
        self.__class__.pickuppartial = bool( self.__class__.pickuppartial )

        updaterargs['command'] = 'updateparameters'
        res = self.talk_to_updater( updaterargs )
        del curstatus['status']
        res['oldsconfig'] = curstatus

        self.__class__.configchangetime = res['configchangetime']
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "UPDATE conductor_config SET instrument_name=%(inst)s, updateargs=%(upda)s, "
                            "                            update_timeout=%(updt)s, pause_updates=%(pause)s, "
                            "                            hold_new_exposures=%(hold)s, configchangetime=%(t)s, "
                            "                            throughstep=%(through)s, pickuppartial=%(partial)s ",
                            { 'inst': res['instrument'],
                              'upda': res['updateargs'],
                              'updt': res['timeout'],
                              'pause': bool( res['pause'] ),
                              'hold': bool( res['hold'] ),
                              't': res['configchangetime'],
                              'through': self.__class__.throughstep,
                              'partial': self.__class__.pickuppartial } )
            conn.commit()

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


class RegisterWorker( ConductorBaseView ):
    def do_the_things( self, argstr=None ):
        args = self.argstr_to_args( argstr, { 'node_id': None, 'replace': 0 } )
        args['replace'] = int( args['replace'] )
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
                    newworker.lastheartbeat = datetime.datetime.now()
                    status = 'updated'
                else:
                    return f"cluster_id {args['cluster_id']} node_id {args['node_id']} already exists", 500

            else:
                newworker = PipelineWorker( cluster_id=args['cluster_id'],
                                            node_id=args['node_id'],
                                            lastheartbeat=datetime.datetime.now() )
                status = 'added'
            session.add( newworker )
            session.commit()
            # Make sure that newworker has the id field loaded
            # session.merge( newworker )
        return { 'status': status,
                 'id': newworker.id,
                 'cluster_id': newworker.cluster_id,
                 'node_id': newworker.node_id }


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
        knownexp_id = None
        with Psycopg2Connection() as dbcon:
            cursor = dbcon.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( "LOCK TABLE knownexposures" )
            # Select the lowest-mjd exposure in the "ready" state (1)
            cursor.execute( "SELECT _id, cluster_id FROM knownexposures "
                            "WHERE cluster_id IS NULL AND _state=1 "
                            "ORDER BY mjd LIMIT 1" )
            rows = cursor.fetchall()
            if len(rows) > 0:
                knownexp_id = rows[0]['_id']
                # Set state to claimed (2), update the claim time and the cluster id
                cursor.execute( "UPDATE knownexposures "
                                "SET cluster_id=%(cluster_id)s, claim_time=NOW(), state=2 "
                                "WHERE _id=%(id)s",
                                { 'id': knownexp_id, 'cluster_id': args['cluster_id'] } )
                cursor.execute( "SELECT throughstep FROM conductor_config" )
                throughstep = cursor.fetchone()[ 'throughstep' ]
                dbcon.commit()

        if knownexp_id is not None:
            return { 'status': 'available',
                     'knownexposure_id': knownexp_id,
                     'through_step': throughstep
                    }
        else:
            return { 'status': 'not available' }


# ======================================================================
# /getknownexposures

class GetKnownExposures( ConductorBaseView ):
    def do_the_things( self, argstr=None ):
        args = self.argstr_to_args( argstr, { "minmjd": None,
                                              "maxmjd": None,
                                              "instrument": None,
                                              "target": None,
                                              "filter": None,
                                              "project": None,
                                              "minexptime": None,
                                              "state": None,
                                              "maxclaimtime": None
                                             } )
        args['minmjd'] = float( args['minmjd'] ) if args['minmjd'] is not None else None
        args['maxmjd'] = float( args['maxmjd'] ) if args['maxmjd'] is not None else None
        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            q = ( "SELECT ke.*,e.filepath FROM knownexposures ke "
                  "LEFT JOIN exposures e ON ke.exposure_id=e._id " )
            _and = "WHERE"
            subdict = {}

            if args['minmjd'] is not None:
                q += f"{_and} mjd >= %(minmjd)s "
                subdict['minmjd'] = float( args['minmjd'] )
                _and = "AND"
            if args['maxmjd'] is not None:
                q += f"{_and} mjd <= %(maxmjd)s "
                subdict['maxmjd'] = float( args['maxmjd'] )
                _and = "AND"
            if args['instrument'] is not None:
                q += f"{_and} instrument = %(instr)s "
                subdict['instr'] = args['instrument']
                _and = "AND"
            if args['target'] is not None:
                q += f"{_and} target = %(target)s "
                subdict['target'] = args['target']
                _and = "AND"
            if args['project'] is not None:
                q += f"{_and} project = %(project)s "
                subdict['project'] = args['project']
                _and = "AND"
            if args['minexptime'] is not None:
                q += f"{_and} exp_time >= %(minexp)s "
                subdict['minexp'] = float( args['minexptime'] )
                _and = "AND"
            if args['state'] is not None:
                q += f"{_and} _state IN %(state)s"
                subdict['state'] = tuple( KnownExposureStateConverter.to_int( s ) for s in args['state'].split(",") )
            if args['maxclaimtime'] is not None:
                claimtime = datetime.datetime.fromisoformat( args['maxclaimtime'] )
                if claimtime.tzinfo is None:
                    claimtime = pytz.utc.localize( claimtime )
                q += f"{_and} claim_time <= %(maxclaimtime)s "
                subdict['maxclaimtime'] = claimtime
                _and = "AND"
            q += "ORDER BY mjd "

            cursor.execute( q, subdict )
            rows = cursor.fetchall()

        retval = { 'status': 'ok',
                   'knownexposures': rows }
        # Add the "id" field that's the same as "_id" for convenience,
        #   make the filter the short name, convert "_state" to a string
        #   in "state", and strip off all but the filename of the
        #   filepath
        for ke in retval['knownexposures']:
            ke['id'] = ke['_id']
            ke['filter'] = get_instrument_instance( ke['instrument'] ).get_short_filter_name( ke['filter'] )
            ke['state'] = KnownExposureStateConverter.to_string( ke['_state'] )
            if ke['filepath'] is not None:
                ke['filepath'] = pathlib.Path( ke['filepath'] ).name
        # We didn't search by filter because we want to make sure we're letting the user
        #   specify short filter names.  Filter by filter now.
        if args['filter'] is not None:
            retval['knownexposures'] = [ ke for ke in retval['knownexposures'] if ke['filter'] == args['filter'] ]
        return retval


# ======================================================================

class SetKnownExposureState( ConductorBaseView ):
    def do_the_things( self ):
        args = self.argstr_to_args( None, { 'knownexposure_ids': [],
                                            'state': None } )
        if ( args['state'] is None ):
            raise ValueError( "must specify state" )
        state = KnownExposureStateConverter.to_int( args['state'] )
        if state not in (0, 1, 2, 3, 4):
            raise ValueError( "state must be one of held, ready, claimed, running, or done, "
                              "or an int in [0, 1, 2, 3, 4]" )
        if not isinstance( args['knownexposure_ids'], list ):
            raise TypeError( "knownexposure_ids must be a list" )
        if len( args['knownexposure_ids'] ) == 0:
            raise ValueError( "Must have at least one in knownexposure_ids to do anything" )

        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "UPDATE knownexposures SET _state=%(state)s WHERE _id IN %(ids)s",
                            { 'state': state, 'ids': tuple( args['knownexposure_ids'] ) } )
            conn.commit()

        return { 'status': 'ok',
                 'state': KnownExposureStateConverter.to_string( state ),
                 'knownexposure_ids': args['knownexposure_ids'] }


# ======================================================================

class DeleteKnownExposures( ConductorBaseView ):
    def do_the_things( self ):
        args = flask.request.json
        if 'knownexposure_ids' not in args:
            return "Error, must pass knownexposure_ids in JSON post body", 500
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "DELETE FROM knownexposures WHERE _id IN %(expids)s",
                            { 'expids': tuple( args['knownexposure_ids'] ) } )
            ndel = cursor.rowcount
            conn.commit()
            return { 'status': 'ok', 'num_deleted': ndel }


# ======================================================================

class FullyClearClusterClaim( ConductorBaseView ):
    def do_the_things( self ):
        args = flask.request.json
        if 'knownexposure_ids' not in args:
            return "Error, must pass knownexposure_ids in JSON post body", 500
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "UPDATE knownexposures SET cluster_id=NULL, node_id=NULL, machine_name=NULL, "
                            "  claim_time=NULL, start_time=NULL, release_time=NULL "
                            "WHERE _id IN %(expids)s",
                            { 'expids': tuple( args['knownexposure_ids'] ) } )
            nmod = cursor.rowcount
            conn.commit()
            return { 'status': 'ok', 'num_cleared': nmod }


# ======================================================================
# Do initialization; create and configure the sub web ap (i.e. flask blueprint)

ConductorBaseView.restore_conductor_state()

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
    "/setknownexposurestate": SetKnownExposureState,
    "/deleteknownexposures": DeleteKnownExposures,
    "/fullyclearclusterclaim": FullyClearClusterClaim,
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
