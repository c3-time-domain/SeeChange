import sys
import pathlib
import socket
import json
import datetime
import pytz
import traceback
import textwrap
import uuid

import psycopg
import psycopg.types.json
from psycopg import sql

import flask

from util.util import asUUID
from models.base import PsycopgConnection, PGDB
from models.enums_and_bitflags import KnownExposureStateConverter, ImageTypeConverter
from models.instrument import Instrument

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

        with PsycopgConnection() as conn:
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
        status[ 'throughstep' ] = ConductorBaseView.throughstep
        status[ 'pickuppartial' ] = ConductorBaseView.pickuppartial
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
            return f"Unknown arguments to UpdateParameters: {unknown}", 422

        for att, val in clsatttoset.items():
            setattr( ConductorBaseView, att, val )
        # Bools will have been passed as ints through the web interface, so make
        #   sure they're really bools.  (This matters when passing to Postgres.)
        ConductorBaseView.pause_updates = bool( ConductorBaseView.pause_updates )
        ConductorBaseView.hold_new_exposures = bool( ConductorBaseView.hold_new_exposures )
        ConductorBaseView.pickuppartial = bool( ConductorBaseView.pickuppartial )

        updaterargs['command'] = 'updateparameters'
        res = self.talk_to_updater( updaterargs )
        del curstatus['status']
        res['oldsconfig'] = curstatus

        ConductorBaseView.configchangetime = res['configchangetime']
        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            cursor.execute( "UPDATE conductor_config SET instrument_name=%(inst)s, updateargs=%(upda)s, "
                            "                            update_timeout=%(updt)s, pause_updates=%(pause)s, "
                            "                            hold_new_exposures=%(hold)s, configchangetime=%(t)s, "
                            "                            throughstep=%(through)s, pickuppartial=%(partial)s ",
                            { 'inst': res['instrument'],
                              'upda': psycopg.types.json.Jsonb(res['updateargs']),
                              'updt': res['timeout'],
                              'pause': bool( res['pause'] ),
                              'hold': bool( res['hold'] ),
                              't': res['configchangetime'],
                              'through': ConductorBaseView.throughstep,
                              'partial': ConductorBaseView.pickuppartial } )
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
            return "cluster_id is required for registerworker", 422
        with PGDB( dictcursor=True ) as pgdb:
            q = sql.SQL( "SELECT * FROM pipelineworkers WHERE cluster_id={cluster} AND node_id={node}"
                        ).format( cluster=args['cluster_id'], node=args['node_id'] )
            rows = pgdb.execute( q )
            newworker = None
            status = None
            if len( rows ) > 0:
                if len( rows ) > 1:
                    return ( f"cluster_id {args['cluster_id']} node_id{args['node_id']} multiply defined, "
                             f"database needs to be cleaned up" ), 422
                if args['replace']:
                    newworker = rows[0]
                    q = sql.SQL( "UPDATE pipelineworkers SET lastheartbeat={now}"
                                ).format( datetime.datetime.now( tz=datetime.UTC ) )
                    pgdb.execute_nofetch( q )
                    status = 'updated'
                else:
                    return f"cluster_id {args['cluster_id']} node_id {args['node_id']} already exists", 422

            else:
                newid = uuid.uuid4()
                q = sql.SQL( textwrap.dedent(
                    """
                    INSERT INTO pipelineworkers(_id, cluster_id, node_id, lastheartbeat)
                    VALUES ({newid}, {cluster_id}, {node_id}, {lastheartbeat})
                    """
                ) ).format( newid=newid,
                            cluster_id=args['cluster_id'],
                            node_id=args['node_id'],
                            lastheartbeat=datetime.datetime.now( tz=datetime.UTC ) )
                pgdb.execute_nofetch( q )
                newworker = { '_id': newid, 'cluster_id': args['cluster_id'], 'node_id': args['node_id'] }
                status = 'added'
            if status in ( 'updated', 'added' ):
                pgdb.commit()

        return { 'status': status,
                 'id': newworker['_id'],
                 'cluster_id': newworker['cluster_id'],
                 'node_id': newworker['node_id'] }


# ======================================================================
# /unregisterworker
#
# Remove a Pipeline Worker registration.  Call with /unregsiterworker/n
# where n is the integer ID of the pipeline worker.

class UnregisterWorker( ConductorBaseView ):
    def do_the_things( self, pipelineworker_id ):
        pipelineworker_id = asUUID( pipelineworker_id )
        with PGDB( dictcursor=True ) as pgdb:
            q = sql.SQL( "SELECT * FROM pipelineworkers WHERE _id={pwid}" ).format( pwid=pipelineworker_id )
            rows = pgdb.execute( q )
            if len(rows) == 0:
                return f"Unknown pipeline worker {pipelineworker_id}", 422
            else:
                q = sql.SQL( "DELETE FROM pipelineworkers WHERE _id={pwid}" ).format( pwid=pipelineworker_id )
                pgdb.execute_nofetch( q )
                pgdb.commit()
        return { "status": "worker deleted" }


# ======================================================================
# /workerheartbeat
#
# Call at /workerheartbeat/n where n is the uuid of the pipeline worker

class WorkerHeartbeat( ConductorBaseView ):
    def do_the_things( self, pipelineworker_id ):
        pipelineworker_id = asUUID( pipelineworker_id )
        with PGDB( dictcursor=True ) as pgdb:
            q = sql.SQL( "SELECT * FROM pipelineworkers WHERE _id={pwid}" ).format( pwid=pipelineworker_id )
            rows = pgdb.execute( q )
            if len(rows) == 0:
                return f"Unknown pipelineworker {pipelineworker_id}", 422
            q = sql.SQL( "UPDATE pipelineworkers SET lastheartbeat={now} WHERE _id={pwid}"
                        ).format( pwid=pipelineworker_id, now=datetime.datetime.now( tz=datetime.UTC ) )
            pgdb.execute_nofetch( q )
            pgdb.commit()
        return { 'status': 'updated' }

# ======================================================================
# /getworkers


class GetWorkers( ConductorBaseView ):
    def do_the_things( self ):
        with PGDB( dictcursor=True ) as pgdb:
            rows = pgdb.execute( "SELECT * FROM pipelineworkers" )
        return { 'status': 'ok',
                 'workers': rows }

# ======================================================================
# /requestexposure


class RequestExposure( ConductorBaseView ):
    def do_the_things( self, argstr=None ):
        args = self.argstr_to_args( argstr )
        if 'cluster_id' not in args.keys():
            return "cluster_id is required for RequestExposure", 422
        if 'types' in args:
            types = args['types'].split( "," )
            if 'all' in [ t.lower() for t in types ]:
                types = list( ImageTypeConverter.dict.keys() )
            else:
                types = [ ImageTypeConverter.to_int( t ) for t in types ]
        else:
            types = [ ImageTypeConverter.to_int( 'Unknown' ), ImageTypeConverter.to_int( 'Sci' ) ]

        # ****
        flask.current_app.logger.debug( f"RequestExposure got argstr={argstr}\n...parsed to {args}\n" )
        # ****

        knownexp_id = None
        with PsycopgConnection() as dbcon:
            cursor = dbcon.cursor( row_factory=psycopg.rows.dict_row )
            cursor.execute( "LOCK TABLE knownexposures" )

            # Select the lowest-mjd exposure in the "ready" state (1)
            q = sql.SQL( "SELECT _id, cluster_id FROM knownexposures\n"
                         "WHERE _state=1\n"
                         "  AND _type=ANY(%(types)s)\n" )
            subdict = { 'types': types }

            if 'instrument' in args:
                q += sql.SQL( "  AND instrument=%(instr)s\n" )
                subdict['instr'] = args['instrument']

            # ****
            flask.current_app.logger.debug( f"Sending query:\n{q}\n...args {subdict}" )
            # ****

            q += sql.SQL( "ORDER BY mjd" )
            cursor.execute( q, subdict )
            rows = cursor.fetchall()
            if len(rows) > 0:
                knownexp_id = rows[0]['_id']
                # Set state to claimed (2), update the claim time and the cluster id
                cursor.execute( "UPDATE knownexposures "
                                "SET cluster_id=%(cluster_id)s, claim_time=NOW(), start_time=NULL, release_time=NULL, "
                                "    _state=2, node_id=NULL, machine_name=NULL "
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
                                              "maxclaimtime": None,
                                              "provtag": None,
                                              "types": None,
                                              "ra": None,
                                              "dec": None,
                                              "radius": None
                                             } )
        args['minmjd'] = float( args['minmjd'] ) if args['minmjd'] is not None else None
        args['maxmjd'] = float( args['maxmjd'] ) if args['maxmjd'] is not None else None


        if args['provtag'] is None:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT ke.*,
                       NULL as matched_exposure_id,
                       NULL as exp_filename,
                       NULL as nimg,
                       NULL as nsrc,
                       NULL as nwcs,
                       NULL as nzp,
                       NULL as nsub,
                       NULL as ngooddets,
                       NULL as ndets
                FROM knownexposures ke
                """
            ) )

        else:
            # Check this out
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT ke.*,
                       e._id AS matched_exposure_id,
                       substring( e.filepath FROM '/?([^/]+)$' ) AS exp_filename,
                       SUM( CASE WHEN i._id IS NULL THEN 0 ELSE 1 END ) AS nimg,
                       SUM( nsrc ) AS nsrc, SUM( nwcs ) AS nwcs, SUM( nzp ) AS nzp, SUM( nsub ) AS nsub,
                       SUM( ngooddets ) AS ngooddets, SUM( ndets ) AS ndets
                FROM knownexposures ke
                LEFT JOIN (
                   SELECT DISTINCT ON(e._id) e._id, e.filepath, e.origin_identifier FROM exposures e
                   INNER JOIN provenance_tags t ON e.provenance_id=t.provenance_id
                                               AND t.tag={provtag}
                ) e ON e.origin_identifier=ke.identifier
                LEFT JOIN (
                   SELECT i._id, i.exposure_id,
                          SUM( CASE WHEN s._id IS NULL THEN 0 ELSE 1 END ) as nsrc,
                          SUM( nwcs ) AS nwcs, SUM( nzp ) AS nzp, SUM( nsub ) AS nsub,
                          SUM( ngooddets ) AS ngooddets, SUM( ndets ) AS ndets
                   FROM (
                     SELECT DISTINCT ON (i._id) i._id, i.exposure_id
                     FROM images i
                     INNER JOIN provenance_tags t ON i.provenance_id=t.provenance_id AND t.tag={provtag}
                   ) i
                   LEFT JOIN (
                      SELECT s._id, s.image_id,
                             SUM( CASE WHEN w._id IS NULL THEN 0 ELSE 1 END ) as nwcs,
                             SUM( nzp ) AS nzp, SUM( nsub ) AS nsub,
                             SUM( ngooddets ) AS ngooddets,
                             SUM( ndets ) AS ndets
                      FROM (
                        SELECT DISTINCT ON (s._id) s._id, s.image_id
                        FROM source_lists s
                        INNER JOIN provenance_tags t ON s.provenance_id=t.provenance_id AND t.tag={provtag}
                      ) s
                      LEFT JOIN (
                         SELECT w._id, w.sources_id,
                                SUM( CASE WHEN z._id IS NULL THEN 0 ELSE 1 END ) as nzp,
                                SUM( nsub ) as nsub, SUM( ngooddets ) AS ngooddets, SUM( ndets ) AS ndets
                         FROM (
                           SELECT DISTINCT ON (w._id) w._id, w.sources_id
                           FROM world_coordinates w
                           INNER JOIN provenance_tags t ON w.provenance_id=t.provenance_id AND t.tag={provtag}
                         ) w
                         LEFT JOIN (
                            SELECT DISTINCT ON (z._id ) z._id, z.wcs_id,
                                   SUM( CASE WHEN sub._id IS NULL THEN 0 ELSE 1 END ) as nsub,
                                   SUM( sub.ngooddets ) AS ngooddets, SUM( sub.ndets ) as ndets
                            FROM zero_points z
                            INNER JOIN provenance_tags t ON z.provenance_id=t.provenance_id AND t.tag={provtag}
                            LEFT JOIN (
                               SELECT DISTINCT ON (sub._id) sub._id, isc.new_zp_id,
                                      SUM( CASE WHEN mset.msetid IS NULL THEN 0 ELSE ngooddets END ) AS ngooddets,
                                      SUM( CASE WHEN mset.msetid IS NULL THEN 0 ELSE ndets END ) AS ndets
                               FROM images sub
                               INNER JOIN provenance_tags t ON sub.provenance_id=t.provenance_id AND t.tag={provtag}
                               INNER JOIN image_subtraction_components isc ON isc.image_id=sub._id
                               LEFT JOIN (
                                  SELECT DISTINCT ON(mset._id) s.image_id AS subid, mset._id AS msetid,
                                                               COUNT( m._id ) AS ndets,
                                                               SUM( CASE WHEN m.is_bad THEN 0 ELSE 1 END ) as ngooddets
                                  FROM source_lists s
                                  INNER JOIN cutouts c ON c.sources_id=s._id
                                  INNER JOIN measurement_sets mset ON mset.cutouts_id=c._id
                                  INNER JOIN provenance_tags t ON mset.provenance_id=t.provenance_id AND t.tag={provtag}
                                  INNER JOIN measurements m ON m.measurementset_id=mset._id
                                  GROUP BY s.image_id, mset._id
                              ) mset ON mset.subid=sub._id
                              GROUP BY sub._id, isc.new_zp_id
                           ) sub ON sub.new_zp_id=z._id
                           GROUP BY z._id, z.wcs_id
                         ) z ON z.wcs_id=w._id
                         GROUP BY w._id, w.sources_id
                      ) w ON w.sources_id=s._id
                      GROUP BY s._id, s.image_id
                   ) s ON s.image_id=i._id
                   GROUP BY i._id, i.exposure_id
                ) i ON i.exposure_id=e._id
                """
            ) ).format( provtag=args['provtag'] )

        _and = sql.SQL( "WHERE" )

        for minarg in [ 'mjd', 'exptime' ]:
            if args[f'min{minarg}'] is not None:
                q += sql.SQL( "{_and} ke.{field} >= {val}\n"
                             ).format( _and=_and, field=sql.Identifier(minarg), val=args[f'min{minarg}'] )
                _and = sql.SQL( "  AND" )
        for eqarg in [ 'instrument', 'target', 'project' ]:
            if args[eqarg] is not None:
                q += sql.SQL( "{_and} ke.{field} = {val}\n"
                              ).format( _and=_and, field=sql.Identifier(eqarg), val=args[eqarg] )
                _and = sql.SQL( "  AND" )

        if args['maxmjd'] is not None:
            q += sql.SQL( "{_and} ke.mjd <= {maxmjd}\n" ).format( _and=_and, maxmjd=float(args['maxmjd']) )
            _and = sql.SQL( "  AND" )
        if args['maxclaimtime'] is not None:
            claimtime = datetime.datetime.fromisoformat( args['maxclaimtime'] )
            if claimtime.tzinfo is None:
                claimtime = pytz.utc.localize( claimtime )
            q += sql.SQL( "{_and} ke.claim_time <= {t}\n" ).format( _and=_and, t=claimtime )
            _and = sql.SQL( "  AND" )
        if args['state'] is not None:
            q += sql.SQL( "{_and} ke._state=ANY(ARRAY[{state}])\n"
                         ).format( _and=_and,
                                   state=sql.SQL(",").join(
                                       KnownExposureStateConverter.to_int( s ) for s in args['state'].split(",") )
                                  )
            _and = sql.SQL( "  AND" )
        if args['types'] is not None:
            types = args['types'].split(",")
            if "all" not in [ t.lower() for t in types ]:
                types = [ ImageTypeConverter.to_int( t ) for t in types ]
                q += sql.SQL( "{_and} ke._type=ANY(ARRAY[{types}])\n"
                             ).format( _and=_and,
                                       types=sql.SQL(",").join( types ) )
                _and = sql.SQL( "  AND" )
        if any( args[a] is not None for a in ('ra', 'dec', 'radius') ):
            if not all( args[a] is not None for a in ('ra', 'dec', 'radius') ):
                raise RuntimeError( "If you give any of ra, dec, radius, must give all." )
            try:
                ra = float( args['ra'] )
                dec = float( args['dec'] )
                radius = float( args['radius'] )
            except Exception:
                raise RuntimeError( f'ra, dec, and radius must all be floats, '
                                    f'got ("{args["ra"]}", "{args["dec"]}", "{args["radius"]}")' )
            if ( ra < 0 ) or ( ra >= 360 ) or ( dec < -90 ) or ( dec > 90 ):
                raise RuntimeError( f"Invalid ra, dec ({ra:.4f}, {dec:.4f})" )
            if radius <= 0:
                raise RuntimeError( f"radius must be positive, got {radius:.4f}" )
            if radius > 20.:
                raise RuntimeError( f"radius must be ≤ 20°, got {radius:.4f}" )
            q += sql.SQL( "{_and} q3c_radial_query(ke.ra,ke.dec,{ra},{dec},{radius})"
                         ).format( ra=ra, dec=dec, radius=radius, _and=_and )
            _and = sql.SQL( "  AND" )

        if args['provtag'] is not None:
            q += sql.SQL( "GROUP BY ke._id, e._id, e.filepath\n" )
        q += sql.SQL( "ORDER BY ke.mjd" )

        flask.current_app.logger.debug( f"Sending query:\n{q.as_string()}" )
        with PGDB( dictcursor=True ) as pgdb:
            rows = pgdb.execute( q )

        # OMGTOOMUCH
        # import io
        # import pprint
        # strio = io.StringIO()
        # pprint.pp( rows, stream=strio )
        # flask.current_app.logger.debug( f"Return from query:\n{strio.getvalue()}" )
        # OMGTOOMUCH

        retval = { 'status': 'ok',
                   'knownexposures': rows }
        # Add the "id" field that's the same as "_id" for convenience,
        #   make the filter the short name, convert "_state" to a string
        #   in "state", "_type" to a string in "type".
        for ke in retval['knownexposures']:
            ke['id'] = ke['_id']
            ke['state'] = KnownExposureStateConverter.to_string( ke['_state'] )
            ke['type'] = ImageTypeConverter.to_string( ke['_type'] )
            ke['filter'] = Instrument.get_instrument_instance( ke['instrument'] ).get_short_filter_name( ke['filter'] )
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

        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            cursor.execute( "UPDATE knownexposures SET _state=%(state)s WHERE _id=ANY(%(ids)s)",
                            { 'state': state, 'ids': args['knownexposure_ids'] } )
            conn.commit()

        return { 'status': 'ok',
                 'state': KnownExposureStateConverter.to_string( state ),
                 'knownexposure_ids': args['knownexposure_ids'] }


# ======================================================================

class DeleteKnownExposures( ConductorBaseView ):
    def do_the_things( self ):
        args = flask.request.json
        if 'knownexposure_ids' not in args:
            return "Error, must pass knownexposure_ids in JSON post body", 422
        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            cursor.execute( "DELETE FROM knownexposures WHERE _id=ANY(%(expids)s)",
                            { 'expids': args['knownexposure_ids'] } )
            ndel = cursor.rowcount
            conn.commit()
            return { 'status': 'ok', 'num_deleted': ndel }


# ======================================================================

class FullyClearClusterClaim( ConductorBaseView ):
    def do_the_things( self ):
        args = flask.request.json
        if 'knownexposure_ids' not in args:
            return "Error, must pass knownexposure_ids in JSON post body", 422
        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            cursor.execute( "UPDATE knownexposures SET cluster_id=NULL, node_id=NULL, machine_name=NULL, "
                            "  claim_time=NULL, start_time=NULL, release_time=NULL "
                            "WHERE _id=ANY(%(expids)s)",
                            { 'expids': args['knownexposure_ids'] } )
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
