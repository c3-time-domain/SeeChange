import sys
import io
import pathlib
import base64

import astropy.visualization
import flask
import h5py
import numpy as np
import PIL
import psycopg2
import psycopg2.extras

from util.config import Config
from util.util import asUUID
from util.logger import SCLogger
from models.base import Psycopg2Connection
# NOTE: for get_instrument_instrance to work, must manually import all
#  known instrument classes we might want to use here.
# If models.instrument gets imported somewhere else before this file
#  is imported, then even this won't work.  There must be a better way....
import models.decam  # noqa: F401
from models.instrument import get_instrument_instance

sys.path.insert( 0, pathlib.Path(__name__).resolve().parent )
from baseview import BaseView


# ======================================================================

class BaseLtcvView( BaseView ):
    def _get_objid( self, objid_or_name, dbcon=None ):
        objid = None
        objname = None
        try:
            objid = asUUID( objid_or_name )
        except ValueError:
            objname = objid_or_name

        with Psycopg2Connection( dbcon ) as conn:
            cursor = conn.cursor()
            if objid is not None:
                cursor.execute( "SELECT _id FROM objects WHERE _id=%(id)s", { 'id': objid } )
            elif objname is not None:
                cursor.execute( "SELECT _id FROM objects WHERE name=%(name)s", { 'name': objname } )
            else:
                raise RuntimeError( "This should never happen." )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError( f"Unknown object {objid_or_name}" )
            return asUUID( row[0] )


# ======================================================================

class Ltcv( BaseLtcvView ):
    def dispatch_request( self, argstr=None ):
        args = self.argstr_to_args( argstr, { 'objid': "", 'provtag': "default", 'zp': 31.4, 'zpunits': 'nJy' } )
        return flask.render_template( "ltcv.html",
                                      objid=args['objid'],
                                      provtag=args['provtag'],
                                      zp=args['zp'],
                                      zpunits=args['zpunits'] )


# ======================================================================

class ObjectInfo( BaseLtcvView ):
    def do_the_things( self, objid_or_name ):
        with Psycopg2Connection() as conn:
            objuuid = self._get_objid( objid_or_name, dbcon=conn )
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( "SELECT _id,name,ra,dec FROM objects WHERE _id=%(id)s", { 'id': objuuid } )
            rows = cursor.fetchall()
            if len(rows) == 0:
                return f"Error: unknown object {objid_or_name}", 500
            if len(rows) > 1:
                return f"Database corruption error: multiple objects {objid_or_name}!", 500

            return { 'status': 'ok',
                     'id': rows[0]['_id'],
                     'name': rows[0]['name'],
                     'ra': rows[0]['ra'],
                     'dec': rows[0]['dec'] }


# ======================================================================

class ObjectLtcv( BaseLtcvView ):
    def do_the_things( self, objid_or_name, provtag, argstr=None ):
        args = self.argstr_to_args( argstr, { 'zp': 31.4,
                                              'zpunits': 'nJy' } )
        with Psycopg2Connection() as conn:
            objid = self._get_objid( objid_or_name, dbcon=conn )
            cursor = conn.cursor()
            cursor.execute( "SELECT m._id,subim.instrument,subim.mjd,subim.filter,m.flux_psf,m.flux_psf_err,z.zp "
                            "FROM measurements m "
                            "INNER JOIN measurement_sets ms ON m.measurementset_id=ms._id "
                            "INNER JOIN cutouts cu ON ms.cutouts_id=cu._id "
                            "INNER JOIN source_lists subsl ON cu.sources_id=subsl._id "
                            "INNER JOIN images subim ON subsl.image_id=subim._id "
                            "INNER JOIN image_subtraction_components isc ON subim._id=isc.image_id "
                            "INNER JOIN zero_points z ON isc.new_zp_id=z._id "
                            "INNER JOIN provenance_tags pt ON ms.provenance_id=pt.provenance_id "
                            "WHERE pt.tag=%(provtag)s AND m.object_id=%(objid)s "
                            "ORDER BY subim.mjd,subim.filter",
                            { 'provtag': provtag, 'objid': objid } )
            columns = { cursor.description[i][0]:i for i in range( len(cursor.description) ) }
            rows = cursor.fetchall()

        # Get a list of instruments and filters, so the two together can be used for sorting.
        unknown_filts = set()
        instrs = []
        filts = []
        known_instrs = set( r[columns['instrument']] for r in rows )
        for instr in known_instrs:
            instrobj = get_instrument_instance( instr )
            curfilts = set( r[columns['filter']] for r in rows if r[columns['instrument']]==instr )
            missing = curfilts - set( instrobj.allowed_filters )
            for m in missing:
                unknown_filts.add( f'{instr}:{m}' )
            instrs.extend( instr for f in instrobj.allowed_filters if f in curfilts )
            filts.extend( f for f in instrobj.allowed_filters if f in curfilts )

        # Factor to convert ADU to nJy
        zpfac = [ 10**( ( r[columns['zp']] - float( args['zp'] ) ) / ( -2.5 ) ) for r in rows ]

        rval = { 'status': 'ok',
                 'objid': objid,
                 'provtag': provtag,
                 'flux_zp': args['zp'],
                 'flux_unit': args['zpunits'],
                 'unknown_filters': list(unknown_filts),
                 'instruments': instrs,
                 'filters': filts,
                 'measid': [ r[columns['_id']] for r in rows ],
                 'mjd': [ r[columns['mjd']] for r in rows ],
                 'instrument': [ r[columns['instrument']] for r in rows ],
                 'filter': [ r[columns['filter']] for r in rows ],
                 'flux_psf': [ r[columns['flux_psf']]*zf for r,zf in zip(rows,zpfac) ],
                 'flux_psf_err': [ r[columns['flux_psf_err']]*zf for r,zf in zip(rows,zpfac) ]
                }
        SCLogger.info( f"Returning {rval}" )
        return rval


# ======================================================================

class ObjectCutouts( BaseLtcvView ):
    def do_the_things( self, objid_or_name, provtag ):
        cfg = Config.get()
        with Psycopg2Connection() as conn:
            objid = self._get_objid( objid_or_name, dbcon=conn )
            cursor = conn.cursor()
            cursor.execute( "SELECT cu.filepath AS cutoutfilepath, "
                            "       m._id AS measid, m.index_in_sources, "
                            "       i.filter, i.mjd, i.filepath AS imgfilepath "
                            "FROM measurements m "
                            "INNER JOIN measurement_sets ms ON m.measurementset_id=ms._id "
                            "INNER JOIN cutouts cu ON ms.cutouts_id=cu._id "
                            "INNER JOIN source_lists subsl ON cu.sources_id=subsl._id "
                            "INNER JOIN images subim ON subsl.image_id=subim._id "
                            "INNER JOIN image_subtraction_components isc ON subim._id=isc.image_id "
                            "INNER JOIN zero_points z ON isc.new_zp_id=z._id "
                            "INNER JOIN world_coordinates w ON z.wcs_id=w._id "
                            "INNER JOIN source_lists s ON w.sources_id=s._id "
                            "INNER JOIN images i ON s.image_id=i._id "
                            "INNER JOIN provenance_tags pt ON ms.provenance_id=pt.provenance_id "
                            "WHERE pt.tag=%(provtag)s AND m.object_id=%(objid)s "
                            "ORDER BY i.mjd,i.filter",
                            { 'provtag': provtag, 'objid': objid } )
            columns = { cursor.description[i][0]:i for i in range( len(cursor.description) ) }
            rows = cursor.fetchall()

        retval = { 'status': 'ok',
                   'cutouts': {
                       'imgfilepath': [],
                       'measid': [],
                       'filter': [],
                       'mjd': [],
                       'w': [],
                       'h': [],
                       'new_png': [],
                       'ref_png': [],
                       'sub_png': []
                   }
                  }

        scaler = astropy.visualization.ZScaleInterval( contrast=0.02 )

        for row in rows:
            with h5py.File( ( pathlib.Path( cfg.value( 'archive.local_read_dir' ) )
                              / row[columns['cutoutfilepath']] ), 'r' ) as h5:
                grp = h5[ f'source_index_{row[columns["index_in_sources"]]}' ]
                vmin, vmax = scaler.get_limits( grp['new_data'] )
                scalednew = ( grp['new_data'] - vmin ) * 255. / ( vmax - vmin )
                scaledref = ( grp['ref_data'] - vmin ) * 255. / ( vmax - vmin )
                vmin, vmax = scaler.get_limits( grp['sub_data'] )
                scaledsub = ( grp['sub_data'] - vmin ) * 255. / ( vmax - vmin )

            scalednew[ scalednew < 0 ] = 0
            scalednew[ scalednew > 255 ] = 0
            scaledref[ scaledref < 0 ] = 0
            scaledref[ scaledref > 255 ] = 255
            scaledsub[ scaledsub < 0 ] = 0
            scaledsub[ scaledsub > 255 ] = 255

            scalednew = np.array( scalednew, dtype=np.uint8 )
            scaledref = np.array( scaledref, dtype=np.uint8 )
            scaledsub = np.array( scaledsub, dtype=np.uint8 )

            # Flip images vertically.  In DS9 and with FITS images,
            #   we call the lower-left pixel (0,0).  Images on
            #   web browsers call the upper-left pixel (0,0).
            #   Flipping vertically will make it display the same
            #   on the web browser as it will in DS9

            newim = io.BytesIO()
            refim = io.BytesIO()
            subim = io.BytesIO()
            PIL.Image.fromarray( scalednew ).transpose( PIL.Image.FLIP_TOP_BOTTOM ).save( newim, format='png' )
            PIL.Image.fromarray( scaledref ).transpose( PIL.Image.FLIP_TOP_BOTTOM ).save( refim, format='png' )
            PIL.Image.fromarray( scaledsub ).transpose( PIL.Image.FLIP_TOP_BOTTOM ).save( subim, format='png' )

            retval['cutouts']['imgfilepath'].append( row[ columns['imgfilepath'] ] )
            retval['cutouts']['measid'].append( str( row[ columns['measid'] ] ) )
            retval['cutouts']['filter'].append( row[ columns['filter'] ] )
            retval['cutouts']['mjd'].append( row[ columns['mjd'] ] )
            retval['cutouts']['w'].append( scalednew.shape[1] )
            retval['cutouts']['h'].append( scalednew.shape[0] )
            retval['cutouts']['new_png'].append( base64.b64encode( newim.getvalue() ).decode( 'ascii' ) )
            retval['cutouts']['ref_png'].append( base64.b64encode( refim.getvalue() ).decode( 'ascii' ) )
            retval['cutouts']['sub_png'].append( base64.b64encode( subim.getvalue() ).decode( 'ascii' ) )

        return retval


# ======================================================================

bp = flask.Blueprint( 'ltcv', __name__, url_prefix='/ltcv' )

urls = {
    '/': Ltcv,
    '/<path:argstr>': Ltcv,
    '/objectinfo/<objid_or_name>': ObjectInfo,
    '/objectltcv/<objid_or_name>/<provtag>': ObjectLtcv,
    '/objectltcv/<objid_or_name>/<provtag>/<path:argstr>': ObjectLtcv,
    '/objectcutouts/<objid_or_name>/<provtag>': ObjectCutouts,
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
