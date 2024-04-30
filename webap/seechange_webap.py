# Put this first so we can be sure that there are no calls that subvert
#  this in other includes.
import matplotlib
matplotlib.use( "Agg" )
# matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# matplotlib.rc('text', usetex=True)  #  Need LaTeX in Dockerfile
from matplotlib import pyplot

# TODO : COUNT(DISTINCT()) can be slow, deal with this if necessary
#   I'm hoping that since they all show up inside a group and the
#   total number of things I expect to have to distinct on within each group is
#   not likely to more than ~10^2, it won't matter.

import sys
import traceback
import math
import io
import re
import json
import pathlib
import logging
import base64

import psycopg2
import psycopg2.extras
import numpy
import h5py
import PIL
import astropy.time
import astropy.visualization

import flask

# Read the database config

from seechange_webap_config import PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_NAME, ARCHIVE_DIR

# Figure out where we are

workdir = pathlib.Path(__name__).resolve().parent

# Create the flask app, which is what gunicorn is going to look for

app = flask.Flask( __name__, instance_relative_config=True )

# app.logger.setLevel( logging.INFO )
app.logger.setLevel( logging.DEBUG )

# **********************************************************************

def dbconn():
    conn = psycopg2.connect( host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, dbname=PG_NAME )
    yield conn
    conn.rollback()
    conn.close()

# **********************************************************************

@app.route( "/", strict_slashes=False )
def mainpage():
    return flask.render_template( "seechange_webap.html" )

# **********************************************************************

@app.route( "/exposures", methods=['POST'], strict_slashes=False )
def exposures():
    try:
        data = { 'startdate': None,
                 'enddate': None
                }
        if flask.request.is_json:
            data.update( flask.request.json )

        app.logger.debug( f"After parsing, data = {data}" )
        t0 = None if data['startdate'] is None else astropy.time.Time( data['startdate'], format='isot' ).mjd
        t1 = None if data['enddate'] is None else astropy.time.Time( data['enddate'], format='isot' ).mjd
        app.logger.debug( f"t0 = {t0}, t1 = {t1}" )

        conn = next( dbconn() )
        cursor = conn.cursor()
        # TODO : deal with provenance!
        q = ( 'SELECT e.id, e.filepath, e.mjd, e.target, e.filter, e.filter_array, e.exp_time, '
              '       COUNT(DISTINCT(i.id)) AS n_images, COUNT(c.id) AS n_sources '
              'FROM exposures e '
              'LEFT JOIN images i ON i.exposure_id=e.id '
              'LEFT JOIN image_upstreams_association ias ON ias.upstream_id=i.id '
              'LEFT JOIN images s ON s.id = ias.downstream_id AND s.is_sub '
              'LEFT JOIN source_lists sl ON sl.image_id=s.id '
              'LEFT JOIN cutouts c ON c.sources_id=sl.id ' )
        subdict = {}
        if ( t0 is not None ) or ( t1 is not None ):
            q += "WHERE "
            if t0 is not None:
                q += 'e.mjd >= %(t0)s'
                subdict['t0'] = t0
            if t1 is not None:
                if t0 is not None: q += ' AND '
                q += 'e.mjd <= %(t1)s'
                subdict['t1'] = t1
        q += ( ' GROUP BY e.id ' # ,e.filepath,e.mjd,e.target,e.filter,e.filter_array,e.exp_time '
               'ORDER BY e.mjd, e.filter, e.filter_array ' )
        cursor.execute( q, subdict )
        columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }

        ids = []
        name = []
        mjd = []
        target = []
        filtername = []
        exp_time = []
        n_images = []
        n_sources = []

        slashre = re.compile( '^.*/([^/]+)$' )
        for row in cursor.fetchall():
            ids.append( row[columns['id']] )
            match = slashre.search( row[columns['filepath']] )
            if match is None:
                name.append( row[columns['filepath']] )
            else:
                name.append( match.group(1) )
            mjd.append( row[columns['mjd']] )
            target.append( row[columns['target']] )
            app.logger.debug( f"filter={row[columns['filter']]} type {row[columns['filter']]}; "
                              f"filter_array={row[columns['filter_array']]} type {row[columns['filter_array']]}" )
            filtername.append( row[columns['filter']] )
            exp_time.append( row[columns['exp_time']] )
            n_images.append( row[columns['n_images']] )
            n_sources.append( row[columns[ 'n_sources']] )

        return { 'status': 'ok',
                 'startdate': t0,
                 'enddate': t1,
                 'exposures': {
                     'id': ids,
                     'name': name,
                     'mjd': mjd,
                     'target': target,
                     'filter': filtername,
                     'exp_time': exp_time,
                     'n_images': n_images,
                     'n_sources': n_sources
                 }
                }
    except Exception as ex:
        # sio = io.StringIO()
        # traceback.print_exc( file=sio )
        # app.logger.debug( sio.getvalue() )
        app.logger.exception( ex )
        return { 'status': 'error',
                 'error': f'Exception: {ex}'
                }

# **********************************************************************

@app.route( "/exposure_images/<expid>", methods=['GET', 'POST'], strict_slashes=False )
def exposure_images( expid ):
    try:
        conn = next( dbconn() )
        cursor = conn.cursor()
        # TODO : deal with provenance!
        q = ( 'SELECT i.id, i.filepath, i.ra, i.dec, i.gallat, i.section_id, i.fwhm_estimate, '
              '       i.zero_point_estimate, i.lim_mag_estimate, i.bkg_mean_estimate, i.bkg_rms_estimate, '
              '       s.id AS subid, COUNT(c.id) AS numsources '
              'FROM images i '
              'LEFT JOIN image_upstreams_association ias ON ias.upstream_id=i.id '
              'LEFT JOIN images s ON s.id = ias.downstream_id AND s.is_sub '
              'LEFT JOIN source_lists sl ON sl.image_id=s.id '
              'LEFT JOIN cutouts c ON  c.sources_id=sl.id '
              'WHERE i.is_sub=false AND i.exposure_id=%(expid)s '
              'GROUP BY i.id,s.id '
              'ORDER BY i.section_id,s.id ' )
        app.logger.debug( f"Getting images for exposure {expid}; query = {cursor.mogrify(q, {'expid': int(expid)})}" )
        cursor.execute( q, { 'expid': int(expid) } )
        columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
        app.logger.debug( f"Got {len(columns)} columns, {cursor.rowcount} rows" )

        fields = ( 'id', 'ra', 'dec', 'gallat', 'section_id', 'fwhm_estimate', 'zero_point_estimate',
                   'lim_mag_estimate', 'bkg_mean_estimate', 'bkg_rms_estimate', 'numsources', 'subid' )

        retval = { 'status': 'ok', 'name': [] }
        for field in fields :
            retval[ field ] = []

        lastimg = -1
        slashre = re.compile( '^.*/([^/]+)$' )
        for row in cursor.fetchall():
            if row[columns['id']] == lastimg:
                app.logger.warning( f'Multiple subtractions for image {i.id}, need to deal with provenance!' )
                continue
            lastimg = row[columns['id']]

            match = slashre.search( row[columns['filepath']] )
            retval['name'].append( row[columns['filepath']] if match is None else match.group(1) )
            for field in fields:
                retval[field].append( row[columns[field]] )

        return retval

    except Exception as ex:
        app.logger.exception( ex )
        return { 'status': 'error',
                 'error': f'Exception: {ex}' }

# **********************************************************************

@app.route( "/png_cutouts_for_sub_image/<int:subid>", methods=['GET', 'POST'], strict_slashes=False )
@app.route( "/png_cutouts_for_sub_image/<int:subid>/<int:limit>", methods=['GET', 'POST'], strict_slashes=False )
@app.route( "/png_cutouts_for_sub_image/<int:subid>/<int:limit>/<int:offset>",
            methods=['GET', 'POST'], strict_slashes=False )
def png_cutouts_for_sub_image( subid, limit=None, offset=0 ):
    try:
        data = { 'sortby': 'index' }
        if flask.request.is_json:
            data.update( flask.request.json )

        conn = next( dbconn() )
        cursor = conn.cursor()
        # TODO : deal with provenance!
        # TODO : r/b and sorting

        q = ( 'SELECT z.zp, z.dzp, z.aper_cor_radii, z.aper_cors, i.id, i.bkg_mean_estimate '
              'FROM images s '
              'INNER JOIN image_upstreams_association ias ON ias.downstream_id=s.id '
              '   AND s.ref_image_id != ias.upstream_id '
              'INNER JOIN images i ON ias.upstream_id=i.id '
              'INNER JOIN source_lists sl ON sl.image_id=i.id '
              'INNER JOIN zero_points z ON sl.id=z.sources_id '
              'WHERE s.id=%(subid)s ')
        cursor.execute( q, { 'subid': subid } )
        cols = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
        rows = cursor.fetchall()
        if len(rows) > 0:
            app.logger.warning( f"Multiple zeropoints for subid {subid}, deal with provenance" )
        if len(rows) == 0:
            app.logger.error( f"Couldn't find a zeropoint for subid {subid}" )
            zp = -99
            dzp = -99
            imageid = -99
        zp = rows[0][cols['zp']]
        dzp = rows[0][cols['dzp']]
        imageid = rows[0][cols['id']]
        newbkg = rows[0][cols['bkg_mean_estimate']]
        aperrads = rows[0][cols['aper_cor_radii']]
        apercors = rows[0][cols['aper_cors']]

        # app.logger.debug( f"Got zp={zp:.2f}±{dzp:.2f}, new bkg={newbkg:.2f} for subid {subid}, imageid {imageid}" )

        app.logger.debug( f"Getting cutouts for sub image {subid}" )
        q = ( 'SELECT c.id, c.filepath, c.ra, c.dec, c.x, c.y, c.index_in_sources, m.best_aperture, '
              '       m.flux_apertures[m.best_aperture+1] AS flux, m.flux_apertures_err[m.best_aperture+1] AS dflux, '
              '       m.ra AS measra, m.dec AS measdec '
              'FROM cutouts c '
              'INNER JOIN source_lists sl ON c.sources_id=sl.id '
              'INNER JOIN images s ON sl.image_id=s.id '
              'INNER JOIN measurements m ON m.cutouts_id=c.id '
              'WHERE s.id=%(subid)s ' )
        if data['sortby'] == 'index':
            q += 'ORDER BY c.index_in_sources '
        else:
            raise RuntimeError( f"Unknown sort criterion {data['sortby']}" )
        if limit is not None:
            q += 'LIMIT %(limit)s OFFSET %(offset)s'
        subdict = { 'subid': subid, 'limit': limit, 'offset': offset }
        cursor.execute( q, subdict );
        cols = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
        rows = cursor.fetchall()
        app.logger.info( f"Got {len(cols)} columns, {len(rows)} rows" )

        hdf5files = {}
        retval = { 'status': 'ok',
                   'sub_id': subid,
                   'image_id': imageid,
                   'cutouts': {
                       'id': [],
                       'ra': [],
                       'dec': [],
                       'measra': [],
                       'measdec': [],
                       'flux': [],
                       'dflux': [],
                       'aperrad': [],
                       'mag': [],
                       'dmag': [],
                       'x': [],
                       'y': [],
                       'w': [],
                       'h': [],
                       'new_png': [],
                       'ref_png': [],
                       'sub_png': []
                   }
                  }

        scaler = astropy.visualization.ZScaleInterval()

        for row in rows:
            if row[cols['filepath']] not in hdf5files:
                hdf5files[row[cols['filepath']]] = h5py.File( ARCHIVE_DIR / row[cols['filepath']], 'r' )
            grp = hdf5files[row[cols['filepath']]][f'source_{row[cols["index_in_sources"]]}']
            vmin, vmax = scaler.get_limits( grp['new_data'] )
            scalednew = ( grp['new_data'] - vmin ) * 255. / ( vmax - vmin )
            # TODO : there's an assumption here that the ref is background
            #   subtracted. They probably usually will be -- that tends to be
            #   part of a coadd process.
            vmin -= newbkg
            vmax -= newbkg
            scaledref = ( grp['ref_data'] - vmin ) * 255. / ( vmax - vmin )
            vmin, vmax = scaler.get_limits( grp['sub_data'] )
            scaledsub = ( grp['sub_data'] - vmin ) * 255. / ( vmax - vmin )

            scalednew[ scalednew < 0 ] = 0
            scalednew[ scalednew > 255 ] = 255
            scaledref[ scaledref < 0 ] = 0
            scaledref[ scaledref > 255 ] = 255
            scaledsub[ scaledsub < 0 ] = 0
            scaledsub[ scaledsub > 255 ] = 255

            scalednew = numpy.array( scalednew, dtype=numpy.uint8 )
            scaledref = numpy.array( scaledref, dtype=numpy.uint8 )
            scaledsub = numpy.array( scaledsub, dtype=numpy.uint8 )

            # TODO : transpose, flip for principle of least surprise
            # Figure out what PIL.Image does.
            #  (this will affect w and h below)

            newim = io.BytesIO()
            refim = io.BytesIO()
            subim = io.BytesIO()
            PIL.Image.fromarray( scalednew ).save( newim, format='png' )
            PIL.Image.fromarray( scaledref ).save( refim, format='png' )
            PIL.Image.fromarray( scaledsub ).save( subim, format='png' )

            retval['cutouts']['new_png'].append( base64.b64encode( newim.getvalue() ).decode('ascii') )
            retval['cutouts']['ref_png'].append( base64.b64encode( refim.getvalue() ).decode('ascii') )
            retval['cutouts']['sub_png'].append( base64.b64encode( subim.getvalue() ).decode('ascii') )
            retval['cutouts']['id'].append( row[cols['id']] )
            retval['cutouts']['ra'].append( row[cols['ra']] )
            retval['cutouts']['dec'].append( row[cols['dec']] )
            retval['cutouts']['measra'].append( row[cols['measra']] )
            retval['cutouts']['measdec'].append( row[cols['measdec']] )
            retval['cutouts']['x'].append( row[cols['x']] )
            retval['cutouts']['y'].append( row[cols['y']] )
            retval['cutouts']['w'].append( scalednew.shape[0] )
            retval['cutouts']['h'].append( scalednew.shape[1] )

            # WARNING : assumption here that the aper cor radii list in the
            #   zero point is the same as was used in the measurements.
            # (I think that's a good assumption, but still.)

            flux = row[cols['flux']]
            dflux = row[cols['dflux']]
            mag = -99
            dmag = -99
            if ( zp > 0 ) and ( flux > 0 ):
                mag = -2.5 * math.log10( flux ) + zp + apercors[ row[cols['best_aperture']] ]
                # Ignore zp and apercor uncertainties
                dmag = 1.0857 * dflux / flux
            retval['cutouts']['flux'].append( flux )
            retval['cutouts']['dflux'].append( dflux )
            retval['cutouts']['aperrad'].append( aperrads[ row[cols['best_aperture']] ] )
            retval['cutouts']['mag'].append( mag )
            retval['cutouts']['dmag'].append( dmag )

        for f in hdf5files.values():
            f.close()

        return retval

    except Exception as ex:
        app.logger.exception( ex )
        return { 'status': 'error',
                 'error': f'Exception: {ex}' }
