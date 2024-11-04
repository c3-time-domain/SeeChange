# MASSIVE TODO : this whole webap doesn't handle provenances at all
# We need a way to choose provenances.  I sugest some sort of tag
# table that allows us to associate tags with provenances so that
# we can choose a set of provenances based a simple tag name.

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

sys.path.append( '/secrets' )
from seechange_webap_config import PG_HOST, PG_PORT, PG_USER, PG_PASS, PG_NAME, ARCHIVE_DIR

# Figure out where we are

workdir = pathlib.Path(__name__).resolve().parent

# Create the flask app, which is what gunicorn is going to look for

app = flask.Flask( __name__, instance_relative_config=True )

# app.logger.setLevel( logging.INFO )
app.logger.setLevel( logging.DEBUG )

# UNHAPPY CODE ORGANIZATION WARNING
# Because the webap doesn't import all of the SeeChange code base, stuff
#    coded there needs to be copied here.  The following table NEEDS TO
#    BE KEPT SYNCED with the combination of
#    enums_and_bitflags.py::DeepScoreAlgorithmConverter and
#    DeepScore.get_rb_cut
#
# Probably I should just give in and have the webap import the SeeChange
#   code base.  The Conductor does, after all.

_rb_cuts = {
    0: 0.5,
    1: 0.99,
    2: 0.55
}

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

@app.route( "/provtags", methods=['POST'], strict_slashes=False )
def provtags():
    try:
        conn = next( dbconn() )
        cursor = conn.cursor()
        cursor.execute( 'SELECT DISTINCT ON(tag) tag FROM provenance_tags ORDER BY tag' )
        return { 'status': 'ok',
                 'provenance_tags': [ row[0] for row in cursor.fetchall() ]
                }
    except Exception as ex:
        app.logger.exception( ex )
        return { 'status': 'error',
                 'error': f'Exception: {ex}' }


# **********************************************************************

@app.route( "/exposures", methods=['POST'], strict_slashes=False )
def exposures():
    try:
        data = { 'startdate': None,
                 'enddate': None,
                 'provenancetag': None,
                }
        if flask.request.is_json:
            data.update( flask.request.json )

        app.logger.debug( f"After parsing, data = {data}" )
        t0 = None if data['startdate'] is None else astropy.time.Time( data['startdate'], format='isot' ).mjd
        t1 = None if data['enddate'] is None else astropy.time.Time( data['enddate'], format='isot' ).mjd
        app.logger.debug( f"t0 = {t0}, t1 = {t1}" )

        conn = next( dbconn() )
        cursor = conn.cursor()

        # Gonna do this in three steps.  First, get all the images with
        #  counts of source lists and counts of measurements in a temp
        #  table, then do the sums and things on that temp table.
        # Filtering on provenance tags makes this more complicated, so
        #  we'll do a different query if we're doing that.  Truthfully,
        #  asking for all provenance tags is going to be a mess for the
        #  user....  perhaps we should disable it?
        haveputinwhere = False
        subdict = {}
        if data['provenancetag'] is None:
            q = ( 'SELECT e._id, e.filepath, e.mjd, e.target, e.filter, e.filter_array, e.exp_time, '
                  '       i._id AS imgid, s._id AS subid, sl._id AS slid, sl.num_sources, '
                  '       COUNT(m._id) AS num_measurements '
                  'INTO TEMP TABLE temp_imgs '
                  'FROM exposures e '
                  'LEFT JOIN images i ON i.exposure_id=e._id '
                  'LEFT JOIN ( '
                  '  SELECT su._id, ias.upstream_id '
                  '  FROM images su '
                  '  INNER JOIN image_upstreams_association ias ON ias.downstream_id=su._id '
                  '  WHERE su.is_sub '
                  ') s ON s.upstream_id=i._id '
                  'LEFT JOIN source_lists sl ON sl.image_id=s._id '
                  'LEFT JOIN cutouts cu ON cu.sources_id=sl._id '
                  'LEFT JOIN measurements m ON m.cutouts_id=cu._id '
                  'GROUP BY e._id, i._id, s._id, sl._id '
                 )
        else:
            q = ( 'SELECT e._id, e.filepath, e.mjd, e.target, e.filter, e.filter_array, e.exp_time, '
                  '       i._id AS imgid, s._id AS subid, sl._id AS slid, sl.num_sources, '
                  '       COUNT(m._id) AS num_measurements '
                  'INTO TEMP TABLE temp_imgs '
                  'FROM exposures e '
                  'LEFT JOIN ( '
                  '  SELECT im._id, im.exposure_id FROM images im '
                  '  INNER JOIN provenance_tags impt ON impt.provenance_id=im.provenance_id AND impt.tag=%(provtag)s '
                  ') i ON i.exposure_id=e._id '
                  'LEFT JOIN ( '
                  '  SELECT su._id, ias.upstream_id FROM images su '
                  '  INNER JOIN image_upstreams_association ias ON ias.downstream_id=su._id AND su.is_sub '
                  '  INNER JOIN provenance_tags supt ON supt.provenance_id=su.provenance_id AND supt.tag=%(provtag)s '
                  ') s ON s.upstream_id=i._id '
                  'LEFT JOIN ( '
                  '  SELECT sli._id, sli.image_id, sli.num_sources FROM source_lists sli '
                  '  INNER JOIN provenance_tags slpt ON slpt.provenance_id=sli.provenance_id AND slpt.tag=%(provtag)s '
                  ') sl ON sl.image_id=s._id '
                  'LEFT JOIN ( '
                  '  SELECT cu._id, cu.sources_id FROM cutouts cu '
                  '  INNER JOIN provenance_tags cupt ON cu.provenance_id=cupt.provenance_id AND cupt.tag=%(provtag)s '
                  ') c ON c.sources_id=sl._id '
                  'LEFT JOIN ( '
                  '  SELECT meas._id, meas.cutouts_id FROM measurements meas '
                  '  INNER JOIN provenance_tags mept ON mept.provenance_id=meas.provenance_id AND mept.tag=%(provtag)s '
                  ') m ON m.cutouts_id=c._id '
                  'INNER JOIN provenance_tags ept ON ept.provenance_id=e.provenance_id AND ept.tag=%(provtag)s '
                  'GROUP BY e._id, i._id, s._id, sl._id, sl.num_sources '
                 )
            subdict['provtag'] = data['provenancetag']
        if ( t0 is not None ) or ( t1 is not None ):
            q += 'WHERE '
            if t0 is not None:
                q += 'e.mjd >= %(t0)s'
                subdict['t0'] = t0
            if t1 is not None:
                if t0 is not None: q += ' AND '
                q += 'e.mjd <= %(t1)s'
                subdict['t1'] = t1

        cursor.execute( q, subdict )

        # Now run a second query to count and sum those things
        # These numbers will be wrong (double-counts) if not filtering on a provenance tag, or if the
        #   provenance tag includes multiple provenances for a given step!
        q = ( 'SELECT t._id, t.filepath, t.mjd, t.target, t.filter, t.filter_array, t.exp_time, '
              '  COUNT(t.subid) AS num_subs, SUM(t.num_sources) AS num_sources, '
              '  SUM(t.num_measurements) AS num_measurements '
              'INTO TEMP TABLE temp_imgs_2 '
              'FROM temp_imgs t '
              'GROUP BY t._id, t.filepath, t.mjd, t.target, t.filter, t.filter_array, t.exp_time '
             )

        cursor.execute( q )

        # Run a third query count reports
        subdict = {}
        q = ( 'SELECT t._id, t.filepath, t.mjd, t.target, t.filter, t.filter_array, t.exp_time, '
              '  t.num_subs, t.num_sources, t.num_measurements, '
              '  SUM( CASE WHEN r.success THEN 1 ELSE 0 END ) as n_successim, '
              '  SUM( CASE WHEN r.error_message IS NOT NULL THEN 1 ELSE 0 END ) AS n_errors '
              'FROM temp_imgs_2 t '
             )
        if data['provenancetag'] is None:
            q += 'LEFT JOIN reports r ON r.exposure_id=t._id '
        else:
            q += ( 'LEFT JOIN ( '
                   '  SELECT re.exposure_id, re.success, re.error_message '
                   '  FROM reports re '
                   '  INNER JOIN provenance_tags rept ON rept.provenance_id=re.provenance_id AND rept.tag=%(provtag)s '
                   ') r ON r.exposure_id=t._id '
                  )
            subdict['provtag'] = data['provenancetag']
        # I wonder if making a primary key on the temp table would be more efficient than
        #    all these columns in GROUP BY?  Investigate this.
        q += ( 'GROUP BY t._id, t.filepath, t.mjd, t.target, t.filter, t.filter_array, t.exp_time, '
               '  t.num_subs, t.num_sources, t.num_measurements ' )

        cursor.execute( q, subdict  )
        columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }

        ids = []
        name = []
        mjd = []
        target = []
        filtername = []
        exp_time = []
        n_subs = []
        n_sources = []
        n_measurements = []
        n_successim = []
        n_errors = []

        slashre = re.compile( '^.*/([^/]+)$' )
        for row in cursor.fetchall():
            ids.append( row[columns['_id']] )
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
            n_subs.append( row[columns['num_subs']] )
            n_sources.append( row[columns['num_sources']] )
            n_measurements.append( row[columns['num_measurements']] )
            n_successim.append( row[columns['n_successim']] )
            n_errors.append( row[columns['n_errors']] )

        return { 'status': 'ok',
                 'startdate': t0,
                 'enddate': t1,
                 'provenance_tag': data['provenancetag'],
                 'exposures': {
                     'id': ids,
                     'name': name,
                     'mjd': mjd,
                     'target': target,
                     'filter': filtername,
                     'exp_time': exp_time,
                     'n_subs': n_subs,
                     'n_sources': n_sources,
                     'n_measurements': n_measurements,
                     'n_successim': n_successim,
                     'n_errors': n_errors,
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

@app.route( "/exposure_images/<expid>/<provtag>", methods=['GET', 'POST'], strict_slashes=False )
def exposure_images( expid, provtag ):
    try:
        conn = next( dbconn() )
        cursor = conn.cursor()

        # Going to do this in a few steps again.  Might be able to write one
        # bigass query, but it's probably more efficient to use temp tables.
        # Easier to build the queries that way too.

        subdict = { 'expid': str(expid), 'provtag': provtag }

        # Step 1: collect image info into temp_exposure_images
        q = ( 'SELECT i._id, i.filepath, i.ra, i.dec, i.gallat, i.exposure_id, i.section_id, i.fwhm_estimate, '
              '       i.zero_point_estimate, i.lim_mag_estimate, i.bkg_mean_estimate, i.bkg_rms_estimate '
              'INTO TEMP TABLE temp_exposure_images '
              'FROM images i '
              'INNER JOIN provenance_tags ipt ON ipt.provenance_id=i.provenance_id '
              'WHERE i.exposure_id=%(expid)s '
              '  AND ipt.tag=%(provtag)s '
             )
        #  app.logger.debug( f"exposure_images finding images; query: {cursor.mogrify(q,subdict)}" )
        cursor.execute( q, subdict )
        cursor.execute( "ALTER TABLE temp_exposure_images ADD PRIMARY KEY(_id)" )
        # ****
        # cursor.execute( "SELECT COUNT(*) FROM temp_exposure_images" )
        # app.logger.debug( f"Got {cursor.fetchone()[0]} images" )
        # ****

        # Step 2: count measurements by joining temp_exposure_images to many things.
        q = ( 'SELECT i._id, s._id AS subid, sl.num_sources AS numsources, COUNT(m._id) AS nummeasurements '
              'INTO TEMP TABLE temp_exposure_images_counts '
              'FROM temp_exposure_images i '
              'INNER JOIN image_upstreams_association ias ON ias.upstream_id=i._id '
              'INNER JOIN images s ON s.is_sub AND s._id=ias.downstream_id '
              'INNER JOIN provenance_tags spt ON spt.provenance_id=s.provenance_id AND spt.tag=%(provtag)s '
              'LEFT JOIN ( '
              '  SELECT sli._id, sli.image_id, sli.num_sources FROM source_lists sli '
              '  INNER JOIN provenance_tags slpt ON slpt.provenance_id=sli.provenance_id AND slpt.tag=%(provtag)s '
              ') sl ON sl.image_id=s._id '
              'LEFT JOIN ('
              '  SELECT cu._id, cu.sources_id FROM cutouts cu '
              '  INNER JOIN provenance_tags cupt ON cupt.provenance_id=cu.provenance_id AND cupt.tag=%(provtag)s '
              ') c ON c.sources_id=sl._id '
              'LEFT JOIN ('
              '  SELECT me._id, me.cutouts_id FROM measurements me '
              '  INNER JOIN provenance_tags mept ON mept.provenance_id=me.provenance_id AND mept.tag=%(provtag)s '
              ') m ON m.cutouts_id=c._id '
              'GROUP BY i._id, s._id, sl.num_sources '
             )
        # app.logger.debug( f"exposure_images counting sources: query {cursor.mogrify(q,subdict)}" )
        cursor.execute( q, subdict )
        # We will get an error here if there are multiple rows for a given image.
        # (Which is good; there shouldn't be multiple rows!  There should only be
        # one (e.g.) source list child of the image for a given provenance tag, etc.)
        cursor.execute( "ALTER TABLE temp_exposure_images_counts ADD PRIMARY KEY(_id)" )
        # ****
        # cursor.execute( "SELECT COUNT(*) FROM temp_exposure_images_counts" )
        # app.logger.debug( f"Got {cursor.fetchone()[0]} rows with counts" )
        # ****

        # Step 3: join to the report table.  This one is probably mergeable with step 1.
        q = ( 'SELECT i._id, r.error_step, r.error_type, r.error_message, r.warnings, '
              '       r.process_memory, r.process_runtime, r.progress_steps_bitflag, r.products_exist_bitflag '
              'INTO TEMP TABLE temp_exposure_images_reports '
              'FROM temp_exposure_images i '
              'INNER JOIN ( '
              '  SELECT re.exposure_id, re.section_id, '
              '         re.error_step, re.error_type, re.error_message, re.warnings, '
              '         re.process_memory, re.process_runtime, re.progress_steps_bitflag, re.products_exist_bitflag '
              '  FROM reports re '
              '  INNER JOIN provenance_tags rept ON rept.provenance_id=re.provenance_id AND rept.tag=%(provtag)s '
              ') r ON r.exposure_id=i.exposure_id AND r.section_id=i.section_id '
             )
        # app.logger.debug( f"exposure_images getting reports; query {cursor.mogrify(q,subdict)}" )
        cursor.execute( q, subdict )
        # Again, we will get an error here if there are multiple rows for a given image
        cursor.execute( "ALTER TABLE temp_exposure_images_reports ADD PRIMARY KEY(_id)" )
        # ****
        # cursor.execute( "SELECT COUNT(*) FROM temp_exposure_images_reports" )
        # app.logger.debug( f"Got {cursor.fetchone()[0]} rows with reports" )
        # ****

        cursor.execute( "SELECT t1.*, t2.*, t3.* "
                        "FROM temp_exposure_images t1 "
                        "LEFT JOIN temp_exposure_images_counts t2 ON t1._id=t2._id "
                        "LEFT JOIN temp_exposure_images_reports t3 ON t1._id=t3._id "
                        "ORDER BY t1.section_id" )
        columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
        rows = cursor.fetchall()
        # app.logger.debug( f"exposure_images got {len(rows)} rows from the final query." )

        fields = ( '_id', 'ra', 'dec', 'gallat', 'section_id', 'fwhm_estimate', 'zero_point_estimate',
                   'lim_mag_estimate', 'bkg_mean_estimate', 'bkg_rms_estimate',
                   'numsources', 'nummeasurements', 'subid',
                   'error_step', 'error_type', 'error_message', 'warnings',
                   'process_memory', 'process_runtime', 'progress_steps_bitflag', 'products_exist_bitflag' )

        retval = { 'status': 'ok',
                   'provenancetag': provtag,
                   'name': [] }

        for field in fields :
            rfield = 'id' if field == '_id' else field
            retval[ rfield ] = []

        lastimg = -1
        multiples = set()
        slashre = re.compile( '^.*/([^/]+)$' )
        for row in rows:
            if row[columns['_id']] == lastimg:
                multiples.add( row[columns['id']] )
                continue
            lastimg = row[columns['_id']]

            match = slashre.search( row[columns['filepath']] )
            retval['name'].append( row[columns['filepath']] if match is None else match.group(1) )
            for field in fields:
                rfield = 'id' if field == '_id' else field
                retval[rfield].append( row[columns[field]] )

        if len(multiples) != 0:
            return { 'status': 'error',
                     'error': ( f'Some images had multiple rows in the query; this probably indicates '
                                f'that the reports table is not well-formed.  Or maybe something else. '
                                f'offending images: {multiples}' ) }

        app.logger.debug( f"exposure_images returning {retval}" )
        return retval

    except Exception as ex:
        app.logger.exception( ex )
        return { 'status': 'error',
                 'error': f'Exception: {ex}' }

# **********************************************************************

@app.route( "/png_cutouts_for_sub_image/<exporsubid>/<provtag>/<int:issubid>/<int:nomeas>",
            methods=['GET', 'POST'], strict_slashes=False )
@app.route( "/png_cutouts_for_sub_image/<exporsubid>/<provtag>/<int:issubid>/<int:nomeas>/<int:limit>",
            methods=['GET', 'POST'], strict_slashes=False )
@app.route( "/png_cutouts_for_sub_image/<exporsubid>/<provtag>/<int:issubid>/<int:nomeas>/"
            "<int:limit>/<int:offset>",
            methods=['GET', 'POST'], strict_slashes=False )
def png_cutouts_for_sub_image( exporsubid, provtag, issubid, nomeas, limit=None, offset=0 ):
    try:
        data = { 'sortby': 'rbdesc_fluxdesc_chip_index' }
        if flask.request.is_json:
            data.update( flask.request.json )

        app.logger.debug( f"Processing {flask.request.url}" )
        if issubid:
            app.logger.debug( f"Looking for cutouts from subid {exporsubid} ({'with' if nomeas else 'without'} "
                              f"missing-measurements)" )
        else:
            app.logger.debug( f"Looking for cutouts from exposure {exporsubid} ({'with' if nomeas else 'without'} "
                              f"missing-measurements)" )

        conn = next( dbconn() )
        cursor = conn.cursor()
        # TODO : r/b and sorting

        # Figure out the subids, zeropoints, backgrounds, and apertures we need

        subids = []
        zps = {}
        dzps = {}
        imageids = {}
        newbkgs = {}
        aperradses = {}
        apercorses = {}

        q = ( 'SELECT s._id AS subid, z.zp, z.dzp, z.aper_cor_radii, z.aper_cors, '
              '  i._id AS imageid, i.bkg_mean_estimate '
              'FROM images s '
              )
        if not issubid:
            # If we got an exposure id, make sure only to get subtractions of the requested provenance
            q += 'INNER JOIN provenance_tags spt ON s.provenance_id=spt.provenance_id AND spt.tag=%(provtag)s '
        q +=  ( 'INNER JOIN image_upstreams_association ias ON ias.downstream_id=s._id '
                '   AND s.ref_image_id != ias.upstream_id '
                'INNER JOIN images i ON ias.upstream_id=i._id '
                'INNER JOIN source_lists sl ON sl.image_id=i._id '
                'INNER JOIN provenance_tags slpt ON sl.provenance_id=slpt.provenance_id AND slpt.tag=%(provtag)s '
                'INNER JOIN zero_points z ON sl._id=z.sources_id ' )
        # (Don't need to check provenance tag of zeropoint since we have a
        # 1:1 relationship between zeropoints and source lists.  Don't need
        # to check image provenance, because there will be a single image id
        # upstream of each sub id.

        if issubid:
            q += 'WHERE s._id=%(subid)s '
            cursor.execute( q, { 'subid': exporsubid, 'provtag': provtag } )
            cols = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
            rows = cursor.fetchall()
            if len(rows) > 1:
                app.logger.error( f"Multiple rows for subid {exporsubid}, provenance tag {provtag} "
                                  f"is not well-defined, or something else is wrong." )
                return { 'status': 'error',
                         'error': ( f"Multiple rows for subid {exporsubid}, provenance tag {provtag} "
                                    f"is not well-defined, or something else is wrong." ) }
            if len(rows) == 0:
                app.logger.error( f"Couldn't find a zeropoint for subid {exporsubid}" )
                return { 'status': 'error',
                         'error': f"Coudn't find zeropoint for subid {exporsubid}" }
            subids.append( exporsubid )
            zps[exporsubid] = rows[0][cols['zp']]
            dzps[exporsubid] = rows[0][cols['dzp']]
            imageids[exporsubid] = rows[0][cols['imageid']]
            newbkgs[exporsubid] = rows[0][cols['bkg_mean_estimate']]
            aperradses[exporsubid] = rows[0][cols['aper_cor_radii']]
            apercorses[exporsubid] = rows[0][cols['aper_cors']]

        else:
            q += ( 'INNER JOIN exposures e ON i.exposure_id=e._id '
                   'WHERE e._id=%(expid)s ORDER BY i.section_id  ' )
            # Don't need to verify provenance here, because there's just going to be one expid!
            cursor.execute( q, { 'expid': exporsubid, 'provtag': provtag } )
            cols = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
            rows = cursor.fetchall()
            for row in rows:
                subid = row[cols['subid']]
                if ( subid in subids ):
                    app.logger.error( f"subid {subid} showed up more than once in zp query" )
                    return { 'status': 'error',
                             'error': f"subid {subid} showed up more than once in zp query" }
                subids.append( subid )
                zps[subid] = row[cols['zp']]
                dzps[subid] = row[cols['dzp']]
                imageids[subid] = row[cols['imageid']]
                newbkgs[subid] = row[cols['bkg_mean_estimate']]
                aperradses[subid] = row[cols['aper_cor_radii']]
                apercorses[subid] = row[cols['aper_cors']]
        app.logger.debug( f'Got {len(subids)} subtractions.' )

        app.logger.debug( f"Getting cutouts files for sub images {subids}" )
        q = ( 'SELECT c.filepath,s._id AS subimageid,sl.filepath AS sources_path '
              'FROM cutouts c '
              'INNER JOIN provenance_tags cpt ON cpt.provenance_id=c.provenance_id AND cpt.tag=%(provtag)s '
              'INNER JOIN source_lists sl ON c.sources_id=sl._id '
              'INNER JOIN images s ON sl.image_id=s._id '
              'WHERE s._id IN %(subids)s ' )
        # Don't have to check the source_lists provenance tag because the cutouts provenance
        # tag cut will limit us to a single source_list for each cutouts
        cursor.execute( q, { 'subids': tuple(subids), 'provtag': provtag } )
        cols = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
        rows = cursor.fetchall()
        cutoutsfiles = { c[cols['subimageid']]: c[cols['filepath']] for c in rows }
        sourcesfiles = { c[cols['subimageid']]: c[cols['sources_path']] for c in rows }
        app.logger.debug( f"Got: {cutoutsfiles}" )

        app.logger.debug( f"Getting measurements for sub images {subids}" )
        q = ( 'SELECT m.ra AS measra, m.dec AS measdec, m.index_in_sources, m.best_aperture, '
              '       m.flux, m.dflux, m.psfflux, m.dpsfflux, m.is_bad, m.name, m.is_test, m.is_fake, '
              '       m.score, m._algorithm, s._id AS subid, s.section_id '
              'FROM cutouts c '
              'INNER JOIN provenance_tags cpt ON cpt.provenance_id=c.provenance_id AND cpt.tag=%(provtag)s '
              'INNER JOIN source_lists sl ON c.sources_id=sl._id '
              'INNER JOIN images s ON sl.image_id=s._id '
              'LEFT JOIN '
              '  ( SELECT meas.cutouts_id AS meascutid, meas.index_in_sources, meas.ra, meas.dec, meas.is_bad, '
              '           meas.best_aperture, meas.flux_apertures[meas.best_aperture+1] AS flux, '
              '           meas.flux_apertures_err[meas.best_aperture+1] AS dflux, '
              '           meas.flux_psf AS psfflux, meas.flux_psf_err AS dpsfflux, '
              '           obj.name, obj.is_test, obj.is_fake, score.score, score._algorithm '
              '    FROM measurements meas '
              '    INNER JOIN provenance_tags mpt ON meas.provenance_id=mpt.provenance_id AND mpt.tag=%(provtag)s '
              '    INNER JOIN objects obj ON meas.object_id=obj._id '
              '    LEFT JOIN '
              '      ( SELECT s.measurements_id, s.score, s._algorithm FROM deepscores s '
              '        INNER JOIN provenance_tags spt ON spt.provenance_id=s.provenance_id AND spt.tag=%(provtag)s '
              '      ) AS score '
              '      ON score.measurements_id=meas._id '
             )
        if not nomeas:
            q += '    WHERE NOT meas.is_bad '
        q += ( '   ) AS m ON m.meascutid=c._id '
               'WHERE s._id IN %(subids)s ' )
        if data['sortby'] == 'fluxdesc_chip_index':
            q += 'ORDER BY flux DESC NULLS LAST,s.section_id,m.index_in_sources '
        elif data['sortby'] == 'rbdesc_fluxdesc_chip_index':
            q += 'ORDER BY score DESC NULLS LAST,flux DESC NULLS LAST,s.section_id,m.index_in_sources '
        else:
            raise RuntimeError( f"Unknown sort criterion {data['sortby']}" )
        if limit is not None:
            q += 'LIMIT %(limit)s OFFSET %(offset)s'
        subdict = { 'subids': tuple(subids), 'provtag': provtag, 'limit': limit, 'offset': offset }
        app.logger.debug( f"Sending query to get measurements: {cursor.mogrify(q,subdict)}" )
        cursor.execute( q, subdict );
        cols = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
        rows = cursor.fetchall()
        app.logger.debug( f"Got {len(cols)} columns, {len(rows)} rows" )

        retval = { 'status': 'ok',
                   'cutouts': {
                       'sub_id': [],
                       'image_id': [],
                       'section_id': [],
                       'source_index': [],
                       'measra': [],
                       'measdec': [],
                       'flux': [],
                       'dflux': [],
                       'aperrad': [],
                       'mag': [],
                       'dmag': [],
                       'rb': [],
                       'rbcut': [],
                       'is_bad': [],
                       'objname': [],
                       'is_test': [],
                       'is_fake': [],
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

        # Open all the hdf5 files

        hdf5files = {}
        for subid in cutoutsfiles.keys():
            hdf5files[ subid ] = h5py.File( ARCHIVE_DIR / cutoutsfiles[subid], 'r' )

        def append_to_retval( subid, index_in_sources, row ):
            grp = hdf5files[ subid ][f'source_index_{row[cols["index_in_sources"]]}']
            vmin, vmax = scaler.get_limits( grp['new_data'] )
            scalednew = ( grp['new_data'] - vmin ) * 255. / ( vmax - vmin )
            # TODO : there's an assumption here that the ref is background
            #   subtracted. They probably usually will be -- that tends to be
            #   part of a coadd process.
            vmin -= newbkgs[subid]
            vmax -= newbkgs[subid]
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

            retval['cutouts']['sub_id'].append( subid )
            retval['cutouts']['image_id'].append( imageids[subid] )
            retval['cutouts']['section_id'].append( row[cols['section_id']] )
            retval['cutouts']['new_png'].append( base64.b64encode( newim.getvalue() ).decode('ascii') )
            retval['cutouts']['ref_png'].append( base64.b64encode( refim.getvalue() ).decode('ascii') )
            retval['cutouts']['sub_png'].append( base64.b64encode( subim.getvalue() ).decode('ascii') )
            # TODO : if we want to return x and y, we also have
            #   to read the source list file...
            # We could also copy them to the cutouts file as attributes
            # retval['cutouts']['x'].append( row[cols['x']] )
            # retval['cutouts']['y'].append( row[cols['y']] )
            retval['cutouts']['w'].append( scalednew.shape[0] )
            retval['cutouts']['h'].append( scalednew.shape[1] )

            retval['cutouts']['rb'].append( row[cols['score']] )
            retval['cutouts']['rbcut'].append( _rb_cuts[ row[cols['_algorithm']] ] )
            retval['cutouts']['is_bad'].append( row[cols['is_bad']] )
            retval['cutouts']['objname'].append( row[cols['name']] )
            retval['cutouts']['is_test'].append( row[cols['is_test']] )
            retval['cutouts']['is_fake'].append( row[cols['is_fake']] )

            if row[cols['psfflux']] is None:
                flux = row[cols['flux']]
                dflux = row[cols['dflux']]
                aperrad = aperradses[subid][ row[cols['best_aperture']] ]
            else:
                flux = row[cols['psfflux']]
                dflux = row[cols['dpsfflux']]
                aperrad = 0.

            if flux is None:
                for field in [ 'flux', 'dflux', 'aperrad', 'mag', 'dmag', 'measra', 'measdec' ]:
                    retval['cutouts'][field].append( None )
            else:
                mag = -99
                dmag = -99
                if ( zps[subid] > 0 ) and ( flux > 0 ):
                    mag = -2.5 * math.log10( flux ) + zps[subid] + apercorses[subid][ row[cols['best_aperture']] ]
                    # Ignore zp and apercor uncertainties
                    dmag = 1.0857 * dflux / flux
                retval['cutouts']['measra'].append( row[cols['measra']] )
                retval['cutouts']['measdec'].append( row[cols['measdec']] )
                retval['cutouts']['flux'].append( flux )
                retval['cutouts']['dflux'].append( dflux )
                retval['cutouts']['aperrad'].append( aperrad )
                retval['cutouts']['mag'].append( mag )
                retval['cutouts']['dmag'].append( dmag )

        # First: put in all the measurements, in the order we got them

        alredy_done = set()
        for row in rows:
            subid = row[cols['subid']]
            index_in_sources = row[ cols['index_in_sources'] ]
            append_to_retval( subid, index_in_sources, row )

        # TODO : things that we don't have measurements of

        for f in hdf5files.values():
            f.close()

        app.logger.debug( f"Returning {len(retval['cutouts']['sub_id'])} cutouts" )
        return retval

    except Exception as ex:
        app.logger.exception( ex )
        return { 'status': 'error',
                 'error': f'Exception: {ex}' }
