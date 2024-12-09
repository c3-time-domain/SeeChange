# Put this first so we can be sure that there are no calls that subvert
#  this in other includes.
import matplotlib
matplotlib.use( "Agg" )
# matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# matplotlib.rc('text', usetex=True)  #  Need LaTeX in Dockerfile, not worth it

import sys
import math
import io
import re
import pathlib
import logging
import base64
import uuid

import numpy
import h5py
import PIL
import astropy.time
import astropy.visualization

import flask
import flask_session
import flask.views

from util.config import Config
from util.util import asUUID
from models.user import AuthUser
from models.deepscore import DeepScore
from models.base import SmartSession


# ======================================================================

class BaseView( flask.views.View ):
    _admin_required = False

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
            if ( self._admin_required ) and ( not self.user.isadmin ):
                return "Action requires admin", 500
            try:
                return self.do_the_things( *args, **kwargs )
            except Exception as ex:
                # sio = io.StringIO()
                # traceback.print_exc( file=sio )
                # app.logger.debug( sio.getvalue() )
                app.logger.exception( str(ex) )
                return f"Exception handling request: {ex}", 500


# ======================================================================

class MainPage( BaseView ):
    def dispatch_request( self ):
        return flask.render_template( "seechange_webap.html" )


# ======================================================================

class ProvTags( BaseView ):
    def do_the_things( self ):
        cursor = self.conn.cursor()
        cursor.execute( 'SELECT DISTINCT ON(tag) tag FROM provenance_tags ORDER BY tag' )
        tags = [ row[0] for row in cursor.fetchall() ]
        tags.sort( key=lambda x: ( '0' if x=='default' else 1 if x=='current' else 2, x ) )
        return { 'status': 'ok',
                 'provenance_tags': tags
                }


# ======================================================================

class ProvTagInfo( BaseView ):
    def do_the_things( self, tag ):
        cursor = self.conn.cursor()
        cursor.execute( 'SELECT p._id, p.process, p.code_version_id, p.parameters, '
                        '       p.is_bad, p.bad_comment, p.is_outdated, p.replaced_by, '
                        '       p.is_testing '
                        'FROM provenance_tags t '
                        'INNER JOIN provenances p ON t.provenance_id=p._id '
                        'WHERE t.tag=%(tag)s ',
                        { 'tag': tag } )
        columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
        rows = cursor.fetchall()

        provorder = { 'download': 0,
                      'import': 0,
                      'manual_import': 0,
                      'import_external_reference': 0,
                      'referencing': 1,
                      'preprocessing': 2,
                      'extraction': 3,
                      'subtraction': 4,
                      'detection': 5,
                      'cutting': 6,
                      'measuring': 7,
                      'scoring': 8,
                      'report': 9 }

        def sorter( row ):
            if row[columns['process']] in provorder.keys():
                val = f'{provorder[row[columns["process"]]]:02d}_{row[columns["_id"]]}'
            else:
                val = f'99_{row[columns["_id"]]}'
            return val

        rows.sort( key=sorter )
        retval = { 'status': 'ok',
                   'tag': tag }
        retval.update( { c: [ r[columns[c]] for r in rows ] for c in columns.keys() } )
        return retval


# ======================================================================

class CloneProvTag( BaseView ):
    _admin_required = True

    def do_the_things( self, existingtag, newtag, clobber=0 ):
        cursor = self.conn.cursor()
        if clobber:
            cursor.execute( "DELETE FROM provenance_tags WHERE tag=%(tag)s", { 'tag': newtag } )
        else:
            cursor.execute( "SELECT COUNT(*) FROM provenance_tags WHERE tag=%(tag)s", { 'tag': newtag } )
            n = cursor.fetchone()[0]
            if n != 0:
                return f"Tag {newtag} already exists and clobber was False", 500

        # I could probably do this with a single SQL command if I were
        #   clever enough, except that I'd need to have a server default
        #   on provenance_tags for generating the primary key uuid, and
        #   right now we don't have that.
        cursor.execute( "SELECT provenance_id FROM provenance_tags WHERE tag=%(tag)s", { 'tag': existingtag } )
        rows = cursor.fetchall()
        for row in rows:
            cursor.execute( "INSERT INTO provenance_tags(_id,tag,provenance_id) "
                            "VALUES(%(id)s,%(tag)s,%(provid)s)",
                            { 'id': uuid.uuid4(), 'tag': newtag, 'provid': row[0] } )
        self.conn.commit()

        return { 'status': 'ok' }


# ======================================================================

class ProvenanceInfo( BaseView ):
    def do_the_things( self, provid ):
        cursor = self.conn.cursor()
        cursor.execute( "SELECT p._id, p.process, p.code_version_id, p.parameters, "
                        "       p.is_bad, p.bad_comment, p.is_outdated, p.replaced_by, p.is_testing "
                        "FROM provenances p "
                        "WHERE p._id=%(provid)s ",
                        { 'provid': provid } )
        columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
        row = cursor.fetchone()
        retval = { 'status': 'ok' }
        retval.update( { c: row[i] for c,i in columns.items() } )

        cursor.execute( "SELECT p._id, p.process FROM provenances p "
                        "INNER JOIN provenance_upstreams pu ON pu.upstream_id=p._id "
                        "WHERE pu.downstream_id=%(provid)s",
                        { 'provid': provid } )
        columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
        rows = cursor.fetchall()
        retval['upstreams'] = { c: [ row[i] for row in rows ] for c, i in columns.items() }
        return retval


# ======================================================================
# This only gets projects from exposures, not images.
#
# Of course, image and exposure both having 'project' means the database
#   isn't normalized... except that we do want to be able to support
#   images that have no exposure.  Or do we?  Maybe we should support
#   the notion of a null exposure for expousre-less images?  That would
#   be a fair bit of refactoring.

class Projects( BaseView ):
    def do_the_things( self ):
        cursor = self.conn.cursor()
        cursor.execute( 'SELECT DISTINCT ON(project) project FROM exposures ORDER BY project' )
        return { 'status': 'ok',
                 'projects': [ row[0] for row in cursor.fetchall() ]
                }


# ======================================================================

class Exposures( BaseView ):
    def do_the_things( self ):
        data = { 'startdate': None,
                 'enddate': None,
                 'provenancetag': None,
                 'projects': None,
                }
        if flask.request.is_json:
            data.update( flask.request.json )

        app.logger.debug( f"After parsing, data = {data}" )
        t0 = None if data['startdate'] is None else astropy.time.Time( data['startdate'], format='isot' ).mjd
        t1 = None if data['enddate'] is None else astropy.time.Time( data['enddate'], format='isot' ).mjd
        app.logger.debug( f"t0 = {t0}, t1 = {t1}" )

        cursor = self.conn.cursor()

        # Gonna do this in three steps.  First, get all the images with
        #  counts of source lists and counts of measurements in a temp
        #  table, then do the sums and things on that temp table.
        # Filtering on provenance tags makes this more complicated, so
        #  we'll do a different query if we're doing that.  Truthfully,
        #  asking for all provenance tags is going to be a mess for the
        #  user....  perhaps we should disable it?
        subdict = {}
        if data['provenancetag'] is None:
            q = ( 'SELECT e._id, e.filepath, e.mjd, e.target, e.project, e.filter, e.filter_array, e.exp_time, '
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
            q = ( 'SELECT e._id, e.filepath, e.mjd, e.target, e.filter, e.project, e.filter_array, e.exp_time, '
                  '       i._id AS imgid, s._id AS subid, sl._id AS slid, sl.num_sources, '
                  '       COUNT(m._id) AS num_measurements '
                  'INTO TEMP TABLE temp_imgs '
                  'FROM exposures e '
                  'LEFT JOIN ( '
                  '  SELECT im._id, im.exposure_id FROM images im '
                  '  INNER JOIN provenance_tags impt ON impt.provenance_id=im.provenance_id '
                  '                                  AND impt.tag=%(provtag)s '
                  ') i ON i.exposure_id=e._id '
                  'LEFT JOIN ( '
                  '  SELECT su._id, ias.upstream_id FROM images su '
                  '  INNER JOIN image_upstreams_association ias ON ias.downstream_id=su._id AND su.is_sub '
                  '  INNER JOIN provenance_tags supt ON supt.provenance_id=su.provenance_id '
                  '                                  AND supt.tag=%(provtag)s '
                  ') s ON s.upstream_id=i._id '
                  'LEFT JOIN ( '
                  '  SELECT sli._id, sli.image_id, sli.num_sources FROM source_lists sli '
                  '  INNER JOIN provenance_tags slpt ON slpt.provenance_id=sli.provenance_id '
                  '                                  AND slpt.tag=%(provtag)s '
                  ') sl ON sl.image_id=s._id '
                  'LEFT JOIN ( '
                  '  SELECT cu._id, cu.sources_id FROM cutouts cu '
                  '  INNER JOIN provenance_tags cupt ON cu.provenance_id=cupt.provenance_id '
                  '                                  AND cupt.tag=%(provtag)s '
                  ') c ON c.sources_id=sl._id '
                  'LEFT JOIN ( '
                  '  SELECT meas._id, meas.cutouts_id FROM measurements meas '
                  '  INNER JOIN provenance_tags mept ON mept.provenance_id=meas.provenance_id '
                  '                                  AND mept.tag=%(provtag)s '
                  ') m ON m.cutouts_id=c._id '
                  'INNER JOIN provenance_tags ept ON ept.provenance_id=e.provenance_id AND ept.tag=%(provtag)s '
                 )
            subdict['provtag'] = data['provenancetag']
        if ( data['projects'] is not None ) or ( t0 is not None ) or ( t1 is not None ):
            q += 'WHERE '
            _and = ''
            if data['projects'] is not None:
                q += f'{_and}e.project IN %(projects)s '
                subdict['projects'] = tuple( data['projects'] )
                _and = 'AND '
            if t0 is not None:
                q += f'{_and}e.mjd >= %(t0)s '
                subdict['t0'] = t0
                _and = 'AND '
            if t1 is not None:
                q += f'{_and}e.mjd <= %(t1)s '
                subdict['t1'] = t1
                _and = 'AND '

        q += 'GROUP BY e._id, i._id, s._id, sl._id, sl.num_sources '

        cursor.execute( q, subdict )

        # Now run a second query to count and sum those things
        # These numbers will be wrong (double-counts) if not filtering on a provenance tag, or if the
        #   provenance tag includes multiple provenances for a given step!
        q = ( 'SELECT t._id, t.filepath, t.mjd, t.target, t.project, t.filter, t.filter_array, t.exp_time, '
              '  COUNT(t.subid) AS num_subs, SUM(t.num_sources) AS num_sources, '
              '  SUM(t.num_measurements) AS num_measurements '
              'INTO TEMP TABLE temp_imgs_2 '
              'FROM temp_imgs t '
              'GROUP BY t._id, t.filepath, t.mjd, t.target, t.project, t.filter, t.filter_array, t.exp_time '
             )

        cursor.execute( q )

        # Run a third query to count reports
        subdict = {}
        q = ( 'SELECT t._id, t.filepath, t.mjd, t.target, t.project, t.filter, t.filter_array, t.exp_time, '
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
                   '  INNER JOIN provenance_tags rept ON rept.provenance_id=re.provenance_id '
                   '                                  AND rept.tag=%(provtag)s '
                   ') r ON r.exposure_id=t._id '
                  )
            subdict['provtag'] = data['provenancetag']
        # I wonder if making a primary key on the temp table would be more efficient than
        #    all these columns in GROUP BY?  Investigate this.
        q += ( 'GROUP BY t._id, t.filepath, t.mjd, t.target, t.project, t.filter, t.filter_array, t.exp_time, '
               '  t.num_subs, t.num_sources, t.num_measurements ' )

        cursor.execute( q, subdict  )
        columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }

        ids = []
        name = []
        mjd = []
        target = []
        project = []
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
            project.append( row[columns['project']] )
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
                 'projects': data['projects'],
                 'exposures': {
                     'id': ids,
                     'name': name,
                     'mjd': mjd,
                     'project': project,
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


# ======================================================================

class ExposureImages( BaseView ):
    def do_the_things( self, expid, provtag ):
        cursor = self.conn.cursor()

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


# ======================================================================

class PngCutoutsForSubImage( BaseView ):
    def do_the_things(  self, exporsubid, provtag, issubid, nomeas, limit=None, offset=0 ):
        exporsubid = asUUID( exporsubid )
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

        cursor = self.conn.cursor()

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
        q = ( 'SELECT c.filepath,s._id AS subimageid,sl.filepath AS sources_path,s.section_id '
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
        sectionids = { c[cols['subimageid']]: c[cols['section_id']] for c in rows }
        cutoutsfiles = { c[cols['subimageid']]: c[cols['filepath']] for c in rows }
        # sourcesfiles = { c[cols['subimageid']]: c[cols['sources_path']] for c in rows }
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
        cursor.execute( q, subdict )
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
            hdf5files[ subid ] = h5py.File( pathlib.Path( cfg.value( 'archive.local_read_dir' ) )
                                            / cutoutsfiles[subid], 'r' )

        def append_to_retval( subid, index_in_sources, section_id, row ):
            retval['cutouts']['source_index'].append( index_in_sources )
            grp = hdf5files[ subid ][f'source_index_{index_in_sources}']
            # In our subtractions, we scale the ref image to the new
            #   image so they share the same zeropoint.  When making
            #   cutouts, we background-subtract both the ref and the
            #   new.  So, we want to share the flux-to-greyscale mapping
            #   for ref and new as that way they can be meaningfully
            #   compared visually.
            vmin, vmax = scaler.get_limits( grp['new_data'] )
            scalednew = ( grp['new_data'] - vmin ) * 255. / ( vmax - vmin )
            scaledref = ( grp['ref_data'] - vmin ) * 255. / ( vmax - vmin )
            # However, use a different mapping for the sub image.  It's
            #   possible that the transient will be a lot dimmer than
            #   the host galaxy, so if we use the same scaling we used
            #   for the new, then the transient won't be visible (all of
            #   the transient data will get mapped to near-sky-level
            #   greys.)
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
            retval['cutouts']['section_id'].append( section_id )
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

            if row is None:
                retval['cutouts']['rb'].append( None )
                retval['cutouts']['rbcut'].append( None )
                retval['cutouts']['is_bad'].append( True )
                retval['cutouts']['objname'].append( None )
                retval['cutouts']['is_test'].append( None )
                retval['cutouts']['is_fake'].append( None )
                flux = None
                dflux = None
                aperrad= 0.
            else:
                retval['cutouts']['rb'].append( row[cols['score']] )
                retval['cutouts']['rbcut'].append( None if row[cols['_algorithm']] is None
                                                   else DeepScore.get_rb_cut( row[cols['_algorithm']] ) )
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
        already_done = set()
        for row in rows:
            subid = row[cols['subid']]
            index_in_sources = row[ cols['index_in_sources'] ]
            section_id = row[ cols['section_id'] ]
            append_to_retval( subid, index_in_sources, section_id, row )
            already_done.add( index_in_sources )

        # Second: if requested, put in detections that didn't pass the initial cuts
        if nomeas:
            for subid, section_id in sectionids.items():
                # WORRY -- if the cutouts files ever have keys other
                #   than one key for each detection (source_index_n keys),
                #   then this next line will break.
                for index_in_sources in range( len( hdf5files[subid] ) ):
                    if index_in_sources not in already_done:
                        append_to_retval( subid, index_in_sources, section_id, None )

        for f in hdf5files.values():
            f.close()

        app.logger.debug( f"Returning {len(retval['cutouts']['sub_id'])} cutouts" )
        return retval


# =====================================================================
# Create and configure the flask app

cfg = Config.get()

app = flask.Flask( __name__, instance_relative_config=True )
# app.logger.setLevel( logging.INFO )
app.logger.setLevel( logging.DEBUG )

secret_key = cfg.value( 'webap.flask_secret_key' )
if secret_key is None:
    with open( cfg.value( 'webap.flask_secret_key_file' ) ) as ifp:
        secret_key = ifp.readline().strip()

app.config.from_mapping(
    SECRET_KEY=secret_key,
    SESSION_COOKIE_PATH='/',
    SESSION_TYPE='filesystem',
    SESSION_PERMANENT=True,
    SESSION_FILE_DIR='/sessions',
    SESSION_FILE_THRESHOLD=1000,
)
server_session = flask_session.Session( app )

# Import and configure the auth subapp
sys.path.insert( 0, pathlib.Path(__name__).resolve().parent )
import rkauth_flask

kwargs = {
    'db_host': cfg.value( 'db.host' ),
    'db_port': cfg.value( 'db.port' ),
    'db_name': cfg.value( 'db.database' ),
    'db_user': cfg.value( 'db.user' ),
    'db_password': cfg.value( 'db.password' )
}
if kwargs['db_password'] is None:
    if cfg.value( 'db.password_file' ) is None:
        raise RuntimeError( 'In config, one of db.password or db.password_file must be specified' )
    with open( cfg.value( 'db.password_file' ) ) as ifp:
        kwargs[ 'db_password' ] = ifp.readline().strip()

for attr in [ 'email_from', 'email_subject', 'email_system_name',
              'smtp_server', 'smtp_port', 'smtp_use_ssl', 'smtp_username', 'smtp_password' ]:
    kwargs[ attr ] = cfg.value( f'email.{attr}' )
if ( kwargs['smtp_password'] ) is None and ( cfg.value('email.smtp_password_file') is not None ):
    with open( cfg.value('email.smtp_password_file') ) as ifp:
        kwargs['smtp_password'] = ifp.readline().strip()

rkauth_flask.RKAuthConfig.setdbparams( **kwargs )

app.register_blueprint( rkauth_flask.bp )


# Configure urls

urls = {
    "/": MainPage,
    "/provtags": ProvTags,
    "/provtaginfo/<tag>": ProvTagInfo,
    "/cloneprovtag/<existingtag>/<newtag>": CloneProvTag,
    "/cloneprovtag/<existingtag>/<newtag>/<int:clobber>": CloneProvTag,
    "/provenanceinfo/<provid>": ProvenanceInfo,
    "/projects": Projects,
    "/exposures": Exposures,
    "/exposure_images/<expid>/<provtag>": ExposureImages,
    "/png_cutouts_for_sub_image/<exporsubid>/<provtag>/<int:issubid>/<int:nomeas>": PngCutoutsForSubImage,
    "/png_cutouts_for_sub_image/<exporsubid>/<provtag>/<int:issubid>/<int:nomeas>/<int:limit>": PngCutoutsForSubImage,
    ( "/png_cutouts_for_sub_image/<exporsubid>/<provtag>/<int:issubid>/<int:nomeas>/"
      "<int:limit>/<int:offset>" ): PngCutoutsForSubImage,
}

usedurls = {}
for url, cls in urls.items():
    if url not in usedurls.keys():
        usedurls[ url ] = 0
        name = url
    else:
        usedurls[ url ] += 1
        name = f"url.{usedurls[url]}"

    app.add_url_rule( url, view_func=cls.as_view(name), methods=["GET", "POST"], strict_slashes=False )
