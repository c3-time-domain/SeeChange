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
import datetime
import pathlib
import logging
import base64
import uuid
import textwrap
import itertools

import numpy
import h5py
import PIL
import astropy.time
import astropy.visualization
from psycopg import sql

import flask
import flask.logging
import flask_session

from util.config import Config
from util.util import asUUID, listify
from models.provenance import Provenance
from models.enums_and_bitflags import ImageTypeConverter
from models.base import PsycopgConnection, PGDB, FileOnDiskMixin
from models.image import Image
from models.deepscore import DeepScoreSet
from models.fakeset import FakeSet, FakeAnalysis
from models.diaforcedphot import DiaForcedPhot

sys.path.insert( 0, pathlib.Path(__name__).resolve().parent )
from baseview import BaseView


# ======================================================================

class MainPage( BaseView ):
    def dispatch_request( self ):
        return flask.render_template( "seechange_webap.html" )


# ======================================================================

class ProvTags( BaseView ):
    def do_the_things( self ):
        with PGDB() as pgdb:
            rows, _cols = pgdb.execute( 'SELECT DISTINCT ON(tag) tag FROM provenance_tags ORDER BY tag' )
            tags = [ row[0] for row in rows ]
            tags.sort( key=lambda x: ( ( 0 if x=='default'
                                         else 1 if x=='current'
                                         else 2 if x=='default_diaforcedphot'
                                         else 3 )
                                       , x ) )
            return { 'status': 'ok',
                     'provenance_tags': tags
                    }


# ======================================================================

class ProvTagInfo( BaseView ):
    def do_the_things( self, tag ):
        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            cursor.execute( 'SELECT p._id, p.process, p.code_version_id, p.parameters, '
                            '       p.is_bad, p.bad_comment, p.is_outdated, p.replaced_by, '
                            '       p.is_testing, c.version_major, c.version_minor, c.version_patch '
                            'FROM provenance_tags t '
                            'INNER JOIN provenances p ON t.provenance_id=p._id '
                            'INNER JOIN code_versions c ON p.code_version_id=c._id '
                            'WHERE t.tag=%(tag)s ',
                            { 'tag': tag } )
            columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
            rows = cursor.fetchall()

            provorder = { 'acquire_exposure': 0,
                          'manual_reference': 0,
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
    _any_group_required = [ 'root', 'admin' ]

    def do_the_things( self, existingtag, newtag, clobber=0 ):
        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            if clobber:
                q = "DELETE FROM provenance_tags WHERE tag=%(tag)s"
                # app.logger.debug( f"CloneProvTag running {cursor.mogrify(q,{'tag':newtag})}" )
                cursor.execute( q, { 'tag': newtag } )
            else:
                q = "SELECT COUNT(*) FROM provenance_tags WHERE tag=%(tag)s"
                # app.logger.debug( f"CloneProvTag running {cursor.mogrify(q,{'tag':newtag})}" )
                cursor.execute( q, { 'tag': newtag } )
                n = cursor.fetchone()[0]
                if n != 0:
                    return f"Tag {newtag} already exists and clobber was False", 500

            # I could probably do this with a single SQL command if I were
            #   clever enough, except that I'd need to have a server default
            #   on provenance_tags for generating the primary key uuid, and
            #   right now we don't have that.
            q = "SELECT provenance_id FROM provenance_tags WHERE tag=%(tag)s"
            # app.logger.debug( f"ConeProvTag running {cursor.mogrify(q,{'tag':existingtag})}" )
            cursor.execute( q, { 'tag': existingtag } )
            rows = cursor.fetchall()
            for row in rows:
                q = "INSERT INTO provenance_tags(_id,tag,provenance_id) VALUES(%(id)s,%(tag)s,%(provid)s)"
                subdict = { 'id': uuid.uuid4(), 'tag': newtag, 'provid': row[0] }
                # app.logger.debug( f"CloneProvTag running {cursor.mogrify(q,subdict)}" )
                cursor.execute( q, subdict )
            # app.logger.debug( "CloneProvTag comitting" )
            conn.commit()

            return { 'status': 'ok' }


# ======================================================================

class ProvenanceInfo( BaseView ):
    def do_the_things( self, provid ):
        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            cursor.execute( "SELECT p._id, p.process, p.code_version_id, p.parameters, "
                            "       p.is_bad, p.bad_comment, p.is_outdated, p.replaced_by, p.is_testing, "
                            "       c.version_major, c.version_minor, c.version_patch "
                            "FROM provenances p "
                            "INNER JOIN code_versions c ON p.code_version_id=c._id "
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
        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            cursor.execute( 'SELECT DISTINCT ON(project) project FROM exposures ORDER BY project' )
            return { 'status': 'ok',
                     'projects': [ row[0] for row in cursor.fetchall() ]
                    }


# ======================================================================
# This one should be updated for more finding

class Images( BaseView ):
    def do_the_things( self, provenancetag, argstr=None ):
        data = self.argstr_to_args( argstr, { 'startdate': None,
                                              'enddate': None,
                                              'projects': None,
                                              } )
        t0 = None if data['startdate'] is None else astropy.time.Time( data['startdate'], format='isot' ).mjd
        t1 = None if data['enddate'] is None else astropy.time.Time( data['enddate'], format='isot' ).mjd
        try:
            projects = listify( data['projects'], require_string=True )
        except Exception:
            return f"Failed to parse projects \"{data['projects']}\"", 422

        with PGDB() as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT i._id, i.filepath, i.provenance_id, i.mjd, i.ra, i.dec, i.exp_time, i._type, i.filter,
                       i.airmass, i.instrument, i.section_id, i.project, i.target, i.is_sub::int,
                       i.zero_point_estimate, i.fwhm_estimate, i.lim_mag_estimate,
                       i.bkg_mean_estimate, i.bkg_rms_estimate
                FROM images i
                INNER JOIN provenance_tags t ON t.provenance_id=i.provenance_id
                                            AND t.tag={provtag}
                """
            ) ).format( provtag=provenancetag )
            where = "WHERE"
            if t0 is not None:
                q += sql.SQL( "{where} i.mjd>={t}\n" ).format( t=t0, where=sql.SQL( where ) )
                where = "  AND"
            if t1 is not None:
                q += sql.SQL( "{where} i.mjd<={t}\n" ).format( t=t1, where=sql.SQL( where ) )
                where = "  AND"
            if data['projects'] is not None:
                q += sql.SQL( "{where} i.projects=ANY(ARRAY[{projects}])\n"
                             ).format( projects=sql.SQL(",").join( projects ) )
            q += sql.SQL( "ORDER BY i.mjd, i.section_id, i._type\n")
            rows, cols = pgdb.execute( q )

        rval = { 'status': 'ok',
                 'startdate': t0,
                 'enddate': t1,
                 'provenance_tag': provenancetag,
                 'projects': projects,
                 'images': { c: [ r[i] for r in rows ] for i, c in enumerate(cols) }
                 }
        slashre = re.compile( '^.*/([^/]+)$' )
        dex = cols.index( 'filepath' )
        rval['images']['filename'] = []
        for row in rows:
            mat = slashre.search( row[dex] )
            if mat is None:
                rval['images']['filename'].append( "????" )
            else:
                rval['images']['filename'].append( mat.group(1) )

        return rval


# ======================================================================

class Exposures( BaseView ):
    def do_the_things( self, provenancetag, argstr=None ):
        data = self.argstr_to_args( argstr, { 'startdate': None,
                                              'enddate': None,
                                              'projects': None,
                                             } )
        app.logger.debug( f"After parsing, data = {data}" )
        t0 = None if data['startdate'] is None else astropy.time.Time( data['startdate'], format='isot' ).mjd
        t1 = None if data['enddate'] is None else astropy.time.Time( data['enddate'], format='isot' ).mjd
        app.logger.debug( f"t0 = {t0}, t1 = {t1}" )

        with PGDB( dictcursor=True ) as pgdb:
            # Gonna do this in three steps.  First, get all the images with
            #  counts of source lists and counts of measurements in a temp
            #  table, then do the sums and things on that temp table.
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT e._id, e.filepath, e.mjd, e.airmass, e.target, e._type, e._filter, e.project,
                       e.filter_array, e.exp_time,
                       i._id AS imgid, i.fwhm_estimate as fwhm_estimate,
                       i.lim_mag_estimate as lim_mag_estimate,
                       s._id AS subid, ssl._id AS slid, ssl.num_sources,
                       ms.num_measurements
                INTO TEMP TABLE temp_imgs
                FROM exposures e
                LEFT JOIN (
                  SELECT im._id, im.exposure_id, im.fwhm_estimate, im.lim_mag_estimate FROM images im
                  INNER JOIN provenance_tags impt ON impt.provenance_id=im.provenance_id
                                                  AND impt.tag={provtag}
                ) i ON i.exposure_id=e._id
                LEFT JOIN (
                  SELECT sli._id, sli.image_id FROM source_lists sli
                  INNER JOIN provenance_tags slipt ON slipt.provenance_id=sli.provenance_id
                                                   AND slipt.tag={provtag}
                ) sl ON sl.image_id=i._id
                LEFT JOIN (
                  SELECT wc._id, wc.sources_id FROM world_coordinates wc
                  INNER JOIN provenance_tags wcpt ON wcpt.provenance_id=wc.provenance_id
                                                  AND wcpt.tag={provtag}
                ) w ON w.sources_id=sl._id
                LEFT JOIN (
                  SELECT zp._id, zp.wcs_id FROM zero_points zp
                  INNER JOIN provenance_tags zppt ON zppt.provenance_id=zp.provenance_id
                                                  AND zppt.tag={provtag}
                ) z ON z.wcs_id=w._id
                LEFT JOIN (
                  SELECT su._id, isc.new_zp_id FROM images su
                  INNER JOIN provenance_tags supt ON supt.provenance_id=su.provenance_id
                                                  AND supt.tag={provtag}
                  INNER JOIN image_subtraction_components isc ON su._id=isc.image_id
                ) s ON s.new_zp_id=z._id
                LEFT JOIN (
                  SELECT ssli._id, ssli.image_id, ssli.num_sources FROM source_lists ssli
                  INNER JOIN provenance_tags sslpt ON sslpt.provenance_id=ssli.provenance_id
                                                   AND sslpt.tag={provtag}
                ) ssl ON ssl.image_id=s._id
                LEFT JOIN (
                  SELECT cu._id, cu.sources_id FROM cutouts cu
                  INNER JOIN provenance_tags cupt ON cu.provenance_id=cupt.provenance_id
                                                  AND cupt.tag={provtag}
                ) c ON c.sources_id=ssl._id
                LEFT JOIN (
                  SELECT sms._id, sms.cutouts_id, COUNT(m._id) AS num_measurements
                  FROM measurement_sets sms
                  INNER JOIN provenance_tags mspt ON sms.provenance_id=mspt.provenance_id
                                                  AND mspt.tag={provtag}
                  INNER JOIN measurements m ON m.measurementset_id=sms._id
                  GROUP BY sms._id, sms.cutouts_id
                ) ms ON ms.cutouts_id=c._id
                INNER JOIN provenance_tags ept ON ept.provenance_id=e.provenance_id AND ept.tag={provtag}
                """
            ) ).format( provtag=provenancetag )
            if ( data['projects'] is not None ) or ( t0 is not None ) or ( t1 is not None ):
                _and = sql.SQL( "WHERE" )
                if data['projects'] is not None:
                    q += sql.SQL( "{_and} e.project=ANY(ARRAY[{projects}]))\n"
                                 ).format( _and=_and, projects=sql.SQL(",".join(data['projects'])) )
                    _and = sql.SQL ("  AND" )
                if t0 is not None:
                    q += sql.SQL( "{_and} e.mjd >= {t0}\n" ).format( _and=_and, t0=t0 )
                    _and = sql.SQL( "  AND" )
                if t1 is not None:
                    q += sql.SQL( "{_and} e.mjd <= {t1}\n" ).format( _and=_and, t1=t1 )
                    _and = sql.SQL( "  AND" )
            app.logger.debug( "Exposures getting images and counts of measurements" )
            pgdb.execute_nofetch( q )

            # Now run a second query to count and sum those things
            # These numbers will be wrong (double-counts) if not filtering on a provenance tag, or if the
            #   provenance tag includes multiple provenances for a given step!
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT t._id, t.filepath, t.mjd, t.airmass, t.target, t.project,
                       t._type, t._filter, t.filter_array, t.exp_time,
                       AVG(t.fwhm_estimate) as seeingavg,
                       AVG(t.lim_mag_estimate) AS limmagavg,
                       SUM( CASE WHEN t.subid IS NULL THEN 0 ELSE 1 END ) AS num_subs,
                       SUM( CASE WHEN t.num_sources IS NULL THEN 0 ELSE t.num_sources END ) AS num_sources,
                       SUM( CASE WHEN t.num_measurements IS NULL THEN 0 ELSE num_measurements END )
                         AS num_measurements
                INTO TEMP TABLE temp_imgs_2
                FROM temp_imgs t
                GROUP BY t._id, t.filepath, t.mjd, t.airmass, t.target, t.project,
                         t._type, t._filter, t.filter_array, t.exp_time
                """
            ) )
            app.logger.debug( "Exposures summing images and measurements" )
            pgdb.execute_nofetch( q )

            # Previous one inserted into a temp table because we will also
            #   need to use it to join in the reports query.  Also,
            #   at some point in the future, we might be able to have
            #   the reports join do everything and only need to get
            #   results at the end of that.
            app.logger.debug( "Exposures getting results" )
            exposureinfo = pgdb.execute( "SELECT * FROM temp_imgs_2 ORDER BY mjd" )

            # Run a third query to count reports.  Tried using
            #   models/report.py::Report.query_for_reports, but it
            #   was horribly slow as it exploded a JSON and then
            #   sorted it all even though it wasn't necessary.
            #   I might be able to make that better.  But, for now,
            #   just pull down *all* reports, and filter in python.
            #   This is still horribly slow because we fetch way too much,
            #   at least if there have been lots of reruns for exposures,
            #   which is the case as of this writing.
            app.logger.debug( f"Getting provenance tags for {provenancetag}" )
            rows = pgdb.execute( sql.SQL( "SELECT provenance_id FROM provenance_tags WHERE tag={tag}" )
                                 .format( tag=provenancetag ) )
            provids = set( r['provenance_id'] for r in rows )
            app.logger.debug( "Exposures getting reports" )
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT r.*
                FROM temp_imgs_2 t
                INNER JOIN reports r ON t._id=r.exposure_id
                ORDER BY t._id, r.section_id, r.start_time DESC
                """
            ) )
            reportinfo = pgdb.execute( q )
            # Filter the reports so that we only keep ones where all of the process
            #   provenance ids are the right ones, keeping just the latest report
            #   that matches for any given id and section.
            app.logger.debug( "Exposures filtering reports" )
            exp_errors = { e['_id']: 0 for e in exposureinfo }
            exp_successim = { e['_id']: 0 for e in exposureinfo }
            curid = ( None, None )
            for row in reportinfo:
                if ( row['exposure_id'], row['section_id'] ) == curid:
                    continue
                if all( v in provids for v in row['process_provid'].values() ):
                    curid = ( row['exposure_id'], row['section_id'] )
                    exp_errors[ row['exposure_id'] ] += 1 if row['error_message'] is not None else 0
                    exp_successim[ row['exposure_id'] ] += 1 if row['success'] else 0

            ids = []
            name = []
            mjd = []
            airmass = []
            target = []
            project = []
            imgtype = []
            filtername = []
            exp_time = []
            seeingavg = []
            limmagavg = []
            n_subs = []
            n_sources = []
            n_measurements = []
            n_successim = []
            n_errors = []

            slashre = re.compile( '^.*/([^/]+)$' )
            for row in exposureinfo:
                ids.append( row['_id'] )
                match = slashre.search( row['filepath'] )
                if match is None:
                    name.append( row['filepath'] )
                else:
                    name.append( match.group(1) )
                mjd.append( row['mjd'] )
                airmass.append( row['airmass'] )
                target.append( row['target'] )
                project.append( row['project'] )
                # app.logger.debug( f"filter={row['_filter']} type {row['_filter']}; "
                #                   f"filter_array={row['filter_array']} type {row['filter_array']}" )
                imgtype.append( ImageTypeConverter.to_string( row['_type'] ) )
                filtername.append( row['_filter'] )
                exp_time.append( row['exp_time'] )
                seeingavg.append( row['seeingavg'] )
                limmagavg.append( row['limmagavg'] )
                n_subs.append( row['num_subs'] )
                n_sources.append( row['num_sources'] )
                n_measurements.append( row['num_measurements'] )
                n_successim.append( exp_successim[ row['_id'] ] )
                n_errors.append( exp_errors[ row['_id'] ] )

            app.logger.debug( "Exposures returning" )
            return { 'status': 'ok',
                     'startdate': t0,
                     'enddate': t1,
                     'provenance_tag': provenancetag,
                     'projects': data['projects'],
                     'exposures': {
                         'id': ids,
                         'name': name,
                         'mjd': mjd,
                         'airmass': airmass,
                         'project': project,
                         'target': target,
                         'imgtype': imgtype,
                         'filter': filtername,
                         'exp_time': exp_time,
                         'seeingavg': seeingavg,
                         'limmagavg': limmagavg,
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
        with PGDB( dictcursor=True ) as pgdb:
            q = sql.SQL( "SELECT *, substring(filepath FROM '/?([^/]+)$/') AS filename\n"
                         "FROM exposures WHERE _id={expid}" ).format( expid=expid )
            rows = pgdb.execute( q )
            if len(rows) == 0:
                raise ValueError( f"Unknown exposure {expid}" )
            if len(rows) > 1:
                raise RuntimeError( f"More than one exposure with id {expid}; this should never happen." )
            exposure_info = rows[0]

            q = sql.SQL( textwrap.dedent(
                """\
                SELECT i._id, i.filter, i.section_id, i.filepath,
                       i.fwhm_estimate, i.lim_mag_estimate, i.zero_point_estimate,
                       substring( i.filepath FROM '/?([^/]+)$' ) AS filename,
                       s._id IS NOT NULL AS has_sources,
                       w._id IS NOT NULL AS has_wcs,
                       z._id IS NOT NULL AS has_zp,
                       sub._id IS NOT NULL AS has_sub,
                       sub._id AS subid,
                       CASE WHEN sub._id IS NULL THEN NULL
                            ELSE substring( sub.filepath FROM '/?([^/]+)$') END AS subfilename,
                       dets.ncutout,
                       dets.ngoodmeas,
                       dets.nmeas
                FROM (
                   SELECT i._id, i.filter, i.section_id, i.filepath,
                          i.fwhm_estimate, i.lim_mag_estimate, i.zero_point_estimate
                   FROM images i
                   INNER JOIN provenance_tags t ON i.provenance_id=t.provenance_id
                   WHERE i.exposure_id={expid}
                     AND t.tag={provtag}
                ) i
                LEFT JOIN (
                   SELECT s._id, s.image_id, s.num_sources
                   FROM source_lists s
                   INNER JOIN provenance_tags t ON s.provenance_id=t.provenance_id AND t.tag={provtag}
                ) s on s.image_id=i._id
                LEFT JOIN (
                   SELECT w._id, w.sources_id
                   FROM world_coordinates w
                   INNER JOIN provenance_tags t ON w.provenance_id=t.provenance_id AND t.tag={provtag}
                ) w ON w.sources_id=s._id
                LEFT JOIN (
                   SELECT z._id, z.wcs_id
                   FROM zero_points z
                   INNER JOIN provenance_tags t ON z.provenance_id=t.provenance_id AND t.tag={provtag}
                ) z ON z.wcs_id=w._id
                LEFT JOIN (
                   SELECT i._id, i.filepath, isc.new_zp_id AS newzpid
                   FROM images i
                   INNER JOIN image_subtraction_components isc ON i._id=isc.image_id
                   INNER JOIN provenance_tags t ON i.provenance_id=t.provenance_id AND t.tag={provtag}
                ) sub ON sub.newzpid=z._id
                LEFT JOIN (
                   SELECT s.image_id, s.num_sources AS ncutout, ms._id,
                          SUM( CASE WHEN m.is_bad THEN 0 ELSE 1 END ) AS ngoodmeas,
                          COUNT( m._id ) AS nmeas
                   FROM source_lists s
                   INNER JOIN cutouts c ON c.sources_id=s._id
                   INNER JOIN measurement_sets ms ON ms.cutouts_id=c._id
                   INNER JOIN measurements m ON m.measurementset_id=ms._id
                   INNER JOIN provenance_tags t ON ms.provenance_id=t.provenance_id AND t.tag={provtag}
                   GROUP BY s.image_id, s.num_sources, ms._id
                ) dets ON dets.image_id=sub._id
                ORDER BY i.section_id
                """
            ) ).format( provtag=provtag, expid=expid )
            imagerows = pgdb.execute( q )

            # Get reports
            # We want the reports were all the provenances in
            #   process_provid are tagged with the right provenance tag.
            #   Report.query_for_reports is supposed to be cleverl SQL
            #   that does all this server side, but I
            #   suspect^H^H^H^H^H^H^H know it has performance problems.
            #   TODO look into that.  In the mean time, just pull down
            #   ALL the reports for ALL of the epxosures, and filter in
            #   python
            allreports = pgdb.execute( sql.SQL( "SELECT * FROM reports WHERE exposure_id={expid}\n"
                                                "ORDER BY modified DESC" )
                                       .format( expid=expid ) )

            # ...aaaaand, we have to get the provenance tags for all the provenances
            #    so we can figure out which ones are OK
            allprovs = set( itertools.chain( *( r['process_provid'].values() for r in allreports ) ) )
            rows = pgdb.execute( sql.SQL( "SELECT provenance_id FROM provenance_tags\n"
                                          "WHERE tag={tag} AND provenance_id=ANY(ARRAY[{provs}])" )
                                 .format( tag=provtag, provs=sql.SQL(",").join( allprovs ) ) )
            okprovs = set( r['provenance_id'] for r in rows )

            reports = [ r for r in allreports if all( v in okprovs for v in r['process_provid'].values() ) ]

        # Attach reports to imagerows.  There could be more than one per image because we might have
        #   stopped and started different stages, so try to merge them
        for imagerow in imagerows:
            myreports = [ r for r in reports if r['section_id'] == imagerow['section_id'] ]
            if len(myreports) == 0:
                imagerow['report'] = { 'exposure_id': None,
                                       'section_id': None,
                                       'start_time': datetime.datetime( 1970, 1, 1 ),
                                       'finish_time': None,
                                       'success': False,
                                       'node_id': None,
                                       'cluster_id': None,
                                       'error_type': None,
                                       'error_step': None,
                                       'error_message': None,
                                       'warnings': None,
                                       'process_memory': {},
                                       'process_runtime': {},
                                       'progress_steps_bitflag': 0,
                                       'products_exist_bitflag': 0,
                                       'products_committed_bitflag': 0,
                                       'created_at': datetime.datetime( 1970, 1, 1 ),
                                       'modiifed': datetime.datetime( 1970, 1, 1 ),
                                       '_id': None,
                                       'image_id': None,
                                       'process_provid': None
                                      }
            else:
                imagerow['report'] = myreports[0]
                for report in myreports[1:]:
                    if ( ( imagerow['report']['warnings'] is not None ) and
                         ( len(imagerow['report']['warnings']) > 0 ) and
                         ( report['warnings'] is not None ) and
                         ( len(report['warnings']) > 0 )
                        ):
                        imagerow['report']['warnings'] = ( report['warnings'] +
                                                           '\n***|***|***\n' +
                                                           imagerow['report']['warnings'] )
                    imagerow['report']['process_memory'].update( report['process_memory'] )
                    imagerow['report']['process_runtime'].update( report['process_runtime'] )
                    imagerow['report']['progress_steps_bitflag'] &= report['progress_steps_bitflag']
                    imagerow['report']['products_exist_bitflag'] &= report['products_exist_bitflag']
                    imagerow['report']['products_committed_bitflag'] &= report['products_committed_bitflag']


        retval = { 'status': 'ok',
                   'provenancetag': provtag,
                   'exposure': exposure_info,
                   'images': imagerows }

        # app.logger.debug( f"exposure_images returning {retval}" )
        return retval


# ======================================================================

class ExposureReports( BaseView ):
    def do_the_things( self, expid, provtag ):
        # As in Exposures, going to pull down ALL reports for the exposure,
        #  then filter later.
        with PGDB( dictcursor=True ) as pgdb:
            rows = pgdb.execute( sql.SQL( "SELECT provenance_id FROM provenance_tags WHERE tag={provtag}" )
                                 .format( provtag=provtag ) )
            okprovs = set( r['provenance_id'] for r in rows )
            rows = pgdb.execute( sql.SQL( "SELECT * FROM reports WHERE exposure_id={expid}\n"
                                          "ORDER BY start_time DESC" )
                                 .format( expid=expid ) )
            retval = { 'status': 'ok',
                       'reports': {} }
            for row in rows:
                if row['section_id'] in retval['reports']:
                    continue
                if all( v in okprovs for v in row['process_provid'].values() ):
                    retval['reports'][row['section_id']] = row

            return retval


# ======================================================================

class ImageInfo( BaseView ):
    def do_the_things( self, imgid ):
        with PGDB( dictcursor=True ) as pgdb:
            try:
                img = Image.get_by_id( imgid, pgdb=pgdb )
            except Exception:
                img = None
            if img is None:
                img = Image.get_by_field_value( "filepath", imgid, pgdb=pgdb )
                if len( img ) > 1:
                    return f"Multiple images with filepath {imgid}; this should never happen.", 422
                elif len(img) == 0:
                    img = None
                else:
                    img = img[0]
            if img is None:
                return f"Could not find image {imgid}", 422

            imgdict = img.to_dict()

            if img.is_sub:
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT i._id AS new_id, wi._id AS warped_ref_id, z.zp AS new_zp, rz.zp AS ref_zp
                    FROM image_subtraction_components isc
                    INNER JOIN zero_points z ON z._id=isc.new_zp_id
                    INNER JOIN world_coordinates w ON w._id=z.wcs_id
                    INNER JOIN source_lists s ON s._id=w.sources_id
                    INNER JOIN images i ON i._id=s.image_id
                    INNER JOIN refs r ON r._id=isc.ref_id
                    INNER JOIN zero_points rz ON rz._id=r.zp_id
                    LEFT JOIN source_lists ws ON isc.warped_ref_sources_id=ws._id
                    LEFT JOIN images wi ON ws.image_id=wi._id
                    WHERE isc.image_id={imid}
                    """
                ) ).format( imid=img.id )
                rows = pgdb.execute( q )
                if len(rows) == 0:
                    return f"Failed to get subtraction parents for subtraction image {imgid}", 422
                elif len(rows) > 1:
                    return "Database corruption in seechange_webap.py::ImageInfo; this should never happen.", 422
                imgdict['new_id'] = rows[0]['new_id']
                imgdict['warped_ref_id'] = rows[0]['warped_ref_id']
                imgdict['new_zp'] = rows[0]['new_zp']
                imgdict['ref_zp'] = rows[0]['ref_zp']

            elif img.is_coadd:
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT i._id AS imgid, w._id AS wcsid, z.zp
                    FROM image_coadd_comonent icc
                    INNER JOIN zero_points z ON icc.zp_id=zp._id
                    INNER JOIN world_coordinates w ON z.wcs_id=w._id
                    INNER JOIN source_lists s ON w.sources_id=s._id
                    INNER JOIN images i ON s.image_id=i._id
                    WHERE icc.coadd_image_id={imid}
                    """
                ) ).format( imid=img.id )
                rows = pgdb.execute( q )
                imgdict[ 'coadd_component_zp' ] = [ r['zp'] for r in rows ]
                imgdict[ 'coadd_component_wcs_id' ] = [ r['imgid'] for r in rows ]
                imgdict[ 'coadd_component_image_id' ] = [ r['wcsid'] for r in rows ]

            elif img.is_trim:
                # ... should this be elif?  I think all of is_sub, is_coadd, is_trim are mutually exclusive
                q = sql.SQL( "SELECT * FROM image_trim_parent WHERE image_id={imgid}" ).format( imgid=img.id )
                rows = pgdb.execute( q )
                if len(rows) == 0:
                    return f"Failed to get trim parents for trimmed image {imgid}", 422
                elif len(rows) > 1:
                    return "Database corruption in seechange_webap.py::ImageInfo; this should never happen.", 422
                imgdict.update( rows[0] )

            return imgdict


# ======================================================================

class ImageData( BaseView ):
    def do_the_things( self, exporimid, provtag=None, sectionid=None, sub=0 ):
        if sectionid is None:
            q = sql.SQL( "SELECT * FROM images WHERE _id={imgid}" ).format( imgid=exporimid )

        else:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT {t}.*
                FROM images i
                INNER JOIN provenance_tags t ON i.provenance_id=t.provenance_id AND t.tag={provtag}
                """
            ) ).format( provtag=provtag, t=sql.Identifier('sub' if sub else 'i') )
            if sub and ( sectionid is not None ):
                q += sql.SQL( textwrap.dedent(
                    """\
                    INNER JOIN (
                       SELECT s._id, s.image_id
                       FROM source_lists s
                       INNER JOIN provenance_tags t ON s.provenance_id=t.provenance_id AND t.tag={provtag}
                    ) s ON s.image_id=i._id
                    INNER JOIN (
                       SELECT w._id, w.sources_id
                       FROM world_coordinates w
                       INNER JOIN provenance_tags t ON w.provenance_id=t.provenance_id AND t.tag={provtag}
                    ) w ON w.sources_id=s._id
                    INNER JOIN (
                       SELECT z._id, z.wcs_id
                       FROM zero_points z
                       INNER JOIN provenance_tags t ON z.provenance_id=t.provenance_id AND t.tag={provtag}
                    ) z ON z.wcs_id=w._id
                    INNER JOIN (
                       SELECT i._id, i.filepath, isc.new_zp_id AS newzpid
                       FROM images i
                       INNER JOIN image_subtraction_components isc ON i._id=isc.image_id
                       INNER JOIN provenance_tags t ON i.provenance_id=t.provenance_id AND t.tag={provtag}
                    ) sub ON sub.newzpid=z._id
                    """
                ) ).format( expid=exporimid, secid=sectionid, provtag=provtag )
            q += sql.SQL( textwrap.dedent(
                """\
                WHERE i.exposure_id={expid} AND i.section_id={secid}
                """
            ) ).format( expid=exporimid, secid=sectionid )

        with PGDB( dictcursor=True ) as pgdb:
            rows = pgdb.execute( q )
            if len(rows) > 1:
                imorsub = ( "image" if ( ( sectionid is None ) or ( not sub ) )
                            else f"subtraction from provtag {provtag}" )
                return ( f"Database corruption: found more than one {imorsub} with "
                         f"{f'id {exporimid}' if sectionid is None else f'exposure {exporimid} seciton {sectionid}'}"
                        ), 422
            imgobj = Image.create( **(rows[0]) ) if len(rows) > 0 else None

        if imgobj is None:
            if sectionid is None:
                return f"Failed to find image with id {exporimid}", 422
            else:
                return f"No {'subtraction' if sub else 'image'} for exposure {exporimid} section {sectionid}", 422

        else:
            # Seechange webap standard is little-endian
            data = imgobj.data.astype( "<f4" )
            barf = bytearray( 4 * data.shape[0] * data.shape[1] + 4 )
            barf[0] = data.shape[0] % 256
            barf[1] = data.shape[0] // 256
            barf[2] = data.shape[1] % 256
            barf[3] = data.shape[1] // 256
            barf[4:] = data.data
            app.logger.debug( f"Returning {len(barf)-4} bytes for a {data.shape[0]}×{data.shape[1]} image" )
            return bytes(barf)


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

        with PGDB( dictcursor=True ) as pgdb:
            # Figure out the subids, zeropoints, backgrounds, and apertures we need

            subids = []
            zps = {}
            dzps = {}
            imageids = {}
            newbkgs = {}
            aperradses = {}
            apercorses = {}

            q = sql.SQL( textwrap.dedent(
                """\
                SELECT s._id AS subid, zp.zp, zp.dzp, zp.aper_cor_radii, zp.aper_cors,
                  i._id AS imageid, i.bkg_mean_estimate
                FROM images s
                INNER JOIN image_subtraction_components isc ON isc.image_id=s._id
                INNER JOIN zero_points zp ON isc.new_zp_id=zp._id
                INNER JOIN world_coordinates wcs ON zp.wcs_id=wcs._id
                INNER JOIN source_lists sl ON wcs.sources_id=sl._id
                INNER JOIN images i ON sl.image_id=i._id
                """
            ) )

            if issubid:
                # Don't need to check provenances; got a subtraction id, so going back
                #   from there will be unique
                q += sql.SQL( "WHERE s._id={subid}\n" ).format( subid=exporsubid )
                # ****
                app.logger.debug( f"Sending query {q.as_string()}" )
                # ****
                rows = pgdb.execute( q )
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
                zps[exporsubid] = rows[0]['zp']
                dzps[exporsubid] = rows[0]['dzp']
                imageids[exporsubid] = asUUID( rows[0]['imageid'] )
                newbkgs[exporsubid] = rows[0]['bkg_mean_estimate']
                aperradses[exporsubid] = rows[0]['aper_cor_radii']
                apercorses[exporsubid] = rows[0]['aper_cors']
            else:
                # If we got an exposure ID, we have to make sure only to get subtractions of the
                #   requested provenance tag
                q += sql.SQL( textwrap.dedent(
                    """\
                    INNER JOIN provenance_tags spt ON s.provenance_id=spt.provenance_id
                                                  AND spt.tag={provtag}
                    INNER JOIN exposures e ON i.exposure_id=e._id
                    WHERE e._id={expid} ORDER BY i.section_id
                    """
                ) ).format( provtag=provtag, expid=exporsubid )
                rows = pgdb.execute( q )
                for row in rows:
                    subid = asUUID( row['subid'] )
                    if ( subid in subids ):
                        app.logger.error( f"subid {subid} showed up more than once in zp query" )
                        return { 'status': 'error',
                                 'error': f"subid {subid} showed up more than once in zp query" }
                    subids.append( subid )
                    zps[subid] = row['zp']
                    dzps[subid] = row['dzp']
                    imageids[subid] = asUUID( row['imageid'] )
                    newbkgs[subid] = row['bkg_mean_estimate']
                    aperradses[subid] = row['aper_cor_radii']
                    apercorses[subid] = row['aper_cors']
            app.logger.debug( f'Got {len(subids)} subtractions.' )

            if len(subids) == 0:
                app.loger.debug( "No subtraction images, skipping getting cutouts and measurements." )
                sectionids = {}
                cutoutsfiles = {}
                rows = []
            else:
                app.logger.debug( f"Getting cutouts files for sub images {subids}" )
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT c.filepath,s._id AS subimageid,sl.filepath AS sources_path,s.section_id
                    FROM cutouts c
                    INNER JOIN provenance_tags cpt ON cpt.provenance_id=c.provenance_id AND cpt.tag={provtag}
                    INNER JOIN source_lists sl ON c.sources_id=sl._id
                    INNER JOIN images s ON sl.image_id=s._id
                    WHERE s._id=ANY(ARRAY[{subids}])
                    """
                ) ).format( provtag=provtag, subids=sql.SQL(",").join(subids) )
                rows = pgdb.execute( q )
                sectionids = { asUUID( r['subimageid'] ): r['section_id'] for r in rows }
                cutoutsfiles = { asUUID( r['subimageid'] ): r['filepath'] for r in rows }
                app.logger.debug( f"Got: {len(cutoutsfiles)} cutouts files" )

                # app.logger.debug( f"Getting measurements for sub images {subids}" )
                app.logger.debug( f"Getting measurements for {len(subids)} sub images" )
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT m.ra AS measra, m.dec AS measdec, m.index_in_sources, m.best_aperture,
                           m.flux, m.dflux, m.psfflux, m.dpsfflux, m.is_bad, m.name, m.is_test,
                           m.score, m._algorithm, m.center_x_pixel, m.center_y_pixel, m.x, m.y, m.gfit_x, m.gfit_y,
                           m.major_width, m.minor_width, m.position_angle, m.nbadpix, m.negfrac, m.negfluxfrac,
                           s._id AS subid, s.section_id
                    FROM cutouts c
                    INNER JOIN provenance_tags cpt ON cpt.provenance_id=c.provenance_id AND cpt.tag={provtag}
                    INNER JOIN source_lists sl ON c.sources_id=sl._id
                    INNER JOIN images s ON sl.image_id=s._id
                    INNER JOIN
                      ( SELECT ms.cutouts_id AS meascutid, meas.index_in_sources, meas.ra, meas.dec, meas.is_bad,
                               meas.best_aperture, meas.flux_apertures[meas.best_aperture+1] AS flux,
                               meas.flux_apertures_err[meas.best_aperture+1] AS dflux,
                               meas.flux_psf AS psfflux, meas.flux_psf_err AS dpsfflux,
                               meas.center_x_pixel, meas.center_y_pixel, meas.x, meas.y, meas.gfit_x, meas.gfit_y,
                               meas.major_width, meas.minor_width, meas.position_angle,
                               meas.nbadpix, meas.negfrac, meas.negfluxfrac,
                               obj.name, obj.is_test, score.score, score._algorithm
                        FROM measurements meas
                        INNER JOIN measurement_sets ms ON meas.measurementset_id=ms._id
                        INNER JOIN provenance_tags mpt ON ms.provenance_id=mpt.provenance_id AND mpt.tag={provtag}
                        INNER JOIN objects obj ON meas.object_id=obj._id
                        LEFT JOIN
                          ( SELECT ss.measurementset_id, ss._algorithm, s.index_in_sources, s.score FROM deepscores s
                            INNER JOIN deepscore_sets ss ON s.deepscoreset_id=ss._id
                            INNER JOIN provenance_tags spt ON spt.provenance_id=ss.provenance_id
                                                           AND spt.tag={provtag}
                          ) AS score
                          ON score.measurementset_id=ms._id AND score.index_in_sources=meas.index_in_sources
                    """
                ) ).format( provtag=provtag )
                if not nomeas:
                    q += sql.SQL( "    WHERE NOT meas.is_bad\n" )
                q += sql.SQL( "   ) AS m ON m.meascutid=c._id\n"
                              "WHERE s._id=ANY(ARRAY[{subids}])\n" ).format( subids=sql.SQL(",").join(subids) )
                if data['sortby'] == 'fluxdesc_chip_index':
                    q += sql.SQL( "ORDER BY flux DESC NULLS LAST,s.section_id,m.index_in_sources\n" )
                elif data['sortby'] == 'rbdesc_fluxdesc_chip_index':
                    q += sql.SQL( "ORDER BY is_bad,score DESC NULLS LAST,flux DESC NULLS LAST,\n"
                                  "         s.section_id,m.index_in_sources\n" )
                else:
                    raise RuntimeError( f"Unknown sort criterion {data['sortby']}" )
                if limit is not None:
                    q += sql.SQL( "LIMIT {limit} OFFSET {offset}" ).format( limit=limit, offset=offset )
                # app.logger.debug( f"Sending query to get measurements: {cursor.mogrify(q,subdict)}" )
                rows = pgdb.execute( q )
                # app.logger.debug( f"Got {len(rows)} rows" )

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
                           'cutout_x': [],
                           'cutout_y': [],
                           'x': [],
                           'y': [],
                           'gfit_x': [],
                           'gfit_y': [],
                           'major_width': [],
                           'minor_width': [],
                           'nbadpix': [],
                           'negfrac': [],
                           'negfluxfrac': [],
                           'w': [],
                           'h': [],
                           'new_png': [],
                           'ref_png': [],
                           'sub_png': []
                       }
                      }

            scaler = astropy.visualization.ZScaleInterval( contrast=0.02 )

            # Open all the hdf5 files

            hdf5files = {}
            for subid in cutoutsfiles.keys():
                hdf5files[ subid ] = h5py.File( FileOnDiskMixin.local_path / cutoutsfiles[subid], 'r' )

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

                retval['cutouts']['sub_id'].append( subid )
                retval['cutouts']['image_id'].append( imageids[subid] )
                retval['cutouts']['section_id'].append( section_id )
                retval['cutouts']['new_png'].append( base64.b64encode( newim.getvalue() ).decode('ascii') )
                retval['cutouts']['ref_png'].append( base64.b64encode( refim.getvalue() ).decode('ascii') )
                retval['cutouts']['sub_png'].append( base64.b64encode( subim.getvalue() ).decode('ascii') )
                retval['cutouts']['w'].append( scalednew.shape[0] )
                retval['cutouts']['h'].append( scalednew.shape[1] )
                retval['cutouts']['cutout_x'].append( grp.attrs['new_x'] )
                retval['cutouts']['cutout_y'].append( grp.attrs['new_y'] )

                if row is None:
                    retval['cutouts']['x'].append( None )
                    retval['cutouts']['y'].append( None )
                    retval['cutouts']['gfit_x'].append( None )
                    retval['cutouts']['gfit_y'].append( None )
                    retval['cutouts']['major_width'].append( None )
                    retval['cutouts']['minor_width'].append( None )
                    retval['cutouts']['nbadpix'].append( None )
                    retval['cutouts']['negfrac'].append( None )
                    retval['cutouts']['negfluxfrac'].append( None )
                    retval['cutouts']['rb'].append( None )
                    retval['cutouts']['rbcut'].append( None )
                    retval['cutouts']['is_bad'].append( True )
                    retval['cutouts']['objname'].append( None )
                    retval['cutouts']['is_test'].append( None )
                    flux = None
                    dflux = None
                    aperrad= 0.
                else:
                    retval['cutouts']['x'].append( row['x'] )
                    retval['cutouts']['y'].append( row['y'] )
                    retval['cutouts']['gfit_x'].append( row['gfit_x'] )
                    retval['cutouts']['gfit_y'].append( row['gfit_y'] )
                    retval['cutouts']['major_width'].append( row['major_width'] )
                    retval['cutouts']['minor_width'].append( row['minor_width'] )
                    retval['cutouts']['nbadpix'].append( row['nbadpix'] )
                    retval['cutouts']['negfrac'].append( row['negfrac'] )
                    retval['cutouts']['negfluxfrac'].append( row['negfluxfrac'] )
                    retval['cutouts']['rb'].append( row['score'] )
                    retval['cutouts']['rbcut'].append( None if row['_algorithm'] is None
                                                       else DeepScoreSet.get_rb_cut( row['_algorithm'] ) )
                    retval['cutouts']['is_bad'].append( row['is_bad'] )
                    retval['cutouts']['objname'].append( row['name'] )
                    retval['cutouts']['is_test'].append( row['is_test'] )

                    if row['psfflux'] is None:
                        flux = row['flux']
                        dflux = row['dflux']
                        aperrad = aperradses[subid][ row['best_aperture'] ]
                    else:
                        flux = row['psfflux']
                        dflux = row['dpsfflux']
                        aperrad = 0.

                if flux is None:
                    for field in [ 'flux', 'dflux', 'aperrad', 'mag', 'dmag', 'measra', 'measdec' ]:
                        retval['cutouts'][field].append( None )
                else:
                    mag = -99
                    dmag = -99
                    if ( zps[subid] > 0 ) and ( flux > 0 ):
                        mag = -2.5 * math.log10( flux ) + zps[subid] + apercorses[subid][ row['best_aperture'] ]
                        # Ignore zp and apercor uncertainties
                        dmag = 1.0857 * dflux / flux
                        retval['cutouts']['measra'].append( row['measra'] )
                        retval['cutouts']['measdec'].append( row['measdec'] )
                        retval['cutouts']['flux'].append( flux )
                        retval['cutouts']['dflux'].append( dflux )
                        retval['cutouts']['aperrad'].append( aperrad )
                        retval['cutouts']['mag'].append( mag )
                        retval['cutouts']['dmag'].append( dmag )

            # First: put in all the measurements, in the order we got them
            already_done = set()
            for row in rows:
                subid = asUUID( row['subid'] )
                index_in_sources = row['index_in_sources']
                section_id = row['section_id']
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


# ======================================================================

class FakeAnalysisData( BaseView ):
    def do_the_things( self, expid, provtag, sectionid=None ):
        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            expid = asUUID( expid )

            # Applying the provenance tag filter to the deepscore set, because
            #   that's the lowest thing down on the chain.  (FakeAnalysis doesn't
            #   have a provenance, and FakeSet's provenance doesn't get tagged.)
            q = ( "SELECT fa._id AS fakeanal_id, fa.filepath AS fakeanal_filepath, "
                  "       fs._id AS fakeset_id, fs.filepath AS fakeset_filepath, "
                  "       i.section_id, zp.zp "
                  "FROM fake_analysis fa "
                  "INNER JOIN ( "
                  "  SELECT dsi._id, dsi.measurementset_id FROM deepscore_sets dsi "
                  "  INNER JOIN provenance_tags dsipt ON dsipt.provenance_id=dsi.provenance_id "
                  "                                   AND dsipt.tag=%(provtag)s "
                  ") ds ON fa.orig_deepscore_set_id=ds._id "
                  "INNER JOIN measurement_sets ms ON ds.measurementset_id=ms._id "
                  "INNER JOIN cutouts cu ON ms.cutouts_id=cu._id "
                  "INNER JOIN source_lists d ON cu.sources_id=d._id "
                  "INNER JOIN images su ON d.image_id=su._id "
                  "INNER JOIN image_subtraction_components isc ON su._id=isc.image_id "
                  "INNER JOIN zero_points zp ON isc.new_zp_id=zp._id "
                  "INNER JOIN world_coordinates wc ON zp.wcs_id=wc._id "
                  "INNER JOIN source_lists s ON wc.sources_id=s._id "
                  "INNER JOIN images i ON s.image_id=i._id "
                  "INNER JOIN fake_sets fs ON fs._id=fa.fakeset_id AND fs.zp_id=zp._id "
                  "WHERE i.exposure_id=%(expid)s "
                 )
            subdict = { 'provtag': provtag, 'expid': expid }
            if sectionid is not None:
                q += " AND i.section_id=%(secid)s"
                subdict['secid'] = sectionid

            cursor.execute( q, subdict )
            columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
            rows = cursor.fetchall()

            # It's possible we'll get multple rows back even with a single section id, because somebody could have
            #   rerun the pipeline using a different random seed for the fake injection.  (But see Issue #444.)
            #   So, each value in the sections dictionary is itself an array.  (Which will contain a dictionary of
            #   values, many of which are arrays.  dict→dict→array→dict→arrays.... ufda.)

            retval = { 'status': 'ok',
                       'sections': {} }

            for row in rows:
                secid = row[columns['section_id']]
                if secid  not in retval['sections']:
                    retval['sections'][secid] = []
                fakeset = FakeSet.get_by_id( row[columns['fakeset_id']], session=self.session )
                fakeset.load()
                fakeanal = FakeAnalysis.get_by_id( row[columns['fakeanal_id']], session=self.session )
                fakeanal.load()
                zp = row[columns['zp']]
                # Just sticking numpy arrays directly, in hopes that the json encoding that flask
                #   does handles this right....
                fakeinfo = { 'random_seed': fakeset.random_seed,
                             'fake_x': fakeset.fake_x,
                             'fake_y': fakeset.fake_y,
                             'fake_mag': fakeset.fake_mag,
                             'is_detected': fakeanal.is_detected,
                             'is_kept': fakeanal.is_kept,
                             'is_bad': fakeanal.is_bad,
                             'mag_psf': -2.5 * numpy.log10( fakeanal.flux_psf ) + zp,
                             'mag_psf_err': 2.5 / numpy.log(10) * fakeanal.flux_psf_err / fakeanal.flux_psf,
                             'center_x_pixel': fakeanal.center_x_pixel,
                             'center_y_pixel': fakeanal.center_y_pixel,
                             'x': fakeanal.x,
                             'y': fakeanal.y,
                             'gfit_x': fakeanal.gfit_x,
                             'gfit_y': fakeanal.gfit_y,
                             'major_width': fakeanal.major_width,
                             'minor_width': fakeanal.minor_width,
                             'position_angle': fakeanal.position_angle,
                             'psf_fit_flags': fakeanal.psf_fit_flags,
                             'nbadpix': fakeanal.nbadpix,
                             'negfrac': fakeanal.negfrac,
                             'negfluxfrac': fakeanal.negfluxfrac,
                             'deepscore_algorithm': fakeanal.deepscore_algorithm,
                             'score': fakeanal.score
                            }
                retval['sections'][secid].append( fakeinfo )

            return retval


# ======================================================================

class DiaForcedPhotForObj( BaseView ):
    def do_the_things( self, objid, provtag ):
        with PGDB() as pgdb:
            prov = Provenance.get_for_tag( provtag, process='diaforcedphot', pgdb=pgdb )
            if prov is None:
                return f"Can't find provenance for tag {provtag} process diaforcedphot", 422

            ltcv, obj = DiaForcedPhot.get_lightcurve_for_object( objid, prov, include_apertures=True,
                                                                 include_raw=True, include_im_ids=True,
                                                                 include_pos=True, include_phot_ids=True,
                                                                 return_format="dictoflists", return_obj=True )
            ltcv['objid'] = str(obj.id)
            ltcv['objname'] = obj.name
            ltcv['provtag'] = provtag
            return ltcv


# ======================================================================

class DiaForcedPhotClips( BaseView ):
    def do_the_things( self, photid, clipsize=25 ):
        if clipsize % 2 == 0:
            clipsize += 1
        with PGDB() as pgdb:
            phot = DiaForcedPhot.get_by_id( photid, pgdb=pgdb )
            clips = phot.get_clips( clip_size=clipsize )

        clipsize = clips.shape[1]
        barf = bytearray( 4 * clipsize * clipsize + 4 )
        barf[0] = clipsize % 256
        barf[1] = clipsize // 256
        barf[2] = clipsize % 256
        barf[3] = clipsize // 256
        barf[4:] = clips.astype( "<f4" ).data

        return bytes(barf)


# ======================================================================

class AllObjsWithDiaForcedPhot( BaseView ):
    def do_the_things( self, provtag ):
        with PGDB() as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT DISTINCT ON(o.name) o._id, o.name, o.ra, o.dec
                FROM dia_forced_photometry p
                INNER JOIN objects o ON p.object_id=o._id
                INNER JOIN provenance_tags t ON p.provenance_id=t.provenance_id
                WHERE t.tag={provtag}
                ORDER BY o.name
                """
            ) ).format( provtag=provtag )
            rows, cols = pgdb.execute( q )

        return { c: [ r[i] for r in rows ] for i, c in enumerate(cols) }


# =====================================================================
# =====================================================================
# =====================================================================
# Create and configure the flask app

app = flask.Flask( __name__, instance_relative_config=True )

# Logger
_formatter = logging.Formatter( '[%(asctime)s - %(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
flask.logging.default_handler.setFormatter( _formatter )

# SeeChange Config.  Put logger in debug so we see logs about what config overrides are read.
app.logger.setLevel( "DEBUG" )
cfg = Config.get()
if not cfg.value( "webap.verbose", True ):
    app.logger.setLevel( "INFO" )
app.logger.debug( "Set log level to DEBUG" )


# Regular flask setup
secret_key = cfg.value( 'webap.flask_secret_key' )
if secret_key is None:
    with open( cfg.value( 'webap.flask_secret_key_file' ) ) as ifp:
        secret_key = ifp.readline().strip()

app.config.from_mapping(
    SECRET_KEY=secret_key,
    SESSION_COOKIE_PATH='/',
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_TYPE='filesystem',
    SESSION_PERMANENT=True,
    SESSION_FILE_DIR='/sessions',
    SESSION_FILE_THRESHOLD=1000,
)
server_session = flask_session.Session( app )

# Import and configure the auth subapp
sys.path.insert( 0, pathlib.Path(__name__).resolve().parent )
import rkauth_flask
import conductor
import ltcv

# Make sure we have the info we need to connect to the database and send password upate email
kwargs = {
    'usegroups': True,
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

# Register urls from subapps
app.register_blueprint( rkauth_flask.bp )
app.register_blueprint( conductor.bp )
app.register_blueprint( ltcv.bp )

# Configure urls
urls = {
    "/": MainPage,
    "/provtags": ProvTags,
    "/provtaginfo/<tag>": ProvTagInfo,
    "/cloneprovtag/<existingtag>/<newtag>": CloneProvTag,
    "/cloneprovtag/<existingtag>/<newtag>/<int:clobber>": CloneProvTag,
    "/provenanceinfo/<provid>": ProvenanceInfo,
    "/projects": Projects,
    "/exposures/<provenancetag>": Exposures,
    "/exposures/<provenancetag>/<path:argstr>": Exposures,
    "/exposure_images/<expid>/<provtag>": ExposureImages,
    "/exposure_reports/<expid>/<provtag>": ExposureReports,
    "/images/<provenancetag>": Images,
    "/images/<provenancetag>/<path:argstr>": Images,
    "/image_info/<imgid>": ImageInfo,
    "/image_data/<exporimid>": ImageData,
    "/image_data/<exporimid>/<provtag>/<sectionid>": ImageData,
    "/image_data/<exporimid>/<provtag>/<sectionid>/<int:sub>": ImageData,
    "/png_cutouts_for_sub_image/<exporsubid>/<provtag>/<int:issubid>/<int:nomeas>": PngCutoutsForSubImage,
    "/png_cutouts_for_sub_image/<exporsubid>/<provtag>/<int:issubid>/<int:nomeas>/<int:limit>": PngCutoutsForSubImage,
    ( "/png_cutouts_for_sub_image/<exporsubid>/<provtag>/<int:issubid>/<int:nomeas>/"
      "<int:limit>/<int:offset>" ): PngCutoutsForSubImage,
    "/fakeanalysisdata/<expid>/<provtag>": FakeAnalysisData,
    "/fakeanalysisdata/<expid>/<provtag>/<sectionid>": FakeAnalysisData,
    "/diaforcedphotforobj/<objid>/<provtag>": DiaForcedPhotForObj,
    "/diaforcedphotclips/<photid>": DiaForcedPhotClips,
    "/diaforcedphotclips/<photid>/<int:clipsize>": DiaForcedPhotClips,
    "/objswithdiaforcedphot/<provtag>": AllObjsWithDiaForcedPhot
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
