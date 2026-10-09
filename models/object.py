import io
import uuid
import numbers
import requests
import functools
import textwrap

import numpy as np

import psycopg
from psycopg import sql
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.schema import UniqueConstraint

from astropy.coordinates import SkyCoord
import astropy.units

from models.base import Base, PGDB, SeeChangeBase, PsycopgConnection, UUIDMixin, SpatiallyIndexed
from models.image import Image
from models.zero_point import ZeroPoint
from models.measurements import Measurements, MeasurementSet
from models.deepscore import DeepScore, DeepScoreSet
from pipeline.catalog_tools import download_gaia_dr3
from util.config import Config
from util.logger import SCLogger
from util.util import parse_dateobs, retry_with_sleep

object_name_max_used = sa.Table(
    'object_name_max_used',
    Base.metadata,
    sa.Column( 'year', sa.Integer, primary_key=True, autoincrement=False ),
    sa.Column( 'maxnum', sa.Integer, server_default=sa.sql.elements.TextClause('0') )
)


class Object(Base, UUIDMixin, SpatiallyIndexed):
    __tablename__ = 'objects'

    @declared_attr
    def __table_args__(cls):  # noqa: N805
        return (
            sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
        )

    name = sa.Column(
        sa.String,
        nullable=False,
        unique=True,
        index=True,
        doc='Name of the object (can be internal nomenclature or external designation, e.g., "SN2017abc")'
    )

    is_test = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default='false',
        doc='Boolean flag to indicate if the object is a test object created during testing. '
    )

    is_bad = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default=sa.text("false"),
        index=True,
        doc='Boolean flag to indicate object is bad; only will ever be set manually.'
    )


    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.calculate_coordinates()


    # By definition, objects have no upstreams
    def get_upstream_ids( self, pgdb=None ):
        return []

    # Objects have positions downstream
    # NOTE: they also have measurements and forced photometry downstream,
    #   but we're not including that here.  Reason: mostly this is used
    #   for cleanup  in tests, and those will clean up separately.  In
    #   particular, deleting measurements is fraught because you shouldn't
    #   do that when you aren't also deleting the parent measurement set.
    def get_downstream_ids( self, pgdb=None ):
        downstreams = []
        with PGDB( pgdb ) as pgdb:
            q = sql.SQL( "SELECT _id FROM object_positions WHERE object_id={me}" ).format( me=self.id )
            rows, _cols = pgdb.execute( q )
            downstreams.extend( [ r[0] for r in rows ] )

        return downstreams


    def get_measurements_et_al( self, measurement_prov_id, deepscore_prov_id=None, omit_measurements=[],
                                mjd_min=None, mjd_max=None, min_deepscore=None, pgdb=None, session=None ):
        """Return lists of sundry objects for this Object.

        Parameters
        ----------
        measurement_prov_id : str
          ID of the Provenance of MeasurementSet to search

        deepscore_prov_id : str
          ID of the Provenance od DeepScoreSet to search, or None to omit deepscores.

        omit_measurements : list of uuid, default None
          IDs of measurements explicitly not to include in the list

        mjd_min : float, default None
          If given, minimum mjd of measurements to return

        mjd_max : float, default None
          If given, maximum mjd of measurements to return

        min_deepscore : float, default None
          If given, minimum deepscore of measurements to return

        Returns
        -------
          dict of lists, all lists having the same length
          First four keys are always there; last two only if deepscore_prov_id is not None
           { 'measurements': list of Measurements,
             'measurementsets': list of MeasurementSet,
             'images': list of Image (the difference images, *not* the original science image!),
             'zeropoints': list of ZeroPoint,
             'deepscores': list of DeepScore,
             'deepscoresets': list of DeepScoreSet
           }

        """
        # In Image.from_new_and_ref, we set a lot of the sub image's
        #   properties (crucially, filter and mjd) to be the same as the
        #   new image.  So, for what we need for alerts, we can just use
        #   the sub image.

        # Get all previous sources with the same provenance.
        # The cutouts parent is a SourceList that is
        #   detections on the sub image, so it's parent
        #   is the sub image.  We need that for mjd and filter.
        # But, also, we need to get the sub image's parent
        #   zeropoint, which is the zeropoint of the new
        #   image that went into the sub image.  In subtraction,
        #   we normalize the sub image so it has the same
        #   zeropoint as the new image, so that's also the
        #   right zeropoint to use with the Measurements
        #   that we pull out.
        # And, finally, we have to get the DeepScore objects
        #   associated with the previous measurements.
        #   That's not upstream of anything, so we have to
        #   include the DeepScore provenance in the join condition.

        if ( min_deepscore is not None ) and ( deepscore_prov_id is None ):
            raise ValueError( "Passing min_deepscore requires passing deepscore_prov_id" )

        mjd_min = None if mjd_min is None else parse_dateobs( mjd_min, output='mjd' )
        mjd_max = None if mjd_max is None else parse_dateobs( mjd_max, output='mjd' )

        with PGDB( pgdb if pgdb is not None else session, dictcursor=True ) as pgdb:
            if deepscore_prov_id is not None:
                extra = sql.SQL( ",\n       to_jsonb(d) AS deepscores,\n"
                                 "       to_jsonb(ds) AS deepscore_set\n" )
                fromclause = sql.SQL( textwrap.dedent(
                    """\
                    FROM deepscore_sets ds
                    INNER JOIN deepscores d ON d.deepscoreset_id=ds._id
                    INNER JOIN measurement_sets ms ON ds.measurementset_id=ms._id
                    INNER JOIN measurements m ON m.measurementset_id=ms._id AND m.index_in_sources=d.index_in_sources
                    """
                ) )
                whereclause = sql.SQL( textwrap.dedent(
                    """\
                    WHERE ds.provenance_id={dsprov}
                      AND ms.provenance_id={msprov}
                    """
                ) ).format( dsprov=deepscore_prov_id, msprov=measurement_prov_id )
            else:
                extra = sql.SQL( "" )
                fromclause = sql.SQL( "FROM measurement_sets ms\n"
                                      "INNER JOIN measurements m ON m.measurementset_id=ms._id\n" )
                whereclause = sql.SQL( "WHERE ms.provenance_id={msprov}\n" ).format( msprov=measurement_prov_id )

            q = sql.SQL( textwrap.dedent(
                """\
                SELECT to_jsonb(m) AS measurements,
                       to_jsonb(ms) AS measurement_set,
                       to_jsonb(i) AS image,
                       to_jsonb(z) AS zeropoint{extra}
                {fromclause}
                INNER JOIN cutouts c ON ms.cutouts_id=c._id
                INNER JOIN source_lists s ON c.sources_id=s._id
                INNER JOIN images i ON s.image_id=i._id
                INNER JOIN image_subtraction_components isc ON isc.image_id=i._id
                INNER JOIN zero_points z ON isc.new_zp_id=z._id
                {whereclause}
                  AND m.object_id={objid}
                """ ) ).format( extra=extra, fromclause=fromclause, whereclause=whereclause, objid=self.id )

            if len(omit_measurements) > 0:
                q += sql.SQL( "  AND m._id NOT IN ({mids})\n" ).format( mids=sql.SQL(",").join(omit_measurements) )
            if mjd_min is not None:
                q += sql.SQL( "  AND i.mjd>={mjd}\n" ).format( mjd=mjd_min )
            if mjd_max is not None:
                q += sql.SQL( "  AND i.mjd<={mjd}\n" ).format( mjd=mjd_max)
            if min_deepscore is not None:
                q += sql.SQL( "  AND d.score>={score}\n" ).format( score=min_deepscore )
            q += sql.SQL( "ORDER BY i.mjd" )

            rows = pgdb.execute( q )

            retval = { 'measurements': [ Measurements.create( **(r['measurements']) ) for r in rows ],
                       'measurementsets': [ MeasurementSet.create( **(r['measurement_set']) ) for r in rows ],
                       'images': [ Image.create( **(r['image']) ) for r in rows ],
                       'zeropoints': [ ZeroPoint.create( **(r['zeropoint']) ) for r in rows ]
                      }
            # Using JSONB will have stringified the ids... just not going to worry about it right now...
            # ...when we're done with Issue #516, maybe we can rename all the database _id columns to
            #    id, and then the id setter in UUIDMixin will be run here even when we use JSON.
            if deepscore_prov_id is not None:
                retval['deepscores'] = [ DeepScore.create( **(r['deepscores']) ) for r in rows ]
                retval['deepscoresets'] = [ DeepScoreSet.create( **(r['deepscore_set']) ) for r in rows ]

            return retval


        # with SmartSession( session ) as sess:
        #     if deepscore_prov_id is not None:
        #         q = sess.query( Measurements, MeasurementSet, Image, ZeroPoint, DeepScore, DeepScoreSet )
        #     else:
        #         q = sess.query( Measurements, MeasurementSet, Image, ZeroPoint )

        #     q = ( q.join( MeasurementSet, sa.and_( Measurements.measurementset_id==MeasurementSet._id,
        #                                            MeasurementSet.provenance_id==measurement_prov_id ) )
        #           .join( Cutouts, MeasurementSet.cutouts_id==Cutouts._id )
        #           .join( SourceList, Cutouts.sources_id==SourceList._id )
        #           .join( Image, SourceList.image_id==Image._id )
        #           .join( image_subtraction_components, image_subtraction_components.c.image_id==Image._id )
        #           .join( ZeroPoint, ZeroPoint._id==image_subtraction_components.c.new_zp_id ) )

        #     if deepscore_prov_id is not None:
        #         q = ( q.join( DeepScoreSet, sa.and_( DeepScoreSet.measurementset_id==MeasurementSet._id,
        #                                              DeepScoreSet.provenance_id==deepscore_prov_id ),
        #                       isouter=True )
        #               .join( DeepScore, sa.and_( DeepScore.deepscoreset_id==DeepScoreSet._id,
        #                                          DeepScore.index_in_sources==Measurements.index_in_sources ),
        #                      isouter=True )
        #              )

        #     q = q.filter( Measurements.object_id==self.id )
        #     if len( omit_measurements ) > 0:
        #         q = q.filter( Measurements._id.not_in( omit_measurements ) )
        #     if mjd_min is not None:
        #         q = q.filter( Image.mjd >= mjd_min )
        #     if mjd_max is not None:
        #         q = q.filter( Image.mjd <= mjd_max )
        #     if min_deepscore is not None:
        #         q = q.filter( DeepScore.score >= min_deepscore )
        #     q = q.order_by( Image.mjd )

        #     mess = q.all()

        #     retval = { 'measurements': [ m[0] for m in mess ],
        #                'measurementsets': [ m[1] for m in mess ],
        #                'images': [ m[2] for m in mess ],
        #                'zeropoints': [ m[3] for m in mess ] }
        #     if deepscore_prov_id is not None:
        #         retval['deepscores'] = [ m[4] for m in mess ]
        #         retval['deepscoresets'] = [ m[5] for m in mess ]

        #     return retval


    def get_mean_coordinates(self, sigma=3.0, iterations=3, measurement_list_kwargs=None):
        """Get the mean coordinates of the object.

        Uses the measurements that are loaded using the get_measurements_list method.
        From these, central ra/dec are calculated, using an aperture flux weighted mean.
        Outliers are removed based on the sigma/iterations parameters.

        Parameters
        ----------
        sigma: float, optional
            The sigma to use for the clipping of the measurements. Default is 3.0.
        iterations: int, optional
            The number of iterations to use for the clipping of the measurements. Default is 3.
        measurement_list_kwargs: dict, optional
            The keyword arguments to pass to the get_measurements_list method.

        Returns
        -------
        float, float
            The mean RA and Dec of the object.
        """

        raise RuntimeError( "This is broken until we fix get_measurements_list" )
        measurements = self.get_measurements_list(**(measurement_list_kwargs or {}))

        ra = np.array([m.ra for m in measurements])
        dec = np.array([m.dec for m in measurements])
        flux = np.array([m.flux for m in measurements])
        fluxerr = np.array([m.flux_err for m in measurements])

        good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(flux) & np.isfinite(fluxerr)
        good &= flux > fluxerr * 3.0  # require a 3-sigma detection
        # make sure that if one of these is bad, all are bad
        ra[~good] = np.nan
        dec[~good] = np.nan
        flux[~good] = np.nan

        points = SkyCoord(ra, dec, unit='deg')

        ra_mean = np.nansum(ra * flux) / np.nansum(flux[good])
        dec_mean = np.nansum(dec * flux) / np.nansum(flux[good])
        center = SkyCoord(ra_mean, dec_mean, unit='deg')

        num_good = np.sum(good)
        if num_good < 3:
            iterations = 0  # skip iterative step if too few points

        # clip the measurements
        for i in range(iterations):
            # the 2D distance from the center
            offsets = points.separation(center).arcsec

            scatter = np.nansum(flux * offsets ** 2) / np.nansum(flux)
            scatter *= num_good / (num_good - 1)
            scatter = np.sqrt(scatter)

            bad_idx = np.where(offsets > sigma * scatter)[0]
            ra[bad_idx] = np.nan
            dec[bad_idx] = np.nan
            flux[bad_idx] = np.nan

            num_good = np.sum(np.isfinite(flux))
            if num_good < 3:
                break

            ra_mean = np.nansum(ra * flux) / np.nansum(flux)
            dec_mean = np.nansum(dec * flux) / np.nansum(flux)
            center = SkyCoord(ra_mean, dec_mean, unit='deg')

        return ra_mean, dec_mean

    @classmethod
    def associate_measurements( cls, measurements, radius=None, year=None, month=None, day=None, no_new=False,
                                no_associate_legacy_survey=False, liumatch_radius=None, liumatch_server=None,
                                no_associate_gaia=False, gaiamatch_radius=None, gaiacat=None,
                                is_testing=False, connection=None, nocommit=False ):
        """Associate an object with each member of a list of measurements.

        Will create new objects (saving them to the database), and will
        create ObjectLegacySurveyMatch and ObjectGaiaMatch rows in the
        database if no_associate_* is True, unless no_new is True, in
        which case nothing is added to the database.

        Does not update any of the measurements in the database.
        Indeed, the measurements probably can't already be in the
        database when this function is called, because the object_id
        field is not nullable; this function would have to have been
        called before the measurements were saved in the first place.
        It is the responsibility of the calling function to actually
        save the all the measurements in the measurements list to the
        database (if it wants them saved).

        Parameters
        ----------
          measurements : list of Measurements
            The measurmentses with which to associate objects.

          radius : float
            The search radius in arseconds.  If an existing object is
            within this distance on the sky of a Measurements' ra/dec,
            then that Measurements will be associated with that object.
            If None, will be set to measuring.association_radius from
            the config.

          year, month, day : int, default None
            The UTC date of the time of exposure of the image from which
            the measurements come.  May be needed to generate object
            names based on the format configured in object.namefmt in
            the config yaml.  May always be omitted if no_new is True.

          no_new : bool, default False
            Normally, if an existing object is not wthin radius of one
            of the Measurements objects in the list given in the
            measurements parameter, then a new object will be created at
            the ra and dec of that Measurements and saved to the
            database.  Set no_new to True to not create any new objects,
            but to leave the object_id field of unassociated
            Measurements objects as is (probably None).

          no_associate_legacy_survey : bool, default False
            Normally, when a new object is created, call
            ObjectLegacySurveyMatch.create_new_object_matches on the
            object. Set this to False to skip that step.

          liumatch_radius : float, default None
            Radius in arcseconds to match to Legacy Survey objects.  If
            no_associate_legacy_survey is False, then something must be
            passed here.

          liumatch_server : str, default None
            Location of the liumatch server for Legacy Survey object
            mathcing.  If no_associate_legacy_survey is False, then
            something must be passed here.

          no_associate_gaia : bool, default False
            Normally, when a new object is created, call
            ObjectGaiaMatch.create_new_object_matches on the object.
            Set this to False to skip that step.

          gaiamatch_radius : bool, default False
            Radius in arcseconds to match to Gaia DR3 objects.  If
            no_associate_gaia is False, then something must be passed
            here.

          gaiacat : CatalogExerpt or None
            Catalog of gaia objects to pass on to the association with
            Gaia objects in the case of new objects.  If None, will
            search the external gaia server in a small patch around the
            object position.

          is_testing : bool, default False
            Never use this.  If True, the only associate measurements
            with objects that have the is_test property set to True, and
            set that property for any newly created objects.  (This
            parameter is used in some of our tests, but should not be
            used outside of that context.)

          connection : psycopg.Connection
            Database connection.  Will create and close one if this is
            None.  Warning : will commit and/or rollback.

          nocommit : bool, default False
            Do not commit to the database at the end of doing all the
            things.  You really want to leave this at False; it's here
            for test_generate_names_race_condition in
            tests/models/test_objects.py

        """

        if radius is None:
            radius = Config.get().value( "measuring.association_radius" )
        else:
            radius = float( radius )

        if not no_associate_legacy_survey:
            if ( liumatch_radius is None ) or ( liumatch_server is None ):
                raise ValueError( "Must supply liumatch_radius and liumatch_server" )

        if not no_associate_gaia:
            if gaiamatch_radius is None:
                raise ValueError( "Must supply gaiamatch_radius" )

        with PsycopgConnection( connection ) as conn:
            neednew = []
            cursor = conn.cursor()
            try:
                # We have to lock the object table for this entire process.
                #   Otherwise, there is a race condition where two processes
                #   with a source at the same RA/Dec (within uncertainty)
                #   generate object names at the same time, and we end up
                #   with two different objects that should only have been
                #   one.
                # This does mean we have to make sure *not* to commit the
                #   database inside any functions called from this function.
                #   (In practice, that is Object.generate_names.)
                if not no_new:
                    if not nocommit:
                        # Rollback the connection before getting the lock to free up any
                        #   shared locks the connection may incidentally have, to reduce
                        #   the chances of two processes deadlock when we try to lock
                        #   the object table.
                        conn.rollback()
                    cursor.execute( "LOCK TABLE objects" )
                for m in measurements:
                    cursor.execute( ( "SELECT _id  FROM objects WHERE "
                                      "  q3c_radial_query( ra, dec, %(ra)s, %(dec)s, %(radius)s ) "
                                      "  AND is_test=%(test)s" ),
                                    { 'ra': m.ra, 'dec': m.dec, 'radius': radius/3600., 'test': is_testing } )
                    rows = cursor.fetchall()
                    if len(rows) > 0:
                        m.object_id = rows[0][0]
                    else:
                        neednew.append( m )

                if ( not no_new ) and ( len(neednew) > 0 ):
                    names = cls.generate_names( number=len(neednew), year=year, month=month, day=day,
                                                ra=m.ra, dec=m.dec, verifyunique=True, connection=conn )
                    for name, m in zip( names, neednew ):
                        objid = uuid.uuid4()
                        cursor.execute( ( "INSERT INTO objects(_id,ra,dec,name,is_test,is_bad) "
                                          "VALUES(%(id)s, %(ra)s, %(dec)s, %(name)s, %(testing)s, FALSE)" ),
                                        { 'id': objid, 'name': name, 'ra': m.ra, 'dec': m.dec, 'testing': is_testing } )
                        m.object_id = objid
                        if not nocommit:
                            # Make sure the object is committed even if one
                            #  of the get_object_matches does a database
                            #  rollback.
                            conn.commit()

                            # Don't care about the return value for the matches,
                            #   just making sure the database is loaded for
                            #   later alert purposes
                            if not no_associate_legacy_survey:
                                SCLogger.debug( "Starting legacy survey matching..." )
                                ObjectLegacySurveyMatch.get_object_matches( objid, con=conn,
                                                                            radius=liumatch_radius,
                                                                            liuserver=liumatch_server )
                                SCLogger.debug( "...done with legacy survey matching." )

                            if not no_associate_gaia:
                                SCLogger.debug( "Starting gaia matching..." )
                                ObjectGaiaMatch.get_object_matches( objid, radius=gaiamatch_radius,
                                                                    gaiacat=gaiacat, con=conn )
                                SCLogger.debug( "...done with gaia matching." )
            finally:
                # Make sure database locks are released
                if not nocommit:
                    conn.rollback()


    @classmethod
    def generate_names( cls, number=1, formatstr=None,  year=None, month=None, day=None, ra=None, dec=None,
                        verifyunique=False, seed=None, connection=None, nocommit=True ):
        """Generate one or more names for an object based on the time of discovery.

        Valid things in format specifier that will be replaced are:
          %y - 2-digit year.  If you use this, then you hate everybody.
          %Y - 4-digit year.
          %m - 2-digit month.
          %d - 2-digit day.
          %R - RA in degrees (using format "08.4f").
          %D - dec in degrees (using format "+08.4f").
          %a - set of lowercase letters, starting with a..z, then aa..az..zz, then aaa..aaz..zzz, etc.  (Sorta.)
          %A - set of uppercase letters, similar.
          %n - an integer that starts at 0 and increments with each object added.
          %l - a randomly generated lowercase letter.  Should probably be used more than once if used at all.
          %% - a literal %.

        It doesn't make sense to use more than one of (%a, %A, %n).

        All of (%a, %A, %n) look at the passed year.  They start over
        for each year, and look at the table object_name_max_used in the
        database to figure out which names have been used for the passed
        year. (That table is locked while in use to guarantee that
        multiple processes won't generate the same name at the same
        time.)

        Parameters
        ----------
          number : int, default 1
            Number of names to generate

          formatstr: str, default None.
            See above.  If None, defaults to Config.get().value(
            'object.namefmt' ), which is usually what you want.

          year : int
            The year in which the object was discovered.  Required if the
            format string includes %y, %Y, %a, %A, or %n.

          month: int
            The month in which the object was discovered.  Required if
            the format string includes %m.

          day: int
            The UTC day of the month on which the object was
            discovered. Required if the format string includes %d.

          verifyunique : bool, default False
            Verify that generated object names don't already exist in
            the database.  Defaults to False, because if you're using a
            %A- or %a-based name, the way the code works guarantees that
            already.  But, if you're doing things like ra and dec, you
            might want this just to be sure.

          seed : random seed for %l in the format string.  Never use
            this (unless you're writing a test and need reproducible
            results).

          connection : psycopg.Connection or None
            Database connection.  Only used if %A, %a, or %n is in the
            format string, or if verifyunique is True.  If %A, %a, or %n
            is in the format string, the connection will be comitted.
            If no connection is passed and one is needed, then a new
            connection will be created and then closed (perhaps twice).

          nocommit : boolean, default True
            Don't commit to the database even if changes are made.  For
            normal use case, you want to leave this as True.  The reason
            to leave this true is that if you commit, you will lose any
            table locks.  Normal use of this function is from
            Object.associate_measurements(), which acquires a table lock
            on the objects table; committing would lose that.  This does
            mean that you need to make sure to commit to the databse in
            the function that called this; otherwise, any modifications
            made to the object_name_max_used table will get lost.

        Returns
        -------
           list of str

        """

        if formatstr is None:
            formatstr = Config.get().value( 'object.namefmt' )

        if ( ( ( "%y" in formatstr ) and ( ( not isinstance( year, numbers.Integral ) )
                                           or ( year < 2000 ) or ( year > 2099 ) ) )
             or
             ( ( "%Y" in formatstr ) and ( ( not isinstance( year, numbers.Integral ) )
                                           or ( year <= 0 ) or ( year > 9999 ) ) )
             or
             ( ( "%m" in formatstr ) and ( ( not isinstance( month, numbers.Integral ) )
                                           or ( month <= 0 ) or ( month > 12 ) ) )
             or
             ( ( "%d" in formatstr ) and ( ( not isinstance( day, numbers.Integral ) )
                                           or ( day <= 0 ) or ( day > 31 ) ) )
            ):
            # We're not going to bother figuring out if the day makes sense for the particular
            #  month.  Let the user claim something happened on February 30 if they want.
            raise ValueError( f"Invalid year/month/day {year}/{month}/{day} given format string {formatstr}" )

        if ( ( ( "%R" in formatstr ) and ( ( not isinstance( ra, numbers.Real ) )
                                           or ( ra < 0. ) or ( ra >= 360. ) ) )
             or
             ( ( "%D" in formatstr ) and ( ( not isinstance( dec, numbers.Real ) )
                                           or ( dec < -90. ) or ( dec > 90. ) ) )
            ):
            raise ValueError( f"Invalid ra/dec {ra}/{dec} given format string {formatstr}" )

        # Figure out incrementing numbers (including letter sequences)
        # by locking the database table and claiming enough new numbers
        firstnum = None
        if ( ( "%a" in formatstr ) or ( "%A" in formatstr ) or ( "%n" in formatstr ) ):
            if not isinstance( year, numbers.Integral ):
                raise ValueError( "Use of %a, %A, or %n requires integer year" )
            with PsycopgConnection( connection ) as conn:
                cursor = conn.cursor()
                cursor.execute( "LOCK TABLE object_name_max_used" )
                cursor.execute( "SELECT year, maxnum FROM object_name_max_used WHERE year=%(year)s",
                                { 'year': year } )
                rows = cursor.fetchall()
                if len(rows) == 0:
                    firstnum = 0
                    cursor.execute( "INSERT INTO object_name_max_used(year, maxnum) VALUES (%(year)s,%(num)s)",
                                    { 'year': year, 'num': number-1 } )
                else:
                    # len(rows) will never be >1 because year is the primary key
                    firstnum = rows[0][1] + 1
                    cursor.execute( "UPDATE object_name_max_used SET maxnum=%(num)s WHERE year=%(year)s",
                                    { 'year': year, 'num': firstnum + number - 1 } )
                if not nocommit:
                    conn.commit()

        # Make a rng if we need it for random letters
        rng = None
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        if  "%l" in formatstr:
            rng = np.random.default_rng( seed )

        names = []

        for i in range( number ):
            if firstnum is not None:
                num = firstnum + i
                # Convert the number to a sequence of letters.  This is not
                # exactly base 26, mapping 0=a to 25=z in each place,
                # beacuse leading a's are *not* leading zeros.  aa is not
                # 00, which is what a straight base26 number using symbols a
                # through z would give.  aa is the first thing after z, so
                # aa is 26.
                # The first 26 work:
                #     a = 0*26⁰
                #     z = 25*26⁰
                # but then:
                #    aa = 1*26¹ + 0*26⁰
                # not 0*26¹ + 0*26⁰.  It gets worse:
                #    za = 26*26¹ + 0*26⁰ = 1*26² + 0*26¹ + 0*26⁰
                # and
                #    zz = 26*26¹ + 25*26⁰ = 1*26² + 0*26¹ + 25*26⁰
                # The sadness only continues:
                #   aaa = 1*26² + 1*26¹ + 0*26⁰
                #   azz = 1*26² + 26*26² + 25*26⁰ = 2*26² + 0*26¹ + 25*26⁰
                #   baa = 2*26² + 1*26¹ + 0*26⁰
                # ... so it's not really a base 26 number.
                #
                # To deal with this, we're not going to use all the
                # available namespace.  Who cares, right?  If somebody
                # cares, they can deal with it.  We're just never going to
                # have a leading a.  So, afer z comes ba.  There is no aa
                # through az.  Except for the very first a, there will never
                # be a leading a.

                letters = ""
                letnum = num
                while letnum > 0:
                    dig26it = letnum % 26
                    thislet = alphabet[ dig26it ]
                    letters = thislet + letters
                    letnum //= 26
                letters = letters if len(letters) > 0 else 'a'

            name = formatstr
            repl = "_percent_"
            while repl in name:
                # Really make sure we're not using a placeholder string that's already there
                repl = f"_{repl}_"
            name = name.replace( "%%", repl )
            if "%y" in formatstr:
                name = name.replace( "%y", f"{year%100:02d}" )
            if "%Y" in formatstr:
                name = name.replace( "%Y", f"{year:04d}" )
            if "%m" in formatstr:
                name = name.replace( "%m", f"{month:02d}" )
            if "%d" in formatstr:
                name = name.replace( "%d", f"{day:02d}" )
            if "%R" in formatstr:
                name = name.replace( "%R", f"{ra:08.4f}" )
            if "%D" in formatstr:
                name = name.replace( "%D", f"{dec:+08.4f}" )
            if "%n" in formatstr:
                name = name.replace( "%n", f"{num}" )
            if "%a" in formatstr:
                name = name.replace( "%a", letters )
            if "%A" in formatstr:
                name = name.replace( "%A", letters.upper() )
            while "%l" in name:
                name = name.replace( "%l", alphabet[ rng.integers(26) ], 1 )
            name = name.replace( repl, "%" )

            names.append( name )

        if verifyunique:
            # First stupid thing, make sure all the generated names are unique
            if  len( set(names) ) != len( names ):
                raise ValueError( f"Newly generated names contain duplicates: {names}" )

            # Make sure none of the names are already in the database.
            with PsycopgConnection( connection ) as conn:
                cursor = conn.cursor()
                cursor.execute( "SELECT name FROM objects WHERE name=ANY(%(names)s)", { 'names': names } )
                rows = cursor.fetchall()
                if len(rows) != 0:
                    raise ValueError( f"{len(rows)} of {len(names)} newly generated names already exist in the "
                                      f"database: {[r[0] for r in rows]}" )

        return names


class ObjectPosition( Base, UUIDMixin, SpatiallyIndexed ):
    """ObjectPosition stores a mean position of an object.

    Because our objects are (mostly) supposed to be immutable once they
    are created in the database, we aren't supposed to update the ra/dec
    field of Object after it's first created.  However, as we get more
    observations of a single object, we (in principle) have better
    measurements of the position of that object.  Jibing that with the
    provenance model is a bit challenging, though.  The provenance of an
    object position depends on exactly which measurements went into it,
    and because that comes from many different images, it's not as
    simple as most of our provenance.

    ObjectPositions are calculated by pipeline/positioner.py

    To make this as reproducible as possible, object position provenance
    will include a date_calculated parameter.  The idea is that when
    calculating an object provenance, only images from before that date
    will be included in the calculation.  This still isn't completely
    reproducible, as it's entirely possible that images from earlier
    dates will be added to the database *after* the object position is
    calculated.  However, with some care from the people running the
    pipeline, this can approximate that.  Certainly for things like Data
    Releases, this is possible, as long as things are done in the same
    order.

    (Note that Objects themselves don't have provenance, but are global
    things.  However, ObjectPosition does have a provenance.  At some
    level, you can think of Object as putting a flag down saying "this
    is the name of an object".  The ra/dec in the objects table is a
    first approximation of the object's position (and, at least as of
    this writing, is what's used for associating measurements).
    ObjectPosition represents an actual measurement.)

    (One might argue that ObjectPosition should be updated based on
    position-variable forced photometry (if that's not an oxymoron), but
    in practice basing it on DIA discoveries will give you pretty much
    the same answer, and it's not worth getting fiddly about the
    difference.)

    """

    __tablename__ = "object_positions"

    @declared_attr
    def __table_args__(cls):  # noqa: N805
        return (
            sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
            UniqueConstraint( 'object_id', 'provenance_id', name='object_position_obj_prov_unique' ),
        )

    object_id = sa.Column(
        sa.ForeignKey( 'objects._id', ondelete='CASCADE', name='object_position_object_id_fkey' ),
        nullable=False,
        index=True,
        doc='ID of the object that this is the position for.'
    )

    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances._id', ondelete='CASCADE', name='object_position_provenance_id_fkey' ),
        nullable=False,
        index=True,
        doc=( "ID of the provenance of this object position." )
    )

    dra = sa.Column( sa.REAL, nullable=False, doc="Uncertainty on RA" )
    ddec = sa.Column( sa.REAL, nullable=False, doc="Uncertainty on Dec" )
    ra_dec_cov = sa.Column( sa.REAL, nullable=True, doc="Covariance on RA/Dec if available" )


class ObjectCatalogMatch:
    """A mixin for tables that match to an external catalog and store an arcsec radius object_match_config.

       Tables of subclasses must have at least four columns:
        object_id : UUID
        {cls.matchidcolumn} : as appropriate for subclass
        match_radius : int ; the search radius in milliarcseconds
        and dist : float ; the distance from the object postiion (in Object, not ObjectPosition) to the thing

       Have indexes on object_id and radius.  Make (object_id,
       {cls.matchidcolumn}, match_radius) unique.

       Subclasses must define the following class properties:
          nomatchtable : the name of the table that flags there are no matches
          matchidcolumn : the name of the column that has the id of the matched catalog row
          tablekeys : list of column names.  Please do not encode SQL injection attacks

       Subclasses must implement _generate_new_matches, and _compare_matches

    """

    @classmethod
    def _get_existing_matches( cls, objid, radius, cursor ):
        mrad = int( radius * 1000. )
        cursor.execute( f"SELECT {','.join(cls.tablekeys)} FROM {cls.__tablename__} "
                        f"WHERE object_id=%(objid)s AND match_radius=%(rad)s "
                        f"ORDER BY dist",
                        { 'objid': objid, 'rad': mrad } )
        rows = cursor.fetchall()

        found_existing = False
        matches = []
        if len(rows) > 0:
            found_existing = True
            matches = [ cls.create( **r ) for r in rows ]
            matches.sort( key=lambda o: o.dist )
        else:
            cursor.execute( f"SELECT * FROM {cls.nomatchtable} WHERE object_id=%(objid)s AND match_radius=%(rad)s ",
                            { 'objid': objid, 'rad': mrad } )
            rows = cursor.fetchall()
            if len(rows) > 0:
                found_existing = True

        return found_existing, matches


    @classmethod
    def get_object_matches( cls, objid, radius=None,
                            ignore_existing=False, verify_existing=False, commit=True,
                            con=None, **kwargs ):
        """Pull object matches to some sort of external catalog (Legacy Survey, Gaia), maybe cached in the database.

        You will never call this from ObjectCatalogMatch directly, but
        rather from a subclass.

        Parameters
        ----------
          objid : uuid
            Object ID

          radius : float
            Radius in arseconds to match around.  You almost always want
            to leave this at None so the configured value will be used.
            If this is not None, you probably want to set
            ignore_existing to False, and commit will be made False
            regardless of what you pass.

          ignore_existing : bool, default False
            Don't look in the database for existing matches to the
            object, but go out to the external server and find the matches
            afresh.  Requires a non-None radius.

          verify_existing : bool, default False
            If True, always go to the external server to get matches and
            verify that any existing matches are the same.  Doesn't make
            sense to set this True if ignore_existing is True.

          commit : bool, default True
            Set this to False to not commit found matches to the
            database.  If radius is non-None, this will be forced to
            True regardless of what you pass.

          con : psycopg.Connection, default None
            Database connection.  If not given, makes and closes a new
            one.  WARNING: even if commit=False, this connection will be
            rolled back!

          **kwargs : passed on to the subclass' _generate_new_matches method

        Returns
        -------
          list of objects of type cls (which will be a subclass of ObjectCatalogMatch)

        """

        if not isinstance( radius, numbers.Real ):
            raise TypeError( f"Radius {radius} isn't a float." )

        with PsycopgConnection( con ) as dbcon:
            try:
                cursor = dbcon.cursor( row_factory=psycopg.rows.dict_row )
                matches = None

                # Get existing matches
                found_existing, matches = cls._get_existing_matches( objid, radius, cursor )
                if found_existing and not verify_existing:
                    return matches

                # Rollback the database so there won't be an idle in
                #   transaction connection sitting around.  This is
                #   necessary because in a bit we might be about to lock
                #   the table we just read, and if one process has an
                #   idle in transaction sitting around that read that
                #   table, it'll stop the other process from locking the
                #   table.
                dbcon.rollback()

                # For one reason or another, we have to find new matches
                cursor.execute( "SELECT ra,dec FROM objects WHERE _id=%(id)s", { 'id': objid } )
                row = cursor.fetchone()
                ra = row['ra']
                dec = row['dec']
                newmatches = cls._generate_new_matches( objid, ra, dec, radius, **kwargs )

                if found_existing:
                    if not cls._compare_matches( matches, newmatches ):
                        raise RuntimeError( f"Catalog matches for {cls.__name__} don't match what is "
                                            f"already saved in the database!" )

                elif commit:
                    # There were no existing matches, so we should save
                    #   the new ones.  But, it's possible another
                    #   process found and saved matches since we last
                    #   searched for existing matches, so we should
                    #   check again for existing matches.  This time
                    #   we're going to lock the table, to avoid the race
                    #   condition of two processes inserting at once.
                    #   We didn't lock the table before the first search
                    #   for existing matches because we didn't want to
                    #   hold a table lock through the whole
                    #   _generate_new_matches process, which could be
                    #   long.  (An external server could be slow, for
                    #   example.)
                    cursor.execute( f"LOCK TABLE {cls.__tablename__}" )
                    found_existing, matches = cls._get_existing_matches( objid, radius, cursor )
                    if found_existing:
                        if not cls._compare_matches( matches, newmatches ):
                            raise RuntimeError( f"Another process acquired and saved catalog matches for "
                                                f"{cls.__name__} as us, but got a different list!" )
                    else:
                        if len(newmatches) == 0:
                            # Create the entry that flags that there were no matches
                            cursor.execute( f"INSERT INTO {cls.nomatchtable}(object_id, match_radius) "
                                            f"VALUES(%(objid)s, %(mrad)s)",
                                            { 'objid': objid, 'mrad': int( radius * 1000 ) } )
                        else:
                            keys = ",".join( cls.tablekeys )
                            valuesmess = ",".join( f"%({k})s" for k in cls.tablekeys )
                            for mat in newmatches:
                                values = { k: getattr( mat, k ) for k in cls.tablekeys }
                                cursor.execute( f"INSERT INTO {cls.__tablename__}({keys}) VALUES({valuesmess})",
                                                values )
                        dbcon.commit()

                return newmatches

            finally:
                # Make sure any held table locks are released
                dbcon.rollback()


class NoLegacySurveyMatches( Base ):
    """Flag that a given object_id / match radius has no gaia matches"""

    __tablename__ = "no_object_legacy_survey_matches"

    object_id = sa.Column(
        sa.ForeignKey( 'objects._id', ondelete='CASCADE', name='no_object_legacy_survey_match_fkey' ),
        nullable=False,
        index=True,
        primary_key=True,
        doc='ID of the object that has no matches'
    )

    match_radius = sa.Column(
        sa.Integer,
        nullable=False,
        index=False,
        primary_key=True,
        doc="Search radius in milliarcsec" )


class ObjectLegacySurveyMatch( Base, ObjectCatalogMatch ):
    """Stores matches bewteen objects and Legacy Survey catalog sources.

    Liu et al., 2025, https://ui.adsabs.harvard.edu/abs/2025arXiv250517174L/abstract
    (submitted to PASP)

    Catalog and "xgboost" score is described in that paper.

    """

    __tablename__ = "object_legacy_survey_match"
    nomatchtable = "no_object_legacy_survey_matches"
    matchidcolumn = 'lsid'
    tablekeys = [ 'object_id', 'match_radius', 'lsid', 'ra', 'dec', 'dist', 'white_mag', 'xgboost', 'is_star' ]

    object_id = sa.Column(
        sa.ForeignKey('objects._id', ondelete='CASCADE', name='object_ls_match_object_id_fkey'),
        nullable=False,
        index=True,
        primary_key=True,
        doc="ID of the object this is a match for"
    )

    match_radius = sa.Column(
        sa.Integer,
        nullable=False,
        index=False,
        primary_key=True,
        doc="Search radius in milliarcsec" )

    lsid = sa.Column( sa.BigInteger, nullable=False, primary_key=True, index=False, doc="Legacy Survey ID" )

    ra = sa.Column( sa.Double, nullable=True, index=False, doc="Legacy Survey object RA" )
    dec = sa.Column( sa.Double, nullable=True, index=False, doc="Legacy Survey object Dec" )
    dist = sa.Column( sa.Double, nullable=True, index=False, doc="Distance from obj to LS obj in arcsec" )

    white_mag = sa.Column( sa.Double, nullable=True, index=False, doc="Legacy Survey object white magnitude" )

    xgboost = sa.Column( sa.REAL, nullable=True, index=False, doc="Legacy Survey object xgboost statistic" )

    is_star = sa.Column( sa.Boolean, nullable=True, index=False, doc="True if xgboost≥0.5, else False" )

    @declared_attr
    def __table_args__(cls):  # noqa: N805
        return ( sa.Index( "ix_object_legacy_survey_match_objidmatchr", 'object_id', 'match_radius' ), )


    @classmethod
    def _generate_new_matches( cls, objid, ra, dec, radius, liuserver=None, **kwargs ):
        """Create new object match entries.

        Searches the liuserver for nearby objects, creates database
        entries.  May or may not commit them.  (If you pass a
        PsycopgConnection in con and don't set commit=True, then
        the added entries will *not* be committed to the database.)

        Parameters
        ----------
          objid : uuid
            ID of the object we're matching to

          ra, dec : double
            Coordinates of the object.  Required.  Nominally, this is
            redundant, because we can get it from the database using
            objid, but it's here for efficiency.  (Also, so we can run
            this routine in case the object isn't yet saved to the
            database.)

          radius : float
            radius in arcseconds to find objects

          liuserver: str
             Location of the liu server.  Required.

          **kwargs : passed on to requests.post (posting to liuserver)

        Returns
        -------
          list of ObjectLegacySurveyMatch, sorted by dist

        """

        if liuserver is None:
            raise ValueError( "Must supply liuserver." )

        url = f"{liuserver}/getsources/{ra}/{dec}/{radius}"

        def _errmsg( res ):
            SCLogger.error( f"Failed to contact {url}; got HTTP status {res.status_code}: \""
                            f"{res.text if len(res.text) < 240 else res.text[:240]}\"." )

        do_the_thing = functools.partial( requests.post, url, **kwargs )
        # ****
        # import random
        # import remote_pdb; remote_pdb.RemotePdb('127.0.0.1', random.randint( 4000, 30000 ) ).set_trace()
        # ****
        res = retry_with_sleep( do_the_thing, sleepmin=0.2, sleept=1.0, sleepmax=32.0, sleepfac=2.,
                                sleepfuzz=0.1, failmessage=f"contacting {liuserver}",
                                good_returns=[200], return_attr='status_code', badreturn_handler=_errmsg )
        res = res.json()

        expected_keys = [ 'lsid', 'ra', 'dec', 'dist', 'white_mag', 'xgboost', 'is_star' ]
        if ( ( not isinstance( res, dict ) ) or
             ( any( k not in res.keys() for k in expected_keys ) )
            ):
            raise ValueError( f"Unexpected response from liuserver; expected a dict with keys {expected_keys}, but "
                              f"got a {type(res)}{f' with keys {res.keys()}' if isinstance(res,dict) else ''}." )

        olsms = []
        for i in range( len( res['lsid'] ) ):
            olsms.append( ObjectLegacySurveyMatch( object_id=objid,
                                                   match_radius=int( radius * 1000 ),
                                                   lsid=res['lsid'][i],
                                                   ra=res['ra'][i],
                                                   dec=res['dec'][i],
                                                   dist=res['dist'][i],
                                                   white_mag=res['white_mag'][i],
                                                   xgboost=res['xgboost'][i],
                                                   is_star=res['is_star'][i] ) )
        olsms.sort( key=lambda o: o.dist )

        return olsms

    @classmethod
    def _compare_matches( cls, Alist, Blist ):
        if len( Alist ) != len( Blist ):
            SCLogger.error( "ObjectLegacySurveyMatch inconsistency: different list lengths." )
            return False

        ok = True
        for a, b in zip( Alist, Blist ):
            cosdec = np.cos( a.dec * np.pi / 180. )
            if any( [ a.object_id != b.object_id,
                      a.lsid != b.lsid,
                      not np.isclose( a.ra, b.ra, atol=2.8e-5/cosdec ),
                      not np.isclose( a.dec, b.dec, atol=2.8e-5 ),
                      not np.isclose( a.dist, b.dist, atol=0.1 ),
                      not np.isclose( a.white_mag, b.white_mag, atol=0.01 ),
                      not np.isclose( a.xgboost, b.xgboost, atol=0.001 ),
                      a.is_star != b.is_star ] ):
                ok = False
                break

        if not ok:
            strio = io.StringIO()
            strio.write( "ObjectLegacySurveyMatch inconsistency:\n" )
            strio.write( f"  {'Old LSID':20s} {'New LSID':20s}  {'Old RA':9s} {'New RA':9s}  "
                         f"{'Old Dec':9s} {'New Dec':9s}  {'Old d':6s} {'New d':6s}  "
                         f"{'Old m':5s} {'New m':5s}  {'Old xg':6s} {'New xg':6s}  "
                         f"{'Old is':6s} {'New is':5s}\n" )
            strio.write( "  ==================== ====================  ========= =========  "
                         "========= =========  ====== ======  ===== =====  ====== ======  ====== ======\n" )
            for oldolsm, newolsm in zip( Alist, Blist ):
                strio.write( f"  {oldolsm.lsid:20d} {newolsm.lsid:20d}  "
                             f"{oldolsm.ra:9.5f} {newolsm.ra:9.5f}  "
                             f"{oldolsm.dec:9.5f} {newolsm.dec:9.5f}  "
                             f"{oldolsm.dist:6.2f} {newolsm.dist:6.2f}  "
                             f"{oldolsm.white_mag:5.2f} {newolsm.white_mag:5.2f}  "
                             f"{oldolsm.xgboost:6.3f} {newolsm.xgboost:6.3f}  "
                             f"{str(oldolsm.is_star):5s} {str(newolsm.is_star):5s}\n" )
            SCLogger.error( strio.getvalue() )

        return ok


class NoGaiaMatches( Base ):
    """Flag that a given object_id / match radius has no gaia matches"""

    __tablename__ = "no_object_gaia_matches"

    object_id = sa.Column(
        sa.ForeignKey( 'objects._id', ondelete='CASCADE', name='no_object_gaia_match_fkey' ),
        nullable=False,
        index=True,
        primary_key=True,
        doc='ID of the object that has no matches'
    )

    match_radius = sa.Column(
        sa.Integer,
        nullable=False,
        index=False,
        primary_key=True,
        doc="Search radius in milliarcsec" )


class ObjectGaiaMatch( Base, ObjectCatalogMatch ):
    """Stores matches between objects and Gaia DR3 catalog sources."""

    __tablename__ = "object_gaia_match"
    nomatchtable = "no_object_gaia_matches"
    matchidcolumn = "gaia_sourceid"
    tablekeys = [ 'object_id', 'match_radius', 'gaia_sourceid', 'dist',
                  'gaia_starprob', 'gaia_quasarprob', 'gaia_galaxyprob' ]

    object_id = sa.Column(
        sa.ForeignKey( 'objects._id', ondelete='CASCADE', name='object_gaia_match_object_id_fkey' ),
        nullable=False,
        index=True,
        primary_key=True,
        doc='ID of the object this is a match for'
    )

    match_radius = sa.Column(
        sa.Integer,
        nullable=False,
        index=False,
        primary_key=True,
        doc="Search radius in milliarcsec" )

    gaia_sourceid = sa.Column( sa.BigInteger, nullable=False, primary_key=True, index=False, doc="Gaia DR3 source_id" )
    dist = sa.Column( sa.Double, nullable=True, index=False, doc="Arcsec between object and Gaia DR3 source" )
    gaia_starprob = sa.Column( sa.REAL, nullable=True, index=False, doc="Gaia DR3 classprob_dsc_combmod_star" )
    gaia_quasarprob = sa.Column( sa.REAL, nullable=True, index=False, doc="Gaia DR3 classprob_dsc_combmod_quasar" )
    gaia_galaxyprob = sa.Column( sa.REAL, nullable=True, index=False, doc="Gaia DR3 classprob_dsc_combmod_galaxy" )

    @declared_attr
    def __table_args__(cls):     #noqa: N805
        return ( sa.Index( "ix_gaia_match_objidmatchr", 'object_id', 'match_radius' ), )


    @classmethod
    def _generate_new_matches( cls, objid, ra, dec, radius, gaiacat=None ):
        """Create new Gaia DR3 matches.

        Searches Gaia DR3 for enarby objects, creates database entries.
        May or may not commit them.  (If you pass a Psycopg2Connection
        and don't set commit=True, then the added entries will *not* be
        committed to the database.)

        Will probably fail very near the poles around that coordinate
        singularity.

        Returns
        -------
          list of ObjectGaiaMatch

        Parametrs
        ---------
          objid : Uuid
            ID of the object we're matching to

          ra, dec : float
            Coordinates of the object.  Required.  Nominally, this is
            redundant, because we can get it from the database using
            objid, but it's here for efficiency.  (Also, so we can run
            this routine in case the object isn't yet saved to the
            database.)

          radius : float, default config item gaiamatch.radius
            Radius in arseconds that objects will be found in.  You
            almost always want to leave this as None so it uses the
            configured default!

          gaiacat : CatalogExcerpt or None
            If gaiacat is none, then this class method will query an
            online gaia database to get dia objects.  If this is not
            None, then it will filter this catalog to find dia objects.

            This is here for efficiency.  If you're going to match a
            bunch of objects on one image to Gaia, just get a gaia
            catalog for the whole image once and pass that here.
            Otherwise, it will make a connection to an external server
            for each object.

            WARNING : catalog_tools.fetch_gaia_dr3_excerpt was written
            with the point of view of finding stars for astrometric
            calibration.  As such, for passing to this function here,
            you may not want to use the defaults from that function, as
            you may want to have (say) no magnitude limit.

        """

        if gaiacat is None:
            cosdec = np.cos( dec * np.pi / 180. )
            minra = ra - radius / 3600. / cosdec
            if minra < 0.:
                minra += 360.
            maxra = ra + radius / 3600. / cosdec
            if maxra > 360.:
                maxra -= 360.
            mindec = dec - radius / 3600.
            if mindec < -90.:
                mindec = -90.
            maxdec = dec + radius / 3600.
            if maxdec > 90.:
                maxdec = 90.

            gaiacat, fpath, _ = download_gaia_dr3( minra, maxra, mindec, maxdec, padding=0., minmag=None, maxmag=None )
            gaiacat.filepath = fpath

        sourceids = gaiacat.data[ 'GAIA_DR3_ID' ]
        starprobs = gaiacat.data[ 'STARPROB' ]
        quasarprobs = gaiacat.data[ 'QUASARPROB' ]
        galaxyprobs = gaiacat.data[ 'GALAXYPROB' ]

        coords = SkyCoord( gaiacat.object_ras, gaiacat.object_decs, unit='deg' )
        ctrcoord = SkyCoord( ra, dec, unit='deg' )
        sep = coords.separation( ctrcoord ).to( astropy.units.arcsec )
        wclose = np.where( sep.value < radius )[0]
        if len(wclose) == 0:
            # Nothing close
            return []

        sourceids = sourceids[ wclose ]
        starprobs = starprobs[ wclose ]
        quasarprobs = quasarprobs[ wclose ]
        galaxyprobs = galaxyprobs[ wclose ]
        sep = sep.value[ wclose ]

        matches = [ ObjectGaiaMatch( object_id=objid,
                                     match_radius=int( radius * 1000 ),
                                     gaia_sourceid=sourceids[i],
                                     dist=sep[i],
                                     gaia_starprob=starprobs[i],
                                     gaia_quasarprob=quasarprobs[i],
                                     gaia_galaxyprob=galaxyprobs[i]
                                    )
                    for i in range(len(sourceids)) ]

        matches.sort( key=lambda o: o.dist )
        return matches


    @classmethod
    def _compare_matches( cls, Alist, Blist ):
        if len( Alist ) != len( Blist ):
            SCLogger.error( "ObjectGaiaMatch inconsistency: different list lengths." )
            return False

        ok = True
        for a, b in zip( Alist, Blist ):
            if any ( [ a.object_id != b.object_id,
                       a.gaia_sourceid != b.gaia_sourceid,
                       not np.isclose( a.gaia_starprob, b.gaia_starprob, atol=1e-4 ),
                       not np.isclose( a.gaia_quasarprob, b.gaia_quasarprob, atol=1e-4 ),
                       not np.isclose( a.gaia_galaxyprob, b.gaia_galaxyprob, atol=1e-4 )
                      ] ):
                ok = False
                break

        if not ok:
            strio = io.StringIO()
            strio.write( "ObjectGaiaMatch inconsistency:\n" )
            strio.write( f"  {'Old Objectid':20s} {'New Objectid':20s}  {'O.Gaia ID':10s} {'N.Gaia ID':10s}  "
                         f"{'O.Star':7s} {'N.Star':7s}  {'O.QSO':7s} {'N.QSO':7s}  {'O.Gal':7s} {'N.Gal':7s}\n" )
            strio.write( "  ==================== ====================  ========== ==========  "
                         "======= =======  ======= =======  ======= =======\n" )
            for a, b in zip( Alist, Blist ):
                strio.write( f"  {a.object_id:20s} {b.object_id:20s}  "
                             f"{a.gaia_sourceid:10s} {b.gaia_sourceid:10d}  "
                             f"{a.gaia_starprob:7.4f} {b.gaia_starprob:7.4f}  "
                             f"{a.gaia_quasarprob:7.4f} {b.gaia_quasarprob:7.4f}  "
                             f"{a.gaia_galaxyprob:7.4f} {b.gaia_galaxyprob:7.4f}\n" )

            SCLogger.error( strio.getvalue() )

        return ok
