import io
import uuid
import operator
import numpy as np
from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.schema import UniqueConstraint

from astropy.time import Time
from astropy.coordinates import SkyCoord

from models.base import Base, SeeChangeBase, SmartSession, Psycopg2Connection, UUIDMixin, SpatiallyIndexed
from models.image import Image
from models.cutouts import Cutouts
from models.source_list import SourceList
from models.zero_point import ZeroPoint
from models.measurements import Measurements, MeasurementSet
from models.deepscore import DeepScore, DeepScoreSet
from models.reference import image_subtraction_components
from util.config import Config
from util.retrypost import retry_post
from util.logger import SCLogger

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


    def get_measurements_et_al( self, measurement_prov_id, deepscore_prov_id, omit_measurements=[],
                                mjd_min=None, mjd_max=None, session=None ):
        """Return a list of sundry objects for this Object.

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

        Returns
        -------
          list of tuples of measurements of this object, sorted by image mjd

          Each tuple has either 3 or 4 entries : Measurements, DeepScore, Image, ZeroPoint
          If deepscore_prov_id is omitted, then it's just Measurements, Image, ZeroPoint

          The Image is the *subtraction image*, not the original science image, of this measurement.
          (The ZeroPoint applies to both.)

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

        with SmartSession( session ) as sess:
            if deepscore_prov_id is not None:
                q = sess.query( Measurements, DeepScore, Image, ZeroPoint )
            else:
                q = sess.query( Measurements, Image, ZeroPoint )

            q = ( q.join( MeasurementSet, Measurements.measurementset_id==Measurements._id )
                  .join( Cutouts, MeasurementSet.cutouts_id==Cutouts._id )
                  .join( SourceList, Cutouts.sources_id==SourceList._id )
                  .join( Image, SourceList.image_id==Image._id )
                  .join( image_subtraction_components, image_subtraction_components.c.image_id==Image._id )
                  .join( ZeroPoint, ZeroPoint._id==image_subtraction_components.c.new_zp_id ) )

            if deepscore_prov_id is not None:
                q = ( q.join( DeepScoreSet, sa.and_( DeepScoreSet.measurementset_id==MeasurementSet._id,
                                                     DeepScoreSet.provenance_id==deepscore_prov_id ),
                              isouter=True )
                      .join( DeepScore, sa.and_( DeepScore.deepscoreset_id==DeepScoreSet._id,
                                                 DeepScore.index_in_sources==Measurements.index_in_sources ),
                             isouter=True )
                     )

            q = q.filter( Measurements.object_id==self.id )
            q = q.filter( MeasurementSet.provenance_id==measurement_prov_id )
            if len( omit_measurements ) > 0:
                q = q.filter( Measurements._id.not_in( omit_measurements ) )
            if mjd_min is not None:
                q = q.filter( Image.mjd >= mjd_min )
            if mjd_max is not None:
                q = q.filter( Image.mjd <= mjd_max )
            q = q.order_by( Image.mjd )

            return q.all()


    def get_filtered_measurements_list(
            self,
            prov_hash_list=None,
            radius=2.0,
            thresholds=None,
            mjd_start=None,
            mjd_end=None,
            time_start=None,
            time_end=None,
    ):
        """Filter the measurements associated with this object.

        Parameters
        ----------
        prov_hash_list: list of strings, optional
            The prov_hash_list is used to choose only some measurements, if they have a matching provenance hash.
            This list is ordered such that the first hash is the most preferred, and the last is the least preferred.
            If not given, will default to the most recently added Measurements object's provenance.
        radius: float, optional
            Will use the radius parameter to narrow down to measurements within a certain distance of the object's
            coordinates (can only narrow down measurements that are already associated with the object).
            Default is to grab all pre-associated measurements.
        thresholds: dict, optional
            Provide a dictionary of thresholds to cut on the Measurements object's disqualifier_scores.
            Can provide keys that match the keys of the disqualifier_scores dictionary, in which case the cuts
            will be applied to any Measurements object that has the appropriate score.
            Can also provide a nested dictionary, where the key is the provenance hash, in which case the value
            is a dictionary with keys matching the disqualifier_scores dictionary of those specific Measurements
            that have that provenance (i.e., you can give different thresholds for different provenances).
            The specific provenance thresholds will override the general thresholds.
            Note that if any of the disqualifier scores is not given, then the threshold saved
            in the Measurements object's Provenance will be used (the original threshold).
            If a disqualifier score is given but no corresponding threshold is given, then the cut will not be applied.
            To override an existing threshold, provide a new value but set it to None.
        mjd_start: float, optional
            The minimum MJD to consider. If not given, will default to the earliest MJD.
        mjd_end: float, optional
            The maximum MJD to consider. If not given, will default to the latest MJD.
        time_start: datetime.datetime or ISO date string, optional
            The minimum time to consider. If not given, will default to the earliest time.
        time_end: datetime.datetime or ISO date string, optional
            The maximum time to consider. If not given, will default to the latest time.

        Returns
        -------
        list of Measurements
        """
        raise RuntimeError( "Issue #346" )
        # this includes all measurements that are close to the discovery measurement
        # measurements = session.scalars(
        #     sa.select(Measurements).where(Measurements.cone_search(self.ra, self.dec, radius))
        # ).all()

        if time_start is not None and mjd_start is not None:
            raise ValueError('Cannot provide both time_start and mjd_start. ')
        if time_end is not None and mjd_end is not None:
            raise ValueError('Cannot provide both time_end and mjd_end. ')

        if time_start is not None:
            mjd_start = Time(time_start).mjd

        if time_end is not None:
            mjd_end = Time(time_end).mjd


        # IN PROGRESS.... MORE THOUGHT REQUIRED
        # THIS WILL BE DONE IN A FUTURE PR  (Issue #346)

        with SmartSession() as session:
            q = session.query( Measurements, Image.mjd ).filter( Measurements.object_id==self._id )

            if ( mjd_start is not None ) or ( mjd_end is not None ):
                q = ( q.join( Cutouts, Measurements.cutouts_id==Cutouts._id )
                      .join( SourceList, Cutouts.sources_id==SourceList._id )
                      .join( Image, SourceList.image_id==Image.id ) )
                if mjd_start is not None:
                    q = q.filter( Image.mjd >= mjd_start )
                if mjd_end is not None:
                    q = q.filter( Image.mjd <= mjd_end )

            if radius is not None:
                q = q.filter( sa.func.q3c_radial_query( Measurements.ra, Measurements.dec,
                                                        self.ra, self.dec,
                                                        radius/3600. ) )

            if prov_hash_list is not None:
                q = q.filter( Measurements.provenance_id.in_( prov_hash_list ) )


        # Further filtering based on thresholds

        # if thresholds is not None:
        # ....stopped here, more thought required


        measurements = []
        if radius is not None:
            for m in self.measurements:  # include only Measurements objects inside the given radius
                delta_ra = np.cos(m.dec * np.pi / 180) * (m.ra - self.ra)
                delta_dec = m.dec - self.dec
                if np.sqrt(delta_ra**2 + delta_dec**2) < radius / 3600:
                    measurements.append(m)

        if thresholds is None:
            thresholds = {}

        if prov_hash_list is None:
            # sort by most recent first
            last_created = max(self.measurements, key=operator.attrgetter('created_at'))
            prov_hash_list = [last_created.provenance.id]

        passed_measurements = []
        for m in measurements:
            local_thresh = m.provenance.parameters.get('thresholds', {}).copy()  # don't change provenance parameters!
            if m.provenance.id in thresholds:
                new_thresh = thresholds[m.provenance.id]  # specific thresholds for this provenance
            else:
                new_thresh = thresholds  # global thresholds for all provenances

            local_thresh.update(new_thresh)  # override the Measurements object's thresholds with the new ones

            for key, value in local_thresh.items():
                if value is not None and m.disqualifier_scores.get(key, 0.0) >= value:
                    break
            else:
                passed_measurements.append(m)  # only append if all disqualifiers are below the threshold

        # group measurements into a dictionary by their MJD
        measurements_per_mjd = defaultdict(list)
        for m in passed_measurements:
            measurements_per_mjd[m.mjd].append(m)

        for mjd, m_list in measurements_per_mjd.items():
            # check if a measurement matches one of the provenance hashes
            for hash in prov_hash_list:
                best_m = [m for m in m_list if m.provenance.id == hash]
                if len(best_m) > 1:
                    raise ValueError('More than one measurement with the same provenance. ')
                if len(best_m) == 1:
                    measurements_per_mjd[mjd] = best_m[0]  # replace a list with a single Measurements object
                    break  # don't need to keep checking the other hashes
            else:
                # if none of the hashes match, don't have anything on that date
                measurements_per_mjd[mjd] = None

        # remove the missing dates
        output = [m for m in measurements_per_mjd.values() if m is not None]

        # remove measurements before mjd_start
        if mjd_start is not None:
            output = [m for m in output if m.mjd >= mjd_start]

        # remove measurements after mjd_end
        if mjd_end is not None:
            output = [m for m in output if m.mjd <= mjd_end]

        return output

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
    def associate_measurements( cls, measurements, radius=None, year=None, no_new=False,
                                no_associate_legacy_survey=False, is_testing=False ):
        """Associate an object with each member of a list of measurements.

        Will create new objects (saving them to the database) unless
        no_new is True.

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
            If None, will be set to measuring.association_radius in the
            config.

          year : int, default None
            The year of the time of exposure of the image from which the
            measurements come.  Needed to generate object names, so it
            may be omitted if no_new is True.

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
            objec.t Set this to False to skip that step.

          is_testing : bool, default False
            Never use this.  If True, the only associate measurements
            with objects that have the is_test property set to True, and
            set that property for any newly created objects.  (This
            parameter is used in some of our tests, but should not be
            used outside of that context.)

        """

        if not no_new:
            if year is None:
                raise ValueError( "Need to pass a year unless no_new is true" )
            else:
                year = int( year )

        if radius is None:
            radius = Config.get().value( "measurements.association_radius" )
        else:
            radius = float( radius )

        with Psycopg2Connection() as conn:
            neednew = []
            cursor = conn.cursor()
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
                names = cls.generate_names( number=len(neednew), year=year, connection=conn )
                # Rollback in order to remove the lock generate_names claimed on object_name_max_used
                conn.rollback()
                cursor = conn.cursor()
                for name, m in zip( names, neednew ):
                    objid = uuid.uuid4()
                    cursor.execute( ( "INSERT INTO objects(_id,ra,dec,name,is_test,is_bad) "
                                      "VALUES(%(id)s, %(ra)s, %(dec)s, %(name)s, %(testing)s, FALSE)" ),
                                    { 'id': objid, 'name': name, 'ra': m.ra, 'dec': m.dec, 'testing': is_testing } )
                    m.object_id = objid
                    if not no_associate_legacy_survey:
                        ObjectLegacySurveyMatch.create_new_object_matches( objid, m.ra, m.dec, con=conn )

                conn.commit()

    @classmethod
    def generate_names( cls, number=1, year=0, month=0, day=0, formatstr=None, connection=None ):
        """Generate one or more names for an object based on the time of discovery.

        Valid things in format specifier that will be replaced are:
          %y - 2-digit year
          %Y - 4-digit year
          %m - 2-digit month (not supported)
          %d - 2-digit day (not supported)
          %a - set of lowercase letters, starting with a..z, then aa..az..zz, then aaa..aaz..zzz, etc.
          %A - set of uppercase letters, similar
          %n - an integer that starts at 0 and increments with each object added
          %l - a randomly generated letter

        It doesn't make sense to use more than one of (%a, %A, %n).

        """

        if formatstr is None:
            formatstr = Config.get().value( 'object.namefmt' )

        if ( ( ( ( "%y" in formatstr ) or ( "%Y" in formatstr ) ) and ( year <= 0 ) )
             or
             ( ( "%m" in formatstr ) and ( year <= 0 ) )
             or
             ( ( "%d" in formatstr ) and ( day <= 0 ) ) ):
            raise ValueError( f"Invalid year/month/day {year}/{month}/{day} given format string {formatstr}" )

        if ( "%m" in formatstr ) or ( "%d" in formatstr ):
            raise NotImplementedError( "Month and day in format string not supported." )

        if ( "%l" in formatstr ):
            raise NotImplementedError( "%l isn't implemented" )

        firstnum = None
        if ( ( "%a" in formatstr ) or ( "%A" in formatstr ) or ( "%n" in formatstr ) ):
            if year <= 0:
                raise ValueError( "Use of %a, %A, or %n requires year > 0" )
            with Psycopg2Connection( connection ) as conn:
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
                conn.commit()

        names = []

        for num in range( firstnum, firstnum + number ):
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
            # available namespace.  who cares, right?  If somebody
            # cares, they can deal with it.  We're just never going to
            # have a leading a.  So, afer z comes ba.  There is no aa
            # through az.  Except for the very first a, there will never
            # be a leading a.

            letters = ""
            letnum = num
            while letnum > 0:
                dig26it = letnum % 26
                thislet = "abcdefghijklmnopqrstuvwxyz"[ dig26it ]
                letters = thislet + letters
                letnum //= 26
            letters = letters if len(letters) > 0 else 'a'

            name = formatstr
            name = name.replace( "%y", f"{year%100:02d}" )
            name = name.replace( "%Y", f"{year:04d}" )
            name = name.replace( "%n", f"{num}" )
            name = name.replace( "%a", letters )
            name = name.replace( "%A", letters.upper() )

            names.append( name )

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



class ObjectLegacySurveyMatch(Base, UUIDMixin):
    """Stores matches bewteen objects and Legacy Survey catalog sources.

    WARNING.  Because this is stored in the database, changes to the
    distance for parameter searches will not be applied to
    already-existing objects without a massive database update procedure
    (for which there is currently no code).

    Liu et al., 2025, https://ui.adsabs.harvard.edu/abs/2025arXiv250517174L/abstract
    (submitted to PASP)

    Catalog and "xgboost" score is described in that paper.

    """

    __tablename__ = "object_legacy_survey_match"

    object_id = sa.Column(
        sa.ForeignKey('objects._id', ondelete='CASCADE', name='object_ls_match_object_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the object this is a match for"
    )

    lsid = sa.Column( sa.BigInteger, nullable=False, index=False, doc="Legacy Survey ID" )
    ra = sa.Column( sa.Double, nullable=False, index=False, doc="Legacy Survey object RA" )
    dec = sa.Column( sa.Double, nullable=False, index=False, doc="Legacy Survey object Dec" )
    dist = sa.Column( sa.Double, nullable=False, index=False, doc="Distance from obj to LS obj in arcsec" )
    white_mag = sa.Column( sa.Double, nullable=False, index=False, doc="Legacy Survey object white magnitude" )
    xgboost = sa.Column( sa.REAL, nullable=False, index=False, doc="Legacy Survey object xgboost statistic" )
    is_star = sa.Column( sa.Boolean, nullable=False, index=False, doc="True if xgboost≥0.5, else False" )


    @classmethod
    def get_object_matches( cls, objid, con=None ):
        """Pull object legacy survey matches from the database.

        Parameters
        ----------
          objid : uuid
            Object ID

          con : Psycopg2Connection, default None
            Database connection.  If not given, makes and closes a new one.

        Returns
        -------
          list of ObjectLegacySurveyMatch

        """
        with Psycopg2Connection( con ) as dbcon:
            # Check for existing matches:
            cursor = dbcon.cursor()
            cursor.execute( "SELECT _id,object_id, lsid, ra, dec, dist, white_mag, xgboost, is_star "
                            "FROM object_legacy_survey_match "
                            "WHERE object_id=%(objid)s",
                            { 'objid': objid } )
            columns = { cursor.description[i][0]: i for i in range( len(cursor.description) ) }
            rows = cursor.fetchall()

        olsms = []
        for i in range( len(rows) ):
            olsms.append( ObjectLegacySurveyMatch( **{ k: rows[i][v] for k, v in columns.items() } ) )

        olsms.sort( key=lambda o: o.dist )
        return olsms


    @classmethod
    def create_new_object_matches( cls, objid, ra, dec, con=None, commit=None, exist_ok=False,
                                   verify_existing=True, **kwargs ):
        """Create new object match entries.

        Searches the liuserver for nearby objects, creates database
        entries.  May or may not commit them.  (If you pass a
        Psycopg2Connection in con and don't set commit=True, then
        the added entries will *not* be committed to the database.)

        Parameters
        ----------
          objid : uuid
            ID of the object we're matching to

          ra, dec : double
            Coordinates of the object.  Nominally, this is redundant,
            because we can get it from the database using objid, but
            it's here for convenience.  (Also, so we can run this
            routine in case the object isn't yet saved to the database.)

          con : Psycopg2Connection, default None
            Database connection to use.  If None, makes and closes a new one.

          commit : boolean, default None
            Should we commit the changes to the database?  If True, then commit,
            if False, then don't.  If None, then if con is None, treat commit as
            True; if con is not None, treat commit as False.

          exist_ok : boolean, default False
            If False, then raise an exception if the database already has matches
            for this object.

          verify_existing : boolean, default True
            Ignored if exist_ok is False.  If exist_ok is True and if
            verify_existing is False, then we just return what's already
            in the database and don't search for new stuff.  This may be
            a bad idea, though if you trust that things have already
            worked, it may be what you want. If exist_ok is True and if
            verify_existing is also True, then raise an exception if the
            new stuff found doesn't match what's in the database.

          retries, timeout0, timeoutfac, timeoutjitter : int, double, double, double
            Passed on to util/retrypost.py::retry_post

        Returns
        -------
          list of ObjectLegacySurveyMatch, sorted by dist

        """

        # Pull down things already in the database, and do checks if necessary

        existing = cls.get_object_matches( objid, con=con )

        if len( existing ) > 0:
            if not exist_ok:
                raise RuntimeError( f"Object {objid} already has {len(existing)} legacy survey matches in the "
                                    f"object_legacy_survey_match table." )
            if not verify_existing:
                return existing

        # Post to the liuserver to get LS object matches

        cfg = Config.get()
        server = cfg.value( "liumatch.server" )
        radius = cfg.value( "liumatch.radius" )
        commit = commit if commit is not None else ( con is None )

        res = retry_post( f"{server}/getsources/{ra}/{dec}/{radius}", returnjson=True, **kwargs )

        expected_keys = [ 'lsid', 'ra', 'dec', 'dist', 'white_mag', 'xgboost', 'is_star' ]
        if ( ( not isinstance( res, dict ) ) or
             ( any( k not in res.keys() for k in expected_keys ) )
            ):
            raise ValueError( f"Unexpected response from liuserver; expected a dict with keys {expected_keys}, but "
                              f"got a {type(res)}{f' with keys {res.keys()}' if isinstance(res,dict) else ''}." )

        olsms = []
        for i in range( len( res['lsid'] ) ):
            olsms.append( ObjectLegacySurveyMatch( _id=uuid.uuid4(),
                                                   object_id=objid,
                                                   lsid=res['lsid'][i],
                                                   ra=res['ra'][i],
                                                   dec=res['dec'][i],
                                                   dist=res['dist'][i],
                                                   white_mag=res['white_mag'][i],
                                                   xgboost=res['xgboost'][i],
                                                   is_star=res['is_star'][i] ) )
        olsms.sort( key=lambda o: o.dist )

        # If there are pre-existing matches in the variable existing,
        #   verify that the things we got from the liuserver (now in
        #   olsms) match them.  (If len(existing) is >0, we know that
        #   verify_existing is True, because earlier we would have
        #   already returned from this class method if len(existing) is
        #   >0 and verify_existing is False.)

        if len( existing ) > 0:
            if len( existing ) != len( olsms ):
                raise ValueError( f"Object {objid} has {len(existing)} legacy survey matches in the "
                                  f"object_legacy_survey_match table, but I just found {len(olsms)}!" )

            ok = True
            for oldolsm, newolsm in zip( existing, olsms ):
                cosdec = np.cos( oldolsm.dec * np.pi / 180. )
                if any( [ oldolsm != newolsm.lsid,
                          not np.isclose( oldolsm.ra, newolsm.ra, atol=2.8e-5/cosdec ),
                          not np.isclose( oldolsm.dec, newolsm.dec, atol=2.8e-5 ),
                          not np.isclose( oldolsm.dist, newolsm.dist, atol=0.1 ),
                          not np.isclose( oldolsm.white_mag, newolsm.white_mag, atol=0.01 ),
                          not np.isclose( oldolsm.xgboost, newolsm.xgboost, atol=0.001 ),
                          oldolsm.is_star == newolsm.is_star ] ):
                    ok = False
                    break

            if not ok:
                strio = io.StringIO()
                strio.write( f"Object {objid} already has legacy survey matches, "
                             f"but they aren't the same as what I found:\n" )
                strio.write( f"  {'Old LSID':20s} {'New LSID':20s}  {'Old RA':9s} {'New RA':9s}  "
                             f"{'Old Dec':9s} {'New Dec':9s}  {'Old d':6s} {'New d':6s}  "
                             f"{'Old m':5s} {'New m':5s}  {'Old xg':6s} {'New xg':6s}  "
                             f"{'Old is':6s} {'New is':5s}\n" )
                strio.write( "  ==================== ====================  ========= =========  "
                             "========= =========  ====== ======  ===== =====  ====== ======  ====== ======\n" )
                for oldolsm, newolsm in zip( existing, olsms ):
                    strio.write( f"  {oldolsm.lsid:20d} {newolsm.lsid:20d}  "
                                 f"{oldolsm.ra:9.5f} {newolsm.ra:9.5f}  "
                                 f"{oldolsm.dec:9.5f} {newolsm.dec:9.5f}  "
                                 f"{oldolsm.dist:6.2f} {newolsm.dist:6.2f}  "
                                 f"{oldolsm.white_mag:5.2f} {newolsm.white_mag:5.2f}  "
                                 f"{oldolsm.xgboost:6.3f} {newolsm.xgboost:6.3f}  "
                                 f"{str(oldolsm.is_star):5s} {str(newolsm.is_star):5s}\n" )
                SCLogger.error( strio.getvalue() )
                raise ValueError( f"Object {objid} already has legacy survey matches, "
                                  f"but they aren't the same as what I found." )

            return existing

        if len(olsms) == 0:
            return []
        else:
            with Psycopg2Connection( con ) as dbcon:
                cursor = dbcon.cursor()
                for olsm in olsms:
                    subdict = { k: getattr( olsm, k ) for k in expected_keys }
                    subdict['object_id'] = olsm.object_id
                    subdict['_id'] = olsm.id
                    cursor.execute( f"INSERT INTO object_legacy_survey_match(_id,object_id,{','.join(expected_keys)}) "
                                    f"VALUES(%(_id)s,%(object_id)s,{','.join(f'%({k})s' for k in expected_keys)})",
                                    subdict )
                if commit:
                    dbcon.commit()

            return olsms
