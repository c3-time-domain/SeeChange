import uuid
import operator
import numpy as np
from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declared_attr

from astropy.time import Time
from astropy.coordinates import SkyCoord

from models.base import Base, SeeChangeBase, SmartSession, Psycopg2Connection, UUIDMixin, SpatiallyIndexed
from models.image import Image
from models.cutouts import Cutouts
from models.source_list import SourceList
from models.measurements import Measurements
from util.config import Config


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

    def get_measurements_list(
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
    def associate_measurements( cls, measurements, radius=None, year=None, no_new=False, is_testing=False ):
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
