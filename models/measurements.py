import uuid

import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declared_attr

from util.util import asUUID
from models.base import ( Base,
                          SeeChangeBase,
                          UUIDMixin,
                          SpatiallyIndexed,
                          HasBitFlagBadness,
                          SmartSession,
                          Psycopg2Connection )
from models.cutouts import Cutouts
from models.image import Image
from models.source_list import SourceList
from models.zero_point import ZeroPoint
from models.reference import image_subtraction_components
from models.enums_and_bitflags import measurements_badness_inverse

# from util.logger import SCLogger


class MeasurementSet( Base, UUIDMixin, HasBitFlagBadness ):
    # A measurement set is a way of having a single upstream for things
    #   that depend on a wholse set of Measurements (like a ScoreSet,
    #   and, depending on the ScoreSet, a Report, or a FakeAnalysis).

    __tablename__ = 'measurement_sets'

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            UniqueConstraint('cutouts_id', 'provenance_id', name='_measurement_sets_uc'),
        )
    cutouts_id = sa.Column(
        sa.ForeignKey('cutouts._id', ondelete="CASCADE", name='meas_set_cutouts_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the cutouts object that this measurements object is associated with. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances._id', ondelete="CASCADE", name='meas_set_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the provenance of this measurement set."
    )

    @property
    def measurements( self ):
        if self._measurements is None:
            with SmartSession() as session:
                self._measurements = list( session.scalars( sa.select( Measurements )
                                                            .where( Measurements.measurementset_id == self._id )
                                                            .order_by( Measurements.index_in_sources )
                                                           ).all() )
        return self._measurements

    @measurements.setter
    def measurements( self, val ):
        if ( not isinstance( val, list ) ) or ( not all( isinstance( m, Measurements ) for m in val ) ):
            raise TypeError( "measurements must be a list of Measurements" )
        self._measurements = val
        for m in self._measurements:
            m.measurementset_id = self.id


    def __init__( self, *args, **kwargs ):
        SeeChangeBase.__init__( self )
        self._measurements = None
        self.set_attributes_from_dict( kwargs )

    @orm.reconstructor
    def init_on_load( self ):
        SeeChangeBase.init_on_load( self )
        self._measurements = None


    def get_upstreams( self, session=None ):
        """Return the upstreams of this MeasurementSet object.

        Will be the Cutouts that these measurements are from.

        """
        with SmartSession( session ) as session:
            return session.scalars( sa.Select( Cutouts ).where( Cutouts._id == self.cutouts_id ) ).all()

    def get_downstreams( self, session=None ):
        """Return the downstreams of this MeasurementSet object.

        Includes any DeepScore objects, plus all Measurements that are
        members of this set.

        """
        from models.deepscore import DeepScoreSet
        with SmartSession( session ) as session:
            downstreams = list( session.scalars( sa.Select( DeepScoreSet )
                                                 .where( DeepScoreSet.measurementset_id == self.id )
                                                ).all() )
            downstreams.extend( list( session.scalars( sa.Select( Measurements )
                                                       .where( Measurements.measurementset_id == self.id )
                                                      ).all() ) )
        return downstreams



class Measurements(Base, UUIDMixin, SpatiallyIndexed, HasBitFlagBadness):

    __tablename__ = 'measurements'

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
            UniqueConstraint('measurementset_id', 'index_in_sources', name='_measurements_uc'),
        )

    measurementset_id = sa.Column(
        sa.ForeignKey( 'measurement_sets._id', ondelete="CASCADE", name="meas_meas_set_id_fkey" ),
        nullable=False,
        index=True,
        doc="ID of the Measurement Set this measurements object is a member of."
    )

    index_in_sources = sa.Column(
        sa.Integer,
        nullable=False,
        index=True,
        doc=( "Index in the source list (of detections in the difference image) "
              "that corresponds to this Measurements; also (effectively) gives the cutout index." )
    )

    object_id = sa.Column(
        sa.ForeignKey('objects._id', ondelete="CASCADE", name='measurements_object_id_fkey'),
        nullable=False,  # every saved Measurements object must have an associated Object
        index=True,
        doc="ID of the object that this measurement is associated with. "
    )

    flux_psf = sa.Column(
        sa.REAL,
        nullable=False,
        doc="PSF flux of the measurement. "
    )

    flux_psf_err = sa.Column(
        sa.REAL,
        nullable=False,
        doc="PSF flux error of the measurement. "
    )

    flux_apertures = sa.Column(
        ARRAY(sa.REAL, zero_indexes=True),
        nullable=False,
        doc="Aperture fluxes of the measurement.  Does not include aperture corrections."
    )

    flux_apertures_err = sa.Column(
        ARRAY(sa.REAL, zero_indexes=True),
        nullable=False,
        doc="Aperture flux errors of the measurement. "
    )

    bkg_per_pix = sa.Column(
        sa.REAL,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '0.0' ),
        doc=( "The background level of the cutout of this measurement.  This background has "
              "already been subtracted from the flux_ properties." )
    )

    aper_radii = sa.Column(
        ARRAY(sa.REAL, zero_indexes=True),
        nullable=False,
        doc="Radii of the apertures used for calculating flux, in pixels. "
    )

    best_aperture = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '-1' ),
        doc="The index of the aperture that was chosen as the best aperture for this measurement. "
            "Set to -1 to select the PSF flux instead of one of the apertures. "
    )

    center_x_pixel = sa.Column(
        sa.Integer,
        nullable=False,
        doc="X pixel coordinate of the center of the cutout (in the full image coordinates),"
            "rounded to nearest integer pixel. "
    )

    center_y_pixel = sa.Column(
        sa.Integer,
        nullable=False,
        doc="Y pixel coordinate of the center of the cutout (in the full image coordinates),"
            "rounded to nearest integer pixel. "
    )

    x = sa.Column(
        sa.REAL,
        nullable=False,
        doc="x pixel coordinate on the subtraction image of the fit psf profile"
    )

    y = sa.Column(
        sa.REAL,
        nullable=False,
        doc="y pixel coordinate on the subtraction image of the fit psf profile"
    )

    gfit_x = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Position of a gaussian fit to the cutout."
    )

    gfit_y = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Position of a gaussian fit to the cutout."
    )

    major_width = sa.Column(
        sa.REAL,
        nullable=False,
        doc=( "Major axis width of the source in the cutout. "
              "Calculated as the FWHM of a Gaussian fit." )
    )

    minor_width = sa.Column(
        sa.REAL,
        nullable=False,
        doc=( "Minor axis width of the source in the cutout. "
              "Calculated as the FWHM of a Gaussian fit." )
    )

    position_angle = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Position angle of the source in the cutout. "
            "Defined as relative to the x-axis, between -pi/2 and pi/2.  From Gaussian fit."
    )

    # Futher diagnostics are nullable, as we may add diagnostics, and
    #   the existing Measurements won't have values to put in when we
    #   migrate.  Ideally, we'll do it right and no provenance that
    #   includes these diagnostics will be working on older Measurements
    #   that don't have them, but deal with that when it comes up.

    psf_fit_flags = sa.Column(
        sa.Integer,
        nullable=True,
        doc=( "The flags returned by photutils PSFPhotometry" )
    )

    nbadpix = sa.Column(
        sa.Integer,
        nullable=True,
        doc=( "Number of bad pixels in a configurable square (usually radius 2FWHM) around (x,y)" )
    )

    negfrac = sa.Column(
        sa.REAL,
        nullable=True,
        doc=( "Number of significantly negative / significantly positive pixels in a configurable "
              "square around (x,y).  Square is the same as that for nbadpix; 'significant' usually=>2sigma" )
    )

    negfluxfrac = sa.Column(
        sa.REAL,
        nullable=True,
        doc=( "|Flux| in the negative pixels divided by flux in the positive pixels used in negfrac" )
    )

    # End of further diagnostics.

    is_bad = sa.Column(
        sa.Boolean,
        nullable=False,
        index=True,
        doc='Boolean flag to indicate if the measurement failed one or more threshold value comparisons. '
    )

    # So many other calculated properties need the zeropoint that we
    # have to be able to find it.  Users would be adivsed to set the
    # zeropoint manually if they are able, so that each and every
    # measurement in a list doesn't query the database separately for
    # that zeropoint.
    @property
    def zp( self ):
        if self._zp is None:
            isc = orm.aliased( image_subtraction_components )
            with SmartSession() as session:
                zps = ( session.query( ZeroPoint )
                        .join( isc, isc.c.new_zp_id==ZeroPoint._id )
                        .join( Image, Image._id==isc.c.image_id )
                        .join( SourceList, SourceList.image_id==Image._id )
                        .join( Cutouts, Cutouts.sources_id==SourceList._id )
                        .join( MeasurementSet, MeasurementSet.cutouts_id==Cutouts._id )
                        .filter( MeasurementSet._id==self.measurementset_id )
                       ).all()
            if len( zps ) > 1:
                raise RuntimeError( "Found multiple zeropoints for Measurements, this shouldn't happen!" )
            if len( zps ) == 0:
                self._zp = None
            else:
                self._zp = zps[0]
        if self._zp is None:
            raise RuntimeError( "Couldn't find ZeroPoint for Measurements in the database.  Make sure the "
                                "ZeroPoint is loaded." )
        return self._zp

    # Normally, I wouldn't have a setter here, but because the query above is
    #  so nasty, put this here for efficiency
    @zp.setter
    def zp( self, val ):
        if not isinstance( val, ZeroPoint ):
            raise TypeError( "Measurements.zp must be a ZeroPoint" )
        self._zp = val

    @property
    def flux(self):

        """The background subtracted aperture flux in the "best" aperture.  Does not aperture correct."""
        if self.best_aperture == -1:
            return self.flux_psf
        else:
            return self.flux_apertures[self.best_aperture]

    @property
    def flux_err(self):
        """The error on the background subtracted aperture flux in the "best" aperture. """
        if self.best_aperture == -1:
            return self.flux_psf_err
        else:
            return self.flux_apertures_err[ self.best_aperture ]

    @property
    def mag_psf(self):
        if self.flux_psf <= 0:
            return np.nan
        return -2.5 * np.log10(self.flux_psf) + self.zp.zp

    @property
    def mag_psf_err(self):
        if self.flux_psf <= 0:
            return np.nan
        return np.sqrt((2.5 / np.log(10) * self.flux_psf_err / self.flux_psf) ** 2 + self.zp.dzp ** 2)

    @property
    def mag_apertures(self):
        mags = []
        for flux, correction in zip(self.flux_apertures, self.zp.aper_cors):
            new_mag = -2.5 * np.log10(flux) + self.zp.zp + correction if flux > 0 else np.nan
            mags.append(new_mag)

        return mags

    @property
    def mag_apertures_err(self):
        errs = []
        for flux, flux_err in zip(self.flux_apertures, self.flux_apertures_err):
            if flux > 0:
                new_err = np.sqrt((2.5 / np.log(10) * flux_err / flux) ** 2 + self.zp.dzp ** 2)
            else:
                new_err = np.nan
            errs.append(new_err)
        return errs

    @property
    def magnitude(self):
        mag = -2.5 * np.log10(self.flux) + self.zp.zp
        if self.best_aperture == -1:
            return mag
        else:
            return mag + self.zp.aper_cors[self.best_aperture]

    @property
    def magnitude_err(self):
        return np.sqrt((2.5 / np.log(10) * self.flux_err / self.flux) ** 2 + self.zp.dzp ** 2)


    @property
    def sub_data(self):
        if self._sub_data is None:
            self.get_data_from_cutouts()
        return self._sub_data

    @sub_data.setter
    def sub_data( self, val ):
        raise RuntimeError( "Don't set sub_data, use get_data_from_cutouts()" )

    @property
    def sub_weight(self):
        if self._sub_weight is None:
            self.get_data_from_cutouts()
        return self._sub_weight

    @sub_weight.setter
    def sub_weight( self, val ):
        raise RuntimeError( "Don't set sub_weight, use get_data_from_cutouts()" )

    @property
    def sub_flags(self):
        if self._sub_flags is None:
            self.get_data_from_cutouts()
        return self._sub_flags

    @sub_flags.setter
    def sub_flags( self, val ):
        raise RuntimeError( "Don't set sub_flags, use get_data_from_cutouts()" )

    @property
    def ref_data(self):
        if self._ref_data is None:
            self.get_data_from_cutouts()
        return self._ref_data

    @ref_data.setter
    def ref_data( self, val ):
        raise RuntimeError( "Don't set ref_data, use get_data_from_cutouts()" )

    @property
    def ref_weight(self):
        if self._ref_weight is None:
            self.get_data_from_cutouts()
        return self._ref_weight

    @ref_weight.setter
    def ref_weight( self, val ):
        raise RuntimeError( "Don't set ref_weight, use get_data_from_cutouts()" )

    @property
    def ref_flags(self):
        if self._ref_flags is None:
            self.get_data_from_cutouts()
        return self._ref_flags

    @ref_flags.setter
    def ref_flags( self, val ):
        raise RuntimeError( "Don't set ref_flags, use get_data_from_cutouts()" )

    @property
    def new_data(self):
        if self._new_data is None:
            self.get_data_from_cutouts()
        return self._new_data

    @new_data.setter
    def new_data( self, val ):
        raise RuntimeError( "Don't set new_data, use get_data_from_cutouts()" )

    @property
    def new_weight(self):
        if self._new_weight is None:
            self.get_data_from_cutouts()
        return self._new_weight

    @new_weight.setter
    def new_weight( self, val ):
        raise RuntimeError( "Don't set new_weight, use get_data_from_cutouts()" )

    @property
    def new_flags(self):
        if self._new_flags is None:
            self.get_data_from_cutouts()
        return self._new_flags

    @new_flags.setter
    def new_flags( self, val ):
        raise RuntimeError( "Don't set new_flags, use get_data_from_cutouts()" )

    @property
    def sub_nandata(self):
        if self.sub_data is None or self.sub_flags is None:
            return None
        return np.where(self.sub_flags > 0, np.nan, self.sub_data)

    @property
    def ref_nandata(self):
        if self.ref_data is None or self.ref_flags is None:
            return None
        return np.where(self.ref_flags > 0, np.nan, self.ref_data)

    @property
    def new_nandata(self):
        if self.new_data is None or self.new_flags is None:
            return None
        return np.where(self.new_flags > 0, np.nan, self.new_data)

    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values
        HasBitFlagBadness.__init__(self)

        self.index_in_sources = None

        self._sub_data = None
        self._sub_weight = None
        self._sub_flags = None
        self._sub_psfflux = None
        self._sub_psffluxerr = None

        self._ref_data = None
        self._ref_weight = None
        self._ref_flags = None

        self._new_data = None
        self._new_weight = None
        self._new_flags = None

        self._zp = None

        # These are server defaults, but we might use them
        #  before saving and reloading
        self.best_aperture = -1

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.calculate_coordinates()

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)

        self._sub_data = None
        self._sub_weight = None
        self._sub_flags = None
        self._sub_psfflux = None
        self._sub_psffluxerr = None

        self._ref_data = None
        self._ref_weight = None
        self._ref_flags = None

        self._new_data = None
        self._new_weight = None
        self._new_flags = None

        self._zp = None

    def __repr__(self):
        return (
            f"<Measurements {self.id} in set {self.measurement_set} "
            f"(number {self.index_in_sources}) "
            f"at x,y= {self.center_x_pixel}, {self.center_y_pixel}>"
        )

    def __setattr__(self, key, value):
        if key in ['flux_apertures', 'flux_apertures_err', 'aper_radii']:
            value = np.array(value)

        if key in ['center_x_pixel', 'center_y_pixel'] and value is not None:
            value = int(np.round(value))

        super().__setattr__(key, value)

    def get_data_from_cutouts( self, cutouts=None, detections=None, session=None ):
        """Populates this object with the cutout data arrays used in
        calculations. This allows us to use, for example, self.sub_data
        without having to look constantly back into the related Cutouts.

        Parameters
        ----------
        cutouts: Cutouts or None
            The Cutouts to load the data from.  load_all_co_data will be
            called on this to make sure the cutouts dictionary is
            loaded.  (That function checks to see if it's there already,
            and doesn't reload it it looks right.)  If None, will try to
            find the cutouts in the database.

        detections: SourceList or None
            The detections associated with cutouts.  Needed because
            laod_all_co_data needs sources.  If you leave this at None,
            it will try to load the SourceList from the database.  Pass
            this for efficiency, or if the cutouts or detections aren't
            already in the database.

        """
        if ( cutouts is None ) or ( detections is None ):
            with SmartSession( session ) as sess:
                if cutouts is None:
                    cutouts = ( sess.query( Cutouts )
                                .join( MeasurementSet, MeasurementSet.cutouts_id==Cutouts._id )
                                .filter( MeasurementSet._id==self.measurementset_id )
                               ).first()
                    if cutouts is None:
                        raise RuntimeError( "Can't find cutouts associated with Measurements, "
                                            "can't load cutouts data." )

                if detections is None:
                    detections = SourceList.get_by_id( cutouts.sources_id )
                    if detections is None:
                        raise RuntimeError( "Can't find detections associated with Measurements, "
                                            "can't load cutouts data." )

        cutouts.load_all_co_data( sources=detections )

        groupname = f'source_index_{self.index_in_sources}'

        if not cutouts.co_dict.get(groupname):
            raise ValueError(f"No subdict found for {groupname}")

        co_data_dict = cutouts.co_dict[groupname] # get just the subdict with data for this

        for att in Cutouts.get_data_array_attributes():
            setattr( self, f"_{att}", co_data_dict.get(att) )
        for att in Cutouts.get_data_scalar_attributes():
            setattr( self, f"_{att}", co_data_dict.get(att) )


    def associate_object(self, radius, is_testing=False, is_fake=False, connection=None):
        """Find or create a new object and associate it with this measurement.

        If no Object is found, a new one is created and saved to the
        database. Its coordinates will be identical to those of this
        Measurements object.

        This should only be done for measurements that have passed
        deletion_threshold preliminary cuts, which mostly rules out
        obvious artifacts.  Measurements which do pass those thresholds
        and are saved still do get an object associated with them.  Good
        objects will eventually have a mix of "good" and "bad"
        measurements assocated with them, as there will be some low S/N
        detections that will randomly not pass some of the thresholds.
        (If the "bad" and "deletion" thresholds are the same, then no
        is_bad measurements will get saved to the database in the first
        place.)

        Parameters
        ----------
          radius: float
            Distance in arcseconds an existing Object must be within
            compared to (self.ra, self.dec) to be considered the same
            object.

          is_testing: bool, default False
            Set to True if the provenance of the measurement is a
            testing provenance.

          is_fake: bool, default False
            Set to True if this is a measurement of a fake.

          connection: psycopg2 connection or None

        """
        from models.object import Object  # avoid circular import

        obj = None
        with Psycopg2Connection( connection ) as conn:
            try:
                cursor = conn.cursor()
                # Avoid race condition of two processes trying to
                #   create the same object at once.
                cursor.execute( "LOCK TABLE objects" )
                cursor.execute( ( "SELECT _id, name, ra, dec, is_test, is_fake, is_bad FROM objects WHERE "
                                  "q3c_radial_query( ra, dec, %(ra)s, %(dec)s, %(radius)s ) "
                                  "AND is_fake=%(fake)s" ),
                                { 'ra': self.ra, 'dec': self.dec, 'radius': radius/3600., 'fake': is_fake } )
                rows = cursor.fetchall()
                if len(rows) == 0:
                    objid = uuid.uuid4()
                    # TODO -- need a way to generate object names.  The way we were
                    #   doing it before no longer works since it depended on numeric IDs.
                    # (Issue #347)
                    objname = str( objid )[-12:]
                    cursor.execute( ( "INSERT INTO objects(_id,ra,dec,name,is_test,is_fake,is_bad) "
                                      "VALUES(%(id)s, %(ra)s, %(dec)s, %(name)s, %(testing)s, FALSE, FALSE)" ),
                                    { 'id': objid, 'name': objname, 'ra': self.ra, 'dec': self.dec,
                                      'testing': is_testing } )
                    conn.commit()
                    obj = Object( _id=objid,
                                  name=objname,
                                  ra=self.ra,
                                  dec=self.dec,
                                  is_test=is_testing,
                                  is_fake=is_fake,
                                  is_bad=False )
                else:
                    obj = Object( _id=asUUID(rows[0][0]),
                                  name=rows[0][1],
                                  ra=rows[0][2],
                                  dec=rows[0][3],
                                  is_test=rows[0][4],
                                  is_fake=rows[0][5],
                                  is_bad=rows[0][6] )

            finally:
                conn.rollback()

        if obj is None:
            raise RuntimeError( "This should never happen." )

        self.object_id = obj.id
        return obj

    def _get_inverse_badness(self):
        return measurements_badness_inverse

    def get_upstreams( self, session=None ):
        """Return the upstreams of this Measurements object.

        Will be the MeasurementSet that this Measurements is a member of.
        """

        with SmartSession( session ) as session:
            return session.scalars( sa.Select( MeasurementSet )
                                    .where( MeasurementSet._id == self.measurementset_id )
                                   ).all()


    def get_downstreams( self, session=None ):
        """Measurements has no downstreams."""
        return []


    @classmethod
    def delete_list(cls, measurements_list):
        """Remove a list of Measurements objects from the database.

        Parameters
        ----------
        measurements_list: list of Measurements
            The list of Measurements objects to remove.
        """
        for m in measurements_list:
            m.delete_from_disk_and_database()
