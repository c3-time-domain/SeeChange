import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.declarative import declared_attr

from models.base import Base, SeeChangeBase, SmartSession, UUIDMixin, SpatiallyIndexed, HasBitFlagBadness
from models.provenance import Provenance, provenance_self_association_table
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.cutouts import Cutouts
from models.image import Image, image_upstreams_association_table
from models.source_list import SourceList
from models.zero_point import ZeroPoint
from models.enums_and_bitflags import measurements_badness_inverse

# from util.logger import SCLogger

from improc.photometry import get_circle


class Measurements(Base, UUIDMixin, SpatiallyIndexed, HasBitFlagBadness):

    __tablename__ = 'measurements'

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
            UniqueConstraint('cutouts_id', 'index_in_sources', 'provenance_id',
                             name='_measurements_cutouts_provenance_uc'),
            sa.Index("ix_measurements_scores_gin", "disqualifier_scores", postgresql_using="gin")
        )

    cutouts_id = sa.Column(
        sa.ForeignKey('cutouts._id', ondelete="CASCADE", name='measurements_cutouts_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the cutouts object that this measurements object is associated with. "
    )

    index_in_sources = sa.Column(
        sa.Integer,
        nullable=False,
        doc="Index of the data for this Measurements"
            "in the source list (of detections in the difference image). "
    )

    object_id = sa.Column(
        sa.ForeignKey('objects._id', ondelete="CASCADE", name='measurements_object_id_fkey'),
        nullable=False,  # every saved Measurements object must have an associated Object
        index=True,
        doc="ID of the object that this measurement is associated with. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances._id', ondelete="CASCADE", name='measurements_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the provenance of this measurement. "
    )

    flux_psf = sa.Column(
        sa.REAL,
        nullable=False,
        doc="PSF flux of the measurement. "
            "This measurement has not had a background from a local annulus subtracted from it. "
    )

    flux_psf_err = sa.Column(
        sa.REAL,
        nullable=False,
        doc="PSF flux error of the measurement. "
    )

    flux_apertures = sa.Column(
        ARRAY(sa.REAL, zero_indexes=True),
        nullable=False,
        doc="Aperture fluxes of the measurement. "
            "This measurement has not had a background from a local annulus subtracted from it. "
    )

    flux_apertures_err = sa.Column(
        ARRAY(sa.REAL, zero_indexes=True),
        nullable=False,
        doc="Aperture flux errors of the measurement. "
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


    # So many other calculated properties need the zeropoint that we
    # have to be able to find it.  Users would be adivsed to set the
    # zeropoint manually if they are able...  otherwise, we're gonna
    # have six table joins to make sure we get the right zeropoint of
    # the upstream new image!  (Note that before the database refactor,
    # underneath many of these table joins were happening, but also it
    # dependend on an image object it was linked to having the manual
    # "zp" field loaded with the right thing.  So, we haven't reduced
    # the need for manual setting in the refactor.)
    @property
    def zp( self ):
        if self._zp is None:
            sub_image = orm.aliased( Image )
            sub_sources = orm.aliased( SourceList )
            imassoc = orm.aliased( image_upstreams_association_table )
            provassoc = orm.aliased( provenance_self_association_table )
            with SmartSession() as session:
                zps = ( session.query( ZeroPoint )
                        .join( SourceList, SourceList._id == ZeroPoint.sources_id )
                        .join( provassoc, provassoc.c.upstream_id == SourceList.provenance_id )
                        .join( imassoc, imassoc.c.upstream_id == SourceList.image_id )
                        .join( sub_image, sa.and_( sub_image.provenance_id == provassoc.c.downstream_id,
                                                   sub_image._id == imassoc.c.downstream_id,
                                                   sub_image.ref_image_id != SourceList.image_id ) )
                        .join( sub_sources, sub_sources.image_id == sub_image._id )
                        .join( Cutouts, sub_sources._id == Cutouts.sources_id )
                        .filter( Cutouts._id==self.cutouts_id )
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
        """The background subtracted aperture flux in the "best" aperture. """
        if self.best_aperture == -1:
            return self.flux_psf - self.bkg_mean * self.area_psf
        else:
            return self.flux_apertures[self.best_aperture] - self.bkg_mean * self.area_apertures[self.best_aperture]

    @property
    def flux_err(self):
        """The error on the background subtracted aperture flux in the "best" aperture. """
        # we divide by the number of pixels of the background as that is how well we can estimate the b/g mean
        if self.best_aperture == -1:
            return np.sqrt(self.flux_psf_err ** 2 + self.bkg_std ** 2 / self.bkg_pix * self.area_psf)
        else:
            err = self.flux_apertures_err[self.best_aperture]
            err += self.bkg_std ** 2 / self.bkg_pix * self.area_apertures[self.best_aperture]
            return np.sqrt(err)

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

    bkg_mean = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Background of the measurement, from a local annulus. Given as counts per pixel. "
    )

    bkg_std = sa.Column(
        sa.REAL,
        nullable=False,
        doc="RMS error of the background measurement, from a local annulus. Given as counts per pixel. "
    )

    bkg_pix = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Annulus area (in pixels) used to calculate the mean/std of the background. "
            "An estimate of the error on the mean would be bkg_std / sqrt(bkg_pix)."
    )

    area_psf = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Area of the PSF used for calculating flux. Remove a * background from the flux measurement. "
    )

    area_apertures = sa.Column(
        ARRAY(sa.REAL, zero_indexes=True),
        nullable=False,
        doc="Areas of the apertures used for calculating flux. Remove a * background from the flux measurement. "
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

    offset_x = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Offset in x from the center of the cutout. "
    )

    offset_y = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Offset in y from the center of the cutout. "
    )

    width = sa.Column(
        sa.REAL,
        nullable=False,
        index=True,
        doc="Width of the source in the cutout. "
            "Given by the average of the 2nd moments of the distribution of counts in the aperture. "
    )

    elongation = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Elongation of the source in the cutout. "
            "Given by the ratio of the 2nd moments of the distribution of counts in the aperture. "
            "Values close to 1 indicate a round source, while values close to 0 indicate an elongated source. "
    )

    position_angle = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Position angle of the source in the cutout. "
            "Given by the angle of the major axis of the distribution of counts in the aperture. "
    )

    is_bad = sa.Column(
        sa.Boolean,
        nullable=False,
        index=True,
        doc='Boolean flag to indicate if the measurement failed one or more threshold value comparisons. '
    )

    disqualifier_scores = sa.Column(
        JSONB,
        nullable=False,
        server_default='{}',
        index=True,
        doc="Values that may disqualify this object, and mark it as not a real source. "
            "This includes all sorts of analytical cuts defined by the provenance parameters. "
            "The higher the score, the more likely the measurement is to be an artefact. "
    )

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
        self.disqualifier_scores = {}

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
            f"<Measurements {self.id} "
            f"from Cutouts {self.cutouts_id} "
            f"(number {self.index_in_sources}) "
            f"at x,y= {self.center_x_pixel}, {self.center_y_pixel}>"
        )

    def __setattr__(self, key, value):
        if key in ['flux_apertures', 'flux_apertures_err', 'aper_radii']:
            value = np.array(value)

        if key in ['center_x_pixel', 'center_y_pixel'] and value is not None:
            value = int(np.round(value))

        super().__setattr__(key, value)

    def get_data_from_cutouts( self, cutouts=None, detections=None ):
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
        if cutouts is None:
            cutouts = Cutouts.get_by_id( self.cutouts_id )
            if cutouts is None:
                raise RuntimeError( "Can't find cutouts associated with Measurements, can't load cutouts data." )

        if detections is None:
            detections = SourceList.get_by_id( cutouts.sources_id )
            if detections is None:
                raise RuntimeError( "Can't find detections associated with Measurements, can't load cutouts data." )

        cutouts.load_all_co_data( sources=detections )

        groupname = f'source_index_{self.index_in_sources}'

        if not cutouts.co_dict.get(groupname):
            raise ValueError(f"No subdict found for {groupname}")

        co_data_dict = cutouts.co_dict[groupname] # get just the subdict with data for this

        for att in Cutouts.get_data_dict_attributes():
            setattr( self, f"_{att}", co_data_dict.get(att) )


    def get_filter_description(self, number=None, psf=None, provenance=None):
        """Use the number of the filter in the filter bank to get a string describing it.

        Parameters
        ----------
          number: int
            The number is from the list of filters, and for a given measurement you can use the
            disqualifier_score['filter bank'] to get the number of the filter that got the best S/N
            (so that filter best describes the shape of the light in the cutout).
            This is the default value for number, if it is not given.

          psf: PSF or None
            The PSF assocated with this measurement.  If not given, loads
            it from the database.  Here for efficiency.

          provenance: Provenance or None
            The provenance of this measurement.  If not given, loads it
            from the database.  Here for efficiency.

        """
        if number is None:
            number = self.disqualifier_scores.get('filter bank', None)

        if number is None:
            raise ValueError('No filter number given, and no filter bank score found. ')

        if number < 0:
            raise ValueError('Filter number must be non-negative.')

        if provenance is None:
            provenance = Provenance.get( self.provenance_id )
        if psf is None:
            with SmartSession() as session:
                psf = ( session.query( PSF )
                        .join( Cutouts, Cutouts.sources_id == PSF.sources_id )
                        .filter( Cutouts._id == self.cutouts_id ) ).first()

        if provenance is None:
            raise ValueError("Can't find for this measurement, cannot recover the parameters used. ")
        if psf is None:
            raise ValueError("Can't find psf for this measurement, cannot recover the PSF width. ")

        mult = provenance.parameters['width_filter_multipliers']
        angles = np.arange(-90.0, 90.0, provenance.parameters['streak_filter_angle_step'])
        fwhm = psf.fwhm_pixels

        if number == 0:
            return f'PSF match (FWHM= 1.00 x {fwhm:.2f})'

        if number < len(mult) + 1:
            return f'PSF mismatch (FWHM= {mult[number - 1]:.2f} x {fwhm:.2f})'

        if number < len(mult) + 1 + len(angles):
            return f'Streaked (angle= {angles[number - len(mult) - 1]:.1f} deg)'

        raise ValueError('Filter number too high for the filter bank. ')

    def associate_object(self, radius, is_testing=False, session=None):
        """Find or create a new object and associate it with this measurement.

        If no Object is found, a new one is created and saved to the
        database. Its coordinates will be identical to those of this
        Measurements object.

        This should only be done for measurements that have passed deletion_threshold
        preliminary cuts, which mostly rules out obvious artefacts. However, measurements
        which passed the deletion_threshold cuts but failed the threshold cuts should still
        be allowed to use this method - in this case, they will create an object with
        attribute is_bad set to True so they are available to review in the db.

        TODO -- this is not the right behavior.  See Issue #345.

        Parameters
        ----------
          radius: float
            Distance in arcseconds an existing Object must be within
            compared to (self.ra, self.dec) to be considered the same
            object.

          is_testing: bool, default False
            Set to True if the provenance of the measurement is a
            testing provenance.

        """
        from models.object import Object  # avoid circular import

        with SmartSession(session) as sess:
            try:
                # Avoid race condition of two processes saving a measurement of
                # the same new object at once.
                self._get_table_lock( sess, 'objects' )
                obj = sess.scalars(sa.select(Object).where(
                    Object.cone_search( self.ra, self.dec, radius, radunit='arcsec' ),
                    Object.is_test.is_(is_testing),  # keep testing sources separate
                    Object.is_bad.is_(self.is_bad),    # keep good objects with good measurements
                )).first()

                if obj is None:  # no object exists, make one based on these measurements
                    obj = Object(
                        ra=self.ra,
                        dec=self.dec,
                        is_bad=self.is_bad
                    )
                    obj.is_test = is_testing

                    # TODO -- need a way to generate object names.  The way we were
                    #   doing it before no longer works since it depended on numeric IDs.
                    # (Issue #347)
                    obj.name = str( obj.id )[-12:]

                    # SCLogger.debug( "Measurements.associate_object calling Object.insert (which will commit)" )
                    obj.insert( session=sess )

                self.object_id = obj.id
            finally:
                # Assure that lock is released
                # SCLogger.debug( "Measurements.associate_object rolling back" )
                sess.rollback()

    def get_flux_at_point( self, ra, dec, aperture=None, wcs=None, psf=None ):
        """Use the given coordinates to find the flux, assuming it is inside the cutout.

        Parameters
        ----------
        ra: float
            The right ascension of the point in degrees.

        dec: float
            The declination of the point in degrees.

        aperture: int, optional
            Use this aperture index in the list of aperture radii to choose
            which aperture to use. Set -1 to get PSF photometry.
            Leave None to use the best_aperture.
            Can also specify "best" or "psf".

        wcs: WorldCoordinates, optional
            The WCS to use to go from ra/dec to x/y.  If not given, will
            try to find it in the database using a rather tortured query.

        psf: PSF, optional
            The PSF from the sub_image this measurement came from.  If
            not given, will try to find it in the database.  (Actually,
            it won't, because that's complicated.  Just pass a PSF if
            aperture is -1.)

        Returns
        -------
        flux: float
            The flux in the aperture.
        fluxerr: float
            The error on the flux.
        area: float
            The area of the aperture.

        """
        if aperture is None:
            aperture = self.best_aperture
        if aperture == 'best':
            aperture = self.best_aperture
        if aperture == 'psf':
            aperture = -1

        if self.sub_data is None:
            raise RuntimeError( "Run get_data_from_cutouts before running get_flux_at_point" )

        im = self.sub_nandata  # the cutouts image we are working with (includes NaNs for bad pixels)

        if wcs is None:
            with SmartSession() as session:
                wcs = ( session.query( WorldCoordinates )
                        .join( Cutouts, WorldCoordinates.sources_id==Cutouts.sources_id )
                        .filter( Cutouts.id==self.cutouts_id ) ).first()
            if wcs is None:
                # There was no WorldCoordiantes for the sub image, so we're going to
                #   make an assumption that we make elsewhere: that the wcs for the
                #   sub image is the same as the wcs for the new image.  This is
                #   almost the same query that's used in zp() above.
                sub_image = orm.aliased( Image )
                sub_sources = orm.aliased( SourceList )
                imassoc = orm.aliased( image_upstreams_association_table )
                provassoc = orm.aliased( provenance_self_association_table )
                wcs = ( session.query( WorldCoordinates )
                        .join( SourceList, SourceList._id == WorldCoordinates.sources_id )
                        .join( provassoc, provassoc.c.upstream_id == SourceList.provenance_id )
                        .join( imassoc, imassoc.c.upstream_id == SourceList.image_id )
                        .join( sub_image, sa.and_( sub_image.provenance_id == provassoc.c.downstream_id,
                                                   sub_image._id == imassoc.c.downstream_id,
                                                   sub_image.ref_image_id != SourceList.image_id ) )
                        .join( sub_sources, sub_sources.image_id == sub_image._id )
                        .join( Cutouts, sub_sources._id == Cutouts.sources_id )
                        .filter( Cutouts._id==self.cutouts_id )
                       ).all()
                if len(wcs) > 1:
                    raise RuntimeError( f"Found more than one WCS for measurements {self.id}, this shouldn't happen!" )
                if len(wcs) == 0:
                    raise RuntimeError( f"Couldn't find a WCS for measurements {self.id}" )
                else:
                    wcs = wcs[0]
        wcs = wcs.wcs

        # these are the coordinates relative to the center of the cutouts
        image_pixel_x = wcs.world_to_pixel_values(ra, dec)[0]
        image_pixel_y = wcs.world_to_pixel_values(ra, dec)[1]

        offset_x = image_pixel_x - self.center_x_pixel
        offset_y = image_pixel_y - self.center_y_pixel

        if abs(offset_x) > im.shape[1] / 2 or abs(offset_y) > im.shape[0] / 2:
            return np.nan, np.nan, np.nan  # quietly return NaNs for large offsets, they will fail the cuts anyway...

        if np.isnan(image_pixel_x) or np.isnan(image_pixel_y):
            return np.nan, np.nan, np.nan  # if we can't use the WCS for some reason, need to fail gracefully

        if aperture == -1:
            # get the subtraction PSF or (if unavailable) the new image PSF
            # NOTE -- right now we're just getting the new image PSF, as we don't
            # currently have code that saves the subtraction PSF
            if psf is None:
                raise ValueError( "Must pass PSF if you want to do PSF photometry." )

            psf_clip = psf.get_clip(x=image_pixel_x, y=image_pixel_y)
            offset_ix = int(np.round(offset_x))
            offset_iy = int(np.round(offset_y))
            # shift the psf_clip by the offset and multiply by the cutouts sub_flux
            # the corner offset between the pixel coordinates of the cutout to that of the psf_clip:
            dx = psf_clip.shape[1] // 2 - im.shape[1] // 2 - offset_ix
            dy = psf_clip.shape[0] // 2 - im.shape[0] // 2 - offset_iy
            start_x = max(0, -dx)  # where (in cutout coordinates) do we start counting the pixels
            end_x = min(im.shape[1], psf_clip.shape[1] - dx)  # where do we stop counting the pixels
            start_y = max(0, -dy)
            end_y = min(im.shape[0], psf_clip.shape[0] - dy)

            # make a mask the same size as the cutout, with the offset PSF and zeros where it is not overlapping
            # before clipping the non overlapping and removing bad pixels, the PSF clip was normalized to 1
            mask = np.zeros_like(im, dtype=float)
            mask[start_y:end_y, start_x:end_x] = psf_clip[start_y + dy:end_y + dy, start_x + dx:end_x + dx]
            mask[np.isnan(im)] = 0  # exclude bad pixels from the mask

            mask_sum = np.nansum(mask ** 2)
            if mask_sum > 0:
                flux = np.nansum(im * mask) / np.nansum(mask ** 2)
                fluxerr = self.bkg_std / np.sqrt(np.nansum(mask ** 2))
                area = np.nansum(mask) / (np.nansum(mask ** 2))
            else:
                flux = fluxerr = area = np.nan
        else:
            radius = self.aper_radii[aperture]
            # get the aperture mask
            mask = get_circle(radius=radius, imsize=im.shape[0], soft=True).get_image(offset_x, offset_y)
            # for aperture photometry we don't normalize, just assume the PSF is in the aperture
            flux = np.nansum(im * mask)
            fluxerr = self.bkg_std * np.sqrt(np.nansum(mask ** 2))
            area = np.nansum(mask)

        return flux, fluxerr, area


    def _get_inverse_badness(self):
        return measurements_badness_inverse

    def get_upstreams( self, session=None ):
        """Return the upstreams of this Measurements object.

        Will be the Cutouts that these measurements are from.
        """

        with SmartSession( session ) as session:
            return session.scalars( sa.Select( Cutouts ).where( Cutouts._id == self.cutouts_id ) ).all()

    def get_downstreams( self, session=None, siblings=False ):
        """Get downstream data products of this Measurements."""

        # Measurements doesn't currently have downstreams; this will
        #  change with the R/B score object.
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
