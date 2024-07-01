import numpy as np

from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

from models.base import Base, SeeChangeBase, SmartSession, AutoIDMixin, SpatiallyIndexed, HasBitFlagBadness
from models.cutouts import Cutouts
from models.enums_and_bitflags import measurements_badness_inverse

from improc.photometry import get_circle


class Measurements(Base, AutoIDMixin, SpatiallyIndexed, HasBitFlagBadness):

    __tablename__ = 'measurements'

    __table_args__ = (
        UniqueConstraint('cutouts_id', 'index_in_sources', 'provenance_id', name='_measurements_cutouts_provenance_uc'),
        sa.Index("ix_measurements_scores_gin", "disqualifier_scores", postgresql_using="gin"),
    )

    cutouts_id = sa.Column(
        sa.ForeignKey('cutouts.id', ondelete="CASCADE", name='measurements_cutouts_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the cutouts object that this measurements object is associated with. "
    )

    cutouts = orm.relationship(
        Cutouts,
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        lazy='selectin',
        doc="The cutouts object that this measurements object is associated with. "
    )

    index_in_sources = sa.Column(
        sa.Integer,
        nullable=False,
        doc="Index of the data for this Measurements"
            "in the source list (of detections in the difference image). "
    )

    object_id = sa.Column(
        sa.ForeignKey('objects.id', ondelete="CASCADE", name='measurements_object_id_fkey'),
        nullable=False,  # every saved Measurements object must have an associated Object
        index=True,
        doc="ID of the object that this measurement is associated with. "
    )

    object = orm.relationship(
        'Object',
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        lazy='selectin',
        doc="The object that this measurement is associated with. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='measurements_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the provenance of this measurement. "
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc="The provenance of this measurement. "
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
        default=-1,
        doc="The index of the aperture that was chosen as the best aperture for this measurement. "
            "Set to -1 to select the PSF flux instead of one of the apertures. "
    )

    mjd = association_proxy('cutouts', 'sources.image.mjd')

    exp_time = association_proxy('cutouts', 'sources.image.exp_time')

    filter = association_proxy('cutouts', 'sources.image.filter')

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

    @property
    def lim_mag(self):
        return self.sources.image.new_image.lim_mag_estimate  # TODO: improve this when done with issue #143

    @property
    def zp(self):
        return self.sources.image.new_image.zp

    @property
    def fwhm_pixels(self):
        return self.sources.image.get_psf().fwhm_pixels

    @property
    def psf(self):
        return self.sources.image.get_psf().get_clip(x=self.center_x_pixel, y=self.center_y_pixel)

    @property
    def pixel_scale(self):
        return self.sources.image.new_image.wcs.get_pixel_scale()

    @property
    def sources(self):
        if self.cutouts is None:
            return None
        return self.cutouts.sources

    @property
    def image(self):
        if self.cutouts is None or self.sources is None:
            return None
        return self.sources.image

    @property
    def instrument_object(self):
        if self.cutouts is None or self.sources is None or self.sources.image is None:
            return None
        return self.sources.image.instrument_object

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
        default={},
        index=True,
        doc="Values that may disqualify this object, and mark it as not a real source. "
            "This includes all sorts of analytical cuts defined by the provenance parameters. "
            "The higher the score, the more likely the measurement is to be an artefact. "
    )

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

    def __repr__(self):
        return (
            f"<Measurements {self.id} "
            f"from SourceList {self.cutouts.sources_id} "
            f"(number {self.index_in_sources}) "
            f"from Image {self.cutouts.sub_image_id} "
            f"at x,y= {self.center_x_pixel}, {self.center_y_pixel}>"
        )

    def __setattr__(self, key, value):
        if key in ['flux_apertures', 'flux_apertures_err', 'aper_radii']:
            value = np.array(value)

        if key in ['center_x_pixel', 'center_y_pixel'] and value is not None:
            value = int(np.round(value))

        super().__setattr__(key, value)

    def get_data_from_cutouts(self):
        """Populates this object with the cutout data arrays used in
        calculations. This allows us to use, for example, self.sub_data
        without having to look constantly back into the related Cutouts.

        Importantly, the data for this measurements should have already
        been loaded by the Co_Dict class
        """
        groupname = f'source_index_{self.index_in_sources}'

        if not self.cutouts.co_dict.get(groupname):
            raise ValueError(f"No subdict found for {groupname}")

        co_data_dict = self.cutouts.co_dict[groupname] # get just the subdict with data for this

        for att in Cutouts.get_data_dict_attributes():
            setattr(self, att, co_data_dict.get(att))


    def get_filter_description(self, number=None):
        """Use the number of the filter in the filter bank to get a string describing it.

        The number is from the list of filters, and for a given measurement you can use the
        disqualifier_score['filter bank'] to get the number of the filter that got the best S/N
        (so that filter best describes the shape of the light in the cutout).
        This is the default value for number, if it is not given.
        """
        if number is None:
            number = self.disqualifier_scores.get('filter bank', None)

        if number is None:
            raise ValueError('No filter number given, and no filter bank score found. ')

        if number < 0:
            raise ValueError('Filter number must be non-negative.')
        if self.provenance is None:
            raise ValueError('No provenance for this measurement, cannot recover the parameters used. ')
        if self.cutouts is None or self.sources is None or self.sources.image is None:
            raise ValueError('No cutouts for this measurement, cannot recover the PSF width. ')

        mult = self.provenance.parameters['width_filter_multipliers']
        angles = np.arange(-90.0, 90.0, self.provenance.parameters['streak_filter_angle_step'])
        fwhm = self.sources.image.get_psf().fwhm_pixels

        if number == 0:
            return f'PSF match (FWHM= 1.00 x {fwhm:.2f})'

        if number < len(mult) + 1:
            return f'PSF mismatch (FWHM= {mult[number - 1]:.2f} x {fwhm:.2f})'

        if number < len(mult) + 1 + len(angles):
            return f'Streaked (angle= {angles[number - len(mult) - 1]:.1f} deg)'

        raise ValueError('Filter number too high for the filter bank. ')

    def associate_object(self, session=None):
        """Find or create a new object and associate it with this measurement.

        Objects must have sufficiently close coordinates to be associated with this
        measurement (set by the provenance.parameters['association_radius'], in arcsec).

        If no Object is found, a new one is created, and its coordinates will be identical
        to those of this Measurements object.

        This should only be done for measurements that have passed deletion_threshold 
        preliminary cuts, which mostly rules out obvious artefacts. However, measurements
        which passed the deletion_threshold cuts but failed the threshold cuts should still
        be allowed to use this method - in this case, they will create an object with
        attribute is_bad set to True so they are available to review in the db.
        
        """
        from models.object import Object  # avoid circular import

        with SmartSession(session) as session:
            obj = session.scalars(sa.select(Object).where(
                Object.cone_search(
                    self.ra,
                    self.dec,
                    self.provenance.parameters['association_radius'],
                    radunit='arcsec',
                ),
                Object.is_test.is_(self.provenance.is_testing),  # keep testing sources separate
                Object.is_bad.is_(self.is_bad),    # keep good objects with good measurements
            )).first()

            if obj is None:  # no object exists, make one based on these measurements
                obj = Object(
                    ra=self.ra,
                    dec=self.dec,
                    is_bad=self.is_bad
                )
                obj.is_test = self.provenance.is_testing

            self.object = obj

    def get_flux_at_point(self, ra, dec, aperture=None):
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

        im = self.sub_nandata  # the cutouts image we are working with (includes NaNs for bad pixels)

        wcs = self.sources.image.new_image.wcs.wcs
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
            psf = self.sources.image.get_psf()
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

    def get_upstreams(self, session=None):
        """Get the image that was used to make this source list. """
        with SmartSession(session) as session:
            return session.scalars(sa.select(Cutouts).where(Cutouts.id == self.cutouts_id)).all()

    def get_downstreams(self, session=None, siblings=False):
        """Get the downstreams of this Measurements"""
        return []

    def _get_inverse_badness(self):
        return measurements_badness_inverse

    @classmethod
    def delete_list(cls, measurements_list, session=None, commit=True):
        """
        Remove a list of Measurements objects from the database.

        Parameters
        ----------
        measurements_list: list of Measurements
            The list of Measurements objects to remove.
        session: Session, optional
            The database session to use. If not given, will create a new session.
        commit: bool
            If True, will commit the changes to the database.
            If False, will not commit the changes to the database.
            If session is not given, commit must be True.
        """
        if session is None and not commit:
            raise ValueError('If session is not given, commit must be True.')

        with SmartSession(session) as session:
            for m in measurements_list:
                m.delete_from_database(session=session, commit=False)
            if commit:
                session.commit()

# use these three functions to quickly add the "property" accessor methods
def load_attribute(object, att):
    """Load the data for a given attribute of the object. Load from Cutouts, but
    if the data needs to be loaded from disk, ONLY load the subdict that contains
    data for this object, not all objects in the Cutouts."""
    if not hasattr(object, f'_{att}'):
        raise AttributeError(f"The object {object} does not have the attribute {att}.")
    if getattr(object, f'_{att}') is None:
        if len(object.cutouts.co_dict) == 0 and object.cutouts.filepath is None:
            return None  # objects just now created and not saved cannot lazy load data!
        
        groupname = f'source_index_{object.index_in_sources}'
        if object.cutouts.co_dict[groupname] is not None:  # will check disk as Co_Dict
            object.get_data_from_cutouts()

    # after data is filled, should be able to just return it
    return getattr(object, f'_{att}')

def set_attribute(object, att, value):
    """Set the value of the attribute on the object. """
    setattr(object, f'_{att}', value)

# add "@property" functions to all the data attributes
for att in Cutouts.get_data_dict_attributes():
    setattr(
        Measurements,
        att,
        property(
            fget=lambda self, att=att: load_attribute(self, att),
            fset=lambda self, value, att=att: set_attribute(self, att, value),
        )
    )
