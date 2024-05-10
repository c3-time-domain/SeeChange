import numpy as np

from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.associationproxy import association_proxy

from models.base import Base, SeeChangeBase, SmartSession, AutoIDMixin, SpatiallyIndexed
from models.cutouts import Cutouts


class Measurements(Base, AutoIDMixin, SpatiallyIndexed):

    __tablename__ = 'measurements'

    __table_args__ = (
        UniqueConstraint('cutouts_id', 'provenance_id', name='_measurements_cutouts_provenance_uc'),
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
        if self.best_aperture == -1:
            return self.mag_psf
        return self.mag_apertures[self.best_aperture]

    @property
    def magnitude_err(self):
        if self.best_aperture == -1:
            return self.mag_psf_err
        return self.mag_apertures_err[self.best_aperture]

    @property
    def lim_mag(self):
        return self.cutouts.sources.image.new_image.lim_mag_estimate  # TODO: improve this when done with issue #143

    @property
    def zp(self):
        return self.cutouts.sources.image.new_image.zp

    @property
    def fwhm_pixels(self):
        return self.cutouts.sources.image.get_psf().fwhm_pixels

    @property
    def psf(self):
        return self.cutouts.sources.image.get_psf().get_clip(x=self.cutouts.x, y=self.cutouts.y)

    @property
    def pixel_scale(self):
        return self.cutouts.sources.image.new_image.wcs.get_pixel_scale()

    @property
    def sources(self):
        if self.cutouts is None:
            return None
        return self.cutouts.sources

    @property
    def image(self):
        if self.cutouts is None or self.cutouts.sources is None:
            return None
        return self.cutouts.sources.image

    @property
    def instrument_object(self):
        if self.cutouts is None or self.cutouts.sources is None or self.cutouts.sources.image is None:
            return None
        return self.cutouts.sources.image.instrument_object

    background = sa.Column(
        sa.REAL,
        nullable=False,
        doc="Background of the measurement, from a local annulus. Given as counts per pixel. "
    )

    background_err = sa.Column(
        sa.REAL,
        nullable=False,
        doc="RMS error of the background measurement, from a local annulus. Given as counts per pixel. "
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

    disqualifier_scores = sa.Column(
        JSONB,
        nullable=False,
        default={},
        index=True,
        doc="Values that may disqualify this object, and mark it as not a real source. "
            "This includes all sorts of analytical cuts defined by the provenance parameters. "
            "The higher the score, the more likely the measurement is to be an artefact. "
    )

    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values
        self._cutouts_list_index = None  # helper (transient) attribute that helps find the right cutouts in a list

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.calculate_coordinates()

    def __repr__(self):
        return (
            f"<Measurements {self.id} "
            f"from SourceList {self.cutouts.sources_id} "
            f"(number {self.cutouts.index_in_sources}) "
            f"from Image {self.cutouts.sub_image_id} "
            f"at x,y= {self.cutouts.x}, {self.cutouts.y}>"
        )

    def __setattr__(self, key, value):
        if key in ['flux_apertures', 'flux_apertures_err', 'aper_radii']:
            value = np.array(value)

        if key == 'cutouts':
            super().__setattr__('cutouts_id', value.id)
            for att in ['ra', 'dec', 'gallon', 'gallat', 'ecllon', 'ecllat']:
                super().__setattr__(att, getattr(value, att))

        super().__setattr__(key, value)

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
        if self.cutouts is None or self.cutouts.sources is None or self.cutouts.sources.image is None:
            raise ValueError('No cutouts for this measurement, cannot recover the PSF width. ')

        mult = self.provenance.parameters['width_filter_multipliers']
        angles = np.arange(-90.0, 90.0, self.provenance.parameters['streak_filter_angle_step'])
        fwhm = self.cutouts.sources.image.get_psf().fwhm_pixels

        if number == 0:
            return f'PSF match (FWHM= 1.00 x {fwhm:.2f})'

        if number < len(mult) + 1:
            return f'PSF mismatch (FWHM= {mult[number - 1]:.2f} x {fwhm:.2f})'

        if number < len(mult) + 1 + len(angles):
            return f'Streaked (angle= {angles[number - len(mult) - 1]:.1f} deg)'

        raise ValueError('Filter number too high for the filter bank. ')

    def find_cutouts_in_list(self, cutouts_list):
        """Given a list of cutouts, find the one that matches this object. """
        # this is faster, and works without needing DB indices to be set
        if self._cutouts_list_index is not None:
            return cutouts_list[self._cutouts_list_index]

        # after loading from DB (or merging) we must use the cutouts_id to associate these
        if self.cutouts_id is not None:
            for i, cutouts in enumerate(cutouts_list):
                if cutouts.id == self.cutouts_id:
                    self._cutouts_list_index = i
                    return cutouts

        raise ValueError('Cutouts not found in the list. ')

    def passes(self):
        """check if there are disqualifiers above the threshold

        Note that if a threshold is missing or None, that disqualifier is not checked
        """
        for key, value in self.provenance.parameters['thresholds'].items():
            if value is not None and self.disqualifier_scores[key] >= value:
                return False
        return True

    def associate_object(self, session=None):
        """Find or create a new object and associate it with this measurement.

        Objects must have sufficiently close coordinates to be associated with this
        measurement (set by the provenance.parameters['association_radius'], in arcsec).

        If no Object is found, a new one is created, and its coordinates will be identical
        to those of this Measurements object.

        This should only be done for measurements that have passed all preliminary cuts,
        which mostly rules out obvious artefacts.
        """
        from models.objects import Object  # avoid circular import

        with SmartSession(session) as session:
            obj = session.scalars(sa.select(Object).where(
                Object.cone_search(
                    self.ra,
                    self.dec,
                    self.provenance.parameters['association_radius'],
                    radunit='arcsec',
                ),
                Object.is_test.is_(self.provenance.is_testing),  # keep testing sources separate
            )).first()

            if obj is None:  # no object exists, make one based on these measurements
                obj = Object(
                    ra=self.ra,
                    dec=self.dec,
                )
                obj.is_test = self.provenance.is_testing

            self.object = obj

    def get_upstreams(self, session=None):
        """Get the image that was used to make this source list. """
        with SmartSession(session) as session:
            return session.scalars(sa.select(Cutouts).where(Cutouts.id == self.cutouts_id)).all()
        
    def get_downstreams(self, session=None):
        """Get the downstreams of this Measurements"""
        return []

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

