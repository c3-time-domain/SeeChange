import os
import base64
import hashlib

import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy.schema import CheckConstraint

from astropy.time import Time
from astropy.wcs import WCS
from astropy.io import fits
import astropy.coordinates
import astropy.units as u

from util.util import read_fits_image, save_fits_image_file

from models.base import (
    Base,
    SeeChangeBase,
    SmartSession,
    AutoIDMixin,
    FileOnDiskMixin,
    SpatiallyIndexed,
    FourCorners,
    HasBitFlagBadness,
)
from models.exposure import Exposure
from models.instrument import get_instrument_instance
from models.enums_and_bitflags import (
    ImageFormatConverter,
    ImageTypeConverter,
    string_to_bitflag,
    bitflag_to_string,
    image_preprocessing_dict,
    image_preprocessing_inverse,
    image_badness_inverse,
)

import util.config as config

from improc.tools import sigma_clipping


# links many-to-many Image to all the Images used to create it
image_upstreams_association_table = sa.Table(
    'image_upstreams_association',
    Base.metadata,
    sa.Column('upstream_id',
              sa.Integer,
              sa.ForeignKey('images.id', ondelete="CASCADE", name='image_upstreams_association_upstream_id_fkey'),
              primary_key=True),
    sa.Column('downstream_id',
              sa.Integer,
              sa.ForeignKey('images.id', ondelete="CASCADE", name='image_upstreams_association_downstream_id_fkey'),
              primary_key=True),
)


class Image(Base, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed, FourCorners, HasBitFlagBadness):

    __tablename__ = 'images'

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=ImageFormatConverter.convert('fits'),
        doc="Format of the file on disk. Should be fits or hdf5. "
    )

    @hybrid_property
    def format(self):
        return ImageFormatConverter.convert(self._format)

    @format.inplace.expression
    @classmethod
    def format(cls):
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(ImageFormatConverter.dict, value=cls._format)

    @format.inplace.setter
    def format(self, value):
        self._format = ImageFormatConverter.convert(value)

    exposure_id = sa.Column(
        sa.ForeignKey('exposures.id', ondelete='SET NULL', name='images_exposure_id_fkey'),
        nullable=True,
        index=True,
        doc=(
            "ID of the exposure from which this image was derived. "
            "Only set for single-image objects."
        )
    )

    exposure = orm.relationship(
        'Exposure',
        cascade='save-update, merge, refresh-expire, expunge',
        doc=(
            "Exposure from which this image was derived. "
            "Only set for single-image objects."
        )
    )

    upstream_images = orm.relationship(
        'Image',
        secondary=image_upstreams_association_table,
        primaryjoin='images.c.id == image_upstreams_association.c.downstream_id',
        secondaryjoin='images.c.id == image_upstreams_association.c.upstream_id',
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        lazy='selectin',
        join_depth=1,  # this enables the eager load of one generation of upstreams
        order_by='images.c.mjd',  # in chronological order of exposure start time
        doc='Images used to produce a multi-image object, like a coadd or a subtraction. '
    )

    downstream_images = orm.relationship(
        'Image',
        secondary=image_upstreams_association_table,
        primaryjoin='images.c.id == image_upstreams_association.c.upstream_id',
        secondaryjoin='images.c.id == image_upstreams_association.c.downstream_id',
        overlaps="upstream_images",
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        order_by='images.c.mjd',  # in chronological order of exposure start time
        doc='Combined Images (like coadds or a subtractions) that use this image in their production. '
    )

    ref_image_id = sa.Column(
        sa.ForeignKey('images.id', ondelete="SET NULL", name='images_ref_image_id_fkey'),
        nullable=True,
        index=True,
        doc="ID of the reference image used to produce this image, in the upstream_images list. "
    )

    ref_image = orm.relationship(
        'Image',
        primaryjoin='Image.ref_image_id == Image.id',
        remote_side='Image.id',
        cascade='save-update, merge, refresh-expire, expunge',
        uselist=False,
        lazy='selectin',
        doc="Reference image used to produce this image, in the upstream_images list. "
    )

    @property
    def new_image(self):
        """Get the image that is NOT the reference image. This only works on subtractions (with ref+new upstreams)"""
        image = [im for im in self.upstream_images if im.id != self.ref_image_id]
        if len(image) == 0 or len(image) > 1:
            return None
        return image[0]

    @new_image.setter
    def new_image(self, value):
        if value is None:
            raise ValueError('Do not assign None to new_image. Simply clear the upstream_images list.')
        if not isinstance(value, Image):
            raise ValueError("The new_image must be an Image object.")
        if len(self.upstream_images) not in [1, 2]:
            raise ValueError("This only works for subtractions that have one or two upstream images.")

        if self.upstream_images[0].id == self.ref_image_id:
            if len(self.upstream_images) == 1:
                self.upstream_images.append(value)
            elif len(self.upstream_images) == 2:
                self.upstream_images[1] = value
            else:
                raise ValueError('This should not happen!')
        elif self.upstream_images[1].id == self.ref_image_id:
            self.upstream_images[0] = value
        else:
            raise ValueError(f"The ref_image_id ({self.ref_image_id}) is not in the upstream_images list.")

    @property
    def new_aligned_image(self):
        """Get the aligned image that is NOT the reference image.
        This only works on subtractions (with ref+new upstreams).
        Will lazy-calculate the aligned images, if they are missing.
        """
        image = [im for im in self.aligned_images if im.info['original_image_id'] != self.ref_image_id]
        if len(image) == 0 or len(image) > 1:
            return None
        return image[0]

    @property
    def ref_aligned_image(self):
        """Get the aligned reference image. Will lazy-calculate the aligned images, if they are missing. """
        image = [im for im in self.aligned_images if im.info['original_image_id'] == self.ref_image_id]
        if len(image) == 0:
            return None
        return image[0]

    is_sub = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc='Is this a subtraction image.'
    )

    is_coadd = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc='Is this image made by stacking multiple images.'
    )

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=ImageTypeConverter.convert('Sci'),
        index=True,
        doc=(
            "Type of image. One of: [Sci, Diff, Bias, Dark, DomeFlat, SkyFlat, TwiFlat, Warped] "
            "or any of the above types prepended with 'Com' for combined "
            "(e.g., a ComSci image is a science image combined from multiple exposures)."
            "Saved as an integer in the database, but converted to a string when read. "
        )
    )

    @hybrid_property
    def type(self):
        return ImageTypeConverter.convert(self._type)

    @type.inplace.expression
    @classmethod
    def type(cls):
        return sa.case(ImageTypeConverter.dict, value=cls._type)

    @type.inplace.setter
    def type(self, value):
        self._type = ImageTypeConverter.convert(value)

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='images_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this image. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this image. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this image. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this image. "
        )
    )

    info = sa.Column(
        JSONB,
        nullable=False,
        default={},
        doc=(
            "Additional information on the this image. "
            "Only keep a subset of the header keywords, "
            "and re-key them to be more consistent. "
        )
    )

    mjd = sa.Column(
        sa.Double,
        nullable=False,
        index=True,
        doc=(
            "Modified Julian date of the exposure (MJD=JD-2400000.5). "
            "Multi-exposure images will have the MJD of the first exposure."
        )
    )

    @property
    def observation_time(self):
        """Translation of the MJD column to datetime object."""
        if self.mjd is None:
            return None
        else:
            return Time(self.mjd, format='mjd').datetime

    @property
    def start_mjd(self):
        """Time of the beginning of the exposure, or set of exposures (equal to mjd). """
        return self.mjd

    @property
    def mid_mjd(self):
        """Time of the middle of the exposures.
        For multiple, coadded exposures (e.g., references), this would
        be the middle between the start_mjd and end_mjd, regarless of
        how the exposures are spaced.
        """
        if self.start_mjd is None or self.end_mjd is None:
            return None
        return (self.start_mjd + self.end_mjd) / 2.0

    end_mjd = sa.Column(
        sa.Double,
        nullable=False,
        index=True,
        doc=(
            "Modified Julian date of the end of the exposure. "
            "Multi-image object will have the end_mjd of the last exposure."
        )
    )

    exp_time = sa.Column(
        sa.REAL,
        nullable=False,
        index=True,
        doc="Exposure time in seconds. Multi-exposure images will have the total exposure time."
    )

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Name of the instrument used to create this image. "
    )

    telescope = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Name of the telescope used to create this image. "
    )

    filter = sa.Column(sa.Text, nullable=True, index=True, doc="Name of the filter used to make this image. ")

    section_id = sa.Column(
        sa.Text,
        nullable=True,
        index=True,
        doc='Section ID of the image, possibly inside a larger mosiaced exposure. '
    )

    project = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the project (could also be a proposal ID). '
    )

    target = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the target object or field id. '
    )

    preproc_bitflag = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=0,
        index=False,
        doc='Bitflag specifying which preprocessing steps have been completed for the image.'
    )

    @property
    def preprocessing_done(self):
        """Return a list of the names of preprocessing steps that have been completed for this image."""
        return bitflag_to_string(self.preproc_bitflag, image_preprocessing_dict)

    @preprocessing_done.setter
    def preprocessing_done(self, value):
        self.preproc_bitflag = string_to_bitflag(value, image_preprocessing_inverse)

    astro_cal_done = sa.Column(
        sa.BOOLEAN,
        nullable=False,
        default=False,
        index=False,
        doc=( 'Has a WCS been solved for this image.  This should be set to true after astro_cal '
              'has been run, or for images (like subtractions) that are derived from other images '
              'with complete WCSes that can be copied.  This does not promise that the "latest and '
              'greatest" astrometric calibration is what is in the image header, only that there is '
              'one from the pipeline that should be good for visual identification of positions.' )
    )

    sky_sub_done = sa.Column(
        sa.BOOLEAN,
        nullable=False,
        default=False,
        index=False,
        doc='Has the sky been subtracted from this image. '
    )

    fwhm_estimate = sa.Column(
        sa.REAL,
        nullable=True,
        index=True,
        doc=(
            'FWHM estimate for the image, in arcseconds, '
            'from the first time the image was processed.'
        )
    )

    zero_point_estimate = sa.Column(
        sa.REAL,
        nullable=True,
        index=True,
        doc=(
            'Zero point estimate for the image, '
            'from the first time the image was processed.'
        )
    )

    lim_mag_estimate = sa.Column(
        sa.REAL,
        nullable=True,
        index=True,
        doc=(
            'Limiting magnitude estimate for the image, '
            'from the first time the image was processed.'
        )
    )

    bkg_mean_estimate = sa.Column(
        sa.REAL,
        nullable=True,
        index=True,
        doc=(
            'Background estimate for the image, '
            'from the first time the image was processed.'
        )
    )

    bkg_rms_estimate = sa.Column(
        sa.REAL,
        nullable=True,
        index=True,
        doc=(
            'Background RMS residual (i.e. sky noise ) '
            'estimate for the image, from the first time '
            'the image was processed.'
        )
    )

    __table_args__ = (
        CheckConstraint(
            sqltext='NOT(md5sum IS NULL AND md5sum_extensions IS NULL)',
            name='md5sum_or_md5sum_extensions_check'
        ),
    )

    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object"""
        return image_badness_inverse

    saved_extensions = [
        'data',
        'flags',
        'weight',
        'score',  # the matched-filter score of the image (e.g., from ZOGY)
        'psfflux',  # the PSF-fitted equivalent flux of the image (e.g., from ZOGY)
        'psffluxerr', # the error in the PSF-fitted equivalent flux of the image (e.g., from ZOGY)
    ]

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        HasBitFlagBadness.__init__(self)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self.raw_data = None  # the raw exposure pixels (2D float or uint16 or whatever) not saved to disk!
        self._header = None  # the header data taken directly from the FITS file

        # these properties must be added to the saved_extensions list of the Image
        self._data = None  # the underlying pixel data array (2D float array)
        self._flags = None  # the bit-flag array (2D int array)
        self._weight = None  # the inverse-variance array (2D float array)
        self._score = None  # the image after filtering with the PSF and normalizing to S/N units (2D float array)
        self._psfflux = None  # the PSF-fitted equivalent flux of the image (2D float array)
        self._psffluxerr = None  # the error in the PSF-fitted equivalent flux of the image (2D float array)

        # additional data products that could go with the Image
        self.sources = None  # the sources extracted from this Image (optionally loaded)
        self.psf = None  # the point-spread-function object (optionally loaded)
        self.bg = None  # the background object (optionally loaded)
        self.wcs = None  # the WorldCoordinates object (optionally loaded)
        self.zp = None  # the zero-point object (optionally loaded)

        self._aligner = None  # an ImageAligner object (lazy loaded using the provenance parameters)
        self._aligned_images = None  # a list of Images that are aligned to one image (lazy calculated, not committed)
        self._nandata = None  # a copy of the image data, only with NaNs at each flagged point. Lazy calculated.
        self._nanscore = None  # a copy of the image score, only with NaNs at each flagged point. Lazy calculated.

        self._instrument_object = None
        self._bitflag = 0
        self.is_sub = False

        if 'header' in kwargs:
            kwargs['_header'] = kwargs.pop('header')

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.calculate_coordinates()  # galactic and ecliptic coordinates

    def __setattr__(self, key, value):
        if key == 'upstream_images':
            # make sure the upstream_images list is sorted by mjd:
            value.sort(key=lambda x: x.mjd)

        super().__setattr__(key, value)

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)
        self.raw_data = None
        self._header = None

        for att in self.saved_extensions:
            setattr(self, f'_{att}', None)

        self.sources = None
        self.psf = None
        self.bg = None
        self.wcs = None
        self.zp = None

        self._aligner = None
        self._aligned_images = None
        self._nandata = None
        self._nanscore = None

        self._instrument_object = None
        this_object_session = orm.Session.object_session(self)
        if this_object_session is not None:  # if just loaded, should usually have a session!
            self.load_upstream_products(this_object_session)

    def merge_all(self, session):
        """Merge self and all its downstream products and assign them back to self.

        This includes: sources, psf, wcs, zp.
        This will also merge relationships, such as exposure or upstream_images,
        but that happens automatically using SQLA's magic.

        Must provide a session to merge into. Need to commit at the end.

        Returns the merged image with all its products on the same session.
        """
        new_image = self.safe_merge(session=session)
        session.flush()  # make sure new_image gets an ID

        if self.sources is not None:
            self.sources.image = new_image
            self.sources.image_id = new_image.id
            self.sources.provenance_id = self.sources.provenance.id if self.sources.provenance is not None else None
            new_image.sources = self.sources.merge_all(session=session)

            if new_image.sources.wcs is not None:
                new_image.wcs = new_image.sources.wcs
                new_image.wcs.sources = new_image.sources
                new_image.wcs.sources_id = new_image.sources.id
                new_image.wcs.provenance_id = new_image.wcs.provenance.id if new_image.wcs.provenance is not None else None

            if new_image.sources.zp is not None:
                new_image.zp = new_image.sources.zp
                new_image.zp.sources = new_image.sources
                new_image.zp.sources_id = new_image.sources.id
                new_image.zp.provenance_id = new_image.zp.provenance.id if new_image.zp.provenance is not None else None

            new_image.cutouts = new_image.sources.cutouts
            new_image.measurements = new_image.sources.measurements
            new_image._aligned_images = self._aligned_images

            # if self.wcs is not None:
            #     self.wcs.sources = new_image.sources
            #     self.wcs.sources_id = new_image.sources.id
            #     self.wcs.provenance_id = self.wcs.provenance.id if self.wcs.provenance is not None else None
            #     # new_image.wcs = self.wcs.safe_merge(session=session)
            #     new_image.wcs = session.merge(self.wcs)
            #
            # if self.zp is not None:
            #     self.zp.sources = new_image.sources
            #     self.zp.sources_id = new_image.sources.id
            #     self.zp.provenance_id = self.zp.provenance.id if self.zp.provenance is not None else None
            #     # new_image.zp = self.zp.safe_merge(session=session)
            #     new_image.zp = session.merge(self.zp)

        if self.psf is not None:
            self.psf.image = new_image
            self.psf.image_id = new_image.id
            self.psf.provenance_id = self.psf.provenance.id if self.psf.provenance is not None else None
            new_image.psf = self.psf.safe_merge(session=session)

        if self.bg is not None:
            self.bg.image = new_image
            self.bg.image_id = new_image.id
            self.bg.provenance_id = self.bg.provenance.id if self.bg.provenance is not None else None
            new_image.bg = self.bg.safe_merge(session=session)

        # take care of the upstream images and their products
        try:
            upstream_list = self.upstream_images  # can use the original images, before merging into new_image
        except DetachedInstanceError as e:
            if "lazy load operation of attribute 'upstream_images' cannot proceed" in str(e):
                upstream_list = []  # can't access the upstream images, so just we have no use calling merge_all
            else:  # other errors should be treated normally
                raise e

        for i, im in enumerate(upstream_list):
            new_image.upstream_images[i] = im.merge_all(session)

        return new_image

    def set_corners_from_header_wcs( self, wcs=None, setradec=False ):
        """Update the image's four corners (and, optionally, RA/Dec) from a WCS.

        Parameters
        ----------
        wcs : astropy.wcs.WCS, default None
           The WCS to use.  If None (default), will use the WCS parsed
           from the image header.

        setradec : bool, default False
           If True, also update the image's ra and dec fields, as well
           as the things calculated from it (galactic, ecliptic
           coordinates).

        """
        if wcs is None:
            wcs = WCS( self._header )
        # Try to detect a bad WCS
        if ( wcs.axis_type_names == ['', ''] ):
            raise ValueError( "Could not find a good WCS" )

        ras = []
        decs = []
        # Note: this used to prefer raw_data; changed it to prefer
        #  data, because we believe that's what we want to prefer,
        #  but left this note here in case things go haywire.
        # data = self.raw_data if self.raw_data is not None else self.data
        data = self.data if self.data is not None else self.raw_data
        width = data.shape[1]
        height = data.shape[0]
        xs = [ 0., width-1., 0., width-1. ]
        ys = [ 0., height-1., height-1., 0. ]
        scs = wcs.pixel_to_world( xs, ys )
        if isinstance( scs[0].ra, astropy.coordinates.Longitude ):
            ras = [ i.ra.to_value() for i in scs ]
            decs = [ i.dec.to_value() for i in scs ]
        else:
            ras = [ i.ra.value_in(u.deg).value for i in scs ]
            decs = [ i.dec.value_in(u.deg).value for i in scs ]
        ras, decs = FourCorners.sort_radec( ras, decs )
        self.ra_corner_00 = ras[0]
        self.ra_corner_01 = ras[1]
        self.ra_corner_10 = ras[2]
        self.ra_corner_11 = ras[3]
        self.dec_corner_00 = decs[0]
        self.dec_corner_01 = decs[1]
        self.dec_corner_10 = decs[2]
        self.dec_corner_11 = decs[3]

        if setradec:
            sc = wcs.pixel_to_world( data.shape[1] / 2., data.shape[0] / 2. )
            self.ra = sc.ra.to(u.deg).value
            self.dec = sc.dec.to(u.deg).value
            self.gallat = sc.galactic.b.deg
            self.gallon = sc.galactic.l.deg
            self.ecllat = sc.barycentrictrueecliptic.lat.deg
            self.ecllon = sc.barycentrictrueecliptic.lon.deg


    @classmethod
    def from_exposure(cls, exposure, section_id):
        """
        Copy the raw pixel values and relevant metadata from a section of an Exposure.

        Parameters
        ----------
        exposure: Exposure
            The exposure object to copy data from.
        section_id: int or str
            The part of the sensor to use when making this image.
            For single section instruments this is usually set to 0.

        Returns
        -------
        image: Image
            The new Image object. It would not have any data variables
            except for raw_data (and all the metadata values).
            To fill out data, flags, weight, etc., the application must
            apply the proper preprocessing tools (e.g., bias, flat, etc).

        """
        if not isinstance(exposure, Exposure):
            raise ValueError(f"The exposure must be an Exposure object. Got {type(exposure)} instead.")

        new = cls()

        same_columns = [
            'type',
            'mjd',
            'end_mjd',
            'exp_time',
            'instrument',
            'telescope',
            'filter',
            'project',
            'target',
            'format',
        ]

        # copy all the columns that are the same
        for column in same_columns:
            setattr(new, column, getattr(exposure, column))

        if exposure.filter_array is not None:
            idx = exposure.instrument_object.get_section_filter_array_index(section_id)
            new.filter = exposure.filter_array[idx]

        exposure.instrument_object.check_section_id(section_id)
        new.section_id = section_id
        new.raw_data = exposure.data[section_id]
        new.instrument_object = exposure.instrument_object

        # read the header from the exposure file's individual section data
        new._header = exposure.section_headers[section_id]

        # Because we will later be writing out float data (BITPIX=-32)
        # -- or whatever the type of raw_data is -- we have to make sure
        # there aren't any vestigal BSCALE and BZERO keywords in the
        # header.
        for delkw in [ 'BSCALE', 'BZERO' ]:
            if delkw in new.header:
                del new.header[delkw]

        # numpy array axis ordering is backwards from FITS ordering
        width = new.raw_data.shape[1]
        height = new.raw_data.shape[0]

        names = ['ra', 'dec'] + new.instrument_object.get_auxiliary_exposure_header_keys()
        header_info = new.instrument_object.extract_header_info(new._header, names)
        # TODO: get the important keywords translated into the searchable header column

        # figure out the RA/Dec of each image

        # first see if this instrument has a special method of figuring out the RA/Dec
        new.ra, new.dec = new.instrument_object.get_ra_dec_for_section(exposure, section_id)

        # Assume that if there's a WCS in the header, it's more reliable than the ra/dec keywords,
        #  (and more reliable than the function call above), so try that:
        try:
            new.set_corners_from_header_wcs( setradec=True )
        except:
            # If the WCS didn't work, and there was no special instrument method, then try other things

            # try to get the RA/Dec from the section header
            if new.ra is None or new.dec is None:
                new.ra = header_info.pop('ra', None)
                new.dec = header_info.pop('dec', None)

            # if we still have nothing, just use the RA/Dec of the global exposure
            # (Ideally, new.instrument_object.get_ra_dec_for_section will
            #  have used known chip offsets, so it will never come to this.)
            if new.ra is None or new.dec is None:
                new.ra = exposure.ra
                new.dec = exposure.dec

            new.calculate_coordinates()  # galactic and ecliptic coordinates

            # Set the corners
            halfwid = new.instrument_object.pixel_scale * width / 2. / np.cos( new.dec * np.pi / 180. ) / 3600.
            halfhei = new.instrument_object.pixel_scale * height / 2. / 3600.
            ra0 = new.ra - halfwid
            ra1 = new.ra + halfwid
            dec0 = new.dec - halfhei
            dec1 = new.dec + halfhei
            new.ra_corner_00 = ra0
            new.ra_corner_01 = ra0
            new.ra_corner_10 = ra1
            new.ra_corner_11 = ra1
            new.dec_corner_00 = dec0
            new.dec_corner_01 = dec1
            new.dec_corner_10 = dec0
            new.dec_corner_11 = dec1

        new.info = header_info  # save any additional header keys into a JSONB column

        # the exposure_id will be set automatically at commit time
        # ...but we have to set it right now because other things are
        # going to check to see if exposure.id matches image.exposure.id
        new.exposure_id = exposure.id
        new.exposure = exposure


        return new

    @classmethod
    def copy_image(cls, image):
        """Make a new Image object with the same data as an existing Image object.

        This new object does not have a provenance or any relationships to other objects.
        It should be used only as a working copy, not to be saved back into the database.
        The filepath is set to None and should be manually set to a new (unique)
        value so as not to overwrite the original.
        """
        copy_attributes = cls.saved_extensions + [
            'header',
            'info',
        ]
        simple_attributes = [
            'ra',
            'dec',
            'gallon',
            'gallat',
            'ecllon',
            'ecllat',
            'mjd',
            'end_mjd',
            'exp_time',
            'instrument',
            'telescope',
            'filter',
            'section_id',
            'project',
            'target',
            'preproc_bitflag',
            'astro_cal_done',
            'sky_sub_done',
            'fwhm_estimate',
            'zero_point_estimate',
            'lim_mag_estimate',
            'bkg_mean_estimate',
            'bkg_rms_estimate',
            'ref_image_id',
            'is_coadd',
            'is_sub',
            '_bitflag',
            '_upstream_bitflag',
            '_format',
            '_type',
        ]
        new = cls()
        for att in copy_attributes:
            value = getattr(image, att)
            if value is not None:
                setattr(new, att, value.copy())

        for att in simple_attributes:
            setattr(new, att, getattr(image, att))

        for axis in ['ra', 'dec']:
            for corner in ['00', '01', '10', '11']:
                setattr(new, f'{axis}_corner_{corner}', getattr(image, f'{axis}_corner_{corner}'))

        return new

    @classmethod
    def from_images(cls, images, index=0):
        """Create a new Image object from a list of other Image objects.
        This is the first step in making a multi-image (usually a coadd).
        Do not use this to make subtractions!  Use from_ref_and_new instead.

        The output image doesn't have any data, and is created with
        nofile=True.  It is up to the calling application to fill in the
        data, flags, weight, etc. using the appropriate preprocessing tools.

        The Image objects used as inputs must have their own data products
        loaded before calling this method, so their provenances will be recorded.
        The provenance of the output object should be generated, then a call to
        output.provenance.upstreams = output.get_upstream_provenances()
        will make sure the provenance has the correct upstreams.

        After that, the data needs to be saved to file, and only then
        can the new Image be added to the database.

        Parameters
        ----------
        images: list of Image objects
            The images to combine into a new Image object.
        index: int
            The image index in the (mjd sorted) list of upstream images
            that is used to set several attributes of the output image.
            Notably this includes the RA/Dec (and corners) of the output image,
            which implies that the indexed source image should be the one that
            all other images are aligned to (when running alignment).

        Returns
        -------
        output: Image
            The new Image object. It would not have any data variables or filepath.
        """
        if len(images) < 1:
            raise ValueError("Must provide at least one image to combine.")

        # sort images by mjd:
        images = sorted(images, key=lambda x: x.mjd)

        output = Image(nofile=True)

        fail_if_not_consistent_attributes = ['filter']
        copy_if_consistent_attributes = ['section_id', 'instrument', 'telescope', 'project', 'target', 'filter']
        copy_by_index_attributes = []  # ['ra', 'dec', 'ra_corner_00', 'ra_corner_01', ...]
        for att in ['ra', 'dec']:
            copy_by_index_attributes.append(att)
            for corner in ['00', '01', '10', '11']:
                copy_by_index_attributes.append(f'{att}_corner_{corner}')

        copy_by_index_attributes += ['gallon', 'gallat', 'ecllon', 'ecllat']

        for att in fail_if_not_consistent_attributes:
            if len(set([getattr(image, att) for image in images])) > 1:
                raise ValueError(f"Cannot combine images with different {att} values: "
                                 f"{[getattr(image, att) for image in images]}")

        # only copy if attribute is consistent across upstreams, otherwise leave as None
        for att in copy_if_consistent_attributes:
            if len(set([getattr(image, att) for image in images])) == 1:
                setattr(output, att, getattr(images[0], att))

        # use the "index" to copy the attributes of that image to the output image
        for att in copy_by_index_attributes:
            setattr(output, att, getattr(images[index], att))

        # exposure time is usually added together
        output.exp_time = sum([image.exp_time for image in images])

        # start MJD and end MJD
        output.mjd = images[0].mjd  # assume sorted by start of exposures
        output.end_mjd = max([image.end_mjd for image in images])  # exposure ends are not necessarily sorted

        # TODO: what about the header? should we combine them somehow?
        output.info = images[index].info
        output.header = images[index].header

        base_type = images[index].type
        if not base_type.startswith('Com'):
            output.type = 'Com' + base_type

        output.upstream_images = images

        # mark as the reference the image used for alignment
        output.ref_image = images[index]
        output.ref_image_id = images[index].id

        output._upstream_bitflag = 0
        for im in images:
            output._upstream_bitflag |= im.bitflag

        # Note that "data" is not filled by this method, also the provenance is empty!
        return output

    @classmethod
    def from_ref_and_new(cls, ref_image, new_image):
        return cls.from_new_and_ref(new_image, ref_image)

    @classmethod
    def from_new_and_ref(cls, new_image, ref_image):
        """Create a new Image object from a reference Image object and a new Image object.
        This is the first step in making a difference image.

        The output image doesn't have any data, and is created with
        nofile=True.  It is up to the calling application to fill in the
        data, flags, weight, etc. using the appropriate preprocessing tools.

        The Image objects used as inputs must have their own data products
        loaded before calling this method, so their provenances will be recorded.
        The provenance of the output object should be generated, then a call to
        output.provenance.upstreams = output.get_upstream_provenances()
        will make sure the provenance has the correct upstreams.

        After that, the data needs to be saved to file, and only then
        can the new Image be added to the database.

        Parameters
        ----------
        new_image: Image object
            The new image to use.
        ref_image: Image object
            The reference image to use.

        Returns
        -------
        output: Image
            The new Image object. It would not have any data variables or filepath.
        """
        if ref_image is None:
            raise ValueError("Must provide a reference image.")
        if new_image is None:
            raise ValueError("Must provide a new image.")

        output = Image(nofile=True)

        # for each attribute, check the two images have the same value
        for att in ['instrument', 'telescope', 'project', 'section_id', 'filter', 'target']:
            ref_value = getattr(ref_image, att)
            new_value = getattr(new_image, att)

            if att == 'section_id':
                ref_value = str(ref_value)
                new_value = str(new_value)

            # TODO: should replace this condition with a check that RA and Dec are overlapping?
            #  in that case: what do we consider close enough? how much overlap is reasonable?
            #  another issue: what happens if the section_id is different, what would be the
            #  value for the subtracted image? can it live without a value entirely?
            #  the same goes for target. what about coadded images? can they have no section_id??
            if att in ['section_id', 'filter', 'target'] and ref_value != new_value:
                raise ValueError(
                    f"Cannot combine images with different {att} values: "
                    f"{ref_value} and {new_value}. "
                )

            # assign the values from the new image
            setattr(output, att, new_value)

        fail_if_not_consistent_attributes = ['filter']

        for att in fail_if_not_consistent_attributes:
            if getattr(ref_image, att) != getattr(new_image, att):
                raise ValueError(f"Cannot combine images with different {att} values: "
                                 f"{getattr(ref_image, att)} and {getattr(new_image, att)}")

        if ref_image.mjd < new_image.mjd:
            output.upstream_images = [ref_image, new_image]
        else:
            output.upstream_images = [new_image, ref_image]

        output.ref_image = ref_image
        output.ref_image_id = ref_image.id

        output._upstream_bitflag = 0
        output._upstream_bitflag |= ref_image.bitflag
        output._upstream_bitflag |= new_image.bitflag

        # get some more attributes from the new image
        for att in ['section_id', 'instrument', 'telescope', 'project', 'target',
                    'exp_time', 'mjd', 'end_mjd', 'info', 'header',
                    'gallon', 'gallat', 'ecllon', 'ecllat', 'ra', 'dec',
                    'ra_corner_00', 'ra_corner_01', 'ra_corner_10', 'ra_corner_11',
                    'dec_corner_00', 'dec_corner_01', 'dec_corner_10', 'dec_corner_11' ]:
            output.__setattr__(att, getattr(new_image, att))

        output.type = 'Diff'
        if new_image.type.startswith('Com'):
            output.type = 'ComDiff'

        # Note that "data" is not filled by this method, also the provenance is empty!
        return output

    def _make_aligned_images(self):
        """Align the upstream_images to one of the images pointed to by image_index.

        The parameters of the alignment must be given in the parameters attribute
        of this Image's Provenance.

        The index to which the images are aligned is given by the "to_index" key in the
        "alignment" dictionary in the parameters of the image provenance; the value can
        be "first" or "last".

        The resulting images are saved in _aligned_images, which are not saved
        to the database. Note that each aligned image is also referred to by
        a global variable under the ImageAligner.temp_images list.
        """
        from improc.alignment import ImageAligner  # avoid circular import
        if self.provenance is None or self.provenance.parameters is None:
            raise RuntimeError('Cannot align images without a Provenance with legal parameters!')
        if 'alignment' not in self.provenance.parameters:
            raise RuntimeError('Cannot align images without an "alignment" dictionary in the Provenance parameters!')

        to_index = self.provenance.parameters['alignment'].get('to_index')
        if to_index == 'first':
            alignment_target = self.upstream_images[0]
        elif to_index == 'last':
            alignment_target = self.upstream_images[-1]
        elif to_index == 'new':
            alignment_target = self.new_image  # only works for a subtraction (or a coadd with exactly 2 upstreams)
        elif to_index == 'ref':
            alignment_target = self.ref_image  # this is not recommended!
        else:
            raise RuntimeError(
                f'Got illegal value for "to_index" ({to_index}) in the Provenance parameters!'
            )

        if self._aligner is None:
            self._aligner = ImageAligner(**self.provenance.parameters['alignment'])
        else:
            self._aligner.pars.override(self.provenance.parameters['alignment'])

        # verify all products are loaded
        for im in self.upstream_images:
            if im.sources is None or im.bg is None or im.wcs is None or im.zp is None:
                raise RuntimeError('Some images are missing data products. Try running load_upstream_products().')

        aligned = []
        for i, image in enumerate(self.upstream_images):
            new_image = self._aligner.run(image, alignment_target)
            aligned.append(new_image)
            # ImageAligner.temp_images.append(new_image)  # keep track of all these images for cleanup purposes

        self._aligned_images = aligned

    def _check_aligned_images(self):
        """Check that the aligned_images loaded in this Image are consistent.

        The aligned_images must have the same provenance parameters as the Image,
        and their "original_image_id" must point to the IDs of the upstream_images.

        If they are inconsistent, they will be removed and the _aligned_images
        attribute will be set to None to be lazy filled by _make_aligned_images().
        """
        if self._aligned_images is None:
            return

        if self.provenance is None or self.provenance.parameters is None:
            raise RuntimeError('Cannot check aligned images without a Provenance with legal parameters!')
        if 'alignment' not in self.provenance.parameters:
            raise RuntimeError(
                'Cannot check aligned images without an "alignment" dictionary in the Provenance parameters!'
            )

        upstream_images_filepaths = [image.filepath for image in self.upstream_images]

        for image in self._aligned_images:
            # im_pars will contain all the default keys and any overrides from self.provenance
            im_pars = image.info.get('alignment_parameters', {})

            # if self.provenance has non-default values, or if im_pars are missing any keys, remake all of them
            for key, value in self.provenance.parameters['alignment'].items():
                if key not in im_pars or im_pars[key] != value:
                    self._aligned_images = None
                    return

            if image.info['original_image_filepath'] not in upstream_images_filepaths:
                self._aligned_images = None
                return

    def _get_alignment_target_image(self):
        """Get the image in upstream_images that is the target to which we align all other images. """
        if self.provenance is None or self.provenance.parameters is None:
            raise RuntimeError('Cannot get alignment target without a Provenance with legal parameters!')
        if 'alignment' not in self.provenance.parameters:
            raise RuntimeError(
                'Cannot get alignment target without an "alignment" dictionary in the Provenance parameters!'
            )

        to_index = self.provenance.parameters['alignment'].get('to_index')
        if to_index == 'first':
            alignment_target = self.upstream_images[0]
        elif to_index == 'last':
            alignment_target = self.upstream_images[-1]
        elif to_index == 'new':
            alignment_target = self.new_image
        elif to_index == 'ref':
            alignment_target = self.ref_image
        else:
            raise RuntimeError(
                f'Got illegal value for "to_index" ({to_index}) in the Provenance parameters!'
            )

        return alignment_target

    def coordinates_to_alignment_target(self):
        """Make sure the coordinates (RA,dec, corners and WCS) all match the alignment target image. """
        target = self._get_alignment_target_image()
        for att in ['ra', 'dec', 'wcs',
                    'ra_corner_00', 'ra_corner_01', 'ra_corner_10', 'ra_corner_11',
                    'dec_corner_00', 'dec_corner_01', 'dec_corner_10', 'dec_corner_11' ]:
            self.__setattr__(att, getattr(target, att))

    @property
    def aligned_images(self):
        """A set of images matching the upstream_images, only aligned (warped) to one of the image. """
        self._check_aligned_images()  # possibly destroy the old aligned images

        if self._aligned_images is None:
            self._make_aligned_images()

        return self._aligned_images

    @aligned_images.setter
    def aligned_images(self, value):
        self._aligned_images = value

    @property
    def instrument_object(self):
        if self.instrument is not None:
            if self._instrument_object is None or self._instrument_object.name != self.instrument:
                self._instrument_object = get_instrument_instance(self.instrument)

        return self._instrument_object

    @instrument_object.setter
    def instrument_object(self, value):
        self._instrument_object = value

    @property
    def filter_short(self):
        """The name of the filtered, shortened for display and for filenames. """
        if self.filter is None:
            return None
        return self.instrument_object.get_short_filter_name(self.filter)

    def __repr__(self):

        output = (
            f"Image(id: {self.id}, "
            f"type: {self.type}, "
            f"exp: {self.exp_time}s, "
            f"filt: {self.filter_short}, "
            f"from: {self.instrument}/{self.telescope}"
        )

        output += ")"

        return output

    def __str__(self):
        return self.__repr__()

    def invent_filepath(self):
        """Create a relative file path for the object.

        Create a file path relative to data root for the object based on its
        metadata.  This is used when saving the image to disk.  Data
        products that depend on an image and are also saved to disk
        (e.g., SourceList) will just append another string to the Image
        filename.

        Coadded or difference images (that have a list of upstream_images)
        will also be appended a "u-tag" which is just the letter u
        (for "upstreams") follwed by the first 6 characters of the
        SHA256 hash of the upstream image filepaths.  This is to make
        sure that the filepath is unique for each combination of
        upstream images.

        """
        prov_hash = inst_name = im_type = date = time = filter = ra = dec = dec_int_pm = ''
        section_id = section_id_int = ra_int = ra_int_h = ra_frac = dec_int = dec_frac = 0

        if self.provenance is not None and self.provenance.id is not None:
            prov_hash = self.provenance.id
        if self.instrument_object is not None:
            inst_name = self.instrument_object.get_short_instrument_name()
        if self.type is not None:
            im_type = self.type

        if self.mjd is not None:
            t = Time(self.mjd, format='mjd', scale='utc').datetime
            date = t.strftime('%Y%m%d')
            time = t.strftime('%H%M%S')

        if self.filter_short is not None:
            filter = self.filter_short

        if self.section_id is not None:
            section_id = str(self.section_id)
            try:
                section_id_int = int(self.section_id)
            except ValueError:
                section_id_int = 0  # TODO: maybe replace with a placeholder like 99?

        if self.ra is not None:
            ra = self.ra
            ra_int, ra_frac = str(float(ra)).split('.')
            ra_int = int(ra_int)
            ra_int_h = ra_int // 15
            ra_frac = int(ra_frac)

        if self.dec is not None:
            dec = self.dec
            dec_int, dec_frac = str(float(dec)).split('.')
            dec_int = int(dec_int)
            dec_int_pm = f'p{dec_int:02d}' if dec_int >= 0 else f'm{-dec_int:02d}'
            dec_frac = int(dec_frac)

        cfg = config.Config.get()
        default_convention = "{inst_name}_{date}_{time}_{section_id}_{filter}_{im_type}_{prov_hash:.6s}"
        name_convention = cfg.value('storage.images.name_convention', default=None)
        if name_convention is None:
            name_convention = default_convention

        filepath = name_convention.format(
            inst_name=inst_name,
            im_type=im_type,
            date=date,
            time=time,
            filter=filter,
            ra=ra,
            ra_int=ra_int,
            ra_int_h=ra_int_h,
            ra_frac=ra_frac,
            dec=dec,
            dec_int=dec_int,
            dec_int_pm=dec_int_pm,
            dec_frac=dec_frac,
            section_id=section_id,
            section_id_int=section_id_int,
            prov_hash=prov_hash,
        )

        # TODO: which elements of the naming convention are really necessary?
        #  and what is a good way to make sure the filename actually depends on them?

        try:
            if self.upstream_images is not None and len(self.upstream_images) > 0:
                utag = hashlib.sha256()
                for image in self.upstream_images:
                    if image.filepath is None:
                        raise RuntimeError('Cannot invent filepath when upstream image has no filepath!')
                    utag.update(image.filepath.encode('utf-8'))
                utag = base64.b32encode(utag.digest()).decode().lower()
                utag = '_u-' + utag[:6]
                filepath += utag
        except DetachedInstanceError:
            pass  # ignore situations where upstream_images is not loaded, it should not happen for a combined image

        return filepath

    def save(self, filename=None, only_image=False, just_update_header=True, **kwargs ):
        """Save the data (along with flags, weights, etc.) to disk.
        The format to save is determined by the config file.
        Use the filename to override the default naming convention.

        Will save the standard image extensions : image, weight, mask.
        Does not save the source list or psf or other things that have
        their own objects; those need to be saved separately.  (Also see
        pipeline.datastore.)

        Parameters
        ----------
        filename: str (optional)
            The filename to use to save the data.  If not provided, will
            use what is in self.filepath; if that is None, then the
            default naming convention willbe used.  self.filepath will
            be updated to this name

        only_image: bool, default False
            If the image is stored as multiple files (i.e. image,
            weight, and flags extensions are all stored as seperate
            files, rather than as HDUs within one file), then _only_
            write the image out.  The use case for this is for
            "first-look" headers with astrometric and photometric
            solutions; the image header gets updated in that case, but
            the weight and flags files stay the same, so they do not
            need to be updated.  You will usually want just_update_header
            to be True when only_image is True.

        just_update_header: bool, default True
            Ignored unless only_image is True and the image is stored as
            multiple files rather than as FITS extensions.  In this
            case, if just_udpate_header is True and the file already
            exists, don't write the data, just update the header.

        **kwargs: passed on to FileOnDiskMixin.save(), include:
            overwrite - bool, set to True if it's OK to overwrite exsiting files
            no_archive - bool, set to True to save only to local disk, otherwise also saves to the archive
            exists_ok, verify_md5 - complicated, see documentation on FileOnDiskMixin

        For images being saved to the database, you probably want to use
        overwrite=True, verify_md5=True, or perhaps overwrite=False,
        exists_ok=True, verify_md5=True.  For temporary images being
        saved as part of local processing, you probably want to use
        verify_md5=False and either overwrite=True (if you're modifying
        and writing the file multiple times), or overwrite=False,
        exists_ok=True (if you might call the save() method more than
        once on the same image, and you want to trust the filesystem to
        have saved it right).

        """
        if self.data is None:
            raise RuntimeError("The image data is not loaded. Cannot save.")

        if self.provenance is None:
            raise RuntimeError("The image provenance is not set. Cannot save.")

        if filename is not None:
            self.filepath = filename
        if self.filepath is None:
            self.filepath = self.invent_filepath()

        cfg = config.Config.get()
        single_file = cfg.value('storage.images.single_file', default=False)
        format = cfg.value('storage.images.format', default='fits')
        extensions = []
        files_written = {}

        if not only_image:
            # In order to ignore just_update_header if only_image is false,
            # we need to pass it as False on to save_fits_image_file
            just_update_header = False

        full_path = os.path.join(self.local_path, self.filepath)

        if format == 'fits':
            # save the imaging data
            extensions.append('.image.fits')
            imgpath = save_fits_image_file(full_path, self.data, self.header,
                                           extname='image', single_file=single_file,
                                           just_update_header=just_update_header)
            files_written['.image.fits'] = imgpath
            # TODO: we can have extensions at the end of the self.filepath (e.g., foo.fits.flags)
            #  or we can have the extension name carry the file extension (e.g., foo.flags.fits)
            #  this should be configurable and will affect how we make the self.filepath and extensions.

            # save the other extensions
            if single_file or ( not only_image ):
                for array_name in self.saved_extensions:
                    if array_name == 'data':
                        continue
                    array = getattr(self, array_name)
                    if array is not None:
                        extpath = save_fits_image_file(
                            full_path,
                            array,
                            self.header,
                            extname=array_name,
                            single_file=single_file
                        )
                        array_name = '.' + array_name
                        if not array_name.endswith('.fits'):
                            array_name += '.fits'
                        extensions.append(array_name)
                        if not single_file:
                            files_written[array_name] = extpath

            if single_file:
                files_written = files_written['.image.fits']
                if not self.filepath.endswith('.fits'):
                    self.filepath += '.fits'

        elif format == 'hdf5':
            # TODO: consider writing a more generic utility to save_image_file that handles either fits or hdf5, etc.
            raise NotImplementedError("HDF5 format is not yet supported.")
        else:
            raise ValueError(f"Unknown image format: {format}. Use 'fits' or 'hdf5'.")

        # Save the file to the archive and update the database record
        # (as well as self.filepath, self.filepath_extensions, self.md5sum, self.md5sum_extensions)
        # (From what we did above, it's already in the right place in the local filestore.)
        if single_file:
            FileOnDiskMixin.save( self, files_written, **kwargs )
        else:
            if just_update_header:
                FileOnDiskMixin.save( self, files_written['.image.fits'], '.image.fits', **kwargs )
            else:
                for ext in extensions:
                    FileOnDiskMixin.save( self, files_written[ext], ext, **kwargs )

    def load(self):
        """Load the image data from disk.
        This includes the _data property,
        but can also load the _flags, _weight,
        _background, _score, _psfflux and _psffluxerr properties.

        """

        if self.filepath is None:
            raise ValueError("The filepath is not set. Cannot load the image.")

        cfg = config.Config.get()
        single_file = cfg.value('storage.images.single_file')

        if single_file:
            filename = self.get_fullpath()
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"Could not find the image file: {filename}")
            self._data, self._header = read_fits_image(filename, ext='image', output='both')
            for att in self.saved_extensions:
                if att == 'data':
                    continue
                array = read_fits_image(filename, ext=att)
                setattr(self, f'_{att}', array)

        else:  # load each data array from a separate file
            if self.filepath_extensions is None:
                self._data, self._header = read_fits_image( self.get_fullpath(), output='both' )
            else:
                gotim = False
                gotweight = False
                gotflags = False
                for extension, filename in zip( self.filepath_extensions, self.get_fullpath(as_list=True) ):
                    if not os.path.isfile(filename):
                        raise FileNotFoundError(f"Could not find the image extension file: {filename}")
                    if extension == '.image.fits':
                        self._data, self._header = read_fits_image(filename, output='both')
                        gotim = True
                    elif extension == '.weight.fits':
                        self._weight = read_fits_image(filename, output='data')
                        gotweight = True
                    elif extension == '.flags.fits':
                        self._flags = read_fits_image(filename, output='data')
                        gotflags = True
                    else:  # other extensions like score and psfflux
                        stripped_ext = extension
                        if stripped_ext.startswith('.'):
                            stripped_ext = stripped_ext[1:]
                        if stripped_ext.endswith('.fits'):
                            stripped_ext = stripped_ext[:-5]
                        if stripped_ext in self.saved_extensions:
                            # e.g., for extension .score.fits, self._score = read_fits_image(filename, output='data')
                            setattr(self, f'_{stripped_ext}', read_fits_image(filename, output='data'))
                        else:
                            raise ValueError( f'Unknown image extension {extension}' )

                if not ( gotim and gotweight and gotflags ):
                    raise FileNotFoundError( "Failed to load at least one of image, weight, flags" )

    def free( self, free_derived_products=True, free_aligned=True, only_free=None ):
        """Free loaded image memory.  Does not delete anything from disk.

        Will wipe out any loaded image, weight, flags, background,
        score, psfflux, psffluxerr, nandata, and nanscore data, for
        purposes of saving memory.  Doesn't make sure anything is saved
        to disk, so only use this when you know you can use it.

        (This is accomplished by setting the parameters of self that
        store that data to None, and otherwise depends on the python
        garbage collector.  If there are other references to the data
        pointed to by those parameters, the memory of course won't
        actually be freed.)

        Parameters
        ----------
          free_derived_products: bool, default True
             If True, will also call free on self.sources, self.psf,
             self.bg and self.wcs.

          free_aligned: bool, default True
             Will call free() on each of the aligned images referenced
             by this image (if any).

          only_free: set or list of strings
             If you pass this string, it will not free everything, but
             only the things you specify here.  Members of the string
             can include raw_data, data, weight, flags, background, score,
             psfflux, psffluxerr, nandata, and nanscore.

        """
        allfree = set( Image.saved_extensions )
        allfree.add( "raw_data" )
        tofree = set( only_free ) if only_free is not None else allfree

        for prop in tofree:
            if prop not in allfree:
                raise RuntimeError( f"Unknown image property to free: {prop}" )
            if prop == 'raw_data':
                self.raw_data = None
            else:
                setattr( self, f'_{prop}', None )

        if free_derived_products:
            if self.sources is not None:
                self.sources.free()
            if self.psf is not None:
                self.psf.free()
            if self.bg is not None:
                self.bg.free()
            if self.wcs is not None:
                self.wcs.free()

        if free_aligned:
            if self._aligned_images is not None:
                for alim in self._aligned_images:
                    alim.free( free_derived_products=free_derived_products, only_free=only_free )

    def get_upstream_provenances(self):
        """Collect the provenances for all upstream objects.

        This does not recursively go back to the upstreams of the upstreams.
        It gets only the provenances of the immediate upstream objects.

        Provenances that are the same (have the same hash) are combined (returned only once).

        This is what would generally be put into a new provenance's upstreams list.

        Note that upstream_images must each have the other related products
        like sources, psf, wcs, etc. already loaded.
        This happens when the objects are used to produce, e.g., a coadd or
        a subtraction image, but they would not necessarily be loaded automatically from the DB.
        To load those products (assuming all were previously committed with their own provenances)
        use the load_upstream_products() method on each of the upstream images.

        IMPORTANT RESTRICTION:
        When putting images in the upstream of a combined image (coadded or subtracted),
        if there are multiple images with the same provenance, they must also have
        loaded downstream products (e.g., SourceList) that have the same provenance.
        This is used to maintain the ability of a downstream to recover its upstreams
        using the provenance (which is the definition of why we need a provenance).
        The images could still be associated with multiple different products with
        different provenances, but not have them loaded into the relevant in-memory
        attributes of the Image objects when creating the coadd.
        Images from different instruments, or a coadded reference vs. a new image,
        would naturally have different provenances, so their products could (and indeed must)
        have different provenances. But images from the same instrument with the same provenance
        should all be produced using the same code and parameters, otherwise it will be impossible
        to know which product was processed in which way.

        Returns
        -------
        list of Provenance objects:
            A list of all the provenances for the upstream objects.
        """
        output = []
        # split the images into groups based on their provenance hash
        im_prov_hashes = list(set([im.provenance.id for im in self.upstream_images]))
        for im_prov_hash in im_prov_hashes:

            im_group = [im for im in self.upstream_images if im.provenance.id == im_prov_hash]
            sources_provs = {}
            psf_provs = {}
            wcs_provs = {}
            zp_provs = {}

            for im in im_group:
                if im.sources is not None:
                    sources_provs[im.sources.provenance.id] = im.sources.provenance
                if im.psf is not None:
                    psf_provs[im.psf.provenance.id] = im.psf.provenance
                if im.wcs is not None:
                    wcs_provs[im.wcs.provenance.id] = im.wcs.provenance
                if im.zp is not None:
                    zp_provs[im.zp.provenance.id] = im.zp.provenance

            if len(sources_provs) > 1:
                raise ValueError(
                    f"Image group with provenance {im_prov_hash} "
                    "has SourceList objects with different provenances."
                )
            if len(psf_provs) > 1:
                raise ValueError(
                    f"Image group with provenance {im_prov_hash} "
                    "has PSF objects with different provenances."
                )
            if len(wcs_provs) > 1:
                raise ValueError(
                    f"Image group with provenance {im_prov_hash} "
                    "has WCS objects with different provenances."
                )
            if len(zp_provs) > 1:
                raise ValueError(
                    f"Image group with provenance {im_prov_hash} "
                    "has ZeroPoint objects with different provenances."
                )
            output += [im_group[0].provenance]
            output += list(sources_provs.values())
            output += list(psf_provs.values())
            output += list(wcs_provs.values())
            output += list(zp_provs.values())

        # because each Image group has a different prov-hash, no products from different groups
        # could ever have the same provenance (it is hashed using the upstreams) so we don't need
        # to also check for repeated provenances between groups
        return output

    def load_upstream_products(self, session=None):
        """Make sure each upstream image has its related products loaded.

        This only works after all the images and products are committed to the database,
        with provenances consistent with what is saved in this Image's provenance
        and its own upstreams.
        """
        if self.provenance is None:
            return
        prov_ids = self.provenance.upstream_ids
        # check to make sure there is any need to load
        need_to_load = False
        for im in self.upstream_images:
            if im.sources is None or im.sources.provenance_id not in prov_ids:
                need_to_load = True
                break
            if im.psf is None or im.psf.provenance_id not in prov_ids:
                need_to_load = True
                break
            if im.bg is None or im.bg.provenance_id not in prov_ids:
                need_to_load = True
                break
            if im.wcs is None or im.wcs.provenance_id not in prov_ids:
                need_to_load = True
                break
            if im.zp is None or im.zp.provenance_id not in prov_ids:
                need_to_load = True
                break

        if not need_to_load:
            return

        from models.source_list import SourceList
        from models.psf import PSF
        from models.background import Background
        from models.world_coordinates import WorldCoordinates
        from models.zero_point import ZeroPoint

        # split the images into groups based on their provenance hash
        im_prov_hashes = list(set([im.provenance.id for im in self.upstream_images]))

        with SmartSession(session) as session:
            for im_prov_hash in im_prov_hashes:
                im_group = [im for im in self.upstream_images if im.provenance.id == im_prov_hash]
                im_ids = [im.id for im in im_group]

                # get all the products for all images in this group
                sources_result = session.scalars(
                    sa.select(SourceList).where(
                        SourceList.image_id.in_(im_ids),
                        SourceList.provenance_id.in_(prov_ids),
                    )
                ).all()
                sources_ids = [s.id for s in sources_result]

                psf_results = session.scalars(
                    sa.select(PSF).where(
                        PSF.image_id.in_(im_ids),
                        PSF.provenance_id.in_(prov_ids),
                    )
                ).all()

                bg_results = session.scalars(
                    sa.select(Background).where(
                        Background.image_id.in_(im_ids),
                        Background.provenance_id.in_(prov_ids),
                    )
                ).all()

                wcs_results = session.scalars(
                    sa.select(WorldCoordinates).where(
                        WorldCoordinates.sources_id.in_(sources_ids),
                        WorldCoordinates.provenance_id.in_(prov_ids),
                    )
                ).all()

                zp_results = session.scalars(
                    sa.select(ZeroPoint).where(
                        ZeroPoint.sources_id.in_(sources_ids),
                        ZeroPoint.provenance_id.in_(prov_ids),
                    )
                ).all()

                for im in im_group:
                    sources = [s for s in sources_result if s.image_id == im.id]  # only get the sources for this image
                    if len(sources) > 1:
                        raise ValueError(
                            f"Image {im.id} has more than one SourceList matching upstream provenance."
                        )
                    elif len(sources) == 1:
                        im.sources = sources[0]

                    psfs = [p for p in psf_results if p.image_id == im.id]  # only get the psfs for this image
                    if len(psfs) > 1:
                        raise ValueError(
                            f"Image {im.id} has more than one PSF matching upstream provenance."
                        )
                    elif len(psfs) == 1:
                        im.psf = psfs[0]

                    bgs = [b for b in bg_results if b.image_id == im.id]  # only get the bgs for this image
                    if len(bgs) > 1:
                        raise ValueError(
                            f"Image {im.id} has more than one Background matching upstream provenance."
                        )
                    elif len(bgs) == 1:
                        im.bg = bgs[0]

                    if im.sources is not None:
                        wcses = [w for w in wcs_results if w.sources_id == im.sources.id]  # the wcses for this image
                        if len(wcses) > 1:
                            raise ValueError(
                                f"SourceList {im.sources.id} has more than one WCS matching upstream provenance."
                            )
                        elif len(wcses) == 1:
                            im.wcs = wcses[0]

                        zps = [z for z in zp_results if z.sources_id == im.sources.id]  # the zps for this image
                        if len(zps) > 1:
                            raise ValueError(
                                f"SourceList {im.sources.id} has more than one ZeroPoint matching upstream provenance."
                            )
                        elif len(zps) == 1:
                            im.zp = zps[0]

    def get_upstreams(self, session=None):
        """Get the upstream images and associated products that were used to make this image.
        This includes the reference/new image (for subtractions) or the set of images
        used to build a coadd.  Each image will have some products that were generated
        from it (source lists, PSFs, etc.) that also count as upstreams to this image.

        Parameters
        ----------
        session: SQLAlchemy session (optional)
            The session to use to query the database.  If not provided,
            will open a new session that automatically closes at
            the end of the function.

        Returns
        -------
        upstreams: list of Image objects
            The upstream images.
        """
        with SmartSession(session) as session:
            self.load_upstream_products(session)
            upstreams = []
            # get the exposure
            try:
                exposure = self.exposure
            except sa.orm.exc.DetachedInstanceError:
                exposure = None
            if exposure is None and self.exposure_id is not None:
                exposure = session.scalars(sa.select(Exposure).where(Exposure.id == self.exposure_id)).first()

            if exposure is not None:
                upstreams.append(exposure)

            # get the upstream images and associated products
            for im in self.upstream_images:
                upstreams.append(im)
                if im.sources is not None:
                    upstreams.append(im.sources)
                if im.psf is not None:
                    upstreams.append(im.psf)
                if im.bg is not None:
                    upstreams.append(im.bg)
                if im.wcs is not None:
                    upstreams.append(im.wcs)
                if im.zp is not None:
                    upstreams.append(im.zp)

        return upstreams

    def get_downstreams(self, session=None, siblings=False):
        """Get all the objects that were created based on this image. """
        # avoids circular import
        from models.source_list import SourceList
        from models.psf import PSF
        from models.background import Background
        from models.world_coordinates import WorldCoordinates
        from models.zero_point import ZeroPoint

        downstreams = []
        with SmartSession(session) as session:
            # get all source lists that are related to this image (regardless of provenance)
            sources = session.scalars(
                sa.select(SourceList).where(SourceList.image_id == self.id)
            ).all()
            downstreams += sources
            if self.sources is not None and self.sources not in sources:  # if not in the session, could be duplicate!
                downstreams.append(self.sources)

            # get all psfs that are related to this image (regardless of provenance)
            psfs = session.scalars(sa.select(PSF).where(PSF.image_id == self.id)).all()
            downstreams += psfs
            if self.psf is not None and self.psf not in psfs:  # if not in the session, could be duplicate!
                downstreams.append(self.psf)

            bgs = session.scalars(sa.select(Background).where(Background.image_id == self.id)).all()
            downstreams += bgs
            if self.bg is not None and self.bg not in bgs:  # if not in the session, could be duplicate!
                downstreams.append(self.bg)

            wcses = []
            zps = []
            for s in sources:
                wcses += session.scalars(
                    sa.select(WorldCoordinates).where(WorldCoordinates.sources_id == s.id)
                ).all()

                zps += session.scalars(
                    sa.select(ZeroPoint).where(ZeroPoint.sources_id == s.id)
                ).all()
            if self.wcs is not None and self.wcs not in wcses:  # if not in the session, could be duplicate!
                wcses.append(self.wcs)
            if self.zp is not None and self.zp not in zps:  # if not in the session, could be duplicate!
                zps.append(self.zp)

            downstreams += wcses
            downstreams += zps

            # now look for other images that were created based on this one
            # ref: https://docs.sqlalchemy.org/en/20/orm/join_conditions.html#self-referential-many-to-many
            images = session.scalars(
                sa.select(Image).join(
                    image_upstreams_association_table, sa.and_(
                        image_upstreams_association_table.c.upstream_id == self.id,
                        image_upstreams_association_table.c.downstream_id == Image.id,
                    )
                ).order_by(Image.mjd).distinct()
            ).all()
            downstreams += images

            return downstreams

    def get_psf(self):
        """Load the PSF object for this image.

        If it is a sub image, it will load the PSF from the new image.
        """
        if self.psf is not None:
            return self.psf
        if self.new_image is not None:
            return self.new_image.psf
        return None

    @property
    def data(self):
        """The underlying pixel data array (2D float array). """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._data

    @data.setter
    def data(self, value):
        self._nandata = None
        self._data = value

    @property
    def header(self):
        if self._header is None and self.filepath is not None:
            self.load()
        if self._header is None:
            self._header = fits.Header()
        return self._header

    @header.setter
    def header(self, value):
        if not isinstance(value, fits.Header):
            raise ValueError(f"data must be a fits.Header object. Got {type(value)} instead. ")
        self._header = value

    @property
    def flags(self):
        """The bit-flag array (2D int array). """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._flags

    @flags.setter
    def flags(self, value):
        self._nandata = None
        self._nanscore = None
        self._flags = value

    @property
    def weight(self):
        """The inverse-variance array (2D float array). """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def score(self):
        """The image after filtering with the PSF and normalizing to S/N units (2D float array). """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._score

    @score.setter
    def score(self, value):
        self._nanscore = None
        self._score = value

    @property
    def psfflux(self):
        """An array containing an estimate for the (PSF-fitted) flux of each point in the image. """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._psfflux

    @psfflux.setter
    def psfflux(self, value):
        self._psfflux = value

    @property
    def psffluxerr(self):
        """The error for each pixel of the PSF-fit flux array. """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._psffluxerr

    @psffluxerr.setter
    def psffluxerr(self, value):
        self._psffluxerr = value

    @property
    def nandata(self):
        """The image data, only masked with NaNs wherever the flag is not zero. """
        if self._nandata is None:
            self._nandata = self.data.copy()
            if self.flags is not None:
                self._nandata[self.flags != 0] = np.nan
        return self._nandata

    @nandata.setter
    def nandata(self, value):
        self._nandata = value

    @property
    def nanscore(self):
        """The image data, only masked with NaNs wherever the flag is not zero. """
        if self._nanscore is None:
            self._nanscore = self.score.copy()
            if self.flags is not None:
                self._nanscore[self.flags != 0] = np.nan
        return self._nanscore

    @nanscore.setter
    def nanscore(self, value):
        self._nanscore = value

    @property
    def data_bgsub(self):
        """The image data, after subtracting the background. If no Background object is loaded, will raise. """
        if self.bg is None:
            raise ValueError("No background is loaded for this image.")
        if self.bg.format == 'scalar':
            return self.data - self.bg.value
        else:
            return self.data - self.bg.counts

    @property
    def nandata_bgsub(self):
        """The image data, after subtracting the background and masking with NaNs wherever the flag is not zero. """
        if self.bg is None:
            raise ValueError("No background is loaded for this image.")
        if self.bg.format == 'scalar':
            return self.nandata - self.bg.value
        else:
            return self.nandata - self.bg.counts

    def show(self, **kwargs):
        """
        Display the image using the matplotlib imshow function.

        Parameters
        ----------
        **kwargs: passed on to matplotlib.pyplot.imshow()
            Additional keyword arguments to pass to imshow.
        """
        import matplotlib.pyplot as plt
        mu, sigma = sigma_clipping(self.data)
        defaults = {
            'cmap': 'gray',
            # 'origin': 'lower',
            'vmin': mu - 3 * sigma,
            'vmax': mu + 5 * sigma,
        }
        defaults.update(kwargs)
        plt.imshow(self.nandata, **defaults)


if __name__ == '__main__':
    filename = '/home/guyn/Dropbox/python/SeeChange/data/DECam_examples/c4d_221104_074232_ori.fits.fz'
    e = Exposure(filename)
    im = Image.from_exposure(e, section_id=1)

