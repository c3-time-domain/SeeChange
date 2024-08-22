import os
import base64
import hashlib
import itertools

import numpy as np

import shapely.geometry

import sqlalchemy as sa
from sqlalchemy import orm

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CheckConstraint

from astropy.time import Time
from astropy.wcs import WCS
from astropy.io import fits
import astropy.coordinates
import astropy.units as u

from util.util import read_fits_image, save_fits_image_file, parse_dateobs, listify, asUUID
from util.radec import parse_ra_hms_to_deg, parse_dec_dms_to_deg
from util.logger import SCLogger

from models.base import (
    Base,
    SeeChangeBase,
    SmartSession,
    UUIDMixin,
    FileOnDiskMixin,
    SpatiallyIndexed,
    FourCorners,
    HasBitFlagBadness,
)
from models.provenance import Provenance
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
              sqlUUID,
              sa.ForeignKey('images._id', ondelete="RESTRICT", name='image_upstreams_association_upstream_id_fkey'),
              primary_key=True),
    sa.Column('downstream_id',
              sqlUUID,
              sa.ForeignKey('images._id', ondelete="CASCADE", name='image_upstreams_association_downstream_id_fkey'),
              primary_key=True),
)


class Image(Base, UUIDMixin, FileOnDiskMixin, SpatiallyIndexed, FourCorners, HasBitFlagBadness):

    __tablename__ = 'images'

    @declared_attr
    def __table_args__( cls ):
        return (
            CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
                               '(md5sum_extensions IS NULL OR array_position(md5sum_extensions, NULL) IS NOT NULL))',
                               name=f'{cls.__tablename__}_md5sum_check' ),
            sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
        )

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( str(ImageFormatConverter.convert('fits')) ),
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
        sa.ForeignKey('exposures._id', ondelete='SET NULL', name='images_exposure_id_fkey'),
        nullable=True,
        index=True,
        doc=(
            "ID of the exposure from which this image was derived. "
            "Only set for single-image objects."
        )
    )

    ref_image_id = sa.Column(
        sa.ForeignKey('images._id', ondelete="SET NULL", name='images_ref_image_id_fkey'),
        nullable=True,
        index=True,
        doc="ID of the reference image used to produce this image, in the upstream_images list. "
    )

    @property
    def new_image_id(self):
        """Get the id of the image that is NOT the reference image. Only for subtractions (with ref+new upstreams)"""
        # TODO : this will return something if it's a coadd of two images.
        # Perhaps we should check self.is_sub, and return None if that's false?
        image = [ i for i in self.upstream_image_ids if i != self.ref_image_id ]
        if len(image) == 0 or len(image) > 1:
            return None
        return image[0]

    @property
    def upstream_image_ids( self ):
        if self._upstream_ids is None:
            with SmartSession() as session:
                them = list ( session.query( image_upstreams_association_table.c.upstream_id,
                                             Image.mjd )
                              .join( Image, Image._id == image_upstreams_association_table.c.upstream_id )
                              .filter( image_upstreams_association_table.c.downstream_id == self.id )
                              .all() )
            them.sort( key=lambda x: x[1] )
            self._upstream_ids = [ t[0] for t in them ]
        return self._upstream_ids

    @upstream_image_ids.setter
    def upstream_image_ids( self, val ):
        raise RuntimeError( "upstream_ids cannot be set directly.  Set it by creating the image with "
                            "from_images() or from_ref_and_new()" )

    is_sub = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default='false',
        index=True,
        doc='Is this a subtraction image.'
    )

    is_coadd = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default='false',
        index=True,
        doc='Is this image made by stacking multiple images.'
    )

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( str(ImageTypeConverter.convert('Sci')) ),
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
        sa.ForeignKey('provenances._id', ondelete="CASCADE", name='images_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this image. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this image. "
        )
    )

    info = sa.Column(
        JSONB,
        nullable=False,
        server_default='{}',
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
        server_default=sa.sql.elements.TextClause( '0' ),
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
        server_default='false',
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
        server_default='false',
        index=False,
        doc='Has the sky been subtracted from this image. '
    )

    airmass = sa.Column(
        sa.REAL,
        nullable=True,
        index=True,
        doc='Airmass of the observation. '
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

        self._nandata = None  # a copy of the image data, only with NaNs at each flagged point. Lazy calculated.
        self._nanscore = None  # a copy of the image score, only with NaNs at each flagged point. Lazy calculated.

        self._upstream_ids = None

        self._instrument_object = None
        self._bitflag = 0
        self.is_sub = False
        self.is_coadd = False
        self.astro_cal_done = False
        self.photo_cal_done = False

        if 'header' in kwargs:
            kwargs['_header'] = kwargs.pop('header')

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.calculate_coordinates()  # galactic and ecliptic coordinates


    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)
        self.raw_data = None
        self._header = None

        for att in self.saved_extensions:
            setattr(self, f'_{att}', None)

        self._nandata = None
        self._nanscore = None

        self._upstream_ids = None

        self._instrument_object = None

        # this_object_session = orm.Session.object_session(self)
        # if this_object_session is not None:  # if just loaded, should usually have a session!
        #     self.load_upstream_products(this_object_session)

    def insert( self, session=None ):
        """Add the Image object to the database.

        In any events, if there are no exceptions, self.id will be set upon
        return.  (As a side effect, may also load self._upstream_ids, but
        that's transparent to the user, and happens when the user accesses
        upstream_image_ids anyway.)

        This calls UUIDMixin.insert, but also will create assocations
        defined in self._upstreams (which will have been set if this
        Image was created with from_images() or from_ref_and_new()).

        Parameters
        ----------
          session: SQLAlchemy Session, default None
            Usually you do not want to pass this; it's mostly for other
            upsert etc. methods that cascade to this.

        """

        with SmartSession( session ) as sess:
            # Insert the image.  If this raises an exception (because the image already exists),
            # then we won't futz with the image_upstreams_association table.
            SeeChangeBase.insert( self, session=sess )

            if ( self._upstream_ids is not None ) and ( len(self._upstream_ids) > 0 ):
                for ui in self._upstream_ids:
                    sess.execute( sa.text( "INSERT INTO "
                                           "image_upstreams_association(upstream_id,downstream_id) "
                                               "VALUES (:them,:me)" ),
                                  { "them": ui, "me": self.id } )
                sess.commit()


    def upsert( self, session=None, load_defaults=False ):
        with SmartSession( session ) as sess:
            SeeChangeBase.upsert( self, session=sess, load_defaults=load_defaults )

            # We're just going to merrily try to set all the upstream associations and not care
            #   if we get already existing errors.  Assume that if we get one, we'll get 'em
            #   all, because somebody else has already loaded all of them.
            # (I hope that's right.  But, in reality, it's extremely unlikely that two processes
            # will be trying to upsert the same image at the same time.)

            if ( self._upstream_ids is not None ) and ( len(self._upstream_ids) > 0 ):
                try:
                    for ui in self._upstream_ids:
                        sess.execute( sa.text( "INSERT INTO "
                                               "image_upstreams_association(upstream_id,downstream_id) "
                                                   "VALUES (:them,:me)" ),
                                      { "them": ui, "me": self.id } )
                        sess.commit()
                except IntegrityError as ex:
                    if 'duplicate key value violates unique constraint "image_upstreams_association_pkey"' in str(ex):
                        sess.rollback()
                    else:
                        raise


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
        self.minra = min( ras )
        self.maxra = max( ras )
        self.dec_corner_00 = decs[0]
        self.dec_corner_01 = decs[1]
        self.dec_corner_10 = decs[2]
        self.dec_corner_11 = decs[3]
        self.mindec = min( decs )
        self.maxdec = max( decs )

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

        if exposure.id is None:
            raise RuntimeError( "Exposure id can't be none to use Image.from_exposure" )

        new = cls()

        new.exposure_id = exposure.id

        same_columns = [
            'type',
            'mjd',
            'end_mjd',
            'exp_time',
            'airmass',
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
            new.minra = ra0
            new.maxra = ra1
            new.dec_corner_00 = dec0
            new.dec_corner_01 = dec1
            new.dec_corner_10 = dec0
            new.dec_corner_11 = dec1
            new.mindec = dec0
            new.maxdec = dec1

        new.info = header_info  # save any additional header keys into a JSONB column


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
            'airmass',
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
            setattr( new, f'min{axis}', getattr( image, f'min{axis}' ) )
            setattr( new, f'max{axis}', getattr( image, f'max{axis}' ) )

        return new

    @classmethod
    def from_images(cls, images, index=0, set_is_coadd=True):
        """Create a new Image object from a list of other Image objects.

        This is the first step in making a multi-image (usually a
        coadd).  Do not use this to make subtractions!  Use
        from_ref_and_new instead.  Make sure to set the is_coadd flag of
        the returned image, as it's not set here (just in case there's
        some eventual usage other than making coadds).

        The output image doesn't have any data, and is created with
        nofile=True.  It is up to the calling application to fill in the data,
        flags, weight, etc. using the appropriate preprocessing tools.  It is
        also up to the calling application to fill in the image's provenance
        (which must include the provenances of the images that went into the
        combination as upstreams!).

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

        set_is_coadd: bool, default True
            Set the is_coadd field of the new image.  This is usually
            what you want, so that's the default.  Make this parameter
            False if for some reason you don't want the created image to
            flagged as a coadd.

        Returns
        -------
        output: Image
            The new Image object. It would not have any data variables or filepath.

        """
        if len(images) < 1:
            raise ValueError("Must provide at least one image to combine.")

        # sort images by mjd:
        images = sorted(images, key=lambda x: x.mjd)

        # Make sure input images all have ids set.  If these images were
        #   loaded from the database, then the ids will be set.  If they
        #   were created fresh, then they don't have ids yet, but
        #   accessing the id property will set them.  This does mean
        #   that the exact image objects passed need to be saved to the
        #   database when the coadded image is saved, so the ids track
        #   properly; otherwise, we'll end up with database integrity
        #   errors.  This is probably not an issue; in practical usage,
        #   most of the time we'll be coadding images from the database.
        #   When we won't is mostly going to be in tests where we don't
        #   want to save, or where we can control this.
        upstream_ids = [ i.id for i in images ]

        output = Image( nofile=True, is_coadd=set_is_coadd )

        fail_if_not_consistent_attributes = ['filter']
        copy_if_consistent_attributes = ['section_id', 'instrument', 'telescope', 'project', 'target', 'filter']
        copy_by_index_attributes = []
        for att in ['ra', 'dec']:
            copy_by_index_attributes.append(att)
            for corner in ['00', '01', '10', '11']:
                copy_by_index_attributes.append(f'{att}_corner_{corner}')
            copy_by_index_attributes.append( f'min{att}' )
            copy_by_index_attributes.append( f'max{att}' )

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

        output._upstream_ids = upstream_ids

        # mark as the reference the image used for alignment
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

        ref_image_id = ref_image.id
        new_image_id = new_image.id

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
            output._upstream_ids = [ ref_image_id, new_image_id ]
        else:
            output._upstream_ids = [ new_image_id, ref_image_id ]

        output.ref_image_id = ref_image.id

        output._upstream_bitflag = 0
        output._upstream_bitflag |= ref_image.bitflag
        output._upstream_bitflag |= new_image.bitflag

        # get some more attributes from the new image
        for att in ['section_id', 'instrument', 'telescope', 'project', 'target',
                    'exp_time', 'airmass', 'mjd', 'end_mjd', 'info', 'header',
                    'gallon', 'gallat', 'ecllon', 'ecllat', 'ra', 'dec',
                    'ra_corner_00', 'ra_corner_01', 'ra_corner_10', 'ra_corner_11',
                    'dec_corner_00', 'dec_corner_01', 'dec_corner_10', 'dec_corner_11',
                    'minra', 'maxra', 'mindec', 'maxdec' ]:
            output.__setattr__(att, getattr(new_image, att))

        output.type = 'Diff'
        if new_image.type.startswith('Com'):
            output.type = 'ComDiff'

        output.is_sub = True

        # Note that "data" is not filled by this method, also the provenance is empty!
        return output


    def set_coordinates_to_match_target( self, target ):
        """Make sure the coordinates (RA,dec, corners and WCS) all match the alignment target image. """

        for att in ['ra', 'dec',
                    'ra_corner_00', 'ra_corner_01', 'ra_corner_10', 'ra_corner_11',
                    'dec_corner_00', 'dec_corner_01', 'dec_corner_10', 'dec_corner_11',
                    'minra', 'maxra', 'mindec', 'maxdec' ]:
            self.__setattr__(att, getattr(target, att))


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
            f"from: {self.instrument}/{self.telescope} {self.section_id}, "
            f"filepath: {self.filepath}"
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

        if self.provenance_id is not None:
            prov_hash = self.provenance_id
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

        if self._upstream_ids is not None and len(self._upstream_ids) > 0:
            utag = hashlib.sha256()
            for id in self._upstream_ids:
                utag.update( str(id).encode('utf-8') )
            utag = base64.b32encode(utag.digest()).decode().lower()
            utag = '_u-' + utag[:6]
            filepath += utag
            # ignore situations where upstream_images is not loaded, it should not happen for a combined image

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

        if self.provenance_id is None:
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

    def free( self, only_free=None ):
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


    def get_upstream_provenances(self):
        """Collect the provenances for all upstream objects.

        This does not recursively go back to the upstreams of the upstreams.
        It gets only the provenances of the immediate upstream objects.

        Provenances that are the same (have the same hash) are combined (returned only once).

        This is what would generally be put into a new provenance's upstreams list.

        Returns
        -------
        list of Provenance objects:
            A list of all the provenances for the upstream objects.
        """
        upstream_objs = self.get_upstreams()
        provids = [ i.provenance_id for i in upstream_objs ]
        provs = Provenance.get_batch( provids )
        return provs


    def get_upstreams(self, only_images=False, session=None):
        """Get the upstream images and associated products that were used to make this image.

        This includes the reference/new image (for subtractions) or the set of
        images used to build a coadd.  Each image will have some products that
        were generated from it (source lists, PSFs, etc.) that also count as
        upstreams to this image.

        Not recursive.  (So, won't get the Exposure upstreams of the images
        that went into a coadd, for instance, and if by some chance you have a
        coadd of coadds (don't do that!), the images that went into the coadd
        that was coadded to produce this coadd won't be loaded.  (Got that?))

        Parameters
        ----------
        only_images: bool, default False
             If True, only get upstream images, not the other assorted data products.

        session: SQLAlchemy session (optional)
            The session to use to query the database.  If not provided,
            will open a new session that automatically closes at
            the end of the function.

        Returns
        -------
        upstreams: list of objects
            The upstream Exposure, Image, SourceList, Background, WCS,
            ZeroPoint, PSF objects that were used to create this image.  For most
            images, it will be (at most) a single Exposure.  For subtraction and
            coadd images, there could be all those other things.

        """

        # Avoid circular imports
        from models.source_list import SourceList
        from models.background import Background
        from models.psf import PSF
        from models.world_coordinates import WorldCoordinates
        from models.zero_point import ZeroPoint

        upstreams = []
        with SmartSession(session) as session:
            # Load the exposure if there is one
            if self.exposure_id is not None:
                upstreams.append( session.query( Exposure ).filter( self.exposure_id == Exposure._id ).first() )

            if ( not self.is_coadd ) and ( not self.is_sub ):
                # We're done!  That wasn't so bad.
                return upstreams

            # This *is* so bad....
            # myprov = session.query( Provenance ).filter( self.provenance_id == Provenance._id ).first()
            myprov = Provenance.get( self.provenance_id, session=session )
            upstrprov = myprov.get_upstreams()
            upstrprovids = [ i.id for i in upstrprov ]

            # Upstream images first
            upstrimages = session.query( Image ).filter( Image._id.in_( self.upstream_image_ids ) ).all()
            # Sort by mjd
            upstrimages.sort( key=lambda i: i.mjd )
            upstreams.extend( upstrimages )

            if not only_images:
                # Get all of the other falderal associated with those images
                upstrsources = ( session.query( SourceList )
                                 .filter( SourceList.image_id.in_( self.upstream_image_ids ) )
                                 .filter( SourceList.provenance_id.in_( upstrprovids ) )
                                 .all() )
                upstrsrcids = [ s.id for s in upstrsources ]

                upstrbkgs = session.query( Background ).filter( Background.sources_id.in_( upstrsrcids ) ).all()
                upstrpsfs = session.query( PSF ).filter( PSF.sources_id.in_( upstrsrcids ) ).all()
                upstrwcses = ( session.query( WorldCoordinates )
                               .filter( WorldCoordinates.sources_id.in_( upstrsrcids ) ) ).all()
                upstrzps = session.query( ZeroPoint ).filter( ZeroPoint.sources_id.in_( upstrsrcids ) ).all()

                upstreams.extend( list(upstrsources) )
                upstreams.extend( list(upstrbkgs) )
                upstreams.extend( list(upstrpsfs) )
                upstreams.extend( list(upstrwcses) )
                upstreams.extend( list(upstrzps) )

        return upstreams

    def get_downstreams(self, session=None, only_images=False, siblings=False):
        """Get all the objects that were created based on this image. """

        # avoids circular import
        from models.source_list import SourceList
        from models.psf import PSF
        from models.background import Background
        from models.world_coordinates import WorldCoordinates
        from models.zero_point import ZeroPoint

        downstreams = []
        with SmartSession(session) as session:
            if not only_images:
                # get all source lists that are related to this image (regardless of provenance)
                sources = session.scalars( sa.select(SourceList).where(SourceList.image_id == self.id) ).all()
                downstreams.extend( list(sources) )
                srcids = [ s.id for s in sources ]

                # Get the bkgs, psfs, wcses, and zps assocated with all of those sources
                bkgs = session.query( Background ).filter( Background.sources_id.in_( srcids ) ).all()
                psfs = session.query( PSF ).filter( PSF.sources_id.in_( srcids ) ).all()
                wcses = session.query( WorldCoordinates ).filter( WorldCoordinates.sources_id.in_( srcids ) ).all()
                zps = session.query( ZeroPoint ).filter( ZeroPoint.sources_id.in_( srcids ) ).all()

                downstreams.extend( list(bkgs) )
                downstreams.extend( list(psfs) )
                downstreams.extend( list(wcses) )
                downstreams.extend( list(zps) )

            # Now get all images that are downstream of this image.

            dsimgs = ( session.query( Image )
                       .join( image_upstreams_association_table,
                              image_upstreams_association_table.c.downstream_id == Image._id )
                       .filter( image_upstreams_association_table.c.upstream_id == self.id )
                      ).all()
            downstreams.extend( list(dsimgs) )

        return downstreams


    @staticmethod
    def find_images(
            ra=None,
            dec=None,
            session=None,
            **kwargs
    ):
        """Return a list of images that match criteria.

        Similar to query_images (and **kwargs is forwarded there),
        except that it returns the actual list rather than an SQLAlchemy
        thingy, and ra/dec searching works.

        Parameters
        ----------
          ra, dec: float (decimal degrees) or str (HH:MM:SS and dd:mm:ss) or None
             Search for images that contain this point.  Must either provide both
             or neither of ra and dec.

          session: Session or None

          *** See query_images for remaining parameters

        Returns
        -------
          list of Image

        """

        if ( ra is None ) != ( dec is None ):
            raise ValueError( "Must provide both or neither of ra/dec" )

        stmt = Image.query_images( ra=ra, dec=dec, **kwargs )

        with SmartSession( session ) as sess:
            images = sess.scalars( stmt ).all()

        if ( ra is not None ) and ( len(images) > 0 ):
            if isinstance( ra, str ):
                ra = parse_ra_hms_to_deg( ra )
            if isinstance( dec, str ):
                dec = parse_dec_dms_to_deg( dec )
            # We selected by minra/maxra mindec/maxdec in query_images()
            #  because there are indexes on those fields.  (We could
            #  have just done a q3c_poly_query using the corners, but
            #  alas the q3c function will use an index on the ra/dec
            #  being searched, not the polygon, so it would not have
            #  used an index and would have been very slow.)  But, if
            #  images aren't square to the sky, that will be a superset
            #  of what we want.  Crop down here.
            keptimages = []
            for img in images:
                poly = shapely.geometry.Polygon( [ ( img.ra_corner_00, img.dec_corner_00 ),
                                                   ( img.ra_corner_01, img.dec_corner_01 ),
                                                   ( img.ra_corner_11, img.dec_corner_11 ),
                                                   ( img.ra_corner_10, img.dec_corner_10 ),
                                                   ( img.ra_corner_00, img.dec_corner_00 ) ] )
                if poly.contains( shapely.geometry.Point( ra, dec ) ):
                    keptimages.append( img )
            images = keptimages

        return images


    @staticmethod
    def query_images(
            ra=None,
            dec=None,
            target=None,
            section_id=None,
            project=None,
            instrument=None,
            filter=None,
            min_mjd=None,
            max_mjd=None,
            min_dateobs=None,
            max_dateobs=None,
            min_exp_time=None,
            max_exp_time=None,
            min_seeing=None,
            max_seeing=None,
            min_lim_mag=None,
            max_lim_mag=None,
            min_airmass=None,
            max_airmass=None,
            min_background=None,
            max_background=None,
            min_zero_point=None,
            max_zero_point=None,
            order_by='latest',
            seeing_quality_factor=3.0,
            provenance_ids=None,
            type=[1, 2, 3, 4],  # TODO: is there a smarter way to only get science images?
    ):
        """Get a SQL alchemy statement object for Image objects, with some filters applied.

        This is a convenience method to get a statement object that can be further filtered.
        If no parameters are given, will happily return all images (be careful with this).

        If you want to filter by ra/dec (which is often what you want to
        do), you may want to use find_images() rather than this
        function, because a query using the result of this function will
        may return a superset of images.  For example, the following
        image (lines) will be returned even though it doesn't include
        the specified RA/dec (asterix):

                       *╱╲
                       ╱  ╲
                       ╲  ╱
                        ╲╱

        The images are sorted either by MJD or by image quality.
        Quality is defined as sum of the limiting magnitude and the seeing,
        multiplied by the negative "seeing_quality_factor" parameter:
          <quality> = <limiting_mag> - <seeing_quality_factor> * <seeing FWHM>
        This means that as the seeing FWHM is smaller, and the limiting magnitude
        is bigger (fainter) the quality is higher.
        Choose a higher seeing_quality_factor to give more weight to the seeing,
        and less weight to the limiting magnitude.

        Parameters
        ----------
        ra, dec: float (decimal degrees) or (HH:MM:SS / dd:mm:ss) or None
            If supplied, will find images that *might* contain this ra
            and dec.  The images you get back will be a susperset of
            images that actually contain this ra and dec.  For
            efficiency, the filtering is done in the
            minra/maxra/mindec/maxdec fields of the database (which have
            indexes).  If the image is not square to the sky, it's
            possible that the image doesn't actually contain the
            requested ra/dec.  If you want to be (more) sure that the
            image actually does contain the ra/dec, use
            Image.find_images() instead of query_images().

        target: str or list of strings (optional)
            Find images that have this target name (e.g., field ID or Object name).
            If given as a list, will match all the target names in the list.

        section_id: int/str or list of ints/strings (optional)
            Find images with this section ID.
            If given as a list, will match all the section IDs in the list.

        project: str or list of strings (optional)
            Find images from this project.
            If given as a list, will match all the projects in the list.

        instrument: str or list of str (optional)
            Find images taken using this instrument.
            Provide a list to match multiple instruments.

        filter: str or list of str (optional)
            Find images taken using this filter.
            Provide a list to match multiple filters.

        min_mjd: float (optional)
            Find images taken after this MJD.

        max_mjd: float (optional)
            Find images taken before this MJD.

        min_dateobs: str (optional)
            Find images taken after this date (use ISOT format or a datetime object).

        max_dateobs: str (optional)
            Find images taken before this date (use ISOT format or a datetime object).

        min_exp_time: float (optional)
            Find images with exposure time longer than this (in seconds).

        max_exp_time: float (optional)
            Find images with exposure time shorter than this (in seconds).

        min_seeing: float (optional)
            Find images with seeing FWHM larger than this (in arcsec).

        max_seeing: float (optional)
            Find images with seeing FWHM smaller than this (in arcsec).

        min_lim_mag: float (optional)
            Find images with limiting magnitude larger (fainter) than this.

        max_lim_mag: float (optional)
            Find images with limiting magnitude smaller (brighter) than this.

        min_airmass: float (optional)
            Find images with airmass larger than this.

        max_airmass: float (optional)
            Find images with airmass smaller than this.

        min_background: float (optional)
            Find images with background rms higher than this.

        max_background: float (optional)
            Find images with background rms lower than this.

        min_zero_point: float (optional)
            Find images with zero point higher than this.

        max_zero_point: float (optional)
            Find images with zero point lower than this.

        order_by: str, default None
            Sort the images by 'earliest', 'latest' or 'quality'.
            The 'earliest' and 'latest' order by MJD, in ascending/descending order, respectively.
            The 'quality' option will try to order the images by quality, as defined above,
            with the highest quality images first.  If None, no order_by clause is included.

        seeing_quality_factor: float, default 3.0
            The factor to multiply the seeing FWHM by in the quality calculation.

        provenance_ids: str or list of strings
            Find images with these provenance IDs.

        type: integer or string or list of integers or strings, default [1,2,3,4]
            List of integer converted types of images to search for.
            This defaults to [1,2,3,4] which corresponds to the
            science images, coadds and subtractions
            (see enums_and_bitflags.ImageTypeConverter for more details).
            Choose 1 to get only the regular (non-coadd, non-subtraction) images.

        Returns
        -------
        stmt: SQL alchemy select statement
            The statement to be executed to get the images.
            Do session.scalars(stmt).all() to get the images.
            Additional filtering can be done on the statement before executing it.

        """
        stmt = sa.select(Image)

        if ( ra is None ) != ( dec is None ):
            raise ValueError( "Must provide both or neither of ra/dec" )

        # Filter by position
        if ( ra is not None ):
            if isinstance( ra, str ):
                ra = parse_ra_hms_to_deg( ra )
            if isinstance( dec, str ):
                dec = parse_dec_dms_to_deg( dec )
            # Select on minra/maxra/mindex/maxdec because there are
            # indexes on those fields.  If the image isn't square to the
            # sky, it's possible that it will be included here even
            # though it doesn't actually contain ra/dec.
            stmt = stmt.where( Image.minra <= ra,
                               Image.maxra >= ra,
                               Image.mindec <= dec,
                               Image.maxdec >= dec )

        # filter by target (e.g., field ID, object name) and possibly section ID and/or project
        targets = listify(target)
        if targets is not None:
            stmt = stmt.where(Image.target.in_(targets))
        section_ids = listify(section_id)
        if section_ids is not None:
            stmt = stmt.where(Image.section_id.in_(section_ids))
        projects = listify(project)
        if projects is not None:
            stmt = stmt.where(Image.project.in_(projects))

        # filter by filter and instrument
        filters = listify(filter)
        if filters is not None:
            stmt = stmt.where(Image.filter.in_(filters))
        instruments = listify(instrument)
        if instruments is not None:
            stmt = stmt.where(Image.instrument.in_(instruments))

        # filter by MJD or dateobs
        if min_mjd is not None:
            if min_dateobs is not None:
                raise ValueError("Cannot filter by both minimal MJD and dateobs.")
            stmt = stmt.where(Image.mjd >= min_mjd)
        if max_mjd is not None:
            if max_dateobs is not None:
                raise ValueError("Cannot filter by both maximal MJD and dateobs.")
            stmt = stmt.where(Image.mjd <= max_mjd)
        if min_dateobs is not None:
            min_dateobs = parse_dateobs(min_dateobs, output='mjd')
            stmt = stmt.where(Image.mjd >= min_dateobs)
        if max_dateobs is not None:
            max_dateobs = parse_dateobs(max_dateobs, output='mjd')
            stmt = stmt.where(Image.mjd <= max_dateobs)

        # filter by exposure time
        if min_exp_time is not None:
            stmt = stmt.where(Image.exp_time >= min_exp_time)
        if max_exp_time is not None:
            stmt = stmt.where(Image.exp_time <= max_exp_time)

        # filter by seeing FWHM
        if min_seeing is not None:
            stmt = stmt.where(Image.fwhm_estimate >= min_seeing)
        if max_seeing is not None:
            stmt = stmt.where(Image.fwhm_estimate <= max_seeing)

        # filter by limiting magnitude
        if max_lim_mag is not None:
            stmt = stmt.where(Image.lim_mag_estimate <= max_lim_mag)
        if min_lim_mag is not None:
            stmt = stmt.where(Image.lim_mag_estimate >= min_lim_mag)

        # filter by airmass
        if max_airmass is not None:
            stmt = stmt.where(Image.airmass <= max_airmass)
        if min_airmass is not None:
            stmt = stmt.where(Image.airmass >= min_airmass)

        # filter by background
        if max_background is not None:
            stmt = stmt.where(Image.bkg_rms_estimate <= max_background)
        if min_background is not None:
            stmt = stmt.where(Image.bkg_rms_estimate >= min_background)

        # filter by zero point
        if max_zero_point is not None:
            stmt = stmt.where(Image.zero_point_estimate <= max_zero_point)
        if min_zero_point is not None:
            stmt = stmt.where(Image.zero_point_estimate >= min_zero_point)

        # filter by provenances
        provenance_ids = listify(provenance_ids)
        if provenance_ids is not None:
            stmt = stmt.where(Image.provenance_id.in_(provenance_ids))

        # filter by image types
        types = listify(type)
        if types is not None:
            int_types = [ImageTypeConverter.to_int(t) for t in types]
            stmt = stmt.where(Image._type.in_(int_types))

        # sort the images
        if order_by == 'earliest':
            stmt = stmt.order_by(Image.mjd)
        elif order_by == 'latest':
            stmt = stmt.order_by(sa.desc(Image.mjd))
        elif order_by == 'quality':
            stmt = stmt.order_by(
                sa.desc(Image.lim_mag_estimate - abs(seeing_quality_factor) * Image.fwhm_estimate)
            )
        elif order_by is not None:
            raise ValueError(f'Unknown order_by parameter: {order_by}. Use "earliest", "latest" or "quality".')

        return stmt


    @staticmethod
    def get_image_from_upstreams(images, prov_id=None, session=None):
        """Finds the combined image that was made from exactly the list of images (with a given provenance).

        Parameters
        ----------
           images: list of Image
             TODO: allow passing just image ids here as an alternative (since id is all we really need).

           prov_id: str

        """

        if ( prov_id is not None ) and ( isinstance( prov_id, Provenance ) ):
            prov_id = prov_id.id

        with SmartSession(session) as session:
            session.execute( sa.text( "DROP TABLE IF EXISTS temp_image_from_upstreams" ) )

            # First get a list of candidate images that are ones whose upstreams
            #   include anything in images, plus a count of how many of
            #   images are in the upstreams.
            q = ( "SELECT i._id AS imgid, COUNT(a.upstream_id) AS nmatchupstr "
                  "INTO TEMP TABLE temp_image_from_upstreams "
                  "FROM images i "
                  "INNER JOIN image_upstreams_association a ON a.downstream_id=i._id "
                  "WHERE a.upstream_id IN :imgids " )
            subdict = { 'imgids': tuple( [ i.id for i in images ] ) }

            if prov_id is not None:  # pick only those with the right provenance id
                q += "AND i.provenance_id=:provid "
                subdict[ 'provid' ] = prov_id

            q += "GROUP BY i._id "
            session.execute( sa.text( q ), subdict )

            # Now go through those images and count *all* of the upstreams.
            # The one (if any) that has len(images) in both the count of
            # matched upstreams and all upstreams is the one we're looking for.
            q = ( "SELECT imgid FROM ("
                  "  SELECT t.imgid, t.nmatchupstr, COUNT(a.upstream_id) AS nupstr "
                  "  FROM temp_image_from_upstreams t "
                  "  INNER JOIN image_upstreams_association a ON a.downstream_id=t.imgid "
                  "  GROUP BY t.imgid, t.nmatchupstr ) subq "
                  "WHERE nmatchupstr=:num AND nupstr=:num " )
            output = session.scalars( sa.text(q), { 'num': len(images) } ).all()

            if len(output) > 1:
                raise ValueError( f"More than one combined image found with provenance ID {prov_id} "
                                  f"and upstreams {images}." )
            elif len(output) == 0:
                return None
            else:
                return Image.get_by_id( output[0], session=session )


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

    # ======================================================================
    # The fields below are things that we've deprecated; these definitions
    #   are here to catch cases in the code where they're still used

    @property
    def provenance( self ):
        raise RuntimeError( "Don't use provenance, use provenance_id" )

    @provenance.setter
    def provenance( self, val ):
        raise RuntimeError( "Don't use provenance, use provenance_id" )

    @property
    def exposure( self ):
        raise RuntimeError( "Don't use exposure, use exposure_id" )

    @exposure.setter
    def exposure( self, val ):
        raise RuntimeError( "Don't use exposure, use exposure_id" )

    @property
    def upstream_images( self ):
        raise RuntimeError( "Don't use upstream_images, use get_upstreams" )

    @upstream_images.setter
    def upstream_images( self, val ):
        raise RuntimeError( "Don't use upstream_images, create image with from_images or from_ref_and_new" )

    @property
    def downstream_images( self ):
        raise RuntimeError( "Don't use downstream_images, use get_downstreams()" )

    @downstream_images.setter
    def downstream_images( self, val ):
        raise RuntimeError( "Can't set downstream images." )

    @property
    def ref_image( self ):
        raise RuntimeError( "Don't use ref_image, use ref_image_id" )

    @ref_image.setter
    def ref_image( self, val ):
        raise RuntimeError( "Don't use ref_image, use ref_image_id" )

    @property
    def new_image( self ):
        raise RuntimeError( "Don't use new_image, use new_image_id" )

    @new_image.setter
    def new_image( self, val ):
        raise RuntimeError( "Don't use new_image, use new_image_id" )

    @property
    def new_aligned_image( self ):
        raise RuntimeError( "aligned images as Image properties are deprecated" )

    @new_aligned_image.setter
    def new_aligned_image( self ):
        raise RuntimeError( "aligned images as Image properties are deprecated" )

    @property
    def ref_aligned_image( self ):
        raise RuntimeError( "aligned images as Image properties are deprecated" )

    @ref_aligned_image.setter
    def ref_aligned_image( self, val ):
        raise RuntimeError( "aligned images as Image properties are deprecated" )

    @property
    def sources( self ):
        raise RuntimeError( f"Image.sources is deprecated, don't use it" )

    @sources.setter
    def sources( self, val ):
        raise RuntimeError( f"Image.sources is deprecated, don't use it" )

    @property
    def psf( self ):
        raise RuntimeError( f"Image.psf is deprecated, don't use it" )

    @psf.setter
    def psf( self, val ):
        raise RuntimeError( f"Image.psf is deprecated, don't use it" )

    @property
    def bg( self ):
        raise RuntimeError( f"Image.bg is deprecated, don't use it" )

    @bg.setter
    def bg( self, val ):
        raise RuntimeError( f"Image.bg is deprecated, don't use it" )

    @property
    def wcs( self ):
        raise RuntimeError( f"Image.wcs is deprecated, don't use it" )

    @wcs.setter
    def wcs( self, val ):
        raise RuntimeError( f"Image.wcs is deprecated, don't use it" )

    @property
    def zp( self ):
        raise RuntimeError( f"Image.zp is deprecated, don't use it" )

    @zp.setter
    def zp( self, val ):
        raise RuntimeError( f"Image.zp is deprecated, don't use it" )

    @property
    def _aligner( self ):
        raise RuntimeError( f"Image._aligner is deprecated, don't use it" )

    @_aligner.setter
    def _aligner( self, val ):
        raise RuntimeError( f"Image._aligner is deprecated, don't use it" )

    @property
    def _aligned_images( self ):
        raise RuntimeError( f"Image._aligned_images is deprecated, don't use it" )

    @_aligned_images.setter
    def _aligned_images( self, val ):
        raise RuntimeError( f"Image._aligned_images is deprecated, don't use it" )

    @property
    def aligned_images( self ):
        raise RuntimeError( f"Image.aligned_images is deprecated, don't use it" )

    @aligned_images.setter
    def aligned_images( self, val ):
        raise RuntimeError( f"Image.aligned_images is deprecated, don't use it" )

    @property
    def get_psf( self ):
        raise RuntimeError( f"Image.get_psf is deprecated, don't use it" )

    @get_psf.setter
    def get_psf( self, val ):
        raise RuntimeError( f"Image.get_psf is deprecated, don't use it" )

    @property
    def get_wcs( self ):
        raise RuntimeError( f"Image.get_wcs is deprecated, don't use it" )

    @get_wcs.setter
    def get_wcs( self, val ):
        raise RuntimeError( f"Image.get_wcs is deprecated, don't use it" )



if __name__ == '__main__':
    SCLogger.warning( "Running image.py doesn't actually do anything." )

