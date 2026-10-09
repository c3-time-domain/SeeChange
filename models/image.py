import os
import base64
import hashlib
import textwrap
import random
import numbers

import numpy as np
from psycopg import sql

import sqlalchemy as sa
from sqlalchemy import orm

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import CheckConstraint, UniqueConstraint

from astropy.time import Time
from astropy.wcs import WCS
from astropy.io import fits

from util.util import parse_dateobs, listify, asUUID
from util.fits import read_fits_image, save_fits_image_file
from util.radec import parse_ra_hms_to_deg, parse_dec_dms_to_deg
from util.logger import SCLogger

from models.base import (
    Base,
    SeeChangeBase,
    PGDB,
    UUIDMixin,
    FileOnDiskMixin,
    SpatiallyIndexed,
    FourCorners,
    HasBitFlagBadness,
)
from models.provenance import Provenance
from models.exposure import Exposure
from models.instrument import Instrument
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


# one-to-one (sorta) link of trimmed images to their parent(s)
# Sorta want a unique constraint here, but then we'd also have to add
#   the trimmed image provenance to this table, and that's gratuitous.
#   So, trust the code not to generate duplicates.  (...scary...)
#   cf: Image.trim
image_trim_parent = sa.Table(
    'image_trim_parent',
    Base.metadata,
    sa.Column( 'image_id',
               sqlUUID,
               sa.ForeignKey('images._id', ondelete="CASCADE", name="image_trim_image_fkey" ),
               index=True,
               nullable=False,
               primary_key=True ),
    sa.Column( 'trim_xcen',
               sa.SmallInteger,
               nullable=False ),
    sa.Column( 'trim_ycen',
               sa.SmallInteger,
               nullable=False ),
    sa.Column( 'parent_image_id',
               sqlUUID,
               sa.ForeignKey('images._id', ondelete="RESTRICT", name="image_trim_parent_image_fkey" ),
               nullable=False,
               index=True ),
    sa.Column( 'parent_wcs_id',
               sqlUUID,
               sa.ForeignKey('world_coordinates._id', ondelete="RESTRICT", name="image_trim_parent_wcs_fkey" ),
               nullable=True,
               index=True )
)


# It is not mandatory to save warped images in subtractions.  If we do, this keeps track of it.
#
# Having warped_provenance_id here violates database normalizaton, since you could get the
#   same value from looking up provenance_id in the Images table with key warped_id.  However,
#   it's here so we can use it in a unique constraint.
image_warp_parent = sa.Table(
    'image_warp_parent',
    Base.metadata,
    sa.Column( 'warp_provenance_id',
               sa.String,
               sa.ForeignKey('provenances._id', ondelete="RESTRICT", name="image_warped_prov_fkey" ),
               index=True,
               nullable=False ),
    sa.Column( 'warped_id',
               sqlUUID,
               sa.ForeignKey('images._id', ondelete="CASCADE", name="image_warped_image_fkey"),
               index=True,
               nullable=False,
               primary_key=True ),
    sa.Column( 'unwarped_zp_id',
               sqlUUID,
               sa.ForeignKey('zero_points._id', ondelete="RESTRICT", name="image_warped_unwarped_zp_fkey"),
               index=True,
               nullable=False ),
    sa.Column( 'target_wcs_id',
               sqlUUID,
               sa.ForeignKey('world_coordinates._id', ondelete="RESTRICT", name="image_warped_target_wcs_fkey"),
               index=True,
               nullable=False ),
    UniqueConstraint( 'unwarped_zp_id', 'target_wcs_id', 'warp_provenance_id', name="warp_unique" )
)


class Image(Base, UUIDMixin, FileOnDiskMixin, SpatiallyIndexed, FourCorners, HasBitFlagBadness):
    __tablename__ = 'images'

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
                               '(md5sum_components IS NULL OR array_position(md5sum_components, NULL) IS NOT NULL))',
                               name=f'{cls.__tablename__}_md5sum_check' ),
            UniqueConstraint( 'provenance_id', 'exposure_id', 'section_id', name='_image_provexpsec_uniq' ),
            sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
        )

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( str(ImageFormatConverter.convert('fits')) ),
        doc="Format of the file on disk. Should be fits, fitsfz, or hdf5. "
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

    is_coadd = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default='false',
        index=True,
        doc=( 'Is this image made by stacking multiple images?  Should only be set if the images stacked '
              'are in the database.  The image should have a _type of ComSomething.  Images with a _type of '
              'ExternComSomething should not have is_coadd set.' )
    )

    coadd_alignment_target = sa.Column(
        sa.ForeignKey('images._id', ondelete="RESTRICT", name="images_coadd_alignment_target_fkey" ),
        nullable=True,
        index=True,
        doc=( "ID of the image that was the alignment target for this coadd image, if appropriate." )
    )

    def _get_coadd_component_zp_ids( self, sort=True, pgdb=None, always_load=False, missing_ok=False ):
        if ( not always_load ) and ( not isinstance(self._coadd_component_zp_ids, config.NoValue) ):
            return self._coadd_component_zp_ids

        with PGDB( pgdb ) as pgdb:
            if sort:
                # We have to join back to image in order to get the mjd for sorting
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT z._id FROM zero_points z
                    INNER JOIN image_coadd_component c ON c.zp_id=z._id
                    INNER JOIN world_coordinates w ON w._id=z.wcs_id
                    INNER JOIN source_lists s ON s._id=w.sources_id
                    INNER JOIN images i ON s.image_id=i._id
                    WHERE c.coadd_image_id={imid}
                    ORDER BY i.mjd
                    """
                ) ).format( imid=self.id )
            else:
                q = ( sql.SQL( "SELECT zp_id FROM image_coadd_component WHERE coadd_image_id={imid}" )
                      .format( imid=self.id ) )
            rows, _cols = pgdb.execute( q )
            zpids = [ asUUID(row[0]) for row in rows ]
            if self.is_coadd:
                if len( zpids ) == 0:
                    if missing_ok:
                        zpids = config.NoValue()
                    else:
                        raise ValueError( "Image is coadd but doesn't have any components in the database!" )
            else:
                if len( zpids ) > 0:
                    raise RuntimeError( "Database corruption, there are coadd components, but image is not a coadd." )
                zpids = None
            return zpids

    @property
    def coadd_component_zp_ids( self ):
        if isinstance( self._coadd_component_zp_ids, config.NoValue ):
            self._coadd_component_zp_ids = self._get_coadd_component_zp_ids( sort=True )
        return self._coadd_component_zp_ids


    # NOTE : is_sub and is_coadd are redundant with _type !
    is_sub = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default='false',
        index=True,
        doc='Is this a subtraction image.'
    )

    def _get_subtraction_components( self, pgdb=None, always_load=False, missing_ok=False ):
        if ( ( not always_load ) and ( not isinstance(self._ref_id, config.NoValue) )
             and ( not isinstance(self._new_zp_id, config.NoValue) ) ):
            return self._ref_id, self._new_zp_id

        with PGDB( pgdb ) as pgdb:
            rows, _cols = pgdb.execute( sql.SQL( "SELECT new_zp_id, ref_id FROM image_subtraction_components "
                                                 "WHERE image_id={me}" )
                                        .format( me=self.id ) )
            if len(rows) > 1:
                raise RuntimeError( f"Database corruption, more than one image_subtraction_components row for "
                                    f"sub image {self.id}" )
            if len(rows) == 0:
                if self.is_sub:
                    if missing_ok:
                        nv = config.NoValue()
                        return ( nv, nv )
                    else:
                        raise ValueError( "Image is a subtraction, but has no subtraction components in the databse." )
                else:
                    return ( None, None )
            else:
                if self.is_sub:
                    return ( rows[0][0], rows[0][1] )
                else:
                    raise ValueError( "Image is not a subtraction, but has subtraction components in the database!" )

        raise RuntimeError( "You should never get here." )

    def _load_subtraction_components( self, pgdb=None, always_load=False ):
        if always_load or isinstance( self._new_zp_id, config.NoValue ) or isinstance( self._ref_id, config.NoValue ):
            self._new_zp_id, self._ref_id = self._get_subtraction_components( pgdb=pgdb, always_load=True )

    @property
    def ref_id( self ):
        if isinstance( self._ref_id, config.NoValue ):
            self._load_subtraction_components()
        return self._ref_id

    @ref_id.setter
    def ref_id( self, val ):
        raise RuntimeError( "Don't" )

    @property
    def new_zp_id( self ):
        if isinstance( self._new_zp_id, config.NoValue ):
            self._load_subtraction_components()
        return self._new_zp_id

    @new_zp_id.setter
    def new_zp_id( self, val ):
        raise RuntimeError( "Don't" )

    @property
    def warped_ref_source_id( self ):
        if isinstance( self._warped_ref_source_id, config.NoValue() ):
            if self.is_sub:
                with PGDB() as pgdb:
                    rows = pgdb.execute( sql.SQL( "SELECT warped_ref_source_id FROM image_subtraction_components "
                                                  "WHERE image_id={imid}" )
                                         .format( imid=self.id ) )
                    if len(rows) == 0:
                        self._warped_ref_source_id = None
                    else:
                        self._warped_ref_source_id = rows[0][0]
            else:
                self._warped_ref_source_id = None
        return self._warped_ref_source_id

    @warped_ref_source_id.setter
    def warped_ref_source_id( self, val ):
        # validate?
        self._warped_ref_source_id = val

    is_trim = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default='false',
        index=False,
        doc='Is this image the result of a call to Image.trim'
    )

    def _get_trim_parent( self, pgdb=None, always_load=False, missing_ok=False ):
        if ( not always_load ) and all( not isinstance( att, config.NoValue )
                                        for att in [ '_trim_image_parent', '_trim_wcs_parent',
                                                     '_trim_xcen', '_trim_ycen' ] ):
            return ( self._trim_image_parent, self._trim_wcs_parent, self._trim_xcen, self._trim_ycen )

        with PGDB( pgdb ) as pgdb:
            rows, _cols = pgdb.execute( sql.SQL( "SELECT parent_image_id, parent_wcs_id, trim_xcen, trim_ycen "
                                                  "FROM image_trim_parent WHERE image_id={me}" )
                                        .format( me=self.id ) )
            if len(rows) > 1:
                raise RuntimeError( f"Database corruption, multiple trim parents for {self.id}" )
            if self.is_trim:
                if len(rows) == 0:
                    if missing_ok:
                        nv = config.NoValue()
                        return ( nv, nv, nv, nv )
                    else:
                        raise ValueError( "Image is trim, but doesn't have trim parent in the database." )
                return tuple( rows[0] )
            else:
                if len(rows) == 1:
                    raise ValueError( "Image is not trim, but has trim parents in the database!" )
                return ( None, None, None, None )

        raise RuntimeError( "You should ever get here." )


    def _load_trim_parent( self, pgdb=None, always_load=False ):
        if always_load or any( isinstance( att, config.NoValue )
                               for att in [ '_trim_image_parent', '_trim_wcs_parent', '_trim_xcen', '_trim_ycen' ] ):
            ( self._trim_image_parent, self._trim_wcs_parent,
              self._trim_xcen, self._trim_ycen ) = self._get_trim_parent( pgdb=pgdb, always_load=True )

    @property
    def trim_image_parent( self ):
        if isinstance( self._trim_image_parent, config.NoValue ):
            self._load_trim_parent()
        return self._trim_image_parent

    @property
    def trim_wcs_parent( self ):
        if isinstance( self._trim_image_parent, config.NoValue ):
            self._load_trim_parent()
        return self._trim_wcs_parent

    @property
    def trim_xcen( self ):
        if isinstance( self._trim_image_parent, config.NoValue ):
            self._load_trim_parent()
        return self._trim_xcen

    @property
    def trim_ycen( self ):
        if isinstance( self._trim_image_parent, config.NoValue ):
            self._load_trim_parent()
        return self._trim_ycen


    def _get_warp_parent( self, pgdb=None, always_load=False ):
        if ( not always_load ) and all( not isinstance( att, config.NoValue )
                                        for att in [ '_warp_parent_source_zp', '_warp_parent_target_wcs' ] ):
            return ( self._warp_parent_source_zp, self._warp_parent_target_wcs )

        with PGDB( pgdb ) as pgdb:
            rows, _cols = pgdb.execute( sql.SQL( "SELECT unwarped_zp_id, target_wcs_id FROM image_warp_parent "
                                                 "WHERE warped_id={me}" ).format( me=self.id ) )
            if len(rows) == 0:
                return ( None, None )
            else:
                return tuple( rows[0] )

    def _load_warp_parent( self, pgdb, always_load=False ):
        if always_load or any( isinstance( att, config.NoValue )
                               for att in [ '_warp_parent_source_zp', '_warp_parent_target_wcs' ] ):
            self._warp_parent_source_zp, self._warp_parent_target_wcs = self._get_warp_parent( pgdb=pgdb,
                                                                                               always_load=True )

    @property
    def warp_parent_source_zp( self ):
        if isinstance( self._warp_parent_source_zp, config.NoValue ):
            self._load_warp_parent()
        return self._warp_parent_source_zp

    @property
    def warp_parent_target_wcs( self ):
        if isinstance( self._warp_parent_target_wcs, config.NoValue ):
            self._load_warp_parent()
        return self._warp_parent_target_wcs

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
        doc=( "A dictionary (key:value) of additional information from the image.  "
              "Some of this may have been pulled from the image header.  This should "
              "NOT include anything that has its own columN (mjd, filter, target, "
              "instrument, etc.).  A few standard keys include: zero, dark, flat, "
              "illumination, linearity.  All of these have values that are UUIDs that "
              "point to the calibrator_files table (though they're not formal SQL foreign "
              "keys).  If that preprocessing step hasn't been done, then probably the key "
              "will not be present in the dict." )
    )

    width = sa.Column(
        sa.SmallInteger,
        nullable=False,
        doc="Width of the image"
    )

    height = sa.Column(
        sa.SmallInteger,
        nullable=False,
        doc="Height of the image"
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

    # These are all the data arrays that (might) get saved for an image
    saved_components = [
        'image',
        'flags',
        'weight',
        'score',  # the matched-filter score of the image (e.g., from ZOGY)
        'psfflux',  # the PSF-fitted equivalent flux of the image (e.g., from ZOGY)
        'psffluxerr', # the error in the PSF-fitted equivalent flux of the image (e.g., from ZOGY)
    ]

    # In the case of saving fits.fz files, the following components should be
    #   saved loselessly
    lossless_components = [ 'flags' ]

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        HasBitFlagBadness.__init__(self)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self.raw_data = None  # the raw exposure pixels (2D float or uint16 or whatever) not saved to disk!
        self._header = None  # the header data taken directly from the FITS file

        # these properties must be added to the saved_components list of the Image
        self._data = None  # the underlying pixel data array (2D float array)
        self._flags = None  # the bit-flag array (2D int array)
        self._weight = None  # the inverse-variance array (2D float array)
        self._score = None  # the image after filtering with the PSF and normalizing to S/N units (2D float array)
        self._psfflux = None  # the PSF-fitted equivalent flux of the image (2D float array)
        self._psffluxerr = None  # the error in the PSF-fitted equivalent flux of the image (2D float array)

        self._nandata = None  # a copy of the image data, only with NaNs at each flagged point. Lazy calculated.
        self._nanscore = None  # a copy of the image score, only with NaNs at each flagged point. Lazy calculated.

        self._coadd_component_zp_ids = config.NoValue()
        self._ref_id = config.NoValue()
        self._ref_image_id = config.NoValue()
        self._new_zp_id = config.NoValue()
        self._warped_ref_source_id = config.NoValue()
        self._trim_image_parent = config.NoValue()
        self._trim_wcs_parent = config.NoValue()
        self._trim_xcen = config.NoValue()
        self._trim_ycen = config.NoValue()
        self._warp_parent_source_zp = config.NoValue()
        self._warp_parent_target_wcs = config.NoValue()

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

        for att in self.saved_components:
            if att == 'image':
                self._data = None
            else:
                setattr(self, f'_{att}', None)

        self._nandata = None
        self._nanscore = None

        self._coadd_component_zp_ids = None
        self._ref_id = None
        self._ref_image_id = None
        self._new_zp_id = None

        self._instrument_object = None

        # this_object_session = orm.Session.object_session(self)
        # if this_object_session is not None:  # if just loaded, should usually have a session!
        #     self.load_upstream_products(this_object_session)

    def insert( self, pgdb=None, session=None, nocommit=False, load_defaults=False ):
        """Add the Image object to the database.

        In any events, if there are no exceptions, self.id will be set upon
        return.

        This calls UUIDMixin.insert, but also will add to the
        image_coadd_component_table if this image is a coadd, and to
        image_subtraction_components if this is a subtraction.

        Parameters
        ----------
          pgdb, session: PGDB, psycopg.Connection, psycogp.Cursor, or sqlalchemy Session, or None
            Usually you do not want to pass this; it's mostly for other
            upsert etc. methods that cascade to this.  The two things
            are synonyms; if both are given, pgdb takes precedence.

          nocommit: bool, default False
            If True, run the statements to insert records to the
            relevante tables, but don't actually commit the database.
            Do this if you want the insert to be inside a transaction
            you've started on pgdb.  It doesn't make sense to set
            nocommit=True unless you've passed something in pgdb.

        """

        with PGDB( pgdb if pgdb is not None else session ) as pgdb:
            SeeChangeBase.insert( self, pgdb=pgdb, nocommit=True, load_defaults=load_defaults )

            if self.is_coadd:
                if isinstance( self._coadd_component_zp_ids, config.NoValue ):
                    raise ValueError( "Error inserting coadd image, missing _coadd_component_zp_ids" )
                else:
                    for ui in self._coadd_component_zp_ids:
                        pgdb.execute( sql.SQL( "INSERT INTO image_coadd_component(zp_id,coadd_image_id) "
                                               "VALUES ({them},{me})"
                                              ).format( them=ui, me=self.id ) )

            if self.is_sub:
                if isinstance( self._ref_id, config.NoValue ) or isinstance( self._new_zp_id, config.NoValue ):
                    raise RuntimeError( "Error inserting sub image, missing subtraction components" )
                wrpsrc = ( None if isinstance( self._warped_ref_source_id, config.NoValue )
                           else self._warped_ref_source_id )
                pgdb.execute( sql.SQL( textwrap.dedent(
                    """\
                    INSERT INTO image_subtraction_components(image_id,new_zp_id,ref_id,warped_ref_source_id)
                    VALUES ({me},{zp},{ref},{wrpsrc})
                    """ ) ).format( me=self.id, zp=self._new_zp_id, ref=self._ref_id, wrpsrc=wrpsrc ) )

            if self.is_trim:
                if any( isinstance( getattr(self, att), config.NoValue )
                        for att in [ '_trim_image_parent', '_trim_wcs_parent', '_trim_xcen', '_trim_ycen' ] ):
                    raise ValueError( "Error inserting trim image, missing expected properties." )
                q = sql.SQL( textwrap.dedent(
                    """\
                    INSERT INTO image_trim_parent(image_id, parent_image_id, parent_wcs_id,
                                                  trim_xcen, trim_ycen )
                    VALUES ({imid},{parid},{wcsid},{xcen},{ycen})
                    """
                ) ).format( imid=self.id, parid=self._trim_image_parent, wcsid=self._trim_wcs_parent,
                            xcen=self._trim_xcen, ycen=self._trim_ycen )
                pgdb.execute_nofetch( q )

            if self.type in ( 'Warped', 'ComWarped', 'DiffWarped', 'ComDiffWarped' ):
                if any( isinstance( getattr(self, att), config.NoValue ) or ( getattr(self, att) is None )
                        for att in [ '_warp_parent_source_zp', '_warp_parent_target_wcs' ] ):
                    raise ValueError( "Error inserting warped image, missing warp parent properties." )
                q = sql.SQL( textwrap.dedent(
                    """\
                    INSERT INTO image_warp_parent(warped_id, unwarped_zp_id, target_wcs_id, warp_provenance_id)
                    VALUES( {imid}, {zpid}, {wcsid}, {prov} )
                    """
                ) ).format( imid=self.id, zpid=self._warp_parent_source_zp, wcsid=self._warp_parent_target_wcs,
                            prov=self.provenance_id )
                pgdb.execute_nofetch( q )

            if not nocommit:
                pgdb.commit()


    def upsert( self, pgdb=None, session=None, load_defaults=False, nocommit=False ):
        with PGDB( pgdb if pgdb is not None else session ) as pgdb:
            SeeChangeBase.upsert( self, pgdb=pgdb, load_defaults=load_defaults, nocommit=True )

            # For all the associated tables, if *all* the various underscore
            #   properties exist, validate them, or, if nothing is there,
            #   insert them.  If they don't all exist, just assume that the
            #   image was already loaded and those values are OK.

            if self.is_coadd:
                if self._coadd_component_zp_ids is None:
                    raise ValueError( "_coadd_component_zp_ids None for coadd image, that shouldn't happen" )
                zpids = self._get_coadd_component_zp_ids( sort=True, pgdb=pgdb, always_load=True, missing_ok=True )
                if isinstance( self._coadd_component_zp_ids, config.NoValue ):
                    if isinstance( zpids, config.NoValue ) or ( zpids is None ):
                        raise ValueError( "Failure upserting coadd image, no coadd components in database, "
                                          "and no coadd components in object." )
                    self._coadd_component_zp_ids = zpids
                else:
                    if isinstance( zpids, config.NoValue ):
                        for ui in self._coadd_component_zp_ids:
                            pgdb.execute( sql.SQL( "INSERT INTO image_coadd_component(zp_id,coadd_image_id) "
                                                   "VALUES ({them},{me})"
                                                  ).format( them=ui, me=self.id ) )
                    else:
                        if zpids != self._coadd_component_zp_ids:
                            raise ValueError( "Error upserting image: coadd components in database do not "
                                              "match what's in the object." )

            if self.is_sub:
                new_zp_id, ref_id = self._get_subtraction_components( pgdb=pgdb, always_load=True, missing_ok=True )
                if isinstance( new_zp_id, config.NoValue ):
                    if isinstance( self._new_zp_id, config.NoValue ) or isinstance( self._ref_id, config.NoValue ):
                        raise ValueError( "Error upserting image, no subtraction components in database or "
                                          "in object" )
                    pgdb.execute( sql.SQL( "INSERT INTO image_subtraction_components(image_id,new_zp_id,ref_id) "
                                           "VALUES ({me},{zp},{ref}) "
                                          ).format( me=self.id, zp=self._new_zp_id, ref=self._ref_id ) )
                else:
                    if isinstance( self._new_zp_id, config.NoValue ):
                        self._new_zp_id = new_zp_id
                    if isinstance( self._ref_id, config.NoValue ):
                        self._ref_id = ref_id
                    if ( self._ref_id != ref_id ) or ( self._new_zp_id != new_zp_id ):
                        raise ValueError( "Subtraction components in database don't match what's in object" )

            if self.is_trim:
                ( trim_image_parent, trim_wcs_parent,
                  trim_xcen, trim_ycen ) = self._get_trim_parent( pgdb=pgdb, always_load=True, missing_ok=True )
                if isinstance( trim_image_parent, config.NoValue ):
                    if any( isinstance( getattr( self, att ), config.NoValue )
                            for att in ( '_trim_image_parent', '_trim_wcs_parent', '_trim_xcen', '_trim_ycen' ) ):
                        raise ValueError( "Error upserting image, no trim components in database and "
                                          "missing trim components in object." )
                    q = sql.SQL( textwrap.dedent(
                        """\
                        INSERT INTO image_trim_parent(image_id, parent_image_id, parent_wcs_id,
                        trim_xcen, trim_ycen )
                        VALUES ({imid},{parid},{wcsid},{xcen},{ycen})
                        """
                    ) ).format( imid=self.id, parid=self._trim_image_parent,
                                wcsid=self._trim_wcs_parent, xcen=self._trim_xcen, ycen=self._trim_ycen )
                    pgdb.execute( q )
                else:
                    uhoh = False
                    for att, val in zip( [ '_trim_image_parent', '_trim_wcs_parent', '_trim_xcen', '_trim_ycen' ],
                                         [ trim_image_parent, trim_wcs_parent, trim_xcen, trim_ycen ] ):
                        if isinstance( getattr( self, att ), config.NoValue() ):
                            setattr( self, att, val )
                        elif getattr( self, att ) != val:
                            uhoh = True
                    if uhoh:
                        raise ValueError( "Error upserting image, image trim components in database do not "
                                          "match what is in object." )

            if self.type in ( 'Warped', 'ComWarped', 'DiffWarped', 'ComDiffWarped' ):
                warp_parent_source_zp, warp_parent_target_wcs = self._get_warp_parent( pgdb=pgdb, always_load=True )
                if ( warp_parent_source_zp is None ) != ( warp_parent_target_wcs is None ):
                    raise RuntimeError( "This should never happen." )
                if warp_parent_source_zp is None:
                    if any( isinstance( getattr(self, att), config.NoValue ) or ( getattr(self, att) is None )
                            for att in [ '_warp_parent_source_zp', '_warp_parent_target_wcs' ] ):
                        raise ValueError( "Error upserting image, missing warp parent info" )
                    q = sql.SQL( textwrap.dedent(
                        """\
                        INSERT INTO image_warp_parent(warped_id, unwarped_zp_id, target_wcs_id)
                        VALUES( {imid}, {zpid}, {wcsid} )
                        """
                    ) ).format( imid=self.id, zpid=self._warp_parent_source_zp, wcsid=self._warp_parent_target_wcs )
                    pgdb.execute_nofetch( q )
                else:
                    if any( getattr( self, att ) != val
                            for att, val in zip( [ '_warp_parent_source_zp', '_warp_parent_target_wcs' ],
                                                 [ warp_parent_source_zp, warp_parent_target_wcs ] ) ):
                        raise ValueError( "Error upserting image, image warp parents in database do not "
                                          "match what is in object." )

            if not nocommit:
                pgdb.commit()


    def set_corners_from_header_wcs( self, wcs=None, setradec=False, width=None, height=None ):
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

        width : int, default self.width
           Width (x-size) of the image

        height : int, default self.height
           Height (y-size) of the image

        """
        if wcs is None:
            wcs = WCS( self._header )
        # Try to detect a bad WCS
        if ( wcs.axis_type_names == ['', ''] ):
            raise ValueError( "Could not find a good WCS" )

        width = self.width if width is None else width
        height = self.height if height is None else height

        FourCorners.set_corners_from_wcs( self, wcs, width, height, setradec=setradec )


    @classmethod
    def from_exposure(cls, exposure, section_id):
        """Copy the raw pixel values and relevant metadata from a section of an Exposure.

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
            The new Image object.  Loads the basic fields and raw_data.

            If the exposure is preprocessed (defined as
            exposure.preproc_bitflags != 0), then data, weight, and
            flags will also be loaded.  In this case, data is exactly
            the same array as raw_data, and "raw_data" is probably
            misnamed.

        """
        if not isinstance(exposure, Exposure):
            raise ValueError(f"The exposure must be an Exposure object. Got {type(exposure)} instead.")

        if exposure.id is None:
            raise RuntimeError( "Exposure id can't be none to use Image.from_exposure" )


        new = cls.create()

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
        ]

        # copy all the columns that are the same
        for column in same_columns:
            setattr(new, column, getattr(exposure, column))

        new.format = config.Config.get().value( 'storage.images.format' )

        if exposure.filter_array is not None:
            idx = exposure.instrument_object.get_section_filter_array_index(section_id)
            new.filter = exposure.filter_array[idx]

        exposure.instrument_object.check_section_id(section_id)
        new.section_id = section_id
        new.instrument_object = exposure.instrument_object
        new.preproc_bitflag = exposure.preproc_bitflag
        new.raw_data = exposure.data[section_id]

        # read the header from the exposure file's individual section data
        new._header = fits.Header( exposure.section_headers[section_id], copy=True )

        # If this is a preprocessed exposure, then we should be able to get weight and flags as well
        if exposure.preproc_bitflag != 0:
            new.data = new.raw_data
            new.weight = exposure.weight[section_id]
            new.flags = exposure.instrument_object.convert_reduced_flags_to_seechange_flags(exposure.flags[section_id])

        # Because we will later be writing out float data (BITPIX=-32)
        # -- or whatever the type of raw_data is -- we have to make sure
        # there aren't any vestigal BSCALE and BZERO keywords in the
        # header.
        for delkw in [ 'BSCALE', 'BZERO' ]:
            if delkw in new.header:
                del new.header[delkw]

        # numpy array axis ordering is backwards from FITS ordering
        new.width = new.raw_data.shape[1]
        new.height = new.raw_data.shape[0]

        names = ['ra', 'dec'] + new.instrument_object.get_auxiliary_exposure_header_keys()
        header_info = new.instrument_object.extract_header_info(new._header, names)
        # TODO: get the important keywords translated into the searchable header column

        # figure out the RA/Dec of each image

        # first see if this instrument has a special method of figuring out the RA/Dec
        new.ra, new.dec = new.instrument_object.get_ra_dec_for_section_of_exposure(exposure, section_id)

        # Assume that if there's a WCS in the header, it's more reliable than the ra/dec keywords,
        #  (and more reliable than the function call above), so try that:
        try:
            new.set_corners_from_header_wcs( setradec=True )
        except Exception:
            # If the WCS didn't work, and there was no special instrument method, then try other things

            # try to get the RA/Dec from the section header
            if new.ra is None or new.dec is None:
                new.ra = header_info.pop('ra', None)
                new.dec = header_info.pop('dec', None)

            # if we still have nothing, just use the RA/Dec of the global exposure
            # (Ideally, new.instrument_object.get_ra_dec_for_section_of_exposure will
            #  have used known chip offsets, so it will never come to this.)
            if new.ra is None or new.dec is None:
                new.ra = exposure.ra
                new.dec = exposure.dec

            new.calculate_coordinates()  # galactic and ecliptic coordinates

            # Set the corners
            halfwid = new.instrument_object.pixel_scale * new.width / 2. / np.cos( new.dec * np.pi / 180. ) / 3600.
            halfhei = new.instrument_object.pixel_scale * new.height / 2. / 3600.
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
    def copy_image (cls, image, no_copy_data=False ):
        """Make a new Image object with the same data as an existing Image object.

        This new object does not have a provenance.  It should be used
        only as a working copy, not to be saved back into the database.
        The filepath is set to None and should be manually set to a new
        (unique) value so as not to overwrite the original.

        Parameters
        ----------
           image : Image
             The Image to copy

           no_copy_data : bool, default False
             Normally, the new Image has its data field (...but not its
             weight or flags field...) filled with a copy of image.data.
             If this is True, leave that field as None.

        Returns
        -------
          Image

        """
        copy_attributes = cls.saved_components + [
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
            'width',
            'height',
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
            'coadd_alignment_target',
            'is_coadd',
            'is_sub',
            'is_trim',
            '_bitflag',
            '_upstream_bitflag',
            '_format',
            '_type',
        ]
        new = cls.create()
        for att in copy_attributes:
            if att == 'image':
                if ( not no_copy_data ) and ( image.data is not None ):
                    new.data = image.data.copy()
                else:
                    new.data = None
            else:
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
    def from_image_zps(cls, zps, width=None, height=None, index=0, images=None, alignment_target=None,
                       ra_corners=None, dec_corners=None, set_is_coadd=True):
        """Create a new Image object from a list of other ZeroPoint objects

        This is the first step in making a coadd image.  You will be
        coadding a bunch of images, which are pointed to by
        zp->wcs->sourcelist->image.  (The reason we pass the zeropoints
        instead of the images is that all of image, background, wcs, and
        zeropoint are needed to actually perform a coadd.  By passing
        the zeropoints, we implicitly specify all of those things.  If
        we just tracked the images, then later we'd have to do all kinds
        of provenance upstream searching to figureout what other data
        products were used in the coadd.)

        The output image doesn't have any data, and is created with
        nofile=True.  It is up to the calling application to fill in the data,
        flags, weight, etc. using the appropriate preprocessing tools.  It is
        also up to the calling application to fill in the image's provenance
        (which must include the provenances of the zeropoints that went into the
        combination as upstreams!).

        Parameters
        ----------
        zps: list of ZeroPoint objects
            The ZeroPoints of the images to combine into a new Image object.

        width: int
            Width of the summed image.  You must set this if you aren't
            going to later set the .data property of the created Image.

        height: int
            Height of the summed image.  You must set this if you aren't
            going to later set the .data property of the created Image.

        index: int, default 0
            The image index in the (mjd sorted) list of upstream images
            that is used to set several attributes of the output image.
            If alignment_target is None, this includes coordinate
            information (ra, dec, minra, maxdec, etc.) unless ra_corners
            and dec_corners are given.

        images: list of Image or None
            A list of Image objects.  If you pass this, it must have the
            same length as zps, and the images in the list must go with
            the zps in the list.  (Otherwise, bad things will happen.)
            If you don't pass this, the function will search the
            database to get the right Image objects.

        alignment_target: Image, default None
            The Image object to which everything will be aligned.  If
            None, then the image specified by index (above) is used as
            the target.  The RA/Dec (and corners) of the output image
            will be taken from this image's header.  Use this when you want
            the alignment target of the coadded images to be an image
            that isn't one of the images you're coadding.

        ra_corners, dec_corners : array of Float or None
            If these aren't None, then totally ignore alignment_target.
            This means that the image didn't *have* an alignment target,
            but was aligned to a manually created WCS covering this
            rectangle on the sky.

        set_is_coadd: bool, default True
            Set the is_coadd field of the new image.  This is usually
            what you want, so that's the default.  Make this parameter
            False if for some reason you don't want the created image to
            flagged as a coadd.  This is almost certainly a bad idea.
            In fact, we should probably remove this parameter.

        Returns
        -------
        output: Image
            The new Image object. It would not have any data variables or filepath.

        """
        if len(zps) < 1:
            raise ValueError("Must provide at least one image to combine.")

        if ( ra_corners is not None ) or ( dec_corners is not None ):
            if ( ra_corners is None ) or ( dec_corners is None ):
                raise ValueError( "If you give ra_corners or dec_corners, you must give both" )
            if alignment_target is not None:
                raise ValueError( "Can't pass an alignment_target when you ra_corners and dec_corners" )

        if images is not None:
            if len(images) != len(zps):
                raise ValueError( "If you pass images, it must match the length of zps" )
        else:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT i.*,z._id AS zpid
                FROM images i
                INNER JOIN source_lists s ON s.image_id=i._id
                INNER JOIN world_coordinates w ON w.sources_id=s._id
                INNER JOIN zero_points z ON z.wcs_id=w._id
                WHERE z._id=ANY(ARRAY[{zpids}])
                """
            ) ) .format( zpids=sql.SQL(",").join( z.id for z in zps ) )
            with PGDB( dictcursor=True ) as conn:
                rows = conn.execute( q )
            imzpdict = {}
            for row in rows:
                zpid = row['zpid']
                del row['zpid']
                imzpdict[ asUUID(zpid) ] = Image( **row )
            images = list( imzpdict.values() )

        # sort component images by mjd:
        dexen = list( range( len(zps) ) )
        dexen = sorted( dexen, key=lambda x: images[x].mjd )
        images = [ images[i] for i in dexen ]
        zps = [ zps[i] for i in dexen ]

        # Make sure input zeropoints all have ids set.  If these images
        #   were loaded from the database, then the ids will be set.  If
        #   they were created fresh, then they don't have ids yet, but
        #   accessing the id property will set them.  This does mean
        #   that the exact zeropoint and image objects passed need to be
        #   saved to the database when the coadded image is saved, so
        #   the ids track properly; otherwise, we'll end up with
        #   database integrity errors.  This is probably not an issue;
        #   in practical usage, most of the time we'll be coadding
        #   images from the database.  When we won't is mostly going to
        #   be in tests where we don't want to save, or where we can
        #   control this.
        upstream_ids = [ z.id for z in zps ]

        output = Image( nofile=True, is_coadd=set_is_coadd, width=width, height=height )

        fail_if_not_consistent_attributes = ['filter']
        copy_if_consistent_attributes = ['section_id', 'instrument', 'telescope', 'project', 'target', 'filter']

        for att in fail_if_not_consistent_attributes:
            if len(set([getattr(image, att) for image in images])) > 1:
                raise ValueError(f"Cannot combine images with different {att} values: "
                                 f"{[getattr(image, att) for image in images]}")

        if ra_corners is not None:
            output.set_corners_minmax( ra_corners, dec_corners )
            if ( output.maxra < output.minra ):
                output.ra = ( output.maxra + output.minra - 360. ) / 2.
                output.ra = output.ra + 360. if output.ra < 0. else output.ra
            else:
                output.ra = ( output.minra + output.maxra ) / 2.
            output.dec = ( output.maxdec + output.mindec ) / 2.
            output.calculate_coordinates()
        else:
            # Copy target image position information
            if alignment_target is None:
                alignment_target = images[index]
            copy_target_attributes = ['gallon', 'gallat', 'ecllon', 'ecllat', 'target']
            for att in ['ra', 'dec']:
                copy_target_attributes.append(att)
                for corner in ['00', '01', '10', '11']:
                    copy_target_attributes.append(f'{att}_corner_{corner}')
                copy_target_attributes.append( f'min{att}' )
                copy_target_attributes.append( f'max{att}' )
            for att in copy_target_attributes:
                setattr(output, att, getattr(alignment_target, att))

        # only copy if attribute is consistent across upstreams, otherwise leave as None
        for att in copy_if_consistent_attributes:
            if len(set([getattr(image, att) for image in images])) == 1:
                setattr(output, att, getattr(images[0], att))

        # exposure time is usually added together
        output.exp_time = sum([image.exp_time for image in images])

        # start MJD and end MJD
        output.mjd = images[0].mjd
        output.end_mjd = max([image.end_mjd for image in images])  # exposure ends are not necessarily sorted
                                                                   # ...but that only matters in pathological cases

        # ...not obvious this is the right thing to do.
        # (In fact, I want to get rid of Image.inf -- see Issue #542
        output.info = images[index].info
        # TODO? : this next one is woeful.  Coordinates should be updated
        #   to come from the alignment target image, but lots of
        #   telescopes do lots of different things for coordinates, so
        #   actually figuring that out is a nightmare.  It's not clear
        #   what we should really do here.  (This header will end
        #   up getting replaced if we use the swarp coaddition method,
        #   and the other methods use an index, so probably we don't
        #   really need to worry about it.)
        output.header = images[index].header
        #
        # ...in fact, let's just set an empty header.  That way, we don't have to
        #   read image, and this will work in tests where there are Image
        #   objects without associated files.
        # output.header = fits.Header()

        output.format = config.Config.get().value( 'storage.images.format' )

        base_type = images[index].type
        if not base_type.startswith('Com'):
            output.type = 'Com' + base_type

        output._coadd_component_zp_ids = upstream_ids
        output.coadd_alignment_target = alignment_target.id if alignment_target is not None else None

        output._upstream_bitflag = 0
        for z in zps:
            output._upstream_bitflag |= z.bitflag

        # Note that "data" is not filled by this method, also the provenance is empty!
        return output

    @classmethod
    def from_ref_and_new(cls, **kwargs):
        return cls.from_new_and_ref( **kwargs )

    @classmethod
    def from_new_and_ref(cls, new_image_zp=None, ref=None, new_image=None, width=None, height=None):
        """Create a new Image object from a Reference object and a new Image object.
        This is the first step in making a difference image.

        The output image doesn't have any data, and is created with
        nofile=True.  It is up to the calling application to fill in the
        data, flags, weight, etc. using the appropriate preprocessing tools.

        Parameters
        ----------
        new_image_zp: ZeroPoint object
            The ZeroPoint of the new image to use.  (The subtraction
            will need to know all of the image, source list, background,
            psf, wcs, and zeropoint.  By specifying zeropoints, we
            implicitly specify all of those things.)

        ref_image: Reference object
            The reference to use

        new_image: Image or None
            If you pass this, then it must be the Image that goes along
            with ZeroPoint.  Normally, this function will search the
            database to find the right Image.  When searching the database,
            it will thrown out difference images and warped images -- those
            will share the same zero_points row as the parent image.  If you
            are trying to work on such an image, then you must pass new_image.

        width, height: int, default None
            You probably never want to set these, because they will
            default to the size of new_image.

        Returns
        -------
        output: Image
            The new Image object. It would not have any data variables or filepath.

        """

        # Avoid circular import
        from models.reference import Reference
        from models.zero_point import ZeroPoint

        if ref is None:
            raise ValueError("Must provide a reference image.")
        if not isinstance( ref, Reference ):
            raise TypeError( f"ref must be a Reference, not a {type(ref)}" )
        if new_image_zp is None:
            raise ValueError("Must provide a new image zeropoint.")
        if not isinstance( new_image_zp, ZeroPoint ):
            raise TypeError( f"new_image_zp must be an ZeroPoint, not a {type(new_image_zp)}" )
        if ( new_image is not None ) and ( not isinstance( new_image, Image ) ):
            raise TypeError( f"If you pass new_image, it must be an Image, not a {type(new_image)}" )

        with PGDB( dictcursor=True ) as pgdb:
            ref._load_ref_data_products( pgdb=pgdb )
            ref_image = Image.get_by_id( ref.image.id, pgdb=pgdb )
            if new_image is None:
                rows = pgdb.execute( sql.SQL( textwrap.dedent(
                    """\
                    SELECT i.* FROM images i
                    INNER JOIN source_lists s ON s.image_id=i._id
                    INNER JOIN world_coordinates w ON w.sources_id=s._id
                    INNER JOIN zero_points z ON z.wcs_id=w._id
                    WHERE z._id={zpid}
                      AND i._type NOT IN ({types})
                    """
                ) ).format( zpid=new_image_zp.id,
                            types=sql.SQL(",").join( [ ImageTypeConverter.to_int(t) for t in ( 'Diff', 'Warped' ) ] )
                           ) )
                if len(rows) == 0:
                    raise RuntimeError( f"Database corruption: Image corresponding to ZeroPoint "
                                        f"{new_image_zp} not found!" )
                elif len(rows) > 1:
                    raise RuntimeError( f"Found more than one image corresponding to ZeroPoint "
                                        f"{new_image_zp}; this should never happen." )
                new_image = Image.create( **(rows[0]) )

        output = Image( nofile=True )

        fail_if_not_consistent_attributes = ['filter']
        # Set various values based on the new image
        # Note that we're assuming that we're going to be aligning
        #   the sub image to the new image here (which is the
        #   default).
        for att in ['instrument', 'telescope', 'project', 'section_id', 'filter', 'target',
                    'exp_time', 'airmass', 'mjd', 'end_mjd', 'info', 'header', 'width', 'height',
                    'gallon', 'gallat', 'ecllon', 'ecllat', 'ra', 'dec',
                    'ra_corner_00', 'ra_corner_01', 'ra_corner_10', 'ra_corner_11',
                    'dec_corner_00', 'dec_corner_01', 'dec_corner_10', 'dec_corner_11',
                    'minra', 'maxra', 'mindec', 'maxdec' ]:
            if ( ( att in fail_if_not_consistent_attributes) and
                 ( getattr( ref_image, att ) != getattr( new_image, att ) ) ):
                raise ValueError(f"Cannot combine images with different {att} values: "
                                 f"{getattr(ref_image, att)} and {getattr(new_image, att)}")
            setattr( output, att, getattr( new_image, att ) )

        output._ref_id = ref.id
        output._ref_image_id = ref_image.id
        output._new_zp_id = new_image_zp.id

        output._upstream_bitflag = 0
        output._upstream_bitflag |= ref_image.bitflag
        output._upstream_bitflag |= new_image_zp.bitflag

        output.format = config.Config.get().value( 'storage.images.format' )
        output.type = 'Diff'
        if new_image.type.startswith('Com'):
            output.type = 'ComDiff'

        output.is_sub = True

        # Note that "data" is not filled by this method, also the provenance is empty!
        return output


    def set_corners_from_wcs( self, wcs=None, width=None, height=None, setradec=False ):
        if wcs is None:
            raise ValueError( "Must provide a wcs" )
        if ( ( ( width is not None ) and ( width != self.width ) ) or
             ( ( height is not None ) and ( height != self.height ) ) ):
            raise ValueError( f"You passed width={width} and height={height}, but this is an image "
                              f"of dimensions {self.width}×{self.height}" )
        width = self.width if width is None else width
        height = self.height if height is None else height
        super().set_corners_from_wcs( wcs, width, height, setradec=setradec )


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
                self._instrument_object = Instrument.get_instrument_instance(self.instrument)

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


    @classmethod
    def get_trim_provs( cls, width, height, wcs_prov=None, zp_prov=None,
                        upstreams=[], save=False, provtag=None, pgdb=None ):
        upstreams = upstreams.copy()

        if wcs_prov is not None:
            if not isinstance( wcs_prov, Provenance ):
                wcs_prov_obj = Provenance.get_by_id( wcs_prov, pgdb=pgdb )
                if wcs_prov_obj is None:
                    raise ValueError( f"Failed to find wcs prov {wcs_prov}" )
                wcs_prov = wcs_prov_obj
            upstreams.append( wcs_prov )
            use_wcs = True
        else:
            use_wcs = False

        trimimprov = Provenance( process='Image.trim',
                                 parameters={ 'width': width, 'height': height, 'used_wcs': use_wcs },
                                 upstreams=upstreams )
        if wcs_prov is not None:
            trimsrcprov = Provenance( process='Image.trim.sources', upstreams=[trimimprov] )
            trimwcsprov = Provenance( process='Image.trim.wcs', upstreams=[trimsrcprov] )
        else:
            trimsrcprov = None
            trimwcsprov = None

        if zp_prov is not None:
            # This one's a little weird... at least as of right now, zeropoint assumes
            #   a constant zeropoint for the whole image.  So, the trimmed zeropoint
            #   will be numerically identical to the original image zeroppoint.  That
            #   means nothing in the trimming is part of the trimmed zeropoint provenance
            #   upstreams.
            if not isinstance( zp_prov, Provenance ):
                zp_prov_obj = Provenance.get_by_id( zp_prov, pgdb=pgdb )
                if zp_prov_obj is None:
                    raise ValueError( f"Failed to get zp prov {zp_prov}" )
                zp_prov = zp_prov_obj
            trimzpprov = Provenance( process='Image.trim.zp', upstreams=[zp_prov] )
        else:
            trimzpprov = None

        if save:
            with PGDB( pgdb ) as pgdb:
                provs = [ trimimprov ]
                trimimprov.insert_if_needed( pgdb=pgdb )
                if wcs_prov is not None:
                    provs.extend( [ trimsrcprov, trimwcsprov ] )
                    trimsrcprov.insert_if_needed( pgdb=pgdb )
                    trimwcsprov.insert_if_needed( pgdb=pgdb )
                if zp_prov is not None:
                    provs.append( trimzpprov )
                    trimzpprov.insert_if_needed( pgdb=pgdb )
                if provtag is not None:
                    Provenance.addtag( provtag, provs, pgdb=pgdb )
        else:
            if provtag is not None:
                SCLogger.warning( "provtag was not None, but save was False, so provtag was ignored." )

        return trimimprov, trimsrcprov, trimwcsprov, trimzpprov


    def trim( self, x0, x1, y0, y1,
              adjust_limits=False,
              sources=None,
              bg=None,
              psf=None,
              wcs=None,
              zp=None,
              save_to_db=False, save_prov=False, provtag=None, no_load=False,
              pgdb=None ):
        """Return an Image (etc.) that's a cutout of this image.

        Returns a new Image with a trimmed data, weight, and flags
        array.  If wcs is given (strongly recommended) will update all
        coordate (ra/dec, etc.) fields, otherwise they will be copied
        from this object... which means that the coordinate fields will
        be quite wrong in the returned Image.

        The new Image will have its filepath set to a call to the result
        of self.invent_filepath(), with "_{x}_{y}" (the center of the
        trim region) appended before the suffix.  (This is sufficient to
        make the image unique, because the width and height are in the
        provenance.)

        If self.provenance_id is not None, then a Provenance will be
        loaded or created for the trimmed image which has process
        "Image.trim" and parameters { "width": x1-x0, "height": y1-y0,
        "used_wcs": wcs is not None }.  It will have
        self.provenance_id as an upstream.  If save_prov is True, this
        provenance will be saved to the databse.  If provtag is not
        None, this provenance will be tagged with the indicated tag.  If
        there are any conflicts in all of this, an exception will be
        raised.

        No FITS header copying is done.

        Parameters
        ----------
          x0, x1, y0, y1: int, required
             The minimum (inclusive)/maximum (exclusive) pixel limits of
             the trimmed region.

          sources: SoureList, default None
             SourceList that goes with wcs.  Required if wcs is not None.

          bg: Background, default None
             Background that goes with SourceList

          psf: PSF, default None
             PSF that goes with SourceList

          wcs: WorldCoordinates, default None
             If given, will update all coordinate fields of the image to
             be right for the trimmed image (assuming the wcs is right).
             Will also return the wcs for the trimmed image.  Requires
             sources.  Warning: if wcs is not given, then ra, dec,
             etc. in the image will all be wrong!

          zp: ZeroPoint, default None
             yadda yadda yadda

          save_prov: bool, default False
             If True, make sure the Provenance of the trimmed image is
             saved to the database.

          provtag: str, deafult None
             If not None, tag the Provenance of the trimmed image with
             this tag.  Requires save_prov to be True.

          save_to_db: bool, default False
             If True, then save everything created to the database.  If
             this is True, requires save_prov to also be True.  (Note
             that provenances may be separately saved if this is False
             based the value of save_prov.)  If you intend on saving to
             the database, it's a good idea to do this here rather than
             yourself, because this makes sure all the fiddly stuff gets
             done.  But, this defaults to False so that callers can
             collect things for efficiency.

             For this to work, both self and wcs (if wcs is not None)
             must already be saved to the database.

          no_load: bool, default False
             Normally, if self.provenance_id is not None, this function
             will try to read the trimmed image from the database, only
             generating it if it's not found.  Set this to False to
             always generate it.  If no_load is True, save_to_db must be
             False.

          adjust_limits: bool, default False
             Normally, if x0, x1, y0, or y1 are outside the bounds of
             the image, an exception will be raised.  Set this to True
             to instead just have it pull in that limit.

          pgdb: PGDB, psycopg.Connection, psycopg.Cursor, or sa Session, default None
             Database connection to use.  If not given, will open and
             close a new one if one is needed.

        Returns
        -------
           dict with keys:
              'image' : Image, the trimmed image
              'sources': SourceList, the trimmed sources, *if* sources was passed
              'bg': Background, the trimmed background, *if* bg was passed
              'psf': PSF, the trimmed PSF, *if* psf was passed
              'wcs': WorldCoordinates, the trimmed wcs, *if* wcs was passed
              'zp': ZeroPoint, the trimmed zeropoint, *if* zp was passed
              provenances: dict of { str: Provenance or None }
                           with keys 'image', 'sources', 'wcs', 'zp'
              limits: ( x0, x1, y0, y1 )

              limits are the limits actually used.  These will be the
              same as the passed x0, x1, y0, y1, unless adjust_limits
              was True and the passed limits went off of the edge of the
              image.

        """

        pgdb_in = pgdb

        # a bit of validation
        if save_to_db and ( not save_prov ):
            raise ValueError( "save_to_db requires save_prov" )
        if save_to_db and no_load:
            raise ValueError( "save_to_db requires no_load to be False" )
        if save_prov and ( self.provenance_id is None ):
            raise ValueError( "save_prov was set but self.provenance_id is None" )
        if ( provtag is not None ) and ( not save_prov ):
            raise ValueError( "Passing a provtag requires save_prov=True" )
        if not all( isinstance( i, numbers.Integral ) for i in [ x0, x1, y0 ,y1 ] ):
            raise TypeError( "x0, x1, y0, y1 must all be integers." )
        for val, name in zip( [ bg, psf, wcs ], [ 'bg', 'psf', 'wcs' ] ):
            if ( val is not None ) and ( sources is None ):
                raise ValueError( f"Passing {name} requires passing sources" )
        if ( zp is not None ) and ( wcs is None ):
            raise ValueError( "Passing zp requires passing wcs" )

        # Calculate xcen and ycen *before* adjusting for limits, because we want
        #   the filepath to reflect the aspirational center.
        xcen = int( np.floor( (x0 + x1) / 2. ) )
        ycen = int( np.floor( (y0 + y1) / 2. ) )
        width = x1 - x0
        height = y1 - y0

        if adjust_limits:
            x0 = max( x0, 0 )
            x1 = min( x1, self.data.shape[1] )
            y0 = max( y0, 0 )
            y1 = min( y1, self.data.shape[0] )
        else:
            if ( x0 < 0 ) or ( x1 > self.data.shape[1] ) or ( y0 < 0 ) or ( y1 > self.data.shape[0] ):
                raise ValueError( "Trim limits [{x0}:{x1}, {y0}:y1}] are outside image borders." )

        trimimprov = trimsrcprov = trimwcsprov = None
        trimim = trimsrc = trimbg = trimpsf = trimwcs = trimzp = None

        if self.provenance_id is not None:
            # Make the provenances
            with PGDB( pgdb ) as tmppgdb:
                provenance = Provenance.get( self.provenance_id, pgdb=tmppgdb )
                wcsprov = None if wcs is None else Provenance.get( wcs.provenance_id, pgdb=tmppgdb )
                zpprov = None if zp is None else Provenance.get( zp.provenance_id, pgdb=tmppgdb )
                ( trimimprov,
                  trimsrcprov,
                  trimwcsprov,
                  trimzpprov ) = self.get_trim_provs( width, height, upstreams=[provenance], wcs_prov=wcsprov,
                                                      zp_prov=zpprov, save=save_prov, provtag=provtag, pgdb=tmppgdb )

            # Try to load the image unless told not to.
            # Note that just having xcen and ycen, without width and height, still uniquely
            #   identifies the image, becasue the trim provenance includes width and height
            #   (see get_trim_provs).
            if not no_load:
                with PGDB( pgdb_in, dictcursor=True ) as pgdb:
                    q = sql.SQL( textwrap.dedent(
                        """\
                        SELECT i.* FROM images i
                        INNER JOIN image_trim_parent t ON i._id=t.image_id
                        WHERE i.provenance_id={prov}
                          AND t.parent_image_id={parentim}
                          AND t.parent_wcs_id{wcscondition}
                          AND t.trim_xcen={xcen}
                          AND t.trim_ycen={ycen}
                        """
                    ) ).format( prov=trimimprov.id, parentim=self.id, xcen=xcen, ycen=ycen,
                                wcscondition=( sql.SQL("={wcsid}").format(wcsid=wcs.id) if wcs is not None
                                               else sql.SQL( " IS NULL" ) ) )
                    rows = pgdb.execute( q )
                    if len(rows) > 1:
                        raise RuntimeError( "Database corruption, trimmed image is in database more than once." )
                    elif len(rows) == 1:
                        trimim = Image.create( **(rows[0]) )

                        # Avoid circular imports
                        from models.source_list import SourceList
                        from models.background import Background
                        from models.psf import PSF
                        from models.world_coordinates import WorldCoordinates
                        from models.zero_point import ZeroPoint

                        def _get_the_thing( pgdb, cls, parentcol, parentid, provid=None ):
                            table = cls.__tablename__
                            q = ( sql.SQL( "SELECT t.* FROM {table} t WHERE t.{parentcol}={parentid}" )
                                  .format( table=sql.Identifier(table), parentcol=sql.Identifier(parentcol),
                                           parentid=parentid ) )
                            if provid is not None:
                                q += sql.SQL( " AND t.provenance_id={prov}" ).format( prov=provid )
                            rows = pgdb.execute( q )
                            if len(rows) > 1:
                                raise RuntimeError( "Databse corrumption trimmed {table} is in the "
                                                    "database more than once." )
                            return cls.create( **(rows[0]) ) if len(rows) == 1 else None

                        if sources is not None:
                            trimsrc = _get_the_thing( pgdb, SourceList, "image_id", trimim.id, trimsrcprov.id )
                            if trimsrc is not None:
                                trimbg = _get_the_thing( pgdb, Background, "sources_id", trimsrc.id )
                                trimpsf = _get_the_thing( pgdb, PSF, "sources_id", trimsrc.id )

                            if ( trimsrc is not None ) and ( wcs is not None ):
                                trimwcs = _get_the_thing( pgdb, WorldCoordinates, "sources_id", trimsrc.id,
                                                          trimwcsprov.id )

                            if ( trimwcs is not None ) and ( zp is not None ):
                                trimzp = _get_the_thing( pgdb, ZeroPoint, "wcs_id", trimwcs.id,
                                                         trimzpprov.id )

        # Make the image stuff we didn't load
        to_save = {}

        if trimim is None:
            to_save['image'] = {}
            trimim = Image.copy_image( self, no_copy_data=True )
            trimim.data = self.data[ y0:y1, x0:x1 ].copy()
            trimim.weight = self.weight[ y0:y1, x0:x1 ].copy() if self.weight is not None else None
            trimim.flags = self.flags[ y0:y1, x0:x1 ].copy() if self.flags is not None else None
            trimim.width = trimim.data.shape[1]
            trimim.height = trimim.data.shape[0]
            trimim.is_trim = True
            if trimimprov is not None:
                trimim.provenance_id = trimimprov.id
                trimim.filepath = trimim.invent_filepath( extra=f"_{xcen}_{ycen}" )
            else:
                trimim.provenance_id = None
                trimim.filepath = None
        trimim._trim_image_parent = self.id
        trimim._trim_wcs_parent = wcs.id if wcs is not None else None
        trimim._trim_xcen = xcen
        trimim._trim_ycen = ycen

        if sources is not None:
            if trimsrc is None:
                to_save['sources'] = { 'image': trimim }
                trimsrc = sources.trim( x0, x1, y0, y1, trimmed_image=trimim )
                trimsrc.provenance_id = None if trimsrcprov is None else trimsrcprov.id

            if trimbg is None:
                to_save['bg'] = { 'image': trimim, 'sources': trimsrc }
                trimbg = bg.trim( x0, x1, y0, y1, trimmed_sources=trimsrc )

            if trimpsf is None:
                to_save['psf'] = { 'image': trimim, 'sources': trimsrc }
                trimpsf = psf.trim( x0, x1, y0, y1, trimmed_sources=trimsrc )

            if ( trimwcs is None ) and ( wcs is not None ):
                to_save['wcs'] = { 'image': trimim }
                trimwcs = wcs.trim( x0, x1, y0, y1 )
                trimwcs.sources_id = trimsrc.id
                trimwcs.provenance_id = None if trimwcsprov is None else trimwcsprov.id
                trimwcs.set_corners_from_wcs( trimim, width=x1-x0, height=y1-y0, setradec=True, mask=trimim.flags )
                # We want the trimmed image ra/dec to have the ra/dec of the thing we TRIED to center on.
                tmp = wcs.wcs.pixel_to_world_values( xcen, ycen )
                # It's very irritating that numpy returns array(5.) instead of 5.
                # Even worse is that numbers.isnstance( array(5.), Real ) is False.
                trimim.ra = float( tmp[0] )
                trimim.dec = float( tmp[1] )
                trimim.calculate_coordinates()
                trimim.set_corners_from_wcs( trimwcs.wcs, width=x1-x0, height=y1-y0, setradec=False )

            if ( trimzp is None ) and ( trimwcs is not None ) and ( zp is not None ):
                to_save['zp'] = None
                from models.zero_point import ZeroPoint
                trimzp = ZeroPoint.create( zp=zp.zp, dzp=zp.dzp, aper_cor_radii=zp.aper_cor_radii,
                                           aper_cors=zp.aper_cors, provenance_id=zp.provenance_id,
                                           wcs_id=trimwcs.id )
                trimzp.provenance_id = None if trimzpprov is None else trimzpprov.id

        retval = { 'image': trimim,
                   'sources': trimsrc,
                   'bg': trimbg,
                   'psf': trimpsf,
                   'wcs': trimwcs,
                   'zp': trimzp,
                   'provenances': { 'image': trimimprov,
                                    'sources': trimsrcprov,
                                    'wcs': trimwcsprov,
                                    'zp': trimzpprov },
                   'limits': ( x0, x1, y0, y1 )
                  }

        if save_to_db:
            # OMG RACE CONDITION
            # It's conceivable that two processes will generate the
            #   same trimmed image at once.
            # I'm just going to hope that hardly ever happens.  If
            #   it does, we'll get an exception below on
            #   pgdb.commit() because of the unique constraint on
            #   filepath.  Perhaps we should catch that exception
            #   and verify that the thing that got saved is the same
            #   as what we just generated.
            for which, kwargs in to_save.items():
                if kwargs is not None:
                    retval[which].save( **kwargs )
            with PGDB( pgdb_in ) as pgdb:
                for which in to_save.keys():
                    retval[which].insert( pgdb=pgdb, nocommit=True )
                pgdb.commit()

        return retval


    def invent_filepath( self, name_convention=None, extra=None, append=None, **overrides ):
        """Create a relative file path for the object.

        Create a file path relative to data root for the object based on its
        metadata.  This is used when saving the image to disk.  Data
        products that depend on an image and are also saved to disk
        (e.g., SourceList) will just append another string to the Image
        filename.

        Coadded or difference images (that have a list of
        upstream_images) will also be appended a "u-tag" which is just
        the letter u (for "upstreams") follwed by the first 6 characters
        of the SHA256 hash of the upstream image filepaths.  This is to
        make sure that the filepath is unique for each combination of
        upstream images.  Furthermore, they will have the RA and Dec (to
        4 decimal places) appended, so that filenames will be unique for
        coadds that happen to include exactly the same upstream images
        but that are centered differently.  (TODO: same center,
        different alignments!  Issue #541.)

        Does NOT set self.filepath, just returns what you might want to
        set it to.

        Parameters
        ----------
           name_convention: str, default None
              A properly-formatted naming convention.  Uses
              storage.image.name_convention from the config file if this
              isn't given.

           extras: str, default None
              If not None, this will be appended ot the end of the filepath
              (but before stuff that's added for coadded or difference
              images, described above).  It can include {field} stuff
              just like the filepath extension.

           append: str, default None
              Added to the very end of the filepath (but before .fits).
              Added as a raw string, so please no spaces, curly braces,
              etc.

           overrides:
              Further keywords can be passed to *override* what's in the
              Image object.  Use this with extreme care.  Arguments can include:
                provenance_id
                inst_name
                im_type
                project
                mjd
                filter_short
                section_id
                ra
                dec


        Returns
        -------
           str

        """
        prov_hash = inst_name = im_type = date = time = filter = ra = dec = dec_int_pm = project = ''
        section_id = section_id_int = ra_int = ra_int_h = ra_frac = dec_int = dec_frac = 0
        mjd = None

        overrideable = { 'prov_hash', 'inst_name', 'im_type', 'project', 'mjd',
                         'filter', 'section_id', 'ra', 'dec' }
        unknown = set( overrides.keys() ) - overrideable
        if len(unknown) > 0:
            raise ValueError( f"Unknown overrides: {unknown}" )
        for prop, val in overrides.items():
            locals()[prop] = overrides[prop]

        if ( self.provenance_id is not None ) and ( 'prov_hash' not in overrides ):
            prov_hash = self.provenance_id

        if ( self.instrument_object is not None ) and ( 'inst_name' not in overrides ):
            inst_name = self.instrument_object.get_short_instrument_name()
        if ( self.type is not None ) and ( 'im_type' not in overrides ):
            im_type = self.type
        if ( self.project is not None ) and ( 'project' not in overrides ):
            project = self.project

        mjd = self.mjd if 'mjd' not in overrides else mjd
        if ( mjd is not None ):
            t = Time(mjd, format='mjd', scale='utc').datetime
            date = t.strftime('%Y%m%d')
            time = t.strftime('%H%M%S')

        if ( self.filter_short is not None ) and ( 'filter' not in overrides ):
            filter = self.filter_short

        tmp_section_id = self.section_id if 'section_id' not in overrides else section_id
        if tmp_section_id is not None:
            section_id = str(tmp_section_id)
            try:
                section_id_int = int(tmp_section_id)
            except ValueError:
                section_id_int = 0  # TODO: maybe replace with a placeholder like 99?

        ra = self.ra if 'ra' not in overrides else ra
        if ra is not None:
            ra_int, ra_frac = str(float(ra)).split('.')
            ra_int = int(ra_int)
            ra_int_h = ra_int // 15
            ra_frac = int(ra_frac)

        dec = self.dec if 'dec' not in overrides else dec
        if dec is not None:
            dec_int, dec_frac = str(float(dec)).split('.')
            dec_int = int(dec_int)
            dec_int_pm = f'p{dec_int:02d}' if dec_int >= 0 else f'm{-dec_int:02d}'
            dec_frac = int(dec_frac)

        cfg = config.Config.get()
        default_convention = "{inst_name}_{date}_{time}_{section_id}_{filter}_{im_type}_{prov_hash:.6s}"
        if name_convention is None:
            name_convention = cfg.value('storage.images.name_convention', default=None)
        if name_convention is None:
            name_convention = default_convention

        if extra is not None:
            name_convention += extra

        filepath = name_convention.format(
            inst_name=inst_name,
            im_type=im_type,
            project=project,
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

        # For coadded and subtracted images, add some more things to the filename.
        # Reason: you can have more than one coadd or subtraction image in the
        #   same provenance that has the same image used as the base for its name,
        #   but they *can* be different images.
        # Two things can be different:
        #   (1) the set of images combined may be different
        #   (2) it's possible that the set of images combined in a coadd
        #       is the same, but the image is centered differently.
        #
        # To keep filenames unique, add two things to the end of the
        #    filename.  First, add a hash of the images (really,
        #    zeropoint ids) that went into the coadd or subtraction.
        #    Second, add a 0.0001-resolution RA and Dec.  (...Yeah,
        #    somebody might coadd things that are centered differently,
        #    but by less than 0.0001° in RA or Dec, but that's a really
        #    perverse edge case.  pipeline/ref_maker.py should usually
        #    be run with an overlap fraction that allows for
        #    misalignment of more than an arcsecond when choosing refs!)
        #
        # ...it's possible that the ra and dec is already in the filename,
        #   given the naming convention.  Probably we should refactor this
        #   to have separate naming conventions for regular, coadded, and
        #   subtracted images!
        if self.is_coadd or self.is_sub:
            utag = hashlib.sha256()

            if self.is_coadd:
                for id in self.coadd_component_zp_ids:
                    utag.update( str(id).encode('utf-8') )
            else:
                utag.update( str(self.ref_id).encode('utf-8') )
                utag.update( str(self.new_zp_id).encode('utf-8') )

            utag = base64.b32encode(utag.digest()).decode().lower()
            utag = '_u-' + utag[:6]
            filepath += utag

            filepath += f"_{ra:08.4f}{dec:+08.4f}"

        if append is not None:
            filepath += append

        return filepath


    def _file_suffix( self, comp=None ):
        if self.format == 'fits':
            return ".fits"
        elif self.format == 'fitsfz':
            return ".fits.fz"
        raise ValueError( f"Don't know suffix for image format {self.format}" )


    def save(self, filename=None, only_image=False, just_update_header=True, **kwargs ):
        """Save the data (along with flags, weights, etc.) to disk.

        The format to save is determined by the config file.
        Use the filename to override the default naming convention.

        Will save the standard image extensions : image, weight, mask.

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
            case, if just_update_header is True and the file already
            exists, don't write the data, just update the header.

        **kwargs: passed on to FileOnDiskMixin.save(), include:
            overwrite - bool, set to True if it's OK to overwrite exsiting files
            no_archive - bool, set to True to save only to local disk, otherwise also saves to the archive
            exists_ok, verify_md5 - complicated, see documentation on FileOnDiskMixin

        For images that will be saved to the database, you probably want
        to use overwrite=True, verify_md5=True, or perhaps
        overwrite=False, exists_ok=True, verify_md5=True.  For temporary
        images being saved as part of local processing, you probably
        want to use verify_md5=False and either overwrite=True (if
        you're modifying and writing the file multiple times), or
        overwrite=False, exists_ok=True (if you might call the save()
        method more than once on the same image, and you want to trust
        the filesystem to have saved it right).

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
        if self.format is None:
            self.format = cfg.value('storage.images.format', default='fits')

        single_file = cfg.value('storage.images.single_file', default=False)
        extensions = []
        files_written = {}

        if not only_image:
            # In order to ignore just_update_header if only_image is false,
            # we need to pass it as False on to save_fits_image_file
            just_update_header = False

        full_path = os.path.join(self.local_path, self.filepath)

        if ( self.format == 'fits' ) or ( self.format == 'fitsfz' ):
            fpack = ( self.format == 'fitsfz' )
            # save the imaging data
            extensions.append('image')
            imgpath, self.header = save_fits_image_file( full_path, self.data, self.header,
                                                         extname='image', single_file=single_file, fpack=fpack,
                                                         just_update_header=just_update_header )
            files_written['image'] = imgpath

            # save the other extensions
            # Write them with an empty header to avoid confusion.  (We're later going
            # to update the image header, but won't update the headers of the
            # associated files, so better just to have nothing in associated files.)
            if single_file or ( not only_image ):
                for array_name in self.saved_components:
                    if array_name == 'image':
                        continue
                    array = getattr(self, array_name)
                    if array is not None:
                        extpath, _nullhdr = save_fits_image_file( full_path,
                                                                  array,
                                                                  fits.Header(),
                                                                  extname=array_name,
                                                                  single_file=single_file,
                                                                  fpack=fpack,
                                                                  lossless=(array_name in self.lossless_components)
                                                                 )
                        extensions.append(array_name)
                        if not single_file:
                            files_written[array_name] = extpath

            if single_file:
                if fpack:
                    raise NotImplementedError( "single_file doesn't currently support fpack" )
                files_written = files_written['image']
                if not self.filepath.endswith('.fits'):
                    self.filepath += '.fits'

        elif self.format == 'hdf5':
            raise NotImplementedError("HDF5 format is not yet supported.")

        else:
            raise ValueError(f"Unknown image format: {self.format}. Use 'fits' or 'fitsfz'.")

        # Save the file to the archive and update the fields of the Image object accordingly.
        # (as well as self.filepath, self.components, self.md5sum, self.md5sum_components)
        # (From what we did above, it's already in the right place in the local filestore.)
        if single_file:
            FileOnDiskMixin.save( self, files_written, **kwargs )
        else:
            if just_update_header:
                FileOnDiskMixin.save( self, files_written['image'], 'image', **kwargs )
            else:
                for ext in extensions:
                    FileOnDiskMixin.save( self, files_written[ext], ext, **kwargs )

    def load(self):
        """Load the image data from disk, including weight, flags, etc.

        Will always load the _data property for the image component.  If
        there is data on disk for flags, weight, or other components,
        will load the relevant "_{comp}" properties as well.  Does *not*
        load the image header.

        """

        if self.filepath is None:
            raise ValueError("The filepath is not set. Cannot load the image.")

        cfg = config.Config.get()
        single_file = cfg.value('storage.images.single_file')

        if single_file:
            filename = self.get_fullpath( nofile=False )
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"Could not find the image file: {filename}")
            self._data, self._header = read_fits_image(filename, ext='image', output='both')
            self.width = self._data.shape[1]
            self.height = self._data.shape[0]
            for att in self.saved_components:
                if att == 'image':
                    continue
                array = read_fits_image(filename, ext=att)
                setattr(self, f'_{att}', array)

        else:  # load each data array from a separate file
            if self.components is None:
                self._data, self._header = read_fits_image( self.get_fullpath(nofile=False), output='both' )
                self.width = self._data.shape[1]
                self.height = self._data.shape[0]
            else:
                gotim = False
                gotweight = False
                gotflags = False
                for comp, filename in zip( self.components, self.get_fullpath(as_list=True, nofile=False) ):
                    if not os.path.isfile(filename):
                        raise FileNotFoundError(f"Could not find the image component file: {filename}")
                    if comp == 'image':
                        self._data, self._header = read_fits_image( filename, output='both' )
                        self.width = self._data.shape[1]
                        self.height = self._data.shape[0]
                    else:
                        setattr( self, f'_{comp}', read_fits_image( filename, output='data' ) )

                    if comp == 'image':
                        gotim = True
                    if comp == 'weight':
                        gotweight = True
                    if comp == 'flags':
                        gotflags = True
                    elif comp not in self.saved_components:
                        raise ValueError( f'Unknown image component {comp}' )

                if not ( gotim and gotweight and gotflags ):
                    raise FileNotFoundError( "Failed to load at least one of image, weight, flags" )

        # Just in case weight value got fiddled during lossy compression, explicitly set all
        #   weights for flagged pixels exactly to 0
        if ( self._weight is not None ) and ( self._flags is not None ):
            self._weight[ self._flags != 0 ] = 0.

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
        allfree = set( Image.saved_components )
        allfree.add( "raw_data" )
        tofree = set( only_free ) if only_free is not None else allfree

        for prop in tofree:
            if prop not in allfree:
                raise RuntimeError( f"Unknown image property to free: {prop}" )
            if prop == 'raw_data':
                self.raw_data = None
            else:
                setattr( self, f'_{prop}', None )


    def get_upstream_ids(self, full_chain=False, _seen=None, pgdb=None):
        """Get the ids immediate upstreams of this image.

        This may include an exposure (for most images), zeropoints (if
        this is subtraction or coadd image), and/or a reference (if this
        is a subtraction).  If full_chain is True, then this can also
        include world coordinates, source lists, and images.

        Parameters
        ----------
          full_chain: bool, default False
            Normally this only returns the immediate upstreams.  Set
            full_chain to True to get everything all the way back to the
            beginning.  WARNING, these are not sorted in any way.

            WARNING : this is not tested, and (I think) not even used in
            the code base right now.  TODO: write tests.

          _seen: set
              Used internally for recursion if full_chain is True.

        Returns
        -------
        upstreams: list of [ ( class, id ) ]
            The upstream Exposure, WorldCoordiantes, ZeroPoint, and
            Reference ids that were used to create this image.  For most
            images, it will be (at most) a single Exposure id.  For
            coadds, it will be a bunch of ZeroPoints.  For a
            subtraction, will be one ZeroPoint and one Reference.  For
            warped images, it will be two WorldCoordiantes.  Don't count
            on the list as being sorted in any particular way.  It's not
            fully deterministic, nor is it random, but in any event it's
            not obvious.

            If full_chain is true, this list can also include
            SourceList, WorldCoordinates, and Image ids.

        """

        # Avoid circular imports
        from models.source_list import SourceList
        from models.world_coordinates import WorldCoordinates
        from models.zero_point import ZeroPoint
        from models.reference import Reference

        seen = set() if _seen is None else _seen

        if self.exposure_id is None:
            upstreams = []
        else:
            seen.add( self.exposure_id )
            upstreams = [ ( Exposure, self.exposure_id ) ]

        with PGDB( pgdb ) as pgdb:
            if self.is_sub:
                if self.is_coadd:
                    raise ValueError( f"Database corruption, image {self.id} is both a sub and a coadd!!!!!" )
                if self.is_trim:
                    raise ValueError( f"Database corruption, image {self.id} is both a sub and a trim!!!!!" )
                q = sql.SQL( "SELECT new_zp_id, ref_id FROM image_subtraction_components WHERE image_id={me}"
                             ).format( me=self.id )
                rows, _cols = pgdb.execute( q )
                for row in rows:
                    if row[0] not in seen:
                        upstreams.append( ( ZeroPoint, row[0] ) )
                        seen.add( row[0] )
                    if row[1] not in seen:
                        upstreams.append( ( Reference, row[1] ) )
                        seen.add( row[1] )

            elif self.is_coadd:
                if self.is_trim:
                    raise ValueError( f"Database corruption, image {self.id} is both a coadd and a trim!!!!!" )
                q = sql.SQL( "SELECT zp_id FROM image_coadd_component WHERE coadd_image_id={me}" ).format( me=self.id )
                rows, _cols = pgdb.execute( q )
                for row in rows:
                    if row[0] not in seen:
                        upstreams.append( ( ZeroPoint, row[0] ) )
                        seen.add( row[0] )

            elif ImageTypeConverter().to_string( self._type ) in ( 'Warped', 'ComWarped' ):
                q = sql.SQL( "SELECT unwarped_zp_id, target_wcs_id FROM image_warp_parent "
                             "WHERE warped_id={me}" ).format( me=self.id )
                rows, _cols = pgdb.execute( q )
                if len(rows) > 1:
                    raise RuntimeError( f"Database corruption, image {self.id} has multiple warp parents!  "
                                        f"This should never happen." )
                elif len(rows) == 1:
                    if rows[0][0] not in seen:
                        upstreams.append( ( ZeroPoint, rows[0][0] ) )
                        seen.add( rows[0][0] )
                    if rows[0][1] not in seen:
                        upstreams.append( ( WorldCoordinates, rows[0][1] ) )
                        seen.add( rows[0][1] )

            elif self.is_trim:
                q = sql.SQL( "SELECT parent_image_id, parent_wcs_id FROM image trim_parent "
                             "WHERE image_id={me}" ).formt( me=self.id )
                rows, _cols = pgdb.execute( q )
                for row in rows:
                    if row[0] not in seen:
                        upstreams.append( ( Image, row[0] ) )
                        seen.add( row[0] )
                    if row[1] not in seen:
                        upstreams.append( ( WorldCoordinates, row[1] ) )
                        seen.add( row[1] )

            if full_chain:
                # Get upstreams of WorldCoordinateses.  These will be from trimmed and warped images.
                wcsupstrs = [ u[1] for u in upstreams if u[0] == WorldCoordinates ]
                if len( wcsupstrs ) > 0:
                    q = sql.SQL( textwrap.dedent(
                        """\
                        SELECT s._id, i._id
                        FROM world_coordinates w
                        INNER JOIN source_lists s ON s._id=w.sources_id
                        INNER JOIN images i ON i._id=s.image_id
                        WHERE w._id=ANY(ARRAY[{wcsids}])
                        """
                    ) ).format( wcsids=sql.SQL(",").join(wcsupstrs) )
                    rows, _cols = pgdb.execute( q )
                    for row in rows:
                        if row[0] not in seen:
                            seen.add( row[0] )
                            upstreams.append( ( SourceList, row[0] ) )
                        if row[1] not in seen:
                            seen.add( row[1] )
                            upstreams.append( ( Image, row[1] ) )

                # Get all upstreams back to Image of ZeroPoints.  This will come from subs and coads.
                zpupstrs = [ u[1] for u in upstreams if u[0] == ZeroPoint ]
                if len( zpupstrs ) > 0:
                    q = sql.SQL( textwrap.dedent(
                        """\
                        SELECT w._id, s._id, i._id
                        FROM zero_points z
                        INNER JOIN world_coordinates w ON w._id=z.wcs_id
                        INNER JOIN source_lists s ON s._id=w.sources_id
                        INNER JOIN images i ON s.image_id=i._id
                        WHERE z._id=ANY(ARRAY[{zpids}])
                        """
                    ) ).format( zpids=sql.SQL(",").join(zpupstrs) )
                    rows, _cols = pgdb.execute( q )
                    for row in rows:
                        if row[0] not in seen:
                            upstreams.append( ( WorldCoordinates, row[0] ) )
                            seen.add( row[0] )
                        if row[1] not in seen:
                            upstreams.append( ( SourceList, row[1] ) )
                            seen.add( row[1] )
                        if row[2] not in seen:
                            seen.add( row[2] )
                            upstreams.append( ( Image, row[2] ) )

                # Recursively get all upstreams of Images we've collected
                for upstream in upstreams:
                    if upstream[0] == Image:
                        img = Image.get_by_id( upstream[1], pgdb=pgdb )
                        upstreams.extend( img.get_upstream_ids( full_chain=True, _seen=seen, pgdb=pgdb ) )

        return upstreams


    def get_downstream_ids(self, pgdb=None):
        """Get ids all the data products that were created based on this image.

        This will include:
           * source lists
           * images (images trimmed from this image)
           * dia forced photometry

        """

        # avoids circular import
        from models.source_list import SourceList
        from models.diaforcedphot import DiaForcedPhot

        downstreams = []

        with PGDB( pgdb ) as pgdb:
            q = sql.SQL( "SELECT _id FROM source_lists WHERE image_id={me}" ).format( me=self.id )
            rows, _cols = pgdb.execute( q )
            downstreams.extend( [ ( SourceList, row[0] ) for row in rows ] )

            q = sql.SQL( "SELECT _id FROM dia_forced_photometry WHERE subtraction_id={me}" ).format( me=self.id )
            rows, _cols = pgdb.execute( q )
            downstreams.extend( [ ( DiaForcedPhot, row[0] ) for row in rows ] )

            q = sql.SQL( "SELECT image_id FROM image_trim_parent WHERE parent_image_id={me}"
                        ).format( me=self.id )
            rows, _cols = pgdb.execute( q )
            downstreams.extend( [ ( Image, row[0] ) for row in rows ] )

        return downstreams

    def get_new_image( self, pgdb=None, return_sources=False, return_wcs=False, return_zp=False ):
        if not self.is_sub:
            raise ValueError( "Can't get new image for image that isn't a subtraction." )
        with PGDB( pgdb ) as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT i._id, s._id, w._id, z._id FROM image_subtracton_components isc
                INNER JOIN zero_points z ON isc.new_zp_id=z._id
                INNER JOIN world_coordinates w ON z.wcs_id=w._id
                INNER JOIN source_lists s ON w.sources_id=s._id
                INNER JOIN images i ON s.image_id=w._id
                WHERE isc.image_id={me}
                """
            ) ).format( me=self.id )
            rows, _cols = pgdb.execute( q )
            if len(rows) != 1:
                raise RuntimeError( f"Failed to find new image for subtracton image {self.id}" )
            rval = { 'image': Image.get_by_id( rows[0][0], pgdb=pgdb ) }
            if return_sources:
                # import here to avoid circular imports
                from models.source_list import SourceList
                rval['sources'] = SourceList.get_by_id( rows[0][1], pgdb=pgdb )
            if return_wcs:
                from models.world_coordinates import WorldCoordinates
                rval['wcs'] = WorldCoordinates.get_by_id( rows[0][2], pgdb=pgdb )
            if return_zp:
                from models.zero_point import ZeroPoint
                rval['zp'] = ZeroPoint.get_by_id( rows[0][3], pgdb=pgdb )

            return rval

    def get_ref_image( self, warp_prov=None, pgdb=None, return_ref=False, return_sources=False, return_wcs=False,
                       return_zp=False, return_warped=False ):
        if not self.is_sub:
            raise ValueError( "Can't get ref image for image that isn't a subtraction" )
        with PGDB( pgdb ) as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT i._id, s._id, w._id, z._id, ref._id FROM image_subtraction_components isc
                INNER JOIN refs ref ON isc.ref_id=ref._id
                INNER JOIN zero_points z ON ref.zp_id=z._id
                INNER JOIN world_coordinates w ON z.wcs_id=w._id
                INNER JOIN source_lists s ON w.sources_id=s._id
                INNER JOIN images i ON s.image_id=i._id
                WHERE isc.image_id={me}
                """
            ) ).format( me=self.id )
            rows, _cols = pgdb.execute( q )
            if len(rows) != 1:
                raise RuntimeError( f"Failed to find ref image for subtraction image {self.id}" )
            rval = { 'image': Image.get_by_id( rows[0][0], pgdb=pgdb ) }
            if return_sources:
                from models.source_list import SourceList
                rval['sources'] = SourceList.get_by_id( rows[0][1], pgdb=pgdb )
            if return_wcs:
                from models.world_coordinates import WorldCoordinates
                rval['wcs'] = WorldCoordinates.get_by_id( rows[0][2], pgdb=pgdb)
            if return_zp:
                from models.zero_point import ZeroPoint
                rval['zp'] = ZeroPoint.get_by_id( rows[0][3], pgdb=pgdb )
            if return_ref:
                from models.reference import Reference
                rval['ref'] = Reference.get_by_id( rows[0][4], pgdb=pgdb )
            if return_warped:
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT i.* FROM image_subtraction_components isc
                    INNER JOIN zero_points z ON isc.new_zp_id=z._id
                    INNER JOIN image_warp_parent iwp ON iwp.target_wcs_id=z.wcs_id
                    INNER JOIN images i ON i._id=iwp.warped_id
                    WHERE iwp.unwarped_zp_id={zpid}
                      AND isc.image_id={me}
                    """
                ) ).format( me=self.id, zpid=rows[0][3] )
                if warp_prov is not None:
                    warp_prov = warp_prov.id if isinstance( warp_prov, Provenance ) else warp_prov
                    q += sql.SQL( "  AND iwp.warp_provenance_id={prov}" ).format( prov=warp_prov )
                rows, _cols = pgdb.execute( q )
                if len(rows) == 0:
                    # Might just not exist; we don't require it to
                    rval['warped' ] = None
                elif warp_prov is not None:
                    rval['warped'] = [ Image.create( **row ) for row in rows ]
                else:
                    rval['warped'] = Image.create( **(rows[0]) )

            return rval


    @staticmethod
    def find_images(
            ra=None,
            dec=None,
            minra=None,
            maxra=None,
            mindec=None,
            maxdec=None,
            image=None,
            overlapfrac=None,
            provenance_ids=None,
            provenance_ids_are_zp=False,
            provenance_ids_are_wcs=False,
            use_good=True,
            type=[1,2,3,4,20,21],
            target=None,
            section_id=None,
            project=None,
            instrument=None,
            filter=None,
            min_mjd=None,
            max_mjd=None,
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
            order_by=None,
            seeing_quality_factor=3.0,
            max_number=None,
            return_zeropoints=False,
            return_wcs=False,
            pgdb=None
    ):
        """Return a list of images that match criteria.

        For position searching, can operate in three modes:

        * Not filtering on position

        * Finding all images that include a point.  Specify ra/dec.

        * Finding all images that overlap a rectangle.  Specify either
          minra/maxra/mindec/maxdec or image.  If overlapfrac is not
          None, only include images that overlap the desired area by at
          least this fraction.  (Note: this isn't really right, as this
          routine treats all images as N-S/E-W aligned rectangles on the
          sky; see doc on overlap frac below.)

        It will combine the position filters with all the other
        conditions, ANDing together all the criteria.

        Parameters
        ----------
        ra, dec: float (decimal degrees) or (HH:MM:SS / dd:mm:ss) or None
            If supplied, will find images that contain this ra
            and dec.  The images you get back will be a susperset of
            images t

        minra, maxra, mindec, maxdec: float (decimal degrees) or (HH:MM:SS / dd:mm:ss) or None
            Specify a rectangle on the sky, find all images that overlap
            this rectangle at all.  You can only give one of ra/dec or
            minra/maxra/mindec/maxdec.  Note that minra is the West side
            of the image, and maxra is the East side of the image, so if
            the image is 2° wide centered around 0°, minra will be 359
            and maxra will be 1.

        image: Image (really, any object that inherits the FourCorners mixin) or None
            If supplied, pull the minra/maxra/mindec/maxdec from this image

        overlapfrac: float or None
            If supplied (which may only happen when min/max ra/dec are
            supplied), only return images that overlap the target
            rectangle by at least this much.  (Sort of.)  NOTE: this
            doesn't do real overlap fractions of images!  Rather, it
            does the overlap fraction of the NS/EW-aligned bounding
            boxes of images!  See docstring for
            FourCorners.get_overlap_frac().  If that is fixed, this
            should be too.  If you don't specify anything, a default of
            0.95 will be used.

        provenance_ids: str or list of strings
            Provenance ids to search on.  By default, it searches on
            Image provenances; if provenance_ids_are_zp is True, then
            search for images with zps with this provenance, or if
            provenance_ds_are_wcs is True, then search for images with
            wcses with this provenacne.

            If either provenance_ids_are_zp or provenance_ids_are_wcs is
            True, then position searches are on WCS corners rather than
            image corners.

        provenance_ids_are_zp: bool, default False
            See provenance_ids

        provenance_ids_are_wcs: bool, default False
            See provenance_ids

        use_good: bool, default True
            Ignored unelss wcs_provenance_ids is not None, or
            provenacne_ids_are_zp is True.  If this is True, then look
            at the ra_good_xx and dec_good_xx fields in the WCS instead
            of the ra_corner_xx and dec_corner_xx fields in the WCS
            and/or image.

        type: integer or string or list of integers or strings, None
            List of image types to search for; see
            enums_and_bitflags.py::ImageTypeConverter for the values.
            Use "Sci" or 1 to get regular (non-coadd, non-subtraction)
            images.  This defaults to [1,2,3,4,20,21], which gets science,
            coadded science, difference, and coadded difference images;
            it omits calibration images (bias, flats, etc.) and warped
            images.  Set this to None to get everything.

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

        min_mjd: float, str, astropy.time.Time, datetime.datetime, or datetime.date (optional)
            Find images taken after this time.  If a float, it is an
            MJD.  If a str, it must be ISO8601 formatted and should
            either have an explicit time zone (ideal) or be UTC.

        max_mjd: float, str, astropy.time.Time, datetime.datetime, or datetime.date (optional)
            Find images taken after this time.

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

        max_number: int or None
            If not None, only return this many images.  Will return the
            first max_number returned from the database, ordered as
            specified in order_by.

        return_zeropoints: bool, default False

        return_wcs: bool, default False

        pgdb: PGDB, psycopg.Connection, psycopg.Cursor, or sa Session, default None
            Database connection to use.  If not given, will open and close a new one.

        Returns
        -------
          One of:
           * list of Image
           * ( list of Image, dict of image_id: WorldCoordinates )
           * ( list of Image, dict of image_id: ZeroPoint )
           * ( list of Image, dict of image_id: WorldCoordinates, dict of image_id: ZeroPoint )

        """

        if provenance_ids_are_zp and provenance_ids_are_wcs:
            raise ValueError( "Can't give both provenance_ids_are_zp and provenance_ids_are_wcs" )

        if provenance_ids is None:
            raise ValueError( "Must specify provenance(s) to search" )
        provenance_ids = listify( provenance_ids, require_string=True )

        if return_zeropoints and ( not provenance_ids_are_zp ):
            raise ValueError( "return_zeropoints requires provenance_ids_are_zp" )

        if return_wcs and ( not ( provenance_ids_are_zp or provenance_ids_are_wcs ) ):
            raise ValueError( "return_wcs requires provenance_ids_are_wcs or provenance_ids_are_zp" )

        # Avoid circular imports
        from models.world_coordinates import WorldCoordinates

        barf = "".join( random.choices( "abcdefghijklmnopqrstuvwxyz", k=6 ) )
        searchcontaining = False
        searchoverlapping = False
        searchtable = None
        fcobj = None

        with PGDB( pgdb, dictcursor=True ) as pgdb:
            # First: position filter (but not including overlapfrac).  This may involve
            #   calling a FourCorners routine to build a temporary table.

            if ( ra is not None ) or ( dec is not None ):
                # Filter by position
                if any( i is not None for i in [ minra, maxra, mindec, maxdec ] ):
                    raise ValueError( "Cannot specify min/max ra/dec and ra/dec together" )
                if ( ra is None ) or ( dec is None ):
                    raise ValueError( "Must provide both or neither of ra/dec" )
                if overlapfrac is not None:
                    raise ValueError( "Can't provide overlap frac with ra/dec" )

                if isinstance( ra, str ):
                    ra = parse_ra_hms_to_deg( ra )
                if isinstance( dec, str ):
                    dec = parse_dec_dms_to_deg( dec )

                searchcontaining = True

                if provenance_ids_are_zp or provenance_ids_are_wcs:
                    corner = "good" if use_good else "corner"
                    limprefix = "good_" if use_good else ""

                    if provenance_ids_are_zp:
                        WorldCoordinates._find_possibly_containing_temptable(
                            ra, dec, session=pgdb, prov_id=provenance_ids, corner=corner, limprefix=limprefix,
                            fromclause=( "FROM world_coordinates i\n"
                                         "INNER JOIN zero_points z ON z.wcs_id=i._id" ),
                            provtable='z',
                            temptable=f"temp_find_containing_{barf}"
                        )
                    else:
                        WorldCoordinates._find_possibly_containing_temptable(
                            ra, dec, session=pgdb, prov_id=provenance_ids, corner=corner, limprefix=limprefix,
                            temptable=f"temp_find_containing_{barf}" )

                    pgdb.execute_nofetch(
                        sql.SQL( textwrap.dedent(
                            """\
                            SELECT i._id,
                            t.ra_corner_00, t.ra_corner_01, t.ra_corner_10, t.ra_corner_11,
                            t.dec_corner_00, t.dec_corner_01, t.dec_corner_10, t.dec_corner_11
                            INTO TEMP TABLE temp_find_image_1_{barf}
                            FROM images i
                            INNER JOIN source_lists s ON i._id=s.image_id
                            INNER JOIN world_coordinates w ON s._id=w.sources_id
                            INNER JOIN temp_find_containing_{barf} t ON t._id=w._id
                            """
                        ) ).format( barf=sql.SQL(barf) )
                    )
                    searchtable = f"temp_find_image_1_{barf}"

                else:
                    Image._find_possibly_containing_temptable( ra, dec, session=pgdb, prov_id=provenance_ids,
                                                               temptable=f"temp_find_containing_{barf}" )
                    searchtable = f"temp_find_containing_{barf}"

            elif any( i is not None for i in [ image, minra, maxra, mindec, maxdec ] ):
                # Filter by rectangle
                if image is not None:
                    if any( i is not None for i in [ minra, maxra, mindec, maxdec ] ):
                        raise ValueError( "May specify either image or min/max ra/dec, not both" )
                    minra = image.minra
                    maxra = image.maxra
                    mindec = image.mindec
                    maxdec = image.maxdec
                else:
                    if any( i is None for i in [ minra, maxra, mindec, maxdec ] ):
                        raise ValueError( "Must specify either all or none of minra/maxra/mindec/maxdec" )
                    if isinstance( minra, str ):
                        minra = parse_ra_hms_to_deg( minra )
                    if isinstance( maxra, str ):
                        maxra = parse_ra_hms_to_deg( maxra )
                    if isinstance( mindec, str ):
                        mindec = parse_dec_dms_to_deg( mindec )
                    if isinstance( maxdec, str ):
                        maxdec = parse_dec_dms_to_deg( maxdec )

                fcobj = FourCorners()
                fcobj.dec = (mindec + maxdec) / 2.
                fcobj.ra_corner_00 = minra
                fcobj.ra_corner_01 = minra
                fcobj.minra = minra
                fcobj.ra_corner_10 = maxra
                fcobj.ra_corner_11 = maxra
                fcobj.maxra = maxra
                fcobj.dec_corner_00 = mindec
                fcobj.dec_corner_10 = mindec
                fcobj.mindec = mindec
                fcobj.dec_corner_01 = maxdec
                fcobj.dec_corner_11 = maxdec
                fcobj.maxdec = maxdec

                searchoverlapping = True
                overlapfrac = overlapfrac if overlapfrac is not None else 0.95

                if provenance_ids_are_zp or provenance_ids_are_wcs:
                    corner = "good" if use_good else "corner"
                    limprefix = "good_" if use_good else ""

                    if provenance_ids_are_zp:
                        WorldCoordinates._find_potential_overlapping_temptable(
                            fcobj, session=pgdb, prov_id=provenance_ids, corner=corner, limprefix=limprefix,
                            fromclause=( "FROM world_coordinates i\n"
                                         "INNER JOIN zero_points z ON z.wcs_id=i._id" ),
                            provtable='z',
                            temptable=f"temp_find_overlapping_{barf}"
                        )
                    else:
                        WorldCoordinates._find_possibly_containing_temptable(
                            fcobj, session=pgdb, prov_id=provenance_ids, corner=corner, limprefix=limprefix,
                            temptable=f"temp_find_overlapping_{barf}" )

                    pgdb.execute_nofetch(
                        sql.SQL( textwrap.dedent(
                            """\
                            SELECT i._id,
                            t.ra_corner_00, t.ra_corner_01, t.ra_corner_10, t.ra_corner_11,
                            t.dec_corner_00, t.dec_corner_01, t.dec_corner_10, t.dec_corner_11
                            INTO TEMP TABLE temp_find_image_2_{barf}
                            FROM images i
                            INNER JOIN source_lists s ON i._id=s.image_id
                            INNER JOIN world_coordinates w ON s._id=w.sources_id
                            INNER JOIN temp_find_overlapping_{barf} t ON t._id=w._id
                            """
                        ) ).format( barf=sql.SQL(barf) )
                    )
                    searchtable = f"temp_find_image_2_{barf}"

                else:
                    Image._find_potential_overlapping_temptable( fcobj, session=pgdb, prov_id=provenance_ids,
                                                                 temptable=f"temp_find_overlapping_{barf}" )
                    searchtable = f"temp_find_overlapping_{barf}"
            else:
                if overlapfrac is not None:
                    raise ValueError( "overlapfrac only makes sense with image or min/max ra/dec" )

            # Second: all filtering other than position

            # Build the query that allows us to search either our temp
            # table or the images table This is a little awkward,
            # because we're using SQLAlchemy.  I don't know an easy way
            # to join to a table that SQLA doesn't know about (i.e. our
            # temp tables) using SQLA constructs, so I'm going to just
            # build a SQL query and hope that I understand SQLA
            # from_statement well enough to do the right thing.  (I'm
            # also not sure how I'd do a q3c_poly_query with SQLA, and
            # it's not worth the effort of trying to figure out the
            # syntax.)

            imgid_temptable = sql.Identifier( f"temp_find_images_image_ids_{barf}" )
            pgdb.execute_nofetch( sql.SQL( "DROP TABLE IF EXISTS {temptable}" )
                                  .format( temptable=imgid_temptable ) )
            if searchtable is None:
                q = sql.SQL( "SELECT i.* INTO TEMP TABLE {temptable} FROM images i\n"
                             ).format( temptable=imgid_temptable )

                if provenance_ids_are_wcs or provenance_ids_are_zp:
                    q += sql.SQL( "INNER JOIN source_lists s ON s.image_id=i._id\n"
                                  "INNER JOIN world_coordinates w ON w.sources_id=s._id\n" )
                    if provenance_ids_are_wcs:
                        subq = sql.SQL( "WHERE w.provenance_id=ANY(ARRAY[{provs}])\n" )
                    else:
                        subq = sql.SQL( "INNER JOIN zero_points z ON z.wcs_id=w._id\n"
                                        "WHERE z.provenance_id=ANY(ARRAY[{provs}])\n" )
                else:
                    subq = sql.SQL( "WHERE i.provenance_id=ANY(ARRAY[{provs}])\n" )

                q += subq.format( provs=sql.SQL(",").join( provenance_ids ) )

            else:
                # Don't have to filter on provenance because that will have happened
                #  in the building of the temp table above
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT i._id
                    INTO TEMP TABLE {temptable}
                    FROM {searchtable} t
                    INNER JOIN images i ON t._id=i._id
                    """
                ) ).format( searchtable=sql.Identifier(searchtable), temptable=imgid_temptable )
                if searchcontaining:
                    # Refine the search from the broad overlap in the temp table
                    #   to a real poly search.
                    q += sql.SQL( textwrap.dedent(
                        """\
                        WHERE q3c_poly_query({ra}, {dec}, ARRAY[ t.ra_corner_00, t.dec_corner_00,
                                                                 t.ra_corner_01, t.dec_corner_01,
                                                                 t.ra_corner_11, t.dec_corner_11,
                                                                 t.ra_corner_10, t.dec_corner_10 ])
                        """
                    ) ).format( ra=ra, dec=dec )

            # A few fields need preprocessing before feeding into the code below
            min_mjd = None if min_mjd is None else parse_dateobs(min_mjd, output='mjd')
            max_mjd = None if max_mjd is None else parse_dateobs(max_mjd, output='mjd')
            types = None if type is None else [ ImageTypeConverter.to_int(t) for t in listify(type) ]

            # Note that we do NOT filter on provenance here, because we already filtered on
            #   provenance in building the temp_find_containing or temp_find_overlapping table.
            fields = [ { 'field': 'project',             'val': project,        'type': 'list' },
                       { 'field': 'target',              'val': target,         'type': 'list' },
                       { 'field': 'section_id',          'val': section_id,     'type': 'list' },
                       { 'field': 'filter',              'val': filter,         'type': 'list' },
                       { 'field': 'instrument',          'val': instrument,     'type': 'list' },
                       { 'field': '_type',               'val': types,          'type': 'list' },
                       { 'field': 'mjd',                 'val': min_mjd,        'type': 'ge' },
                       { 'field': 'mjd',                 'val': max_mjd,        'type': 'le' },
                       { 'field': 'exp_time',            'val': min_exp_time,   'type': 'ge' },
                       { 'field': 'exp_time',            'val': max_exp_time,   'type': 'le' },
                       { 'field': 'fwhm_estimate',       'val': min_seeing,     'type': 'ge' },
                       { 'field': 'fwhm_estimate',       'val': max_seeing,     'type': 'le' },
                       { 'field': 'lim_mag_estimate',    'val': min_lim_mag,    'type': 'ge' },
                       { 'field': 'lim_mag_estimate',    'val': max_lim_mag,    'type': 'le' },
                       { 'field': 'airmass',             'val': min_airmass,    'type': 'ge' },
                       { 'field': 'airmass',             'val': max_airmass,    'type': 'le' },
                       { 'field': 'zero_point_estimate', 'val': min_zero_point, 'type': 'ge' },
                       { 'field': 'zero_point_estimate', 'val': max_zero_point, 'type': 'le' },
                       { 'field': 'bkg_rms_estimate',    'val': min_background, 'type': 'ge' },
                       { 'field': 'bkg_rms_estimate',    'val': max_background, 'type': 'le' } ]
            for field in fields:
                if field['val'] is not None:
                    val = field['val']
                    q += sql.SQL( "  AND i.{field}" ).format( field=sql.Identifier(field['field']) )
                    if field['type'] == 'list':
                        val = listify( val )
                        q += sql.SQL( " = ANY(ARRAY[{mess}])\n" ).format( mess=sql.SQL(",").join( val ) )
                    elif field['type'] == 'ge':
                        q += sql.SQL( " >= {val}\n" ).format( val=val )
                    elif field['type'] == 'le':
                        q += sql.SQL( " <= {val}\n" ).format( val=val )
                    else:
                        raise RuntimeError( f"Unknown field type {field['type']}; this should never happen." )

            # Third: Sort

            if order_by == 'earliest':
                q += sql.SQL( "ORDER BY i.mjd\n" )
            elif order_by == 'latest':
                q += sql.SQL( "ORDER BY i.mjd DESC\n" )
            elif order_by == 'quality':
                q += sql.SQL( "ORDER BY i.lim_mag_estimate - {qf}*i.fwhm_estimate DESC NULLS LAST\n"
                             ).format( qf=np.abs(seeing_quality_factor) )
            elif order_by is not None:
                raise ValueError(f'Unknown order_by parameter: {order_by}. Use "earliest", "latest" or "quality".')

            # Run it to identify image ids of interest
            pgdb.execute_nofetch( q )

            # Get the images
            rows = pgdb.execute( sql.SQL( "SELECT i.* FROM images i "
                                          "INNER JOIN {temptable} t ON i._id=t._id" )
                                 .format( temptable=imgid_temptable ) )
            images = [ Image(**r) for r in rows ]

            # Get the WCSen if necessary
            if return_wcs:
                if provenance_ids_are_wcs:
                    q = sql.SQL( textwrap.dedent(
                        """\
                        SELECT s.image_id, w.* FROM world_coordinates w
                        INNER JOIN source_lists s ON w.sources_id=s._id
                        INNER JOIN {temptable} t ON s.image_id=t._id
                        WHERE w.provenance_id=ANY(ARRAY[{provs}])
                        """
                    ) ).format( provs=sql.SQL(",").join( provenance_ids ), temptable=imgid_temptable )
                elif provenance_ids_are_zp:
                    q = sql.SQL( textwrap.dedent(
                        """\
                        SELECT s.image_id, z.provenance_id AS zpprov, w.* FROM world_coordinates w
                        INNER JOIN zero_points z ON z.wcs_id=w._id
                        INNER JOIN source_lists s ON w.sources_id=s._id
                        INNER JOIN {temptable} t ON s.image_id=t._id
                        WHERE z.provenance_id=ANY(ARRAY[{provs}])
                        """
                    ) ).format( provs=sql.SQL(",").join( provenance_ids ), temptable=imgid_temptable )
                rows = pgdb.execute( q )
                wcsen = {}
                for row in rows:
                    # Gonna treat provenance_ids as a rank-ordered list
                    mustsave = False
                    if row['image_id'] in wcsen:
                        mustsave = ( ( provenance_ids_are_zp and
                                       ( provenance_ids.index( row['zpprov'] )
                                         < provenance_ids.index( wcsen[row['image_id']]['zpprov'] ) ) )
                                     or
                                     ( provenance_ids_are_wcs and
                                       ( provenance_ids.index( row['provenance_id'] )
                                         < provenance_ids.index( wcsen[row['image_id']]['provenance_id'] ) ) )
                                    )
                    else:
                        mustsave = True

                    if mustsave:
                        wcsen[row['image_id']] = row
                        del wcsen[row['image_id']]['image_id']

                if provenance_ids_are_zp:
                    for v in wcsen.values():
                        del v['zpprov']

                # Just in case; avoid circular imports
                from models.world_coordinates import WorldCoordinates
                wcsen = { k: WorldCoordinates(**v) for k, v in wcsen.items() }

            if return_zeropoints:
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT s.image_id, z.* FROM zero_points z
                    INNER JOIN world_coordinates w ON z.wcs_id=w._id
                    INNER JOIN source_lists s ON w.sources_id=s._id
                    INNER JOIN {temptable} t ON s.image_id=t._id
                    WHERE z.provenance_id=ANY(ARRAY[{provs}])
                    """
                ) ).format( provs=sql.SQL(",").join( provenance_ids ), temptable=imgid_temptable )
                rows = pgdb.execute( q )
                zps = {}
                for row in rows:
                    # Again, treat provenance_ids as a rank-ordered list
                    mustsave = False
                    if row['image_id'] in zps:
                        mustsave = ( provenance_ids.index( row['provenance_id'] )
                                     < provenance_ids.index( zps[row['image_id']]['provenance_id'] ) )
                    else:
                        mustsave = True

                    if mustsave:
                        zps[row['image_id']] = row
                        del zps[row['image_id']]['image_id']

                # Just in case; avoid circular imports
                from models.zero_point import ZeroPoint
                zps = { k: ZeroPoint(**v) for k, v in zps.items() }

        # Should we delete temp tables?  They ought to get dropped automatically when the session closes.

        # Fourth: remove things with too small overlap fraction if relevant

        if searchoverlapping:
            retimages = []
            retimgids = set()
            for im in images:
                if FourCorners.get_overlap_frac( fcobj, im ) >= overlapfrac:
                    retimages.append( im )
                    retimgids.add( im._id )
            if return_wcs:
                wcsen = { k: v for k, v in wcsen.items() if k in retimgids }
            if return_zeropoints:
                zps = { k: v for k, v in zps.items() if k in retimgids }
        else:
            retimages = images

        if return_zeropoints:
            if return_wcs:
                return retimages, wcsen, zps
            else:
                return retimages, zps
        elif return_wcs:
            return retimages, wcsen
        else:
            return retimages


    @staticmethod
    def get_coadd_from_components(zps, prov_id=None, pgdb=None, session=None):
        """Finds the combined image with a given provenance that was made from exactly a list of images.

        (The zeropoints point back to wcs which point to sources which point to images.)

        Parameters
        ----------
           zps: list of ZeroPoint or str/UUID
             The zeropoints, or ids of same, that go with the images
             whose sum we're looking for.  Does not need to be sorted.

           prov_id: str or UUID
             The provenance of the coadd image.

        """

        from models.zero_point import ZeroPoint

        if ( prov_id is not None ) and ( isinstance( prov_id, Provenance ) ):
            prov_id = prov_id.id
        zpids = [ i.id if isinstance(i, ZeroPoint) else asUUID(i) for i in zps ]

        with PGDB( pgdb if pgdb is not None else session, dictcursor=True ) as pgdb:
            pgdb.execute( "DROP TABLE IF EXISTS temp_image_from_upstreams" )

            # First get a list of candidate coadd images that are ones whose upstreams
            #   include anything in images, plus a count of how many of
            #   images are in the upstreams.
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT i._id AS imgid, COUNT(c.zp_id) AS nmatchupstr
                INTO TEMP TABLE temp_image_from_upstreams
                FROM images i
                INNER JOIN image_coadd_component c ON c.coadd_image_id=i._id
                WHERE c.zp_id=ANY(ARRAY[{zipds}])
                """
            ) ).format( zipds=sql.SQL(",").join( zpids ) )

            if prov_id is not None:  # pick only those coadds with the right provenance id
                q += sql.SQL( "  AND i.provenance_id={provid}\n" ).format( provid=prov_id )

            q += sql.SQL( "GROUP BY i._id " )
            pgdb.execute( q )

            # Now go through those images and count *all* of the upstreams.
            # (The previous table only counted upstreams that were in zps.)
            # The one (if any) that has len(images) in both the count of
            # matched upstreams and all upstreams is the one we're looking for.
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT imgid FROM (
                  SELECT t.imgid, t.nmatchupstr, COUNT(c.zp_id) AS nupstr
                  FROM temp_image_from_upstreams t
                  INNER JOIN image_coadd_component c ON c.coadd_image_id=t.imgid
                  GROUP BY t.imgid, t.nmatchupstr ) subq
                WHERE nmatchupstr={num} AND nupstr={num}
                """
            ) ).format( num=len(zpids) )
            rows = pgdb.execute( q )

            if len(rows) > 1:
                raise ValueError( f"More than one combined image found with provenance ID {prov_id} "
                                  f"and upstreams {zps}." )
            elif len(rows) == 0:
                return None
            else:
                return Image.get_by_id( rows[0]['imgid'], session=session )


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
        if value is not None:
            self.width = value.shape[1]
            self.height = value.shape[0]
            # zero out inconsistent weight and flags.  This isn't obviously
            # the right thing to do, but doing nothing is obviously wrong.
            if ( self._weight is not None ) and ( self._weight.shape != value.shape ):
                self._weight = None
            if ( self._flags is not None ) and ( self._flags.shape != value.shape ):
                self._flags = None

    @property
    def header(self):
        if ( self._header is None ) and ( self.filepath is not None ):
            self.load()
        # These next two lines are troublesome.  If we later set
        #   a filepath, intending to hook this image up to it, but
        #   we've already accessed the header property, then it won't
        #   load the header from the filepath.  Removing them... and
        #   we'll see if there are things that depend on finding
        #   an empty header if there isn't one set already.
        # if self._header is None:
        #     self._header = fits.Header()
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
        if ( self._data is not None ) and ( value is not None ) and ( self._data.shape != value.shape ):
            raise ValueError( f"Tried to set flags array of shape {value.shape} "
                              f"for image with shape {self._data.shape}" )
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
        if ( self._data is not None ) and ( value is not None ) and ( self._data.shape != value.shape ):
            raise ValueError( f"Tried to set weight array of shape {value.shape} "
                              f"for image with shape {self._data.shape}" )
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


if __name__ == '__main__':
    SCLogger.warning( "Running image.py doesn't actually do anything." )
