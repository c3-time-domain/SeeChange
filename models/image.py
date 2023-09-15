import os
import math
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import aliased
from sqlalchemy.sql.functions import coalesce

from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u

from pipeline.utils import read_fits_image, save_fits_image_file

from models.base import SeeChangeBase, Base, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed, FourCorners
from models.exposure import Exposure
from models.instrument import get_instrument_instance
from models.enums_and_bitflags import (
    ImageFormatConverter,
    ImageTypeConverter,
    image_badness_inverse,
    data_badness_dict,
    string_to_bitflag,
    bitflag_to_string,
)

import util.config as config

image_source_self_association_table = sa.Table(
    'image_sources',
    Base.metadata,
    sa.Column('source_id',
              sa.Integer,
              sa.ForeignKey('images.id', ondelete="CASCADE", name='image_sources_source_id_fkey'),
              primary_key=True),
    sa.Column('combined_id',
              sa.Integer,
              sa.ForeignKey('images.id',ondelete="CASCADE", name='image_sources_combined_id_fkey'),
              primary_key=True),
)


class Image(Base, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed, FourCorners):

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

    source_images = orm.relationship(
        "Image",
        secondary=image_source_self_association_table,
        primaryjoin='images.c.id == image_sources.c.combined_id',
        secondaryjoin='images.c.id == image_sources.c.source_id',
        passive_deletes=True,
        lazy='selectin',  # should be able to get source_images without a session!
        order_by='images.c.mjd',  # in chronological order (of the exposure beginnings)
        doc=(
            "Images used to produce a multi-image object "
            "(e.g., an images stack, reference, difference, super-flat, etc)."
        )
    )

    ref_image_id = sa.Column(
        sa.ForeignKey('images.id', ondelete="CASCADE", name='images_ref_image_id_fkey'),
        nullable=True,
        index=True,
        doc=(
            "ID of the reference image used to produce a difference image. "
            "Only set for difference images. This usually refers to a coadd image. "
        )
    )

    ref_image = orm.relationship(
        'Image',
        cascade='save-update, merge, refresh-expire, expunge',
        primaryjoin='images.c.ref_image_id == images.c.id',
        uselist=False,
        remote_side='Image.id',
        doc=(
            "Reference image used to produce a difference image. "
            "Only set for difference images. This usually refers to a coadd image. "
        )
    )

    new_image_id = sa.Column(
        sa.ForeignKey('images.id', ondelete="CASCADE", name='images_new_image_id_fkey'),
        nullable=True,
        index=True,
        doc=(
            "ID of the new image used to produce a difference image. "
            "Only set for difference images. This usually refers to a regular image. "
        )
    )

    new_image = orm.relationship(
        'Image',
        cascade='save-update, merge, refresh-expire, expunge',
        primaryjoin='images.c.new_image_id == images.c.id',
        uselist=False,
        remote_side='Image.id',
        doc=(
            "New image used to produce a difference image. "
            "Only set for difference images. This usually refers to a regular image. "
        )
    )

    @property
    def is_coadd(self):
        try:
            if self.source_images is not None and len(self.source_images) > 0:
                return True
        except DetachedInstanceError:
            if not self.is_sub and self.exposure_id is None:
                return True

        return False

    @hybrid_property
    def is_sub(self):
        try:
            if self.ref_image is not None and self.new_image is not None:
                return True
        except DetachedInstanceError:
            if self.ref_image_id is not None and self.new_image_id is not None:
                return True

        return False

    @is_sub.inplace.expression
    @classmethod
    def is_sub(cls):
        return cls.ref_image_id.isnot(None) & cls.new_image_id.isnot(None)

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=ImageTypeConverter.convert('Sci'),
        index=True,
        doc=(
            "Type of image. One of: Sci, Diff, Bias, Dark, DomeFlat, SkyFlat, TwiFlat, "
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
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this image. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this image. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this image. "
        )
    )

    header = sa.Column(
        JSONB,
        nullable=False,
        default={},
        doc=(
            "Header of the specific image for one section of the instrument. "
            "Only keep a subset of the keywords, "
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
        """
        Time of the middle of the exposures.
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
        sa.Float,
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

    filter = sa.Column(sa.Text, nullable=False, index=True, doc="Name of the filter used to make this image. ")

    section_id = sa.Column(
        sa.Text,
        nullable=False,
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

    _bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        default=0,
        index=True,
        doc='Bitflag for this image. Good images have a bitflag of 0. '
            'Bad images are each bad in their own way (i.e., have different bits set). '
            'The bitflag will include this value, bit-wise-or-ed with the bitflags of the '
            'exposure or images used to make this image. '
    )

    @hybrid_property
    def bitflag(self):
        bf = self._bitflag
        if self.source_images is not None and len(self.source_images) > 0:
            for img in self.source_images:
                if img.bitflag is not None:
                    bf |= img.bitflag
        elif self.ref_image is not None and self.new_image is not None:
            if self.ref_image.bitflag is not None:
                bf |= self.ref_image.bitflag
            if self.new_image.bitflag is not None:
                bf |= self.new_image.bitflag
        elif self.exposure is not None:
            if self.exposure.bitflag is not None:
                bf |= self.exposure.bitflag
        else:
            raise RuntimeError('Cannot get bitflag without source images, ref_image, new_image, or exposure.')

        return bf

    @bitflag.inplace.expression
    @classmethod
    def bitflag(cls):

        class ImTable:
            """
            This is a helper class that defines the recursive relationship
            between Image and its parents. Some images will have an Exposure parent,
            some will have a ref/new pair of images as parents, and some will have
            a list of source images as parents. Each instance of ImTable will contain
            aliased versions of the Image and Exposure tables, and will have a role
            to play (is it a new image, a ref image, a source image, etc).
            The role helps tell the following code how to join the SQL tables together
            when we construct the select expression.


            """
            def __init__(self, role):
                """
                Create a new ImTable with a given role.
                The roles can be "first", "ref", "new", and "src".
                The first is the Image table we are querying on.
                The ref/new are images that act as the pair used for subtraction.
                The src are a list of images used to build up a coadd.

                The self.dad attribute links back to the ImTable above this one.
                The self.ref, self.new and self.src are references to the ImTable
                objects below this one.
                Thus we get a linked tree of ImTables that forms the blueprint for
                the massive join we want to make in the end.
                """
                self.role = role
                self.img = aliased(Image)
                self.exp = aliased(Exposure)
                self.dad = None
                self.ref = None
                self.new = None
                self.src = None
                if role == 'src':
                    self.association_table = aliased(image_source_self_association_table)

            def make_children(self, iterations=3):
                """
                Recursively make the child ImTable objects for this ImTable.
                Will stop when iterations reach 0.
                """
                if iterations < 1:
                    return

                self.ref = ImTable(role='ref')
                self.ref.dad = self
                self.ref.make_children(iterations=iterations-1)

                self.new = ImTable(role='new')
                self.new.dad = self
                self.new.make_children(iterations=iterations-1)

                self.src = ImTable(role='src')
                self.src.dad = self
                self.src.make_children(iterations=iterations-1)

            def add_to_columns(self, col=None):
                """
                Add the column that we want to select. If col=None will just create a column
                based on the current img._bitflag and its associated exp._bitflag.
                This only happens for the first ImTable (the one we are querying on).

                If col is given, it is "or'ed" together using op.('|')(bit_or(...)).
                The internal bit_or is used to aggregate the bitflags of an array of images
                (e.g., the source images).

                The resulting col is returned, so it can be used recursively
                or just put into the select statement in the end.

                Since we use coalesce(bitflag, 0) and bit_or(bitflag) any null
                values in these tables will be ignored/replaced by zero.
                So only if we have a non-zero value in any of the bitflags
                will the final bitflag be non-zero (it will be the bit or of all bitflags).
                """
                if self.role == 'first':
                    return self.img._bitflag.op('|')(coalesce(self.exp._bitflag, 0))
                else:
                    return col.op('|')(
                        coalesce(sa.func.bit_or(self.img._bitflag), 0).op('|')(
                            coalesce(sa.func.bit_or(self.exp._bitflag), 0)
                        )
                    )

            def add_children_to_columns(self, col):
                """
                Applies the add_to_column to the children ImTables.
                This is re-applied recursively to the children's children
                until a generation is reached that has None as children.
                """
                if self.ref is None or self.new is None or self.src is None:
                    return col

                col = self.ref.add_to_columns(col)
                col = self.new.add_to_columns(col)
                col = self.src.add_to_columns(col)

                col = self.ref.add_children_to_columns(col)
                col = self.new.add_children_to_columns(col)
                col = self.src.add_children_to_columns(col)

                return col

            def add_to_joins(self, stmt):
                """
                Updates the "stmt" (select statement) with the joins needed
                to get the data from more and more aliased versions of Image
                and Exposure tables, so we get the bitflag of all the objects
                that were used to make the image we are querying on.

                We outerjoin the exposures, ref/new pairs and the source images,
                and wherever there is no parent of that type we just allow a null row.
                Those null rows are not included in the bit_or aggregate function,
                and in case all rows are null, we use coalesce to get 0 instead of null.

                Each role has a different join condition, and the joins are applied
                recursively to the children ImTables.

                For any of the roles, after we join the correct Image table,
                we need to also join the Exposures used to make (some) of those images.

                The function returns the "stmt" after modifying it, so it can be used
                recursively and at the end it can be used as a select statement for
                the hybrid_property expression.
                """
                if self.role == 'first':
                    stmt = stmt.select_from(self.img)
                elif self.role == 'ref':
                    stmt = stmt.outerjoin(
                        self.img, self.dad.img.ref_image_id == self.img.id
                    )
                elif self.role == 'new':
                    stmt = stmt.outerjoin(
                        self.img, self.dad.img.new_image_id == self.img.id
                    )
                elif self.role == 'src':
                    stmt = stmt.outerjoin(
                        self.association_table, self.association_table.c.combined_id == self.dad.img.id
                    ).outerjoin(
                        self.img, self.association_table.c.source_id == self.img.id
                    )
                # also add the exposures for each image
                stmt = stmt.outerjoin(
                        self.exp, self.img.exposure_id == self.exp.id
                    )
                return stmt

            def add_children_to_joins(self, stmt):
                """
                Recursively apply the joins of add_to_joins
                to all of this ImTable's children, and to their
                children, until a generation of children that has
                None as their children.
                """
                if self.ref is None or self.new is None or self.src is None:
                    return stmt

                stmt = self.ref.add_to_joins(stmt)
                stmt = self.new.add_to_joins(stmt)
                stmt = self.src.add_to_joins(stmt)

                stmt = self.ref.add_children_to_joins(stmt)
                stmt = self.new.add_children_to_joins(stmt)
                stmt = self.src.add_children_to_joins(stmt)

                return stmt

        # end of ImTable class
        # start of the bitflag expression built up from those ImTables:

        # define a recursive structure with X number of layers
        # each Image has zero or one Exposures, but can also
        # have one ref/new pair of images, or a list of source images
        # each of those images (ref, new, sources) has their own parents,
        # up to the number of recursions allowed.
        # This step only builds up the blueprint for the relationships
        # between the different aliased versions of the Image table (and Exposure table).
        first_table = ImTable(role='first')
        first_table.make_children(iterations=3)  # recursively add three layers of children

        # this is where we add the column that needs to get selected
        # e.g., the _bitflag of each table, using op('|') to combine them
        columns = first_table.add_to_columns()
        columns = first_table.add_children_to_columns(columns)

        # here we join more and more tables in, including
        # the exposures associated with each image,
        # and the images recursively associated with other images
        stmt = sa.select(columns)
        stmt = first_table.add_to_joins(stmt)
        stmt = first_table.add_children_to_joins(stmt)

        # since we aggregate every associated image/exposure using bit_or,
        # we need to make sure to group by the top level image and exposure combination
        stmt = stmt.group_by(first_table.img.id, first_table.exp.id)

        # this last where makes sure that when querying on Image externally
        # each Image row will correspond to one result, hence the scalar_subquery
        stmt = stmt.where(cls.id == first_table.img.id).scalar_subquery()

        return stmt

    @bitflag.inplace.setter
    def bitflag(self, value):
        allowed_bits = 0
        for i in image_badness_inverse.values():
            allowed_bits += 2 ** i
        if value & ~allowed_bits != 0:
            raise ValueError(f'Bitflag value {bin(value)} has bits set that are not allowed.')
        self._bitflag = value

    @property
    def badness(self):
        """
        A comma separated string of keywords describing
        why this data is not good, based on the bitflag.
        This includes all the reasons this data is bad,
        including the parent data models that were used
        to create this data (e.g., the Exposure underlying
        the Image).
        """
        return bitflag_to_string(self.bitflag, data_badness_dict)

    @badness.setter
    def badness(self, value):
        """Set the badness for this image using a comma separated string. """
        self.bitflag = string_to_bitflag(value, image_badness_inverse)

    def append_badness(self, value):
        """Add some keywords (in a comma separated string)
        describing what is bad about this image.
        The keywords will be added to the list "badness"
        and the bitflag for this image will be updated accordingly.
        """
        self.bitflag |= string_to_bitflag(value, image_badness_inverse)

    description = sa.Column(
        sa.Text,
        nullable=True,
        doc='Free text comment about this image, e.g., why it is bad. '
    )

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self.raw_data = None  # the raw exposure pixels (2D float or uint16 or whatever) not saved to disk!
        self._raw_header = None  # the header data taken directly from the FITS file
        self._data = None  # the underlying pixel data array (2D float array)
        self._flags = None  # the bit-flag array (2D int array)
        self._weight = None  # the inverse-variance array (2D float array)
        self._background = None  # an estimate for the background flux (2D float array)
        self._score = None  # the image after filtering with the PSF and normalizing to S/N units (2D float array)
        self._psf = None  # a small point-spread-function image (2D float array)

        self._instrument_object = None
        self._bitflag = 0

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.ra is not None and self.dec is not None:
            self.calculate_coordinates()  # galactic and ecliptic coordinates

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)
        self.raw_data = None
        self._raw_header = None
        self._data = None
        self._flags = None
        self._weight = None
        self._background = None
        self._score = None
        self._psf = None

        self._instrument_object = None

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

        new.section_id = section_id
        new.raw_data = exposure.data[section_id]
        new.instrument_object = exposure.instrument_object

        # read the header from the exposure file's individual section data
        new._raw_header = exposure.section_headers[section_id]

        # numpy array axis ordering is backwards from FITS ordering
        width = new.raw_data.shape[1]
        height = new.raw_data.shape[0]

        names = ['ra', 'dec'] + new.instrument_object.get_auxiliary_exposure_header_keys()
        header_info = new.instrument_object.extract_header_info(new._raw_header, names)
        # TODO: get the important keywords translated into the searchable header column

        # figure out the RA/Dec of each image

        # first see if this instrument has a special method of figuring out the RA/Dec
        new.ra, new.dec = new.instrument_object.get_ra_dec_for_section(exposure, section_id)

        # if not (which is true for most instruments!), try to read the WCS from the header:
        try:
            wcs = WCS(new._raw_header)
            sc = wcs.pixel_to_world(width // 2, height // 2)
            new.ra = sc.ra.to(u.deg).value
            new.dec = sc.dec.to(u.deg).value
        except:
            pass  # can't do it, just leave RA/Dec as None

        # if that fails, try to get the RA/Dec from the section header
        if new.ra is None or new.dec is None:
            new.ra = header_info.pop('ra', None)
            new.dec = header_info.pop('dec', None)

        # if that fails, just use the RA/Dec of the global exposure
        if new.ra is None or new.dec is None:
            new.ra = exposure.ra
            new.dec = exposure.dec

        new.header = header_info  # save any additional header keys into a JSONB column

        # Figure out the 4 corners  Start by trying to use the WCS
        gotcorners = False
        try:
            wcs = WCS( new._raw_header )
            ras = []
            decs = []
            xs = [ 0., width-1., 0., width-1. ]
            ys = [ 0., height-1., height-1., 0. ]
            scs = wcs.pixel_to_world( xs, ys )
            ras = [ i.ra.value_in(u.deg).value for i in scs ]
            decs = [ i.dec.value_in(u.deg).value for i in scs ]
            ras, decs = FourCorners.sort_radec( ras, decs )
            new.ra_corner_00 = ras[0]
            new.ra_corner_01 = ras[1]
            new.ra_corner_10 = ras[2]
            new.ra_corner_11 = ras[3]
            new.dec_corner_00 = decs[0]
            new.dec_corner_01 = decs[1]
            new.dec_corner_10 = decs[2]
            new.dec_corner_11 = decs[3]
            _logger.debug( 'Got corners from WCS' )
            gotcorners = True
        except:
            pass

        # If that didn't work, then use ra and dec and the instrument scale
        # TODO : take into account standard instrument orientation!
        # (Can be done after the decam_pull PR is merged)

        if not gotcorners:
            halfwid = new.instrument_object.pixel_scale * width / 2. / math.cos( new.dec * math.pi / 180. ) / 3600.
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
            gotcorners = True

        # the exposure_id will be set automatically at commit time
        new.exposure = exposure

        return new

    @classmethod
    def from_images(cls, images):
        """
        Create a new Image object from a list of other Image objects.
        This is the first step in making a multi-image (usually a coadd).
        The output image doesn't have any data, and is created with
        nofile=True. It is up to the calling application to fill in the
        data, flags, weight, etc. using the appropriate preprocessing tools.
        After that, the data needs to be saved to file, and only then
        can the new Image be added to the database.

        Parameters
        ----------
        images: list of Image objects
            The images to combine into a new Image object.

        Returns
        -------
        output: Image
            The new Image object. It would not have any data variables or filepath.
        """
        if len(images) < 1:
            raise ValueError("Must provide at least one image to combine.")

        output = Image(nofile=True)

        # for each attribute, check that all the images have the same value
        for att in ['section_id', 'instrument', 'telescope', 'type', 'filter', 'project', 'target']:
            values = set([getattr(image, att) for image in images])
            if len(values) != 1:
                raise ValueError(f"Cannot combine images with different {att} values: {values}")
            output.__setattr__(att, values.pop())
        # TODO: should RA and Dec also be exactly the same??
        output.ra = images[0].ra
        output.dec = images[0].dec
        output.ra_corner_00 = images[0].ra_corner_00
        output.ra_corner_01 = images[0].ra_corner_01
        output.ra_corner_10 = images[0].ra_corner_10
        output.ra_corner_11 = images[0].ra_corner_11
        output.dec_corner_00 = images[0].dec_corner_00
        output.dec_corner_01 = images[0].dec_corner_01
        output.dec_corner_10 = images[0].dec_corner_10
        output.dec_corner_11 = images[0].dec_corner_11

        # exposure time is usually added together
        output.exp_time = sum([image.exp_time for image in images])

        # start MJD and end MJD
        output.mjd = min([image.mjd for image in images])
        output.end_mjd = max([image.end_mjd for image in images])

        # TODO: what about the header? should we combine them somehow?
        output.header = images[0].header
        output.raw_header = images[0].raw_header

        output.source_images = images
        base_type = images[0].type
        if not base_type.startswith('Com'):
            output.type = 'Com' + base_type

        # Note that "data" is not filled by this method, also the provenance is empty!
        return output

    @classmethod
    def from_ref_and_new(cls, ref, new):
        """
        Create a new Image object from a reference Image object and a new Image object.
        This is the first step in making a difference image.
        The output image doesn't have any data, and is created with
        nofile=True. It is up to the calling application to fill in the
        data, flags, weight, etc. using the appropriate preprocessing tools.
        After that, the data needs to be saved to file, and only then
        can the new Image be added to the database.

        Parameters
        ----------
        ref: Image object
            The reference image to use.
        new: Image object
            The new image to use.

        Returns
        -------
        output: Image
            The new Image object. It would not have any data variables or filepath.
        """
        output = Image(nofile=True)

        # for each attribute, check the two images have the same value
        for att in ['section_id', 'instrument', 'telescope', 'filter', 'project', 'target']:
            ref_value = getattr(ref, att)
            new_value = getattr(new, att)

            if att == 'section_id':
                ref_value = str(ref_value)
                new_value = str(new_value)
            if ref_value != new_value:
                raise ValueError(
                    f"Cannot combine images with different {att} values: "
                    f"{ref_value} and {new_value}. "
                )
            output.__setattr__(att, new_value)
        # TODO: should RA and Dec also be exactly the same??

        # get some more attributes from the new image
        for att in ['exp_time', 'mjd', 'end_mjd', 'header', 'raw_header', 'ra', 'dec',
                    'ra_corner_00', 'ra_corner_01', 'ra_corner_10', 'ra_corner_11',
                    'dec_corner_00', 'dec_corner_01', 'dec_corner_10', 'dec_corner_11' ]:
            output.__setattr__(att, getattr(new, att))

        output.ref_image = ref
        output.new_image = new
        output.type = 'Diff'
        if new.type.startswith('Com'):
            output.type = 'ComDiff'

        # Note that "data" is not filled by this method, also the provenance is empty!
        return output

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

    def invent_filename(self):
        """
        Create a filename for the object based on its metadata.
        This is used when saving the image to disk.
        Data products that depend on an image and are also
        saved to disk (e.g., SourceList) will just append
        another string to the Image filename.
        """
        prov_hash = inst_name = im_type = date = time = filter = ra = dec = dec_int_pm = ''
        section_id = section_id_int = ra_int = ra_int_h = ra_frac = dec_int = dec_frac = 0

        if self.provenance is not None:
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
                section_id_int = 0

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
            dec_int_pm = f'p{dec_int:02d}' if dec_int >= 0 else f'm{dec_int:02d}'
            dec_frac = int(dec_frac)

        cfg = config.Config.get()
        default_convention = "{inst_name}_{date}_{time}_{section_id}_{filter}_{im_type}_{prov_hash:.6s}"
        name_convention = cfg.value('storage.images.name_convention', default=None)
        if name_convention is None:
            name_convention = default_convention

        filename = name_convention.format(
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
        return filename

    def save(self, filename=None, **kwargs ):
        """Save the data (along with flags, weights, etc.) to disk.
        The format to save is determined by the config file.
        Use the filename to override the default naming convention.

        Parameters
        ----------
        filename: str (optional)
            The filename to use to save the data.
            If not provided, the default naming convention will be used.
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

        self.filepath = filename if filename is not None else self.invent_filename()

        cfg = config.Config.get()
        single_file = cfg.value('storage.images.single_file', default=False)
        format = cfg.value('storage.images.format', default='fits')
        extensions = []
        files_written = {}

        full_path = os.path.join(self.local_path, self.filepath)

        if format == 'fits':
            # save the imaging data
            extensions.append('.image.fits')  # assume the primary extension has no name
            imgpath = save_fits_image_file(full_path, self.data, self.raw_header,
                                           extname='image', single_file=single_file)
            files_written['.image.fits'] = imgpath
            # TODO: we can have extensions at the end of the self.filepath (e.g., foo.fits.flags)
            #  or we can have the extension name carry the file extension (e.g., foo.flags.fits)
            #  this should be configurable and will affect how we make the self.filepath and extensions.

            # save the other extensions
            array_list = ['flags', 'weight', 'background', 'score', 'psf']
            for array_name in array_list:
                array = getattr(self, array_name)
                if array is not None:
                    extpath = save_fits_image_file(
                        full_path,
                        array,
                        self.raw_header,
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
            raise RuntimeError("HDF5 format is not yet supported.")
        else:
            raise ValueError(f"Unknown image format: {format}. Use 'fits' or 'hdf5'.")

        # Save the file to the archive and update the database record
        # (as well as self.filepath, self.filepath_extensions, self.md5sum, self.md5sum_extensions)
        # (From what we did above, it's already in the right place in the local filestore.)
        if single_file:
            super().save( files_written, **kwargs )
        else:
            for ext in extensions:
                super().save( files_written[ext], ext, **kwargs )

    def load(self):
        """
        Load the image data from disk.
        This includes the _data property,
        but can also load the _flags, _weight,
        _background, _score, and _psf properties.

        """

        if self.filepath is None:
            raise ValueError("The filepath is not set. Cannot load the image.")

        cfg = config.Config.get()
        single_file = cfg.value('storage.images.single_file')

        if single_file:
            filename = self.get_fullpath()
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"Could not find the image file: {filename}")
            self._data, self._raw_header = read_fits_image(filename, ext='image', output='both')
            self._flags = read_fits_image(filename, ext='flags')
            self._weight = read_fits_image(filename, ext='weight')
            self._background = read_fits_image(filename, ext='background')
            self._score = read_fits_image(filename, ext='score')
            # TODO: add more if needed!

        else:  # save each data array to a separate file
            # assumes there are filepath_extensions so get_fullpath() will return a list (as_list guarantees this)
            for filename in self.get_fullpath(as_list=True):
                if not os.path.isfile(filename):
                    raise FileNotFoundError(f"Could not find the image file: {filename}")

                self._data, self._raw_header = read_fits_image(filename, output='both')

    @property
    def data(self):
        """
        The underlying pixel data array (2D float array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def raw_header(self):
        if self._raw_header is None and self.filepath is not None:
            self.load()
        return self._raw_header

    @raw_header.setter
    def raw_header(self, value):
        self._raw_header = value

    @property
    def flags(self):
        """
        The bit-flag array (2D int array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._flags

    @flags.setter
    def flags(self, value):
        self._flags = value

    @property
    def weight(self):
        """
        The inverse-variance array (2D float array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def background(self):
        """
        An estimate for the background flux (2D float array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._background

    @background.setter
    def background(self, value):
        self._background = value

    @property
    def score(self):
        """
        The image after filtering with the PSF and normalizing to S/N units (2D float array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def psf(self):
        """
        A small point-spread-function image (2D float array).
        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._psf

    @psf.setter
    def psf(self, value):
        self._psf = value


if __name__ == '__main__':
    filename = '/home/guyn/Dropbox/python/SeeChange/data/DECam_examples/c4d_221104_074232_ori.fits.fz'
    e = Exposure(filename)
    im = Image.from_exposure(e, section_id=1)

# this is an example for the recursive select emitted by the Image.bitflag hybrid property expression
# if you set the recursion level to 1, this is the expression you are supposed to get
# I wrote this first, then made it into a recursive function using the ImTable helper class
# if you need some help figuring out what is going on in that class then this may be useful:

# from sqlalchemy import func
# from sqlalchemy.orm import aliased
# from sqlalchemy.sql import coalesce

# first_images = aliased(Image)
# first_exposures = aliased(Exposure)
# ref_images = aliased(Image)
# ref_exposures = aliased(Exposure)
# new_images = aliased(Image)
# new_exposures = aliased(Exposure)
# source_images = aliased(Image)
# source_exposures = aliased(Exposure)
#
# stmt = sa.select(
#     coalesce(first_images._bitflag, 0).op('|')(
#         coalesce(first_exposures._bitflag, 0)
#     ).op('|')(
#         coalesce(bit_or(ref_images._bitflag), 0)
#     ).op('|')(
#         coalesce(bit_or(ref_exposures._bitflag), 0)
#     ).op('|')(
#         coalesce(bit_or(new_images._bitflag), 0)
#     ).op('|')(
#         coalesce(bit_or(new_exposures._bitflag), 0)
#     ).op('|')(
#         coalesce(bit_or(source_images._bitflag), 0)
#     ).op('|')(
#         coalesce(bit_or(source_exposures._bitflag), 0)
#     )
# ).select_from(Image).outerjoin(
#     first_exposures, first_exposures.id == first_images.exposure_id
# ).outerjoin(
#     ref_images, ref_images.id == first_images.ref_image_id
# ).outerjoin(
#     ref_exposures, ref_exposures.id == ref_images.exposure_id
# ).outerjoin(
#     new_images, new_images.id == first_images.new_image_id
# ).outerjoin(
#     new_exposures, new_exposures.id == new_images.exposure_id
# ).outerjoin(
#     image_source_self_association_table, image_source_self_association_table.c.combined_id == first_images.id,
# ).outerjoin(
#     source_images, sa.and_(image_source_self_association_table.c.source_id == source_images.id)
# ).outerjoin(
#     source_exposures, source_exposures.id == source_images.exposure_id
# ).group_by(
#     first_images.id, first_exposures.id
# ).where(Image.id == first_images.id).scalar_subquery()
