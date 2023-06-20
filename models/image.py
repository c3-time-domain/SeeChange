import os
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.types import Enum
from sqlalchemy.dialects.postgresql import JSONB

from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u

from pipeline.utils import read_fits_image, save_fits_image_file

from models.base import SeeChangeBase, Base, FileOnDiskMixin, SpatiallyIndexed
from models.exposure import Exposure, im_type_enum
from models.instrument import get_instrument_instance
from models.provenance import Provenance

import util.config as config

image_source_self_association_table = sa.Table(
    'image_sources',
    Base.metadata,
    sa.Column('source_id', sa.Integer, sa.ForeignKey('images.id', ondelete="CASCADE"), primary_key=True),
    sa.Column('combined_id', sa.Integer, sa.ForeignKey('images.id', ondelete="CASCADE"), primary_key=True),
)


class Image(Base, FileOnDiskMixin, SpatiallyIndexed):

    __tablename__ = 'images'

    exposure_id = sa.Column(
        sa.ForeignKey('exposures.id'),
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

    ref_id = sa.Column(
        sa.ForeignKey('images.id', ondelete="CASCADE"),
        nullable=True,
        index=True,
        doc=(
            "ID of the reference image used to produce a difference image. "
            "Only set for difference images. This usually refers to a coadd image. "
        )
    )

    ref = orm.relationship(
        'Image',
        cascade='save-update, merge, refresh-expire, expunge',
        primaryjoin='images.c.ref_id == images.c.id',
        uselist=False,
        doc=(
            "Reference image used to produce a difference image. "
            "Only set for difference images. This usually refers to a coadd image. "
        )
    )

    new_id = sa.Column(
        sa.ForeignKey('images.id', ondelete="CASCADE"),
        nullable=True,
        index=True,
        doc=(
            "ID of the new image used to produce a difference image. "
            "Only set for difference images. This usually refers to a regular image. "
        )
    )

    new = orm.relationship(
        'Image',
        cascade='save-update, merge, refresh-expire, expunge',
        primaryjoin='images.c.new_id == images.c.id',
        uselist=False,
        doc=(
            "New image used to produce a difference image. "
            "Only set for difference images. This usually refers to a regular image. "
        )
    )

    @property
    def is_multi_image(self):
        if self.exposure is not None:
            return False
        elif self.source_images is not None and len(self.source_images) > 0:
            return True
        elif self.ref_id is not None and self.new_id is not None:
            return True
        else:
            return None  # for new objects that have not defined either exposure or source_images

    @property
    def combine_method(self):
        if self.ref_id is not None and self.new_id is not None:
            return "subtraction"
        elif self.source_images is not None and len(self.source_images) > 0:
            return "coadd"
        elif self.exposure is not None:
            return "single"
        else:
            return None  # for new objects that have not defined either exposure or source_images

    type = sa.Column(
        im_type_enum,  # defined in models/exposure.py
        nullable=False,
        default="science",
        index=True,
        doc=(
            "Type of image. One of: science, reference, difference, bias, dark, flat. "
        )
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE"),
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
            'target'
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

        names = ['ra', 'dec'] + new.instrument_object.get_auxiliary_exposure_header_keys()
        header_info = new.instrument_object.extract_header_info(new._raw_header, names)
        # TODO: get the important keywords translated into the searchable header column

        # figure out the RA/Dec of each image

        # first see if this instrument has a special method of figuring out the RA/Dec
        new.ra, new.dec = new.instrument_object.get_ra_dec_for_section(exposure, section_id)

        # if that fails (which is true for most instruments!), get the RA/Dec from the section header
        if new.ra is None or new.dec is None:
            new.ra = header_info.pop('ra', None)
            new.dec = header_info.pop('dec', None)

        # if that doesn't work, maybe we can read the WCS from the header:
        try:
            wcs = WCS(new._raw_header)
            sc = wcs.pixel_to_world(new._raw_header['NAXIS2'] // 2, new._raw_header['NAXIS1'] // 2)
            new.ra = sc.ra.to(u.deg).value
            new.dec = sc.dec.to(u.deg).value
        except:
            pass  # can't do it, just leave RA/Dec as None

        # just use the RA/Dec of the global exposure
        if new.ra is None or new.dec is None:
            new.ra = exposure.ra
            new.dec = exposure.dec

        new.header = header_info  # save any additional header keys into a JSONB column

        # the exposure_id will be set automatically at commit time
        new.exposure = exposure

        return new

    @property
    def instrument_object(self):
        if self.instrument is not None:
            if self._instrument_object is None or self._instrument_object.name != self.instrument:
                self._instrument_object = get_instrument_instance(self.instrument)

        return self._instrument_object

    @instrument_object.setter
    def instrument_object(self, value):
        self._instrument_object = value

    def __repr__(self):

        output = (
            f"Image(id: {self.id}, "
            f"type: {self.type}, "
            f"exp: {self.exp_time}s, "
            f"filt: {self.filter}, "
            f"from: {self.instrument}/{self.telescope}"
        )

        multi_type = str(self.combine_method) if self.is_multi_image else None

        if multi_type is not None:
            output += f", multi_type: {multi_type}"

        output += ")"

        return output

    def __str__(self):
        return self.__repr__()

    def invent_filename(self):
        """
        Create a filename for the image based on the metadata.
        This is used when saving the image to disk.
        """
        # TODO: we may want to get the naming convention from the config file
        #  in which case we will need to parse it somehow, e.g., using some blocks
        #  like <instrument>, <filter>, etc.

        t = Time(self.mjd, format='mjd', scale='utc').datetime

        short_name = self.instrument_object.get_short_instrument_name()
        date = t.strftime('%Y%m%d')
        time = t.strftime('%H%M%S')
        filter = self.instrument_object.get_short_filter_name(self.filter)

        ra = self.ra
        ra_int, ra_frac = str(float(ra)).split('.')
        ra_int = int(ra_int)
        ra_int_h = ra_int // 15
        ra_frac = int(ra_frac)

        dec = self.dec
        dec_int, dec_frac = str(float(dec)).split('.')
        dec_int = int(dec_int)
        dec_int_pm = f'p{dec_int:02d}' if dec_int >= 0 else f'm{dec_int:02d}'
        dec_frac = int(dec_frac)

        section_id = self.section_id
        prov_id = self.provenance_id

        default_convention = "{short_name}_{date}_{time}_{section_id:02d}_{filter}_{prov_id:03d}"

        cfg = config.Config.get()
        name_convention = cfg.value('storage.images.name_convention', default=None)
        if name_convention is None:
            name_convention = default_convention

        filename = name_convention.format(
            short_name=short_name,
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
            prov_id=prov_id,
        )

        return filename

    def save(self, filename=None):
        """
        Save the data (along with flags, weights, etc.) to disk.
        The format to save is determined by the config file.
        Use the filename to override the default naming convention.

        Parameters
        ----------
        filename: str (optional)
            The filename to use to save the data.
            If not provided, the default naming convention will be used.
        """
        if self.data is None:
            raise RuntimeError("The image data is not loaded. Cannot save.")

        if self.provenance_id is None:
            raise RuntimeError("The image provenance_id is not set. Cannot save.")

        if filename is None:
            filename = self.invent_filename()

        cfg = config.Config.get()
        single_file = cfg.value('storage.images.single_file', default=False)
        format = cfg.value('storage.images.format', default='fits')
        extensions = []

        full_path = os.path.join(self.local_path, filename)

        if format == 'fits':
            # save the imaging data
            extensions.append('.image.fits')  # assume the primary extension has no name
            save_fits_image_file(full_path, self.data, self.raw_header, extname='image', single_file=single_file)
            # TODO: we can have extensions at the end of the filename (e.g., foo.fits.flags)
            #  or we can have the extension name carry the file extension (e.g., foo.flags.fits)
            #  this should be configurable and will affect how we make the filename and extensions.

            # save the other extensions
            array_list = ['flags', 'weight', 'background', 'score', 'psf']
            for array_name in array_list:
                array = getattr(self, array_name)
                if array is not None:
                    save_fits_image_file(
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

            if single_file and not filename.endswith('.fits'):
                filename += '.fits'

        elif format == 'hdf5':
            # TODO: consider writing a more generic utility to save_image_file that handles either fits or hdf5, etc.
            raise RuntimeError("HDF5 format is not yet supported.")
        else:
            raise ValueError(f"Unknown image format: {format}. Use 'fits' or 'hdf5'.")

        self.filepath = filename

        if not single_file:
            self.filepath_extensions = extensions

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
            self._data, self._raw_header = read_fits_image(filename, ext=0, output='both')
            # TODO: do we know what the extensions are for weight/flags/etc? are they ordered or named?
            self._flags = read_fits_image(filename, ext=1)  # TODO: is the flags always extension 1??
            self._weight = read_fits_image(filename, ext=2)  # TODO: is the weight always extension 2??

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
