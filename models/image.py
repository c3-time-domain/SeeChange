import os
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.types import Enum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm.exc import DetachedInstanceError

from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u

from pipeline.utils import read_fits_image, save_fits_image_file

from models.base import SeeChangeBase, Base, FileOnDiskMixin, SpatiallyIndexed
from models.exposure import Exposure, im_type_enum, im_format_enum
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

    format = sa.Column(
        im_format_enum,
        nullable=False,
        default='fits',
        doc="Format of the image on disk. Should be fits or hdf5. "
    )

    exposure_id = sa.Column(
        sa.ForeignKey('exposures.id', ondelete='SET NULL'),
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
        sa.ForeignKey('images.id', ondelete="CASCADE"),
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
        sa.ForeignKey('images.id', ondelete="CASCADE"),
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

    @property
    def is_sub(self):
        try:
            if self.ref_image is not None and self.new_image is not None:
                return True
        except DetachedInstanceError:
            if self.ref_image_id is not None and self.new_image_id is not None:
                return True

        return False

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

        # exposure time is usually added together
        output.exp_time = sum([image.exp_time for image in images])

        # start MJD and end MJD
        output.mjd = min([image.mjd for image in images])
        output.end_mjd = max([image.end_mjd for image in images])

        # TODO: what about the header? should we combine them somehow?
        output.header = images[0].header
        output.raw_header = images[0].raw_header

        output.source_images = images

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
        for att in ['section_id', 'instrument', 'telescope', 'type', 'filter', 'project', 'target']:
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
        for att in ['exp_time', 'mjd', 'end_mjd', 'header', 'raw_header', 'ra', 'dec']:
            output.__setattr__(att, getattr(new, att))

        output.ref_image = ref
        output.new_image = new

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

    def __repr__(self):

        type_str = self.type
        if self.is_coadd:
            type_str += " (coadd)"

        if self.is_sub:
            type_str += " (sub)"

        output = (
            f"Image(id: {self.id}, "
            f"type: {type_str}, "
            f"exp: {self.exp_time}s, "
            f"filt: {self.filter}, "
            f"from: {self.instrument}/{self.telescope}"
        )

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

        if self.provenance is None:
            raise ValueError("Cannot invent filename for image without provenance.")

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
        prov_hash = self.provenance.unique_hash

        default_convention = "{short_name}_{date}_{time}_{section_id}_{filter}_{prov_hash:.6s}"

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
            prov_hash=prov_hash,
        )

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
            self._data, self._raw_header = read_fits_image(filename, ext=0, output='both')
            # TODO: do we know what the extensions are for weight/flags/etc? are they ordered or named?
            # Rob: we should have standard names for them; there is already a standard set of names
            #   in save() above: ['flags', 'weight', 'background', 'score', 'psf']
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
