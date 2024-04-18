import pathlib
from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import CheckConstraint
from sqlalchemy.orm.session import object_session
from sqlalchemy.ext.hybrid import hybrid_property

from astropy.time import Time
from astropy.io import fits

from util.config import Config
from util.util import read_fits_image 
from util.radec import parse_ra_hms_to_deg, parse_dec_dms_to_deg

from models.base import (
    Base,
    SeeChangeBase,
    AutoIDMixin,
    FileOnDiskMixin,
    SpatiallyIndexed,
    SmartSession,
    HasBitFlagBadness,
    _logger
)
from models.instrument import guess_instrument, get_instrument_instance
from models.provenance import Provenance

from models.enums_and_bitflags import (
    ImageFormatConverter,
    ImageTypeConverter,
    image_badness_inverse,
    data_badness_dict,
    string_to_bitflag,
    bitflag_to_string,
)

# columns key names that must be loaded from the header for each Exposure
EXPOSURE_COLUMN_NAMES = [
    'ra',
    'dec',
    'mjd',
    'project',
    'target',
    'exp_time',
    'filter',
    'telescope',
    'instrument'
]

# these are header keywords that are not stored as columns of the Exposure table,
# but are still useful to keep around inside the "info" JSONB column.
EXPOSURE_HEADER_KEYS = ['gain']  # TODO: add more here


class SectionData:
    """
    A helper class that lazy loads the section data from the database.
    When requesting one of the section IDs it will fetch that data from
    disk and store it in memory.
    To clear the memory cache, call the clear_cache() method.
    """
    def __init__(self, filepath, instrument):
        """
        Must initialize this object with a filepath
        (or list of filepaths) and an instrument object.
        These two things will control how data is loaded
        from the disk.

        Parameters
        ----------
        filepath: str or list of str
            The filepath of the exposure to load.
            If each section is in a different file, then
            this should be a list of filepaths.
        instrument: Instrument
            The instrument object that describes the
            sections and how to load them from disk.

        """
        self.filepath = filepath
        self.instrument = instrument
        self._data = defaultdict(lambda: None)

    def __getitem__(self, section_id):
        if self._data[section_id] is None:
            self._data[section_id] = self.instrument.load_section_image(self.filepath, section_id)
        return self._data[section_id]

    def __setitem__(self, section_id, value):
        self._data[section_id] = value

    def clear_cache(self):
        self._data = defaultdict(lambda: None)


class SectionHeaders:
    """
    A helper class that lazy loads the section header from the database.
    When requesting one of the section IDs it will fetch the header
    for that section, load it from disk and store it in memory.
    To clear the memory cache, call the clear_cache() method.
    """
    def __init__(self, filepath, instrument):
        """
        Must initialize this object with a filepath
        (or list of filepaths) and an instrument object.
        These two things will control how data is loaded
        from the disk.

        Parameters
        ----------
        filepath: str or list of str
            The filepath of the exposure to load.
            If each section is in a different file, then
            this should be a list of filepaths.
        instrument: Instrument
            The instrument object that describes the
            sections and how to load them from disk.

        """
        self.filepath = filepath
        self.instrument = instrument
        self._header = defaultdict(lambda: None)  # each item here is a lazy-loaded fits.Header

    def __getitem__(self, section_id):
        if self._header[section_id] is None:
            self._header[section_id] = self.instrument.read_header(self.filepath, section_id)
        return self._header[section_id]

    def __setitem__(self, section_id, value):
        self.header[section_id] = value

    def clear_cache(self):
        self._header = defaultdict(lambda: None)


class ExposureImageIterator:
    """A class to iterate through the HDUs of an exposure, one for each SensorSection."""

    def __iter__( self, exposure ):
        self.exposure = exposure

        self.instrument = get_instrument_instance( self.exposure.instrument )
        self.section_ids = self.instrument.get_section_ids()
        self.dex = 0
        return self

    def __next__( self ):
        from models.image import Image  # avoid circular import
        if self.dex < len( self.section_ids ):
            img = Image.from_exposure( self.exposure, self.section_ids[ self.dex ] )
            self.dex += 1
            return img
        else:
            raise StopIteration


class Exposure(Base, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed, HasBitFlagBadness):

    __tablename__ = "exposures"

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=ImageTypeConverter.convert('Sci'),
        index=True,
        doc=(
            "Type of image. One of: Sci, Diff, Bias, Dark, DomeFlat, SkyFlat, TwiFlat, "
            "or any of the above types prepended with 'Com' for combined "
            "(e.g., a ComSci image is a science image combined from multiple exposures)."
            "The value is saved as SMALLINT but translated to a string when read. "
        )
    )

    @hybrid_property
    def type(self):
        return ImageTypeConverter.convert(self._type)

    @type.expression
    def type(cls):
        return sa.case(ImageTypeConverter.dict, value=cls._type)

    @type.setter
    def type(self, value):
        self._type = ImageTypeConverter.convert(value)

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=ImageFormatConverter.convert('fits'),
        doc="Format of the file on disk. Should be fits or hdf5. "
            "The value is saved as SMALLINT but translated to a string when read. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this exposure. "
            "The provenance will containe a record of the code version "
            "and the parameters used to obtain this exposure."
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this exposure. "
            "The provenance will containe a record of the code version "
            "and the parameters used to obtain this exposure."
        )
    )

    @hybrid_property
    def format(self):
        return ImageFormatConverter.convert(self._format)

    @format.expression
    def format(cls):
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(ImageFormatConverter.dict, value=cls._format)

    @format.setter
    def format(self, value):
        self._format = ImageFormatConverter.convert(value)

    info = sa.Column(
        JSONB,
        nullable=False,
        default={},
        doc=(
            "Subset of the raw exposure's header. "
            "Only keep a subset of the keywords, "
            "and re-key them to be more consistent. "
            "This will only include global values, "
            "not those associated with a specific section. "
        )
    )

    mjd = sa.Column(
        sa.Double,
        nullable=False,
        index=True,
        doc="Modified Julian date of the start of the exposure (MJD=JD-2400000.5)."
    )

    exp_time = sa.Column(sa.Float, nullable=False, index=True, doc="Exposure time in seconds. ")

    filter = sa.Column(sa.Text, nullable=True, index=True, doc="Name of the filter used to make this exposure. ")

    @property
    def filter_short(self):
        if self.filter is None:
            return None
        return self.instrument_object.get_short_filter_name(self.filter)

    filter_array = sa.Column(
        sa.ARRAY(sa.Text),
        nullable=True,
        index=True,
        doc="Array of filter names, if multiple filters were used. "
    )

    __table_args__ = (
        CheckConstraint(
            sqltext='NOT(filter IS NULL AND filter_array IS NULL)',
            name='exposures_filter_or_array_check'
        ),
    )

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='Name of the instrument used to take the exposure. '
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
        doc='Bitflag for this exposure. Good exposures have a bitflag of 0. '
            'Bad exposures are each bad in their own way (i.e., have different bits set). '
    )

    origin_identifier = sa.Column(
        sa.Text,
        nullable=True,
        index=True,
        doc='Opaque string used by InstrumentOriginExposures to identify this exposure remotely'
    )

    def __init__(self, current_file=None, invent_filepath=True, **kwargs):
        """Initialize the exposure object.

        If the filepath is given (as a keyword argument), it will parse the instrument name
        from the filename.  The header will be read out from the file.

        Parameters
        ----------
        All the properties of Exposure (i.e. columns of the exposures table), plus:

        current_file: Path or str
           The path to the file where the exposure currently is (which
           may or may not be the same as the filepath it will have in
           the database).  If you don't specify this, then the file must
           exist at filepath (either the one you pass or the one that is
           determined automatically if invent_filepath is True).

        invent_filepath: bool
           Will be ignored if you specify a filepath as an argument.
           Otherwise, if this is True, call invent_filepath() to create
           the filepath for this Exposure, based on all the other
           properties.  If this is False, then you must specify filepath
           unless the global property Exposure.nofile is True (but you
           really shouldn't be playing around with that).

        """
        FileOnDiskMixin.__init__(self, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self._data = None  # the underlying image data for each section
        self._section_headers = None  # the headers for individual sections, directly from the FITS file
        self._header = None  # the global (exposure level) header, directly from the FITS file
        self.type = 'Sci'  # default, can override using kwargs
        self._instrument_object = None
        self._bitflag = 0

        if 'header' in kwargs:
            kwargs['_header'] = kwargs.pop('header')

        # manually set all properties (columns or not, but don't
        # overwrite instance methods) Do this once here, because some of
        # the values are going to be needed by upcoming function calls.
        # We will run this exact code again later so that the keywords
        # can override what's detected from the header.
        self.set_attributes_from_dict( kwargs )

        # must have Instrument to invent a filename (and initialize Provenance)
        # but if not given, it can be guessed from the filepath
        if self.filepath is None and self.instrument is None:
            raise ValueError( "Exposure.__init__: must give at least a filepath or an instrument" )

        if self.filepath is None:
            # in this case, the instrument must have been given
            if self.provenance is None:
                self.make_provenance()  # a default provenance for exposures

            if invent_filepath:
                self.filepath = self.invent_filepath()
            elif not self.nofile:
                raise ValueError("Exposure.__init__: must give a filepath to initialize an Exposure object. ")

        if self.instrument is None:
            self.instrument = guess_instrument(self.filepath)

        # this can happen if the instrument is not given, but the filepath is
        if self.provenance is None:
            self.make_provenance()  # a default provenance for exposures

        # instrument_obj is lazy loaded when first getting it
        if current_file is None:
            current_file = pathlib.Path( FileOnDiskMixin.local_path ) / self.filepath
        if self.instrument_object is not None:
            self.use_instrument_to_read_header_data( fromfile=current_file )

        # Allow passed keywords to override what's detected from the header
        self.set_attributes_from_dict( kwargs )

        self.calculate_coordinates()  # galactic and ecliptic coordinates

    def make_provenance(self):
        """Generate a Provenance for this exposure.

        The provenance will have only one parameter,
        which is the instrument name.

        Why use the instrument as a parameter of the exposure?
        So we never have data products from different instruments
        with the same provenance (this is important to how we group
        the upstream images when e.g., making a coadd).
        """
        codeversion = Provenance.get_code_version()
        self.provenance = Provenance(
            code_version=codeversion,
            process='load_exposure',
            parameters={'instrument': self.instrument},
            upstreams=[],
        )

    @sa.orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)
        self._data = None
        self._section_headers = None
        self._header = None
        self._instrument_object = None
        session = object_session(self)
        if session is not None:
            self.update_instrument(session=session)

    def __setattr__(self, key, value):
        if key == 'ra' and isinstance(value, str):
            value = parse_ra_hms_to_deg(value)
        if key == 'dec' and isinstance(value, str):
            value = parse_dec_dms_to_deg(value)

        super().__setattr__(key, value)

    def use_instrument_to_read_header_data(self, fromfile=None):
        """
        Use the instrument object to read the header data from the file.
        This will set the column attributes from these values.
        Additional header values will be stored in the header JSONB column.
        """
        # get the header from the file in its raw form as a fits.Header object
        if fromfile is None:
            fromfile = self.get_fullpath()
        raw_header_dictionary = self.instrument_object.read_header( fromfile )

        # read and rename/convert units for all the column attributes:
        critical_info = self.instrument_object.extract_header_info(
            header=raw_header_dictionary,
            names=EXPOSURE_COLUMN_NAMES,
        )

        # verify some attributes match, and besides set the column attributes from these values
        for k, v in critical_info.items():
            if k == 'instrument':
                if self.instrument != v:
                    raise ValueError(f"Header instrument {v} does not match Exposure instrument {self.instrument}")
            elif k == 'telescope':
                if self.telescope != v:
                    raise ValueError(
                        f"Header telescope {v} does not match Exposure telescope {self.telescope}"
                    )
            elif k == 'filter' and isinstance(v, list):
                self.filter_array = v
            elif k == 'filter' and isinstance(v, str):
                self.filter = v
            else:
                setattr(self, k, v)

        # these additional keys go into the header only
        auxiliary_names = EXPOSURE_HEADER_KEYS + self.instrument_object.get_auxiliary_exposure_header_keys()
        self.info = self.instrument_object.extract_header_info(
            header=raw_header_dictionary,
            names=auxiliary_names,
        )

    def check_required_attributes(self):
        """Check that this exposure has all the required attributes."""

        missing = []
        required = EXPOSURE_COLUMN_NAMES
        required.pop('filter')  # check this manually after the loop
        for name in required:
            if getattr(self, name) is None:
                missing.append(name)

        # one of these must be defined:
        if self.filter is None and self.filter_array is None:
            missing.append('filter')

        if len(missing) > 0:
            raise ValueError(f"Missing required attributes: {missing}")

    @property
    def instrument_object(self):
        if self.instrument is not None:
            if self._instrument_object is None or self._instrument_object.name != self.instrument:
                self._instrument_object = get_instrument_instance(self.instrument)

        return self._instrument_object

    @property
    def telescope(self):
        return self.instrument_object.telescope

    @instrument_object.setter
    def instrument_object(self, value):
        self._instrument_object = value

    @property
    def start_mjd(self):
        """Time of the beginning of the exposure (equal to mjd). """
        return self.mjd

    @property
    def mid_mjd(self):
        """Time of the middle of the exposure. """
        if self.mjd is None or self.exp_time is None:
            return None
        return (self.start_mjd + self.end_mjd) / 2.0

    @property
    def end_mjd(self):
        """The time when the exposure ended. """
        if self.mjd is None or self.exp_time is None:
            return None
        return self.mjd + self.exp_time / 86400.0

    def __repr__(self):

        filter_str = '--'
        if self.filter is not None:
            filter_str = self.filter
        if self.filter_array is not None:
            filter_str = f"[{', '.join(self.filter_array)}]"

        return (
            f"Exposure(id: {self.id}, "
            f"exp: {self.exp_time}s, "
            f"filt: {filter_str}, "
            f"from: {self.instrument}/{self.telescope})"
        )

    def __str__(self):
        return self.__repr__()

    def invent_filepath( self ):
        """Create a filepath (relative to data root) for the exposure based on metadata.

        This is used when saving the exposure to disk

        """

        # Much code redundancy with Image.invent_filepath; move to a mixin?

        if self.provenance is None:
            raise ValueError("Cannot invent filepath for exposure without provenance.")
        prov_hash = self.provenance.id

        t = Time(self.mjd, format='mjd', scale='utc').datetime
        date = t.strftime('%Y%m%d')
        time = t.strftime('%H%M%S')

        short_name = self.instrument_object.get_short_instrument_name()
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

        default_convention = "{short_name}_{date}_{time}_{filter}_{prov_hash:.6s}"
        cfg = Config.get()
        name_convention = cfg.value( 'storage.exposures.name_convention', default=None )
        if name_convention is None:
            name_convention = default_convention

        filepath = name_convention.format(
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
            prov_hash=prov_hash,
        )

        if self.format == 'fits':
            filepath += ".fits"
        else:
            raise ValueError( f"Unknown format for exposures: {self.format}" )

        return filepath

    def save( self, *args, **kwargs ):
        """Save an exposure to the local file store and the archive.

        One optional positional parameter is the data to save.  This can
        either be a binary blob, or a file path (str or pathlib.Path).
        This is passed as the first parameter to FileOnDiskMixin.save().
        If nothing is given, then we will assume that the exposure is
        already in the right place in the local filestore, and will use
        self.get_fullpath(nofile=True) to figure out where it is.  (In
        that case, the only real reason to call this is to make sure
        things get pushed to the archive.)

        Keyword parmeters are passed on to FileOnDiskMixin.save().

        """
        if len(args) > 0:
            data = args[0]
        else:
            data = self.get_fullpath( nofile=True )
        FileOnDiskMixin.save( self, data, **kwargs )

    def load(self, section_ids=None):
        # Thought required: if exposures are going to be on the archive,
        #  then we're going to need to call self.get_fullpath() to make
        #  sure the exposure has been downloaded from the archive to
        #  local storage.
        if section_ids is None:
            section_ids = self.instrument_object.get_section_ids()

        if not isinstance(section_ids, list):
            section_ids = [section_ids]

        if not all([isinstance(sec_id, (str, int)) for sec_id in section_ids]):
            raise ValueError("section_ids must be a list of integers. ")

        if self.filepath is not None:
            for i in section_ids:
                self.data[i]  # use the SectionData __getitem__ method to load the data
        else:
            raise ValueError("Cannot load data from database without a filepath! ")

    @property
    def data(self):
        if self._data is None:
            if self.instrument is None:
                raise ValueError("Cannot load data without an instrument! ")
            self._data = SectionData(self.get_fullpath(), self.instrument_object)
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, SectionData):
            raise ValueError(f"data must be a SectionData object. Got {type(value)} instead. ")
        self._data = value

    @property
    def section_headers(self):
        if self._section_headers is None:
            if self.instrument is None:
                raise ValueError("Cannot load headers without an instrument! ")
            self._section_headers = SectionHeaders(self.get_fullpath(), self.instrument_object)
        return self._section_headers

    @section_headers.setter
    def section_headers(self, value):
        if not isinstance(value, SectionHeaders):
            raise ValueError(f"data must be a SectionHeaders object. Got {type(value)} instead. ")
        self._section_headers = value

    @property
    def header(self):
        if self._header is None:
            self._header = read_fits_image(self.get_fullpath(), ext=0, output='header')
        if self._header is None:
            self._header = fits.Header()
        return self._header

    @header.setter
    def header(self, value):
        if not isinstance(value, fits.Header):
            raise ValueError(f"data must be a fits.Header object. Got {type(value)} instead. ")
        self._header = value

    def update_instrument(self, session=None):
        """
        Make sure the instrument object is up-to-date with the current database session.

        This will call the instrument's fetch_sections() method,
        using the given session and the exposure's MJD as dateobs.

        If there are SensorSections for this instrument on the DB,
        and if their validity range is consistent with this exposure's MJD,
        those sections will be loaded to the instrument.
        This must be called before loading any data.

        This function is called automatically when an exposure
        is loaded from the database.

        Parameters
        ----------
        session: sqlalchemy.orm.Session
            The database session to use.
            If None, will open a new session
            and close it at the end of the function.
        """
        if self.instrument is None:
            return
        with SmartSession(session) as session:
            self.instrument_object.fetch_sections(session=session, dateobs=self.mjd)

    @staticmethod
    def _do_not_require_file_to_exist():
        """
        By default, new Exposure objects are generated
        with nofile=False, which means the file must exist
        at the time the Exposure object is created.
        This is the opposite default from the base class
        FileOnDiskMixin behavior.
        """
        return False

    def get_upstreams(self, session=None):
        """An exposure does not have any upstreams. """
        return []

    def get_downstreams(self, session=None):
        """An exposure has only Image objects as direct downstreams. """
        from models.image import Image

        with SmartSession(session) as session:
            images = session.scalars(sa.select(Image).where(Image.exposure_id == self.id)).all()

        return images


if __name__ == '__main__':
    import os
    ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(ROOT_FOLDER, 'data/DECam_examples/c4d_221104_074232_ori.fits.fz')
    e = Exposure(filepath)
    _logger.debug(e)
