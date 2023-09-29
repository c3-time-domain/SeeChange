import os
import re
import io
import math
import time
import traceback
import pathlib
import pytz
from enum import Enum
from datetime import datetime, timedelta

import numpy as np

import sqlalchemy as sa

import astropy.time

from models.base import Base, AutoIDMixin, SmartSession

from models.base import Base, SmartSession, FileOnDiskMixin,_logger
from models.enums_and_bitflags import CalibratorTypeConverter
from models.provenance import Provenance
from pipeline.utils import parse_dateobs, read_fits_image
from util.config import Config
import util.radec

# dictionary of regex for filenames, pointing at instrument names
INSTRUMENT_FILENAME_REGEX = None

# dictionary of names of instruments, pointing to the relevant class
INSTRUMENT_CLASSNAME_TO_CLASS = None

# dictionary of instrument object instances, lazy loaded to be shared between exposures
INSTRUMENT_INSTANCE_CACHE = None


# Orientations for those instruments that have a permanent orientation square to the sky
# x increases to the right, y increases upward
class InstrumentOrientation(Enum):
    NupEleft = 0          # No rotation
    NrightEup = 1         # 90° clockwise
    NdownEright = 2       # 180°
    NleftEdown = 3        # 270° clockwise
    NupEright = 4         # flip-x
    NrightEdown = 5       # flip-x, then 90° clockwise
    NdownEleft = 6        # flip-x, then 180°
    NleftEup = 7          # flip-x, then 270° clockwise


# from: https://stackoverflow.com/a/5883218
def get_inheritors(klass):
    """Get all classes that inherit from klass. """
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def register_all_instruments():
    """
    Go over all subclasses of Instrument and register them in the global dictionaries.
    """
    global INSTRUMENT_FILENAME_REGEX, INSTRUMENT_CLASSNAME_TO_CLASS

    if INSTRUMENT_FILENAME_REGEX is None:
        INSTRUMENT_FILENAME_REGEX = {}
    if INSTRUMENT_CLASSNAME_TO_CLASS is None:
        INSTRUMENT_CLASSNAME_TO_CLASS = {}

    inst = get_inheritors(Instrument)
    for i in inst:
        INSTRUMENT_CLASSNAME_TO_CLASS[i.__name__] = i
        if i.get_filename_regex() is not None:
            for regex in i.get_filename_regex():
                INSTRUMENT_FILENAME_REGEX[regex] = i.__name__


def guess_instrument(filename):
    """
    Find the name of the instrument from the filename.
    Uses the regex of each instrument (if it exists)
    to try to match the filename with the expected
    instrument's file name convention.
    If multiple instruments match, raises an error.

    If no instruments match, returns None.
    TODO: add a fallback method that lets each instrument
      run its own load method and see if it can load the file.

    """
    if filename is None:
        raise ValueError("Cannot guess instrument without a filename! ")

    filename = os.path.basename(filename)  # only scan the file name itself!

    if INSTRUMENT_FILENAME_REGEX is None:
        register_all_instruments()

    instrument_list = []
    for k, v in INSTRUMENT_FILENAME_REGEX.items():
        if re.search(k, filename):
            instrument_list.append(v)

    if len(instrument_list) == 0:
        # TODO: maybe add a fallback of looking into the file header?
        # raise ValueError(f"Could not guess instrument from filename: {filename}. ")
        return None  # leave empty is the right thing? should probably go to a fallback method
    elif len(instrument_list) == 1:
        return instrument_list[0]
    else:
        raise ValueError(f"Found multiple instruments matching filename: {filename}. ")

    # TODO: add fallback method that runs all instruments
    #  (or only those on the short list) and checks if they can load the file


def get_instrument_instance(instrument_name):
    """
    Get an instance of the instrument class, given the name of the instrument.
    Will store that instance in the INSTRUMENT_INSTANCE_CACHE dictionary,
    so the instruments can be re-used for e.g., loading multiple exposures.
    """
    if INSTRUMENT_CLASSNAME_TO_CLASS is None:
        register_all_instruments()

    global INSTRUMENT_INSTANCE_CACHE
    if INSTRUMENT_INSTANCE_CACHE is None:
        INSTRUMENT_INSTANCE_CACHE = {}

    if instrument_name not in INSTRUMENT_INSTANCE_CACHE:
        INSTRUMENT_INSTANCE_CACHE[instrument_name] = INSTRUMENT_CLASSNAME_TO_CLASS[instrument_name]()

    return INSTRUMENT_INSTANCE_CACHE[instrument_name]


class SensorSection(Base, AutoIDMixin):
    """
    A class to represent a section of a sensor.
    This is most often associated with a CCD chip, but could be any
    section of a sensor. For example, a section of a CCD chip that
    is read out independently, or different channels in a dichroic imager.

    Any properties that are not set (e.g., set to None) on the sensor section
    will be replaced by the global value of the parent Instrument object.
    E.g., if the DemoInstrument has gain=2.0, and it's sensor section has
    gain=None, then the sensor section will have gain=2.0.
    If at any time the instrument changes, add a new SensorSection object
    (with appropriate validity range) to the database to capture the new
    instrument properties.
    Thus, a SensorSection can override global instrument values either for
    specific parts of the sensor (spatial variability) or for specific times
    (temporal variability).
    """

    __tablename__ = "sensor_sections"

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='The name of the instrument this section belongs to. '
    )

    identifier = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc='A unique identifier for this section. Can be, e.g., the CCD ID. '
    )

    validity_start = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=(
            'The time when this section object becomes valid. '
            'If None, this section is valid from the beginning of time. '
            'Use the validity range to get updated versions of sections, '
            'e.g., after a change of CCD. '
        )
    )

    validity_end = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=(
            'The time when this section object becomes invalid. '
            'If None, this section is valid until the end of time. '
            'Use the validity range to get updated versions of sections, '
            'e.g., after a change of CCD. '
        )
    )

    size_x = sa.Column(
        sa.Integer,
        nullable=True,
        doc='Number of pixels in the x direction. '
    )

    size_y = sa.Column(
        sa.Integer,
        nullable=True,
        doc='Number of pixels in the y direction. '
    )

    offset_x = sa.Column(
        sa.Integer,
        nullable=True,
        doc='Offset of the center of the section in the x direction (in pixels). '
    )

    offset_y = sa.Column(
        sa.Integer,
        nullable=True,
        doc='Offset of the center of the section in the y direction (in pixels). '
    )

    filter_array_index = sa.Column(
        sa.Integer,
        nullable=True,
        doc='Index in the filter array that specifies which filter this section is located under in the array. '
    )

    # Note that read_noise, dark_current, gain, saturation_limit, and
    #  non_linearity_limit can vary by lot (10s of %) between amps on a
    #  single chip.  The values here will be at best "nominal" values,
    #  and shouldn't be used for any image reduction, but only for
    #  general low-precision instrument comparison.

    read_noise = sa.Column(
        sa.Float,
        nullable=True,
        doc='Read noise of the sensor section (in electrons). '
    )

    dark_current = sa.Column(
        sa.Float,
        nullable=True,
        doc='Dark current of the sensor section (in electrons/pixel/second). '
    )

    gain = sa.Column(
        sa.Float,
        nullable=True,
        doc='Gain of the sensor section (in electrons/ADU). '
    )

    saturation_limit = sa.Column(
        sa.Float,
        nullable=True,
        doc='Saturation level of the sensor section (in electrons). '
    )

    non_linearity_limit = sa.Column(
        sa.Float,
        nullable=True,
        doc='Non-linearity of the sensor section (in electrons). '
    )

    defective = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        index=True,
        doc='Whether this section is defective (i.e., if True, do not use it!). '
    )

    def __init__(self, identifier, instrument, **kwargs):
        """
        Create a new SensorSection object.
        Some parameters must be filled out for this object.
        Others (e.g., offsets) can be left at the default value.

        Parameters
        ----------
        identifier: str or int
            A unique identifier for this section. Can be, e.g., the CCD ID.
            Integers will be converted to strings.
        instrument: str
            Name of the instrument this section belongs to.
        kwargs: dict
            Additional values like gain, saturation_limit, etc.
        """
        if not isinstance(identifier, (str, int)):
            raise ValueError(f"identifier must be a string or an integer. Got {type(identifier)}.")

        self.identifier = str(identifier)
        self.instrument = instrument

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"<SensorSection {self.identifier} ({self.size_x}x{self.size_y})>"

    def __eq__(self, other):
        """
        Check if the sensor section is identical to the other one.
        Returns True if all attributes are the same,
        not including database level attributes like id, created_at, etc.
        """

        for att in self.get_attribute_list():
            if getattr(self, att) != getattr(other, att):
                return False

        return True

class Instrument:
    """
    Base class for an instrument.
    Instruments contain all the information about the instrument and telescope,
    that were used to produce an exposure.

    Subclass this base class to add methods that are unique to each instrument,
    e.g., loading files, reading headers, etc.

    Each instrument can have one or more SensorSection objects,
    each corresponding to a part of the focal plane (e.g., a CCD chip).
    These include additional info like the chip offset, size, etc.
    The sections can be generated dynamically (using hard-coded values),
    or loaded from the database using fetch_sections().

    If a sensor section has a non-null value for a given parameter
    (e.g., gain) then that value is used instead of the Instrument
    object's global value. Thus, a sensor section can be used to
    override the global parameter values.
    Sections can also be defined with a validity range,
    to reflect changes in the instrument (e.g., replacement of a CCD).
    Thus the sensor sections act as a way to override the global
    values either in time or in space.

    """
    def __init__(self, **kwargs):
        """
        Create a new Instrument. This should only be called
        at the end of the __init__() method of a subclass.
        Any attributes that do not have any definitions in the
        subclass __init__ will be set to None (or the default value).
        In general, kwargs will be passed into the attributes
        of the object.
        """
        self.name = getattr(self, 'name', None)  # name of the instrument (e.g., DECam)

        # telescope related properties
        self.telescope = getattr(self, 'telescope', None)  # name of the telescope it is mounted on (e.g., Blanco)
        self.focal_ratio = getattr(self, 'focal_ratio', None)  # focal ratio of the telescope (e.g., 2.7)
        self.aperture = getattr(self, 'aperture', None)  # telescope aperture in meters (e.g., 4.0)
        self.pixel_scale = getattr(self, 'pixel_scale', None)  # number of arc-seconds per pixel (e.g., 0.2637)

        # sensor related properties
        # these are average value for all sensor sections,
        # and if no sections can be loaded, or if the sections
        # do not define these properties, then the global values are used
        self.size_x = getattr(self, 'size_x', None)  # number of pixels in the x direction
        self.size_y = getattr(self, 'size_y', None)  # number of pixels in the y direction
        self.read_time = getattr(self, 'read_time', None)  # read time in seconds (e.g., 20.0)
        # read_noise, dark_currnet, gain, saturation_limit, and non_linearity_limit can
        #   vary by a lot between chips (and between amps on a single chip).  The numbers
        #   here should only be used for low-precision instrument comparision, not
        #   for any data reduction.  (Same comment in SensorSection.)
        self.read_noise = getattr(self, 'read_noise', None)  # read noise in electrons (e.g., 7.0)
        self.dark_current = getattr(self, 'dark_current', None)  # dark current in electrons/pixel/second (e.g., 0.2)
        self.gain = getattr(self, 'gain', None)  # gain in electrons/ADU (e.g., 4.0)
        self.saturation_limit = getattr(self, 'saturation_limit', None)  # saturation limit in electrons (e.g., 100000)
        self.non_linearity_limit = getattr(self, 'non_linearity_limit', None)  # non-linearity limit in electrons

        self.allowed_filters = getattr(self, 'allowed_filters', None)  # list of allowed filter (e.g., ['g', 'r', 'i'])

        self.orientation_fixed = ( self, 'orientation_fixed', False ) # True if sensor never rotates
        self.orientation = ( self, 'orientation', None ) # If orientation_fixed is True, one of InstrumentOrientation

        self.sections = getattr(self, 'sections', None)  # populate this using fetch_sections(), then a dict
        self._dateobs_for_sections = getattr(self, '_dateobs_for_sections', None)  # dateobs when sections were loaded
        self._dateobs_range_days = getattr(self, '_dateobs_range_days', 1.0)  # how many days from dateobs to reload

        # List of the preprocessing steps to apply to images from this
        # instrument, in order overscan must always be first.  The
        # values here (in the Instrument class) are all possible values.
        # Subclasses should redefine this with the subset that they
        # actually need.  If a subclass has to add a new preprocessing
        # step, then it should add that step to this list, and (if it's
        # a step that includes a calibraiton image or datafile) to the
        # CalibratorTypeConverter dict in enums_and_bitflags.py
        self.preprocessing_steps = [ 'overscan', 'zero','dark', 'linearity', 'flat', 'fringe', 'illumination' ]
        # nofile_steps are ones that don't have an associated file
        self.preprocessing_nofile_steps = [ 'overscan' ]
        
        # set the attributes from the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # add this instrument to the cache, if there isn't one already
        global INSTRUMENT_INSTANCE_CACHE
        if INSTRUMENT_INSTANCE_CACHE is None:
            INSTRUMENT_INSTANCE_CACHE = {}
        if self.name not in INSTRUMENT_INSTANCE_CACHE:
            INSTRUMENT_INSTANCE_CACHE[self.__class__.__name__] = self

    def __repr__(self):
        ap = None if self.aperture is None else f'{self.aperture:.1f}m'
        sc = None if self.pixel_scale is None else f'{self.pixel_scale:.2f}"/pix'
        filts = [] if self.allowed_filters is None else [",".join(self.allowed_filters)]
        return f'<Instrument {self.name} on {self.telescope} ({ap}, {sc}, {filts})'

    @classmethod
    def get_section_ids(cls):
        """
        Get a list of SensorSection identifiers for this instrument.

        THIS METHOD MUST BE OVERRIDEN BY THE SUBCLASS.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    @classmethod
    def check_section_id(cls, section_id):
        """
        Check that the type and value of the section is compatible with the instrument.
        For example, many instruments will key the section by a running integer (e.g., CCD ID),
        while others will use a string (e.g., channel 'A').

        Will raise a meaningful error if not compatible.

        Subclasses should override this method to be more specific
        (e.g., to test if an integer is in range).

        THIS METHOD CAN BE OVERRIDEN TO MAKE THE CHECK MORE SPECIFIC
        (e.g., to check only for integers, to check the id is in range).
        """
        if not isinstance(section_id, (int, str)):
            raise ValueError(f"The section_id must be an integer or string. Got {type(section_id)}. ")

    def _make_new_section(self, identifier):
        """
        Make a new SensorSection object for this instrument.
        The new sections can be generated with hard-coded values,
        including the most up-to-date information about the instrument.
        If that information changes, a new section should be added,
        with the old section saved to the DB with some validity range
        added manually.

        Any properties of the section that are the same as the global
        value of the instrument can be left as None, and the global
        value will be used when calling get_property() on the instrument.

        Often the offsets of a section will be non-zero and hard-coded
        based on the physical layout of a tiled-CCD focal plane.
        Other properties like the gain, read noise, etc. can be measured
        and hard-coded here, be read out from a table, etc.
        It is the user's responsibility to maintain updated values
        for the sections.

        THIS METHOD MUST BE OVERRIDEN BY THE SUBCLASS.

        Parameters
        ----------
        identifier: str or int
            The identifier for the section. This is usually an integer,
            but for some instruments it could be a string
            (e.g., for multi-channel instruments).

        Returns
        -------
        section: SensorSection
            The new section object.
        """
        raise NotImplementedError("Subclass this base class to add methods that are unique to each instrument.")

    def get_section(self, section_id):
        """
        Get a section from the sections dictionary.

        The section_id is first checked for type and value compatibility,
        and then the section is loaded from the dictionary of sections.
        This method does not access the database or generate new sections.
        To make sections, use fetch_sections().

        THIS METHOD SHOULD GENERALLY NOT BE OVERRIDEN BY SUBCLASSES.
        """
        self.check_section_id(section_id)

        if self.sections is None:
            raise RuntimeError("No sections loaded for this instrument. Use fetch_sections() first.")

        return self.sections.get(section_id)

    def fetch_sections(self, session=None, dateobs=None):
        """
        Get the sensor section objects associated with this instrument.

        Will try to get sections that are valid during the given date.
        If any sections are missing, they will be created using the
        hard coded values in _make_new_section().
        If multiple valid sections are found, use the latest one
        (the one with the most recent "modified" value).

        Will populate the self.sections attribute,
        and will lazy load that before checking against the DB.
        If the dateobs value is too far from that used the last time
        the sections were populated, then they will be cleared and reloaded.
        The time delta for this is set by self._dateobs_range_days (=1 by default).

        Parameters
        ----------
        session: sqlalchemy.orm.Session (optional)
            The database session to use. If None, will create a new session.
            Use session=False to avoid using the database entirely.
        dateobs: datetime or Time or float (as MJD) or string (optional)
            The date of the observation. If None, will use the current date.
            If there are multiple instances of a sensor section on the DB,
            only choose the ones valid during the observation.

        THIS METHOD SHOULD GENERALLY NOT BE OVERRIDEN BY SUBCLASSES.

        Returns
        -------
        sections: list of SensorSection
            The sensor sections associated with this instrument.
        """
        dateobs = parse_dateobs(dateobs, output='datetime')

        # if dateobs is too far from last time we loaded sections, reload
        if self._dateobs_for_sections is not None:
            if abs(self._dateobs_for_sections - dateobs) < timedelta(self._dateobs_range_days):
                self.sections = None

        # this should never happen, but still
        if self._dateobs_for_sections is None:
            self.sections = None

        # we need to get new sections
        if self.sections is None:
            self.sections = {}
            self._dateobs_for_sections = dateobs  # track the date used to load
            if session is False:
                all_sec = []
            else:
                # load sections from DB
                with SmartSession(session) as session:
                    all_sec = session.scalars(
                        sa.select(SensorSection).where(
                            SensorSection.instrument == self.name,
                            sa.or_(SensorSection.validity_start.is_(None), SensorSection.validity_start <= dateobs),
                            sa.or_(SensorSection.validity_end.is_(None), SensorSection.validity_end >= dateobs),
                        ).order_by(SensorSection.modified.desc())
                    ).all()

            for sid in self.get_section_ids():
                sec = [s for s in all_sec if s.identifier == str(sid)]
                if len(sec) > 0:
                    self.sections[sid] = sec[0]
                else:
                    self.sections[sid] = self._make_new_section(sid)

        return self.sections

    def commit_sections(self, session=None, validity_start=None, validity_end=None):
        """
        Commit the sensor sections associated with this instrument to the database.
        This is used to update or add missing sections that were created from
        hard-coded values (i.e., using the _make_new_section() method).

        THIS METHOD SHOULD GENERALLY NOT BE OVERRIDEN BY SUBCLASSES.

        Parameters
        ----------
        session: sqlalchemy.orm.Session (optional)
            The database session to use. If None, will create a new session.
        validity_start: datetime or Time or float (as MJD) or string (optional)
            The start of the validity range for these sections.
            Only changes the validity start of sections that have validity_start=None.
            If None, will not modify any of the validity start values.
        validity_end: datetime or Time or float (as MJD) or string (optional)
            The end of the validity range for these sections.
            Only changes the validity end of sections that have validity_end=None.
            If None, will not modify any of the validity end values.
        """
        with SmartSession(session) as session:
            for sec in self.sections.values():
                if sec.validity_start is None and validity_start is not None:
                    sec.validity_start = validity_start
                if sec.validity_end is None and validity_end is not None:
                    sec.validity_end = validity_end
                session.add(sec)

            session.commit()

    def get_property(self, section_id, prop):
        """
        Get the value of a property for a given section of the instrument.
        If that property is not defined on the sensor section
        (e.g., if it is None) then the global value from the Instrument is used.

        Will raise an error if no sections were loaded (if sections=None).
        If sections were loaded but no section with the required id is found,
        will quietly use the global value.

        THIS METHOD SHOULD GENERALLY NOT BE OVERRIDEN BY SUBCLASSES.

        Parameters
        ----------
        section_id: int or str
            The identifier of the section to get the property for.
        prop: str
            The name of the property to get.

        """

        section = self.get_section(section_id)
        if section is not None:
            if hasattr(section, prop) and getattr(section, prop) is not None:
                return getattr(section, prop)

        # first check if we can recover these properties from hard-coded functions:
        if prop == 'offsets':
            return self.get_section_offsets(section_id)
        elif prop == 'offset_x':
            return self.get_section_offsets(section_id)[0]
        elif prop == 'offset_y':
            return self.get_section_offsets(section_id)[1]
        elif prop == 'filter_array_index':
            return self.get_section_filter_array_index(section_id)
        else:  # just get the value from the object
            return getattr(self, prop)

    def get_section_offsets(self, section_id):
        """
        Get the offset of the given section from the origin of the detector.
        This can be used if the SensorSection object itself does not have
        values for offset_x and offset_y. Use this function in subclasses
        to hard-code the offsets.
        If the offsets need to be updated over time, they should be
        added to the SensorSection objects on the database.

        THIS METHOD SHOULD BE OVERRIDEN BY SUBCLASSES WITH NON-ZERO OFFSETS.
        (e.g., if the instrument has a tiled focal plane, each section should
        have a different offset, where the hard-coded values are given by
        the override of this function).

        Parameters
        ----------
        section_id: int or str
            The identifier of the section.

        Returns
        -------
        offset: tuple of floats
            The offsets in the x and y direction.
        """
        self.check_section_id(section_id)
        # this simple instrument defaults to zero offsets for ALL sections
        offset_x = 0
        offset_y = 0
        return offset_x, offset_y

    def get_section_filter_array_index(self, section_id):
        """
        Get the index in the filter array under which this section is placed.
        This can be used if the SensorSection object itself does not have
        a value for filter_array_index. Use this function in subclasses
        to hard-code the array index.
        If the array index need to be updated over time, it should be
        added to the SensorSection objects on the database.

        THIS METHOD SHOULD BE OVERRIDEN ONLY FOR INSTRUMENTS WITH A FILTER ARRAY.

        Parameters
        ----------
        section_id: int or str
            The identifier of the section.

        Returns
        -------
        idx: int
            The index in the filter array.
        """
        self.check_section_id(section_id)
        # this simple instrument has no filter array, so return zero
        idx = 0
        return idx

    def load(self, filepath, section_ids=None):
        """
        Load a part of an exposure file, based on the section identifier.
        If the instrument does not have multiple sections, set section_ids=0.

        THIS FUNCTION SHOULD GENERALLY NOT BE OVERRIDEN BY SUBCLASSES.

        Parameters
        ----------
        filepath: str
            The filepath of the exposure file.
        section_ids: str, int, or list of str or int (optional)
            Choose which section to load.
            The section_id is the identifier of the SensorSection object.
            This can be a serial number which is converted to a string.
            If given as a list, will load all the sections mentioned in the list,
            and returns a list of data arrays.
            If None (or not given) will load all the sections in the instrument,
            and return a list of arrays.

        Returns
        -------
        data: np.ndarray or list of np.ndarray
            The data from the exposure file.
        """
        if section_ids is None:
            section_ids = self.get_section_ids()

        if isinstance(section_ids, (int, str)):
            return self.load_section_image(filepath, section_ids)

        elif isinstance(section_ids, list):
            return [self.load_section_image(filepath, section_id) for section_id in section_ids]

        else:
            raise ValueError(
                f"section_ids must be a string, int, or list of strings or ints. Got {type(section_ids)}"
            )

    def load_section_image(self, filepath, section_id):
        """
        Load one section of an exposure file.
        The default loader uses the pipeline.utils.read_fits_image function,
        which is a basic FITS reader utility. More advanced instruments should
        override this function to use more complex file reading code.

        THIS FUNCTION CAN BE OVERRIDEN BY EACH INSTRUMENT IMPLEMENTATION.

        Parameters
        ----------
        filepath: str
            The filename (with full path) of the exposure file.
        section_id: str or int
            The identifier of the SensorSection object.
            This can be a serial number which is converted to a string.

        Returns
        -------
        data: np.ndarray
            The data from the exposure file.
        """
        self.check_section_id(section_id)
        idx = self._get_fits_hdu_index_from_section_id(section_id)
        return read_fits_image(filepath, idx)

    @classmethod
    def get_filename_regex(cls):
        """
        Get the regular expressions used to match filenames for this instrument.
        This is used to guess the correct instrument class to load the file
        based only on the filename. Must return a list of regular expressions.

        THIS FUNCTION MUST BE OVERRIDEN BY EACH SUBCLASS.

        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def read_header(self, filepath, section_id=None):
        """
        Load the header from file.

        By default, instruments use a "standard" FITS header that is read
        out using pipeline.utils.read_fits_image.
        Subclasses can override this method to use a different header format.
        Note that all keyword translations and value conversions happen later,
        in the extract_header_info function.

        THIS FUNCTION CAN BE OVERRIDEN OR EXTENDED BY SUBCLASSES IF NECESSARY.

        Parameters
        ----------
        filepath: str, Path or list of str or Path
            The filename (and full path) of the exposure file.
            If an Exposure is associated with multiple files,
            this will be a list of filenames.
        section_id: int or str (optional)
            The identifier of the section to load.
            If None (default), will load the header for the entire detector,
            which could be a generic header for that exposure, that doesn't
            capture any of the section-specific information.

        Returns
        -------
        header: dict or astropy.io.fits.Header
            The header from the exposure file, as a dictionary
            (or the more complex astropy.io.fits.Header object).
        """
        if isinstance(filepath, (str, pathlib.Path)):
            if section_id is None:
                return read_fits_image(filepath, ext=0, output='header')
            else:
                self.check_section_id(section_id)
                idx = self._get_fits_hdu_index_from_section_id(section_id)
                return read_fits_image(filepath, ext=idx, output='header')
        elif isinstance(filepath, list) and all( (isinstance(f, (str, pathlib.Path))) for f in filepath):
            if section_id is None:
                # just read the header of the first file
                return read_fits_image(filepath[0], ext=0, output='header')
            else:
                self.check_section_id(section_id)
                idx = self._get_file_index_from_section_id(section_id)
                return read_fits_image(filepath[idx], ext=0, output='header')
        else:
            raise ValueError(
                f"filepath must be a string or list of strings. Got {type(filepath)}"
            )

    @staticmethod
    def normalize_keyword(key):
        """
        Normalize the header keyword to be all uppercase and
        remove spaces and underscores.

        THIS FUNCTION MAY BE OVERRIDEN BY SUBCLASSES IN RARE CASES.
        """
        return key.upper().replace(' ', '').replace('_', '').replace('-', '')

    @classmethod
    def extract_header_info(cls, header, names):
        """
        Get information from the raw header into common column names.
        This includes keywords that are required for non-nullable columns (like MJD),
        or optional header keywords that can be included but are not critical.
        Will only extract keywords that have a translation
        (which is defined in _get_header_keyword_translations()).

        THIS FUNCTION SHOULD NOT BE OVERRIDEN BY SUBCLASSES.
        To override the header keyword translation, use _get_header_keyword_translations(),
        to add unit conversions use _get_header_values_converters().

        Parameters
        ----------
        header: dict
            The raw header as loaded from the file.
        names: list of str
            The names of the columns to extract.

        Returns
        -------
        output_values: dict
            A dictionary with some of the required values from the header.
        """
        header = {cls.normalize_keyword(key): value for key, value in header.items()}
        output_values = {}
        translations = cls._get_header_keyword_translations()
        converters = cls._get_header_values_converters()
        for name in names:
            translation_list = translations.get(name, [])
            if isinstance(translation_list, str):
                translation_list = [translation_list]
            for key in translation_list:
                if key in header:
                    value = header[key]
                    if name in converters:
                        value = converters[name](value)
                    output_values[name] = value
                    break

        return output_values

    @classmethod
    def get_auxiliary_exposure_header_keys(cls):
        """
        Additional header keys that can be useful to have on the
        Exposure header. This could include instrument specific
        items that are saved to the global exposure header,
        in addition to the keys in Exposure.EXPOSURE_HEADER_KEYS.

        THIS METHOD SHOULD BE OVERRIDEN BY SUBCLASSES, TO ADD MORE ITEMS
        """

        return []

    def get_ra_dec_for_section(self, exposure, section_id):
        """
        Get the RA and Dec of the center of the section.
        If there is no clever way to figure out the section
        coordinates, just leave it to return (None, None).
        In that case, the RA/Dec will be read out of the
        individual section headers.

        This function should only be overriden by instruments
        where (a) the RA/Dec in the individual section headers
        is not good / does not exist, and (b) there is a clever
        way to figure out the RA/Dec of the section from the
        global exposure header, e.g., using the offsets and
        pixel scale to calculate the center of the section
        relative to the center of the detector.

        THIS METHOD CAN BE OVERRIDEN BY SUBCLASSES.

        Parameters
        ----------
        exposure: Exposure
            The exposure object.
        section_id: int or str
            The identifier of the section.

        Returns
        -------
        ra: float or None
            The RA of the center of the section, in degrees.
        dec: float or None
            The Dec of the center of the section, in degrees.
        """
        self.check_section_id(section_id)
        return None, None

    def get_default_section_flags( self, section_id ):
        """Get the default flags image for the given SensorSection of this instrument.

        This is for loading, say, an observatory-standard flags image.
        It should be overriden by any subclass that has such a default
        flags image.  By default, it returns a data array of all zeros.

        Parameters
        ----------
        section_id: int or str
          The identifier of the section

        Returns
        -------
        A 2d numpy array of uint16 with shape [ sensorsection.size_y, sensorsection.size_x ]

        """

        sec = self.get_section( section_id )
        return np.zeros( [ sec.size_y, sec.size_x ], dtype=numpy.uint16 )

    def get_gain_at_pixel( self, image, x, y, section_id=None ):
        """Get the gain of an image at a given pixel position.

        THIS SHOULD USUALLY BE OVERRIDDEN BY SUBCLASSES.  By default,
        it's going to assume that the gain property of the sensor
        section is good, or if it's null, that the gain property of the
        instrument is good.  Subclasses should use the image header
        information.

        Parameters
        ----------
        image: Image or None
          The Image.  If None, section_id must not be none.
        x, y: int or float
          Position on the image in C-coordinates (0 offset).  Remember
          that numpy arrays are indexed [y, x].
        section_id:
          If Image is None, pass a non-null section_id to get the default gain.

        Returns
        -------
        float

        """
        sec = self.get_section( section_id if image is None else image.section_id )
        if sec.gain is None:
            return self.gain
        else:
            return sec.gain

    @classmethod
    def _get_header_keyword_translations(cls):

        """
        Get a dictionary that translates the header keywords into normalized column names.
        Each column name has a list of possible header keywords that can be used to populate it.
        When parsing the header, look for each one of these keywords, and use the first one that is found.

        THIS METHOD SHOULD BE EXTENDED BY SUBCLASSES, OR REPLACED COMPLETELY.
        """
        t = dict(
            ra=['RA', 'RADEG'],
            dec=['DEC', 'DECDEG'],
            mjd=['MJD', 'MJDOBS'],
            project=['PROJECT', 'PROJID', 'PROPOSID', 'PROPOSAL', 'PROPID'],
            target=['TARGET', 'OBJECT', 'FIELD', 'FIELDID'],
            width=['WIDTH', 'NAXIS1'],
            height=['HEIGHT', 'NAXIS2'],
            exp_time=['EXPTIME', 'EXPOSURE'],
            filter=['FILTER', 'FILT', 'FILTER_ARRAY', 'FILTERA'],
            instrument=['INSTRUME', 'INSTRUMENT'],
            telescope=['TELESCOP', 'TELESCOPE'],
        )
        return t
        # TODO: add more!

    @classmethod
    def _get_header_values_converters(cls):
        """
        Get a dictionary with some keywords
        and the conversion functions needed to turn the
        raw header values into the correct units.
        For example, if this instrument uses milliseconds
        as the exposure time units, the output dictionary
        would be: {'exp_time': lambda t: t/1000.0}.

        The base class does not assume any unit conversions
        are needed, so it returns an empty dict.
        Subclasses can override this method to add conversions.

        THIS METHOD SHOULD BE OVERRIDEN BY SUBCLASSES, TO ADD MORE ITEMS
        """
        return {}

    @classmethod
    def _get_fits_hdu_index_from_section_id(cls, section_id):
        """
        Translate the section_id into the index of the HDU in the FITS file.
        For example, if we have an instrument with 10 CCDs, numbered 0 to 9,
        the HDU list will probably contain a generic HDU at index 0,
        and the individual section information in 1 through 10, so
        the function should return section_id+1.
        Another example could have section_id=A give an index 1,
        and a section_id=B give an index 2 (so the function will read
        from a dictionary to translate the values).

        THIS METHOD SHOULD BE OVERRIDEN BY SUBCLASSES,
        in particular when section_id is a string.

        Parameters
        ----------
        section_id: int or str
            The identifier of the section.

        Returns
        -------
        index: int
            The index of the HDU in the FITS file.
            Note that the index is 0-based, as this value is
            used in the astropy.io.fits functions/objects,
            not in the native FITS format (which is 1-based).
        """
        cls.check_section_id(section_id)
        return int(section_id) + 1

    @classmethod
    def _get_file_index_from_section_id(cls, section_id):
        """
        Translate the section_id into the file index in an array of filenames.
        For example, if we have an instrument with 10 CCDs, numbered 0 to 9,
        then we would probably have 10 filenames in some list.
        In this case the function should return the section_id.
        Another example could have section_id=A give an index 0,
        and a section_id=B give an index 1 (so the function will read
        from a dictionary to translate the values).

        Parameters
        ----------
        section_id: int or str
            The identifier of the section.

        Returns
        -------
        index: int
            The list index for the file that corresponds to the section_id.
            The list of filenames must be in the correct order for this to work.
        """
        cls.check_section_id(section_id)
        return int(section_id)

    @classmethod
    def get_short_instrument_name(cls):
        """
        Get a short name used for e.g., making filenames.
        """

        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def get_short_filter_name(cls, filter):
        """
        Translate the full filter name into a shorter version,
        e.g., for using in filenames.
        The default is to just return the filter name,
        but instruments that have very long filter names
        should really override this with a table lookup with short names.
        """

        return filter

    # ----------------------------------------
    # Preprocessing functions.  These live here rather than
    # in pipeline/preprocessing.py because individual instruments
    # may need specific overrides for some of the steps.  For many
    # instruments, the defaults should work.

    def _get_default_calibrator( self, mjd, section, calibtype='dark', filter=None, session=None ):
        """Acquire (if possible) the default externally-supplied CalibratorFile.

        Will load it into the database as both an Image and a Calibrator
        Image; may also load other default calibrator images if they
        come as a pack. WILL CALL session.commit()!

        Should not be called from outside Instrument; instead, use
        preprocessing_calibrator_params.  _get_default_calibrator method
        will assume that the default is not already in the database, and
        load it without checking first.  That other method searches the
        database first.

        WILL NEED TO BE OVERRIDEEN FOR EVERY SUBCLASS, unless the
        instrument doesn't use any calibrator images, or doesn't have
        any default externally supplied calibrator images.

        Parameters
        ----------
        mjd: float
          mjd of validity of the dark frame
        section: SensorSection
          The sensor section for the dark frame
        calibtype: str
          One of 'zero', 'dark', 'flat', 'illumination', 'fringe', or 'linearity'
        session: Session
          database session

        Returns
        -------
        CalibratorFile or None

        """

        # Note: subclasses will need to import CalibratorFile here.  We
        # can't import it at the top of the file because calibrator.py
        # imports image.py, image.py imports exposure.py, and
        # exposure.py imports instrument.py -- leading to a circular import

        # from models.calibratorfile import CalibratorFile

        return None

    def preprocessing_calibrator_params( self, calibset, flattype, section, filter, mjd, nofetch=False, session=None ):
        """Get a dictionary of calibrator images/datafiles for a given mjd and sensor section.

        MIGHT call session.commit(); see below.
        
        Instruments *may* need to override this.

        If a calibrator file doesn't exist for calibset 'default', will
        call the instrument's _get_default_calibrator, which will call
        session.commit().

        If a calibrator file isn't found (potentially after calling the
        instrument's _get_default_calibrator), then the _isimage and
        _fileid values in the return dictionary for that calibrator file
        will be None.
        
        Parameters
        ----------
        calibset: str
          The calibrator set, one of the values in the CalibratorSetConverter enum
        flattype: str
          The flatfield type, one of the values in the FlatTypeConverter
          enum; if and only if calibset is externally_supplied, then
          flattype must be externally_supplied.
        section: str
          The name of the SensorSection
        filter: str
          The filter (can be None for some types, e.g. zero, linearity)
        mjd: float
          The mjd where the calibrator params are valid
        nofetch: bool
          If True, will only search the database for an
          externally_supplied calibrator.  If False (default), will call
          the instrument's _get_default_calibrators method if an
          externally_supplied calibrator isn't found in the database.
          Ignored if calibset is not externally_supplied.
        session: Session

        Returns
        -------
        dict with up to 12 keys:
           (zero|flat|dark|fringe|illumination|linearity)_isimage: bool
               True if the calibrator id is an image (other wise is none, or a miscellaneous data file)
           (zero|flat|dark|fringe|illumination|linearity)_fileid: int
               Either the image_id or datafile_id of the calibrator file, or None if not found

        keys will only be included for steps in the instrument's
        preprocessing_steps list (which is set during instrument object
        construction).

        """

        if ( calibset == 'externally_supplied' ) != ( flattype == 'externally_supplied' ):
            raise ValueError( "Doesn't make sense to have only one of calibset and flattype be externally_supplied" )
        
        # Import CalibratorFile here.  We can't import it at the top of
        # the file because calibrator.py imports image.py, image.py
        # imports exposure.py, and exposure.py imports instrument.py --
        # leading to a circular import
        from models.calibratorfile import CalibratorFile
        from models.image import Image

        params = {}

        expdatetime = pytz.utc.localize( astropy.time.Time( mjd, format='mjd' ).datetime )

        with SmartSession(session) as session:
            for calibtype in self.preprocessing_steps:
                if calibtype in self.preprocessing_nofile_steps:
                    continue
                # To avoid a race condition in _get_default_calibrator, we need to
                # lock the CalibratorFile table.  Reason: if multiple processes are
                # running at once, they might both look for the same calibrator file
                # at once, both not find it, and both call _get_default_calibrator
                # to try to add the same calibrator file.
                try:
                    if ( calibset == 'externally_supplied' ) and ( not nofetch ):
                        session.connection().execute( sa.text( 'LOCK TABLE calibrator_files' ) )
                    calibquery = ( session.query( CalibratorFile )
                                   .filter( CalibratorFile.calibrator_set == calibset )
                                   .filter( CalibratorFile.instrument == self.name )
                                   .filter( CalibratorFile.type == calibtype )
                                   .filter( CalibratorFile.sensor_section == section )
                                   .filter( sa.or_( CalibratorFile.validity_start == None,
                                                    CalibratorFile.validity_start <= expdatetime ) )
                                   .filter( sa.or_( CalibratorFile.validity_end == None,
                                                    CalibratorFile.validity_end >= expdatetime ) )
                                  )
                    if calibtype == 'flat':
                        calibquery = calibquery.filter( CalibratorFile.flat_type == flattype )
                    if ( calibtype in [ 'flat', 'fringe', 'illumination' ] ) and ( filter is not None ):
                        calibquery = calibquery.join( Image ).filter( Image.filter == filter )

                    calib = None
                    if ( calibquery.count() == 0 ) and ( calibset == 'externally_supplied' ) and ( not nofetch ):
                        calib = self._get_default_calibrator( mjd, section, calibtype=calibtype,
                                                              filter=self.get_short_filter_name( filter ),
                                                              session=session )
                    else:
                        if calibquery.count() > 1:
                            _logger.warning( f"Found {calibquery.count()} valid {calibtype}s for "
                                             f"{instrument.name} {section}, randomly using one." )
                        calib = calibquery.first()
                finally:
                    # Make sure that the calibrator_files table gets
                    # unlocked.  If _get_default_calibrator had to add
                    # something to the database, it will have called a
                    # commit() already, which would have already
                    # unlocked the table.  But, if it wasn't called,
                    # then the table would remain locked; to avoid that,
                    # rollback here.
                    if ( calibset == 'externally_supplied' ) and ( not nofetch ):
                        session.rollback()

                if calib is None:
                    params[ f'{calibtype}_isimage' ] = False
                    params[ f'{calibtype}_fileid'] = None
                else:
                    if calib.image_id is not None:
                        params[ f'{calibtype}_isimage' ] = True
                        params[ f'{calibtype}_fileid' ] = calib.image_id
                    elif calib.datafile_id is not None:
                        params[ f'{calibtype}_isimage' ] = False
                        params[ f'{calibtype}_fileid' ] = calib.datafile_id
                    else:
                        raise RuntimeError( f'Data corruption: CalibratorFile {calib.id} has neither '
                                            f'image_id nor datafile_id' )

        return params

    def overscan_sections( self, header ):
        """Return the overscan sections for a raw image given its header.

        You probably want to call overscan_and_data_sections() rather than this.

        Most instruments will need to override this to look at the right
        header keywords.  This default routine uses the keywords from
        DECam, i.e. BIASSECA, BIASSECB, DATASECA, DATASECB.

        Parameters
        ----------
        header: dict
          The header of the image in question.
          NOTE: this needs to be the full header of the image,
          i.e. Image.raw_header rahter than Image.header.

        Returns
        -------
        list of dicts; each element has fields
          'secname': str,
          'biassec': { 'x0': int, 'x1': int, 'y0': int, 'y1': int },
          'datasec': { 'x0': int, 'x1': int, 'y0': int, 'y1': int }
        Secname is some subsection identifier, which is instrument
        specific.  By default, it will be 'A' or 'B'.  Sections are in
        C-coordinates (0-offset), using the numpy standard (i.e. x1 is
        one past the end of the region); remember that numpy arrays are
        indexed [y, x].

        """

        arrparse = re.compile( '^\s*\[\s*(?P<x0>\d+)\s*:\s*(?P<x1>\d+)\s*,\s*(?P<y0>\d+)\s*:\s*(?P<y1>\d+)\s*\]\s*$' )
        retval = []
        for letter in [ 'A', 'B' ]:
            if ( f'BIASSEC{letter}' not in header ) or ( f'DATASEC{letter}' not in header ):
                raise RuntimeError( f"Can't find BAISSEC{letter} and/or DATASEC{letter} in header" )
            biassec = header[ f'BIASSEC{letter}' ]
            datasec = header[ f'DATASEC{letter}' ]
            biasmatch = arrparse.search( biassec )
            datamatch = arrparse.search( datasec )
            if not biasmatch:
                raise ValueError( f"Error parsing header BIASSEC{letter} entry {biassec}" )
            if not datamatch:
                raise ValueError( f"Error parsing header DATASEC{letter} entry {biassec}" )
            # The -1 are to convert from FITS/Fortran coordinates to C coordinates
            # There is no -1 high limits because the header keywords
            # give inclusive right-side limits, but numpy wants exclusive right-side
            # limits, and we're using numpy.  (The -1 offsets the +1 to
            # go from inclusive to exclusive.)
            retval.append( { 'secname': letter,
                             'biassec': { 'x0': int(biasmatch.group('x0'))-1, 'x1': int(biasmatch.group('x1')),
                                          'y0': int(biasmatch.group('y0'))-1, 'y1': int(biasmatch.group('y1')) },
                             'datasec': { 'x0': int(datamatch.group('x0'))-1, 'x1': int(datamatch.group('x1')),
                                          'y0': int(datamatch.group('y0'))-1, 'y1': int(datamatch.group('y1')) }
                            } )
        return retval


    def overscan_and_data_sections( self, header ):
        """Return the data sections where they appear in the image after trimming.

        Returns everything overscan_sections does, plus more.

        It's possible (likely?) that an instrument that overrides
        overscan_sections won't need to override this.

        Parameters
        ----------
        header: dict
          The header of the image in question; use Image.raw_header not Image.header.

        Returns
        ------
        list of dicts, each dict being:
           { 'secname': str,
             'biassec': { 'x0': int, 'x1': int, 'y0': int, 'y1': int },
             'datasec': { 'x0': int, 'x1': int, 'y0': int, 'y1': int },
             'destsec': { 'x0': int, 'x1', int, 'y0': int, 'y1': int }
           }
        secname is a string identifying the subsection.  (Not to be
        confused with SensorSection, as each Image has a single
        SensorSection.)  biassec and datasec are regions on the raw
        image; destsec are regions on the trimmed image.  Sections are
        in C-coordiates(0-offset), using the numpy standard (i.e. x1 and
        y1 are one past the end of the region); remember that numpy
        arrays are indexed [y, x].

        """
        # This whole thing turns out to be surprisingly complicated

        secs = self.overscan_sections( header )

        # Figure out all the data ranges, making sure they don't overlap
        xranges = []
        yranges = []
        for sec in secs:
            x0 = sec['datasec']['x0']
            x1 = sec['datasec']['x1']
            y0 = sec['datasec']['y0']
            y1 = sec['datasec']['y1']
            foundx = False
            for xr in xranges:
                if ( x0 == xr[0] ) and ( x1 == xr[1] ):
                    foundx = True
                    continue
                if ( ( ( x0 > xr[0] ) and ( x0 < xr[1] ) )
                     or
                     ( ( x1 > xr[0] ) and ( x1 < xr[1] ) ) ):
                    raise ValueError( f"Error, data sections aren't in a grid" )
            if not foundx:
                xranges.append( [ x0, x1 ] )
            foundy = False
            for yr in yranges:
                if ( y0 == yr[0] ) and ( y1 == yr[1] ):
                    foundy = True
                    continue
                if ( ( ( y0 > yr[0] ) and ( y0 < yr[1] ) )
                     or
                     ( ( y1 > yr[0] ) and ( y1 < yr[1] ) ) ):
                    raise ValueError( f"Error, data sections aren't in a grid" )
            if not foundy:
                yranges.append( [ y0, y1 ] )

        # Figure out destination data ranges.  There's a built in
        # assumption here that the ordering of the data sections on the
        # raw image is the same as the ordering of the data sections on
        # the trimmed image.
        xranges.sort( key = lambda a: a[0] )
        yranges.sort( key = lambda a: a[0] )
        xdestranges = []
        ydestranges = []
        xsize = 0
        for xr in xranges:
            xdestranges.append( [ xsize, xsize + xr[1] - xr[0] ] )
            xsize += xr[1] - xr[0]
        ysize = 0
        for yr in yranges:
            ydestranges.append( [ ysize, ysize + yr[1] - yr[0] ] )
            ysize += yr[1] - yr[0]

        # Now map sections to ranges
        for sec in secs:
            # Figure out where the data section goes inthe trimmed image
            xr = None
            xrdest = None
            for xrcand, xrdestcand in zip( xranges, xdestranges ):
                if ( xrcand[0] == sec['datasec']['x0'] ) and ( xrcand[1] == sec['datasec']['x1'] ):
                    xr = xrcand
                    xrdest = xrdestcand
            yr = None
            yrdest = None
            for yrcand, yrdestcand in zip( yranges, ydestranges ):
                if ( yrcand[0] == sec['datasec']['y0'] ) and ( yrcand[1] == sec['datasec']['y1'] ):
                    yr = yrcand
                    yrdest = yrdestcand
            sec[ 'destsec' ] = { 'x0': xrdest[0], 'x1': xrdest[1], 'y0': yrdest[0], 'y1': yrdest[1] }

        # whew
        return secs

    def overscan_and_trim( self, *args ):
        """Overscan and trim image.

        Parameters
        ----------
        Can pass either one or two positional parmeters

        If one: image
        image: Image
          The Image to process.  Will use Image.raw_header for header
          and Image.raw_data for data

          --- OR ---

        If two: header, data
        header: dict (NOTE THIS DOESN'T WORK -- SEE ISSUE #92)
          Image header.  Need the full header, i.e. Image.raw_header not Image.header.
        data: numpy array
          Image data.  Must not be trimmed, i.e. must include the overscan section

        Hopefully most instruments won't have to override this (only
        overscan_sections), but this routine does assume that the data
        sections are laid out in a grid (1x2, 2x2, 2x2, 3x3, etc), and
        that the relative positions of the data sections on the raw
        images (i.e. what is to the left of what) is the same as on the
        reconstructed trimmed image.   Any more complicated layout of
        data sections (e.g. interleaving, out of order data sections,
        etc.) will require custom code.

        Returns
        -------
        numpy array
          The bias-corrected and trimmed image data

        """
        # import image here because importing it at the top
        # of the file leads to circular imports
        from models.image import Image

        if len(args) == 1:
            if not isinstance( args[0], Image ):
                raise TypeError( 'overscan_and_trim: pass either an Image as one argument, '
                                 'or header and data as two arguments' )
            # THOUGHT REQUIRED
            # The raw FITS data as loaded might be integer data (based on what
            # the observatory does), but we want to use floats once we start
            # processing.  Are we limiting ourselves too much by hardcoding
            # in float32, or should we put in an option somewhere to
            # use float64?  That's probably a back-burner low-priority TODO.
            data = args[0].raw_data.astype( np.float32 )
            header = args[0].raw_header
        elif len(args) == 2:
            # if not isinstance( args[0], <whatever the right header datatype is>:
            #     raise TypeError( "header isn't a <header>" )
            if not isinstance( args[1], np.ndarray ):
                raise TypeError( "data isn't a numpy array" )
            header = args[0]
            data = args[1].astype( np.float32 )
        else:
            raise RuntimeError( 'overscan_and_trim: pass either an Image as one argument, '
                                'or header and data as two arguments' )

        sections = self.overscan_and_data_sections( header )
        xsize = 0
        ysize = 0
        for sec in sections:
            ysize = max( sec['destsec']['y1'], ysize )
            xsize = max( sec['destsec']['x1'], xsize )

        # Figure out bias values by taking the median of the appropriate overscan sections
        for sec in sections:
            # Have to figure out whether the overscan strip is offset in x or offset in y
            if sec['biassec']['y1'] - sec['biassec']['y0'] == sec['datasec']['y1'] - sec['datasec']['y0']:
                sec['bias'] = np.median( data[ sec['biassec']['y0']:sec['biassec']['y1'] ,
                                               sec['biassec']['x0']:sec['biassec']['x1'] ],
                                         axis=1 )[ :, np.newaxis ]
            elif sec['biassec']['x1'] - sec['biassec']['x0'] == sec['datasec']['x1'] - sec['datasec']['x0']:
                sec['bias'] = np.median( data[ sec['biassec']['y0']:sec['biassec']['y1'] ,
                                               sec['biassec']['x0']:sec['biassec']['x1'] ],
                                         axis=0 )[ np.newaxis, : ]
            else:
                err = ( f"Bias/Data section size mismatch: biassec=["
                        f"{sec['biassec']['x0']}:{sec['biassec']['x1']},"
                        f"{sec['biassec']['y0']}:{sec['biassec']['y1']}], datasec=["
                        f"{sec['datasec']['x0']}:{sec['datasec']['x1']},"
                        f"{sec['datasec']['y0']}:{sec['datasec']['y1']}]" )
                _logger.error( err )
                raise ValueError( err )

        # Actually subtract overscan and trim
        trimmedimage = np.zeros( [ ysize, xsize ], dtype=data.dtype )
        for sec in sections:
            # Make a bunch of temp variables for clarity
            xsrc0 = sec['datasec']['x0']
            xsrc1 = sec['datasec']['x1']
            ysrc0 = sec['datasec']['y0']
            ysrc1 = sec['datasec']['y1']
            xdst0 = sec['destsec']['x0']
            xdst1 = sec['destsec']['x1']
            ydst0 = sec['destsec']['y0']
            ydst1 = sec['destsec']['y1']
            trimmedimage[ ydst0:ydst1, xdst0:xdst1 ] = data[ ysrc0:ysrc1, xsrc0:xsrc1 ] - sec[ 'bias' ]

        return trimmedimage

    def linearity_correct( self, *args, linearitydata=None ):
        """Linearity correct image.

        Pass an image that has already been overscanned and trimmed.

        Parameters
        ----------
        Can pass either one or two positional parmeters

        If one: image
        image: Image
          The Image to process.  Will use Image.raw_header for header
          and Image.data for data

          --- OR ---

        If two: header, data
        header: dict (NOTE THIS DOESN'T WORK -- SEE ISSUE #92)
          Image header.  Need the full header, i.e. Image.raw_header not Image.header.
        data: numpy array
          Image data.  Must not be trimmed, i.e. must include the overscan section

        In addition, there's one keyword parameter, linearitydata, which
        should be either an Image or a DataFile (which one is needed
        depends on the instrument) holding the data that the instrument
        needs to linearity correct this image.

        Returns
        -------
        numpy array
          The linearity-corrected data

        """
        raise NotImplementedError( f"{self.__class__.__name__} needs to impldment linearity_correct" )


class DemoInstrument(Instrument):

    def __init__(self, **kwargs):
        self.name = 'DemoInstrument'
        self.telescope = 'DemoTelescope'
        self.aperture = 2.0
        self.focal_ratio = 5.0
        self.square_degree_fov = 0.5
        self.pixel_scale = 0.41
        self.read_time = 2.0
        self.read_noise = 1.5
        self.dark_current = 0.1
        self.gain = 2.0
        self.non_linearity_limit = 10000.0
        self.saturation_limit = 50000.0
        self.allowed_filters = ["g", "r", "i", "z", "Y"]

        # will apply kwargs to attributes, and register instrument in the INSTRUMENT_INSTANCE_CACHE
        Instrument.__init__(self, **kwargs)

        # DemoInstrument doesn't know how to preprocess
        self.preprocessing_steps = []

    @classmethod
    def get_section_ids(cls):

        """
        Get a list of SensorSection identifiers for this instrument.
        """
        return [0]

    @classmethod
    def check_section_id(cls, section_id):
        """
        Check if the section_id is valid for this instrument.
        The demo instrument only has one section, so the section_id must be 0.
        """
        if not isinstance(section_id, int):
            raise ValueError(f"section_id must be an integer. Got {type(section_id)} instead.")
        if section_id != 0:
            raise ValueError(f"section_id must be 0 for this instrument. Got {section_id} instead.")

    def _make_new_section(self, identifier):
        """
        Make a single section for the DEMO instrument.
        The identifier must be a valid section identifier.

        Returns
        -------
        section: SensorSection
            A new section for this instrument.
        """
        return SensorSection(identifier, self.name, size_x=512, size_y=1024)

    def load_section_image(self, filepath, section_id):
        """
        A spoof load method for this demo instrument.
        The data is just a random array.
        The instrument only has one section,
        so the section_id must be 0.

        Will fail if sections were not loaded using fetch_sections().

        Parameters
        ----------
        filepath: str
            The filename (and full path) of the exposure file.
            In this case the filepath is not used.
        section_id: str or int
            The identifier of the SensorSection object.
            This instrument only has one section, so this must be 0.

        Returns
        -------
        data: np.ndarray
            The data from the exposure file.
        """

        section = self.get_section(section_id)

        return np.array( np.random.poisson(10., (section.size_y, section.size_x)), dtype='=f4' )

    def read_header(self, filepath, section_id=None):
        # return a spoof header
        return {
            'RA': np.random.uniform(0, 360),
            'DEC': np.random.uniform(-90, 90),
            'EXPTIME': 30.0,
            'FILTER': np.random.choice(self.allowed_filters),
            'MJD': np.random.uniform(50000, 60000),
            'PROPID': '2020A-0001',
            'OBJECT': 'crab nebula',
            'TELESCOP': self.telescope,
            'INSTRUME': self.name,
            'GAIN': np.random.normal(self.gain, 0.01),
        }

    @classmethod
    def get_filename_regex(cls):
        return [r'Demo']

    @classmethod
    def get_short_instrument_name(cls):
        """
        Get a short name used for e.g., making filenames.
        """
        return 'Demo'

    def find_origin_exposures( self, skip_exposures_in_database=True,
                               minmjd=None, maxmjd=None, filters=None,
                               containing_ra=None, containing_dec=None,
                               minexptime=None ):
        """Search the external repository associated with this instrument.

        Search the external image/exposure repository for this
        instrument for exposures that the database doesn't know about
        already.  For example, for DECam, this searches the noirlab data
        archive.

        WARNING : do not call this without some parameters that limit
        the search; otherwise, too many things will be returned, and the
        query is likely to time out or get an error from the external
        repository. E.g., a good idea is to search only for exposure from the last week. 

        Parameters
        ----------
        skip_exposures_in_databse: bool
           If True (default), will filter out any exposures that (as
           best can be determined) are already known in the SeeChange
           database.  If False, will include all exposures. 
        minmjd: float
           The earliest time of exposure to search (default: no limit)
        maxmjd: float
           The latest time of exposure to search (default: no limit)
        filters: str or list of str
           Filters to search.  The actual strings are
           instrument-dependent, and will match what is expected on the
           external repository.  By default, doesn't limit by filter.
        containing_ra: float
           Search for exposures that include this RA (degrees, J2000);
           default, no RA constraint.
        containing_dec: float
           Search for exposures that include this Dec (degrees, J2000);
           default, no Dec constraint.
        minexptime: float
           Search for exposures that have this minimum exposure time in
           seconds; default, no limit.

        Returns
        -------
        A InstrumentOriginExposures object, or None if nothing is found.

        """
        raise NotImplementedError( f"Instrument class {self.__class__.__name__} hasn't "
                                   f"implemented find_origin_exposures." )

class InstrumentOriginExposures:
    """A class encapsulating the response from Instrument.find_origin_exposures()

    Never instantiate one of these (or a subclass) directly; get it from
    find_origin_exposures().

    Must be subclassed by each instrument that defines
    find_origin_exposures().  The internal storage of the exposures will
    differ for each instrument, and no external assumptions should be
    made about it other than that it's a sequence (and so can be indexed).

    """

    def download_exposures( self, outdir=".", indexes=None, onlyexposures=True,
                            clobber=False, existing_ok=False, session=None ):
        """Download exposures from the origin.

        Parameters
        ----------
        outdir: Path or str
           Directory where to save the files.  Filenames will be
           straight from the origin.
        indexes: list of int or None
           List of indexes into the set of origin exposures to download;
           None means download them all.
        onlyexposures: bool default True
           If True, only download the exposure.  If False, and there are
           anciallary exposure (e.g. for the DECam instrument, when
           reducing prod_type='instcal' images, there are weight and
           data quality mask exposure), download those as well.
        clobber: bool
           If True, will always download and overwrite existing files.
           If False, will trust that the file is the right thing if existing_ok=True,
           otherwise will throw an exception.
        existing_ok: bool
           Only matters if clobber=False (see above)
        session: Session
           Database session to use.  (A new one will be created if this
           is None, but that will lead to the returned exposures and
           members of those exposures not being bound to a session, so
           lazy-loading won't work).

        Returns
        -------
        A list of dictionaries, each element of which is { prod_type: pathlib.Path }
        prod_type will generally only be "exposure", but if there are ancillary
        exposures, there may be others.  E.g., when downloading reduced DECam
        images from NOIRlab, prod_type may be all of 'exposure', 'weight', 'dqmask'.

        """
        raise NotImplementedError( f"Instrument class {self.__class__.__name__} hasn't "
                                   f"implemented download_exposure." )

    def download_and_commit_exposures( self, indexes=None, clobber=False, existing_ok=False,
                                       delete_downloads=True, skip_existing=True ):
        """Download exposures and load them into the database.

        Files will first be downloaded to FileOnDiskMixin.local_path
        with the filename that the origin gave them.  The headers of
        files will be used to construct Exposure objects.  When each
        Exposure object is saved, it will copy the file to the file
        named by Exposure.invent_filpath (relative to
        FileOnDiskMixin.local_path) and upload the exposure to the
        archive.

        Parmaeters
        ----------
        indexes: list of int or None
           List of indexes into the set of origin exposures to download;
           None means download them all.
        clobber: bool
           Applies to the originally downloaded file
           (i.e. FileOnDiskMixin.local_path/{origin_filename}) already
           exists.  If clobber is True, that originally downloaded file
           will always be deleted and written over with a redownload.
           If clobber is False, then if existing_ok is True it will
           assume that that file is correct, otherwise it throws an
           exception.
        existing_ok: bool
           Applies to the originally downloaded file; see clobber.
        delete_downloads: bool
           If True (the default), will delete the originally downloaded
           files after they have been copied to their final location.
           (This mainly exists for testing purposes to avoid repeated
           downloads.)
        skip_existing: bool
           If True, will silently skip loading exposures that already exist in the
           database.  If False, will raise an exception on an attempt to load
           an exposure that already exists in the database.

        Returns
        -------
        A list of Exposure objects.  Depending on skip_existing, the length of this list may
        not be the same as the length of indexes.

        """
        raise NotImplementedError( f"Instrument class {self.__class__.__name__} hasn't "
                                   f"implemented download_and_commit_exposures." )

    def __len__( self ):
        """The number of exposures this object encapsulates."""
        raise NotImplementedError( f"Instrument class {self.__class__.__name__} hasn't implemented __len__." )




if __name__ == "__main__":
    inst = DemoInstrument()
