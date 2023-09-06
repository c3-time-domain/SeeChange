import os
import re
import io
import math
import copy
import time
import traceback
import collections.abc
import requests
import pathlib
import logging
import hashlib
from enum import Enum
from datetime import datetime, timedelta

import numpy as np
import pandas

import sqlalchemy as sa

import astropy.time
from astropy.io import fits

from models.base import Base, SmartSession, FileOnDiskMixin,_logger
from util.config import Config
from models.provenance import Provenance
from pipeline.utils import parse_dateobs, read_fits_image
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


class SensorSection(Base):
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
    #  general low-precision# instrument comparision.

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
        if isinstance(filepath, str) or isinstance(filepath, pathlib.Path):
            if section_id is None:
                return read_fits_image(filepath, ext=0, output='header')
            else:
                self.check_section_id(section_id)
                idx = self._get_fits_hdu_index_from_section_id(section_id)
                return read_fits_image(filepath, ext=idx, output='header')
        elif isinstance(filepath, list) and all( (isinstance(f, str) or isinstance(f,pathlib.Path)) for f in filepath):
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

        return np.random.poisson(10, (section.size_y, section.size_x))

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
        """Search the external repository for this instrument.

        Search the external image/exposure repository for this
        instrument for exposures that the database doesn't know about
        already.  For example, for DECam, this searches the noirlab data
        archive.

        WARNING : do not call this without some parameters that limit
        the search; otherwise, too many things will be returned, and the
        query is likely to time out or get an error from the external
        repository.

        Parameters
        ----------
        skip_exposures_in_databse: bool
           If True (default), will filter out any exposures that (as
           best can be determined) are already known in the SeeChange
           database.  If False, will include any 
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
    made about it other than that it's a sequence.

    """

    def download_exposures( self, outdir=".", indexes=None, clobber=False, existing_ok=False, session=None ):
        """Download exposures from the origin.

        Parameters
        ----------
        outdir: Path or str
           Directory where to save the files.  Filenames will be
           straight from the origin.
        indexes: list of int or None
           List of indexes into the set of origin exposures to download;
           None means download them all.
        clobber: bool
           If True, will always download and overwrite existing files.
           If False, will trust that the file is the right thing if existing_ok=True,
           otherwise will throw an exception.
        existing_ok: bool
           Only matters if clobber=False (see above)
        session: models.base.SmartSession
           Database session to use.  (A new one will be created if this
           is None, but that will lead to the returned exposures and
           members of those exposures not being bound to a session, so
           lazy-loading won't work).

        Returns
        -------
        A list of pathlib.Path for the files that were downloaded.

        """
        raise NotImplementedError( f"Instrument class {self.__class__.__name__} hasn't "
                                   f"implemented download_exposure." )

    def download_and_load_exposures( self, indexes=None, clobber=False, existing_ok=False,
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
           If True, will delete the originally downloaded files after they have
           been copied to their final location.  (This mainly exists for testing
           purposes to avoid repeated downloads.)
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
                                   f"implemented download_and_load_exposures." )

    def __len__( self ):
        """The number of exposures this object encapsulates."""
        raise NotImplementedError( f"Instrument class {self.__class__.__name__} hasn't implemented __len__." )



class DECam(Instrument):

    def __init__(self, **kwargs):
        self.name = 'DECam'
        self.telescope = 'CTIO 4.0-m telescope'
        self.aperture = 4.0
        self.focal_ratio = 2.7
        self.square_degree_fov = 3.0
        self.pixel_scale = 0.263
        self.read_time = 20.0
        self.orientation_fixed = True
        self.orientation = InstrumentOrientation.NleftEup
        # read_noise, dark_current, gain, saturation_limit, non_linearity_limit
        # are all approximate values for DECam; it varies by a lot
        # between chips
        self.read_noise = 7.0
        self.dark_current = 0.1
        self.gain = 4.0
        self.saturation_limit = 44000
        self.non_linearity_limit = 44000
        self.allowed_filters = ["g", "r", "i", "z", "Y"]

        # These numbers were measured off of the WCS solution to
        #  c4d_230804_031607_ori.fits as saved by the
        #  decat lensgrinder pipeline.
        # Ra offsets are approximately *linear* degrees -- that is, they
        #  are ΔRA * cos( dec ), where dec is the exposure dec.
        # Chips 31 and 60 are the "bad" chips, and weren't in the
        #  decat database, so their centers used centers of
        #  the nearest aligned chips along each axis.
        #
        # Notice that the "N" chips are to the south and the "S" chips
        # are to the north; this is correct! See:
        # https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/characteristics
        # will apply kwargs to attributes, and register instrument in the INSTRUMENT_INSTANCE_CACHE
        self._chip_radec_off = {
            'S29': { 'ccdnum':  1, 'dra':  -0.30358, 'ddec':   0.90579 },
            'S30': { 'ccdnum':  2, 'dra':   0.00399, 'ddec':   0.90370 },
            'S31': { 'ccdnum':  3, 'dra':   0.31197, 'ddec':   0.90294 },
            'S25': { 'ccdnum':  4, 'dra':  -0.45907, 'ddec':   0.74300 },
            'S26': { 'ccdnum':  5, 'dra':  -0.15079, 'ddec':   0.74107 },
            'S27': { 'ccdnum':  6, 'dra':   0.15768, 'ddec':   0.73958 },
            'S28': { 'ccdnum':  7, 'dra':   0.46585, 'ddec':   0.73865 },
            'S20': { 'ccdnum':  8, 'dra':  -0.61496, 'ddec':   0.58016 },
            'S21': { 'ccdnum':  9, 'dra':  -0.30644, 'ddec':   0.57787 },
            'S22': { 'ccdnum': 10, 'dra':   0.00264, 'ddec':   0.57605 },
            'S23': { 'ccdnum': 11, 'dra':   0.31167, 'ddec':   0.57493 },
            'S24': { 'ccdnum': 12, 'dra':   0.62033, 'ddec':   0.57431 },
            'S14': { 'ccdnum': 13, 'dra':  -0.77134, 'ddec':   0.41738 },
            'S15': { 'ccdnum': 14, 'dra':  -0.46272, 'ddec':   0.41468 },
            'S16': { 'ccdnum': 15, 'dra':  -0.15310, 'ddec':   0.41266 },
            'S17': { 'ccdnum': 16, 'dra':   0.15678, 'ddec':   0.41097 },
            'S18': { 'ccdnum': 17, 'dra':   0.46634, 'ddec':   0.41032 },
            'S19': { 'ccdnum': 18, 'dra':   0.77533, 'ddec':   0.41018 },
            'S8':  { 'ccdnum': 19, 'dra':  -0.77343, 'ddec':   0.25333 },
            'S9':  { 'ccdnum': 20, 'dra':  -0.46437, 'ddec':   0.25010 },
            'S10': { 'ccdnum': 21, 'dra':  -0.15423, 'ddec':   0.24804 },
            'S11': { 'ccdnum': 22, 'dra':   0.15631, 'ddec':   0.24661 },
            'S12': { 'ccdnum': 23, 'dra':   0.46667, 'ddec':   0.24584 },
            'S13': { 'ccdnum': 24, 'dra':   0.77588, 'ddec':   0.24591 },
            'S1':  { 'ccdnum': 25, 'dra':  -0.93041, 'ddec':   0.09069 },
            'S2':  { 'ccdnum': 26, 'dra':  -0.62099, 'ddec':   0.08716 },
            'S3':  { 'ccdnum': 27, 'dra':  -0.31067, 'ddec':   0.08417 },
            'S4':  { 'ccdnum': 28, 'dra':   0.00054, 'ddec':   0.08241 },
            'S5':  { 'ccdnum': 29, 'dra':   0.31130, 'ddec':   0.08122 },
            'S6':  { 'ccdnum': 30, 'dra':   0.62187, 'ddec':   0.08113 },
            'S7':  { 'ccdnum': 31, 'dra':   0.93180, 'ddec':   0.08113 },
            'N1':  { 'ccdnum': 32, 'dra':  -0.93285, 'ddec':  -0.07360 },
            'N2':  { 'ccdnum': 33, 'dra':  -0.62288, 'ddec':  -0.07750 },
            'N3':  { 'ccdnum': 34, 'dra':  -0.31207, 'ddec':  -0.08051 },
            'N4':  { 'ccdnum': 35, 'dra':  -0.00056, 'ddec':  -0.08247 },
            'N5':  { 'ccdnum': 36, 'dra':   0.31077, 'ddec':  -0.08351 },
            'N6':  { 'ccdnum': 37, 'dra':   0.62170, 'ddec':  -0.08335 },
            'N7':  { 'ccdnum': 38, 'dra':   0.93180, 'ddec':  -0.08242 },
            'N8':  { 'ccdnum': 39, 'dra':  -0.77988, 'ddec':  -0.24010 },
            'N9':  { 'ccdnum': 40, 'dra':  -0.46913, 'ddec':  -0.24376 },
            'N10': { 'ccdnum': 41, 'dra':  -0.15732, 'ddec':  -0.24624 },
            'N11': { 'ccdnum': 42, 'dra':   0.15476, 'ddec':  -0.24786 },
            'N12': { 'ccdnum': 43, 'dra':   0.46645, 'ddec':  -0.24819 },
            'N13': { 'ccdnum': 44, 'dra':   0.77723, 'ddec':  -0.24747 },
            'N14': { 'ccdnum': 45, 'dra':  -0.78177, 'ddec':  -0.40426 },
            'N15': { 'ccdnum': 46, 'dra':  -0.47073, 'ddec':  -0.40814 },
            'N16': { 'ccdnum': 47, 'dra':  -0.15836, 'ddec':  -0.41091 },
            'N17': { 'ccdnum': 48, 'dra':   0.15385, 'ddec':  -0.41244 },
            'N18': { 'ccdnum': 49, 'dra':   0.46623, 'ddec':  -0.41260 },
            'N19': { 'ccdnum': 50, 'dra':   0.77755, 'ddec':  -0.41164 },
            'N20': { 'ccdnum': 51, 'dra':  -0.62766, 'ddec':  -0.57063 },
            'N21': { 'ccdnum': 52, 'dra':  -0.31560, 'ddec':  -0.57392 },
            'N22': { 'ccdnum': 53, 'dra':  -0.00280, 'ddec':  -0.57599 },
            'N23': { 'ccdnum': 54, 'dra':   0.30974, 'ddec':  -0.57705 },
            'N24': { 'ccdnum': 55, 'dra':   0.62187, 'ddec':  -0.57650 },
            'N25': { 'ccdnum': 56, 'dra':  -0.47298, 'ddec':  -0.73648 },
            'N26': { 'ccdnum': 57, 'dra':  -0.16038, 'ddec':  -0.73922 },
            'N27': { 'ccdnum': 58, 'dra':   0.15280, 'ddec':  -0.74076 },
            'N28': { 'ccdnum': 59, 'dra':   0.46551, 'ddec':  -0.74086 },
            'N29': { 'ccdnum': 60, 'dra':  -0.31779, 'ddec':  -0.90199 },
            'N30': { 'ccdnum': 61, 'dra':  -0.00280, 'ddec':  -0.90348 },
            'N31': { 'ccdnum': 62, 'dra':   0.30889, 'ddec':  -0.90498 },
        }

        Instrument.__init__(self, **kwargs)

    @classmethod
    def get_section_ids(cls):

        """
        Get a list of SensorSection identifiers for this instrument.
        We are using the names of the FITS extensions (e.g., N12, S22, etc.).
        See ref: https://noirlab.edu/science/sites/default/files/media/archives/images/DECamOrientation.png
        """
        n_list = [f'N{i}' for i in range(1, 32)]
        s_list = [f'S{i}' for i in range(1, 32)]
        return n_list + s_list

    @classmethod
    def check_section_id(cls, section_id):
        """
        Check that the type and value of the section is compatible with the instrument.
        In this case, it must be an integer in the range [0, 63].
        """
        if not isinstance(section_id, str):
            raise ValueError(f"The section_id must be a string. Got {type(section_id)}. ")

        letter = section_id[0]
        number = int(section_id[1:])

        if letter not in ['N', 'S']:
            raise ValueError(f"The section_id must start with either 'N' or 'S'. Got {letter}. ")

        if not 1 <= number <= 31:
            raise ValueError(f"The section_id number must be in the range [1, 31]. Got {number}. ")

    def get_section_offsets(self, section_id):
        """Find the offset for a specific section.

        For DECam, these offests were determined by using the WCS
        solutions for a given exposure, and then calculated from the
        nominal pixel scale of the instrument.  The exposure center was
        taken as the average of S4 (x=2047, y=2048) and N4 (x=0,
        y=2048).  These pixel offsets do *not* correspond exactly to the
        pixel offsets that you'd get if you laid the chips down flat on
        a table positioned exactly where they are in the camera, and
        measured the centers of each chip.  But, they are the ones you'd
        use to figure out the RA and Dec of the chip centers starting
        with the exposure ra/dec (ASSUMING that it's centered between N4
        and S4) and using the noiminal instrument pixel scale (properly
        including cos(dec)); as of this writing, that nominal instrument
        pixel scale was coded to be 0.263"/pixel, which is the
        three-digit average of what
        https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/characteristics
        cites as the center pixel scale (0.2637) and edge pixel scale
        (0.2626).

        Parameters
        ----------
        section_id: int
            The identifier of the section.

        Returns
        -------
        offset_x: int
            The x offset of the section.
        offset_y: int
            The y offset of the section.

        """

        self.check_section_id(section_id)
        if section_id not in self._chip_radec_off:
            raise ValueError( f'Failed to find {section_id} in dictionary of chip offsets' )

        # x increases to the south, y increases to the east
        return ( -self._chip_radec_off[section_id]['ddec'] * 3600. / self.pixel_scale ,
                 self._chip_radec_off[section_id]['dra'] * 3600. / self.pixel_scale )

    def _make_new_section(self, section_id):
        """
        Make a single section for the DECam instrument.
        The section_id must be a valid section identifier (Si or Ni, where i is an int in [1,31])

        Returns
        -------
        section: SensorSection
            A new section for this instrument.
        """
        (dx, dy) = self.get_section_offsets(section_id)
        defective = section_id in { 'N30', 'S7' }
        return SensorSection(section_id, self.name, size_x=2048, size_y=4096,
                             offset_x=dx, offset_y=dy, defective=defective)

    def get_ra_dec_for_section( self, exposure, section_id ):
        if section_id not in self._chip_radec_off:
            raise ValueError( f"Unknown DECam section_id {section_id}" )
        return ( exposure.ra + self._chip_radec_off[section_id]['dra'] / math.cos( exposure.dec * math.pi / 180. ),
                 exposure.dec + self._chip_radec_off[section_id]['ddec'] )

    @classmethod
    def _get_fits_hdu_index_from_section_id(cls, section_id):
        """
        Return the index of the HDU in the FITS file for the DECam files.
        Since the HDUs have extension names, we can use the section_id directly
        to index into the HDU list.
        """
        cls.check_section_id(section_id)
        return section_id

    @classmethod
    def get_filename_regex(cls):
        return [r'c4d.*\.fits']

    @classmethod
    def get_short_instrument_name(cls):
        """
        Get a short name used for e.g., making filenames.
        """
        return 'c4d'

    @classmethod
    def get_short_filter_name(cls, filter):
        """
        Return the short version of each filter used by DECam.
        In this case we just return the first character of the filter name,
        e.g., shortening "g DECam SDSS c0001 4720.0 1520.0" to "g".
        """
        return filter[0:1]

    def find_origin_exposures( self, skip_exposures_in_database=True,
                               minmjd=None, maxmjd=None, filters=None,
                               containing_ra=None, containing_dec=None,
                               minexptime=None, proc_type='raw',
                               proposals=None ):
        """Search the NOIRLab data archive for exposures.

        See Instrument.find_origin_exposures for documentation; in addition:

        Parameters
        ----------
        filters: str or list of str
           The short (i.e. single character) filter names ('g', 'r',
           'i', 'z', or 'Y') to search for.  If not given, will
           return images from all filters.
        proposals: str or list of str
           The NOIRLab proposal ids to limit the search to.  If not
           given, will not filter based on proposal id.
        proc_type: str
           'raw' or 'instcal' : the processing type to get 
           from the NOIRLab data archive.

        """

        if ( containing_ra is None ) != ( containing_dec is None ):
            raise RuntimeError( f"Must specify both or neither of (containing_ra, containing_dec)" )
        if ( ( containing_ra is None ) or ( containing_dec is None ) ) and ( minmjd is None ):
            raise RuntimeError( f"Must specify either a containing ra,dec or a minmjd to find DECam exposures." )
        if containing_ra is not None:
            raise NotImplementedError( f"containing_(ra|dec) is not implemented yet for DECam" )

        # Convert mjd to iso format for astroarchive
        starttime, endtime = astropy.time.Time( [ minmjd, maxmjd ], format='mjd').to_value( 'isot' )
        # Make sure there's a Z at the end of these times; astropy
        # doesn't seem to do it, but might in the future
        if starttime[-1] != 'Z': starttime += 'Z'
        if endtime[-1] != 'Z': endtime += 'Z'
        spec = {
            "outfields" : [
                "archive_filename",
                "url",
                "instrument",
                "telescope",
                "proposal",
                "proc_type",
                "prod_type",
                "caldat",
                "dateobs_center",
                "ifilter",
                "exposure",
                "md5sum",
                "MJD-OBS",
                "DATE-OBS",
                "AIRMASS",
            ],
            "search" : [
                [ "instrument", "decam" ],
                [ "proc_type", proc_type ],
                [ "prod_type", "image" ],
                [ "dateobs_center", starttime, endtime ],
            ]
        }

        if filters is not None:
            if not isinstance( filters, collections.abc.Sequence ):
                raise TypeError( f"Error, filters must be a list or a string" )
            if isinstance( filters, str ):
                filters = [ filters ]
            else:
                filters = list( filters )

        if proposals is not None:
            if not isinstance( proposals, collections.abc.Sequence ):
                raise TypeError( f"Error, proposals must be a list or a string" )
            if isinstance( proposals, str ):
                proposals = [ proposals ]
            else:
                proposals = list( proposals )
            spec["search"].append( [ "proposal" ] + proposals )

        # TODO : implement the ability to log in via a configured username and password
        # For now, will only get exposures that are public
        apiurl = f'https://astroarchive.noirlab.edu/api/adv_search/find/?format=json&limit=0'

        def getoneresponse( json ):
            _logger.debug( f"Sending NOIRLab search query to {apiurl} with json={json}" )
            response = requests.post( apiurl, json=json )
            response.raise_for_status()
            if response.status_code == 200:
                files = pandas.DataFrame( response.json()[1:] )
            else:
                _logger.error( response.json()['errorMessage'] )
                # _logger.error( response.json()['traceback'] )     # Uncomment for API developer use
                raise RuntimeError( response.json()['errorMessage'] )
            return files

        if filters is None:
            files = getoneresponse( spec )
        else:
            files = None
            for filt in filters:
                filtspec = copy.deepcopy( spec )
                filtspec["search"].append( [ "ifilter", filt, "startswith" ] )
                newfiles = getoneresponse( filtspec )
                if not newfiles.empty:
                    files = newfiles if files is None else pandas.concat( [files, newfiles] )

        if files.empty or files is None:
            _logger.warning( f"DECam exposure search found no files." )
            return None

        if minexptime is not None:
            files = files[ files.exposure >= minexptime ]
        files.sort_values( by='dateobs_center', inplace=True )
        files['filtercode'] = files.ifilter.str[0]

        if skip_exposures_in_database:
            raise NotImplementedError( "TODO: implement skip_exposures_in_database" )

        return DECamOriginExposures( proc_type, files )


class DECamOriginExposures:
    """An object that encapsulates what was found by DECam.find_origin_exposures()"""

    def __init__( self, proc_type, frame ):
        """Should only be instantiated from DECam.find_origin_exposures()

        Parameters
        ----------
        proc_type: str
           'raw' or 'instcal'
        frame: pandas.DataFrame

        """
        self.proc_type = proc_type
        self._frame = frame

    def __len__( self ):
        return len(self._frame)

    def download_exposures( self, outdir=".", indexes=None, clobber=False, existing_ok=False ):
        outdir = pathlib.Path( outdir )
        if indexes is None:
            indexes = range( len(self._frame) )
        if not isinstance( indexes, collections.abc.Sequence ):
            indexes = [ indexes ]

        downloaded = []

        for dex in indexes:
            expinfo = self._frame.iloc[dex]
            fname = pathlib.Path( expinfo.archive_filename ).name
            fpath = pathlib.Path( outdir / fname )
            if fpath.exists():
                if clobber:
                    if not fpath.is_file():
                        _logger.error( f"download_exposures: {fpath} exists and is not a file, not overwriting." )
                        raise FileExistsError( f"{fpath} exists and is not a file, not overwriting." )
                    else:
                        fpath.unlink()
                elif existing_ok:
                    _logger.info( f"download_exposures: {fpath} exists; trusting it's the right thing" )
                    downloaded.append( fpath )
                    continue
                else:
                    _logger.error( f"download_exposures: {fpath} exists but clobber is False" )
                    raise FileExistsError( f"{fpath} exists but clobber is False" )
            countdown = 5
            success = False
            while not success:
                try:
                    starttime = time.perf_counter()
                    renew = False
                    if _logger.getEffectiveLevel() >= logging.DEBUG:
                        _logger.info( f"download_exposures: Downloading {fname} from {expinfo.url}" )
                    else:
                        _logger.info( f"download_exposures: Downloading {fname}" )
                    response = requests.get( expinfo.url )
                    response.raise_for_status()
                    midtime = time.perf_counter()
                    size = len(response.content) / 1024 / 1024 / 1024
                    _logger.info( f"...downloaded {size:.3f} GiB in {midtime-starttime:.2f} sec" )
                    with open( fpath, "wb" ) as ofp:
                        ofp.write( response.content )
                    endtime = time.perf_counter()
                    _logger.info( f"...written to disk in {endtime-midtime:.2f} sec" )
                    success = True
                except Exception as e:
                    strio = io.StringIO("")
                    traceback.print_exc( file=strio )
                    _logger.warning( f"Exception downloading from {expinfo.url}:\n{strio.getvalue()}" )
                    countdown -= 1
                    if countdown >= 0:
                        _logger.warning( f"download_exposures: Failed to download {fname}, waiting 5s and retrying." )
                        time.sleep( 5 )
                    else:
                        _logger.error( f"download_exposures: Repeated exceptions trying to download {fname}" )
                        raise e

            downloaded.append( fpath )

        return downloaded

    def download_and_load_exposures( self, indexes=None, clobber=False, existing_ok=False,
                                     delete_downloads=True, skip_existing=True, session=None ):
        outdir = pathlib.Path( FileOnDiskMixin.local_path )
        if indexes is None:
            indexes = range( len(self._frame) )
        if not isinstance( indexes, collections.abc.Sequence ):
            indexes = [ index ]

        exposures = []

        # This import is here rather than at the top of the file
        #  because Exposure imports Instrument, so we've got
        #  a circular import.  Here, instrument will have been
        #  fully initialized before we try to import Exposure,
        #  so we should be OK.
        from models.exposure import Exposure

        with SmartSession(session) as dbsess:
            codeversion = Provenance.get_code_version( session=dbsess )
            provenance = Provenance.create_or_load( code_version=codeversion, process='download',
                                                    parameters={ 'proc_type': self.proc_type },
                                                    session=dbsess )

            downloaded = self.download_exposures( outdir=outdir, indexes=indexes,
                                                  clobber=clobber, existing_ok=existing_ok )
            for dex, expfile in zip( range(len(indexes)), downloaded ):
                with fits.open( expfile ) as ifp:
                    hdr = { k: v for k, v in ifp[0].header.items()
                            if k in ( 'PROCTYPE', 'PRODTYPE', 'FILENAME', 'TELESCOP', 'OBSERVAT', 'INSTRUME'
                                      'OBS-LONG', 'OBS-LAT', 'EXPTIME', 'DARKTIME', 'OBSID',
                                      'DATE-OBS', 'TIME-OBS', 'MJD-OBS', 'OBJECT', 'PROGRAM',
                                      'OBSERVER', 'PROPID', 'FILTER', 'RA', 'DEC', 'HA', 'ZD', 'AIRMASS',
                                      'VSUB', 'GSKYPHOT', 'LSKYPHOT' ) }
                    pass
                mjd = hdr['MJD-OBS']
                exp_time = hdr['EXPTIME']
                filter = hdr['FILTER']
                project = hdr['PROPID']
                target = hdr['OBJECT']
                origin_identifier = pathlib.Path( self._frame.iloc[dex].archive_filename ).name

                ra = util.radec.parse_sexigesimal_degrees( hdr['RA'] )
                dec = util.radec.parse_sexigesimal_degrees( hdr['DEC'] )

                q = dbsess.query( Exposure ).filter( Exposure.origin_identifier==origin_identifier )
                existing = q.first()
                # Maybe check that q.count() isn't >1; if it is, throw an exception
                #  about database corruption?
                if existing is not None:
                    if skip_existing:
                        _logger.info( f"download_and_load_exposures: exposure with origin identifier "
                                      f"{origin_identifier} is already in the database, skipping. "
                                      f"({existing.filepath})" )
                        continue
                    else:
                        raise FileExistsError( f"Exposure with origin identifier {origin_identifier} "
                                               f"already exists in the database. ({existing.filepath})" )
                expobj = Exposure( current_file=expfile, invent_filepath=True,
                                   type='Sci', format='fits', provenance=provenance, ra=ra, dec=dec, 
                                   header=hdr, mjd=mjd, exp_time=exp_time, filter=filter, instrument='DECam',
                                   project=project, target=target, origin_identifier=origin_identifier )
                dbpath = outdir / expobj.filepath
                expobj.save( expfile )
                dbsess.add( expobj )
                dbsess.commit()
                if delete_downloads and ( dbpath.resolve() != expfile.resolve() ):
                    expfile.unlink()
                exposures.append( expobj )

        return exposures

if __name__ == "__main__":
    inst = DemoInstrument()
