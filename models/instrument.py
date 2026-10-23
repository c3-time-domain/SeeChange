import os
import re
import copy
import pathlib
import pytz
from enum import Enum

import numpy as np
import numpy.polynomial.chebyshev

import psycopg
from psycopg import sql
import sqlalchemy as sa

import astropy.time
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance

from models.base import Base, SmartSession, UUIDMixin, PsycopgConnection
from models.provenance import Provenance
from models.enums_and_bitflags import CalibratorSetConverter, CalibratorTypeConverter, FlatTypeConverter

from pipeline.catalog_tools import Bandpass
from util.fits import read_fits_image
from util.logger import SCLogger


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


class SensorSection(Base, UUIDMixin):
    """A class to represent a section of a sensor.

    NOTE : right now, SensorSections should *not* actually be stored in
    the database.  We may start doing that again based on how we address
    Issue #487.

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
        sa.REAL,
        nullable=True,
        doc='Read noise of the sensor section (in electrons). '
    )

    dark_current = sa.Column(
        sa.REAL,
        nullable=True,
        doc='Dark current of the sensor section (in electrons/pixel/second). '
    )

    gain = sa.Column(
        sa.REAL,
        nullable=True,
        doc='Gain of the sensor section (in electrons/ADU). '
    )

    saturation_limit = sa.Column(
        sa.REAL,
        nullable=True,
        doc='Saturation level of the sensor section (in electrons). '
    )

    non_linearity_limit = sa.Column(
        sa.REAL,
        nullable=True,
        doc='Non-linearity of the sensor section (in electrons). '
    )

    defective = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default='false',
        index=True,
        doc='Whether this section is defective (i.e., if True, do not use it!). '
    )

    def __init__(self, identifier, instrument, **kwargs):
        """Create a new SensorSection object.

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
        """Check if the sensor section is identical to the other one.

        Returns True if all attributes are the same,
        not including database level attributes like id, created_at, etc.
        """

        for att in self.get_attribute_list():
            if getattr(self, att) != getattr(other, att):
                return False

        return True


class Instrument:
    """Base class for an instrument.
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
    Thus, the sensor sections act as a way to override the global
    values either in time or in space.

    """

    # dictionary of regex for filenames, pointing at instrument names
    _instrument_filename_regex = {}

    # dictionary of names of instruments, pointing to the relevant class
    _instrument_classname_to_class = {}

    # dictionary of instrument object instances, lazy loaded to be shared between exposures
    _instrument_instance_cache = {}

    @classmethod
    def register_all_instruments( cls ):
        # Any module that defines classes that derive from Instrument is
        # supposed to call <class>.regsiter_this_instrument at the
        # bottom.
        #
        # Import all the classes here so that Instrument knows about
        # them, and so that get_instrument_instance() and
        # guess_instrument() will work.
        #
        # This does mean we have to edit this
        # class method every time we add a new instrument.
        #
        # Doing this in a function rather than at the bottom of the
        # class to avoid circular imports.  Individual instruments
        # import Exposure, which imports Instrument....

        import models.ptf     # noqa: F401
        import models.decam   # noqa: F401
        import models.ls4cam  # noqa: F401


    @classmethod
    def register_this_instrument( cls ):
        """Register this instrument in the Instrument class dictionaries.

        Any module that defines a subclass of Instrument needs to call
        this method on that subclass after the class is defined.
        Otherwise, Instrument won't know about that instrument and won't
        be able to find it in get_instrument_instance or
        guess_instrument.

        """

        if cls.__name__ not in Instrument._instrument_classname_to_class:
            Instrument._instrument_classname_to_class[cls.__name__] = cls
            if cls.get_filename_regex() is not None:
                for regex in cls.get_filename_regex():
                    if not isinstance( regex, re.Pattern ):
                        raise TypeError( r"Coding error: at least one of the returns of get_instrument_regex() for "
                                         r"{cls.__name__} is not a re.Pattern" )
                    Instrument._instrument_filename_regex[regex] = cls.__name__


    @classmethod
    def get_instrument_instance( cls, instrument_name ):
        """Get the singleton instrument instance for an instrument."""

        Instrument.register_all_instruments()
        if instrument_name not in Instrument._instrument_instance_cache:
            if instrument_name not in Instrument._instrument_classname_to_class:
                raise ValueError( f"Unknown instrument {instrument_name}" )
            SCLogger.debug( f"Making instrument cache for {instrument_name}" )
            Instrument._instrument_instance_cache[ instrument_name ] = (
                Instrument._instrument_classname_to_class[ instrument_name ]() )

        return Instrument._instrument_instance_cache[ instrument_name ]


    @classmethod
    def guess_instrument( cls, filename ):
        """Find the name of the instrument from the filename.

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

        Instrument.register_all_instruments()
        instrument_list = []
        for k, v in cls._instrument_filename_regex.items():
            if k.search( filename ):
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


    def __init__(self, **kwargs):
        """Create a new Instrument.

        This should only be called at the end of the __init__() method
        of a subclass.  Any attributes that do not have any definitions
        in the subclass __init__ will be set to None (or the default
        value).  In general, kwargs will be passed into the attributes
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
        self.background_box_size = getattr( self, 'background_box_size', 256 ) # Box size for sep background estimation
        self.background_filt_size = getattr( self, 'background_filt_size', 3 ) # Filter size for sep background

        self.allowed_filters = getattr(self, 'allowed_filters', None)  # list of allowed filter (e.g., ['g', 'r', 'i'])
                                                                       # Sort this the way they shoudl be displayed

        self.orientation_fixed = ( self, 'orientation_fixed', False ) # True if sensor never rotates
        self.orientation = ( self, 'orientation', None ) # If orientation_fixed is True, one of InstrumentOrientation

        self.sections = getattr(self, 'sections', None)  # populate this using fetch_sections(), then a dict
        self._dateobs_for_sections = getattr(self, '_dateobs_for_sections', None)  # dateobs when sections were loaded
        self._dateobs_range_days = getattr(self, '_dateobs_range_days', 1.0)  # how many days from dateobs to reload

        # List of the preprocessing steps that can be applied to exposures from this
        # instrument, in order.  'overscan' must always be first.
        # All preprocessing steps that are available for an instrument are listed under preprocessing_steps_available.
        # Use image_preprocessing_dict, defined in the enums_and_bitflags file to see all possible values.
        # Subclasses of Instrument should redefine this with the subset that they
        # actually need to apply. So, if an instrument has exposures
        # that already have overscan removed, that instrument should remove 'overscan' from this list.
        # If a subclass has to add a new preprocessing step,
        # then it should add that step to enum_and_bitflags.image_preprocessing_dict,
        # and (if it's a step that includes a calibraiton image or datafile)
        # to the CalibratorTypeConverter dict in enums_and_bitflags.
        self.preprocessing_steps_available = ['overscan', 'zero', 'dark', 'linearity', 'flat', 'fringe', 'illumination']
        # If a type isn't listed, it does all steps.  Otherwise, just the steps listed.
        self.preprocessing_steps_by_type = {}
        # a list of preprocessing steps that are pre-applied to the exposure data
        self.preprocessing_steps_done = []
        self.preprocessing_step_skip_by_filter = {}  # e.g., {'g': ['fringe', 'illumination']} will skip those for g

        # nofile_steps are ones that don't have an associated file
        self.preprocessing_nofile_steps = [ 'overscan' ]

        # set the attributes from the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # add this instrument to the cache, if there isn't one already
        if self.name not in Instrument._instrument_instance_cache:
            Instrument._instrument_instance_cache[self.__class__.__name__] = self

    def __repr__(self):
        ap = None if self.aperture is None else f'{self.aperture:.1f}m'
        sc = None if self.pixel_scale is None else f'{self.pixel_scale:.2f}"/pix'
        filts = [] if self.allowed_filters is None else [",".join(self.allowed_filters)]
        return f'<Instrument {self.name} on {self.telescope} ({ap}, {sc}, {filts})'


    @classmethod
    def get_filename_regex(cls):
        """Get the regular expressions used to match filenames for this instrument.

        This is used to guess the correct instrument class to load the file
        based only on the filename. Must return a list of regular expressions.

        THIS FUNCTION MUST BE OVERRIDEN BY EACH SUBCLASS.

        """
        raise NotImplementedError("This method must be implemented by the subclass.")


    def get_section_ids(self, includebad=False):
        """Get a list of SensorSection identifiers for this instrument.

        Parameters
        ----------
          includebad: bool, default False
             Normally, instruments will omit hopeless chips (i.e. things
             that have data in data files but whose data is not usable)
             from the list.  This is not per-exposure, but "known" for
             the instrument.  (...This also assumes the list of bad
             chips is static.)  Set this to True to include those section_ids.

        Returns
        -------
           list of str

        THIS METHOD MUST BE OVERRIDEN BY THE SUBCLASS.

        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def check_section_id(self, section_id, mustbegood=False):
        """Check that the type and value of the section is compatible with the instrument.

        For example, many instruments will key the section by a running
        integer (e.g., CCD ID), while others will use a string (e.g.,
        channel 'A').

        Will raise a meaningful error if not compatible.

        Subclasses should override this method to be more specific
        (e.g., to test if an integer is in range).  IMPORTANT NOTE:
        subclasses must *always* be able to handle a string section_id,
        even if that subclass only uses integer section_ids.  (Reason:
        the "section_id" field of the Image model is type string, so
        sometimes strings will come in.)  See DemoInstrument as an example.

        THIS METHOD CAN BE OVERRIDEN TO MAKE THE CHECK MORE SPECIFIC
        (e.g., to verify that the actual section_id is valid for the
        instrument)

        """
        if not isinstance(section_id, (int, str)):
            raise ValueError(f"The section_id must be an integer or string. Got {type(section_id)}. ")

    def _make_new_section(self, identifier):
        """Make a new SensorSection object for this instrument.

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

    def get_section(self, section_id, mustbegood=False):
        """Get a section from the sections dictionary.

        The section_id is first checked for type and value compatibility,
        and then the section is loaded from the dictionary of sections.
        This method does not access the database or generate new sections.
        To make sections, use fetch_sections().

        THIS METHOD SHOULD GENERALLY NOT BE OVERRIDEN BY SUBCLASSES.
        """
        self.check_section_id(section_id, mustbegood=mustbegood)

        if self.sections is None:
            self.fetch_sections()

        return self.sections.get( str(section_id) )

    def fetch_sections(self, session=None, dateobs=None):
        """Get the sensor section objects associated with this instrument.

        TEMPORARILY REPLACED.  See Issue #487.  This was causing a
        gigantic performance hit in searching for NOIRLab references.
        Each exposure searched was on a new dateobs, so was not in the
        section cache, so was causing the database to be hit *every*
        *single* *time*... even though we know for DECam that it was
        never ever loading anything new.  So, currently, just ignore the
        database.

        We should think about the concept of date-dependent sensor
        sections and do a better implemtnation of them than one that
        caches sensor sections for exactly the dateobs we asked for.
        (Cache by ranges!)  For now, though, I've just removed this and
        put in the assumption that the sensor sections are static, for
        performance reasons.

        (It looks like it sorta tries to do range caching right now, but
        it has a dateobs_range_days thing.  Since the sensor_sections
        table has validity_start and validity_end, we should cache in
        memory here based on *that*, not on the dateobs we were looking
        for.)

        ========
        OLD DOCS

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

        END OF OLD DOCS
        ===============

        THIS METHOD SHOULD GENERALLY NOT BE OVERRIDEN BY SUBCLASSES.



        Returns
        -------
        sections: list of SensorSection
            The sensor sections associated with this instrument.

        """

        if self.sections is None:
            self.sections = { s: self._make_new_section(s) for s in self.get_section_ids(includebad=True) }

        return self.sections

        # dateobs = parse_dateobs(dateobs, output='datetime')

        # # if dateobs is too far from last time we loaded sections, reload
        # if self._dateobs_for_sections is not None:
        #     if abs(self._dateobs_for_sections - dateobs) < timedelta(self._dateobs_range_days):
        #         self.sections = None

        # # this should never happen, but still
        # if self._dateobs_for_sections is None:
        #     self.sections = None

        # # we need to get new sections
        # if self.sections is None:
        #     self.sections = {}
        #     self._dateobs_for_sections = dateobs  # track the date used to load
        #     if session is False:
        #         all_sec = []
        #     else:
        #         # load sections from DB
        #         SCLogger.debug( f"Loading sensor sections from database for {self.name}" )
        #         with SmartSession(session) as session:
        #             all_sec = session.scalars(
        #                 sa.select(SensorSection).where(
        #                     SensorSection.instrument == self.name,
        #                     sa.or_(SensorSection.validity_start.is_(None), SensorSection.validity_start <= dateobs),
        #                     sa.or_(SensorSection.validity_end.is_(None), SensorSection.validity_end >= dateobs),
        #                 ).order_by(SensorSection.modified.desc())
        #             ).all()

        #     for sid in self.get_section_ids():
        #         sec = [s for s in all_sec if s.identifier == sid]
        #         if len(sec) > 0:
        #             self.sections[sid] = sec[0]
        #         else:
        #             self.sections[sid] = self._make_new_section(sid)

        # return self.sections

    def commit_sections(self, session=None, validity_start=None, validity_end=None):
        """Commit the sensor sections associated with this instrument to the database.

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
        raise RuntimeError( "Do not commit sections until we address Issue #487." )

        with SmartSession(session) as session:
            for sec in self.sections.values():
                if sec.validity_start is None and validity_start is not None:
                    sec.validity_start = validity_start
                if sec.validity_end is None and validity_end is not None:
                    sec.validity_end = validity_end
                session.add(sec)

            session.commit()

    def get_property(self, section_id, prop):
        """Get the value of a property for a given section of the instrument.

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
        section_id = section.identifier
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
        """Get the offset of the given section from the origin of the detector.

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
            The offsets in the x and y direction (in pixels).
        """
        self.check_section_id(section_id)
        # this simple instrument defaults to zero offsets for ALL sections
        offset_x = 0
        offset_y = 0
        return offset_x, offset_y

    def get_section_filter_array_index(self, section_id):
        """Get the index in the filter array under which this section is placed.

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

    def get_section_filter(self, section_id):
        raise NotImplementedError( f"get_section_filter not implemented for {self.__class__.__name__}" )


    def load(self, filepath, section_ids=None):
        """Load a part of an exposure file, based on the section identifier.

        If the instrument does not have multiple sections, set section_ids=0.

        THIS FUNCTION SHOULD GENERALLY NOT BE OVERRIDEN BY SUBCLASSES.
        Instead, override load_section_image if necessary; often you can
        get by by just overriding _get_fits_hdu_index_from_section_id.

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
        """Load one section of an exposure file.

        The default loader uses the util.fits.read_fits_image function,
        which is a basic FITS reader utility. More advanced instruments should
        override this function to use more complex file reading code.

        THIS FUNCTION CAN BE OVERRIDEN BY EACH INSTRUMENT
        IMPLEMENTATION.  You may be able to get by just overriding
        _get_fits_hdu_index_from_section_id.

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

    def read_header(self, filepath, section_id=None):
        """Load the header from file.

        By default, instruments use a "standard" FITS header that is read
        out using util.fits.read_fits_image.
        Subclasses can override this method to use a different header format.
        Note that all keyword translations and value conversions happen later,
        in the extract_header_info function.

        THIS FUNCTION CAN BE OVERRIDEN OR EXTENDED BY SUBCLASSES IF NECESSARY.

        Parameters
        ----------
        filepath: str, Path or list of str or Path
            The filename (and full path) of the exposure file.  If an
            Exposure is associated with multiple files, this will be a
            list of filenames.  In that case, the header of the first
            file in the list is what's read.

        section_id: int or str (optional)
            The identifier of the section to load.
            If None (default), will load the header for the entire detector,
            which could be a generic header for that exposure, that doesn't
            capture any of the section-specific information.

        Returns
        -------
        header: astropy.io.fits.Header
            The header from the exposure file, as an astropy.io.fits.Header object.

        """
        if isinstance( filepath, list ):
            if not all( isinstance( f, (str, pathlib.Path) ) for f in filepath ):
                raise TypeError( "If you pass a list to read_header, it must be a list of file paths." )
            filepath = filepath[0]

        if not isinstance( filepath, (str, pathlib.Path) ):
            raise TypeError( f"filepath must be a string or path. Got {type(filepath)}" )

        if section_id is None:
            return read_fits_image(filepath, ext=0, output='header')
        else:
            self.check_section_id(section_id)
            idx = self._get_fits_hdu_index_from_section_id(section_id)
            return read_fits_image(filepath, ext=idx, output='header')


    @staticmethod
    def normalize_keyword(key):
        """Normalize the header keyword to be all uppercase and remove spaces and underscores.

        THIS FUNCTION MAY BE OVERRIDEN BY SUBCLASSES IN RARE CASES.
        """
        return key.upper().replace(' ', '').replace('_', '').replace('-', '')

    def extract_header_info(self, header, names):
        """Get information from the raw header into common column names.

        This includes keywords that are required for non-nullable columns (like MJD),
        or optional header keywords that can be included but are not critical.
        Will only extract keywords that have a translation
        (which is defined in _get_header_keyword_translations()).

        THIS FUNCTION SHOULD NOT BE OVERRIDEN BY SUBCLASSES.
        To override the header keyword translation, use _get_header_keyword_translations(),
        to add unit conversions use _get_header_values_converters().

        (However, if you must, consider using super() for most of the
        work.  As an example, see::

          ls4cam.py::LS4Cam.extract_header_info

        .)

        Parameters
        ----------
        header: fits.Header
            The raw header as loaded from the file.
        names: list of str
            The names of the columns to extract.

        Returns
        -------
        output_values: dict
            A dictionary with some of the required values from the header.

        """
        header = {self.normalize_keyword(key): value for key, value in header.items()}
        output_values = {}
        translations = self._get_header_keyword_translations()
        converters = self._get_header_values_converters()
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

    def get_auxiliary_exposure_header_keys(self):
        """Additional header keys that can be useful to have on the Exposure header.

        These will be parsed from the expousre header and put in the
        info database column.  Should *not* include anything that
        already has its own database column (mjd, exp_time, filter,
        etc.).

        THIS METHOD SHOULD BE OVERRIDEN BY SUBCLASSES, TO ADD MORE ITEMS

        """

        return []

    def get_ra_dec_for_section( self, ra, dec, section_id ):
        """Given an exposure at ra, dec, return the ra, dec of the given section.

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

    def get_ra_dec_corners_for_section( self, ra, dec, section_id ):
        """Get nominal corners of a sensor section given overall array RA and Dec.

        Parameters
        ----------
           ra, dec: float
              Position of the entire array (not of the specific section).

           section_id: str

        Returns
        -------
           dict, 12 keys: (ra|dec)_corner_(0|1)(0|1), (min|max)(ra|dec)

        """
        raise NotImplementedError( f"{self.__class__.__name__} needs to implement get_ra_dec_corners_for_section" )


    def get_ra_dec_for_section_of_exposure(self, exposure, section_id):
        """Get the RA and Dec of the center of the section of an exposure.

        See get_ra_dec_for_section

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
        return self.get_ra_dec_for_section( exposure.ra, exposure.dec, section_id )


    def get_standard_flags_image( self, section_id ):
        """Get the default flags image for the given SensorSection of this instrument.

        This is for loading, say, an observatory-standard flags image.
        It should be overriden by any subclass that has such a default
        flags image.  By default, it returns a data array of all zeros.

        NOTE: SeeChange flags images are int16, not uint16, because the
        FITS standard doesn't have a unsigned int type . Yes, yes, we
        could BSCALE it, but it's safer and simpler to just have 15 bits
        of flags to play with.

        Parameters
        ----------
        section_id: int or str
          The identifier of the section

        Returns
        -------
        A 2d numpy array of int16 with shape [ sensorsection.size_y, sensorsection.size_x ]

        """

        sec = self.get_section( section_id )
        return np.zeros( [ sec.size_y, sec.size_x ], dtype=np.int16 )

    def convert_reduced_flags_to_seechange_flags( self, flagdata ):
        """Convert the dqmask flags for source-supplied reduced images to the SeeChange bitmask.

        The default method returns an int16 array with everything !=0
        set to 1 ('bad pixel' in flag_image_bits in
        enums_and_bitfags.py).  Individual instruments should override
        this if they want to be more careful about this.

        """

        newflags = np.zeros_like( flagdata, dtype=np.int16 )
        newflags[ flagdata != 0 ] = 1
        return newflags


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
          The Image.  If None, will look at section_id
        x, y: int or float
          Position on the image in C-coordinates (0 offset).  Remember
          that numpy arrays are indexed [y, x].
        section_id:
          Ignored if image is not None.  If image is None, will get the
          default gain for this section.  If both this and image are
          None, will return the instrument default gain, or 1 if there
          is no instrument default gain.

        Returns
        -------
        float

        """
        raise NotImplementedError( f"{self.__class__.__name___} needs to implement get_gain_at_pixel" )

        # Leaving the previous implementation here commented out
        #   in case for some reason we need to fall back to it.
        # For now, all of DemoInstrument, PTF, and DECam have
        #   their own implementations.  We either need to solve
        #   Issue #369, or insist that every Instrument subclass
        #   implement get_gain_at_pixel (which is the current behavior).
        # if image is not None:
        #     return self.get_section( image.section_id ).gain
        # elif section_id is not None:
        #     return self.get_section( section_id ).gain
        # elif self.gain is not None:
        #     return self.gain
        # else:
        #     return 1.

    def average_gain( self, image, section_id=None ):
        """Get an average gain for the image.

        THIS SHOULD USUALLY BE OVERRIDDEN BY SUBCLASSES.  By default,
        it's going to assume that the gain property of the sensor
        section is good, or if it's null, that the gain property of the
        instrument is good.  Subclasses should use the image header
        information.

        Parameters
        ----------
        image: Image or None
          The Image.  If None, section_id must not be none
        section_id:
          If Image is None, pass a non-null section_id to get the default gain.

        Returns
        -------
        float

        """
        return self.get_gain_at_pixel( image, 0, 0, section_id=section_id )

    def average_saturation_limit( self, image, section_id=None ):
        """Get an average saturation limit in ADU for the image.

        THIS SHOULD USUALLY BE OVERRIDDEN BY SUBCLASSES, for the same
        reason as average_gain.

        Parameters
        ----------
        image: Image or None
          The Image.  If None, section_id must not be none
        section_id: int or str
          If Image is None, pass a non-null section_id to get the default gain.

        Returns
        -------
        float

        """
        return self.saturation_limit

    def _get_header_keyword_translations(self):
        """Get a dictionary that translates the header keywords into normalized column names.

        Each column name has a list of possible header keywords that can be used to populate it.
        When parsing the header, look for each one of these keywords, and use the first one that is found.

        THIS METHOD SHOULD BE EXTENDED BY SUBCLASSES, OR REPLACED COMPLETELY.
        """
        t = dict(
            ra=['RA', 'RADEG'],
            dec=['DEC', 'DECDEG'],
            mjd=['MJD', 'MJDOBS', 'OBSMJD'],
            project=['PROJECT', 'PROJID', 'PROPOSID', 'PROPOSAL', 'PROPID'],
            target=['TARGET', 'OBJECT', 'FIELD', 'FIELDID'],
            width=['WIDTH', 'NAXIS1'],
            height=['HEIGHT', 'NAXIS2'],
            exp_time=['EXPTIME', 'EXPOSURE'],
            filter=['FILTER', 'FILT', 'FILTER_ARRAY', 'FILTERA'],
            instrument=['INSTRUME', 'INSTRUMENT'],
            telescope=['TELESCOP', 'TELESCOPE'],
            gain=['GAIN'],
            airmass=['AIRMASS'],
            sec_id=['SEC_ID'],
        )
        return t
        # TODO: add more!

    def _get_header_values_converters(self):
        """Get a dictionary with information needed to turn raw header values into values with correct units.

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

    def _get_fits_hdu_index_from_section_id(self, section_id):
        """Translate the section_id into the index of the HDU in the FITS file.

        For example, if we have an instrument with 10 CCDs, numbered 0
        to 9, the HDU list will probably contain a generic HDU at index
        0, and the individual section information in 1 through 10, so
        the function should return section_id+1.  Another example could
        have section_id=A give an index 1, and a section_id=B give an
        index 2 (so the function will read from a dictionary to
        translate the values).

        THIS METHOD SHOULD USUALLY BE OVERRIDEN BY SUBCLASSES.  Only in
        the special case where the section IDs are [0, 1, 2, 3, ...]
        and in an exposure those correspond to FITS HDUs [1, 2, 3, 4,
        ...] (or indexes into an astropy.io.fits HDU that is the same as
        the section ID).  will this implementation work.

        Parameters
        ----------
        section_id: int or str
            The identifier of the section.

        Returns
        -------
        index: int
            The index of the HDU in the FITS file.  Note that the index
            is 1-based, as it corresponds to the index in the native
            FITS file.

        """
        self.check_section_id(section_id)
        return int(section_id) + 1

    def _get_file_index_from_section_id(self, section_id):
        """Translate the section_id into the file index in an array of filenames.

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
        self.check_section_id(section_id)
        return int(section_id)

    def get_short_instrument_name(self):
        """Get a short name used for e.g., making filenames.

        The default spits out the property __name__, or, if that doesn't
        exist, the class name.
        """
        if hasattr( self, "__name__" ):
            return self.__name__
        else:
            return self.__class__.__name__

    def get_short_filter_name(self, filter):
        """Translate the full filter name into a shorter version,

        e.g., for using in filenames.
        The default is to just return the filter name,
        but instruments that have very long filter names
        should really override this with a table lookup with short names.
        """

        return filter

    def get_full_filter_name(self, shortfilter):
        """Translate the short filter name into the full version.

        The default is to just return the short filter name,
        unless an instrument subclass has overridden these functions.
        """

        # should be overridden: default is to return the input
        return shortfilter

    def standard_apertures( self ):
        """Return standard photometry aperture radii in FWHMs.

        The first aperture on the list is the one that will be used for
        most photometry.  By default, this is 0.6732 FWHMs, which is the
        optimum for a Gaussian profile when you are sky-noise-limited.
        (If sky noise is insignificant, then the formal optimum is very
        large, but practically speaking is a little less than 2 FWHMs.)
        Undersampled instruments may want to use a larger standard
        aperture, and so should override this method.

        Returns
        -------
           list of float

        """
        return RuntimeError('We should no longer depend on instruments to give the standard apertures')
        return [ 0.6732, 1., 2., 3., 4., 5., 7., 10. ]

    def fiducial_aperture( self ):
        """Return the aperture number assumed to be 'infinite' for aperture corrections.

        Defaults to 5, which is 5*FWHM if using the base
        standard_apertures.  Instruments that override
        standard_apertures should also override this.

        Returns
        -------
          int : the index into standard_apertures() that should be used
          as the "full flux" aperture for purposes of determining
          aperture corrections.

        """
        # By default, we want to use 5*FWHM radius as the "infinite"
        # aperture.  Empirically, if we get bigger, things seem to
        # get increasingly pathological (e.g. comparing sextractor to
        # photutils).  This may be because the isophotal radius no
        # longer includes all (or even most) of the pixels in the
        # aperture, so edge effects, bad pixels, etc. aren't getting
        # flagged by sextractor.  Note that for a 2d Gaussian,
        # r=5*FWHM has 1-10^-30 of the flux.  Keep the bigger ones,
        # though, for diagnostic purposes.

        # Note that this 5 is an index, not the value... it's coincidence that the index number is 5.
        return RuntimeError('We should no longer depend on instruments to give the fiducial aperture')
        return 5

    # Gaia specific methods
    # For gaia_dr3, catdata has fields:
    # X_WORLD, Y_WORLD, MAG_G, MAGERR_G, MAG_BP, MAGERR_BP, MAG_RP, MAGERR_RP, STARPROB

    def gaia_dr3_prune_star_cat(self, catdata, gaiaminbp_rp=0.5, gaiamaxbp_rp=3.0):
        """Choose only rows from a catalog that have stars.

        Usually this is done by choosing a subset of the catalog
        data with reasonable colors, and with a high enough STARPROB.
        We also remove rows that have NaN values in any of the
        critical columns (coordinates or magnitudes)

        Parameters
        ----------
        catdata: dict or pandas.DataFrame or numpy.recarray or astropy.Table
            Must be a data structure with the following keys:
            MAG_G, MAGERR_G, MAG_BP, MAGERR_BP, MAG_RP, MAGERR_RP, STARPROB
            The data structure, when indexed on those keys, should
            return a 1D numpy array.
        gaiaminbp_rp: float, optional
            The minimum BP-RP color to keep.
        gaiamaxbp_rp: float, optional
            The maximum BP-RP color to keep.

        Returns
        -------
        Returns a copy of the input data structure, with only the
        rows that are likely to be stars.
        """
        output = copy.deepcopy(catdata)

        color = catdata['MAG_BP'] - catdata['MAG_RP']
        dex = (color >= gaiaminbp_rp) & (color <= gaiamaxbp_rp)
        dex &= catdata['STARPROB'] > 0.95
        dex &= ~np.isnan(catdata['MAG_G'])
        dex &= ~np.isnan(color)
        dex &= ~np.isnan(catdata['X_WORLD'])
        dex &= ~np.isnan(catdata['Y_WORLD'])
        dex &= ~np.isnan(catdata['PMRA'])
        dex &= ~np.isnan(catdata['PMDEC'])
        # TODO: should we also look for FLAGS??

        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = value[dex]
        else:
            output = output[dex]

        return output

    def gaia_dr3_get_skycoords(self, catdata, image_mjd=None):
        """Use the RA/Dec from a Gaia catalog data array to initialize an array of SkyCoord objects

        Parameters
        ----------
        catdata: dict or pandas.DataFrame or numpy.recarray or astropy.Table
            Must be a data structure with the following keys:
            X_WORLD, Y_WORLD, PMRA, PMDEC,
            The data structure, when indexed on those keys, should
            return a 1D numpy array.
        image_mjd: float, optional
            If given, will adjust the coordinates according to proper
            motion during the time between the Gaia epoch and the image MJD.

        Returns
        -------
        coords: astropy.coordinates.SkyCoord
            An array of SkyCoord objects
        """
        coords = SkyCoord(
            ra=catdata['X_WORLD'] * u.deg,
            dec=catdata['Y_WORLD'] * u.deg,
            # distance=Distance(parallax=wd_cat['parallax'][i] * u.mas) if wd_cat['parallax'][i] > 0 else None,
            distance=Distance(1 * u.kpc),  # the distance is not really relevant for on-sky motion
            pm_ra_cosdec=catdata['PMRA'] * u.mas / u.yr,
            pm_dec=catdata['PMDEC'] * u.mas / u.yr,
            obstime=astropy.time.Time('2016.0', format='jyear', scale='tdb'),
        )
        # 2016.0 is the reference epoch for Gaia DR3 (TODO: should be 2015.5?)
        # coordinate transform ref:
        # https://docs.astropy.org/en/stable/coordinates/apply_space_motion.html#example-use-velocity-to-compute-sky-position-at-different-epochs
        image_time = astropy.time.Time(image_mjd, format='mjd', scale='tdb')
        coords = coords.apply_space_motion(image_time)

        return coords

    def gaia_dr3_to_instrument_mag( self, filter, catdata ):
        """Transform Gaia DR3 magnitudes to instrument magnitudes.

        Could use a polynomial based on the colors, or any other method.

        SUBCLASS MUST IMPLEMENT THIS METHOD!

        Parameters
        ----------
        filter: str
            The (short) filter name of the magnitudes we want.
        catdata: dict or pandas.DataFrame or numpy.recarray or astropy.Table
            A data structure that holds the relevant data,
            that can be indexed on the following keys:
            MAG_G, MAGERR_G, MAG_BP, MAGERR_BP, MAG_RP, MAGERR_RP
            If a single magnitude is required, can pass a dict.
            If an array of magnitudes is required, can be any
            data structure that when indexed on those keys
            returns a 1D numpy array (e.g., a pandas DataFrame,
            or a named structured numpy array, or even a dict
            with ndarray values).

        Returns
        -------
        trans_mag: float or numpy array
            The instrument magnitude(s).
        trans_magerr: float or numpy array
            The instrument magnitude error(s).
        """
        raise NotImplementedError( f"{self.__name__} needs to implement gaia_dr3_to_instrument_mag" )

    # ----------------------------------------
    # Preprocessing functions.  These live here rather than
    # in pipeline/preprocessing.py because individual instruments
    # may need specific overrides for some of the steps.  For many
    # instruments, the defaults should work.

    def get_filter_bandpasses(self):
        """Get a dictionary of filter name -> Bandpass object for a list of common filters.

        The default Instrument just gives some generic filters and their bandpasses,
        but subclasses should override (or update) this dictionary with their own
        filters and bandpasses.
        """

        # we can probably do better than this, but I don't know if it makes any difference
        # ref: https://en.wikipedia.org/wiki/Photometric_system
        # wikipedia = dict(
        #     U=Bandpass(332, 398),
        #     R=Bandpass(589, 727),
        #     V=Bandpass(507, 595),
        #     G=Bandpass(400, 528),
        #     B=Bandpass(398, 492),
        #     I=Bandpass(731, 880),
        #     Z=Bandpass(824, 976),
        #     Y=Bandpass(960, 1080),
        #     J=Bandpass(1110, 1326),
        #     H=Bandpass(1476, 1784),
        #     K=Bandpass(1995, 2385),
        #     L=Bandpass(3214, 3686),
        # )

        # maybe better to use LSST filters?
        # ref: https://www.lsst.org/sites/default/files/docs/sciencebook/SB_2.pdf Table 2.1 page 11
        lsst = dict(
            u=Bandpass(320, 400),
            g=Bandpass(400, 552),
            r=Bandpass(552, 691),
            i=Bandpass(691, 818),
            z=Bandpass(818, 922),
            y=Bandpass(950, 1080),
        )

        values = lsst
        # values.update(wikipedia)  # TODO: should we remove these or add them as options as well?

        return values

    def _get_default_calibrator( self, mjd, section, calibtype='dark', filter=None, session=None ):
        """Acquire (if possible) the default externally-supplied CalibratorFile.

        Will load it into the database as both an Image and a Calibrator
        Image; may also load other default calibrator images if they
        come as a pack. WILL CALL session.commit()!

        Should not be called from outside Instrument; instead, use
        preprocessing_calibrator_files.  _get_default_calibrator method
        will assume that the default is not already in the database, and
        load it without checking first.  That other method searches the
        database first.

        WILL NEED TO BE OVERRIDDEN FOR EVERY SUBCLASS, unless the
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

    def preprocessing_calibrator_files( self, calibset, flattype, section, filter, mjd,
                                        zero_provtag=None, flat_provtag=None, fringe_provtag=None,
                                        nofetch=False ):
        """Get a dictionary of calibrator images/datafiles for a given mjd and sensor section.

        Don't call this when you're holding open a database session, as
        it may take a while to run -- it may have to download files.
        Plus, internally, it will open database sessions, and we don't
        want them to pile up.

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
          The filter (can be None for some types, e.g. zero, linearity). Should
          be the short filter name, not the long filter name.
        mjd: float
          The mjd where the calibrator params are valid
        zero_provtag: str or None
          If given, the provenance tag of the bias image to get.
        flat_provtag: str or None
          If given, the provenance tag of the flat image to get.
        fringe_provtag: str or None
          If given, the provenance tag of the fringe image to get.
        nofetch: bool
          If True, will only search the database for an
          externally_supplied calibrator.  If False (default), will call
          the instrument's _get_default_calibrators method if an
          externally_supplied calibrator isn't found in the database.
          Ignored if calibset is not externally_supplied.

        Returns
        -------
        dict with up to 12 keys:
           (zero|flat|dark|fringe|illumination|linearity)_isimage: bool
               True if the calibrator id is an image (otherwise is none, or a miscellaneous data file)
           (zero|flat|dark|fringe|illumination|linearity)_fileid: int
               Either the image_id or datafile_id of the calibrator file, or None if not found

        keys will only be included for steps in the instrument's
        preprocessing_steps list (which is set during instrument object
        construction).

        """
        section = str(section)
        # SCLogger.debug( f'Looking for calibrators for {calibset} {section}' )

        if ( calibset == 'externally_supplied' ) != ( flattype == 'externally_supplied' ):
            raise ValueError( "Doesn't make sense to have only one of calibset and flattype be externally_supplied" )

        params = {}

        expdatetime = pytz.utc.localize( astropy.time.Time( mjd, format='mjd' ).datetime )

        for calibtype in self.preprocessing_steps_available:
            if calibtype in self.preprocessing_nofile_steps:
                continue

            provtag = ( zero_provtag if calibtype=='zero'
                        else flat_provtag if calibtype=='flat'
                        else fringe_provtag if calibtype=='fringe'
                        else None )

            SCLogger.debug( f'Looking for calibrators for {section} type {calibtype}'
                            f'{f" provtag {provtag}" if provtag is not None else ""}' )

            calib = None
            with PsycopgConnection() as conn:
                cursor = conn.cursor( row_factory=psycopg.rows.dict_row )
                q = sql.SQL( "SELECT c.* FROM calibrator_files c\n"
                             "LEFT JOIN images i ON c.image_id=i._id\n"
                             "LEFT JOIN data_files d ON c.datafile_id=d._id\n"
                            )
                if provtag is not None:
                    q += sql.SQL( "LEFT JOIN provenance_tags it ON i.provenance_id=it.provenance_id\n"
                                  "LEFT JOIN provenance_tags dt ON d.provenance_id=dt.provenance_id\n" )
                q += sql.SQL( "WHERE c._calibrator_set={calibset}\n"
                              "  AND c._type={calibtype}\n"
                              "  AND c.instrument={instr}\n"
                              "  AND c.sensor_section={sec}\n"
                              "  AND ( c.validity_start IS NULL OR c.validity_start <= {expdatetime} )\n"
                              "  AND ( c.validity_end IS NULL OR c.validity_end >= {expdatetime} )\n"
                             ).format( calibset=CalibratorSetConverter.to_int(calibset),
                                       calibtype=CalibratorTypeConverter.to_int(calibtype),
                                       instr=self.name,
                                       sec=section,
                                       expdatetime=expdatetime )
                if calibtype == "flat":
                    q += ( sql.SQL( "  AND _flat_type={flat_type}\n" )
                           .format( flat_type=FlatTypeConverter.to_int(flattype) ) )
                if ( calibtype in [ 'flat', 'fringe', 'illumination' ] ) and ( filter is not None ):
                    q += sql.SQL( "  AND i.filter={filter}\n" ).format( filter=filter )
                if provtag is not None:
                    q += sql.SQL( "  AND ( it.tag={provtag} OR dt.tag={provtag} )\n" ).format( provtag=provtag )

                # Select thing with most recent validity start; if there are more than one with the same
                #   validity_start, choose the one made most recently.
                q += sql.SQL( "ORDER BY c.validity_start DESC, c.created_at DESC\n" )

                cursor.execute( q )
                rows = cursor.fetchall()

            matchedrow = None
            if len(rows) > 1:
                SCLogger.warning( f"Found {len(rows)} valid {calibtype}s for "
                                  f"{self.name} {section}, picking the latest one, or, failing "
                                  f"that, picking a 'random' one." )
            if len(rows) > 0:
                SCLogger.debug( f"Got an existing valid {calibtype} for {self.name} {section}" )
                for row in rows:
                    if row['validity_start'] is not None:
                        matchedrow = row
                        break
                if matchedrow is None:
                    matchedrow = rows[0]

            if ( ( matchedrow is None ) and
                 ( CalibratorSetConverter.to_string( calibset ) == 'externally_supplied' ) and
                 ( not nofetch )
                ):
                calib = self._get_default_calibrator( mjd, section, calibtype=calibtype, filter=filter )
                SCLogger.debug( f"Got default calibrator {calib} for {calibtype} {section}" )
                matchedrow = calib.to_dict() if calib is not None else None

            if matchedrow is None:
                params[ f'{calibtype}_isimage' ] = False
                params[ f'{calibtype}_fileid' ] = None
            else:
                if matchedrow['image_id'] is not None:
                    params[ f'{calibtype}_isimage' ] = True
                    params[ f'{calibtype}_fileid' ] = matchedrow['image_id']
                elif matchedrow['datafile_id'] is not None:
                    params[ f'{calibtype}_isimage' ] = False
                    params[ f'{calibtype}_fileid' ] = matchedrow['datafile_id']
                else:
                    raise RuntimeError( f'Data corruption: CalibratorFile {matchedrow["_id"]} has neither '
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
        header: fits.Header
          The header of the image in question.

        Returns
        -------
        list of dicts; each element has fields
          'secname': str,
          'biassec': { 'x0': int, 'x1': int, 'y0': int, 'y1': int },
          'datasec': { 'x0': int, 'x1': int, 'y0': int, 'y1': int },
        Secname is some subsection identifier, which is instrument
        specific.  By default, it will be 'A' or 'B'.  Sections are in
        C-coordinates (0-offset), using the numpy standard (i.e. x1 is
        one past the end of the region); remember that numpy arrays are
        indexed [y, x].

        """

        arrparse = re.compile( r'^\s*\[\s*(?P<x0>\d+)\s*:\s*(?P<x1>\d+)\s*,\s*(?P<y0>\d+)\s*:\s*(?P<y1>\d+)\s*\]\s*$' )
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
        header: fits.Header
          The header of the image in question.

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
                    raise ValueError( "Error, data sections aren't in a grid" )
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
                    raise ValueError( "Error, data sections aren't in a grid" )
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
            # Figure out where the data section goes in the trimmed image
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

    def overscan_trim_keywords_to_strip( self ):
        """Returns a list of header keywords that should be removed after overscan/trim.

        This should be overridden by specific instrument classes.

        """

        return []


    def overscan_and_trim( self, *args, method='median', **kwargs ):
        """Overscan and trim image.

        Positional Parameters
        ---------------------
        Can pass either one or two positional parameters

        If one: image
        image: Image
          The Image to process.  Will use Image.header
          and Image.raw_data for data

          --- OR ---

        If two: header, data
        header: fits.Header
          Image header.  Need the full header, i.e. Image.header not Image.info.
        data: numpy array
          Image data.  Must not be trimmed, i.e. must include the overscan section

        Keyword Parameters
        ------------------
          method : str, default 'median'
             If 'median', then the overscan for an image's row/column is
             the median of the overscan region for that row/column.

             If 'polymedrej', then the overscan will first calculate the
             median of all the rows/cols, then fit a kwargs['order']
             (default: 3) Chebychev polynomial along the cols/rows of
             those resultant medians, will reject points that are more
             than kwargs['sigcut'] (default 3) σ away from the fit, and
             will iterate the fit until there are no more rejections.
             The overscan value for an image's row/col is then
             interpolated from that polynomial.

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
            header = args[0].header
        elif len(args) == 2:
            if not isinstance( args[0], fits.Header ):
                raise TypeError( "header isn't a fits.Header" )
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

        for sec in sections:
            # Have to figure out whether the overscan strip is offset in x or offset in y
            isrow = ( sec['biassec']['y1'] - sec['biassec']['y0'] == sec['datasec']['y1'] - sec['datasec']['y0'] )
            iscol = ( sec['biassec']['x1'] - sec['biassec']['x0'] == sec['datasec']['x1'] - sec['datasec']['x0'] )
            axis = 1 if isrow else 0 if iscol else None
            if axis is None:
                err = ( f"Bias/Data section size mismatch: biassec=["
                        f"{sec['biassec']['x0']}:{sec['biassec']['x1']},"
                        f"{sec['biassec']['y0']}:{sec['biassec']['y1']}], datasec=["
                        f"{sec['datasec']['x0']}:{sec['datasec']['x1']},"
                        f"{sec['datasec']['y0']}:{sec['datasec']['y1']}]" )
                SCLogger.error( err )
                raise ValueError( err )

            ovscn_meds = np.median( data[ sec['biassec']['y0']:sec['biassec']['y1'] ,
                                          sec['biassec']['x0']:sec['biassec']['x1'] ],
                                    axis=axis )

            if method == 'median':
                # Figure out bias values by taking the median of the appropriate overscan sections
                pass

            elif method == 'polymedrej':
                order = kwargs['order'] if 'order' in kwargs else 3
                sigcut = kwargs['sigcut'] if 'sigcut' in kwargs else 3.
                allxvals = numpy.arange( 0, ovscn_meds.shape[0], dtype=ovscn_meds.dtype )
                xvals = allxvals.copy()
                done = False
                while not done:
                    poly = np.polynomial.chebyshev.Chebyshev.fit( xvals, ovscn_meds, order )
                    resid = ovscn_meds - poly(xvals)
                    sig = resid.std()
                    keeps = np.where( np.fabs(resid) < sigcut * sig )[0]
                    if len(keeps) == len(xvals):
                        done = True
                    else:
                        xvals = xvals[keeps]
                        ovscn_meds = ovscn_meds[keeps]
                ovscn_meds = poly( allxvals )

            else:
                raise ValueError( f"Unknown overscan method {method}" )

            # By now, the right overscan values as a function of col/row are in ovscn_meds
            if isrow:
                sec['bias'] = ovscn_meds[:, np.newaxis]
            elif iscol:
                sec['bias'] = ovscn_meds[np.newaxis, :]
            else:
                raise RuntimeError( "This should never happen." )

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
          The Image to process.  Will use Image.header for header
          and Image.data for data

          --- OR ---

        If two: header, data
        header: fits.Header
          Image header.  Need the full header, i.e. Image.header not Image.info.
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


    def get_exposure_provenance( self, proc_type='raw', method='download', code_version=None, **kwargs ):
        """Get the provenance for an exposure from this instrument.

        Also makes sure it's in the database.

        The process name will be "acquire_exposure".  The process will
        have three parameters: proc_type, method, and instrument.  The
        latter is so that images extracted from exposures of different
        instruments will have different provenances (because of
        upstreams) even though the preprocessing parameters may be
        exactly the same.

        Parameters
        ----------
          proc_type: str, default 'raw'
             What processing has this exposure been through before
             loading.  Defaults to 'raw', which indicates it's a raw
             telescope exposure.  (For DECam, this could also be
             'instcal'.)

          method: str, default 'download'
             Can be anything, but should be some indication of how the
             exposure was produced. Default 'download' indicates it was
             downloaded from some exposure source somewhere.

         code_version: CodeVersion or None
             If None, will use Provenance.get_code_version().  This is
             probably what you want, so generally don't specify this.

         Additional parameters are added to the provenance parameters.

        """

        if code_version is None:
            code_version = Provenance.get_code_version( process='acquire_exposure' )
        params = { 'instrument': self.name, 'proc_type': proc_type, 'method': method }
        params.update( kwargs )
        provenance = Provenance(
            code_version_id=code_version.id,
            process='acquire_exposure',
            parameters=params,
            upstreams=[],
        )
        provenance.insert_if_needed()
        return provenance


    def manually_load_exposure( self, filepath, origin_identifier=None, params=None,
                                proc_type='raw', method='manual_load', code_version=None,
                                exists_ok=False ):
        """Load an exposure into the database from a file on disk.

        USE THIS WITH CARE.  EXTREME CARE.  Subclasses that implement
        this need to make sure that what they do with origin_identifier
        is the right thing!  The conductor will use this field to decide
        if an exposure it finds via find_origin_exposures is the same as
        one that's already loaded into the database.  So, if you're
        manually loading exposures rather than going througha ll the
        acquire_and_commit_origin_exposures process, make sure that you
        aren't going to end up with duplicate exposires!

        Parameters
        ----------
          filepath : str or pathlib.Path
              Path to file with exposure to load

          origin_identifier : str
              VERY IMPORTANT.  See warning above.

          params : ...something
              Instrument dependent.

          proc_type, method, code_version: all passed on to get_exposure_provenance

          exists_ok: bool, default False
             If True, and an exposure with the same origin_identifier
             already exists, and the provenance is as expected, assume
             that what's there is right.  If the exposure in the
             database has a different provenance from what we would have
             loaded, raise an exception.  If exists_ok is False
             (default) and an exposure with the same origin_identifier
             exists at all, regardless of whether the provenance matches
             or not, raise an exception.

        Returns
        -------
          Exposure

        """
        raise NotImplementedError( f"{self.__class__.__name__} hasn't implemented manually_load_exposure" )


    def acquire_origin_exposure( self, identifier, params, outdir=None ):
        """Does the same thing as InstrumentOriginExposures.download_exposures.

        Works outside of the context of find_origin exposures.

        Parameters
        ----------
          identifier : str
            Identifies the image at the source of exposures.  (See
            KnownExposure.identifier or Exposure.origin_identifier.)

          params : defined differently for each subclass
            Necessary parameters for this instrument to download an
            origin exposure

          outdir : str or Path
             Directory where to write the downloaded file.  Defaults to
             FileOnDiskMixin.temp_path.

        Returns
        -------
          outpath : pathlib.Path
            The written file.

        """
        raise NotImplementedError( f"Instrument class {self.__class__.__name__} hasn't "
                                   f"implemented acquire_origin_exposure" )


    def acquire_and_commit_origin_exposure( self, identifier, params ):
        """Call acquire_origin_exposure and add the exposure to the database.

        Parameters
        ----------
          identifier : str
            Identifies the image at the source of exposures.  (See
            KnownExposure.identfier or Exposure.origin_identifier.)

          params : defined differently for each subclass
            Necessary parameters for this instrument to download an
            origin exposure

        Returns
        -------
          Exposure

        """
        raise NotImplementedError( f"Instrument class {self.__class__.__name__} hasn't "
                                   f"implemented acquire_and_commit_origin_exposure" )


    def find_origin_exposures( self,
                               skip_exposures_in_database=True,
                               skip_known_exposures=True,
                               minmjd=None,
                               maxmjd=None,
                               filters=None,
                               containing_ra=None,
                               containing_dec=None,
                               ctr_ra=None,
                               ctr_dec=None,
                               radius=None,
                               minexptime=None,
                               projects=None
                              ):
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

        skip_known_exposures: bool
           If True (default), will filter out any exposures that are
           already in the knownexposures table in the database.

        minmjd: float
           The earliest time of exposure to search (default: no limit)

        maxmjd: float
           The latest time of exposure to search (default: no limit)

        filters: str or list of str
           Filters to search.  The actual strings are
           instrument-dependent, and will match what is expected on the
           external repository.  By default, doesn't limit by filter.

        containing_ra, containing_dec: float
           Search for exposures that include this position (degrees, J2000);
           default, no position constraint.

        ctr_ra, ctr_dec, radius: float
           Search for exposures within radius degrees of these coordinates.

        minexptime: float
           Search for exposures that have this minimum exposure time in
           seconds; default, no limit.

        projects: str or list of str
           Name of the projects to search for exposures from

        Returns
        -------
        A InstrumentOriginExposures object, or None if nothing is found.

        """
        raise NotImplementedError( f"Instrument class {self.__class__.__name__} hasn't "
                                   f"implemented find_origin_exposures." )


class DemoInstrument(Instrument):
    fake_image_size_x = 512
    fake_image_size_y = 1024

    def __init__(self, **kwargs):
        self.name = 'DemoInstrument'
        self.telescope = 'DemoTelescope'
        self.aperture = 2.0
        self.focal_ratio = 5.0
        self.square_degree_fov = 0.5
        self.max_rad_degree = 0.4
        self.pixel_scale = 0.41
        self.read_time = 2.0
        self.read_noise = 1.5
        self.dark_current = 0.1
        self.gain = 2.0
        self.non_linearity_limit = 10000.0
        self.saturation_limit = 10000.0
        self.allowed_filters = ["g", "r", "i", "z", "Y"]

        # will apply kwargs to attributes, and register instrument in the INSTRUMENT_INSTANCE_CACHE
        Instrument.__init__(self, **kwargs)

        # DemoInstrument doesn't know how to preprocess
        self.preprocessing_steps_available = []
        self.preprocessing_steps_done = ['overscan', 'linearity', 'flat', 'fringe']


    @classmethod
    def get_filename_regex(cls):
        return [ re.compile(r'^Demo') ]


    def get_section_ids(self, includebad=False):
        """Get a list of SensorSection identifiers for this instrument.

        See Instrument.get_section_ids for interface.
        """

        return [ '0' ]

    def check_section_id(self, section_id, mustbegood=False):
        """Check if the section_id is valid for this instrument.

        The demo instrument only has one section, so the section_id must be 0.
        """
        if ( not isinstance(section_id, (str, int)) ) or ( int(section_id) != 0 ):
            raise ValueError(f"section_id must be 0 for this instrument. Got {section_id} instead.")

    def _make_new_section(self, identifier):
        """Make a single section for the DEMO instrument.

        The identifier must be a valid section identifier.

        Returns
        -------
        section: SensorSection
            A new section for this instrument.
        """
        return SensorSection(identifier, self.name, size_x=self.fake_image_size_x, size_y=self.fake_image_size_y)

    def load_section_image(self, filepath, section_id):
        """A spoof load method for this demo instrument.

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

        rng = np.random.default_rng()
        return np.array( rng.poisson(10., (section.size_y, section.size_x)), dtype='=f4' )

    def read_header(self, filepath, section_id=None):
        # return a spoof header
        rng = np.random.default_rng()
        return fits.Header( {
            'RA': rng.uniform(0, 360),
            'DEC': rng.uniform(-90, 90),
            'EXPTIME': 30.0,
            'FILTER': rng.choice(self.allowed_filters),
            'MJD': rng.uniform(50000, 60000),
            'PROPID': '2020A-0001',
            'OBJECT': 'crab nebula',
            'TELESCOP': self.telescope,
            'INSTRUME': self.name,
            'GAIN': rng.normal(self.gain, 0.01),
        } )

    def get_gain_at_pixel( self, image, x, y, section_id=None ):
        return self.gain

    def get_short_instrument_name(self):
        """Get a short name used for e.g., making filenames."""
        return 'Demo'


# Register the instrument in the Instrument dictionaries
DemoInstrument.register_this_instrument()


class InstrumentOriginExposures:
    """A class encapsulating the response from Instrument.find_origin_exposures()

    Never instantiate one of these (or a subclass) directly; get it from
    find_origin_exposures().

    Must be subclassed by each instrument that defines
    find_origin_exposures().  The internal storage of the exposures will
    differ for each instrument, and no external assumptions should be
    made about it other than that it's a sequence (and so can be indexed).

    """

    def exposure_coords( self, index ):
        """Return central ra, dec of exposure.

        Parameters
        ----------
          index: int
            Index into encaspulated exposures

        Returns
        -------
          ra, dec : float, float

        """
        raise NotImplementedError( f"{self.__class__.__name__} hasn't implemented exposure_coords" )

    def exposure_depth( self, index ):
        """Return a number in magnitudes that relates to depth of exposure.

        The meaning of this is likely to be instrument-dependent.

        Parameters
        ----------
          index: int
            Index into encapsulated exposures

        Returns
        -------
          depth : float

        """
        raise NotImplementedError( f"{self.__class__.__name__} has't implemented exposure_depth" )


    def exposure_filter( self, index ):
        """Return the filter of an exposure.

        Parameters
        ----------
          index: int
            Index into encapsulated exposures

        Returns
        -------
          filter: str

        """
        raise NotImplementedError( f"{self.__class__.__name__} has't implemented exposure_filter" )


    def exposure_seeing( self, index ):
        """Return the seeing (arcsec) of the exposure.

        Parameters
        ----------
          index: int
            Index into encapsulated exposures

        Returns
        -------
          seeing: float

        """
        raise NotImplementedError( f"{self.__class__.__name__} has't implemented exposure_seeing" )


    def exposure_exptime( self, index ):
        """Return the exposure time (sec) of the exposure.

        Parameters
        ----------
          index: int
            Index into encapsulated exposures

        Returns
        -------
          exptime: float

        """
        raise NotImplementedError( f"{self.__class__.__name__} hasn't implemented exposure_exptime" )


    def exposure_origin_identifier( self, index ):
        """Return the origin identifier of the exposure.

        This is the thing that eventually goes into KnownExposure.identifier and Exposure.origin_identifier .

        Parameters
        ----------
          index: int
            Index into encapsulated exposures

        Returns
        -------
          identifier: str

        """
        raise NotImplementedError( f"{self.__class__.__name__} hasn't implemented exposure_origin_identifier" )


    def add_to_known_exposures( self,
                                indexes=None,
                                hold=False,
                                skip_loaded_exposures=True,
                                skip_duplicates=True,
                                session=None ):
        """Add exposures to the knownexposures table.

        Parameters
        ----------
        indexes: list of int or None, default None
          List of indexes into the set of origin exposures to add;
          None means add them all.

        hold: bool, default False
          The "hold" field to set in the KnownExposures table.  (The
          conductor will not hand out exposures to pipeline processes
          for rows where hold is True.)

        skip_duplicates: bool, default True
          Don't create duplicate entries in the knownexposures table.
          If the exposure is one that's already in the table, don't add
          a new record for it.  You probably always want to leave this
          as True.

        skip_loaded_exposures: bool, default True
          If True, then try to figure out if this exposure is one that
          is already loaded into the exposures table in the database.
          If it is, then don't add it to known_exposures.  Generally,
          you will want this to be true.

        session: Session, default None
          Database session to use, or None.

        """
        raise NotImplementedError( f"Instrument class {self.__class__.__name__} hasn't "
                                   f"implemented add_to_known_exposures." )


    def download_exposures( self, outdir=".", indexes=None, onlyexposures=True,
                            clobber=False, existing_ok=False, skip_existing=True,
                            skip_failures=False, session=None ):
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
           Only matters if clobber=False (see above).  Note that this is
           a different thing from skip_existing; the parmaeter names are
           perhaps unfortuate.

        skip_existing: bool
           If True, will silently skip downloading exposures that
           already exist in the database.  If False, will raise an
           exception on an attempt to load an exposure that already
           exists in the database.

        session: Session
           Database session to use.  (A new one will be created if this
           is None, but that will lead to the returned exposures and
           members of those exposures not being bound to a session, so
           lazy-loading won't work).

        Returns
        -------
        indexes, downloaded

        indexes are the indexes that were actually downloaded.  If
        skip_failures is False and indexes was passed, this should be
        exactly the same.  If skip_failures is True, then the ones whose
        downloads failed will not be included.

        downloaded is aw list of dictionaries, each element of which is {
        prod_type: pathlib.Path } prod_type will generally only be
        "exposure", but if there are ancillary exposures, there may be
        others.  E.g., when downloading reduced DECam images from
        NOIRlab, prod_type may be all of 'exposure', 'weight', 'dqmask'.

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
