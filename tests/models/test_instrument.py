import pytest
import time
import re
import uuid
import datetime
import hashlib
import logging

import numpy as np

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession, _logger
from models.instrument import SensorSection, Instrument, DemoInstrument, DECam
from models.exposure import Exposure


def test_base_instrument_not_implemented():
    inst = Instrument()

    with pytest.raises(NotImplementedError):
        inst.get_section_ids()

    with pytest.raises(NotImplementedError):
        inst.load('foo.bar')

    with pytest.raises(NotImplementedError):
        inst.get_filename_regex()

@pytest.mark.xfail( reason="Mostly tests fields that no longer exist in SensorSection" )
def test_global_vs_sections_values():
    inst = DemoInstrument()
    inst.name = 'TestInstrument' + uuid.uuid4().hex
    assert inst.gain == 2.0  # default value
    assert inst.read_noise == 1.5  # default value

    # make sure there are no sections matching this instrument on the DB
    with SmartSession() as session:
        sections = session.scalars(sa.select(SensorSection).where(SensorSection.instrument == inst.name)).all()
        assert len(sections) == 0

    # instrument is generated without any sections (use fetch_sections() to get them)
    assert inst.sections is None

    # cannot use get_property() without checking if there are SensorSections on the DB...
    with pytest.raises(RuntimeError, match='No sections loaded for this instrument'):
        inst.get_property(0, 'gain')

    # generate a section with null values
    with SmartSession() as session:
        inst.fetch_sections(session=session)  # must generate a new section (there are none in the DB)

    assert inst.sections is not None
    assert len(inst.sections) == 1

    # new section is created with null values
    assert inst.sections[0].gain is None
    assert inst.sections[0].read_noise is None
    assert inst.get_property(0, 'gain') == 2.0
    assert inst.get_property(0, 'read_noise') == 1.5

    # now adjust the values on that section:
    inst.sections[0].gain = 2.5
    inst.sections[0].read_noise = 1.3
    assert inst.get_property(0, 'gain') == 2.5
    assert inst.get_property(0, 'read_noise') == 1.3

    # add the new section to the DB:
    with SmartSession() as session:
        inst.commit_sections(session=session)
        sections = session.scalars(sa.select(SensorSection).where(SensorSection.instrument == inst.name)).all()
        assert len(sections) == 1  # now it is on the DB

    # make a new instrument and fetch sections
    inst2 = DemoInstrument()
    inst2.name = inst.name
    with SmartSession() as session:
        inst2.fetch_sections(session=session)
        assert inst2.get_property(0, 'gain') == 2.5
        assert inst2.get_property(0, 'read_noise') == 1.3

        t0 = datetime.datetime.utcnow()
        # re-commit the section with a validity range
        inst2.commit_sections(session=session, validity_start=t0, validity_end=t0 + datetime.timedelta(days=1))

    # new instrument should be able to fetch that section TODAY
    inst3 = DemoInstrument()
    inst3.name = inst.name

    with SmartSession() as session:
        inst3.fetch_sections(session=session)
        assert inst3.get_property(0, 'gain') == 2.5
        assert inst3.get_property(0, 'read_noise') == 1.3

    # but not if we ask for a date in the past (e.g., an image taken last week)
    inst4 = DemoInstrument()
    inst4.name = inst.name

    with SmartSession() as session:
        inst4.fetch_sections(session=session, dateobs=t0 - datetime.timedelta(days=7))
        assert inst4.get_property(0, 'gain') == 2.0
        assert inst4.get_property(0, 'read_noise') == 1.5


def test_instrument_offsets_and_filter_array_index():
    inst = DemoInstrument()
    inst.name = 'TestInstrument' + uuid.uuid4().hex
    # assert inst.gain == 2.0

    inst.fetch_sections()
    assert inst.sections is not None
    assert len(inst.sections) == 1

    # assert inst.get_property(0, 'gain') == 2.0

    # check that there's also a default offsets list
    offsets = inst.get_property(0, 'offsets')
    assert offsets is not None
    assert isinstance(offsets, tuple)
    assert offsets == (0, 0)

    offsets_x = inst.get_property(0, 'offset_x')
    assert offsets_x == 0

    offsets_y = inst.get_property(0, 'offset_y')
    assert offsets_y == 0

    # check that there's also a default filter array index list
    idx = inst.get_property(0, 'filter_array_index')
    assert idx == 0

    # for the DECam instrument, the offsets are different
    inst = DECam()
    inst.name = 'TestInstrument' + uuid.uuid4().hex
    inst.fetch_sections()

    assert inst.sections is not None
    assert len(inst.sections) > 1

    # check that there are default (no zero) offsets for other sections
    offsets = inst.get_property('N4', 'offsets')
    assert isinstance(offsets, tuple)
    assert offsets != (0, 0)

    # the filter array for DECam is also just 0 for any section
    idx = inst.get_property('N4', 'filter_array_index')
    assert idx == 0

    # Spot check the offsets of a couple of DECam chips
    offN19 = inst.get_section_offsets( 'N19' )
    offS21 = inst.get_section_offsets( 'S21' )
    assert offN19[0] == pytest.approx( 5635, abs=1 )
    assert offN19[1] == pytest.approx( 10643, abs=1 )
    assert offS21[0] == pytest.approx( -7910, abs=1 )
    assert offS21[1] == pytest.approx( -4195, abs=1 )

def test_instrument_inheritance_full_example():
    # define a new instrument class and make all the necessary overrides
    class TestInstrument(Instrument):
        def __init__(self, **kwargs):
            self.name = 'TestInstrument' + uuid.uuid4().hex
            self.telescope = 'TestTelescope'
            self.focal_ratio = np.random.uniform(1.5, 2.5)
            self.aperture = np.random.uniform(0.5, 1.5)
            self.pixel_scale = np.random.uniform(0.1, 0.2)
            self.square_degree_fov = 0.5
            self.read_noise = 1.5
            self.dark_current = 0.1
            self.size_x = 2048
            self.size_y = 4096
            self.gain = 1.2
            self.read_time = 10.0
            self.non_linearity_limit = 10000.0
            self.saturation_limit = 50000.0

            self.allowed_filters = ['r', 'g', 'b']

            # will apply kwargs to attributes, and register instrument in the INSTRUMENT_INSTANCE_CACHE
            Instrument.__init__(self, **kwargs)

        @classmethod
        def get_section_ids(cls):
            """
            Get a list of SensorSection identifiers for this instrument.
            """
            return range(10)  # let's assume this instrument has 10 sections

        @classmethod
        def check_section_id(cls, section_id):
            """
            Check if the section_id is valid for this instrument.
            The section identifier must be between 0 and 9.
            """
            if not isinstance(section_id, int):
                raise ValueError(f"section_id must be an integer. Got {type(section_id)} instead.")
            if section_id < 0 or section_id > 9:
                raise ValueError(f"section_id must be between 0 and 9. Got {section_id} instead.")

        def _make_new_section(self, identifier):
            return SensorSection(
                identifier=identifier,
                instrument=self.name,
                offset_x=0,
                offset_y=identifier*(self.size_y + 100),
            )

        def load_section_image(self, filepath, section_id):
            size_x = self.get_property(section_id, 'size_x')
            size_y = self.get_property(section_id, 'size_y')
            return np.random.poisson(10, (size_y, size_x))

        def read_header(self, filepath):
            # return a spoof header
            return {
                'RA': np.random.uniform(0, 360),
                'DEC': np.random.uniform(-90, 90),
                'EXPTIME': 25.0,  # milliseconds!!!
                'FILTER': 'r',
                'MJD': np.random.uniform(50000, 60000),
                'PROPID': '2020A-0001',
                'OBJECT': 'crab nebula',
                'TELESCOP': 'TestTelescope',
                'INSTRUME': 'TestInstrument',
                'SHUTMODE': 'ROLLING',
                'GAIN': np.random.normal(self.gain, 0.01),
            }

        @classmethod
        def get_filename_regex(cls):
            return [r'TestInstrument']

        @classmethod
        def get_auxiliary_exposure_header_keys(cls):
            return ['shutter_mode']

        @classmethod
        def _get_header_keyword_translations(cls):
            translations = Instrument._get_header_keyword_translations()
            translations.update({'shutter_mode': 'SHUTMODE'})
            return translations

        @classmethod
        def _get_header_values_converters(cls):
            # convert exp_time from ms to s:
            return {'exp_time': lambda t: t/1000.0}

    from models.instrument import register_all_instruments
    register_all_instruments()  # make sure this instrument is registered to global dictionaries

    # create an instance of the new instrument class
    inst = TestInstrument()
    inst.fetch_sections()  # there are no sections on DB, so make new ones

    assert len(inst.sections) == 10
    for i in range(10):
        assert inst.sections[i].identifier == str(i)
        assert inst.sections[i].offset_x == 0
        if i > 0:
            assert inst.sections[i].offset_y > 0

    with pytest.raises(ValueError, match='section_id must be an integer'):
        inst.get_section('0')

    with pytest.raises(ValueError, match='section_id must be between 0 and 9'):
        inst.get_section(10)

    inst.sections[1].gain = 1.6
    assert inst.get_property(0, 'gain') == 1.2
    assert inst.get_property(1, 'gain') == 1.6

    # check that the exposure object gets the correct header
    e = Exposure(filepath='TestInstrument.fits', nofile=True)
    assert e.instrument == 'TestInstrument'
    assert isinstance(e.instrument_object, TestInstrument)
    assert e.exp_time == 0.025  # needs to be converted from ms to s
    assert e.mjd is not None
    assert e.header.get('shutter_mode') == 'ROLLING'

    # allow the instrument to update with SensorSections consistent with the exposure's MJD
    with SmartSession() as session:
        e.update_instrument(session)

    im_data = e.data[0]  # load the first CCD image
    assert isinstance(im_data, np.ndarray)
    assert im_data.shape == (4096, 2048)
    assert im_data.sum() > 0
    assert abs(np.mean(im_data) - 10) < 0.1  # random numbers with Poisson distribution around lambda=10

def test_demoim_search_notimplemented():
    inst = DemoInstrument()
    with pytest.raises( NotImplementedError ):
        inst.find_origin_exposures()

@pytest.fixture(scope='module')
def decam_origin_exposures():
    decam = DECam()
    yield decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                       proposals='2023A-716082',
                                       skip_exposures_in_database=False )
    
def test_decam_search_noirlab( decam_origin_exposures ):
    origloglevel = _logger.getEffectiveLevel()
    try:
        # Make sure we'll show the things sent to the noirlab API
        # so that we have a hope of figuring out what went wrong
        # if something does go wrong.
        _logger.setLevel( logging.DEBUG )

        decam = DECam()

        # Make sure it yells at us if we don't give any constraints
        with pytest.raises( RuntimeError ):
            decam.find_origin_exposures()

        # Make sure we can find some raw exposures without filter/proposals
        originexposures = decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                                       skip_exposures_in_database=False )
        assert len(originexposures._frame) == 9
        assert set(originexposures._frame.filtercode) == { 'g', 'V', 'i', 'r', 'z' }
        assert set(originexposures._frame.proposal) == { '2023A-716082', '2023A-921384' }


        # Make sure we can find raw exposures limiting by proposal
        originexposures = decam_origin_exposures
        assert set(originexposures._frame.filtercode) == { 'i', 'r', 'g', 'z' }
        assert set(originexposures._frame.proposal) == { '2023A-716082' }

        # Make sure we can find raw exposures limiting by filter
        originexposures = decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                                       filters='r',
                                                       skip_exposures_in_database=False )
        assert len(originexposures._frame) == 2
        assert set(originexposures._frame.filtercode) == { 'r' }

        originexposures = decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                                       filters=[ 'r', 'g' ],
                                                       skip_exposures_in_database=False )
        assert len(originexposures._frame) == 4
        assert set(originexposures._frame.filtercode) == { 'r', 'g' }
    finally:
        _logger.setLevel( origloglevel )

def test_decam_download_origin_exposure( decam_origin_exposures ):
    downloaded = decam_origin_exposures.download_exposures( indexes=[ 1, 3 ] )
    try:
        assert len(downloaded) == 2
        for path, dex in zip( downloaded, [ 1, 3 ] ):
            md5 = hashlib.md5()
            with open( path, "rb") as ifp:
                md5.update( ifp.read() )
            assert md5.hexdigest() == decam_origin_exposures._frame.iloc[dex].md5sum
        import pdb; pdb.set_trace()
        pass
    finally:
        for path in downloaded:
            path.unlink()


        
    
# TODO: add more tests for e.g., loading FITS headers
