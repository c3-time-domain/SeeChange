import os
import pytest
import re

import numpy as np
from datetime import datetime

from astropy.time import Time

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession, CODE_ROOT
from models.exposure import Exposure, SectionData

from models.instrument import Instrument, DECam, DemoInstrument


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


def test_exposure_no_null_values():

    # cannot create an exposure without a filepath!
    with pytest.raises(ValueError, match='Must give a filepath'):
        _ = Exposure()

    required = {
        'mjd': 58392.1,
        'exp_time': 30,
        'filter': 'r',
        'ra': np.random.uniform(0, 360),
        'dec': np.random.uniform(-90, 90),
        'instrument': 'DemoInstrument',
        'project': 'foo',
        'target': 'bar',
    }

    added = {}

    # use non-capturing groups to extract the column name from the error message
    expr = r'(?:null value in column )(".*")(?: of relation "exposures" violates not-null constraint)'

    try:
        exposure_id = None  # make sure to delete the exposure if it is added to DB
        e = Exposure(f"Demo_test_{rnd_str(5)}.fits", nofile=True)
        with SmartSession() as session:
            for i in range(len(required)):
                # set the exposure to the values in "added" or None if not in "added"
                for k in required.keys():
                    setattr(e, k, added.get(k, None))

                # without all the required columns on e, it cannot be added to DB
                with pytest.raises(IntegrityError) as exc:
                    session.add(e)
                    session.commit()
                    exposure_id = e.id
                session.rollback()

                if "check constraint" in str(exc.value):
                    # the constraint on the filter is either filter or filter array must be not-null
                    colname = 'filter'
                else:
                    # a constraint on a column being not-null was violated
                    match_obj = re.search(expr, str(exc.value))
                    assert match_obj is not None

                    # find which column raised the error
                    colname = match_obj.group(1).replace('"', '')

                # add missing column name:
                added.update({colname: required[colname]})

        for k in required.keys():
            setattr(e, k, added.get(k, None))
        session.add(e)
        session.commit()
        exposure_id = e.id
        assert exposure_id is not None

    finally:
        # cleanup
        with SmartSession() as session:
            exposure = None
            if exposure_id is not None:
                exposure = session.scalars(sa.select(Exposure).where(Exposure.id == exposure_id)).first()
            if exposure is not None:
                session.delete(exposure)
                session.commit()


def test_exposure_guess_demo_instrument():
    e = Exposure(f"Demo_test_{rnd_str(5)}.fits", exp_time=30, mjd=58392.0, filter="F160W", ra=123, dec=-23,
                 project='foo', target='bar', nofile=True)

    assert e.instrument == 'DemoInstrument'
    assert e.telescope == 'DemoTelescope'
    assert isinstance(e.instrument_object, DemoInstrument)

    # check that we can override the RA value:
    assert e.ra == 123


def test_exposure_guess_decam_instrument():

    t = datetime.now()
    mjd = Time(t).mjd

    e = Exposure(f"DECam_examples/c4d_20221002_040239_r_v1.24.fits", exp_time=30, mjd=mjd, filter="r", ra=123, dec=-23,
                 project='foo', target='bar', nofile=True)

    assert e.instrument == 'DECam'
    assert isinstance(e.instrument_object, DECam)


def test_exposure_coordinates():
    e = Exposure('foo.fits', ra=None, dec=None, nofile=True)
    assert e.ecllat is None
    assert e.ecllon is None
    assert e.gallat is None
    assert e.gallon is None

    with pytest.raises(ValueError, match='Object must have RA and Dec set'):
        e.calculate_coordinates()

    e = Exposure('foo.fits', ra=123.4, dec=None, nofile=True)
    assert e.ecllat is None
    assert e.ecllon is None
    assert e.gallat is None
    assert e.gallon is None

    e = Exposure('foo.fits', ra=123.4, dec=56.78, nofile=True)
    assert abs(e.ecllat - 35.846) < 0.01
    assert abs(e.ecllon - 111.838) < 0.01
    assert abs(e.gallat - 33.542) < 0.01
    assert abs(e.gallon - 160.922) < 0.01


def test_exposure_load_demo_instrument_data(exposure):
    # this is a new exposure, created as a fixture (not from DB):
    assert exposure.from_db == 0

    # the data is a SectionData object that lazy loads from file
    assert isinstance(exposure.data, SectionData)
    assert exposure.data.filepath == exposure.get_fullpath()
    assert exposure.data.instrument == exposure.instrument_object

    # must go to the DB to get the SensorSections first:
    exposure.instrument_object.fetch_sections()

    # loading the first (and only!) section data gives a random array
    array = exposure.data[0]
    assert isinstance(array, np.ndarray)

    # this array is random integers, but it is not re-generated each time:
    assert np.array_equal(array, exposure.data[0])

    inst = exposure.instrument_object
    assert array.shape == (inst.sections[0].size_y, inst.sections[0].size_x)

    # check that we can clear the cache and "re-load" data:
    exposure.data.clear_cache()
    assert not np.array_equal(array, exposure.data[0])


def test_exposure_comes_loaded_with_instrument_from_db(exposure):
    with SmartSession() as session:
        session.add(exposure)
        session.commit()
        eid = exposure.id

    assert eid is not None

    with SmartSession() as session:
        e2 = session.scalars(sa.select(Exposure).where(Exposure.id == eid)).first()
        assert e2 is not None
        assert e2.instrument_object is not None
        assert isinstance(e2.instrument_object, DemoInstrument)
        assert e2.instrument_object.sections is not None


def test_exposure_spatial_indexing(exposure):
    pass  # TODO: complete this test


def test_decam_exposure(decam_example_file):
    assert os.path.isfile(decam_example_file)

    # verify we don't already have an Exposure like this on DB
    decam_example_file_short = decam_example_file[len(CODE_ROOT + '/data/'):]
    with SmartSession() as session:
        session.execute(sa.delete(Exposure).where(Exposure.filepath == decam_example_file_short))
        session.commit()

    e = Exposure(decam_example_file)

    assert e.instrument == 'DECam'
    assert isinstance(e.instrument_object, DECam)
    assert e.telescope == 'CTIO 4.0-m telescope'
    assert e.mjd == 59887.32121458
    assert e.end_mjd == 59887.32232569111
    assert e.ra == 116.32024583333332
    assert e.dec == -26.25
    assert e.exp_time == 96.0
    assert e.filepath == 'DECam_examples/c4d_221104_074232_ori.fits.fz'
    assert e.filter == 'g DECam SDSS c0001 4720.0 1520.0'
    assert not e.from_db
    assert e.header == {}
    assert e.id is None
    assert e.target == 'DECaPS-West'
    assert e.project == '2022A-724693'

    # check that we can lazy load the header from file
    assert len(e.raw_header) == 150
    assert e.raw_header['NAXIS'] == 0

    with pytest.raises(ValueError, match=re.escape('The section_id must be a string. ')):
        _ = e.data[0]

    assert isinstance(e.data['N4'], np.ndarray)
    assert e.data['N4'].shape == (4146, 2160)
    assert e.data['N4'].dtype == 'uint16'

    with pytest.raises(ValueError, match=re.escape('The section_id must be a string. ')):
        _ = e.section_headers[0]

    assert len(e.section_headers['N4']) == 100
    assert e.section_headers['N4']['NAXIS'] == 2
    assert e.section_headers['N4']['NAXIS1'] == 2160
    assert e.section_headers['N4']['NAXIS2'] == 4146

    try:
        exp_id = None
        with SmartSession() as session:
            session.add(e)
            session.commit()
            exp_id = e.id
            assert exp_id is not None

        with SmartSession() as session:
            e2 = session.scalars(sa.select(Exposure).where(Exposure.id == exp_id)).first()
            assert e2 is not None
            assert e2.id == exp_id
            assert e2.from_db

    finally:
        if exp_id is not None:
            with SmartSession() as session:
                session.delete(e)
                session.commit()


