import os
import pytest
import re
import shutil
import uuid

import numpy as np
from datetime import datetime

from astropy.time import Time

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession, CODE_ROOT
from models.exposure import Exposure, SectionData
from models.instrument import Instrument, DemoInstrument
from models.decam import DECam

from tests.conftest import rnd_str


def test_exposure_instrument_provenance(sim_exposure1):
    with SmartSession() as session:
        sim_exposure1 = session.merge(sim_exposure1)
        assert sim_exposure1.id is not None
        assert sim_exposure1.provenance is not None
        assert sim_exposure1.provenance.id is not None
        assert sim_exposure1.provenance.code_version is not None
        assert sim_exposure1.provenance.parameters == {'instrument': 'DemoInstrument'}


def test_exposure_no_null_values():
    # cannot create an exposure without a filepath!
    with pytest.raises(ValueError, match='Exposure.__init__: must give at least a filepath or an instrument'):
        _ = Exposure()

    required = {
        'mjd': 58392.1,
        'exp_time': 30,
        'filter': 'r',
        'md5sum': uuid.UUID('00000000-0000-0000-0000-000000000000'),
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
        e = Exposure(filepath=f"Demo_test_{rnd_str(5)}.fits", nofile=True)
        with SmartSession() as session:
            for i in range(len(required)):
                # set the exposure to the values in "added" or None if not in "added"
                for k in required.keys():
                    setattr(e, k, added.get(k, None))

                # without all the required columns on e, it cannot be added to DB
                with pytest.raises(IntegrityError) as exc:
                    e = session.merge(e)
                    # session.merge( e.provenance.code_version )
                    # session.merge( e.provenance )
                    # session.add(e)
                    session.commit()
                    exposure_id = e.id
                session.rollback()

                if 'check constraint "exposures_filter_or_array_check"' in str(exc.value):
                    # the constraint on the filter is either filter or filter array must be not-null
                    colname = 'filter'
                elif 'check constraint "exposures_md5sum_check"' in str(exc.value):
                    # the constraint on the md5sum is that it must be not-null or md5sum_extensions must be non-null
                    colname = 'md5sum'
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
        assert e.provenance.process == 'load_exposure'
        assert e.provenance.parameters == {'instrument': e.instrument}

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
    e = Exposure(filepath=f"Demo_test_{rnd_str(5)}.fits", exp_time=30, mjd=58392.0, filter="F160W", ra=123, dec=-23,
                 project='foo', target='bar', nofile=True)

    assert e.instrument == 'DemoInstrument'
    assert e.telescope == 'DemoTelescope'
    assert isinstance(e.instrument_object, DemoInstrument)

    # check that we can override the RA value:
    assert e.ra == 123


def test_exposure_guess_decam_instrument(decam_fits_image_filename, cache_dir, data_dir):
    cache_dir = os.path.join(cache_dir, 'DECam')
    t = datetime.now()
    mjd = Time(t).mjd
    # basename = "c4d_20221002_040239_r_v1.24.fits"
    shutil.copy2(os.path.join(cache_dir, decam_fits_image_filename), os.path.join(data_dir, decam_fits_image_filename))
    e = Exposure(filepath=decam_fits_image_filename, exp_time=30, mjd=mjd,
                 filter="r", ra=123, dec=-23, project='foo', target='bar', nofile=True)

    assert e.instrument == 'DECam'
    assert isinstance(e.instrument_object, DECam)
    if os.path.isfile(e.get_fullpath()):
        os.remove(e.get_fullpath())


def test_exposure_coordinates():
    e = Exposure(filepath='foo.fits', ra=None, dec=None, nofile=True)
    assert e.ecllat is None
    assert e.ecllon is None
    assert e.gallat is None
    assert e.gallon is None

    e = Exposure(filepath='foo.fits', ra=123.4, dec=None, nofile=True)
    assert e.ecllat is None
    assert e.ecllon is None
    assert e.gallat is None
    assert e.gallon is None

    e = Exposure(filepath='foo.fits', ra=123.4, dec=56.78, nofile=True)
    assert abs(e.ecllat - 35.846) < 0.01
    assert abs(e.ecllon - 111.838) < 0.01
    assert abs(e.gallat - 33.542) < 0.01
    assert abs(e.gallon - 160.922) < 0.01


def test_exposure_load_demo_instrument_data(sim_exposure1):
    # the data is a SectionData object that lazy loads from file
    assert isinstance(sim_exposure1.data, SectionData)
    assert sim_exposure1.data.filepath == sim_exposure1.get_fullpath()
    assert sim_exposure1.data.instrument == sim_exposure1.instrument_object

    # must go to the DB to get the SensorSections first:
    sim_exposure1.instrument_object.fetch_sections()

    # loading the first (and only!) section data gives a random array
    array = sim_exposure1.data[0]
    assert isinstance(array, np.ndarray)

    # this array is random integers, but it is not re-generated each time:
    assert np.array_equal(array, sim_exposure1.data[0])

    inst = sim_exposure1.instrument_object
    assert array.shape == (inst.sections['0'].size_y, inst.sections['0'].size_x)

    # check that we can clear the cache and "re-load" data:
    sim_exposure1.data.clear_cache()
    assert not np.array_equal(array, sim_exposure1.data[0])


def test_exposure_comes_loaded_with_instrument_from_db(sim_exposure1):
    with SmartSession() as session:
        sim_exposure1 = session.merge(sim_exposure1)
        eid = sim_exposure1.id

    assert eid is not None

    # now reload this exposure from the DB:
    with SmartSession() as session:
        e2 = session.scalars(sa.select(Exposure).where(Exposure.id == eid)).first()
        assert e2 is not None
        assert e2.instrument_object is not None
        assert isinstance(e2.instrument_object, DemoInstrument)
        assert e2.instrument_object.sections is not None


def test_exposure_spatial_indexing(sim_exposure1):
    pass  # TODO: complete this test
