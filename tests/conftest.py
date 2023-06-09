import os
import pytest
import uuid
import wget

import numpy as np

import sqlalchemy as sa

from models.base import SmartSession, CODE_ROOT
from models.provenance import CodeVersion, Provenance
from models.exposure import Exposure
from models.image import Image

def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


@pytest.fixture(scope="session", autouse=True)
def code_version():
    cv = CodeVersion(version="test_v1.0.0")
    cv.update()

    yield cv

    with SmartSession() as session:
        session.execute(sa.delete(CodeVersion).where(CodeVersion.version == 'test_v1.0.0'))
        session.commit()


@pytest.fixture
def provenance_base(code_version):
    p = Provenance(
        process="test_base_process",
        code_version=code_version,
        parameters={"test_key": uuid.uuid4().hex},
        upstreams=[],
    )

    with SmartSession() as session:
        session.add(p)
        session.commit()
        pid = p.id

    yield p

    with SmartSession() as session:
        session.execute(sa.delete(Provenance).where(Provenance.id == pid))
        session.commit()


@pytest.fixture
def provenance_extra(code_version, provenance_base):
    p = Provenance(
        process="test_base_process",
        code_version=code_version,
        parameters={"test_key": uuid.uuid4().hex},
        upstreams=[provenance_base],
    )

    with SmartSession() as session:
        session.add(p)
        session.commit()
        pid = p.id

    yield p

    with SmartSession() as session:
        session.execute(sa.delete(Provenance).where(Provenance.id == pid))
        session.commit()


@pytest.fixture
def exposure_factory():
    def factory():
        e = Exposure(
            f"Demo_test_{rnd_str(5)}.fits",
            section_id=0,
            exp_time=np.random.randint(1, 4) * 10,  # 10 to 40 seconds
            mjd=np.random.uniform(58300, 58500),
            filter=np.random.choice(list('grizY')),
            ra=np.random.uniform(0, 360),
            dec=np.random.uniform(-90, 90),
            project='foo',
            target=rnd_str(6),
            nofile=True,
        )
        return e

    return factory


def make_exposure_file(exposure):
    fullname = None
    try:  # make sure to remove file at the end
        fullname = exposure.get_fullpath()
        open(fullname, 'a').close()
        exposure.nofile = False

        yield exposure

    finally:
        with SmartSession() as session:
            exposure = session.merge(exposure)
            if exposure.id is not None:
                session.execute(sa.delete(Exposure).where(Exposure.id == exposure.id))
                session.commit()

        if fullname is not None and os.path.isfile(fullname):
            os.remove(fullname)

@pytest.fixture
def exposure(exposure_factory):
    e = exposure_factory()
    make_exposure_file(e)
    yield e


@pytest.fixture
def exposure2(exposure_factory):
    e = exposure_factory()
    make_exposure_file(e)
    yield e


@pytest.fixture
def exposure_filter_array(exposure_factory):
    e = exposure_factory()
    e.filter = None
    e.filter_array = ['r', 'g', 'r', 'i']
    make_exposure_file(e)
    yield e


@pytest.fixture
def decam_example_file():
    filename = os.path.join(CODE_ROOT, 'data/DECam_examples/c4d_221104_074232_ori.fits.fz')
    if not os.path.isfile(filename):
        url = 'https://astroarchive.noirlab.edu/api/retrieve/004d537b1347daa12f8361f5d69bc09b/'
        response = wget.download(
            url=url,
            out=os.path.join(CODE_ROOT, 'data/DECam_examples/c4d_221104_074232_ori.fits.fz')
        )
        assert response == filename

    yield filename


@pytest.fixture
def demo_image(exposure):
    exposure.update_instrument()
    im = Image.from_exposure(exposure, section_id=0)

    yield im

    with SmartSession() as session:
        im = session.merge(im)
        if im.id is not None:
            session.execute(sa.delete(Image).where(Image.id == im.id))
            session.commit()
        im.remove_data_from_disk(remove_folders=True)
