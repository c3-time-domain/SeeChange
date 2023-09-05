import os
import pytest
import uuid
import wget
import shutil
import pathlib

import numpy as np

import sqlalchemy as sa

from astropy.time import Time

from util.config import Config
from models.base import SmartSession, CODE_ROOT
from models.provenance import CodeVersion, Provenance
from models.exposure import Exposure
from models.image import Image
from models.references import ReferenceEntry
from util import config
from util.archive import Archive


# idea taken from: https://shay-palachy.medium.com/temp-environment-variables-for-pytest-7253230bd777
# this fixture should be the first thing loaded by the test suite
@pytest.fixture(scope="session", autouse=True)
def tests_setup_and_teardown():
    # Will be executed before the first test
    # print('Initial setup fixture loaded! ')

    # make sure to load the test config
    test_config_file = str((pathlib.Path(__file__).parent.parent / 'tests' / 'seechange_config_test.yaml').resolve())

    Config.get(configfile=test_config_file, setdefault=True)

    yield
    # Will be executed after the last test
    # print('Final teardown fixture executed! ')


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


@pytest.fixture
def config_test():
    return Config.get()


@pytest.fixture(scope="session", autouse=True)
def code_version():
    with SmartSession() as session:
        cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.version == 'test_v1.0.0')).first()
    if cv is None:
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
        is_testing=True,
    )

    with SmartSession() as session:
        session.add(p)
        session.commit()
        session.refresh(p)
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
        is_testing=True,
    )

    with SmartSession() as session:
        session.add(p)
        session.commit()
        session.refresh(p)
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
            mjd=np.random.uniform(58000, 58500),
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
def decam_example_exposure(decam_example_file):
    # always destroy this Exposure object and make a new one, to avoid filepath unique constraint violations
    decam_example_file_short = decam_example_file[len(CODE_ROOT+'/data/'):]
    with SmartSession() as session:
        session.execute(sa.delete(Exposure).where(Exposure.filepath == decam_example_file_short))
        session.commit()

    exposure = Exposure(decam_example_file)
    return exposure


@pytest.fixture
def decam_example_image(decam_example_exposure):
    image = Image.from_exposure(decam_example_exposure, section_id='N1')
    image.data = image.raw_data.astype(np.float32)  # TODO: add bias/flat corrections at some point
    return image


@pytest.fixture
def decam_small_image(decam_example_image):
    image = decam_example_image
    image.data = image.data[256:256+512, 256:256+512].copy()  # make it C-contiguous
    return image


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
        im.remove_data_from_disk(remove_folders=True, purge_archive=True, session=session)


@pytest.fixture
def reference_entry(exposure_factory, provenance_base, provenance_extra):
    ref_entry = None
    try:  # remove files and DB entries at the end
        filter = np.random.choice(list('grizY'))
        target = rnd_str(6)
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        images = []

        for i in range(5):
            exp = exposure_factory()

            exp.filter = filter
            exp.target = target
            exp.project = "coadd_test"
            exp.ra = ra
            exp.dec = dec

            exp.update_instrument()
            im = Image.from_exposure(exp, section_id=0)
            im.data = im.raw_data - np.median(im.raw_data)
            im.provenance = provenance_base
            im.ra = ra
            im.dec = dec
            im.save()
            images.append(im)

        # TODO: replace with a "from_images" method?
        ref = Image.from_images(images)
        ref.data = np.mean(np.array([im.data for im in images]), axis=0)

        provenance_extra.process = 'coaddition'
        ref.provenance = provenance_extra
        ref.save()

        ref_entry = ReferenceEntry()
        ref_entry.image = ref
        ref_entry.validity_start = Time(50000, format='mjd', scale='utc').isot
        ref_entry.validity_end = Time(58500, format='mjd', scale='utc').isot
        ref_entry.section_id = 0
        ref_entry.filter = filter
        ref_entry.target = target

        with SmartSession() as session:
            session.add(ref_entry)
            session.commit()

        yield ref_entry

    finally:  # cleanup
        if ref_entry is not None:
            with SmartSession() as session:
                ref_entry = session.merge(ref_entry)
                ref = ref_entry.image
                for im in ref.source_images:
                    exp = im.exposure
                    exp.remove_data_from_disk(purge_archive=True, session=session)
                    im.remove_data_from_disk(purge_archive=True, session=session)
                    session.delete(exp)
                    session.delete(im)
                ref.remove_data_from_disk(purge_archive=True, session=session)
                session.delete(ref)  # should also delete ref_entry

                session.commit()


@pytest.fixture
def archive():
    cfg = config.Config.get()
    archive_specs = cfg.value('archive')
    if archive_specs is None:
        raise ValueError( "archive in config is None" )
    archive = Archive( **archive_specs )
    yield archive

    # To tear down, we need to blow away the archive server's directory.
    # For the test suite, we've also mounted that directory locally, so
    # we can do that
    archivebase = f"{os.getenv('SEECHANGE_TEST_ARCHIVE_DIR')}/{cfg.value('archive.path_base')}"
    try:
        shutil.rmtree( archivebase )
    except FileNotFoundError:
        pass


