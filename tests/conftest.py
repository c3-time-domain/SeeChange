import os
import warnings
import pytest
import uuid
import shutil
import pathlib
import logging

import numpy as np

import sqlalchemy as sa

from util.config import Config
from models.base import (
    FileOnDiskMixin,
    SmartSession,
    CODE_ROOT,
    get_all_database_objects,
    setup_warning_filters
)
from models.provenance import CodeVersion, Provenance
from models.catalog_excerpt import CatalogExcerpt
from models.exposure import Exposure
from models.objects import Object

from util.archive import Archive
from util.util import remove_empty_folders
from util.retrydownload import retry_download
from util.logger import SCLogger


pytest_plugins = [
    'tests.fixtures.simulated',
    'tests.fixtures.decam',
    'tests.fixtures.ztf',
    'tests.fixtures.ptf',
    'tests.fixtures.pipeline_objects',
]

ARCHIVE_PATH = None

# We may want to turn this on only for tests, as it may add a lot of runtime/memory overhead
# ref: https://www.mail-archive.com/python-list@python.org/msg443129.html
# os.environ["SEECHANGE_TRACEMALLOC"] = "1"


# this fixture should be the first thing loaded by the test suite
# (session is the pytest session, not the SQLAlchemy session)
def pytest_sessionstart(session):
    # Will be executed before the first test

    # this is only to make the warnings into errors, so it is easier to track them down...
    # warnings.filterwarnings('error', append=True)  # comment this out in regular usage

    setup_warning_filters()  # load the list of warnings that are to be ignored (not just in tests)
    # below are additional warnings that are ignored only during tests:

    # ignore warnings from photometry code that occur for cutouts with mostly zero values
    warnings.filterwarnings('ignore', message=r'.*Background mean=.*, std=.*, normalization skipped!.*')

    # make sure to load the test config
    test_config_file = str((pathlib.Path(__file__).parent.parent / 'tests' / 'seechange_config_test.yaml').resolve())
    Config.get(configfile=test_config_file, setdefault=True)
    FileOnDiskMixin.configure_paths()
    # SCLogger.setLevel( logging.INFO )


# This will be executed after the last test (session is the pytest session, not the SQLAlchemy session)
def pytest_sessionfinish(session, exitstatus):
    # SCLogger.debug('Final teardown fixture executed! ')
    with SmartSession() as dbsession:
        # first get rid of any Exposure loading Provenances, if they have no Exposures attached
        provs = dbsession.scalars(sa.select(Provenance).where(Provenance.process == 'load_exposure'))
        for prov in provs:
            exp = dbsession.scalars(sa.select(Exposure).where(Exposure.provenance_id == prov.id)).all()
            if len(exp) == 0:
                dbsession.delete(prov)
        dbsession.commit()

        objects = get_all_database_objects(session=dbsession)
        any_objects = False
        for Class, ids in objects.items():
            # TODO: check that surviving provenances have test_parameter
            if Class.__name__ in ['CodeVersion', 'CodeHash', 'SensorSection', 'CatalogExcerpt', 'Provenance', 'Object']:
                SCLogger.debug(f'There are {len(ids)} {Class.__name__} objects in the database. These are OK to stay.')
            elif len(ids) > 0:
                SCLogger.info(
                    f'There are {len(ids)} {Class.__name__} objects in the database. Please make sure to cleanup!'
                )
                for id in ids:
                    obj = dbsession.scalars(sa.select(Class).where(Class.id == id)).first()
                    SCLogger.info(f'  {obj}')
                    any_objects = True

        # delete the CodeVersion object (this should remove all provenances as well)
        dbsession.execute(sa.delete(CodeVersion).where(CodeVersion.id == 'test_v1.0.0'))

        # remove any Object objects from tests, as these are not automatically cleaned up:
        dbsession.execute(sa.delete(Object).where(Object.is_test.is_(True)))

        dbsession.commit()

        verify_archive_database_empty = False  # set to False to avoid spurious errors at end of tests (when debugging)

        if any_objects and verify_archive_database_empty:
            raise RuntimeError('There are objects in the database. Some tests are not properly cleaning up!')

        # remove empty folders from the archive
        if ARCHIVE_PATH is not None:
            # remove catalog excerpts manually, as they are meant to survive
            with SmartSession() as session:
                catexps = session.scalars(sa.select(CatalogExcerpt)).all()
                for catexp in catexps:
                    if os.path.isfile(catexp.get_fullpath()):
                        os.remove(catexp.get_fullpath())
                    archive_file = os.path.join(ARCHIVE_PATH, catexp.filepath)
                    if os.path.isfile(archive_file):
                        os.remove(archive_file)

            remove_empty_folders( ARCHIVE_PATH, remove_root=False )

            # check that there's nothing left in the archive after tests cleanup
            if os.path.isdir(ARCHIVE_PATH):
                files = list(pathlib.Path(ARCHIVE_PATH).rglob('*'))

                if len(files) > 0 and verify_archive_database_empty:
                    raise RuntimeError(f'There are files left in the archive after tests cleanup: {files}')


@pytest.fixture(scope='session')
def download_url():
    return 'https://portal.nersc.gov/cfs/m4616/SeeChange_testing_data'


@pytest.fixture(scope='session')
def download_url():
    return 'https://portal.nersc.gov/cfs/m4616/SeeChange_testing_data'


# data that is included in the repo and should be available for tests
@pytest.fixture(scope="session")
def persistent_dir():
    return os.path.join(CODE_ROOT, 'data')


# this is a cache folder that should survive between test runs
@pytest.fixture(scope="session", autouse=True)
def cache_dir():
    path = os.path.join(CODE_ROOT, 'data/cache')
    os.makedirs(path, exist_ok=True)
    return path


# this will be configured to FileOnDiskMixin.local_path, and used as temporary data location
@pytest.fixture(scope="session")
def data_dir():
    temp_data_folder = FileOnDiskMixin.local_path
    os.makedirs(temp_data_folder, exist_ok=True)
    with open(os.path.join(temp_data_folder, 'placeholder'), 'w'):
        pass  # make an empty file inside this folder to make sure it doesn't get deleted on "remove_data_from_disk"

    # SCLogger.debug(f'temp_data_folder: {temp_data_folder}')

    yield temp_data_folder

    # remove all the files created during tests
    # make sure the test config is pointing the data_dir
    # to a different location than the rest of the data
    # shutil.rmtree(temp_data_folder)


@pytest.fixture(scope="session")
def blocking_plots():
    """
    Control how and when plots will be generated.
    There are three options for the environmental variable "INTERACTIVE".
     - It is not set: do not make any plots. blocking_plots returns False.
     - It is set to a False value: make plots, but save them, and do not show on screen/block execution.
       In this case the blocking_plots returns False, but the tests that skip if INTERACTIVE is None will run.
     - It is set to a True value: make the plots, but stop the test execution until the figure is closed.

    If a test only makes plots and does not test functionality, it should be marked with
    @pytest.mark.skipif( os.getenv('INTERACTIVE') is None, reason='Set INTERACTIVE to run this test' )

    If a test makes a diagnostic plot, that is only ever used to visually inspect the results,
    then it should be surrounded by an if blocking_plots: statement. It will only run in interactive mode.

    If a test makes a plot that should be saved to disk, it should either have the skipif mentioned above,
    or have an if os.getenv('INTERACTIVE'): statement surrounding the plot itself.
    You may want to add plt.show(block=blocking_plots) to allow the figure to stick around in interactive mode,
    on top of saving the figure at the end of the test.
    """
    import matplotlib
    backend = matplotlib.get_backend()

    # make sure there's a folder to put the plots in
    if not os.path.isdir(os.path.join(CODE_ROOT, 'tests/plots')):
        os.makedirs(os.path.join(CODE_ROOT, 'tests/plots'))

    inter = os.getenv('INTERACTIVE', False)
    if isinstance(inter, str):
        inter = inter.lower() in ('true', '1', 'on', 'yes')

    if not inter:  # for non-interactive plots, use headless plots that just save to disk
        # ref: https://stackoverflow.com/questions/15713279/calling-pylab-savefig-without-display-in-ipython
        matplotlib.use("Agg")

    yield inter

    matplotlib.use(backend)


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


@pytest.fixture(scope='session', autouse=True)
def test_config():
    return Config.get()


@pytest.fixture(scope="session", autouse=True)
def code_version():
    with SmartSession() as session:
        cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.id == 'test_v1.0.0')).first()
        if cv is None:
            cv = CodeVersion(id="test_v1.0.0")
            cv.update()
            session.add( cv )
            session.commit()
        cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.id == 'test_v1.0.0')).first()

    yield cv


@pytest.fixture
def provenance_base(code_version):
    with SmartSession() as session:
        code_version = session.merge(code_version)
        p = Provenance(
            process="test_base_process",
            code_version=code_version,
            parameters={"test_parameter": uuid.uuid4().hex},
            upstreams=[],
            is_testing=True,
        )
        p = session.merge(p)

        session.commit()

    yield p

    with SmartSession() as session:
        session.delete(p)
        session.commit()


@pytest.fixture
def provenance_extra( provenance_base ):
    with SmartSession() as session:
        provenance_base = session.merge(provenance_base)
        p = Provenance(
            process="test_base_process",
            code_version=provenance_base.code_version,
            parameters={"test_parameter": uuid.uuid4().hex},
            upstreams=[provenance_base],
            is_testing=True,
        )
        p = session.merge(p)
        session.commit()

    yield p

    with SmartSession() as session:
        session.delete(p)
        session.commit()


# use this to make all the pre-committed Image fixtures
@pytest.fixture(scope="session")
def provenance_preprocessing(code_version):
    with SmartSession() as session:
        code_version = session.merge(code_version)
        p = Provenance(
            process="preprocessing",
            code_version=code_version,
            parameters={"test_parameter": "test_value"},
            upstreams=[],
            is_testing=True,
        )

        p = session.merge(p)
        session.commit()

    yield p

    with SmartSession() as session:
        session.delete(p)
        session.commit()


@pytest.fixture(scope="session")
def provenance_extraction(code_version):
    with SmartSession() as session:
        code_version = session.merge(code_version)
        p = Provenance(
            process="extraction",
            code_version=code_version,
            parameters={"test_parameter": "test_value"},
            upstreams=[],
            is_testing=True,
        )

        p = session.merge(p)
        session.commit()

    yield p

    with SmartSession() as session:
        session.delete(p)
        session.commit()


@pytest.fixture(scope="session", autouse=True)
def archive_path(test_config):
    if test_config.value('archive.local_read_dir', None) is not None:
        archivebase = test_config.value('archive.local_read_dir')
    elif os.getenv('SEECHANGE_TEST_ARCHIVE_DIR') is not None:
        archivebase = os.getenv('SEECHANGE_TEST_ARCHIVE_DIR')
    else:
        raise ValueError('No archive.local_read_dir in config, and no SEECHANGE_TEST_ARCHIVE_DIR env variable set')

    # archive.path_base is usually /test
    archivebase = pathlib.Path(archivebase) / pathlib.Path(test_config.value('archive.path_base'))
    global ARCHIVE_PATH
    ARCHIVE_PATH = archivebase
    return archivebase


@pytest.fixture(scope="session")
def archive(test_config, archive_path):
    archive_specs = test_config.value('archive')
    if archive_specs is None:
        raise ValueError( "archive in config is None" )
    archive = Archive( **archive_specs )

    archive.test_folder_path = archive_path  # track the place where these files actually go in the test suite
    yield archive

    # try:
    #     # To tear down, we need to blow away the archive server's directory.
    #     # For the test suite, we've also mounted that directory locally, so
    #     # we can do that
    #     try:
    #         shutil.rmtree( archivebase )
    #     except FileNotFoundError:
    #         pass
    #
    # except Exception as e:
    #     warnings.warn(str(e))


@pytest.fixture( scope="module" )
def catexp(data_dir, cache_dir, download_url):
    filename = "Gaia_DR3_151.0926_1.8312_17.0_19.0.fits"
    cachepath = os.path.join(cache_dir, filename)
    filepath = os.path.join(data_dir, filename)

    if not os.path.isfile(cachepath):
        retry_download(os.path.join(download_url, filename), cachepath)

    if not os.path.isfile(filepath):
        shutil.copy2(cachepath, filepath)

    yield CatalogExcerpt.create_from_file( filepath, 'gaia_dr3' )

    if os.path.isfile(filepath):
        os.remove(filepath)

