import os
import io
import warnings
import pytest
import uuid
import shutil
import pathlib

import numpy as np

import sqlalchemy as sa

import selenium.webdriver

from util.config import Config
from models.base import (
    FileOnDiskMixin,
    SmartSession,
    CODE_ROOT,
    get_all_database_objects,
    setup_warning_filters
)
from models.knownexposure import KnownExposure, PipelineWorker
from models.provenance import CodeVersion, CodeHash, Provenance
from models.catalog_excerpt import CatalogExcerpt
from models.exposure import Exposure
from models.object import Object
from models.refset import RefSet
from models.calibratorfile import CalibratorFileDownloadLock

from util.archive import Archive
from util.util import remove_empty_folders, env_as_bool
from util.retrydownload import retry_download
from util.logger import SCLogger

# Set this to False to avoid errors about things left over in the database and archive
#   at the end of tests.  In general, we want this to be True, so we can make sure
#   that our tests are properly cleaning up after themselves.  However, the errors
#   from this can hide other errors and failures, so when debugging, set it to False.
verify_archive_database_empty = True
# verify_archive_database_empty = False


pytest_plugins = [
    'tests.fixtures.simulated',
    'tests.fixtures.decam',
    'tests.fixtures.ztf',
    'tests.fixtures.ptf',
    'tests.fixtures.pipeline_objects',
    'tests.fixtures.datastore_factory',
    'tests.fixtures.conductor',
]

ARCHIVE_PATH = None

SKIP_WARNING_TESTS = False

# We may want to turn this on only for tests, as it may add a lot of runtime/memory overhead
# ref: https://www.mail-archive.com/python-list@python.org/msg443129.html
# os.environ["SEECHANGE_TRACEMALLOC"] = "1"


# this fixture should be the first thing loaded by the test suite
# (session is the pytest session, not the SQLAlchemy session)
def pytest_sessionstart(session):
    # Will be executed before the first test
    global SKIP_WARNING_TESTS

    if False:  # this is only to make the warnings into errors, so it is easier to track them down...
        warnings.filterwarnings('error', append=True)  # comment this out in regular usage
        SKIP_WARNING_TESTS = True

    setup_warning_filters()  # load the list of warnings that are to be ignored (not just in tests)
    # below are additional warnings that are ignored only during tests:

    # ignore warnings from photometry code that occur for cutouts with mostly zero values
    warnings.filterwarnings('ignore', message=r'.*Background mean=.*, std=.*, normalization skipped!.*')

    # make sure to load the test config
    test_config_file = str((pathlib.Path(__file__).parent.parent / 'tests' / 'seechange_config_test.yaml').resolve())
    Config.get(configfile=test_config_file, setdefault=True)
    FileOnDiskMixin.configure_paths()
    # SCLogger.setLevel( logging.INFO )

    # get rid of any catalog excerpts from previous runs:
    with SmartSession() as session:
        catexps = session.scalars(sa.select(CatalogExcerpt)).all()
        for catexp in catexps:
            if os.path.isfile(catexp.get_fullpath()):
                os.remove(catexp.get_fullpath())
            session.delete(catexp)
        session.commit()

def any_objects_in_database( dbsession ):
    """Look in the database, print errors and return False if things are left behind.

    The "allowed" tables (CodeVersion, CodeHash, SensorSection,
    CatalogExcerpt, Provenance, Object, PasswordLink) will not cause
    False to be returned, but will just print a debug message.

    Parameters
    ----------
      dbsession: Session

    Returns
    -------
      True if there are only database rows in allowed tables.
      False if there are any databse rows in non-allowed tables.

    """

    objects = get_all_database_objects( session=dbsession )
    any_objects = False
    for Class, ids in objects.items():
        # TODO: check that surviving provenances have test_parameter
        if Class.__name__ in ['CodeVersion', 'CodeHash', 'SensorSection', 'CatalogExcerpt',
                              'Provenance', 'Object', 'PasswordLink']:
            SCLogger.debug(f'There are {len(ids)} {Class.__name__} objects in the database. These are OK to stay.')
        elif len(ids) > 0:
            any_objects = True
            strio = io.StringIO()
            strio.write( f'There are {len(ids)} {Class.__name__} objects in the database. '
                         f'Please make sure to cleanup!')
            for id in ids:
                obj = dbsession.scalars(sa.select(Class).where(Class._id == id)).first()
                strio.write( f'\n    {obj}' )
            SCLogger.error( strio.getvalue() )
    return any_objects

# Uncomment this fixture to run the "empty database" check after each
# test.  This can be useful in figuring out which test is leaving stuff
# behind.  Because of session scope fixtures, it will cause nearly every
# (or every) test to fail, but at least you'll have enough debug output
# to (hopefully) find the tests that are leaving behind extra stuff.
#
# NOTE -- for this to work, ironically, you have to set
# verify_archive_database_empty to False at the top of this file.
# Otherwise, at the end of all the tests, the things left over in the
# databse you are looking for will cause everything to fail, and you
# *only* get that message instead of all the error messages from here
# that you wanted to get!  (Oh, pytest.)
#
# (This is probably not practical, becasuse there is *so much* module
# and session scope stuff that lots of things are left behind by tests.
# You will have to sift through a lot of output to find what you're
# looking for.  We need a better way.)
# @pytest.fixture(autouse=True)
# def check_empty_database_at_end_of_each_test():
#     yield True
#     with SmartSession() as dbsession:
#         assert not any_objects_in_database( dbsession )

# This will be executed after the last test (session is the pytest session, not the SQLAlchemy session)
def pytest_sessionfinish(session, exitstatus):
    global verify_archive_database_empty

    # SCLogger.debug('Final teardown fixture executed! ')
    with SmartSession() as dbsession:
        # first get rid of any Exposure loading Provenances, if they have no Exposures attached
        provs = dbsession.scalars(sa.select(Provenance).where(Provenance.process == 'load_exposure'))
        for prov in provs:
            exp = dbsession.scalars(sa.select(Exposure).where(Exposure.provenance_id == prov.id)).all()
            if len(exp) == 0:
                dbsession.delete(prov)
        dbsession.commit()

        any_objects = any_objects_in_database( dbsession )

        # delete the CodeVersion object (this should remove all provenances as well,
        # and that should cascade to almost everything else)
        dbsession.execute(sa.delete(CodeVersion).where(CodeVersion._id == 'test_v1.0.0'))

        # remove any Object objects from tests, as these are not automatically cleaned up:
        dbsession.execute(sa.delete(Object).where(Object.is_test.is_(True)))

        # make sure there aren't any CalibratorFileDownloadLock rows
        # left over from tests that failed or errored out
        dbsession.execute(sa.delete(CalibratorFileDownloadLock))

        # remove RefSets, because those won't have been deleted by the provenance cascade
        dbsession.execute(sa.delete(RefSet))

        # remove any residual KnownExposures and PipelineWorkers
        dbsession.execute( sa.delete( KnownExposure ) )
        dbsession.execute( sa.delete( PipelineWorker ) )

        dbsession.commit()

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

                if len(files) > 0:
                    if verify_archive_database_empty:
                        raise RuntimeError(f'There are files left in the archive after tests cleanup: {files}')
                    else:
                        warnings.warn( f'There are files left in the archive after tests cleanup: {files}' )


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
    tdf = pathlib.Path( temp_data_folder )
    tdf.mkdir( exist_ok=True, parents=True )
    with open( tdf / 'placeholder', 'w' ):
        pass  # make an empty file inside this folder to make sure it doesn't get deleted on "remove_data_from_disk"

    # SCLogger.debug(f'temp_data_folder: {temp_data_folder}')

    yield temp_data_folder

    ( tdf / 'placeholder' ).unlink( missing_ok=True )

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
    @pytest.mark.skipif( not env_as_bool('INTERACTIVE'), reason='Set INTERACTIVE to run this test' )

    If a test makes a diagnostic plot, that is only ever used to visually inspect the results,
    then it should be surrounded by an if blocking_plots: statement. It will only run in interactive mode.

    If a test makes a plot that should be saved to disk, it should either have the skipif mentioned above,
    or have an if env_as_bool('INTERACTIVE'): statement surrounding the plot itself.
    You may want to add plt.show(block=blocking_plots) to allow the figure to stick around in interactive mode,
    on top of saving the figure at the end of the test.
    """
    import matplotlib
    backend = matplotlib.get_backend()

    # make sure there's a folder to put the plots in
    if not os.path.isdir(os.path.join(CODE_ROOT, 'tests/plots')):
        os.makedirs(os.path.join(CODE_ROOT, 'tests/plots'))

    inter = env_as_bool('INTERACTIVE')
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
    cv = CodeVersion( id="test_v1.0.0" )
    # cv.insert()
    # A test was failing on this line saying test_v1.0.0 already
    # existed.  This happened on github actions, but *not* locally.  I
    # can't figure out what's up.  So, for now, work around by just
    # doing upsert.
    cv.upsert()

    with SmartSession() as session:
        newcv = session.scalars( sa.select(CodeVersion ) ).first()
        assert newcv is not None

    yield cv

    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM code_versions WHERE _id='test_v1.0.0'" ) )
        # Verify that the code hashes got cleaned out too
        them = session.query( CodeHash ).filter( CodeHash.code_version_id == 'test_v1.0.0' ).all()
        assert len(them) == 0

@pytest.fixture
def provenance_base(code_version):
    p = Provenance(
        process="test_base_process",
        code_version_id=code_version.id,
        parameters={"test_parameter": uuid.uuid4().hex},
        upstreams=[],
        is_testing=True,
    )
    p.insert()

    yield p

    with SmartSession() as session:
        session.execute( sa.delete( Provenance ).where( Provenance._id==p.id ) )
        session.commit()


@pytest.fixture
def provenance_extra( provenance_base ):
    p = Provenance(
        process="test_base_process",
        code_version_id=provenance_base.code_version_id,
        parameters={"test_parameter": uuid.uuid4().hex},
        upstreams=[provenance_base],
        is_testing=True,
    )
    p.insert()

    yield p

    with SmartSession() as session:
        session.execute( sa.delete( Provenance ).where( Provenance._id==p.id ) )
        session.commit()


# use this to make all the pre-committed Image fixtures
@pytest.fixture(scope="session")
def provenance_preprocessing(code_version):
    p = Provenance(
        process="preprocessing",
        code_version_id=code_version.id,
        parameters={"test_parameter": "test_value"},
        upstreams=[],
        is_testing=True,
    )
    p.insert()

    yield p

    with SmartSession() as session:
        session.execute( sa.delete( Provenance ).where( Provenance._id==p.id ) )
        session.commit()


@pytest.fixture(scope="session")
def provenance_extraction(code_version):
    p = Provenance(
        process="extraction",
        code_version_id=code_version.id,
        parameters={"test_parameter": "test_value"},
        upstreams=[],
        is_testing=True,
    )
    p.insert()

    yield p

    with SmartSession() as session:
        session.execute( sa.delete( Provenance ).where( Provenance._id==p.id ) )
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
    archive_specs[ 'logger' ] = SCLogger
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


@pytest.fixture
def browser():
    opts = selenium.webdriver.FirefoxOptions()
    opts.add_argument( "--headless" )
    ff = selenium.webdriver.Firefox( options=opts )
    # This next line lets us use self-signed certs on test servers
    ff.accept_untrusted_certs = True
    yield ff
    ff.close()
    ff.quit()


@pytest.fixture( scope="session" )
def webap_url():
    return "http://webap:8081/"
