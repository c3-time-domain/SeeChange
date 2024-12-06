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
    Psycopg2Connection,
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
from models.user import AuthUser, PasswordLink

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
    'tests.fixtures.webap',
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
        # ...I don't think this should be a TODO.  Check this, but I
        #    think the pipeline will automatically add provenances if
        #    they don't exist.  As such, the tests may implicitly
        #    add provenances they don't explicitly track.
        if Class.__name__ in ['CodeVersion', 'CodeHash', 'SensorSection', 'CatalogExcerpt',
                              'Provenance', 'Object', 'PasswordLink']:
            SCLogger.debug(f'There are {len(ids)} {Class.__name__} objects in the database. These are OK to stay.')
            continue

        # Special case handling for the 'current' Provenance Tag, which may have
        #   been added automatically by top_level.py
        if Class.__name__ == "ProvenanceTag":
            currents = []
            notcurrents = []
            for id in ids:
                obj = Class.get_by_id( id, session=dbsession )
                if obj.tag == 'current':
                    currents.append( obj )
                else:
                    notcurrents.append( obj )
            if len(currents) > 0:
                SCLogger.debug( f'There are {len(currents)} {Class.__name__} "current" objects in the database. '
                                F'These are OK to stay.' )
            objs = notcurrents
        else:
            objs = [ Class.get_by_id( i, session=dbsession) for i in ids ]

        if len(objs) > 0:
            any_objects = True
            strio = io.StringIO()
            strio.write( f'There are {len(objs)} {Class.__name__} objects in the database. '
                         f'Please make sure to cleanup!')
            for obj in objs:
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

@pytest.fixture
def provenance_tags_loaded( provenance_base, provenance_extra ):
    try:
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "INSERT INTO provenance_tags(_id,tag,provenance_id) "
                            "VALUES (%(id)s,%(tag)s,%(provid)s)",
                            { 'id': uuid.uuid4(), 'tag': 'xyzzy', 'provid': provenance_base.id } )
            cursor.execute( "INSERT INTO provenance_tags(_id,tag,provenance_id) "
                            "VALUES (%(id)s,%(tag)s,%(provid)s)",
                            { 'id': uuid.uuid4(), 'tag': 'plugh', 'provid': provenance_base.id } )
            cursor.execute( "INSERT INTO provenance_tags(_id,tag,provenance_id) "
                            "VALUES (%(id)s,%(tag)s,%(provid)s)",
                            { 'id': uuid.uuid4(), 'tag': 'plugh', 'provid': provenance_extra.id } )
            conn.commit()
        yield True
    finally:
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "DELETE FROM provenance_tags WHERE tag IN ('xyzzy', 'plugh')" )
            conn.commit()


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
    p.insert_if_needed()

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
def user():
    # username test, password test_password
    with SmartSession() as session:
        user = AuthUser( id='fdc718c3-2880-4dc5-b4af-59c19757b62d',
                         username='test',
                         displayname='Test User',
                         email='testuser@mailhog'
                        )
        user.pubkey = '''-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEArBn0QI7Z2utOz9VFCoAL
+lWSeuxOprDba7O/7EBxbPev/MsayA+MB+ILGo2UycGHs9TPBWihC9ACWPLG0tJt
q5FrqWaHPmvXMT5rb7ktsAfpZSZEWdrPfLCvBdrFROUwMvIaw580mNVm4PPb5diG
pM2b8ZtAr5gHWlBH4gcni/+Jv1ZKYh0b3sUOru9+IStvFs6ijySbHFz1e/ejP0kC
LQavMj1avBGfaEil/+NyJb0Ufdy8+IdgGJMCwFIZ15HPiIUFDRYWPsilX8ik+oYU
QBZlFpESizjEzwlHtdnLrlisQR++4dNtaILPqefw7BYMRDaf1ggYiy5dl0+ZpxYO
puvcLQlPqt8iO1v3IEuPCdMqhmmyNno0AQZq+Fyc21xRFdwXvFReuOXcgvZgZupI
XtYQTStR9t7+HL5G/3yIa1utb3KRQbFkOXRXHyppUEIr8suK++pUORrAablj/Smj
9TCCe8w5eESmQ+7E/h6M84nh3t8kSBibOlcLaNywKm3BEedQXmtu4KzLQbibZf8h
Ll/jFHv5FKYjMBbVw3ouvMZmMU+aEcaSdB5GzQWhpHtGmp+fF0bPztgTdQZrArja
Y94liagnjIra+NgHOzdRd09sN9QGZSHDanANm24lZHVWvTdMU+OTAFckY560IImB
nRVct/brmHSH0KXam2bLZFECAwEAAQ==
-----END PUBLIC KEY-----
'''
        user.privkey = {"iv": "pXz7x5YA79o+Qg4w",
                        "salt": "aBtXrLT7ds9an38nW7EgbQ==",
                        "privkey": "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"
                        }
        session.add( user )
        session.commit()

    yield True

    with SmartSession() as session:
        user = session.query( AuthUser ).filter( AuthUser.username=='test' ).all()
        for u in user:
            session.delete( u )
        session.commit()

@pytest.fixture
def admin_user():
    # In the noble tradition of routers everywhere, username admin, password admin
    with SmartSession() as session:
        user = AuthUser( id='684ece1d-c33a-4f80-b90e-646ae54021b7',
                         username='admin',
                         displayname='Admin User',
                         email='admin@mailhog',
                         isadmin=True
                        )
        user.pubkey = '''-----BEGIN PUBLIC KEY-----
 MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAw+8D5eIkY1BKHkjmiifL
 XTzpitg84xW292YWoLNd5nYCzudzZLb0wCZmIyB9Go7Qy/1vEAHctpD7u74QCevN
 eDhfryTYHwD4Cfj1756htLtT4/M6UVxBV+gpQAQjko+otM6y900Rpkq8sidvNEWO
 yZlgYU3jUDO4Xa6Pxxzfuf8W+hU1beIZosIVjqiJr4aX2ceItHJfEjKh/LtZjFsj
 2HUjPFLuU+AIFUxVdfHqPmX25OhMmCE/TffaUZUfNIE0B+kV5m9nlhdccwlhPrO0
 3mVS5rSOz4TE6TYEVmACaMYhMFI79tBk4/ld1p0ngW6VhAwW1GqQwOCUoXs9alUJ
 Zh7qmSZy3t7XK/IUIAJdPmwfBF5tKwmoahCjn2LxzS3OCXoRwKPL0hvHUpmvfJKk
 XTNX/gJ21mk4tncdbwy7y+TcfP38amLvC9axIm7rmXOL22K1FGsxBFxllzpYiRY2
 dSOtSY3roih06f4wcyOfrWhWCbsR3zLZQ54L+pn4r1jae+RqSBqhslvsJQopnsMK
 YTEaYDKcP9d+SsIh2N71VURYSVBEUko4QTBIxisPwgWewhKMN/vOOpiY1kXp2CQv
 YoIHDVu28bZHOry9/T7nw5iwrqymXr9P2cYPosjmPj2Mee3XhgMlaPxj2B6z4qiT
 lgq/YFNl4MiGzKRlUCURCgMCAwEAAQ==
 -----END PUBLIC KEY-----
'''
        user.privkey = {"iv": "P3iF0nASKzuz/9BR",
                        "salt": "tZXnze39AoYqUsGYQoAIWw==",
                        "privkey": "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"
                        }
        session.add( user )
        session.commit()

    yield True

    with SmartSession() as session:
        user = session.query( AuthUser ).filter( AuthUser.username=='admin' ).all()
        for u in user:
            session.delete( u )
        session.commit()

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
