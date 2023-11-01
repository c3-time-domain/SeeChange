import os
import warnings
import pytest
import uuid
import wget
import shutil
import pathlib

import numpy as np

import sqlalchemy as sa

from astropy.time import Time
from astropy.io import fits

from util.config import Config
from models.base import FileOnDiskMixin, SmartSession, CODE_ROOT, _logger
from models.provenance import CodeVersion, Provenance
from models.exposure import Exposure
from models.image import Image
from models.datafile import DataFile
from models.references import ReferenceEntry
from models.instrument import Instrument, get_instrument_instance
from models.decam import DECam
from models.source_list import SourceList
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
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

    with SmartSession() as session:
        # Tests are leaving behind (at least) exposures and provenances.
        # Ideally, they should all clean up after themselves.  Finding
        # all of this is a worthwhile TODO, but recursive_merge probably
        # means that finding all of them is going to be a challenge.
        # So, make sure that the database is wiped.  Deleting just
        # provenances and codeversions should do it, because most things
        # have a cascading foreign key into provenances.
        session.execute( sa.text( "DELETE FROM provenances" ) )
        session.execute( sa.text( "DELETE FROM code_versions" ) )
        session.commit()


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


@pytest.fixture
def config_test():
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

    try:
        with SmartSession() as session:
            session.execute(sa.delete(CodeVersion).where(CodeVersion.id == 'test_v1.0.0'))
            session.commit()
    except Exception as e:
        warnings.warn(str(e))


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

    try:
        with SmartSession() as session:
            session.execute(sa.delete(Provenance).where(Provenance.id == pid))
            session.commit()
    except Exception as e:
        warnings.warn(str(e))


@pytest.fixture
def provenance_extra(code_version, provenance_base):
    p = Provenance(
        process="test_base_process",
        code_version=code_version,
        parameters={"test_key": uuid.uuid4().hex},
        upstreams=[provenance_base],
        is_testing=True,
    )
    p.update_id()

    with SmartSession() as session:
        session.add(p)
        session.commit()
        session.refresh(p)
        pid = p.id

    yield p

    try:
        with SmartSession() as session:
            session.execute(sa.delete(Provenance).where(Provenance.id == pid))
            session.commit()
    except Exception as e:
        warnings.warn(str(e))


@pytest.fixture
def exposure_factory():
    def factory():
        e = Exposure(
            filepath=f"Demo_test_{rnd_str(5)}.fits",
            section_id=0,
            exp_time=np.random.randint(1, 4) * 10,  # 10 to 40 seconds
            mjd=np.random.uniform(58000, 58500),
            filter=np.random.choice(list('grizY')),
            ra=np.random.uniform(0, 360),
            dec=np.random.uniform(-90, 90),
            project='foo',
            target=rnd_str(6),
            nofile=True,
            md5sum=uuid.uuid4(),  # this should be done when we clean up the exposure factory a little more
        )
        return e

    return factory


def make_exposure_file(exposure):
    fullname = None
    fullname = exposure.get_fullpath()
    open(fullname, 'a').close()
    exposure.nofile = False

    yield exposure

    try:
        with SmartSession() as session:
            exposure = exposure.recursive_merge( session )
            if exposure.id is not None:
                session.execute(sa.delete(Exposure).where(Exposure.id == exposure.id))
                session.commit()

        if fullname is not None and os.path.isfile(fullname):
            os.remove(fullname)
    except Exception as e:
        warnings.warn(str(e))


@pytest.fixture
def exposure(exposure_factory):
    e = exposure_factory()
    make_exposure_file(e)
    yield e


# idea taken from: https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
def generate_exposure():
    @pytest.fixture
    def new_exposure(exposure_factory):
        e = exposure_factory()
        make_exposure_file(e)
        yield e

    return new_exposure


def inject_exposure_fixture(name):
    globals()[name] = generate_exposure()


for i in range(2, 10):
    inject_exposure_fixture(f'exposure{i}')


@pytest.fixture
def exposure_filter_array(exposure_factory):
    e = exposure_factory()
    e.filter = None
    e.filter_array = ['r', 'g', 'r', 'i']
    make_exposure_file(e)
    yield e


def get_decam_example_file():
    filename = os.path.join(CODE_ROOT, 'data/test_data/DECam_examples/c4d_221104_074232_ori.fits.fz')
    if not os.path.isfile(filename):
        cachedfilename = f'{filename}_cached'
        if not os.path.isfile( cachedfilename ):
            url = 'https://astroarchive.noirlab.edu/api/retrieve/004d537b1347daa12f8361f5d69bc09b/'
            response = wget.download( url=url, out=cachedfilename )
            assert response == cachedfilename
        os.symlink( cachedfilename, filename )
    return filename

@pytest.fixture
def decam_example_file():
    yield get_decam_example_file()

def get_decam_example_exposure():
    filename = get_decam_example_file()
    decam_example_file_short = filename[len(CODE_ROOT+'/data/'):]
    with SmartSession() as session:
        # always destroy this Exposure object and make a new one, to avoid filepath unique constraint violations
        session.execute(sa.delete(Exposure).where(Exposure.filepath == decam_example_file_short))
        session.commit()

    with fits.open( filename, memmap=False ) as ifp:
        hdr = ifp[0].header
    exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter', 'project', 'target' ] )

    exposure = Exposure( filepath=filename, instrument='DECam', **exphdrinfo )
    return exposure


@pytest.fixture
def decam_example_exposure(decam_example_file):
    return get_decam_example_exposure()

@pytest.fixture
def decam_example_raw_image( decam_example_exposure ):
    image = Image.from_exposure(decam_example_exposure, section_id='N1')
    image.data = image.raw_data.astype(np.float32)
    return image


@pytest.fixture(scope="session")
def decam_example_reduced_image_ds():
    """Returns a datastore with an image that's been loaded into the database."""
    exposure = get_decam_example_exposure()
    # Have to spoof the md5sum field to let us add it to the database even
    # though we're not really saving it to the archive
    exposure.md5sum = uuid.uuid4()
    with SmartSession() as session:
        # Spoof md5sum so we can add it
        # Think: may be better just to actually save the exposure
        # to the test archive?
        exposure = exposure.recursive_merge( session )
        session.add( exposure )
        session.commit()
        prepper = Preprocessor()
        # NOTE: this has a side effect of loading some DECam
        # calibration files (flats, etc.) into the databse
        ds = prepper.run( exposure, 'N1', session=session )
        ds.save_and_commit( session=session )
    yield ds
    ds.delete_everything()

@pytest.fixture(scope="session")
def decam_example_reduced_image_source_list_ds( decam_example_reduced_image_ds ):
    """Returns the same datastore from decam_example_reduced_image_ds, only now with a source list too"""
    det = Detector()
    ds = det.run( decam_example_reduced_image_ds )
    ds.save_and_commit()
    yield ds
    # This next line may not be necessary since
    # decam_example_reduced_image_ds will run it
    # ...indeed, it looks like data_store.delete_everyting()
    # is not robust to being called more than once
    # ds.delete_everything()

@pytest.fixture
def decam_small_image(decam_example_raw_image):
    image = decam_example_raw_image
    image.data = image.data[256:256+512, 256:256+512].copy()  # make it C-contiguous
    return image


class ImageCleanup:
    """
    Helper function that allows you to take an Image object
    with fake data (for testing) and save it to disk,
    while also making sure that the data is removed from disk
    when the object goes out of scope.

    Usage:
    >> im_clean = ImageCleanup.save_image(image)
    at end of test the im_clean goes out of scope and removes the file
    """

    @classmethod
    def save_image(cls, image, archive=True):
        """
        Save the image to disk, and return an ImageCleanup object.

        Parameters
        ----------
        image: models.image.Image
            The image to save (that is used to call remove_data_from_disk)
        archive:
            Whether to save to the archive or not. Default is True.
            Controls the save(no_archive) flag and whether the file
            will be cleaned up from database and archive at the end.

        Returns
        -------
        ImageCleanup:
            An object that will remove the image from disk when it goes out of scope.
            This should be put into a variable that goes out of scope at the end of the test.
        """
        if image.data is None:
            if image.raw_data is None:
                image.raw_data = np.random.uniform(0, 100, size=(100, 100))
            image.data = np.float32(image.raw_data)

        if image.instrument is None:
            image.instrument = 'DemoInstrument'

        if image._raw_header is None:
            image._raw_header = fits.Header()

        image.save(no_archive=not archive)

        # if not archive:
        #     image.md5sum = uuid.uuid4()  # spoof the md5 sum
        return cls(image, archive=archive)  # don't use this, but let it sit there until going out of scope of the test

    def __init__(self, image, archive=True):
        self.image = image
        self.archive = archive

    def __del__(self):
        # print('removing file at end of test!')
        try:
            if self.archive:
                self.image.delete_from_disk_and_database()
            else:
                self.image.remove_data_from_disk()
        except:
            pass


@pytest.fixture
def demo_image(exposure):
    exposure.update_instrument()
    im = Image.from_exposure(exposure, section_id=0)

    yield im

    try:
        with SmartSession() as session:
            im = session.merge(im)
            im.delete_from_disk_and_database(session=session, commit=True)

    except Exception as e:
        warnings.warn(str(e))


# idea taken from: https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
def generate_image():

    @pytest.fixture
    def new_image(exposure_factory):
        exp = exposure_factory()
        make_exposure_file(exp)
        exp.update_instrument()
        im = Image.from_exposure(exp, section_id=0)

        yield im

        with SmartSession() as session:
            im = session.merge(im)
            im.delete_from_disk_and_database(session=session, commit=True)
            if sa.inspect( im ).persistent:
                session.execute(sa.delete(Image).where(Image.id == im.id))
                session.commit()

    return new_image


def inject_demo_image_fixture(image_name):
    globals()[image_name] = generate_image()


for i in range(2, 10):
    inject_demo_image_fixture(f'demo_image{i}')


@pytest.fixture
def reference_entry(exposure_factory, provenance_base, provenance_extra):
    ref_entry = None
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
        ref_entry.image = session.merge( ref_entry.image )
        session.add(ref_entry)
        session.commit()

    yield ref_entry

    try:
        if ref_entry is not None:
            with SmartSession() as session:
                ref_entry = session.merge(ref_entry)
                ref = ref_entry.image
                for im in ref.source_images:
                    exp = im.exposure
                    exp.delete_from_disk_and_database(session=session, commit=False)
                    im.delete_from_disk_and_database(session=session, commit=False)
                ref.delete_from_disk_and_database(session=session, commit=False)

                session.commit()

    except Exception as e:
        warnings.warn(str(e))


@pytest.fixture
def sources(demo_image):
    num = 100
    x = np.random.uniform(0, demo_image.raw_data.shape[1], num)
    y = np.random.uniform(0, demo_image.raw_data.shape[0], num)
    flux = np.random.uniform(0, 1000, num)
    flux_err = np.random.uniform(0, 100, num)
    rhalf = np.abs(np.random.normal(0, 3, num))

    data = np.array(
        [x, y, flux, flux_err, rhalf],
        dtype=([('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('flux_err', 'f4'), ('rhalf', 'f4')])
    )
    s = SourceList(image=demo_image, data=data, format='sepnpy')

    yield s

    try:
        with SmartSession() as session:
            s = session.merge(s)
            s.delete_from_disk_and_database(session=session, commit=True)
    except Exception as e:
        warnings.warn(str(e))


@pytest.fixture
def archive():
    cfg = config.Config.get()
    archive_specs = cfg.value('archive')
    if archive_specs is None:
        raise ValueError( "archive in config is None" )
    archive = Archive( **archive_specs )
    yield archive

    try:
        # To tear down, we need to blow away the archive server's directory.
        # For the test suite, we've also mounted that directory locally, so
        # we can do that
        archivebase = f"{os.getenv('SEECHANGE_TEST_ARCHIVE_DIR')}/{cfg.value('archive.path_base')}"
        try:
            shutil.rmtree( archivebase )
        except FileNotFoundError:
            pass

    except Exception as e:
        warnings.warn(str(e))


# Get the flat, fringe, and linearity for
# a couple of DECam chips and filters
# Need session scope; otherwise, things
# get mixed up when _get_default_calibrator
# is called from within another function.
@pytest.fixture( scope='session' )
def decam_default_calibrators():
    decam = get_instrument_instance( 'DECam' )
    sections = [ 'N1', 'S1' ]
    filters = [ 'r', 'i', 'z' ]
    for sec in sections:
        for calibtype in [ 'flat', 'fringe' ]:
            for filt in filters:
                decam._get_default_calibrator( 60000, sec, calibtype=calibtype, filter=filt )
    decam._get_default_calibrator( 60000, sec, calibtype='linearity' )

    yield sections, filters

    imagestonuke = set()
    datafilestonuke = set()
    with SmartSession() as session:
        for sec in [ 'N1', 'S1' ]:
            for filt in [ 'r', 'i', 'z' ]:
                info = decam.preprocessing_calibrator_files( 'externally_supplied', 'externally_supplied',
                                                             sec, filt, 60000, nofetch=True, session=session )
                for filetype in [ 'zero', 'flat', 'dark', 'fringe', 'illumination', 'linearity' ]:
                    if ( f'{filetype}_fileid' in info ) and ( info[ f'{filetype}_fileid' ] is not None ):
                        if info[ f'{filetype}_isimage' ]:
                            imagestonuke.add( info[ f'{filetype}_fileid' ] )
                        else:
                            datafilestonuke.add( info[ f'{filetype}_fileid' ] )
        for imid in imagestonuke:
            im = session.get( Image, imid )
            im.delete_from_disk_and_database( session=session, commit=False )
        for dfid in datafilestonuke:
            df = session.get( DataFile, dfid )
            df.delete_from_disk_and_database( session=session, commit=False )
        session.commit()

@pytest.fixture
def example_image_with_sources_and_psf_filenames():
    image = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.fits"
    weight = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.weight.fits"
    flags = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.flags.fits"
    sources = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.sources.fits"
    psf = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.psf"
    psfxml = pathlib.Path( FileOnDiskMixin.local_path ) / "test_data/test_ztf_image.psf.xml"
    return image, weight, flags, sources, psf, psfxml

@pytest.fixture
def example_source_list_filename( example_image_with_sources_and_psf_filenames ):
    image, weight, flags, sources, psf, psfxml = example_image_with_sources_and_psf_filenames
    return sources

@pytest.fixture
def example_psfex_psf_files():
    psfpath = ( pathlib.Path( FileOnDiskMixin.local_path )
                / "test_data/ztf_20190317307639_000712_zg_io.083_sources.psf" )
    psfxmlpath = ( pathlib.Path( FileOnDiskMixin.local_path )
                   / "test_data/ztf_20190317307639_000712_zg_io.083_sources.psf.xml" )
    if not ( psfpath.is_file() and psfxmlpath.is_file() ):
        raise FileNotFoundErrro( f"Can't read at least one of {psfpath}, {psfxmlpath}" )
    return psfpath, psfxmlpath
