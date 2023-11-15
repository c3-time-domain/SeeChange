import os
import re
import io
import warnings
import pytest
import uuid
import wget
import shutil
import pathlib
import hashlib
import yaml
import subprocess

import numpy as np

import sqlalchemy as sa

from astropy.time import Time
from astropy.io import fits, votable
from astropy.wcs import WCS

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
from models.psf import PSF
from pipeline.data_store import DataStore
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from util import config
from util.archive import Archive
from util.retrydownload import retry_download
from util.exceptions import SubprocessFailure


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


@pytest.fixture
def headless_plots():
    import matplotlib

    backend = matplotlib.get_backend()
    # ref: https://stackoverflow.com/questions/15713279/calling-pylab-savefig-without-display-in-ipython
    matplotlib.use("Agg")

    yield None

    matplotlib.use(backend)


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
        p.code_version=session.merge(code_version)
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
def provenance_extra( provenance_base ):
    p = Provenance(
        process="test_base_process",
        code_version=provenance_base.code_version,
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


def make_new_exposure():
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


def add_file_to_exposure(exposure):
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
def exposure():
    e = make_new_exposure()
    add_file_to_exposure(e)
    yield e


# idea taken from: https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
def generate_exposure_fixture():
    @pytest.fixture
    def new_exposure():
        e = make_new_exposure()
        add_file_to_exposure(e)
        yield e

    return new_exposure


def inject_exposure_fixture(name):
    globals()[name] = generate_exposure_fixture()


for i in range(2, 10):
    inject_exposure_fixture(f'exposure{i}')


@pytest.fixture
def exposure_filter_array():
    e = make_new_exposure()
    e.filter = None
    e.filter_array = ['r', 'g', 'r', 'i']
    add_file_to_exposure(e)
    yield e


def get_decam_example_file():
    """Pull a DECam exposure down from the NOIRLab archives.

    Because this is a slow process (depending on the NOIRLab archive
    speed, it can take up to minutes), first look for
    "{filename}_cached", and if it exists, just symlink to it.  If not,
    actually download the image from NOIRLab, rename it to
    "{filename}_cached", and create the symlink.  That way, until the
    user manually deletes the _cached file, we won't have to redo the
    slow NOIRLab download again.

    """
    filename = os.path.join(CODE_ROOT, 'data/test_data/DECam_examples/c4d_221104_074232_ori.fits.fz')
    if not os.path.isfile(filename):
        cachedfilename = f'{filename}_cached'
        if not os.path.isfile( cachedfilename ):
            url = 'https://astroarchive.noirlab.edu/api/retrieve/004d537b1347daa12f8361f5d69bc09b/'
            response = wget.download( url=url, out=cachedfilename )
            assert response == cachedfilename
        os.symlink( cachedfilename, filename )
    return filename


@pytest.fixture(scope="session")
def decam_example_file():
    yield get_decam_example_file()


@pytest.fixture
def decam_example_exposure(decam_example_file):
    filename = get_decam_example_file()
    decam_example_file_short = filename[len(CODE_ROOT+'/data/'):]

    with fits.open( filename, memmap=True ) as ifp:
        hdr = ifp[0].header
    exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter', 'project', 'target' ] )

    exposure = Exposure( filepath=filename, instrument='DECam', **exphdrinfo )

    yield exposure

    # Just in case this exposure got loaded into the database
    with SmartSession() as session:
        session.execute(sa.delete(Exposure).where(Exposure.filepath == decam_example_file_short))
        session.commit()


@pytest.fixture
def decam_example_raw_image( decam_example_exposure ):
    image = Image.from_exposure(decam_example_exposure, section_id='N1')
    image.data = image.raw_data.astype(np.float32)
    return image


@pytest.fixture
def decam_example_reduced_image_ds( code_version, decam_example_exposure ):
    """Provides a datastore with an image, source list, and psf.

    Preprocessing, source extraction, and PSF estimation were all
    performed as this fixture was written, and saved to
    data/test_data/DECam_examples, so that this fixture will return
    quickly.  That does mean that if the code has evolved since those
    test files were created, the files may not be exactly consistent
    with the code_version in the provenances they have attached to them,
    but hopefully the actual data and database records are still good
    enough for the tests.

    Has an image (with weight and flags), sources, and psf.  The data
    has not been loaded into the image, sources, or psf fields, but of
    course that will happen if you access (for instance)
    decam_example_reduced_image_ds.image.data.  The provenances *have*
    been loaded into the session (and committed to the database), but
    the image, sources, and psf have not been added to the session.

    The DataStore has a session inside it.

    """

    exposure = decam_example_exposure
    # Have to spoof the md5sum field to let us add it to the database even
    # though we're not really saving it to the archive
    exposure.md5sum = uuid.uuid4()

    datadir = pathlib.Path( FileOnDiskMixin.local_path )
    filepathbase = 'test_data/DECam_examples/c4d_20221104_074232_N1_g_Sci_VWQNR2'
    fileextensions = { '.image.fits', '.weight.fits', '.flags.fits', '.sources.fits', '.psf', '.psf.xml' }

    # This next block of code generates the files read in this test.  We
    # don't want to rerun the code to generate them every single time,
    # because it's a fairly slow process and this is a function-scope
    # fixture.  As such, generate files with names ending in "_cached"
    # that we won't delete at fixture teardown.  Then, we'll copy those
    # files to the ones that the fixture really needs, and those copies
    # will be deleted at fixture teardown.  (We need this to be a
    # function-scope fixture because the tests that use the DataStore we
    # return may well modify it, and may well modify the files on disk.)

    # (Note: this will not regenerate the .yaml files
    # test_data/DECam_examples, which are used to reconstruct the
    # database object fields.  If tests fail after regenerating these
    # files, review those .yaml files to make sure they're still
    # current.)

    # First check if those files exist; if they don't, generate them.
    mustregenerate = False
    for ext in fileextensions:
        if not ( datadir / f'{filepathbase}{ext}_cached' ).is_file():
            _logger.info( f"{filepathbase}{ext}_cached missing, decam_example_reduced_image_ds fixture "
                          f"will regenerate all _cached files" )
            mustregenerate = True

    if mustregenerate:
        with SmartSession() as session:
            exposure = exposure.recursive_merge( session )
            session.add( exposure )
            session.commit()              # This will get cleaned up in the decam_example_exposure teardown
            prepper = Preprocessor()
            ds = prepper.run( exposure, 'N1', session=session )
            try:
                det = Detector( measure_psf=True )
                ds = det.run( ds )
                ds.save_and_commit()

                paths = []
                for obj in [ ds.image, ds.sources, ds.psf ]:
                    paths.extend( obj.get_fullpath( as_list=True ) )

                extextract = re.compile( '^(?P<base>.*)(?P<extension>\\..*\\.fits|\\.psf|\\.psf.xml)$' )
                extscopied = set()
                for src in paths:
                    match = extextract.search( src )
                    if match is None:
                        raise RuntimeError( f"Failed to parse {src}" )
                    if match.group('extension') not in fileextensions:
                        raise RuntimeError( f"Unexpected file extension on {src}" )
                    shutil.copy2( src, datadir/ f'{filepathbase}{match.group("extension")}_cached' )
                    extscopied.add( match.group('extension') )

                if extscopied != fileextensions:
                    raise RuntimeError( f"Extensions copied {extcopied} doesn't match expected {extstocopy}" )
            finally:
                ds.delete_everything()
    else:
        _logger.info( f"decam_example_reduced_image_ds fixture found all _cached files, not regenerating" )

    # Now make sure that the actual files needed are there by copying
    # the _cached files

    copiesmade = []
    try:
        for ext in [ '.image.fits', '.weight.fits', '.flags.fits', '.sources.fits', '.psf', '.psf.xml' ]:
            actual = datadir / f'{filepathbase}{ext}'
            if actual.exists():
                raise FileExistsError( f"{actual} exists, but at this point in the tests it's not supposed to" )
            shutil.copy2( datadir / f"{filepathbase}{ext}_cached", actual )
            copiesmade.append( actual )

        with SmartSession() as session:
            # The filenames will not match the provenance, because the filenames
            # are what they are, but as the code evoles the provenance tag is
            # going to change.  What's more, the provenances are different for
            # the images and for sources/psf.
            imgprov = Provenance( process="preprocessing", code_version=code_version, upstreams=[], is_testing=True )
            srcprov = Provenance( process="extraction", code_version=code_version,
                                  upstreams=[imgprov], is_testing=True )
            session.add( imgprov )
            session.add( srcprov )
            session.commit()
            session.refresh( imgprov )
            session.refresh( srcprov )

            with open( datadir / f'{filepathbase}.image.yaml' ) as ifp:
                imageyaml = yaml.safe_load( ifp )
            with open( datadir / f'{filepathbase}.sources.yaml' ) as ifp:
                sourcesyaml = yaml.safe_load( ifp )
            with open( datadir / f'{filepathbase}.psf.yaml' ) as ifp:
                psfyaml = yaml.safe_load( ifp )
            ds = DataStore( session=session )
            ds.image = Image( **imageyaml )
            ds.image.provenance = imgprov
            ds.image.filepath = filepathbase
            ds.sources = SourceList( **sourcesyaml )
            ds.sources.image = ds.image
            ds.sources.provenance = srcprov
            ds.sources.filepath = f'{filepathbase}.sources.fits'
            ds.psf = PSF( **psfyaml )
            ds.psf.image = ds.image
            ds.psf.provenance = srcprov
            ds.psf.filepath = filepathbase

            yield ds

            ds.delete_everything()
            session.delete( imgprov )
            session.delete( srcprov )
            session.commit()
            session.close()

    finally:
        for f in copiesmade:
            f.unlink( missing_ok=True )

# TODO : cache the results of this just like in
# decam_example_reduced_image_ds so they don't have to be regenerated
# every time this fixture is used.
@pytest.fixture
def decam_example_reduced_image_ds_with_wcs( decam_example_reduced_image_ds ):
    ds = decam_example_reduced_image_ds
    with open( ds.image.get_fullpath()[0], "rb" ) as ifp:
        md5 = hashlib.md5()
        md5.update( ifp.read() )
        origmd5 = uuid.UUID( md5.hexdigest() )

    xvals = [ 0, 0, 2047, 2047 ]
    yvals = [ 0, 4095, 0, 4095 ]
    origwcs = WCS( ds.image.raw_header )

    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[22.], mag_range=4.,
                                  min_stars=50, max_resid=0.15, crossid_radius=[2.0],
                                  min_frac_matched=0.1, min_matched_stars=10 )
    ds = astrometor.run( ds )

    return ds, origwcs, xvals, yvals, origmd5

    # Don't need to do any cleaning up, because no files were written
    # doing the WCS (it's all database), and the
    # decam_example_reduced_image_ds is going to do a
    # ds.delete_everything()

@pytest.fixture
def decam_example_reduced_image_ds_with_zp( decam_example_reduced_image_ds_with_wcs ):
    ds = decam_example_reduced_image_ds_with_wcs[0]
    ds.save_and_commit()
    photomotor = PhotCalibrator( cross_match_catalog='GaiaDR3' )
    ds = photomotor.run( ds )

    return ds, photomotor

@pytest.fixture
def ref_for_decam_example_image( provenance_base ):
    datadir = pathlib.Path( FileOnDiskMixin.local_path ) / 'test_data/DECam_examples'
    filebase = 'DECaPS-West_20220112.g.32'

    urlmap = { '.image.fits': '.fits.fz',
               '.weight.fits': '.weight.fits.fz',
               '.flags.fits': '.bpm.fits.fz' }
    for ext in [ '.image.fits', '.weight.fits', '.flags.fits' ]:
        path = datadir / f'{filebase}{ext}'
        cachedpath = datadir / f'{filebase}{ext}_cached'
        fzpath = datadir / f'{filebase}{ext}_cached.fz'
        if cachedpath.is_file():
            _logger.info( f"{path} exists, not redownloading." )
        else:
            url = ( f'https://portal.nersc.gov/cfs/m2218/decat/decat/templatecache/DECaPS-West_20220112.g/'
                    f'{filebase}{urlmap[ext]}' )
            retry_download( url, fzpath )
            res = subprocess.run( [ 'funpack', '-D', fzpath ] )
            if res.returncode != 0:
                raise SubprocessFailure( res )
        shutil.copy2( cachedpath, path )

    prov = provenance_base

    with open( datadir / f'{filebase}.image.yaml' ) as ifp:
        refyaml = yaml.safe_load( ifp )
    image = Image( **refyaml )
    image.provenance = prov
    image.filepath = f'test_data/DECam_examples/{filebase}'

    yield image

    # Just in case the image got added to the database:
    image.delete_from_disk_and_database()

    # And just in case the image was added to the database with a different name:
    for ext in [ '.image.fits', '.weight.fits', '.flags.fits' ]:
        ( datadir / f'{filebase}{ext}' ).unlink( missing_ok=True )

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
def generate_image_fixture():

    @pytest.fixture
    def new_image():
        exp = make_new_exposure()
        add_file_to_exposure(exp)
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
    globals()[image_name] = generate_image_fixture()


for i in range(2, 10):
    inject_demo_image_fixture(f'demo_image{i}')


@pytest.fixture
def reference_entry(provenance_base, provenance_extra):
    ref_entry = None
    filter = np.random.choice(list('grizY'))
    target = rnd_str(6)
    ra = np.random.uniform(0, 360)
    dec = np.random.uniform(-90, 90)
    images = []

    for i in range(5):
        exp = make_new_exposure()

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
def example_ds_with_sources_and_psf( example_image_with_sources_and_psf_filenames ):
    image, weight, flags, sources, psf, psfxml = example_image_with_sources_and_psf_filenames
    ds = DataStore()

    ds.image = Image( filepath=str( image.relative_to( FileOnDiskMixin.local_path ) ), format='fits' )
    with fits.open( image ) as hdul:
        ds.image._data = hdul[0].data
        ds.image._raw_header = hdul[0].header
    with fits.open( weight ) as hdul:
        ds.image._weight = hdul[0].data
    with fits.open( flags ) as hdul:
        ds.image_flags = hdul[0].data
    ds.image.set_corners_from_header_wcs()
    ds.image.ra = ( ds.image.ra_corner_00 + ds.image.ra_corner_01 +
                    ds.image.ra_corner_10 + ds.image.ra_corner_11 ) / 4.
    ds.image.dec = ( ds.image.dec_corner_00 + ds.image.dec_corner_01 +
                     ds.image.dec_corner_00 + ds.image.dec_corner_11 ) / 4.
    ds.image.calculate_coordinates()

    ds.sources = SourceList( filepath=str( sources.relative_to( FileOnDiskMixin.local_path ) ), format='sextrfits' )
    ds.sources.load( sources )
    ds.sources.num_sources = len( ds.sources.data )

    ds.psf = PSF( filepath=str( psf.relative_to( FileOnDiskMixin.local_path ) ), format='psfex' )
    ds.psf.load( download=False, psfpath=psf, psfxmlpath=psfxml )
    bio = io.BytesIO( ds.psf.info.encode( 'utf-8' ) )
    tab = votable.parse( bio ).get_table_by_index( 1 )
    ds.psf.fwhm_pixels = float( tab.array['FWHM_FromFluxRadius_Mean'][0] )

    return ds


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
        raise FileNotFoundError( f"Can't read at least one of {psfpath}, {psfxmlpath}" )
    return psfpath, psfxmlpath
