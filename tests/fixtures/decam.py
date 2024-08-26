import pytest
import os
import wget
import yaml
import shutil
import pathlib

import sqlalchemy as sa
import numpy as np

from astropy.io import fits
from astropy.time import Time

from models.base import SmartSession
from models.instrument import Instrument, get_instrument_instance
from models.decam import DECam  # need this import to make sure DECam is added to the Instrument list
from models.provenance import Provenance, ProvenanceTag
from models.exposure import Exposure
from models.image import Image
from models.datafile import DataFile
from models.reference import Reference
from models.refset import RefSet

from improc.alignment import ImageAligner

from util.retrydownload import retry_download
from util.logger import SCLogger
from util.cache import copy_to_cache, copy_from_cache
from util.util import env_as_bool


@pytest.fixture(scope='session')
def decam_cache_dir(cache_dir):
    output = os.path.join(cache_dir, 'DECam')
    if not os.path.isdir(output):
        os.makedirs(output)

    yield output


# Get the flat, fringe, and linearity for
# a couple of DECam chips and filters
# Need session scope; otherwise, things
# get mixed up when _get_default_calibrator
# is called from within another function.
@pytest.fixture( scope='session' )
def decam_default_calibrators(cache_dir, data_dir):
    try:
        # try to get the calibrators from the cache folder
        if not env_as_bool( "LIMIT_CACHE_USAGE" ):
            if os.path.isdir(os.path.join(cache_dir, 'DECam_default_calibrators')):
                shutil.copytree(
                    os.path.join(cache_dir, 'DECam_default_calibrators'),
                    os.path.join(data_dir, 'DECam_default_calibrators'),
                    dirs_exist_ok=True,
                )

        decam = get_instrument_instance( 'DECam' )
        sections = [ 'S3', 'N16' ]
        filters = [ 'r', 'i', 'z', 'g']
        for sec in sections:
            for calibtype in [ 'flat', 'fringe' ]:
                for filt in filters:
                    decam._get_default_calibrator( 60000, sec, calibtype=calibtype, filter=filt )
        decam._get_default_calibrator( 60000, sec, calibtype='linearity' )

        # store the calibration files in the cache folder
        if not env_as_bool( "LIMIT_CACHE_USAGE" ):
            if not os.path.isdir(os.path.join(cache_dir, 'DECam_default_calibrators')):
                os.makedirs(os.path.join(cache_dir, 'DECam_default_calibrators'), exist_ok=True)
            for folder in os.listdir(os.path.join(data_dir, 'DECam_default_calibrators')):
                if not os.path.isdir(os.path.join(cache_dir, 'DECam_default_calibrators', folder)):
                    os.makedirs(os.path.join(cache_dir, 'DECam_default_calibrators', folder), exist_ok=True)
                for file in os.listdir(os.path.join(data_dir, 'DECam_default_calibrators', folder)):
                    shutil.copy2(
                        os.path.join(data_dir, 'DECam_default_calibrators', folder, file),
                        os.path.join(cache_dir, 'DECam_default_calibrators', folder, file)
                    )

        yield sections, filters

    finally:
        # SCLogger.debug('tear down of decam_default_calibrators')
        imagestonuke = set()
        datafilestonuke = set()
        with SmartSession() as session:
            for sec in [ 'S3', 'N16' ]:
                for filt in [ 'r', 'i', 'z', 'g' ]:
                    info = decam.preprocessing_calibrator_files( 'externally_supplied', 'externally_supplied',
                                                                 sec, filt, 60000, nofetch=True, session=session )
                    for filetype in [ 'zero', 'flat', 'dark', 'fringe', 'illumination', 'linearity' ]:
                        if ( f'{filetype}_fileid' in info ) and ( info[ f'{filetype}_fileid' ] is not None ):
                            if info[ f'{filetype}_isimage' ]:
                                imagestonuke.add( info[ f'{filetype}_fileid' ] )
                            else:
                                datafilestonuke.add( info[ f'{filetype}_fileid' ] )

        for imid in imagestonuke:
            im = Image.get_by_id( imid )
            im.delete_from_disk_and_database()

        for dfid in datafilestonuke:
            df = DataFile.get_by_id( dfid )
            df.delete_from_disk_and_database()

        with SmartSession() as session:
            provs = session.scalars(
                sa.select(Provenance).where(Provenance.process == 'DECam Default Calibrator')
            ).all()
            for prov in provs:
                datafiles = session.scalars(sa.select(DataFile).where(DataFile.provenance_id == prov.id)).all()
                images = session.scalars(sa.select(Image).where(Image.provenance_id == prov.id)).all()
                if len(datafiles) == 0 and len(images) == 0:
                    session.delete(prov)
            session.commit()


@pytest.fixture(scope='session')
def provenance_decam_prep(code_version):
    p = Provenance(
        process="preprocessing",
        code_version_id=code_version.id,
        parameters={
            'steps': None,
            'calibset': None,
            'flattype': None,
            'test_parameter': 'test_value',
            'preprocessing_steps': ['overscan', 'linearity', 'flat', 'fringe'],
            'use_sky_subtraction': False,
        },
        upstreams=[],
        is_testing=True,
    )
    p.insert()

    yield p

    with SmartSession() as session:
        session.execute( sa.delete( Provenance ).where( Provenance.id==p.id ) )
        session.commit()


@pytest.fixture(scope='module')
def decam_reduced_origin_exposures():
    decam = DECam()
    yield decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                       projects='2023A-716082',
                                       skip_exposures_in_database=False,
                                       proc_type='instcal' )


@pytest.fixture(scope='session')
def decam_raw_origin_exposures_parameters():
    return { 'minmjd': 60127.33819,
             'maxmjd': 60127.36319,
             'projects': [ '2023A-716082' ] ,
             'proc_type': 'raw' }

@pytest.fixture(scope='module')
def decam_raw_origin_exposures( decam_raw_origin_exposures_parameters ):
    decam = DECam()
    yield decam.find_origin_exposures( **decam_raw_origin_exposures_parameters )


@pytest.fixture(scope="session")
def decam_exposure_name():
    return 'c4d_230702_080904_ori.fits.fz'

@pytest.fixture(scope="session")
def decam_filename(download_url, data_dir, decam_exposure_name, decam_cache_dir):
    """Secure a DECam exposure.

    Pulled from the SeeChange test data cache maintained on the web at
    NERSC (see download_url in conftest.py).

    Because this is a slow process (depending on the NOIRLab archive
    speed, it can take up to minutes), first look for this file
    in the cache_dir, and if it exists, and copy it. If not,
    actually download the image from NOIRLab into the cache_dir,
    and create a symlink to the temp_dir. That way, until the
    user manually deletes the cached file, we won't have to redo the
    slow NOIRLab download again.

    This exposure is the same as the one pulled down by the
    test_decam_download_and_commit_exposure test (with expdex 1) in
    tests/models/test_decam.py, so whichever runs first will load the
    cache.

    """
    base_name = decam_exposure_name
    filename = os.path.join(data_dir, base_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    url = os.path.join(download_url, 'DECAM', base_name)

    if not os.path.isfile(filename):
        if env_as_bool( "LIMIT_CACHE_USAGE" ):
            SCLogger.debug( f"Downloading {filename}" )
            wget.download( url=url, out=filename )
        else:
            cachedfilename = os.path.join(decam_cache_dir, base_name)
            os.makedirs(os.path.dirname(cachedfilename), exist_ok=True)

            if not os.path.isfile(cachedfilename):
                SCLogger.debug( f"Downloading {filename}" )
                response = wget.download(url=url, out=cachedfilename)
                assert response == cachedfilename
            else:
                SCLogger.debug( f"Cached file {filename} exists, not redownloading." )

            shutil.copy2(cachedfilename, filename)

    yield filename

    if os.path.isfile(filename):
        os.remove(filename)

# Making this a session fixture means running a single
#   fast test starts kinda slow.  (It seems that
#   the fixture is run even if the one test doesn't
#   ask for it.)
# Make it not a session fixture will make tests that
#   reuse it kinda slow.
# There is no good answer.
@pytest.fixture(scope="session")
def decam_exposure(decam_filename, data_dir):
    filename = decam_filename

    with fits.open( filename, memmap=True ) as ifp:
        hdr = ifp[0].header
    exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter', 'project', 'target' ] )

    exposure = Exposure( filepath=filename, instrument='DECam', **exphdrinfo )
    exposure.save()  # save to archive and get an MD5 sum
    with SmartSession() as session:
        exposure.insert()

    yield exposure

    exposure.delete_from_disk_and_database()

@pytest.fixture
def decam_raw_image_provenance( provenance_base ):
    return provenance_base

@pytest.fixture
def decam_raw_image( decam_exposure, provenance_base ):
    image = Image.from_exposure(decam_exposure, section_id='S3')
    image.data = image.raw_data.astype(np.float32)
    # These next two don't mean anything, but put them there for things
    #   that require those files to be there for reading purposes
    image.weight = np.full_like( image.data, image.data.std() )
    image.flags = np.zeros_like( image.data )
    image.provenance_id = provenance_base.id
    image.save()

    yield image

    image.delete_from_disk_and_database()


@pytest.fixture
def decam_small_image(decam_raw_image):
    image = decam_raw_image
    image.data = image.data[256:256+512, 256:256+512].copy()  # make it C-contiguous

    yield image


@pytest.fixture(scope='session')
def decam_refset():
    refset = RefSet( name='test_refset_decam' )
    refset.insert()

    yield refset

    with SmartSession() as session:
        session.execute( sa.delete( RefSet ).where( RefSet.name=='test_refset_decam' ) )
        session.commit()

# Don't use the decam_datastore and decam_datastore_through_* fixtures in the same test.
@pytest.fixture
def decam_datastore(
        datastore_factory,
        decam_cache_dir,
        decam_exposure,
        decam_default_calibrators,  # not used directly, but makes sure this is pre-fetched from cache
        decam_reference,
):
    """Provide a datastore with all the products based on the DECam exposure

    To use this data store in a test where new data is to be generated,
    simply change the pipeline object's "test_parameter" value to a unique
    new value, so the provenance will not match and the data will be regenerated.

    EXAMPLE
    -------
    extractor.pars.test_parameter = uuid.uuid().hex
    extractor.run(datastore)
    assert extractor.has_recalculated is True
    """
    ds = datastore_factory(
        decam_exposure,
        'S3',
        cache_dir=decam_cache_dir,
        cache_base_name='007/c4d_20230702_080904_S3_r_Sci_NBXRIO',
        overrides={ 'subtraction': { 'refset': 'test_refset_decam' } },
        save_original_image=True,
        provtag='decam_datastore'
    )
    # This save is redundant, as the datastore_factory calls save_and_commit
    # However, I leave this here because it is a good test that calling it twice
    # does not cause any problems.
    ds.save_and_commit()

    deletion_list = [
        ds.image, ds.sources, ds.psf, ds.wcs, ds.zp, ds.sub_image, ds.detections, ds.cutouts, ds.measurements
    ]

    yield ds

    # cleanup
    if 'ds' in locals():
        ds.delete_everything()

    # # make sure that these individual objects have their files cleaned up,
    # # even if the datastore is cleared and all database rows are deleted.
    # for obj in deletion_list:
    #     if isinstance(obj, list) and len(obj) > 0 and hasattr(obj[0], 'delete_list'):
    #         obj[0].delete_list(obj)
    #     if obj is not None and hasattr(obj, 'delete_from_disk_and_database'):
    #         obj.delete_from_disk_and_database(archive=True)

    # Because save_original_image was True in the call to datastore_factory above
    os.unlink( ds.path_to_original_image )

    ImageAligner.cleanup_temp_images()

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'decam_datastore' } )
        session.commit()

@pytest.fixture
def decam_datastore_through_preprocessing(
        datastore_factory,
        decam_cache_dir,
        decam_exposure,
        decam_reference,                  # Needed so that the right provenances get loaded into the prov_tree
        decam_default_calibrators
):
    ds = datastore_factory(
        decam_exposure,
        'S3',
        cache_dir=decam_cache_dir,
        cache_base_name='007/c4d_20230702_080904_S3_r_Sci_NBXRIO',
        overrides={ 'subtraction': { 'refset': 'test_refset_decam' } },
        save_original_image=True,
        provtag='decam_datastore',
        through_step='preprocessing'
    )
    ds.save_and_commit()

    yield ds

    if 'ds' in locals():
        ds.delete_everything()

    # Because save_original_image as True in the call to datastore_factory
    os.unlink( ds.path_to_original_image )

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'decam_datastore' } )
        session.commit()


@pytest.fixture
def decam_datastore_through_extraction(
        datastore_factory,
        decam_cache_dir,
        decam_exposure,
        decam_reference,                  # Needed so that the right provenances get loaded into the prov_tree
        decam_default_calibrators
):
    ds = datastore_factory(
        decam_exposure,
        'S3',
        cache_dir=decam_cache_dir,
        cache_base_name='007/c4d_20230702_080904_S3_r_Sci_NBXRIO',
        overrides={ 'subtraction': { 'refset': 'test_refset_decam' } },
        save_original_image=True,
        provtag='decam_datastore',
        through_step='extraction'
    )
    ds.save_and_commit()

    yield ds

    if 'ds' in locals():
        ds.delete_everything()

    # Because save_original_image as True in the call to datastore_factory
    os.unlink( ds.path_to_original_image )

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'decam_datastore' } )
        session.commit()


@pytest.fixture
def decam_datastore_through_bg(
        datastore_factory,
        decam_cache_dir,
        decam_exposure,
        decam_reference,                  # Needed so that the right provenances get loaded into the prov_tree
        decam_default_calibrators
):
    ds = datastore_factory(
        decam_exposure,
        'S3',
        cache_dir=decam_cache_dir,
        cache_base_name='007/c4d_20230702_080904_S3_r_Sci_NBXRIO',
        overrides={ 'subtraction': { 'refset': 'test_refset_decam' } },
        save_original_image=True,
        provtag='decam_datastore',
        through_step='bg'
    )
    ds.save_and_commit()

    yield ds

    if 'ds' in locals():
        ds.delete_everything()

    # Because save_original_image as True in the call to datastore_factory
    os.unlink( ds.path_to_original_image )

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'decam_datastore' } )
        session.commit()


@pytest.fixture
def decam_datastore_through_wcs(
        datastore_factory,
        decam_cache_dir,
        decam_exposure,
        decam_reference,                  # Needed so that the right provenances get loaded into the prov_tree
        decam_default_calibrators
):
    ds = datastore_factory(
        decam_exposure,
        'S3',
        cache_dir=decam_cache_dir,
        cache_base_name='007/c4d_20230702_080904_S3_r_Sci_NBXRIO',
        overrides={ 'subtraction': { 'refset': 'test_refset_decam' } },
        save_original_image=True,
        provtag='decam_datastore',
        through_step='wcs'
    )
    ds.save_and_commit()

    yield ds

    if 'ds' in locals():
        ds.delete_everything()

    # Because save_original_image as True in the call to datastore_factory
    os.unlink( ds.path_to_original_image )

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'decam_datastore' } )
        session.commit()

@pytest.fixture
def decam_datastore_through_zp(
        datastore_factory,
        decam_cache_dir,
        decam_exposure,
        decam_reference,                  # Needed so that the right provenances get loaded into the prov_tree
        decam_default_calibrators
):
    ds = datastore_factory(
        decam_exposure,
        'S3',
        cache_dir=decam_cache_dir,
        cache_base_name='007/c4d_20230702_080904_S3_r_Sci_NBXRIO',
        overrides={ 'subtraction': { 'refset': 'test_refset_decam' } },
        save_original_image=True,
        provtag='decam_datastore',
        through_step='zp'
    )
    ds.save_and_commit()

    yield ds

    if 'ds' in locals():
        ds.delete_everything()

    # Because save_original_image as True in the call to datastore_factory
    os.unlink( ds.path_to_original_image )

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'decam_datastore' } )
        session.commit()

@pytest.fixture
def decam_datastore_through_subtraction(
        datastore_factory,
        decam_cache_dir,
        decam_exposure,
        decam_reference,
        decam_default_calibrators
):
    ds = datastore_factory(
        decam_exposure,
        'S3',
        cache_dir=decam_cache_dir,
        cache_base_name='007/c4d_20230702_080904_S3_r_Sci_NBXRIO',
        overrides={ 'subtraction': { 'refset': 'test_refset_decam' } },
        save_original_image=True,
        provtag='decam_datastore',
        through_step='subtraction'
    )
    ds.save_and_commit()

    yield ds

    if 'ds' in locals():
        ds.delete_everything()

    # Because save_original_image as True in the call to datastore_factory
    os.unlink( ds.path_to_original_image )

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'decam_datastore' } )
        session.commit()

@pytest.fixture
def decam_datastore_through_detection(
        datastore_factory,
        decam_cache_dir,
        decam_exposure,
        decam_reference,
        decam_default_calibrators
):
    ds = datastore_factory(
        decam_exposure,
        'S3',
        cache_dir=decam_cache_dir,
        cache_base_name='007/c4d_20230702_080904_S3_r_Sci_NBXRIO',
        overrides={ 'subtraction': { 'refset': 'test_refset_decam' } },
        save_original_image=True,
        provtag='decam_datastore',
        through_step='detection'
    )
    ds.save_and_commit()

    yield ds

    if 'ds' in locals():
        ds.delete_everything()

    # Because save_original_image as True in the call to datastore_factory
    os.unlink( ds.path_to_original_image )

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'decam_datastore' } )
        session.commit()

@pytest.fixture
def decam_datastore_through_cutouts(
        datastore_factory,
        decam_cache_dir,
        decam_exposure,
        decam_reference,
        decam_default_calibrators
):
    ds = datastore_factory(
        decam_exposure,
        'S3',
        cache_dir=decam_cache_dir,
        cache_base_name='007/c4d_20230702_080904_S3_r_Sci_NBXRIO',
        overrides={ 'subtraction': { 'refset': 'test_refset_decam' } },
        save_original_image=True,
        provtag='decam_datastore',
        through_step='cutting'
    )
    ds.save_and_commit()

    yield ds

    if 'ds' in locals():
        ds.delete_everything()

    # Because save_original_image as True in the call to datastore_factory
    os.unlink( ds.path_to_original_image )

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'decam_datastore' } )
        session.commit()

@pytest.fixture
def decam_processed_image(decam_datastore):

    ds = decam_datastore

    yield ds.image

    # the datastore should delete everything, so we don't need to do anything here


@pytest.fixture
def decam_fits_image_filename(download_url, decam_cache_dir):
    download_url = os.path.join(download_url, 'DECAM')
    filename = 'c4d_20221002_040239_r_v1.24.fits'
    filepath = os.path.join(decam_cache_dir, filename)
    if not os.path.isfile(filepath):
        url = os.path.join(download_url, filename)
        response = wget.download(url=url, out=filepath)

    yield filename

    if env_as_bool( "LIMIT_CACHE_USAGE" ):
        try:
            os.unlink( filepath )
        except FileNotFoundError:
            pass


@pytest.fixture
def decam_fits_image_filename2(download_url, decam_cache_dir):
    download_url = os.path.join(download_url, 'DECAM')

    filename = 'c4d_20221002_040434_i_v1.24.fits'
    filepath = os.path.join(decam_cache_dir, filename)
    if not os.path.isfile(filepath):
        url = os.path.join(download_url, filename)
        response = wget.download(url=url, out=filepath)

    yield filename

    if env_as_bool( "LIMIT_CACHE_USAGE" ):
        try:
            os.unlink( filepath )
        except FileNotFoundError:
            pass


@pytest.fixture
def decam_elais_e1_two_refs_datastore( code_version, download_url, decam_cache_dir, data_dir,
                                       datastore_factory, decam_refset ):
    SCLogger.debug( "Starting decam_elais_e1_two_refs_datastore fixture" )

    filebase = 'ELAIS-E1-r-templ'

    prov = Provenance(
        code_version_id=code_version.id,
        process='import_external_reference',
        parameters={},
    )
    prov.insert_if_needed()

    dses = []
    delete_list = []
    for dsindex, chip in enumerate( [ 27, 47 ] ):
        for ext in [ 'image.fits', 'weight.fits', 'flags.fits', 'image.yaml' ]:
            cache_path = os.path.join( decam_cache_dir, f'007/{filebase}.{chip:02d}.{ext}' )
            if os.path.isfile( cache_path ):
                SCLogger.info( f"{cache_path} exists, not redownloading" )
            else:
                url = os.path.join( download_url, 'DECAM', f'{filebase}.{chip:02d}.{ext}' )
                SCLogger.info( f"Downloading {cache_path}" )
                retry_download( url, cache_path )
                if not os.path.isfile( cache_path ):
                    raise FileNotFoundError( f"Can't find downloaded file {cache_path}" )

            if not ext.endswith('.yaml'):
                destination = os.path.join(data_dir, f'007/{filebase}.{chip:02d}.{ext}')
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                if os.getenv( "LIMIT_CACHE_USAGE" ):
                    shutil.move( cache_path, destination )
                else:
                    shutil.copy2( cache_path, destination )


        # the JSON file is generated by our cache system, not downloaded from the NERSC archive
        json_path = os.path.join( decam_cache_dir, f'007/{filebase}.{chip:02d}.image.fits.json' )
        if not env_as_bool( "LIMIT_CACHE_USAGE" ) and os.path.isfile( json_path ):
            image = copy_from_cache(Image, decam_cache_dir, json_path)
            image.provenance_id = prov.id
            image.save(verify_md5=False)  # make sure to upload to archive as well
        else:  # no cache, must create a new image object
            yaml_path = os.path.join(decam_cache_dir, f'007/{filebase}.{chip:02d}.image.yaml')

            with open( yaml_path ) as ifp:
                refyaml = yaml.safe_load( ifp )

            image = Image(**refyaml)
            image.provenance_id = prov.id
            image.filepath = f'007/{filebase}.{chip:02d}'
            image.is_coadd = True
            image.save()  # make sure to upload to archive as well

            if not env_as_bool( "LIMIT_CACHE_USAGE" ):  # save a copy of the image in the cache
                copy_to_cache( image, decam_cache_dir )

        # the datastore factory will load from cache or recreate all the other products
        # Use skip_sub because we don't want to try to find a reference for or subtract
        #   from this reference!
        ds = datastore_factory( image,
                                cache_dir=decam_cache_dir,
                                cache_base_name=f'007/{filebase}.{chip:02d}',
                                skip_sub=True,
                                provtag='decam_elais_e1_two_refs_datastore_datastore_factory')

        for filename in image.get_fullpath(as_list=True):
            assert os.path.isfile(filename)

        ds.save_and_commit()

        dses.append( ds )
        delete_list.extend( [ ds.image, ds.sources, ds.psf, ds.wcs, ds.zp,
                              ds.sub_image, ds.detections, ds.cutouts, ds.measurements ] )

    yield dses

    for ds in dses:
        ds.delete_everything()

    # Clean out the provenance tag that may have been created by the datastore_factory
    with SmartSession() as session:
        for tag in [ 'decam_elais_e1_two_refs_datastore',
                     'decam_elais_e1_two_refs_datastore_datastore_factory' ]:
            session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': tag } )
        session.commit()

    ImageAligner.cleanup_temp_images()

@pytest.fixture
def decam_ref_datastore( decam_elais_e1_two_refs_datastore ):
    return decam_elais_e1_two_refs_datastore[0]

@pytest.fixture
def decam_elais_e1_two_references( decam_elais_e1_two_refs_datastore ):
    refs = []

    # This doesn't work right, because the refmaker makes assumptions
    #    about the provenance of References that are wrong.
    # prov = maker.refset.provenances[0]
    # maker = refmaker_factory('test_refset_decam', 'DECam', 'decam_elais_e1_two_references' )
    # maker.make_refset()

    ds = decam_elais_e1_two_refs_datastore[0]
    upstrs = Provenance.get_batch( [ ds.image.provenance_id, ds.sources.provenance_id ] )
    refprov = Provenance(
        process='referencing',
        upstreams=upstrs,
        parameters={},
    )
    refprov.insert_if_needed()
    refset = RefSet.get_by_name( 'test_refset_decam' )
    refset.append_provenance( refprov )

    for ds in decam_elais_e1_two_refs_datastore:
        ref = Reference(
            image_id = ds.image.id,
            provenance_id = refprov.id,
            instrument = ds.image.instrument,
            section_id = ds.image.section_id,
            filter = ds.image.filter,
            target = ds.image.target,
        )
        ref.insert()
        refs.append( ref )

    yield refs

    with SmartSession() as session:
        session.execute( sa.delete( Reference ).where( Reference._id.in_( [ r.id for r in refs ] ) ) )
        session.execute( sa.delete( Provenance ).where( Provenance._id==refprov.id ) )
        session.commit()


@pytest.fixture
def decam_reference( decam_elais_e1_two_references ):
    return decam_elais_e1_two_references[0]

@pytest.fixture
def decam_ref_datastore( decam_elais_e1_two_refs_datastore ):
    return decam_elais_e1_two_refs_datastore[0]
