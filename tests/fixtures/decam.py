import pytest
import os
import wget
import yaml
import subprocess
import shutil
import warnings

import sqlalchemy as sa
import numpy as np

from astropy.io import fits
from astropy.time import Time

from models.base import SmartSession, _logger
from models.instrument import Instrument, get_instrument_instance
from models.decam import DECam  # need this import to make sure DECam is added to the Instrument list
from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.datafile import DataFile
from models.reference import Reference

from util.retrydownload import retry_download
from util.exceptions import SubprocessFailure


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
        if os.path.isdir(os.path.join(cache_dir, 'DECam_default_calibrators')):
            shutil.copytree(
                os.path.join(cache_dir, 'DECam_default_calibrators'),
                os.path.join(data_dir, 'DECam_default_calibrators'),
                dirs_exist_ok=True,
            )

        decam = get_instrument_instance( 'DECam' )
        sections = [ 'N1', 'S1' ]
        filters = [ 'r', 'i', 'z', 'g']
        for sec in sections:
            for calibtype in [ 'flat', 'fringe' ]:
                for filt in filters:
                    decam._get_default_calibrator( 60000, sec, calibtype=calibtype, filter=filt )
        decam._get_default_calibrator( 60000, sec, calibtype='linearity' )

        # store the calibration files in the cache folder
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
        # _logger.debug('tear down of decam_default_calibrators')
        imagestonuke = set()
        datafilestonuke = set()
        with SmartSession() as session:
            for sec in [ 'N1', 'S1' ]:
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
                im = session.scalars( sa.select(Image).where(Image.id == imid )).first()
                im.delete_from_disk_and_database( session=session, commit=False )

            for dfid in datafilestonuke:
                df = session.scalars( sa.select(DataFile).where(DataFile.id == dfid )).first()
                df.delete_from_disk_and_database( session=session, commit=False )

            session.commit()

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
    with SmartSession() as session:
        code_version = session.merge(code_version)
        p = Provenance(
            process="preprocessing",
            code_version=code_version,
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
        p.update_id()
        p = session.merge(p)
        session.commit()

    yield p

    with SmartSession() as session:
        session.delete(p)
        session.commit()


@pytest.fixture(scope='module')
def decam_reduced_origin_exposures():
    decam = DECam()
    yield decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                       proposals='2023A-716082',
                                       skip_exposures_in_database=False,
                                       proc_type='instcal' )


@pytest.fixture(scope='module')
def decam_raw_origin_exposures():
    decam = DECam()
    yield decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                       proposals='2023A-716082',
                                       skip_exposures_in_database=False,
                                       proc_type='raw' )


@pytest.fixture(scope="session")
def decam_filename(download_url, data_dir, decam_cache_dir):
    """Pull a DECam exposure down from the NOIRLab archives.

    Because this is a slow process (depending on the NOIRLab archive
    speed, it can take up to minutes), first look for this file
    in the cache_dir, and if it exists, and copy it. If not,
    actually download the image from NOIRLab into the cache_dir,
    and create a symlink to the temp_dir. That way, until the
    user manually deletes the cached file, we won't have to redo the
    slow NOIRLab download again.
    """
    base_name = 'c4d_221104_074232_ori.fits.fz'
    filename = os.path.join(data_dir, base_name)
    if not os.path.isfile(filename):
        cachedfilename = os.path.join(decam_cache_dir, base_name)
        os.makedirs(os.path.dirname(cachedfilename), exist_ok=True)

        if not os.path.isfile(cachedfilename):
            url = os.path.join(download_url, 'DECAM', base_name)
            response = wget.download(url=url, out=cachedfilename)
            assert response == cachedfilename

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        shutil.copy2(cachedfilename, filename)

    yield filename

    if os.path.isfile(filename):
        os.remove(filename)


@pytest.fixture(scope="session")
def decam_exposure(decam_filename, data_dir):
    filename = decam_filename

    with fits.open( filename, memmap=True ) as ifp:
        hdr = ifp[0].header
    exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter', 'project', 'target' ] )

    exposure = Exposure( filepath=filename, instrument='DECam', **exphdrinfo )
    exposure.save()  # save to archive and get an MD5 sum

    with SmartSession() as session:
        session.add(exposure)
        session.commit()

    yield exposure

    exposure.delete_from_disk_and_database()


@pytest.fixture
def decam_raw_image( decam_exposure, provenance_base ):
    image = Image.from_exposure(decam_exposure, section_id='N1')
    image.data = image.raw_data.astype(np.float32)
    image.provenance = provenance_base
    image.save()

    yield image

    image.delete_from_disk_and_database()


@pytest.fixture
def decam_small_image(decam_raw_image):
    image = decam_raw_image
    image.data = image.data[256:256+512, 256:256+512].copy()  # make it C-contiguous

    yield image


@pytest.fixture
def decam_datastore(
        datastore_factory,
        decam_cache_dir,
        decam_exposure,
        decam_default_calibrators,  # not used directly, but makes sure this is pre-fetched from cache
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
        'N1',
        cache_dir=decam_cache_dir,
        cache_base_name='115/c4d_20221104_074232_N1_g_Sci_FVOSOC'
    )
    # This save is redundant, as the datastore_factory calls save_and_commit
    # However, I leave this here because it is a good test that calling it twice
    # does not cause any problems.
    ds.save_and_commit()

    delete_list = [
        ds.image, ds.sources, ds.psf, ds.wcs, ds.zp, ds.sub_image, ds.detections, ds.cutouts, ds.measurements
    ]

    yield ds

    # cleanup
    if 'ds' in locals():
        ds.delete_everything()

    # make sure that these individual objects have their files cleaned up,
    # even if the datastore is cleared and all database rows are deleted.
    for obj in delete_list:
        if obj is not None and hasattr(obj, 'delete_from_disk_and_database'):
            obj.delete_from_disk_and_database(archive=True)


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


@pytest.fixture
def decam_fits_image_filename2(download_url, decam_cache_dir):
    download_url = os.path.join(download_url, 'DECAM')

    filename = 'c4d_20221002_040434_i_v1.24.fits'
    filepath = os.path.join(decam_cache_dir, filename)
    if not os.path.isfile(filepath):
        url = os.path.join(download_url, filename)
        response = wget.download(url=url, out=filepath)

    yield filename


@pytest.fixture
def decam_ref_datastore( code_version, download_url, decam_cache_dir, data_dir, datastore_factory ):
    filebase = 'DECaPS-West_20220112.g.32'

    # I added this mirror so the tests will pass, and we should remove it once the decam image goes back up to NERSC
    # TODO: should we leave these as a mirror in case NERSC is down?
    dropbox_urls = {
        '.image.fits': 'https://www.dropbox.com/scl/fi/x8rzwfpe4zgc8tz5mv0e2/DECaPS-West_20220112.g.32.image.fits?rlkey=5wse43bby3tce7iwo2e1fm5ru&dl=1',
        '.weight.fits': 'https://www.dropbox.com/scl/fi/dfctqqj3rjt09wspvyzb3/DECaPS-West_20220112.g.32.weight.fits?rlkey=tubr3ld4srf59hp0cuxrv2bsv&dl=1',
        '.flags.fits': 'https://www.dropbox.com/scl/fi/y693ckhcs9goj1t7s0dty/DECaPS-West_20220112.g.32.flags.fits?rlkey=fbdyxyzjmr3g2t9zctcil7106&dl=1',
    }

    for ext in [ '.image.fits', '.weight.fits', '.flags.fits', '.image.yaml' ]:
        cache_path = os.path.join(decam_cache_dir, f'115/{filebase}{ext}')
        fzpath = cache_path + '.fz'
        if os.path.isfile(cache_path):
            _logger.info( f"{cache_path} exists, not redownloading." )
        else:  # need to download!
            url = os.path.join(download_url, 'DECAM', filebase + ext)
            retry_download( url, cache_path )
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(f'Cannot find downloaded file: {cache_path}')

        destination = os.path.join(data_dir, f'115/{filebase}{ext}')
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2( cache_path, destination )

    yaml_path = os.path.join(decam_cache_dir, f'115/{filebase}.image.yaml')

    with open( yaml_path ) as ifp:
        refyaml = yaml.safe_load( ifp )

    with SmartSession() as session:
        code_version = session.merge(code_version)
        prov = Provenance(
            process='preprocessing',
            code_version=code_version,
            parameters={},
            upstreams=[],
            is_testing=True,
        )
        # check if this Image is already in the DB
        existing = session.scalars(
            sa.select(Image).where(Image.filepath == f'115/{filebase}')
        ).first()
        if existing is None:
            image = Image(**refyaml)
        else:
            # overwrite the existing row data using the YAML
            for key, value in refyaml.items():
                if (
                        key not in ['id', 'image_id', 'created_at', 'modified'] and
                        value is not None
                ):
                    setattr(existing, key, value)
            image = existing  # replace with the existing object

        image.provenance = prov
        image.filepath = f'115/{filebase}'
        image.is_coadd = True
        image.save(verify_md5=False)  # make sure to upload to archive as well

        image.copy_to_cache( decam_cache_dir )

        ds = datastore_factory(image, cache_dir=decam_cache_dir, cache_base_name=f'115/{filebase}')

        for filename in image.get_fullpath(as_list=True):
            assert os.path.isfile(filename)

        ds.save_and_commit(session)
        session.commit()

    delete_list = [
        ds.image, ds.sources, ds.psf, ds.wcs, ds.zp, ds.sub_image, ds.detections, ds.cutouts, ds.measurements
    ]

    yield ds

    ds.delete_everything()

    # make sure that these individual objects have their files cleaned up,
    # even if the datastore is cleared and all database rows are deleted.
    for obj in delete_list:
        if obj is not None and hasattr(obj, 'delete_from_disk_and_database'):
            obj.delete_from_disk_and_database(archive=True)


@pytest.fixture
def decam_reference(decam_ref_datastore):
    ds = decam_ref_datastore
    with SmartSession() as session:
        prov = Provenance(
            code_version=ds.image.provenance.code_version,
            process='reference',
            parameters={'test_parameter': 'test_value'},
            upstreams=[
                ds.image.provenance,
                ds.sources.provenance,
                ds.psf.provenance,
                ds.wcs.provenance,
                ds.zp.provenance,
            ],
            is_testing=True,
        )
        prov = session.merge(prov)

        ref = Reference()
        ref.image = ds.image
        ref.provenance = prov
        ref.validity_start = Time(50000, format='mjd', scale='utc').isot
        ref.validity_end = Time(65000, format='mjd', scale='utc').isot
        ref.section_id = ds.image.section_id
        ref.filter = ds.image.filter
        ref.target = ds.image.target
        ref.project = ds.image.project

        ref = ref.merge_all(session=session)
        if not sa.inspect(ref).persistent:
            session.add(ref)
        session.commit()

    yield ref

    if 'ref' in locals():
        with SmartSession() as session:
            ref = session.merge(ref)
            if sa.inspect(ref).persistent:
                session.delete(ref.provenance)  # should also delete the reference image
            session.commit()


@pytest.fixture

def decam_subtraction(decam_reference, decam_processed_image, subtractor, decam_cache_dir):
    filepath = '115/c4d_20221104_074232_N1_g_Diff_7EGWL3_u-4ea5cc.image.fits'

    upstreams = []
    for im in [decam_reference.image, decam_processed_image]:
        upstreams.append(im.provenance)
        for att in ['sources', 'psf', 'wcs', 'zp']:
            upstreams.append(getattr(im, att).provenance)

    prov = Provenance(
        process='subtraction',
        code_version=decam_processed_image.provenance.code_version,
        parameters=subtractor.pars.get_critical_pars(),
        upstreams=upstreams,
        is_testing=True,
    )

    if prov.id[:6] not in filepath:
        warnings.warn(f"Provenance ID {prov.id[:6]} not in filepath {filepath}")

    if os.path.isfile(os.path.join(decam_cache_dir, filepath)):
        sub_im = Image.copy_from_cache(decam_cache_dir, filepath)
        sub_im.upstream_images = [decam_reference.image, decam_processed_image]

        if sub_im._aligned_images is None:
            align_ref_prov = Provenance(
                code_version=decam_reference.image.provenance.code_version,
                process='alignment',
                parameters=prov.parameters['alignment'],
                upstreams=[
                    decam_reference.image.provenance,
                    decam_reference.sources.provenance,
                    decam_reference.psf.provenance,
                    decam_reference.wcs.provenance,
                    decam_reference.zp.provenance,
                    decam_processed_image.provenance,
                    decam_processed_image.sources.provenance,
                    decam_processed_image.psf.provenance,
                    decam_processed_image.wcs.provenance,
                ],
            )
            align_new_prov = Provenance(
                code_version=decam_reference.image.provenance.code_version,
                process='alignment',
                parameters=prov.parameters['alignment'],
                upstreams=[
                    decam_processed_image.provenance,
                    decam_processed_image.sources.provenance,
                    decam_processed_image.wcs.provenance,
                    decam_processed_image.zp.provenance,
                ],
            )
            aligned_ref = None
            aligned_new = None
            aligned_ref_file = '115/c4d_20220113_050224_N1_g_Warped_ZBELGP'
            if os.path.isfile(os.path.join(decam_cache_dir, aligned_ref_file + '.image.fits.json')):
                aligned_ref = Image.copy_from_cache(decam_cache_dir, aligned_ref_file + '.image.fits.json')
                aligned_ref.info['original_image_id'] = decam_reference.image.id
                aligned_ref.provenance = align_ref_prov

            aligned_new_file = '115/c4d_20221104_074232_N1_g_Warped_JZNHWZ'
            if os.path.isfile(os.path.join(decam_cache_dir, aligned_new_file + '.image.fits.json')):
                aligned_new = Image.copy_from_cache(decam_cache_dir, aligned_new_file + '.image.fits.json')
                aligned_new.info['original_image_id'] = decam_processed_image.id
                aligned_new.provenance = align_new_prov

            if aligned_ref is not None and aligned_new is not None:
                sub_im._aligned_images = [aligned_ref, aligned_new]

        sub_im.ref_image_id = decam_reference.image.id
        sub_im.ref_image = decam_reference.image
        sub_im.provenance = prov
        sub_im.load_upstream_products()  # make sure stuff like WCS is loaded for upstream_images
        sub_im.coordinates_to_alignment_target()  # propagate WCS to sub_im

    else:
        ds = subtractor.run(decam_processed_image)

        ds.sub_image.save()
        sub_im = ds.sub_image
        sub_im.copy_to_cache(decam_cache_dir)

    for im in sub_im.aligned_images:  # also save the aligned images...
        im.save()
        im.copy_to_cache(decam_cache_dir)

    yield sub_im

    if 'sub_im' in locals():
        for im in sub_im.aligned_images:
            im.delete_from_disk_and_database(archive=True)
        sub_im.delete_from_disk_and_database(archive=True)


@pytest.fixture
def decam_detection_list(decam_subtraction, detector, decam_cache_dir):
    prov = Provenance(
        process='detection',
        code_version=decam_subtraction.provenance.code_version,
        parameters=detector.pars.get_critical_pars(),
        upstreams=[decam_subtraction.provenance],
        is_testing=True,
    )
    filepath = decam_subtraction.filepath + f'.detections_{prov.id[:6]}.npy'

    if os.path.isfile(filepath):
        detections = SourceList.copy_from_cache(decam_cache_dir, filepath)
        detections.image_id = decam_subtraction.id
        detections.provenance = prov
    else:  # no cache, need to product a new object
        ds = detector.run(decam_subtraction)
        detections = ds.sub_image.sources
        detections.provenance = prov
        detections.save()
        detections.copy_to_cache(decam_cache_dir)

    yield detections

    # must delete the detections (especially the file) because I'm not sure the datastore will delete it
    detections.delete_from_disk_and_database(archive=True)
