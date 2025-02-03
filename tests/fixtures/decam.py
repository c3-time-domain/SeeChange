import pytest
import os
import re
import wget
import yaml
import shutil
import pathlib

import sqlalchemy as sa
import numpy as np

from astropy.io import fits

from models.base import SmartSession
from models.instrument import Instrument, get_instrument_instance
from models.decam import DECam  # need this import to make sure DECam is added to the Instrument list
from models.provenance import Provenance
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
from util.fits import read_fits_image
import util.radec


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
        sections = [ 'S2', 'S3', 'N16' ]
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
            for sec in [ 'S2', 'S3', 'N16' ]:
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


@pytest.fixture(scope='module')
def decam_reduced_origin_exposure_files( decam_cache_dir, download_url, data_dir ):
    fnames = []
    for which in [ 'i', 'w', 'd' ]:
        outname = f'c4d_170925_083024_oo{which}_r_v1.fits.fz'
        outpath = os.path.join( data_dir, outname )
        fnames.append( outpath )
        url = os.path.join( download_url, 'DECAM', outname )

        if not os.path.isfile( outpath ):
            if env_as_bool( 'LIMIT_CACHE_USAGE' ):
                SCLogger.debug( f"Downloading {outname}" )
                wget.download( url=url, out=outpath )
            else:
                cachedpath = os.path.join( decam_cache_dir, outname )
                os.makedirs( os.path.dirname( cachedpath ), exist_ok=True )
                if not os.path.isfile( cachedpath ):
                    SCLogger.debug( f"Downloading {outname}" )
                    response = wget.download( url=url, out=cachedpath )
                    assert response == cachedpath
                else:
                    SCLogger.debug( f"Cached file {outname} exists, not redownloading." )

                shutil.copy2( cachedpath, outpath )

    yield fnames

    for f in fnames:
        if os.path.isfile( f ):
            os.remove( f )


@pytest.fixture
def decam_reduced_origin_exposure_loaded_in_db( decam_reduced_origin_exposure_files, provenance_base ):
    expfile, wtfile, flgfile = decam_reduced_origin_exposure_files
    expobj = None

    try:
        with fits.open( expfile ) as ifp:
            hdr = { k: v for k, v in ifp[0].header.items()
                    if k in ( 'PROCTYPE', 'PRODTYPE', 'FILENAME', 'TELESCOP', 'OBSERVAT', 'INSTRUME'
                              'OBS-LONG', 'OBS-LAT', 'EXPTIME', 'DARKTIME', 'OBSID',
                              'DATE-OBS', 'TIME-OBS', 'MJD-OBS', 'OBJECT', 'PROGRAM',
                              'OBSERVER', 'PROPID', 'FILTER', 'RA', 'DEC', 'HA', 'ZD', 'AIRMASS',
                              'VSUB', 'GSKYPHOT', 'LSKYPHOT' ) }
        exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter',
                                                            'project', 'target' ] )
        ra = util.radec.parse_sexigesimal_degrees( hdr['RA'], hours=True )
        dec = util.radec.parse_sexigesimal_degrees( hdr['DEC'] )

        expobj = Exposure( current_file=expfile, invent_filepath=True, type='Sci', origin_identifier='foo',
                           instrument='DECam', header=hdr, ra=ra, dec=dec, preproc_bitflag=127, format='fits',
                           components=['image', 'weight', 'flags'],
                           **exphdrinfo )
        expobj.save( expfile, wtfile, flgfile )
        provenance_base.insert_if_needed()
        expobj.insert()

        yield expobj

    finally:
        if expobj is not None:
            expobj.delete_from_disk_and_database()


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
def decam_exposure_factory(download_url, data_dir, decam_cache_dir):
    exposurestodelete = []

    def make_decam_exposure( decam_exposure_name ):
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

        with fits.open( filename, memmap=True ) as ifp:
            hdr = ifp[0].header
        exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter', 'project', 'target',
                                                            'ra','dec' ] )

        exposure = Exposure( filepath=filename, instrument='DECam', **exphdrinfo )
        exposure.save()  # save to archive and get an MD5 sum
        exposure.insert()

        exposurestodelete.append( exposure )
        exposure.data.clear_cache()
        return exposure

    yield make_decam_exposure

    for exposure in exposurestodelete:
        exposure.delete_from_disk_and_database()


# All three of the exposures are r-band, and
#   of the same field (the field that goes with
#   decam_reference).
#
# We make these session fixtures because they're slow,
#   we don't want them to run over and over again,
#   and because exposures are the sort of thing you load
#   in once anyway.
@pytest.fixture(scope="session")
def decam_exposure( decam_exposure_factory ):
    # 2025-01-06 RKNOP... changing the
    #   exposure we're using, which will be painful
    #   as lots of tests will need to be updated,
    #   but the new one has a nice SN in it that
    #   was found for DECAT-DDF with lensgrinder.
    #   (DC21cyddn).
    # (Why I didn't choose a test exposure that way
    # in the first place is a mystery.)
    # Lots of 'S3' in other fixtures have been
    #  changed to 'S2' to go along with this.
    # return decam_exposure_factory( 'c4d_230702_080904_ori.fits.fz' )
    return decam_exposure_factory( 'c4d_211025_044847_ori.fits.fz' )


@pytest.fixture
def decam_raw_image_provenance( provenance_base ):
    return provenance_base


@pytest.fixture
def decam_raw_image( decam_exposure, provenance_base ):
    image = Image.from_exposure(decam_exposure, section_id='S2')
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
        'S2',
        cache_dir=decam_cache_dir,
        cache_base_name='007/c4d_20211025_044847_S2_r_Sci_IDDLGQ',
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


# This next fixture returns a factory that's
# not general; it will only work for DECam
# fields that use decam_reference (so they
# have to be r-band, in the right specific
# field, and on chip S2.)
@pytest.fixture
def decam_partial_datastore_factory( datastore_factory, decam_cache_dir,
                                     decam_reference, decam_default_calibrators ):
    dses = []

    def decam_partial_datastore_maker( exposure, through_step ):
        # This is a little ugly, though it's replacing something that
        #   used to just be hardcoded, so it's probably not any less
        #   ugly than that.  Look at the exposure filename, and worm
        #   it around to figure out the image filename.
        mat = re.search( r'^c4d_(?P<yymmdd>\d{6})_(?P<hhmmss>\d{6})_ori\.fits\.fz$', str(exposure.filepath) )
        if mat is None:
            raise ValueError( f"Failed to match {exposure.filepath}" )
        cache_base_name = f'007/c4d_20{mat.group("yymmdd")}_{mat.group("hhmmss")}_S2_r_Sci_IDDLGQ'

        ds = datastore_factory(
            exposure,
            'S2',
            cache_dir=decam_cache_dir,
            cache_base_name=cache_base_name,
            overrides={ 'subtraction': { 'refset': 'test_refset_decam' } },
            save_original_image=True,
            provtag='decam_datastore',
            through_step=through_step
        )
        ds.save_and_commit()

        # Here's hoping I really understand the python scoping rules
        dses.append( ds )
        return ds

    yield decam_partial_datastore_maker

    for ds in dses:
        ds.delete_everything()
        # Because save_original_image as True in the call to datastore_factory
        os.unlink( ds.path_to_original_image )

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ),
                         {'tag': 'decam_datastore' } )
        session.commit()


@pytest.fixture
def decam_datastore_through_preprocessing( decam_exposure, decam_partial_datastore_factory ):
    return decam_partial_datastore_factory( decam_exposure, 'preprocessing' )


@pytest.fixture
def decam_datastore_through_extraction( decam_exposure, decam_partial_datastore_factory ):
    return decam_partial_datastore_factory( decam_exposure, 'extraction' )


@pytest.fixture
def decam_datastore_through_bg( decam_exposure, decam_partial_datastore_factory ):
    return decam_partial_datastore_factory( decam_exposure, 'bg' )


@pytest.fixture
def decam_datastore_through_wcs( decam_exposure, decam_partial_datastore_factory ):
    return decam_partial_datastore_factory( decam_exposure, 'wcs' )


@pytest.fixture
def decam_datastore_through_zp( decam_exposure, decam_partial_datastore_factory ):
    return decam_partial_datastore_factory( decam_exposure, 'zp' )


@pytest.fixture
def decam_datastore_through_subtraction( decam_exposure, decam_partial_datastore_factory ):
    return decam_partial_datastore_factory( decam_exposure, 'subtraction' )


@pytest.fixture
def decam_datastore_through_detection( decam_exposure, decam_partial_datastore_factory ):
    return decam_partial_datastore_factory( decam_exposure, 'detection' )


@pytest.fixture
def decam_datastore_through_cutouts( decam_exposure, decam_partial_datastore_factory ):
    return decam_partial_datastore_factory( decam_exposure, 'cutting' )


@pytest.fixture
def decam_datastore_through_measurements( decam_exposure, decam_partial_datastore_factory ):
    return decam_partial_datastore_factory( decam_exposure, 'measuring' )


@pytest.fixture
def decam_datastore_through_scoring( decam_exposure, decam_partial_datastore_factory ):
    return decam_partial_datastore_factory( decam_exposure, 'scoring' )


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
        wget.download(url=url, out=filepath)

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
        wget.download(url=url, out=filepath)

    yield filename

    if env_as_bool( "LIMIT_CACHE_USAGE" ):
        try:
            os.unlink( filepath )
        except FileNotFoundError:
            pass


@pytest.fixture
def decam_elais_e1_two_refs_datastore( code_version, download_url, decam_cache_dir, data_dir, datastore_factory ):
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
    for dsindex, chip in enumerate( [ 26, 47 ] ):
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
        json_path = os.path.join( decam_cache_dir, f'007/{filebase}.{chip:02d}.json' )
        if not env_as_bool( "LIMIT_CACHE_USAGE" ) and os.path.isfile( json_path ):
            image = copy_from_cache(Image, decam_cache_dir, json_path)
            image.provenance_id = prov.id
            # This is a bit gratuitous; it's going to overwrite the file on idks that
            #  we just copied from cache.
            image.save(verify_md5=False)  # make sure to upload to archive as well
        else:  # no cache, must create a new image object
            yaml_path = os.path.join(decam_cache_dir, f'007/{filebase}.{chip:02d}.image.yaml')

            with open( yaml_path ) as ifp:
                refyaml = yaml.safe_load( ifp )

            image = Image(**refyaml)
            image.filter = image.filter_short
            image.provenance_id = prov.id
            image.filepath = f'007/{filebase}.{chip:02d}'
            image.is_coadd = True
            image.components = [ 'image', 'weight', 'flags' ]
            image.md5sum_components = [ None, None, None ]
            for comp, path in zip( image.components, image.get_fullpath() ):
                data, header = read_fits_image( path, output='both' )
                if comp == 'image':
                    image.data = data
                    image.header = header
                else:
                    setattr( image, comp, data )
            image.save()  # make sure to upload to archive as well

            # NOTE -- if we aren't using the cache, this fixture is kind of broken.
            # The datastore_factory is going to run preprocessing on this image,
            # but this image has already been preprocessed.  If the image is in the
            # cache, datastore_factory will just read it, and this not double-apply
            # the preprocessing step.
            # (...I think?  datastore_factory is such a beast.)
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


# TODO -- the provenance upstreams are now wrong after PR #407
# (It may be possible to go back to refmaker_factory?)
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

    # Have to have reproducible Reference ids for the cache to work
    for refid, ds in zip( [ '74c10990-51f0-49d2-9e92-cb9c3fe04c18', '80a00042-ac0c-4ebe-9fde-435c8b6b65b7' ],
                          decam_elais_e1_two_refs_datastore ):
        ref = Reference(
            _id = refid,
            zp_id = ds.zp.id,
            provenance_id = refprov.id
        )
        ref.insert()
        refs.append( ref )

    # Create the test_refset_decam reference set
    refset = RefSet( name='test_refset_decam', provenance_id=refprov.id )
    refset.insert()

    yield refs

    with SmartSession() as session:
        session.execute( sa.delete( Reference ).where( Reference._id.in_( [ r.id for r in refs ] ) ) )
        session.execute( sa.delete( Provenance ).where( Provenance._id==refprov.id ) )
        session.execute( sa.delete( RefSet ).where( RefSet.name=='test_refset_decam' ) )
        session.commit()


@pytest.fixture
def decam_reference( decam_elais_e1_two_references ):
    return decam_elais_e1_two_references[0]


@pytest.fixture(scope='session')
def get_cached_decam_image( code_version, decam_cache_dir, download_url, datastore_factory ):
    improv = Provenance( process="acquired",
                         parameters={},
                         code_version_id=code_version.id,
                         upstreams=[],
                         is_testing=True )
    improv.insert_if_needed()
    provbarf = improv.id[:6]

    def cached_decam_image_getter( fname ):
        """Utility function to make a Datastore through zp of a downloaded image.

        Paramters
        ---------
          fname : Path
            The relative path of the image in the file store, but
            *without* the provenance tag component of the filename.
            (See decam_four_offset_refs as an example.)  The image will
            be found at download_url/DECAM/fname.name*, where * is
            .image.yaml, .image.fits, .weight.fits, and .flags.fits.

        """

        cache_json = f"{fname}.image.fits.json"
        if not env_as_bool( "LIMIT_CACHE_USAGE" ) and os.path.isfile( os.path.join( decam_cache_dir, cache_json ) ):
            # If the json file exists in the cache, restore the image (+weight and flags) from there
            SCLogger.debug( f'Getting decam image {fname} from cache' )
            img = copy_from_cache( Image, decam_cache_dir, cache_json )
        else:
            # Otherwise, download the files, and copy to cache
            SCLogger.debug( f'Downloading decam images for {fname}' )
            tmppaths = {}
            for ext in [ '.image.yaml', '.image.fits', '.weight.fits', '.flags.fits' ]:
                url = f"{download_url}/DECAM/{fname.name}{ext}"
                tmppaths[ext] = pathlib.Path( decam_cache_dir ) / f'{fname.name}{ext}'
                if not tmppaths[ext].is_file():
                    wget.download( url=url, out=str(tmppaths[ext].resolve()) )
            kwargs = yaml.safe_load( open( tmppaths['.image.yaml'] ) )
            kwargs['provenance_id'] = improv.id
            img = Image( **kwargs )
            with fits.open( tmppaths['.image.fits'] ) as hdu:
                # Have to do the astype here to convert from >f4 to native type,
                #  because sep.Background seems to need that (!?!)
                img.data = hdu[0].data.astype( np.float32 )
                img.header = hdu[0].header
                img.set_corners_from_header_wcs()
                img.calculate_coordinates()
            with fits.open( tmppaths['.weight.fits'] ) as hdu:
                img.weight = hdu[0].data.astype( np.float32 )
            with fits.open( tmppaths['.flags.fits'] ) as hdu:
                img.flags = hdu[0].data
            img.filepath = img.invent_filepath()
            expectedname = f'{str(fname)}_{provbarf}'
            assert img.filepath[-len(expectedname):] == expectedname
            img.save()
            # Don't save the image to the cache here; that will happen
            #   in the datastore factory below, and by then the image
            #   will have the right minra/maxra/etc. fields loaded
            #   (from post wcs solution)
            for tmppath in tmppaths.values():
                tmppath.unlink( missing_ok=True )

        SCLogger.debug( f"Running datastore_factory for decam image {fname}" )
        ds = datastore_factory( img, cache_dir=decam_cache_dir,
                                cache_base_name=f'{fname}_{provbarf}', through_step='zp',
                                provtag='cached_decam_image_getter' )
        SCLogger.debug( f"Done getting decam image {fname}" )

        return ds

    yield cached_decam_image_getter

    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag='cached_decam_image_getter'" ) )
        session.commit()


@pytest.fixture
def decam_four_offset_refs( get_cached_decam_image ):
    files = [ pathlib.Path( i )
              for i in [ '029/c4d_20221017_054830_S31_r_Sci',
                         '029/c4d_20181130_024059_S8_r_Sci',
                         '030/c4d_20181130_024059_S9_r_Sci',
                         '029/c4d_20181130_024059_S2_r_Sci' ] ]
    dses = [ get_cached_decam_image(i) for i in files ]

    yield dses

    for ds in dses:
        ds.delete_everything()


@pytest.fixture
def decam_four_refs_alignment_target( get_cached_decam_image ):
    ds = get_cached_decam_image( pathlib.Path( '030/c4d_20240924_061837_S4_r_Sci' ) )
    yield ds
    ds.delete_everything()


# This fixture takes minutes and minutes to run; any tests that
#    use it should probably be marked for skipping unles
#    RUN_SLOW_TESTS is set.
@pytest.fixture
def decam_17_offset_refs( get_cached_decam_image ):
    files = [ pathlib.Path( i )
              for i in [ '029/c4d_20221017_054113_S31_r_Sci',
                         '029/c4d_20221017_055008_S31_r_Sci',
                         '030/c4d_20171116_030256_S26_r_Sci',
                         '029/c4d_20221017_053359_S31_r_Sci',
                         '029/c4d_20221017_054253_S31_r_Sci',
                         '029/c4d_20140819_085520_S4_r_Sci',
                         '030/c4d_20140819_085520_S5_r_Sci',
                         '029/c4d_20221017_053539_S31_r_Sci',
                         '029/c4d_20221017_054830_S31_r_Sci',
                         '030/c4d_20171207_024824_N31_r_Sci',
                         '030/c4d_20171207_024824_N27_r_Sci',
                         '029/c4d_20140924_064323_N21_r_Sci',
                         '030/c4d_20140924_064323_N22_r_Sci',
                         '030/c4d_20140924_064323_N26_r_Sci',
                         '029/c4d_20181130_024059_S8_r_Sci',
                         '030/c4d_20181130_024059_S9_r_Sci',
                         '029/c4d_20181130_024059_S2_r_Sci' ] ]
    dses = [ get_cached_decam_image(i) for i in files ]

    yield dses

    for ds in dses:
        ds.delete_everything()
