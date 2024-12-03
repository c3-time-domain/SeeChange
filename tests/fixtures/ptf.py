import pytest
import warnings
import uuid
import os
import re
import shutil
import base64
import hashlib
import requests

import numpy as np

import sqlalchemy as sa
from bs4 import BeautifulSoup
from datetime import datetime
from astropy.io import fits

from models.base import SmartSession
from models.ptf import PTF  # need this import to make sure PTF is added to the Instrument list
from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference
from models.refset import RefSet

from improc.alignment import ImageAligner

from pipeline.data_store import DataStore
from pipeline.coaddition import Coadder

from util.retrydownload import retry_download
from util.logger import SCLogger
from util.cache import copy_to_cache, copy_from_cache
from util.util import env_as_bool


@pytest.fixture(scope='session')
def ptf_cache_dir(cache_dir):
    output = os.path.join(cache_dir, 'PTF')
    if not os.path.isdir(output):
        os.makedirs(output)

    yield output


@pytest.fixture(scope='session')
def ptf_bad_pixel_map(download_url, data_dir, ptf_cache_dir):
    filename = 'C11/masktot.fits'  # TODO: add more CCDs if needed
    # url = 'https://portal.nersc.gov/project/m2218/pipeline/test_images/2012021x/'
    url = os.path.join(download_url, 'PTF/10cwm/2012021x/')
    data_dir = os.path.join(data_dir, 'PTF_calibrators')
    data_path = os.path.join(data_dir, filename)

    if env_as_bool( "LIMIT_CACHE_USAGE" ):
        if not os.path.isfile( data_path ):
            os.makedirs( os.path.dirname( data_path ), exist_ok=True )
            retry_download( url + filename, data_path )
        if not os.path.isfile( data_path ):
            raise FileNotFoundError( f"Can't read {data_path}.  It should have been downloaded!" )
    else:
        # is this file already on the cache? if not, download it
        cache_path = os.path.join(ptf_cache_dir, filename)
        if not os.path.isfile(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            retry_download(url + filename, cache_path)

        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"Can't read {cache_path}. It should have been downloaded!")

        if not os.path.isfile(data_path):
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            shutil.copy2(cache_path, data_path)

    with fits.open(data_path) as hdul:
        data = (hdul[0].data == 0).astype('uint16')  # invert the mask (good is False, bad is True)

    data = np.roll(data, -1, axis=0)  # shift the mask by one pixel (to match the PTF data)
    data[-1, :] = 0  # the last row that got rolled seems to be wrong
    data = np.roll(data, 1, axis=1)  # shift the mask by one pixel (to match the PTF data)
    data[:, 0] = 0  # the last column that got rolled seems to be wrong

    yield data

    os.remove(data_path)

    # remove (sub)folder if empty
    dirname = os.path.dirname(data_path)
    for i in range(2):
        if os.path.isdir(dirname) and len(os.listdir(dirname)) == 0:
            os.removedirs(dirname)
            dirname = os.path.dirname(dirname)


@pytest.fixture(scope='session')
def ptf_downloader(provenance_preprocessing, download_url, data_dir, ptf_cache_dir):
    """Downloads an image for ptf.

    At the end, only count on the file being in data_dir.  It *might*
    have also put the file in ptf_cache_dir, depending on an environment
    variable setting; don't count on the file being in cache_dir outside
    of this function.

    """

    def download_ptf_function(filename='PTF201104291667_2_o_45737_11.w.fits'):

        os.makedirs(ptf_cache_dir, exist_ok=True)
        cachedpath = os.path.join(ptf_cache_dir, filename)
        destination = os.path.join(data_dir, filename)
        # url = f'https://portal.nersc.gov/project/m2218/pipeline/test_images/{filename}'
        url = os.path.join(download_url, 'PTF/10cwm', filename)

        if env_as_bool( "LIMIT_CACHE_USAGE" ):
            retry_download( url, destination )
            if not os.path.isfile( destination ):
                raise FileNotFoundError( f"Can't read {destination}.  It should have been downloaded!" )
        else:
            # first make sure file exists in the cache
            if os.path.isfile(cachedpath):
                SCLogger.info(f"{cachedpath} exists, not redownloading.")
            else:
                retry_download(url, cachedpath)  # make the cached copy

            if not os.path.isfile(cachedpath):
                raise FileNotFoundError(f"Can't read {cachedpath}. It should have been downloaded!")

            # copy the PTF exposure from cache to local storage:

            if not os.path.isfile(destination):
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy(cachedpath, destination)

        md5sum = hashlib.md5()
        with open( destination, "rb" ) as ifp:
            md5sum.update( ifp.read() )
        exposure = Exposure( filepath=filename, md5sum=uuid.UUID(md5sum.hexdigest()) )

        return exposure

    return download_ptf_function


@pytest.fixture
def ptf_exposure(ptf_downloader):

    exposure = ptf_downloader()
    exposure.upsert()

    yield exposure

    exposure.delete_from_disk_and_database()

@pytest.fixture
def ptf_datastore_through_cutouts( datastore_factory, ptf_exposure, ptf_ref, ptf_cache_dir, ptf_bad_pixel_map ):
    ptf_exposure.instrument_object.fetch_sections()
    ds = datastore_factory(
        ptf_exposure,
        11,
        cache_dir=ptf_cache_dir,
        cache_base_name='187/PTF_20110429_040004_11_R_Sci_EM7WTT',
        overrides={'extraction': {'threshold': 5}, 'subtraction': {'refset': 'test_refset_ptf'}},
        bad_pixel_map=ptf_bad_pixel_map,
        provtag='ptf_datastore',
        through_step='cutting'
    )

    # Just make sure through_step did what it was supposed to
    assert ds.cutouts is not None
    assert ds.measurements is None

    yield ds

    ds.delete_everything()

    ImageAligner.cleanup_temp_images()

    # Clean out the provenance tag that may have been created by the datastore_factory
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'ptf_datastore' } )
        session.commit()

@pytest.fixture
def ptf_datastore_through_zp( datastore_factory, ptf_exposure, ptf_ref, ptf_cache_dir, ptf_bad_pixel_map ):
    ptf_exposure.instrument_object.fetch_sections()
    ds = datastore_factory(
        ptf_exposure,
        11,
        cache_dir=ptf_cache_dir,
        cache_base_name='187/PTF_20110429_040004_11_R_Sci_LYQY3W',
        overrides={'extraction': {'threshold': 5}, 'subtraction': {'refset': 'test_refset_ptf'}},
        bad_pixel_map=ptf_bad_pixel_map,
        provtag='ptf_datastore',
        through_step='zp'
    )

    # Just make sure through_step did what it was supposed to
    assert ds.zp is not None
    assert ds.sub_image is None

    yield ds

    ds.delete_everything()

    ImageAligner.cleanup_temp_images()

    # Clean out the provenance tag that may have been created by the datastore_factory
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'ptf_datastore' } )
        session.commit()


@pytest.fixture
def ptf_datastore(datastore_factory, ptf_exposure, ptf_ref, ptf_cache_dir, ptf_bad_pixel_map):
    ptf_exposure.instrument_object.fetch_sections()
    ds = datastore_factory(
        ptf_exposure,
        11,
        cache_dir=ptf_cache_dir,
        cache_base_name='187/PTF_20110429_040004_11_R_Sci_LYQY3W',
        overrides={'extraction': {'threshold': 5}, 'subtraction': {'refset': 'test_refset_ptf'}},
        bad_pixel_map=ptf_bad_pixel_map,
        provtag='ptf_datastore'
    )
    yield ds
    ds.delete_everything()

    ImageAligner.cleanup_temp_images()

    # Clean out the provenance tag that may have been created by the datastore_factory
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'ptf_datastore' } )
        session.commit()


@pytest.fixture(scope='session')
def ptf_urls(download_url):
    # base_url = 'https://portal.nersc.gov/project/m2218/pipeline/test_images/'
    base_url = os.path.join(download_url, 'PTF/10cwm')
    r = requests.get(base_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    links = soup.find_all('a')
    filenames = [
        link.get('href') for link in links
        if link.get('href').endswith('.fits') and link.get('href').startswith('PTF')
    ]
    bad_files = [
        'PTF200904053266_2_o_19609_11.w.fits',
        'PTF200904053340_2_o_19614_11.w.fits',
        'PTF201002163703_2_o_18626_11.w.fits',
    ]
    for file in bad_files:
        if file in filenames:
            filenames.pop(filenames.index(file))
    yield filenames


@pytest.fixture(scope='session')
def ptf_images_datastore_factory(ptf_urls, ptf_downloader, datastore_factory, ptf_cache_dir, ptf_bad_pixel_map):

    def factory( start_date='2009-04-04', end_date='2013-03-03',
                 max_images=None, provtag='ptf_images_factory',
                 overrides={'extraction': {'threshold': 5}} ):
        # see if any of the cache names were saved to a manifest file
        cache_names = {}
        if (   ( not env_as_bool( "LIMIT_CACHE_USAGE" ) ) and
               ( os.path.isfile(os.path.join(ptf_cache_dir, 'manifest.txt')) )
            ):
            with open(os.path.join(ptf_cache_dir, 'manifest.txt')) as f:
                text = f.read().splitlines()
            for line in text:
                filename, cache_name = line.split()
                cache_names[filename] = cache_name

        # translate the strings into datetime objects
        start_time = datetime.strptime(start_date, '%Y-%m-%d') if start_date is not None else datetime(1, 1, 1)
        end_time = datetime.strptime(end_date, '%Y-%m-%d') if end_date is not None else datetime(3000, 1, 1)

        # choose only the urls that are within the date range (and no more than max_images)
        urls = []
        for url in ptf_urls:
            if not url.startswith('PTF20'):
                continue
            obstime = datetime.strptime(url[3:11], '%Y%m%d')
            if start_time <= obstime <= end_time:
                urls.append(url)

        # download the images and make a datastore for each one
        dses = []
        for url in urls:
            exp = ptf_downloader(url)
            exp.insert()
            exp.instrument_object.fetch_sections()
            exp.md5sum = uuid.uuid4()  # this will save some memory as the exposures are not saved to archive
            try:
                # produce an image
                ds = datastore_factory(
                    exp,
                    11,
                    cache_dir=ptf_cache_dir,
                    cache_base_name=cache_names.get(url, None),
                    overrides=overrides,
                    bad_pixel_map=ptf_bad_pixel_map,
                    provtag=provtag,
                    skip_sub=True
                )

                if (
                        not env_as_bool( "LIMIT_CACHE_USAGE" ) and
                        hasattr(ds, 'cache_base_name') and ds.cache_base_name is not None
                ):
                    cache_name = ds.cache_base_name
                    if cache_name.startswith(ptf_cache_dir):
                        cache_name = cache_name[len(ptf_cache_dir) + 1:]
                    if cache_name.endswith('.image.fits.json'):
                        cache_name = cache_name[:-len('.image.fits.json')]
                    cache_names[url] = cache_name

                    # save the manifest file (save each iteration in case of failure)
                    with open(os.path.join(ptf_cache_dir, 'manifest.txt'), 'w') as f:
                        for key, value in cache_names.items():
                            f.write(f'{key} {value}\n')

            except Exception as e:
                # I think we should fix this along with issue #150

                # this will also leave behind exposure and image data on disk only
                SCLogger.debug(f'Error processing {url}')
                raise e

                # TODO: should we be worried that some of these images can't complete their processing?
                # SCLogger.debug(e)
                # continue

            dses.append( ds )
            if max_images is not None and len(dses) >= max_images:
                break

        return dses

    return factory


@pytest.fixture(scope='session')
def ptf_reference_image_datastores(ptf_images_datastore_factory):
    dses = ptf_images_datastore_factory('2009-04-05', '2009-05-01', max_images=5, provtag='ptf_reference_images')

    # Sort them by mjd
    dses.sort( key=lambda d: d.image.mjd )

    yield dses

    with SmartSession() as session:
        expsrs = session.query( Exposure ).filter( Exposure._id.in_( [ d.image.exposure_id for d in dses ] ) ).all()

    for ds in dses:
        ds.delete_everything()

    for expsr in expsrs:
        expsr.delete_from_disk_and_database()

    # Clean out the provenance tag that may have been created by the datastore_factory
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'ptf_reference_images' } )
        session.commit()


@pytest.fixture
def ptf_supernova_image_datastores(ptf_images_datastore_factory):
    dses = ptf_images_datastore_factory('2010-02-01', '2013-12-31', max_images=2, provtag='ptf_supernova_images')

    yield dses

    with SmartSession() as session:
        expsrs = session.query( Exposure ).filter( Exposure._id.in_( [ d.image.exposure_id for d in dses ] ) ).all()

    for ds in dses:
        ds.delete_everything()

    for expsr in expsrs:
        expsr.delete_from_disk_and_database()

    # Clean out the provenance tag that may have been created by the datastore_factory
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'ptf_supernova_images' } )
        session.commit()


@pytest.fixture(scope='session')
def ptf_aligned_image_datastores(request, ptf_reference_image_datastores, ptf_cache_dir, data_dir, code_version):
    cache_dir = os.path.join(ptf_cache_dir, 'aligned_images')

    # try to load from cache
    if (    ( not env_as_bool( "LIMIT_CACHE_USAGE" ) ) and
            ( os.path.isfile(os.path.join(cache_dir, 'manifest.txt')) )
        ):

        aligner = ImageAligner( method='swarp' )
        # Going to assume that the upstream provenances are the same for all
        # of the images.  That will be true here by construction... I think.
        ds = ptf_reference_image_datastores[0]
        improv = Provenance.get( ds.image.provenance_id )
        srcprov = Provenance.get( ds.sources.provenance_id )
        warped_prov, warped_sources_prov = aligner.get_provenances( [improv, srcprov], srcprov )

        with open(os.path.join(cache_dir, 'manifest.txt')) as f:
            filenames = f.read().splitlines()
        output_dses = []
        for filename in filenames:
            imfile, sourcesfile, bgfile, psffile, wcsfile = filename.split()
            image = copy_from_cache( Image, cache_dir, imfile + '.image.fits' )
            image.provenance_id = warped_prov.id
            ds = DataStore( image )
            ds.sources = copy_from_cache( SourceList, cache_dir, sourcesfile )
            ds.sources.provenance_id = warped_sources_prov.id
            ds.bg = copy_from_cache( Background, cache_dir, bgfile, add_to_dict={ 'image_shape': ds.image.data.shape } )
            ds.psf = copy_from_cache( PSF, cache_dir, psffile + '.fits' )
            ds.wcs = copy_from_cache( WorldCoordinates, cache_dir, wcsfile )
            ds.zp = copy_from_cache( ZeroPoint, cache_dir, imfile + '.zp' )

            output_dses.append( ds )

    else:
        # no cache available, must regenerate

        # ref: https://stackoverflow.com/a/75337251
        # ptf_reference_image_datastores = request.getfixturevalue('ptf_reference_image_datastores')

        coadder = Coadder( alignment_index='last', alignment={ 'method': 'swarp' } )
        coadder.run_alignment( ptf_reference_image_datastores, len(ptf_reference_image_datastores)-1 )

        for ds in coadder.aligned_datastores:
            ds.image.save( overwrite=True )
            ds.sources.save( image=ds.image, overwrite=True )
            ds.bg.save( image=ds.image, sources=ds.sources, overwrite=True )
            ds.psf.save( image=ds.image, sources=ds.sources, overwrite=True )
            ds.wcs.save( image=ds.image, sources=ds.sources, overwrite=True )

            if not env_as_bool( "LIMIT_CACHE_USAGE" ):
                copy_to_cache( ds.image, cache_dir )
                copy_to_cache( ds.sources, cache_dir )
                copy_to_cache( ds.bg, cache_dir )
                copy_to_cache( ds.psf, cache_dir )
                copy_to_cache( ds.wcs, cache_dir )
                copy_to_cache( ds.zp, cache_dir, filepath=ds.image.filepath+'.zp.json' )

        if not env_as_bool( "LIMIT_CACHE_USAGE" ):
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, 'manifest.txt'), 'w') as f:
                for ds in coadder.aligned_datastores:
                    f.write( f'{ds.image.filepath} {ds.sources.filepath} {ds.bg.filepath} '
                             f'{ds.psf.filepath} {ds.wcs.filepath}\n' )

        output_dses = coadder.aligned_datastores

    yield output_dses

    for ds in output_dses:
        ds.delete_everything()


@pytest.fixture
def ptf_ref(
        refmaker_factory,
        ptf_reference_image_datastores,
        ptf_aligned_image_datastores,
        ptf_cache_dir,
        data_dir,
        code_version
):
    refmaker = refmaker_factory('test_ref_ptf', 'PTF', provtag='ptf_ref')
    pipe = refmaker.coadd_pipeline

    ds0 = ptf_reference_image_datastores[0]
    origimprov = Provenance.get( ds0.image.provenance_id )
    origsrcprov = Provenance.get( ds0.sources.provenance_id )
    upstream_provs = [ origimprov, origsrcprov ]
    im_prov = Provenance(
        process='coaddition',
        parameters=pipe.coadder.pars.get_critical_pars(),
        upstreams=upstream_provs,
        code_version_id=code_version.id,
        is_testing=True,
    )
    im_prov.insert_if_needed()

    # Copying code from Image.invent_filepath so that
    #   we know what the filenames will be
    utag = hashlib.sha256()
    for id in [ d.image.id for d in ptf_reference_image_datastores ]:
        utag.update( str(id).encode('utf-8') )
    utag = base64.b32encode(utag.digest()).decode().lower()
    utag = f'u-{utag[:6]}'

    cache_base_name = f'187/PTF_20090405_073932_11_R_ComSci_{im_prov.id[:6]}_{utag}'

    # this provenance is used for sources, psf, wcs, zp
    sources_prov = Provenance(
        process='extraction',
        parameters=pipe.extractor.pars.get_critical_pars(),
        upstreams=[ im_prov ],
        code_version_id=code_version.id,
        is_testing=True,
    )
    sources_prov.insert_if_needed()

    extensions = [
        'image.fits',
        f'sources_{sources_prov.id[:6]}.fits',
        f'psf_{sources_prov.id[:6]}.fits',
        f'bg_{sources_prov.id[:6]}.h5',
        f'wcs_{sources_prov.id[:6]}.txt',
        'zp'
    ]
    filenames = [os.path.join(ptf_cache_dir, cache_base_name) + f'.{ext}.json' for ext in extensions]

    if not env_as_bool( "LIMIT_CACHE_USAGE" ) and all( [ os.path.isfile(filename) for filename in filenames ] ):
        # can load from cache

        # get the image:
        coadd_image = copy_from_cache(Image, ptf_cache_dir, cache_base_name + '.image.fits')
        # We're supposed to load this property by running Image.from_images(), but directly
        # access the underscore variable here as a hack since we loaded from the cache.
        coadd_image._upstream_ids = [ d.image.id for d in ptf_reference_image_datastores ]
        coadd_image.provenance_id = im_prov.id
        coadd_image.ref_image_id = ptf_reference_image_datastores[-1].image.id

        coadd_datastore = DataStore( coadd_image )

        # get the source list:
        coadd_datastore.sources = copy_from_cache(
            SourceList, ptf_cache_dir, cache_base_name + f'.sources_{sources_prov.id[:6]}.fits'
        )
        coadd_datastore.sources.image_id = coadd_image.id
        coadd_datastore.sources.provenance_id = sources_prov.id

        # get the PSF:
        coadd_datastore.psf = copy_from_cache( PSF, ptf_cache_dir,
                                               cache_base_name + f'.psf_{sources_prov.id[:6]}.fits' )
        coadd_datastore.psf.sources_id = coadd_datastore.sources.id

        # get the background:
        coadd_datastore.bg = copy_from_cache( Background, ptf_cache_dir,
                                              cache_base_name + f'.bg_{sources_prov.id[:6]}.h5',
                                              add_to_dict={ 'image_shape': coadd_datastore.image.data.shape } )
        coadd_datastore.bg.sources_id = coadd_datastore.sources.id

        # get the WCS:
        coadd_datastore.wcs = copy_from_cache( WorldCoordinates, ptf_cache_dir,
                                               cache_base_name + f'.wcs_{sources_prov.id[:6]}.txt' )
        coadd_datastore.wcs.sources_id = coadd_datastore.sources.id

        # get the zero point:
        coadd_datastore.zp = copy_from_cache( ZeroPoint, ptf_cache_dir, cache_base_name + '.zp' )
        coadd_datastore.zp.sources_id = coadd_datastore.sources.id

        # Make sure it's all in the database
        coadd_datastore.save_and_commit()

    else:  # make a new reference image

        coadd_datastore = pipe.run( ptf_reference_image_datastores, aligned_datastores=ptf_aligned_image_datastores )
        coadd_datastore.save_and_commit()

        # Check that the filename came out what we expected above
        mtch = re.search( r'_([a-zA-Z0-9\-]+)$', coadd_datastore.image.filepath )
        if mtch.group(1) != utag:
            raise ValueError( f"fixture cache error: filepath utag is {mtch.group(1)}, expected {utag}" )

        if not env_as_bool( "LIMIT_CACHE_USAGE" ):
            # save all products into cache:
            copy_to_cache(coadd_datastore.image, ptf_cache_dir)
            copy_to_cache(coadd_datastore.sources, ptf_cache_dir)
            copy_to_cache(coadd_datastore.psf, ptf_cache_dir)
            copy_to_cache(coadd_datastore.bg, ptf_cache_dir)
            copy_to_cache(coadd_datastore.wcs, ptf_cache_dir)
            copy_to_cache(coadd_datastore.zp, ptf_cache_dir, cache_base_name + '.zp.json')

    parms = dict( refmaker.pars.get_critical_pars() )
    parms[ 'test_parameter' ] = 'test_value'
    refprov = Provenance(
        code_version_id=code_version.id,
        process='referencing',
        parameters=parms,
        upstreams = [ im_prov, sources_prov ],
        is_testing=True
    )
    refprov.insert_if_needed()
    ref = Reference(
        image_id=coadd_datastore.image.id,
        target=coadd_datastore.image.target,
        instrument=coadd_datastore.image.instrument,
        filter=coadd_datastore.image.filter,
        section_id=coadd_datastore.image.section_id,
        provenance_id=refprov.id
    )
    ref.provenance_id=refprov.id
    ref.insert()

    # Since we didn't actually run the RefMaker we got from refmaker_factory, we may still need
    #   to create the reference set and tag the reference we built.
    # (Not bothering with locking here because we know our tests are single-threaded.)
    must_delete_refset = False
    with SmartSession() as sess:
        refset = RefSet.get_by_name( 'test_refset_ptf' )
        if refset is None:
            refset = RefSet( name='test_refset_ptf' )
            refset.insert()
            must_delete_refset = True
        refset.append_provenance( refprov )

    yield ref

    coadd_datastore.delete_everything()

    with SmartSession() as session:
        ref_in_db = session.scalars(sa.select(Reference).where(Reference._id == ref.id)).first()
        assert ref_in_db is None  # should have been deleted by cascade when image is deleted

        # Clean up the ref set
        if must_delete_refset:
            session.execute( sa.delete( RefSet ).where( RefSet._id==refset.id ) )
            session.commit()


@pytest.fixture
def ptf_ref_offset(ptf_ref):
    ptf_ref_image = Image.get_by_id( ptf_ref.image_id )
    offset_image = Image.copy_image( ptf_ref_image )
    offset_image.ra_corner_00 -= 0.5
    offset_image.ra_corner_01 -= 0.5
    offset_image.ra_corner_10 -= 0.5
    offset_image.ra_corner_11 -= 0.5
    offset_image.minra -= 0.5
    offset_image.maxra -= 0.5
    offset_image.ra -= 0.5
    offset_image.filepath = ptf_ref_image.filepath + '_offset'
    offset_image.provenance_id = ptf_ref_image.provenance_id
    offset_image.md5sum = uuid.uuid4()  # spoof this so we don't have to save to archive

    new_ref = Reference( target=ptf_ref.target,
                         filter=ptf_ref.filter,
                         instrument=ptf_ref.instrument,
                         section_id=ptf_ref.section_id,
                         image_id=offset_image.id
                        )
    refprov = Provenance.get( ptf_ref.provenance_id )
    pars = refprov.parameters.copy()
    pars['test_parameter'] = uuid.uuid4().hex
    refprov = Provenance.get( ptf_ref.provenance_id )
    prov = Provenance(
        process='referencing',
        parameters=pars,
        upstreams=refprov.upstreams,
        code_version_id=refprov.code_version_id,
        is_testing=True,
    )
    prov.insert_if_needed()
    new_ref.provenance_id = prov.id

    offset_image.insert()
    new_ref.insert()

    yield new_ref

    offset_image.delete_from_disk_and_database()
    # (Database cascade will also delete new_ref)


@pytest.fixture(scope='session')
def ptf_refset(refmaker_factory):
    refmaker = refmaker_factory('test_refset_ptf', 'PTF', 'ptf_refset')
    refmaker.pars.save_new_refs = True

    refmaker.make_refset()  # this makes a refset without making any references

    yield refmaker.refset

    # delete all the references and the refset
    with SmartSession() as session:
        for prov in refmaker.refset.provenances:
            refs = session.scalars(sa.select(Reference).where(Reference.provenance_id == prov.id)).all()
            for ref in refs:
                session.delete(ref)

        session.execute( sa.delete( RefSet ).where( RefSet.name == refmaker.refset.name ) )

        session.commit()

    # Clean out the provenance tag that may have been created by the refmaker_factory
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'ptf_refset' } )
        session.commit()


@pytest.fixture
def ptf_subtraction1_datastore( ptf_ref, ptf_supernova_image_datastores, subtractor, ptf_cache_dir, code_version ):
    subtractor.pars.refset = 'test_refset_ptf'
    ds = ptf_supernova_image_datastores[0]
    ds.set_prov_tree( { 'referencing': Provenance.get( ptf_ref.provenance_id ) } )
    prov = ds.get_provenance( 'subtraction', pars_dict=subtractor.pars.get_critical_pars(), replace_tree=True )
    cache_path = os.path.join(
        ptf_cache_dir,
        f'187/PTF_20100216_075004_11_R_Diff_{prov.id[:6]}_u-iig7a2.image.fits.json'
    )

    if ( not env_as_bool( "LIMIT_CACHE_USAGE" ) ) and ( os.path.isfile(cache_path) ):  # try to load this from cache
        im = copy_from_cache( Image, ptf_cache_dir, cache_path )
        refim = Image.get_by_id( ptf_ref.image_id )
        im._upstream_ids = [ refim.id, ptf_supernova_images[0].id ]
        im.ref_image_id = ptf_ref.image.id
        im.provenance_id = prov.id
        ds.sub_image = im
        ds.sub_image.insert()

    else:  # cannot find it on cache, need to produce it, using other fixtures
        ds = subtractor.run( ptf_supernova_image_datastores[0] )
        ds.sub_image.save()
        ds.sub_image.insert()

        if not env_as_bool( "LIMIT_CACHE_USAGE" ) :
            copy_to_cache(ds.sub_image, ptf_cache_dir)

    yield ds

    # Don't have to clean up, everything we have done will be cleaned up by cascade.
    # (Except for the provenance, but we don't demand those be cleaned up.)
