import pytest
import os
import shutil
import requests

import numpy as np

import sqlalchemy as sa
from bs4 import BeautifulSoup
from datetime import datetime
from astropy.io import fits

from models.base import SmartSession, _logger
from models.ptf import PTF  # need this import to make sure PTF is added to the Instrument list
from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.psf import PSF
from models.zero_point import ZeroPoint

from util.retrydownload import retry_download


@pytest.fixture(scope='session')
def ptf_bad_pixel_map(data_dir, cache_dir):
    cache_dir = os.path.join(cache_dir, 'PTF')
    filename = 'C11/masktot.fits'  # TODO: add more CCDs if needed
    url = 'https://portal.nersc.gov/project/m2218/pipeline/test_images/2012021x/'

    # is this file already on the cache? if not, download it
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.isfile(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        retry_download(url + filename, cache_path)

    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Can't read {cache_path}. It should have been downloaded!")

    data_dir = os.path.join(data_dir, 'PTF_calibrators')
    data_path = os.path.join(data_dir, filename)
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
def ptf_downloader(provenance_preprocessing, data_dir, cache_dir):
    cache_dir = os.path.join(cache_dir, 'PTF')

    def download_ptf_function(filename='PTF201104291667_2_o_45737_11.w.fits'):

        os.makedirs(cache_dir, exist_ok=True)
        cachedpath = os.path.join(cache_dir, filename)

        # first make sure file exists in the cache
        if os.path.isfile(cachedpath):
            _logger.info(f"{cachedpath} exists, not redownloading.")
        else:
            url = f'https://portal.nersc.gov/project/m2218/pipeline/test_images/{filename}'
            retry_download(url, cachedpath)  # make the cached copy

        if not os.path.isfile(cachedpath):
            raise FileNotFoundError(f"Can't read {cachedpath}. It should have been downloaded!")

        # copy the PTF exposure from cache to local storage:
        destination = os.path.join(data_dir, filename)

        if not os.path.isfile(destination):
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy(cachedpath, destination)

        exposure = Exposure(filepath=filename)

        return exposure

    return download_ptf_function


@pytest.fixture
def ptf_exposure(ptf_downloader):

    exposure = ptf_downloader()
    # check if this Exposure is already on the database
    with SmartSession() as session:
        existing = session.scalars(sa.select(Exposure).where(Exposure.filepath == exposure.filepath)).first()
        if existing is not None:
            _logger.info(f"Found existing Image on database: {existing}")
            # overwrite the existing row data using the JSON cache file
            for key in sa.inspect(exposure).mapper.columns.keys():
                value = getattr(exposure, key)
                if (
                        key not in ['id', 'image_id', 'created_at', 'modified'] and
                        value is not None
                ):
                    setattr(existing, key, value)
            exposure = existing  # replace with the existing row
        else:
            exposure = session.merge(exposure)
            exposure.save()  # make sure it is up on the archive as well
            session.add(exposure)
            session.commit()

    yield exposure

    exposure.delete_from_disk_and_database()


@pytest.fixture
def ptf_datastore(datastore_factory, ptf_exposure, cache_dir, ptf_bad_pixel_map):
    cache_dir = os.path.join(cache_dir, 'PTF')
    ptf_exposure.instrument_object.fetch_sections()
    ds = datastore_factory(
        ptf_exposure,
        11,
        cache_dir=cache_dir,
        cache_base_name='187/PTF_20110429_040004_11_R_Sci_5F5TAU',
        overrides={'extraction': {'threshold': 5}},
        bad_pixel_map=ptf_bad_pixel_map,
    )
    yield ds
    ds.delete_everything()


@pytest.fixture(scope='session')
def ptf_urls():
    base_url = f'https://portal.nersc.gov/project/m2218/pipeline/test_images/'
    r = requests.get(base_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    links = soup.find_all('a')
    filenames = [link.get('href') for link in links if link.get('href').endswith('.fits')]

    bad_files = ['PTF200904053266_2_o_19609_11.w.fits', 'PTF200904053340_2_o_19614_11.w.fits']
    for file in bad_files:
        if file in filenames:
            filenames.pop(filenames.index(file))
    yield filenames


@pytest.fixture(scope='session')
def ptf_images_factory(ptf_urls, ptf_downloader, datastore_factory, cache_dir, ptf_bad_pixel_map):
    cache_dir = os.path.join(cache_dir, 'PTF')

    def factory(start_date='2009-04-04', end_date='2013-03-03', max_images=None):
        # see if any of the cache names were saved to a manifest file
        cache_names = {}
        if os.path.isfile(os.path.join(cache_dir, 'manifest.txt')):
            with open(os.path.join(cache_dir, 'manifest.txt')) as f:
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
            obstime = datetime.strptime(url[3:11], '%Y%m%d')
            if start_time <= obstime <= end_time:
                urls.append(url)

        # download the images and make a datastore for each one
        images = []
        for url in urls:
            exp = ptf_downloader(url)
            exp.instrument_object.fetch_sections()
            try:
                # produce an image
                ds = datastore_factory(
                    exp,
                    11,
                    cache_dir=cache_dir,
                    cache_base_name=cache_names.get(url, None),
                    overrides={'extraction': {'threshold': 5}},
                    bad_pixel_map=ptf_bad_pixel_map,
                )

                if hasattr(ds, 'cache_base_name') and ds.cache_base_name is not None:
                    cache_name = ds.cache_base_name
                    if cache_name.startswith(cache_dir):
                        cache_name = cache_name[len(cache_dir) + 1:]
                    if cache_name.endswith('.image.fits.json'):
                        cache_name = cache_name[:-len('.image.fits.json')]
                    cache_names[url] = cache_name

                    # save the manifest file (save each iteration in case of failure)
                    with open(os.path.join(cache_dir, 'manifest.txt'), 'w') as f:
                        for key, value in cache_names.items():
                            f.write(f'{key} {value}\n')

            except Exception as e:
                raise e
                # I think we should fix this along with issue #150
                print(f'Error processing {url}')  # this will also leave behind exposure and image data on disk only
                # print(e)  # TODO: should we be worried that some of these images can't complete their processing?
                continue
            images.append(ds.image)
            if max_images is not None and len(images) >= max_images:
                break

        return images

    return factory


@pytest.fixture(scope='session')
def ptf_reference_images(ptf_images_factory):
    images = ptf_images_factory('2009-04-05', '2009-05-01', max_images=10)

    yield images

    with SmartSession() as session:
        for image in images:
            image = image.recursive_merge(session)
            image.exposure.delete_from_disk_and_database(session=session, commit=False)
            image.delete_from_disk_and_database(session=session, commit=False, remove_downstream_data=True)
        session.commit()


# conditionally call the ptf_reference_images fixture if cache is not there:
# ref: https://stackoverflow.com/a/75337251
@pytest.fixture(scope='session')
def ptf_aligned_images(request, cache_dir, data_dir, code_version):
    cache_dir = os.path.join(cache_dir, 'PTF/aligned_images')

    # try to load from cache
    if os.path.isfile(os.path.join(cache_dir, 'manifest.txt')):
        with open(os.path.join(cache_dir, 'manifest.txt')) as f:
            filenames = f.read().splitlines()
        output_images = []
        for filename in filenames:
            output_images.append(Image.copy_from_cache(cache_dir, filename + '.image.fits'))
            output_images[-1].psf = PSF.copy_from_cache(cache_dir, filename + '.psf')
            output_images[-1].zp = ZeroPoint.copy_from_cache(cache_dir, filename + '.zp')
    else:  # no cache available
        ptf_reference_images = request.getfixturevalue('ptf_reference_images')
        images_to_align = ptf_reference_images[:9]  # speed things up using fewer images
        prov = Provenance(
            code_version=code_version,
            parameters={'alignment': {'method': 'swarp', 'to_index': 'last'}, 'test_parameter': 'test_value'},
            upstreams=[],
            process='coaddition',
            is_testing=True,
        )
        prov.update_id()
        new_image = Image.from_images(images_to_align, index=-1)
        new_image.provenance = prov
        new_image.provenance.upstreams = new_image.get_upstream_provenances()

        filenames = []
        for image in new_image.aligned_images:
            image.save()
            filepath = image.copy_to_cache(cache_dir)
            image.psf.copy_to_cache(cache_dir, filepath=filepath[:-len('.image.fits.json')])
            image.zp.copy_to_cache(cache_dir, filepath=filepath[:-len('.image.fits.json')]+'.zp.json')
            filenames.append(image.filepath)

        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, 'manifest.txt'), 'w') as f:
            for filename in filenames:
                f.write(f'{filename}\n')
        output_images = new_image.aligned_images

    yield output_images

    if 'output_images' in locals():
        for image in output_images:
            image.psf.delete_from_disk_and_database()
            image.delete_from_disk_and_database()

    if 'new_image' in locals():
        new_image.delete_from_disk_and_database()
