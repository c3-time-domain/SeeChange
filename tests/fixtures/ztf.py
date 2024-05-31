import os
import pytest
import pathlib
import shutil
import io


from astropy.io import fits, votable

from models.base import SmartSession, FileOnDiskMixin
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF

from pipeline.data_store import DataStore
from pipeline.catalog_tools import fetch_gaia_dr3_excerpt

from util.retrydownload import retry_download


@pytest.fixture(scope='session')
def ztf_cache_dir(cache_dir):
    output = os.path.join(cache_dir, 'ZTF')
    if not os.path.isdir(output):
        os.makedirs(output)

    yield output


@pytest.fixture
def ztf_filepaths_image_sources_psf(data_dir, ztf_cache_dir, download_url):
    download_url = download_url + '/ZTF/'

    image = "test_ztf_image.fits"
    weight = "test_ztf_image.weight.fits"
    flags = "test_ztf_image.flags.fits"
    sources = "test_ztf_image.sources.fits"
    psf = "test_ztf_image.psf"
    psfxml = "test_ztf_image.psf.xml"
    output = image, weight, flags, sources, psf, psfxml

    for filepath in output:
        if os.getenv( "LIMIT_CACHE_USAGE" ):
            retry_download( download_url + filepath, os.path.join( data_dir, filepath ) )
        else:
            if not os.path.isfile(os.path.join(ztf_cache_dir, filepath)):
                retry_download(download_url + filepath, os.path.join(ztf_cache_dir, filepath))
            if not os.path.isfile(os.path.join(ztf_cache_dir, filepath)):
                raise FileNotFoundError(f"Can't read {filepath}. It should have been downloaded! ")
            if not os.path.isfile(os.path.join(data_dir, filepath)):
                shutil.copy2(os.path.join(ztf_cache_dir, filepath), os.path.join(data_dir, filepath))

    output = tuple(
        pathlib.Path(os.path.join(data_dir, filepath)) for filepath in output
    )
    yield output

    for filepath in output:
        if os.path.isfile(filepath):
            os.remove(filepath)


@pytest.fixture
def ztf_datastore_uncommitted( ztf_filepaths_image_sources_psf ):
    image, weight, flags, sources, psf, psfxml = ztf_filepaths_image_sources_psf
    ds = DataStore()

    ds.image = Image( filepath=str( image.relative_to( FileOnDiskMixin.local_path ) ), format='fits' )
    with fits.open( image ) as hdul:
        ds.image._data = hdul[0].data
        ds.image._header = hdul[0].header
    with fits.open( weight ) as hdul:
        ds.image._weight = hdul[0].data
    with fits.open( flags ) as hdul:
        ds.image._flags = hdul[0].data
    ds.image.set_corners_from_header_wcs()
    ds.image.ra = ( ds.image.ra_corner_00 + ds.image.ra_corner_01 +
                    ds.image.ra_corner_10 + ds.image.ra_corner_11 ) / 4.
    ds.image.dec = ( ds.image.dec_corner_00 + ds.image.dec_corner_01 +
                     ds.image.dec_corner_00 + ds.image.dec_corner_11 ) / 4.
    ds.image.calculate_coordinates()
    ds.image.instrument = 'ZTF'
    ds.image.telescope = 'P48'
    ds.image.project = 'Program 1'
    ds.image.section_id = ds.image.header['CCD_ID']
    ds.image.target = f'field {ds.image.header["FIELDID"]}'
    ds.image.exp_time = ds.image.header['EXPTIME']
    ds.image.mjd = ds.image.header['OBSMJD']
    ds.image.end_mjd = ds.image.mjd + ds.image.header['EXPTIME'] / 86400.
    ds.image.filter = ds.image.header['FILTER']

    ds.sources = SourceList( filepath=str( sources.relative_to( FileOnDiskMixin.local_path ) ), format='sextrfits' )
    ds.sources.load( sources )
    ds.sources.num_sources = len( ds.sources.data )
    ds.sources.image = ds.image

    ds.psf = PSF( filepath=str( psf.relative_to( FileOnDiskMixin.local_path ) ), format='psfex' )
    ds.psf.load( download=False, psfpath=psf, psfxmlpath=psfxml )
    bio = io.BytesIO( ds.psf.info.encode( 'utf-8' ) )
    tab = votable.parse( bio ).get_table_by_index( 1 )
    ds.psf.fwhm_pixels = float( tab.array['FWHM_FromFluxRadius_Mean'][0] )
    ds.psf.image = ds.image

    yield ds

    ds.delete_everything()


@pytest.fixture
def ztf_filepath_sources( ztf_filepaths_image_sources_psf ):
    image, weight, flags, sources, psf, psfxml = ztf_filepaths_image_sources_psf
    return sources


@pytest.fixture
def ztf_gaia_dr3_excerpt( ztf_datastore_uncommitted ):
    ds = ztf_datastore_uncommitted
    catexp = fetch_gaia_dr3_excerpt( ds.image, minstars=50, maxmags=20, magrange=4 )
    assert catexp is not None

    yield catexp

    with SmartSession() as session:
        catexp = session.merge(catexp)
        catexp.delete_from_disk_and_database( session=session )
