import pathlib
import pytest

import numpy as np
import sqlalchemy as sa

from astropy.wcs import WCS

from util.exceptions import CatalogNotFoundError
from models.base import SmartSession, FileOnDiskMixin
from models.catalog_excerpt import CatalogExcerpt
from pipeline.astro_cal import AstroCalibrator

@pytest.fixture( scope='module' )
def astrometor():
    return AstroCalibrator( catalog='GaiaDR3' )

@pytest.fixture
def gaiadr3_excerpt( astrometor, example_ds_with_sources_and_psf ):
    ds = example_ds_with_sources_and_psf
    catexp = astrometor.secure_GaiaDR3_excerpt( ds.image, maxmags=(20.,), magrange=4., numstars=50 )
    assert catexp is not None
    
    yield catexp

    with SmartSession() as session:
        catexp = catexp.recursive_merge( session )
        catexp.delete_from_disk_and_database( session=session )
    
def test_download_GaiaDR3( astrometor ):
    firstfilepath = None
    secondfilepath = None
    basepath = pathlib.Path( FileOnDiskMixin.local_path )
    try:
        catexp, firstfilepath, dbfile = astrometor.download_GaiaDR3( 150.9427, 151.2425, 1.75582, 1.90649,
                                                                     padding=0.1, minmag=18., maxmag=22. )
        assert firstfilepath == str( basepath / 'GaiaDR3_excerpt/94/Gaia_DR3_151.0926_1.8312_18.0_22.0.fits' )
        assert dbfile == firstfilepath
        assert catexp.num_items == 178
        assert catexp.format == 'fitsldac'
        assert catexp.origin == 'GaiaDR3'
        assert catexp.minmag == 18.
        assert catexp.maxmag == 22.
        assert ( catexp.dec_corner_11 - catexp.dec_corner_00 ) == pytest.approx( 1.2 * (1.90649-1.75582), abs=1e-4 )
        catexp, secondfilepath, dbfile = astrometor.download_GaiaDR3( 150.9427, 151.2425, 1.75582, 1.90649,
                                                                      padding=0.1, minmag=17., maxmag=19. )
        assert secondfilepath == str( basepath / 'GaiaDR3_excerpt/94/Gaia_DR3_151.0926_1.8312_17.0_19.0.fits' )
        assert dbfile == secondfilepath
        assert catexp.num_items == 59
        assert catexp.minmag == 17.
        assert catexp.maxmag == 19.
    finally:
        if firstfilepath is not None:
            pathlib.Path( firstfilepath ).unlink( missing_ok=True )
        if secondfilepath is not None:
            pathlib.Path( secondfilepath ).unlink( missing_ok=True )

def test_gaiadr3_excerpt( astrometor, gaiadr3_excerpt, example_ds_with_sources_and_psf ):
    catexp = gaiadr3_excerpt
    ds = example_ds_with_sources_and_psf

    assert catexp.num_items == 172
    assert catexp.num_items == len( catexp.data )
    assert catexp.filepath == 'GaiaDR3_excerpt/30/Gaia_DR3_153.6459_39.0937_16.0_20.0.fits'
    assert pathlib.Path( catexp.get_fullpath() ).is_file()
    assert catexp.object_ras.min() == pytest.approx( 153.413563, abs=0.1/3600. )
    assert catexp.object_ras.max() == pytest.approx( 153.877110, abs=0.1/3600. )
    assert catexp.object_decs.min() == pytest.approx( 38.914110, abs=0.1/3600. )
    assert catexp.object_decs.max() == pytest.approx( 39.274596, abs=0.1/3600. )
    assert ( catexp.data['X_WORLD'] == catexp.object_ras ).all()
    assert ( catexp.data['Y_WORLD'] == catexp.object_decs ).all()
    assert catexp.data['MAG_G'].min() == pytest.approx( 16.076, abs=0.001 )
    assert catexp.data['MAG_G'].max() == pytest.approx( 19.994, abs=0.001 )
    assert catexp.data['MAGERR_G'].min() == pytest.approx( 0.0004, abs=0.0001 )
    assert catexp.data['MAGERR_G'].max() == pytest.approx( 0.018, abs=0.001 )
    
    # Test reading of cache
    newcatexp = astrometor.secure_GaiaDR3_excerpt( ds.image, maxmags=(20.,), magrange=4., numstars=50,
                                                   onlycached=True )
    assert newcatexp.id == catexp.id
    
    # Make sure we can't read the cache for something that doesn't exist
    with pytest.raises( CatalogNotFoundError, match='Failed to secure Gaia DR3 stars' ):
        newcatexp = astrometor.secure_GaiaDR3_excerpt( ds.image, maxmags=(22.,), magrange=4., numstars=50,
                                                       onlycached=True )

def test_solve_wcs_scamp( astrometor, gaiadr3_excerpt, example_ds_with_sources_and_psf ):
    catexp = gaiadr3_excerpt
    ds = example_ds_with_sources_and_psf

    orighdr = ds.image._raw_header.copy()

    catexp.ds9_regfile( 'catexp.reg', radius=4 )
    ds.sources.ds9_regfile( 'sources.reg', radius=3 )
    
    astrometor.solve_wcs_scamp( ds.image, ds.sources, catexp )

    # Because this was a ZTF image that had a WCS already, the new WCS
    # should be damn close, but not identical (since there's no way we
    # used exactly the same set of sources and stars, plus this was a
    # cropped ZTF image, not the full image).
    allsame = True
    for i in [ 1, 2 ]:
        for j in range( 17 ):
            Δ = np.abs( ( orighdr[f'PV{i}_{j}'] - ds.image._raw_header[f'PV{i}_{j}'] ) / orighdr[f'PV{i}_{j}'] )
            if Δ > 1e-6:
                allsame = False
                break
    assert not allsame

    #...but check that they are close
    wcsold = WCS( orighdr )
    scolds = wcsold.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    wcsnew = WCS( ds.image._raw_header )
    scnews = wcsnew.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    for scold, scnew in zip( scolds, scnews ):
        assert scold.ra.value == pytest.approx( scnew.ra.value, abs=1./3600. )
        assert scold.dec.value == pytest.approx( scnew.dec.value, abs=1./3600. )
    
    import pdb; pdb.set_trace()
    pass
