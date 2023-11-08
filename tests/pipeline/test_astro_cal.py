import pathlib
import shutil
import pytest
import hashlib
import uuid

import numpy as np
import sqlalchemy as sa

from astropy.wcs import WCS
from astropy.io import fits

from util.exceptions import CatalogNotFoundError, BadMatchException
from models.base import SmartSession, FileOnDiskMixin
from models.catalog_excerpt import CatalogExcerpt
from models.image import Image
from models.world_coordinates import WorldCoordinates
from pipeline.astro_cal import AstroCalibrator

@pytest.fixture
def gaiadr3_excerpt( example_ds_with_sources_and_psf ):
    ds = example_ds_with_sources_and_psf
    astrometor = AstroCalibrator( catalog='GaiaDR3', max_mag=[20.], mag_range=4., min_stars=50 )
    catexp = astrometor.fetch_GaiaDR3_excerpt( ds.image )
    assert catexp is not None

    yield catexp

    with SmartSession() as session:
        catexp = catexp.recursive_merge( session )
        catexp.delete_from_disk_and_database( session=session )

def test_download_GaiaDR3():
    firstfilepath = None
    secondfilepath = None
    basepath = pathlib.Path( FileOnDiskMixin.local_path )
    try:
        astrometor = AstroCalibrator( catalog='GaiaDR3' )
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

def test_gaiadr3_excerpt_failures( example_ds_with_sources_and_psf, gaiadr3_excerpt ):
    ds = example_ds_with_sources_and_psf

    # Make sure it fails if we give it a ridiculous max mag
    with pytest.raises( CatalogNotFoundError, match="Failed to fetch Gaia DR3 stars at" ):
        astrometor = AstroCalibrator( catalog='GaiaDR3', max_mag=[5.], mag_range=4., min_stars=50 )
        catexp = astrometor.fetch_GaiaDR3_excerpt( ds.image )

    # ...but make sure it succeeds if we also give it a reasonable max mag
    astrometor = AstroCalibrator( catalog='GaiaDR3', max_mag=[5., 20.], mag_range=4., min_stars=50 )
    catexp = astrometor.fetch_GaiaDR3_excerpt( ds.image )
    assert catexp.id == gaiadr3_excerpt.id

    # Make sure it fails if we ask for too many stars
    with pytest.raises( CatalogNotFoundError, match="Failed to fetch Gaia DR3 stars at" ):
        astrometor = AstroCalibrator( catalog='GaiaDR3', max_mag=[20.], mag_range=4., min_stars=50000 )
        catexp = astrometor.fetch_GaiaDR3_excerpt( ds.image )

    # Make sure it fails if mag range is too small
    with pytest.raises( CatalogNotFoundError, match="Failed to fetch Gaia DR3 stars at" ):
        astrometor = AstroCalibrator( catalog='GaiaDR3', max_mag=[20.], mag_range=0.01, min_stars=5 )
        catexp = astrometor.fetch_GaiaDR3_excerpt( ds.image )


def test_gaiadr3_excerpt( gaiadr3_excerpt, example_ds_with_sources_and_psf ):
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
    astrometor = AstroCalibrator( catalog='GaiaDR3', max_mag=[20.], mag_range=4., min_stars=50 )
    newcatexp = astrometor.fetch_GaiaDR3_excerpt( ds.image, onlycached=True )
    assert newcatexp.id == catexp.id

    # Make sure we can't read the cache for something that doesn't exist
    with pytest.raises( CatalogNotFoundError, match='Failed to fetch Gaia DR3 stars' ):
        astrometor = AstroCalibrator( catalog='GaiaDR3', max_mag=[20.5], mag_range=4., min_stars=50 )
        newcatexp = astrometor.fetch_GaiaDR3_excerpt( ds.image, onlycached=True )


def test_solve_wcs_scamp_failures( gaiadr3_excerpt, example_ds_with_sources_and_psf ):
    catexp = gaiadr3_excerpt
    ds = example_ds_with_sources_and_psf

    # Make sure it fails if we give too stringent of a minimum residual
    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[20.], mag_range=4., min_stars=50,
                                  max_resid=0.01 )
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    # Make sure it fails if we give it too small of a crossid radius.
    # Note that this one is passed directly to _solve_wcs_scamp.
    # _solve_wcs_scamp doesn't read what we pass to AstroCalibrator
    # constructor, because that is an array of crossid_rad values to
    # try, whereas _solve_wcs_scamp needs a single value.  (The
    # iteration happens outsice _solve_wcs_scamp.)
    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[20.], mag_range=4., min_stars=50 )
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp, crossid_rad=0.01 )

    # Make sure it fails if min_frac_matched is too high
    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[20.], mag_range=4., min_stars=50,
                                  min_frac_matched=0.8 )
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    # Make sure it fails if min_matched_stars is too high
    # (For this test image, there's only something like 120 objects in the source list.)
    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[20.], mag_range=4., min_stars=50,
                                  min_matches=50 )
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

def test_solve_wcs_scamp( gaiadr3_excerpt, example_ds_with_sources_and_psf ):
    catexp = gaiadr3_excerpt
    ds = example_ds_with_sources_and_psf

    # Make True for visual testing purposes
    if False:
        catexp.ds9_regfile( 'catexp.reg', radius=4 )
        ds.sources.ds9_regfile( 'sources.reg', radius=3 )

    orighdr = ds.image._raw_header.copy()

    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[20.], mag_range=4., min_stars=50 )
    astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    # Because this was a ZTF image that had a WCS already, the new WCS
    # should be damn close, but not identical (since there's no way we
    # used exactly the same set of sources and stars, plus this was a
    # cropped ZTF image, not the full image).
    allsame = True
    for i in [ 1, 2 ]:
        for j in range( 17 ):
            diff = np.abs( ( orighdr[f'PV{i}_{j}'] - ds.image._raw_header[f'PV{i}_{j}'] ) / orighdr[f'PV{i}_{j}'] )
            if diff > 1e-6:
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

def actually_run_scamp( ds, astrometor ):
    with open( ds.image.get_fullpath()[0], "rb" ) as ifp:
        md5 = hashlib.md5()
        md5.update( ifp.read() )
        origmd5 = uuid.UUID( md5.hexdigest() )

    xvals = [ 0, 0, 2047, 2047 ]
    yvals = [ 0, 4095, 0, 4095 ]
    origwcs = WCS( ds.image.raw_header )

    ds = astrometor.run( ds )

    wcs = ds.wcs.wcs

    # Make sure that the new WCS is different from the original wcs
    # (since we know the one that came in the decam exposure is approximate)
    # BUT, make sure that it's within 40", because the original one, while
    # not great, is *something*
    origscs = origwcs.pixel_to_world( xvals, yvals )
    newscs = wcs.pixel_to_world( xvals, yvals )
    for origsc, newsc in zip( origscs, newscs ):
        assert not origsc.ra.value == pytest.approx( newsc.ra.value, abs=1./3600. )
        assert not origsc.dec.value == pytest.approx( newsc.dec.value, abs=1./3600. )
        assert origsc.ra.value == pytest.approx( newsc.ra.value, abs=40./3600. )   # cos(dec)...
        assert origsc.dec.value == pytest.approx( newsc.dec.value, abs=40./3600. )

    # These next few lines will need to be done after astrometry is done.  Right now,
    # we don't do saving and committing inside the Astrometor.run method.
    update_image_header = False
    if not ds.image.astro_cal_done:
        ds.image.astro_cal_done = True
        update_image_header = True
    ds.save_and_commit( update_image_header=update_image_header, overwrite=True )

    with SmartSession() as session:
        # Make sure the WCS made it into the databse
        q = ( session.query( WorldCoordinates )
              .filter( WorldCoordinates.source_list_id==ds.sources.id )
              .filter( WorldCoordinates.provenance_id==ds.wcs.provenance.id ) )
        assert q.count() == 1
        dbwcs = q.first()
        dbscs = dbwcs.wcs.pixel_to_world( xvals, yvals )
        for newsc, dbsc in zip( newscs, dbscs ):
            assert dbsc.ra.value == pytest.approx( newsc.ra.value, abs=0.01/3600. )
            assert dbsc.dec.value == pytest.approx( newsc.dec.value, abs=0.01/3600. )

        # Make sure the image got updated properly on the database
        # and on disk
        q = session.query( Image ).filter( Image.id==ds.image.id )
        assert q.count() == 1
        foundim = q.first()
        assert foundim.md5sum_extensions[0] == ds.image.md5sum_extensions[0]
        assert foundim.md5sum_extensions[0] != origmd5
        with open( foundim.get_fullpath()[0], 'rb' ) as ifp:
            md5 = hashlib.md5()
            md5.update( ifp.read() )
            assert uuid.UUID( md5.hexdigest() ) == foundim.md5sum_extensions[0]
        # This is probably redundant given the md5sum test we just did....
        ds.image._raw_header = None
        for kw in foundim.raw_header:
            # SIMPLE can't be an index to a Header.  (This is sort
            # of a weird thing in the astropy Header interface.)
            # BITPIX doesn't match because the ds.image raw header
            # was constructed from the exposure that had been
            # BSCALEd, even though the image we wrote to disk fully
            # a float (BITPIX=-32).
            if kw in [ 'SIMPLE', 'BITPIX' ]:
                continue
            assert foundim.raw_header[kw] == ds.image.raw_header[kw]

        # Make sure the new WCS got written to the FITS file
        with fits.open( foundim.get_fullpath()[0] ) as hdul:
            imhdr = hdul[0].header
        imwcs = WCS( hdul[0].header )
        imscs = imwcs.pixel_to_world( xvals, yvals )
        for newsc, imsc in zip( newscs, imscs ):
            assert newsc.ra.value == pytest.approx( imsc.ra.value, abs=0.01/3600. )
            assert newsc.dec.value == pytest.approx( imsc.dec.value, abs=0.01/3600. )

        # Make sure the archive has the right md5sum
        info = foundim.archive.get_info( f'{foundim.filepath}.image.fits' )
        assert info is not None
        assert uuid.UUID( info['md5sum'] ) == foundim.md5sum_extensions[0]

def test_run_scamp( decam_example_reduced_image_ds ):
    ds = decam_example_reduced_image_ds

    # Do a run that we know should succeed

    astrometor = AstroCalibrator( catalog='GaiaDR3', method='scamp', max_mag=[22.], mag_range=4.,
                                  min_stars=50, max_resid=0.15, crossid_radius=[2.0],
                                  min_frac_matched=0.1, min_matched_stars=10 )
    actually_run_scamp( ds, astrometor )

# TODO : test that it fails when it's supposed to

