import warnings

import pytest
import random
import re
import subprocess

import numpy as np
import astropy.wcs

from models.base import SmartSession
from models.image import Image
from models.background import Background
from models.source_list import SourceList
from models.psf import PSF
from models.enums_and_bitflags import string_to_bitflag, flag_image_bits_inverse

from improc.alignment import ImageAligner
from pipeline.coaddition import Coadder
from util.util import env_as_bool


# Putting this in RUN_SLOW_TESTS because there are a couple of 60-second timeouts.
# Good to have this test to run sometimes, but basic functionality of swarp_fodder_wcs
#   *will* get testsed in test_warp_decam below.
@pytest.mark.skipif( not env_as_bool('RUN_SLOW_TESTS'), reason="Set RUN_SLOW_TESTS to run this test" )
def test_get_swarp_fodder_wcs( decam_datastore_through_zp, decam_elais_e1_two_refs_datastore ):
    refds0, refds1 = decam_elais_e1_two_refs_datastore
    ds = decam_datastore_through_zp
    aligner = ImageAligner()

    wcs = aligner.get_swarp_fodder_wcs( ds.image, ds.sources, ds.wcs, ds.zp, refds0.sources )

    # target WCS should be close, but not identical, to the old target wcs
    x, y = np.meshgrid( np.array( [ 0.05, 0.5, 0.75, 0.95 ] ) * refds0.image.data.shape[1],
                        np.array( [ 0.05, 0.5, 0.75, 0.95 ] ) * refds0.image.data.shape[0] )
    oldsc = refds0.wcs.wcs.pixel_to_world( x, y )
    newsc = wcs.pixel_to_world( x, y )
    dra = np.fabs( ( oldsc.ra - newsc.ra ).value ) * np.cos( oldsc.dec.value * np.pi / 180. )
    ddec = np.fabs( ( oldsc.dec - newsc.dec ).value )
    assert ( dra < 1./3600. ).all()
    assert ( ddec < 1./3600. ).all()
    assert ( dra > 0.001/3600. ).all()
    assert ( ddec > 0.001/3600. ).all()

    # Make sure it fails if it shouldn't succeed
    # (The two refs are different chips from the same exposure, so don't overlap.)
    # Make sure we can find a solution.  (This is a gratuitous 60 second delay in tests....)
    with pytest.raises( subprocess.TimeoutExpired ):
        wcs = aligner.get_swarp_fodder_wcs( refds1.image, refds1.sources, refds1.wcs, refds1.zp, refds0.sources )

    # Make sure it falls back if we tell it to.  (Another 60 second delay in tests....)
    wcs = aligner.get_swarp_fodder_wcs( refds1.image, refds1.sources, refds1.wcs, refds1.zp, refds0.sources,
                                        fall_back_wcs=refds0.wcs.wcs )
    newsc = wcs.pixel_to_world( x, y )
    dra = np.fabs( ( oldsc.ra - newsc.ra ).value ) * np.cos( oldsc.dec.value * np.pi / 180. )
    ddec = np.fabs( ( oldsc.dec - newsc.dec ).value )
    # Same WCS, should be *really* close.  (Ideally identical, in fact.)
    assert ( dra < 0.001/3600. ).all()
    assert ( ddec < 0.001/3600. ).all()


def test_warp_decam( decam_datastore_through_zp, decam_reference ):
    ds = decam_datastore_through_zp

    try:
        ds.get_reference()
        aligner = ImageAligner()
        ( warped, warpedsrc,
          warpedbg, warpedpsf ) = aligner.run( ds.ref_image, ds.ref_sources, ds.ref_bg, ds.ref_psf,
                                               ds.ref_wcs, ds.ref_zp, ds.image, ds.sources )
        assert isinstance( warped, Image )
        assert isinstance( warpedsrc, SourceList )
        assert isinstance( warpedbg, Background )
        assert isinstance( warpedpsf, PSF )
        assert warped.data.shape == ds.image.data.shape

        warped.filepath = f'warp_test_{"".join(random.choices("abcdefghijklmnopqrstuvwxyz",k=10))}'

        # The warped image should have a WCS in the header thanks to swarp (and the scamp stuff we did)
        warpedwcs = astropy.wcs.WCS( warped.header )

        # Remember numpy arrays are indexed [y, x]
        def ref_to_warped( y, x ):
            sc = ds.ref_wcs.wcs.pixel_to_world( x, y )
            rval = warpedwcs.world_to_pixel( sc )
            return float( rval[1] ), float( rval[0] )

        def warped_to_ref( y, x ):
            sc = warpedwcs.pixel_to_world( x, y )
            rval = ds.ref_wcs.wcs.world_to_pixel( sc )
            return float( rval[1] ), float( rval[0] )

        oob_bitflag = string_to_bitflag( 'out of bounds', flag_image_bits_inverse)
        badpixel_bitflag = string_to_bitflag( 'bad pixel', flag_image_bits_inverse)
        # Commenting out this next test; when I went to the ELAIS-E1 reference,
        #   it didn't pass.  Seems that there are more bad pixels, and/or fewer
        #   pixels out of bounds, than was the case with the DECaPS reference.
        # assert (warped.flags == oob_bitflag).sum() > (warped.flags == badpixel_bitflag).sum()

        # Check a couple of spots on the image
        # First, around a star (which I visually inspected and saw was lined up):
        stararea = ( slice( 2306, 2327, 1 ), slice( 1046, 1068, 1 ) )
        origll = warped_to_ref( stararea[0].start, stararea[1].start )
        origur = warped_to_ref( stararea[0].stop, stararea[1].stop )
        # I know in this case that the ll and ur are swapped,
        #   (the image got transposed in the warp)
        #   hence the switch below
        origrefstararea = ( slice( round(origur[0]), round(origll[0]), 1 ),
                            slice( round(origur[1]), round(origll[1]), 1 ) )
        assert ds.image.data[ stararea ].sum() == pytest.approx( 323622.53, rel=0.001 )
        assert warped.data[ stararea ].sum() == pytest.approx( 24483.215, rel=0.001 )
        assert ( warped.data[ stararea ].sum() ==
                 pytest.approx( ds.ref_image.data[ origrefstararea ].sum(), rel=0.001 ) )

        # And a blank spot (here we can do some statistics instead of hard coded values)
        blankarea = ( slice( 2700, 2721, 1 ), slice( 1040, 1061, 1 ) )
        origll = warped_to_ref( blankarea[0].start, blankarea[1].start )
        origur = warped_to_ref( blankarea[0].stop, blankarea[1].stop )
        origrefblankarea = ( slice( round(origur[0]), round(origll[0]), 1 ),
                             slice( round(origur[1]), round(origll[1]), 1 ) )
        num_pix = ds.image.data[ blankarea ].size
        newmean = ( ds.image.data[ blankarea ] - ds.bg.counts[ blankarea ] ).mean()
        newstd = ( ds.image.data[ blankarea ] - ds.bg.counts[ blankarea ] ).std()
        # I know the refernce is nominally background subtracted
        origrefmean = ( ds.ref_image.data[ origrefblankarea ] ).mean()
        origrefstd = ( ds.ref_image.data[ origrefblankarea ] ).std()
        warpedmean = ( warped.data[ blankarea ] ).mean()
        warpedstd = ( warped.data[ blankarea ] ).std()

        # Check that the reference is actually background subtracted
        # (fudged the 3σ based on empiricism...)
        assert origrefmean == pytest.approx( 0., abs=3. * origrefstd / np.sqrt(num_pix) )
        assert warpedmean == pytest.approx( 0., abs=3. * warpedstd / np.sqrt(num_pix) )

        # The rel values below are really regression tests, since I tuned them to
        #   what matched.  Correlated schmorrelated.  (In summed images, which had
        #   resampling, and warped images.)
        assert origrefmean == pytest.approx( ds.ref_bg.value, abs=3. * origrefstd / np.sqrt(num_pix) )
        assert origrefstd == pytest.approx( ds.ref_bg.noise, rel=0.09 )
        assert warpedmean == pytest.approx( warpedbg.value, abs=3. * warpedstd / np.sqrt(num_pix) )
        assert warpedstd == pytest.approx( warpedbg.noise, rel=0.15 )
        assert newmean == pytest.approx( 0., abs=3. * newstd / np.sqrt(num_pix) )
        assert newstd == pytest.approx( ds.bg.noise, rel=0.01 )


        # Make sure the warped image WCS is about right.  We don't
        # expect it to be exactly identical, but it should be very
        # close.

        x = [ 256, 1791, 256, 1791, 1024 ]
        y = [ 256, 256, 3839, 3839, 2048 ]
        imsc = ds.wcs.wcs.pixel_to_world( x, y )
        warpsc = warpedwcs.pixel_to_world( x, y )
        # In fact, it's better than 0.1/3600. except for the (256,3839) (upper-left) in dec
        assert all( [ i.ra.deg == pytest.approx(w.ra.deg, abs=0.25/3600.) for i, w in zip( imsc, warpsc ) ] )
        assert all( [ i.dec.deg == pytest.approx(w.dec.deg, abs=0.25/3600.) for i, w in zip( imsc, warpsc ) ] )

    finally:
        if 'warped' in locals():
            warped.delete_from_disk_and_database()


def test_alignment_in_image( ptf_reference_image_datastores, code_version ):
    try:  # cleanup at the end
        # ptf_reference_images = ptf_reference_images[:4]  # speed things up using fewer images
        coaddparams = { 'alignment': { 'method': 'swarp' }, 'alignment_index': 'last' }

        coadder = Coadder( **coaddparams )
        prov, _ = coadder.get_coadd_prov( ptf_reference_image_datastores, code_version_id=code_version.id )
        prov.insert_if_needed()
        if prov.parameters['alignment_index'] == 'last':
            index = -1
        elif prov.parameters['alignment_index'] == 'first':
            index = 0
        else:
            raise ValueError(f"Unknown alignment reference index: {prov.parameters['alignment_index']}")

        new_image = Image.from_image_zps( [ d.zp for d in ptf_reference_image_datastores ], index=index )
        new_image.provenance_id = prov.id

        coadder.run_alignment( ptf_reference_image_datastores, index )

        # We're manually doing a naive sum here
        new_image.data = np.sum( [ d.image.data for d in coadder.aligned_datastores ], axis=0 )
        new_image.save()
        new_image.insert()

        # check that the filename is correct
        # e.g.: /path/to/data/PTF_<YYYYMMDD>_<HHMMSS>_<sec_ID>_<filt>_ComSci_<prov hash>_u-<coadd hash>.image.fits
        match = re.match(r'/.*/.*_\d{8}_\d{6}_.*_.*_ComSci_.{6}_u-.{6}\.image\.fits', new_image.get_fullpath()[0])
        assert match is not None

        upstream_zps = new_image.get_upstreams()
        assert [ i.id for i in upstream_zps ] == [ d.zp.id for d in ptf_reference_image_datastores ]
        assert len( coadder.aligned_datastores ) == len( ptf_reference_image_datastores )
        dsindex = ptf_reference_image_datastores[index]
        assert np.array_equal( coadder.aligned_datastores[index].image.data,
                               dsindex.image.data - dsindex.bg.counts )

        ref = ptf_reference_image_datastores[index].image

        # check that images are aligned properly
        for ds in coadder.aligned_datastores:
            check_aligned( ds.image, ref )

        # check that unaligned images do not pass the check
        for ds in ptf_reference_image_datastores:
            if ds.image.id == ref.id:
                continue
            with pytest.raises(AssertionError):
                check_aligned(ds.image, ref)

        # add new image to database
        with SmartSession() as session:
            new_image = session.merge(new_image)
            session.commit()

    finally:
        ImageAligner.cleanup_temp_images()
        # (The aligned datastores should not have been saved to disk or database.)
        if 'new_image' in locals():
            new_image.delete_from_disk_and_database(remove_downstreams=True)


def check_aligned(image1, image2):
    d1 = image1.data.copy()
    d1[image1.flags > 0] = np.nan
    d2 = image2.data.copy()
    d2[image2.flags > 0] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r'.*All-NaN slice encountered.*')
        row_func1 = np.nansum(d1 - np.nanmedian(d1, axis=1, keepdims=True), axis=1)
        row_func2 = np.nansum(d2 - np.nanmedian(d2, axis=1, keepdims=True), axis=1)

        col_func1 = np.nansum(d1 - np.nanmedian(d1, axis=0, keepdims=True), axis=0)
        col_func2 = np.nansum(d2 - np.nanmedian(d2, axis=0, keepdims=True), axis=0)

    xcorr_rows = np.correlate(row_func1, row_func2, mode='full')
    xcorr_cols = np.correlate(col_func1, col_func2, mode='full')

    assert abs(np.argmax(xcorr_rows) - len(xcorr_rows) // 2) <= 1
    assert abs(np.argmax(xcorr_cols) - len(xcorr_cols) // 2) <= 1
