import pytest
import random

import astropy.wcs

from pipeline.data_store import DataStore
from pipeline.alignment import ImageAligner
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator

def test_warp_decam( decam_example_reduced_image_ds_with_zp, ref_for_decam_example_image ):
    ds = decam_example_reduced_image_ds_with_zp[0]
    ds.save_and_commit()
    refds = DataStore()
    refds.image = ref_for_decam_example_image
    warpds = None

    try:
        # Need a source list and wcs for the ref image before we can warp it
        det = Detector( measure_psf=True )
        refds = det.run( refds )
        refds.sources.filepath = f'{refds.image.filepath}.sources.fits'
        refds.psf.filepath = refds.image.filepath
        refds.save_and_commit( no_archive=True )
        ast = AstroCalibrator()
        refds = ast.run( refds )
        refds.save_and_commit( no_archive=True )
        phot = PhotCalibrator()
        refds = phot.run( refds )

        Aligner = ImageAligner()
        warpds = Aligner.run( refds, ds )
        warpds.image.filepath = f'warp_test_{"".join(random.choices("abcdefghijklmnopqrstuvwxyz",k=10))}'

        assert warpds.image.data.shape == ds.image.data.shape

        # Check a couple of spots on the image
        # First, around a star:
        assert ds.image.data[ 2223:2237, 545:559 ].sum() == pytest.approx( 58014.1, abs=1 )
        assert warpds.image.data[ 2223:2237, 545:559 ].sum() == pytest.approx( 22597.9, abs=1 )
        # And a blank spot
        assert ds.image.data[ 2243:2257, 575:589 ].sum() == pytest.approx( 35298.6, abs=1 )    # sky not subtracted
        assert warpds.image.data[ 2243:2257, 575:589 ].sum() == pytest.approx( 971.7, abs=1 )

        # Make sure the warped image WCS is about right.  We don't
        # expect it to be exactly identical, but it should be very
        # close.
        imwcs = ds.wcs.wcs
        warpwcs = astropy.wcs.WCS( warpds.image.raw_header )
        x = [ 256, 1791, 256, 1791, 1024 ]
        y = [ 256, 256, 3839, 3839, 2048 ]
        imsc = imwcs.pixel_to_world( x, y )
        warpsc = warpwcs.pixel_to_world( x, y )
        assert all( [ i.ra.deg == pytest.approx(w.ra.deg,abs=0.05/3600.) for i, w in zip( imsc, warpsc ) ] )
        assert all( [ i.dec.deg == pytest.approx(w.dec.deg,abs=0.05/3600.) for i, w in zip( imsc, warpsc ) ] )

    finally:
        refds.delete_everything()
        if warpds is not None:
            warpds.delete_everything()

