import logging
import warnings

import pytest
import random
import re

import numpy as np
import astropy.wcs

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import Provenance
from models.image import Image

from models.enums_and_bitflags import string_to_bitflag, flag_image_bits_inverse
from improc.alignment import ImageAligner


def test_warp_decam( decam_datastore, decam_reference ):
    ds = decam_datastore

    try:
        ds.get_reference()
        aligner = ImageAligner()
        warped = aligner.run( ds.reference.image, ds.image )
        warped.filepath = f'warp_test_{"".join(random.choices("abcdefghijklmnopqrstuvwxyz",k=10))}'

        assert warped.data.shape == ds.image.data.shape

        oob_bitflag = string_to_bitflag( 'out of bounds', flag_image_bits_inverse)
        badpixel_bitflag = string_to_bitflag( 'bad pixel', flag_image_bits_inverse)
        assert (warped.flags == oob_bitflag).sum() > (warped.flags == badpixel_bitflag).sum()

        # Check a couple of spots on the image
        # First, around a star:
        assert ds.image.data[ 2223:2237, 545:559 ].sum() == pytest.approx( 58014.1, rel=0.01 )
        assert warped.data[ 2223:2237, 545:559 ].sum() == pytest.approx( 22597.9, rel=0.01 )
        # And a blank spot
        assert ds.image.data[ 2243:2257, 575:589 ].sum() == pytest.approx( 35298.6, rel=0.01 )    # sky not subtracted
        assert warped.data[ 2243:2257, 575:589 ].sum() == pytest.approx( 971.7, rel=0.01 )

        # Make sure the warped image WCS is about right.  We don't
        # expect it to be exactly identical, but it should be very
        # close.
        imwcs = ds.wcs.wcs
        warpwcs = astropy.wcs.WCS( warped.header )
        x = [ 256, 1791, 256, 1791, 1024 ]
        y = [ 256, 256, 3839, 3839, 2048 ]
        imsc = imwcs.pixel_to_world( x, y )
        warpsc = warpwcs.pixel_to_world( x, y )
        assert all( [ i.ra.deg == pytest.approx(w.ra.deg, abs=0.1/3600.) for i, w in zip( imsc, warpsc ) ] )
        assert all( [ i.dec.deg == pytest.approx(w.dec.deg, abs=0.1/3600.) for i, w in zip( imsc, warpsc ) ] )

    finally:
        if 'warped' in locals():
            warped.delete_from_disk_and_database()


def test_alignment_in_image( ptf_reference_images, code_version ):
    try:  # cleanup at the end
        # ptf_reference_images = ptf_reference_images[:4]  # speed things up using fewer images
        prov = Provenance(
            code_version=code_version,
            parameters={'alignment': {'method': 'swarp', 'to_index': 'last'}, 'test_parameter': 'test_value'},
            upstreams=[],
            process='coaddition',
            is_testing=True,
        )
        if prov.parameters['alignment']['to_index'] == 'last':
            index = -1
        elif prov.parameters['alignment']['to_index'] == 'first':
            index = 0
        else:
            raise ValueError(f"Unknown alignment reference index: {prov.parameters['alignment']['to_index']}")

        new_image = Image.from_images(ptf_reference_images, index=index)
        new_image.provenance = prov
        new_image.provenance.upstreams = new_image.get_upstream_provenances()
        new_image.data = np.sum([image.data for image in new_image.aligned_images], axis=0)
        new_image.save()

        # check that the filename is correct
        # e.g.: /path/to/data/PTF_<YYYYMMDD>_<HHMMSS>_<sec_ID>_<filt>_ComSci_<prov hash>_u-<coadd hash>.image.fits
        match = re.match(r'/.*/.*_\d{8}_\d{6}_.*_.*_ComSci_.{6}_u-.{6}\.image\.fits', new_image.get_fullpath()[0])
        assert match is not None

        aligned = new_image.aligned_images
        assert new_image.upstream_images == ptf_reference_images
        assert len(aligned) == len(ptf_reference_images)
        assert np.array_equal(aligned[index].data, ptf_reference_images[index].data)
        ref = ptf_reference_images[index]

        # check that images are aligned properly
        for image in new_image.aligned_images:
            check_aligned(image, ref)

        # check that unaligned images do not pass the check
        for image in new_image.upstream_images:
            if image == ref:
                continue
            with pytest.raises(AssertionError):
                check_aligned(image, ref)

        # add new image to database
        with SmartSession() as session:
            new_image = session.merge(new_image)
            session.commit()

        # should be able to recreate aligned images from scratch
        with SmartSession() as session:
            loaded_image = session.scalars(sa.select(Image).where(Image.id == new_image.id)).first()
            assert loaded_image is not None
            assert len(loaded_image.aligned_images) == len(ptf_reference_images)
            assert np.array_equal(loaded_image.aligned_images[-1].data, ptf_reference_images[-1].data)

            # check that images are aligned properly
            for image in loaded_image.aligned_images:
                check_aligned(image, ref)

    finally:
        ImageAligner.cleanup_temp_images()
        for im in new_image.aligned_images:
            im.delete_from_disk_and_database()
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
