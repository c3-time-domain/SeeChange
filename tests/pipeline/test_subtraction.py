import pytest
import uuid

import numpy as np
from scipy import ndimage

from improc.tools import sigma_clipping
from util.util import asUUID

from tests.conftest import SKIP_WARNING_TESTS


def test_subtraction_data_products( ptf_ref, ptf_supernova_image_datastores ):
    assert len(ptf_supernova_image_datastores) == 2
    ds1, _ = ptf_supernova_image_datastores

    assert ds1.sources is not None
    assert ds1.psf is not None
    assert ds1.wcs is not None
    assert ds1.zp is not None
    subtractor = ds1._pipeline.subtractor

    # run the subtraction like you'd do in the real pipeline (calls get_reference and get_sub_image internally)
    subtractor.pars.test_parameter = uuid.uuid4().hex
    subtractor.pars.method = 'naive'
    subtractor.pars.refset = 'test_refset_ptf'
    assert subtractor.pars['alignment_index'] == 'new'  # make sure alignment is configured to new, not latest image
    ds1._pipeline.make_provenance_tree( ds1, no_provtag=True )
    ds = subtractor.run( ds1 )
    assert len( ds.exceptions ) == 0      # Make sure no exceptions from subtractor run

    # check that we don't lazy load a subtracted image, but recalculate it
    assert subtractor.has_recalculated

    # check that the image is correctly ingested with all products
    assert ds.image is not None
    assert ds.sources is not None
    assert ds.psf is not None
    assert ds.wcs is not None
    assert ds.zp is not None

    # check that reference is loaded automatically when calling the subtraction
    assert ds.reference is not None
    assert asUUID( ds.reference.id ) == asUUID( ptf_ref.id )
    assert ds.reference != ptf_ref  # not the same object, it was loaded from DB!
    assert ds.reference.from_db  # it was loaded from DB!

    # check that the diff image is created
    assert ds.sub_image is not None
    assert ds.sub_image.data is not None


def test_subtraction_ptf_zogy(ptf_ref, ptf_supernova_image_datastores):
    assert len(ptf_supernova_image_datastores) == 2
    ds1, _ = ptf_supernova_image_datastores
    subtractor = ds1._pipeline.subtractor

    # run the subtraction like you'd do in the real pipeline (calls get_reference and get_sub_image internally)
    subtractor.pars.test_parameter = uuid.uuid4().hex
    subtractor.pars.method = 'zogy'  # this is the default, but it might not always be
    subtractor.pars.refset = 'test_refset_ptf'
    assert subtractor.pars['alignment_index'] == 'new'  # make sure alignment is configured to new, not latest image
    ds1._pipeline.make_provenance_tree( ds1, no_provtag=True )
    ds = subtractor.run( ds1 )
    assert len( ds.exceptions ) == 0      # Make sure no exceptions from subtractor run

    assert ds.sub_image is not None
    assert ds.sub_image.data is not None
    assert ds.sub_image.weight is not None

    # make sure there are not too many masked pixels
    mask = ds.sub_image.flags > 0
    labels, num_masked_regions = ndimage.label(mask)
    all_idx = np.arange(1, num_masked_regions + 1)
    region_pixel_counts = ndimage.sum(mask, labels, all_idx)
    region_pixel_counts.sort()
    region_pixel_counts = region_pixel_counts[:-1]  # remove that last region, which is the largest one

    # no region should have more than 6000 pixels masked
    assert max(region_pixel_counts) < 6000
    # No more than 1.6% pixels masked.  (This used to be 1%, but I think we masked a few more
    # pixels in the ref with the fixing of caodd zogy weights.)
    assert np.sum(region_pixel_counts) / ds.sub_image.data.size < 0.016

    # check that a visually-identified blank region really is 0, and that
    #   the subtraction weight makes sense.
    y0 = 1908
    y1 = 1934
    x0 = 1049
    x1 = 1075
    assert ( np.abs( ds.sub_image.data[y0:y1,x0:x1].mean() )
             < 3. * ( ds.sub_image.data[y0:y1,x0:x1].std() / np.sqrt( ds.sub_image.data[y0:y1,x0:x1].size ) ) )
    assert ( ds.sub_image.data[y0:y1,x0:x1].std()
             == pytest.approx( 1. / np.sqrt( ds.sub_image.weight[y0:y1,x0:x1] ).mean(), rel=0.1 ) )

    # isolate the score, masking the bad pixels
    S = ds.zogy_score.copy()
    S[ds.sub_image.flags > 0] = np.nan

    mu, sigma = sigma_clipping(S)
    assert abs(mu) < 0.1  # the mean should be close to zero
    assert abs(sigma - 1) < 0.1  # the standard deviation should be close to 1


def test_subtraction_ptf_hotpants( ptf_ref, ptf_supernova_image_datastores ):
    assert len( ptf_supernova_image_datastores ) == 2
    ds1, _ = ptf_supernova_image_datastores
    subtractor = ds1._pipeline.subtractor
    detector = ds1._pipeline.detector

    subtractor.pars.method = 'hotpants'
    subtractor.pars.refset = 'test_refset_ptf'
    detector.pars.method = 'sextractor'
    ds1._pipeline.make_provenance_tree( ds1, no_provtag=True )
    ds = subtractor.run( ds1 )
    assert len( ds.exceptions ) == 0      # Make sure no exceptions from subtractor run

    assert ds.sub_image is not None
    assert ds.sub_image.data is not None
    assert ds.sub_image.weight is not None

    # make sure there are not too many masked pixels
    mask = ds.sub_image.flags > 0
    labels, num_masked_regions = ndimage.label(mask)
    all_idx = np.arange(1, num_masked_regions + 1)
    region_pixel_counts = ndimage.sum(mask, labels, all_idx)
    region_pixel_counts.sort()
    region_pixel_counts = region_pixel_counts[:-1]  # remove that last region, which is the largest one

    # no region should have more than 10000 pixels masked
    assert max(region_pixel_counts) < 10000
    # No more than ~3% pixels masked.  (Hotpants method seems to mask more than zogy.)
    assert np.sum(region_pixel_counts) / ds.sub_image.data.size < 0.032

    # check that a visually-identified blank region really is 0, and that
    #   the subtraction weight makes sense.
    y0 = 1908
    y1 = 1934
    x0 = 1049
    x1 = 1075
    assert ( np.abs( ds.sub_image.data[y0:y1,x0:x1].mean() )
             < 3.2 * ( ds.sub_image.data[y0:y1,x0:x1].std() / np.sqrt( ds.sub_image.data[y0:y1,x0:x1].size ) ) )
    assert ( ds.sub_image.data[y0:y1,x0:x1].std()
             == pytest.approx( 1. / np.sqrt( ds.sub_image.weight[y0:y1,x0:x1] ).mean(), rel=0.1 ) )


def test_warnings_and_exceptions( decam_datastore_through_zp, decam_reference, decam_default_calibrators):
    ds = decam_datastore_through_zp
    subtractor = ds._pipeline.subtractor

    if not SKIP_WARNING_TESTS:
        subtractor.pars.inject_warnings = 1
        subtractor.pars.refset = 'test_refset_decam'
        ds._pipeline.make_provenance_tree( ds )

        with pytest.warns(UserWarning) as record:
            subtractor.run( ds )
        assert len(record) > 0
        assert any("Warning injected by pipeline parameters in process 'subtraction'." in str(w.message)
                   for w in record)

    subtractor.pars.inject_warnings = 0
    subtractor.pars.inject_exceptions = 1
    ds.sub_image = None
    ds._pipeline.make_provenance_tree( ds )
    with pytest.raises(Exception, match="Exception injected by pipeline parameters in process 'subtraction'."):
        ds = subtractor.run( ds )
