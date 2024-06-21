import pytest
import uuid

import numpy as np
from scipy import ndimage

from improc.tools import sigma_clipping


def test_subtraction_data_products(ptf_ref, ptf_supernova_images, subtractor):
    assert len(ptf_supernova_images) == 2
    image1, image2 = ptf_supernova_images

    assert image1.sources is not None
    assert image1.psf is not None
    assert image1.wcs is not None
    assert image1.zp is not None

    # run the subtraction like you'd do in the real pipeline (calls get_reference and get_subtraction internally)
    subtractor.pars.test_parameter = uuid.uuid4().hex
    subtractor.pars.method = 'naive'
    assert subtractor.pars.alignment['to_index'] == 'new'  # make sure alignment is configured to new, not latest image
    ds = subtractor.run(image1)

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
    assert ds.reference.id == ptf_ref.id
    assert ds.reference != ptf_ref  # not the same object, it was loaded from DB!
    assert ds.reference.from_db  # it was loaded from DB!

    # check that the diff image is created
    assert ds.sub_image is not None
    assert ds.sub_image.data is not None


def test_subtraction_ptf_zogy(ptf_ref, ptf_supernova_images, subtractor):
    assert len(ptf_supernova_images) == 2
    image1, image2 = ptf_supernova_images

    # run the subtraction like you'd do in the real pipeline (calls get_reference and get_subtraction internally)
    subtractor.pars.test_parameter = uuid.uuid4().hex
    subtractor.pars.method = 'zogy'  # this is the default, but it might not always be
    assert subtractor.pars.alignment['to_index'] == 'new'  # make sure alignment is configured to new, not latest image
    ds = subtractor.run(image1)

    assert ds.sub_image is not None
    assert ds.sub_image.data is not None

    # make sure there are not too many masked pixels
    mask = ds.sub_image.flags > 0
    labels, num_masked_regions = ndimage.label(mask)
    all_idx = np.arange(1, num_masked_regions + 1)
    region_pixel_counts = ndimage.sum(mask, labels, all_idx)
    region_pixel_counts.sort()
    region_pixel_counts = region_pixel_counts[:-1]  # remove that last region, which is the largest one

    assert max(region_pixel_counts) < 5000  # no region should have more than 5000 pixels masked
    assert np.sum(region_pixel_counts) / ds.sub_image.data.size < 0.01  # no more than 1% of the pixels should be masked

    # isolate the score, masking the bad pixels
    S = ds.sub_image.score.copy()
    S[ds.sub_image.flags > 0] = np.nan

    mu, sigma = sigma_clipping(S)
    assert abs(mu) < 0.1  # the mean should be close to zero
    assert abs(sigma - 1) < 0.1  # the standard deviation should be close to 1


def test_warnings_and_exceptions(decam_datastore, decam_reference, subtractor, decam_default_calibrators):
    subtractor.pars.inject_warnings = 1

    with pytest.warns(UserWarning) as record:
        subtractor.run(decam_datastore)
    assert len(record) > 0
    assert any("Warning injected by pipeline parameters in process 'subtraction'." in str(w.message) for w in record)

    subtractor.pars.inject_warnings = 0
    subtractor.pars.inject_exceptions = 1
    with pytest.raises(Exception) as excinfo:
        ds = subtractor.run(decam_datastore)
        ds.reraise()
    assert "Exception injected by pipeline parameters in process 'subtraction'." in str(excinfo.value)
    ds.read_exception()