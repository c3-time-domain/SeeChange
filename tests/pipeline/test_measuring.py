import time

import pytest
import uuid

import numpy as np


from improc.tools import make_gaussian


@pytest.mark.flaky(max_runs=3)
def test_measuring(measurer, decam_cutouts):
    measurer.pars.test_parameter = uuid.uuid4().hex
    measurer.pars.bad_pixel_exclude = ['saturated']
    sz = decam_cutouts[0].sub_data.shape
    fwhm = decam_cutouts[0].sources.image.get_psf().fwhm_pixels

    # clear any flags for the fake data we are using
    for i in range(12):
        decam_cutouts[i].sub_flags = np.zeros_like(decam_cutouts[i].sub_flags)

    # delta function
    decam_cutouts[0].sub_data = np.zeros_like(decam_cutouts[0].sub_data)
    decam_cutouts[0].sub_data[sz[0] // 2, sz[1] // 2] = 100.0

    # shifted delta function
    decam_cutouts[1].sub_data = np.zeros_like(decam_cutouts[0].sub_data)
    decam_cutouts[1].sub_data[sz[0] // 2 + 2, sz[1] // 2 + 3] = 200.0

    # gaussian
    decam_cutouts[2].sub_data = make_gaussian(imsize=sz[0], sigma_x=fwhm / 2.355, norm=1) * 1000

    # shifted gaussian
    decam_cutouts[3].sub_data = make_gaussian(
        imsize=sz[0], sigma_x=fwhm / 2.355, norm=1, offset_x=-2, offset_y=-3
    ) * 500

    # dipole
    decam_cutouts[4].sub_data = np.zeros_like(decam_cutouts[4].sub_data)
    decam_cutouts[4].sub_data += make_gaussian(
        imsize=sz[0], sigma_x=fwhm / 2.355, norm=1, offset_x=-1, offset_y=-0.8
    ) * 500
    decam_cutouts[4].sub_data -= make_gaussian(
        imsize=sz[0], sigma_x=fwhm / 2.355, norm=1, offset_x=1, offset_y=0.8
    ) * 500

    # shifted gaussian with noise
    decam_cutouts[5].sub_data = decam_cutouts[3].sub_data + np.random.normal(0, 1, size=sz)

    # dipole with noise
    decam_cutouts[6].sub_data = decam_cutouts[4].sub_data + np.random.normal(0, 1, size=sz)

    # delta function with bad pixel
    decam_cutouts[7].sub_data = np.zeros_like(decam_cutouts[0].sub_data)
    decam_cutouts[7].sub_data[sz[0] // 2, sz[1] // 2] = 100.0
    decam_cutouts[7].sub_flags[sz[0] // 2 + 2, sz[1] // 2 + 2] = 1  # bad pixel

    # delta function with bad pixel and saturated pixel
    decam_cutouts[8].sub_data = np.zeros_like(decam_cutouts[0].sub_data)
    decam_cutouts[8].sub_data[sz[0] // 2, sz[1] // 2] = 100.0
    decam_cutouts[8].sub_flags[sz[0] // 2 + 2, sz[1] // 2 + 1] = 1  # bad pixel
    decam_cutouts[8].sub_flags[sz[0] // 2 - 2, sz[1] // 2 + 1] = 4  # saturated should be ignored!

    # delta function with offset that makes it far from the bad pixel
    decam_cutouts[9].sub_data = np.zeros_like(decam_cutouts[0].sub_data)
    decam_cutouts[9].sub_data[sz[0] // 2 + 3, sz[1] // 2 + 3] = 100.0
    decam_cutouts[9].sub_flags[sz[0] // 2 - 2, sz[1] // 2 - 2] = 1  # bad pixel

    # gaussian that is too wide
    decam_cutouts[10].sub_data = make_gaussian(imsize=sz[0], sigma_x=fwhm / 2.355 * 2, norm=1) * 1000
    decam_cutouts[10].sub_data += np.random.normal(0, 1, size=sz)

    # streak
    decam_cutouts[11].sub_data = make_gaussian(
        imsize=sz[0], sigma_x=fwhm / 2.355, sigma_y=20, rotation=25, norm=1
    ) * 1000
    decam_cutouts[11].sub_data += np.random.normal(0, 1, size=sz)

    ds = measurer.run(decam_cutouts)

    assert len(ds.all_measurements) == len(ds.cutouts)

    # verify all scores have been assigned
    for score in measurer.pars.analytical_cuts:
        assert score in ds.measurements[0].disqualifier_scores

    m = ds.all_measurements[0]  # delta function
    assert m.disqualifier_scores['negatives'] == 0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] < 0.01
    assert m.disqualifier_scores['filter bank'] == 1
    assert m.get_filter_description() == f'PSF mismatch (FWHM= 0.25 x {fwhm:.2f})'

    assert np.allclose(m.flux_apertures, 100)  # aperture is irrelevant for delta function
    assert m.background == 0
    assert m.background_err == 0
    for i in range(3):  # check only the last apertures, that are smaller than cutout square
        assert m.area_apertures[i] == pytest.approx(np.pi * (m.aper_radii[i] + 0.5) ** 2, rel=0.1)

    m = ds.all_measurements[1]  # shifted delta function
    assert m.disqualifier_scores['negatives'] == 0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] == pytest.approx(np.sqrt(2 ** 2 + 3 ** 2), abs=0.1)
    assert m.disqualifier_scores['filter bank'] == 1
    assert m.get_filter_description() == f'PSF mismatch (FWHM= 0.25 x {fwhm:.2f})'

    assert np.allclose(m.flux_apertures, 200)
    assert m.background == 0
    assert m.background_err == 0

    m = ds.all_measurements[2]  # gaussian
    assert m.disqualifier_scores['negatives'] < 1.0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] < 0.1
    assert m.disqualifier_scores['filter bank'] == 0
    assert m.get_filter_description() == f'PSF match (FWHM= 1.00 x {fwhm:.2f})'

    assert m.flux_apertures[0] < 900
    assert m.flux_apertures[1] < 1000
    for i in range(2, len(m.flux_apertures)):
        assert m.flux_apertures[i] == pytest.approx(1000, rel=0.1)
    assert m.background == pytest.approx(0, abs=0.01)
    assert m.background_err == pytest.approx(0, abs=0.01)

    # TODO: add test for PSF flux when it is implemented

    m = ds.all_measurements[3]  # shifted gaussian
    assert m.disqualifier_scores['negatives'] < 1.0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] == pytest.approx(np.sqrt(2 ** 2 + 3 ** 2), abs=1.0)
    assert m.disqualifier_scores['filter bank'] == 0

    assert m.flux_apertures[0] < 450
    assert m.flux_apertures[1] < 500
    for i in range(2, len(m.flux_apertures)):
        assert m.flux_apertures[i] == pytest.approx(500, rel=0.1)
    assert m.background == pytest.approx(0, abs=0.01)
    assert m.background_err == pytest.approx(0, abs=0.01)

    m = ds.all_measurements[4]  # dipole
    assert m.disqualifier_scores['negatives'] == pytest.approx(1.0, abs=0.1)
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] > 100
    assert m.disqualifier_scores['filter bank'] > 0

    # the dipole's large offsets will short-circuit the iterative repositioning of the aperture (should be flagged!)
    assert np.allclose(m.flux_apertures, 0)
    assert np.allclose(m.area_apertures, sz[0] * sz[1])
    assert m.background == pytest.approx(0, abs=0.01)
    assert m.background_err > 1.0
    assert m.background_err < 10.0

    m = ds.all_measurements[5]  # shifted gaussian with noise
    assert m.disqualifier_scores['negatives'] < 1.0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] == pytest.approx(np.sqrt(2 ** 2 + 3 ** 2), rel=0.1)
    assert m.disqualifier_scores['filter bank'] == 0
    assert m.get_filter_description() == f'PSF match (FWHM= 1.00 x {fwhm:.2f})'

    assert m.flux_apertures[0] < 450
    assert m.flux_apertures[1] < 500
    for i in range(2, len(m.flux_apertures)):
        assert m.flux_apertures[i] == pytest.approx(500, rel=0.1)

    m = ds.all_measurements[6]  # dipole with noise
    assert m.disqualifier_scores['negatives'] == pytest.approx(1.0, abs=0.2)
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] > 10
    assert m.disqualifier_scores['filter bank'] > 0

    m = ds.all_measurements[7]  # delta function with bad pixel
    assert m.disqualifier_scores['negatives'] == 0
    assert m.disqualifier_scores['bad pixels'] == 1
    assert m.disqualifier_scores['offsets'] < 0.01
    assert m.disqualifier_scores['filter bank'] == 1
    assert m.get_filter_description() == f'PSF mismatch (FWHM= 0.25 x {fwhm:.2f})'

    m = ds.all_measurements[8]  # delta function with bad pixel and saturated pixel
    assert m.disqualifier_scores['negatives'] == 0
    assert m.disqualifier_scores['bad pixels'] == 1  # we set to ignore the saturated pixel!
    assert m.disqualifier_scores['offsets'] < 0.01
    assert m.disqualifier_scores['filter bank'] == 1
    assert m.get_filter_description() == f'PSF mismatch (FWHM= 0.25 x {fwhm:.2f})'

    m = ds.all_measurements[9]  # delta function with offset that makes it far from the bad pixel
    assert m.disqualifier_scores['negatives'] == 0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] == pytest.approx(np.sqrt(3 ** 2 + 3 ** 2), abs=0.1)
    assert m.disqualifier_scores['filter bank'] == 1

    m = ds.all_measurements[10]  # gaussian that is too wide
    assert m.disqualifier_scores['negatives'] < 1.0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] < 0.5
    assert m.disqualifier_scores['filter bank'] == 2
    assert m.get_filter_description() == f'PSF mismatch (FWHM= 2.00 x {fwhm:.2f})'

    assert m.flux_apertures[0] < 400
    assert m.flux_apertures[1] < 600
    for i in range(2, len(m.flux_apertures)):
        assert m.flux_apertures[i] == pytest.approx(1000, rel=1)

    assert m.background == pytest.approx(0, abs=0.2)
    assert m.background_err == pytest.approx(1.0, abs=0.2)

    m = ds.all_measurements[11]  # streak
    # TODO: this fails because background is too high, need to fix this by using a better background estimation
    #  one way this could work is by doing a hard-edge annulus and taking sigma_clipping (or median) of the pixel
    #  values, instead of the weighted mean we are using now.
    # assert m.disqualifier_scores['negatives'] < 1.0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] < 0.7
    assert m.disqualifier_scores['filter bank'] == 28
    assert m.get_filter_description() == 'Streaked (angle= 25.0 deg)'
    assert m.background < 1.0  # see TODO above
    assert m.background_err < 3.0  # TODO: above
