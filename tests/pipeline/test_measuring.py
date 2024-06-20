import time

import pytest
import uuid

import numpy as np

from models.base import SmartSession

from improc.tools import make_gaussian


@pytest.mark.flaky(max_runs=3)
def test_measuring(measurer, decam_cutouts, decam_default_calibrators):
    measurer.pars.test_parameter = uuid.uuid4().hex
    measurer.pars.bad_pixel_exclude = ['saturated']  # ignore saturated pixels
    measurer.pars.bad_flag_exclude = ['satellite']  # ignore satellite cutouts

    sz = decam_cutouts[0].sub_data.shape
    fwhm = decam_cutouts[0].sources.image.get_psf().fwhm_pixels

    # clear any flags for the fake data we are using
    for i in range(14):
        decam_cutouts[i].sub_flags = np.zeros_like(decam_cutouts[i].sub_flags)
        # decam_cutouts[i].filepath = None  # make sure the cutouts don't re-load the original data
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
    decam_cutouts[11].sub_data = make_gaussian(imsize=sz[0], sigma_x=fwhm / 2.355, sigma_y=20, rotation=25, norm=1)
    decam_cutouts[11].sub_data *= 1000
    decam_cutouts[11].sub_data += np.random.normal(0, 1, size=sz)

    # a regular cutout but we'll put some bad flag on the cutout
    decam_cutouts[12].badness = 'cosmic ray'

    # a regular cutout with a bad flag that we are ignoring:
    decam_cutouts[13].badness = 'satellite'

    # run the measurer
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
    assert m.flux_psf > 150  # flux is more focused than the PSF, so it will bias the flux to be higher than 100
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
    assert m.flux_psf > 300  # flux is more focused than the PSF, so it will bias the flux to be higher than 100
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
    assert m.flux_psf == pytest.approx(1000, rel=0.1)
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
    assert m.flux_psf == pytest.approx(500, rel=0.1)
    assert m.background == pytest.approx(0, abs=0.01)
    assert m.background_err == pytest.approx(0, abs=0.01)

    m = ds.all_measurements[4]  # dipole
    assert m.disqualifier_scores['negatives'] == pytest.approx(1.0, abs=0.1)
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] > 100
    assert m.disqualifier_scores['filter bank'] > 0

    # the dipole's large offsets will short-circuit the iterative repositioning of the aperture (should be flagged!)
    assert all(np.isnan(m.flux_apertures))
    assert all(np.isnan(m.area_apertures))
    assert m.background == 0
    assert m.background_err == 0
    assert m.background_err == 0

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
    assert m.flux_psf < 500  # flux is more spread out than the PSF, so it will bias the flux to be lower

    assert m.background == pytest.approx(0, abs=0.2)
    assert m.background_err == pytest.approx(1.0, abs=0.2)

    m = ds.all_measurements[11]  # streak
    assert m.disqualifier_scores['negatives'] < 0.5
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] < 0.7
    assert m.disqualifier_scores['filter bank'] == 28
    assert m.get_filter_description() == 'Streaked (angle= 25.0 deg)'
    assert m.background < 0.5
    assert m.background_err < 3.0

    m = ds.all_measurements[12]  # regular cutout with a bad flag
    assert m.disqualifier_scores['bad_flag'] == 2 ** 41  # this is the bit for 'cosmic ray'

    m = ds.all_measurements[13]  # regular cutout with a bad flag that we are ignoring
    assert m.disqualifier_scores['bad_flag'] == 0  # we've included the satellite flag in the ignore list

    # check that coordinates have been modified:
    for i in range(14):
        m = ds.all_measurements[i]
        if m.offset_x != 0 and m.offset_y != 0:
            assert m.ra != m.cutouts.ra
            assert m.dec != m.cutouts.dec


def test_propagate_badness(decam_datastore):
    ds = decam_datastore
    with SmartSession() as session:
        ds.measurements[0].badness = 'cosmic ray'
        # find the index of the cutout that corresponds to the measurement
        idx = [i for i, c in enumerate(ds.cutouts) if c.id == ds.measurements[0].cutouts_id][0]
        ds.cutouts[idx].badness = 'cosmic ray'
        ds.cutouts[idx].update_downstream_badness(session=session)
        m = session.merge(ds.measurements[0])

        assert m.badness == 'cosmic ray'  # note that this does not change disqualifier_scores!


def test_warnings_and_exceptions(decam_datastore, measurer):
    measurer.pars.inject_warnings = 1

    with pytest.warns(UserWarning) as record:
        measurer.run(decam_datastore)
    assert len(record) > 0
    assert any("Warning injected by pipeline parameters in process 'measuring'." in str(w.message) for w in record)

    measurer.pars.inject_exceptions = 1
    measurer.pars.inject_warnings = 0
    with pytest.raises(Exception) as excinfo:
        ds = measurer.run(decam_datastore)
        ds.reraise()
    assert "Exception injected by pipeline parameters in process 'measuring'." in str(excinfo.value)
    ds.read_exception()

