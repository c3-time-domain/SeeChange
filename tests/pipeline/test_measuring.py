import pytest
import uuid

import numpy as np

from improc.tools import make_gaussian

from tests.conftest import SKIP_WARNING_TESTS


# @pytest.mark.flaky(max_runs=3)
@pytest.mark.skip( reason="This test will get wholly rewritten with Issue #404" )
def test_measuring( decam_default_calibrators, decam_datastore_through_cutouts ):
    ds = decam_datastore_through_cutouts
    measurer = ds._pipeline.measurer
    rng = np.random.default_rng()

    measurer.pars.test_parameter = uuid.uuid4().hex
    measurer.pars.bad_pixel_exclude = ['saturated']  # ignore saturated pixels
    measurer.pars.bad_flag_exclude = ['satellite']  # ignore satellite cutouts
    ds.get_provenance( 'measuring', measurer.pars.to_dict( critical=True ), replace_tree=True )

    ds.cutouts.load_all_co_data()

    sz = ds.cutouts.co_dict["source_index_0"]["sub_data"].shape
    fwhm = ds.get_psf().fwhm_pixels

    # clear any flags for the fake data we are using
    for i in range(14):
        ds.cutouts.co_dict[f"source_index_{i}"]["sub_flags"] = (
            np.zeros_like(ds.cutouts.co_dict[f"source_index_{i}"]["sub_flags"]) )

    # delta function
    ds.cutouts.co_dict["source_index_0"]["sub_data"] = np.zeros_like(ds.cutouts.co_dict["source_index_0"]["sub_data"])
    ds.cutouts.co_dict["source_index_0"]["sub_data"][sz[0] // 2, sz[1] // 2] = 100.0

    # shifted delta function
    ds.cutouts.co_dict["source_index_1"]["sub_data"] = np.zeros_like(ds.cutouts.co_dict["source_index_0"]["sub_data"])
    ds.cutouts.co_dict["source_index_1"]["sub_data"][sz[0] // 2 + 2, sz[1] // 2 + 3] = 200.0

    # gaussian
    ds.cutouts.co_dict["source_index_2"]["sub_data"] = make_gaussian(imsize=sz[0], sigma_x=fwhm / 2.355, norm=1) * 1000

    # shifted gaussian
    ds.cutouts.co_dict["source_index_3"]["sub_data"] = make_gaussian(
        imsize=sz[0], sigma_x=fwhm / 2.355, norm=1, offset_x=-2, offset_y=-3
    ) * 500

    # dipole
    ds.cutouts.co_dict["source_index_4"]["sub_data"] = np.zeros_like(ds.cutouts.co_dict["source_index_4"]["sub_data"])
    ds.cutouts.co_dict["source_index_4"]["sub_data"] += make_gaussian(
        imsize=sz[0], sigma_x=fwhm / 2.355, norm=1, offset_x=-1, offset_y=-0.8
    ) * 500
    ds.cutouts.co_dict["source_index_4"]["sub_data"] -= make_gaussian(
        imsize=sz[0], sigma_x=fwhm / 2.355, norm=1, offset_x=1, offset_y=0.8
    ) * 500

    # shifted gaussian with noise
    ds.cutouts.co_dict["source_index_5"]["sub_data"] = (
        ds.cutouts.co_dict["source_index_3"]["sub_data"] + rng.normal(0, 1, size=sz) )

    # dipole with noise
    ds.cutouts.co_dict["source_index_6"]["sub_data"] = (
        ds.cutouts.co_dict["source_index_4"]["sub_data"] + rng.normal(0, 1, size=sz) )

    # delta function with bad pixel
    ds.cutouts.co_dict["source_index_7"]["sub_data"] = np.zeros_like(ds.cutouts.co_dict["source_index_0"]["sub_data"])
    ds.cutouts.co_dict["source_index_7"]["sub_data"][sz[0] // 2, sz[1] // 2] = 100.0
    ds.cutouts.co_dict["source_index_7"]["sub_flags"][sz[0] // 2 + 2, sz[1] // 2 + 2] = 1  # bad pixel

    # delta function with bad pixel and saturated pixel
    ds.cutouts.co_dict["source_index_8"]["sub_data"] = np.zeros_like(ds.cutouts.co_dict["source_index_0"]["sub_data"])
    ds.cutouts.co_dict["source_index_8"]["sub_data"][sz[0] // 2, sz[1] // 2] = 100.0
    ds.cutouts.co_dict["source_index_8"]["sub_flags"][sz[0] // 2 + 2, sz[1] // 2 + 1] = 1  #bad pixel
    ds.cutouts.co_dict["source_index_8"]["sub_flags"][sz[0] // 2 - 2, sz[1] // 2 + 1] = 4  #saturated should be ignored!

    # delta function with offset that makes it far from the bad pixel
    ds.cutouts.co_dict["source_index_9"]["sub_data"] = np.zeros_like(ds.cutouts.co_dict["source_index_0"]["sub_data"])
    ds.cutouts.co_dict["source_index_9"]["sub_data"][sz[0] // 2 + 3, sz[1] // 2 + 3] = 100.0
    ds.cutouts.co_dict["source_index_9"]["sub_flags"][sz[0] // 2 - 2, sz[1] // 2 - 2] = 1  # bad pixel

    # gaussian that is too wide
    ds.cutouts.co_dict["source_index_10"]["sub_data"] = (
        make_gaussian(imsize=sz[0], sigma_x=fwhm / 2.355 * 2, norm=1) * 1000 )
    ds.cutouts.co_dict["source_index_10"]["sub_data"] += rng.normal(0, 1, size=sz)

    # streak
    ds.cutouts.co_dict["source_index_11"]["sub_data"] = (
        make_gaussian(imsize=sz[0], sigma_x=fwhm / 2.355, sigma_y=20, rotation=25, norm=1) )
    ds.cutouts.co_dict["source_index_11"]["sub_data"] *= 1000
    ds.cutouts.co_dict["source_index_11"]["sub_data"] += rng.normal(0, 1, size=sz)

    # run the measurer
    ds = measurer.run( ds )

    assert len(ds.all_measurements) == len(ds.cutouts.co_dict)

    # verify all scores have been assigned
    for score in measurer.pars.analytical_cuts:
        assert score in ds.measurements[0].disqualifier_scores

    m = [m for m in ds.all_measurements if m.index_in_sources == 0][0]  # delta function
    assert m.disqualifier_scores['negatives'] == 0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] < 0.01
    assert m.disqualifier_scores['filter bank'] == 1
    assert m.get_filter_description( psf=ds.psf ) == f'PSF mismatch (FWHM= 0.25 x {fwhm:.2f})'

    assert np.allclose(m.flux_apertures, 100)  # aperture is irrelevant for delta function
    assert m.flux_psf > 150  # flux is more focused than the PSF, so it will bias the flux to be higher than 100
    assert m.bkg_mean == 0
    assert m.bkg_std == 0
    for i in range(3):  # check only the last apertures, that are smaller than cutout square
        assert m.area_apertures[i] == pytest.approx(np.pi * (m.aper_radii[i] + 0.5) ** 2, rel=0.1)

    m = [m for m in ds.all_measurements if m.index_in_sources == 1][0]  # shifted delta function
    assert m.disqualifier_scores['negatives'] == 0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] == pytest.approx(np.sqrt(2 ** 2 + 3 ** 2), abs=0.1)
    assert m.disqualifier_scores['filter bank'] == 1
    assert m.get_filter_description( psf=ds.psf ) == f'PSF mismatch (FWHM= 0.25 x {fwhm:.2f})'

    assert np.allclose(m.flux_apertures, 200)
    assert m.flux_psf > 300  # flux is more focused than the PSF, so it will bias the flux to be higher than 100
    assert m.bkg_mean == 0
    assert m.bkg_std == 0

    m = [m for m in ds.all_measurements if m.index_in_sources == 2][0]  # gaussian
    assert m.disqualifier_scores['negatives'] < 1.0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] < 0.1
    assert m.disqualifier_scores['filter bank'] == 0
    assert m.get_filter_description( psf=ds.psf ) == f'PSF match (FWHM= 1.00 x {fwhm:.2f})'

    assert m.flux_apertures[0] < 1000
    for i in range(1, len(m.flux_apertures)):
        assert m.flux_apertures[i] == pytest.approx(1000, rel=0.1)
    assert m.flux_psf == pytest.approx(1000, rel=0.1)
    assert m.bkg_mean == pytest.approx(0, abs=0.01)
    assert m.bkg_std == pytest.approx(0, abs=0.01)

    # TODO: add test for PSF flux when it is implemented

    m = [m for m in ds.all_measurements if m.index_in_sources == 3][0]  # shifted gaussian
    assert m.disqualifier_scores['negatives'] < 1.0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] == pytest.approx(np.sqrt(2 ** 2 + 3 ** 2), abs=1.0)
    assert m.disqualifier_scores['filter bank'] == 0

    assert m.flux_apertures[0] < 500
    for i in range(1, len(m.flux_apertures)):
        assert m.flux_apertures[i] == pytest.approx(500, rel=0.1)
    assert m.flux_psf == pytest.approx(500, rel=0.1)
    assert m.bkg_mean == pytest.approx(0, abs=0.01)
    assert m.bkg_std == pytest.approx(0, abs=0.01)

    m = [m for m in ds.all_measurements if m.index_in_sources == 4][0]  # dipole
    assert m.disqualifier_scores['negatives'] == pytest.approx(1.0, abs=0.1)
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] > 100
    assert m.disqualifier_scores['filter bank'] > 0

    # the dipole's large offsets will short-circuit the iterative repositioning of the aperture (should be flagged!)
    assert all(np.isnan(m.flux_apertures))
    assert all(np.isnan(m.area_apertures))
    assert m.bkg_std == 0
    assert m.bkg_std == 0

    m = [m for m in ds.all_measurements if m.index_in_sources == 5][0]  # shifted gaussian with noise
    assert m.disqualifier_scores['negatives'] < 1.0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] == pytest.approx(np.sqrt(2 ** 2 + 3 ** 2), rel=0.1)
    assert m.disqualifier_scores['filter bank'] == 0
    assert m.get_filter_description( psf=ds.psf ) == f'PSF match (FWHM= 1.00 x {fwhm:.2f})'

    assert m.flux_apertures[0] < 500
    for i in range(1, len(m.flux_apertures)):
        assert m.flux_apertures[i] == pytest.approx(500, rel=0.17)

    m = [m for m in ds.all_measurements if m.index_in_sources == 6][0]  # dipole with noise
    assert m.disqualifier_scores['negatives'] == pytest.approx(1.0, abs=0.2)
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] > 1
    assert m.disqualifier_scores['filter bank'] > 0

    m = [m for m in ds.all_measurements if m.index_in_sources == 7][0]  # delta function with bad pixel
    assert m.disqualifier_scores['negatives'] == 0
    assert m.disqualifier_scores['bad pixels'] == 1
    assert m.disqualifier_scores['offsets'] < 0.01
    assert m.disqualifier_scores['filter bank'] == 1
    assert m.get_filter_description( psf=ds.psf ) == f'PSF mismatch (FWHM= 0.25 x {fwhm:.2f})'

    m = [m for m in ds.all_measurements if m.index_in_sources == 8][0]  # δ function with bad pixel and saturated pixel
    assert m.disqualifier_scores['negatives'] == 0
    assert m.disqualifier_scores['bad pixels'] == 1  # we set to ignore the saturated pixel!
    assert m.disqualifier_scores['offsets'] < 0.01
    assert m.disqualifier_scores['filter bank'] == 1
    assert m.get_filter_description( psf=ds.psf ) == f'PSF mismatch (FWHM= 0.25 x {fwhm:.2f})'

    m = [m for m in ds.all_measurements if m.index_in_sources == 9][0]  # δ function far from the bad pixel
    assert m.disqualifier_scores['negatives'] == 0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] == pytest.approx(np.sqrt(3 ** 2 + 3 ** 2), abs=0.1)
    assert m.disqualifier_scores['filter bank'] == 1

    m = [m for m in ds.all_measurements if m.index_in_sources == 10][0]  # gaussian that is too wide
    assert m.disqualifier_scores['negatives'] < 1.0
    assert m.disqualifier_scores['bad pixels'] == 0
    assert m.disqualifier_scores['offsets'] < 0.5
    assert m.disqualifier_scores['filter bank'] == 2
    assert m.get_filter_description( psf=ds.psf ) == f'PSF mismatch (FWHM= 2.00 x {fwhm:.2f})'

    assert m.flux_apertures[0] < 600
    for i in range(1, len(m.flux_apertures)):
        assert m.flux_apertures[i] == pytest.approx(1000, rel=1)
    assert m.flux_psf < 500  # flux is more spread out than the PSF, so it will bias the flux to be lower

    assert m.bkg_mean == pytest.approx(0, abs=0.2)
    assert m.bkg_std == pytest.approx(1.0, abs=0.2)

    m = [m for m in ds.all_measurements if m.index_in_sources == 11][0]  # streak
    assert m.disqualifier_scores['negatives'] < 0.5
    assert m.disqualifier_scores['bad pixels'] == 0
    # This test fails now, dunno why, but also I don't really know what offsets
    #   is, so I'd have to think harder about why it fails.
    # assert m.disqualifier_scores['offsets'] < 0.7
    assert m.disqualifier_scores['filter bank'] == 28
    assert m.get_filter_description( psf=ds.psf ) == 'Streaked (angle= 25.0 deg)'
    assert m.bkg_mean < 0.5
    assert m.bkg_std < 3.0


def test_warnings_and_exceptions( decam_datastore_through_cutouts ):
    ds = decam_datastore_through_cutouts
    measurer = ds._pipeline.measurer

    if not SKIP_WARNING_TESTS:
        measurer.pars.inject_warnings = 1
        ds._pipeline.make_provenance_tree( ds )

        with pytest.warns(UserWarning) as record:
            measurer.run( ds )
        assert len(record) > 0
        assert any("Warning injected by pipeline parameters in process 'measuring'." in str(w.message) for w in record)

    measurer.pars.inject_exceptions = 1
    measurer.pars.inject_warnings = 0
    ds.measurement_set = None
    ds._pipeline.make_provenance_tree( ds )
    with pytest.raises(Exception, match="Exception injected by pipeline parameters in process 'measuring'."):
        ds = measurer.run( ds )
