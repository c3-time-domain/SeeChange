import pytest
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from improc.tools import sigma_clipping, make_gaussian, make_cutouts

# os.environ['INTERACTIVE'] = '1'  # for diagnostics only

CUTOUT_SIZE = 15
BIG_CUTOUT_SIZE = 51
PSF_SIGMA = 1.5


def make_template_bank(imsize=15, psf_sigma=1.0):
    """Make some templates to check if the cutout contains a point-source or extended object.

    Parameters
    ----------
    imsize: int
        The size of the templates. Default is 25.
    psf_sigma: float
        The width of the PSF in pixels. Default is 2.0.

    Returns
    -------
    templates: list
        A list of templates. Each template is a 2D numpy array, of the same size as the cutouts.
        The first template is for a point-source, the rest are for extended sources and streaks.
        All templates are normalized by sqrt(sum(template**2)), so the matched-filter results
        can be compared directly.
        If the cutout is centered on the object, it is enough to just multiply by the template
        and sum the result. Otherwise, use cross-correlation.
    """
    templates = []
    templates.append(make_gaussian(imsize=imsize, sigma_x=psf_sigma, sigma_y=psf_sigma, norm=2))

    # narrow gaussian to trigger on cosmic rays
    templates.append(make_gaussian(imsize=imsize, sigma_x=psf_sigma * 0.5, sigma_y=psf_sigma * 0.5, norm=2))

    # bigger templates for extended sources
    templates.append(make_gaussian(imsize=imsize, sigma_x=psf_sigma * 2.5, sigma_y=psf_sigma * 2.5, norm=2))
    templates.append(make_gaussian(imsize=imsize, sigma_x=psf_sigma * 5.0, sigma_y=psf_sigma * 5.0, norm=2))
    templates.append(make_gaussian(imsize=imsize, sigma_x=psf_sigma * 10.0, sigma_y=psf_sigma * 10.0, norm=2))

    # add some streaks:
    (y, x) = np.meshgrid(range(-(imsize // 2), imsize // 2 + 1), range(-(imsize // 2), imsize // 2 + 1))
    for angle in np.arange(-90.0, 90.0, 5):

        if angle == 90:
            d = np.abs(x)  # distance from line
        else:
            a = np.tan(np.radians(angle))
            b = 0  # impact parameter is zero for centered streak
            d = np.abs(a * x - y + b) / np.sqrt(1 + a ** 2)  # distance from line
        streak = (1 / np.sqrt(2.0 * np.pi) / psf_sigma) * np.exp(
            -0.5 * d ** 2 / psf_sigma ** 2
        )
        streak /= np.sum(streak ** 2)  # verify that the template is normalized

        templates.append(streak)

    return templates


def test_detection_ptf_supernova(detector, ptf_subtraction1, blocking_plots, cache_dir):
    ds = detector.run(ptf_subtraction1)
    try:
        assert ds.detections is not None
        assert ds.detections.num_sources > 0
        ds.detections.save()

        if blocking_plots:
            ds.detections.show()
            plt.show(block=True)

        # make cutouts to see if we can filter out the bad subtractions
        data = ptf_subtraction1.nandata
        det = ds.detections.data
        cutouts = make_cutouts(data, det['x'], det['y'], size=CUTOUT_SIZE)
        big_cutouts = make_cutouts(data, det['x'], det['y'], size=BIG_CUTOUT_SIZE)
        hist_ok = np.ones((cutouts.shape[0],), dtype=bool)
        nans_ok = np.ones((cutouts.shape[0],), dtype=bool)
        templates = make_template_bank(imsize=CUTOUT_SIZE, psf_sigma=PSF_SIGMA)
        scores = np.zeros((cutouts.shape[0], (len(templates))))

        for i, cutout in enumerate(cutouts):
            mu, sigma = sigma_clipping(big_cutouts[i], nsigma=3, iterations=3)
            cutout = (cutout - mu) / sigma

            # analytical cuts:
            positives = np.sum(cutout > 3)
            negatives = np.sum(cutout < -3)
            if positives == 0 or negatives / positives > 0.3:  # too many negative pixels
                hist_ok[i] = False

            if np.sum(np.isnan(cutout)) > 0.1 * cutout.size:  # too many bad pixels
                nans_ok[i] = False

            cutout[np.isnan(cutout)] = 0
            cutouts[i] = cutout  # save this for later debugging

            # cross correlate with all templates:
            for j, temp in enumerate(templates):
                scores[i, j] = np.max(scipy.signal.correlate(abs(cutout), temp, mode='same'))

        # if the first template is the best match, it's a point source
        score_ok = np.argmax(scores, axis=1) == 0

        good = hist_ok & score_ok & nans_ok

        # try to find the supernova (this is PTF10cwm)
        # see: https://www.wiserep.org/object/7876
        # convert the coordinates from RA, Dec to pixel coordinates
        sn_coords = SkyCoord(188.230866 * u.deg, 4.48647 * u.deg)
        sn_x, sn_y = ds.image.wcs.wcs.world_to_pixel(sn_coords)

        coords = ds.image.wcs.wcs.pixel_to_world(det['x'], det['y'])
        sep = coords.separation(sn_coords).value
        mndx = np.argmin(sep)  # minimum index

        if blocking_plots:
            plt.imshow(data, vmin=-3, vmax=10)
            plt.plot(sn_x, sn_y, 'ro', fillstyle='none', label='PTF10cwm')
            square_xs = [
                det[mndx]['x'] + CUTOUT_SIZE // 2,
                det[mndx]['x'] - CUTOUT_SIZE // 2,
                det[mndx]['x'] - CUTOUT_SIZE // 2,
                det[mndx]['x'] + CUTOUT_SIZE // 2,
                det[mndx]['x'] + CUTOUT_SIZE // 2,
            ]
            square_ys = [
                det[mndx]['y'] + CUTOUT_SIZE // 2,
                det[mndx]['y'] + CUTOUT_SIZE // 2,
                det[mndx]['y'] - CUTOUT_SIZE // 2,
                det[mndx]['y'] - CUTOUT_SIZE // 2,
                det[mndx]['y'] + CUTOUT_SIZE // 2,
            ]
            plt.plot(square_xs, square_ys, 'g-', label=f'cutout {mndx}')
            plt.legend()
            plt.show(block=True)  # the SN's host galaxy is detected but the subtraction is too bad to see the SN

        # TODO: after cleaning up the subtraction, need to make sure at least
        #  one of the surviving detections is the supernova (has close enough coordinates).

    finally:
        ds.detections.delete_from_disk_and_database()


def test_warnings_and_exceptions(decam_datastore, detector):
    detector.pars.inject_warnings = 1

    with pytest.warns(UserWarning) as record:
        detector.run(decam_datastore)
    assert len(record) > 0
    assert any("Warning injected by pipeline parameters in process 'detection'." in str(w.message) for w in record)

    detector.pars.inject_warnings = 0
    detector.pars.inject_exceptions = 1
    with pytest.raises(Exception) as excinfo:
        ds = detector.run(decam_datastore)
        ds.reraise()
    assert "Exception injected by pipeline parameters in process 'detection'." in str(excinfo.value)
    ds.read_exception()