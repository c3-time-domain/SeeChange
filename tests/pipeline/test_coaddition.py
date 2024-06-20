import pytest
import uuid

from astropy.io import fits

import numpy as np
from numpy.fft import fft2, ifft2, fftshift

from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from improc.simulator import Simulator
from improc.tools import sigma_clipping

from pipeline.coaddition import Coadder, CoaddPipeline
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator


def estimate_psf_width(data, sz=15, upsampling=25):
    """Extract a bright star and estimate its FWHM.

    This is a very rough-and-dirty method used only for testing.

    Assumes the data array has NaNs at all masked pixel locations.

    Parameters
    ----------
    data: ndarray
        The image data.
    sz: int
        The size of the box to extract around the star.
        Default is 15.
    upsampling: int
        The factor by which to up-sample the PSF.
        Default is 25.

    Returns
    -------
    float
        The estimated FWHM.
    """
    data = data.copy()
    # add a nan border, so we can get a PSF not from the edge
    data[0:sz, :] = np.nan
    data[-sz:, :] = np.nan
    data[:, 0:sz] = np.nan
    data[:, -sz:] = np.nan

    psf = extract_psf_surrogate(data, sz=sz, upsampling=upsampling)
    flux = []
    area = []
    radii = np.array(range(1, psf.shape[0], 2))
    x, y = np.meshgrid(np.arange(psf.shape[0]), np.arange(psf.shape[1]))
    rmap = np.sqrt((x - psf.shape[0] // 2) ** 2 + (y - psf.shape[1] // 2) ** 2)

    for r in radii:
        mask = (rmap <= r + 1) & (rmap > r - 1)
        area.append(np.sum(mask))
        flux.append(np.sum(psf[mask]))

    flux = np.array(flux)
    area = np.array(area, dtype=float)
    area[area == 0] = np.nan
    flux_n = flux / area  # normalize by the area of the annulus

    # go over the flux difference curve and find where it drops below half the peak flux:
    peak = np.nanmax(flux_n)
    idx = np.where(flux_n <= peak / 2)[0][0]

    fwhm = radii[idx] * 2 / upsampling

    return fwhm


def extract_psf_surrogate(data, sz=15, upsampling=25):
    """Extract a rough estimate for the PSF from the brightest (non-flagged) star in the image.

    This is a very rough-and-dirty method used only for testing.

    Assumes the data array has NaNs at all masked pixel locations.

    Parameters
    ----------
    data: ndarray
        The image data.
    sz: int
        The size of the box to extract around the star.
        Default is 15.
    upsampling: int
        The factor by which to up-sample the PSF.
        Default is 25.

    Returns
    -------
    ndarray
        The PSF surrogate.
    """
    # find the brightest star in the image:
    y, x = np.unravel_index(np.nanargmax(data), data.shape)
    data[np.isnan(data)] = 0

    # extract a 21x21 pixel box around the star:
    edge_x1 = max(0, x - sz)
    edge_x2 = min(data.shape[1], x + sz)
    edge_y1 = max(0, y - sz)
    edge_y2 = min(data.shape[0], y + sz)

    psf = data[edge_y1:edge_y2, edge_x1:edge_x2]

    # up-sample the PSF by the given factor:
    psf = ifft2(fftshift(np.pad(fftshift(fft2(psf)), sz*upsampling))).real
    if np.sum(psf) == 0:
        raise RuntimeError("PSF is all zeros")
    psf /= np.sum(np.abs(psf))

    if all(np.isnan(psf.flatten())):
        raise RuntimeError("PSF is all NaNs")

    # roll the psf so the max is at the center of the image:
    y, x = np.unravel_index(np.nanargmax(psf), psf.shape)
    psf = np.roll(psf, psf.shape[0] // 2 - y, axis=0)
    psf = np.roll(psf, psf.shape[1] // 2 - x, axis=1)

    return psf


@pytest.mark.flaky(max_runs=3)
def test_zogy_simulation(coadder, blocking_plots):
    num_images = 10
    sim = Simulator(
        image_size_x=256,  # make smaller images to make the test faster
        star_number=100,  # the smaller images require a smaller number of stars to avoid crowding
        seeing_mean=5.0,
        seeing_std=1.0,
        seeing_minimum=0.5,
        gain_std=0,  # leave the gain at 1.0
        read_noise=1,
        optic_psf_pars={'sigma': 0.1},  # make the optical PSF much smaller than the seeing
    )
    images = []
    weights = []
    flags = []
    truths = []
    psfs = []
    fwhms = []
    zps = []
    bkg_means = []
    bkg_stds = []
    for i in range(num_images):
        sim.make_image(new_sky=True, new_stars=False)
        images.append(sim.apply_bias_correction(sim.image))
        weights.append(np.ones_like(sim.image, dtype=float))
        flags.append(np.zeros_like(sim.image, dtype=np.int16))
        flags[-1][100, 100] = 1  # just to see what happens to a flagged pixel
        truths.append(sim.truth)
        psfs.append(sim.truth.psf_downsampled)
        fwhms.append(sim.truth.atmos_psf_fwhm)
        zps.append(sim.truth.transmission_instance)
        bkg_means.append(sim.truth.background_instance)
        bkg_stds.append(np.sqrt(sim.truth.background_instance + sim.truth.read_noise ** 2))

    # figure out the width of the PSFs in the original images:
    fwhms_est = []
    for im, fl in zip(images, flags):
        im = im.copy()
        im[fl > 0] = np.nan
        fwhms_est.append(estimate_psf_width(im))

    # check that fwhm estimator is ballpark correct:
    fwhms = np.array(fwhms)
    fwhms_est = np.array(fwhms_est)
    fwhms_est2 = np.sqrt(fwhms_est ** 2 - 1.5)  # add the pixelization width
    deltas = np.abs((fwhms - fwhms_est2) / fwhms)
    # SCLogger.debug(
    #     f'max(deltas) = {np.max(deltas) * 100:.1f}%, '
    #     f'mean(deltas) = {np.mean(deltas) * 100:.1f}%, '
    #     f'std(deltas)= {np.std(deltas) * 100 :.1f}% '
    # )
    assert np.all(deltas < 0.3)  # the estimator should be within 30% of the truth

    # now that we know the estimator is good, lets check the coadded images vs. the originals:
    outim, outwt, outfl, outpsf, score = coadder._coadd_zogy(  # calculate the ZOGY coadd
        images,
        weights,
        flags,
        psfs,
        fwhms,
        zps,
        bkg_means,
        bkg_stds,
    )

    assert outim.shape == (256, 256)
    assert outwt.shape == (256, 256)
    assert outfl.shape == (256, 256)

    assert np.sum(outfl) > 1  # there should be more than one flagged pixel (PSF splash)
    assert np.sum(outfl) < 200  # there should be fewer than 200 flagged pixels

    zogy_im_nans = outim.copy()
    zogy_im_nans[outfl > 0] = np.nan
    zogy_fwhm = estimate_psf_width(zogy_im_nans)

    assert zogy_fwhm < np.quantile(fwhms_est, 0.75)  # the ZOGY PSF should be narrower than most original PSFs

    mu, sigma = sigma_clipping(outim)
    assert abs(mu) == pytest.approx(0, abs=0.5)  # the coadd should be centered on zero
    assert sigma == pytest.approx(1.0, abs=0.1)  # the coadd should have a background rms of about 1

    if blocking_plots:
        import matplotlib.pyplot as plt

        mx = max(np.max(fwhms), np.max(fwhms_est))
        plt.plot(fwhms, fwhms_est, 'b.', label='FWHM estimate')
        plt.plot(fwhms, np.sqrt(fwhms_est ** 2 - 1.5), 'r.', label='pixelization corrected')
        plt.plot([0, mx], [0, mx], 'k--')
        plt.xlabel('FWHM truth [pixels]')
        plt.ylabel('FWHM estimate [pixels]')
        plt.legend()
        plt.show(block=True)

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(images[0], vmin=0, vmax=100)
        ax[0, 0].set_title('original 0')
        ax[0, 1].imshow(images[1], vmin=0, vmax=100)
        ax[0, 1].set_title('original 1')

        ax[1, 0].imshow(outim)
        ax[1, 0].set_title('coadded')

        ax[1, 1].imshow(score)
        ax[1, 1].set_title('score')

        plt.show(block=True)


def test_zogy_vs_naive(ptf_aligned_images, coadder):
    assert all([im.psf is not None for im in ptf_aligned_images])
    assert all([im.zp is not None for im in ptf_aligned_images])

    naive_im, naive_wt, naive_fl = coadder._coadd_naive(ptf_aligned_images)

    zogy_im, zogy_wt, zogy_fl, zogy_psf, zogy_score = coadder._coadd_zogy(ptf_aligned_images)

    assert naive_im.shape == zogy_im.shape

    # ZOGY must dilate the bad pixels to account for PSF match-filtering:
    assert np.sum(naive_fl == 1) < np.sum(zogy_fl == 1)  # more bad pixels
    assert np.sum(naive_fl == 0) > np.sum(zogy_fl == 0)  # less good pixels

    mu, sigma = sigma_clipping(zogy_im)
    assert abs(mu) == pytest.approx(0, abs=0.5)  # the coadd should be centered on zero
    assert sigma == pytest.approx(1.0, abs=0.2)  # the coadd should have a background rms of about 1

    # get the FWHM estimate for the regular images and for the coadd
    fwhms = []
    for im in ptf_aligned_images:
        im_nan = im.data.copy()
        im_nan[im.flags > 0] = np.nan
        fwhms.append(estimate_psf_width(im_nan))

    fwhms = np.array(fwhms)

    zogy_im_nans = zogy_im.copy()
    zogy_im_nans[zogy_fl > 0] = np.nan
    zogy_fwhm = estimate_psf_width(zogy_im_nans)
    naive_im_nans = naive_im.copy()
    naive_im_nans[naive_fl > 0] = np.nan
    naive_fwhm = estimate_psf_width(naive_im_nans)

    assert all(zogy_fwhm <= fwhms)  # the ZOGY PSF should be narrower than original PSFs
    assert zogy_fwhm < naive_fwhm


def test_coaddition_run(coadder, ptf_reference_images, ptf_aligned_images):
    # first make sure the "naive" coadd method works
    coadder.pars.test_parameter = uuid.uuid4().hex
    coadder.pars.method = 'naive'

    ref_image = coadder.run(ptf_reference_images, ptf_aligned_images)
    ref_image.provenance.is_testing = True

    # now check that ZOGY works and verify the output
    coadder.pars.test_parameter = uuid.uuid4().hex
    coadder.pars.method = 'zogy'

    ref_image = coadder.run(ptf_reference_images, ptf_aligned_images)
    ref_image.provenance.is_testing = True

    assert isinstance(ref_image, Image)
    assert ref_image.filepath is None
    assert ref_image.type == 'ComSci'
    assert ref_image.provenance.id != ptf_reference_images[0].provenance.id
    assert ref_image.instrument == 'PTF'
    assert ref_image.telescope == 'P48'
    assert ref_image.filter == 'R'
    assert str(ref_image.section_id) == '11'

    assert isinstance(ref_image.info, dict)
    assert isinstance(ref_image.header, fits.Header)

    # check a random value from the header, should have been taken from the last image
    assert ref_image.header['TELDEC'] == ptf_reference_images[-1].header['TELDEC']
    # the coordinates have also been grabbed from the last image
    assert ref_image.ra == ptf_reference_images[-1].ra
    assert ref_image.dec == ptf_reference_images[-1].dec
    assert ref_image.ra_corner_00 == ptf_reference_images[-1].ra_corner_00  # check one of the corners

    assert ref_image.start_mjd == min([im.start_mjd for im in ptf_reference_images])
    assert ref_image.end_mjd == max([im.end_mjd for im in ptf_reference_images])
    assert ref_image.exp_time == sum([im.exp_time for im in ptf_reference_images])

    assert ref_image.is_coadd
    assert not ref_image.is_sub
    assert ref_image.exposure_id is None
    assert ref_image.exposure is None

    assert ref_image.upstream_images == ptf_reference_images
    assert ref_image.ref_image_id == ptf_reference_images[-1].id
    assert ref_image.new_image is None

    assert ref_image.data is not None
    assert ref_image.data.shape == ptf_reference_images[0].data.shape
    assert ref_image.weight is not None
    assert ref_image.weight.shape == ref_image.data.shape
    assert ref_image.flags is not None
    assert ref_image.flags.shape == ref_image.data.shape
    assert ref_image.zogy_psf is not None
    assert ref_image.zogy_score is not None
    assert ref_image.zogy_score.shape == ref_image.data.shape


def test_coaddition_pipeline_inputs(ptf_reference_images):
    pipe = CoaddPipeline()
    assert pipe.pars.date_range == 7
    assert isinstance(pipe.coadder, Coadder)
    assert pipe.coadder.pars.method == 'zogy'
    assert isinstance(pipe.extractor, Detector)
    assert pipe.extractor.pars.threshold == 3.0
    assert isinstance(pipe.astrometor, AstroCalibrator)
    assert pipe.astrometor.pars.max_catalog_mag == [22.0]
    assert isinstance(pipe.photometor, PhotCalibrator)
    assert pipe.photometor.pars.max_catalog_mag == [22.0]

    # make a new pipeline with modified parameters
    pipe = CoaddPipeline(pipeline={'date_range': 5}, coaddition={'method': 'naive'})
    assert pipe.pars.date_range == 5
    assert isinstance(pipe.coadder, Coadder)
    assert pipe.coadder.pars.method == 'naive'

    # now modify it after initialization:
    pipe.coadder.pars.method = 'zogy'
    assert pipe.coadder.pars.method == 'zogy'

    # now run the pipeline:
    pipe.parse_inputs(ptf_reference_images)
    assert pipe.images == ptf_reference_images

    # make sure you can grab these using the target and other parameters:
    pipe.parse_inputs(
        target="100014",
        instrument="PTF",
        filter="R",
        section_id="11",
        provenance_ids=ptf_reference_images[0].provenance_id,
    )

    # without giving a start/end time, all these images will not be selected!
    assert len(pipe.images) == 0

    # try with a time too far in the past
    pipe.parse_inputs(
        target="100014",
        instrument="PTF",
        filter="R",
        section_id="11",
        provenance_ids=ptf_reference_images[0].provenance_id,
        start_time='2000-01-01',
        end_time='2007-01-01',
    )
    assert len(pipe.images) == 0

    # without an end_time, should use "now" so it would include the images
    pipe.parse_inputs(
        target="100014",
        instrument="PTF",
        filter="R",
        section_id="11",
        provenance_ids=ptf_reference_images[0].provenance_id,
        start_time='2000-01-01',
    )
    im_ids = set([im.id for im in pipe.images])
    ptf_im_ids = set([im.id for im in ptf_reference_images])
    assert ptf_im_ids == im_ids

    ptf_ras = [im.ra for im in ptf_reference_images]
    ptf_decs = [im.dec for im in ptf_reference_images]
    center_ra = np.mean(ptf_ras)
    center_dec = np.mean(ptf_decs)

    # make sure we can grab these images using coordinates as well:
    pipe.parse_inputs(
        target=None,
        ra=center_ra,
        dec=center_dec,
        instrument="PTF",
        filter="R",
        section_id="11",
        provenance_ids=ptf_reference_images[0].provenance_id,
        start_time='2000-01-01',
    )

    im_ids = set([im.id for im in pipe.images])
    ptf_im_ids = set([im.id for im in ptf_reference_images])
    assert ptf_im_ids == im_ids


def test_coaddition_pipeline_outputs(ptf_reference_images, ptf_aligned_images):
    try:
        pipe = CoaddPipeline()
        coadd_image = pipe.run(ptf_reference_images, ptf_aligned_images)

        # check that the second list input was ingested
        assert pipe.aligned_images == ptf_aligned_images

        assert isinstance(coadd_image, Image)
        assert coadd_image.filepath is None
        assert coadd_image.type == 'ComSci'
        assert coadd_image.provenance.id != ptf_reference_images[0].provenance.id
        assert coadd_image.instrument == 'PTF'
        assert coadd_image.telescope == 'P48'
        assert coadd_image.filter == 'R'
        assert str(coadd_image.section_id) == '11'
        assert coadd_image.start_mjd == min([im.start_mjd for im in ptf_reference_images])
        assert coadd_image.end_mjd == max([im.end_mjd for im in ptf_reference_images])
        assert coadd_image.provenance_id is not None
        assert coadd_image.aligned_images == ptf_aligned_images  # use the same images from the input to pipeline

        # check that all output products are there
        assert isinstance(coadd_image.sources, SourceList)
        assert isinstance(coadd_image.psf, PSF)
        assert isinstance(coadd_image.wcs, WorldCoordinates)
        assert isinstance(coadd_image.zp, ZeroPoint)

        # check that the ZOGY PSF width is similar to the PSFex result
        assert np.max(coadd_image.zogy_psf) == pytest.approx(np.max(coadd_image.psf.get_clip()), abs=0.01)
        zogy_fwhm = estimate_psf_width(coadd_image.zogy_psf)
        psfex_fwhm = estimate_psf_width(np.pad(coadd_image.psf.get_clip(), 20))  # pad so extract_psf_surrogate works
        assert zogy_fwhm == pytest.approx(psfex_fwhm, rel=0.1)

        # check that the S/N is consistent with a coadd
        flux_zp = [10 ** (0.4 * im.zp.zp) for im in ptf_reference_images]  # flux in ADU of a magnitude 0 star
        bkgs = [im.bkg_rms_estimate for im in ptf_reference_images]
        snrs = np.array(flux_zp) / np.array(bkgs)
        mean_snr = np.mean(snrs)

        flux_zp_zogy = 10 ** (0.4 * coadd_image.zp.zp)
        _, bkg_zogy = sigma_clipping(coadd_image.data)
        snr_zogy = flux_zp_zogy / bkg_zogy

        # zogy background noise is normalized by construction
        assert bkg_zogy == pytest.approx(1.0, abs=0.1)

        # S/N should be sqrt(N) better # TODO: why is the zogy S/N 15% better than expected??
        assert snr_zogy == pytest.approx(mean_snr * np.sqrt(len(ptf_reference_images)), rel=0.2)

    finally:
        if 'coadd_image' in locals():
            coadd_image.delete_from_disk_and_database(commit=True, remove_downstreams=True)


def test_coadded_reference(ptf_ref):
    ref_image = ptf_ref.image
    assert isinstance(ref_image, Image)
    assert ref_image.filepath is not None
    assert ref_image.type == 'ComSci'
    assert isinstance(ref_image.sources, SourceList)
    assert isinstance(ref_image.psf, PSF)
    assert isinstance(ref_image.wcs, WorldCoordinates)
    assert isinstance(ref_image.zp, ZeroPoint)

    assert ptf_ref.target == ref_image.target
    assert ptf_ref.filter == ref_image.filter
    assert ptf_ref.section_id == ref_image.section_id

    assert ptf_ref.validity_start is None
    assert ptf_ref.validity_end is None

    assert ptf_ref.provenance.upstreams[0].id == ref_image.provenance_id
    assert ptf_ref.provenance.process == 'reference'

    assert ptf_ref.provenance.parameters['test_parameter'] == 'test_value'

