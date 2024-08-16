import os
import pytest

import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt

from models.base import CODE_ROOT, safe_mkdir
from improc.simulator import Simulator
from improc.zogy import zogy_subtract
from util.util import env_as_bool

imsize = 256

low_threshold = 4.31  # this is the maximum value we expect to get from a 256x256 image with unit noise
assert abs(scipy.special.erfc(low_threshold / np.sqrt(2)) * imsize ** 2 - 1) < 0.1
threshold = 6.01  # this should be high enough to avoid false positives at the 1/1000 level
assert scipy.special.erfc(threshold / np.sqrt(2)) * imsize ** 2 < 1e-3


@pytest.mark.flaky(max_runs=5)
def test_subtraction_no_stars():
    # this simulator creates images with the same b/g and seeing, so the stars will easily be subtracted
    sim = Simulator(
        image_size_x=imsize,  # not too big, but allow some space for stars
        background_std=0,  # keep background constant between images
        background_mean=1,  # keep the background to a simple value
        background_minimum=0.5,
        read_noise=0,  # make it easier to input the background noise (w/o source noise)
        dark_current=0,  # make it easier to input the background noise (w/o source noise)
        seeing_mean=2.0,  # make the seeing a little more pronounced
        seeing_std=0,  # keep seeing constant between images
        seeing_minimum=0.5,  # assume seeing never gets much better than this
        transmission_std=0,  # keep transmission constant between images
        pixel_qe_std=0,  # keep pixel QE constant between images
        gain_std=0,  # keep pixel gain constant between images
        star_number=0,
    )

    sim.pars.seeing_mean = 2.5
    sim.make_image(new_sky=True)
    im1 = sim.apply_bias_correction(sim.image)
    im1 -= sim.truth.background_instance
    truth1 = sim.truth

    sim.pars.seeing_mean = 2.5
    sim.make_image(new_sky=True)
    im2 = sim.apply_bias_correction(sim.image)
    im2 -= sim.truth.background_instance
    truth2 = sim.truth

    # what is the best value for the "flux-based zero point"?
    # the flux of a star needs to be this much to provide S/N=1
    # since the PSF is unit normalized, we only need to figure out how much noise in a measurement:
    # the sum over PSF squared times the noise variance gives the noise in a measurement
    F1 = 1 / np.sqrt(np.sum(truth1.psf_downsampled ** 2) * truth1.background_instance)
    F2 = 1 / np.sqrt(np.sum(truth2.psf_downsampled ** 2) * truth2.background_instance)

    # zogy_diff, zogy_psf, zogy_score, zogy_score_corr, alpha, alpha_err = zogy_subtract(
    output = zogy_subtract(
        im1,
        im2,
        truth1.psf_downsampled,
        truth2.psf_downsampled,
        np.sqrt(truth1.total_bkg_var),
        np.sqrt(truth2.total_bkg_var),
        F1,
        F2,
    )

    assert abs(np.std(output['sub_image']) - 1) < 0.1  # the noise should be unit variance
    assert np.max(abs(output['sub_image'])) < threshold  # we should not have anything get to the high threshold
    assert abs( np.max(abs(output['sub_image'])) - low_threshold ) < 1.5  # peak should be close to the low threshold

    # currently this doesn't work, I need to figure out the correct normalization for F_r and F_n
    # assert abs(np.std(zogy_score) - 1) < 0.1  # the noise should be unit variance
    # assert np.max(abs(zogy_score)) < threshold  # we should not have anything get to the high threshold
    # assert abs(np.max(abs(zogy_score)) - low_threshold) < 1.5  # some value should be close to the low threshold

    assert abs(np.std(output['score_corr']) - 1) < 0.1  # the noise should be unit variance
    assert np.max(abs(output['score_corr'])) < threshold  # we should not have anything get to the high threshold
    assert abs( np.max(abs(output['score_corr'])) - low_threshold ) < 1.5  # the peak should be close to the low threshold


# @pytest.mark.skip( reason="This test frequently fails even with the flaky.  Can we use a random seed?" )
@pytest.mark.flaky(max_runs=5)
def test_subtraction_no_new_sources():
    sim = Simulator(
        image_size_x=imsize,  # not too big, but allow some space for stars
        background_std=0,  # keep background constant between images
        read_noise=0,  # make it easier to input the background noise (w/o source noise)
        seeing_mean=2.0,  # make the seeing a little more pronounced
        seeing_std=0,  # keep seeing constant between images
        seeing_minimum=0.5,  # assume seeing never gets much better than this
        transmission_std=0,  # keep transmission constant between images
        pixel_qe_std=0,  # keep pixel QE constant between images
        gain_std=0,  # keep pixel gain constant between images
        saturation_limit=1e9,  # make it impossible to saturate
        star_number=1000,
    )

    seeing = np.arange(1.0, 3.0, 0.3)
    naive_successes = 0
    zogy_successes = 0
    zogy_failures = 0

    for which in ('R', 'N'):
        for i, s in enumerate(seeing):
            sim.pars.seeing_mean = s if which == 'R' else 1.5
            sim.make_image(new_sky=True)
            truth1 = sim.truth
            im1 = sim.apply_bias_correction(sim.image)
            im1 -= truth1.background_instance
            psf1 = truth1.psf_downsampled
            bkg1 = truth1.total_bkg_var

            sim.pars.seeing_mean = s if which == 'N' else 1.5
            sim.make_image(new_sky=True)
            truth2 = sim.truth
            im2 = sim.apply_bias_correction(sim.image)
            im2 -= truth2.background_instance
            psf2 = truth2.psf_downsampled
            bkg2 = truth2.total_bkg_var

            # need to figure out better values for this
            F1 = 1.0
            F2 = 1.0

            diff = im2 - im1
            diff /= np.sqrt(bkg1 + bkg2)  # adjust the image difference by the noise in both images

            # check that peaks in the matched filter image also obey the same statistics (i.e., we don't find anything)
            matched1 = scipy.signal.convolve(diff, psf1, mode='same') / np.sqrt(np.sum(psf1 ** 2))
            matched2 = scipy.signal.convolve(diff, psf2, mode='same') / np.sqrt(np.sum(psf2 ** 2))

            if (
                np.max(abs(diff)) <= threshold and
                np.max(abs(matched1)) <= threshold and
                np.max(abs(matched2)) <= threshold
            ):
                naive_successes += 1

            # now try ZOGY
            output = zogy_subtract(
                im1, im2, psf1, psf2, np.sqrt(bkg1), np.sqrt(bkg2), F1, F2,
            )

            # must ignore the edges where sometimes stars are off one image but on the other (if PSF is wide)
            edge = int(np.ceil(max(s, 1.5) * 2))
            if np.max(abs(output['score_corr'][edge:-edge, edge:-edge])) <= threshold:
                zogy_successes += 1
            else:
                zogy_failures += 1

    assert naive_successes == 0
    assert zogy_failures == 0


@pytest.mark.skipif( not env_as_bool('INTERACTIVE'), reason='Set INTERACTIVE to run this test' )
def test_subtraction_snr_histograms(blocking_plots):
    background = 5.0
    seeing = 3.0
    iterations = 300

    sim = Simulator(
        image_size_x=imsize,  # not too big, but allow some space for stars
        background_mean=background,  # some average background value
        background_std=0.0,  # keep background constant between images
        background_minimum=1.0,  # allow very low background for the reference image
        read_noise=0,  # make it easier to input the background noise (w/o source noise)
        seeing_mean=seeing,  # make the seeing a little more pronounced
        seeing_std=0,  # keep seeing constant between images
        seeing_minimum=0.1,  # assume seeing never gets much better than this
        transmission_std=0,  # keep transmission constant between images
        pixel_qe_std=0,  # keep pixel QE constant between images
        gain_std=0,  # keep pixel gain constant between images
        saturation_limit=1e9,  # make it impossible to saturate
        star_number=0,
    )

    # add a few sources
    fluxes = [100, 1000, 10000, 100000]
    pos = np.linspace(0, imsize, len(fluxes) + 1, endpoint=False)[1:]

    sim.make_stars()
    # make a new image, with transients
    for f, p in zip(fluxes, pos):
        sim.add_extra_stars(flux=f, x=p, y=p)

    sim.make_image()

    transients_overlay = sim.flux_top  # no noise, only the transients

    expected = np.zeros((iterations, len(fluxes)))
    measured = np.zeros((iterations, len(fluxes)))

    for j in range(iterations):
        psf1 = sim.psf_downsampled
        bkg1 = background  # TODO: make this slightly variable?
        im1 = np.random.normal(0, np.sqrt(bkg1), size=transients_overlay.shape)

        psf2 = sim.psf_downsampled
        bkg2 = background  # TODO: make this slightly variable?
        im2 = np.random.normal(transients_overlay, np.sqrt(bkg2 + transients_overlay))

        # need to figure out better values for this
        F1 = 1.0
        F2 = 1.0

        output = zogy_subtract(im1, im2, psf1, psf2, np.sqrt(bkg1), np.sqrt(bkg2), F1, F2)
        edge = 8  # do not trigger on stars too close to the edge
        S = np.ones(output['score_corr'].shape) * np.nan
        S[edge:-edge, edge:-edge] = output['score_corr'][edge:-edge, edge:-edge].copy()
        B = bkg1 + bkg2
        P = output['sub_psf']

        for i, f in enumerate(fluxes):
            x = y = int(pos[i])
            c = S[y - edge:y + edge, x - edge:x + edge]
            # In the denominator we replace B with B + f * np.sum(P ** 2) to account for
            # the combination of background variance with source-noise variance
            expected[j, i] = f * np.sqrt(np.sum(P ** 2) / (B + f * np.sum(P ** 2)))
            measured[j, i] = np.nanmax(c)

    # exp = np.mean(expected, axis=0)
    # mea = np.mean(measured, axis=0)
    # p1 = np.polyfit(exp, mea, 1)
    # SCLogger.debug(p1)

    matplotlib.rcParams["font.size"] = 22
    fig, axes = plt.subplots(2, 2, figsize=[14, 10], )
    for i, ax in enumerate(axes.flatten()):
        ax.hist(measured[:, i], bins=20, label='measured')
        ylim = ax.get_ylim()
        ax.plot(np.nanmean(expected[:, i])*np.array([1, 1]), ylim, label='expected')
        ax.set_title(f'flux: {fluxes[i]}')
        ax.set_xlabel('Measured S/N')
        ax.legend(loc='upper right')
        plt.subplots_adjust(hspace=0.5)
    plt.show(block=blocking_plots)

    filename = os.path.join(CODE_ROOT, "tests/plots/zogy_snr_histograms")
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".pdf")


# @pytest.mark.skip( reason="This test frequently fails even with the flaky.  Can we use a random seed?" )
@pytest.mark.flaky(max_runs=5)
def test_subtraction_new_sources_snr(blocking_plots):
    num_stars = 300
    sim = Simulator(
        image_size_x=imsize,  # not too big, but allow some space for stars
        background_std=0,  # keep background constant between images
        background_minimum=0.01,  # allow very low background for the reference image
        read_noise=0,  # make it easier to input the background noise (w/o source noise)
        seeing_mean=2.0,  # make the seeing a little more pronounced
        seeing_std=0,  # keep seeing constant between images
        seeing_minimum=0.1,  # assume seeing never gets much better than this
        transmission_std=0,  # keep transmission constant between images
        pixel_qe_std=0,  # keep pixel QE constant between images
        gain_std=0,  # keep pixel gain constant between images
        saturation_limit=1e9,  # make it impossible to saturate
        star_number=num_stars,
    )

    sim.pars.seeing_mean = 0.7  # good seeing reference image
    sim.pars.background_mean = 2.0  # low background reference image
    sim.make_image(new_sky=True)
    truth1 = sim.truth
    im1 = sim.apply_bias_correction(sim.image)
    im1 -= truth1.background_instance
    psf1 = truth1.psf_downsampled
    bkg1 = truth1.total_bkg_var

    # add a few sources
    fluxes = [100, 500, 1000, 2000, 5000, 10000, 50000, 100000]
    pos = np.linspace(0, imsize, len(fluxes) + 1, endpoint=False)[1:]
    for f, p in zip(fluxes, pos):
        sim.add_extra_stars(flux=f, x=p, y=p)

    sim.pars.seeing_mean = 2.5  # seeing is much worse in new image
    sim.pars.background_mean = 30.0  # background is much higher in new image

    iterations = 10
    expected = np.zeros((iterations, len(fluxes)))
    measured = np.zeros((iterations, len(fluxes)))

    for j in range(iterations):
        sim.make_image(new_sky=True)
        truth2 = sim.truth
        im2 = sim.apply_bias_correction(sim.image)
        im2 -= truth2.background_instance
        psf2 = truth2.psf_downsampled
        bkg2 = truth2.total_bkg_var

        # need to figure out better values for this
        F1 = 1.0
        F2 = 1.0

        output = zogy_subtract(
            im1, im2, psf1, psf2, np.sqrt(bkg1), np.sqrt(bkg2), F1, F2,
        )
        edge = 8  # do not trigger on stars too close to the edge
        S = np.ones(output['score_corr'].shape) * np.nan
        S[edge:-edge, edge:-edge] = output['score_corr'][edge:-edge, edge:-edge].copy()
        B = truth1.background_instance + truth2.background_instance
        P = output['sub_psf']

        for i, f in enumerate(fluxes):
            x = y = int(pos[i])
            c = S[y-edge:y+edge, x-edge:x+edge]
            # In the denominator we replace B with B + f * np.sum(P ** 2) to account for
            # the combination of background variance with source-noise variance
            expected[j, i] = f * np.sqrt(np.sum(P ** 2) / (B + f * np.sum(P ** 2)))
            measured[j, i] = np.nanmax(c)

    mean = np.nanmean(measured, axis=0)
    err = np.nanstd(measured, axis=0)
    exp = np.nanmean(expected, axis=0)
    p1 = np.polyfit(exp, mean, 1)

    chi2 = np.sum((mean - np.polyval(p1, exp)) ** 2 / err ** 2)
    assert chi2 / (len(fluxes) - 2) < 2.0  # the fit should be good
    assert p1[0] > 0.8  # should be close to one, but we are losing about 15% S/N and I don't know why

    if blocking_plots:
        plt.errorbar(exp, mean, err, fmt='.', label='measured vs. expected')
        plt.plot(exp, np.polyval(p1, exp), 'r-', label=f'fit: {p1[0]:.3f}x + {p1[1]:.3f}')
        plt.xlabel('expected S/N')
        plt.ylabel('measured S/N')
        plt.legend()
        plt.show(block=True)


# @pytest.mark.skip( reason="This test frequently fails even with the flaky.  Can we use a random seed?" )
@pytest.mark.flaky(max_runs=5)
def test_subtraction_seeing_background():
    num_stars = 300
    sim = Simulator(
        image_size_x=imsize,  # not too big, but allow some space for stars
        background_std=0,  # keep background constant between images
        background_minimum=0.01,  # allow very low background for the reference image
        read_noise=0,  # make it easier to input the background noise (w/o source noise)
        seeing_mean=2.0,  # make the seeing a little more pronounced
        seeing_std=0,  # keep seeing constant between images
        seeing_minimum=0.1,  # assume seeing never gets much better than this
        transmission_std=0,  # keep transmission constant between images
        pixel_qe_std=0,  # keep pixel QE constant between images
        gain_std=0,  # keep pixel gain constant between images
        saturation_limit=1e9,  # make it impossible to saturate
        star_number=num_stars,
    )
    seeing_values = [2.0, 3.0, 5.0]
    background_values = [1.0, 3.0, 10.0]

    fluxes = [300, 500, 1000]
    pos = np.linspace(0, imsize, len(fluxes) + 1, endpoint=False)[1:]

    sim.make_stars()

    # verify no stars are on top of our injection positions
    # ZOGY still detects the new source on top of the star, but the S/N is reduced
    for p in pos:
        for i in range(len(sim.stars.star_mean_fluxes)):
            if abs(p - sim.stars.star_mean_x_pos[i]) < 10:
                sim.stars.star_mean_x_pos[i] += 15  # bump this star out of the way
            if abs(p - sim.stars.star_mean_y_pos[i]) < 10:
                sim.stars.star_mean_y_pos[i] += 15  # bump this star out of the way

    # beware the 4-loop!
    for ref_seeing in seeing_values:
        for new_seeing in seeing_values:
            for ref_bkg in background_values:
                for new_bkg in background_values:
                    # SCLogger.debug(f'seeing: {ref_seeing}, {new_seeing} | bkg: {ref_bkg}, {new_bkg}')

                    sim.pars.seeing_mean = ref_seeing
                    sim.pars.background_mean = ref_bkg

                    # remove the additional stars from the reference image
                    sim.stars.remove_stars(len(sim.stars.star_mean_fluxes) - sim.pars.star_number)

                    sim.make_image(new_sky=True)
                    truth1 = sim.truth
                    im1 = sim.apply_bias_correction(sim.image)
                    im1 -= truth1.background_instance
                    psf1 = truth1.psf_downsampled
                    bkg1 = truth1.total_bkg_var

                    # add a few sources
                    for f, p in zip(fluxes, pos):
                        sim.add_extra_stars(flux=f, x=p, y=p)

                    sim.pars.seeing_mean = new_seeing
                    sim.pars.background_mean = new_bkg

                    sim.make_image(new_sky=True)
                    truth2 = sim.truth
                    im2 = sim.apply_bias_correction(sim.image)
                    im2 -= truth2.background_instance
                    psf2 = truth2.psf_downsampled
                    bkg2 = truth2.total_bkg_var

                    # need to figure out better values for this
                    F1 = 1.0
                    F2 = 1.0

                    output = zogy_subtract(im1, im2, psf1, psf2, np.sqrt(bkg1), np.sqrt(bkg2), F1, F2)

                    edge = 8  # do not trigger on stars too close to the edge
                    S = np.ones(output['score_corr'].shape) * np.nan
                    S[edge:-edge, edge:-edge] = output['score_corr'][edge:-edge, edge:-edge].copy()
                    B = ref_bkg + new_bkg
                    P = output['sub_psf']

                    for i, f in enumerate(fluxes):
                        x = y = int(pos[i])
                        c = S[y - edge:y + edge, x - edge:x + edge]
                        # In the denominator we replace B with B + f * np.sum(P ** 2) to account for
                        # the combination of background variance with source-noise variance
                        expected = f * np.sqrt(np.sum(P ** 2) / (B + f * np.sum(P ** 2)))
                        measured = np.nanmax(c)
                        # SCLogger.debug((expected - measured) / expected)
                        # TODO: figure out why we still have cases where the S/N is reduced by 35%
                        if abs((expected - measured) / expected) > 0.35:
                            raise ValueError(
                                f'seeing: ({ref_seeing:.2f}, {new_seeing:.2f}), '
                                f'background: ({ref_bkg:.2f}, {new_bkg:.2f}), '
                                f'expected/measured: ({expected:.3f}, {measured:.3f}), '
                                f'loss: {(expected - measured) / expected:.3f}'
                            )


@pytest.mark.flaky(max_runs=5)
def test_subtraction_jitter_noise():
    num_stars = 300
    ref_seeing = 2.0
    ref_bkg = 1.0
    new_seeing = 3.0
    new_bkg = 10.0
    sim = Simulator(
        image_size_x=imsize,  # not too big, but allow some space for stars
        background_mean=ref_bkg,  # the ref image has a low background
        background_std=0,  # keep background constant between images
        background_minimum=0.01,  # allow very low background for the reference image
        read_noise=0,  # make it easier to input the background noise (w/o source noise)
        seeing_mean=ref_seeing,  # make the seeing a little more pronounced
        seeing_std=0,  # keep seeing constant between images
        seeing_minimum=0.1,  # assume seeing never gets much better than this
        transmission_std=0,  # keep transmission constant between images
        pixel_qe_std=0,  # keep pixel QE constant between images
        gain_std=0,  # keep pixel gain constant between images
        saturation_limit=1e9,  # make it impossible to saturate
        star_number=num_stars,
    )

    jitter_values = [0.0, 0.1, 0.25, 0.5]
    fluxes = [300, 500, 1000]
    pos = np.linspace(0, imsize, len(fluxes) + 1, endpoint=False)[1:]

    sim.make_stars()
    # verify no stars are on top of our injection positions
    # ZOGY still detects the new source on top of the star, but the S/N is reduced
    for p in pos:
        for i in range(len(sim.stars.star_mean_fluxes)):
            if abs(p - sim.stars.star_mean_x_pos[i]) < 10:
                sim.stars.star_mean_x_pos[i] += 15  # bump this star out of the way
            if abs(p - sim.stars.star_mean_y_pos[i]) < 10:
                sim.stars.star_mean_y_pos[i] += 15  # bump this star out of the way

    sim.make_image(new_sky=True)
    truth1 = sim.truth
    im1 = sim.apply_bias_correction(sim.image)
    im1 -= truth1.background_instance
    psf1 = truth1.psf_downsampled
    bkg1 = truth1.total_bkg_var

    # add a few sources
    for f, p in zip(fluxes, pos):
        sim.add_extra_stars(flux=f, x=p, y=p)

    for jitter in jitter_values:
        # SCLogger.debug(f'jitter: {jitter}')
        sim.pars.seeing_mean = new_seeing  # a little worse seeing than the ref
        sim.pars.background_mean = new_bkg  # a little worse background than the ref
        sim.pars.star_position_std = jitter  # add some jitter to the star positions

        sim.make_image(new_sky=True)
        truth2 = sim.truth
        im2 = sim.apply_bias_correction(sim.image)
        im2 -= truth2.background_instance
        psf2 = truth2.psf_downsampled
        bkg2 = truth2.total_bkg_var

        # need to figure out better values for this
        F1 = 1.0
        F2 = 1.0

        output = zogy_subtract(im1, im2, psf1, psf2, np.sqrt(bkg1), np.sqrt(bkg2), F1, F2, dx=jitter)

        edge = 8  # do not trigger on stars too close to the edge
        S = np.ones(output['score_corr'].shape) * np.nan
        S[edge:-edge, edge:-edge] = output['score_corr'][edge:-edge, edge:-edge].copy()
        B = ref_bkg + new_bkg
        P = output['sub_psf']

        # check that the background stars are not detected
        S2 = S.copy()
        for i, f in enumerate(fluxes):
            x = y = int(pos[i])
            S2[y - edge:y + edge, x - edge:x + edge] = np.nan  # remove the injected sources

        assert np.nanmax(abs(S2)) < threshold  # we should not have anything get to the high threshold

        # check that the sources are recovered correctly
        for i, f in enumerate(fluxes):
            x = y = int(pos[i])
            c = S[y - edge:y + edge, x - edge:x + edge]
            # In the denominator we replace B with B + f * np.sum(P ** 2) to account for
            # the combination of background variance with source-noise variance
            expected = f * np.sqrt(np.sum(P ** 2) / (B + f * np.sum(P ** 2)))
            measured = np.nanmax(c)
            # SCLogger.debug((expected - measured) / expected)
            # TODO: figure out why we still have cases where the S/N is reduced by 50%
            if abs((expected - measured) / expected) > 0.5:
                raise ValueError(
                    f'seeing: ({ref_seeing:.2f}, {new_seeing:.2f}), '
                    f'background: ({ref_bkg:.2f}, {new_bkg:.2f}), '
                    f'expected/measured: ({expected:.3f}, {measured:.3f})'
                )
