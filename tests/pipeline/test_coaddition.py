import pytest
import uuid

from astropy.io import fits

import numpy as np
from numpy.fft import fft2, ifft2, fftshift

from models.base import Psycopg2Connection
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from improc.simulator import Simulator
from improc.tools import sigma_clipping

from pipeline.data_store import DataStore
from pipeline.coaddition import Coadder, CoaddPipeline

from util.util import env_as_bool


def estimate_psf_width(data, sz=7, upsampling=50, num_stars=20):
    """Extract a few bright stars and estimate their median FWHM.

    This is a very rough-and-dirty method used only for testing.

    Assumes the data array has NaNs at all masked pixel locations.

    Parameters
    ----------
    data: ndarray
        The image data.
    sz: int
        The size of the box to extract around the star.
        Default is 7.
    upsampling: int
        The factor by which to up-sample the PSF.
        Default is 50.
    num_stars: int
        The number of stars to use to estimate the FWHM.
        Default is 20.

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

    fwhms = []
    for i in range(num_stars):
        psf = extract_psf_surrogate(data, sz=sz, upsampling=upsampling)
        flux = []
        area = []
        radii = np.array(range(1, psf.shape[0] // 2, 2))
        x, y = np.meshgrid(np.arange(psf.shape[0]), np.arange(psf.shape[1]))
        rmap = np.sqrt((x - psf.shape[1] // 2) ** 2 + (y - psf.shape[0] // 2) ** 2)

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
        fwhms.append(fwhm)

    fwhm = np.nanmedian(fwhms)
    print(f'fwhm median= {fwhm}, fwhm_err= {np.std(fwhms)}')

    return fwhm


def extract_psf_surrogate(data, sz=7, upsampling=50):
    """Extract a rough estimate for the PSF from the brightest (non-flagged) star in the image.

    This is a very rough-and-dirty method used only for testing.

    Assumes the data array has NaNs at all masked pixel locations.

    Will mask the area of the chosen star so that the same array can be
    re-used to find progressively fainter stars.

    Parameters
    ----------
    data: ndarray
        The image data.
    sz: int
        The size of the box to extract around the star.
        Default is 7.
    upsampling: int
        The factor by which to up-sample the PSF.
        Default is 50.

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

    psf = data[edge_y1:edge_y2, edge_x1:edge_x2].copy()
    data[edge_y1:edge_y2, edge_x1:edge_x2] = np.nan  # can re-use this array to find other stars

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
    outim, outwt, outfl, _, score = coadder._coadd_zogy(  # calculate the ZOGY coadd
        images,
        weights=weights,
        flags=flags,
        psf_clips=psfs,
        psf_fwhms=fwhms,
        flux_zps=zps,
        bkg_means=bkg_means,
        bkg_sigmas=bkg_stds,
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

        _, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(images[0], vmin=0, vmax=100)
        ax[0, 0].set_title('original 0')
        ax[0, 1].imshow(images[1], vmin=0, vmax=100)
        ax[0, 1].set_title('original 1')

        ax[1, 0].imshow(outim)
        ax[1, 0].set_title('coadded')

        ax[1, 1].imshow(score)
        ax[1, 1].set_title('score')

        plt.show(block=True)


def test_zogy_vs_naive( ptf_aligned_image_datastores, coadder ):
    assert all( [d.bg is not None for d in ptf_aligned_image_datastores] )
    assert all( [d.psf is not None for d in ptf_aligned_image_datastores] )
    assert all( [d.zp is not None for d in ptf_aligned_image_datastores] )

    aligned_images = [ d.image for d in ptf_aligned_image_datastores ]
    aligned_bgs = [ d.bg for d in ptf_aligned_image_datastores ]
    aligned_psfs = [ d.psf for d in ptf_aligned_image_datastores ]
    aligned_zps = [ d.zp for d in ptf_aligned_image_datastores ]

    naive_im, _, naive_fl = coadder._coadd_naive( aligned_images )

    zogy_im, _, zogy_fl, _, _ = coadder._coadd_zogy( aligned_images,
                                                                     aligned_bgs,
                                                                     aligned_psfs,
                                                                     aligned_zps )

    assert naive_im.shape == zogy_im.shape

    # ZOGY must dilate the bad pixels to account for PSF match-filtering:
    assert np.sum(naive_fl == 1) < np.sum(zogy_fl == 1)  # more bad pixels
    assert np.sum(naive_fl == 0) > np.sum(zogy_fl == 0)  # less good pixels

    mu, sigma = sigma_clipping(zogy_im)
    assert abs(mu) == pytest.approx(0, abs=0.5)  # the coadd should be centered on zero
    assert sigma == pytest.approx(1.0, abs=0.2)  # the coadd should have a background rms of about 1

    # get the FWHM estimate for the regular images and for the coadd
    fwhms = []
    for im in aligned_images:
        # choose an area in the middle of the image
        fwhms.append(estimate_psf_width(im.nandata[1800:2600, 600:1400]))

    fwhms = np.array(fwhms)

    zogy_im_nans = zogy_im.copy()
    zogy_im_nans[zogy_fl > 0] = np.nan
    zogy_fwhm = estimate_psf_width(zogy_im_nans[1800:2600, 600:1400])
    naive_im_nans = naive_im.copy()
    naive_im_nans[naive_fl > 0] = np.nan
    naive_fwhm = estimate_psf_width(naive_im_nans[1800:2600, 600:1400])

    assert zogy_fwhm < np.mean(fwhms)  # the ZOGY PSF should be narrower than original PSFs
    assert zogy_fwhm < naive_fwhm


def test_coaddition_run(coadder, ptf_reference_image_datastores, ptf_aligned_image_datastores):
    refim0 = ptf_reference_image_datastores[0].image
    refimlast = ptf_reference_image_datastores[-1].image

    # first make sure the "naive" coadd method works
    coadder.pars.test_parameter = uuid.uuid4().hex
    coadder.pars.method = 'naive'

    ref_image = coadder.run( ptf_reference_image_datastores, aligned_datastores=ptf_aligned_image_datastores )

    # now check that ZOGY works and verify the output
    coadder.pars.test_parameter = uuid.uuid4().hex
    coadder.pars.method = 'zogy'

    ref_image = coadder.run( ptf_reference_image_datastores, aligned_datastores=ptf_aligned_image_datastores )

    assert isinstance(ref_image, Image)
    assert ref_image.filepath is None
    assert ref_image.type == 'ComSci'
    assert ref_image.provenance_id != refim0.provenance_id
    assert ref_image.instrument == 'PTF'
    assert ref_image.telescope == 'P48'
    assert ref_image.filter == 'R'
    assert str(ref_image.section_id) == '11'

    assert isinstance(ref_image.info, dict)
    assert isinstance(ref_image.header, fits.Header)

    # check a random value from the header, should have been taken from the last image
    assert ref_image.header['TELDEC'] == refimlast.header['TELDEC']
    # the coordinates have also been grabbed from the last image
    assert ref_image.ra == refimlast.ra
    assert ref_image.dec == refimlast.dec
    for coord in [ 'ra', 'dec' ]:
        for corner in [ '00', '01', '10', '11' ]:
            # Even though it should nominally should be exactly the same
            #   value, we need to pytest.approx because of conversions
            #   betwen numpy.float64 and float (and, if using the cache, string).
            #   (Plus the database has 32-bit floats for these.)
            assert ( getattr( ref_image, f'{coord}_corner_{corner}' ) ==
                     pytest.approx( getattr( refimlast, f'{coord}_corner_{corner}' ), abs=0.1/3600. ) )
        assert ( getattr ( ref_image, f'min{coord}' ) ==
                 pytest.approx( getattr( refimlast, f'min{coord}' ), abs=0.1/3600. ) )
        assert ( getattr ( ref_image, f'max{coord}' ) ==
                 pytest.approx( getattr( refimlast, f'max{coord}' ), abs=0.1/3600. ) )

    assert ref_image.start_mjd == min( [d.image.start_mjd for d in ptf_reference_image_datastores] )
    assert ref_image.end_mjd == max( [d.image.end_mjd for d in ptf_reference_image_datastores] )
    assert ref_image.exp_time == sum( [d.image.exp_time for d in ptf_reference_image_datastores] )

    assert ref_image.is_coadd
    assert not ref_image.is_sub
    assert ref_image.exposure_id is None

    try:
        # Insert so that we can search for upstreams in the database
        ref_image.filepath='foo'
        ref_image.md5sum = uuid.uuid4()
        ref_image.insert()
        upstrzps = ref_image.get_upstreams()
        assert [ i.id for i in upstrzps ] == [ d.zp.id for d in ptf_reference_image_datastores ]
        assert ref_image.coadd_alignment_target == refimlast.id
        assert ref_image.is_coadd
        assert not ref_image.is_sub
        assert ref_image.data is not None
        assert ref_image.data.shape == refimlast.data.shape
        assert ref_image.weight is not None
        assert ref_image.weight.shape == ref_image.data.shape
        assert ref_image.flags is not None
        assert ref_image.flags.shape == ref_image.data.shape
        assert ref_image.zogy_psf is not None
        assert ref_image.zogy_score is not None
        assert ref_image.zogy_score.shape == ref_image.data.shape

        # The zogy coaddition should have left the sky noise at 1, and
        # the weights are all 1 (under the zogy assumption of sky noise
        # domination).  Look at at a visually-selected "blank spot"
        # on the image:
        assert ref_image.data[ 1985:2015, 915:945 ].std() == pytest.approx( 1.0, rel=0.1 )
        assert np.all( ref_image.weight[ ref_image.flags == 0 ] == 1 )

    finally:
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "DELETE FROM images WHERE _id=%(id)s", { 'id': ref_image.id } )
            conn.commit()


def test_coaddition_pipeline_outputs(ptf_reference_image_datastores, ptf_aligned_image_datastores):
    try:
        pipe = CoaddPipeline( coaddition={ 'cleanup_alignment': False } )
        coadd_ds = pipe.run( ptf_reference_image_datastores, ptf_aligned_image_datastores )

        # check that the second list input was ingested
        assert pipe.aligned_datastores == ptf_aligned_image_datastores

        assert isinstance(coadd_ds, DataStore)
        assert coadd_ds.image.filepath is None
        assert coadd_ds.image.type == 'ComSci'
        assert coadd_ds.image.provenance_id is not None
        assert coadd_ds.image.provenance_id != ptf_reference_image_datastores[0].image.provenance_id
        assert coadd_ds.image.instrument == 'PTF'
        assert coadd_ds.image.telescope == 'P48'
        assert coadd_ds.image.filter == 'R'
        assert str(coadd_ds.image.section_id) == '11'
        assert coadd_ds.image.start_mjd == min([ d.image.start_mjd for d in ptf_reference_image_datastores ])
        assert coadd_ds.image.end_mjd == max([ d.image.end_mjd for d in ptf_reference_image_datastores])

        # check that all output products are there
        assert isinstance(coadd_ds.sources, SourceList)
        assert isinstance(coadd_ds.psf, PSF)
        assert isinstance(coadd_ds.wcs, WorldCoordinates)
        assert isinstance(coadd_ds.zp, ZeroPoint)

        # check that the ZOGY PSF width is similar to the PSFex result
        # NOTE -- see comment Issue #350 in coaddition.py.  Right now,
        # we're storing zogy_psf and zogy_score in the Image
        # object, but that's vestigal from when the Image object had all
        # kinds of contingent data proucts (sometimes) in it.  It would
        # be better to store these in the DataStore; refactor the code
        # necessary to do that.
        assert np.max(coadd_ds.image.zogy_psf) == pytest.approx(np.max(coadd_ds.psf.get_clip()), abs=0.01)
        zogy_fwhm = estimate_psf_width(coadd_ds.image.zogy_psf, num_stars=1)
        # pad so extract_psf_surrogate works
        psfex_fwhm = estimate_psf_width(np.pad(coadd_ds.psf.get_clip(), 20), num_stars=1)
        assert zogy_fwhm == pytest.approx(psfex_fwhm, rel=0.1)

        # check that the S/N is consistent with a coadd
        flux_zp = [10 ** (0.4 * d.zp.zp) for d in ptf_reference_image_datastores]  # flux in ADU of a magnitude 0 star
        bkgs = [ d.image.bkg_rms_estimate for d in ptf_reference_image_datastores ]
        snrs = np.array(flux_zp) / np.array(bkgs)
        mean_snr = np.mean(snrs)

        flux_zp_zogy = 10 ** (0.4 * coadd_ds.zp.zp)
        _, bkg_zogy = sigma_clipping(coadd_ds.image.data)
        snr_zogy = flux_zp_zogy / bkg_zogy

        # zogy background noise is normalized by construction
        assert bkg_zogy == pytest.approx(1.0, abs=0.1)

        # S/N should be sqrt(N) better # TODO: why is the zogy S/N 20% better than expected??
        assert snr_zogy == pytest.approx(mean_snr * np.sqrt(len(ptf_reference_image_datastores)), rel=0.5)

    finally:
        if 'coadd_ds' in locals():
            coadd_ds.delete_everything()


def test_coadded_reference(ptf_ref):
    ref_image = Image.get_by_id( ptf_ref.image_id )
    assert ref_image.filepath is not None
    assert ref_image.type == 'ComSci'

    ref_sources, ref_bg, ref_psf, ref_wcs, ref_zp = ptf_ref.get_ref_data_products()

    assert isinstance(ref_sources, SourceList)
    assert isinstance(ref_psf, PSF)
    assert isinstance(ref_bg, Background)
    assert isinstance(ref_wcs, WorldCoordinates)
    assert isinstance(ref_zp, ZeroPoint)

    ref_prov = Provenance.get( ptf_ref.provenance_id )
    # refimg_prov = Provenance.get( ref_image.provenance_id )

    assert ref_image.provenance_id in [ p.id for p in ref_prov.upstreams ]
    assert ref_sources.provenance_id in [ p.id for p in ref_prov.upstreams ]
    assert ref_prov.process == 'referencing'

    assert ref_prov.parameters['test_parameter'] == 'test_value'


def test_coadd_partial_overlap_swarp( decam_four_offset_refs, decam_four_refs_alignment_target ):

    coadder = Coadder( method='swarp',
                       alignment_index='other',
                       alignment={ 'min_frac_matched': 0.025, 'min_matched': 50 }
                      )
    img = coadder.run( data_store_list=decam_four_offset_refs,
                       alignment_target_datastore=decam_four_refs_alignment_target )

    assert img.data.shape == ( 4096, 2048 )
    assert img.flags.shape == img.data.shape
    assert img.weight.shape == img.data.shape

    # What else to check?

    # Spot check a few points on the image.
    # (I manually looked at the image and picked out a few spots)

    # Check that the weight is higher in a region where two images actually overlapped
    wtmean1 = img.weight[ 550:640, 975:1140 ].mean()
    wtmean2 = img.weight[ 690:770, 930:1050 ].mean()
    assert  wtmean1 == pytest.approx( 0.021, abs=0.001 )
    assert  wtmean2 == pytest.approx( 0.013, abs=0.001 )
    # And, while we're here, check that those weights make sense.  (These are blank regions
    #   on the images, so the weights should be approx 1/(sdev(image)**2).)
    # ...but, no.  Because the pixels were resampled when warped, there
    #   is now correlated noise between the pixels, which will (I
    #   believe) tend to reduce the measured standard deviation.  So, we
    #   expect the weights to be "too low" here (though, really, they're
    #   probably right).  But, they should be in the same general range.
    #   Somebody who knows statistics better than me can determine a
    #   better value to use for the empirical 0.3 or 0.4 that I have
    #   below.
    assert wtmean1 < 1. / np.std( img.data[550:640, 957:1140] ) ** 2
    assert wtmean1 == pytest.approx( 1. / np.std(img.data[550:640, 957:1140])**2, rel=0.4 )
    assert wtmean2 < 1. / np.std( img.data[690:770, 930:1050] ) ** 2
    assert wtmean2 == pytest.approx( 1. / np.std(img.data[690:770, 930:1050])**2, rel=0.3 )

    # Look at a spot with a star, and a nearby sky, in a place where there was only
    #   one image in the coadd
    assert img.data[ 3217:3231, 479:491 ].sum() == pytest.approx( 82560.88, abs=25. )
    assert img.data[ 3217:3231, 509:521 ].sum() == pytest.approx( 177.70, abs=25. )
    # ...for reasons I don't understand, acutal numbers here aren't stable.
    #   Is it a cache thing?  I don't know.  Issue #361.
    # For now, just verify that the spot with the star is a lot brighter
    #   than the neighboring spot
    assert ( img.data[ 3217:3231, 479:491 ].sum() / img.data[ 3217:3231, 509:521 ].sum() ) >= 400.

    # Look at a spot with a galaxy and a nearby sky, in a place where there were
    #   two images in the sum
    # assert img.data[ 237:266, 978:988 ].sum() == pytest.approx( 7968., abs=10. )
    # assert img.data[ 237:266, 1008:1018 ].sum() == pytest.approx( 70., abs=10. )
    assert ( img.data[ 237:266, 978:988 ].sum() / img.data[ 237:266, 1008:1018 ].sum() ) >= 100.


# This next test is very slow (9 minutes on github), and also perhaps a
#   bit much given that it downloads and swarps together 17 images.  As
#   such, put in two conditions to skip it; this means it won't be run
#   by default either when you just do "pytest -v" or on github (where
#   SKIP_BIG_MEMORY and RUN_SLOW_TESTS are both 1).  To actually run it,
#   use
#           RUN_SLOW_TESTS=1 pytest ...
#
# (The same code is tested in the previous test, so it's not a big deal
#   to routinely skip this test.  It's here because I wanted an example
#   of an actual ref that might approximate something we'd use, and I
#   wanted to have the code to generate the image so I could look at
#   it.)
@pytest.mark.skipif( ( not env_as_bool('RUN_SLOW_TESTS') ) or ( env_as_bool('SKIP_BIG_MEMORY' ) ),
                     reason="Set RUN_SLOW_TESTS and unset SKIP_BIG_MEMORY to run this test" )
def test_coadd_17_decam_images_swarp( decam_17_offset_refs, decam_four_refs_alignment_target ):
    coadder = Coadder( method='swarp',
                       alignment_index='other',
                       alignment={ 'min_frac_matched': 0.025,
                                   'min_matched': 50,
                                   'max_arcsec_residual': 0.25 },
                      )
    img = coadder.run( data_store_list=decam_17_offset_refs,
                       alignment_target_datastore=decam_four_refs_alignment_target )

    assert img.data.shape == ( 4096, 2048 )
    assert img.flags.shape == img.data.shape
    assert img.weight.shape == img.weight.shape
