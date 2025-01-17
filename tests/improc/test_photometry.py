import pytest

import numpy as np
from astropy.io import fits

from improc.photometry import photometry_and_diagnostics

# This test has a slow startup because making the psf palette takes ~20-30 seconds, and
#   then we run sextractor (fast) psfex (slow).  Overall a ~1 min fixture setup.
def test_photometry_and_diagnostics( psf_palette ):

    # Test photometry on something where we know all the fluxes and shapes
    # TODO : run a test with a much noisier psf palette
    with fits.open( psf_palette.imagename ) as hdul:
        image = hdul[0].data
    with fits.open( psf_palette.flagsname ) as hdul:
        mask = np.full_like( hdul[0].data, False, dtype=bool )
        mask[ hdul[0].data != 0 ] = True
    with fits.open( psf_palette.weightname ) as hdul:
        weight = hdul[0].data
        noise = 1. / np.sqrt( hdul[0].data )
        mask[ weight <= 0 ] = True

    mesh = np.meshgrid( psf_palette.xpos, psf_palette.ypos )
    positions = [ ( x, y ) for x, y, in zip( mesh[0].flatten(), mesh[1].flatten() ) ]
    apers = [ 1.25, 2.5, 5. ]

    meas = photometry_and_diagnostics( image, noise, mask, positions, psf_palette.psf, apers )

    # I'm a little disappointed that I had to set this to 3.5%, especially
    #  since the flux_psf_err all came out to 0.01%....  (Which sounds
    #  about right, given 200000. flux and noise of 5. per pixel, or
    #  about an overall noise of 17.)
    # It seems to be systematically low by a coupleof percent in
    #  addition to scattering.  Perhaps more investigation is needed.
    #  (psfex psfs bad?  My reconstruction of psfex psfs bad?  The
    #  interpolation assumptions documented in the comments of
    #  photometry.py are biting us?  Something else?)
    assert all( [ m.flux_psf == pytest.approx( 200000., rel=0.035 ) for m in meas ] )
    assert all( [ m.flux_psf_err == pytest.approx( 26., abs=2. ) for m in meas ] )
    assert all( [ m.flux_apertures[2] > m.flux_apertures[1] for m in meas ] )
    assert all( [ m.flux_apertures[1] > m.flux_apertures[0] for m in meas ] )
    assert all( [ m.flux_apertures[2] / m.flux_psf == pytest.approx( 0.998, abs=0.02 ) for m in meas ] )
    assert all( [ m.flux_apertures_err[2] > m.flux_apertures_err[1] for m in meas ] )
    assert all( [ m.flux_apertures_err[1] > m.flux_apertures_err[0] for m in meas ] )
    assert all( [ m.bkg_per_pix == 0 for m in meas ] )
    assert all( [ ( m.aper_radii == np.array([1.25, 2.5, 5.]) ).all() for m in meas ] )
    assert all( [ m.center_x_pixel == pytest.approx( m.x, abs=0.5 ) for m in meas ] )
    assert all( [ m.center_y_pixel == pytest.approx( m.y, abs=0.5 ) for m in meas ] )
    # PSFPalette did not line up all things right at the center of pixels
    assert np.median( np.abs( [ m.center_x_pixel - m.x for m in meas ] ) ) == pytest.approx( 0.3, abs=0.1 )
    assert np.median( np.abs( [ m.center_y_pixel - m.y for m in meas ] ) ) == pytest.approx( 0.3, abs=0.1 )
    # Gaussian fit positions should be very close to psf fit positions
    assert all( [ m.gfit_x == pytest.approx( m.x, abs=0.01 ) for m in meas ] )
    assert all( [ m.gfit_y == pytest.approx( m.y, abs=0.01 ) for m in meas ] )

    # With PSF palette:
    #   σx = 1.25 + 0.5 * ( x - 512 ) / 1024
    #   σy = 1.75 - 0.5 * ( x - 512 ) / 1024
    #   θ = 0. + π/2 * ( y - 512 ) / 1024
    #
    # Expect, for pure gaussians, without the convolving effect of pixelization:
    #   Lower left : σx = 1.0, σy = 2.0, θ = -π/4
    #   Lower right : σx = 1.5, σy = 1.5, θ = -π/4
    #   Middle : σx = 1.25, σy = 1.75, θ = 0.
    #   Upper left : σx = 1.0, σy = 2.0, θ = π/4
    #   Upper right : σx = 1.5, οy = 1.5, θ = π/4

    # Empirically, the major widths are systemtically about 0.05 bigger
    #   than the σy values; likewise for minor widths and σx (diff
    #   0.07).  is this the result of the convolution from
    #   pixellization?  Perhaps if I fit GaussianPRF instead of
    #   GaussianPSF it would be better?  Not going to worry about it
    assert all( [ np.abs( m.major_width - 2.35482 * ( 1.75 - 0.5 * (m.x-512) / 1024. ) ) < 0.07 for m in meas ] )
    assert all( [ np.abs( m.minor_width - 2.35482 * ( 1.25 + 0.5 * (m.x-512) / 1024. ) ) < 0.10 for m in meas ] )

    assert all( [ np.abs( m.position_angle - ( np.pi / 2 * (m.y-512) / 1024 ) ) < 0.005 for m in meas ] )

    # NEXT.  Make sure background subtraction works right, at least for a very basic
    #  background.  Should get all the same photometry back out.

    bgmeas = photometry_and_diagnostics( image + 20., noise, mask, positions, psf_palette.psf, apers,
                                         dobgsub=True, innerrad=17., outerrad=20. )
    assert all( [ b.bkg_per_pix == pytest.approx( 20., abs=1. ) for b in bgmeas ] )
    # For a ring of inner radius 17, outer radius 20, area~350 pixels, expect bg to be good to 5./sqrt(350) = ~0.27
    # The average of the averages should be good to ~0.27 / sqrt(100) = ~0.027
    assert np.mean( [ b.bkg_per_pix - 20. for b in bgmeas ] ) == pytest.approx( 0., abs=0.03 )
    assert np.std( [ b.bkg_per_pix - 20. for b in bgmeas ] ) == pytest.approx( 0.27, rel=0.3 )
    assert all( [ b.flux_psf / m.flux_psf == pytest.approx( 1.0, 0.001 ) for b, m in zip( bgmeas, meas ) ] )
    assert all( [ b.flux_psf_err / m.flux_psf_err == pytest.approx( 1.0, 0.001 ) for b, m in zip( bgmeas, meas ) ] )
    for i in range(3):
        assert all( [ b.flux_apertures[i] / m.flux_apertures[i] == pytest.approx( 1.0, 0.001 )
                      for b, m in zip( bgmeas, meas ) ] )
    assert all( [ b.center_x_pixel == m.center_x_pixel for b, m in zip( bgmeas, meas ) ] )
    assert all( [ b.center_y_pixel == m.center_y_pixel for b, m in zip( bgmeas, meas ) ] )
    assert all( [ b.x == pytest.approx( m.x, abs=0.001 ) for b, m in zip( bgmeas, meas ) ] )
    assert all( [ b.y == pytest.approx( m.y, abs=0.001 ) for b, m in zip( bgmeas, meas ) ] )

    assert all( [ np.abs( m.major_width - b.major_width ) < 0.001 for m, b in zip( meas, bgmeas ) ] )
    assert all( [ np.abs( m.minor_width - b.minor_width ) < 0.001 for m, b in zip( meas, bgmeas ) ] )
    assert all( [ np.abs( m.position_angle - b.position_angle ) < 1e-5 for m, b in zip( meas, bgmeas ) ] )
