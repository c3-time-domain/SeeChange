import pytest

import numpy as np
from astropy.io import fits
import photutils.psf

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

    meas = photometry_and_diagnostics( image, noise, mask, positions, apers, psfobj=psf_palette.psf )

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

    bgmeas = photometry_and_diagnostics( image + 20., noise, mask, positions, apers,
                                         psfobj=psf_palette.psf, dobgsub=True, innerrad=17., outerrad=20. )
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


def test_diagnostics():
    # Make a noise image, and add certain specific failure modes to make sure the diagnostics
    #   that we expect to catch these do in fact catch these.
    # Note that widths and position angles were tested with all our gaussians
    #   on the PSFPalette in test_photometry_and_diagnostics
    rng = np.random.default_rng( seed=42 )
    bg = 100.
    skynoise = 5.
    image = rng.normal( loc=bg, scale=skynoise, size=(1024, 1024) )
    noise = np.full_like( image, skynoise )
    mask = np.full_like( image, False, dtype=bool )

    # We'll be working in 51 by 51 patches, so it's a little bigger than the 41x41 patch
    #   that we'll be cutting out

    _sqrt2 = np.sqrt(2.)
    _2sqrt2ln2 = 2. * np.sqrt( 2. * np.log( 2. ) )
    wid = 51
    xvals, yvals = np.meshgrid( np.array(range(wid)) - wid // 2, np.array(range(wid)) - wid // 2 )
    sigma = 2.0
    fwhm = _2sqrt2ln2 * sigma
    flux = 20000.
    inner = 4. * fwhm
    outer = 5. * fwhm
    ringarea = np.pi * ( outer**2 - inner**2 )

    positions = []
    expected = []

    # At 100, 100 : a normal psf
    x0 = 100 - wid // 2
    y0 = 100 - wid // 2
    image[ y0:y0+wid, x0:x0+wid ] += ( flux / ( 2. * np.pi * sigma**2 )
                                       * np.exp( -( xvals**2 + yvals**2 ) / ( 2. * sigma**2 ) ) )
    positions.append( (x0 + wid//2, y0 + wid//2) )
    expected.append( { 'flux': pytest.approx( flux, abs=3. * np.sqrt(2*np.pi) * fwhm * skynoise ),
                       'center_x_pixel': 100,
                       'center_y_pixel': 100,
                       'bkg_per_pix': pytest.approx( bg, abs=3. * ( skynoise / np.sqrt(ringarea) ) ),
                       'x': pytest.approx( 100, abs=0.05 ),
                       'y': pytest.approx( 100, abs=0.05 ),
                       'gfit_x': pytest.approx( 100, abs=0.05 ),
                       'gfit_y': pytest.approx( 100, abs=0.05 ),
                       'major_width': pytest.approx( fwhm, rel=0.01 ),
                       'minor_width': pytest.approx( fwhm, rel=0.01 ),
                       'psf_fit_flags': 0,
                       'nbadpix': 0,
                       'negfrac': ( "lt", 0.05 ),
                       'negfluxfrac': ( "lt", 0.05 )
                      } )



    # At 100, 200 : a negative psf
    x0 = 100 - wid // 2
    y0 = 200 - wid // 2
    image[ y0:y0+wid, x0:x0+wid ] -= ( flux / ( 2. * np.pi * sigma**2 )
                                       * np.exp( -( xvals**2 + yvals**2 ) / ( 2. * sigma**2 ) ) )
    positions.append( (x0, y0) )
    expected.append( { 'psf_fit_flags': ( "and", 4 ),
                       'negfrac': ( "gt", 0.75 ),
                       'negfluxfrac': ( "gt", 0.75 ) } )


    # At 100, 300 : a dipole with 1/2 the flux negative, separated by 1σ
    x0 = 100 - wid // 2
    y0 = 300 - wid // 2
    image[ y0:y0+wid, x0:x0+wid ] += ( flux / ( 2. * np.pi * sigma**2 )
                                       * np.exp( -( xvals**2 + yvals**2 ) / ( 2. * sigma**2 ) )
                                       -
                                       flux / ( 4. * np.pi * sigma**2 )
                                       * np.exp( -( (xvals - sigma/_sqrt2)**2 + (yvals - sigma/_sqrt2)**2 )
                                                 / ( 2. * sigma**2 ) ) )
    positions.append( (x0 + wid//2, y0 + wid//2) )
    expected.append( { 'psf_fit_flags': 0,
                       'negfrac': ( "gt", 0.45 ),
                       'negfluxfrac': ( "gt", 0.1 ) } )

    # At 100, 400 : a dipole with full flux negative, separated by 2.5σ
    x0 = 100 - wid // 2
    y0 = 400 - wid // 2
    image[ y0:y0+wid, x0:x0+wid ] += ( flux / ( 2. * np.pi * sigma**2 )
                                       * np.exp( -( xvals**2 + yvals**2 ) / ( 2. * sigma**2 ) )
                                       -
                                       flux / ( 2. * np.pi * sigma**2 )
                                       * np.exp( -( (xvals - sigma/(2.5*_sqrt2))**2 +
                                                    (yvals - sigma/(2.5*_sqrt2))**2 )
                                                 / ( 2. * sigma**2 ) ) )
    positions.append( (x0 + wid//2, y0 + wid//2) )
    expected.append( { 'psf_fit_flags': ( "notand", 1 & 2 & 32 ),
                       'negfrac': pytest.approx( 1.0, rel=0.1 ),
                       'negfluxfrac': pytest.approx( 1.0, rel=0.1 ) } )

    # At 100, 500 : a normal psf, but with some masked pixels
    x0 = 100 - wid // 2
    y0 = 500 - wid // 2
    image[ y0:y0+wid, x0:x0+wid ] += ( flux / ( 2. * np.pi * sigma**2 )
                                       * np.exp( -( xvals**2 + yvals**2 ) / ( 2. * sigma**2 ) ) )
    mask[ y0+wid//2, x0+wid//2-1:x0+wid//2+2 ] = True
    positions.append( (x0 + wid//2, y0 + wid//2) )
    expected.append( { 'psf_fit_flags': ( "and", 1 ),
                       'flux': pytest.approx( flux, abs=3. * np.sqrt(2*np.pi) * fwhm * skynoise ),
                       'x': pytest.approx( 100, abs=0.05 ),
                       'y': pytest.approx( 500, abs=0.05 ),
                       'major_width': pytest.approx( fwhm, rel=0.01 ),
                       'minor_width': pytest.approx( fwhm, rel=0.01 ),
                       'negfrac': ( "lt", 0.05 ),
                       'negfluxfrac': ( "lt", 0.05 ),
                       'nbadpix': 3 } )

    # A big blob at 100, 600
    x0 = 100 - wid // 2
    y0 = 600 - wid // 2
    image[ y0:y0+wid, x0:x0+wid ] += ( flux / ( 2. * np.pi * (3.*sigma)**2 )
                                       * np.exp( -( xvals**2 + yvals**2 ) / ( 2. * (3.*sigma)**2 ) ) )
    positions.append( (x0 + wid//2, y0 + wid//2) )
    expected.append( { 'psf_fit_flags': 0,
                       'center_x_pixel': 100,
                       'center_y_pixel': 600,
                       'gfit_x': pytest.approx( 100, abs=0.1 ),
                       'gfit_y': pytest.approx( 600, abs=0.1 ),
                       'bkg_per_pix': pytest.approx( bg, abs=3. * ( skynoise / np.sqrt(ringarea) ) ),
                       'x': pytest.approx( 100, abs=1. ),       # fit of 1fwhm gaussian won't be so good
                       'y': pytest.approx( 600, abs=1. ),
                       'negfrac': ( "lt", 0.05 ),
                       'negfluxfrac': ( "lt", 0.05 ),
                       'nbadpix': 0,
                       'major_width': pytest.approx( 3*fwhm, rel=0.03 ),
                       'minor_width': pytest.approx( 3*fwhm, rel=0.03 ) } )


    # TODO : more!
    #   * vertical and/or horizontal streak (1-pixel wide)
    #   * streak at an angle

    psf = photutils.psf.CircularGaussianPSF( flux=1., fwhm=fwhm )
    measurements = photometry_and_diagnostics( image, noise, mask, positions, [ fwhm, 2*fwhm ],
                                               photutils_psf=psf, fwhm_pixels=fwhm,
                                               dobgsub=True, innerrad=inner, outerrad=outer )
    for exp, m in zip( expected, measurements ):
        for attr, val in exp.items():
            m_attr = getattr( m, attr )
            if isinstance( val, tuple ):
                if val[0] == 'lt':
                    assert m_attr < val[1]
                elif val[0] == 'gt':
                    assert m_attr > val[1]
                elif val[0] == 'and':
                    assert m_attr & val[1]
                elif val[0] == 'notand':
                    assert m_attr & val[1] == 0
                else:
                    raise ValueError( f"Unknown comparator {val[0]}" )
            else:
                assert m_attr == val
