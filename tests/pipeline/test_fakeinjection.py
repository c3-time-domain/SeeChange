import pytest
import numpy as np

from models.fakeset import FakeSet


def test_hostless_fakeinjection( bogus_datastore, fakeinjector ):
    ds = bogus_datastore

    # This test is with fully random positions
    fakeinjector.pars.hostless_frac = 1.

    # We want reproducible tests so we don't have to muck with flaky tests
    fakeinjector.pars.random_seed = 42
    # Generate a *lot* of fakes to make statistical tests below stronger
    n = 1000
    fakeinjector.pars.num_fakes = n

    # Start tests with a flat magnitude probabilty distro, mag rel. limmag
    fakeinjector.pars.mag_prob_ratio = 1.
    minmag = ds.image.lim_mag_estimate + fakeinjector.pars.min_fake_mag
    maxmag = ds.image.lim_mag_estimate + fakeinjector.pars.max_fake_mag

    # Do
    ds = fakeinjector.run( ds )
    seed0 = ds.fakes.random_seed
    assert isinstance( ds.fakes, FakeSet )
    assert len( ds.fakes.fake_x ) == n
    assert len( ds.fakes.fake_y ) == n
    assert len( ds.fakes.fake_mag ) == n
    assert np.all( ds.fakes.host_dex == -1 )
    assert np.all( ds.fakes.fake_x >= 0 )
    assert np.all( ds.fakes.fake_x < ds.image.data.shape[1] )
    assert np.all( ds.fakes.fake_y >= 0 )
    assert np.all( ds.fakes.fake_y < ds.image.data.shape[0] )
    assert np.all( ds.fakes.fake_mag >= minmag )
    assert np.all( ds.fakes.fake_mag <= maxmag )
    # Flat probability distro: divide mag range into 10 bins, each one should have n/10±√n/10 (1σ)
    hist, _binedges = np.histogram( ds.fakes.fake_mag, range=(minmag, maxmag) )
    assert np.all( np.isclose( hist, n/10., atol=2.*np.sqrt(n/10.), rtol=0.  ) )
    assert hist.mean() == pytest.approx( n/10., abs=2.*np.sqrt(n)/10. )

    # Put in a dim/bright ratio of 2
    fakeinjector.pars.mag_prob_ratio = 2.
    ds = fakeinjector.run( ds )
    assert ds.fakes.random_seed == seed0
    assert len( ds.fakes.fake_x ) == n
    assert len( ds.fakes.fake_y ) == n
    assert len( ds.fakes.fake_mag ) == n
    assert np.all( ds.fakes.fake_x >= 0 )
    assert np.all( ds.fakes.fake_x < ds.image.data.shape[1] )
    assert np.all( ds.fakes.fake_y >= 0 )
    assert np.all( ds.fakes.fake_y < ds.image.data.shape[0] )
    assert np.all( ds.fakes.fake_mag >= minmag )
    assert np.all( ds.fakes.fake_mag <= maxmag )
    hist, _binedges = np.histogram( ds.fakes.fake_mag, range=(minmag, maxmag) )
    assert hist[-1] / hist[0] == pytest.approx( 2., rel=2.*np.sqrt( 1./hist[-1] + 1./hist[0] ) )

    # ... and 0.5
    fakeinjector.pars.mag_prob_ratio = 0.5
    ds = fakeinjector.run( ds )
    hist, _binedges = np.histogram( ds.fakes.fake_mag, range=(minmag, maxmag) )
    assert hist[0] / hist[-1] == pytest.approx( 2., rel=2.*np.sqrt( 1./hist[-1] + 1./hist[0] ) )

    # Absolute magnitude range
    fakeinjector.pars.min_fake_mag = 23.
    fakeinjector.pars.max_fake_mag = 25.
    fakeinjector.pars.mag_rel_limmag = False
    ds = fakeinjector.run( ds )
    assert np.all( ds.fakes.fake_mag >= 23. )
    assert np.all( ds.fakes.fake_mag <= 25. )

    # Random random seed
    fakeinjector.pars.random_seed = 0
    ds = fakeinjector.run( ds )
    seed1 = ds.fakes.random_seed
    ds = fakeinjector.run( ds )
    # Technically, this is flaky, but it will only fail something like 1/2³¹ of the time, so whatevs.
    assert ds.fakes.random_seed != seed1


def test_fakeinjection_on_host( decam_datastore_through_zp, fakeinjector ):
    ds = decam_datastore_through_zp

    # Put only on hosts w/in 1 magnitudes of the fake's mag, scale parameter 1.
    fakeinjector.pars.hostless_frac = 0.
    fakeinjector.pars.host_minmag = -4.
    fakeinjector.pars.host_maxmag = 0.5
    fakeinjector.pars.host_distscale = 1.
    fakeinjector.pars.num_fakes = 100
    fakeinjector.pars.random_seed = 31337

    ds = fakeinjector.run( ds )
    fakes = ds.fakes
    sources = ds.get_sources()
    zp = ds.get_zp()
    sourcemags = -2.5 * np.log10( sources.data['FLUX_AUTO'] ) + zp.zp
    hostdist = np.sqrt( ( fakes.fake_x - sources.x[fakes.host_dex] )**2 +
                        ( fakes.fake_y - sources.y[fakes.host_dex] )**2 )

    assert len( fakes.fake_x ) == 100
    assert np.all( fakes.host_dex >= 0 )
    assert np.all( fakes.host_dex < len( sources.data ) )
    assert np.all( sourcemags[fakes.host_dex] >= fakes.fake_mag + fakeinjector.pars.host_minmag )
    assert np.all( sourcemags[fakes.host_dex] <= fakes.fake_mag + fakeinjector.pars.host_maxmag )
    # This next number is kinda arbitrary.  I could dig into the host sizes and really look at it,
    #   but for now just "is the fake close to its host".  (One could be really ambitious and
    #   make sure that the distribution of positions relative to the host, taking into account
    #   the host ellipticity and position angle, stastically matches an exponential distribution
    #   with the desired distance scale.)
    assert np.all( hostdist < 10. )

    # Uncomment the following lines to visually inspect the inserted fakes.  Run
    #    ds9 -zscale -lock frame image im.fits faked.fits diff.fits
    # Look at "diff.fits" (scale it to 0:50 or 0:100 for a better view) to find
    # where fakes were injected, and then look at im.fits (the original image) and
    # fake.fits (the image plus the fake)
    # from astropy.io import fits
    # im, _ = fakes.inject_on_to_image()
    # fits.writeto( 'faked.fits', im, overwrite=True )
    # fits.writeto( 'im.fits', ds.get_image().data, overwrite=True )
    # fits.writeto( 'diff.fits', im - ds.image.data, overwrite=True )

    # Make sure we can only inject a fraction of fakes near hosts if we want
    for nearhostfrac in [ 0.35, 0.5, 0.75 ]:
        fakeinjector.pars.hostless_frac = 1. - nearhostfrac
        fakeinjector.pars.random_seed += 101
        ds = fakeinjector.run( ds )
        nwithhosts = ( fakes.host_dex >= 0 ).sum()
        assert nwithhosts == pytest.approx( fakeinjector.pars.num_fakes * nearhostfrac, 2.*np.sqrt( nwithhosts ) )
