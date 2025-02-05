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
