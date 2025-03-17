import pytest
import pathlib
import numpy as np

from models.fakeset import FakeSet, FakeAnalysis


def test_hostless_fakeinjection( decam_datastore_through_zp, fakeinjector ):
    origds = decam_datastore_through_zp
    origimmean = origds.image.data.mean()
    origimstd = origds.image.data.std()
    origwtmean = origds.image.weight.mean()
    origwtstd = origds.image.weight.std()

    # This test is with fully random positions
    fakeinjector.pars.hostless_frac = 1.

    # We want reproducible tests so we don't have to muck with flaky tests
    fakeinjector.pars.random_seed = 42
    # Generate a *lot* of fakes to make statistical tests below stronger
    n = 1000
    fakeinjector.pars.num_fakes = n

    # Start tests with a flat magnitude probabilty distro, mag rel. limmag
    fakeinjector.pars.mag_prob_ratio = 1.
    minmag = origds.image.lim_mag_estimate + fakeinjector.pars.min_fake_mag
    maxmag = origds.image.lim_mag_estimate + fakeinjector.pars.max_fake_mag

    # Do
    ds = fakeinjector.run( origds )
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

    # Make sure the image actually changed.  Image mean should have gone up from
    #   the injection sources, weight should have gone down from increased variance
    #   from injected sources.
    assert ds.image.data.mean() > origimmean
    assert ds.image.weight.mean() < origwtmean

    # Make sure that the fake data store's provenance tree got properly edited
    assert ds.prov_tree != origds.prov_tree
    assert 'fakeinjection' in ds.prov_tree
    assert 'fakeinjection' in ds.prov_tree.upstream_steps
    assert ds.prov_tree.upstream_steps['fakeinjection' ] == [ 'zp' ]
    assert [ p.id for p in ds.prov_tree['fakeinjection'].upstreams ] == [ ds.prov_tree['zp'].id ]
    assert ds.prov_tree.upstream_steps['subtraction'] == [ 'referencing', 'fakeinjection' ]
    assert ( set( p.id for p in ds.prov_tree['subtraction'].upstreams )
             ==  { ds.prov_tree['referencing'].id, ds.prov_tree['fakeinjection'].id } )
    for step in [ 'subtraction', 'detection', 'cutting', 'measuring', 'scoring' ]:
        assert ds.prov_tree[step].id != origds.prov_tree[step].id

    # Make sure the original datastore's image didn't get munged
    assert origds.image.data.mean() == origimmean
    assert origds.image.data.std() == origimstd
    assert origds.image.weight.mean() == origwtmean
    assert origds.image.weight.std() == origwtstd

    # Put in a dim/bright ratio of 2
    fakeinjector.pars.mag_prob_ratio = 2.
    ds = fakeinjector.run( origds )
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
    ds = fakeinjector.run( origds )
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
    ds = fakeinjector.run( origds )
    seed1 = ds.fakes.random_seed
    ds = fakeinjector.run( origds )
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
    assert np.all( hostdist < 15. )
    assert np.median( hostdist < 2. )

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


def test_fake_analysis( decam_datastore ):
    ds = decam_datastore

    # Just running the pipeline that datastore_factory put in the datastore should do what we want
    ds._pipeline.pars.inject_fakes = True
    ds._pipeline.pars.save_at_finish = True
    ds._pipeline.subtractor.pars.trust_aligned_images = True
    ds._pipeline.fakeinjector.pars.hostless_frac=0.35
    ds._pipeline.fakeinjector.pars.random_seed = 42
    ds._pipeline.fakeinjector.pars.num_fakes = 100
    ds._pipeline.fakeinjector.pars.host_distscale = 1.
    ds._pipeline.fakeinjector.pars.host_minmag = -4.
    ds._pipeline.fakeinjector.pars.host_maxmag = 0.5

    # This will be relatively fast, since all of the things before fake injection have
    #   already been run.  It will now just run the fake injection and analysis step
    #   as we set ds._pipeline.pars.inject_fakes to True.  (It should have defaulted
    #   to False unless the config files have been messed up.)
    ds = ds._pipeline.run( ds )

    props = [ 'is_detected', 'is_kept', 'is_bad', 'flux_psf', 'flux_psf_err', 'best_aperture',
              'bkg_per_pix', 'center_x_pixel', 'center_y_pixel', 'x', 'y', 'gfit_x', 'gfit_y',
              'major_width', 'minor_width', 'position_angle', 'psf_fit_flags',
              'nbadpix', 'negfrac', 'negfluxfrac', 'is_bad', 'deepscore_algorithm', 'score' ]
    origprop = {}
    for prop in props:
        assert isinstance( getattr( ds.fakeanal, prop ), np.ndarray )
        assert getattr( ds.fakeanal, prop ).shape == ds.fakes.fake_mag.shape
        origprop[ prop ] = getattr( ds.fakeanal, prop )

    # Make sure the file was saved and that loading it works.
    # (This test, perhaps, belongs in models/test_fakeset.py,
    # but that would probably mean running the fixture again,
    # so may as well put it here.)

    filepath = pathlib.Path( ds.fakeanal.get_fullpath() )
    assert filepath.is_file()
    newanal = FakeAnalysis()
    newanal.load( filepath=filepath, download=False, always_verify_md5=False )
    for prop in newanal.arrayprops:
        assert np.all( ( np.isclose( getattr( newanal, prop ), getattr( ds.fakeanal, prop ), rtol=1e-5, atol=0. ) )
                         | ( np.isnan( getattr( newanal, prop ) ) & np.isnan( getattr( ds.fakeanal, prop ) ) ) )
    del newanal

    # Check some actual values.  Things that weren't detected
    #   will have NaN in the fake analysis arrays, so chuck
    #   those.

    wbad = np.isnan( ds.fakeanal.flux_psf )
    wgood = ~wbad
    injectm = ds.fakes.fake_mag[wgood]
    m = -2.5 * np.log10( ds.fakeanal.flux_psf[wgood] ) + ds.zp.zp
    dm = 2.5 / np.log(10) * ds.fakeanal.flux_psf_err[wgood] / ds.fakeanal.flux_psf[wgood]
    diffm = m - injectm
    reldiffm = diffm / dm

    # Magnitude residual should mostly be within 3σ of injected fake
    #   mag.  As of the writing of this comment, using the random seed
    #   above, there are four residuals around ~4σ
    assert ( np.abs(reldiffm) <= 3. ).sum() >= ( wgood.sum() - 4 )
    assert np.abs( reldiffm.mean() ) < 3.* ( reldiffm.std() / np.sqrt( len(reldiffm) ) )

    # Most of the things not detected should be dim
    assert ( ds.fakes.fake_mag[wbad] >= ds.image.lim_mag_estimate ).sum() >= wbad.sum() - 4

    # Things detected should be brighter on average than things not detected
    assert ds.fakes.fake_mag[wgood].mean() < ds.fakes.fake_mag[wbad].mean() - 1.

    # Look at the things with the "kept" flag.  (is_bad will just be ~is_kept given the default thresholds)

    wkept = ds.fakeanal.is_kept
    # Nothing "kept" should have a NaN analysis flux
    assert not np.any( np.isnan( ds.fakeanal.flux_psf[wkept] ) )

    # Things kept should be brighter on average than things not kept
    assert ds.fakes.fake_mag[wkept].mean() < ds.fakes.fake_mag[ wgood & ~wkept ].mean() - 0.7

    injectm = ds.fakes.fake_mag[wkept]
    m = -2.5 * np.log10( ds.fakeanal.flux_psf[wkept] ) + ds.zp.zp
    dm = 2.5 / np.log(10) * ds.fakeanal.flux_psf_err[wkept] / ds.fakeanal.flux_psf[wkept]
    diffm = m - injectm
    reldiffm = diffm / dm

    assert ( np.abs( reldiffm ) >= 3 ).sum() <= 2
    assert np.abs( reldiffm.mean() ) < 2. * reldiffm.std() / np.sqrt( len(reldiffm) )
