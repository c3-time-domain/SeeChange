from astropy.coordinates import SkyCoord
import astropy.units as u

from pipeline.asteroid_checker import AsteroidChecker, default_radius_threshold


def test_cross_match_basic():
    # Create one asteroid and one source within matching radius
    ast_ra = [10.0]
    ast_dec = [0.0]
    asteroid_skycoords = SkyCoord(ast_ra, ast_dec, unit='deg')
    asteroid_designations = ['2025 AB']

    # create a source very close (10 arcsec away)
    src = SkyCoord([10.0 + (10.0/3600.0)/ (u.deg).to(u.deg)], [0.0], unit='deg')

    ac = object.__new__(AsteroidChecker)
    # call the cross-match method directly
    matched = ac._cross_match_sources_with_asteroids(asteroid_skycoords, asteroid_designations, src,
                                                    radius_threshold=30*u.arcsec)

    assert len(matched) == 1
    assert matched[0] == '2025 AB'
    print("Matched asteroid designation:", matched[0])


def test_cross_match_no_asteroids():
    # No asteroids -> all None
    asteroid_skycoords = []
    asteroid_designations = []
    srcs = SkyCoord([10.0, 11.0], [0.0, 1.0], unit='deg')

    ac = object.__new__(AsteroidChecker)
    matched = ac._cross_match_sources_with_asteroids(asteroid_skycoords, asteroid_designations, srcs,
                                                    radius_threshold=default_radius_threshold)

    assert matched == [None, None]
    print("No asteroids matched, as expected:", matched)


def test_all_neighbors_sorted_and_ra_wrap():
    asteroids = SkyCoord([359.999, 0.002, 1.], [0., 0., 0.], unit='deg')
    sources = SkyCoord([0., 2.], [0., 0.], unit='deg')
    matches = AsteroidChecker._find_neighbors(asteroids, ['closest', 'second', 'far'], sources)
    assert [m['designation'] for m in matches[0]] == ['closest', 'second']
    assert abs(matches[0][0]['distanceArcsec'] - 3.6) < 1e-6
    assert abs(matches[0][1]['distanceArcsec'] - 7.2) < 1e-6
    assert matches[1] == []


def test_empty_sources():
    assert AsteroidChecker._find_neighbors(SkyCoord([0.], [0.], unit='deg'), ['test'],
                                          SkyCoord([], [], unit='deg')) == []


def test_parameter_validation():
    import pytest
    from pipeline.asteroid_checker import ParsAsteroidChecker
    for value in [0, -1, float('nan')]:
        with pytest.raises(ValueError):
            ParsAsteroidChecker(radius_arcsec=value)


def test_orbit_prediction_and_empty_catalog():
    import kete
    from astropy.wcs import WCS

    checker = AsteroidChecker()
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [512., 256.]
    wcs.wcs.cdelt = [-0.00027, 0.00027]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.array_shape = (512, 1024)
    coordinates, names = checker._get_asteroid_list(wcs, 2461000.5)
    assert len(coordinates) == 0 and names == []

    state = kete.State('synthetic', 2461000.5, [2., 0., 0.], [0., 0.012, 0.])
    observer = kete.spice.get_state('Earth', state.jd)
    pos = (state.pos - observer.pos).change_frame(kete.Frames.Equatorial)
    wcs.wcs.crval = [pos.ra, pos.dec]
    checker.mpc_states = [state]
    coordinates, names = checker._get_asteroid_list(wcs, state.jd)
    assert names == ['synthetic']
    # Cross-check against kete's FOV implementation, which also accounts for light delay.
    ra, dec = wcs.pixel_to_world_values([0, 0, 1023, 1023], [0, 511, 511, 0])
    fov = kete.fov.RectangleFOV.from_corners(
        [kete.Vector.from_ra_dec(r, d) for r, d in zip(ra, dec)], observer)
    visible = kete.fov.fov_state_check([state], [fov])
    assert len(visible) == 1
    assert len(visible[0].states) == 1
    expected = (visible[0].states[0].pos - observer.pos).change_frame(kete.Frames.Equatorial)
    separation = coordinates[0].separation(SkyCoord(expected.ra, expected.dec, unit='deg'))
    assert separation.arcsec < 0.1

    wcs.wcs.crval = [180., 0.]
    coordinates, names = checker._get_asteroid_list(wcs, state.jd)
    assert len(coordinates) == 0 and names == []
