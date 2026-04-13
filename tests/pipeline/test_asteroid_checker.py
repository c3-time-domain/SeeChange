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
    matched = ac._cross_match_sources_with_asteroids(asteroid_skycoords, asteroid_designations, src, radius_threshold=30*u.arcsec)

    assert len(matched) == 1
    assert matched[0] == '2025 AB'
    print("Matched asteroid designation:", matched[0])


def test_cross_match_no_asteroids():
    # No asteroids -> all None
    asteroid_skycoords = []
    asteroid_designations = []
    srcs = SkyCoord([10.0, 11.0], [0.0, 1.0], unit='deg')

    ac = object.__new__(AsteroidChecker)
    matched = ac._cross_match_sources_with_asteroids(asteroid_skycoords, asteroid_designations, srcs, radius_threshold=default_radius_threshold)

    assert matched == [None, None]
    print("No asteroids matched, as expected:", matched)
