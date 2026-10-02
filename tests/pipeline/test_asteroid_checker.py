import kete
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from pipeline.asteroid_checker import AsteroidChecker


def test_all_neighbors_sorted_and_ra_wrap():
    asteroids = SkyCoord([359.999, 0.002, 1.], [0., 0., 0.], unit='deg')
    sources = SkyCoord([0., 2.], [0., 0.], unit='deg')
    matches = AsteroidChecker._find_neighbors(asteroids, ['closest', 'second', 'far'], sources)
    assert [m['designation'] for m in matches[0]] == ['closest', 'second']
    assert matches[0][0]['distanceArcsec'] == pytest.approx(3.6)
    assert matches[0][1]['distanceArcsec'] == pytest.approx(7.2)
    assert matches[1] == []
    assert AsteroidChecker._find_neighbors(asteroids, ['closest', 'second', 'far'], sources, 3 * u.arcsec) == [[], []]


def test_empty_inputs():
    empty = SkyCoord([], [], unit='deg')
    sources = SkyCoord([0., 1.], [0., 0.], unit='deg')
    assert AsteroidChecker._find_neighbors(empty, [], sources) == [[], []]
    assert AsteroidChecker._find_neighbors(sources, ['A', 'B'], empty) == []


def test_orbit_prediction_and_rectangular_footprint():
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
    # A second kete path checks light delay and the rectangular WCS footprint.
    # This is an internal consistency test, not an independent ephemeris check.
    ra, dec = wcs.pixel_to_world_values([0, 0, 1023, 1023], [0, 511, 511, 0])
    fov = kete.fov.RectangleFOV.from_corners(
        [kete.Vector.from_ra_dec(r, d) for r, d in zip(ra, dec)], observer)
    visible = kete.fov.fov_state_check([state], [fov])
    assert len(visible) == 1 and len(visible[0].states) == 1
    expected = (visible[0].states[0].pos - observer.pos).change_frame(kete.Frames.Equatorial)
    assert coordinates[0].separation(SkyCoord(expected.ra, expected.dec, unit='deg')).arcsec < 0.1
    wcs.wcs.crval = [180., 0.]
    coordinates, names = checker._get_asteroid_list(wcs, state.jd)
    assert len(coordinates) == 0 and names == []
