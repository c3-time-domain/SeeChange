"""Annotate detections with nearby known asteroids; never change alert selection."""

import kete
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.time import Time

from models.base import PsycopgConnection
from util.config import Config
from util.logger import SCLogger


default_radius_threshold = 30 * u.arcsec


def _valid_states(states):
    return [s for s in states if np.all(np.isfinite([*s.pos, *s.vel]))]


def download_and_update_db(time_jd):
    """Replace the orbit snapshot atomically. time_jd is a TDB Julian Date.

    Download and propagation happen before the database transaction. A failed
    refresh leaves the previous snapshot intact.
    """
    if not np.isfinite(time_jd):
        raise ValueError('Orbit epoch must be finite')
    orbits = kete.mpc.fetch_known_orbit_data(force_download=True)
    states = _valid_states(kete.propagate_n_body(kete.mpc.table_to_states(orbits), time_jd))
    if not states:
        raise RuntimeError('Refusing to replace the catalog with an empty orbit snapshot')
    tomorrow = kete.propagate_two_body(states, time_jd + 1)

    def rows():
        for state, later in zip(states, tomorrow):
            pos, later_pos = state.as_equatorial.pos, later.as_equatorial.pos
            pm_ra = (later_pos.ra - pos.ra + 180) % 360 - 180
            pm_dec = later_pos.dec - pos.dec
            # Failed next-day propagation must not insert NaNs into diagnostics.
            if not np.isfinite(pm_ra) or not np.isfinite(pm_dec):
                pm_ra = pm_dec = 0.
            yield (state.desig, state.jd, pos.ra, pos.dec, pm_ra, pm_dec, *state.pos, *state.vel)

    with PsycopgConnection() as conn:
        cursor = conn.cursor()
        # DELETE (rather than TRUNCATE) preserves the old snapshot for MVCC readers.
        cursor.execute('DELETE FROM mpc_table;')
        cursor.executemany(
            'INSERT INTO mpc_table (designation, jd, ra, dec, pm_ra, pm_dec, '
            'position_x, position_y, position_z, velocity_x, velocity_y, velocity_z) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', rows())
        conn.commit()
    SCLogger.info(f'Refreshed {len(states)} MPC orbits at TDB JD {time_jd}')


class AsteroidChecker:
    def __init__(self):
        config = Config.get().value('asteroid_checking', {})
        self.enabled = config.get('enabled', True)
        self.radius_arcsec = config.get('radius_arcsec', 30.)
        self.max_epoch_distance_days = config.get('max_epoch_distance_days', 3.)
        self.prefilter_padding_deg = config.get('prefilter_padding_deg', 1.)
        self.observatory_code = config.get('observatory_code')
        for value in [self.radius_arcsec, self.max_epoch_distance_days, self.prefilter_padding_deg]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Asteroid radius, epoch window, and prefilter padding must be finite and positive')
        self.mpc_states = []

    def _load_states_from_db(self):
        # Explicit columns exclude the table's automatic timestamp columns.
        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT designation, jd, position_x, position_y, position_z, '
                           'velocity_x, velocity_y, velocity_z FROM mpc_table;')
            rows = cursor.fetchall()
        self.mpc_states = [kete.State(row[0], row[1], list(row[2:5]), list(row[5:8])) for row in rows]
        if not self.mpc_states or len(_valid_states(self.mpc_states)) != len(self.mpc_states):
            raise RuntimeError('MPC orbit cache is empty or contains invalid states')

    def check(self, ds):
        """Return annotations by measurement UUID without changing the datastore.

        Read one orbit snapshot per alert batch; no downloads or saved results.
        Failures are reported as status information so alert sending can continue.
        """
        measurements = ds.get_measurement_set().measurements
        neighbors = [[] for _ in measurements]
        status = 'disabled'
        if self.enabled:
            status = 'unavailable'
            try:
                self._load_states_from_db()
                image = ds.get_image()
                observation_jd = Time(image.mid_mjd, format='mjd', scale='utc').tdb.jd
                if any(abs(observation_jd - state.jd) > self.max_epoch_distance_days for state in self.mpc_states):
                    status = 'stale'
                else:
                    wcs = ds.get_wcs().wcs.deepcopy()
                    if wcs.array_shape is None:
                        wcs.array_shape = image.data.shape
                    coordinates, designations = self._get_asteroid_list(wcs, observation_jd)
                    sources = SkyCoord([m.ra for m in measurements], [m.dec for m in measurements], unit='deg')
                    neighbors = self._find_neighbors(coordinates, designations, sources, self.radius_arcsec * u.arcsec)
                    status = 'checked'
            except Exception as exc:
                SCLogger.warning(f'Asteroid annotation unavailable: {type(exc).__name__}: {exc}')
        return {str(m.id): {'mpcDesignation': matches[0]['designation'] if matches else None,
                            'mpcMatches': matches, 'mpcMatchStatus': status}
                for m, matches in zip(measurements, neighbors)}

    def _get_asteroid_list(self, frame_wcs, time_jd):
        if not self.mpc_states:
            return SkyCoord([], [], unit='deg'), []
        observer = (kete.spice.get_state('Earth', time_jd) if self.observatory_code is None
                    else kete.spice.mpc_code_to_ecliptic(self.observatory_code, time_jd))
        # Derive the cone from the actual footprint, including rectangular CCD corners.
        height, width = frame_wcs.array_shape
        center = frame_wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2)
        corners = frame_wcs.pixel_to_world([0, 0, width - 1, width - 1], [0, height - 1, height - 1, 0])
        radius = center.separation(corners).max() + self.prefilter_padding_deg * u.deg
        radius += self.radius_arcsec * u.arcsec

        def skycoords(states):
            vectors = [(state.pos - observer.pos).change_frame(kete.Frames.Equatorial) for state in states]
            return SkyCoord([v.ra for v in vectors], [v.dec for v in vectors], unit='deg')

        approximate = kete.propagate_two_body(self.mpc_states, time_jd, observer_pos=observer.pos)
        if len(_valid_states(approximate)) != len(approximate):
            raise RuntimeError('Approximate orbit propagation failed for part of the catalog')
        candidate_indices = np.where(center.separation(skycoords(approximate)) <= radius)[0]
        if not len(candidate_indices):
            return SkyCoord([], [], unit='deg'), []
        candidates = [self.mpc_states[i] for i in candidate_indices]
        SCLogger.debug(f'Running n-body propagation for {len(candidates)} asteroid candidates')
        refined = kete.propagate_n_body(candidates, time_jd)
        # N-body propagation is at reception time. Apply kete's short two-body
        # light-time correction relative to the actual observer for apparent positions.
        apparent = kete.propagate_two_body(refined, time_jd, observer_pos=observer.pos)
        if len(_valid_states(apparent)) != len(apparent):
            raise RuntimeError('Refined orbit propagation failed for asteroid candidates')
        return skycoords(apparent), [s.desig for s in apparent]

    @staticmethod
    def _find_neighbors(asteroid_coordinates, designations, source_coordinates,
                        radius_threshold=default_radius_threshold):
        neighbors = [[] for _ in range(len(source_coordinates))]
        if not len(asteroid_coordinates) or not len(source_coordinates):
            return neighbors
        sources, asteroids, separations, _ = search_around_sky(
            source_coordinates, asteroid_coordinates, radius_threshold)
        for source, asteroid, separation in zip(sources, asteroids, separations):
            neighbors[source].append({'designation': str(designations[asteroid]),
                                      'distanceArcsec': float(separation.arcsec)})
        for matches in neighbors:
            matches.sort(key=lambda m: (m['distanceArcsec'], m['designation']))
        return neighbors
