"""Annotate detections with nearby known asteroids; never change alert selection."""

import time
import uuid

import kete
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.time import Time

from models.base import PsycopgConnection
from pipeline.parameters import Parameters
from util.config import Config
from util.logger import SCLogger


default_radius_threshold = 30 * u.arcsec


class CatalogUnavailable(RuntimeError):
    pass


class CatalogStale(CatalogUnavailable):
    pass


class ParsAsteroidChecker(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.enabled = self.add_par('enabled', True, bool, 'Annotate detections with known asteroid neighbors.')
        self.radius_arcsec = self.add_par('radius_arcsec', 30., (float, int), 'Angular matching radius in arcseconds.')
        self.max_epoch_distance_days = self.add_par(
            'max_epoch_distance_days', 3., (float, int),
            'Maximum separation between image time and cached orbit epoch, in days.')
        self.prefilter_padding_deg = self.add_par(
            'prefilter_padding_deg', 1., (float, int),
            'Padding around the image footprint for approximate orbit preselection.')
        self.observatory_code = self.add_par(
            'observatory_code', None, (str, None), 'MPC ground observatory code; null uses the Earth center.')
        self._enforce_no_new_attrs = True
        self.override(kwargs)
        for name in ['radius_arcsec', 'max_epoch_distance_days', 'prefilter_padding_deg']:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')

    def get_process_name(self):
        return 'asteroid_checking'


def _valid_states(states):
    return [s for s in states if np.all(np.isfinite([*s.pos, *s.vel]))]


def download_and_update_db(time_jd):
    """Replace the orbit snapshot atomically. time_jd is a TDB Julian Date.

    Download and propagation happen before the database transaction. A failed
    refresh leaves the previous snapshot intact, including its version metadata.
    """
    if not np.isfinite(time_jd):
        raise ValueError('Orbit epoch must be finite')
    orbits = kete.mpc.fetch_known_orbit_data(force_download=True)
    states = _valid_states(kete.propagate_n_body(kete.mpc.table_to_states(orbits), time_jd))
    if not states:
        raise CatalogUnavailable('Refusing to replace the catalog with an empty orbit snapshot')
    tomorrow = kete.propagate_two_body(states, time_jd + 1)
    version = str(uuid.uuid4())

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
        cursor.execute('SELECT pg_advisory_xact_lock(736234019);')
        # DELETE (rather than TRUNCATE) preserves the old snapshot for MVCC readers.
        cursor.execute('DELETE FROM mpc_table;')
        cursor.executemany(
            'INSERT INTO mpc_table (designation, jd, ra, dec, pm_ra, pm_dec, '
            'position_x, position_y, position_z, velocity_x, velocity_y, velocity_z) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', rows())
        cursor.execute(
            "INSERT INTO mpc_orbit_catalog (name, version, epoch_jd, object_count) VALUES ('current', %s, %s, %s) "
            'ON CONFLICT (name) DO UPDATE SET version=EXCLUDED.version, epoch_jd=EXCLUDED.epoch_jd, '
            'object_count=EXCLUDED.object_count, modified=now()', (version, time_jd, len(states)))
        conn.commit()
    SCLogger.info(f'Refreshed {len(states)} MPC orbits at TDB JD {time_jd}, version {version}')
    return version


class AsteroidChecker:
    def __init__(self, **kwargs):
        config = Config.get().value('asteroid_checking', {})
        config.update(kwargs)
        self.pars = ParsAsteroidChecker(**config)
        self.mpc_states = []
        self.catalog_version = None
        self.catalog_epoch_jd = None
        self.has_recalculated = False

    def _load_states_from_db(self):
        # Keep metadata and orbit rows consistent even if a refresh commits while reading.
        with PsycopgConnection() as conn:
            cursor = conn.cursor()
            cursor.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;')
            cursor.execute("SELECT version, epoch_jd, object_count FROM mpc_orbit_catalog WHERE name='current';")
            catalog = cursor.fetchone()
            if catalog is None or catalog[2] == 0:
                raise CatalogUnavailable('MPC orbit catalog has not been populated')
            version, epoch, count = catalog
            if version != self.catalog_version:
                cursor.execute('SELECT designation, jd, position_x, position_y, position_z, '
                               'velocity_x, velocity_y, velocity_z FROM mpc_table ORDER BY designation;')
                rows = cursor.fetchall()
                if len(rows) != count:
                    raise CatalogUnavailable('MPC orbit snapshot and metadata disagree')
                states = [kete.State(row[0], row[1], list(row[2:5]), list(row[5:8])) for row in rows]
                if len(_valid_states(states)) != count:
                    raise CatalogUnavailable('MPC orbit snapshot contains invalid states')
                self.mpc_states = states
                self.catalog_version, self.catalog_epoch_jd = version, epoch

    def run(self, ds):
        from models.asteroid_match import AsteroidMatchSet

        started = time.perf_counter()
        self.has_recalculated = False
        self.pars.do_warning_exception_hangup_injection_here()
        measurement_set = ds.get_measurement_set()
        if measurement_set is None:
            raise ValueError(f'Cannot find a measurement set for {ds.inputs_str}')
        prov = ds.get_provenance('asteroid_checking', self.pars.get_critical_pars())
        existing = ds.get_asteroid_match_set(provenance=prov)
        # A saved result is a reproducible annotation, even after the live catalog changes.
        # Retry unavailable/stale checks so a recovered catalog can improve those annotations.
        if existing is not None and existing.status in ['checked', 'disabled']:
            return ds
        measurements = measurement_set.measurements
        image = ds.get_image()
        observation_jd = Time(image.mid_mjd, format='mjd', scale='utc').tdb.jd
        result = AsteroidMatchSet(
            measurementset_id=measurement_set.id, provenance_id=prov.id,
            status='disabled', observation_jd=observation_jd, matches={},
        )
        if existing is not None:
            result._id = existing.id
        if self.pars.enabled:
            try:
                self._load_states_from_db()
                result.catalog_version = self.catalog_version
                result.catalog_epoch_jd = self.catalog_epoch_jd
                if abs(observation_jd - self.catalog_epoch_jd) > self.pars.max_epoch_distance_days:
                    raise CatalogStale('Image time lies outside the configured orbit snapshot window')
                if measurements:
                    world_coordinates = ds.get_wcs()
                    if world_coordinates is None:
                        raise CatalogUnavailable('Image has no astrometric solution')
                    wcs = world_coordinates.wcs.deepcopy()
                    if wcs.array_shape is None:
                        wcs.array_shape = image.data.shape
                    coordinates, designations = self._get_asteroid_list(wcs, observation_jd)
                    source_coordinates = SkyCoord([m.ra for m in measurements], [m.dec for m in measurements],
                                                  unit='deg')
                    neighbors = self._find_neighbors(coordinates, designations, source_coordinates,
                                                     self.pars.radius_arcsec * u.arcsec)
                    result.matches = {str(m.id): matches for m, matches in zip(measurements, neighbors)}
                result.status = 'checked'
            except Exception as exc:
                # Annotation failures must never suppress otherwise eligible alerts.
                result.status = 'stale' if isinstance(exc, CatalogStale) else 'unavailable'
                result.matches = {}
                SCLogger.warning(f'Asteroid annotation {result.status}: {type(exc).__name__}: {exc}')
        ds.asteroid_match_set = result
        self.has_recalculated = True
        if ds.update_runtimes:
            ds.runtimes['asteroid_checking'] = time.perf_counter() - started
        return ds

    def _get_asteroid_list(self, frame_wcs, time_jd):
        if not self.mpc_states:
            return SkyCoord([], [], unit='deg'), []
        observer = (kete.spice.get_state('Earth', time_jd) if self.pars.observatory_code is None
                    else kete.spice.mpc_code_to_ecliptic(self.pars.observatory_code, time_jd))
        # Derive the cone from the actual footprint, including rectangular CCD corners.
        height, width = frame_wcs.array_shape
        center = frame_wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2)
        corners = frame_wcs.pixel_to_world([0, 0, width - 1, width - 1], [0, height - 1, height - 1, 0])
        radius = center.separation(corners).max() + self.pars.prefilter_padding_deg * u.deg
        radius += self.pars.radius_arcsec * u.arcsec

        def skycoords(states):
            vectors = [(state.pos - observer.pos).change_frame(kete.Frames.Equatorial) for state in states]
            return SkyCoord([v.ra for v in vectors], [v.dec for v in vectors], unit='deg')

        approximate = kete.propagate_two_body(self.mpc_states, time_jd, observer_pos=observer.pos)
        if len(_valid_states(approximate)) != len(approximate):
            raise CatalogUnavailable('Approximate orbit propagation failed for part of the catalog')
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
            raise CatalogUnavailable('Refined orbit propagation failed for asteroid candidates')
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

    def _cross_match_sources_with_asteroids(self, asteroid_sky_coords, asteroid_designations, sources_sky_coords,
                                            radius_threshold=default_radius_threshold):
        """Compatibility helper returning the closest designation per source."""
        return [matches[0]['designation'] if matches else None
                for matches in self._find_neighbors(asteroid_sky_coords, asteroid_designations,
                                                    sources_sky_coords, radius_threshold)]
