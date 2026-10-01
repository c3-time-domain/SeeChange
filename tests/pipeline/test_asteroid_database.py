"""Exercise real SQL migrations and persistence in a disposable PostgreSQL schema.

Set SEECHANGE_ASTEROID_TEST_DSN to run independently of the full pipeline fixtures.
"""

import importlib.util
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import kete
import psycopg
from psycopg import sql
import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from models import base
from models.measurements import MeasurementSet, Measurements
from models.object import Object  # noqa: F401 -- registers the Measurements object_id FK type
from models.asteroid_match import AsteroidMatchSet
from pipeline import asteroid_checker as module
from pipeline.data_store import DataStore


def migration(filename):
    path = Path(__file__).parents[2] / 'alembic/versions' / filename
    spec = importlib.util.spec_from_file_location('asteroid_migration', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def database(monkeypatch):
    dsn = os.getenv('SEECHANGE_ASTEROID_TEST_DSN')
    if not dsn:
        pytest.skip('Set SEECHANGE_ASTEROID_TEST_DSN for disposable PostgreSQL integration checks')
    schema = 'asteroid_test_' + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    parameters = psycopg.conninfo.conninfo_to_dict(dsn)
    parameters.update(options=f'-csearch_path={schema}', client_encoding='UTF8')
    engine = sa.create_engine('postgresql+psycopg://', connect_args=parameters)
    monkeypatch.setattr(base, '_psycopg_params', parameters)
    monkeypatch.setattr(base, '_engine', engine)
    monkeypatch.setattr(base, '_Session', sa.orm.sessionmaker(bind=engine, expire_on_commit=False))
    try:
        with engine.begin() as conn:
            conn.execute(sa.text('CREATE TABLE provenances (_id varchar PRIMARY KEY)'))
            conn.execute(sa.text('CREATE TABLE cutouts (_id uuid PRIMARY KEY)'))
            conn.execute(sa.text('CREATE TABLE objects (_id uuid PRIMARY KEY)'))
            MeasurementSet.__table__.create(conn)
            # These focused tests do not require the production q3c spatial index.
            conn.execute(sa.schema.CreateTable(Measurements.__table__))
            with Operations.context(MigrationContext.configure(conn)):
                migration('2025_12_18_0539-8c3f73f0ede0_create_mpc_table.py').upgrade()
                migration('2026_10_01_0001-a61e0d3cf902_asteroid_annotations.py').upgrade()
        yield engine
    finally:
        engine.dispose()
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


def refresh(monkeypatch, states):
    monkeypatch.setattr(module.kete.mpc, 'fetch_known_orbit_data', lambda **kwargs: None)
    monkeypatch.setattr(module.kete.mpc, 'table_to_states', lambda data: states)
    return module.download_and_update_db(2461000.5)


def test_catalog_refresh_reload_and_atomic_rollback(database, monkeypatch):
    state = kete.State('fixture', 2461000.5, [2., 0., 0.], [0., 0.012, 0.])
    version = refresh(monkeypatch, [state])
    checker = module.AsteroidChecker()
    checker._load_states_from_db()
    assert checker.catalog_version == version
    assert [s.desig for s in checker.mpc_states] == ['fixture']
    cached_states = checker.mpc_states
    checker._load_states_from_db()
    assert checker.mpc_states is cached_states
    # A duplicate designation fails after DELETE, exercising actual transaction rollback.
    with pytest.raises(psycopg.errors.UniqueViolation):
        refresh(monkeypatch, [state, state])
    checker._load_states_from_db()
    assert checker.catalog_version == version
    assert len(checker.mpc_states) == 1
    new_version = refresh(monkeypatch, [state])
    checker._load_states_from_db()
    assert checker.catalog_version == new_version != version
    assert checker.mpc_states is not cached_states


def test_datastore_save_reload_and_measurement_identity(database):
    upstream, checking, cutouts = "upstream_fixture", "checking_fixture", uuid.uuid4()
    with database.begin() as conn:
        conn.execute(sa.text('INSERT INTO provenances VALUES (:id)'), [{'id': upstream}, {'id': checking}])
        conn.execute(sa.text('INSERT INTO cutouts VALUES (:id)'), {'id': cutouts})
    ms = MeasurementSet(cutouts_id=cutouts, provenance_id=upstream)
    ms.measurements = []
    first, second = SimpleNamespace(id=uuid.uuid4()), SimpleNamespace(id=uuid.uuid4())
    ds = DataStore()
    ds._measurement_set = ms
    ds.asteroid_match_set = AsteroidMatchSet(
        measurementset_id=ms.id, provenance_id=checking, status='checked',
        catalog_version='fixture', catalog_epoch_jd=2461000.5, observation_jd=2461000.5,
        matches={str(first.id): [{'designation': 'nearby', 'distanceArcsec': 4.}], str(second.id): []},
    )
    ds.save_and_commit()
    assert 'asteroid_match_set' in ds.products_committed
    reloaded = DataStore()
    reloaded._measurement_set = ms
    reloaded.prov_tree = {'asteroid_checking': SimpleNamespace(id=checking, _id=checking)}
    result = reloaded.get_asteroid_match_set()
    assert result.from_db
    assert result.annotation(first)['mpcDesignation'] == 'nearby'
    assert result.annotation(second)['mpcDesignation'] is None
    # Annotation lookup must follow UUIDs, including when source indices have gaps or order changes.
    ms._measurements = [second, first]
    assert reloaded.get_mpc_designations() == [None, 'nearby']
    # The saved annotation also survives alert-schema serialization after reload.
    import io
    import fastavro
    from tests.pipeline.test_asteroid_annotations import source_packet
    packet = source_packet(result.annotation(first))
    schema = fastavro.schema.load_schema(str(Path(__file__).parents[2] / 'share/avsc/ls4.v0_2.diaSource.avsc'))
    encoded = io.BytesIO()
    fastavro.schemaless_writer(encoded, schema, packet)
    encoded.seek(0)
    decoded = fastavro.schemaless_reader(encoded, schema)
    assert decoded['mpcDesignation'] == 'nearby'
    assert decoded['mpcMatches'] == [{'designation': 'nearby', 'distanceArcsec': 4.}]
    ms._measurements = []
    # A changed matching configuration/provenance must not reuse an old cached result.
    reloaded.prov_tree = {'asteroid_checking': SimpleNamespace(id='new_config', _id='new_config')}
    assert reloaded.get_asteroid_match_set() is None
    reloaded.prov_tree = {'asteroid_checking': SimpleNamespace(id=checking, _id=checking)}
    assert reloaded.get_asteroid_match_set().annotation(first)['mpcDesignation'] == 'nearby'
    # Replacement measurement sets invalidate the cached annotation.
    reloaded.measurement_set = None
    assert reloaded.asteroid_match_set is None


def test_migration_downgrade(database):
    with database.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            migration('2026_10_01_0001-a61e0d3cf902_asteroid_annotations.py').downgrade()
        inspector = sa.inspect(conn)
        assert 'asteroid_match_sets' not in inspector.get_table_names()
        assert 'mpc_orbit_catalog' not in inspector.get_table_names()
        assert 'mpc_table' in inspector.get_table_names()


def test_prediction_to_persisted_alert(database, monkeypatch):
    import io
    import fastavro
    from astropy.time import Time
    from astropy.wcs import WCS
    from pipeline.alerting import Alerting

    state = kete.State('fixture-asteroid', 2461000.5, [2., 0., 0.], [0., 0.012, 0.])
    refresh(monkeypatch, [state])
    observer = kete.spice.get_state('Earth', state.jd)
    apparent = kete.propagate_two_body([state], state.jd, observer_pos=observer.pos)[0]
    pos = (apparent.pos - observer.pos).change_frame(kete.Frames.Equatorial)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [512., 256.]
    wcs.wcs.crval = [pos.ra, pos.dec]
    wcs.wcs.cdelt = [-0.00027, 0.00027]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.array_shape = (512, 1024)
    upstream, checking = 'measurement_fixture', 'checker_fixture'
    cutouts, object_id = uuid.uuid4(), uuid.uuid4()
    with database.begin() as conn:
        conn.execute(sa.text('INSERT INTO provenances VALUES (:id)'), [{'id': upstream}, {'id': checking}])
        conn.execute(sa.text('INSERT INTO cutouts VALUES (:id)'), {'id': cutouts})
        conn.execute(sa.text('INSERT INTO objects VALUES (:id)'), {'id': object_id})
    ms = MeasurementSet(cutouts_id=cutouts, provenance_id=upstream)
    ms.measurements = [Measurements(
        ra=pos.ra, dec=pos.dec, index_in_sources=7, object_id=object_id,
        flux_psf=1., flux_psf_err=0.1, flux_apertures=[1.], flux_apertures_err=[0.1], aper_radii=[1.],
        center_x_pixel=512, center_y_pixel=256, x=512., y=256., gfit_x=512., gfit_y=256.,
        major_width=1., minor_width=1., position_angle=0., is_bad=False,
    )]
    ms.upsert(load_defaults=True)
    checker = module.AsteroidChecker()
    prov = SimpleNamespace(id=checking, _id=checking, parameters=checker.pars.get_critical_pars())
    ds = DataStore()
    ds._measurement_set = ms
    ds.prov_tree = {'asteroid_checking': prov}
    image = SimpleNamespace(mid_mjd=Time(state.jd, format='jd', scale='tdb').utc.mjd, filter='r')
    ds._image = image
    monkeypatch.setattr(ds, 'get_wcs', lambda: SimpleNamespace(wcs=wcs))
    assert checker.run(ds) is ds
    assert ds.asteroid_match_set.status == 'checked'
    assert ds.asteroid_match_set.annotation(ms.measurements[0])['mpcDesignation'] == 'fixture-asteroid'
    ds._image = None  # The synthetic image is an input fixture, not a file product to save.
    ds.save_and_commit()
    reloaded = DataStore()
    reloaded._measurement_set = MeasurementSet.get_by_id(ms.id)
    reloaded.prov_tree = ds.prov_tree
    measurement = reloaded.measurements[0]
    annotation = reloaded.get_asteroid_match_set().annotation(measurement)
    packet = Alerting(send_alerts=False, methods={}).dia_source_alert(
        measurement, None, image, None, aperdex=0, fluxscale=1., mpc_annotation=annotation)
    schema = fastavro.schema.load_schema(str(Path(__file__).parents[2] / 'share/avsc/ls4.v0_2.diaSource.avsc'))
    encoded = io.BytesIO()
    fastavro.schemaless_writer(encoded, schema, packet)
    encoded.seek(0)
    decoded = fastavro.schemaless_reader(encoded, schema)
    assert decoded['diaSourceId'] == str(measurement.id)
    assert decoded['mpcDesignation'] == 'fixture-asteroid'
    assert decoded['mpcMatches'][0]['distanceArcsec'] < 0.1
