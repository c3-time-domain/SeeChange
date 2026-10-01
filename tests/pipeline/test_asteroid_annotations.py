"""Annotation, serialization, and selection tests without the external pipeline services."""

import io
import uuid
from types import SimpleNamespace

import fastavro
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from models.asteroid_match import AsteroidMatchSet
from models.measurements import MeasurementSet, Measurements
from pipeline.alerting import Alerting
from pipeline.asteroid_checker import AsteroidChecker, CatalogUnavailable
from pipeline.data_store import DataStore


def make_datastore():
    measurement = Measurements(ra=10., dec=0., index_in_sources=3)
    ms = MeasurementSet()
    ms.measurements = [measurement]
    ds = DataStore()
    ds._measurement_set = ms
    ds._image = SimpleNamespace(mid_mjd=61000., mjd=60999.9)
    ds.prov_tree = {'asteroid_checking': SimpleNamespace(id=uuid.uuid4())}
    ds.get_provenance = lambda *args: ds.prov_tree['asteroid_checking']
    ds.get_asteroid_match_set = lambda **kwargs: ds.asteroid_match_set
    ds.get_wcs = lambda: SimpleNamespace(wcs=SimpleNamespace(deepcopy=lambda: SimpleNamespace(array_shape=(10, 10))))
    return ds, measurement


def loaded_checker(monkeypatch):
    checker = AsteroidChecker()
    checker.catalog_version = 'fixture-version'
    checker.catalog_epoch_jd = Time(61000., format='mjd', scale='utc').tdb.jd
    monkeypatch.setattr(checker, '_load_states_from_db', lambda: None)
    monkeypatch.setattr(checker, '_get_asteroid_list',
                        lambda wcs, jd: (SkyCoord([10.001, 10.002], [0., 0.], unit='deg'), ['A', 'B']))
    return checker


def test_run_attaches_all_matches_and_does_not_filter(monkeypatch):
    ds, measurement = make_datastore()
    checker = loaded_checker(monkeypatch)
    assert checker.run(ds) is ds
    annotation = ds.asteroid_match_set.annotation(measurement)
    assert annotation['mpcMatchStatus'] == 'checked'
    assert annotation['mpcDesignation'] == 'A'
    assert [m['designation'] for m in annotation['mpcMatches']] == ['A', 'B']
    assert annotation['mpcCatalogVersion'] == 'fixture-version'
    assert ds.measurements == [measurement]
    assert ds.asteroid_match_set.observation_jd == pytest.approx(checker.catalog_epoch_jd)
    assert checker.run(ds) is ds
    assert not checker.has_recalculated


@pytest.mark.parametrize('mode,status', [('disabled', 'disabled'), ('failure', 'unavailable'), ('stale', 'stale')])
def test_annotations_never_remove_measurements(monkeypatch, mode, status):
    ds, measurement = make_datastore()
    checker = loaded_checker(monkeypatch)
    if mode == 'disabled':
        checker.pars.enabled = False
        monkeypatch.setattr(checker, '_load_states_from_db', lambda: pytest.fail('Disabled checker accessed database'))
    elif mode == 'failure':
        def unavailable():
            raise CatalogUnavailable('offline')
        monkeypatch.setattr(checker, '_load_states_from_db', unavailable)
    else:
        checker.catalog_epoch_jd -= 10
    assert checker.run(ds) is ds
    assert ds.asteroid_match_set.status == status
    assert ds.asteroid_match_set.annotation(measurement)['mpcMatches'] == []
    assert ds.measurements == [measurement]


def test_unavailable_check_retried_when_catalog_recovers(monkeypatch):
    ds, measurement = make_datastore()
    checker = loaded_checker(monkeypatch)
    original_load = checker._load_states_from_db

    def offline():
        raise CatalogUnavailable('offline')
    monkeypatch.setattr(checker, '_load_states_from_db', offline)
    checker.run(ds)
    previous_id = ds.asteroid_match_set.id
    assert ds.asteroid_match_set.status == 'unavailable'
    monkeypatch.setattr(checker, '_load_states_from_db', original_load)
    checker.run(ds)
    assert ds.asteroid_match_set.id == previous_id
    assert ds.asteroid_match_set.annotation(measurement)['mpcDesignation'] == 'A'


def source_packet(annotation=None):
    measurement = SimpleNamespace(id=uuid.uuid4(), object_id=uuid.uuid4(), ra=10., dec=0.,
                                  flux_apertures=[1.], flux_apertures_err=[0.1], flux_psf=1., flux_psf_err=0.1)
    image = SimpleNamespace(mid_mjd=61000., filter='r')
    return Alerting(send_alerts=False, methods={}).dia_source_alert(
        measurement, None, image, None, aperdex=0, fluxscale=1., mpc_annotation=annotation)


def test_source_packet_avro_round_trip():
    from pathlib import Path
    schema = fastavro.schema.load_schema(str(Path(__file__).parents[2] / 'share/avsc/ls4.v0_2.diaSource.avsc'))
    match_set = AsteroidMatchSet(status='checked', catalog_version='fixture', catalog_epoch_jd=2461000., matches={})
    measurement = SimpleNamespace(id=uuid.uuid4())
    match_set.matches = {str(measurement.id): [{'designation': 'A', 'distanceArcsec': 3.6},
                                             {'designation': 'B', 'distanceArcsec': 7.2}]}
    packet = source_packet(match_set.annotation(measurement))
    encoded = io.BytesIO()
    fastavro.schemaless_writer(encoded, schema, packet)
    encoded.seek(0)
    decoded = fastavro.schemaless_reader(encoded, schema)
    assert decoded['mpcDesignation'] == 'A'
    assert decoded['mpcMatches'] == packet['mpcMatches']
    assert decoded['mpcMatchStatus'] == 'checked'
    assert decoded['mpcCatalogVersion'] == 'fixture'
    assert source_packet()['mpcMatchStatus'] == 'unavailable'
    # Defaults allow the updated schema to read older source packets.
    import copy
    old_schema = copy.deepcopy(schema)
    added = {'mpcMatches', 'mpcMatchStatus', 'mpcCatalogVersion', 'mpcCatalogEpoch'}
    old_schema['fields'] = [field for field in old_schema['fields'] if field['name'] not in added]
    older_packet = io.BytesIO()
    fastavro.schemaless_writer(older_packet, old_schema, packet)
    older_packet.seek(0)
    resolved = fastavro.schemaless_reader(older_packet, old_schema, schema)
    assert resolved['mpcDesignation'] == 'A'
    assert resolved['mpcMatches'] == []
    assert resolved['mpcMatchStatus'] is None



def test_kafka_sends_matched_and_unmatched_alerts(monkeypatch):
    sent = []

    class Producer:
        def __init__(self, config): pass
        def produce(self, topic, value): sent.append(value)
        def flush(self): pass
    monkeypatch.setattr('pipeline.alerting.confluent_kafka.Producer', Producer)
    packets = [{'diaSource': {'rb': 1., 'rbtype': 'allperfect', 'mpcDesignation': designation}}
               for designation in ['A', None]]
    schema = {'type': 'record', 'name': 'testAlert', 'fields': [
        {'name': 'diaSource', 'type': {'type': 'record', 'name': 'source', 'fields': [
            {'name': 'rb', 'type': 'double'}, {'name': 'rbtype', 'type': 'string'},
            {'name': 'mpcDesignation', 'type': ['string', 'null']}]}}]}
    alerter = Alerting(send_alerts=False, methods={})
    method = {'kafka_server': 'fixture', 'topic': 'fixture', 'deepcut': 0.5, 'schema': schema}
    assert alerter.send_kafka_alerts(packets, method) == 2
    assert len(sent) == 2


def test_pipeline_checks_before_save_and_alerting(monkeypatch):
    from pipeline.top_level import Pipeline

    ds, measurement = make_datastore()
    ds._image.id = uuid.uuid4()
    ds._image.filepath = 'fixture.fits'
    ds._image.preproc_bitflag = 0
    pipeline = Pipeline(pipeline={'generate_report': False, 'save_before_subtraction': False})
    monkeypatch.setattr(pipeline, 'setup_datastore', lambda *args, **kwargs: ds)
    for name in ['preprocessor', 'extractor', 'astrometor', 'photometor', 'subtractor',
                 'detector', 'cutter', 'measurer', 'scorer']:
        monkeypatch.setattr(getattr(pipeline, name), 'run', lambda current: current)
    pipeline.asteroid_checker = loaded_checker(monkeypatch)
    events = []

    def save(step, current):
        assert current.asteroid_match_set.status == 'checked'
        events.append('save')

    def alert(current):
        assert events == ['save']
        assert current.asteroid_match_set.annotation(measurement)['mpcDesignation'] == 'A'
        events.append('alert')
        return current

    monkeypatch.setattr(pipeline, 'save_data_products', save)
    monkeypatch.setattr(pipeline.alerter, 'run', alert)
    assert pipeline.run(ds) is ds
    assert events == ['save', 'alert']
