"""Test alert annotations without requiring pipeline services."""

import copy
import io
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import fastavro
import kete
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from pipeline.alerting import Alerting
from pipeline.asteroid_checker import AsteroidChecker


def make_datastore():
    measurements = [SimpleNamespace(id=uuid.uuid4(), object_id=uuid.uuid4(), ra=ra, dec=0.,
                                   index_in_sources=index, is_bad=False, aper_radii=[1.],
                                   flux_apertures=[1.], flux_apertures_err=[0.1], flux_psf=1., flux_psf_err=0.1)
                    for ra, index in [(10., 3), (20., 7)]]
    ms = SimpleNamespace(measurements=measurements, provenance_id='measurement')
    image = SimpleNamespace(mid_mjd=61000., filter='r', fwhm_estimate=1.,
                            instrument_object=SimpleNamespace(pixel_scale=1.))
    return SimpleNamespace(get_measurement_set=lambda: ms, get_image=lambda: image,
                           get_wcs=lambda: SimpleNamespace(wcs=SimpleNamespace(
                               deepcopy=lambda: SimpleNamespace(array_shape=(10, 10)))))


def loaded_checker(monkeypatch):
    checker = AsteroidChecker()
    epoch = Time(61000., format='mjd', scale='utc').tdb.jd
    checker.mpc_states = [kete.State('A', epoch, [2., 0., 0.], [0., 0.012, 0.])]
    monkeypatch.setattr(checker, '_load_states_from_db', lambda: None)
    monkeypatch.setattr(checker, '_get_asteroid_list',
                        lambda wcs, jd: (SkyCoord([10.001, 10.002], [0., 0.], unit='deg'), ['A', 'B']))
    return checker


@pytest.mark.parametrize('mode', ['checked', 'disabled', 'unavailable', 'stale'])
def test_annotation_status_and_measurement_identity(monkeypatch, mode):
    ds = make_datastore()
    measurements = list(ds.get_measurement_set().measurements)
    checker = loaded_checker(monkeypatch)
    if mode == 'disabled':
        checker.enabled = False
        monkeypatch.setattr(checker, '_load_states_from_db', lambda: pytest.fail('Disabled checker accessed database'))
    elif mode == 'unavailable':
        def offline():
            raise RuntimeError('offline')
        monkeypatch.setattr(checker, '_load_states_from_db', offline)
    elif mode == 'stale':
        checker.mpc_states = [kete.State('A', 2400000., [2., 0., 0.], [0., 0.012, 0.])]
    annotations = checker.check(ds)
    assert all(a['mpcMatchStatus'] == mode for a in annotations.values())
    assert ds.get_measurement_set().measurements == measurements
    first = annotations[str(measurements[0].id)]
    assert first['mpcDesignation'] == ('A' if mode == 'checked' else None)
    assert [m['designation'] for m in first['mpcMatches']] == (['A', 'B'] if mode == 'checked' else [])
    assert annotations[str(measurements[1].id)]['mpcMatches'] == []


def test_build_alerts_and_avro_round_trip(monkeypatch):
    ds = make_datastore()
    measurements = ds.get_measurement_set().measurements
    ds.get_sub_image = lambda: SimpleNamespace(provenance_id='subtraction')
    ds.get_zp = lambda: SimpleNamespace(zp=31.4)
    ds.get_detections = lambda: None
    ds.get_deepscore_set = lambda: SimpleNamespace(
        deepscores=[SimpleNamespace(score=1.) for m in measurements], algorithm='allperfect', provenance_id='score')
    cutout = {key: np.zeros((2, 2)) for key in ['new_data', 'ref_data', 'sub_data',
                                             'new_flags', 'ref_flags', 'sub_flags']}
    ds.get_cutouts = lambda: SimpleNamespace(
        load_all_co_data=lambda **kwargs: None,
        co_dict={f'source_index_{m.index_in_sources}': cutout for m in measurements})
    monkeypatch.setattr('pipeline.alerting.SmartSession', lambda: nullcontext())
    monkeypatch.setattr('pipeline.alerting.Object.get_by_id', lambda *args, **kwargs: SimpleNamespace(
        get_measurements_et_al=lambda *args, **kwargs: {'measurements': []}))
    monkeypatch.setattr('pipeline.alerting.Image.find_images', lambda **kwargs: [])
    alerter = Alerting(send_alerts=False, methods={})
    alerter.asteroid_checker = loaded_checker(monkeypatch)
    monkeypatch.setattr(alerter, 'dia_object_alert', lambda *args, **kwargs: None)
    packets = alerter.build_avro_alert_structures(ds)
    assert len(packets) == 2  # Both matched and unmatched sources still produce alerts.
    assert packets[0]['diaSource']['mpcDesignation'] == 'A'
    assert packets[1]['diaSource']['mpcMatches'] == []
    schema = fastavro.schema.load_schema(str(Path(__file__).parents[2] / 'share/avsc/ls4.v0_2.alert.avsc'))
    encoded = io.BytesIO()
    fastavro.schemaless_writer(encoded, schema, packets[0])
    encoded.seek(0)
    decoded = fastavro.schemaless_reader(encoded, schema)
    assert decoded['diaSource']['mpcMatches'] == packets[0]['diaSource']['mpcMatches']
    assert decoded['diaSource']['mpcMatchStatus'] == 'checked'
    # A failed check must also leave both alerts available.

    def offline():
        raise RuntimeError('offline')
    monkeypatch.setattr(alerter.asteroid_checker, '_load_states_from_db', offline)
    packets = alerter.build_avro_alert_structures(ds)
    assert len(packets) == 2
    assert all(p['diaSource']['mpcMatchStatus'] == 'unavailable' for p in packets)


def test_schema_reads_old_sources():
    schema = fastavro.schema.load_schema(str(Path(__file__).parents[2] / 'share/avsc/ls4.v0_2.diaSource.avsc'))
    old_schema = copy.deepcopy(schema)
    added = {'mpcDesignation', 'mpcMatches', 'mpcMatchStatus'}
    old_schema['fields'] = [f for f in old_schema['fields'] if f['name'] not in added]
    ds = make_datastore()
    packet = Alerting(send_alerts=False, methods={}).dia_source_alert(
        ds.get_measurement_set().measurements[0], None, ds.get_image(), None, aperdex=0, fluxscale=1.)
    encoded = io.BytesIO()
    fastavro.schemaless_writer(encoded, old_schema, packet)
    encoded.seek(0)
    decoded = fastavro.schemaless_reader(encoded, old_schema, schema)
    assert decoded['mpcDesignation'] is None
    assert decoded['mpcMatches'] == []
    assert decoded['mpcMatchStatus'] is None
