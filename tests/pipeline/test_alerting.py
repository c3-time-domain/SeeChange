import random
import pytest
import re
import io

import fastavro
import confluent_kafka

from models.object import Object
from models.deepscore import DeepScore
from util.config import Config
from pipeline.alerting import Alerting

def test_build_avro_alert_structures( test_config, decam_datastore_through_scoring ):
    ds = decam_datastore_through_scoring
    fluxscale = 10** ( ( ds.zp.zp - 27.5 ) / -2.5 )

    alerter = Alerting()
    alerts = alerter.build_avro_alert_structures( ds )

    assert len(alerts) == len(ds.measurements)
    assert all( isinstance( a['alertId'], str ) for a in alerts )
    assert all( len(a['alertId']) == 36 for a in alerts )

    assert all( a['diaSource']['diaSourceId'] == str(m.id) for a, m in zip( alerts, ds.measurements ) )
    assert all( a['diaSource']['diaObjectId'] == str(m.object_id) for a, m in zip( alerts, ds.measurements ) )
    assert all( a['diaSource']['MJD'] == pytest.approx( ds.image.mid_mjd, abs=0.0001 ) for a in alerts )
    assert all( a['diaSource']['ra'] == pytest.approx( m.ra, abs=0.1/3600. )
                for a, m in zip( alerts, ds.measurements ) )
    assert all( a['diaSource']['dec'] == pytest.approx( m.dec, abs=0.1/3600. )
                for a, m in zip( alerts, ds.measurements ) )
    assert all( a['diaSource']['fluxZeroPoint'] == 27.5 for a in alerts )
    assert all( a['diaSource']['psfFlux'] == pytest.approx( m.flux_psf * fluxscale, rel=1e-5 )
                for a, m in zip( alerts, ds.measurements ) )
    assert all( a['diaSource']['psfFluxErr'] == pytest.approx( m.flux_psf_err * fluxscale, rel=1e-5 )
                for a, m in zip( alerts, ds.measurements ) )
    assert all( a['diaSource']['apFlux'] == pytest.approx( m.flux_apertures[0] * fluxscale, rel=1e-5 )
                for a, m in zip( alerts, ds.measurements ) )
    assert all( a['diaSource']['apFluxErr'] == pytest.approx( m.flux_apertures_err[0] * fluxscale, rel=1e-5 )
                for a, m in zip( alerts, ds.measurements ) )

    assert all( a['diaSource']['rbtype'] == s.algorithm for a, s in zip( alerts, ds.scores ) )
    assert all( a['diaSource']['rbcut'] == DeepScore.get_rb_cut( s.algorithm ) for a, s in zip( alerts, ds.scores ) )
    assert all( a['diaSource']['rb'] == pytest.approx( s.score, rel=0.001 ) for a, s in zip( alerts, ds.scores ) )

    assert all( a['diaObject']['diaObjectId'] == str( m.object_id ) for a, m in zip( alerts, ds.measurements ) )
    for a, m in zip( alerts, ds.measurements ):
        obj = Object.get_by_id( m.object_id )
        assert a['diaObject']['name'] == obj.name
        assert a['diaObject']['ra'] == pytest.approx( obj.ra, abs=0.1/3600. )
        assert a['diaObject']['dec'] == pytest.approx( obj.dec, abs=0.1/3600. )

    assert all( len(a['cutoutScience']) == 41 * 41 * 4 for a in alerts )
    assert all( len(a['cutoutTemplate']) == 41 * 41 * 4 for a in alerts )
    assert all( len(a['cutoutDifference']) == 41 * 41 * 4 for a in alerts )

    # TODO : test the actual values in the cutouts

    # TODO : this will require bigger and longer fixtures to really test, sadly!  We need to
    #   test prvDiaSources and prvNonDetectionLimits.
    # (prvDiaForcedSources is an even bigger ball of wax since at the moment we have no plans
    # for a forced source pipeline.  So, right now, verifing that prvDiaForcedSources is None
    # is actually a complete test of expected pipeline behavior.)

    assert all( len(a['prvDiaSources']) == 0 for a in alerts )
    assert all( a['prvDiaForcedSources'] is None for a in alerts )
    assert all( len(a['prvDiaNonDetectionLimits']) == 0 for a in alerts )


def test_send_alerts( test_config,decam_datastore_through_scoring ):
    ds = decam_datastore_through_scoring

    alerter = Alerting()
    # The test config has "{barf}" in the kafka topic.  The reason for
    #  this is that It isn't really possible to clean up after the tests
    #  by clearing out the kafka server, so instead of cleaning up after
    #  ourselves, we'll just point to a different topic each time, which
    #  for testing purposes should be close enough.  (It does mean if you
    #  leave a test environment open for a long time, stuff will build up
    #  on the test kafka server.)
    topic = alerter.methods[0]['topic']
    assert re.search( '^test_topic_[a-z]{6}$', topic )

    alerter.send( ds )

    groupid = f'test_{"".join(random.choices("abcdefghijklmnopqrstuvwxyz",k=10))}'
    consumer = confluent_kafka.Consumer( { 'bootstrap.servers': test_config.value('alerts.methods.0.kafka_server'),
                                           'auto.offset.reset': 'earliest',
                                           'group.id': groupid } )
    consumer.subscribe( [ alerter.methods[0]['topic'] ] )
    msgs = consumer.consume( 100, timeout=1 )

    measurements_seen = set()
    for msg in msgs:
        alert = fastavro.schemaless_reader( io.BytesIO( msg.value() ), alerter.methods[0]['schema'] )
        dex = [ i for i in range(len(ds.measurements))
                if str( ds.measurements[i].id ) == alert['diaSource']['diaSourceId'] ]
        assert len(dex) > 0
        dex = dex[0]
        measurements_seen.add( ds.measurements[dex].id )

        assert alert['diaSource']['MJD'] == pytest.approx( ds.image.mid_mjd, abs=0.0001 )
        assert alert['diaSource']['ra'] == pytest.approx( ds.measurements[dex].ra, abs=0.1/3600. )
        assert alert['diaSource']['dec'] == pytest.approx( ds.measurements[dex].dec, abs=0.1/3600. )
        assert alert['diaSource']['fluxZeroPoint'] == 27.5
        fluxscale = 10 ** ( ( ds.zp.zp - 27.5 ) / -2.5 )
        assert alert['diaSource']['psfFlux'] == pytest.approx( ds.measurements[dex].flux_psf * fluxscale, rel=1e-5 )
        assert alert['diaSource']['psfFluxErr'] == pytest.approx( ds.measurements[dex].flux_psf_err * fluxscale,
                                                                  rel=1e-5 )
        assert alert['diaSource']['apFlux'] == pytest.approx( ds.measurements[dex].flux_apertures[0] * fluxscale,
                                                              rel=1e-5 )
        assert alert['diaSource']['apFluxErr'] == pytest.approx( ds.measurements[dex].flux_apertures_err[0]
                                                                 * fluxscale, rel=1e-5 )
        assert alert['diaObject']['diaObjectId'] == str( ds.measurements[dex].object_id )

        assert alert['diaSource']['rbtype'] == ds.scores[dex].algorithm
        assert alert['diaSource']['rbcut'] == DeepScore.get_rb_cut( ds.scores[dex].algorithm )
        assert alert['diaSource']['rb'] == pytest.approx( ds.scores[dex].score, rel=0.001 )

        assert len(alert['cutoutScience']) == 41 * 41 * 4
        assert len(alert['cutoutTemplate']) == 41 * 41 * 4
        assert len(alert['cutoutDifference']) == 41 * 41 * 4

        # TODO : check the actual image cutout data

        assert len( alert['prvDiaSources'] ) == 0
        assert alert['prvDiaForcedSources'] is None
        assert len( alert['prvDiaNonDetectionLimits'] ) == 0

    # The test had None configured for its r/b cutoff, so it should be using the default
    cut = DeepScore.get_rb_cut( ds.scores[0].algorithm )
    assert measurements_seen == set( m.id for m, s in zip( ds.measurements, ds.scores ) if s.score >= cut )
