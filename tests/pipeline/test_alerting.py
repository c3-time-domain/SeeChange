import random
import pytest
import re
import io

import numpy as np

import fastavro
import confluent_kafka

from models.object import Object, ObjectLegacySurveyMatch
from models.deepscore import DeepScoreSet
from pipeline.alerting import Alerting


def test_build_avro_alert_structures( test_config, decam_datastore_through_scoring ):
    ds = decam_datastore_through_scoring
    fluxscale = 10** ( ( ds.zp.zp - 31.4 ) / -2.5 )

    alerter = Alerting()
    alerts = alerter.build_avro_alert_structures( ds, skip_bad=False )

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
    assert all( a['diaSource']['fluxZeroPoint'] == pytest.approx( 31.4, rel=1e-5 ) for a in alerts )
    assert all( a['diaSource']['psfFlux'] == pytest.approx( m.flux_psf * fluxscale, rel=1e-5 )
                for a, m in zip( alerts, ds.measurements ) if not np.isnan(m.flux_psf)  )
    assert all( a['diaSource']['psfFluxErr'] == pytest.approx( m.flux_psf_err * fluxscale, rel=1e-5 )
                for a, m in zip( alerts, ds.measurements ) if not np.isnan(m.flux_psf_err) )
    assert all( a['diaSource']['apFlux'] == pytest.approx( m.flux_apertures[0] * fluxscale, rel=1e-5 )
                for a, m in zip( alerts, ds.measurements ) if not np.isnan(m.flux_psf) )
    assert all( a['diaSource']['apFluxErr'] == pytest.approx( m.flux_apertures_err[0] * fluxscale, rel=1e-5 )
                for a, m in zip( alerts, ds.measurements ) if not np.isnan(m.flux_psf) )

    assert all( a['diaSource']['rbtype'] == ds.deepscore_set.algorithm for a in alerts )
    assert all( a['diaSource']['rbcut'] == DeepScoreSet.get_rb_cut( ds.deepscore_set.algorithm ) for a in alerts )
    assert all( a['diaSource']['rb'] == pytest.approx( s.score, rel=0.001 ) for a, s in zip( alerts, ds.deepscores ) )

    assert all( a['diaObject']['diaObjectId'] == str( m.object_id ) for a, m in zip( alerts, ds.measurements ) )
    at_least_some_had_matches = False
    for a, m in zip( alerts, ds.measurements ):
        obj = Object.get_by_id( m.object_id )
        lsmatches = ObjectLegacySurveyMatch.get_object_matches( obj.id )
        if len(lsmatches) > 0:
            at_least_some_had_matches = True
        assert a['diaObject']['name'] == obj.name
        assert a['diaObject']['ra'] == pytest.approx( obj.ra, abs=0.1/3600. )
        assert a['diaObject']['dec'] == pytest.approx( obj.dec, abs=0.1/3600. )
        assert a['diaObject']['xgmatchRadius'] == test_config.value( 'liumatch.radius' )
        # Matches should always be sorted by distance
        zippy = zip( lsmatches, a['diaObject']['ls-xgboost'] )
        assert all(m.lsid == am['lsid'] for m, am in zippy )
        assert all(m.ra == pytest.approx( am['ra'], abs=0.1/3600 ) for m, am in zippy )
        assert all(m.dec == pytest.approx( am['dec'], abs=0.1/3600 ) for m, am in zippy )
        assert all(m.dist == pytest.approx( am['dist'], abs=0.1/3600 ) for m, am in zippy )
        assert all(m.white_mag == pytest.approx( am['white_mag'], abs=0.001 ) for m, am in zippy )
        assert all(m.xgboost == pytest.approx( am['ls-xbgoost'], abs=0.001 ) for m, am in zippy )
        assert all(m.is_star == am['is_star'] for m, am in zippy )
    assert at_least_some_had_matches
    assert all( len(a['cutoutScience']) == 41 * 41 * 4 for a in alerts )
    assert all( len(a['cutoutTemplate']) == 41 * 41 * 4 for a in alerts )
    assert all( len(a['cutoutDifference']) == 41 * 41 * 4 for a in alerts )

    # TODO : test the actual values in the cutouts

    assert all( len(a['prvDiaSources']) == 0 for a in alerts )
    assert all( a['prvDiaForcedSources'] is None for a in alerts )
    assert all( len(a['prvDiaNonDetectionLimits']) == 0 for a in alerts )

    # Make sure that if we skip is_bad measurements (which is the
    # default), we will get fewer (but still some).  (Actually,
    # we may not get fewer: if the deletion thresholds are the
    # same as the bad thresholds, then no is_bad measurements
    # will have been saved in the first place.)
    alerts = alerter.build_avro_alert_structures( ds )
    assert len(alerts) > 2
    assert len(alerts) <= len(ds.measurements)


def test_send_alerts( test_config, decam_datastore_through_scoring ):
    ds = decam_datastore_through_scoring

    nalerts = {}
    for skip_bad in [ False, True ]:
        alerter = Alerting()
        # The test config has "{barf}" in the kafka topic.  The reason for
        #  this is that It isn't really possible to clean up after the tests
        #  by clearing out the kafka server, so instead of cleaning up after
        #  ourselves, we'll just point to a different topic each time, which
        #  for testing purposes should be close enough.  (It does mean if you
        #  leave a test environment open for a long time, stuff will build up
        #  on the test kafka server.)
        topic = alerter.methods['test_alert_stream']['topic']
        assert re.search( '^test_topic_[a-z]{6}$', topic )

        alerter.send( ds, skip_bad=skip_bad )

        groupid = f'test_{"".join(random.choices("abcdefghijklmnopqrstuvwxyz",k=10))}'
        consumer = confluent_kafka.Consumer(
            { 'bootstrap.servers': test_config.value('alerts.methods.test_alert_stream.kafka_server'),
              'auto.offset.reset': 'earliest',
              'group.id': groupid }
        )
        consumer.subscribe( [ alerter.methods['test_alert_stream']['topic'] ] )

        # I have noticed that the very first time I run this test within a
        #   docker compose environment, and the very first time I send
        #   alerts to the kafka server, they don't show up here with a
        #   consumer.consume call command.  If I rerun the tests, it works a
        #   second time.  If I rerun the consume task, if finds them.  In
        #   practical usage, one would be repeatedly polling the consumer,
        #   so if they don't come through one time, but do the next, then
        #   things are basically working.  So, to hack around the error I've
        #   seen, put in a poll loop that continues until we get messages
        #   once, and then stops once we aren't.
        gotsome = False
        done = False
        msgs = []
        while not done:
            newmsgs = consumer.consume( 100, timeout=1 )
            if gotsome and ( len(newmsgs) == 0 ):
                done = True
            if len( newmsgs ) > 0:
                msgs.extend( newmsgs )
                gotsome = True
        nalerts[skip_bad] = len(msgs)

        if skip_bad:
            measurements = [ m for m in ds.measurements if not m.is_bad ]
            scores = [ s for s, m in zip( ds.deepscores, ds.measurements ) if not m.is_bad ]
        else:
            measurements = ds.measurements
            scores = ds.deepscores

        measurements_seen = set()
        for msg in msgs:
            alert = fastavro.schemaless_reader( io.BytesIO( msg.value() ),
                                                alerter.methods['test_alert_stream']['schema'] )
            dex = [ i for i in range(len(measurements))
                    if str( measurements[i].id ) == alert['diaSource']['diaSourceId'] ]
            assert len(dex) > 0
            dex = dex[0]
            measurements_seen.add( measurements[dex].id )

            assert alert['diaSource']['MJD'] == pytest.approx( ds.image.mid_mjd, abs=0.0001 )
            assert alert['diaSource']['ra'] == pytest.approx( measurements[dex].ra, abs=0.1/3600. )
            assert alert['diaSource']['dec'] == pytest.approx( measurements[dex].dec, abs=0.1/3600. )
            assert alert['diaSource']['fluxZeroPoint'] == pytest.approx( 31.4, rel=1e-5 )
            fluxscale = 10 ** ( ( ds.zp.zp - 31.4 ) / -2.5 )
            if not np.isnan( measurements[dex].flux_psf ):
                assert alert['diaSource']['psfFlux'] == pytest.approx( measurements[dex].flux_psf * fluxscale,
                                                                       rel=1e-5 )
                assert alert['diaSource']['psfFluxErr'] == pytest.approx( measurements[dex].flux_psf_err * fluxscale,
                                                                          rel=1e-5 )
                assert alert['diaSource']['apFlux'] == pytest.approx( measurements[dex].flux_apertures[0] * fluxscale,
                                                                      rel=1e-5 )
                assert alert['diaSource']['apFluxErr'] == pytest.approx( measurements[dex].flux_apertures_err[0]
                                                                         * fluxscale, rel=1e-5 )
            assert alert['diaObject']['diaObjectId'] == str( measurements[dex].object_id )
            lsmatches = ObjectLegacySurveyMatch.get_object_matches( measurements[dex].object_id )
            zippy = zip( lsmatches, alert['diaObject']['ls-xgboost'] )
            assert all(m.lsid == am['lsid'] for m, am in zippy )
            assert all(m.ra == pytest.approx( am['ra'], abs=0.1/3600 ) for m, am in zippy )
            assert all(m.dec == pytest.approx( am['dec'], abs=0.1/3600 ) for m, am in zippy )
            assert all(m.dist == pytest.approx( am['dist'], abs=0.1/3600 ) for m, am in zippy )
            assert all(m.white_mag == pytest.approx( am['white_mag'], abs=0.001 ) for m, am in zippy )
            assert all(m.xgboost == pytest.approx( am['ls-xbgoost'], abs=0.001 ) for m, am in zippy )
            assert all(m.is_star == am['is_star'] for m, am in zippy )

            assert alert['diaSource']['rbtype'] == ds.deepscore_set.algorithm
            assert alert['diaSource']['rbcut'] == pytest.approx( DeepScoreSet.get_rb_cut( ds.deepscore_set.algorithm ),
                                                                 rel=1e-6 )
            assert alert['diaSource']['rb'] == pytest.approx( scores[dex].score, rel=0.001 )

            assert len(alert['cutoutScience']) == 41 * 41 * 4
            assert len(alert['cutoutTemplate']) == 41 * 41 * 4
            assert len(alert['cutoutDifference']) == 41 * 41 * 4

            # TODO : check the actual image cutout data

            assert len( alert['prvDiaSources'] ) == 0
            assert alert['prvDiaForcedSources'] is None
            assert len( alert['prvDiaNonDetectionLimits'] ) == 0

        # The test had None configured for its r/b cutoff, so it should be using the default
        cut = DeepScoreSet.get_rb_cut( ds.deepscore_set.algorithm )
        assert measurements_seen == set( m.id for m, s in zip( measurements, scores ) if s.score >= cut )

    assert nalerts[True] > 0
    assert nalerts[False] > 0
    assert nalerts[True] <= nalerts[False]


# The simulated lightcurve will have previous detections, so use that
#  fixture here.  Think about Issue #367, but don't do anything about it.
def test_alerts_with_previous( test_config, sim_lightcurve_complete_dses ):
    # The fixture will have sent alerts
    _ref, _refds, newdsen, pips = sim_lightcurve_complete_dses

    # OK.  If I understand my fixture right, it will have run the
    # pipelines on newdsen all the way through alerting.  So, the
    # previous detections of a given object in alerts from a given ds
    # should only be the ones from the newdsen earlier in the array.

    some_did_have_previous = False
    some_did_have_nondet = False
    for curdex in range( len(newdsen) ):
        pip = pips[ curdex ]
        curds = newdsen[ curdex ]
        topic = pip.alerter.methods['test_alert_stream']['topic']
        groupid = f'test_{"".join(random.choices("abcdefghijklmnopqrstuvwxyz",k=10))}'
        consumer = confluent_kafka.Consumer(
            { 'bootstrap.servers': test_config.value('alerts.methods.test_alert_stream.kafka_server'),
              'auto.offset.reset': 'earliest',
              'group.id': groupid } )
        consumer.subscribe( [ topic ] )
        gotsome = False
        done = False
        msgs = []
        while not done:
            # Every ds' pipeline should have sent alerts, so this shouldn't be an infinite loop
            newmsgs = consumer.consume( 100, timeout=1 )
            if gotsome and ( len(newmsgs) == 0 ):
                done = True
            if len( newmsgs ) > 0:
                msgs.extend( newmsgs )
                gotsome = True

        measfound = set()
        for msg in msgs:
            alert = fastavro.schemaless_reader( io.BytesIO( msg.value() ),
                                                pip.alerter.methods['test_alert_stream']['schema'] )
            # Figure out which measurement this alert goes with
            dex = [ i for i in range( len(curds.measurements))
                    if str( curds.measurements[i].id ) == alert['diaSource']['diaSourceId'] ]
            assert len(dex) > 0
            dex = dex[0]
            measfound.add( curds.measurements[dex].id )

            oldmeas = []
            nondet = []
            for prevdex in range( 0, curdex ):
                prevds = newdsen[ prevdex ]
                thisoldmeas = [ m for m in prevds.measurements
                                if str(m.object_id) == str(curds.measurements[dex].object_id) ]
                assert len(thisoldmeas) < 2
                if len(thisoldmeas) > 0:
                    oldmeas.append( thisoldmeas[0] )
                else:
                    nondet.append( ( prevds.sub_image.mjd,
                                     prevds.sub_image.filter,
                                     prevds.sub_image.lim_mag_estimate ) )

            if len( oldmeas ) > 0 :
                some_did_have_previous = True
            if len( nondet ) > 0:
                some_did_have_nondet = True

            # I'm depending on newdsen being sorted by mjd here, so that
            #    the previous measurements and previous nondetections#
            #    will be found in the same order as they arein the alerts.
            zippy = zip( oldmeas, alert['prvDiaSources'] )
            assert all( str(m.id) == a['diaSourceId'] for m, a in zippy )
            # Can't compare flux directly without getting more stuff out of the
            #   database; we'd need the sub image zeropoint.  (The alert has
            #   converted to nJy.)  TODO, do this.
            assert all( m.ra == pytest.approx( a['ra'], abs=0.1/3600. ) for m, a in zippy )
            assert all( m.dec == pytest.approx( a['dec'], abs=0.1/3600. ) for m, a in zippy )

            zippy = zip( nondet, alert['prvDiaNonDetectionLimits'] )
            assert all( nondet[0] == pytest.approx( a['MJD'], abs=1./3600./24. ) for n, a in zippy )
            assert all( nondet[1] == a['band'] for n, a in zippy )
            assert all( nondet[2] == pytest.approx( a['limitingMag'], abs=0.01 ) for n, a in zippy )

    assert some_did_have_previous
    assert some_did_have_nondet
