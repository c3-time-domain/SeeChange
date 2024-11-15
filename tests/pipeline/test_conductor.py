import pytest
import time

import datetime
import dateutil.parser
import requests
# Disable warnings from urllib, since there will be lots about insecure connections
#  given that we're using a self-signed cert for the server in the test environment
requests.packages.urllib3.disable_warnings()

import sqlalchemy as sa

import selenium
import selenium.webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.remote.webelement import WebElement

from models.base import SmartSession
from models.knownexposure import KnownExposure, PipelineWorker

# TODO : write tests for hold/release

def test_conductor_not_logged_in( conductor_url ):
    res = requests.post( f"{conductor_url}/status", verify=False )
    assert res.status_code == 500
    assert res.text == "Not logged in"

def test_conductor_uninitialized( conductor_connector ):
    data = conductor_connector.send( 'status' )
    assert data['status'] == 'status'
    assert data['instrument'] is None
    assert data['timeout'] == 120
    assert data['updateargs'] is None

def test_force_update_uninitialized( conductor_connector ):
    data = conductor_connector.send( 'forceupdate' )
    assert data['status'] == 'forced update'

    data = conductor_connector.send( 'status' )
    assert data['status'] == 'status'
    assert data['instrument'] is None
    assert data['timeout'] == 120
    assert data['updateargs'] is None

def test_update_missing_args( conductor_connector ):
    with pytest.raises( RuntimeError, match=( r"Got response 500: Error return from updater: "
                                              r"Either both or neither of instrument and updateargs "
                                              r"must be None; instrument=no_such_instrument, updateargs=None" ) ):
        res = conductor_connector.send( "updateparameters/instrument=no_such_instrument" )

    with pytest.raises( RuntimeError, match=( r"Got response 500: Error return from updater: "
                                              r"Either both or neither of instrument and updateargs "
                                              r"must be None; instrument=None, updateargs={'thing': 1}" ) ):
        res = conductor_connector.send( "updateparameters", { "updateargs": { "thing": 1 } } )

def test_update_unknown_instrument( conductor_connector ):
    with pytest.raises( RuntimeError, match=( r"Got response 500: Error return from updater: "
                                              r"Failed to find instrument no_such_instrument" ) ):
        res = conductor_connector.send( "updateparameters/instrument=no_such_instrument",
                                        { "updateargs": { "thing": 1 } } )

    data = conductor_connector.send( "status" )
    assert data['status'] == 'status'
    assert data['instrument'] is None
    assert data['timeout'] == 120
    assert data['updateargs'] is None
    assert data['hold'] == 0

def test_pull_decam( conductor_connector, conductor_config_for_decam_pull ):
    req = conductor_config_for_decam_pull

    mjd0 = 60127.33819
    mjd1 = 60127.36319

    # Verify that the right things are in known exposures
    # (Do this here rather than in a test because we need
    # to clean it up after the yield.)

    with SmartSession() as session:
        kes = ( session.query( KnownExposure )
                .filter( KnownExposure.mjd >= mjd0 )
                .filter( KnownExposure.mjd <= mjd1 ) ).all()
        assert len(kes) == 18
        assert all( [ not i.hold for i in kes ] )
        assert all( [ i.project == '2023A-716082' for i in kes ] )
        assert min( [ i.mjd for i in kes ] ) == pytest.approx( 60127.33894, abs=1e-5 )
        assert max( [ i.mjd for i in kes ] ) == pytest.approx( 60127.36287, abs=1e-5 )
        assert set( [ i.exp_time for i in kes ] ) == { 60, 86, 130 }
        assert set( [ i.filter for i in kes ] ) == { 'g DECam SDSS c0001 4720.0 1520.0',
                                                     'r DECam SDSS c0002 6415.0 1480.0',
                                                     'i DECam SDSS c0003 7835.0 1470.0' }

    # Run another forced update to make sure that additional knownexposures aren't added

    data = conductor_connector.send( 'forceupdate' )
    assert data['status'] == 'forced update'
    data = conductor_connector.send( 'status' )
    first_updatetime = dateutil.parser.parse( data['lastupdate'] )

    with SmartSession() as session:
        kes = ( session.query( KnownExposure )
                .filter( KnownExposure.mjd >= mjd0 )
                .filter( KnownExposure.mjd <= mjd1 ) ).all()
        assert len(kes) == 18


    # Make sure that if *some* of what is found is already in known_exposures, only the others are added

        delkes = ( session.query( KnownExposure )
                   .filter( KnownExposure.mjd > 60127.338 )
                   .filter( KnownExposure.mjd < 60127.348 ) ).all()
        for delke in delkes:
            session.delete( delke )
        session.commit()

        kes = ( session.query( KnownExposure )
                .filter( KnownExposure.mjd >= mjd0 )
                .filter( KnownExposure.mjd <= mjd1 ) ).all()
        assert len(kes) == 11

    time.sleep(1)  # So we can resolve the time difference
    data = conductor_connector.send( 'forceupdate' )
    assert data['status'] == 'forced update'
    data = conductor_connector.send( 'status' )
    assert dateutil.parser.parse( data['lastupdate'] ) > first_updatetime

    with SmartSession() as session:
        kes = ( session.query( KnownExposure )
                .filter( KnownExposure.mjd >= mjd0 )
                .filter( KnownExposure.mjd <= mjd1 ) ).all()
        assert len(kes) == 18

    # Make sure holding by default works

    data = conductor_connector.send( "updateparameters/hold=true" )
    assert data['status'] == 'updated'
    assert data['instrument'] == 'DECam'
    assert data['hold'] == 1

    with SmartSession() as session:
        delkes = ( session.query( KnownExposure )
                   .filter( KnownExposure.mjd > mjd0 )
                   .filter( KnownExposure.mjd < mjd1 ) ).all()
        for delke in delkes:
            session.delete( delke )
        session.commit()

    data = conductor_connector.send( 'forceupdate' )
    assert data['status'] == 'forced update'

    with SmartSession() as session:
        kes = ( session.query( KnownExposure )
                .filter( KnownExposure.mjd >= mjd0 )
                .filter( KnownExposure.mjd <= mjd1 ) ).all()
        assert len(kes) == 18
        assert all( [ i.hold for i in kes ] )


def test_request_knownexposure_get_none( conductor_connector ):
    with pytest.raises( RuntimeError, match=( r"Got response 500: cluster_id is required for RequestExposure" ) ):
        res = conductor_connector.send( "requestexposure" )

    data = conductor_connector.send( 'requestexposure/cluster_id=test_cluster' )
    assert data['status'] == 'not available'


def test_request_knownexposure( conductor_connector, conductor_config_for_decam_pull ):
    previous = set()
    for i in range(3):
        data = conductor_connector.send( 'requestexposure/cluster_id=test_cluster' )
        assert data['status'] == 'available'
        assert data['knownexposure_id'] not in previous
        previous.add( data['knownexposure_id'] )

        with SmartSession() as session:
            kes = session.query( KnownExposure ).filter( KnownExposure._id==data['knownexposure_id'] ).all()
            assert len(kes) == 1
            assert kes[0].cluster_id == 'test_cluster'

    # Make sure that we don't get held exposures

    with SmartSession() as session:
        session.execute( sa.text( 'UPDATE knownexposures SET hold=true' ) )
        session.commit()

    data = conductor_connector.send( 'requestexposure/cluster_id=test_cluster' )
    assert data['status'] == 'not available'


def test_register_worker( conductor_connector ):
    """Tests registerworker, unregisterworker, and heartbeat """
    try:
        data = conductor_connector.send( 'registerworker/cluster_id=test/node_id=testnode/nexps=10' )
        assert data['status'] == 'added'
        assert data['cluster_id'] == 'test'
        assert data['node_id'] == 'testnode'
        assert data['nexps'] == 10

        with SmartSession() as session:
            pw = session.query( PipelineWorker ).filter( PipelineWorker._id==data['id'] ).first()
            assert pw.cluster_id == 'test'
            assert pw.node_id == 'testnode'
            assert pw.nexps == 10
            firstheartbeat = pw.lastheartbeat

        hb = conductor_connector.send( f'workerheartbeat/{data["id"]}' )
        assert hb['status'] == 'updated'

        with SmartSession() as session:
            pw = session.query( PipelineWorker ).filter( PipelineWorker._id==data['id'] ).first()
            assert pw.cluster_id == 'test'
            assert pw.node_id == 'testnode'
            assert pw.nexps == 10
            assert pw.lastheartbeat > firstheartbeat

        done = conductor_connector.send( f'unregisterworker/{data["id"]}' )
        assert done['status'] == 'worker deleted'

        with SmartSession() as session:
            pw = session.query( PipelineWorker ).filter( PipelineWorker._id==data['id'] ).all()
            assert len(pw) == 0

    finally:
        with SmartSession() as session:
            pws = session.query( PipelineWorker ).filter( PipelineWorker.cluster_id=='test' ).all()
            for pw in pws:
                session.delete( pw )
            session.commit()



# ======================================================================
# The tests below use selenium to test the interactive part of the
# conductor web ap

def test_main_page( browser, conductor_url ):
    browser.get( conductor_url )
    WebDriverWait( browser, timeout=10 ).until( lambda d: d.find_element(By.ID, 'login_username' ) )
    el = browser.find_element( By.TAG_NAME, 'h1' )
    assert el.text == "SeeChange Conductor"
    authdiv = browser.find_element( By.ID, 'authdiv' )
    el = browser.find_element( By.CLASS_NAME, 'link' )
    assert el.text == 'Request Password Reset'

def test_log_in( conductor_browser_logged_in ):
    browser = conductor_browser_logged_in
    # The fixture effectively has all the necessary tests.
    # Perhaps rename this test to something else and
    # do something else with it.

# TODO : more UI tests
