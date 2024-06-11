import pytest

import datetime
import dateutil.parser
import requests

import selenium
import selenium.webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.remote.webelement import WebElement

from models.base import SmartSession
from models.knownexposure import KnownExposure

def test_conductor_not_logged_in( conductor_url ):
    res = requests.post( f"{conductor_url}/status", json={ "command": "status" }, verify=False )
    assert res.status_code == 500
    assert res.text == "Not logged in"
    res = requests.post( f"{conductor_url}/status", json={ "command": "status" }, verify=False )

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
    updatetime = dateutil.parser.parse( data['lastupdate'] )
    dt = updatetime - datetime.datetime.now( tz=datetime.timezone.utc )
    # Should be safe to make this short, because the conductor sets the update
    #   time at the *beginning* of it's call to update... and this is a null
    #   call anyway, because there's no instrument set.
    assert dt.total_seconds() < 2

def test_update_missing_args( conductor_connector ):
    with pytest.raises( RuntimeError, match=( r"Got response 500 from conductor: Error return from updater: "
                                              r"Either both or neither of instrument and updateargs "
                                              r"must be None; instrument=no_such_instrument, updateargs=None" ) ):
        res = conductor_connector.send( "updateparameters/instrument=no_such_instrument" )

    with pytest.raises( RuntimeError, match=( r"Got response 500 from conductor: Error return from updater: "
                                              r"Either both or neither of instrument and updateargs "
                                              r"must be None; instrument=None, updateargs={'thing': 1}" ) ):
        res = conductor_connector.send( "updateparameters", { "updateargs": { "thing": 1 } } )

def test_update_unknown_instrument( conductor_connector ):
    with pytest.raises( RuntimeError, match=( r"Got response 500 from conductor: Error return from updater: "
                                              r"Failed to find instrument no_such_instrument" ) ):
        res = conductor_connector.send( "updateparameters/instrument=no_such_instrument",
                                        { "updateargs": { "thing": 1 } } )

    data = conductor_connector.send( "status" )
    assert data['status'] == 'status'
    assert data['instrument'] is None
    assert data['timeout'] == 120
    assert data['updateargs'] is None

def test_pull_decam( conductor_connector, conductor_config_for_decam_pull ):
    req = conductor_config_for_decam_pull

    # Verify that the right things are in known exposures
    # (Do this here rather than in a test because we need
    # to clean it up after the yield.)

    with SmartSession() as session:
        kes = ( session.query( KnownExposure )
                .filter( KnownExposure.mjd >= 60159.157 )
                .filter( KnownExposure.mjd <= 60159.167 ) ).all()
        assert len(kes) == 9
        assert set( [ i.project for i in kes ] ) == { '2023A-921384', '2023A-716082' }
        assert min( [ i.mjd for i in kes ] ) == pytest.approx( 60159.15722, abs=1e-5 )
        assert max( [ i.mjd for i in kes ] ) == pytest.approx( 60159.16662, abs=1e-5 )
        assert set( [ i.exp_time for i in kes ] ) == { 100, 96, 50, 30 }
        assert set( [ i.filter for i in kes ] ) == { 'VR DECam c0007 6300.0 2600.0',
                                                     'g DECam SDSS c0001 4720.0 1520.0',
                                                     'r DECam SDSS c0002 6415.0 1480.0',
                                                     'i DECam SDSS c0003 7835.0 1470.0',
                                                     'z DECam SDSS c0004 9260.0 1520.0' }

    # Run another forced update to make sure that additional knownexposures aren't added

    data = conductor_connector.send( 'forceupdate' )
    assert data['status'] == 'forced update'

    with SmartSession() as session:
        kes = ( session.query( KnownExposure )
                .filter( KnownExposure.mjd >= 60159.157 )
                .filter( KnownExposure.mjd <= 60159.167 ) ).all()
        assert len(kes) == 9


    # Make sure that if *some* of what is found is already in known_exposures, only the others are added

        delkes = ( session.query( KnownExposure )
                   .filter( KnownExposure.mjd > 60159.160 )
                   .filter( KnownExposure.mjd < 60159.166 ) ).all()
        for delke in delkes:
            session.delete( delke )
        session.commit()

        kes = ( session.query( KnownExposure )
                .filter( KnownExposure.mjd >= 60159.157 )
                .filter( KnownExposure.mjd <= 60159.167 ) ).all()
        assert len(kes) == 3

    data = conductor_connector.send( 'forceupdate' )
    assert data['status'] == 'forced update'

    with SmartSession() as session:
        kes = ( session.query( KnownExposure )
                .filter( KnownExposure.mjd >= 60159.157 )
                .filter( KnownExposure.mjd <= 60159.167 ) ).all()
        assert len(kes) == 9

def test_request_knownexposure_get_none( conductor_connector ):
    with pytest.raises( RuntimeError, match=( r"Got response 500 from conductor: "
                                              r"cluster_id is required for RequestExposure" ) ):
        res = conductor_connector.send( "requestexposure" )

    data = conductor_connector.send( 'requestexposure/cluster_id=test_cluster' )
    assert data['status'] == 'not available'


def test_request_knownexposure( conductor_connector, conductor_config_for_decam_pull ):
    data = conductor_connector.send( 'requestexposure/cluster_id=test_cluster' )
    assert data['status'] == 'available'

    with SmartSession() as session:
        kes = session.query( KnownExposure ).filter( KnownExposure.id==data['knownexposure_id'] ).all()
        assert len(kes) == 1
        assert kes[0].cluster_id == 'test_cluster'


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

