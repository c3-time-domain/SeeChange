import pytest
import requests
# Disable warnings from urllib, since there will be lots about insecure connections
#  given that we're using a self-signed cert for the server in the test environment
requests.packages.urllib3.disable_warnings()
import re

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

from models.base import SmartSession
from models.knownexposure import KnownExposure

from util.conductor_connector import ConductorConnector
from util.config import Config


@pytest.fixture
def conductor_url():
    return Config.get().value( 'conductor.conductor_url' )


@pytest.fixture
def conductor_user( admin_user ):
    return admin_user


@pytest.fixture
def conductor_browser_logged_in( browser, conductor_user ):
    cfg = Config.get()
    conductor_url = cfg.value( 'conductor.conductor_url' )
    username = cfg.value( 'conductor.username' )
    password = cfg.value( 'conductor.password' )
    browser.get( conductor_url )
    # Possible race conditions here that would not happen with a real
    #  user with human reflexes....  The javascript inserts the
    #  login_username element before the login_password element, and
    #  while I'm not fully sure how selenium works, it's possible that
    #  if I waited on login_username and read it, I'd get to finding the
    #  login_password element before the javascript had actually
    #  inserted it.  This is why I wait on password, because I know
    #  it's inserted second.
    input_password = WebDriverWait( browser, timeout=5 ).until( lambda d: d.find_element( By.ID, 'login_password' ) )
    input_password.clear()
    input_password.send_keys( password )
    input_user = browser.find_element( By.ID, 'login_username' )
    input_user.clear()
    input_user.send_keys( username )
    buttons = browser.find_elements( By.TAG_NAME, 'button' )
    button = None
    for possible_button in buttons:
        if possible_button.get_attribute( "innerHTML" ) == "Log In":
            button = possible_button
            break
    assert button is not None
    button.click()

    def check_logged_in( d ):
        authdiv = browser.find_element( By.ID, 'authdiv' )
        if authdiv is not None:
            p = authdiv.find_element( By.TAG_NAME, 'p' )
            if p is not None:
                if re.search( r'^Logged in as test \(Test User\)', p.text ):
                    return True
        return False

    WebDriverWait( browser, timeout=5 ).until( check_logged_in )

    yield browser

    authdiv = browser.find_element( By.ID, 'authdiv' )
    logout = authdiv.find_element( By.TAG_NAME, 'span' )
    assert logout.get_attribute( "innerHTML" ) == "Log Out"
    logout.click()


@pytest.fixture
def conductor_connector( conductor_user ):
    conductcon = ConductorConnector( verify=False )
    yield conductcon
    conductcon.send( 'auth/logout' )


@pytest.fixture
def conductor_connector_wrong_user( user ):
    wrongconductcon = ConductorConnector( username='test', password='test_password', verify=False )
    yield wrongconductcon
    wrongconductcon.send( 'auth/logout' )


@pytest.fixture
def conductor_config_for_decam_pull( conductor_connector, decam_raw_origin_exposures_parameters ):
    origstatus = conductor_connector.send( 'conductor/status' )
    del origstatus[ 'status' ]
    del origstatus[ 'lastupdate' ]
    del origstatus[ 'configchangetime' ]

    data = conductor_connector.send( 'conductor/updateparameters/timeout=120/instrument=DECam/pause=true',
                                     { 'updateargs': decam_raw_origin_exposures_parameters } )
    assert data['status'] == 'updated'
    assert data['instrument'] == 'DECam'
    assert data['timeout'] == 120
    assert data['updateargs'] == decam_raw_origin_exposures_parameters
    assert data['hold'] == 0
    assert data['pause'] == 1

    data = conductor_connector.send( 'conductor/forceupdate' )
    assert data['status'] == 'forced update'

    yield True

    # Reset the conductor to no instrument

    data = conductor_connector.send( 'conductor/updateparameters', origstatus )
    assert data['status'] == 'updated'
    for kw in [ 'instrument', 'timeout', 'updateargs' ]:
        assert data[kw] == origstatus[kw]

    # Clean up known exposures

    with SmartSession() as session:
        kes = ( session.query( KnownExposure )
                .filter( KnownExposure.mjd >= decam_raw_origin_exposures_parameters['minmjd'] )
                .filter( KnownExposure.mjd <= decam_raw_origin_exposures_parameters['maxmjd'] ) ).all()
        for ke in kes:
            session.delete( ke )
        session.commit()


# This next one doesn't restore state, be aware.  Treat it as
#   a variant of conductor_config_for_decam_pull; don't use
#   both fixtures in the same test.
@pytest.fixture
def conductor_config_decam_pull_all_held( conductor_connector, conductor_config_for_decam_pull ):
    data = conductor_connector.send( "conductor/getknownexposures" )
    tohold = [ ke['id'] for ke in data['knownexposures'] ]
    conductor_connector.send( "conductor/holdexposures/", { 'knownexposure_ids': tohold } )

    # Make sure they all got held
    with SmartSession() as session:
        kes = session.query( KnownExposure ).all()
    assert all( [ ke.hold for ke in kes ] )

    return True
