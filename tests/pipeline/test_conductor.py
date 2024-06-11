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

def test_conductor_not_logged_in( conductor_url ):
    res = requests.post( f"{conductor_url}/status", json={ "command": "status" }, verify=False )
    assert res.status_code == 500
    assert res.text == "Not logged in"
    res = requests.post( f"{conductor_url}/status", json={ "command": "status" }, verify=False )

def test_conductor_uninitialized( conductor_url, conductor_logged_in ):
    req = conductor_logged_in
    res = req.post( f"{conductor_url}/status", verify=False )
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'status'
    assert data['instrument'] is None
    assert data['timeout'] == 120
    assert data['updateargs'] is None

def test_force_update_uninitialized( conductor_url, conductor_logged_in ):
    req = conductor_logged_in
    res = req.post( f"{conductor_url}/forceupdate", verify=False )
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'forced update'

    res = req.post( f"{conductor_url}/status", verify=False )
    assert res.status_code == 200
    data = res.json()
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

def test_update_missing_args( conductor_url, conductor_logged_in ):
    req = conductor_logged_in
    res = req.post( f"{conductor_url}/updateparameters/instrument=no_such_instrument", verify=False )
    assert res.status_code == 500
    assert res.text == ( 'Error return from updater: Either both or neither of instrument and updateargs '
                         'must be None; instrument=no_such_instrument, updateargs=None' )
    res = req.post( f"{conductor_url}/updateparameters", verify=False, json={ "updateargs": { "thing": 1 } } )
    assert res.status_code == 500
    assert res.text == ( 'Error return from updater: Either both or neither of instrument and updateargs '
                         'must be None; instrument=None, updateargs={\'thing\': 1}' )
    pass
    
    
def test_update_unknown_instrument( conductor_url, conductor_logged_in ):
    req = conductor_logged_in
    res = req.post( f"{conductor_url}/updateparameters/instrument=no_such_instrument", verify=False,
                    json={ "updateargs": { "thing": 1 } } )
    assert res.status_code == 500
    assert res.text == "Error return from updater: Failed to find instrument no_such_instrument"
    
    res = req.post( f"{conductor_url}/status", verify=False )
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'status'
    assert data['instrument'] is None
    assert data['timeout'] == 120
    assert data['updateargs'] is None

def test_pull_decam( conductor_config_for_decam_pull ):
    import pdb; pdb.set_trace()
    pass
    
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

