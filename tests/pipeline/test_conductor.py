import requests

import selenium
import selenium.webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.remote.webelement import WebElement


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
    

def test_conductor_not_logged_in( conductor_url ):
    res = requests.post( f"{conductor_url}/status", json={ "command": "status" }, verify=False )
    assert res.status_code == 500
    assert res.text == "Not logged in"


def test_conductor_uninitialized( conductor_url, conductor_logged_in ):
    req = conductor_logged_in
    res = req.post( f"{conductor_url}/status", json={ "command": "status" }, verify=False )
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'status'
    assert data['instrument_name'] is None
    assert data['timeout'] == 120
    assert data['updateargs'] is None
    
