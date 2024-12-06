import pytest
import re
import requests
# Disable warnings from urllib, since there will be lots about insecure connections
#  given that we're using a self-signed cert for the server in the test environment
requests.packages.urllib3.disable_warnings()

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

from util.config import Config
from util.rkauth_client import rkAuthClient


@pytest.fixture( scope='session' )
def webap_url():
    return Config.get().value( 'webap.webap_url' )


@pytest.fixture
def webap_rkauth_client( webap_url, user ):
    client = rkAuthClient( webap_url, 'test', 'test_password', verify=False )
    client.verify_logged_in()
    return client


@pytest.fixture
def webap_admin_client( webap_url, admin_user ):
    client = rkAuthClient( webap_url, 'admin', 'admin', verify=False )
    client.verify_logged_in()
    return client


@pytest.fixture
def webap_browser_logged_in( browser, user ):
    cfg = Config.get()
    webap_url = cfg.value( 'webap.webap_url' )
    username = cfg.value( 'conductor.username' )
    password = cfg.value( 'conductor.password' )
    browser.get( webap_url )
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
        authdiv = browser.find_element( By.ID, "authdiv" )
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
