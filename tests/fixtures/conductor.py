import pytest
import requests
import re

import selenium.webdriver
from selenium.webdriver.common.by import By

from models.user import AuthUser
from models.base import SmartSession

@pytest.fixture
def conductor_url():
    return "http://conductor:8082"

@pytest.fixture
def conductor_logged_in():
    pass

@pytest.fixture
def conductor_user():
    with SmartSession() as session:
        user = AuthUser( id='fdc718c3-2880-4dc5-b4af-59c19757b62d',
                         username='test',
                         displayname='Test User',
                         email='testuser@mailhog'
                        )
        user.pubkey = '''-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEArBn0QI7Z2utOz9VFCoAL
+lWSeuxOprDba7O/7EBxbPev/MsayA+MB+ILGo2UycGHs9TPBWihC9ACWPLG0tJt
q5FrqWaHPmvXMT5rb7ktsAfpZSZEWdrPfLCvBdrFROUwMvIaw580mNVm4PPb5diG
pM2b8ZtAr5gHWlBH4gcni/+Jv1ZKYh0b3sUOru9+IStvFs6ijySbHFz1e/ejP0kC
LQavMj1avBGfaEil/+NyJb0Ufdy8+IdgGJMCwFIZ15HPiIUFDRYWPsilX8ik+oYU
QBZlFpESizjEzwlHtdnLrlisQR++4dNtaILPqefw7BYMRDaf1ggYiy5dl0+ZpxYO
puvcLQlPqt8iO1v3IEuPCdMqhmmyNno0AQZq+Fyc21xRFdwXvFReuOXcgvZgZupI
XtYQTStR9t7+HL5G/3yIa1utb3KRQbFkOXRXHyppUEIr8suK++pUORrAablj/Smj
9TCCe8w5eESmQ+7E/h6M84nh3t8kSBibOlcLaNywKm3BEedQXmtu4KzLQbibZf8h
Ll/jFHv5FKYjMBbVw3ouvMZmMU+aEcaSdB5GzQWhpHtGmp+fF0bPztgTdQZrArja
Y94liagnjIra+NgHOzdRd09sN9QGZSHDanANm24lZHVWvTdMU+OTAFckY560IImB
nRVct/brmHSH0KXam2bLZFECAwEAAQ==
-----END PUBLIC KEY-----
'''
        user.privkey = {"iv": "pXz7x5YA79o+Qg4w",
                        "salt": "aBtXrLT7ds9an38nW7EgbQ==",
                        "privkey": "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"
                        }
        session.add( user )
        session.commit()

    yield 'test', 'test_password'

    with SmartSession() as session:
        user = session.query( AuthUser ).filter( username='test' ).all()
        for u in user:
            u.delete()
        session.commit()
        

@pytest.fixture
def browser():
    opts = selenium.webdriver.FirefoxOptions()
    opts.add_argument( "--headless" )
    ff = selenium.webdriver.Firefox( options=opts )
    # This next line lets us use self-signed certs on test servers
    ff.accept_untrusted_certs = True
    yield ff
    ff.close()
    ff.quit()

@pytest.fixture
def conductor_browser_logged_in( conductor_url, conductor_user, browser ):
    username, password = conductor_user
    browser.get( conductor_url )
    el = WebDriverWait( browser, timeout=5 ).until( lambda d: d.find_element(By.TAG_NAME, 'h1' ) )
    assert el.text == "SeeChange Conductor"
    input_user = browser.find_element( By.ID, 'login_username' )
    input_user.setAttribute( "value", username )
    input_password = browser.find_element( By.ID, 'login_password' )
    input_password.setAttribute( "vaue", password )
    buttons = browser.find_element( By.TAG_NAME, 'button' )
    button = None
    for possible_button in buttons:
        if possible_button.getAttribute( "innerHTML" ) == "Log In":
            button = possible_button
            break
    assert button is not None
    button.click()

    def check_logged_in( d ):
        authdiv = browser.find_element( By.ID, 'authdiv' )
        if authdiv is not None:
            p = authdiv.find_element( By.TAG_NAME, 'p' )
            if p is not None:
                if re.search( '^Logged in as test (Test User)', p.getAttribute( "innerHTML" ) ):
                    return True
        return False
        
    WebDriverWait( browser, timeout=5 ).until( check_logged_in )

    yield True

    logout = authdiv.find_element( By.TAG_NAME, 'span' )
    assert logout.getAttribute( "innerHTML" ) == "Log Out"
    logout.click()
        
    
@pytest.fixture
def conductor_noirlab_listening( conductor_url ):
    res = requests.post( f"{conductor_url}/update" )
