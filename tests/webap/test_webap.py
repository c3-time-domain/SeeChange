import pytest

import selenium
import selenium.webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.remote.webelement import WebElement

def test_webap( browser, webap_url, decam_datastore ):
    import pdb; pdb.set_trace()

    browser.get( webap_url )
    WebDriverWait( browser, timeout=10 ).until(
        lambda d: d.find_element(By.ID, 'seechange_context_render_page_complete' ) )
    
    
    pass

    
