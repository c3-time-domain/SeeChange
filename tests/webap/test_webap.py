import re
import time
import pytest

import sqlalchemy as sa

import selenium
import selenium.webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.remote.webelement import WebElement

from models.base import SmartSession
from models.provenance import CodeVersion, Provenance, ProvenanceTag

from util.logger import SCLogger

def test_webap( browser, webap_url, decam_datastore ):
    ds = decam_datastore
    junkprov = None

    try:
        # Create a new provenance tag, tagging the provenances that are in decam_datastore
        ProvenanceTag.newtag( 'test_webap',
                              [ ds.exposure.provenance_id,
                                ds.image.provenance_id,
                                ds.sources.provenance_id,
                                ds.reference.provenance_id,
                                ds.sub_image.provenance_id,
                                ds.detections.provenance_id,
                                ds.cutouts.provenance_id,
                                ds.measurements[0].provenance_id ] )

        # Create a throwaway provenance and provenance tag so we can test
        #  things *not* being found
        cv = Provenance.get_code_version()
        junkprov = Provenance( process='no_process', code_version_id=cv.id, is_testing=True )
        junkprov.insert()
        ProvenanceTag.newtag( 'no_such_tag', [ junkprov ] )

        browser.get( webap_url )
        WebDriverWait( browser, timeout=10 ).until(
            lambda d: d.find_element(By.ID, 'seechange_context_render_page_complete' ) )

        # The "test_webap" option in the provtag_wid select widget won't necessarily
        # be there immediately, because it's filled in with a callback from a web request
        tries = 5
        while ( tries > 0 ):
            provtag_wid = browser.find_element( By.ID, "provtag_wid" )
            options = provtag_wid.find_elements( By.TAG_NAME, "option" )
            if any( [ o.text == 'test_webap' for o in options ] ):
                break
            tries -= 1
            if tries < 0:
                assert False, "Failed to find the test_webap option in the provenances select widget"
            else:
                SCLogger.debug( "Didn't find test_webap in the provtag_wid select, sleeping 1s and retrying" )
                time.sleep( 1 )

        buttons = browser.find_elements( By.XPATH, "//input[@type='button']" )
        buttons = { b.get_attribute("value") : b for b in buttons }

        # Make sure we get no exposures if we ask for the junk tag

        select = Select( provtag_wid )
        select.select_by_visible_text( 'no_such_tag' )
        buttons['Show Exposures'].click()

        WebDriverWait( browser, timeout=10 ).until(
            lambda d: d.find_element( By.XPATH, "//h2[contains(.,'Exposures from')]" ) )

        # Make sure that the "Exposure List" div is what's shown
        # WARNING -- this absolute xpath might change if the page layout is changed!
        tabcontentdiv = browser.find_element( By.XPATH, "html/body/div/div/div/div/div/div/div[2]" )
        assert tabcontentdiv.text[:15] == 'Exposures from '
        explist = browser.find_element( By.ID, "exposure_list_table" )
        rows = explist.find_elements( By.TAG_NAME, "tr" )
        assert len(rows) == 1   # Just the header row

        # Now ask for the test_webap tag, see if we get the one exposure we expect

        select.select_by_visible_text( 'test_webap' )
        buttons['Show Exposures'].click()
        # Give it half a second to go at least get to the "loading" screen; that's
        #  all javascript with no server communcation, so should be fast.
        time.sleep( 0.5 )
        WebDriverWait( browser, timeout=10 ).until(
            lambda d: d.find_element( By.XPATH, "//h2[contains(.,'Exposures from')]" ) )

        tabcontentdiv = browser.find_element( By.XPATH, "html/body/div/div/div/div/div/div/div[2]" )
        assert tabcontentdiv.text[:15] == 'Exposures from '
        explist = browser.find_element( By.ID, "exposure_list_table" )
        rows = explist.find_elements( By.TAG_NAME, "tr" )
        assert len(rows) == 2   # Just the header row

        cols = rows[1].find_elements( By.XPATH, "./*" )
        assert cols[0].text == 'c4d_230702_080904_ori.fits.fz'
        assert cols[2].text == 'ELAIS-E1'
        assert cols[5].text == '1'    # n_images
        assert cols[6].text == '187'  # detections
        assert cols[7].text == '8'    # sources

        # Try to click on the exposure name, make sure we get the exposure details
        expnamelink = cols[0].find_element( By.TAG_NAME, 'a' )
        expnamelink.click()
        WebDriverWait( browser, timeout=10 ).until(
            lambda d: d.find_element( By.XPATH, "//h2[contains(.,'Exposure c4d_230702_080904_ori.fits.fz')]" ) )

        # OMG I nest a lot of divs
        tabcontentdiv = browser.find_element( By.XPATH, "html/body/div/div/div/div/div/div/div[2]" )
        imagesdiv = tabcontentdiv.find_element( By.XPATH, "./div/div/div/div[2]/div" )
        assert re.search( r"^Exposure has 1 images and 1 completed subtractions.*"
                          r"8 out of 187 detections pass preliminary cuts",
                          imagesdiv.text, re.DOTALL ) is not None


        imagestab = imagesdiv.find_element( By.TAG_NAME, 'table' )
        rows = imagestab.find_elements( By.TAG_NAME, 'tr' )
        assert len(rows) == 2
        cols = rows[1].find_elements( By.XPATH, "./*" )
        assert re.search( r'^c4d_20230702_080904_S3_r_Sci', cols[1].text ) is not None

        # Find the sources tab and click on that
        tabbuttonsdiv = tabcontentdiv.find_element( By.XPATH, "./div/div/div/div[1]" )
        sourcestab = tabbuttonsdiv.find_element( By.XPATH, "//.[.='Sources']" )
        sourcestab.click()
        # Give it half a second to go at least get to the "loading" screen; that's
        #  all javascript with no server communcation, so should be fast.
        time.sleep( 0.5 )
        WebDriverWait( browser, timeout=10 ).until(
            lambda d: d.find_element( By.XPATH, "//p[contains(.,'Sources for all successfully completed chips')]" ) )

        # Now imagesdiv should have information about the sources
        tabcontentdiv = browser.find_element( By.XPATH, "html/body/div/div/div/div/div/div/div[2]" )
        imagesdiv = tabcontentdiv.find_element( By.XPATH, "./div/div/div/div[2]/div" )

        sourcestab = imagesdiv.find_element( By.TAG_NAME, 'table' )
        rows = sourcestab.find_elements( By.TAG_NAME, 'tr' )
        assert len(rows) == 9
        # check stuff about the rows?

        # There is probably more we should be testing here.  Definitely.

    finally:
        # Clean up the junk Provenance, and the ProvenanceTags we created
        with SmartSession() as session:
            session.execute( sa.text( "DELETE FROM provenance_tags "
                                      "WHERE tag IN ('test_webap', 'no_such_tag')" ) )
            if junkprov is not None:
                session.execute( sa.text( "DELETE FROM provenances WHERE _id=:id" ), { 'id': junkprov.id } )
            session.commit()


