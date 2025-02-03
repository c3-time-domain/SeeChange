import re
import time
import pytest
import requests
# Disable warnings from urllib, since there will be lots about insecure connections
#  given that we're using a self-signed cert for the server in the test environment
requests.packages.urllib3.disable_warnings()

import sqlalchemy as sa

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support.ui import Select

from models.base import SmartSession, Psycopg2Connection
from models.provenance import Provenance, ProvenanceTag

from util.logger import SCLogger


def test_webap_not_logged_in( webap_url ):
    res = requests.post( f"{webap_url}/provtags", verify=False )
    assert res.status_code == 500
    assert res.text == "Not logged in"


def test_webap_admin_required( webap_rkauth_client ):
    # SUGGESTION : whenever adding *any* endpoint to
    #   the webap that requires admin, hit that endpoint
    #   in this test, just to be sure.
    client = webap_rkauth_client
    res = client.post( "provtags" )
    assert res.status_code == 200
    with pytest.raises( RuntimeError, match="Got response 500: Action requires admin" ):
        res = client.post( "cloneprovtag/foo/bar" )


def test_webap_provtags( webap_rkauth_client, provenance_base, provenance_extra, provenance_tags_loaded ):
    res = webap_rkauth_client.send( "provtags" )
    assert isinstance( res, dict )
    assert ( 'status' in res ) and ( res['status'] == 'ok' )
    assert ( 'provenance_tags' in res ) and ( { 'xyzzy', 'plugh' } <= set( res['provenance_tags'] ) )


def test_webap_provtaginfo( webap_rkauth_client, provenance_base, provenance_extra, provenance_tags_loaded ):
    res = webap_rkauth_client.send( "/provtaginfo/plugh" )
    assert isinstance( res, dict )
    assert res['status'] == 'ok'
    assert res['tag'] == 'plugh'
    assert len( res['_id'] ) == 2
    for key in [ '_id', 'process', 'code_version_id', 'bad_comment' ]:
        assert set( res[key] ) == { getattr( provenance_base, key ), getattr( provenance_extra, key ) }
    for key in [ 'is_outdated', 'replaced_by', 'is_testing' ]:
        assert set( bool(r) for r in res[key] ) == { bool( getattr( provenance_base, key ) ),
                                                     bool( getattr( provenance_extra, key ) ) }

    res = webap_rkauth_client.send( "/provtaginfo/this_prov_tag_does_not_exist" )
    assert isinstance( res, dict )
    assert res['status'] == 'ok'
    assert res['tag'] == 'this_prov_tag_does_not_exist'
    assert len( res['_id'] ) == 0


def test_webap_provinfo( webap_rkauth_client, provenance_base, provenance_extra, code_version ):
    res = webap_rkauth_client.send( f"/provenanceinfo/{provenance_base.id}" )
    assert isinstance( res, dict )
    assert res['status'] == 'ok'
    assert res['_id'] == provenance_base.id
    assert res['code_version_id'] == code_version.id
    assert res['process'] == 'test_base_process'
    assert res['parameters'] == provenance_base.parameters
    assert res['is_testing']
    assert not res['is_bad']
    assert not res['is_outdated']
    assert len( res['upstreams']['_id'] ) == 0

    res = webap_rkauth_client.send( f"/provenanceinfo/{provenance_extra.id}" )
    assert len( res['upstreams']['_id'] ) == 1
    assert res['upstreams']['_id'][0] == provenance_base.id
    assert res['upstreams']['process'][0] == 'test_base_process'


def test_webap_clone_provtag( webap_admin_client, provenance_base, provenance_extra, provenance_tags_loaded ):
    try:
        # Make sure we can clone to a non-existent current
        res = webap_admin_client.send( '/cloneprovtag/xyzzy/current' )
        assert 'status' in res and res['status'] == 'ok'

        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "SELECT tag, provenance_id FROM provenance_tags "
                            "WHERE tag IN ('xyzzy', 'current')" )
            rows = cursor.fetchall()
            assert set( r[0] for r in rows ) == { 'xyzzy', 'current' }
            assert all( r[1] == provenance_base.id for r in rows )

        # Make sure that we can't clone to an existing provenance if we don't ask to
        with pytest.raises( RuntimeError, match="Got response 500: Tag current already exists and clobber was False" ):
            res = webap_admin_client.send( '/cloneprovtag/plugh/current' )

        # Make sure we can clone an existing provenance if we ask to
        res = webap_admin_client.send( '/cloneprovtag/plugh/current/1' )
        assert 'status' in res and res['status'] == 'ok'

        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "SELECT tag, provenance_id FROM provenance_tags "
                            "WHERE tag IN ( 'xyzzy', 'plugh', 'current' )" )
            rows = cursor.fetchall()
            foundtags = {}
            for row in rows:
                if row[0] in foundtags:
                    foundtags[row[0]].add( row[1] )
                else:
                    foundtags[row[0]] = { row[1] }
            assert set( foundtags.keys() ) ==  { 'xyzzy', 'plugh', 'current' }
            assert len( foundtags['xyzzy'] ) == 1
            assert len( foundtags['plugh'] ) == 2
            assert foundtags['plugh'] == foundtags['current']

    finally:
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "DELETE FROM provenance_tags WHERE tag='current'" )
            conn.commit()



# This test fails because there are other session-scope (I think) fixtures that create projects
#  in the database
@pytest.mark.skip( reason="comment out this skip when running test_webap.py by itself to run this test" )
def test_webap_empty_projects( webap_rkauth_client ):
    res = webap_rkauth_client.send( "/projects" )
    assert isinstance( res, dict )
    assert res['status'] == 'ok'
    assert res['projects'] == []


def test_webap_projects( webap_rkauth_client, sim_exposure1 ):
    res = webap_rkauth_client.send( "/projects" )
    assert isinstance( res, dict )
    assert res['status'] == 'ok'
    assert 'foo' in res['projects']


# TODO : test /exposures, /exposureimages, /pngcutoutsforsubimage
# (Maybe not urgent, because these are tested via browser in test_webap below.)


# This test is brobdingnagian because I just want to run the
#    fixture once.  It will troll through the web interface a lot.
def test_webap( webap_browser_logged_in, webap_url, decam_datastore ):
    browser = webap_browser_logged_in
    ds = decam_datastore
    junkprov = None

    try:

        # ======================================================================
        # Some setup

        # Create a new provenance tag, tagging the provenances that are in decam_datastore
        provs = Provenance.get_batch( [ ds.exposure.provenance_id,
                                        ds.image.provenance_id,
                                        ds.sources.provenance_id,
                                        ds.bg.provenance_id,
                                        ds.wcs.provenance_id,
                                        ds.zp.provenance_id,
                                        ds.reference.provenance_id,
                                        ds.sub_image.provenance_id,
                                        ds.detections.provenance_id,
                                        ds.cutouts.provenance_id,
                                        ds.measurements[0].provenance_id,
                                        ds.scores[0].provenance_id,
                                        ds.report.provenance_id ] )
        ProvenanceTag.addtag( 'test_webap', provs )

        # Create a throwaway provenance and provenance tag so we can test
        #  things *not* being found
        cv = Provenance.get_code_version()
        junkprov = Provenance( process='no_process', code_version_id=cv.id, is_testing=True )
        junkprov.insert()
        ProvenanceTag.addtag( 'no_such_tag', [ junkprov ] )


        # ======================================================================
        # ======================================================================
        # ======================================================================
        # Test selecting exposures

        browser.get( webap_url )
        WebDriverWait( browser, timeout=10 ).until(
            lambda d: d.find_element(By.ID, 'seechange_context_render_page_complete' ) )
        mainbuttonbox = browser.find_element( By.XPATH, "//div[@id='frontpagetabs']/div[1]" )
        maincontentbox = browser.find_element( By.XPATH, "//div[@id='frontpagetabs']/div[2]" )
        assert 'buttonboxdiv' in mainbuttonbox.get_attribute('class')
        assert 'tabcontentdiv' in maincontentbox.get_attribute('class')

        # Make sure the "Exposure Search" tag is selected
        but = mainbuttonbox.find_element( By.XPATH, "./button[1]" )
        assert but.text == 'Exposure Search'
        assert 'tabsel' in but.get_attribute('class')
        div = maincontentbox.find_element( By.XPATH, "./div" )
        assert div.get_attribute('id') == 'exposuresearchdiv'

        # exposurediv is the div that holds the tabbed results of the exposure search
        exposurediv = div.find_element( By.XPATH, ".//div[@id='exposuresearchsubdiv']" )

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

        # ======================================================================
        # ======================================================================
        # Get nothing selecting a tag with no exposures

        select = Select( provtag_wid )
        select.select_by_visible_text( 'no_such_tag' )
        buttons['Show Exposures'].click()

        WebDriverWait( exposurediv, timeout=10 ).until(
            lambda d: d.find_element( By.XPATH, ".//h2[contains(.,'Exposures from')]" ) )

        # Make sure that the "Exposure List" div is what's shown
        tabcontentdiv = exposurediv.find_element( By.XPATH, ".//div[contains(@class, 'tabcontentdiv')]" )
        exposurelistdiv = tabcontentdiv.find_element( By.TAG_NAME, "div" )
        assert exposurelistdiv.get_attribute( 'id' ) == 'exposurelistlistdiv'
        assert exposurelistdiv.find_element( By.XPATH, "./h2" ).text[0:15] == "Exposures from "
        explist = exposurelistdiv.find_element( By.ID, "exposure_list_table" )
        rows = explist.find_elements( By.TAG_NAME, "tr" )
        assert len(rows) == 1   # Just the header row

        # ======================================================================
        # ======================================================================
        # List of exposures tagged with test_webap

        select.select_by_visible_text( 'test_webap' )
        buttons['Show Exposures'].click()
        # Give it half a second to go at least get to the "loading" screen; that's
        #  all javascript with no server communcation, so should be fast.
        time.sleep( 0.5 )
        WebDriverWait( exposurediv, timeout=10 ).until(
            lambda d: d.find_element( By.XPATH, "//h2[contains(.,'Exposures from')]" ) )

        tabcontentdiv = exposurediv.find_element( By.XPATH, ".//div[contains(@class, 'tabcontentdiv')]" )
        exposurelistdiv = tabcontentdiv.find_element( By.TAG_NAME, "div" )
        assert exposurelistdiv.get_attribute( 'id' ) == 'exposurelistlistdiv'
        assert exposurelistdiv.find_element( By.XPATH, "./h2" ).text[0:15] == "Exposures from "
        explist = exposurelistdiv.find_element( By.ID, "exposure_list_table" )
        rows = explist.find_elements( By.TAG_NAME, "tr" )
        assert len(rows) == 2

        cols = rows[1].find_elements( By.XPATH, "./*" )
        assert cols[0].text == 'c4d_211025_044847_ori.fits.fz'
        assert cols[2].text == 'ELAIS-E1'
        assert cols[5].text == '1'    # n_images
        assert cols[6].text == '266'  # detections
        assert cols[7].text == '9'    # sources

        # ======================================================================
        # ======================================================================
        # Exposure details

        # Try to click on the exposure name, make sure we get the exposure details
        expnamelink = cols[0].find_element( By.TAG_NAME, 'a' )
        expnamelink.click()
        WebDriverWait( exposurediv, timeout=10 ).until(
            lambda d: d.find_element( By.XPATH, "//h2[contains(.,'Exposure c4d_211025_044847_ori.fits.fz')]" ) )

        buttonbox = exposurediv.find_element( By.XPATH, ".//div[contains(@class, 'buttonboxdiv')]" )
        assert 'tabsel' in ( buttonbox
                             .find_element( By.XPATH, "./button[text()='Exposure Details']" )
                             .get_attribute('class') )

        tabcontentdiv = exposurediv.find_element( By.XPATH, ".//div[contains(@class, 'tabcontentdiv')]" )
        expdetailsdiv = tabcontentdiv.find_element( By.XPATH, "./div" )
        assert expdetailsdiv.get_attribute('id') == 'exposurelistexposurediv'

        subbuttonbox = expdetailsdiv.find_element( By.XPATH, ".//div[contains(@class, 'buttonboxdiv')]" )
        assert 'tabsel' in ( subbuttonbox
                             .find_element( By.XPATH, "./button[text()='Images']" )
                             .get_attribute('class') )
        subcontentdiv = expdetailsdiv.find_element( By.XPATH, ".//div[contains(@class, 'tabcontentdiv')]" )

        # ======================================================================
        # Images

        imagesdiv = subcontentdiv.find_element( By.XPATH, "./div" )
        assert imagesdiv.get_attribute('id') == 'exposureimagesdiv'
        assert re.search( r"^Exposure has 1 images and 1 completed subtractions.*"
                          r"\s9 out of 266 detections pass preliminary cuts",
                          imagesdiv.text, re.DOTALL ) is not None
        imagestab = imagesdiv.find_element( By.TAG_NAME, 'table' )
        rows = imagestab.find_elements( By.TAG_NAME, 'tr' )
        assert len(rows) == 2
        cols = rows[1].find_elements( By.XPATH, "./*" )
        assert re.search( r'^c4d_20211025_044847_S2_r_Sci', cols[1].text ) is not None

        # ======================================================================
        # Sources

        # Find the sources tab and click on that
        subbuttonbox.find_element( By.XPATH, "./button[text()='Sources']" ).click()
        # Give it half a second to go at least get to the "loading" screen; that's
        #  all javascript with no server communcation, so should be fast.
        time.sleep( 0.5 )
        WebDriverWait( subcontentdiv, timeout=10 ).until(
            lambda d: d.find_element( By.XPATH, ".//p[contains(.,'Sources for all successfully completed chips')]" ) )

        # Now the tab content div should have information about the sources
        sourcesdiv = subcontentdiv.find_element( By.XPATH, "./div" )
        assert sourcesdiv.get_attribute('id') == "exposurecutoutsdiv"
        sourcestable = sourcesdiv.find_element( By.TAG_NAME, 'table' )
        rows = sourcestable.find_elements( By.TAG_NAME, 'tr' )
        assert len(rows) == 10

        # OMG writing these tests is exhausting.  There is still lots more to do:
        # * actually look at the rows of the sources table
        # * go back to exposure list
        # * check whether you're searching for a single image vs. whole exposure sources
        # * provenance tags at top of page
        # * other things

    finally:
        # Clean up the junk Provenance, and the ProvenanceTags we created
        with SmartSession() as session:
            session.execute( sa.text( "DELETE FROM provenance_tags "
                                      "WHERE tag IN ('test_webap', 'no_such_tag')" ) )
            if junkprov is not None:
                session.execute( sa.text( "DELETE FROM provenances WHERE _id=:id" ), { 'id': junkprov.id } )
            session.commit()
