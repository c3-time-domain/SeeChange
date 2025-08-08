import pytest
import uuid
import time
from multiprocessing import Process

import numpy as np
import sqlalchemy as sa
import psycopg2.errors

from models.base import SmartSession, Psycopg2Connection
from models.image import Image
from models.zero_point import ZeroPoint
from models.object import Object, ObjectLegacySurveyMatch
from models.measurements import Measurements, MeasurementSet
from models.deepscore import DeepScore, DeepScoreSet
from util.util import asUUID


def test_object_creation():
    ra = 7.01241
    dec = -42.96943
    obj = Object(ra=ra, dec=dec, is_test=True, is_bad=False)

    try:
        with pytest.raises( psycopg2.errors.NotNullViolation, match='null value in column "name"' ):
            obj.insert()

        obj.name = "foo"
        obj.insert()

        assert obj._id is not None

        with SmartSession() as session:
            obj2 = session.scalars(sa.select(Object).where(Object._id == obj.id)).first()
            assert obj2.ra == ra
            assert obj2.dec == dec
            assert obj2.name is not None
            assert obj2.name == obj.name

    finally:
        if obj._id is not None:
            with Psycopg2Connection() as conn:
                cursor = conn.cursor()
                cursor.execute( "DELETE FROM objects WHERE _id=%(id)s", { 'id': obj._id } )
                conn.commit()


def test_generate_names():
    ra0 = 87.5873836658953
    dec0 = 72.93697309565587
    try:
        # make sure that things in the year range 3500 aren't in the object_name_max_used table
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "SELECT * FROM object_name_max_used WHERE year>=2090 AND year<3600" )
            rows = cursor.fetchall()
            if len(rows) > 0:
                raise RuntimeError( "object_name_max_used table has rows in the range [3500,3600). "
                                    "This shouldn't be.  Make sure another test isn't doing this." )

        names = Object.generate_names( number=3, year=2090, formatstr="test_gen_name%y%a", nocommit=False )
        assert names == [ "test_gen_name90a", "test_gen_name90b", "test_gen_name90c" ]
        names = Object.generate_names( number=3, year=2091, formatstr="test_gen_name%y%a", nocommit=False )
        assert names == [ "test_gen_name91a", "test_gen_name91b", "test_gen_name91c" ]
        names = Object.generate_names( number=3, year=2091, formatstr="test_gen_name%Y%a", nocommit=False )
        assert names == [ "test_gen_name2091d", "test_gen_name2091e", "test_gen_name2091f" ]
        names = Object.generate_names( number=3, year=2091, formatstr="test_gen_name%Y%a", nocommit=False )
        assert names == [ "test_gen_name2091g", "test_gen_name2091h", "test_gen_name2091i" ]
        names = Object.generate_names( number=3, year=2091, formatstr="test_gen_name%Y_%n", nocommit=False )
        assert names == [ "test_gen_name2091_9", "test_gen_name2091_10", "test_gen_name2091_11" ]
        names = Object.generate_names( number=1, year=3502, formatstr="test_gen_name%Y%A", nocommit=False )
        assert names == [ "test_gen_name3502A" ]

        names = Object.generate_names( number=26**3, year=3503, formatstr="test_gen_name%Y%a", nocommit=False )
        assert names[0] == "test_gen_name3503a"
        assert names[25] == "test_gen_name3503z"
        assert names[1*26 + 0] == "test_gen_name3503ba"
        assert names[1*26 + 25] == "test_gen_name3503bz"
        assert names[2*26 + 0] == "test_gen_name3503ca"
        assert names[25*26 + 25] == "test_gen_name3503zz"
        assert names[1*(26**2) + 0*26 + 0] == "test_gen_name3503baa"
        assert names[25*(26**2) + 0*26 + 0] ==  "test_gen_name3503zaa"
        assert names[25*(26**2) + 25*26 + 25] == "test_gen_name3503zzz"

        names = Object.generate_names( number=1, year=3503, month=6, day=26, formatstr="test_gen_names_%Y_%m_%d" )
        assert names == [ 'test_gen_names_3503_06_26' ]

        names = Object.generate_names( number=1, ra=ra0, dec=dec0, formatstr="test_gen_names_%R_%D" )
        assert names == [ 'test_gen_names_087.5874_+72.9370' ]

        names = Object.generate_names( number=2, formatstr="test_gen_name_%l%l", seed=42 )
        assert names == [ 'test_gen_name_cu', 'test_gen_name_rl' ]

        names = Object.generate_names( number=2, formatstr="test_gen_name_%Y_%a_%%", year=3505, nocommit=False )
        assert names == [ "test_gen_name_3505_a_%", "test_gen_name_3505_b_%" ]

        # Test name collisions
        with pytest.raises( ValueError, match=( r"Newly generated names contain duplicates: "
                                                r"\['test_gen_names', 'test_gen_names'\]" ) ):
            names = Object.generate_names( number=2, formatstr="test_gen_names", verifyunique=True )

        objobj = Object( name="test_gen_names", ra=ra0, dec=dec0, is_bad=False )
        objobj.calculate_coordinates()
        objobj.insert()

        with pytest.raises( ValueError, match=( r"1 of 1 newly generated names already exist in the database: "
                                                r"\['test_gen_names'\]" ) ):
            names = Object.generate_names( number=1, formatstr="test_gen_names", verifyunique=True )

        with pytest.raises( ValueError, match="1 of 1 newly generated names already exist in the database: " ):
            for i in range(27):
                names = Object.generate_names( number=1, formatstr="test_gen_names_%Y_%l",
                                               year=3504, verifyunique=True )
                objobj = Object( name=names[0], ra=ra0+i*0.01, dec=dec0+i*0.01, is_bad=False )
                objobj.calculate_coordinates()
                objobj.insert()

        with pytest.raises( ValueError, match="Newly generated names contain duplicates: " ):
            names = Object.generate_names( number=27, formatstr="test_gen_name_%Y_%l", year=3505, verifyunique=True )

        # TODO: test more failures?

    finally:
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "DELETE FROM object_name_max_used WHERE year>=2090 AND year<3600" )
            cursor.execute( "DELETE FROM objects WHERE name LIKE 'test_gen_name%'" )
            conn.commit()


def test_generate_names_race_condition():
    # If this ra and dec is used in any other test
    #   and they leave behind an object, it will
    #   break this test, and also that object
    #   will get deleted by cleanup of this test.
    ra=154.4032444540051
    dec=-15.32714749020748
    measur = Measurements( ra=ra, dec=dec  )

    def associator():
        with Psycopg2Connection() as conn:
            Object.associate_measurements( [ measur ], year=2025, connection=conn, nocommit=True )
            # Put a sleep here in order to trigger the race condition if
            #   the table lock in associate_measurements is commented
            #   out.  (I have verified that in fact the
            #     assert newnobj == orignobj + 1
            #   below fails if the table lock is commented out;
            #   there are two new objects in that case.)
            time.sleep(1)
            conn.commit()

    try:
        # Count how many objects there are before we begin
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "SELECT COUNT(*) FROM objects" )
            orignobj = cursor.fetchone()[0]

        # Have two different processes try to generate an object at the same time for
        #   a measurement at the same ra/dec.

        p1 = Process( target=associator )
        p2 = Process( target=associator )
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        # Count how many objects there are now
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "SELECT COUNT(*) FROM objects" )
            newnobj = cursor.fetchone()[0]

        # Only one object should have been created by the two processes.
        assert newnobj == orignobj + 1

    finally:
        # Delete the objects we created
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "DELETE FROM objects WHERE q3c_radial_query(ra, dec, %(ra)s, %(dec)s, 1./3600.)",
                            { "ra": ra, "dec": dec } )
            conn.commit()


# This next test is a little weird, because the actual stuff
#   that does the work is all in fixtures, and the fixtures do
#   more than that.  But, object/measurement assocation requiers
#   measurements, and measurements reuqire all kinds of previous
#   data products, so we're stuck with either running a lot, or
#   building a giant stack of fake data structures.  We'll run
#   all this stuff off the simulated image, and probe around a bit
#   to see if things look right.
# (This is really sort of a test of the whole pipeline in the
#   oversimplified case of the new images being perfectly aligned with
#   the ref image and having exactly the same seeing.)
def test_associate_measurements( sim_lightcurve_complete_dses_module,
                                 sim_lightcurve_persistent_sources,
                                 sim_lightcurve_image_parameters ):
    _ref, refds, newdses, _pips = sim_lightcurve_complete_dses_module
    sources = sim_lightcurve_persistent_sources

    dses_detected_for_sources = []
    sourceids_for_sources = []
    allsourceids = set()
    # Figure out which of the subtractions detected which of the sources
    # (Ignoring cos(dec) since I know dec is -3.)
    for source in sources:
        thissourceids = set()
        thisdet = []
        for ds in newdses:
            flux = source['maxflux'] * np.exp( -( ds.image.mjd - ( refds.image.mjd + source['mjdmaxoff'] ) )**2
                                               / ( 2 * source['sigmadays']**2 ) )
            found = False
            for meas in ds.measurements:
                if ( ( np.abs( meas.ra - source['ra'] ) < 1./3600. )
                     and ( np.abs( meas.dec - source['dec'] ) < 1./3600. )
                    ):
                    assert asUUID( meas.id ) not in allsourceids
                    thissourceids.add( asUUID( meas.id ) )
                    allsourceids.add( asUUID( meas.id ) )
                    found = True
                    break
            #  Image FWHM is 2.72 pixels.  That means that the
            #    source is spread over ~23 pixels (for an aperture of r=1FWHM).
            #  Sky noise is about 57 ADU.  So, noise in 1 aperture is about
            #    275 ADU.  A 5σ object is at flux:
            #      f = n sqrt( f/g + s² ) where n
            #    where f is the flux, n is the number of sigma cutoff,
            #    g is the instrument gain (2.0 for DemoInstrument),
            #    and s is the sky noise in the aperture.  Quadraticify,
            #      f = ( n*sqrt( 4g²s² + n² ) + n² ) / 2g
            #    Put in n=5, s=275, g=2, get f = 1381.
            #
            # Determining the actual S/N cutoff of what we should detect
            #   is challenging, because some of the supernovae
            #   are on very bright host galaxies.  Maybe what
            #   I should do is turn off the deletion thresholds,
            #   and then look at is_bad?  For now, just do ~1.5 * 1381.
            assert found or ( flux < 2000. )
            if found:
                thisdet.append( ds )

        dses_detected_for_sources.append( thisdet )
        sourceids_for_sources.append( thissourceids )

    # If Object.associate_measurements worked right, then all of each detection
    #   will be associated with the same object, and that object will not
    #   be associated with anything else.
    # Start by going through sourcesdetected and checking those.

    sourcesobjects = set()
    objsourceids = set()
    with Psycopg2Connection() as conn:
        cursor = conn.cursor()
        for i, source in enumerate( sources ):
            cursor.execute( "SELECT _id, name, ra, dec FROM objects "
                            "WHERE q3c_radial_query( ra, dec, %(ra)s, %(dec)s, 1./3600. )",
                            { 'ra': source['ra'], 'dec': source['dec'] } )
            columns = { cursor.description[i][0]: i for i in range(len(cursor.description)) }
            rows = cursor.fetchall()
            assert len(rows) == 1
            objid = rows[0][columns['_id']]
            sourcesobjects.add( objid )

            cursor.execute( "SELECT _id FROM measurements WHERE object_id=%(objid)s", { 'objid': objid } )
            rows = cursor.fetchall()
            assert len(rows) == len( dses_detected_for_sources[i] )
            cursourceids = set( asUUID(r[0]) for r in rows )
            assert cursourceids == sourceids_for_sources[i]
            objsourceids = objsourceids.union( cursourceids )

        assert allsourceids == objsourceids
        assert len( sourcesobjects ) == len( sources )

        # Look at all the other objects and make sure that none of them are associated with
        #   any of our measurements

        cursor.execute( "SELECT _id FROM objects WHERE _id NOT IN %(objids)s", { 'objids': tuple( sourcesobjects ) } )
        rows = cursor.fetchall()
        assert len( rows ) > 0
        for row in rows:
            cursor.execute( "SELECT _id FROM measurements WHERE object_id=%(objid)s", { 'objid': rows[0] } )
            meases = set( r[0] for r in cursor.fetchall() )
            assert len( meases.intersection( objsourceids ) ) == 0


# This also has a quick test of Object.at_ra_dec
# Interestingly, the function we're testing will have been called
#   in all the fixtures because it's used to generate alerts,
#   which we really don't need for this test, but, eh,
#   it's part of the pipeline, so it's easier to just do it.
def test_get_measurements_et_al( sim_lightcurve_complete_dses_module,
                                 sim_lightcurve_persistent_sources,
                                 sim_lightcurve_image_parameters ):
    _ref, _refds, newdsen, _pips = sim_lightcurve_complete_dses_module
    measprovid = newdsen[0].measurement_set.provenance_id
    deepprovid = newdsen[0].deepscore_set.provenance_id

    # Get the object of the 3rd injected persistent source
    objobj = Object.at_ra_dec( sim_lightcurve_persistent_sources[2]['ra'],
                               sim_lightcurve_persistent_sources[2]['dec'],
                               1.0 )[0]
    assert objobj.ra == pytest.approx( sim_lightcurve_persistent_sources[2]['ra'], abs=1./3600. )
    assert objobj.dec == pytest.approx( sim_lightcurve_persistent_sources[2]['dec'], abs=1./3600. )

    # Get all measurements for the object; there should be 5, ordered by mjd:
    expected_mjd = np.array(  [ 60032.,   60037.,    60040.,    60045.,    60055. ] )
    expected_flux = np.array( [ 1914.028, 2193.6316, 2734.4426, 2939.5178, 2692.7202 ] )
    expected_rb = np.array(   [ 0.5408,   0.5471,    0.5510,    0.6644,    0.5116 ] )

    mess = objobj.get_measurements_et_al( measprovid )
    assert len( mess['measurements'] ) == 5
    assert all( isinstance( m, Measurements ) for m in mess['measurements'] )
    assert all( isinstance( i, Image ) for i in mess['images'] )
    assert all( isinstance( z, ZeroPoint ) for z in mess['zeropoints'] )
    assert all( isinstance( m, MeasurementSet ) for m in mess['measurementsets'] )

    oldmess = mess
    mess = objobj.get_measurements_et_al( measprovid, deepprovid )
    assert len( mess['measurements'] ) == 5
    assert all( isinstance( d, DeepScore ) for d in mess['deepscores'] )
    assert all( isinstance( d, DeepScoreSet ) for d in mess['deepscoresets'] )
    assert all( o.id == m.id for o, m in zip ( oldmess['measurements'], mess['measurements'] ) )
    assert all( o.id == m.id for o, m in zip ( oldmess['measurementsets'], mess['measurementsets'] ) )
    assert all( o.id == m.id for o, m in zip ( oldmess['images'], mess['images'] ) )
    assert all( o.id == m.id for o, m in zip ( oldmess['zeropoints'], mess['zeropoints'] ) )
    mjds = np.array( [ m.mjd for m in mess['images'] ] )
    fluxen = np.array( [ m.flux_psf for m in mess['measurements'] ] )
    rbs = np.array( [ m.score for m in mess['deepscores'] ] )
    assert np.all( np.isclose( mjds, expected_mjd, atol=0.1 ) )
    assert np.all( np.isclose( fluxen, expected_flux, rtol=1e-6 ) )
    assert np.all( np.isclose( rbs, expected_rb, atol=0.001 ) )

    # Try to get everything between mjd 60035 and 60047
    mess = objobj.get_measurements_et_al( measprovid, deepprovid, mjd_min=60035, mjd_max=60047 )
    mjds = np.array( [ m.mjd for m in mess['images'] ] )
    fluxen = np.array( [ m.flux_psf for m in mess['measurements'] ] )
    rbs = np.array( [ m.score for m in mess['deepscores'] ] )
    assert np.all( np.isclose( mjds, expected_mjd[[1,2,3]], atol=0.1 ) )
    assert np.all( np.isclose( fluxen, expected_flux[[1,2,3]], rtol=1e-6 ) )
    assert np.all( np.isclose( rbs, expected_rb[[1,2,3]], atol=0.001 ) )

    # Try doing the same search only using datstrings
    mess = objobj.get_measurements_et_al( measprovid, deepprovid, mjd_min='2023-04-01', mjd_max='2023-04-13' )
    mjds = np.array( [ m.mjd for m in mess['images'] ] )
    fluxen = np.array( [ m.flux_psf for m in mess['measurements'] ] )
    rbs = np.array( [ m.score for m in mess['deepscores'] ] )
    assert np.all( np.isclose( mjds, expected_mjd[[1,2,3]], atol=0.1 ) )
    assert np.all( np.isclose( fluxen, expected_flux[[1,2,3]], rtol=1e-6 ) )
    assert np.all( np.isclose( rbs, expected_rb[[1,2,3]], atol=0.001 ) )

    # Try to get everything with r/b > 0.55
    mess = objobj.get_measurements_et_al( measprovid, deepprovid, min_deepscore=0.55 )
    mjds = np.array( [ m.mjd for m in mess['images'] ] )
    fluxen = np.array( [ m.flux_psf for m in mess['measurements'] ] )
    rbs = np.array( [ m.score for m in mess['deepscores'] ] )
    assert np.all( np.isclose( mjds, expected_mjd[[2,3]], atol=0.1 ) )
    assert np.all( np.isclose( fluxen, expected_flux[[2,3]], rtol=1e-6 ) )
    assert np.all( np.isclose( rbs, expected_rb[[2,3]], atol=0.001 ) )

    # Combine the previous two
    mess = objobj.get_measurements_et_al( measprovid, deepprovid, mjd_min=60044, mjd_max=60046,
                                          min_deepscore=0.55 )
    mjds = np.array( [ m.mjd for m in mess['images'] ] )
    fluxen = np.array( [ m.flux_psf for m in mess['measurements'] ] )
    rbs = np.array( [ m.score for m in mess['deepscores'] ] )
    assert np.all( np.isclose( mjds, expected_mjd[[3]], atol=0.1 ) )
    assert np.all( np.isclose( fluxen, expected_flux[[3]], rtol=1e-6 ) )
    assert np.all( np.isclose( rbs, expected_rb[[3]], atol=0.001 ) )

    # TODO : test thresholds when those are implmeneted


def test_object_legacy_survey_match():
    objid = uuid.uuid4()
    try:
        obj = Object( _id=objid, ra=7.01241, dec=-42.96943, name='test_olsm_object',
                      is_test=True, is_bad=False )
        obj.calculate_coordinates()
        with SmartSession() as sess:
            # This is the RA and Dec of a known SN from DECAT-DDF, also used
            #   in several other tests
            sess.add( obj )
            sess.commit()

        # Try to find and commit new objects
        firstmatches = ObjectLegacySurveyMatch.create_new_object_matches( obj.id, obj.ra, obj.dec )
        assert len(firstmatches) == 4
        assert all( asUUID(m.object_id) == asUUID(objid) for m in firstmatches )
        assert firstmatches[0].lsid == 10995226531340506
        assert firstmatches[0].dist == pytest.approx( 1.131, abs=0.01 )
        assert firstmatches[0].white_mag == pytest.approx( 19.93, abs=0.01 )
        assert not firstmatches[0].is_star
        assert firstmatches[0].xgboost < 0.002

        # See if we get the same matches when we ask for matches
        matches = ObjectLegacySurveyMatch.get_object_matches( obj.id )
        for first, mat in zip( firstmatches, matches ):
            assert asUUID(first._id) == asUUID(mat._id)
            assert first.lsid == mat.lsid
            assert asUUID(first.object_id) == asUUID(mat.object_id)
            assert first.ra == pytest.approx( mat.ra, abs=0.1/3600. / np.cos( first.dec * np.pi / 180. ) )
            assert first.dec == pytest.approx( mat.dec, abs=0.1/3600. )
            assert first.dist == pytest.approx( mat.dist, abs=0.01 )
            assert first.xgboost == pytest.approx( mat.xgboost, abs=0.001 )
            assert first.is_star == mat.is_star

        # Make sure we get yelled at if we try to match when there are existing matches
        with pytest.raises( RuntimeError, match="Object .* already has 4 legacy survey matches" ):
            _ = ObjectLegacySurveyMatch.create_new_object_matches( obj.id, obj.ra, obj.dec )

        # Twiddle one of the magnitudes and make sure we get yelled at if we try to verify_existing
        with Psycopg2Connection() as con:
            cursor = con.cursor()
            cursor.execute( "UPDATE object_legacy_survey_match SET white_mag=19.90 "
                            "WHERE object_id=%(oid)s AND lsid=%(lsid)s",
                            { 'oid': objid, 'lsid': 10995226531340506 } )
            con.commit()
        with pytest.raises( ValueError, match="Object .* already has legacy survey matches, but they aren't" ):
            _ = ObjectLegacySurveyMatch.create_new_object_matches( obj.id, obj.ra, obj.dec, exist_ok=True )

        # But if we turn off verify_existing, we should get the thing I just munged
        matches = ObjectLegacySurveyMatch.create_new_object_matches( obj.id, obj.ra, obj.dec,
                                                                       exist_ok=True, verify_existing=False )
        assert asUUID(matches[0]._id) == asUUID(firstmatches[0]._id)
        assert matches[0].lsid == firstmatches[0].lsid
        assert matches[0].white_mag != firstmatches[0].white_mag

        # Unmung, and add a new match to verify that it yells at us if the number doesn't match
        with Psycopg2Connection() as con:
            cursor = con.cursor()
            cursor.execute( "UPDATE object_legacy_survey_match SET white_mag=19.93 "
                            "WHERE object_id=%(oid)s AND lsid=%(lsid)s",
                            { 'oid': objid, 'lsid': 10995226531340506 } )
            cursor.execute( "INSERT INTO object_legacy_survey_match(_id,object_id,lsid,ra,dec,dist,"
                            "                                       white_mag,xgboost, is_star) "
                            "VALUES(%(id)s,%(objid)s,%(lsid)s,%(ra)s,%(dec)s,%(dist)s,%(mag)s,%(xgb)s,%(iss)s)",
                            { 'id': uuid.uuid4(),
                              'objid': objid,
                              'lsid': 666,
                              'ra': obj.ra,
                              'dec': obj.dec,
                              'dist': 0.,
                              'mag': 19.99,
                              'xgb': 1.0,
                              'iss': True } )
            con.commit()

        with pytest.raises( ValueError, match="Object .* has 5 legacy survey matches.*but I just found 4" ):
            _ = ObjectLegacySurveyMatch.create_new_object_matches( obj.id, obj.ra, obj.dec, exist_ok=True )

        # Make sure we get what I patched in if we don't verify existing:
        matches = ObjectLegacySurveyMatch.create_new_object_matches( obj.id, obj.ra, obj.dec,
                                                                       exist_ok=True, verify_existing=False )
        assert len(matches) == 5
        assert matches[0].white_mag == pytest.approx( 19.99, abs=0.01 )
        assert matches[1].white_mag == pytest.approx( 19.93, abs=0.01 )

    finally:
        with Psycopg2Connection() as con:
            cursor = con.cursor()
            cursor.execute( "DELETE FROM object_legacy_survey_match WHERE object_id=%(id)s", { 'id': objid } )
            cursor.execute( "DELETE FROM objects WHERE _id=%(id)s", { 'id': objid } )
            con.commit()
