import pytest
import uuid
import re

import numpy as np
import sqlalchemy as sa
import psycopg2.errors

from astropy.time import Time

from models.base import SmartSession, Psycopg2Connection
from models.provenance import Provenance
from models.measurements import Measurements
from models.object import Object, ObjectLegacySurveyMatch
from util.util import asUUID


def test_object_creation():
    obj = Object(ra=7.01241, dec=-42.96943, is_test=True, is_bad=False)

    try:
        with pytest.raises( psycopg2.errors.NotNullViolation, match='null value in column "name"' ):
            obj.insert()

        obj.name = "foo"
        obj.insert()

        assert obj._id is not None
        assert re.match( r'^obj\d{4}\w+$', obj.name )

        with SmartSession() as session:
            obj2 = session.scalars(sa.select(Object).where(Object._id == obj.id)).first()
            assert obj2.ra == 1.0
            assert obj2.dec == 2.0
            assert obj2.name is not None
            assert obj2.name == obj.name
            assert re.match( r'^obj\d{4}\w+$', obj.name )

    finally:
        if obj._id is not None:
            with Psycopg2Connection() as conn:
                cursor = conn.cursor()
                cursor.execute( "DELETE FROM objects WHERE _id=%(id)s", { 'id': obj._id } )
                conn.commit()


def test_generate_names():
    try:
        # make sure that things in the year range 3500 aren't in the object_name_max_used table
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "SELECT * FROM object_name_max_used WHERE year>=3500 AND year<3600" )
            rows = cursor.fetchall()
            if len(rows) > 0:
                raise RuntimeError( "object_name_max_used table has rows in the range [3500,3600). "
                                    "This shouldn't be.  Make sure another test isn't doing this." )

        names = Object.generate_names( number=3, year=3500, formatstr="test_gen_name%y%a" )
        assert names == [ "test_gen_name00a", "test_gen_name00b", "test_gen_name00c" ]
        names = Object.generate_names( number=3, year=3501, formatstr="test_gen_name%y%a" )
        assert names == [ "test_gen_name01a", "test_gen_name01b", "test_gen_name01c" ]
        names = Object.generate_names( number=3, year=3501, formatstr="test_gen_name%Y%a" )
        assert names == [ "test_gen_name3501d", "test_gen_name3501e", "test_gen_name3501f" ]
        names = Object.generate_names( number=3, year=3501, formatstr="test_gen_name%Y%a" )
        assert names == [ "test_gen_name3501g", "test_gen_name3501h", "test_gen_name3501i" ]
        names = Object.generate_names( number=3, year=3501, formatstr="test_gen_name%Y_%n" )
        assert names == [ "test_gen_name3501_9", "test_gen_name3501_10", "test_gen_name3501_11" ]
        names = Object.generate_names( number=1, year=3502, formatstr="test_gen_name%Y%A" )
        assert names == [ "test_gen_name3502A" ]

        names = Object.generate_names( number=26**3, year=3503, formatstr="test_gen_name%Y%a" )
        assert names[0] == "test_gen_name3503a"
        assert names[25] == "test_gen_name3503z"
        assert names[1*26 + 0] == "test_gen_name3503ba"
        assert names[1*26 + 25] == "test_gen_name3503bz"
        assert names[2*26 + 0] == "test_gen_name3503ca"
        assert names[25*26 + 25] == "test_gen_name3503zz"
        assert names[1*(26**2) + 0*26 + 0] == "test_gen_name3503baa"
        assert names[25*(26**2) + 0*26 + 0] ==  "test_gen_name3503zaa"
        assert names[25*(26**2) + 25*26 + 25] == "test_gen_name3503zzz"

    finally:
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "DELETE FROM object_name_max_used WHERE year>=3500 AND year<3600" )
            cursor.execute( "DELETE FROM objects WHERE name LIKE 'test_gen_name35%' "
                            "                       OR name LIKE 'test_gen_name00%'"
                            "                       OR name LIKE 'test_gen_name01%'" )
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
def test_associate_measurements( sim_lightcurve_complete_dses,
                                 sim_lightcurve_persistent_sources,
                                 sim_lightcurve_image_parameters ):
    _ref, refds, newdses = sim_lightcurve_complete_dses
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
            #  Image FWHM is 1.11 pixels.  That means that the
            #    source is spread over ~4 pixels (for an aperture of r=1FWHM).
            #  Sky noise is about 57 ADU.  So, noise in 1 aperture is about
            #    115 ADU.  Ideally we'd detect things up to 5σ or 7σ,
            #    but in practice we're getting 10σ. :(
            assert found or ( flux < 1150 )
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


# ...what does this next test have to do with Object?
# @pytest.mark.flaky(max_runs=5)
@pytest.mark.xfail( reason="Issue #346" )
def test_lightcurves_from_measurements(sim_lightcurves):
    assert False
    for lc in sim_lightcurves:
        expected_flux = []
        expected_error = []
        measured_flux = []

        for m in lc:
            measured_flux.append(m.flux_apertures[3] - m.bkg_mean * m.area_apertures[3])
            expected_flux.append(m.sources.data['flux'][m.index_in_sources])
            expected_error.append(m.sources.data['flux_err'][m.index_in_sources])

        assert len(expected_flux) == len(measured_flux)
        for i in range(len(measured_flux)):
            assert measured_flux[i] == pytest.approx(expected_flux[i], abs=expected_error[i] * 3)


# @pytest.mark.flaky(max_runs=5)
@pytest.mark.xfail( reason="Issue #346" )
def test_filtering_measurements_on_object(sim_lightcurves):
    assert False
    assert len(sim_lightcurves) > 0
    assert len(sim_lightcurves[0]) > 3

    idx = 0
    mjds = [m.mjd for m in sim_lightcurves[idx]]
    mjds.sort()

    with SmartSession() as session:
        # make a new provenance for the measurements
        prov = Provenance(
            process=sim_lightcurves[idx][0].provenance.process,
            upstreams=sim_lightcurves[idx][0].provenance.upstreams,
            code_version_id=sim_lightcurves[idx][0].provenance.code_version_id,
            parameters=sim_lightcurves[idx][0].provenance.parameters.copy(),
            is_testing=True,
        )
        prov.parameters['test_parameter'] = uuid.uuid4().hex
        prov.update_id()
        obj = session.merge(sim_lightcurves[idx][0].object)

        measurements = sim_lightcurves[idx]
        new_measurements = []
        for i, m in enumerate(measurements):
            m2 = Measurements()
            for key, value in m.__dict__.items():
                if key not in [
                    '_sa_instance_state',
                    'id',
                    'created_at',
                    'modified',
                    'from_db',
                    'provenance',
                    'provenance_id',
                    'object',
                    'object_id',
                ]:
                    setattr(m2, key, value)
            m2.provenance = prov
            m2.provenance_id = prov.id
            m2.ra += 0.05 * i / 3600.0  # move the RA by less than one arcsec
            m2.ra = m2.ra % 360.0  # make sure RA is in range
            new_measurements.append(m2)

        Object.associate_measurements( measurements, year=2000 )
        # Make sure measurement objects are properly saved to the database after association
        # (Or, anyway, the mysterious and rather annoying SQLAlchemy equivalent.)
        new_new_measurements = []
        for m in new_measurements:
            m2 = session.merge( m )
            new_new_measurements.append( m2 )
        new_measurements = new_new_measurements

        session.commit()
        session.refresh(obj)
        all_ids = [m.id for m in new_measurements + measurements]
        all_ids.sort()
        assert set([m.id for m in obj.measurements]) == set(all_ids)


    assert all([m.id is not None for m in new_measurements])
    assert all([m.id != m2.id for m, m2 in zip(new_measurements, measurements)])

    with SmartSession() as session:
        obj = session.merge(obj)
        # by default will load the most recently created Measurements objects
        found_ids = [m.id for m in obj.get_measurements_list()]
        assert set(found_ids) == set([m.id for m in new_measurements])

        # explicitly ask for the measurements with the provenance hash
        found_ids = [m.id for m in obj.get_measurements_list(prov_hash_list=[prov.id])]
        assert set(found_ids) == set([m.id for m in new_measurements])

        # ask for the old measurements
        found_ids = [m.id for m in obj.get_measurements_list(prov_hash_list=[measurements[0].provenance.id])]
        assert set(found_ids) == set([m.id for m in measurements])

        # get measurements up to a date:
        found = obj.get_measurements_list(mjd_end=mjds[1])
        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(m.mjd <= mjds[1] for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        # get measurements after a date:
        found = obj.get_measurements_list(mjd_start=mjds[1])
        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(m.mjd >= mjds[1] for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        # check this works with datetimes too
        time_start = Time(mjds[1], format='mjd').isot
        found = obj.get_measurements_list(time_start=time_start)
        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(m.mjd >= mjds[1] for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        time_end = Time(mjds[1], format='mjd').isot
        found = obj.get_measurements_list(time_end=time_end)
        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(m.mjd <= mjds[1] for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        # get measurements between two dates
        t1 = Time(mjds[1] - 0.01, format='mjd').isot
        t2 = Time(mjds[2] + 0.01, format='mjd').isot
        found = obj.get_measurements_list(time_start=t1, time_end=t2)

        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(mjds[1] <= m.mjd <= mjds[2] for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        # get measurements that are very close to the source
        found = obj.get_measurements_list(radius=0.1)  # should include only 1-3 measurements
        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(m.distance_to(obj) <= 0.1 for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        # filter on all the offsets disqualifier score
        offsets = [m.disqualifier_scores['offsets'] for m in measurements]  # should be the same as the new measurements
        offsets.sort()
        found = obj.get_measurements_list(thresholds={'offsets': offsets[1]})  # should be only one lower than this
        assert len(found) == 1
        assert found[0].disqualifier_scores['offsets'] < offsets[1]
        assert found[0].id in [m.id for m in new_measurements]

        offsets = [m.disqualifier_scores['offsets'] for m in measurements]  # should be the same as the new measurements
        offsets.sort()
        found = obj.get_measurements_list(thresholds={'offsets': offsets[-1]})  # all but one will pass
        assert len(found) == len(new_measurements) - 1
        assert found[0].disqualifier_scores['offsets'] < offsets[-1]
        for m2 in found:
            assert m2.id in [m.id for m in new_measurements]

        # now give different thresholds for different provenances
        thresholds = {
            measurements[0].provenance.id: {'offsets': offsets[1]},  # will only get one from old list
            prov.id: {'offsets': offsets[-1]},  # should get all but one from the new list, minus one from the old list
        }
        found = obj.get_measurements_list(
            thresholds=thresholds,
            prov_hash_list=[measurements[0].provenance.id, prov.id],  # prefer old measurements
        )
        assert len([m.id for m in found if m.id in [m2.id for m2 in measurements]]) == 1
        assert len([m.id for m in found if m.id in [m2.id for m2 in new_measurements]]) == len(new_measurements) - 2

        # now delete some measurements from each provenance and see if we can get the subsets
        delete_old_id = measurements[0].id
        delete_new_id = new_measurements[-1].id
        old_id_list = [m.id for m in measurements if not m.id == delete_old_id] + [new_measurements[0].id]
        old_id_list.sort()

        new_id_list = [m.id for m in new_measurements if not m.id == delete_new_id] + [measurements[-1].id]
        new_id_list.sort()

        # removing the deleted measurements from the object should remove them from DB as well
        obj.measurements = [m for m in obj.measurements if m.id != delete_old_id and m.id != delete_new_id]
        m = session.scalars(sa.select(Measurements).where(Measurements.id == delete_old_id)).first()
        assert m is None
        m = session.scalars(sa.select(Measurements).where(Measurements.id == delete_new_id)).first()
        assert m is None

        # get the old and only if not found go to the new
        found = obj.get_measurements_list(prov_hash_list=[measurements[0].provenance.id, prov.id])
        assert set([m.id for m in found]) == set(old_id_list)

        # get the new and only if not found go to the old
        found = obj.get_measurements_list(prov_hash_list=[prov.id, measurements[0].provenance.id])
        assert set([m.id for m in found]) == set(new_id_list)


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
