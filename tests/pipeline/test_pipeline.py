import os
import copy
import pytest
import shutil
import datetime
import textwrap

import numpy as np
import psycopg.sql as sql

import sqlalchemy as sa

from models.base import SmartSession, PGDB, FileOnDiskMixin, PsycopgConnection
from models.provenance import Provenance, ProvenanceTag
from models.exposure import Exposure
from models.image import Image
from models.measurements import MeasurementSet
from models.deepscore import DeepScoreSet
from models.report import Report

from pipeline.data_store import DataStore
from pipeline.top_level import Pipeline

from util.logger import SCLogger
from util.util import env_as_bool, asUUID

from tests.conftest import SKIP_WARNING_TESTS


def check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, ds):
    """Check that all the required objects are saved on the database and in the datastore.

    (After running the entire pipeline.)

    Parameters
    ----------
    exp_id: UUID
        The Exposure ID.

    sec_id: str or int
        The section_id of the image from the exposure.

    ref_id: UUID
        The Reference ID.

    ds: datastore.DataStore
        The datastore object

    """

    with PGDB( dictcursor=True ) as pgdb:
        # check the image in the database
        rows = pgdb.execute( sql.SQL( textwrap.dedent(
            """\
            SELECT _id FROM images
            WHERE exposure_id={expid}
              AND section_id={secid}
              AND provenance_id={provid}
            """
        ) ).format( expid=exp_id, secid=str(sec_id), provid=ds.image.provenance_id ) )
        assert len(rows) == 1
        assert ds.image.id == rows[0]['_id']

        # Check lots of other stuff in the database

        # ( prob, table, match_col, match_val )
        matches = [ ( 'sources', 'source_lists', 'image_id', ds.image.id ),
                    ( 'wcs', 'world_coordinates', 'sources_id', ds.sources.id ),
                    ( 'zp', 'zero_points', 'wcs_id', ds.wcs.id ),
                    ( 'detections', 'source_lists', 'image_id', ds.sub_image.id ),
                    ( 'cutouts', 'cutouts', 'sources_id', ds.detections.id ),
                    ( 'measurement_set', 'measurement_sets', 'cutouts_id', ds.cutouts.id ),
                    ( 'deepscore_set', 'deepscore_sets', 'measurementset_id', ds.measurement_set.id )
                   ]
        for prop, tab, matchcol, matchval in matches:
            rows = pgdb.execute(
                sql.SQL( "SELECT _id FROM {table} WHERE {col}={val} AND provenance_id={provid}" )
                .format( table=sql.Identifier(tab), col=sql.Identifier(matchcol),
                         val=matchval, provid=getattr(ds, prop).provenance_id )
            )
            assert len(rows) == 1
            assert rows[0]['_id'] == getattr( ds, prop ).id

        # Things that don't follow the pattern...

        # psf and bg
        rows = pgdb.execute( sql.SQL( "SELECT _id FROM psfs WHERE sources_id={sourcesid}" )
                             .format( sourcesid=ds.sources.id ) )
        assert len(rows) == 1
        assert ds.psf.id == rows[0]['_id']
        rows = pgdb.execute( sql.SQL( "SELECT _id FROM backgrounds WHERE sources_id={sourcesid}" )
                             .format( sourcesid=ds.sources.id ) )
        assert len(rows) == 1
        assert ds.bg.id == rows[0]['_id']

        # sub_image
        rows = pgdb.execute( sql.SQL( textwrap.dedent(
            """\
            SELECT s._id
            FROM images s
            INNER JOIN image_subtraction_components isc ON isc.image_id=s._id
            WHERE isc.ref_id={refid} AND isc.new_zp_id={zpid}
            """
        ) ).format( refid=ref_id, zpid=ds.zp.id ) )
        assert len(rows) == 1
        assert ds.sub_image.id == rows[0]['_id']

        # measurements
        rows = pgdb.execute( sql.SQL( textwrap.dedent(
            """\
            SELECT _id FROM measurements
            WHERE measurementset_id={setid}
            ORDER BY index_in_sources
            """
        ) ).format( setid=ds.measurement_set.id ) )
        assert len(rows) > 0
        assert len(rows) == len(ds.measurements)
        assert all( rows[i]['_id'] == ds.measurements[i].id for i in range( len(rows) ) )

        # deepscores
        rows = pgdb.execute( sql.SQL( textwrap.dedent(
            """\
            SELECT _id FROM deepscores
            WHERE deepscoreset_id={setid}
            ORDER BY index_in_sources
            """
        ) ).format( setid=ds.deepscore_set.id ) )
        assert len(rows) > 0
        assert len(rows) == len(ds.measurements)
        assert len(rows) == len(ds.deepscores)
        assert all( r['_id'] == d.id for r, d in zip( rows, ds.deepscores ) )


def test_parameters( test_config ):
    """Test that pipeline parameters are being set properly"""

    # Verify that we _enforce_no_new_attrs works
    kwargs = { 'pipeline': { 'keyword_does_not_exist': 'testing' } }
    with pytest.raises( AttributeError, match='object has no attribute' ):
        _ = Pipeline( **kwargs )

    # Verify that we can override from the yaml config file
    pipeline = Pipeline()
    assert pipeline.astrometor.pars['cross_match_catalog'] == 'gaia_dr3'
    assert pipeline.astrometor.pars['catalog'] == 'gaia_dr3'
    assert pipeline.subtractor.pars['method'] == 'zogy'

    # TODO: this is based on a temporary "example_pipeline_parameter" that will be removed later
    pipeline = Pipeline( pipeline={ 'example_pipeline_parameter': -999 } )
    assert pipeline.pars['example_pipeline_parameter'] == -999

    # Verify that manual override works for all parts of pipeline
    overrides = {
        'preprocessing': { 'steps': [ 'overscan', 'linearity'] },
        'extraction': {'snr_threshold': 3.14 },
        'astrocal': {'cross_match_catalog': 'override'},
        'photocal': {'cross_match_catalog': 'override'},
        'subtraction': { 'method': 'override' },
        'detection': { 'snr_threshold': 3.14 },
        'cutting': { 'cutout_size': 666 },
        'measuring': { 'negatives_n_sigma_outlier': 3.5 }
    }

    def check_override( new_values_dict, pars ):
        for key, value in new_values_dict.items():
            if pars[key] != value:
                return False
        return True

    pipeline = Pipeline( **overrides )

    assert check_override(overrides['preprocessing'], pipeline.preprocessor.pars)
    assert check_override(overrides['extraction'], pipeline.extractor.pars)
    assert check_override(overrides['astrocal'], pipeline.astrometor.pars)
    assert check_override(overrides['photocal'], pipeline.photometor.pars)
    assert check_override(overrides['subtraction'], pipeline.subtractor.pars)
    assert check_override(overrides['detection'], pipeline.detector.pars)
    assert check_override(overrides['cutting'], pipeline.cutter.pars)
    assert check_override(overrides['measuring'], pipeline.measurer.pars)


# TODO : This really tests that there are no reference provenances defined for the refet
# Also write a test where provenances exist but no reference exists, and then one where
# a reference exists for a different field but not for this field.
def test_running_without_reference(decam_exposure, decam_default_calibrators, pipeline_for_tests):
    p = pipeline_for_tests
    p.subtractor.pars.refset = 'test_refset_decam'  # choosing ref set doesn't mean we have an actual reference
    p.pars.save_before_subtraction = True  # need this so images get saved even though it crashes on "no reference"

    with pytest.raises( RuntimeError, match=( "Failed to create the provenance tree: No reference set "
                                              "with name test_refset_decam found in the database!" ) ):
        # Use the 'N1' sensor section since that's not one of the ones used in the regular
        #  DECam fixtures, so we don't have to worry about any session scope fixtures that
        #  load refererences.  (Though I don't think there are any.)
        _ = p.run(decam_exposure, 'N1')

    with PGDB( dictcursor=True ) as pgdb:
        rows = pgdb.execute( textwrap.dedent(
            """\
            SELECT i.*
            FROM calibrator_files c
            INNER JOIN images i ON i._id=c.image_id
            WHERE c.instrument='DECam'
              AND c.sensor_section='N1'
              AND c.image_id IS NOT NULL
            """
        ) )
        imgtodel = [ Image.create( **row ) for row in rows ]
        for i in imgtodel:
            i.delete_from_disk_and_database()


def check_full_run_results( ds, exposure, sec_id, ref, expected ):
    # Make sure that the provenances are all in the database
    with PGDB() as pgdb:
        for prov in ds.prov_tree.values():
            rows, _cols = pgdb.execute( sql.SQL( "SELECT _id FROM provenances WHERE _id={provid}" )
                                        .format( provid=prov.id ) )
            assert len(rows) == 1

    # Check that all the data products are in the database
    check_datastore_and_database_have_everything( exposure.id, sec_id, ref.id, ds )

    # Get the measurements and deepscores.
    # Filter only on the DeepScoreSet provenance since that's the lowest
    #   one down, so by going at upstreams from the DeepScoreSet we know
    #   we're getting the right things.
    # (This is perhaps gratuitous, since the measurements and deescores
    #   should aready be in DataStore ds, but, well, I guess this is
    #   also checking it all got saved right to the database.)
    with PGDB( dictcursor=True ) as pgdb:
        rows = pgdb.execute( sql.SQL( textwrap.dedent(
            """\
            SELECT m.*, to_jsonb(d) AS deepscore_set
            FROM deepscore_sets d
            INNER JOIN measurement_sets m ON d.measurementset_id=m._id
            INNER JOIN cutouts c ON m.cutouts_id=c._id
            INNER JOIN source_lists dets ON c.sources_id=dets._id
            INNER JOIN images sub ON dets.image_id=sub._id
            INNER JOIN image_subtraction_components isc ON isc.image_id=sub._id
            INNER JOIN zero_points z ON isc.new_zp_id=z._id
            INNER JOIN world_coordinates w ON z.wcs_id=w._id
            INNER JOIN source_lists s ON w.sources_id=s._id
            INNER JOIN images i ON s.image_id=i._id
            WHERE d.provenance_id={detprov}
              AND i.exposure_id={expid}
            """
        ) ).format( detprov=ds.prov_tree['scoring'].id, expid=exposure.id ) )
        assert len(rows) == 1
        deep = DeepScoreSet.create( **(rows[0]['deepscore_set']))
        # The JSON games will have turned the UUID into a string
        deep._id = asUUID( deep._id )
        del rows[0]['deepscore_set']
        meas = MeasurementSet.create( **(rows[0]) )

    # ---> REGRESSION TEST : look at some of the results to make
    #   sure that things behaved as expected.  If not, then either
    #   find the errors, or, if there no errors and it's legitimate
    #   changes, then *some* process version numbers should be
    #   bumped.  Good luck figuring out which one(s)!  (Hopefully,
    #   other test failures will give you a clue.)  (This is
    #   probably not a sufficient regression test; also
    #   test_pipeline_exposure_launcher, with some checking of
    #   measurements and scores there, should be run, at the very
    #   least!)
    #
    # Since this is a regression test, you might argue that the abs= and rel=
    # values in the pytest.approx calls below should be smaller.

    assert len( meas.measurements ) == len( expected['x'] )
    assert len( deep.deepscores ) == len( meas.measurements )

    # We don't care about things being in the same order, but the set of
    #   measurements and deepscores should match.  Find the match based
    #   on x and y being both within 0.05 of what's expected.
    dexen = []
    for m in meas.measurements:
        w = np.where( ( np.fabs( expected['x'] - m.x ) < 0.05 ) & ( np.fabs( expected['y'] - m.y ) < 0.05 ) )[0]
        assert len(w) == 1
        dexen.append( w[0] )
    assert set( dexen ) == set( range(len(expected['x'])) )

    for m, d, i in zip( meas.measurements, deep.deepscores, dexen ):
        assert expected['gfit_x'][i] == pytest.approx( m.gfit_x, abs=0.05 )
        assert expected['gfit_y'][i] == pytest.approx( m.gfit_y, abs=0.05 )
        assert expected['major_width'][i] == pytest.approx( m.major_width, abs=0.05 )
        assert expected['minor_width'][i] == pytest.approx( m.minor_width, abs=0.05 )
        assert expected['neg_frac'][i] == pytest.approx( m.negfrac, abs=0.02 )
        assert expected['neg_flux_frac'][i] == pytest.approx( m.negfluxfrac, abs=0.05 )
        assert expected['psf_flux_err'][i] == pytest.approx( m.flux_psf_err, rel=0.05 )
        assert expected['psf_flux'][i] == pytest.approx( m.flux_psf, abs=m.flux_psf_err / 20. )
        assert expected['aper_flux_err'][i] == pytest.approx( m.flux_apertures_err[0], rel=0.05 )
        assert expected['aper_flux'][i] == pytest.approx( m.flux_apertures[0], abs=m.flux_apertures_err[0] / 20. )
        assert expected['rb'][i] == pytest.approx( d.score, abs=0.03 )


# The user fixture is here because this is a convenient test for looking
#  to see what we've got on the weabp.  This will only work if you're
#  running the tests on the dekstop or laptop you're sitting at.  Put a
#  breakpoint at the end of the test, and then go log into the webap
#  (which will be at https://localhost:8081, with 8081 replaced with the
#  value of WEBAP_PORT if you set that when running the docker compose
#  file), log in with username "test" and password "test_password",
#  and check things out.
def test_full_run_zogy( decam_exposure, decam_reference, decam_default_calibrators, user ):
    # The source at 1619.22, 1881.40 is a real SN
    expected = {
        'x':      np.array( [ 1984.90,  785.16, 1409.14, 1452.95, 1917.67, 1439.39, 1451.49, 1619.22, 1441.19,
                              1928.02, 1358.37,  457.96, 1518.36 ] ),
        'y':      np.array( [  221.94,  468.16,  757.69,  807.57, 1394.23, 1463.61, 1627.44, 1881.40, 2048.38,
                               2093.13, 3491.88, 4008.20, 4043.33 ] ),
        'gfit_x':           [ 1985.56,  785.19, 1408.73, 1451.45, 1917.77, 1438.04, 1450.82, 1619.22, 1441.06,
                              1927.90, 1358.17,  457.86, 1516.75 ],
        'gfit_y':           [  221.53,  468.16,  757.71,  809.32, 1394.10, 1464.79, 1628.07, 1881.28, 2048.19,
                               2092.87, 3492.08, 4008.47, 4041.56 ],
        'major_width': [  5.07,  2.54,  6.71,  8.60,  2.21,  7.88,  6.79,  4.05,  3.92,  6.19,  9.74,  4.99,  7.73 ],
        'minor_width': [  2.15,  2.04,  4.53,  6.99,  1.85,  7.42,  6.15,  3.18,  2.35,  2.96,  4.25,  2.48,  7.50 ],
        'neg_frac':      [ 0.24, 0.06, 0.23, 0.07, 0.25, 0.02, 0.11, 0.00, 0.23, 0.00, 0.00, 0.05, 0.02 ],
        'neg_flux_frac': [ 0.29, 0.01, 0.10, 0.09, 0.19, 0.02, 0.12, -0.00, 0.24, 0.00, -0.00, 0.07, 0.05 ],
        'psf_flux':      [   2336.,   8647.,  94666.,  10838.,    518.,   3444.,  11195.,   2299.,   1335.,
                              483.,    821.,   3003.,  21014. ],
        'psf_flux_err':  [    130.,    110.,    467.,    185.,     89.,    133.,    178.,     97.,    111.,
                               92.,     87.,    123.,    220. ],
        'aper_flux':     [   1410.,   5861.,  71652.,  11512.,    346.,   3872.,   9973.,   1900.,    761.,
                              431.,   1050.,   2925.,  27991. ],
        'aper_flux_err': [    145.,    106.,    554.,    206.,     99.,    145.,    188.,    105.,    124.,
                              104.,    100.,    149.,    265. ],
        'rb': [ 0.624, 0.880, 0.439, 0.388, 0.493, 0.561, 0.457, 0.791, 0.626, 0.519, 0.488, 0.608, 0.452 ]
    }

    try:
        # subtraction.method zogy and detection.method filter should already
        #   be in the defaults from the config file, but be explicit
        #   here for clarity (and comparison to test_full_run_hotpants)
        pipeline = Pipeline( pipeline={ 'provenance_tag': 'test_full_run_zogy' },
                             subtraction={ 'method': 'zogy',
                                           'refset': 'test_refset_decam' },
                             detection={ 'method': 'filter' } )
        ds = pipeline.run( decam_exposure, decam_reference.image.section_id )
        ds.save_and_commit()

        # # Uncomment this bit to produce the "expected" regression results above;
        # #   will need to fix indentation and (maybe) add newlines.
        # # Run the test with --capture=tee-sys to see the output.
        # import io
        # strio = io.StringIO()
        # strio.write(
        #     f"""
        #     expected = {{
        #         'x':      np.array( [ {', '.join(f"{m.x:7.2f}" for m in ds.measurements)} ] ),
        #         'y':      np.array( [ {', '.join(f"{m.y:7.2f}" for m in ds.measurements)} ] ),
        #         'gfit_x':           [ {', '.join(f"{m.gfit_x:7.2f}" for m in ds.measurements)} ],
        #         'gfit_y':           [ {', '.join(f"{m.gfit_y:7.2f}" for m in ds.measurements)} ],
        #         'major_width': [ {', '.join(f"{m.major_width:5.2f}" for m in ds.measurements)} ],
        #         'minor_width': [ {', '.join(f"{m.minor_width:5.2f}" for m in ds.measurements)} ],
        #         'neg_frac':      [ {', '.join(f"{m.negfrac:4.2f}" for m in ds.measurements)} ],
        #         'neg_flux_frac': [ {', '.join(f"{m.negfluxfrac:4.2f}" for m in ds.measurements)} ],
        #         'psf_flux':      [ {', '.join(f"{m.flux_psf:6.0f}." for m in ds.measurements)} ],
        #         'psf_flux_err':  [ {', '.join(f"{m.flux_psf_err:6.0f}." for m in ds.measurements)} ],
        #         'aper_flux':     [ {', '.join(f"{m.flux_apertures[0]:6.0f}." for m in ds.measurements)} ],
        #         'aper_flux_err': [ {', '.join(f"{m.flux_apertures_err[0]:6.0f}." for m in ds.measurements)} ],
        #         'rb': [ {', '.join(f"{d.score:5.3f}" for d in ds.deepscores)} ]
        #     }}
        #     """
        # )
        # SCLogger.info( f"Here it is:\n{strio.getvalue()}" )
        # import pdb; pdb.set_trace()

        check_full_run_results( ds, decam_exposure, decam_reference.image.section_id, decam_reference, expected )

    finally:
        if 'ds' in locals():
            ds.delete_everything()
        with PsycopgConnection() as con:
            cursor = con.cursor()
            cursor.execute( "DELETE FROM provenance_tags WHERE tag='test_full_run_zogy'" )
            con.commit()


# See comment on test_full_run_zogy for the reason for the user fixture
def test_full_run_hotpants( decam_exposure, decam_reference, decam_default_calibrators, user, test_config ):
    # A lot of the stuff that pass the cuts is really bad -- lots of CRs getting through.
    # I think they didn't get through with zogy because zogy effectively convolved them out
    # a bit (as the ref had worse seeing the the new here), and the blurring led to
    # Gaussian fits that did a better job of seeing very non-round things.  Perhaps thought
    # required on our analytic cuts.  But, also, R/B really needs to be trained better to
    # get rid of all of that, as a lot of these 27 are junk with high R/B.
    #
    # ALSO: I think we *might* be able to be more stringent on dipole detection with Alard/Luption
    # than we are with zogy.  Investigation required.
    #
    # The source at 1619.21, 1881.42 is a real SN.

    expected = {
        'x':      np.array( [ 1519.08,  458.19,  458.25,  593.75, 1192.19, 1265.52, 1108.80,  192.91, 1645.96,
                               949.14,  903.47, 1752.72,  998.42,  983.28,  291.46, 1984.85, 1563.97, 1619.21,
                               397.92, 1581.49, 1451.79, 1143.22, 1741.33, 1439.00, 1437.29,  616.39, 1185.98,
                               698.96,  246.24, 1451.11, 1409.10,  177.78, 1435.24,  785.16, 1405.95, 1610.92,
                               1984.46, 1576.61, 1573.30,  786.63 ] ),
        'y':      np.array( [ 4039.62, 4008.88, 4008.75, 3527.51, 3450.32, 3409.28, 3402.39, 3280.49, 2929.00,
                              2847.72, 2747.91, 2620.64, 2278.11, 2273.60, 2228.04, 2026.02, 2006.98, 1881.41,
                              1732.86, 1665.81, 1627.63, 1571.47, 1485.58, 1464.84, 1462.08, 1310.13, 1085.24,
                               948.00,  933.89,  812.40,  758.22,  704.00,  477.55,  468.16,  442.23,  374.36,
                               220.79,  181.67,  180.33,  168.42 ] ),
        'gfit_x':           [ 1517.14,  457.84,  457.84,  593.42, 1192.27, 1265.73, 1108.94,  192.78, 1645.87,
                               949.17,  903.51, 1752.83,  998.43,  983.31,  291.49, 1984.95, 1564.23, 1619.22,
                               398.00, 1581.51, 1450.96, 1143.42, 1741.41, 1438.29, 1438.29,  616.41, 1186.01,
                               698.82,  246.33, 1451.52, 1408.76,  177.89, 1435.30,  785.21, 1405.99, 1610.86,
                               1984.66, 1576.84, 1575.26,  786.68 ],
        'gfit_y':           [ 4041.87, 4008.77, 4008.77, 3527.21, 3450.46, 3409.50, 3402.59, 3280.40, 2929.27,
                              2847.63, 2747.88, 2620.62, 2278.16, 2273.17, 2228.10, 2025.94, 2006.85, 1881.32,
                              1733.23, 1665.88, 1628.44, 1571.61, 1485.61, 1465.15, 1465.15, 1310.15, 1085.21,
                               948.13,  933.92,  809.42,  758.05,  703.85,  477.70,  468.15,  442.27,  374.45,
                               220.54,  181.84,  180.84,  168.45 ],
        'major_width': [  7.95,  5.87,  5.87,  1.56,  1.62,  4.68,  2.17,  1.85,  0.90,  3.70,  2.19,  2.48,
                          3.53,  2.51,  1.20,  4.86,  0.57,  3.94,  0.43,  3.01,  7.47,  1.84,  2.07,  7.96,
                          7.96,  2.18,  2.34,  0.64,  2.08,  8.44,  7.15,  0.77,  1.40,  2.50,  1.59,  0.88,
                          3.80,  2.74,  6.29,  0.99 ],
        'minor_width': [  6.74,  2.02,  2.02,  0.62,  0.82,  1.81,  0.95,  0.74,  0.43,  1.41,  1.37,  1.56,
                          1.21,  1.04,  1.06,  3.19,  0.39,  3.19,  0.31,  1.51,  6.40,  0.68,  1.62,  7.52,
                          7.52,  1.76,  1.67,  0.54,  1.53,  6.90,  5.13,  0.66,  0.96,  1.96,  1.55,  0.51,
                          1.77,  1.74,  2.94,  0.62 ],
        'neg_frac':      [ 0.01, 0.14, 0.14, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.03, 0.00, 0.00, 0.00,
                           0.14, 0.25, 0.00, 0.06, 0.00, 0.00, 0.07, 0.00, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00,
                           0.00, 0.07, 0.17, 0.00, 0.00, 0.00, 0.00, 0.00, 0.24, 0.00, 0.01, 0.00 ],
        'neg_flux_frac': [ 0.00, 0.18, 0.18, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.02, 0.27, 0.00, 0.03, 0.00, 0.00, 0.06, 0.00, 0.00, 0.04, 0.04, 0.00, 0.00, 0.00,
                           0.00, 0.07, 0.07, 0.00, 0.00, 0.00, 0.00, 0.00, 0.19, 0.00, 0.00, 0.00 ],
        'psf_flux':      [  19090.,   2340.,   2969.,   4050.,   6993.,  12925.,   4356.,   4815.,    620.,   7405.,
                             8262.,  13013.,   6467.,   5645.,   2859.,   1390.,   1468.,   2347.,    578.,   8752.,
                            10268.,   6364.,   7924.,   3624.,   2223.,   7568.,   8401.,   1203.,   9584.,   8622.,
                           103492.,    504.,   8016.,   8550.,   4852.,   2299.,   1757.,  16487.,  14246.,   3416. ],
        'psf_flux_err':  [    190.,    109.,    110.,     96.,    105.,    113.,    101.,    100.,     90.,    104.,
                              106.,    116.,    102.,     99.,     97.,    107.,     91.,     96.,     90.,    108.,
                              168.,    104.,    107.,    145.,    109.,    107.,    108.,     92.,    110.,    158.,
                              454.,     92.,    109.,    109.,    102.,     97.,    107.,    121.,    116.,    102. ],
        'aper_flux':     [  16938.,   2340.,   1478.,   4678.,   5765.,  13466.,   4356.,   4159.,    620.,   5630.,
                             6590.,  11471.,   5367.,   5365.,   1844.,   1236.,   1468.,   1909.,    578.,   6765.,
                             9402.,   5908.,   5486.,   3507.,   2321.,   4958.,   6365.,   1203.,   7272.,   7841.,
                             84008.,    504.,   9071.,   5709.,   3052.,   2299.,   1599.,  15938.,  14476.,   3416. ],
        'aper_flux_err': [    214.,    133.,    144.,    104.,    105.,    114.,    104.,    104.,     99.,    105.,
                              106.,    113.,    105.,    105.,    103.,    119.,    100.,    104.,     99.,    107.,
                              171.,    106.,    105.,    138.,    123.,    105.,    107.,    100.,    108.,    161.,
                              439.,    100.,    110.,    106.,    102.,    101.,    123.,    117.,    115.,    103. ],
        'rb': [ 0.452, 0.654, 0.426, 0.628, 0.880, 0.788, 0.904, 0.899, 0.716, 0.778, 0.891, 0.852, 0.601, 0.613,
                0.791, 0.819, 0.741, 0.779, 0.766, 0.777, 0.489, 0.838, 0.921, 0.600, 0.495, 0.847, 0.841, 0.795,
                0.809, 0.695, 0.528, 0.761, 0.812, 0.881, 0.891, 0.815, 0.548, 0.814, 0.325, 0.894 ]
    }

    try:
        detconfigs = test_config.value( 'detection' )
        detconfigs['subconfigs']['subconfigs']['extragalactic']['method'] = 'sextractor'
        detconfigs['subconfigs']['subconfigs']['galactic']['method'] = 'sextractor'
        pipeline = Pipeline( pipeline={ 'provenance_tag': 'test_full_run_hotpants' },
                             subtraction={ 'method': 'hotpants',
                                           'refset': 'test_refset_decam' },
                             detection=detconfigs )
        ds = pipeline.run( decam_exposure, decam_reference.image.section_id )
        ds.save_and_commit()

        # # Uncomment this bit to produce the "expected" regression results above;
        # #   will need to fix indentation and (maybe) add newlines
        # import io
        # strio = io.StringIO()
        # strio.write(
        #     f"""
        #     expected = {{
        #         'x':      np.array( [ {', '.join(f"{m.x:7.2f}" for m in ds.measurements)} ] ),
        #         'y':      np.array( [ {', '.join(f"{m.y:7.2f}" for m in ds.measurements)} ] ),
        #         'gfit_x':           [ {', '.join(f"{m.gfit_x:7.2f}" for m in ds.measurements)} ],
        #         'gfit_y':           [ {', '.join(f"{m.gfit_y:7.2f}" for m in ds.measurements)} ],
        #         'major_width': [ {', '.join(f"{m.major_width:5.2f}" for m in ds.measurements)} ],
        #         'minor_width': [ {', '.join(f"{m.minor_width:5.2f}" for m in ds.measurements)} ],
        #         'neg_frac':      [ {', '.join(f"{m.negfrac:4.2f}" for m in ds.measurements)} ],
        #         'neg_flux_frac': [ {', '.join(f"{m.negfluxfrac:4.2f}" for m in ds.measurements)} ],
        #         'psf_flux':      [ {', '.join(f"{m.flux_psf:6.0f}." for m in ds.measurements)} ],
        #         'psf_flux_err':  [ {', '.join(f"{m.flux_psf_err:6.0f}." for m in ds.measurements)} ],
        #         'aper_flux':     [ {', '.join(f"{m.flux_apertures[0]:6.0f}." for m in ds.measurements)} ],
        #         'aper_flux_err': [ {', '.join(f"{m.flux_apertures_err[0]:6.0f}." for m in ds.measurements)} ],
        #         'rb': [ {', '.join(f"{d.score:5.3f}" for d in ds.deepscores)} ]
        #     }}
        #     """
        # )
        # SCLogger.info( f"Here it is:\n{strio.getvalue()}" )
        # import pdb; pdb.set_trace()

        check_full_run_results( ds, decam_exposure, decam_reference.image.section_id, decam_reference, expected )

    finally:
        if 'ds' in locals():
            ds.delete_everything()
        with PsycopgConnection() as con:
            cursor = con.cursor()
            cursor.execute( "DELETE FROM provenance_tags WHERE tag='test_full_run_hotpants'" )
            con.commit()



@pytest.mark.skipif( not env_as_bool('RUN_SLOW_TESTS'), reason="Set RUN_SLOW_TESTS=1 to run this test" )
def test_data_flow(decam_exposure, decam_reference, decam_default_calibrators, pipeline_for_tests, archive):
    """Test that the pipeline runs end-to-end.

    Also check that it regenerates things that are missing. The
    iteration of that makes this a slow test....

    TODO : given test_full_run_zogy above, we should make a faster
    version of this to check that things are regenerated or not as
    expected, e.g. using a small sample image that will run through very
    quickly.

    """
    exposure = decam_exposure

    ref = decam_reference
    sec_id = ref.image.section_id
    try:  # cleanup the file at the end
        p = pipeline_for_tests
        p.subtractor.pars.refset = 'test_refset_decam'
        assert p.extractor.pars.snr_threshold != 3.14
        assert p.detector.pars.snr_threshold != 3.14

        ds = p.run(exposure, sec_id)
        ds.save_and_commit()

        with SmartSession() as session:
            # check that everything is in the database
            provs = session.scalars(sa.select(Provenance)).all()
            assert len(provs) > 0
            prov_processes = [p.process for p in provs]
            expected_processes = ['preprocessing', 'extraction', 'subtraction', 'detection',
                                  'cutting', 'measuring', 'scoring']
            for process in expected_processes:
                assert process in prov_processes

            check_datastore_and_database_have_everything(exposure.id, sec_id, ref.id, ds)

        # feed the pipeline the same data, but missing the upstream data.
        attributes = ['image', 'sources', 'sub_image', 'detections', 'cutouts', 'measurement_set', 'deepscore_set']

        # TODO : put in the loop below a verification that the processes were
        #   not rerun, but products were just loaded from the database
        for i in range(len(attributes)):
            SCLogger.debug( f"test_data_flow: testing removing everything up through {attributes[i]}" )
            for j in range(i + 1):
                setattr(ds, attributes[j], None)  # get rid of all data up to the current attribute
            # SCLogger.debug(f'removing attributes up to {attributes[i]}')
            ds = p.run(ds)  # for each iteration, we should be able to recreate the data
            ds.save_and_commit()

            check_datastore_and_database_have_everything(exposure.id, sec_id, ref.id, ds)

        # make sure we can remove the data from the end to the beginning and recreate it
        # TODO : this is a test that the pipeline can pick up if it's partially done.
        #   put in checks to verify the earlier processes weren't rerun.
        # Maybe also create a test where partial products exist in the database to verify
        #   that the pipeline doesn't recreate those but does recreate the later ones.
        for i in range(len(attributes)):
            SCLogger.debug( f"test_data_flow: testing removing everything after {attributes[-i-1]}" )
            for j in range(i):
                obj = getattr(ds, attributes[-j-1])
                if isinstance(obj, FileOnDiskMixin):
                    obj.delete_from_disk_and_database()

                setattr(ds, attributes[-j-1], None)

            ds = p.run(ds)  # for each iteration, we should be able to recreate the data
            ds.save_and_commit()

            check_datastore_and_database_have_everything(exposure.id, sec_id, ref.id, ds)

    finally:
        if 'ds' in locals():
            ds.delete_everything()


def test_bitflag_propagation(decam_exposure, decam_reference, decam_default_calibrators, pipeline_for_tests, archive):
    """Test that adding a bitflag to the exposure propagates to all downstreams as they are created.

    Does not check measurements, as they do not have the HasBitflagBadness Mixin.
    """
    exposure = decam_exposure
    ref = decam_reference
    sec_id = ref.image.section_id

    try:  # cleanup the file at the end
        p = Pipeline( pipeline={'provenance_tag': 'test_bitflag_propagation'} )
        p.subtractor.pars.refset = 'test_refset_decam'
        exposure.set_badness( 'banding' )  # add a bitflag to check for propagation

        # first run the pipeline and check for basic propagation of the single bitflag
        ds = p.run(exposure, sec_id)

        assert ds.exposure._bitflag == 2     # 2**1 is the bitflag for 'banding'
        assert ds.image._upstream_bitflag == 2
        assert ds.sources._upstream_bitflag == 2
        assert ds.psf._upstream_bitflag == 2
        assert ds.bg._upstream_bitflag == 2
        assert ds.wcs._upstream_bitflag == 2
        assert ds.zp._upstream_bitflag == 2
        assert ds.sub_image._upstream_bitflag == 2
        assert ds.detections._upstream_bitflag == 2
        assert ds.cutouts._upstream_bitflag == 2
        assert ds.measurement_set._upstream_bitflag == 2
        for m in ds.measurements:
            assert m._upstream_bitflag == 2
        assert ds.deepscore_set._upstream_bitflag == 2
        for d in ds.deepscores:
            assert d._upstream_bitflag == 2

        # test part 2: Add a second bitflag partway through and check it propagates to downstreams

        # delete downstreams of ds.sources
        # The only immediate downstream is wcs, and both delete_from_disk_and_database and wcs.setter will
        #   recursively clear out the downstreams
        ds.wcs.delete_from_disk_and_database()
        ds.wcs = None

        # (Make sure other downstreams got cleared... i.e., make sure I didn't lie in the last comment)
        # (If I was really good, I'd also make sure they weren't in the database or on disk.)
        for prop in ( '_zp', '_sub_image', '_detections', '_cutouts', '_measurement_set',
                      '_deepscore_set', '_aligned_ref_image', '_aligned_ref_sources',
                      '_aligned_ref_bg', '_aligned_ref_psf' ):
            assert getattr( ds, prop ) is None

        ds.sources._set_bitflag( 2 ** 17 )  # bitflag 2**17 is 'many sources'
        desired_bitflag = 2 ** 1 + 2 ** 17  # bitflag for 'banding' and 'many sources'
        ds = p.run(ds)

        assert ds.sources.bitflag == desired_bitflag
        assert ds.wcs._upstream_bitflag == desired_bitflag
        assert ds.zp._upstream_bitflag == desired_bitflag
        assert ds.sub_image._upstream_bitflag == desired_bitflag
        assert ds.detections._upstream_bitflag == desired_bitflag
        assert ds.cutouts._upstream_bitflag == desired_bitflag
        assert ds.measurement_set._upstream_bitflag == desired_bitflag
        for m in ds.measurements:
            assert m._upstream_bitflag == desired_bitflag
        assert ds.deepscore_set._upstream_bitflag == desired_bitflag
        for d in ds.deepscores:
            assert d._upstream_bitflag == desired_bitflag
        assert ds.image.bitflag == 2  # not in the downstream of sources

        # test part 3: test update_downstream_badness() function by adding and removing flags
        # and observing propagation

        ds.save_and_commit()       # Redundant, already happened in p.run(ds) above

        # add a bitflag and check that it appears in downstreams

        ds.image._set_bitflag( 2 ** 4 )  # bitflag for 'bad subtraction'
        ds.image.upsert()
        ds.exposure.update_downstream_badness()

        desired_bitflag = 2 ** 1 + 2 ** 4 + 2 ** 17  # 'banding' 'bad subtraction' 'many sources'

        assert Exposure.get_by_id( ds.exposure.id )._bitflag == 2 ** 1
        assert ds.get_image( reload=True ).bitflag == 2 ** 1 + 2 ** 4  # 'banding' and 'bad subtraction'
        assert ds.get_sources( reload=True ).bitflag == desired_bitflag
        assert ds.get_psf( reload=True ).bitflag == desired_bitflag
        assert ds.get_wcs( reload=True ).bitflag == desired_bitflag
        assert ds.get_zp( reload=True ).bitflag == desired_bitflag
        assert ds.get_sub_image( reload=True ).bitflag == desired_bitflag
        assert ds.get_detections( reload=True ).bitflag == desired_bitflag
        assert ds.get_cutouts( reload=True ).bitflag == desired_bitflag
        assert ds.get_measurement_set( reload=True ).bitflag == desired_bitflag
        for m in ds.measurements:
            assert m.bitflag == desired_bitflag
        assert ds.get_deepscore_set( reload=True ).bitflag == desired_bitflag
        for d in ds.deepscores:
            assert d.bitflag == desired_bitflag

        # remove the bitflag and check that it disappears in downstreams
        ds.image._set_bitflag( 0 )  # remove 'bad subtraction'
        ds.exposure.update_downstream_badness()

        desired_bitflag = 2 ** 1 + 2 ** 17  # 'banding' 'many sources'
        assert ds.exposure.bitflag == 2 ** 1
        assert ds.get_image( reload=True ).bitflag == 2 ** 1  # just 'banding' left on image
        assert ds.get_sources( reload=True ).bitflag == desired_bitflag
        assert ds.get_psf( reload=True ).bitflag == desired_bitflag
        assert ds.get_wcs( reload=True ).bitflag == desired_bitflag
        assert ds.get_zp( reload=True ).bitflag == desired_bitflag
        assert ds.get_sub_image( reload=True ).bitflag == desired_bitflag
        assert ds.get_detections( reload=True ).bitflag == desired_bitflag
        assert ds.get_cutouts( reload=True ).bitflag == desired_bitflag
        assert ds.get_measurement_set( reload=True ).bitflag == desired_bitflag
        for m in ds.measurements:
            assert m.bitflag == desired_bitflag
        assert ds.get_deepscore_set( reload=True ).bitflag == desired_bitflag
        for d in ds.deepscores:
            assert d.bitflag == desired_bitflag


        # TODO : adjust ds.sources's bitflag, and make sure that it
        # propagates to sub_image.  (I believe right now in the code it
        # won't, but it should!)


    finally:
        if 'ds' in locals():
            ds.delete_everything()
        # added this cleanup to make sure the temp data folder is cleaned up
        # this should be removed after we add datastore failure modes (issue #150)
        shutil.rmtree(os.path.join(os.path.dirname(exposure.get_fullpath()), '115'), ignore_errors=True)
        shutil.rmtree(os.path.join(archive.test_folder_path, '115'), ignore_errors=True)

        # Reset the exposure bitflag since this is a session fixture
        exposure._set_bitflag( 0 )
        exposure.upsert()

        # Remove the ProvenanceTag that will have been created
        with SmartSession() as session:
            session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag='test_bitflag_propagation'" ) )
            session.commit()


def test_get_upstreams_and_downstreams(decam_exposure, decam_reference, decam_default_calibrators, archive):
    """Test that get_upstreams() and get_downstreams() return the proper objects."""
    exposure = decam_exposure
    ref = decam_reference
    sec_id = ref.image.section_id

    try:  # cleanup the file at the end
        p = Pipeline( pipeline={'provenance_tag': 'test_get_upstreams_and_downstreams'} )
        p.subtractor.pars.refset = 'test_refset_decam'
        ds = p.run(exposure, sec_id)

        ds.save_and_commit()
        with PGDB() as pgdb:
            # test get_upstreams()
            assert ds.exposure.get_upstreams() == []
            assert [upstream.id for upstream in ds.image.get_upstreams(pgdb=pgdb)] == [ds.exposure.id]
            assert [upstream.id for upstream in ds.sources.get_upstreams(pgdb=pgdb)] == [ds.image.id]
            assert [upstream.id for upstream in ds.wcs.get_upstreams(pgdb=pgdb)] == [ds.sources.id]
            assert [upstream.id for upstream in ds.psf.get_upstreams(pgdb=pgdb)] == [ds.sources.id]
            assert [upstream.id for upstream in ds.zp.get_upstreams(pgdb=pgdb)] == [ds.wcs.id]
            assert ( set([ upstream.id for upstream in ds.sub_image.get_upstreams( pgdb=pgdb ) ])
                     == { ds.reference.id, ds.zp.id } )
            assert [upstream.id for upstream in ds.detections.get_upstreams(pgdb=pgdb)] == [ds.sub_image.id]
            assert [upstream.id for upstream in ds.cutouts.get_upstreams(pgdb=pgdb)] == [ds.detections.id]
            assert [upstream.id for upstream in ds.measurement_set.get_upstreams(pgdb=pgdb)] == [ds.cutouts.id]
            for measurement in ds.measurements:
                assert ( [upstream.id for upstream in measurement.get_upstreams(pgdb=pgdb)]
                         == [ds.measurement_set.id] )
            assert ( [upstream.id for upstream in ds.deepscore_set.get_upstreams(pgdb=pgdb)]
                     == [ds.measurement_set.id] )
            for deepscore in ds.deepscores:
                assert ( [upstream.id for upstream in deepscore.get_upstreams(pgdb=pgdb)]
                         == [ds.deepscore_set.id] )

            # test get_downstreams
            # When this test is run by itself, the exposure only has a
            #   single downstream.  When it's run in the context of
            #   other tests, it has two downstreams.  I'm a little
            #   surprised by this, because the decam_reference fixture
            #   ultimately (tracking it back) runs the
            #   decam_elais_e1_two_refs_datastore fixture, which should
            #   create two downstreams for the exposure.  However, it
            #   probably has to do with when things get committed to the
            #   actual database and with the whole mess around
            #   SQLAlchemy pgdbs.  Making decam_exposure a
            #   function-scope fixture (rather than the pgdb-scope
            #   fixture it is right now) would almost certainly make
            #   this test work the same in whether run by itself or run
            #   in context, but for now I've just commented out the check
            #   on the length of the exposure downstreams.
            exp_downstreams = [ downstream.id for downstream in ds.exposure.get_downstreams(pgdb=pgdb) ]
            # assert len(exp_downstreams) == 2
            assert ds.image.id in exp_downstreams

            assert [downstream.id for downstream in ds.image.get_downstreams(pgdb=pgdb)] == [ds.sources.id]
            assert ( set( downstream.id for downstream in ds.sources.get_downstreams(pgdb=pgdb) )
                     == { ds.wcs.id, ds.bg.id, ds.psf.id } )
            assert [downstream.id for downstream in ds.psf.get_downstreams(pgdb=pgdb)] == []
            assert ( set( downstream.id for downstream in ds.wcs.get_downstreams(pgdb=pgdb) )
                     == { ds.zp.id, ds.aligned_ref_image.id } )
            assert [downstream.id for downstream in ds.zp.get_downstreams(pgdb=pgdb)] == [ds.sub_image.id]
            assert ( [ downstream.id for downstream in ds.aligned_ref_image.get_downstreams(pgdb=pgdb) ]
                     == [ ds.aligned_ref_sources.id ] )
            assert ( set( downstream.id for downstream in ds.aligned_ref_sources.get_downstreams(pgdb=pgdb) )
                     == { ds.aligned_ref_bg.id, ds.aligned_ref_psf.id } )
            assert [downstream.id for downstream in ds.reference.get_downstreams(pgdb=pgdb)] == [ds.sub_image.id]
            assert [downstream.id for downstream in ds.sub_image.get_downstreams(pgdb=pgdb)] == [ds.detections.id]
            assert [downstream.id for downstream in ds.detections.get_downstreams(pgdb=pgdb)] == [ds.cutouts.id]
            assert ( [downstream.id for downstream in ds.cutouts.get_downstreams(pgdb=pgdb)] ==
                     [ds.measurement_set.id] )
            ms_dstrs = set( measurement.id for measurement in ds.measurements )
            ms_dstrs.add( ds.deepscore_set.id )
            assert ( set( downstream.id for downstream in ds.measurement_set.get_downstreams(pgdb=pgdb) )
                     == ms_dstrs )
            assert all ( m.get_downstreams(pgdb=pgdb) == [] for m in ds.measurements )
            ds_dstrs = set( deepscore.id for deepscore in ds.deepscores )
            assert ( set( downstream.id for downstream in ds.deepscore_set.get_downstreams(pgdb=pgdb) )
                     == ds_dstrs )
            assert all( d.get_downstreams(pgdb=pgdb) == [] for d in ds.deepscores )


    finally:
        if 'ds' in locals():
            ds.delete_everything()
        # Clean up the provenance tag created by the pipeline
        with PGDB() as pgdb:
            pgdb.execute( sql.SQL( "DELETE FROM provenance_tags WHERE tag={tag}" )
                          .format( tag='test_get_upstreams_and_downstreams' ) )
            pgdb.commit()
        # added this cleanup to make sure the temp data folder is cleaned up
        # this should be removed after we add datastore failure modes (issue #150)
        shutil.rmtree(os.path.join(os.path.dirname(exposure.get_fullpath()), '115'), ignore_errors=True)
        shutil.rmtree(os.path.join(archive.test_folder_path, '115'), ignore_errors=True)


def test_provenance_tree(pipeline_for_tests, decam_exposure, decam_datastore, decam_reference):
    p = pipeline_for_tests
    p.subtractor.pars.refset = 'test_refset_decam'

    def check_prov_tag( provs, ptagname ):
        with SmartSession() as session:
            ptags = session.query( ProvenanceTag ).filter( ProvenanceTag.tag==ptagname ).all()
        provids = []
        for prov in provs:
            if isinstance( prov, list ):
                provids.extend( [ i.id for i in prov ] )
            else:
                provids.append( prov.id )
        ptagprovids = [ ptag.provenance_id for ptag in ptags ]
        assert all( [ pid in provids for pid in ptagprovids ] )
        assert all( [ pid in ptagprovids for pid in provids ] )
        return ptags

    ds = DataStore( decam_exposure, 'S2' )
    provs = p.make_provenance_tree( ds )
    assert isinstance(provs, dict)
    assert provs == ds.prov_tree

    # Make sure the ProvenanceTag got created properly
    ptags = check_prov_tag( provs.values(), 'pipeline_for_tests' )

    t_start = datetime.datetime.utcnow()
    ds = p.run( ds )    # the data should all be there so this should be quick
    t_end = datetime.datetime.utcnow()

    assert decam_exposure.provenance_id == provs['starting_point'].id
    assert ds.image.provenance_id == provs['preprocessing'].id
    assert ds.sources.provenance_id == provs['extraction'].id
    assert ds.reference.provenance_id == provs['referencing'].id
    assert ds.sub_image.provenance_id == provs['subtraction'].id
    assert ds.detections.provenance_id == provs['detection'].id
    assert ds.cutouts.provenance_id == provs['cutting'].id
    assert ds.measurement_set.provenance_id == provs['measuring'].id
    assert ds.deepscore_set.provenance_id == provs['scoring'].id

    with SmartSession() as session:
        report = session.scalars(
            sa.select(Report).where(Report.exposure_id == decam_exposure.id).order_by(Report.start_time.desc())
        ).first()
        assert report is not None
        assert report.success
        assert abs(report.start_time - t_start) < datetime.timedelta(seconds=1)
        assert abs(report.finish_time - t_end) < datetime.timedelta(seconds=1)

    # Make sure that the provenance tags are reused if we ask for the same thing
    newprovs = p.make_provenance_tree( ds )
    provids = []
    for prov in provs.values():
        if isinstance( prov, list ):
            provids.extend( [ i.id for i in prov ] )
        else:
            provids.append( prov.id )
    newprovids = []
    for prov in newprovs.values():
        if isinstance( prov, list ):
            newprovids.extend( [ i.id for i in prov ] )
        else:
            newprovids.append( prov.id )
    assert set( newprovids ) == set( provids )
    newptags = check_prov_tag( newprovs.values(), 'pipeline_for_tests' )
    assert set( [ i.id for i in newptags ] ) == set( [ i.id for i in ptags ] )

    # Make sure that we get an exception if we ask for a mismatched provenance tree
    # Do this by creating a new pipeline with inconsistent parameters but asking
    # for the same provenance tag.
    extrsubconfigs = copy.deepcopy( p.extractor.pars.subconfigs )
    extrsubconfigs['subconfigs']['extragalactic']['snr_threshold'] = 42.
    newp = Pipeline( pipeline={'provenance_tag': 'pipeline_for_tests'}, extraction={'subconfigs': extrsubconfigs} )
    with pytest.raises( RuntimeError,
                        match=( 'The following provenances do not match the existing provenance '
                                'for tag pipeline_for_tests' ) ):
        newp.make_provenance_tree( ds )


def test_inject_warnings_errors( sim_lightcurve_one_complete_ds ):
    ref, refds, ds, p = sim_lightcurve_one_complete_ds
    p.subtractor.pars.refset = 'sim_lightcurve_reference'

    try:
        # This next dict and the code that uses it took me a while to
        #   get right, so I'm writing the convoluted trail down here in
        #   case we ever come back and have to think about it again.
        #   The goal is reconstruct what text shows up in the warning
        #   recorded by the report.  In pipeline/parameters.py::
        #   Parameters.do_warning_exception_hangup_injection_here, there
        #   is a warnings.warn("...{self.get_process_name()}").  The
        #   warnings get added to the report when
        #   DataStore.update_report is called, whose firist parameter is
        #   process_step; this calls Report.scan_datastore, passing
        #   along process_step.  That method sets the warnings field of
        #   the report to a string via read_warnings, where each line of
        #   the warning starts with process_step, and has other stuff
        #   after that.  process_step is originally set in
        #   top_level.py::Pipeline.run when it calls
        #   DataStore.update_report after each step.

        # All of which would be fine for human consumption, but now we
        #   want to write a for loop to check that the right warnings showed up.
        #   This dictionary reproduces the process_step values used in
        #   top_level.py

        obj_to_process_step = {
            # This test no longer works on preprocessing because
            #   top_level.py skips running the preprocessor if the
            #   datastore already has an image and that image's
            #   preproc_bitflag matches what the preprocessor would have
            #   set.  (Search for comment "SPECIAL CASE" in
            #   top_level.py.)
            #   (This was also true before I changed the fixture from
            #   decam to sim_lightcurve, as the decam fixture also has a
            #   fully preprocessed image.)
            # 'preprocessor': 'preprocessing',
            'extractor': 'extraction',
            'astrometor': 'astrocal',
            'photometor': 'photocal',
            'subtractor': 'subtraction',
            'detector': 'detection',
            'cutter': 'cutting',
            'measurer': 'measuring',
            'scorer': 'scoring'
        }

        for obj, process in obj_to_process_step.items():
            # first reset all warnings and errors
            for obj2 in obj_to_process_step.keys():
                getattr(p, obj2).pars.inject_exceptions = False
                getattr(p, obj2).pars.inject_warnings = False

            process_name = getattr( p, obj ).pars.get_process_name()
            process_step = obj_to_process_step[ obj ]

            if not SKIP_WARNING_TESTS:
                # set the warning:
                getattr(p, obj).pars.inject_warnings = True

                # run the pipeline
                ds = p.run( ds )
                expected = ( f"{process_step}: <class 'UserWarning'> Warning injected by pipeline parameters "
                             f"in process '{process_name}'" )
                assert expected in ds.report.warnings
                # NOTE -- should really add a test that there are no other "Warning injected"
                #   lines.  The report should be this separated by ...***... lines.

            # these are used to find the report later on
            exp_id = ds.exposure_id
            sec_id = ds.section_id

            # set the error instead
            getattr(p, obj).pars.inject_warnings = False
            getattr(p, obj).pars.inject_exceptions = True
            # run the pipeline again, this time with an exception

            with pytest.raises( RuntimeError,
                                match=f"Exception injected by pipeline parameters in process '{process_name}'" ):
                ds = p.run( ds )

            # fetch the report object
            ds.update_report( process_step )
            with SmartSession() as session:
                reports = session.scalars(
                    sa.select(Report).where(
                        Report.exposure_id == exp_id,
                        Report.section_id == sec_id,
                    ).order_by(Report.start_time.desc())
                ).all()
                report = reports[0]  # the last report is the one we just generated
                assert not report.success
                assert report.error_step == process_step
                assert report.error_type == 'RuntimeError'
                assert 'Exception injected by pipeline parameters' in report.error_message

    finally:
        if 'ds' in locals():
            ds.delete_everything()


def test_multiprocessing_make_provenances_and_exposure(decam_exposure, decam_reference, pipeline_for_tests):
    from multiprocessing import SimpleQueue, Process
    process_list = []
    pipeline_for_tests.subtractor.pars.refset = 'test_refset_decam'

    def make_provenances(exposure, pipeline, queue):
        ds = DataStore( exposure, 'S2' )
        provs = pipeline.make_provenance_tree( ds )
        queue.put(provs)

    queue = SimpleQueue()
    for i in range(3):  # github has 4 CPUs for testing, so 3 sub-processes and 1 main process
        p = Process(target=make_provenances, args=(decam_exposure, pipeline_for_tests, queue))
        p.start()
        process_list.append(p)

    # also run this on the main process
    ds = DataStore( decam_exposure, 'S2' )
    provs = pipeline_for_tests.make_provenance_tree( ds )

    for p in process_list:
        p.join()
        assert not p.exitcode

    # check that the provenances are the same
    for _ in process_list:  # order is not kept but all outputs should be the same
        output_provs = queue.get()
        assert output_provs['measuring'].id == provs['measuring'].id
