import os
import pytest
import shutil
import datetime

import numpy as np

import sqlalchemy as sa
import sqlalchemy.orm as orm

from models.base import SmartSession, FileOnDiskMixin
from models.provenance import Provenance, ProvenanceTag
from models.exposure import Exposure
from models.image import Image
from models.reference import image_subtraction_components
from models.calibratorfile import CalibratorFile
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements, MeasurementSet
from models.deepscore import DeepScore, DeepScoreSet
from models.report import Report

from pipeline.data_store import DataStore
from pipeline.top_level import Pipeline

from util.logger import SCLogger
from util.util import env_as_bool

from tests.conftest import SKIP_WARNING_TESTS


def check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, ds):
    """Check that all the required objects are saved on the database and in the datastore.

    (After running the entire pipeline.)

    Parameters
    ----------
    exp_id: int
        The Exposure ID.

    sec_id: str or int
        The section_id of the image from the exposure.

    ref_id: int
        The Reference ID.

    session: sqlalchemy.orm.session.Session
        The database session

    ds: datastore.DataStore
        The datastore object

    """

    with SmartSession() as session:
        # find the image
        im = session.scalars(
            sa.select(Image).where(
                Image.exposure_id == exp_id,
                Image.section_id == str(sec_id),
                Image.provenance_id == ds.image.provenance_id,
            )
        ).first()
        assert im is not None
        assert ds.image.id == im.id

        # find the extracted sources
        sources = session.scalars(
            sa.select(SourceList).where(
                SourceList.image_id == im.id,
                SourceList.provenance_id == ds.sources.provenance_id,
            )
        ).first()
        assert sources is not None
        assert ds.sources.id == sources.id

        # find the PSF
        psf = session.scalars( sa.select(PSF).where(PSF.sources_id == sources.id) ).first()
        assert psf is not None
        assert ds.psf.id == psf.id

        # find the WorldCoordinates object
        wcs = session.scalars( sa.select(WorldCoordinates).where(WorldCoordinates.sources_id == sources.id) ).first()
        assert wcs is not None
        assert ds.wcs.id == wcs.id

        # find the ZeroPoint object
        zp = session.scalars( sa.select(ZeroPoint).where(ZeroPoint.wcs_id == wcs.id) ).first()
        assert zp is not None
        assert ds.zp.id == zp.id

        # find the subtraction image
        sub = ( session.query( Image )
                .join( image_subtraction_components,
                       sa.and_( image_subtraction_components.c.image_id==Image._id,
                                image_subtraction_components.c.ref_id==ref_id ) )
                .filter( image_subtraction_components.c.new_zp_id==zp._id ) ).first()
        assert sub is not None
        assert ds.sub_image.id == sub.id

        # find the detections SourceList
        det = session.scalars(
            sa.select(SourceList).where(
                SourceList.image_id == sub.id,
                SourceList.provenance_id == ds.detections.provenance_id,
            )
        ).first()

        assert det is not None
        assert ds.detections.id == det.id

        # find the Cutouts
        cutouts = session.scalars(
            sa.select(Cutouts).where(
                Cutouts.sources_id == det.id,
                Cutouts.provenance_id == ds.cutouts.provenance_id,
            )
        ).first()
        assert ds.cutouts.id == cutouts.id

        # Measurements
        measurement_set = session.scalars( sa.select( MeasurementSet )
                                           .where( MeasurementSet.cutouts_id == cutouts.id,
                                                   MeasurementSet.provenance_id == ds.measurement_set.provenance_id )
                                          ).first()
        assert ds.measurement_set.id == measurement_set.id

        measurements = session.scalars( sa.select( Measurements )
                                        .where( Measurements.measurementset_id == measurement_set.id )
                                        .order_by( Measurements.index_in_sources )
                                       ).all()
        assert len(measurements) > 0
        assert len(ds.measurements) == len(measurements)
        assert all( ds.measurements[i].id == measurements[i].id for i in range(len(measurements)) )

        # deepscores
        deepscore_set = session.scalars( sa.select( DeepScoreSet )
                                         .where( DeepScoreSet.measurementset_id == measurement_set.id,
                                                 DeepScoreSet.provenance_id == ds.deepscore_set.provenance_id )
                                        ).first()
        assert ds.deepscore_set.id == deepscore_set.id

        deepscores = session.scalars( sa.select( DeepScore )
                                      .where( DeepScore.deepscoreset_id == deepscore_set.id )
                                      .order_by( DeepScore.index_in_sources )
                                     ).all()
        assert len(deepscores) == len(measurements)
        assert all( d.index_in_sources == m.index_in_sources for d, m in zip( deepscores, measurements ) )
        assert len(deepscores) == len(ds.deepscores)
        assert all( d.id == dsd.id for d, dsd in zip( deepscores, ds.deepscores ) )


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
        'extraction': {'threshold': 3.14 },
        'astrocal': {'cross_match_catalog': 'override'},
        'photocal': {'cross_match_catalog': 'override'},
        'subtraction': { 'method': 'override' },
        'detection': { 'threshold': 3.14 },
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

    with SmartSession() as session:
        # The N1 decam calibrator files will have been automatically added
        # in the pipeline run above; need to clean them up.  However,
        # *don't* remove the linearity calibrator file, because that will
        # have been added in session fixtures used by other tests.  (Tests
        # and automatic cleanup become very fraught when you have automatic
        # loading of stuff....)

        cfs = ( session.query( CalibratorFile )
                .filter( CalibratorFile.instrument == 'DECam' )
                .filter( CalibratorFile.sensor_section == 'N1' )
                .filter( CalibratorFile.image_id is not None ) )
        imdel = [ c.image_id for c in cfs ]
        imgtodel = session.query( Image ).filter( Image._id.in_( imdel ) )
        for i in imgtodel:
            i.delete_from_disk_and_database()

        session.commit()


def check_full_run_results( ds, exposure, sec_id, ref, expected ):
    # Make sure that the provenances are all in the database
    with SmartSession() as session:
        for prov in ds.prov_tree.values():
            provs = session.query( Provenance ).filter( Provenance._id==prov.id ).all()
            assert len(provs) == 1

    # Check that all the data products are in the database
    check_datastore_and_database_have_everything( exposure.id, sec_id, ref.id, ds )

    # Get the measurements and deepscores.
    # Filter only on the DeepScoreSet provenance since that's the lowest
    #   one down, so by going at upstreams from the DeepScoreSet we know
    #   we're getting the right things.
    # (This is perhaps gratuitous, since the measurements and deescores
    #   should aready be in DataStore ds, but, well, I guess this is
    #   also checking it all got saved right to the database.)
    detections = orm.aliased( SourceList )
    subim = orm.aliased( Image )
    with SmartSession() as session:
        res = ( session.query( MeasurementSet, DeepScoreSet )
                .join( Cutouts, MeasurementSet.cutouts_id==Cutouts._id )
                .join( detections, Cutouts.sources_id==detections._id )
                .join( subim, detections.image_id==subim._id )
                .join( image_subtraction_components, subim._id==image_subtraction_components.c.image_id )
                .join( ZeroPoint, image_subtraction_components.c.new_zp_id==ZeroPoint._id )
                .join( WorldCoordinates, ZeroPoint.wcs_id==WorldCoordinates._id )
                .join( SourceList, WorldCoordinates.sources_id==SourceList._id )
                .join( Image, SourceList.image_id==Image._id )
                .filter( DeepScoreSet.measurementset_id==MeasurementSet._id )
                .filter( DeepScoreSet.provenance_id==ds.prov_tree['scoring'].id )
                .filter( Image.exposure_id==exposure.id )
               ).all()
        assert len(res) == 1
        meas, deep = res[0]

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
    for m, d in zip( meas.measurements, deep.deepscores ):
        # We don't care about things being in the same order, but the set of
        #   measurements and deepscores should match.  Find the match based
        #   on x and y being both within 0.05 of what's expected.
        w = np.where( ( np.fabs( expected['x'] - m.x ) < 0.05 ) & ( np.fabs( expected['y'] - m.y ) < 0.05 ) )[0]
        assert len(w) == 1
        i = w[0]
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
        assert expected['rb'][i] == pytest.approx( d.score, abs=0.1 )


# The user fixture is here because this is a convenient test for looking
#  to see what we've got on the weabp.  This will only work if you're
#  running the tests on the dekstop or laptop you're sitting at.  Put a
#  breakpoint at the end of the test, and then go log into the webap
#  (which will be at https://localhost:8081, with 8081 replaced with the
#  value of WEBAP_PORT If you set that when running the docker compose
#  file), login in with username "test" and password "test_password",
#  and check things out.
def test_full_run_zogy( decam_exposure, decam_reference, decam_default_calibrators, user ):
    # THis source at 1618.23, 1880,40 is a real SN
    expected = {
        'x':      np.array( [ 1407.86, 1451.27, 1450.29, 1618.23, 1440.11, 1357.60,  457.35, 1516.72 ] ),
        'y':      np.array( [  756.24,  805.64, 1626.01, 1880.40, 2047.02, 3493.38, 4007.04, 4042.66 ] ),
        'gfit_x':           [ 1407.53, 1450.10, 1449.61, 1618.22, 1439.95, 1357.17,  457.15, 1515.51 ],
        'gfit_y':           [  756.24,  808.16, 1625.93, 1880.29, 2046.99, 3491.08, 4007.34, 4040.48 ],
        'major_width': [ 7.00, 8.99, 6.12, 4.04, 4.04, 9.76, 4.57, 8.07 ],
        'minor_width': [ 4.20, 7.11, 3.38, 3.10, 2.16, 4.25, 2.48, 7.66 ],
        'neg_frac':      [ 0.27, 0.12, 0.11, 0.07, 0.28, 0.12, 0.21, 0.03 ],
        'neg_flux_frac': [ 0.10, 0.12, 0.12, 0.03, 0.27, 0.10, 0.17, 0.05 ],
        'psf_flux':      [ 78176., 8227., 9229., 2249., 835., 810., 3121., 21820. ],
        'psf_flux_err':  [   390.,  146.,  145.,   96., 101.,  87.,  143.,   281. ],
        'aper_flux':     [ 62767., 9852., 8466., 1728., 534., 987., 2447., 24947. ],
        'aper_flux_err': [   559.,  207.,  188.,  106., 124., 101.,  148.,   266. ],
        'rb': [ 0.420, 0.463, 0.507, 0.734, 0.624, 0.498, 0.595, 0.410 ]
    }

    try:
        # subtraction.method zogy and detection.method filter should already
        #   be in the defaults from the config file, but be explicit
        #   here for clarity (and comparison to test_full_run_hotpants)
        pipeline = Pipeline( subtraction={ 'method': 'zogy',
                                           'refset': 'test_refset_decam' },
                             detection={ 'method': 'filter' } )
        ds = pipeline.run( decam_exposure, decam_reference.image.section_id )
        ds.save_and_commit()
        check_full_run_results( ds, decam_exposure, decam_reference.image.section_id, decam_reference, expected )

    finally:
        if 'ds' in locals():
            ds.delete_everything()


# See comment on test_full_run_zogy for the reason for the user fixture
def test_full_run_hotpants( decam_exposure, decam_reference, decam_default_calibrators, user ):
    # A lot of the stuff that pass the cuts is really bad -- lots of CRs getting through.
    # I think they didn't get through with zogy because zogy effectively convolved them out
    # a bit (as the ref had worse seeing the the new here), and the blurring led to
    # Gaussian fits that did a better job of seeing very non-round things.  Perhaps thought
    # required on our analytic cuts.  But, also, R/B really needs to be trained better to
    # get rid of all of that, as a lot of these 26 are junk with high R/B.
    #
    # The source at 1619.24, 1881.37 is a real SN.  I am distressed that the position is 1 off
    # from the zogy positions.  This is probably a bug in reading sextractor positions.
    # See Issue #462
    #
    # ...visibly, I think the 1619.24, 1881.37 position is more correct!  So is the problem
    # somewhere in zogy???
    # ...yes!  Visual inspection shows that the residual in the difference image is shifted
    # down and to the left from the search image!

    expected = {
        'x': np.array( [  593.75, 1358.50, 1358.13, 1265.51,  949.14,  903.47, 1752.72,  998.42,  983.28,
                          1985.58, 1619.21, 1581.49, 1451.51, 1451.43, 1143.22, 1435.00, 1441.26, 1438.02,
                          1185.98,  246.24, 1451.70, 1451.92, 1408.90, 1435.25, 1576.60, 1573.30 ] ),
        'y': np.array( [ 3527.50, 3494.76, 3492.19, 3409.27, 2847.72, 2747.91, 2620.64, 2278.12, 2273.60,
                         2026.23, 1881.41, 1665.81, 1632.57, 1627.75, 1571.47, 1464.65, 1466.73, 1465.25,
                         1085.23,  933.89,  812.72,  809.64,  758.32,  477.55,  181.67,  180.33 ] ),
        'gfit_x':      [  593.56, 1358.18, 1358.18, 1265.73,  949.17,  903.51, 1752.83,  998.43,  983.31,
                         1987.86, 1619.22, 1581.51, 1450.86, 1450.86, 1143.42, 1438.10, 1438.10, 1438.10,
                         1186.01,  246.33, 1451.50, 1451.50, 1408.65, 1435.30, 1576.84, 1575.26 ],
        'gfit_y':      [ 3527.32, 3492.07, 3492.07, 3409.50, 2847.63, 2747.88, 2620.62, 2278.16, 2273.17,
                         2026.20, 1881.32, 1665.88, 1628.53, 1628.53, 1571.61, 1465.25, 1465.25, 1465.25,
                         1085.21,  933.92,  809.51,  809.51,  758.29,  477.70,  181.84,  180.84 ],
        'major_width': [ 1.53, 9.73, 9.73, 4.68, 3.70, 2.19, 2.48, 3.53, 2.51,11.50, 3.91, 3.01, 7.61,
                         7.60, 1.84, 8.30, 8.30, 8.30, 2.34, 2.08, 8.18, 8.18, 7.21, 1.40, 2.74, 6.30 ],
        'minor_width': [ 0.58, 4.16, 4.16, 1.81, 1.41, 1.37, 1.56, 1.21, 1.04, 4.36, 3.18, 1.51, 6.78,
                         6.78, 0.68, 7.47, 7.47, 7.47, 1.67, 1.53, 7.24, 7.24, 5.74, 0.96, 1.74, 2.94 ],
        'neg_frac':      [ 0.227, 0.192, 0.179, 0.233, 0.152, 0.149, 0.133, 0.220, 0.116, 0.207, 0.095,
                           0.106, 0.125, 0.091, 0.235, 0.064, 0.114, 0.077, 0.140, 0.214, 0.119, 0.109,
                           0.173, 0.219, 0.059, 0.067 ],
        'neg_flux_frac': [ 0.019, 0.158, 0.146, 0.019, 0.019, 0.020, 0.010, 0.031, 0.014, 0.214, 0.060,
                           0.016, 0.053, 0.046, 0.021, 0.046, 0.066, 0.051, 0.021, 0.017, 0.045, 0.044,
                           0.067, 0.016, 0.004, 0.005 ],
        'psf_flux':      [   4039,    786,    818,  12910,   7402,   8255,  13014,   6465,   5639,   1103,
                             2336,   8753,   4753,   9514,   6368,   2136,   1982,   3768,   8406,   9597,
                             7772,  12470, 101766,   8022,  16498,  14251 ],
        'psf_flux_err':  [     95,     86,     86,    113,    104,    106,    116,    102,     99,    107,
                               96,    108,    124,    177,    104,    112,    103,    152,    108,    111,
                              146,    219,    468,    109,    121,    116 ],
        'aper_flux':     [   4703,    867,   1055,  13486,   5638,   6608,  11533,   5390,   5387,   1105,
                             1901,   6789,   4690,   9141,   5909,   3242,   2420,   3661,   6383,   7292,
                             7295,  10623,  82179,   9113,  16071,  14676 ],
        'aper_flux_err': [    105,    100,    100,    115,    106,    107,    113,    105,    105,    119,
                              105,    108,    151,    173,    107,    134,    124,    140,    107,    108,
                              160,    192,    440,    110,    118,    116 ],
        'rb': [ 0.637, 0.519, 0.404, 0.790, 0.779, 0.889, 0.852, 0.602, 0.614, 0.782, 0.780, 0.774, 0.551,
                0.516, 0.837, 0.561, 0.537, 0.716, 0.840, 0.809, 0.601, 0.527, 0.537, 0.812, 0.815, 0.325 ]
    }

    try:
        pipeline = Pipeline( subtraction={ 'method': 'hotpants',
                                           'refset': 'test_refset_decam' },
                             detection={ 'method': 'sextractor' } )
        ds = pipeline.run( decam_exposure, decam_reference.image.section_id )
        ds.save_and_commit()
        check_full_run_results( ds, decam_exposure, decam_reference.image.section_id, decam_reference, expected )

    finally:
        if 'ds' in locals():
            ds.delete_everything()


@pytest.mark.skipif( not env_as_bool('RUN_SLOW_TESTS') )
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
        assert p.extractor.pars.threshold != 3.14
        assert p.detector.pars.threshold != 3.14

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
        # Gotta do the sources siblings individually,
        #   but doing those will catch everything else
        #   with remove_downstreams defaulting to True
        ds.bg.delete_from_disk_and_database()
        ds.bg = None
        ds.wcs.delete_from_disk_and_database()
        ds.wcs = None
        ds.zp.delete_from_disk_and_database()
        ds.zp = None

        ds.sub_image = None
        ds.detections = None
        ds.cutouts = None
        ds.measurement_set = None
        ds.deepscore_set = None

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
        with SmartSession() as session:
            # test get_upstreams()
            assert ds.exposure.get_upstreams() == []
            assert [upstream.id for upstream in ds.image.get_upstreams(session=session)] == [ds.exposure.id]
            assert [upstream.id for upstream in ds.sources.get_upstreams(session=session)] == [ds.image.id]
            assert [upstream.id for upstream in ds.wcs.get_upstreams(session=session)] == [ds.sources.id]
            assert [upstream.id for upstream in ds.psf.get_upstreams(session=session)] == [ds.sources.id]
            assert [upstream.id for upstream in ds.zp.get_upstreams(session=session)] == [ds.wcs.id]
            assert ( set([ upstream.id for upstream in ds.sub_image.get_upstreams( session=session ) ])
                     == { ds.reference.id, ds.zp.id } )
            assert [upstream.id for upstream in ds.detections.get_upstreams(session=session)] == [ds.sub_image.id]
            assert [upstream.id for upstream in ds.cutouts.get_upstreams(session=session)] == [ds.detections.id]
            assert [upstream.id for upstream in ds.measurement_set.get_upstreams(session=session)] == [ds.cutouts.id]
            for measurement in ds.measurements:
                assert ( [upstream.id for upstream in measurement.get_upstreams(session=session)]
                         == [ds.measurement_set.id] )
            assert ( [upstream.id for upstream in ds.deepscore_set.get_upstreams(session=session)]
                     == [ds.measurement_set.id] )
            for deepscore in ds.deepscores:
                assert ( [upstream.id for upstream in deepscore.get_upstreams(session=session)]
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
            #   SQLAlchemy sessions.  Making decam_exposure a
            #   function-scope fixture (rather than the session-scope
            #   fixture it is right now) would almost certainly make
            #   this test work the same in whether run by itself or run
            #   in context, but for now I've just commented out the check
            #   on the length of the exposure downstreams.
            exp_downstreams = [ downstream.id for downstream in ds.exposure.get_downstreams(session=session) ]
            # assert len(exp_downstreams) == 2
            assert ds.image.id in exp_downstreams

            assert [downstream.id for downstream in ds.image.get_downstreams(session=session)] == [ds.sources.id]
            assert ( set( [downstream.id for downstream in ds.sources.get_downstreams(session=session)] )
                     == { ds.wcs.id, ds.bg.id, ds.psf.id } )
            assert [downstream.id for downstream in ds.psf.get_downstreams(session=session)] == []
            assert [downstream.id for downstream in ds.wcs.get_downstreams(session=session)] == [ds.zp.id]
            assert [downstream.id for downstream in ds.zp.get_downstreams(session=session)] == [ds.sub_image.id]
            assert [downstream.id for downstream in ds.reference.get_downstreams(session=session)] == [ds.sub_image.id]
            assert [downstream.id for downstream in ds.sub_image.get_downstreams(session=session)] == [ds.detections.id]
            assert [downstream.id for downstream in ds.detections.get_downstreams(session=session)] == [ds.cutouts.id]
            assert ( [downstream.id for downstream in ds.cutouts.get_downstreams(session=session)] ==
                     [ds.measurement_set.id] )
            ms_dstrs = set( measurement.id for measurement in ds.measurements )
            ms_dstrs.add( ds.deepscore_set.id )
            assert ( set( downstream.id for downstream in ds.measurement_set.get_downstreams(session=session) )
                     == ms_dstrs )
            assert all ( m.get_downstreams(session=session) == [] for m in ds.measurements )
            ds_dstrs = set( deepscore.id for deepscore in ds.deepscores )
            assert ( set( downstream.id for downstream in ds.deepscore_set.get_downstreams(session=session) )
                     == ds_dstrs )
            assert all( d.get_downstreams(session=session) == [] for d in ds.deepscores )


    finally:
        if 'ds' in locals():
            ds.delete_everything()
        # Clean up the provenance tag created by the pipeline
        with SmartSession() as session:
            session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ),
                            { 'tag': 'test_get_upstreams_and_downstreams' } )
            session.commit()
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
    newp = Pipeline( pipeline={'provenance_tag': 'pipeline_for_tests'},
                     extraction={ 'threshold': 42. } )
    with pytest.raises( RuntimeError,
                        match=( 'The following provenances do not match the existing provenance '
                                'for tag pipeline_for_tests' ) ):
        newp.make_provenance_tree( ds )


# This test is really slow because it runs the pipeline repeatedly to test
#   warnings and exceptions at each step.
@pytest.mark.skipif( not env_as_bool('RUN_SLOW_TESTS'), reason="Set RUN_SLOW_TESTS to run this test" )
def test_inject_warnings_errors(decam_datastore, decam_reference, pipeline_for_tests):
    p = pipeline_for_tests
    p.subtractor.pars.refset = 'test_refset_decam'

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
            'preprocessor': 'preprocessing',
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
                ds = p.run(decam_datastore)
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
                ds = p.run(decam_datastore)

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
