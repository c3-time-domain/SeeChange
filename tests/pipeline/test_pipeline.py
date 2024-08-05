import os
import pytest
import shutil
import datetime

import sqlalchemy as sa
import numpy as np

from models.base import SmartSession, FileOnDiskMixin
from models.provenance import Provenance, ProvenanceTag
from models.image import Image, image_upstreams_association_table
from models.calibratorfile import CalibratorFile
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.report import Report

from pipeline.top_level import Pipeline

from tests.conftest import SKIP_WARNING_TESTS


def check_datastore_and_database_have_everything(exp_id, sec_id, ref_id, session, ds):
    """
    Check that all the required objects are saved on the database
    and in the datastore, after running the entire pipeline.

    Parameters
    ----------
    exp_id: int
        The exposure ID.
    sec_id: str or int
        The section ID.
    ref_id: int
        The reference image ID.
    session: sqlalchemy.orm.session.Session
        The database session
    ds: datastore.DataStore
        The datastore object
    """
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
            SourceList.is_sub.is_(False),
            SourceList.provenance_id == ds.sources.provenance_id,
        )
    ).first()
    assert sources is not None
    assert ds.sources.id == sources.id

    # find the PSF
    psf = session.scalars(
        sa.select(PSF).where(PSF.image_id == im.id, PSF.provenance_id == ds.psf.provenance_id)
    ).first()
    assert psf is not None
    assert ds.psf.id == psf.id

    # find the WorldCoordinates object
    wcs = session.scalars(
        sa.select(WorldCoordinates).where(
            WorldCoordinates.sources_id == sources.id,
            WorldCoordinates.provenance_id == ds.wcs.provenance_id,
        )
    ).first()
    assert wcs is not None
    assert ds.wcs.id == wcs.id

    # find the ZeroPoint object
    zp = session.scalars(
        sa.select(ZeroPoint).where(ZeroPoint.sources_id == sources.id, ZeroPoint.provenance_id == ds.zp.provenance_id)
    ).first()
    assert zp is not None
    assert ds.zp.id == zp.id

    # find the subtraction image
    aliased_table = sa.orm.aliased(image_upstreams_association_table)
    sub = session.scalars(
        sa.select(Image).join(
            image_upstreams_association_table,
            sa.and_(
                image_upstreams_association_table.c.upstream_id == ref_id,
                image_upstreams_association_table.c.downstream_id == Image.id,
            )
        ).join(
            aliased_table,
            sa.and_(
                aliased_table.c.upstream_id == im.id,
                aliased_table.c.downstream_id == Image.id,
            )
        )
    ).first()

    assert sub is not None
    assert ds.sub_image.id == sub.id

    # find the detections SourceList
    det = session.scalars(
        sa.select(SourceList).where(
            SourceList.image_id == sub.id,
            SourceList.is_sub.is_(True),
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
    measurements = session.scalars(
        sa.select(Measurements).where(
            Measurements.cutouts_id == cutouts.id,
            Measurements.provenance_id == ds.measurements[0].provenance_id,
        )
    ).all()
    assert len(measurements) > 0
    assert len(ds.measurements) == len(measurements)


def test_parameters( test_config ):
    """Test that pipeline parameters are being set properly"""

    # Verify that we _enforce_no_new_attrs works
    kwargs = { 'pipeline': { 'keyword_does_not_exist': 'testing' } }
    with pytest.raises( AttributeError, match='object has no attribute' ):
        failed = Pipeline( **kwargs )

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
        'extraction': {
            'sources': {'threshold': 3.14 },
            'wcs': {'cross_match_catalog': 'override'},
            'zp': {'cross_match_catalog': 'override'},
        },
        'subtraction': { 'method': 'override' },
        'detection': { 'threshold': 3.14 },
        'cutting': { 'cutout_size': 666 },
        'measuring': { 'outlier_sigma': 3.5 }
    }

    def check_override( new_values_dict, pars ):
        for key, value in new_values_dict.items():
            if pars[key] != value:
                return False
        return True

    pipeline = Pipeline( **overrides )

    assert check_override(overrides['preprocessing'], pipeline.preprocessor.pars)
    assert check_override(overrides['extraction']['sources'], pipeline.extractor.pars)
    assert check_override(overrides['extraction']['wcs'], pipeline.astrometor.pars)
    assert check_override(overrides['extraction']['zp'], pipeline.photometor.pars)
    assert check_override(overrides['subtraction'], pipeline.subtractor.pars)
    assert check_override(overrides['detection'], pipeline.detector.pars)
    assert check_override(overrides['cutting'], pipeline.cutter.pars)
    assert check_override(overrides['measuring'], pipeline.measurer.pars)


def test_running_without_reference(decam_exposure, decam_refset, decam_default_calibrators, pipeline_for_tests):
    p = pipeline_for_tests
    p.subtractor.pars.refset = 'test_refset_decam'  # choosing ref set doesn't mean we have an actual reference
    p.pars.save_before_subtraction = True  # need this so images get saved even though it crashes on "no reference"

    with pytest.raises(ValueError, match='Cannot find a reference image corresponding to.*'):
        # Use the 'N1' sensor section since that's not one of the ones used in the regular
        #  DECam fixtures, so we don't have to worry about any session scope fixtures that
        #  load refererences.  (Though I don't think there are any.)
        ds = p.run(decam_exposure, 'N1')
        ds.reraise()

    # make sure the data is saved, but then clean it up
    with SmartSession() as session:
        im = session.scalars(sa.select(Image).where(Image.id == ds.image.id)).first()
        assert im is not None
        im.delete_from_disk_and_database( remove_downstreams=True, session=session )

        # The N1 decam calibrator files will have been automatically added
        # in the pipeline run above; need to clean them up.  However,
        # *don't* remove the linearity calibrator file, because that will
        # have been added in session fixtures used by other tests.  (Tests
        # and automatic cleanup become very fraught when you have automatic
        # loading of stuff....)

        cfs = ( session.query( CalibratorFile )
                .filter( CalibratorFile.instrument == 'DECam' )
                .filter( CalibratorFile.sensor_section == 'N1' )
                .filter( CalibratorFile.image_id != None ) )
        imdel = [ c.image_id for c in cfs ]
        imgtodel = session.query( Image ).filter( Image.id.in_( imdel ) )
        for i in imgtodel:
            i.delete_from_disk_and_database( session=session )

        session.commit()

def test_data_flow(decam_exposure, decam_reference, decam_default_calibrators, pipeline_for_tests, archive):
    """Test that the pipeline runs end-to-end."""
    exposure = decam_exposure

    ref = decam_reference
    sec_id = ref.section_id
    try:  # cleanup the file at the end
        p = pipeline_for_tests
        p.subtractor.pars.refset = 'test_refset_decam'
        assert p.extractor.pars.threshold != 3.14
        assert p.detector.pars.threshold != 3.14

        ds = p.run(exposure, sec_id)

        # commit to DB using this session
        with SmartSession() as session:
            ds.save_and_commit(session=session)

        # use a new session to query for the results
        with SmartSession() as session:
            # check that everything is in the database
            provs = session.scalars(sa.select(Provenance)).all()
            assert len(provs) > 0
            prov_processes = [p.process for p in provs]
            expected_processes = ['preprocessing', 'extraction', 'subtraction', 'detection', 'cutting', 'measuring']
            for process in expected_processes:
                assert process in prov_processes

            check_datastore_and_database_have_everything(exposure.id, sec_id, ref.image.id, session, ds)

        # feed the pipeline the same data, but missing the upstream data.
        attributes = ['image', 'sources', 'sub_image', 'detections', 'cutouts', 'measurements', 'scores']

        for i in range(len(attributes)):
            for j in range(i + 1):
                setattr(ds, attributes[j], None)  # get rid of all data up to the current attribute
            # SCLogger.debug(f'removing attributes up to {attributes[i]}')
            ds = p.run(ds)  # for each iteration, we should be able to recreate the data

            # commit to DB using this session
            with SmartSession() as session:
                ds.save_and_commit(session=session)

            # use a new session to query for the results
            with SmartSession() as session:
                check_datastore_and_database_have_everything(exposure.id, sec_id, ref.image.id, session, ds)

        # make sure we can remove the data from the end to the beginning and recreate it
        for i in range(len(attributes)):
            for j in range(i):
                obj = getattr(ds, attributes[-j-1])
                if isinstance(obj, FileOnDiskMixin):
                    obj.delete_from_disk_and_database(session=session, commit=True)

                setattr(ds, attributes[-j-1], None)

            ds = p.run(ds)  # for each iteration, we should be able to recreate the data

            # commit to DB using this session
            with SmartSession() as session:
                ds.save_and_commit(session=session)

            # use a new session to query for the results
            with SmartSession() as session:
                check_datastore_and_database_have_everything(exposure.id, sec_id, ref.image.id, session, ds)

    finally:
        if 'ds' in locals():
            ds.delete_everything()
        # added this cleanup to make sure the temp data folder is cleaned up
        # this should be removed after we add datastore failure modes (issue #150)
        shutil.rmtree(os.path.join(os.path.dirname(exposure.get_fullpath()), '115'), ignore_errors=True)
        shutil.rmtree(os.path.join(archive.test_folder_path, '115'), ignore_errors=True)


def test_bitflag_propagation(decam_exposure, decam_reference, decam_default_calibrators, pipeline_for_tests, archive):
    """
    Test that adding a bitflag to the exposure propagates to all downstreams as they are created
    Does not check measurements, as they do not have the HasBitflagBadness Mixin.
    """
    exposure = decam_exposure
    ref = decam_reference
    sec_id = ref.section_id

    try:  # cleanup the file at the end
        p = Pipeline( pipeline={'provenance_tag': 'test_bitflag_propagation'} )
        p.subtractor.pars.refset = 'test_refset_decam'
        p.pars.save_before_subtraction = False
        exposure.badness = 'banding'  # add a bitflag to check for propagation

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
        for m in ds.measurements:
            assert m._upstream_bitflag == 2

        # test part 2: Add a second bitflag partway through and check it propagates to downstreams

        # delete downstreams of ds.sources
        ds.bg = None
        ds.wcs = None
        ds.zp = None
        ds.sub_image = None
        ds.detections = None
        ds.cutouts = None
        ds.measurements = None

        ds.sources._bitflag = 2 ** 17  # bitflag 2**17 is 'many sources'
        desired_bitflag = 2 ** 1 + 2 ** 17  # bitflag for 'banding' and 'many sources'
        ds = p.run(ds)

        assert ds.sources.bitflag == desired_bitflag
        assert ds.wcs._upstream_bitflag == desired_bitflag
        assert ds.zp._upstream_bitflag == desired_bitflag
        assert ds.sub_image._upstream_bitflag == desired_bitflag
        assert ds.detections._upstream_bitflag == desired_bitflag
        assert ds.cutouts._upstream_bitflag == desired_bitflag
        for m in ds.measurements:
            assert m._upstream_bitflag == desired_bitflag
        assert ds.image.bitflag == 2  # not in the downstream of sources

        # test part 3: test update_downstream_badness() function by adding and removing flags
        # and observing propagation

        # commit to DB using this session
        with SmartSession() as session:
            ds.save_and_commit(session=session)
            ds.image = session.merge(ds.image)

            # add a bitflag and check that it appears in downstreams

            ds.image._bitflag = 2 ** 4  # bitflag for 'bad subtraction'
            session.add(ds.image)
            session.commit()
            ds.image.exposure.update_downstream_badness(session=session)
            session.commit()

            desired_bitflag = 2 ** 1 + 2 ** 4 + 2 ** 17  # 'banding' 'bad subtraction' 'many sources'
            assert ds.exposure.bitflag == 2 ** 1
            assert ds.image.bitflag == 2 ** 1 + 2 ** 4  # 'banding' and 'bad subtraction'
            assert ds.sources.bitflag == desired_bitflag
            assert ds.psf.bitflag == 2 ** 1 + 2 ** 4
            assert ds.wcs.bitflag == desired_bitflag
            assert ds.zp.bitflag == desired_bitflag
            assert ds.sub_image.bitflag == desired_bitflag
            assert ds.detections.bitflag == desired_bitflag
            assert ds.cutouts.bitflag == desired_bitflag
            for m in ds.measurements:
                assert m.bitflag == desired_bitflag

            # remove the bitflag and check that it disappears in downstreams
            ds.image._bitflag = 0  # remove 'bad subtraction'
            session.commit()
            ds.image.exposure.update_downstream_badness(session=session)
            session.commit()
            desired_bitflag = 2 ** 1 + 2 ** 17  # 'banding' 'many sources'
            assert ds.exposure.bitflag == 2 ** 1
            assert ds.image.bitflag == 2 ** 1  # just 'banding' left on image
            assert ds.sources.bitflag == desired_bitflag
            assert ds.psf.bitflag == 2 ** 1
            assert ds.wcs.bitflag == desired_bitflag
            assert ds.zp.bitflag == desired_bitflag
            assert ds.sub_image.bitflag == desired_bitflag
            assert ds.detections.bitflag == desired_bitflag
            assert ds.cutouts.bitflag == desired_bitflag
            for m in ds.measurements:
                assert m.bitflag == desired_bitflag

    finally:
        if 'ds' in locals():
            ds.delete_everything()
        # added this cleanup to make sure the temp data folder is cleaned up
        # this should be removed after we add datastore failure modes (issue #150)
        shutil.rmtree(os.path.join(os.path.dirname(exposure.get_fullpath()), '115'), ignore_errors=True)
        shutil.rmtree(os.path.join(archive.test_folder_path, '115'), ignore_errors=True)
        with SmartSession() as session:
            ds.exposure.bitflag = 0
            session.merge(ds.exposure)
            session.commit()
            # Remove the ProvenanceTag that will have been created
            session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag='test_bitflag_propagation'" ) )
            session.commit()


def test_get_upstreams_and_downstreams(decam_exposure, decam_reference, decam_default_calibrators, archive):
    """
    Test that get_upstreams() and get_downstreams() return the proper objects.
    """
    exposure = decam_exposure
    ref = decam_reference
    sec_id = ref.section_id

    try:  # cleanup the file at the end
        p = Pipeline( pipeline={'provenance_tag': 'test_get_upstreams_and_downstreams'} )
        p.subtractor.pars.refset = 'test_refset_decam'
        ds = p.run(exposure, sec_id)

        # commit to DB using this session
        with SmartSession() as session:
            ds.save_and_commit(session=session)

            # test get_upstreams()
            assert ds.exposure.get_upstreams() == []
            assert [upstream.id for upstream in ds.image.get_upstreams(session)] == [ds.exposure.id]
            assert [upstream.id for upstream in ds.sources.get_upstreams(session)] == [ds.image.id]
            assert [upstream.id for upstream in ds.wcs.get_upstreams(session)] == [ds.sources.id]
            assert [upstream.id for upstream in ds.psf.get_upstreams(session)] == [ds.image.id]
            assert [upstream.id for upstream in ds.zp.get_upstreams(session)] == [ds.sources.id]
            assert set([upstream.id for upstream in ds.sub_image.get_upstreams(session)]) == set([
                ref.image.id,
                ref.image.sources.id,
                ref.image.psf.id,
                ref.image.bg.id,
                ref.image.wcs.id,
                ref.image.zp.id,
                ds.image.id,
                ds.sources.id,
                ds.psf.id,
                ds.bg.id,
                ds.wcs.id,
                ds.zp.id,
            ])
            assert [upstream.id for upstream in ds.detections.get_upstreams(session)] == [ds.sub_image.id]
            assert [upstream.id for upstream in ds.cutouts.get_upstreams(session)] == [ds.detections.id]

            for measurement in ds.measurements:
                assert [upstream.id for upstream in measurement.get_upstreams(session)] == [ds.cutouts.id]


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
            exp_downstreams = [ downstream.id for downstream in ds.exposure.get_downstreams(session) ]
            # assert len(exp_downstreams) == 2
            assert ds.image.id in exp_downstreams

            assert set([downstream.id for downstream in ds.image.get_downstreams(session)]) == set([
                ds.sources.id,
                ds.psf.id,
                ds.bg.id,
                ds.wcs.id,
                ds.zp.id,
                ds.sub_image.id
            ])
            assert [downstream.id for downstream in ds.sources.get_downstreams(session)] == [ds.sub_image.id]
            assert [downstream.id for downstream in ds.psf.get_downstreams(session)] == [ds.sub_image.id]
            assert [downstream.id for downstream in ds.wcs.get_downstreams(session)] == [ds.sub_image.id]
            assert [downstream.id for downstream in ds.zp.get_downstreams(session)] == [ds.sub_image.id]
            assert [downstream.id for downstream in ds.sub_image.get_downstreams(session)] == [ds.detections.id]
            assert [downstream.id for downstream in ds.detections.get_downstreams(session)] == [ds.cutouts.id]
            measurement_ids = set([measurement.id for measurement in ds.measurements])
            assert set([downstream.id for downstream in ds.cutouts.get_downstreams(session)]) == measurement_ids

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


def test_datastore_delete_everything(decam_datastore):
    im = decam_datastore.image
    im_paths = im.get_fullpath(as_list=True)
    sources = decam_datastore.sources
    sources_path = sources.get_fullpath()
    psf = decam_datastore.psf
    psf_paths = psf.get_fullpath(as_list=True)
    sub = decam_datastore.sub_image
    sub_paths = sub.get_fullpath(as_list=True)
    det = decam_datastore.detections
    det_path = det.get_fullpath()
    cutouts = decam_datastore.cutouts
    cutouts_file_path = cutouts.get_fullpath()
    measurements_list = decam_datastore.measurements

    # make sure we can delete everything
    decam_datastore.delete_everything()

    # make sure everything is deleted
    for path in im_paths:
        assert not os.path.exists(path)

    assert not os.path.exists(sources_path)

    for path in psf_paths:
        assert not os.path.exists(path)

    for path in sub_paths:
        assert not os.path.exists(path)

    assert not os.path.exists(det_path)

    assert not os.path.exists(cutouts_file_path)

    # check these don't exist on the DB:
    with SmartSession() as session:
        assert session.scalars(sa.select(Image).where(Image.id == im.id)).first() is None
        assert session.scalars(sa.select(SourceList).where(SourceList.id == sources.id)).first() is None
        assert session.scalars(sa.select(PSF).where(PSF.id == psf.id)).first() is None
        assert session.scalars(sa.select(Image).where(Image.id == sub.id)).first() is None
        assert session.scalars(sa.select(SourceList).where(SourceList.id == det.id)).first() is None
        assert session.scalars(sa.select(Cutouts).where(Cutouts.id == cutouts.id)).first() is None
        if len(measurements_list) > 0:
            assert session.scalars(
                sa.select(Measurements).where(Measurements.id == measurements_list[0].id)
            ).first() is None


def test_provenance_tree(pipeline_for_tests, decam_refset, decam_exposure, decam_datastore, decam_reference):
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

    provs = p.make_provenance_tree( decam_exposure )
    assert isinstance(provs, dict)

    # Make sure the ProvenanceTag got created properly
    ptags = check_prov_tag( provs.values(), 'pipeline_for_tests' )

    t_start = datetime.datetime.utcnow()
    ds = p.run(decam_exposure, 'S3')  # the data should all be there so this should be quick
    t_end = datetime.datetime.utcnow()

    assert ds.image.provenance_id == provs['preprocessing'].id
    assert ds.sources.provenance_id == provs['extraction'].id
    assert ds.psf.provenance_id == provs['extraction'].id
    assert ds.wcs.provenance_id == provs['extraction'].id
    assert ds.zp.provenance_id == provs['extraction'].id
    assert ds.sub_image.provenance_id == provs['subtraction'].id
    assert ds.detections.provenance_id == provs['detection'].id
    assert ds.cutouts.provenance_id == provs['cutting'].id
    assert ds.measurements[0].provenance_id == provs['measuring'].id

    with SmartSession() as session:
        report = session.scalars(
            sa.select(Report).where(Report.exposure_id == decam_exposure.id).order_by(Report.start_time.desc())
        ).first()
        assert report is not None
        assert report.success
        assert abs(report.start_time - t_start) < datetime.timedelta(seconds=1)
        assert abs(report.finish_time - t_end) < datetime.timedelta(seconds=1)

    # Make sure that the provenance tags are reused if we ask for the same thing
    newprovs = p.make_provenance_tree( decam_exposure )
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
                     extraction={'sources': { 'threshold': 42. } } )
    with pytest.raises( RuntimeError,
                        match='The following provenances are not associated with provenance tag pipeline_for_tests' ):
        newp.make_provenance_tree( decam_exposure )


def test_inject_warnings_errors(decam_datastore, decam_reference, pipeline_for_tests):
    from pipeline.top_level import PROCESS_OBJECTS
    p = pipeline_for_tests
    p.subtractor.pars.refset = 'test_refset_decam'

    try:
        obj_to_process_name = {
            'preprocessor': 'preprocessing',
            'extractor': 'detection',
            'backgrounder': 'backgrounding',
            'astrometor': 'astro_cal',
            'photometor': 'photo_cal',
            'subtractor': 'subtraction',
            'detector': 'detection',
            'cutter': 'cutting',
            'measurer': 'measuring',
            'scorers': 'scoring',
        }
        for process, objects in PROCESS_OBJECTS.items():
            if isinstance(objects, str):
                objects = [objects]
            elif isinstance(objects, dict):
                objects = list(set(objects.values()))  # e.g., "extractor", "astrometor", "photometor"

            # first reset all warnings and errors
            for obj in objects:
                for _, objects2 in PROCESS_OBJECTS.items():
                    if isinstance(objects2, str):
                        objects2 = [objects2]
                    elif isinstance(objects2, dict):
                        objects2 = list(set(objects2.values()))  # e.g., "extractor", "astrometor", "photometor"
                    for obj2 in objects2:
                        if isinstance(getattr(p, obj2), list): # [scorer1, scorer2]
                            for obj3 in getattr(p, obj2):
                                obj3.pars.inject_exceptions = False
                                obj3.pars.inject_warnings = False
                        else:
                            getattr(p, obj2).pars.inject_exceptions = False
                            getattr(p, obj2).pars.inject_warnings = False

                if not SKIP_WARNING_TESTS:
                    # set the warning:
                    if isinstance(getattr(p, obj), list): # [scorer1, scorer2]
                        for o in getattr(p, obj):
                            o.pars.inject_warnings = True
                    else:
                        getattr(p, obj).pars.inject_warnings = True

                    # run the pipeline
                    ds = p.run(decam_datastore)
                    expected = (f"{process}: <class 'UserWarning'> Warning injected by pipeline parameters "
                                f"in process '{obj_to_process_name[obj]}'")
                    assert expected in ds.report.warnings

                # these are used to find the report later on
                exp_id = ds.exposure_id
                sec_id = ds.section_id
                prov_id = ds.report.provenance_id

                # set the error instead
                if isinstance(getattr(p, obj), list): # [scorer1, scorer2]
                    for o in getattr(p, obj):
                        o.pars.inject_warnings = False
                        o.pars.inject_exceptions = True
                else:
                    getattr(p, obj).pars.inject_warnings = False
                    getattr(p, obj).pars.inject_exceptions = True
                # run the pipeline again, this time with an exception

                with pytest.raises(
                        RuntimeError,
                        match=f"Exception injected by pipeline parameters in process '{obj_to_process_name[obj]}'"
                ):
                    ds = p.run(decam_datastore)
                    ds.reraise()

                # fetch the report object
                with SmartSession() as session:
                    reports = session.scalars(
                        sa.select(Report).where(
                            Report.exposure_id == exp_id,
                            Report.section_id == sec_id,
                            Report.provenance_id == prov_id
                        ).order_by(Report.start_time.desc())
                    ).all()
                    report = reports[0]  # the last report is the one we just generated
                    assert len(reports) - 1 == report.num_prev_reports
                    assert not report.success
                    assert report.error_step == process
                    assert report.error_type == 'RuntimeError'
                    assert 'Exception injected by pipeline parameters' in report.error_message

    finally:
        if 'ds' in locals():
            ds.read_exception()
            ds.delete_everything()


def test_multiprocessing_make_provenances_and_exposure(decam_exposure, decam_reference, pipeline_for_tests):
    from multiprocessing import SimpleQueue, Process
    process_list = []
    pipeline_for_tests.subtractor.pars.refset = 'test_refset_decam'
    def make_provenances(exposure, pipeline, queue):
        provs = pipeline.make_provenance_tree(exposure)
        queue.put(provs)

    queue = SimpleQueue()
    for i in range(3):  # github has 4 CPUs for testing, so 3 sub-processes and 1 main process
        p = Process(target=make_provenances, args=(decam_exposure, pipeline_for_tests, queue))
        p.start()
        process_list.append(p)

    # also run this on the main process
    provs = pipeline_for_tests.make_provenance_tree(decam_exposure)

    for p in process_list:
        p.join()
        assert not p.exitcode

    # check that the provenances are the same
    for _ in process_list:  # order is not kept but all outputs should be the same
        output_provs = queue.get()
        assert output_provs['measuring'].id == provs['measuring'].id
