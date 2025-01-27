import pytest
import os
import time
import logging

from models.base import SmartSession
from models.knownexposure import KnownExposure
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.knownexposure import PipelineWorker
from pipeline.pipeline_exposure_launcher import ExposureLauncher

from util.logger import SCLogger


# NOTE -- this test gets killed on github actions; googling about a bit
# suggests that it uses too much memory.  Given that it launches two
# image processes tasks, and that we still are allocating more memory
# than we think we should be, this is perhaps not a surprise.  Put in an
# env var that will cause it to get skipped on github actions, but to be
# run by default when run locally.  This env var is set in the github
# actions workflows.

@pytest.mark.skipif( os.getenv('SKIP_BIG_MEMORY') is not None, reason="Uses too much memory for github actions" )
def test_exposure_launcher( conductor_connector,
                            conductor_config_for_decam_pull,
                            decam_elais_e1_two_references ):
    # This is just a basic test that the exposure launcher runs.  It does
    # run in parallel, but only two chips.  On my desktop, it takes about 2
    # minutes.  There aren't tests of failure modes written (yet?).

    decam_exposure_name = 'c4d_230702_080904_ori.fits.fz'

    # Hold all exposures
    data = conductor_connector.send( "getknownexposures" )
    tohold = []
    idtodo = None
    for ke in data['knownexposures']:
        if ke['identifier'] == decam_exposure_name:
            idtodo = ke['id']
        else:
            tohold.append( ke['id'] )
    assert idtodo is not None
    res = conductor_connector.send( "holdexposures/", { 'knownexposure_ids': tohold } )

    # Make sure the right things got held
    with SmartSession() as session:
        kes = session.query( KnownExposure ).all()
    assert all( [ ke.hold for ke in kes if str(ke.id) != idtodo ] )
    assert all( [ not ke.hold for ke in kes if str(ke.id) == idtodo ] )

    elaunch = ExposureLauncher( 'testcluster', 'testnode', numprocs=2, onlychips=['S2', 'N16'], verify=False,
                                worker_log_level=logging.DEBUG )
    elaunch.register_worker()

    try:
        # Make sure the worker got registered properly
        res = conductor_connector.send( "getworkers" )
        assert len( res['workers'] ) == 1
        assert res['workers'][0]['cluster_id'] == 'testcluster'
        assert res['workers'][0]['node_id'] == 'testnode'
        assert res['workers'][0]['nexps'] == 1

        t0 = time.perf_counter()
        elaunch( max_n_exposures=1, die_on_exception=True )
        dt = time.perf_counter() - t0

        SCLogger.debug( f"Running exposure processor took {dt} seconds" )

        # Find the exposure that got processed
        with SmartSession() as session:
            expq = session.query( Exposure ).join( KnownExposure ).filter( KnownExposure.exposure_id==Exposure._id )
            assert expq.count() == 1
            exposure = expq.first()
            imgq = session.query( Image ).filter( Image.exposure_id==exposure.id ).order_by( Image.section_id )
            assert imgq.count() == 2
            images = imgq.all()
            sub0 = session.query( Image ).filter( Image.new_image_id==images[0].id ).first()
            sub1 = session.query( Image ).filter( Image.new_image_id==images[1].id ).first()
            assert sub0 is not None
            assert sub1 is not None

            measq = session.query( Measurements ).join( Cutouts ).join( SourceList ).join( Image )
            meas0 = measq.filter( Image._id==sub0.id ).all()
            meas1 = measq.filter( Image._id==sub1.id ).all()
            assert len(meas0) == 2
            assert len(meas1) == 6

    finally:
        # Try to clean up everything.  If we delete the exposure, the two images and two subtraction images,
        #   that should cascade to most everything else.
        with SmartSession() as session:
            exposure = ( session.query( Exposure ).join( KnownExposure )
                         .filter( KnownExposure.exposure_id==Exposure._id ) ).first()
            images = session.query( Image ).filter( Image.exposure_id==exposure.id ).all()
            imgids = [ i.id for i in images ]
            subs = session.query( Image ).filter( Image.new_image_id.in_( imgids ) ).all()
        for sub in subs:
            sub.delete_from_disk_and_database( remove_folders=True, remove_downstreams=True, archive=True )
        for img in images:
            img.delete_from_disk_and_database( remove_folders=True, remove_downstreams=True, archive=True )
        # Before deleting the exposure, we have to make sure it's not referenced in the
        #  knownexposures table
        with SmartSession() as session:
            kes = session.query( KnownExposure ).filter( KnownExposure.exposure_id==exposure.id ).all()
        for ke in kes:
            ke.exposure_id = None
            ke.upsert()

        # WORRY -- I think this is deleting something that shouldn't get deleted until
        #  the decam_exposure session fixture cleans up.  Because this test tends to be
        #  one of the last ones that runs, this hasn't bitten us, but it could.
        exposure.delete_from_disk_and_database( remove_folders=True, remove_downstreams=True, archive=True )

        # There will also have been a whole bunch of calibrator files.
        # Don't delete those, because the decam_default_calibrators
        # fixture will clean them up, and is a session-scope fixture;
        # deleting them here would undermine that fixture.  (I wanted
        # not to have that as a fixture for this test so that running
        # this test by itself tested two processes downloading those at
        # the same time-- and, indeed, in so doing found some problems
        # that needed to be fixed.)  That means that if you run this
        # test by itself, the fixture teardown will complain about stuff
        # left over in the database.  However, if you run all the tests,
        # it'll be fine, because the decam_default_calibrators fixture
        # will have been run and its teardown does the necessary
        # cleanup.

        # Finally, remove the pipelineworker that got created
        # (Don't bother cleaning up knownexposures, the fixture will do that)
        with SmartSession() as session:
            pws = session.query( PipelineWorker ).filter( PipelineWorker.cluster_id=='testcluster' ).all()
            for pw in pws:
                session.delete( pw )
            session.commit()
