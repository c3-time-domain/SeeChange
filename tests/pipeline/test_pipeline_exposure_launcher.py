import pytest
import time
import logging

from models.base import SmartSession
from models.knownexposure import KnownExposure
from models.exposure import Exposure
from models.image import Image, image_upstreams_association_table
from models.source_list import SourceList
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.knownexposure import PipelineWorker
from models.calibratorfile import CalibratorFile
from models.datafile import DataFile
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
                            decam_elais_e1_two_references,
                            decam_exposure_name ):
    # This is just a basic test that the exposure launcher runs.  It does
    # run in parallel, but only two chips.  On my desktop, it takes about 2
    # minutes.  There aren't tests of failure modes written (yet?).

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
    res = conductor_connector.send( f"holdexposures/", { 'knownexposure_ids': tohold } )

    elaunch = ExposureLauncher( 'testcluster', 'testnode', numprocs=2, onlychips=['S3', 'N16'], verify=False,
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
            expq = session.query( Exposure ).join( KnownExposure ).filter( KnownExposure.exposure_id==Exposure.id )
            assert expq.count() == 1
            exposure = expq.first()
            imgq = session.query( Image ).filter( Image.exposure_id==exposure.id ).order_by( Image.section_id )
            assert imgq.count() == 2
            images = imgq.all()
            # There is probably a cleverl sqlalchemy way to do this
            #  using the relationship, but searching for a bit didn't
            #  find anything that worked, so just do it manually
            subq = ( session.query( Image ).join( image_upstreams_association_table,
                                                  Image.id==image_upstreams_association_table.c.downstream_id ) )
            sub0 = subq.filter( image_upstreams_association_table.c.upstream_id==images[0].id ).first()
            sub1 = subq.filter( image_upstreams_association_table.c.upstream_id==images[1].id ).first()
            assert sub0 is not None
            assert sub1 is not None

            measq = session.query( Measurements ).join( Cutouts ).join( SourceList ).join( Image )
            meas0 = measq.filter( Image.id==sub0.id ).all()
            meas1 = measq.filter( Image.id==sub1.id ).all()
            assert len(meas0) == 3
            assert len(meas1) == 6

            assert False

    finally:
        # Try to clean up everything.  If we delete the exposure, the two images and two subtraction images,
        #   that should cascade to most everything else.
        with SmartSession() as session:
            exposure = ( session.query( Exposure ).join( KnownExposure )
                         .filter( KnownExposure.exposure_id==Exposure.id ) ).first()
            images = session.query( Image ).filter( Image.exposure_id==exposure.id ).all()
            imgids = [ i.id for i in images ]
            subs = ( session.query( Image ).join( image_upstreams_association_table,
                                                  Image.id==image_upstreams_association_table.c.downstream_id )
                     .filter( image_upstreams_association_table.c.upstream_id.in_( imgids ) ) ).all()
            for sub in subs:
                sub.delete_from_disk_and_database( session=session, commit=True, remove_folders=True,
                                                   remove_downstreams=True, archive=True )
            for img in images:
                img.delete_from_disk_and_database( session=session, commit=True, remove_folders=True,
                                                   remove_downstreams=True, archive=True )
            # Before deleting the exposure, we have to make sure it's not referenced in the
            #  knownexposures table
            kes = session.query( KnownExposure ).filter( KnownExposure.exposure_id==exposure.id ).all()
            for ke in kes:
                ke.exposure_id = None
                session.merge( ke )
            session.commit()
            exposure.delete_from_disk_and_database( session=session, commit=True, remove_folders=True,
                                                    remove_downstreams=True, archive=True )

            # There will also have been a whole bunch of calibrator files

            # PROBLEM : the fixtures/decam.py:decam_default_calibrator
            #   fixture is a scope-session fixture that loads these
            #   things!  So, don't delete them here, that would
            #   undermine the fixture.  (I wanted to not have this test
            #   depend on that fixture so that running this test by
            #   itself tested two processes downloading those at the
            #   same time-- and indeed, in so doing found some problems
            #   that needed to be fixed.)  This means that if you run
            #   this test by itself, the fixture teardown will complain
            #   about stuff left over in the database.  But, if you run
            #   all the tests, that other fixture will end up having
            #   been run and will have loaded anything we would have
            #   loaded here.
            #
            #   Leave the code commented out so one can uncomment it
            #   when running just this test, if one wishes.

            # deleted_images = set()
            # deleted_datafiles = set()
            # cfs = session.query( CalibratorFile ).filter( CalibratorFile.instrument=='DECam' )
            # for cf in cfs:
            #     if cf.image_id is not None:
            #         if cf.image_id not in deleted_images:
            #             cf.image.delete_from_disk_and_database( session=session, commit=True, remove_folders=True,
            #                                                     remove_downstreams=True, archive=True )
            #             # Just in case more than one CalibratorFile entry refers to the same image
            #             deleted_images.add( cf.image_id )
            #             session.delete( cf )
            #     elif cf.datafile_id is not None:
            #         if cf.datafile_id not in deleted_datafiles:
            #             cf.datafile.delete_from_disk_and_database( session=session, commit=True, remove_folders=True,
            #                                                        remove_downstreams=True, archive=True )
            #             # Just in case more than one CalibratorFile entry refers to the same datafile
            #             deleted_datafiles.add( cf.datafile_id )
            #             session.delete( cf )
            #     # don't need to delete the cf, because it will have cascaded from above

        # Finally, remove the pipelineworker that got created
        # (Don't bother cleaning up knownexposures, the fixture will do that)
        with SmartSession() as session:
            pws = session.query( PipelineWorker ).filter( PipelineWorker.cluster_id=='testcluster' ).all()
            for pw in pws:
                session.delete( pw )
            session.commit()


