import pytest
import os
import time
import logging

from models.base import SmartSession, Psycopg2Connection
from models.knownexposure import KnownExposure
from models.exposure import Exposure
from models.image import Image
from models.reference import image_subtraction_components
from models.source_list import SourceList
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements, MeasurementSet
from models.knownexposure import PipelineWorker
from pipeline.pipeline_exposure_launcher import ExposureLauncher

from util.logger import SCLogger


def unhold_decam_exposure( conductor_connector, identifier ):
    data = conductor_connector.send( "conductor/getknownexposures" )
    idtodo = None
    for ke in data['knownexposures']:
        if ke['identifier'] == identifier:
            idtodo = ke['id']
    assert idtodo is not None
    conductor_connector.send( "conductor/releaseexposures/", { 'knownexposure_ids': [ idtodo ] } )

    # Make sure the right things are held
    with SmartSession() as session:
        kes = session.query( KnownExposure ).all()
    assert all( [ ke.hold for ke in kes if str(ke.id) != idtodo ] )
    assert all( [ not ke.hold for ke in kes if str(ke.id) == idtodo ] )

    return idtodo


def delete_exposure( exposure_name ):
    # IMPORTANT.  Make sure that the tests that call this do NOT
    #   use an exposure that is referenced by the session
    #   fixture decam_exposure (in tests/fixtures/decam.py), or
    #   in any other session fixtures, because it will undermine
    #   that fixture!

    # Before deleting the exposure, we have to make sure it's not referenced in the
    #  knownexposures table
    if exposure_name is None:
        return

    expids = []
    with SmartSession() as session:
        kes = session.query( KnownExposure ).filter( KnownExposure.identifier==exposure_name ).all()

        for ke in kes:
            expids.append( ke.exposure_id )
            ke.exposure_id = None
            ke.upsert( session=session )


        for expid in expids:
            exposure = Exposure.get_by_id( expid )
            if exposure is not None:
                exposure.delete_from_disk_and_database( remove_folders=True, remove_downstreams=True, archive=True )


def verify_exposure_image_etc( numimages=2, numsourcelists=2, numzps=2, numsubs=2 ):
    """Check that the expected number of images etc. are in the database for processed Exposures.

    Looks for all Exposures that have an entry in the KnownExposures table.

    Returns the Image objects pulled from the query on subtractions.

    """
    if ( numimages == 0 ) and any( numsourcelists > 0, numzps > 0, numsubs > 0 ):
        raise RuntimeError( "You asked me to check something that makes no sense.  Stop that." )

    with SmartSession() as session:
        expq = session.query( Exposure ).join( KnownExposure ).filter( KnownExposure.exposure_id==Exposure._id )
        assert expq.count() == 1
        exposure = expq.first()

        imgq = session.query( Image ).filter( Image.exposure_id==exposure.id ).order_by( Image.section_id )
        assert imgq.count() == numimages
        images = imgq.all()
        imgids = [ i.id for i in images ]

        slq = session.query(SourceList).filter( SourceList.image_id.in_( imgids ) )
        assert slq.count() == numsourcelists

        zpq = ( session.query(ZeroPoint)
                .join( WorldCoordinates, WorldCoordinates._id==ZeroPoint.wcs_id )
                .join( SourceList, SourceList._id==WorldCoordinates.sources_id )
                .filter( SourceList.image_id.in_( imgids ) ) )
        assert zpq.count() == numzps

        subq = ( session.query( Image )
                 .join( image_subtraction_components, image_subtraction_components.c.image_id==Image._id )
                 .join( ZeroPoint, ZeroPoint._id==image_subtraction_components.c.new_zp_id )
                 .join( WorldCoordinates, WorldCoordinates._id==ZeroPoint.wcs_id )
                 .join( SourceList, SourceList._id==WorldCoordinates.sources_id )
                 .filter( SourceList.image_id.in_( [ images[0].id, images[1].id ] ) )
                 .order_by( Image.section_id ) )
        assert subq.count() == numsubs

        return subq.all()


# See comment on test_exposure_launcher re: user and admin_user fixtures
def test_exposure_launcher_conductor_through_step( conductor_connector,
                                                   conductor_config_decam_pull_all_held,
                                                   decam_elais_e1_two_references,
                                                   user, admin_user ):
    decam_exposure_name = 'c4d_230702_080904_ori.fits.fz'

    try:
        unhold_decam_exposure( conductor_connector, decam_exposure_name )

        conductor_connector.send( "/conductor/updateparameters/throughstep=photocal" )
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "SELECT throughstep FROM conductor_config" )
            assert cursor.fetchone()[0] == 'photocal'

        elaunch = ExposureLauncher( 'testcluster', 'testnode', numprocs=2, onlychips=['S2', 'N16'], verify=False,
                                    worker_log_level=logging.DEBUG )
        elaunch.register_worker()
        elaunch( max_n_exposures=1, die_on_exception=True )
        verify_exposure_image_etc( numimages=2, numsourcelists=2, numzps=2, numsubs=0 )

    finally:
        delete_exposure( decam_exposure_name )

        # See comment in test_exposure_launcher re: leftover calibrator files

        # Finally, remove the pipelineworker that got created
        # (Don't bother cleaning up knownexposures, the fixture will do that)
        with SmartSession() as session:
            pws = session.query( PipelineWorker ).filter( PipelineWorker.cluster_id=='testcluster' ).all()
            for pw in pws:
                session.delete( pw )
            session.commit()

        # Reset the conductor through step to 'scoring' (the default)
        conductor_connector.send( "/conductor/updateparameters/throughstep=scoring" )


# See comment on test_exposure_launcher re: user and admin_user fixtures
def test_exposure_launcher_through_step( conductor_connector,
                                         conductor_config_decam_pull_all_held,
                                         decam_elais_e1_two_references,
                                         user, admin_user ):
    decam_exposure_name = 'c4d_230702_080904_ori.fits.fz'

    try:
        unhold_decam_exposure( conductor_connector, decam_exposure_name )

        conductor_connector.send( "/conductor/updateparameters/throughstep=photocal" )
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "SELECT throughstep FROM conductor_config" )
            assert cursor.fetchone()[0] == 'photocal'

        elaunch = ExposureLauncher( 'testcluster', 'testnode', numprocs=2, onlychips=['S2', 'N16'], verify=False,
                                    through_step='extraction', worker_log_level=logging.DEBUG )
        elaunch.register_worker()
        elaunch( max_n_exposures=1, die_on_exception=True )
        verify_exposure_image_etc( numimages=2, numsourcelists=2, numzps=0, numsubs=0 )

    finally:
        delete_exposure( decam_exposure_name )

        # See comment in test_exposure_launcher re: leftover calibrator files

        # Finally, remove the pipelineworker that got created
        # (Don't bother cleaning up knownexposures, the fixture will do that)
        with SmartSession() as session:
            pws = session.query( PipelineWorker ).filter( PipelineWorker.cluster_id=='testcluster' ).all()
            for pw in pws:
                session.delete( pw )
            session.commit()

        # Reset the conductor through step to 'scoring' (the default)
        conductor_connector.send( "/conductor/updateparameters/throughstep=scoring" )


# NOTE -- this next test gets killed on github actions; googling about a bit
# suggests that it uses too much memory.  Given that it launches two
# image processes tasks, and that we still are allocating more memory
# than we think we should be, this is perhaps not a surprise.  Put in an
# env var that will cause it to get skipped on github actions, but to be
# run by default when run locally.  This env var is set in the github
# actions workflows.
#
# ...while the memory has been reduced a while back, for reasons I don't
# understand, if you run this test in the context of all the other tests,
# it hangs on the R/B step.  If you run this test all by itself, it
# does not hang.  So, for now, keep skipping it on github, and run it
# individually manually.

# The user and admin_user fixtures are included not because they are needed,
# but because setting a breakpoint at the end of this test and running it
# is a convenient way to set something up for playing around with the
# web ap interactively.  Including those users as fixtures doesn't slow
# the test down significantly, but does mean that every time I want to
# do this, I don't have to remember to put the users there in addition
# to putting in the breakpoint.
@pytest.mark.skipif( os.getenv('SKIP_BIG_MEMORY') is not None, reason="Uses too much memory for github actions" )
def test_exposure_launcher( conductor_connector,
                            conductor_config_decam_pull_all_held,
                            decam_elais_e1_two_references,
                            user, admin_user ):
    # This is just a basic test that the exposure launcher runs.  It does
    # run in parallel, but only two chips.  On my desktop, it takes about 2
    # minutes.  There aren't tests of failure modes written (yet?).

    decam_exposure_name = 'c4d_230702_080904_ori.fits.fz'

    try:
        # Figure out our known exposure id and unhold our exposure
        unhold_decam_exposure( conductor_connector, decam_exposure_name )

        # Make our launcher
        elaunch = ExposureLauncher( 'testcluster', 'testnode', numprocs=2, onlychips=['S2', 'N16'], verify=False,
                                    worker_log_level=logging.DEBUG )
        elaunch.register_worker()

        # Make sure the worker got registered properly
        res = conductor_connector.send( "conductor/getworkers" )
        assert len( res['workers'] ) == 1
        assert res['workers'][0]['cluster_id'] == 'testcluster'
        assert res['workers'][0]['node_id'] == 'testnode'

        t0 = time.perf_counter()
        elaunch( max_n_exposures=1, die_on_exception=True )
        dt = time.perf_counter() - t0

        SCLogger.debug( f"Running exposure processor took {dt} seconds" )

        # Make sure that two subtractions were created, and extract them
        subs = verify_exposure_image_etc( numimages=2, numsourcelists=2, numzps=2, numsubs=2 )

        # Find the exposure that got processed
        with SmartSession() as session:
            measq = ( session.query( Measurements )
                      .join( MeasurementSet, Measurements.measurementset_id==MeasurementSet._id )
                      .join( Cutouts, MeasurementSet.cutouts_id==Cutouts._id )
                      .join( SourceList, Cutouts.sources_id==SourceList._id )
                      .join( Image, SourceList.image_id==Image._id ) )
            meas0 = measq.filter( Image._id==subs[0].id ).all()
            meas1 = measq.filter( Image._id==subs[1].id ).all()
            assert len(meas0) == 2
            assert len(meas1) == 4

    finally:
        # Deleting the exposure should cascade to everything else

        delete_exposure( decam_exposure_name )

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
