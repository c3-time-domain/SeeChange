import time
import logging

from models.base import SmartSession
from models.knownexposure import PipelineWorker
from pipeline.pipeline_exposure_launcher import ExposureLauncher

# This is just a basic test that the exposure launcher runs.
# It does run in parallel, but only two chips.
# There aren't tests of failure modes written (yet?).
def test_exposure_launcher( conductor_connector, conductor_config_for_decam_pull, decam_elais_e1_two_references ):
    # Hold all exposures
    data = conductor_connector.send( "getknownexposures" )
    tohold = [ ke['id'] for ke in data['knownexposures'][1:] ]
    idtodo = data['knownexposures'][0]['id']
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

        # import pdb; pdb.set_trace()
        assert False
        
    finally:
        # Clean up all the data products from the subtractions run... we hope...
        # Clean up the pipeline worker entry if it got created
        with SmartSession() as session:
            pws = session.query( PipelineWorker ).filter( PipelineWorker.cluster_id=='testcluster' ).all()
            for pw in pws:
                session.delete( pw )
            session.commit()
            
    
