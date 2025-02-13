import sqlalchemy as sa

from models.base import SmartSession
from models.deepscore import DeepScore, DeepScoreSet

from pipeline.top_level import Pipeline


def test_deepscore_saving(ptf_datastore, scorer):
    ds = ptf_datastore
    ##  delete the scores from the datastore
    ds.scores = None
    ## run the scorer on the measurements in the datastore
    ds = scorer.run(ds)
    ## check the scores are there
    assert isinstance( ds.deepscore_set, DeepScoreSet )
    assert ds.deepscores == ds.deepscore_set.deepscores
    assert len(ds.deepscores) == len(ds.measurements)
    assert all( d.index_in_sources == m.index_in_sources for d, m in zip( ds.deepscores, ds.measurements ) )
    ## assert that the scores have not recalculated ( thus they were found on DB )
    assert not scorer.has_recalculated
    ## try to commit and confirm there are no errors
    ds.save_and_commit()

    return None


def test_multiple_algorithms(decam_exposure, decam_reference, decam_default_calibrators):
    # NOTE: decam_default_calibrators is included in this test in order to trigger proper
    # cleanup afterwards. Removing it will cause objects to be left in database

    exposure = decam_exposure
    ref = decam_reference
    sec_id = ref.image.section_id

    try:  # cleanup the file at the end.
        p1 = Pipeline( pipeline={'provenance_tag': 'test_multiple_algorithms1'} )
        p1.subtractor.pars.refset = 'test_refset_decam'
        p1.scorer.pars.algorithm = "random"
        ds1 = p1.run(exposure, sec_id)
        ds1.save_and_commit()

        # try and find all the existing objects, check they are right
        with SmartSession() as session:
            dbscores = ( session.query( DeepScore )
                         .join( DeepScoreSet, DeepScore.deepscoreset_id==DeepScoreSet._id )
                         .filter( DeepScoreSet.measurementset_id==ds1.measurement_set.id )
                         .order_by( DeepScore.index_in_sources )
                        ).all()
            assert len(dbscores) == len(ds1.measurements)
            assert all( d.index_in_sources == m.index_in_sources for d, m in zip( dbscores, ds1.measurements ) )

        p2 = Pipeline( pipeline={'provenance_tag': 'test_multiple_algorithms2'} )
        p2.subtractor.pars.refset = 'test_refset_decam'
        p2.scorer.pars.algorithm = "allperfect"
        ds2 = p2.run(exposure, sec_id)
        ds2.save_and_commit()

        assert ds2.deepscore_set.id != ds1.deepscore_set.id
        assert ds2.deepscore_set.provenance_id != ds1.deepscore_set.provenance_id

        # check that the proper number of scores are saved to db
        # try and find all the existing objects, check they are right
        with SmartSession() as session:
            dbscores = ( session.query( DeepScore )
                         .join( DeepScoreSet, DeepScore.deepscoreset_id==DeepScoreSet._id )
                         .filter( DeepScoreSet.measurementset_id==ds2.measurement_set.id )
                         .order_by( DeepScore.index_in_sources )
                        ).all()
            assert len(dbscores) == 2 * len(ds2.measurements)   # 2x the deepscores for the same measurement set

    finally:
        if 'ds1' in locals():
            ds1.delete_everything()
        if 'ds2' in locals():
            ds2.delete_everything()
        # Clean up the provenance tag created by the pipeline
        with SmartSession() as session:
            session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ),
                            { 'tag': 'test_multiple_algorithms1' } )
            session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ),
                            { 'tag': 'test_multiple_algorithms2' } )
            session.commit()
