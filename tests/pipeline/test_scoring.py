import pytest

from pipeline.scoring import Scorer

def test_rbbot( decam_datastore_through_measurements ):
    ds = decam_datastore_through_measurements
    scorer = Scorer( algorithm='RBbot-quiet-shadow-131' )
    scorer.run()
    import pdb; pdb.set_trace()
    pass

    
