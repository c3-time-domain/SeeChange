import pytest

from models.provenance import Provenance
from pipeline.scoring import Scorer


def test_rbbot( decam_datastore_through_measurements, code_version ):
    ds = decam_datastore_through_measurements
    scorer = Scorer( algorithm='RBbot-quiet-shadow-131-cut0.55' )
    # Need to update the DataStore's provenance tree because
    #   it was created with a different scorer algorithm
    scoreprov = Provenance( code_version_id=code_version.id,
                            process='scoring',
                            parameters=scorer.pars.get_critical_pars(),
                            upstreams=[ ds.prov_tree['measuring'] ]
                           )
    ds.prov_tree['scoring'] = scoreprov

    expected_scores = [ 0.412, 0.468, 0.511, 0.527, 0.768, 0.484, 0.488, 0.669, 0.350 ]
    scorer.run( ds )
    for scobj, expect in zip( ds.scores, expected_scores ):
        assert scobj.score == pytest.approx( expect, abs=0.002 )
