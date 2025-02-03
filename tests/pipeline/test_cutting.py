import pytest

from tests.conftest import SKIP_WARNING_TESTS


# The actual operation of cutting is tested in models/test_cutouts.py

def test_warnings_and_exceptions( decam_datastore_through_detection ):
    ds = decam_datastore_through_detection
    cutter = ds._pipeline.cutter

    if not SKIP_WARNING_TESTS:
        cutter.pars.inject_warnings = 1
        ds._pipeline.make_provenance_tree( ds )

        with pytest.warns(UserWarning) as record:
            cutter.run( ds )
        assert len(record) > 0
        assert any("Warning injected by pipeline parameters in process 'cutting'." in str(w.message) for w in record)

    cutter.pars.inject_warnings = 0
    cutter.pars.inject_exceptions = 1
    ds.cutouts = None
    ds._pipeline.make_provenance_tree( ds )
    with pytest.raises(Exception, match="Exception injected by pipeline parameters in process 'cutting'."):
        ds = cutter.run( ds )
