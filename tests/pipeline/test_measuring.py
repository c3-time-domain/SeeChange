import pytest


@pytest.mark.skip( reason="This test will get wholly rewritten with Issue #404" )
def test_measuring( decam_default_calibrators, decam_datastore_through_cutouts ):
    raise NotImplementedError( "See Issue #404" )
