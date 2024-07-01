import pytest

from tests.conftest import SKIP_WARNING_TESTS


def test_warnings_and_exceptions(decam_datastore, cutter):
    if not SKIP_WARNING_TESTS:
        cutter.pars.inject_warnings = 1

        with pytest.warns(UserWarning) as record:
            cutter.run(decam_datastore)
        assert len(record) > 0
        assert any("Warning injected by pipeline parameters in process 'cutting'." in str(w.message) for w in record)

    cutter.pars.inject_warnings = 0
    cutter.pars.inject_exceptions = 1
    with pytest.raises(Exception) as excinfo:
        ds = cutter.run(decam_datastore)
        ds.reraise()
    assert "Exception injected by pipeline parameters in process 'cutting'." in str(excinfo.value)
    ds.read_exception()