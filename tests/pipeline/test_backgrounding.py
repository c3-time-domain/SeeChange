import pytest
import uuid

import numpy as np

from improc.tools import sigma_clipping


def test_measuring_background(decam_processed_image, backgrounder):
    backgrounder.pars.test_parameter = uuid.uuid4().hex  # make sure there is no hashed value
    ds = backgrounder.run(decam_processed_image)

    # check that the background is statistically similar to the image stats
    mu, sig = sigma_clipping(ds.image.nandata)
    assert mu == pytest.approx(ds.bg.value, rel=0.01)
    assert sig == pytest.approx(ds.bg.noise, rel=0.2)  # this is really a very rough estimate

    # is the background subtracted image a good representation?
    mu, sig = sigma_clipping(ds.image.nandata_bgsub)  # also checks that nandata_bgsub exists
    assert mu == pytest.approx(0, abs=sig)
    assert sig < 10

    # most of the pixels are inside a 3 sigma range
    assert np.sum(np.abs(ds.image.nandata_bgsub) < 3 * sig) > 0.9 * ds.image.nandata.size

    # this is not true of the original image
    assert np.sum(np.abs(ds.image.nandata) < 3 * sig) < 0.001 * ds.image.nandata.size

    # try to do the background again, but this time using the "zero" method
    backgrounder.pars.method = 'zero'
    ds = backgrounder.run(ds)
    assert ds.bg.method == 'zero'
    assert ds.bg.value == 0
    assert ds.bg.noise == 0
    assert np.array_equal(ds.image.data, ds.image.data_bgsub)


def test_warnings_and_exceptions(decam_datastore, backgrounder):
    backgrounder.pars.inject_warnings = 1

    with pytest.warns(UserWarning) as record:
        backgrounder.run(decam_datastore)
    assert len(record) > 0
    assert any("Warning injected by pipeline parameters in process 'backgrounding'." in str(w.message) for w in record)

    backgrounder.pars.inject_warnings = 0
    backgrounder.pars.inject_exceptions = 1
    with pytest.raises(Exception) as excinfo:
        ds = backgrounder.run(decam_datastore)
        ds.reraise()
    assert "Exception injected by pipeline parameters in process 'backgrounding'." in str(excinfo.value)
    ds.read_exception()
