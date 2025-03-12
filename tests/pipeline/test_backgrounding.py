import pytest

import numpy as np

from models.base import SmartSession
from models.background import Background
from models.source_list import SourceList

from improc.tools import sigma_clipping

from tests.conftest import SKIP_WARNING_TESTS


def test_measuring_background( decam_datastore_through_preprocessing ):
    ds = decam_datastore_through_preprocessing

    # Verify that the background isn't in the database already
    with SmartSession() as session:
        assert ( session.query( Background )
                 .join( SourceList, Background.sources_id==SourceList._id )
                 .filter( SourceList.image_id==ds.image.id )
                ).first() is None

    backgrounder = ds._pipeline.extractor.backgrounder
    bg = backgrounder.run( ds )

    # check that the background is statistically similar to the image stats
    mu, sig = sigma_clipping(ds.image.nandata)
    assert mu == pytest.approx(bg.value, rel=0.01)
    assert sig == pytest.approx(bg.noise, rel=0.2)  # this is really a very rough estimate

    # is the background subtracted image a good representation?
    nanbgsub = ds.image.nandata - bg.counts
    mu, sig = sigma_clipping( nanbgsub )
    assert mu == pytest.approx(0, abs=sig)
    assert sig < 25

    # most of the pixels are inside a 3 sigma range
    assert np.sum(np.abs(nanbgsub) < 3 * sig) > 0.9 * ds.image.nandata.size

    # this is not true of the original image
    assert np.sum(np.abs(ds.image.nandata) < 3 * sig) < 0.001 * ds.image.nandata.size

    # Try to do the background again, but this time using the "zero" method
    backgrounder.pars.method = 'zero'
    bg = backgrounder.run(ds)
    assert bg.method == 'zero'
    assert bg.value == 0
    assert bg.noise == 0


def test_warnings_and_exceptions( decam_datastore_through_preprocessing ):
    ds = decam_datastore_through_preprocessing
    backgrounder = ds._pipeline.extractor.backgrounder

    if not SKIP_WARNING_TESTS:
        backgrounder.pars.inject_warnings = 1

        with pytest.warns(UserWarning) as record:
            backgrounder.run( ds )
        assert len(record) > 0
        assert any("Warning injected by pipeline parameters in process 'backgrounding'." in str(w.message)
                   for w in record)

    backgrounder.pars.inject_warnings = 0
    backgrounder.pars.inject_exceptions = 1
    with pytest.raises(Exception, match="Exception injected by pipeline parameters in process 'backgrounding'."):
        ds = backgrounder.run( ds )
