import pytest

import numpy as np

from models.base import SmartSession
from models.background import Background

from improc.tools import sigma_clipping

from tests.conftest import SKIP_WARNING_TESTS


def test_measuring_background( decam_datastore_through_extraction ):
    ds = decam_datastore_through_extraction

    # NOTE -- we had to get the decam_datastore_through_extraction because
    # right now background is considered a sources sibling, so lots of
    # code gets upset if you try to set background when there are no
    # sources.  In practice, at the moment, we don't do backgrounding
    # and source extraction in the same step, so we may want to consider
    # not making backgrounding a source sibling.

    # Verify that the background isn't in the database already
    with SmartSession() as session:
        assert session.query( Background ).filter( Background.sources_id==ds.sources.id ).first() is None

    backgrounder = ds._pipeline.backgrounder
    ds = backgrounder.run( ds )

    # check that the background is statistically similar to the image stats
    mu, sig = sigma_clipping(ds.image.nandata)
    assert mu == pytest.approx(ds.bg.value, rel=0.01)
    assert sig == pytest.approx(ds.bg.noise, rel=0.2)  # this is really a very rough estimate

    # is the background subtracted image a good representation?
    nanbgsub = ds.image.nandata - ds.bg.counts
    mu, sig = sigma_clipping( nanbgsub )
    assert mu == pytest.approx(0, abs=sig)
    assert sig < 25

    # most of the pixels are inside a 3 sigma range
    assert np.sum(np.abs(nanbgsub) < 3 * sig) > 0.9 * ds.image.nandata.size

    # this is not true of the original image
    assert np.sum(np.abs(ds.image.nandata) < 3 * sig) < 0.001 * ds.image.nandata.size

    # Try to do the background again, but this time using the "zero" method

    # Hack the provenance tree so this will run.  Never do this!  Except
    # in tests like this.  Doing this in actual code to get past an
    # error will break all kinds of things; figure out where the error
    # really came from.  (A better way to do this would be to set
    # backgrounder.pars.method to 'zero', then run
    # ds.prov_tree=ds._pipeline.make_provenance_tree(ds.exposure).)
    ds.prov_tree['backgrounding'].parameters['method'] = 'zero'
    backgrounder.pars.method = 'zero'

    # Because we did a horribly ugly wrong never-do-this-hack to the provenances, the backgrounder
    #   will just load the old background from the database because it thinks the old provenance
    #   is right even though we changed the parameters.  To make sure that doesn't happen, wipe
    #   out the old background.
    ds.bg.delete_from_disk_and_database()
    ds.bg = None

    ds = backgrounder.run(ds)
    assert ds.bg.method == 'zero'
    assert ds.bg.value == 0
    assert ds.bg.noise == 0


def test_warnings_and_exceptions( decam_datastore_through_extraction ):
    ds = decam_datastore_through_extraction
    backgrounder = ds._pipeline.backgrounder

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
