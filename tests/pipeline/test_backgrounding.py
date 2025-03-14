import pytest

import numpy as np

from astropy.io import fits

from models.base import SmartSession
from models.image import Image
from models.background import Background
from models.source_list import SourceList

from pipeline.data_store import DataStore
from pipeline.backgrounding import Backgrounder

from improc.tools import sigma_clipping, pepper_stars

from tests.conftest import SKIP_WARNING_TESTS


# This tests the default method, which is 'sep'
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


def test_compare_sep_sextr( provenance_base, provenance_extra ):
    rng = np.random.default_rng( 42 )
    imagedata, var = pepper_stars( 2048, 2048, 42., 4.2, 1.0, 2000, 0., 20000, rng=rng )

    image = Image( provenance_id=provenance_base.id )
    image.header = fits.Header()
    image.data = imagedata
    image.weight = 1. / var
    image.flags = np.zeros_like( imagedata, dtype=np.int16 )
    ds = DataStore( image )
    # Data Store gets all pissy in places if it doesn't have a provenance tree
    ds.prov_tree = { 'preprocessing': provenance_base, 'extraction': provenance_extra }

    sepbger = Backgrounder( method='sep', format='map', box_size=256, filt_size=3 )
    sextrbger = Backgrounder( method='sextr', format='map', box_size=256, filt_size=3 )
    iterbger = Backgrounder( method='iter_sextr', format='map', box_size=256, filt_size=3, iter_sextr_iterations=3 )

    sepbg = sepbger.run( ds )
    sextrbg = sextrbger.run( ds )
    iterbg = iterbger.run( ds )

    # There were not many stars in this sample image, so all these things should be very similar.
    assert sepbg.noise == pytest.approx( sextrbg.noise, rel=0.01 )
    assert sepbg.noise == pytest.approx( iterbg.noise, rel=0.01 )
    assert sepbg.value == pytest.approx( sextrbg.value, abs=sextrbg.noise * 0.01 )
    assert sepbg.value == pytest.approx( iterbg.value, abs=sextrbg.noise * 0.01 )

    assert np.all( ( ( sepbg.counts - sextrbg.counts ) / sepbg.rms ) < 0.02 )
    assert np.all( ( ( sepbg.counts - iterbg.counts ) / sepbg.rms ) < 0.02 )


def test_compare_sep_sextr_crowded_image( provenance_base, provenance_extra ):
    rng = np.random.default_rng( 42 )
    imagedata, var = pepper_stars( 2048, 2048, 42., 4.2, 1.0, 200000, 0., 20000, rng=rng )
    skynoise = 42.
    skylevel = skynoise ** 2

    image = Image( provenance_id=provenance_base.id )
    image.header = fits.Header()
    image.data = imagedata
    image.weight = 1. / var
    image.flags = np.zeros_like( imagedata, dtype=np.int16 )
    ds = DataStore( image )
    # Data Store gets all pissy in places if it doesn't have a provenance tree
    ds.prov_tree = { 'preprocessing': provenance_base, 'extraction': provenance_extra }

    sepbger = Backgrounder( method='sep', format='map', box_size=256, filt_size=3 )
    sextrbger = Backgrounder( method='sextr', format='map', box_size=256, filt_size=3 )
    iterbger = Backgrounder( method='iter_sextr', format='map', box_size=256, filt_size=3, iter_sextr_iterations=3 )

    sepbg = sepbger.run( ds )
    sextrbg = sextrbger.run( ds )
    iterbg = iterbger.run( ds )

    # In the croweded field, the iterative method should have done better, but
    #   all will have overestimated both sky level and sky noise

    assert sepbg.value > skylevel
    assert sextrbg.value > skylevel
    assert iterbg.value > skylevel
    assert sepbg.noise > skynoise
    assert sextrbg.noise > skynoise
    assert iterbg.noise > skynoise

    assert iterbg.value < sepbg.value
    assert ( iterbg.value - skylevel ) / ( sepbg.value - skylevel ) < 0.3
    assert iterbg.value < sextrbg.value
    assert ( iterbg.value - skylevel ) / ( sextrbg.value - skylevel ) < 0.3
    assert iterbg.noise < sepbg.noise
    assert ( iterbg.noise - skynoise ) / ( sepbg.noise - skynoise ) < 0.1
    assert iterbg.noise < sextrbg.noise
    assert ( iterbg.noise - skynoise ) / ( sextrbg.noise - skynoise ) < 0.1


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
