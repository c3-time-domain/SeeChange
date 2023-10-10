import pytest

import numpy
from astropy.io import fits

from pipeline.preprocessing import Preprocessor

# Gotta include this to make sure decam gets registered
# in Instrument's list of classes
import models.decam


def test_preprocessing( decam_example_exposure, decam_default_calibrators ):
    # The decam_default_calibrators fixture is included so that
    # _get_default_calibrators won't be called as a side effect of calls
    # to Preprocessor.run().  (To avoid committing.)

    preppor = Preprocessor()
    ds = preppor.run( decam_example_exposure, 'N1' )

    # Check some Preprocesor internals
    assert preppor._calibset == 'externally_supplied'
    assert preppor._flattype == 'externally_supplied'
    assert preppor._stepstodo == [ 'overscan', 'linearity', 'flat', 'fringe' ]
    assert preppor._ds.exposure.filter[:1] == 'g'
    assert preppor._ds.section_id == 'N1'
    assert set( preppor.stepfiles.keys() ) == { 'flat', 'linearity' }

    # Flatfielding should have improved the sky noise, though for DECam
    # it looks like this is a really small effect.  I've picked out a
    # section that's all sky (though it may be in the wings of a bright
    # star, but, whatever).

    # 56 is how much got trimmed from this image
    rawsec = ds.image.raw_data[ 2226:2267, 267+56:308+56 ]
    flatsec = ds.image.data[ 2226:2267, 267:308 ]
    assert flatsec.std() < rawsec.std()

    # Make sure that some bad pixels got masked, but not too many
    assert numpy.all( ds.image._flags[ 1390:1400, 1430:1440 ] == 4 )
    assert numpy.all( ds.image._flags[ 4085:4093, 1080:1100 ] == 1 )
    assert ( ds.image._flags != 0 ).sum() / ds.image.data.size < 0.03

    # Make sure that the weight is reasonable
    assert not numpy.any( ds.image._weight < 0 )
    assert ( ds.image.data[3959:3980, 653:662].std() ==
             pytest.approx( 1./numpy.sqrt(ds.image._weight[3959:3980, 653:662]), rel=0.2 ) )

    # TODO : other checks that preprocessing did what it was supposed to do?
    # (Look at image header once we have HISTORY adding in there.)

    # Test some overriding

    preppor = Preprocessor()
    ds = preppor.run( decam_example_exposure, 'N1', steps=['overscan','linearity'] )
    assert preppor._calibset == 'externally_supplied'
    assert preppor._flattype == 'externally_supplied'
    assert preppor._stepstodo == [ 'overscan', 'linearity' ]
    assert preppor._ds.exposure.filter[:1] == 'g'
    assert preppor._ds.section_id == 'N1'
    assert set( preppor.stepfiles.keys() ) == { 'linearity' }

