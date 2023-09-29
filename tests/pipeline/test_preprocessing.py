from pipeline.preprocessing import Preprocessor

from astropy.io import fits

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

