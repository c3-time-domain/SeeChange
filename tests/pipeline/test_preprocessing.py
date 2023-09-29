from pipeline.preprocessing import Preprocessor

from astropy.io import fits

# Gotta include this to make sure decam gets registered
# in Instrument's list of classes
import models.decam


def test_preprocessing( decam_example_exposure ):
    preppor = Preprocessor( 'DECam', 'N1', decam_example_exposure )
    ds = preppor.run()

    # fits.writeto( 'preprocessed.fits', ds.image.data, header=ds.image.raw_header, overwrite=True )

    # Flatfielding should have improved the sky noise, though for DECam
    # it looks like this is a really small effect.  I've picked out a
    # seciton that's all sky (though it may be in the wings of a bright
    # star, but, whatever).

    # 56 is how much got trimmed from this image
    rawsec = ds.image.raw_data[ 2226:2267, 267+56:308+56 ]
    flatsec = ds.image.data[ 2226:2267, 267:308 ]
    assert flatsec.std() < rawsec.std()
