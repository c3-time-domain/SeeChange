import pytest
import hashlib

from astropy.io import fits
from astropy.wcs import WCS

from models.world_coordinates import WorldCoordinates

def test_world_coordinates( example_image_with_sources_and_psf_filenames ):
    image = example_image_with_sources_and_psf_filenames[0]

    with fits.open( image ) as ifp:
        hdr = ifp[0].header

    origwcs = WCS( hdr )
    origscs = origwcs.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )

    # Make sure we can construct a WorldCoordinates object from a WCS object

    wcobj = WorldCoordinates()
    wcobj.wcs = origwcs
    md5 = hashlib.md5( wcobj.header_excerpt.encode('ascii') )
    assert md5.hexdigest() == 'a13d6bdd520c5a0314dc751025a62619'

    # Make sure that we can construct a WCS from a WorldCoordinates

    hdrkws = wcobj.header_excerpt
    wcobj = WorldCoordinates()
    wcobj.header_excerpt = hdrkws
    scs = wcobj.wcs.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    for sc, origsc in zip( scs, origscs ):
        assert sc.ra.value == pytest.approx( origsc.ra.value, abs=0.01/3600. )
        assert sc.dec.value == pytest.approx( origsc.dec.value, abs=0.01/3600. )
