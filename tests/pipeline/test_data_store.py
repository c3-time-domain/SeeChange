import pytest
import uuid

from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference
from models.cutouts import Cutouts
from models.measurements import Measurements

# The fixture gets us a datastore with everything saved and committed
# The fixture takes some time to build (even from cache), so glom
# all the tests together in one function.

# (TODO: think about test fixtures, see if we could easily (without too
# much repeated code) have module scope (and even session scope)
# fixtures with decam_datastore alongside the function scope fixture.)

def test_data_store( decam_datastore ):
    ds = decam_datastore

    # ********** Test basic attributes **********

    origexp = ds._exposure_id
    assert ds.exposure_id == origexp

    tmpuuid = uuid.uuid4()
    ds.exposure_id = tmpuuid
    assert ds._exposure_id == tmpuuid
    assert ds.exposure_id == tmpuuid

    with pytest.raises( Exception ) as ex:
        ds.exposure_id = 'this is not a valid uuid'
        import pdb; pdb.set_trace()
        pass

    ds.exposure_id = origexp

    origimg = ds._image_id
    assert ds.image_id == origimg

    tmpuuid = uuid.uuid4()
    ds.image_id = tmpuuid
    assert ds._image_id == tmpuuid
    assert ds.image_id == tmpuuid

    with pytest.raises( Exception ) as ex:
        ds.image_id = 'this is not a valud uuid'
        import pdb; pdb.set_trace()
        pass

    exp = ds.exposure
    import pdb; pdb.set_trace()
    # ROB, write tests

    assert ds._section is None
    sec = ds.section
    import pdb; pdb.set_trace()
    # ROB, write tests

    assert isinstance( ds.image, Image )
    assert isinstance( ds.sources, SourceList )
    assert isinstance( ds.bg, Background )
    assert isinstance( ds.psf, PSF )
    assert isinstance( ds.wcs, WorldCoordinates )
    assert isinstance( ds.zp, ZeroPoint )
    assert isinstance( ds.sub_image, Image )
    assert isinstance( ds.detections, SourceList )
    assert isinstance( ds.cutouts, Cutouts )
    assert isinstance( ds.measurements, list )
    assert all( [ isinstance( m, Measurements ) for m in ds.measurements ] )
    assert isinstance( ds.aligned_ref_image, Image )
    assert isinstance( ds.aligned_new_image, Image )

    # Test that if we set a property to None, the dependent properties cascade to None

    props = [ 'image', 'sources', 'sub_image', 'detections', 'cutouts', 'measurements' ]
    sourcesiblings = [ 'bg', 'psf', 'wcs', 'zp' ]
    origprops = { prop: getattr( ds, prop ) for prop in props }
    origprops.update( { prop: getattr( ds, prop ) for prop in sourcesiblings } )

    def resetprops():
        for k, v in origprops.items():
            setattr( ds, k, v )

    for i, prop in enumerate( props ):
        setattr( ds, prop, None )
        for subprop in props[i+1:]:
            assert getattr( ds, prop ) is None
            if subprop == 'sources':
                assert all( [ getattr( ds, p ) is None for p in sourcesiblings ] )
        resetprops()
        if prop == 'sources':
            for sibling in sourcesiblings:
                setattr( ds, sibling, None )
                for subprop in props[ props.index('sources')+1: ]:
                    assert getattr( ds, prop ) is None
                resetprops()

    # MORE

