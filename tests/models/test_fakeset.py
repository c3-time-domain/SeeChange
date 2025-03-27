import pytest
import uuid

import numpy as np

from models.fakeset import FakeSet
from models.zero_point import ZeroPoint
from models.world_coordinates import WorldCoordinates
from models.source_list import SourceList
from models.psf import PSF
from models.image import Image


def test_save_and_load( bogus_fakeset_saved ):
    fakeset = FakeSet.get_by_id( bogus_fakeset_saved.id )
    assert fakeset.zp_id == bogus_fakeset_saved.zp_id
    assert fakeset.provenance_id == bogus_fakeset_saved.provenance_id
    assert fakeset.random_seed is None
    assert fakeset.fake_x is None
    assert fakeset.fake_y is None
    assert fakeset.fake_mag is None
    assert fakeset.host_dex is None
    assert fakeset.md5sum is not None

    # TODO: check that it got saved to the archive

    fakeset.load()
    assert fakeset.random_seed == bogus_fakeset_saved.random_seed
    assert np.all( fakeset.fake_x == bogus_fakeset_saved.fake_x )
    assert np.all( fakeset.fake_y == bogus_fakeset_saved.fake_y )
    assert np.all( fakeset.fake_mag == bogus_fakeset_saved.fake_mag )
    assert np.all( fakeset.host_dex == bogus_fakeset_saved.host_dex )


def test_properties( bogus_fakeset_saved ):
    fakeset = FakeSet.get_by_id( bogus_fakeset_saved.id )
    assert fakeset._zp is None
    assert fakeset._wcs is None
    assert fakeset._sources is None
    assert fakeset._psf is None
    assert fakeset.fake_x is None
    assert fakeset.fake_y is None
    assert fakeset.fake_mag is None
    assert fakeset.host_dex is None
    assert fakeset.random_seed is None

    def reset_fakeset_props():
        fakeset._zp = None
        fakeset._wcs = None
        fakeset._sources = None
        fakeset._psf = None
        fakeset._image = None
        fakeset.random_seed = None
        fakeset.fake_x = None
        fakeset.fake_y = None
        fakeset.host_dex = None
        fakeset.fake_mag = None

    # Make sure we can't do bad things
    for prop, cls in zip( [ 'zp', 'wcs', 'sources', 'psf', 'image' ],
                          [ ZeroPoint, WorldCoordinates, SourceList, PSF, Image ] ):
        reset_fakeset_props()
        with pytest.raises( TypeError, match=f"{prop} must be a.*not a" ):
            setattr( fakeset, prop, 5 )
        obj = cls()
        _ = obj.id
        # PSF needs a bit of special handling in this test
        if prop == 'psf':
            obj.sources_id = uuid.uuid4()
        with pytest.raises( ValueError, match=f"FakeSet's {prop} property has wrong id!" ):
            setattr( fakeset, prop, obj )

    # Now zero all those out and make sure we can do good things
    for prop, cls in zip( [ 'zp', 'wcs', 'sources', 'psf', 'image' ],
                          [ ZeroPoint, WorldCoordinates, SourceList, PSF, Image ] ):
        reset_fakeset_props()
        setattr( fakeset, prop, getattr( bogus_fakeset_saved, prop ) )

    # If we set the image, the others should get loaded as it checks stuff
    reset_fakeset_props()
    fakeset.image = bogus_fakeset_saved.image
    assert fakeset._zp is not None
    assert fakeset._wcs is not None
    assert fakeset._sources is not None
    assert fakeset._image is not None

    # Finally, make sure we can lazy load all the things
    for prop in [ 'zp', 'wcs','sources', 'psf', 'image' ]:
        reset_fakeset_props()
        assert getattr( fakeset, prop ).id == getattr( bogus_fakeset_saved, prop ).id


def test_inject_fakes( decam_fakeset ):
    fakeset = decam_fakeset

    imagedata, weight = fakeset.inject_on_to_image()

    diffdata = imagedata - fakeset.image.data
    diffweight = weight - fakeset.image.weight

    tmpimg = diffdata.copy()
    tmpwgt = diffweight.copy()
    for x, y, mag in zip( fakeset.fake_x, fakeset.fake_y, fakeset.fake_mag ):
        xc = int( np.round( x ) )
        yc = int( np.round( y ) )
        clip = fakeset.psf.get_clip( x, y )
        xmin = xc - clip.shape[1] // 2
        ymin = yc - clip.shape[0] // 2
        xmax = xmin + clip.shape[1]
        ymax = ymin + clip.shape[0]
        xmin = max( xmin, 0 )
        xmax = min( xmax, diffdata.shape[1] )
        ymin = max( ymin, 0 )
        ymax = min( ymax, diffdata.shape[0] )

        flux = diffdata[ ymin:ymax, xmin:xmax ].sum()
        wpos = fakeset.image.weight[ ymin:ymax, xmin:xmax ] > 0.
        # This is wonky. Because we've done a *perfect* subtraction,
        #   we're just looking at the statistics of the injected fake.
        #   So, the dflux is going to be smaller than the actual
        #   uncertainty, because it doesn't include sky noise, and
        #   we're usually sky dominated.
        #   (Real tests of subtractons will come later when we actually
        #   run subtractions with refs that aren't the image on to whic
        #   the fake was injected.)
        dflux = np.sqrt( ( 1. / weight[ ymin:ymax, xmin:xmax ][wpos]
                           - 1. / fakeset.image.weight[ ymin:ymax, xmin:xmax ][wpos] ).sum() )
        expectedflux = 10 ** ( ( mag - fakeset.zp.zp ) / -2.5 )
        assert flux == pytest.approx( expectedflux, abs=3.*dflux )

        # do this to make sure nothing outside the place where stuff got added was changed
        tmpimg[ ymin:ymax, xmin:xmax ] = 0.
        tmpwgt[ ymin:ymax, xmin:xmax ] = 0.

    # Now make sure nothing outside the place where stuff got added was changed
    assert np.all( tmpimg == 0. )
    assert np.all( tmpwgt == 0. )


# FakeAnalysis is tested in pipeline/test_fakeinjection.py
