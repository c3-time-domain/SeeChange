import pytest

import numpy as np

from models.fakeset import FakeSet


def test_inject_fakes( decam_datastore_through_zp ):
    ds = decam_datastore_through_zp

    # Hand-picked locations
    # lim mag estimate is 23.22
    xs = [  1127.68, 1658.71, 1239.56, 1601.83, 1531.19, 921.57 ]
    ys = [  2018.91, 1998.84, 2503.77, 2898.47, 3141.27, 630.95  ]
    mags = [ 20.32,  19.90,   23.00,   22.00,  21.00,    23.30 ]

    fakeset = FakeSet( zp_id=ds.zp.id )
    # Load up the properties so we can cope with fakeset not being in the database
    fakeset.zp = ds.get_zp()
    fakeset.wcs = ds.get_wcs()
    fakeset.sources = ds.get_sources()
    fakeset.psf = ds.get_psf()
    fakeset.image = ds.get_image()

    fakeset.random_seed = 42
    fakeset.fake_x = np.array( xs )
    fakeset.fake_y = np.array( ys )
    fakeset.fake_mag = np.array( mags )

    import pdb; pdb.set_trace()
    imagedata, weight = fakeset.inject_on_to_image()

    diffdata = imagedata - fakeset.image.data
    diffweight = weight - fakeset.image.weight

    tmpimg = diffdata.copy()
    tmpwgt = diffweight.copy()
    for x, y, mag in zip( xs, ys, mags ):
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
        dflux = np.sqrt( ( 1. / weight[ ymin:ymax, xmin:xmax ][wpos]
                           - 1. / fakeset.image.weight[ ymin:ymax, xmin:xmax ][wpos] ).sum() )
        expectedflux = 10 ** ( ( mag - fakeset.zp.zp ) / -2.5 )
        import pdb; pdb.set_trace()
        assert flux == pytest.approx( expectedflux, abs=3.*dflux )

        # do this to make sure nothing outside the place where stuff got added was changed
        tmpimg[ ymin:ymax, xmin:xmax ] = 0.
        tmpwgt[ ymin:ymax, xmin:xmax ] = 0.

    # Now make sure nothing outside the place where stuff got added was changed
    assert np.all( tmpimg == 0. )
    assert np.all( tmpwgt == 0. )
