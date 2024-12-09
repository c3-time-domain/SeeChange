import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from improc.tools import strip_wcs_keywords, make_cutouts


def test_strip_wcs_keywords():
    hdr = fits.Header()

    keeps = {
        'NAXIS': 2,
        'NAXIS1': 1024,
        'NAXIS2': 2048,
        'AKW': 0,
        'A_1': 1,
        'PV1_1A': 3,
        'APb1_2': 4,
        'CDELT0': 5
    }
    loses = {
        'CRVAL1': 180.,
        'CRVAL2': -2.,
        'CRPIX1': 512.,
        'CRPIX2': 1024.,
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CTYPE1': 'something',
        'CTYPE2': 'something',
        'CD1_1': 1.,
        'CD1_2': 0.,
        'CD2_1': 0.,
        'CD2_2': 1.,
        'PC1_1': 1.,
        'PC1_2': 0.,
        'PC2_1': 0.,
        'PC2_2': 1.,
        'A_ORDER': 3,
        'AP_ORDER': 3,
        'B_ORDER': 3,
        'BP_ORDER': 3,
    }
    for i in range(0,2):
        for j in range(0,7):
            loses[f'PV{i}_{j}'] = 1.
            loses[f'AP_{i}_{j}'] = 1.
            loses[f'A_{i}_{j}'] = 1.
            loses[f'BP_{i}_{j}'] = 1.
            loses[f'B_{i}_{j}'] = 1.


    for kw, val in keeps.items():
        hdr[kw] = val
    for kw, val in loses.items():
        hdr[kw] = val

    strip_wcs_keywords(hdr)

    hdrkeys = [ k for k in hdr.keys() ]  # Oh, astropy
    for kw in loses.keys():
        assert kw not in hdrkeys

    for kw, val in keeps.items():
        assert hdr[kw] == val


def test_make_cutouts():
    rng = np.random.default_rng( seed=42 )
    imsize = 64
    image = rng.uniform( -100., 100., size=(imsize, imsize) )
    xs = [ 0, 25, 50, 60 ]
    ys = [ 4, 25, 50, 63 ]
    mess = np.meshgrid( ys, xs )
    xs = mess[1].flatten()
    ys = mess[0].flatten()

    # Tolerances : 32-bit floats have 24 bits of mantissa,
    #   2^24 = 1.68×10⁷ , there are ~7 sig figs.
    #   For numbers up to 100 (which is what we did random
    #   above), we should expect better than an absolute
    #   tolerance of 1e-4 (as that's 6 sig figs), so use that.
    atol = 1e-4

    def check_cuts( cuts, xs, ys, size, imshape, dtype, atol ):
        dtype = np.dtype( dtype )
        assert cuts.dtype == dtype
        assert cuts.shape[1] == size
        assert cuts.shape[2] == size
        halfsize = int(np.floor(size/2.))
        for i, xy in enumerate( zip( xs, ys ) ):
            x, y = xy
            x = int( np.round(x) )
            y = int( np.round(y) )
            left = -halfsize if x >= halfsize else -x
            right = halfsize if x <= imshape[1]-1 - halfsize else imshape[1]-1 - x
            bottom = -halfsize if y >= halfsize else -y
            top = halfsize if y <= imshape[0]-1 - halfsize else imshape[0]-1 - y
            x0 = x + left
            x1 = x + right + 1
            y0 = y + bottom
            y1 = y + top + 1
            cx0 = halfsize + left
            cx1 = halfsize + right + 1
            cy0 = halfsize + bottom
            cy1 = halfsize + top + 1

            assert_allclose( image[y0:y1, x0:x1] , cuts[i][cy0:cy1, cx0:cx1] , atol=atol, rtol=0. )
            if dtype.kind == 'f':
                if left > -halfsize:
                    assert np.all( np.isnan( cuts[i][:, 0:halfsize+left] ) )
                if bottom > -halfsize:
                    assert np.all( np.isnan( cuts[i][0:halfsize+bottom, :] ) )
                if right < halfsize:
                    assert np.all( np.isnan( cuts[i][:, -(halfsize-right)] ) )
                if top < halfsize:
                    assert np.all( np.isnan( cuts[i][-(halfsize-top), :] ) )
            else:
                if left > -halfsize:
                    assert np.all( cuts[i][:, 0:halfsize+left] == 0 )
                if bottom > -halfsize:
                    assert np.all( cuts[i][0:halfsize+bottom, :] == 0 )
                if right < halfsize:
                    assert np.all( cuts[i][:, -(halfsize-right)] == 0 )
                if top < halfsize:
                    assert np.all( cuts[i][-(halfsize-top), :] == 0 )



    # Cutouts with same dtype as image
    for size in [ 5, 15, 25 ]:
        if size == 15:
            # Make sure that's the default
            cuts = make_cutouts( image, xs, ys )
        else:
            cuts = make_cutouts( image, xs, ys, size=size )
        check_cuts( cuts, xs, ys, size, image.shape, image.dtype, atol )

    # 32-bit big-endian floats
    for size in [5, 15, 25]:
        cuts = make_cutouts( image, xs, ys, size=size, dtype='>f4' )
        check_cuts( cuts, xs, ys, size, image.shape, '>f4', atol )
        assert cuts.dtype.kind == 'f'
        # It will say "=" if big-endian is native, so swap twice to get numpy to really say big-endian
        assert cuts.dtype.newbyteorder().newbyteorder().byteorder == '>'
        assert cuts.dtype.itemsize == 4

    # 32-bit little-endian floats
    for size in [5, 15, 25]:
        cuts = make_cutouts( image, xs, ys, size=size, dtype='<f4' )
        check_cuts( cuts, xs, ys, size, image.shape, '<f4', atol )
        assert cuts.dtype.kind == 'f'
        assert cuts.dtype.newbyteorder().newbyteorder().byteorder == '<'
        assert cuts.dtype.itemsize == 4

    # 32-bit native floats
    for size in [5, 15, 25]:
        cuts = make_cutouts( image, xs, ys, size=size, dtype='f4' )
        check_cuts( cuts, xs, ys, size, image.shape, 'f4', atol )
        assert cuts.dtype.kind == 'f'
        assert cuts.dtype.isnative
        assert cuts.dtype.itemsize == 4

    # 64-bit big-endian floats
    for size in [5, 15, 25]:
        cuts = make_cutouts( image, xs, ys, size=size, dtype='>f8' )
        check_cuts( cuts, xs, ys, size, image.shape, '>f8', atol )
        assert cuts.dtype.kind == 'f'
        assert cuts.dtype.newbyteorder().newbyteorder().byteorder == '>'
        assert cuts.dtype.itemsize == 8

    # 16-bit integers
    # Tolerance now has to be 1, because of integers!
    for size in [5, 15, 25]:
        cuts = make_cutouts( image, xs, ys, size=size, dtype='i2' )
        check_cuts( cuts, xs, ys, size, image.shape, 'i2', 1 )
        assert cuts.dtype.kind == 'i'
        assert cuts.dtype.isnative
        assert cuts.dtype.itemsize == 2

    # This should fail -- can't make cutouts of type timedelta!
    with pytest.raises( TypeError, match="make_cutouts: can only make float or integer cutouts" ):
        cuts = make_cutouts( image, xs, ys, dtype='m' )
