import numpy as np

from astropy.io import fits
import sep_pjw as sep

from improc.bijaouisky import estimate_single_sky, estimate_smooth_sky
from util.logger import SCLogger


def test_pure_sky():
    rng = np.random.default_rng( 42 )
    image = rng.normal( 1337., 42., size=( 2048, 2048 ) )
    a, sigma, s = estimate_single_sky( image, figname="plot.png", maxiterations=40 )
    import pdb; pdb.set_trace()
    pass

def pepper_stars( image, var, seeing, nstars, minflux, fluxscale, rng=None ):
    if rng is None:
        rng = np.default_rng()
    sigma = seeing / 2.35482
    halfwid = int( np.ceil( 4. * seeing ) )
    xvals, yvals = np.meshgrid( range(-halfwid, halfwid+1), range(-halfwid, halfwid+1) )
    for n in range(nstars):
        x = rng.uniform( -2.*seeing, image.shape[1] + 2*seeing )
        y = rng.uniform( -2.*seeing, image.shape[0] + 2*seeing )
        ix = int( np.floor( x + 0.5 ) )
        iy = int( np.floor( y + 0.5 ) )

        flux = minflux + rng.exponential( fluxscale )
        star = flux / ( 2. * np.pi * sigma**2 ) * np.exp( -( xvals**2 + yvals**2 ) / ( 2. *sigma**2 ) )
        dstar = np.zeros_like( star )
        dstar[ star >= 0. ] = np.sqrt( star[ star >=0. ] )

        x0 = 0
        ix0 = ix - halfwid
        x1 = 2 * halfwid + 1
        ix1 = ix + halfwid + 1
        y0 = 0
        iy0 = iy - halfwid
        y1 = 2 * halfwid + 1
        iy1 = iy + halfwid + 1

        if ix0 < 0:
            x0 -= ix0
            ix0 = 0
        if iy0 < 0:
            y0 -= iy0
            iy0 = 0
        if ix1 > image.shape[1]:
            x1 -= ( ix1 - image.shape[1] )
            ix1 = image.shape[1]
        if iy1 > image.shape[0]:
            y1 -= ( iy1 - image.shape[0] )
            iy1 = image.shape[0]
            
        image[ iy0:iy1, ix0:ix1 ] += star[ y0:y1, x0:x1 ]
        var[ iy0:iy1, ix0:ix1 ] += dstar[ y0:y1, x0:x1 ] ** 2
        image[ iy0:iy1, ix0:ix1 ] += rng.normal( 0., dstar[ y0:y1, x0:x1 ] )
    

def test_sky_with_some_stars():
    rng = np.random.default_rng( 42 )
    image = rng.normal( 1764., 42., size=( 2048, 2048 ) )
    var = np.full_like( image, 42. * 42. )
    seeingpix = 4.2
    nstars = 20000
    pepper_stars( image, var, seeingpix, nstars, 0., 20000., rng )

    # For testing purposes
    fits.writeto( "test.fits", data=image, overwrite=True )

    SCLogger.debug( "first sep background" )
    sep_raw_bg = sep.Background( image, bw=256, bh=256, fw=3, fh=3 )

    SCLogger.debug( "sep extracting" )
    sep.set_extract_pixstack( 3000000 )
    sep.set_sub_object_limit( 20480 )
    stars, segment = sep.extract( image - sep_raw_bg.back(), var=var, thresh=5., segmentation_map=True )
    fits.writeto( "seg.fits", data=segment, overwrite=True )
    SCLogger.debug( "masked sep background" )
    sep_mask_bg = sep.Background( image, mask=segment, bw=256, bh=256, fw=3, fh=3 )
    
    a, sigma, s = estimate_single_sky( image, figname="plot.png", maxiterations=40, converge=0.001,
                                       sigcut=3., lowsigcut=5. )

    import pdb; pdb.set_trace()
    pass
