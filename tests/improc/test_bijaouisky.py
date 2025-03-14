import io

import numpy as np

from astropy.io import fits
import sep

from improc.bijaouisky import estimate_single_sky, estimate_smooth_sky
from improc.sextrsky import sextrsky
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


def test_various_algorithms():
    rng = np.random.default_rng( 42 )
    seeingpix = 4.2

    strio = io.StringIO()
    for nstars in [ 0, 2000, 20000, 200000 ]:
        image = rng.normal( 1764., 42., size=( 2048, 2048 ) )
        var = np.full_like( image, 42. * 42. )
        SCLogger.debug( f"Putting {nstars} stars on noise image" )
        pepper_stars( image, var, seeingpix, nstars, 0., 20000., rng )

        # For testing purposes
        fits.writeto( f"test_{nstars}.fits", data=image, overwrite=True )

        SCLogger.debug( f"first sep background w/ nstars={nstars}" )
        sep_raw_bg = sep.Background( image, bw=256, bh=256, fw=3, fh=3 )
        fits.writeto( f"raw_seg_bkg_{nstars}.fits", sep_raw_bg.back(), overwrite=True )

        sep_mask_globalbacks = []
        sep_mask_globalrmses = []
        sextr_skys = []
        sextr_skysigs = []
        lastback = sep_raw_bg.back()
        for sepiter in range(8):
            SCLogger.debug( f"sep extracting w/ nstars={nstars}, iteration {sepiter}" )
            sep.set_extract_pixstack( 10000000 )
            sep.set_sub_object_limit( 40960 )
            stars, segment = sep.extract( image - lastback, var=var, thresh=5., segmentation_map=True )
            fits.writeto( f"sep_seg_{nstars}_{sepiter}.fits", data=segment, overwrite=True )

            SCLogger.debug( f"masked sep background w/ nstars={nstars}, iteration {sepiter}" )
            sep_mask_bg = sep.Background( image, mask=segment, bw=256, bh=256, fw=3, fh=3 )
            fits.writeto( f"mask_sep_bkg_{nstars}_{sepiter}.fits", sep_mask_bg.back(), overwrite=True )
            fits.writeto( f"raw-mask_sep_bkg_{nstars}_{sepiter}.fits",
                          sep_raw_bg.back() - sep_mask_bg.back(), overwrite=True )
            sep_mask_globalbacks.append( sep_mask_bg.globalback )
            sep_mask_globalrmses.append( sep_mask_bg.globalrms )

            SCLogger.debug( f"masked sextrsky background w/ nstars={nstars}, iteration {sepiter}" )
            masked_sextrsky, sextrskysig = sextrsky( image, maskdata=segment, sigcut=3, boxsize=256, filtsize=3 )
            fits.writeto( f"mask_sextr_bkg_{nstars}_{sepiter}.fits", masked_sextrsky, overwrite=True )
            sextr_skys.append( np.median( masked_sextrsky ) )
            sextr_skysigs.append( sextrskysig )

            lastback = masked_sextrsky

        SCLogger.debug( f"bijaoui background w/ nstars={nstars}" )
        a, sigma, s = estimate_single_sky( image, figname="plot.png", maxiterations=40, converge=0.001,
                                           sigcut=3., lowsigcut=5. )

        SCLogger.debug( f"masked bijaoui background w/ nstars={nstars}" )
        m_a, m_sigma, m_s = estimate_single_sky( image, bpm=segment, figname="plot.png", maxiterations=40,
                                                 converge=0.001, sigcut=3., lowsigcut=5. )

        strio.write( f"\nBACKGROUND RESULTS FOR nstars={nstars}:\n" )
        strio.write( f"          raw sep: bkg = {sep_raw_bg.globalback:8.2f}  rms = {sep_raw_bg.globalrms:8.2f}\n" )
        for i in range( len( sep_mask_globalbacks ) ):
            strio.write( f"    masked sep {i:2d}: bkg = {sep_mask_globalbacks[i]:8.2f}  "
                         f"rms = {sep_mask_globalrmses[i]:8.2f}\n" )
            strio.write( f"  masked sextr {i:2d}: bkg = {sextr_skys[i]:8.2f}  rms = {sextr_skysigs[i]:8.2f}\n" )
        strio.write( f"          bijoaui: bkg = {s:8.2f}  rms = {sigma:8.2f}\n" )
        strio.write( f"   masked bijaoui: bkg = {m_s:8.2f}  rms = {m_sigma:8.2f}\n" )

    SCLogger.info( strio.getvalue() )

    import pdb; pdb.set_trace()
    pass
