import io

import numpy as np

from astropy.io import fits
import sep

from improc.bijaouisky import estimate_single_sky, estimate_smooth_sky
from improc.sextrsky import sextrsky
from improc.sectractor import run_sextractor
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

        SCLogger.debug( f"First sextractor run w/ nstars={nstars}..." )
        first_sextr = run_sextractor( fits.Header(), image, 1./var, maskdata=None,
                                      outbase="first_sextr_{nstars}",
                                      seeing_fwhm=4.2, pixel_scale=1.0,
                                      back_type="AUTO", back_size=256, back_filtersize=3,
                                      writebg=True, seg=True, timeout=300 )
        SCLogger.debug( f"...done first sextractor run w/ nstars={nstars}" )
        with fits.open( first_sextr['segmentation'] ) as ifp:
            objmask = ifp[0].data
        with fits.open( first_sextr['bkg'] ) as ifp:
            bg = ifp[0].data

        sextr_skys = []
        sextr_skysigs = []
        niters = 8
        for iter in range( niters ):
            fmasked = ( objmask > 0 ).sum() / objmask.size
            SCLogger.debug( f"sextrsky w/ nstars={nstars}, {fmasked:.2f} masked, iteration {iter}..." )
            sky, skysig = sextrsky( image - bg, maskdata=objmask, boxsize=256, filtsize=3 )
            fits.writeto( "setrsky_{nstars}_{iter}.fits", data=sky )
            sextr_skys.append( np.median( sky ) )
            sextr_skysigs.append( skysig )
            SCLogger.debug( f"...done" )

            if iter < niters - 1:
                SCLogger.debug( f"Sextractor iteration {iter} w/ nstars={nstars}..." )
                sextr_res = run_sextractor( fits.Header(), image - bg, 1./var, maskdata=objmask,
                                            outbase="sextractor_{nstars}_{iter}",
                                            seeing_fwhm=4.2, pixel_scale=1.0,
                                            back_type="AUTO", back_size=256, back_filtersize=3,
                                            writebg=True, seg=True, timeout=300 )
                SCLogger.debug( f"...done sextractor iteration {iter} w/ nstars={nstars}" )
                with fits.open( sextr_res['segmentation'] ) as ifp:
                    objmask = ifp[0].data
                with fits.open( sextr_res['bkg'] ) as ifp:
                    bg = ifp[0].data

        SCLogger.debug( f"bijaoui background w/ nstars={nstars}" )
        a, sigma, s = estimate_single_sky( image, figname="plot.png", maxiterations=40, converge=0.001,
                                           sigcut=3., lowsigcut=5. )

        SCLogger.debug( f"masked bijaoui background w/ nstars={nstars}" )
        m_a, m_sigma, m_s = estimate_single_sky( image, bpm=segment, figname="plot.png", maxiterations=40,
                                                 converge=0.001, sigcut=3., lowsigcut=5. )

        strio.write( f"\nBACKGROUND RESULTS FOR nstars={nstars}:\n" )
        strio.write( f"  first sextractor: bkg = {first_sextr['bkg_mean']:8.2f}  "
                     f"rms = {first_sextr['bkg_sig']:8.2f}\n" )
        for i in range( niters ):
            strio.write( f"   sextractor {i:2d}: bkg = {sextr_skys[i]:8.2f}  rms = {sextr_skysigs[i]:8.2f}"
        strio.write( f"           bijoaui: bkg = {s:8.2f}  rms = {sigma:8.2f}\n" )
        strio.write( f"    masked bijaoui: bkg = {m_s:8.2f}  rms = {m_sigma:8.2f}\n" )

    SCLogger.info( strio.getvalue() )

    import pdb; pdb.set_trace()
    pass
