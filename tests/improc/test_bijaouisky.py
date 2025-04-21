import pytest
import io

import numpy as np

from astropy.io import fits

from improc.bijaouisky import estimate_single_sky
from improc.sextrsky import sextrsky
from improc.sextractor import run_sextractor
from improc.tools import pepper_stars
from util.logger import SCLogger


# Uncomment this next skip line to actually run this "test"
@pytest.mark.skip( reason='Test is only for user inspection; uncomment the skip to run it' )
def test_various_algorithms():
    rng = np.random.default_rng( 42 )
    seeingpix = 4.2

    strio = io.StringIO()
    for nstars in [ 0, 2000, 20000, 200000 ]:
        SCLogger.debug( f"Putting {nstars} stars on noise image" )
        image, var = pepper_stars( 2048, 2048, 42., seeingpix, 1.0, nstars, 0., 20000., rng )

        # For testing purposes
        fits.writeto( f"test_{nstars}.fits", data=image, overwrite=True )

        sextr_skys = []
        sextr_skysigs = []
        niters = 8
        sky = 0.
        for iter in range( niters ):
            SCLogger.debug( f"Running sextractor iteration {iter} w/ nstars={nstars}" )
            if iter == 0:
                sextr_res = run_sextractor( fits.Header(), image, 1./var, maskdata=None,
                                            outbase=f"sextractor_{nstars}_{iter}",
                                            seeing_fwhm=4.2, pixel_scale=1.0,
                                            back_type="AUTO", back_size=256, back_filtersize=3,
                                            writebg=True, writeseg=True, timeout=300,
                                            mem_pixstack=10000000 )
            else:
                sextr_res = run_sextractor( fits.Header(), image - sky, 1./var, maskdata=None,
                                            outbase=f"sextractor_{nstars}_{iter}",
                                            seeing_fwhm=4.2, pixel_scale=1.0,
                                            back_type="MANUAL", back_value=0.,
                                            writebg=True, writeseg=True, timeout=300,
                                            mem_pixstack=10000000 )
            with fits.open( sextr_res['segmentation'] ) as ifp:
                objmask = ifp[0].data
            SCLogger.debug( f"...done sextractor iteration {iter} w/ nstars={nstars}" )

            fmasked = ( objmask > 0 ).sum() / objmask.size
            SCLogger.debug( f"sextrsky w/ nstars={nstars}, {fmasked:.2f} masked, iteration {iter}..." )
            sky, skysig = sextrsky( image, maskdata=objmask, boxsize=256, filtsize=3 )
            fits.writeto( f"sextrsky_{nstars}_{iter}.fits", data=sky, overwrite=True )
            sextr_skys.append( np.median( sky ) )
            sextr_skysigs.append( skysig )
            SCLogger.debug( "...done" )

        SCLogger.debug( f"bijaoui background w/ nstars={nstars}" )
        _a, sigma, s = estimate_single_sky( image, figname="plot.png", maxiterations=40, converge=0.001,
                                           sigcut=3., lowsigcut=5. )

        SCLogger.debug( f"masked bijaoui background w/ nstars={nstars}" )
        _m_a, m_sigma, m_s = estimate_single_sky( image, bpm=objmask, figname="plot.png", maxiterations=40,
                                                 converge=0.001, sigcut=3., lowsigcut=5. )

        strio.write( f"\nBACKGROUND RESULTS FOR nstars={nstars}; nominal sky=1764, noise=42:\n" )
        for i in range( niters ):
            strio.write( f"     sextractor {i:2d}: bkg = {sextr_skys[i]:8.2f}  rms = {sextr_skysigs[i]:8.2f}\n" )
        strio.write( f"           bijoaui: bkg = {s:8.2f}  rms = {sigma:8.2f}\n" )
        strio.write( f"    masked bijaoui: bkg = {m_s:8.2f}  rms = {m_sigma:8.2f}\n" )

    SCLogger.info( strio.getvalue() )


    # CONCLUSIONS:
    #    - bijaouisky is a disaster.  It always underestimates the sky
    #      value.  For the most crowded field it gets a lot closer than
    #      any of the other algorithms, for both sky level and sky sig
    #      (which is why I used it for galactic fields in lensgrinder),
    #      but even then it's underestimating the sky by a few adu.
    #
    #    - sextrsky seems to converge after 4-5 iterations; further
    #      iterations are almost certainly not worth it.  It will
    #      overestimate the sky and sky sigma in very crowded fields
    #      still, but not as badly as in the first run.
