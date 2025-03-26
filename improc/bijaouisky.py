import os
import math
import argparse
import logging

import numpy as np
from scipy.signal import convolve2d
from scipy.special import erfc
from scipy.optimize import differential_evolution
from scipy.optimize import curve_fit
from numpy.polynomial.chebyshev import chebval2d

from astropy.io import fits

from util.logger import SCLogger


# NOTE: in the functions below, "adu" is used as a synonym for I.
# Reason: ruff objects to the use of I as a variable, presumably
# because depending on your font, it might look just like 1 or l.
# I or adu = intensity
# nI = number in the histogram bin for this intensity


def grow_bpm( bpm, bpmgrow ):
    # Make a copy of bpm so that we don't mung up any data passed in
    bpm = np.copy( bpm )
    bpmgrow = int( math.floor( bpmgrow + 0.5 ) )
    bpmkernel = np.ones( (2*bpmgrow+1, 2*bpmgrow+1) )
    bpmconv = convolve2d( bpm, bpmkernel )
    bpmconv = bpmconv[ bpmgrow:-bpmgrow, bpmgrow:-bpmgrow ]
    w = np.where( bpmconv > 0 )
    bpm[w] = 1
    return bpm


def calc_pI( params, adu ):
    a = params[0]
    σ = params[1]
    s = params[2]
    sqrt2 = math.sqrt(2)
    # The 0.5 and sqrt2 come from Bijaoui's erfc being different from scipy's
    # (The 0.5 doesn't matter much since we normalize pI anyway.)
    pI = 0.5/a * np.exp(σ**2 / (2*a**2)) * np.exp(-(adu-s)/a) * erfc( ( σ/a - (adu-s)/σ ) / sqrt2 )
    pI /= pI.sum()
    return pI


def fitfunc( params, adu, nI ):
    pI = calc_pI( params, adu )
    val = -( nI * np.log( pI ) ).sum()
    if math.isnan(val):
        val = 1e32
    return val


# This function is not used but leaving it here because
#   it would be needed if moving from scipy.optimize.differential_evolution
#   to some sort of gradient descent fitting method.
def fitfunc_jac( params, adu, nI ):
    # NOTE!  Doesn't handle the hard edge for adu<0
    a = params[0]
    σ = params[1]
    s = params[2]
    sqrt2 = math.sqrt(2)
    sqrtπ = math.sqrt(math.pi)
    # The 0.5 and sqrt2 come from Bijaoui's erfc being different from scipy's
    # pI = 0.5/a * np.exp(σ**2 / (2*a**2) ) * np.exp(-(adu-s)/a) * erfc( (σ/a - (adu-s) / σ ) / sqrt2 )

    expσaterm = np.exp(σ**2 / (2*a**2) )
    dexpσatermda = expσaterm * ( -σ**2 / a**3 )
    dexpσatermdσ = expσaterm * ( σ / a**2 )
    expIsaterm = np.exp(-(adu-s)/a)
    dexpIsatermda = (adu-s)/(a**2) * expIsaterm
    dexpIsatermds = 1/a * expIsaterm
    erfarg =  (σ/a - (adu-s) / σ ) / sqrt2
    erfintegralarg = np.exp( -erfarg**2 )
    derfargda = -σ/a**2 / sqrt2
    derfargdσ = 1/sqrt2 * ( 1/a + (adu-s)/σ**2 )
    derfargds = 1/σ / sqrt2
    erfcterm = 0.5*erfc( erfarg )
    derfctermda = -1/sqrtπ * erfintegralarg * derfargda
    derfctermdσ = -1/sqrtπ * erfintegralarg * derfargdσ
    derfctermds = -1/sqrtπ * erfintegralarg * derfargds

    pI = 1/a * expσaterm * expIsaterm * erfcterm
    dpIda = pI * ( -1/a + dexpσatermda/expσaterm + dexpIsatermda/expIsaterm + derfctermda/erfcterm )
    dpIdσ = pI * ( dexpσatermdσ/expσaterm + derfctermdσ/erfcterm )
    dpIds = pI * ( dexpIsatermds/expIsaterm + derfctermds/erfcterm )

    pIsum = pI.sum()
    dpIda = dpIda / pIsum - pI / pIsum**2 * dpIda.sum()
    dpIdσ = dpIdσ / pIsum - pI / pIsum**2 * dpIdσ.sum()
    dpIds = dpIds / pIsum - pI / pIsum**2 * dpIds.sum()
    pI /= pIsum

    val = -( nI * np.log( pI ) ).sum()
    dvalda = -( nI/pI * dpIda ).sum()
    dvaldσ = -( nI/pI * dpIdσ ).sum()
    dvalds = -( nI/pI * dpIds ).sum()

    if math.isnan( val ):
        val = 1e32
        dvalda = -1e32
        dvaldσ = -1e32
        dvalds = -1e32
    # print( f'Returning {val}' )
    return ( val, np.array( [dvalda, dvaldσ, dvalds] ) )


def estimate_single_sky( image, bpm=None, bpmgrow=3, sigcut=5.0, lowsigcut=None, nbins=None,
                         converge=0.01, maxiterations=5, workers=None, figname=None ):
    """Estimate the sky level of an image using Bijaoui, 1980, A&A, 84, 81

    https://adsabs.harvard.edu/full/1980A%26A....84...81B

    DON'T USE THIS.  It systematically underestimates the sky value,
    especially in sparse fields.  In crowded fields, it gets a whole
    hell of a lot closer for both the sky background and the sky noise
    than anything else I've tried, but if the field isn't crowded, this
    algorithm as implemented here is a disaster.  This source file is left
    here because it's used in
      tests/improc/test_bijaouisky.py::test_various_algorithms
    where you can see for yourself how it performs.

    The Bijaoui algorithm fits three parameters (s, σ, and a) to a
    histogram of pixel values in the image.  s is an estimator of the
    sky level, σ of the sky noise, and a of a parameter for the
    distribution of light from soruces on top of the source.  How well
    the algorithm works depends on how good the model of light above the
    sky level is.  For this reason, we do sigma cuts of the pixel values
    included in the histogram to limit how much bright pixels drive the
    fit.

    This routine iterates the Bijaoui fit, in hopes of converging on a
    good sigma-cut for pixels to include, as well as a good binning of
    the histogram so that bins both aren't too sparse, but also aren't
    too small to capture the sky distribution.  Each iteration, the
    value of s from the previous iteration is used as the center of the
    sigma cuts, and a factor times σ from the previous iteration around
    this center is used as the cut limits.  The values of s, σ, and a
    are used as the starting point of the fit for the next iteration.
    For the first iteration, where we don't have a previous iteration to
    pull s, σ, and a from, we use the median non-masked pixel value as
    s, the standard devian of the non-masked pxiels as σ, and 3*σ as a.

    Uses scipy.optimize.differential_evolution for the fit, which
    parallelizes (ses "workers" below).

    Empirically, at least in extragalactic fields, this algorithm seems
    to *underestimate* the sky value by a little bit (>0.1 times the
    sigma).  TODO : investigate.

    Parameters
    ----------
      image: 2d numpy array of float
        Image data

      bpm: 2d numpy aray of int, or None
        bad pixel mask.  0 = good pixel, anything else = bad pixel

      bpmgrow: int, default 3
        Expand the mask around bad pixels by this much

      sigcut: float, default 5.
        Defines the cutoff used for pixel values that go into the
        histogram to which the Bijaoui function is fit.  The low cut is
        sigcut * σ, and the high cut is sigcut * σ.

      lowsigcut: float or None
        If not None, then the histogram goes to this many σ below the
        estimated sky level of the image, and sigcut is just used for
        the high side cut.  (You might want a more generous lowsigcut
        for crowded images to make sure you haven't thrown out all of
        the sky, particularly on the first iteration.)

      nbins: int or None
        Number of bins in the histogram to which the Bijaoui function is
        fit.  If None, then nbins is the number of non-masked pixel in
        image // 8000, or 40 if that is larger.

      converge: float, default 0.01
        If two iterations in a row produce values of s, σ, and a that
        are within this factor of σ of the previous iteration, it's
        assumed to have converged.

      maxiterations: int, default 5
        Will iterate at most this many times before arbitrarily
        declaring convergence.

      workers: int or None
        If not None, the number of workers to tell
        scipy.optimize.differential_evolution to use.  If None, uses the
        value the environment variable OMP_NUM_THREADS, or 1 if that env
        var is not set.

    Returns
    -------
     a, σ, s : float, float, float

       These are the three parameters from Bijaoui 1980.  s is the sky
       level estimate.  σ is the estimation of sky noise (more or less).
       a is a parameter indicating what fraction of pixels are brighter
       and how bright they are— really a nuisance paramaeter

    """

    if lowsigcut is None:
        lowsigcut = sigcut

    if bpm is not None:
        if bpmgrow > 0:
            bpm = grow_bpm( bpm, bpmgrow )
        w = np.where( bpm == 0 )
        newimage = image[w]
        if newimage.size < 0.1*image.size:
            SCLogger.warning( f'Biajouisky masked more than 90% of {image.size} pixels in image segment; not masking '
                              f'this segment for sky subtraction' )
            newimage = None
        else:
            image = newimage

    # Initial guesses
    s = np.median( image )
    σ = np.std( image )
    a = 3 * σ
    SCLogger.debug( f"Image segment has median {s} and stdev {σ}" )

    for iteration in range(maxiterations):
        sguess = s
        σguess = σ
        aguess = a

        minrange = sguess - lowsigcut * σguess
        maxrange = sguess + sigcut * σguess
        if nbins is None:
            nbins = max( 40, image.size // 8000 )
        hist, bins = np.histogram( image, bins=nbins, range=( minrange, maxrange ) )
        pixvals = (bins[:-1] + bins[1:]) / 2.
        SCLogger.debug( f'Iteration {iteration} binsize: {pixvals[1]-pixvals[0]}' )

        # ****
        # if iteration == 0:
        #     np.save( "pixvals0.npy", pixvals )
        #     np.save( "hist0.npy", hist )
        # elif iteration == iterations-1:
        #     np.save( "pixvalsn.npy", pixvals )
        #     np.save( "histn.npy", hist )
        # ****

        _paramguess = ( aguess, σguess, sguess )

        if iteration == 0:
            bounds = [ (0., 100.*aguess),
                       (0.1*σguess, 10.*σguess),
                       (sguess - 10*σguess, sguess + 10*σguess) ]
        else:
            bounds = [ (0., 10*aguess),
                       (0.5*σguess, 2.*σguess),
                       (sguess - 2.*σguess, sguess + 2.*σguess) ]

        # ****
        # SCLogger.debug( f'In bijaouisky; initial guess: a={aguess}, σ={σguess}, s={sguess}' )
        # f, g = fitfunc_jac(_paramguess, pixvals, hist)
        # fa, junk = fitfunc_jac( ( aguess*1.001, σguess, sguess ), pixvals, hist )
        # fσ, junk = fitfunc_jac( ( aguess, σguess*1.001, sguess ), pixvals, hist )
        # fs, junk = fitfunc_jac( ( aguess, σguess, sguess*1.001 ), pixvals, hist )
        # dfda = ( fa - f ) / ( 0.001*aguess )
        # dfdσ = ( fσ - f ) / ( 0.001*σguess )
        # dfds = ( fs - f ) / ( 0.001*sguess )
        # SCLogger.debug( f'fitfunc_jac of initial guess : {f}' )
        # SCLogger.debug( f'g[0] = {g[0]}, cheesy dfda = {dfda}' )
        # SCLogger.debug( f'g[1] = {g[1]}, cheesy dfdσ = {dfdσ}' )
        # SCLogger.debug( f'g[2] = {g[2]}, cheesy dfds = {dfds}' )
        # SCLogger.debug( f'Sleeping for a long time.' )
        # time.sleep( 36000 )
        # ****

        workers = workers if workers is not None else os.getenv( "OMP_NUM_THREADS", 1 )
        res = differential_evolution( fitfunc, bounds, args=(pixvals, hist), tol=1e-6,
                                      workers=workers )
        if not res.success:
            raise Exception( f'Bijaouisky minimization failiure: {res.message}' )
        a, σ, s = res.x

        has_converged =  ( ( np.fabs( σ - σguess ) < converge * np.fabs( σ ) ) and
                           ( np.fabs( s - sguess ) < converge * np.fabs( σ ) ) and
                           ( np.fabs( a - aguess ) < converge * np.fabs( σ ) ) )

        SCLogger.debug( f'Iteration {iteration}: a={a:.3f}, σ={σ:.3f}, s={s:.3f}'
                        f'{" : converged!" if has_converged else ""}' )

        if figname is not None:
            # Notice this is plotted every iteration.  That's so that if
            #   we're doing this inside pdb with appropriate
            #   breakpoints, we can look at the plot every iteration.
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot
            fig = pyplot.figure( figsize=(8,6), tight_layout=True )
            ax = fig.add_subplot()
            ax.set_xlabel('Pixel value')
            ax.set_ylabel('N')
            ax.plot( pixvals, hist, color='blue' )
            fitvals = calc_pI( (a,σ,s), pixvals )
            norm = hist.sum() / fitvals.sum()
            ax.plot( pixvals, norm*fitvals, color='red' )
            fig.savefig(figname)
            pyplot.close( fig )

        if has_converged:
            break

    return a, σ, s


# ======================================================================

def cheb( coords, order, shape, *params ):
    if len(params) == 1:
        params = params[0]
    x = ( coords[0] - shape[0]/2. ) / (shape[0]/2.)
    y = ( coords[1] - shape[1]/2. ) / (shape[1]/2.)
    return chebval2d( x, y, np.asarray(params).reshape( order+1, order+1 ) )


def estimate_smooth_sky( image, bpm=None, bpmgrow=3, sigcut=5.0, lowsigcut=None, nbins=None,
                         converge=0.01, maxiterations=5, workers=None,
                         order=2, cutshort=10 ):
    """Fit a 2-dimensional smooth sky to image.

    DON'T USE THIS.  See docs on estimate_single_sky for explanation.

    Cuts image into roughly square boxes such that the short side of the
    image is divded into cutshort boxes.  Calls estimate_single_sky on
    each of those boxes, and then fits a 2d chebyshev polynomial of the
    specified order to the results.

    Parameters
    ----------
      image, bpm, bpmgrow, sigcut, lowsigcut, converge, maxiterations, nbins, workers
         All of these are passed on to estimate_single_sky.  image and
         bpm are chopped into boxes whose sizes is determined by
         cutshort, and estimate_single_sky is called once for each box.

         (...sort of.  bpmgrow is applied once ahead of time, and then
         estimate_single_sky is called with bpmgrow=0.)

      order: int, default 2
         Order of the polynomial fit.

      cutshort: int, default 10
         Chop the short side of the image into this many square boxes.
         (So, if the image were 2048×4096 and this was 10, cut the image
         into boxes of size ~205×205.)  Try to make this so that the
         remainder of either image dimension times the box size is
         either small, or close to the box size.  (Or don't worry
         too much if that sounds painful.)

    Returns
    -------
      sky, a, σ, s : 2d numpy array, float, float, float

        sky is the resultant sky image
        a, σ, s are the means of the estimate_single_sky returned values

    """

    # Do bpm growing once to avoid seam issues
    if bpm is not None and bpmgrow > 0:
        bpm = grow_bpm( bpm, bpmgrow )

    # First, determine the coordinates of the boxes on the image.  As is always
    #  the case in these kinds of things, most of the code is handling the special
    #  case of including the remainder pixels in the right and top boxes, making
    #  sure that the edge boxes are between (0.5, 1.5) time the regular edge size.
    if image.shape[0] < image.shape[1]:
        cut0 = cutshort
        cut1 = cut0 * ( image.shape[1] // image.shape[0] )
    else:
        cut1 = cutshort
        cut0 = cut1 * ( image.shape[0] // image.shape[1] )

    xstart = np.arange( 0, image.shape[0], image.shape[0] // cut0, dtype=int )
    if image.shape[0] - xstart[-1] < image.shape[0] // (cut0 * 2):
        xstart = xstart[:-1]
    xstop = np.empty_like( xstart )
    xstop[:-1] = xstart[1:]
    xstop[-1] = image.shape[0]
    ystart = np.arange( 0, image.shape[1], image.shape[1] // cut1, dtype=int )
    if image.shape[1] - ystart[-1] < image.shape[1] // (cut1 * 2):
        ystart = ystart[:-1]
    ystop = np.empty_like( ystart )
    ystop[:-1] = ystart[1:]
    ystop[-1] = image.shape[1]

    xstartgrid, ystartgrid = np.meshgrid( xstart, ystart )
    xstopgrid, ystopgrid = np.meshgrid( xstop, ystop )
    agrid = np.zeros_like( xstartgrid, dtype=float )
    σgrid = np.zeros_like( xstartgrid, dtype=float )
    sgrid = np.zeros_like( xstartgrid, dtype=float )

    # Next, run estimate_single_sky on each one of these boxes,
    #  collecting the results in agrid, σgrid, and sgrid

    SCLogger.debug( f'Bijaouisky estimating sky in {agrid.size} boxes with size '
                    f'{xstop[0]-xstart[0]} × {ystop[0]-ystart[0]}' )

    for i in range( agrid.shape[0] ):
        for j in range( agrid.shape[1] ):
            subimage = image[ xstartgrid[i,j]:xstopgrid[i,j], ystartgrid[i,j]:ystopgrid[i,j] ]
            subbpm = None if bpm is None else bpm[ xstartgrid[i,j]:xstopgrid[i,j], ystartgrid[i,j]:ystopgrid[i,j] ]
            a, σ, s = estimate_single_sky( subimage,
                                           subbpm,
                                           bpmgrow=0,
                                           sigcut=sigcut,
                                           lowsigcut=lowsigcut,
                                           nbins=nbins,
                                           converge=converge,
                                           maxiterations=maxiterations,
                                           workers=workers )
            agrid[i, j] = a
            σgrid[i, j] = σ
            sgrid[i, j] = s

    # Now that we've estimated the sky in the various grids, we need to fit a smooth sky to it

    SCLogger.debug( f'Done getting sky in tiles, fitting order {order} polynomial to the tiles.' )

    smoothfitfunc = lambda coords, *params : cheb( coords, order, image.shape, *params )

    meansky = sgrid.mean()
    σsky = sgrid.std()
    meanσ = σgrid.mean()
    σσ = σgrid.std()
    meana = agrid.mean()
    σa = agrid.std()
    xcoord = ( xstartgrid + xstopgrid ) / 2.
    ycoord = ( ystartgrid + ystopgrid ) / 2.
    params = np.zeros( (order+1)**2 )
    params[0] = meansky

    params, _pcov = curve_fit( smoothfitfunc, ( xcoord.reshape( [xcoord.size] ),
                                                ycoord.reshape( [ycoord.size] ) ),
                               sgrid.reshape( [sgrid.size] ), p0=params )

    allyvals, allxvals = np.meshgrid( np.arange(image.shape[1]), np.arange(image.shape[0]) )
    allxvals = allxvals.reshape( [allxvals.size] )
    allyvals = allyvals.reshape( [allyvals.size] )
    sky = np.empty( [ image.size ] )
    sky = smoothfitfunc( (allxvals,allyvals), *params )
    sky = sky.reshape( image.shape )

    SCLogger.debug( f'Bijaouisky.estimate_smooth_sky done; '
                    f'fit 0-term: {params[0]:.2f}; sky grid means: a={meana:.2f}±{σa:.2f}, '
                    f'σ={meanσ:.2f}±{σσ:.2f}, s={meansky:.2f}±{σsky:.2f}' )

    return sky, meana, meanσ, meansky


# ======================================================================

def main():
    parser = argparse.ArgumentParser( description="Estimate image sky" )
    parser.add_argument( "image", help="Image filename" )
    parser.add_argument( "bpm", help="Bad Pixel Mask filename" )
    parser.add_argument( "-n", "--hdunum", default=0, type=int,
                         help="HDU number to find image and mask data (default: 0)" )
    parser.add_argument( "-g", "--growbpm", type=int, default=3,
                         help="Grow bad pixel mask by this many pixels (def: 3)" )
    parser.add_argument( "-s", "--sigcut", type=float, default=5.0,
                         help="Histogram within this many sigma of median (default: 5.0)" )
    parser.add_argument( "-l", "--low-sigcut", type=float, default=None,
                         help="Histogram cutoff on low side (default: same as sigcut)" )
    parser.add_argument( "-o", "--order", type=int, default=0,
                         help="Order of fit (default 0, single-value sky)" )
    parser.add_argument( "-c", "--converge", type=float, default=0.01,
                         help="Converge when values are within this factor of sky noise (default 0.01)" )
    parser.add_argument( "-m", "--max-iterations", type=int, default=5,
                         help="Max Number of iterations" )
    parser.add_argument( "-w", "--write-sky", default=None,
                         help="Write sky to this file (default: don't write)" )
    parser.add_argument( "-b", "--write-bgsub", default=None,
                         help="Write sky-subtracted image to this file (default: don't write)" )
    parser.add_argument( "--save-figure", default=None,
                         help="Matplotlib figure of histogram and fit.  (Ignored if order isn't 0.)" )
    parser.add_argument( "-v", "--verbose", action='store_true', default=False,
                         help="Show debug info?" )
    args = parser.parse_args()

    if args.verbose:
        SCLogger.set_level( logging.DEBUG )
    else:
        SCLogger.set_level( logging.INFO )

    with fits.open( args.image, memmap=False ) as imfp, fits.open( args.bpm, memmap=False ) as bpmfp:
        im = imfp[args.hdunum].data
        imhdr = imfp[args.hdunum].header
        bpm = bpmfp[args.hdunum].data

    if args.order == 0:
        a, σ, s = estimate_single_sky( im, bpm, bpmgrow=args.growbpm, sigcut=args.sigcut,
                                       lowsigcut=args.low_sigcut, converge=args.converge,
                                       maxiterations=args.max_iterations, figname=args.save_figure )
        skyim = np.full_like( im, s )

    else:
        # TODO : add an argument for cutshort
        skyim, a, σ, s = estimate_smooth_sky( im, bpm, bpmgrow=args.growbpm, sigcut=args.sigcut,
                                              lowsigcut=args.low_sigcut, converge=args.converge,
                                              maxiterations=args.max_iterations, order=args.order )

    if args.write_sky is not None:
        header = fits.Header( imhdr, copy=True )
        header['COMMENT'] = f'Bijaouisky image for {os.path.basename(args.image)}'
        header['COMMENT'] = ( f'Bijaouisky growbpm={args.growbpm}, sigcut={args.sigcut:.1f}'
                              + ( f', low-sigcut={args.low_sigcut:.1f}' if args.low_sigcut is not None else '' ) )
        header['COMMENT'] = ( f'Bijaouisky order={args.order}' )
        header['COMMENT'] = ( f'Bijaouisky converge={args.converge}, max_iterations={args.iterations}' )
        hdu = fits.PrimaryHDU( data=skyim, header=header )
        hdu.writeto( args.write_sky, overwrite=True )
        SCLogger.info( f'Wrote {args.write_sky}' )

    if args.write_bgsub is not None:
        header=fits.Header( imhdr, copy=True )
        header['COMMENT'] = 'Sky subtracted with bijaouisky'
        header['COMMENT'] = ( f'Bijaouisky growbpm={args.growbpm}, sigcut={args.sigcut:.1f}'
                              + ( f', low-sigcut={args.low_sigcut:.1f}' if args.low_sigcut is not None else '' ) )
        header['COMMENT'] = ( f'Bijaouisky order={args.order}' )
        header['COMMENT'] = ( f'Bijaouisky converge={args.converge}, max_iterations={args.iterations}' )
        hdu = fits.PrimaryHDU( data=im-skyim, header=header )
        hdu.writeto( args.write_bgsub, overwrite=True )
        SCLogger.info( f'Wrote {args.write_bgsub}' )

    print( f'a={a}; σ={σ}; s={s}' )


# ======================================================================

if __name__ == "__main__":
    main()
