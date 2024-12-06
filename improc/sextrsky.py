import math
import argparse
import logging
import numpy as np
from scipy.signal import medfilt2d
from scipy.interpolate import RectBivariateSpline
from astropy.io import fits

from util.logger import SCLogger


def single_sextrsky( imagedata, maskdata=None, sigcut=3 ):
    """Estimate sky and sky sigma of imagedata (ignoreing nonzero maskdata pixels)

    Iteratively sigma clips from the median until no more pixels get
    clipped.  Then, from the remaining pixels, estimates the sky as
    2.5*med - 1.5*mean, unless the mean and median differ by more than
    0.3 of the mean, in which case estimate it just as the median.
    Estimate sky sigma as 1.4826*median(|data-sky|).  (This is the
    SExtractor algorithm.)

    Parameters
    ----------
      imagedata: 2d numpy array
         image data

      maskdata: 2d numpy array, optional
         flags array where 0=good, otherwise bad

      sigcut: float, default 3
         Number of sigmas to clip each iteration (see above).

    Returns:
    -------
      sky, skysig

      sky: float
        The sky value

      skysig:
        The 1σ sky noise

    """
    done = False
    if maskdata is None:
        maskdata = np.zeros_like( imagedata, dtype=np.uint8 )
    w = maskdata == 0
    lastn = imagedata.size
    while not done:
        med = np.median( imagedata[ w ] )
        mean = np.mean( imagedata[ w ] )
        sdev = np.std( imagedata[ w ] )
        w = ( ( maskdata == 0 ) & ( np.abs( imagedata - med ) < sigcut * sdev ) )
        SCLogger.debug( f'single_sextrsky: med={med:.2f}, mean={mean:.2f}, sdev={sdev:.2f}, n={w.sum()}' )
        if w.sum() > lastn:
            SCLogger.warning( "single_sextrsky: n increased" )
        if w.sum() == lastn:
            done = True
        lastn = w.sum()

    if math.fabs( mean - med ) / mean > 0.3:
        SCLogger.debug( f'mean={mean}, med={med}, using just median for sky estimate' )
        sky = med
    else:
        sky = 2.5*med - 1.5*mean
    skysig = 1.4826 * ( np.median( np.abs( imagedata[w] - sky ) ) )
    return sky, skysig


def sextrsky( imagedata, maskdata=None, sigcut=3, boxsize=200, filtsize=3 ):
    """Estimate sky using an approximation of the SExtractor algorithm.

    Divides the image into boxes of size boxsize. (For the last box
    along each axis, it will be between 0.5 and 1.5 the boxsize.)  Calls
    single_sextrsky for each box, then median filters the results, then
    fits a bicubic spline to the median filtered results to generate a
    sky image.

    Parameters
    ----------
      imagedata: 2d numpy array
         image data

      maskdata: 2d numpy array, optional
         flags array where 0=good, otherwise bad

      sigcut: float, default 3.
         See single_sextrsky

      boxsize: int, default 200
         The size of boxes into which to divide the image

      filtsize: int, default 3
         The size of the median filter to apply to the sky values
         determined in each box.  Currently must be either 1 or 3.

    Returns:
    -------
      sky, skysig

      sky: 2d numpy array
        The sky image

      skysig: float
        The median 1σ sky noises from all the boxes

    """

    filtsize = int(filtsize)
    if filtsize%2 == 0:
        filtsize += 1

    if ( filtsize != 1 ) and ( filtsize != 3 ):
        raise ValueError( "Code currently has hardcoded filtsize=1 or 3 assumption" )

    xgrid0 = np.arange( 0, imagedata.shape[0], boxsize )
    if imagedata.shape[0] - xgrid0[-1] < boxsize/2:
        xgrid0 = xgrid0[:-1]
    xgrid1 = xgrid0 + boxsize
    if xgrid1[-1] > imagedata.shape[0]:
        xgrid1[boxsize] = imagedata.shape[0]
    ygrid0 = np.arange( 0, imagedata.shape[1], boxsize )
    if imagedata.shape[1] - ygrid0[-1] < boxsize/2:
        ygrid0 = ygrid0[:-1]
    ygrid1 = ygrid0 + boxsize
    if ygrid1[-1] > imagedata.shape[1]:
        ygrid1[boxsize] = imagedata.shape[1]
    xgrid = ( xgrid0 + (xgrid1-1) ) / 2.
    ygrid = ( ygrid0 + (ygrid1-1) ) / 2.

    backvals = np.empty( [ len(xgrid), len(ygrid) ] )
    skysigvals = np.empty( [ len(xgrid), len(ygrid) ] )

    for i in range(len(xgrid0)):
        for j in range(len(ygrid0)):
            subim = imagedata[ xgrid0[i]:xgrid1[i], ygrid0[j]:ygrid1[j] ]
            submask = None if maskdata is None else maskdata[ xgrid0[i]:xgrid1[i], ygrid0[j]:ygrid1[j] ]
            sky, skysig = single_sextrsky( subim, submask, sigcut=sigcut )
            backvals[i, j] = sky
            skysigvals[i, j] = skysig

    if ( filtsize > 1 ):
        filt_backvals = medfilt2d( backvals, kernel_size=filtsize )
        # scipy's medfilt2d is going to put 0 at the corners, so put something in
        # TODO : this is only true for filtsize=3.  If filtsize is bigger,
        #   more things get set to 0!
        filt_backvals[ 0,  0] = np.median( backvals[:filtsize//2+1, :filtsize//2+1] )
        filt_backvals[ 0, -1] = np.median( backvals[:filtsize//2+1, -filtsize//2-2:] )
        filt_backvals[-1,  0] = np.median( backvals[-filtsize//2-2:, :filtsize//2+1] )
        filt_backvals[-1, -1] = np.median( backvals[-filtsize//2-2:, -filtsize//2-2:] )
        backvals = filt_backvals

    # Spline extrapolation is a complete disaster.  Avoid this by anchoring the spline

    anchored_backvals = np.empty( [ backvals.shape[0]+2, backvals.shape[1]+2 ] )
    anchored_backvals[ 1:-1, 1:-1 ] = backvals
    anchored_backvals[ 0, 1:-1] = backvals[ 0, :]
    anchored_backvals[-1, 1:-1] = backvals[-1, :]
    anchored_backvals[1:-1,  0] = backvals[:,  0]
    anchored_backvals[1:-1, -1] = backvals[:, -1]
    anchored_backvals[ 0, 0] = ( anchored_backvals[ 1, 0] + anchored_backvals[ 0, 1] ) / 2.
    anchored_backvals[ 0,-1] = ( anchored_backvals[ 1,-1] + anchored_backvals[ 0,-2] ) / 2.
    anchored_backvals[-1, 0] = ( anchored_backvals[-1, 1] + anchored_backvals[-2, 0] ) / 2.
    anchored_backvals[-1,-1] = ( anchored_backvals[-1,-2] + anchored_backvals[-2,-1] ) / 2.
    anchored_xgrid = np.empty( xgrid.size+2 )
    anchored_xgrid[1:-1] = xgrid
    anchored_xgrid[0] = 0
    anchored_xgrid[-1] = imagedata.shape[0]
    anchored_ygrid = np.empty( ygrid.size+2 )
    anchored_ygrid[1:-1] = ygrid
    anchored_ygrid[0] = 0
    anchored_ygrid[-1] = imagedata.shape[1]

    interpoler = RectBivariateSpline( anchored_xgrid, anchored_ygrid, anchored_backvals,
                                      bbox=[ 0, imagedata.shape[0], 0, imagedata.shape[1] ] )
    sky = interpoler( np.arange(imagedata.shape[0]), np.arange(imagedata.shape[1]) )

    return sky, np.median(skysigvals)

# ======================================================================


def main():
    parser = argparse.ArgumentParser( description="Estimate image sky using sextractor algorithm" )
    parser.add_argument( "image", help="Image filename" )
    parser.add_argument( "-m", "--mask", help="Bad Pixel Mask filename (default: None)", default=None )
    parser.add_argument( "-c", "--sigcut", default=3., type=float,
                         help="σ cut for clipping about median (default 3)" )
    parser.add_argument( "-n", "--hdunum", default=0, type=int,
                         help="Which HDU of image and bpm to use (default: 0)" )
    parser.add_argument( "-b", "--boxwid", default=200, type=int,
                         help="Width of box to use (0 to get one sky for whole image; default 200)" )
    parser.add_argument( "-f", "--filtsize", default=3, type=int,
                         help="Size of median filter (default 3; ignored if boxwid=0)" )
    parser.add_argument( "-o", "--output", default=None,
                         help="Write sky-subtracted image to this file (default: don't save)" )
    parser.add_argument( "-s", "--write-sky", default=None,
                         help="Write sky image to this file (default: don't save)" )
    parser.add_argument( "-v", "--verbose", default=False, action="store_true",
                         help="Show extra log info" )
    args = parser.parse_args()

    if args.verbose:
        SCLogger.setLevel( logging.DEBUG )
    else:
        SCLogger.setLevel( logging.INFO )

    with fits.open( args.image ) as hdu:
        imagedata = hdu[args.hdunum].data
        imageheader = hdu[args.hdunum].header
    if args.mask is None:
        bpmdata = None
    else:
        with fits.open( args.mask ) as hdu:
            bpmdata = hdu[args.hdunum].data

    if args.boxwid == 0:
        sky, sig = single_sextrsky( imagedata, bpmdata, sigcut=args.sigcut )
        skyim = np.full_like( imagedata, sky )
    else:
        skyim, sig = sextrsky( imagedata, bpmdata, sigcut=args.sigcut,
                               filtsize=args.filtsize, boxsize=args.boxwid )
        sky = np.median( skyim )
    SCLogger.debug( f'Sky: {sky}; σ: {sig}' )

    if args.output is not None:
        hdr = imageheader.copy()
        hdr['COMMENT'] = "Sky subtracted with sextractor sky algorithm"
        hdr['COMMENT'] = f"boxwid={args.boxwid}, filtsize={args.filtsize}, sigcut={args.sigcut:.2f}"
        hdr['COMMENT'] = f"Median sky: {sky:.2f}, sigma: {sig:.2f}"
        hdu = fits.PrimaryHDU( data=imagedata-skyim, header=hdr )
        hdu.writeto( args.output, overwrite=True )
    if args.write_sky is not None:
        hdr = imageheader.copy()
        hdr['COMMENT'] = "Sky estimated with sextractor sky algorithm"
        hdr['COMMENT'] = f"boxwid={args.boxwid}, filtsize={args.filtsize}, sigcut={args.sigcut:.2f}"
        hdr['COMMENT'] = f"Median sky: {sky:.2f}, sigma: {sig:.2f}"
        hdu = fits.PrimaryHDU( data=skyim, header=hdr )
        hdu.writeto( args.write_sky, overwrite=True )


# ======================================================================

if __name__ == "__main__":
    main()
