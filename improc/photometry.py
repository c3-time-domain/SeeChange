import numpy as np

import photutils.background
import photutils.psf
import photutils.aperture

import astropy.modeling.fitting
from astropy.table import QTable

from improc.tools import make_cutouts
from models.measurements import Measurements
from util.logger import SCLogger


def photometry( image, noise, mask, positions, apers, measurements=None,
                psfobj=None, photutils_psf=None, fwhm_pixels=None,
                dobgsub=False, innerrad=None, outerrad=None,
                cutouts=None, noise_cutouts=None, mask_cutouts=None, cutouts_size=41,
                return_cutouts=False ):
    """Do PSF and and Aperture photometry on an image.

    Some properties of the returned Measurements objects are filled and
    calculated as follows:

      bkg_per_pix : The background level measured in an annulus around
        (center_x_pixel, center_y_pixel) with inner radius innerrad and
        outer radius outerrad, or just 0. if dobgsub is False.  else 0.

      center_x_pixel, center_y_pixel : The center pixel of the cutout.
        This is pulled from what was passed in positions.

      x, y : The position of the source from psf fitting.

      flux_psf : The flux of the source from psf fitting.  If more than
        75% of the pixels in a square 2*FWHM on a side around
        (center_x_pixel,center_y_pixel) (minimum 7×7) are masked, then
        this will be NaN.

      flux_psf_err : Uncertainty on flux_psf (or NaN if flux_psf is NaN).

      psf_fit_flags : Flags returned by photutils' psf fit.  See
         https://photutils.readthedocs.io/en/stable/api/photutils.psf.PSFPhotometry.html
         and search for "flags" under "Returns" for method __call__.

      aper_radii : the list of aperture radii in pixels.  This is
        specified in the config (though perhaps in fwhm, so the pixel
        size will be calculated here).

      flux_apertures : an array of aperture-photometry flux
        corresponding to the elements of aper_radii.  These are
        background-subtracted if dobgsub is True.  Does not include an
        aperture correctin.  In units of ADU.

      flux_apertures_err : uncertainties in flux_apertures.

    Parameters
    ----------
        image : 2d ndarray
          The image data.  Indexed so that the pixel at position (ix, iy)
          is image[iy, ix].  The center of the lower-left pixel is (0.0,
          0.0).

        noise : 2d ndarray
          The 1σ noise data for image.  Whereas in most of the pipeline we
          use 1/σ² weight images, we use the noise image here because that's
          what photutils uses (and it's convenient for other cuts).

        mask : 2d ndarray of booleans
          A mask image the same size as image and noise which like what
          photutils expects: False = not masked, True = masked.

        positions : List of 2-element tuples (or similar)
          A sequence of positions.  Must be such that positions[n] is a
          2-element sequence with (x,y) positions.  (It may be that
          it must be exactly a list of tuples, depending on how
          photutils behaves.)

        psfobj : PSF
          A PSF object that goes with image.  Will use the fwhm_pixels
          attribute and get_clip method of this object.  Must specify
          either this, or both of photutils_psf and fwhm_pixels.

        photutils_psf : astropy.modeling.Model
          One of the photutils PSF classes derived from
          astropy.modeling.Model (e.g. GaussianPRF, ImagePSF).  May
          specify this instead of psfobj.  If this is specified, must
          also give fwhm_pixels.

        fwhm_pixels : float
          The PSF fwhm.  (A single value for the whole image, so a PSF
          that varies isn't fully encapsulated here).  Not needed if
          psfobj is given, needed if photutils_psf is needed.

        apers : list of float
          Aperture radii in pixels in which to do photometry.

        measurements : list of Measurement or None
          If passed, modify the appropriate fields of these Measurements
          objects based on the photometry and return that; must have the
          same length as the number of positions.  If not passed, a new
          list of Measurements will be allocated.

        dobgsub : bool, default False
          If True, do annulus background subtraction.

        innerrad, outerrad : float
          Ignored if dobgsub is false.  Required if dobgsub is true.  The
          size in pixels of the background annulus.

        cutouts : list of 2d ndarray or 3d ndarray
          Small cutouts on the image which will be used for diagnostic
          calculations.  cutouts[i] must be a 2d ndarray square image
          centered on positions[i] on the image.  If None, cutouts will
          be manually constructed here.  If given, then noise_cutouts
          and mask_cutouts must also be given.

        noise_cutouts : list of 2d ndarray
          Cutouts with 1σ noise.

        mask_cutouts : list of 2d ndarray of bool
          Masks on cutouts

        cutouts_size : int, default 41
          If cutouts is None, construct cutouts by making squares this
          length on a side.

        return_cutouts : bool, default False
           If True, also return a dictionary with the lists of cutouts.

    Returns
    -------
      measurements or measurements, cutouts

      measurements : a list of Measurement objects
        These have not been saved to the database.  Unless a
        "measurements" parameters was passed with the right fields
        pre-filled, they are not yet in a state to be saved to the
        database.  They need to have cutouts_id, index_in_sources,
        best_aperture, provenance_id, ra, dec, gallat, gallon, ecllat,
        ecllon all filled before they can be saved to the database, in
        addition to parameters that are filled by a call to
        diagnostics().

      cutouts : a dictionary with keys "image", "noise", "mask"
        Only returned if parameter return_cutouts is True.  The value of
        each element is one of the cutouts lists either passed to the
        function or created inside the function.  (If created inside the
        function, will be a 3d ndarray, not a list of 2d ndarrays.)

    """

    xvals = [ p[0] for p in positions ]
    yvals = [ p[1] for p in positions ]
    if cutouts is None:
        cutouts = make_cutouts( image, xvals, yvals, size=cutouts_size )
    if noise_cutouts is None:
        noise_cutouts = make_cutouts( noise, xvals, yvals, size=cutouts_size )
    if mask_cutouts is None:
        mask_cutouts = make_cutouts( mask, xvals, yvals, size=cutouts_size, fillvalue=True )

    if ( ( len(cutouts) != len(positions) )
         or ( len(noise_cutouts) != len(positions) )
         or ( len(mask_cutouts) != len(positions ) )
        ):
        raise ValueError( f"Length of positions, cutouts, noise_cutouts, and mask_cutouts must all match; "
                          f"Got positions={len(positions)}, cutouts={len(cutouts)}, "
                          f"noise_cutouts={len(noise_cutouts)}, mask_cutouts={len(mask_cutouts)}" )

    if dobgsub:
        # photutils LocalBackground does a sigma-clipped median
        SCLogger.debug( "Determining backgrounds..." )
        backgrounder = photutils.background.LocalBackground( innerrad, outerrad )
        bkgs = backgrounder( image, xvals, yvals, mask=mask )
    else:
        bkgs = np.zeros( ( len(positions), ), dtype=np.float32 )

    SCLogger.debug( "Getting pfs for photometry..." )
    psfs = []
    clipwid = None
    if psfobj is not None:
        if photutils_psf is not None:
            raise ValueError( "Can't specify both psfobj and photutils_psf" )
        if fwhm_pixels is not None:
            raise ValueError( "Can't specify fwhm_pixels and psfobj together." )
        for pos in positions:
            clip = psfobj.get_clip( x=np.round(pos[0]), y=np.round(pos[1]) )
            clipwid = clipwid if clipwid is not None else clip.shape[0]
            psfs.append( photutils.psf.ImagePSF( clip ) )
            fwhm_pixels = psfobj.fwhm_pixels
    elif photutils_psf is not None:
        if fwhm_pixels is None:
            raise ValueError( "photutils_psf requires fwhm_pixels" )
        clipwid = cutouts_size
        for pos in positions:
            # I hope I don't regret storing the same object repeatedly.
            # I don't *think* PSFPhotometry modifies the psf object.
            psfs.append( photutils_psf )
    else:
        raise ValueError( "Must give either psfobj or photutils_psf" )

    # Psf photometry first, to find optimum positions.  Because
    #    photutils doesn't understand psfex psfs, it's not actually
    #    going to interpolate right.  It will be using a bicubic spline
    #    interpolation on the image-scale PSF, instead of the
    #    sync-interpolation on the optimum-scale PSF that psfex defines.
    #    It's a pity that there isn't a PSF model for photutils that
    #    understands PSFEX psfs.  Hopefully what we have here will be
    #    good enough for our purposes, but we should verify that!
    # Sadly, we can't just pass all the positions in a single call the
    #   way we do with aperture photometry, because the interface to
    #   photutils lets us pass in a single PSF model, but we're using a
    #   different one for each detection because we used a
    #   position-variable PSF from psfex.  In practice, it doesn't seem
    #   to be too terribly slow.
    # (It should be possible to fix both these issues by writing a
    #   Fittable2dModel class to use with photutils' PSF photometry
    #   using our models/psf.py class and get_clip.  It might be slower.
    #   It would be effort to write it.)

    SCLogger.debug( "Doing psf photometry...." )

    measurements = []
    for i, pos in enumerate( positions ):
        photor = photutils.psf.PSFPhotometry(
            psfs[i],
            clipwid,
            aperture_radius=fwhm_pixels
        )
        init_params = QTable()
        init_params['x'] = [ pos[0] ]
        init_params['y'] = [ pos[1] ]
        init_params['local_bkg'] = [ bkgs[i] ]

        m = Measurements( aper_radii=apers,
                          flux_apertures=[np.nan] * len(apers),
                          flux_apertures_err=[np.nan] * len(apers),
                          bkg_per_pix=bkgs[i],
                          center_x_pixel=int(np.round(pos[0])),
                          center_y_pixel=int(np.round(pos[1])),
                         )

        # PSFPhotometry is going to spit at us if we feed it a fully masked
        #   object.  Try to detect that ahead of time and skip the photometry
        #   if that might happen.
        r = max( 3, int( np.round( 2. * fwhm_pixels ) ) )
        r = r if ( r % 2 == 1 ) else r + 1
        ninr = ( 2*r + 1 ) **2
        nmask = mask[ m.center_y_pixel-r:m.center_y_pixel+r+1,
                      m.center_x_pixel-r:m.center_x_pixel+r+1 ].sum()
        if nmask > 0.75 * ninr:
            m.x = pos[0]
            m.y = pos[1]
            m.flux_psf = np.nan
            m.flux_psf_err = np.nan
            m.psf_fit_flags = 9         # 1 = "≥1 pixel masked", 8 = "fit did not converge"
        else:
            photresult = photor( image, mask=mask, error=noise, init_params=init_params )
            m.x = photresult['x_fit'][0]
            m.y = photresult['y_fit'][0]
            m.flux_psf = photresult['flux_fit'][0]
            m.flux_psf_err = photresult['flux_err'][0]
            m.psf_fit_flags = photresult['flags'][0]
        measurements.append( m )

    SCLogger.debug( "...done doing psf photometry." )

    # Aperture photometry we can do in one go for each
    #   aperture, since we're using the same aperture radii
    #   for everybody.
    # This does point out an inconsitency in our approach:
    #   we have a single aperture correction for the whole
    #   image, which really only is true if the psf doesn't
    #   vary with position, but above we did PSF photometry
    #   with a position-variable PSF.
    # This inconsistency, together with limitations in the
    #   PSF interpolation, can be tossed into the basket
    #   labeled "This is a detction pipeline, we aren't
    #   promising photometrty to better than a couple of
    #   percent."

    SCLogger.debug( "Building apertures for photometry..." )
    aperobjs = [ photutils.aperture.CircularAperture( positions, r=r ) for r in apers ]
    for i in range(len(apers)):
        SCLogger.debug( f"...aperture photometry with r={apers[i]}" )
        apphot = photutils.aperture.aperture_photometry( image,
                                                         aperobjs[i],
                                                         error=noise,
                                                         mask=mask )
        for j in range( len(measurements) ):
            if dobgsub:
                measurements[j].flux_apertures[i] = apphot['aperture_sum'][j] - bkgs[j] * aperobjs[i].area
            else:
                measurements[j].flux_apertures[i] = apphot['aperture_sum'][j]
            measurements[j].flux_apertures_err[i] = apphot['aperture_sum_err'][j]
    SCLogger.debug( "...done with aperture photometry" )

    if return_cutouts:
        return measurements, { 'image': cutouts,
                               'noise': noise_cutouts,
                               'mask': mask_cutouts }
    else:
        return measurements


def diagnostics( measurements, cutouts, noise_cutouts, mask_cutouts, fwhm_pixels,
                 diagdist=2, distunit='fwhm', n_sigma_outlier=2. ):
    """Measure morphological and diagnostic properties of cutouts and Measurements.

    At the end of this function, several fields of each Measurement
    object in the measurements list will be filled, including gfit_x,
    gfit_y, major_width, minor_width, position_angle, nbadpix, negfrac,
    negfluxfrac.  (The is_bad field will not be set; that needs to be
    set by something that has thresholds for diagnostics
    (e.g. pipline/measuring.py), whereas this routine just calculates
    them.)

    The fields of Measurements are calculated as follows.

      gfit_x, gfit_y : Center position of a 2d Gaussian fit over the
        source.

      minor_width, major_width : The minor axis and major axis FWHM from
        the 2d Gaussian fit.

      position_angle : The angle of the major axis of the Gaussian fit
        relative to the x-axis.  (...if the code below properly
        interprets the return from astropy's fit.)

      nbadpix : The number of masked pixels within diagdist of
        round(x,y) that are masked.  (Within that distance along either
        x or y; it looks in a square of side 2*diag_dist+1.)

      negfrac : The number of significantly negative pixels divided by
        significantly positive pixels within the same square used for
        nbadpix.  "Signficantly" is defined more than as n_sigma_outlier
        times the noise value of the pixel (from noise_cutouts) away
        from 0 on the background subtracted cutout.  Does not include
        masked pixels.  Larger values indicate more significantly
        negative pixels (and thus a sign of a likely bad candidate,
        e.g. a dipole).  Set to 1e32 if there are no significantly
        positive pixels in the diagnostic square.

      negfluxfrac : Looking at the same pixels selected in negfrac, the
        absolute value of the sum of the negative pixel values divided by
        the sum of the positive pixel values.  Set to 1e32 if there are no
        significantly positive pixels in the diagnostic square.

    Parameters
    ----------
      measurements : list of Measurement
        Such as might have been returned from photometry (above).  The
        Measurement objects in this list will be modified; see above.

      cutouts : list of 2d ndarrays or 3d ndarray
        cutouts[i] is a square cutout from the image that goes with
        measurements[i].

      noise_cutouts : list of 2d ndarrays or 3d ndarray
        1σ noise (not weights!) on cutouts.

      mask_cutouts : list of 2d ndarrays of bool or 3d ndarray of bool
        Masks on cutouts; True = pixel is masked, False = pixel is not
        masked.  (Using this rather than our flags bitmask because this
        is what photutils expects.)

      fwhm_pixels : The FWHM of the seeing in pixels

      diagdist : float, default 2
        For some diagnostics, look in a square on the cutout that is
        2*diagdist+1 on a side, centered on the pixel where the object is
        found.  (This is the pixel defined by measurements[i].x,
        measurements[i].y, not the center of the cutout, though it
        should be close. )  This square is used for the bad pixel
        counts and the negative / positive pixel ratios

      distunit : "fwhm" or "pixel"
        The units of baddist.

      n_sigma_outlier : float, default 2.
        A pixel will be counted in the negfrac and negfluxfrac
        diagnostics if its flux value is at least this many times the
        noise away from zero.

    Returns
    -------
      nothing, but measurements is modified; see above

    """

    if ( ( len(cutouts) != len(measurements) )
         or ( len(noise_cutouts) != len(measurements) )
         or ( len(mask_cutouts) != len(measurements ) )
        ):
        raise ValueError( f"Number of cutouts, noise_cutouts, and mask_cutouts must match number of measurements; got "
                          f"measurements={len(measurements)}, cutouts={len(cutouts)}, "
                          f"noise_cutouts={len(noise_cutouts)}, mask_cutouts={len(mask_cutouts)}" )

    if distunit not in [ "fwhm", "pixel" ]:
        raise ValueError( f"distunit must be 'fwhm' or 'pixel', not '{distunit}'" )

    SCLogger.debug( "Calculating morphological parameters and diagnostics..." )
    dist = diagdist if distunit == 'pixel' else diagdist * fwhm_pixels
    dist = int( np.round( dist ) )
    xvals, yvals = np.meshgrid( range(0,cutouts[0].shape[1]), range(0,cutouts[0].shape[0]) )
    for i, ( m, cutout, cutout_noise, cutout_mask ) in enumerate( zip( measurements, cutouts, noise_cutouts,
                                                                       mask_cutouts ) ):
        # Leave this code here for now; we'll probably
        #   remove it later, but I'm hedging my bets.
        # I found that the moment calculations were very
        #   unstable.  They're a nice idea, but highly
        #   susceptible to noisy images, which means
        #   they won't be very good on anything that's
        #   anywhere near our detection limit.
        if False:
            # Notice that photutils.morphology.data_properties doesn't take
            #   a noise image....  The moments are defined in terms of just
            #   the image, yes, but notice that there *is* a mask, and you
            #   could define (at least) a centroid that weights by noise.
            # morpho = photutils.morphology.data_properties( cutout - m.bkg_per_pix, mask=cutout_mask )
            # ...
            # photutils.morphology.data_properties sets all negative pixels to positive for purposes
            #   of moment calculation.  I *THINK*.  Hard to say, because the documentation on
            #   SourceCatalog, which data_properties points to, talks about source segments, but
            #   data_properties has no concept of that.  As such, I'm not 100% sure what it's actually
            #   doing.  Alas.  Calculate them ourselves, so we know what's happened.

            m00 = ( cutout[ ~cutout_mask ] - m.bkg_per_pix ).sum()
            m10 = ( ( cutout - m.bkg_per_pix ) * yvals )[ ~cutout_mask ].sum()
            m01 = ( ( cutout - m.bkg_per_pix ) * xvals )[ ~cutout_mask ].sum()
            m20 = ( ( cutout - m.bkg_per_pix ) * yvals * yvals )[ ~cutout_mask ].sum()
            m02 = ( ( cutout - m.bkg_per_pix ) * xvals * xvals )[ ~cutout_mask ].sum()
            m11 = ( ( cutout - m.bkg_per_pix ) * xvals * yvals )[ ~cutout_mask ].sum()

            # Dealing with moments.
            # See: https://en.wikipedia.org/wiki/Image_moment
            #  and: hacks/rknop/moments.wxmx (a wxMaxima file)
            #
            # For a Gaussian, the image is:
            #   im(x,y) = ( A / (2π σx σy) ) exp( - ( xr^2/(2 σx^2) + ( yr^2/(2 σy^2) ) ) )
            # where
            #   xr =  x cos(θ) + y sin(θ)
            #   yr = -x sin(θ) + y cos(θ)
            #
            # A is the total flux in the Gaussian
            # θ is the rotation of the profile; consider it the angle
            #    between the x-axis and the profile, in a counter-clockwise
            #    fashion (i.e. rotate from the +x axis towards the +y axis)
            #
            # Define some moments by:
            #   A = Σ f(x,y)
            #   cxx = Σ (x - x0)^2 * f(x,y)
            #   cyy = Σ (y - y0)^2 * f(x,y)
            #   cxy = Σ (x - x0) * (y - y0) * f(x,y0)
            # where sums are over x and y and (x0,y0) is the centroid.
            #
            # These can be found in the photutils.morpology result (see above):
            #   A = morpho.moments_central[0][0]
            #   cxx = morpho.moments_central[0][2]
            #   cyy = morpho.moments_central[2][0]
            #   cxy = morpho.moments_central[1][1]
            #
            # Define reduced moments by rab = cab / A
            #
            # You can reconstruct σx and σy with:
            #
            # σ1² = ( rxx + ryy ) / 2 + sqrt( 4 rxy^2 + ( rxx - ryy )^2 ) / 2
            # σ2² = ( rxx + ryy ) / 2 - sqrt( 4 rxy^2 + ( rxx - ryy )^2 ) / 2
            #
            # Those give the major (σ1) and minor (σ2) axes sizes, corresponding
            # to the 1σ Gaussian widths.
            #
            # Define φ as the angle between the major axis of the distribution
            #   and the (y if cyy > cxx else x)-axis, in a counter-clockwise direction.
            #
            # With this definition, then
            #   ( π/4 < φ < π/4 ) if cxx != cyy else π/4 if cxy > 0 else -π/4 if cxy < 0 else 0
            # (if cxy = 0, φ is not well-defined and we may as well call it 0.)
            #
            # You can calculate φ with:
            #   φ = 1/2 arctan( 2 rxy / ( rxx - ryy ) )
            # if rxx != ryy, else:
            #   φ = π/4
            #
            # You can then get θ back (sorta... choose a thing between -π/2 and π that looks the same)
            #   θ = φ if rxx > ryy
            #   else θ = π/2 - θ

            m.centroid_x = m01 / m00
            m.centroid_y = m10 / m00
            rxx = ( m02 - m.centroid_x * m01 ) / m00
            ryy = ( m20 - m.centroid_y * m10 ) / m00
            rxy = ( m11 - m.centroid_y * m01 ) / m00

            m.centroid_x += m.center_x_pixel - cutout.shape[1] // 2
            m.centroid_y += m.center_y_pixel - cutout.shape[0] // 2

            m.major_width = np.sqrt( ( ( rxx + ryy ) + np.sqrt( 4 * rxy**2 + ( rxx - ryy )**2 ) ) / 2. )
            m.minor_width = ( ( rxx + ryy ) - np.sqrt( 4 * rxy**2 + ( rxx - ryy )**2 ) ) / 2.
            m.minor_width = np.sqrt( m.minor_width ) if m.minor_width > 0. else 0.
            if rxx == ryy:
                theta = np.pi/4. if rxy > 0 else -np.pi/4. if rxy < 0 else 0.
            else:
                phi = 0.5 * np.atan( 2 * rxy / ( rxx - ryy ) )
                theta = phi if rxx > rxy else np.pi/2. - phi
                # Make the angle between -π/2 and π/2
                if theta > np.pi / 2.:
                    theta -= np.pi
            m.position_angle = theta

        # Fit a 2d Gaussian over the detected source to get some
        # diagnostics.

        # First, gotta turn the cutout into what astropy fitting
        # expects, which isn't exactly the same as what photutils
        # wanted.  It can't handle any nans, and it wants a 1/σ image it
        # calls weight (yes, 1/σ, not 1/σ²).
        gcutout = np.copy( cutout ) - m.bkg_per_pix
        bad = ( ( cutout_mask ) |
                ( np.isnan( gcutout ) ) |
                ( np.isinf( gcutout ) ) |
                ( cutout_noise <= 0. ) |
                ( np.isnan( cutout_noise ) ) |
                ( np.isinf( cutout_noise ) ) )
        gcutout[ bad ] = 0.
        gcutout_weight = 1. / cutout_noise
        gcutout_weight[ bad ] = 0.

        fluxguess = m.flux_psf if ( not np.isnan(m.flux_psf) ) else gcutout[~bad].sum()
        initgauss = photutils.psf.GaussianPSF( flux=fluxguess, x_0=cutout.shape[1] // 2, y_0=cutout.shape[0] // 2,
                                               x_fwhm=fwhm_pixels, y_fwhm=fwhm_pixels, theta=0. )
        initgauss.x_fwhm.fixed = False
        initgauss.y_fwhm.fixed = False
        initgauss.theta.fixed = False
        gfitter = astropy.modeling.fitting.TRFLSQFitter()
        try:
            fitgauss = gfitter( initgauss, xvals, yvals, gcutout, weights=gcutout_weight )
            theta = fitgauss.theta.value * np.pi / 180.
            m.gfit_x = fitgauss.x_0.value + m.center_x_pixel - cutout.shape[1] // 2
            m.gfit_y = fitgauss.y_0.value + m.center_y_pixel - cutout.shape[0] // 2
            if fitgauss.x_fwhm > fitgauss.y_fwhm:
                m.major_width = fitgauss.x_fwhm.value
                m.minor_width = fitgauss.y_fwhm.value
                theta -= np.pi/2.
            else:
                m.major_width = fitgauss.y_fwhm.value
                m.minor_width = fitgauss.x_fwhm.value
            # Try to make theta betwen -π/2 and π/2.
            while theta >= np.pi:
                theta -= 2. * np.pi
            while theta < -np.pi:
                theta += 2. * np.pi
            if theta > np.pi/2:
                theta -= np.pi
            if theta <= -np.pi/2.:
                theta += np.pi
            m.position_angle = theta
        except Exception:
            # Sometimes the gaussian fitter fails.  I'm getting
            #   exceptions saying that there are nan or inf values, but
            #   I'm not seeing it when I look at the actual contents of
            #   the array, so I don't know what's going on.  We're just
            #   going to assume that this is a pathological case where
            #   things are bad anyway and just fill in some values that
            #   probably fail any cuts. It would be better to set a flag
            #   saying the gaussian fit failed.  See Issue #422
            m.gfit_x = -9999
            m.gfit_y = -9999
            m.major_width = 0.
            m.minor_width = 0.
            m.position_angle = 0.

        # Figure out positions on cutouts for the bad pixel and negative fraction diagnostics
        ixc = int( np.round( m.x ) - m.center_x_pixel + ( cutouts[i].shape[1] // 2 ) )
        iyc = int( np.round( m.y ) - m.center_y_pixel + ( cutouts[i].shape[0] // 2 ) )
        x0 = max( 0, ixc - dist )
        x1 = min( ixc + dist + 1, cutouts[i].shape[1] )
        y0 = max( 0, iyc - dist )
        y1 = min( iyc + dist + 1, cutouts[i].shape[0] )

        # Make subsetted cutouts that include just this size for those diagnostics
        sub_cutout = cutouts[i][ y0:y1, x0:x1 ]
        sub_noise = noise_cutouts[i][ y0:y1, x0:x1 ]
        sub_mask = mask_cutouts[i][ y0:y1, x0:x1 ]

        m.nbadpix = sub_mask.sum()
        wneg = ( ~sub_mask ) & ( sub_cutout - m.bkg_per_pix < -n_sigma_outlier * sub_noise )
        wpos = ( ~sub_mask ) & ( sub_cutout - m.bkg_per_pix > n_sigma_outlier * sub_noise )
        nneg = wneg.sum()
        npos = wpos.sum()
        fluxneg = -(sub_cutout[ wneg ] - m.bkg_per_pix).sum()
        fluxpos = (sub_cutout[ wpos ] - m.bkg_per_pix).sum()
        m.negfrac = ( 0 if ( npos == 0 and nneg == 0 )
                      else 1e32 if npos == 0
                      else nneg / npos )
        m.negfluxfrac = ( 0 if ( fluxpos == 0 and fluxneg == 0 )
                          else 1e32 if fluxpos == 0
                          else fluxneg / fluxpos )

    SCLogger.debug( "...done calculating diagnostics." )



def photometry_and_diagnostics( image, noise, mask, positions, apers, measurements=None,
                                psfobj=None, photutils_psf=None, fwhm_pixels=None,
                                dobgsub=False, innerrad=None, outerrad=None,
                                cutouts=None, noise_cutouts=None, mask_cutouts=None, cutouts_size=41,
                                diagdist=2, distunit='fwhm' ):

    """Calculate photometry and associated diagnostics at various positions on an image.

    All parameters are passed on to photometry() or diagnostics(); see those functions for documentation.

    Returns
    -------
      measurements : list of Measurements
        The return value from photometry(), with further fields filled by diagnostics()

    """

    measurements, co = photometry( image, noise, mask, positions, apers, measurements=measurements,
                                   psfobj=psfobj, photutils_psf=photutils_psf, fwhm_pixels=fwhm_pixels,
                                   dobgsub=dobgsub, innerrad=innerrad, outerrad=outerrad,
                                   cutouts=cutouts, noise_cutouts=noise_cutouts, mask_cutouts=mask_cutouts,
                                   cutouts_size=cutouts_size, return_cutouts=True )

    fwhm_pixels = psfobj.fwhm_pixels if psfobj is not None else fwhm_pixels
    diagnostics( measurements, co['image'], co['noise'], co['mask'], fwhm_pixels, diagdist=diagdist, distunit=distunit )

    return measurements
