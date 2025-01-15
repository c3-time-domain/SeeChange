import numpy as np

import photutils.background
import photutils.psf
import photutils.aperture

from astropy.table import QTable

from models.measurements import Measurements
from util.logger import SCLogger

def photometry_and_diagnostics( image, noise, mask, positions, psfobj, apers, baddist=None,
                                dobgsub=False, innerrad=None, outerrad=None,
                                cutouts=None, noise_cutouts=None, mask_cutouts=None, cutouts_rad=20 ):
    """Calculate photometry and associated diagnostics at various positions on an image.

    Parameters
    ----------
        image : 2d ndarray
          The image data.

        noise : 2d ndarray
          The 1σ noise data for image.  Whereas in most of the pipeline we
          use 1/σ² weight images, we use the noise image here because that's
          what photutils uses (and it's convenient for other cuts).

        mask : 2d ndarray of booleans
          A mask image the same size as image and noise which like what
          photutils expects: False = not masked, True = masked.

        psfobj : PSF
          A PSF object that goes with image.  Will use the fwhm_pixels
          attribute and get_clip method of this object.

        apers : list of float
          Aperture radii in pixels in which to do photometry.

        baddist : int or None
          See the "diagnostics" return below.

        positions : list of (x, y)
          Positions at which to do photometry.  These are initial positions.
          The lower-left corner of the lower-left pixel is at (-0.5, -0.5).
          The pixel at position (ix, iy) on image is image[iy, ix].

        dobgsub : bool, default False
          If True, do annulus background subtraction.

        innerrad, outerrad : float
          Ignored if dobgsub is false.  Required if dobgsub is true.  The
          size in pixels of the background annulus.

        cutouts : list of 2d ndarray
          Small cutouts on the image which will be used for diagnostic
          calculatoins.  If None, cutouts will be manually constructed here.
          If given, then noise_cutouts and mask_cutouts must also be given.

        noise_cutouts : list of 2d ndarray
          Cutouts with 1σ noise.

        mask_cutouts : list of 2d ndarray of bool
          Masks on cutouts

        cutouts_rad : int, default 20
          If cutouts is None, construct cutouts by making squares ±this many
          pixels in x and y on the image.

    Returns
    -------
      measurements, diagnostics

      measurements : a list of Measurement objects
        These have not been saved to the database, and are not yet in a
        state to be saved to the database.  They need to have
        cutouts_id, index_in_sources, best_aperture, and provenance_id
        filled before they can be saved to the database.

      diagnostics : a dictionary of ndarrays, each of length len(positions)
        Diagnostic quantities, designed so that larger = worse.  Keys
        in the dictionary:

          width_ratio : 2√(2ln2) * width / image FHWM_pix  (ratio of object size to seeing)
          elongation : major axis / minor axis, where the axes are determined from the moments on the cutouts
                Set to 1e32 if the minor axis length comes out to zero (somehow)
          numbadpix : number of bad pixels within baddist in either x or y of the center of the cutout.
                      if baddist is None, look at the whole cutout
          negfrac : number of <-2σ pixels divided by number of >2σ pixels.  Set to 0 if there are no
             pixels either <-2σ or >2σ, else set to 1e32 if there are no pixels >2σ.
          negfluxfrac : |total flux| in <-2σ pixels divded by total fix in >2σ pixels.  Set to 0 if there
             are no pixels either <-2σ or >2σ, else set to 1e32 if there are no pixels >2σ

    """

    if cutouts is None:
        SCLogger.debug( "photometry_and_diagnostics making cutouts..." )
        cutouts = []
        noise_cutouts = []
        mask_cutouts = []
        for (x, y) in positions:
            im = np.zeros( ( 2*cutouts_rad+1, 2*cutouts_rad+1 ), dtype=image.dtype )
            no = np.zeros( ( 2*cutouts_rad+1, 2*cutouts_rad+1 ), dtype=noise.dtype )
            ma = np.full( ( 2*cutouts_rad+1, 2*cutouts_rad+1 ), False, dtype=bool )
            # Figure out the limits with all the usual special-case pain of detecting
            #   cutouts that hang off of the image
            x0 = int( np.round(x) - cutouts_rad )
            outx0 = 0
            x1 = x0 + 2 * cutouts_rad + 1
            outx1 = 2 * cutouts_rad + 1
            y0 = int( np.round(y) - cutouts_rad )
            outy0 = 0
            y1 = y0 + 2 * cutouts_rad + 1
            outy1 = 2 * cutouts_rad + 1
            if x0 < 0:
                outx0 = -x0
                x0 = 0
                ma[ :, 0:outx0 ] = True
                no[ :, 0:outx0 ] = np.nan
            if x1 > image.shape[1]:
                outx1 -= ( x1 - image.shape[1] )
                x1 = image.shape[1]
                ma[ :, outx1:2*cutouts_rad+1 ] = False
                no[ :, outx1:2*cuyouts_rad+1 ] = np.nan
            if y0 < 0:
                outy0 = -y0
                y0 = 0
                ma[ 0:outy0, : ] = True
                no[ 0:outy0, : ] = np.nan
            if y1 > image.shape[0]:
                outy1 -= ( y1 - image.shape[0] )
                y1 = image.shape[0]
                ma[ outy1:2*cutouts_rad+1, : ] = False
                no[ outy1:2*cuyouts_rad+1, : ] = np.nan
            im[outy0:outy1, outx0:outx1] = image[y0:y1, x0:x1]
            no[outy0:outy1, outx0:outx1] = noise[y0:y1, x0:x1]
            ma[outy0:outy1, outx0:outx1] = mask[y0:y1, x0:x1]
            cutouts.append( im )
            noise_cutouts.append( no )
            mask_cutouts.append( ma )
        SCLogger.debug( "...photometry_and_diagnostics done making cutouts." )

    if ( noise_cutouts is None ) or ( mask_cutouts is None ):
        raise ValueError( "If you pass cutouts, you must also pass noise_cutouts and mask_cutouts" )

    if ( ( len(cutouts) != len(positions) ) or ( len(noise_cutouts) != len(positions) )
         or ( len(mask_cutouts) != len(positions) ) ):
        raise ValueError( "Number of cutouts, noise_cutouts, and mask_cutouts must all match number of positions." )

    if dobgsub:
        # photutils LocalBackground does a sigma-clipped median
        backgrounder = photutils.background.LocalBackground( inner_annulus_px, outer_annulus_px )

    SCLogger.debug( "Getting pfs for photometry..." )
    psfs = []
    clipwid = None
    for pos in positions:
        clip = psfobj.get_clip( x=pos[0], y=pos[1] )
        clipwid = clipwid if clipwid is not None else clip.shape[0]
        psfs.append( photutils.psf.ImagePSF( clip ) )

    # Psf photometry first, to find optimum positions.  Because
    #    photutils doesn't understand psfex psfs, it's not actually
    #    going to interpolate right.  It will be using a bicubic spline
    #    interpolation on the image-scale PSF, instead of the
    #    sync-interpolation on the optimum-scale PSF that psfex defines.
    #    It's a pity that there isn't a PSF model for photutils that
    #    understands PSFEX psfs.  Hopefully what we have here will be
    #    good enough for our purposes, but we should verify that!
    # Sadly, we can't just pass all the positions in a single call the
    #   way we do with aperture photometry, because the interace to
    #   photutils lets us pass in a single PSF model, but we're using a
    #   different one for each detection because we used a
    #   position-variable PSF from psfex.
    # (It should be possible to fix both these issues by writing a
    #   Fittable2dModel class to use with photutils' PSF photometry
    #   using our models/psf.py class and get_clip.  It might be slow.
    #   It would be effort to do.)
    SCLogger.debug( "Doing psf photometry...." )
    measurements = []
    for i, pos in enumerate( positions ):
        photor = photutils.psf.PSFPhotometry(
            psfs[i],
            clipwid,
            localbkg_estimator=backgrounder if dobgsub else None,
            aperture_radius=psfobj.fwhm_pixels
        )
        init_params = QTable()
        init_params['x'] = [ pos[0] ]
        init_params['y'] = [ pos[1] ]
        photresult = photor( image, mask=mask, error=noise, init_params=init_params )
        m = Measurements( aper_radii=apers,
                          flux_apertures=[np.nan] * len(apers),
                          flux_apertures_err=[np.nan] * len(apers),
                          x=photresult['x_fit'][0],
                          y=photresult['y_fit'][0],
                          flux_psf=photresult['flux_fit'][0],
                          flux_psf_err=photresult['flux_err'][0] )
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
    if dobgsub:
        SCLogger.debug( "...measuring backgrounds on sub image..." )
        bkgs = backgrounder( image,
                             [p[0] for p in positions],
                             [p[1] for p in positions],
                             mask=mask )
    for i in range(len(apers)):
        SCLogger.debug( f"...aperture on sub image photometry with r={apers[i]}" )
        apphot = photutils.aperture.aperture_photometry( image,
                                                         aperobjs[i],
                                                         error=noise,
                                                         mask=mask )
        for j in range( len(measurements) ):
            if dobgsub:
                measurements[j].flux_apertures[i] = apphot['aperture_sum'][j] - bkgs[j] * apers[i].area
            else:
                measurements[j].flux_apertures[i] = apphot['aperture_sum'][j]
            measurements[j].flux_apertures_err[i] = apphot['aperture_sum_err'][j]

    # Calculate morphological parameters.

    diagnostics = { 'width_ratio': np.zeros( len(measurements) ),
                    'ellipticity': np.zeros( len(measurements) ),
                    'negfrac': np.zeros( len(measurements) ),
                    'negfluxfrac': np.zeros( len(measurements) ) }

    SCLogger.debug( "Calculating moments for morphological parameters..." )
    for i, ( m, cutout, cutout_noise, cutout_mask ) in enumerate( zip( measurements, cutouts, noise_cutouts,
                                                                       mask_cutouts ) ):
        if dobgsub:
            morpho = photutils.morphology.data_properties( cutout - bkgs[i], mask=mask )
        else:
            morpho = photutils.morphology.data_properties( cutout, mask=cutout_mask )

        m.centroid_x = morpho.centroid[0]
        m.centroid_y = morpho.centroid[1]

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
        # Define moments by:
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
        # Those give the 1σ major and minor axes sizes.
        #
        # Define φ as the angle between the major axis of the distribution
        #   and the (y if cyy > cxx else x)-axis, in a counter-clockwise direction.
        #
        # With this definition, then
        #   ( π/4 < φ < π/4 ) if cxx != cyy else π/4 if cxy > 0 else -π/4 if cxy < 0 else 0
        # (if cxy = 0, φ is not well-defined and we may as well call it 0.)
        #
        # You can calculate φ with:
        #   φ = 1/2 arctan( 2 rxy / ( rxx^2 - ryy^2 ) )
        # if rxx != ryy, else:
        #   φ = π/4
        #
        # You can then get θ back (sorta... choose a thing between -π/2 and π that looks the same)
        #   θ = φ if rxx > ryy
        #   else θ = π/2 - θ

        rxx = morpho.moments_central[0][2] / morpho.moments_central[0][0]
        ryy = morpho.moments_central[2][0] / morpho.moments_central[0][0]
        rxy = morpho.moments_central[1][1] / morpho.moments_central[0][0]
        major = ( ( rxx + ryy ) + np.sqrt( 4 * rxy**2 + ( rxx - ryy ) ) ) / 2.
        minor = ( ( rxx + ryy ) - np.sqrt( 4 * rxy**2 + ( rxx - ryy ) ) ) / 2.
        if rxx == ryy:
            theta = np.pi/4. if rxy > 0 else -np.pi/4. if rxy < 0 else 0.
        else:
            phi = 0.5 * np.atan( 2 * rxy / ( rxx**2 - ryy**2 ) )
            theta = phi if rxx > rxy else np.pi/2. - phi
            # Make the angle between -π/2 and π/2
            if theta > np.pi / 2.:
                theta -= np.pi

        m.width = ( major + minor ) / 2.
        m.elongation = major / minor if minor > 0. else 1e32
        m.position_angle = theta

    SCLogger.debug( "...done calculating moments and morphological parameters." )

    # Diagnostic scores.  All of these are defined so that you can
    #   set a maximum threshold above which you decide a candidate is bad.
    #
    # * with_ratio = 2√(2ln2) * width / image FHWM_pix   (ratio of object size to seeing)
    # * elongation (bigger = more elliptical)
    # * negfrac = negpix / pospix:
    #     In a box of size 4fwhm on a side centered on center of the cutout,
    #       pospix = the number of pixels that are >2σ
    #       negpix = the number of pixels that are <-2σ
    #     Higher = more negative pixels, more likely to be a dipole
    #  * negfluxfrac = negflux / posflux
    #     In a box of size 4fwhm on a side centered on the cutout
    #       posflux = sum of flux values of all pixels that are >2σ
    #       negflux = -sum of flux values of all pixels that are <-2σ
    #  * nbad = # of bad (flagged) pixels within nbaddist of obj center along x or y

    SCLogger.debug( "Calculating diagnostics..." )
    _2sqrt2ln2 = 2.35482
    # dist =
    x0 = int( np.round(cutout.shape[0] / 2 - 2 * psfobj.fwhm_pixels ) )
    x1 = int( np.round(cutout.shape[0] / 2 + 2 * psfobj.fwhm_pixels ) ) + 1
    x0 = x0 if x0 >= 0 else 0
    x1 = x1 if x1 <= cutout.shape[0] else cutout.shape[0]
    for i, m in enumerate( measurements ):
        diagnostics['width_ratio'][i] = _2sqrt2ln2 * m.width / psfobj.fwhm_pixels
        diagnostics['ellipticity'][i] = m.elongation
        sub_cutout = cutout[ x0:x1, x0:x1 ]
        sub_noise = cutout_noise[ x0:x1, x0:x1 ]
        sub_mask = cutout_mask[ x0:x1, x0:x1 ]
        wneg = ( ~sub_mask ) & ( sub_cutout < 2.*sub_noise )
        wpos = ( ~sub_mask ) & ( sub_cutout > 2.*sub_noise )
        nneg = wneg[x0:x1, x0:x1].sum()
        npos = wpos[x0:x1, x0:x1].sum()
        fluxneg = -sub_cutout[ wneg ].sum()
        fluxpos = sub_cutout[ wpos ].sum()
        diagnostics['negfrac'][i] = ( 0 if ( npos == 0 and nneg == 0 )
                                      else 1e32 if npos == 0
                                      else nneg / npos )
        diagnostics['negfluxfrac'][i] = ( 0 if ( fluxpos == 0 and fluxneg == 0 )
                                          else 1e32 if fluxpos == 0
                                          else fluxneg / fluxpos )

    SCLogger.debug( "...done calculating diagnostics." )


    return measurements, diagnostics


# ======================================================================
# OLD BELOW

# import numpy as np

# from improc.tools import sigma_clipping

# # caching the soft-edge circles for faster calculations
# CACHED_CIRCLES = []
# CACHED_RADIUS_RESOLUTION = 0.01


# def get_circle(radius, imsize=15, oversampling=100, soft=True):

#     """Get a soft-edge circle.

#     This function will return a 2D array with a soft-edge circle of the given radius.

#     Parameters
#     ----------
#     radius: float
#         The radius of the circle.
#     imsize: int
#         The size of the 2D array to return. Must be square. Default is 15.
#     oversampling: int
#         The oversampling factor for the circle.
#         Default is 100.
#     soft: bool
#         Toggle the soft edge of the circle. Default is True (soft edge on).

#     Returns
#     -------
#     circle: np.ndarray
#         A 2D array with the soft-edge circle.

#     """
#     # Check if the circle is already cached
#     for circ in CACHED_CIRCLES:
#         if np.abs(circ.radius - radius) < CACHED_RADIUS_RESOLUTION and circ.imsize == imsize and circ.soft == soft:
#             return circ

#     # Create the circle
#     circ = Circle(radius, imsize=imsize, oversampling=oversampling, soft=soft)

#     # Cache the circle
#     CACHED_CIRCLES.append(circ)

#     return circ


# class Circle:
#     def __init__(self, radius, imsize=15, oversampling=100, soft=True):
#         self.radius = radius
#         self.imsize = imsize
#         self.datasize = max(imsize, 1 + 2 * int(radius + 1))
#         self.oversampling = oversampling
#         self.soft = soft

#         # these include the circle, after being moved by sub-pixel shifts for all possible positions in x and y
#         self.datacube = np.zeros((oversampling ** 2, self.datasize, self.datasize))

#         for i in range(oversampling):
#             for j in range(oversampling):
#                 x = i / oversampling
#                 y = j / oversampling
#                 self.datacube[i * oversampling + j] = self._make_circle(x, y)

#     def _make_circle(self, x, y):
#         """Generate the circles for a given sub-pixel shift in x and y. """

#         if x < 0 or x > 1 or y < 0 or y > 1:
#             raise ValueError("x and y must be between 0 and 1")

#         # Create the circle
#         xgrid, ygrid = np.meshgrid(np.arange(self.datasize), np.arange(self.datasize))
#         xgrid = xgrid - self.datasize // 2 - x
#         ygrid = ygrid - self.datasize // 2 - y
#         r = np.sqrt(xgrid ** 2 + ygrid ** 2)
#         if self.soft:
#             im = 1 + self.radius - r
#             im[r <= self.radius] = 1
#             im[r > self.radius + 1] = 0
#         else:
#             im = r
#             im[r <= self.radius] = 1
#             im[r > self.radius] = 0

#         # TODO: improve this with a better soft-edge function

#         return im

#     def get_image(self, dx, dy):
#         """Get the circle with the given pixel shifts, dx and dy.

#         Parameters
#         ----------
#         dx: float
#             The shift in the x direction. Can be a fraction of a pixel.
#         dy: float
#             The shift in the y direction. Can be a fraction of a pixel.

#         Returns
#         -------
#         im: np.ndarray
#             The circle with the given shifts.
#         """
#         if not np.isfinite(dx):
#             dx = 0
#         if not np.isfinite(dy):
#             dy = 0

#         # Get the integer part of the shifts
#         ix = int(np.floor(dx))
#         iy = int(np.floor(dy))

#         # Get the fractional part of the shifts
#         fx = dx - ix
#         fx = int(fx * self.oversampling)  # convert to oversampled pixels
#         fy = dy - iy
#         fy = int(fy * self.oversampling)  # convert to oversampled pixels

#         # Get the circle
#         im = self.datacube[(fx * self.oversampling + fy) % self.datacube.shape[0], :, :]

#         # roll and crop the circle to the correct position
#         im = np.roll(im, ix, axis=1)
#         if ix >= 0:
#             im[:, :ix] = 0
#         else:
#             im[:, ix:] = 0
#         im = np.roll(im, iy, axis=0)
#         if iy >= 0:
#             im[:iy, :] = 0
#         else:
#             im[iy:, :] = 0

#         if self.imsize != self.datasize:  # crop the image to the correct size
#             im = im[
#                 (self.datasize - self.imsize) // 2 : (self.datasize + self.imsize) // 2,
#                 (self.datasize - self.imsize) // 2 : (self.datasize + self.imsize) // 2,
#             ]

#         return im


# def iterative_cutouts_photometry(
#         image, weight, flags, radii=[3.0, 5.0, 7.0], annulus=[7.5, 10.0], iterations=2, local_bg=True
# ):
#     """Perform aperture photometry on an image, at slowly updating positions, using a list of apertures.

#     The "iterative" part means that it will use the starting positions but move the aperture centers
#     around based on the centroid found using the last aperture.

#     Parameters
#     ----------
#     image: np.ndarray
#         The image to perform photometry on.
#     weight: np.ndarray
#         The weight map for the image.
#     flags: np.ndarray
#         The flags for the image.
#     radii: list or 1D array
#         The apertures to use for photometry.
#         Must be a list of positive numbers.
#         In units of pixels!
#         Default is [3, 5, 7].
#     annulus: list or 1D array
#         The inner and outer radii of the annulus in pixels.
#     iterations: int
#         The number of repositioning iterations to perform.
#         For each aperture, will measure and reposition the centroid
#         this many times before moving on to the next aperture.
#         After the final centroid is found, will measure the flux
#         and second moments using the best centroid, over all apertures.
#         Default is 2.
#     local_bg: bool
#         Toggle the use of a local background estimate.
#         When True, will use the measured background in the annulus
#         when calculating the centroids. If the background is really
#         well subtracted before sending the cutout into this function,
#         the results will be a little more accurate with this set to False.
#         If the area in the annulus is very crowded,
#         it's better to set this to False as well.
#         Default is True.

#     Returns
#     -------
#     photometry: dict
#         A dictionary with the output of the photometry.

#     """
#     # Make sure the image is a 2D array
#     if len(image.shape) != 2:
#         raise ValueError("Image must be a 2D array")

#     # Make sure the weight is a 2D array
#     if len(weight.shape) != 2:
#         raise ValueError("Weight must be a 2D array")

#     # Make sure the flags is a 2D array
#     if len(flags.shape) != 2:
#         raise ValueError("Flags must be a 2D array")

#     # Make sure the apertures are a list or 1D array
#     radii = np.atleast_1d(radii)
#     if not np.all(radii > 0):
#         raise ValueError("Apertures must be positive numbers")

#     # order the radii in descending order:
#     radii = np.sort(radii)[::-1]

#     xgrid, ygrid = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
#     xgrid -= image.shape[1] // 2
#     ygrid -= image.shape[0] // 2

#     nandata = np.where(flags > 0, np.nan, image)

#     if np.all(nandata == 0 | np.isnan(nandata)):
#         cx = cy = cxx = cyy = cxy = 0.0
#         need_break = True  # skip the iterative mode if there's no data
#     else:
#         need_break = False
#         # find a rough estimate of the centroid using an unmasked cutout
#         if local_bg:
#             bkg_estimate = np.nanmedian(nandata)
#         else:
#             bkg_estimate = 0.0

#         denominator = np.nansum(nandata - bkg_estimate)
#         # prevent division by zero and other rare cases
#         epsilon = 0.01
#         if denominator == 0:
#             denominator = epsilon
#         elif abs(denominator) < epsilon:
#             denominator = epsilon * np.sign(denominator)

#         cx = np.nansum(xgrid * (nandata - bkg_estimate)) / denominator
#         cy = np.nansum(ygrid * (nandata - bkg_estimate)) / denominator
#         cxx = np.nansum((xgrid - cx) ** 2 * (nandata - bkg_estimate)) / denominator
#         cyy = np.nansum((ygrid - cy) ** 2 * (nandata - bkg_estimate)) / denominator
#         cxy = np.nansum((xgrid - cx) * (ygrid - cy) * (nandata - bkg_estimate)) / denominator

#     # get some very rough estimates just so we have something in case of immediate failure of the loop
#     fluxes = [np.nansum(nandata - bkg_estimate)] * len(radii)
#     areas = [float(np.nansum(~np.isnan(nandata)))] * len(radii)
#     norms = [float(np.nansum(~np.isnan(nandata)))] * len(radii)

#     background = 0.0
#     variance = np.nanvar(nandata)

#     photometry = dict(
#         radii=radii,
#         fluxes=fluxes,
#         areas=areas,
#         normalizations=norms,
#         background=background,
#         variance=variance,
#         n_pix_bg=nandata.size,
#         offset_x=cx,
#         offset_y=cy,
#         moment_xx=cxx,
#         moment_yy=cyy,
#         moment_xy=cxy,
#     )

#     if abs(cx) > nandata.shape[1] or abs(cy) > nandata.shape[0]:
#         need_break = True  # skip iterations if the centroid measurement is outside the cutouts

#     # in case any of the iterations fail, go back to the last centroid
#     prev_cx = cx
#     prev_cy = cy

#     for j, r in enumerate(radii):  # go over radii in order (from large to small!)
#         # short circuit if one of the measurements failed
#         if need_break:
#             break

#         # for each radius, do 1-3 rounds of repositioning the centroid
#         for i in range(iterations):
#             flux, area, background, variance, n_pix_bg, norm, cx, cy, cxx, cyy, cxy, failure = calc_at_position(
#                 nandata, r, annulus, xgrid, ygrid, cx, cy, local_bg=local_bg, full=False  # reposition only!
#             )

#             if failure:
#                 need_break = True
#                 cx = prev_cx
#                 cy = prev_cy
#                 break

#             # keep this in case any of the iterations fail
#             prev_cx = cx
#             prev_cy = cy

#     fluxes = np.full(len(radii), np.nan)
#     areas = np.full(len(radii), np.nan)
#     norms = np.full(len(radii), np.nan)

#     # no more updating of the centroids!
#     best_cx = cx
#     best_cy = cy

#     # go over each radius again and this time get all outputs (e.g., cxx) using the best centroid
#     for j, r in enumerate(radii):
#         flux, area, background, variance, n_pix_bg, norm, cx, cy, cxx, cyy, cxy, failure = calc_at_position(
#             nandata,
#             r,
#             annulus,
#             xgrid,
#             ygrid,
#             best_cx,
#             best_cy,
#             local_bg=local_bg,
#             soft=True,
#             full=True,
#             fixed=True,
#         )

#         if failure:
#             break

#         fluxes[j] = flux
#         areas[j] = area
#         norms[j] = norm

#     # update the output dictionary
#     photometry['radii'] = radii[::-1]  # return radii and fluxes in increasing order
#     photometry['fluxes'] = fluxes[::-1]  # return radii and fluxes in increasing order
#     photometry['areas'] = areas[::-1]  # return radii and areas in increasing order
#     photometry['background'] = background
#     photometry['variance'] = variance
#     photometry['n_pix_bg'] = n_pix_bg
#     photometry['normalizations'] = norms[::-1]  # return radii and areas in increasing order
#     photometry['offset_x'] = best_cx
#     photometry['offset_y'] = best_cy
#     photometry['moment_xx'] = cxx
#     photometry['moment_yy'] = cyy
#     photometry['moment_xy'] = cxy

#     # calculate from 2nd moments the width, ratio and angle of the source
#     # ref: https://en.wikipedia.org/wiki/Image_moment
#     major = 2 * (cxx + cyy + np.sqrt((cxx - cyy) ** 2 + 4 * cxy ** 2))
#     major = np.sqrt(major) if major > 0 else 0
#     minor = 2 * (cxx + cyy - np.sqrt((cxx - cyy) ** 2 + 4 * cxy ** 2))
#     minor = np.sqrt(minor) if minor > 0 else 0

#     angle = np.arctan2(2 * cxy, cxx - cyy) / 2
#     elongation = major / minor if minor > 0 else 0

#     photometry['major'] = major
#     photometry['minor'] = minor
#     photometry['angle'] = angle
#     photometry['elongation'] = elongation

#     return photometry


# def calc_at_position(data, radius, annulus, xgrid, ygrid, cx, cy, local_bg=True, soft=True, full=True, fixed=False):
#     """Calculate the photometry at a given position.

#     Parameters
#     ----------
#     data: np.ndarray
#         The image to perform photometry on.
#         Any bad pixels in the image are replaced by NaN.
#     radius: float
#         The radius of the aperture in pixels.
#     annulus: list or 1D array
#         The inner and outer radii of the annulus in pixels.
#     xgrid: np.ndarray
#         The x grid for the image.
#     ygrid: np.ndarray
#         The y grid for the image.
#     cx: float
#         The x position of the aperture center.
#     cy: float
#         The y position of the aperture center.
#     local_bg: bool
#         Toggle the use of a local background estimate.
#         When True, will use the measured background in the annulus
#         when calculating the centroids. If the background is really
#         well subtracted before sending the cutout into this function,
#         the results will be a little more accurate with this set to False.
#         If the area in the annulus is very crowded,
#         it's better to set this to False as well, though in that case
#         you really want to have background-subtracted before sending
#         data to this function.
#         Default is True.
#     soft: bool
#         Toggle the use of a soft-edged aperture.
#         Default is True.
#     full: bool
#         Toggle the calculation of the fluxes and second moments.
#         If set to False, will only calculate the centroids.
#         Default is True.
#     fixed: bool
#         If True, do not update the centroid position (assume it is fixed).
#         Default is False.

#     Returns
#     -------
#     flux: float
#         The flux in the aperture.
#     area: float
#         The area of the aperture.
#     background: float
#         The background level.
#     variance: float
#         The variance of the background.
#     n_pix_bg: float
#         Number of pixels in the background annulus.
#     norm: float
#         The normalization factor for the flux error
#         (this is the sqrt of the sum of squares of the aperture mask).
#     cx: float
#         The x position of the centroid.
#     cy: float
#         The y position of the centroid.
#     cxx: float
#         The second moment in x.
#     cyy: float
#         The second moment in y.
#     cxy: float
#         The cross moment.
#     failure: bool
#         A flag to indicate if the calculation failed.
#         This means the centroid is outside the cutout,
#         or the aperture is empty, or things like that.
#         If True, it flags to the outer scope to stop
#         the iterative process.
#     """
#     flux = area = background = variance = n_pix_bg = norm = cxx = cyy = cxy = 0
#     if np.all(np.isnan(data)):
#         return flux, area, background, variance, n_pix_bg, norm, cx, cy, cxx, cyy, cxy, True

#     # make a circle-mask based on the centroid position
#     if not np.isfinite(cx) or not np.isfinite(cy):
#         raise ValueError("Centroid is not finite, cannot proceed with photometry")

#     # get a circular mask
#     mask = get_circle(radius=radius, imsize=data.shape[0], soft=soft).get_image(cx, cy)
#     if np.nansum(mask) == 0:
#         return flux, area, background, variance, n_pix_bg, norm, cx, cy, cxx, cyy, cxy, True

#     masked_data = data * mask

#     flux = np.nansum(masked_data)  # total flux, not per pixel!
#     area = np.nansum(mask)  # save the number of pixels in the aperture

#     # get an offset annulus to get a local background estimate
#     if local_bg:
#         inner = get_circle(radius=annulus[0], imsize=data.shape[0], soft=False).get_image(cx, cy)
#         outer = get_circle(radius=annulus[1], imsize=data.shape[0], soft=False).get_image(cx, cy)
#         annulus_map = outer - inner
#         annulus_map[annulus_map == 0.] = np.nan  # flag pixels outside annulus as nan

#         if np.nansum(annulus_map) == 0:  # this can happen if annulus is too large
#             return flux, area, background, variance, n_pix_bg, norm, cx, cy, cxx, cyy, cxy, True

#         annulus_map_sum = np.nansum(annulus_map)
#         if annulus_map_sum == 0 or np.all(np.isnan(annulus_map)):
#             # this should only happen in tests or if the annulus is way too large or if all pixels are NaN
#             background = 0
#             variance = 0
#             norm = 0
#         else:
#             # b/g mean and variance (per pixel)
#             background, standard_dev = sigma_clipping(data * annulus_map, nsigma=5.0, median=True)
#             variance = standard_dev ** 2
#             norm = np.sqrt(np.nansum(mask ** 2))

#         if local_bg:  # update these to use the local background
#             denominator = (flux - background * area)
#             masked_data_bgsub = (data - background) * mask

#     else:
#         denominator = flux
#         masked_data_bgsub = masked_data
#         annulus_map_sum = 0.
#         background = np.nan
#         variance = np.nan
#         norm = np.sqrt( np.nansum( mask ** 2 ) )


#     if denominator == 0:  # this should only happen in pathological cases
#         return flux, area, background, variance, n_pix_bg, norm, cx, cy, cxx, cyy, cxy, True

#     if not fixed:  # update the centroids
#         cx = np.nansum(xgrid * masked_data_bgsub) / denominator
#         cy = np.nansum(ygrid * masked_data_bgsub) / denominator

#         # check that we got reasonable values!
#         if np.isnan(cx) or abs(cx) > data.shape[1] / 2 or np.isnan(cy) or abs(cy) > data.shape[0] / 2:
#             return flux, area, background, variance, n_pix_bg, norm, cx, cy, cxx, cyy, cxy, True

#     if full:
#         # update the second moments
#         cxx = np.nansum((xgrid - cx) ** 2 * masked_data_bgsub) / denominator
#         cyy = np.nansum((ygrid - cy) ** 2 * masked_data_bgsub) / denominator
#         cxy = np.nansum((xgrid - cx) * (ygrid - cy) * masked_data_bgsub) / denominator

#     n_pix_bg = annulus_map_sum
#     return flux, area, background, variance, n_pix_bg, norm, cx, cy, cxx, cyy, cxy, False


# if __name__ == '__main__':
#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#     c = get_circle(radius=3.0)
#     plt.imshow(c.get_image(0.0, 0.0))
#     plt.show()
