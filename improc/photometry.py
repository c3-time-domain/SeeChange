
import numpy as np

from improc.tools import make_gaussian, sigma_clipping

# caching the soft-edge circles for faster calculations
CACHED_CIRCLES = []
CACHED_RADIUS_RESOLUTION = 0.01


def get_circle(radius, imsize=15, oversampling=100, soft=True):
    """Get a soft-edge circle.

    This function will return a 2D array with a soft-edge circle of the given radius.

    Parameters
    ----------
    radius: float
        The radius of the circle.
    imsize: int
        The size of the 2D array to return. Must be square. Default is 15.
    oversampling: int
        The oversampling factor for the circle.
        Default is 100.
    soft: bool
        Toggle the soft edge of the circle. Default is True (soft edge on).

    Returns
    -------
    circle: np.ndarray
        A 2D array with the soft-edge circle.

    """
    # Check if the circle is already cached
    for circ in CACHED_CIRCLES:
        if np.abs(circ.radius - radius) < CACHED_RADIUS_RESOLUTION and circ.imsize == imsize and circ.soft == soft:
            return circ

    # Create the circle
    circ = Circle(radius, imsize=imsize, oversampling=oversampling, soft=soft)

    # Cache the circle
    CACHED_CIRCLES.append(circ)

    return circ


class Circle:
    def __init__(self, radius, imsize=15, oversampling=100, soft=True):
        self.radius = radius
        self.imsize = imsize
        self.datasize = max(imsize, 1 + 2 * int(radius + 1))
        self.oversampling = oversampling
        self.soft = soft

        # these include the circle, after being moved by sub-pixel shifts for all possible positions in x and y
        self.datacube = np.zeros((oversampling ** 2, self.datasize, self.datasize))

        for i in range(oversampling):
            for j in range(oversampling):
                x = i / oversampling
                y = j / oversampling
                self.datacube[i * oversampling + j] = self._make_circle(x, y)

    def _make_circle(self, x, y):
        """Generate the circles for a given sub-pixel shift in x and y. """

        if x < 0 or x > 1 or y < 0 or y > 1:
            raise ValueError("x and y must be between 0 and 1")

        # Create the circle
        xgrid, ygrid = np.meshgrid(np.arange(self.datasize), np.arange(self.datasize))
        xgrid = xgrid - self.datasize // 2 - x
        ygrid = ygrid - self.datasize // 2 - y
        r = np.sqrt(xgrid ** 2 + ygrid ** 2)
        if self.soft:
            im = 1 + self.radius - r
            im[r <= self.radius] = 1
            im[r > self.radius + 1] = 0
        else: 
            im = r
            im[r <= self.radius] = 1
            im[r > self.radius] = 0

        # TODO: improve this with a better soft-edge function

        return im

    def get_image(self, dx, dy):
        """Get the circle with the given pixel shifts, dx and dy.

        Parameters
        ----------
        dx: float
            The shift in the x direction. Can be a fraction of a pixel.
        dy: float
            The shift in the y direction. Can be a fraction of a pixel.

        Returns
        -------
        im: np.ndarray
            The circle with the given shifts.
        """
        if not np.isfinite(dx):
            dx = 0
        if not np.isfinite(dy):
            dy = 0

        # Get the integer part of the shifts
        ix = int(np.floor(dx))
        iy = int(np.floor(dy))

        # Get the fractional part of the shifts
        fx = dx - ix
        fx = int(fx * self.oversampling)  # convert to oversampled pixels
        fy = dy - iy
        fy = int(fy * self.oversampling)  # convert to oversampled pixels

        # Get the circle
        im = self.datacube[(fx * self.oversampling + fy) % self.datacube.shape[0], :, :]

        # roll and crop the circle to the correct position
        im = np.roll(im, ix, axis=1)
        if ix >= 0:
            im[:, :ix] = 0
        else:
            im[:, ix:] = 0
        im = np.roll(im, iy, axis=0)
        if iy >= 0:
            im[:iy, :] = 0
        else:
            im[iy:, :] = 0

        if self.imsize != self.datasize:  # crop the image to the correct size
            im = im[
                (self.datasize - self.imsize) // 2 : (self.datasize + self.imsize) // 2,
                (self.datasize - self.imsize) // 2 : (self.datasize + self.imsize) // 2,
            ]

        return im


def iterative_cutouts_photometry(
        image, weight, flags, radii=[3.0, 5.0, 7.0], annulus=[7.5, 10.0], iterations=2, local_bg=True
):
    """Perform aperture photometry on an image, at slowly updating positions, using a list of apertures.

    The "iterative" part means that it will use the starting positions but move the aperture centers
    around based on the centroid found using the last aperture.

    Parameters
    ----------
    image: np.ndarray
        The image to perform photometry on.
    weight: np.ndarray
        The weight map for the image.
    flags: np.ndarray
        The flags for the image.
    radii: list or 1D array
        The apertures to use for photometry.
        Must be a list of positive numbers.
        In units of pixels!
        Default is [3, 5, 7].
    annulus: list or 1D array
        The inner and outer radii of the annulus in pixels.
    iterations: int
        The number of repositioning iterations to perform.
        For each aperture, will measure and reposition the centroid
        this many times before moving on to the next aperture.
        After the final centroid is found, will measure the flux
        and second moments using the best centroid, over all apertures.
        Default is 2.
    local_bg: bool
        Toggle the use of a local background estimate.
        When True, will use the measured background in the annulus
        when calculating the centroids. If the background is really
        well subtracted before sending the cutout into this function,
        the results will be a little more accurate with this set to False.
        If the area in the annulus is very crowded,
        it's better to set this to False as well.
        Default is True.

    Returns
    -------
    photometry: dict
        A dictionary with the output of the photometry.

    """
    # Make sure the image is a 2D array
    if len(image.shape) != 2:
        raise ValueError("Image must be a 2D array")

    # Make sure the weight is a 2D array
    if len(weight.shape) != 2:
        raise ValueError("Weight must be a 2D array")

    # Make sure the flags is a 2D array
    if len(flags.shape) != 2:
        raise ValueError("Flags must be a 2D array")

    # Make sure the apertures are a list or 1D array
    radii = np.atleast_1d(radii)
    if not np.all(radii > 0):
        raise ValueError("Apertures must be positive numbers")

    # order the radii in descending order:
    radii = np.sort(radii)[::-1]

    xgrid, ygrid = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xgrid -= image.shape[1] // 2
    ygrid -= image.shape[0] // 2

    nandata = np.where(flags > 0, np.nan, image)

    if np.all(nandata == 0 | np.isnan(nandata)):
        cx = cy = cxx = cyy = cxy = 0.0
        need_break = True  # skip the iterative mode if there's no data
    else:
        need_break = False
        # find a rough estimate of the centroid using an unmasked cutout
        if local_bg:
            bkg_estimate = np.nanmedian(nandata)
        else:
            bkg_estimate = 0.0

        denominator = np.nansum(nandata - bkg_estimate)
        epsilon = 0.01
        if denominator == 0:
            denominator = epsilon
        elif abs(denominator) < epsilon:
            denominator = epsilon * np.sign(denominator)  # prevent division by zero and other rare cases

        cx = np.nansum(xgrid * (nandata - bkg_estimate)) / denominator
        cy = np.nansum(ygrid * (nandata - bkg_estimate)) / denominator
        cxx = np.nansum((xgrid - cx) ** 2 * (nandata - bkg_estimate)) / denominator
        cyy = np.nansum((ygrid - cy) ** 2 * (nandata - bkg_estimate)) / denominator
        cxy = np.nansum((xgrid - cx) * (ygrid - cy) * (nandata - bkg_estimate)) / denominator

    # get some very rough estimates just so we have something in case of immediate failure of the loop
    fluxes = [np.nansum((nandata - bkg_estimate))] * len(radii)
    areas = [float(np.nansum(~np.isnan(nandata)))] * len(radii)
    norms = [float(np.nansum(~np.isnan(nandata)))] * len(radii)

    background = 0.0
    variance = np.nanvar(nandata)

    photometry = dict(
        radii=radii,
        fluxes=fluxes,
        areas=areas,
        normalizations=norms,
        background=background,
        variance=variance,
        offset_x=cx,
        offset_y=cy,
        moment_xx=cxx,
        moment_yy=cyy,
        moment_xy=cxy,
    )

    if abs(cx) > nandata.shape[1] or abs(cy) > nandata.shape[0]:
        need_break = True  # skip iterations if the centroid measurement is outside the cutouts

    # in case any of the iterations fail, go back to the last centroid
    prev_cx = cx
    prev_cy = cy

    for j, r in enumerate(radii):  # go over radii in order (from large to small!)
        # short circuit if one of the measurements failed
        if need_break:
            break

        # for each radius, do 1-3 rounds of repositioning the centroid
        for i in range(iterations):
            flux, area, background, variance, norm, cx, cy, cxx, cyy, cxy, failure = calc_at_position(
                nandata, r, annulus, xgrid, ygrid, cx, cy, local_bg=local_bg, full=False  # reposition only!
            )

            if failure:
                need_break = True
                cx = prev_cx
                cy = prev_cy
                break

            # keep this in case any of the iterations fail
            prev_cx = cx
            prev_cy = cy

    fluxes = np.full(len(radii), np.nan)
    areas = np.full(len(radii), np.nan)
    norms = np.full(len(radii), np.nan)

    # no more updating of the centroids!
    best_cx = cx
    best_cy = cy

    # go over each radius again and this time get all outputs (e.g., cxx) using the best centroid
    for j, r in enumerate(radii):
        flux, area, background, variance, norm, cx, cy, cxx, cyy, cxy, failure = calc_at_position(
            nandata,
            r,
            annulus,
            xgrid,
            ygrid,
            best_cx,
            best_cy,
            local_bg=local_bg,
            soft=True,
            full=True,
            fixed=True,
        )

        if failure:
            break

        fluxes[j] = flux
        areas[j] = area
        norms[j] = norm

    # update the output dictionary
    photometry['radii'] = radii[::-1]  # return radii and fluxes in increasing order
    photometry['fluxes'] = fluxes[::-1]  # return radii and fluxes in increasing order
    photometry['areas'] = areas[::-1]  # return radii and areas in increasing order
    photometry['background'] = background
    photometry['variance'] = variance
    photometry['normalizations'] = norms[::-1]  # return radii and areas in increasing order
    photometry['offset_x'] = best_cx
    photometry['offset_y'] = best_cy
    photometry['moment_xx'] = cxx
    photometry['moment_yy'] = cyy
    photometry['moment_xy'] = cxy

    # calculate from 2nd moments the width, ratio and angle of the source
    # ref: https://en.wikipedia.org/wiki/Image_moment
    major = 2 * (cxx + cyy + np.sqrt((cxx - cyy) ** 2 + 4 * cxy ** 2))
    major = np.sqrt(major) if major > 0 else 0
    minor = 2 * (cxx + cyy - np.sqrt((cxx - cyy) ** 2 + 4 * cxy ** 2))
    minor = np.sqrt(minor) if minor > 0 else 0

    angle = np.arctan2(2 * cxy, cxx - cyy) / 2
    elongation = major / minor if minor > 0 else 0

    photometry['major'] = major
    photometry['minor'] = minor
    photometry['angle'] = angle
    photometry['elongation'] = elongation

    return photometry


def calc_at_position(data, radius, annulus, xgrid, ygrid, cx, cy, local_bg=True, soft=True, full=True, fixed=False):
    """Calculate the photometry at a given position.

    Parameters
    ----------
    data: np.ndarray
        The image to perform photometry on.
        Any bad pixels in the image are replaced by NaN.
    radius: float
        The radius of the aperture in pixels.
    annulus: list or 1D array
        The inner and outer radii of the annulus in pixels.
    xgrid: np.ndarray
        The x grid for the image.
    ygrid: np.ndarray
        The y grid for the image.
    cx: float
        The x position of the aperture center.
    cy: float
        The y position of the aperture center.
    local_bg: bool
        Toggle the use of a local background estimate.
        When True, will use the measured background in the annulus
        when calculating the centroids. If the background is really
        well subtracted before sending the cutout into this function,
        the results will be a little more accurate with this set to False.
        If the area in the annulus is very crowded,
        it's better to set this to False as well.
        Default is True.
    soft: bool
        Toggle the use of a soft-edged aperture.
        Default is True.
    full: bool
        Toggle the calculation of the fluxes and second moments.
        If set to False, will only calculate the centroids.
        Default is True.
    fixed: bool
        If True, do not update the centroid position (assume it is fixed).
        Default is False.

    Returns
    -------
    flux: float
        The flux in the aperture.
    area: float
        The area of the aperture.
    background: float
        The background level.
    variance: float
        The variance of the background.
    norm: float
        The normalization factor for the flux error
        (this is the sqrt of the sum of squares of the aperture mask).
    cx: float
        The x position of the centroid.
    cy: float
        The y position of the centroid.
    cxx: float
        The second moment in x.
    cyy: float
        The second moment in y.
    cxy: float
        The cross moment.
    failure: bool
        A flag to indicate if the calculation failed.
        This means the centroid is outside the cutout,
        or the aperture is empty, or things like that.
        If True, it flags to the outer scope to stop
        the iterative process.
    """
    flux = area = background = variance = norm = cxx = cyy = cxy = 0

    # make a circle-mask based on the centroid position
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise ValueError("Centroid is not finite, cannot proceed with photometry")

    # get a circular mask
    mask = get_circle(radius=radius, imsize=data.shape[0], soft=soft).get_image(cx, cy)
    if np.nansum(mask) == 0:
        return flux, area, background, variance, norm, cx, cy, cxx, cyy, cxy, True

    masked_data = data * mask

    flux = np.nansum(masked_data)  # total flux, not per pixel!
    area = np.nansum(mask)  # save the number of pixels in the aperture
    denominator = flux
    masked_data_bg = masked_data

    # get an offset annulus to get a local background estimate
    if full or local_bg:
        inner = get_circle(radius=annulus[0], imsize=data.shape[0], soft=False).get_image(cx, cy)
        outer = get_circle(radius=annulus[1], imsize=data.shape[0], soft=False).get_image(cx, cy)
        annulus_map = outer - inner
        annulus_map[annulus_map == 0.] = np.nan  # flag pixels outside annulus as nan

        if np.nansum(annulus_map) == 0:  # this can happen if annulus is too large
            return flux, area, background, variance, norm, cx, cy, cxx, cyy, cxy, True

        annulus_map_sum = np.nansum(annulus_map)
        if annulus_map_sum == 0:  # this should only happen in tests or if the annulus is way too large
            background = 0
            variance = 0
            norm = 0
        else:
            # b/g mean and variance (per pixel)
            background, standard_dev = sigma_clipping(data * annulus_map, nsigma=5.0, median=True)
            variance = standard_dev ** 2
            norm = np.sqrt(np.nansum(mask ** 2))

        if local_bg:  # update these to use the local background
            denominator = (flux - background * area)
            masked_data_bg = (data - background) * mask

    if denominator == 0:  # this should only happen in pathological cases
        return flux, area, background, variance, norm, cx, cy, cxx, cyy, cxy, True

    if not fixed:  # update the centroids
        cx = np.nansum(xgrid * masked_data_bg) / denominator
        cy = np.nansum(ygrid * masked_data_bg) / denominator

        # check that we got reasonable values!
        if np.isnan(cx) or abs(cx) > data.shape[1] / 2 or np.isnan(cy) or abs(cy) > data.shape[0] / 2:
            return flux, area, background, variance, norm, cx, cy, cxx, cyy, cxy, True

    if full:
        # update the second moments
        cxx = np.nansum((xgrid - cx) ** 2 * masked_data_bg) / denominator
        cyy = np.nansum((ygrid - cy) ** 2 * masked_data_bg) / denominator
        cxy = np.nansum((xgrid - cx) * (ygrid - cy) * masked_data_bg) / denominator

    return flux, area, background, variance, norm, cx, cy, cxx, cyy, cxy, False


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    c = get_circle(radius=3.0)
    plt.imshow(c.get_image(0.0, 0.0))
    plt.show()
