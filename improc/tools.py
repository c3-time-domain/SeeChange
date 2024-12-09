# various functions and tools used for image processing

import numpy as np
import re


def sigma_clipping(values, nsigma=3.0, iterations=5, axis=None, median=False):
    """Calculate the robust mean and rms by iterative exclusion of outliers.

    Parameters
    ----------
    values: numpy.ndarray
        The values to calculate the mean and rms for.
        Can be a vector, image, or cube.
    nsigma: float
        The number of sigma to use for the sigma clipping procedure.
        Values further from this many standard deviations are removed.
        Default is 3.0.
    iterations: int
        The number of iterations to use for the sigma clipping procedure.
        Default is 5. If the procedure converges it may do fewer iterations.
    axis: int or tuple of ints
        The axis or axes along which to calculate the mean and rms.
        Default is None, which means the function will attempt to guess
        the right axis.
        For vectors and 3D image cubes, will use axis=0 by default,
        which produces a scalar mean/rms for a vector,
        and a 2D image for a cube.
        For a 2D image input, will use axis=(0,1) by default,
        which will produce a scalar mean/rms for the image.
    median: bool
        If True, use the median instead of the mean for the all iterations
        beyond the first one (first iteration always uses median).

    Returns
    -------
    mean: float or numpy.ndarray
        The mean of the values after sigma clipping.
    rms: float or numpy.ndarray
        The rms of the values after sigma clipping.
    """
    # parse arguments
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a numpy.ndarray")

    if axis is None:
        if values.ndim == 1 or values.ndim == 3:
            axis = 0
        elif values.ndim == 2:
            axis = (0, 1)
        else:
            raise ValueError("values must be a vector, image, or cube")

    values = values.copy()

    # how many nan values?
    nans = np.isnan(values).sum()
    if nans == values.size:
        return np.nan, np.nan

    # first iteration:
    mean = np.nanmedian(values, axis=axis)
    rms = np.nanstd(values, axis=axis)

    for i in range(iterations):
        # remove pixels that are more than nsigma from the median
        clipped = np.abs(values - mean) > nsigma * rms
        values[clipped] = np.nan

        # recalculate the sky flat and noise
        if median:  # use median to calculate the mean estimate
            mean = np.nanmedian(values, axis=axis)
        else:  # only use median on the first iteration
            mean = np.nanmean(values, axis=axis)
        rms = np.nanstd(values, axis=axis)

        new_nans = np.isnan(values).sum()

        if new_nans == nans:
            break
        else:
            nans = new_nans

    return mean, rms


def make_gaussian(sigma_x=2.0, sigma_y=None, offset_x=0.0, offset_y=0.0, rotation=0.0, norm=1, imsize=None):
    """Create a small image of a Gaussian centered around the middle of the image.

    Parameters
    ----------
    sigma_x: float
        The sigma width parameter.
        If sigma_x and sigma_y are specified, this will be for the x-axis.
    sigma_y: float or None
        The sigma width parameter.
        If None, will use sigma_x for both axes.
    offset_x: float
        The offset in the x direction.
    offset_y: float
        The offset in the y direction.
    rotation: float
        The rotation angle in degrees.
        The Gaussian will be rotated counter-clockwise by this angle.
        If sigma_y is equal to sigma_x (or None) this has no effect.
    norm: int
        Normalization of the Gaussian. Choose value:
        0- do not normalize, peak will have a value of 1.0
        1- normalize so the sum of the image is equal to 1.0
        2- normalize the squares: the sqrt of the sum of squares is equal to 1.0
    imsize: int or 2-tuple of ints (optional)
        Number of pixels on a side for the output.
        If None, will automatically choose the smallest odd integer that is larger than max(sigma_x, sigma_y) * 10.

    Returns
    -------
    output: array
        A 2D array of the Gaussian.
    """
    if sigma_y is None:
        sigma_y = sigma_x

    if imsize is None:
        imsize = int(max(sigma_x, sigma_y) * 10)
        if imsize % 2 == 0:
            imsize += 1

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if norm not in [0, 1, 2]:
        raise ValueError('norm must be 0, 1, or 2')

    x = np.arange(imsize[1])
    y = np.arange(imsize[0])
    x, y = np.meshgrid(x, y)

    x0 = imsize[1] // 2 + offset_x
    y0 = imsize[0] // 2 + offset_y
    # TODO: what happens if imsize is even?

    x = x - x0
    y = y - y0

    rotation = rotation * np.pi / 180.0  # TODO: add option to give rotation in different units?

    x_rot = x * np.cos(rotation) - y * np.sin(rotation)
    y_rot = x * np.sin(rotation) + y * np.cos(rotation)

    output = np.exp(-0.5 * (x_rot ** 2 / sigma_x ** 2 + y_rot ** 2 / sigma_y ** 2))

    if norm == 1:
        output /= np.sum(output)
    elif norm == 2:
        output /= np.sqrt(np.sum(output ** 2))

    return output


def make_cutouts(data, x, y, size=15, fillvalue=None, dtype=None, yes_i_know_my_size_is_even=False ):
    """Make square cutouts around the given positions in the data.

    Parameters
    ----------
    data: numpy.ndarray
        The image to make the cutouts from.

    x: numpy.ndarray or list
        The x positions of the cutouts.

    y: numpy.ndarray or list
        The y positions of the cutouts.

    size: int
        The size along one edge of the cutouts. Default is 15.

    fillvalue: float or int
        The value to fill the cutouts with if they are partially off the
        edge of the data.  Default is np.nan, for floats, 0 for integers.

    dtype: numpy datatype or None
        The datatype for the returned cutouts.  If None, will use the
        datatype of the passed data.  WARNING: you may get perverse
        results for some combinations of data and cutout datatypes.
        For instance, converting floats with negative values to
        unsigned ints will create large positive integers where the
        floats are negative.  Use with care.

    yes_i_know_my_size_is_even: bool, default False
        Normally, make_cutouts will object if size is even.  (Odd size
        means there is a center pixel.)  If you really know you want
        cutouts that have an even size, set this to True.

    Returns
    -------
    cutouts: 3D np.ndarray
        The cutouts, with shape (len(x), size, size).

    """
    if dtype is None:
        dtype = data.dtype
    else:
        dtype = np.dtype( dtype )

    if dtype.kind not in ( 'f', 'i', 'u' ):
        raise TypeError( "make_cutouts: can only make float or integer cutouts" )

    if fillvalue is None:
        if dtype.kind == 'f':
            fillvalue = np.nan
        else:
            fillvalue = 0

    size = int( size )
    if ( size % 2 != 1 ) and ( not yes_i_know_my_size_is_even ):
        raise ValueError( f"cutouts size must be even, and {size} is not." )

    cutouts = np.full((len(x), size, size), fillvalue, dtype=dtype)
    down = int(np.floor((size - 1) / 2))
    up = int(np.ceil((size - 1) / 2))

    for i, (x0, y0) in enumerate(zip(x, y)):
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        if x0 - down < 0:
            if x0 + up < 0:
                continue  # leave the cutout as fillvalue
            left = 0
            offset_left = down - x0
        else:
            left = x0 - down
            offset_left = 0

        if x0 + up >= data.shape[1]:
            if x0 - down >= data.shape[1]:
                continue  # leave the cutout as fillvalue
            right = data.shape[1]
            offset_right = size - (x0 + up - data.shape[1] + 1)
        else:
            right = x0 + up
            offset_right = size

        if y0 - down < 0:
            if y0 + up < 0:
                continue  # leave the cutout as fillvalue
            bottom = 0
            offset_bottom = down - y0
        else:
            bottom = y0 - down
            offset_bottom = 0

        if y0 + up >= data.shape[0]:
            if y0 - down >= data.shape[0]:
                continue  # leave the cutout as fillvalue
            top = data.shape[0]
            offset_top = size - (y0 + up - data.shape[0] + 1)
        else:
            top = y0 + up
            offset_top = size

        cutouts[i][offset_bottom:offset_top, offset_left:offset_right] = data[bottom:top + 1, left:right + 1]

    return cutouts


def strip_wcs_keywords( hdr ):
    """Attempt to strip all WCS information from a FITS header.

    This may not be complete, as it pattern matches expected keywords.
    If it's missing some patterns, those won't get stripped.

    Parameters
    ----------
      hdr: The header from which to strip all WCS-related keywords.

    """

    basematch = re.compile( r"^C(RVAL|RPIX|UNIT|DELT|TYPE)[12]$" )
    cdmatch = re.compile( r"^CD[12]_[12]$" )
    sipmatch = re.compile( r"^[AB]P?_(ORDER|(\d+)_(\d+))$" )
    tpvmatch = re.compile( r"^P[CV]\d+_\d+$" )

    tonuke = set()
    for kw in hdr.keys():
        if ( basematch.search(kw) or cdmatch.search(kw) or sipmatch.search(kw) or tpvmatch.search(kw) ):
            tonuke.add( kw )

    for kw in tonuke:
        del hdr[kw]
