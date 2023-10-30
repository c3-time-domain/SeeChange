# various functions and tools used for image processing

import numpy as np


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

    # first iteration:
    mean = np.nanmedian(values, axis=axis)
    rms = np.nanstd(values, axis=axis)

    # how many nan values?
    nans = np.isnan(values).sum()

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