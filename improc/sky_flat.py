import numpy as np


from improc.simulator import Simulator
from improc.tools import sigma_clipping


def calc_sky_flat(images, iterations=3, nsigma=3.0, median=True):
    """Calculate the sky flat for a set of images.

    Parameters
    ----------
    images : list of 2D numpy.ndarrays or single 3D numpy.ndarray
        The images to calculate the sky flat for.
    iterations : int
        The number of iterations to use for the sigma clipping procedure.
        Default is 3. If the procedure converges it may do fewer iterations.
    nsigma : float
        The number of sigma to use for the sigma clipping procedure.
        Values further from this many standard deviations are removed.
        Default is 5.0.
    median: bool
        If True, use the median instead of the mean for the all iterations
        of the sigma clipping algorithm. Default is True.
        TODO: does the use of median cause a bias?

    Returns
    -------
    sky_flat : numpy.ndarray
        The sky flat image. The value in each pixel represents how much light was
        lost between the sky and the detector (including quantum efficiency, and digitization).
        Divide an image by the flat to correct for pixel-to-pixel sensitivity variations
        and camera vignetting.
    """
    # TODO: we may need to chop the images into smaller pieces to avoid memory issues

    if isinstance(images, np.ndarray) and images.ndim == 3:
        pass
    elif isinstance(images, list) and all(isinstance(im, np.ndarray) for im in images):
        images = np.array(images)
    else:
        raise TypeError("images must be a list of 2D numpy arrays or a 3D numpy array")

    # use the middle half of the image to calculate the sky level (to avoid vignetting)
    idx1_1 = images.shape[1] // 4
    idx1_2 = images.shape[1] // 4 * 3
    idx2_1 = images.shape[2] // 4
    idx2_2 = images.shape[2] // 4 * 3

    # normalize all images to the same mean sky level
    mean_sky = np.array([sigma_clipping(im[idx1_1:idx1_2, idx2_1:idx2_2], nsigma=3.0)[0] for im in images])
    # mean_sky = np.array([sigma_clipping(im)[0] for im in images])
    mean_sky = np.reshape(mean_sky, (images.shape[0], 1, 1))

    im = images.copy() / mean_sky

    mean, _ = sigma_clipping(im, nsigma=nsigma, iterations=iterations, axis=0, median=median)

    return mean


if __name__ == '__main__':
    sim = Simulator(image_size_x=128)
    sim.make_image()
