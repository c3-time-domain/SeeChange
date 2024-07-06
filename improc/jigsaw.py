import numpy as np


def cut(im, cut_size, overlap=0, pad_value=None):
    """Cut an image into small cutouts, with set intervals and some overlap.

    Parameters
    ----------
    im: np.ndarray
        A 2D image that needs to be cut into pieces.
    cut_size: scalar int or 2-tuple of int
        The size of the output cutouts.
        Can be a scalar (in which case the cutouts are square)
        or a 2-tuple so that the shape of each cutout is equal
        to this parameter.
    overlap: scalar float or integer or string
        The fraction of overlap between the cutouts, in the range [0,1).
        If given as an integer (or float larger or equal to 1.0) will be
        interpreted as the number of pixels that overlap between adjacent cutouts.
        The default is 0 (no overlap).
        For fractional overlap, the number of pixels that overlap
        between adjacent cutouts (in each dimension) is ceil(cut_size * overlap).
        That means the next cutout will start inside the current cutout
        at pixel number floor(cut_size * (1 - overlap)).
        If given as a string, will be interpreted as a window function.
        Currently only "hanning" is supported.
        See the stitch() function for more details.
    pad_value: scalar float or string, optional
        If given a value, will use that as
        a filler for any part of the cutouts
        that lie outside the original image
        (this can happen if the last pixels
        reach out of the image).
        Default is NaN.
        If given as a string equal to "nan" will also
        use it as NaN. If given as "zero" will use 0.

    Returns
    -------
    cutouts: 3D np.ndarray
        A 3D array where the first dimension is the number of cutouts,
        and the other two dimensions are the cutout height and width.
    corners: list of 2-tuples
        Each item on the list contains the lower corner (x,y) pixel positions
        of the relevant cutout in the coordinates of the full image.
    """
    S = im.shape  # size of the input
    if np.isscalar(cut_size):
        C = (cut_size, cut_size)
    elif isinstance(cut_size, tuple) and len(cut_size) == 2:
        C = cut_size
    else:
        raise TypeError("cut_size must be a scalar or 2-tuple")

    if not isinstance(overlap, (int, float, np.number, str)):
        raise ValueError("Overlap must be a scalar, a 2D array, or a string")

    if isinstance(overlap, str):
        if overlap == 'hanning':
            overlap = 0.5  # we don't apply the window here, only set up the correct overlap for it
        else:
            raise ValueError(f'Unknown overlap type "{overlap}". Use "hanning" or a scalar or a 2D array.')

    if overlap < 0:
        raise ValueError("Overlap must be a non-negative number")
    if overlap < 1:
        pix_overlap = tuple(int(np.ceil(c * overlap)) for c in C)
    else:
        pix_overlap = (int(overlap), int(overlap))

    if isinstance(pad_value, str):
        if pad_value == 'nan':
            pad_value = np.nan
        elif pad_value == 'zero':
            pad_value = 0
        else:
            raise ValueError('pad_value must be a scalar or "nan" or "zero"')

    # estimate the number of cutouts in each dimension
    num_x = int(np.ceil(S[1] / (C[1] - pix_overlap[1]))) + 1
    num_y = int(np.ceil(S[0] / (C[0] - pix_overlap[0]))) + 1

    corners = []

    for i in range(num_x):
        corner_x = i * (C[1] - pix_overlap[1]) - pix_overlap[1]
        if corner_x >= S[1]:
            break

        for j in range(num_y):
            corner_y = j * (C[0] - pix_overlap[0]) - pix_overlap[0]
            if corner_y >= S[0]:
                break

            corners.append((corner_x, corner_y))

    output = np.empty((len(corners), C[0], C[1]))

    for i, (x, y) in enumerate(corners):
        low_x = max(x, 0)
        high_x = min(x + C[1], S[1])
        low_y = max(y, 0)
        high_y = min(y + C[0], S[0])

        cutout = np.full(C, pad_value)
        cutout[low_y - y:high_y - y, low_x - x:high_x - x] = im[low_y:high_y, low_x:high_x]

        output[i] = cutout

    return output, corners


def stitch(cutouts, im_shape, corners=None, overlap=None):
    """Rebuild a full image from cutouts.

    Parameters
    ----------
    cutouts: np.ndarray
        A 3D array where the first dimension is the number of cutouts,
        and the other two dimensions are the cutout height and width.
    im_shape: scalar int or 2-tuple of int
        The shape of the full image.
    corners: list of 2-tuples (optional)
        Each item on the list contains the lower corner (x,y) pixel positions
        of the relevant cutout in the coordinates of the full image.
        If not given, will try to guess this from the im_shape,
        by assuming overlap=0.
    overlap: float scalar or np.ndarray or string (optional)
        The overlap fraction, or the overlap number of pixels.
        Will zero out the top and right edges of the cutouts.
        If given as a fraction, the number of pixels that are not
        zeroed out, from the beginning of each cutout,
        is floor(cut_size * (1 - overlap)).

        Another option is to give an array of the same size as one of the cutouts.
        The cutouts are multiplied by this array before adding them
        to make the full image. This is useful for overlapping images,
        as it allows to blend the cutouts in a smooth way.

        Can also give a string, which will be interpreted as one of these options:
        - 'hanning': a Hanning window, which is a cosine window, defined as
            0.5 * (1 - cos(2 * pi * x / (cut_size - 1))), where x is the pixel position.

        If not given, will not apply any overlap window, which will alter the pixel values
        of the output image in any case where there is overlap!

    Returns
    -------
        A 2D array with the full image.
    """
    C = cutouts.shape[1:]  # size of the cutouts
    N = cutouts.shape[0]  # number of cutouts
    S = im_shape
    if isinstance(S, (int, float, np.number)):
        S = (S, S)
    if S is not None and len(S) != 2:
        raise ValueError("im_shape must be a scalar or 2-tuple")

    if corners is None:  # infer corner_list from im_shape
        corners = []
        for i in range(N):
            corners.append((i % (S[1] // C[1]) * C[1], i // (S[1] // C[1]) * C[0]))

    # check if we can interpret the "overlap" input to make a window
    window = None  # by default there is no window array
    if isinstance(overlap, str):
        if overlap == 'hanning':
            window = np.hanning(C[0])[:, None] * np.hanning(C[1])[None, :]
        else:
            raise ValueError(f'Unknown overlap type "{overlap}". Use "hanning" or a scalar or a 2D array.')
    elif np.isscalar(overlap):
        if overlap < 0:
            raise ValueError("Overlap must be a non-negative number")
        if overlap < 1:
            pix_overlap = tuple(int(np.ceil(overlap * c)) for c in C)
        else:
            pix_overlap = (int(overlap), int(overlap))
        window = np.ones(C)
        window[-pix_overlap[0]:, :] = 0
        window[:, -pix_overlap[1]:] = 0
    elif overlap is not None:
        raise ValueError('overlap must be a scalar, a 2D array, or a string')

    if window is not None:
        if window.shape != C:
            raise ValueError('window must have the same shape as the cutouts')

        window = window[None, :, :]  # add a dimension to broadcast
        cutouts = cutouts * window

    im = np.zeros(S)
    for i, (x, y) in enumerate(corners):
        low_x = max(x, 0)
        high_x = min(x + C[1], S[1])
        low_y = max(y, 0)
        high_y = min(y + C[0], S[0])
        im[low_y:high_y, low_x:high_x] += cutouts[i, low_y - y:high_y - y, low_x - x:high_x - x]

    return im




