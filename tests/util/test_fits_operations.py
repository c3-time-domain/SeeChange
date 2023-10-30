import os

import numpy as np

from models.base import CODE_ROOT
from pipeline.utils import read_fits_image


def test_read_fits_image():
    filename = os.path.join(CODE_ROOT, 'data/test_data/DECam_examples/c4d_20221002_040239_r_v1.24.fits')

    # by default only get the data
    data = read_fits_image(filename)

    assert data.shape == (4094, 2046)
    assert np.array_equal(data[0, 0:10], np.array([303.96603, 304.1212, 291.13953, 301.72168, 306.76215, 314.1882,
                                                   323.1495, 324.92514, 336.54843, 369.187], dtype=np.float32))

    # get only the header
    header = read_fits_image(filename, output='header')

    assert header['LMT_MG'] == 25.37038556706342

    # get both as a tuple
    data, header = read_fits_image(filename, output='both')

    assert data.shape == (4094, 2046)
    assert np.array_equal(data[0, 0:10], np.array([303.96603, 304.1212, 291.13953, 301.72168, 306.76215, 314.1882,
                                                   323.1495, 324.92514, 336.54843, 369.187], dtype=np.float32))

    assert header['LMT_MG'] == 25.37038556706342

