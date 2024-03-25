import pytest

import numpy as np

from improc.inpainting import Inpainter
from models.base import _logger


def test_trivial_inpaint():
    im = np.ones((10, 10))
    flags = np.zeros((10, 10), dtype='uint16')
    flags[5, 5] = 1
    im[5, 5] = 100
    weight = np.ones((10, 10))

    assert np.mean(im) == pytest.approx(2, abs=0.1)

    inp = Inpainter(single_image_method='biharmonic')
    im2 = inp.run(im, flags, weight)

    assert np.mean(im2) == pytest.approx(1, abs=0.1)
    assert im2[5, 5] == pytest.approx(1, abs=0.1)

    # check that we can ignore this flag if needed
    inp = Inpainter(single_image_method='biharmonic', ignore_flags=1)
    im2 = inp.run(im, flags, weight)
    assert np.array_equal(im, im2)

    # make a datacube

    im = np.ones((3, 10, 10))
    # make the images not exactly the same
    im[1] *= 1.2
    im[2] *= 1.4
    flags = np.zeros((3, 10, 10), dtype='uint16')

    # make a bad pixel but assume it is bad across all images (repair this using in-image inpainting)
    im[1, 5, 5] = 100
    flags[:, 5, 5] = 1

    # make another bad pixel but only in one image (repair by using other images)
    im[2, 8, 8] = 100
    flags[2, 8, 8] = 1

    weight = np.ones((3, 10, 10))

    assert np.mean(im) == pytest.approx(1.86, abs=0.1)

    inp = Inpainter(single_image_method='biharmonic', multi_image_method='mean', rescale_method='none')
    im2 = inp.run(im, flags, weight)

    # all bad pixels eliminated
    assert np.mean(im2) == pytest.approx(1.2, abs=0.1)
    assert im2[0, 5, 5] == pytest.approx(1, abs=0.1)
    assert im2[0, 8, 8] == pytest.approx(1, abs=0.1)  # original value
    assert im2[1, 8, 8] == pytest.approx(1.2, abs=0.1)  # original value
    assert im2[2, 8, 8] == pytest.approx(1.1, abs=0.1)  # fixed based on 1.0 and 1.2 means

    # this should rescale each image before inpainting by other images
    inp = Inpainter(single_image_method='biharmonic', multi_image_method='mean', rescale_method='median')
    im2 = inp.run(im, flags, weight)

    assert im2[2, 8, 8] == pytest.approx(1.4, abs=0.1)  # rescaled to each image

    # don't use image to image inpainting at all
    inp = Inpainter(single_image_method='biharmonic', multi_image_method='none', rescale_method='median')
    im2 = inp.run(im, flags, weight)

    assert im2[2, 8, 8] == pytest.approx(1.4, abs=0.1)  # inpainted from nearby values in the same image

    # try to make it fail on purpose by adding too many bad pixels
    im = np.ones((3, 5, 5))
    flags = np.zeros((3, 5, 5), dtype='uint16')
    # can't fix these bad pixels using other images
    im[:, :, 2] = 100
    flags[:, :, 2] = 1
    weight = np.ones((3, 5, 5))

    inp = Inpainter(single_image_method='biharmonic', multi_image_method='mean', ignore_flags=0)
    inp.run(im, flags, weight)

    assert np.mean(im) > 2
    assert np.all(im[:, :, 2] == 100)  # was not fixed!


def test_inpaint_aligned_images(ptf_aligned_images, blocking_plots):

    imcube = np.array([im.data for im in ptf_aligned_images])
    flagcube = np.array([im.flags for im in ptf_aligned_images])
    weightcube = np.array([im.weight for im in ptf_aligned_images])

    inp = Inpainter(single_image_method='biharmonic', multi_image_method='mean', rescale_method='median')
    im2 = inp.run(imcube, flagcube, weightcube)

    if blocking_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(imcube[0][0:160, 0:150], vmin=1500, vmax=3500)
        ax[0].set_title('original')
        ax[1].imshow(im2[0][0:160, 0:150], vmin=1500, vmax=3500)
        ax[1].set_title('inpaint')
        ax[2].imshow(flagcube[0][0:160, 0:150])
        ax[2].set_title('flags')
        plt.show(block=True)
        _logger.debug('done')
