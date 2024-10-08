import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from improc.photometry import iterative_cutouts_photometry, get_circle, Circle

#coordinates nearby a star (pix), such that it appears within the annulus
clipCentX = 2391
clipCentY = 1323
clipHalfWidth = 12 #half-width of clipped image

# uncomment this to run the plotting tests interactively
# os.environ['INTERACTIVE'] = '1'

#TODO: We should write this once we have a better soft-edge function
# def test_circle_soft():
    # pass

def test_circle_hard():
    circTst = get_circle(radius=3,imsize=7,soft=False).get_image(0,0)
    assert np.array_equal(circTst, np.array([[0., 0., 0., 1., 0., 0., 0.],
                                             [0., 1., 1., 1., 1., 1., 0.],
                                             [0., 1., 1., 1., 1., 1., 0.],
                                             [1., 1., 1., 1., 1., 1., 1.],
                                             [0., 1., 1., 1., 1., 1., 0.],
                                             [0., 1., 1., 1., 1., 1., 0.],
                                             [0., 0., 0., 1., 0., 0., 0.]]))

def test_background_sigma_clip(ptf_datastore):
    imgClip = ptf_datastore.image.data[     clipCentX - clipHalfWidth : clipCentX + clipHalfWidth,
                                            clipCentY - clipHalfWidth : clipCentY + clipHalfWidth]
    weightClip = ptf_datastore.image.weight[clipCentX - clipHalfWidth : clipCentX + clipHalfWidth,
                                            clipCentY - clipHalfWidth : clipCentY + clipHalfWidth]
    flagsClip = ptf_datastore.image.flags[  clipCentX - clipHalfWidth : clipCentX + clipHalfWidth,
                                            clipCentY - clipHalfWidth : clipCentY + clipHalfWidth]
    result = iterative_cutouts_photometry(imgClip, weightClip, flagsClip)
    assert result['background'] == pytest.approx(1199.1791, rel=1e-2)

@pytest.mark.skipif( os.getenv('INTERACTIVE') is None, reason='Set INTERACTIVE to run this test' )
def test_plot_annulus(ptf_datastore):
    imgClip = ptf_datastore.image.data[clipCentX-clipHalfWidth:clipCentX+clipHalfWidth,
                                        clipCentY-clipHalfWidth:clipCentY+clipHalfWidth]

    inner = get_circle(radius=7.5, imsize=imgClip.shape[0], soft=False).get_image(0, 0)
    outer = get_circle(radius=10.0, imsize=imgClip.shape[0], soft=False).get_image(0, 0)
    annulus_map = outer - inner

    vmin = np.percentile(imgClip, 1)
    vmax = np.percentile(imgClip, 99)

    plt.imshow(imgClip, vmin=vmin, vmax=vmax)
    plt.savefig('plots/annulus_test.png')
    plt.imshow(annulus_map,alpha=0.5)
    plt.savefig('plots/annulus_test_2.png')
    plt.show()
