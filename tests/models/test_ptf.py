
from models.image import Image
from models.source_list import SourceList
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements


def test_get_ptf_exposure(ptf_exposure):
    assert ptf_exposure.filter == 'R'
    assert ptf_exposure.exp_time == 60
    assert ptf_exposure.instrument == 'PTF'
    assert ptf_exposure.telescope == 'P48'


def test_ptf_datastore(ptf_datastore):
    assert ptf_datastore.exposure.filter == 'R'
    assert ptf_datastore.exposure.exp_time == 60

    assert isinstance(ptf_datastore.image, Image)
    assert isinstance(ptf_datastore.sources, SourceList)
    assert isinstance(ptf_datastore.wcs, WorldCoordinates)
    assert isinstance(ptf_datastore.zp, ZeroPoint)
    assert isinstance(ptf_datastore.sub_image, Image)
    assert isinstance(ptf_datastore.detections, SourceList)
    assert isinstance(ptf_datastore.cutouts, Cutouts)
    assert all([isinstance(m, Measurements) for m in ptf_datastore.measurements])

    # using that bad row of pixels from the mask image
    assert all(ptf_datastore.image.flags[0:120, 94] > 0)
    assert all(ptf_datastore.image.weight[0:120, 94] == 0)


def test_ptf_urls(ptf_urls):
    assert len(ptf_urls) == 393


def test_ptf_images(ptf_reference_images):
    assert len(ptf_reference_images) == 5
