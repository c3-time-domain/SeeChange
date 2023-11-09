from models.base import SmartSession
from models.zero_point import ZeroPoint

from pipeline.photo_cal import PhotCalibrator


def test_decam_photo_cal( decam_example_reduced_image_ds_with_wcs ):
    ds = decam_example_reduced_image_ds_with_wcs[0]
    ds.save_and_commit()
    with SmartSession() as session:
        photomotor = PhotCalibrator( cross_match_catalog='GaiaDR3' )  # add other parameters
        ds = photomotor.run( ds )
        import pdb; pdb.set_trace()
        pass


