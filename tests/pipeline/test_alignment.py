from pipeline.data_store import DataStore
from pipeline.alignment import ImageAligner
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator

def test_warp_decam( decam_example_reduced_image_ds_with_zp, ref_for_decam_example_image ):
    ds = decam_example_reduced_image_ds_with_zp[0]
    ds.save_and_commit()
    refds = DataStore()
    refds.image = ref_for_decam_example_image

    try:
        # Need a source list and wcs for the ref image before we can warp it
        det = Detector( measure_psf=True )
        refds = det.run( refds )
        refds.sources.filepath = f'{refds.image.filepath}.sources.fits'
        refds.psf.filepath = refds.image.filepath
        refds.save_and_commit( no_archive=True )
        ast = AstroCalibrator()
        refds = ast.run( refds )
        refds.save_and_commit( no_archive=True )
        phot = PhotCalibrator()
        refds = phot.run( refds )

        Aligner = ImageAligner()
        warpds = Aligner.run( refds, ds )
        warpds.image.filepath = 'warpy'
        import pdb; pdb.set_trace()
        warpds.save_and_commit( no_archive=True )
        pass
    finally:
        refds.delete_everything()
