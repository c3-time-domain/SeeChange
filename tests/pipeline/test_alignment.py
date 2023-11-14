from pipeline.data_store import DataStore
from pipeline.alignment import ImageAligner
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator

def test_warp_decam( decam_example_reduced_image_ds_with_wcs, ref_for_decam_example_image ):
    ds = decam_example_reduced_image_ds_with_wcs[0]
    refds = DataStore()
    refds.image = ref_for_decam_example_image

    import pdb; pdb.set_trace()
    # Need a source list and wcs for the ref image before we can warp it
    det = Detector( measure_psf=True )
    refds = det.run( refds )
    refds.sources.filepath = f'{refds.image.filepath}.sources.fits'
    refds.save_and_commit( no_archive=True )
    ast = AstroCalibrator()
    refds = ast.run( refds )
    refds.save_and_commit( no_archive=True )
    
    Aligner = ImageAligner()
    warpds = Aligner.run( refds, ds )
    wrapds.image.filepath = 'warpy'
    warpds.save_and_commit( no_archive=True )
    import pdb; pdb.set_trace()
    pass
