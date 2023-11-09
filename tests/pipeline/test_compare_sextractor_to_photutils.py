import pytest
import os
import numpy as np
import sep
from photutils.aperture import CircularAperture, aperture_photometry

from models.base import _logger

@pytest.mark.skipif( os.getenv('MUCKABOUT') is None, reason='Set MUCKABOUT to run this test' )
def test_compare_sextr_photutils( decam_example_reduced_image_ds ):
    ds = decam_example_reduced_image_ds
    image = ds.get_image()
    sources = ds.get_sources()
    mask = np.full( image.flags.shape, False )
    mask[ image.flags != 0 ] = True
    error = 1. / np.sqrt( image.weight )
    
    sky = sep.Background(image.data )
    skysubdata = image.data - sky

    pos = np.empty( ( sources.num_sources, 2 ) )
    pos[ :, 0 ] = sources.x
    pos[ :, 1 ] = sources.y

    import pdb; pdb.set_trace()
    
    phot = np.empty( ( sources.num_sources, len(sources.aper_rads) ), dtype=float )
    dphot = np.empty( ( sources.num_sources, len(sources.aper_rads) ), dtype=float )
    
    for i, aperrad in enumerate( sources.aper_rads ):
        _logger.info( f"Doing aperture radius {aperrad}..." )
        apers = CircularAperture( pos, r=aperrad )
        res = aperture_photometry( skysubdata, apers, error=error, mask=mask )
        phot[ :, i ] = res['aperture_sum']
        dphot[ :, i ] = res['aperture_sum_err']

    _logger.info( "Done with aperture photometry." )

    import pdb; pdb.set_trace()
    pass

