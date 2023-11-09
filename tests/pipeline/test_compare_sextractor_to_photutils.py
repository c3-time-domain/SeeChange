import pytest
import os

import numpy as np
from matplotlib import pyplot

import sep
from photutils.aperture import CircularAperture, aperture_photometry

from models.base import _logger



@pytest.mark.skipif( os.getenv('INTERACTIVE') is None, reason='Set INTERACTIVE to run this test' )
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

    phot = np.empty( ( sources.num_sources, len(sources.aper_rads) ), dtype=float )
    dphot = np.empty( ( sources.num_sources, len(sources.aper_rads) ), dtype=float )
    
    for i, aperrad in enumerate( sources.aper_rads ):
        _logger.info( f"Doing aperture radius {aperrad}..." )
        apers = CircularAperture( pos, r=aperrad )
        res = aperture_photometry( skysubdata, apers, error=error, mask=mask )
        phot[ :, i ] = res['aperture_sum']
        dphot[ :, i ] = res['aperture_sum_err']

    _logger.info( "Done with aperture photometry." )

    for i, aperrad in enumerate( sources.aper_rads ):
        wgood = ( dphot[ :, i ] > 0 ) & ( sources.good )
        sextrphot = sources.apfluxadu( apnum=i )[0][wgood]
        sextrdphot = sources.apfluxadu( apnum=i )[1][wgood]
        puphot = phot[ :, i ][wgood]
        pudphot = dphot[ :, i ][wgood]

        reldiff = ( sextrphot - puphot ) / puphot
        dreldiff = np.sqrt( ( sextrdphot / puphot ) **2 + ( ( sextrphot / puphot**2 ) * pudphot ) **2 )
        import pdb; pdb.set_trace()
        
        fig = pyplot.figure()
        ax = fig.add_subplot( 1, 1, 1 )
        ax.set_title( f"Aperrad = {aperrad}" )
        ax.set_xlabel( "PhotUtils ADU" )
        ax.set_ylabel( "( Sextractor - Photutils ) / PhotUtils" )
        ax.plot( [ puphot.min(), puphot.max() ], [ 0., 0. ] )
        ax.errorbar( puphot, reldiff, dreldiff, linestyle='none', marker='.' )
        ax.set_ylim( -1., 1. )
        fig.savefig( f'{i}.svg' )
        pyplot.close()
