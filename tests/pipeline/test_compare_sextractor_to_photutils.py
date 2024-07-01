import pytest
import os

import numpy as np
from matplotlib import pyplot

from photutils.aperture import CircularAperture, aperture_photometry

from models.base import CODE_ROOT
from improc.sextrsky import sextrsky

from util.logger import SCLogger
from util.util import env_as_bool


@pytest.mark.skipif( not env_as_bool('INTERACTIVE'), reason='Set INTERACTIVE to run this test' )
def test_compare_sextr_photutils( decam_datastore ):
    plot_dir = os.path.join(CODE_ROOT, 'tests/plots/sextractor_comparison')
    os.makedirs( plot_dir, exist_ok=True)

    ds = decam_datastore
    image = ds.get_image()
    sources = ds.get_sources()
    mask = np.full( image.flags.shape, False )
    mask[ image.flags != 0 ] = True
    error = 1. / np.sqrt( image.weight )

    # Write a region file for each aperture
    for aperrad in sources.aper_rads:
        sources.ds9_regfile( f'{plot_dir}/{aperrad:.2f}.reg', radius=aperrad )

    # Run some photutils aperture photometry

    sky, skysig = sextrsky( image.data, image.flags,
                            boxsize=image.instrument_object.background_box_size,
                            filtsize=image.instrument_object.background_filt_size )
    skysubdata = image.data - sky

    pos = np.empty( ( sources.num_sources, 2 ) )
    pos[ :, 0 ] = sources.x
    pos[ :, 1 ] = sources.y

    phot = np.empty( ( sources.num_sources, len(sources.aper_rads) ), dtype=float )
    dphot = np.empty( ( sources.num_sources, len(sources.aper_rads) ), dtype=float )

    for i, aperrad in enumerate( sources.aper_rads ):
        SCLogger.info( f"Doing aperture radius {aperrad}..." )
        apers = CircularAperture( pos, r=aperrad )
        res = aperture_photometry( skysubdata, apers, error=error, mask=mask )
        phot[ :, i ] = res['aperture_sum']
        dphot[ :, i ] = res['aperture_sum_err']

    SCLogger.info( "Done with photutils aperture photometry." )

    # Futz around a whole lot doing comparisons

    sextr0, dsextr0 = sources.apfluxadu( apnum=0 )
    pu0 = phot[ :, 0 ]
    dpu0 = dphot[ :, 0 ]

    wbig = pu0 > 200000

    sexapercor = []
    puapercor = []

    naper = len(sources.aper_rads)
    nfigh = naper // 2 + 1 if naper %2 == 1 else naper // 2
    fig = pyplot.figure( figsize=(12, 4*nfigh), dpi=150, layout='tight' )
    figzoom = pyplot.figure( figsize=(12, 4*nfigh), dpi=150, layout='tight' )
    for i, aperrad in enumerate( sources.aper_rads ):
        sextrphot, sextrdphot = sources.apfluxadu( apnum=i )
        wgood = ( ( dphot[ :, i ] > 0 ) & ( sources.good ) &
                  ( phot[ :, i ] > 5.*dphot[ :, i] ) &
                  ( sextrphot > 5.*sextrdphot ) )
        sextrphot = sextrphot[wgood]
        sextrdphot = sextrdphot[wgood]
        puphot = phot[ :, i ][wgood]
        pudphot = dphot[ :, i ][wgood]

        reldiff = ( sextrphot - puphot ) / puphot
        # Really, the two uncertainties are highly correlated because they come
        # from the same underlying image data, but oh well.
        # dreldiff = np.sqrt( ( ( sextrdphot * puphot ) / (sextrphot**2) ) ** 2 +
        #                     ( pudphot / sextrphot ) **2 )
        dreldiff = np.sqrt( ( sextrdphot / puphot ) **2 +
                            ( ( pudphot * sextrphot ) / (puphot**2) ) **2 )
        ax = fig.add_subplot( nfigh, 2, i+1 )
        ax.set_title( f"Aperrad = {aperrad:.3f}" )
        ax.set_xlabel( "PhotUtils ADU" )
        ax.set_ylabel( "( SExtractor - PhotUtils ) / PhotUtils" )
        ax.plot( [ puphot.min(), puphot.max() ], [ 0., 0. ] )
        ax.errorbar( puphot, reldiff, dreldiff, linestyle='none', marker='.' )
        ax.set_ylim( -1.25, 1.25 )

        axzoom = figzoom.add_subplot( nfigh, 2, i+1 )
        axzoom.set_title( f"Aperrad = {aperrad:.3f}" )
        axzoom.set_xlabel( "PhotUtils ADU" )
        axzoom.set_ylabel( "( SExtractor - PhotUtils ) / PhotUtils" )
        axzoom.plot( [ puphot.min(), puphot.max() ], [ 0., 0. ] )
        axzoom.errorbar( puphot, reldiff, dreldiff, linestyle='none', marker='.' )
        axzoom.set_ylim( -0.1, 0.1 )

        sexac = np.median( 2.5 * np.log10( sources.apfluxadu( apnum=i )[0][wgood&wbig] / sextr0[wgood&wbig] ) )
        puac = np.median( 2.5 * np.log10( phot[ :, i ][wgood&wbig] / pu0[wgood&wbig] ) )
        sexapercor.append( sexac )
        puapercor.append( puac )

    fig.savefig( f'{plot_dir}/reldiff.svg' )
    pyplot.close( fig )
    figzoom.savefig( f'{plot_dir}/reldiff_zoom.svg' )
    pyplot.close( figzoom )

    fig = pyplot.figure( figsize=(8, 6), dpi=150, layout='tight' )
    ax = fig.add_subplot( 1, 1, 1 )
    ax.set_title( "Aperture correction" )
    ax.set_ylabel( "Corr" )
    ax.set_xlabel( "Ap Rad" )
    ax.plot( sources.aper_rads, sexapercor, marker='.', label='sextractor' )
    ax.plot( sources.aper_rads, puapercor, marker='.', label='photutils' )
    ax.legend()
    fig.savefig( f'{plot_dir}/apcor.svg' )
    pyplot.close( fig )

    fig = pyplot.figure( figsize=(8, 6), dpi=150, layout='tight' )
    ax = fig.add_subplot( 1, 1, 1 )
    ax.set_title( "SExtractor PSF vs. aperture" )
    ax.set_ylabel( f"PSF Flux / Aperture={sources.aper_rads[4]:.2f}" )
    ax.set_xlabel( f"Aperture={sources.aper_rads[4]:.2f} ADU" )
    ax.plot( sources.apfluxadu( apnum=4 )[0], sources.psffluxadu()[0] / sources.apfluxadu( apnum=4 )[0],
             linestyle='none', marker='.' )
    ax.plot( ax.get_xlim(), [ 1, 1 ] )
    ax.set_ylim( 0., 1.2 )
    fig.savefig( f'{plot_dir}/psf_ap_vs_ap_.svg' )
    pyplot.close( fig )

    for big in range( 1, len( sources.aper_rads ) ):
        # Variables are called _6 because originally
        #  this was not in a for loop and I hardcoded
        #  the big apnum to 6.
        s6, ds6 = sources.apfluxadu( apnum=big )
        s0, ds0 = sources.apfluxadu( apnum=0 )
        p6 = phot[ :, big ]
        dp6 = dphot[ :, big ]
        p0 = phot[ :, 0 ]
        dp0 = dphot[ :, 0 ]

        sextr_6_0 = s6 / s0
        dsextr_6_0 = np.sqrt( ( ds6 / s0 ) **2 + ( ds0 * s6 / (s0**2) ) **2 )
        pu_6_0 = p6 / p0
        dpu_6_0 = np.sqrt( ( dp6 / p0 ) **2 + ( dp0 * p6 / (p0**2) ) **2 )

        wgood = sources.good & ( dphot[ :, big ] > 0 ) & ( dphot[ :, 0 ] > 0 )
        smallerr = dpu_6_0 < 0.05
        # wuse = wgood & smallerr
        wuse = wgood

        sextr_6_0 = sextr_6_0[wuse]
        dsextr_6_0 = dsextr_6_0[wuse]
        pu_6_0 = pu_6_0[wuse]
        dpu_6_0 = dpu_6_0[wuse]

        fig = pyplot.figure( figsize=(8, 6), dpi=150, layout='tight' )
        ax = fig.add_subplot( 2, 1, 1 )
        ax.set_title( "photutils" )
        ax.set_xlabel( f'Aper={sources.aper_rads[0]:.3f} ADU' )
        ax.set_ylabel( f'Aper={sources.aper_rads[big]:.3f} / Aper={sources.aper_rads[0]:.3f}' )
        # ax.set_ylim( 0.6, 1.4 )
        ax.set_ylim( 1.15, 1.23 )
        ax.errorbar( p0[wuse], pu_6_0, dpu_6_0, linestyle='none', marker='.' )
        wtmean = ( pu_6_0 / (dpu_6_0**2) ).sum() / ( 1. / (dpu_6_0**2) ).sum()
        dwtmean = ( 1. / (dpu_6_0**2) ).sum()
        ax.plot( ax.get_xlim(), [ wtmean, wtmean ], label=f'wtmean={wtmean:.2f}±{dwtmean:.1e}' )
        med = np.median( pu_6_0 )
        ax.plot( ax.get_xlim(), [ med, med ], label=f'median={med:.2f}' )
        ax.legend()

        ax = fig.add_subplot( 2, 1, 2 )
        ax.set_title( "sextractor" )
        ax.set_xlabel( f'Aper={sources.aper_rads[0]:.3f} ADU' )
        ax.set_ylabel( f'Aper={sources.aper_rads[big]:.3f} / Aper={sources.aper_rads[0]:.3f}' )
        # ax.set_ylim( 0.6, 1.4 )
        ax.set_ylim( 1.15, 1.23 )
        ax.errorbar( s0[wuse], sextr_6_0, dsextr_6_0, linestyle='none', marker='.' )
        wtmean = ( sextr_6_0 / (dsextr_6_0**2) ).sum() / ( 1. / (dsextr_6_0**2) ).sum()
        dwtmean = ( 1. / (dsextr_6_0**2) ).sum()
        ax.plot( ax.get_xlim(), [ wtmean, wtmean ], label=f'wtmean={wtmean:.2f}±{dwtmean:.1e}' )
        med = np.median( sextr_6_0 )
        ax.plot( ax.get_xlim(), [ med, med ], label=f'med={med:.2f}' )
        ax.legend()

        fig.savefig( f'{plot_dir}/twoaperratio_vs_oneaper_ap{big}.svg' )
        pyplot.close( fig )

        fig = pyplot.figure()
        ax = fig.add_subplot( 1, 1, 1 )
        ax.set_ylabel( f'Aper={sources.aper_rads[big]:.3f} ADU / Aper={sources.aper_rads[0]:.3f} ADU' )
        ax.set_xlabel( "sextractor" )
        ax.set_ylabel( "photutils" )
        wbigflux = p0[wuse] > 200000
        ax.plot( sextr_6_0[wbigflux], pu_6_0[wbigflux], linestyle='none', marker='.' )

        fig.savefig( f'{plot_dir}/twoaperratio_sexvsphot_ap{big}.svg' )
        pyplot.close( fig )

