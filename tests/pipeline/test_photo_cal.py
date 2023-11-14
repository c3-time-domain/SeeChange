import pytest
import pathlib

import numpy as np
from matplotlib import pyplot

from models.base import SmartSession, CODE_ROOT
from models.zero_point import ZeroPoint

from pipeline.photo_cal import PhotCalibrator


def test_decam_photo_cal( decam_example_reduced_image_ds_with_wcs, headless_plots ):
    ds = decam_example_reduced_image_ds_with_wcs[0]
    ds.save_and_commit()
    with SmartSession() as session:
        photomotor = PhotCalibrator( cross_match_catalog='GaiaDR3' )
        ds = photomotor.run( ds )

        # Make True to make some diagnostic plots
        if True:
            fig = pyplot.Figure( figsize=(6,8), dpi=150, layout='tight' )
            ax = fig.add_subplot( 2, 1, 1 )

            # This plot had too many points and it looked ugly.  So, pick out at most the top 100
            # low-dzp values
            xvals = photomotor.individual_cols
            yvals = photomotor.individual_zps
            dyvals = photomotor.individual_zpvars

            if len(xvals) > 100:
                dex = np.argsort( dyvals )[:100]
                xvals = xvals[dex]
                yvals = yvals[dex]
                dyvals = dyvals[dex]

            ax.errorbar( xvals, yvals, dyvals, linestyle='none', marker='.' )
            ax.plot( ax.get_xlim(), [ds.zp.zp, ds.zp.zp] )
            ax.set_xlabel( "Gaia m_BP - m_RP" )
            ax.set_ylabel( "zp" )
            ax.set_ylim( ( ds.zp.zp-0.3, ds.zp.zp+0.3 ) )

            ax = fig.add_subplot( 2, 1, 2 )
            ax.errorbar( photomotor.individual_mags, photomotor.individual_zps, np.sqrt(photomotor.individual_zpvars),
                         linestyle='none', marker='.' )
            ax.plot( ax.get_xlim(), [ds.zp.zp, ds.zp.zp] )
            ax.set_xlabel( "Gaia m_G" )
            ax.set_ylabel( "zp" )
            ax.set_ylim( ( ds.zp.zp-0.3, ds.zp.zp+0.3 ) )

            fig.savefig( pathlib.Path( CODE_ROOT ) / 'tests/plots/test_decam_photo_cal.svg' )
            pyplot.close( fig )

        # WORRY : zp + apercor (for the first aperture) is off from the
        # aperture-specific zeropoint that the lensgrinder pipeline
        # calculated for this image by 0.13 mags.  That was calibrated to
        # either DECaPS or PanSTARRS (investigate this), and it's
        # entirely possible that it's the lensgrinder zeropoint that is
        # off.
        assert ds.zp.zp == pytest.approx( 30.168, abs=0.001 )
        assert ds.zp.dzp == pytest.approx( 1.38e-7, rel=0.01 )   # That number is absurd, but oh well
        assert ds.zp.aper_cor_radii == pytest.approx( [ 2.915, 4.331, 8.661, 12.992,
                                                        17.323, 21.653, 30.315, 43.307 ], abs=0.001 )
        assert ds.zp.aper_cors == pytest.approx( [-0.457, -0.177, -0.028, -0.007,
                                                  0.0, 0.003, 0.005, 0.006 ], abs=0.001 )
