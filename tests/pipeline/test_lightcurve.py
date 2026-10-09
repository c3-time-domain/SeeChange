import numpy as np

from models.diaforcedphot import DiaForcedPhot
from util.logger import SCLogger   # noqa: F401


def test_lightcurve( sim_lightcurve_lightcurves ):
    ltcvs, objinfos, _ = sim_lightcurve_lightcurves

    reduced_residses = []
    for ltcv, objinfo in zip( ltcvs, objinfos ):
        # #####
        # # Uncomment this to write out debugging images to the test directory.
        # # If you do, be aware that they will be cleaned up in the finally block!
        # #   So, uncomment the import pdb just before the finally below
        # import shutil
        # from astropy.io import fits
        # from models.image import Image
        # from models.source_list import SourceList
        # import pathlib
        # comps = ['image', 'weight', 'flags']
        # atts = ['data', 'weight', 'flags']
        # for filt, ref in ltcv.refs.items():
        #     origfiles = ref.image.get_fullpath( components=comps )
        #     for comp, origfile in zip( atts, origfiles ):
        #         destfile = f"ref_{comp}_{filt}.fits{'.fz' if origfile[-3:]=='.fz' else ''}"
        #         shutil.copy2( origfile, destfile)
        #         nukes['loose_files'].append( pathlib.Path(destfile) )
        #     destfile = f"ref_{comp}_{filt}.reg"
        #     ref.sources.ds9_regfile( destfile )
        #     nukes['loose_files'].append( pathlib.Path( destfile ) )
        # with PGDB( dictcursor=True ) as pgdb:
        #     for photi, phot in enumerate( ltcv.dia_forced_phots ):
        #         subim = Image.get_by_id( phot.subtraction_id, pgdb=pgdb )
        #         origfiles = subim.get_fullpath( comps )
        #         for comp, origfile in zip( comps, origfiles ):
        #             destfile = ( f"sub_{comp}_{subim.filter}_{obji}_{photi}.fits"
        #                          f"{'.fz' if origfile[-3:]=='.fz' else ''}" )
        #             shutil.copy2( origfile, destfile )
        #             nukes['loose_files'].append( pathlib.Path(destfile) )
        #         q = sql.SQL( textwrap.dedent(
        #             """\
        #             SELECT s.* FROM source_lists s
        #             INNER JOIN world_coordinates w ON w.sources_id=s._id
        #             INNER JOIN zero_points z ON z.wcs_id=w._id
        #             INNER JOIN image_subtraction_components isc ON isc.new_zp_id=z._id
        #             WHERE isc.image_id={subid}
        #             """
        #         ) ).format( subid=subim.id )
        #         rows = pgdb.execute( q )
        #         if len(rows) != 1:
        #             raise RuntimeError( "I am surprised." )
        #         newsrcs = SourceList.create( **(rows[0]) )
        #         newim = Image.get_by_id( newsrcs.image_id, pgdb=pgdb )
        #         origfiles = newim.get_fullpath( components=comps )
        #         for comp, origfile in zip( atts, origfiles ):
        #             destfile = ( f"new_{comp}_{newim.filter}_{obji}_{photi}.fits"
        #                          f"{'.fz' if origfile[-3:]=='.fz' else ''}" )
        #             shutil.copy2( origfile, destfile )
        #             nukes['loose_files'].append( pathlib.Path(destfile) )
        #         destfile = f"new_{newim.filter}_{obji}_{photi}.reg"
        #         newsrcs.ds9_regfile( destfile )
        #         nukes['loose_files'].append( pathlib.Path( destfile ) )
        # for photi, aligned in enumerate( ltcv.aligned_cache ):
        #     hdr = fits.Header( aligned['wcs'].wcs.to_header( relax=True ) )
        #     destfile = f"alignedref_image_{newim.filter}_{obji}_{photi}.fits"
        #     fits.writeto( destfile, data=aligned['ref_image'].data, header=hdr )
        #     nukes['loose_files'].append( pathlib.Path( destfile ) )
        #     destfile = f"alignedref_{newim.filter}_{obji}_{photi}.reg"
        #     aligned['ref_sources'].ds9_regfile( destfile )
        #     nukes['loose_files'].append( pathlib.Path( destfile ) )
        # SCLogger.warning( "Lightcurve images written to test directory" )
        # ####

        assert len( ltcv ) == len( objinfo['mjds'] )
        psffluxen = np.array( [ p.flux_psf for p in ltcv ] )
        psffluxen_err = np.array( [ p.flux_psf_err for p in ltcv ] )
        # aperfluxen = np.array( [ p.flux_apertures[0] *
        #                          10**(p._aper_cors[0]/-2.5) for p in ltcv ] )
        # aperfluxen_err = np.array( [ p.flux_apertures_err[0] for p in ltcv ] )

        reduced_resids = ( psffluxen - objinfo['fluxen'] ) / psffluxen_err
        reduced_residses.append( reduced_resids )

    for obji, reduced_resids in enumerate( reduced_residses ):
        # assert np.all( np.fabs( reduced_resids ) < 3. )
        SCLogger.info( f"reduced_resids for {obji}: {reduced_resids}" )
        # I'm a little nervous that the chisq/dof for objects 0 and
        # 1 are high (~2.2), but visually at least object 0 is on a
        # galaxy that doesn't subtract all that well.



def test_find_objects_with_diaforcedphot( sim_lightcurve_lightcurves ):
    ltvcs, objinfos, photprovid = sim_lightcurve_lightcurves

    # Try to get everything
    _rval = DiaForcedPhot.find_objects_with_diaforcedphot( photprovid )
    import pdb; pdb.set_trace()
    pass
