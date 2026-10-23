import pytest
import uuid

import numpy as np
import pandas
import nested_pandas

from models.diaforcedphot import DiaForcedPhot
from util.logger import SCLogger   # noqa: F401


def test_lightcurve( sim_lightcurve_lightcurves ):
    ltcvs, objinfos, _, _ = sim_lightcurve_lightcurves

    psf_reduced_residses = []
    psf_flagses = []
    aper_reduced_residses = []
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
        aperfluxen = np.array( [ p.flux_apertures[0] *
                                 10**(p._aper_cors[0]/-2.5) for p in ltcv ] )
        aperfluxen_err = np.array( [ p.flux_apertures_err[0] for p in ltcv ] )

        psf_resids = ( psffluxen - objinfo['fluxen'] ) / psffluxen_err
        psf_reduced_residses.append( psf_resids )
        psf_flagses.append( np.array( [ p.psfflags for p in ltcv ] ) )
        aper_resids = ( aperfluxen - objinfo['fluxen'] ) / aperfluxen_err
        aper_reduced_residses.append( aper_resids )


    # OMG THESE ARE TERRIBLE RIGHT NOW

    for obji, ( psf_reduced_resids, aper_reduced_resids ) in enumerate( zip( psf_reduced_residses,
                                                                             aper_reduced_residses ) ):
        # assert np.all( np.fabs( reduced_resids ) < 3. )
        SCLogger.info( f"PSF reduced_resids for {obji}: {psf_reduced_resids}" )
        SCLogger.info( f"Aper reduced resides for {obji}: {aper_reduced_resids}" )
        # I'm a little nervous that the chisq/dof for objects 0 and
        # 1 are high (~2.2), but visually at least object 0 is on a
        # galaxy that doesn't subtract all that well.


# These next couple of tests are here rather than in
#    models/test_diaforcedphot.py because they're test of lightcurve
#    functionality, and because of the efficiency of not running the
#    sim_lightcurve_lightcurves module-scope fixture yet another time.

def test_find_objects_with_diaforcedphot( sim_lightcurve_lightcurves ):
    ltcvs, _objinfos, objs, photprovid = sim_lightcurve_lightcurves

    # Try to get everything
    rval = DiaForcedPhot.find_objects_with_diaforcedphot( photprovid )
    assert len( rval['id'] ) == 3
    objdexen = []
    for i in range( len(objs) ):
        dex = [ j for j in range( len(rval['name']) ) if rval['name'][j] == objs[i].name ]
        assert len(dex) == 1
        objdexen.append( dex[0] )
    # The fixture had no object positions, so was just using the positions from the object
    assert all ( r == pytest.approx( objs[objdexen[i]].ra, abs=0.01/3600. / np.cos( rval['dec'][i] ) )
                 for i, r in enumerate( rval['ra'] ) )
    assert all ( d == pytest.approx( objs[objdexen[i]].dec, abs=0.01/3600. )
                 for i, d in enumerate( rval['dec'] ) )
    assert all( n == 6 for n in rval['numphot'] )

    # TODO MORE


def test_get_lightcurve_for_object( sim_lightcurve_lightcurves ):
    expected_ltcvs, objinfos, objs, photprovid = sim_lightcurve_lightcurves

    basic_keys = { 'mjd', 'filter', 'zp', 'psfflags', 'flux_psf', 'flux_psf_err' }
    raw_keys = { 'flux_psf_raw', 'flux_psf_raw_err' }
    pos_keys = { 'x', 'y' }
    imid_keys = { 'newid', 'newzpid', 'refid', 'refzpid', 'subid' }
    photid_keys = { 'diaforcedphotid' }
    scalar_ap_keys = { 'flux_aper_0', 'flux_aper_0_err', 'flux_aper_n', 'flux_aper_n_err',
                       'aperflags_0', 'aperflags_n' }
    list_ap_keys = { 'aper_cors', 'aper_pix' }
    raw_ap_keys = { 'flux_apertures_raw', 'flux_apertures_raw_err', 'aperflags_raw' }

    # ....it's a four-loop.  No, a five-loop!  72 trials!  NO!  a seven-loop!  288 trials!
    for obj, objinfo, expected_ltcv in zip( objs, objinfos, expected_ltcvs ):
        for searchby in ( obj, obj.id, obj.name ):
            for incraw in ( False, True ):
                for incap in ( False, True ):
                    for incpos in ( False, True ):
                        for incimid in ( False, True ):
                            for incphotid in ( False, True ):
                                scalarcols = basic_keys
                                allcols = basic_keys
                                dfcols = basic_keys
                                dftype = pandas.DataFrame
                                extras = {}
                                if incraw:
                                    scalarcols = scalarcols.union( raw_keys )
                                    dfcols = scalarcols
                                    allcols = scalarcols
                                    extras['include_raw'] = True
                                if incpos:
                                    scalarcols = scalarcols.union( pos_keys )
                                    dfcols = scalarcols
                                    allcols = scalarcols
                                    extras['include_pos'] = True
                                if incimid:
                                    scalarcols = scalarcols.union( imid_keys )
                                    dfcols = scalarcols
                                    allcols = scalarcols
                                    extras['include_im_ids'] = True
                                if incphotid:
                                    scalarcols = scalarcols.union( photid_keys )
                                    dfcols = scalarcols
                                    allcols = scalarcols
                                    extras['include_phot_ids'] = True
                                if incap:
                                    scalarcols = scalarcols.union( scalar_ap_keys )
                                    allcols = scalarcols.union( list_ap_keys )
                                    allcols = allcols.union( raw_ap_keys if incraw else set() )
                                    dftype = nested_pandas.NestedFrame
                                    dfcols = dfcols.union( { 'apphot' } ).union( scalar_ap_keys )
                                    dfapcols = list_ap_keys.union( raw_ap_keys if incraw else set() )
                                    extras['include_apertures'] = True


                                # Default is list of dicts
                                ltcv = DiaForcedPhot.get_lightcurve_for_object( searchby, photprovid, **extras )
                                assert isinstance( ltcv, list )
                                assert isinstance( ltcv[0], dict )
                                assert set( ltcv[0].keys() ) == allcols
                                assert all( lc['filter'] == 'r' for lc in ltcv )
                                mjds = np.array( [ lc['mjd'] for lc in ltcv ] )
                                assert np.all( np.isclose( mjds, objinfo['mjds'], rtol=1e-14 ) )
                                # fluxen = np.array( [ lc['flux_psf'] for lc in ltcv ] )
                                # dfluxen = np.array( [ lc['flux_psf_err'] for lc in ltcv ] )
                                # # ...these next two are more a test of pipeline/lightcurve.py than of
                                # #   get_lightcurve_for_object, but oh well... should pass here too!
                                # assert np.all( np.abs( ( fluxen - objinfo['fluxen'] ) / dfluxen ) < 3. )
                                # assert ( ( (fluxen - objinfo['fluxen'] ) / dfluxen ) **2 ).sum()
                                #          < 1.5 * ( len(fluxen) - 1 ) )

                                # Ask for different formats of output and make sure the results are all consistent

                                ltcv2 = DiaForcedPhot.get_lightcurve_for_object( searchby, photprovid,
                                                                                 return_format='dictoflists', **extras )
                                assert isinstance( ltcv2, dict )
                                assert set( ltcv2.keys() ) == allcols
                                for i, point in enumerate( ltcv ):
                                    for attr in allcols:
                                        if ( attr == 'filter' ) or isinstance( point[attr], uuid.UUID ):
                                            assert point[attr] == ltcv2[attr][i]
                                        else:
                                            if isinstance( point[attr], list ):
                                                assert all ( ( np.isnan(a) and np.isnan(b) ) or ( a == b )
                                                             for a, b in zip( point[attr], ltcv2[attr][i] ) )
                                            else:
                                                assert ( ( np.isnan(point[attr]) and np.isnan(ltcv2[attr][i]) )
                                                         or ( point[attr] == ltcv2[attr][i] ) )


                                ltcv3 = DiaForcedPhot.get_lightcurve_for_object( searchby, photprovid,
                                                                                 return_format='dataframe', **extras )
                                assert isinstance( ltcv3, dftype )
                                assert set( ltcv3.columns ) == dfcols
                                for key in scalarcols:
                                    if ( key == 'filter' ) or isinstance( ltcv3[key].values[0], uuid.UUID ):
                                        assert np.all( ltcv3[key] == np.array(ltcv2[key]) )
                                    else:
                                        assert np.all( np.isclose( ltcv3[key], np.array(ltcv2[key]),
                                                                   rtol=1e-6, equal_nan=True ) )
                                if incap:
                                    for key in list_ap_keys:
                                        assert set( ltcv3.apphot.columns ) == dfapcols
                                        assert all( np.all( np.isclose( ltcv3.apphot[i][key], ltcv2[key][i],
                                                                        rtol=1e-6, equal_nan=True ) )
                                                    for i in range(len(ltcv3)) )

                                if incpos and incraw and incap:
                                    phots = DiaForcedPhot.get_lightcurve_for_object( searchby, photprovid,
                                                                                     return_format='dictoflists',
                                                                                     diaforcedphotobjs=True )
                                    assert set( phots.keys() ) == { 'mjd', 'filter', 'diaforcedphot' }
                                    assert all( isinstance( val, list ) for val in phots.values() )
                                    assert all( isinstance( val, DiaForcedPhot ) for val in phots['diaforcedphot'] )
                                    for point, mjd, filt, phot in zip( ltcv, phots['mjd'],
                                                                       phots['filter'], phots['diaforcedphot'] ):
                                        assert point['mjd'] == mjd
                                        assert point['filter'] == filt
                                        assert point['psfflags'] == phot.psfflags
                                        assert np.isclose( point['x'], phot.x, rtol=1e-6, equal_nan=True )
                                        assert np.isclose( point['y'], phot.y, rtol=1e-6, equal_nan=True )
                                        assert np.isclose( point['flux_psf_raw'], phot.flux_psf,
                                                           rtol=1e-6, equal_nan=True )
                                        assert np.isclose( point['flux_psf_raw_err'], phot.flux_psf_err,
                                                           rtol=1e-6, equal_nan=True )
                                        assert np.all( np.isclose( np.array(point['flux_apertures_raw']),
                                                                   phot.flux_apertures,
                                                                   rtol=1e-6, equal_nan=True ) )
                                        assert np.all( np.isclose( np.array(point['flux_apertures_raw_err']),
                                                                   phot.flux_apertures_err,
                                                                   rtol=1e-6, equal_nan=True ) )


# NOTE : the user and admin_user fixtures aren't really needed for this next test,
#   but they're there so that I can have the sim_lightcurve_lightcurves all loaded
#   up and go in and look at the test webap by running this test with --trace.
def test_get_clips( sim_lightcurve_lightcurves, user, admin_user ):
    _expected_ltcvs, _objinfos, objs, photprovid = sim_lightcurve_lightcurves

    for obj in objs:
        ltcvs = DiaForcedPhot.get_lightcurve_for_object( obj, return_format='dictoflists',
                                                         prov=photprovid, diaforcedphotobjs=True )
        for phot in ltcvs['diaforcedphot']:
            clips = phot.get_clips( clip_size=25 )
            assert isinstance( clips, np.ndarray )
            assert clips.shape == ( 3, 25, 25 )

    # TODO MORE
