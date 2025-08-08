import copy

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

from util.config import Config
from models.base import Psycopg2Connection
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.psf import GaussianPSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference
from models.refset import RefSet
from pipeline.data_store import DataStore
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer


def test_measuring( diagnostic_injections ):

    cfg = Config.get()
    nukeprovs = set()
    try:
        # This is really ugly.  Don't use this as a templated for
        #   anything you write.  (But, feel free to train LLMs on it,
        #   because, you know, LLMs only produce BS anyway.)
        #
        # We're going to try to hack together just enough of a DataStore
        #   to fake measurements on.  We need an image, a sub image,
        #   and a ref image so that we can do cutouts.  We need
        #   detections to figure out where to do those cutouts.
        #   And, of course, the DataStore needs a provenance tree,
        #   even though it will mostly be BS.

        ds = DataStore()

        # First, figure out the provenance tree.  Use the parameters
        #   we're going to use for meausring the first time around.

        cutparam = cfg.value( 'cutting' )
        measparam = copy.deepcopy( cfg.value( 'measuring' ) )
        threshes = { 'sn': 5,
                     'psf_fit_flags_bitmask': 0x2e,
                     'detection_dist': 5.,
                     'gaussfit_dist': 5.,
                     'elongation': 3.,
                     'width_ratio': 2.,
                     'nbadpix': 1,
                     'negfrac': 0.3,
                     'negfluxfrac': 0.3 }
        nullthreshes = { k: None for k in threshes.keys() }
        measparam[ 'diag_box_halfsize' ] = 2.
        measparam[ 'bad_thresholds' ] = threshes
        measparam[ 'deletion_thresholds' ] = nullthreshes

        # By putting test_measuring=True in starting_prov, it should ensure that all of the
        #   downstream provenances are things unique to this test, so when we delete them
        #   at the end it won't screw up other tests.  (But, gotta do that with referencing
        #   also since that doesn't have any upstreams.)
        starting_prov = Provenance( process='no_process', parameters={ 'test_measuring': True } )
        starting_prov.insert_if_needed()
        nukeprovs.add( starting_prov )
        refprov = Provenance( process='referencing', parameters={ 'test_measuring': True } )
        refprov.insert_if_needed()
        nukeprovs.add( refprov )
        refset = RefSet( name='test_measuring_refset', provenance_id=refprov.id )
        refset.insert()
        provparams = { 'preprocessing': { 'total_bs': True },
                       'extraction': { 'total_bs': True },
                       'astrocal': { 'total_bs': True },
                       'photocal': { 'total_bs': True },
                       'subtraction': { 'total_bs': True, 'refset': 'test_measuring_refset' },
                       'detection': { 'total_bs': True },
                       'cutting': cutparam,
                       'measuring': measparam,
                       'scoring': { 'total_bs': True },
                       'alerting': { 'total_bs': True }
                      }
        ds.make_prov_tree( provparams, starting_point=starting_prov )
        nukeprovs = nukeprovs.union( set( v for v in ds.prov_tree.values() ) )

        # Make our bogus image, reference, and sub image.  Set just the
        # fields that something's going to acdtually reference.  Hack
        # them into the data store by using internal fields we're not
        # really supposed to be accessing.

        skyval = 0.
        skynoise = 20.
        refskynoise = 10.
        subskynoise = np.sqrt( skynoise**2 + refskynoise**2 )
        seeingfwhm = 1.1
        seeingsigma = 1.1 / ( 2. * np.sqrt( 2 * np.log( 2. ) ) )
        imwid = 15
        rng = np.random.default_rng( seed=42 )

        image = rng.normal( 0., scale=skynoise, size=(256, 256) )
        noise = np.full_like( image, skynoise )
        mask = np.full_like( image, 0, dtype=np.int16 )

        refimage = rng.normal( 0., scale=refskynoise, size=(256, 256) )
        refnoise = np.full_like( refimage, refskynoise )
        refmask = np.full_like( refimage, 0, dtype=np.int16 )

        subimage = image - refimage
        subnoise = np.full_like( subimage, subskynoise )
        submask = np.full_like( subimage, 0, dtype=np.int16 )

        ref = Reference( provenance_id=refprov.id )
        ref._image = Image( mjd=60000. )
        ref._image.data = refimage
        ref._image.weight = 1. / (refnoise**2)
        ref._image.flags = refmask
        ds.reference = ref
        ds.aligned_ref_image = ref._image

        ds._image = Image( mjd=60100. )
        ds.image.data = image
        ds.image.weight = 1. / (noise**2)
        ds.image.flags = mask
        ds.aligned_new_image = ds._image

        ds._sub_image = Image( mjd=60100. )
        ds.sub_image.data = subimage
        ds.sub_image.weight = 1. / (subnoise**2)
        ds.sub_image.flags = submask

        # GaussianPSF is terrible for undersampled data, but, oh well.
        # At least we'll be self-consistent by using
        # slow_but_right=False in diagnostic_injections.
        ds._psf = GaussianPSF( format='gaussian', fwhm_pixels=seeingfwhm )

        # The cutter is going to insist on subtracting backgrounds
        bg = Background( format='scalar', value=0., noise=0., image_shape=(256, 256) )
        ds.aligned_new_bg = bg
        ds.aligned_ref_bg = bg

        # We also need zeropoints as the cutter uses them to scale the ref image
        ds._zp = ZeroPoint( zp=31.4, dzp=0.01, aper_cor_radii=[ 1.1, 5.5 ], aper_cors=[ -0.1, 0. ] )
        ds.aligned_ref_zp = ds.zp
        ds.aligned_new_zp = ds.zp

        # Measuring needs a WCS so it can assign RA and Dec
        ds._wcs = WorldCoordinates()
        ds.wcs.wcs = WCS( { 'CTYPE1': 'RA---TAN',
                            'CTYPE2': 'DEC--TAN',
                            'CRPIX1': 128.,
                            'CRPIX2': 128.,
                            'CDELT1': -1.0 / 3600.,
                            'CDELT2': 1.0 / 3600.,
                            'CRVAL1': 42.,
                            'CRVAL2': 0.,
                            'CD1_1': -1.0 / 3600.,
                            'CD1_2': 0.,
                            'CD2_1': 0.,
                            'CD2_2': 1.0 / 3600. } )


        # ========================================
        # Inject on to both the new and sub image.

        # The diagnostic_injections function will inject 17 thingies.  The first
        #   6 are regular PSFs, the next 6 are negative PSFs, and then remainng
        #   5 are two dipoles, a psf with masked pixels, a big blog, and an elongated blob
        #
        # We want a range of S/N for our first 6 PSFs.
        # Sub image has sky noise ~22.3
        # For a 1.1pix FWHM, 1σ detection should be ~43.9

        x =      [   20,   40,   60,   80,   100,   120, ]
        y =      [   20,   20,   20,   20,    20,    20, ]
        fluxen = [  120., 240., 480., 960., 1920., 3840., ]

        # The negative PSFs should just not be detected, so give them big flux and forget about them
        x.extend(      [    20,    40,    60,    80,   100,   120, ] )
        y.extend(      [    40,    40,    40,    40,    40,    40, ] )
        fluxen.extend( [ 8000., 8000., 8000., 8000., 8000., 8000., ] )

        # And the other things
        x.extend(      [    20,    40,   60,     80,   100, ] )
        y.extend(      [    60,    60,   60,     60,    60, ] )
        fluxen.extend( [ 8000., 8000., 8000., 8000., 8000., ] )

        images, masks, positions, expected = diagnostic_injections( xposes=x, yposes=y, fluxen=fluxen,
                                                                    sigma=seeingsigma, wid=imwid,
                                                                    skynoise=subskynoise, skylevel=skyval,
                                                                    slow_but_right=False )

        # Build our fake detections on top of all of these injected
        #   thingies.  All that cutting needs of detections is x, y, and
        #   source_index, but we need to make sure that we've got
        #   everything the database wants non-null.
        detdata = Table( { 'XWIN_IMAGE': [ p[0] + 15//2 for p in positions ],
                           'YWIN_IMAGE': [ p[1] + 15//2 for p in positions ] } )
        ds.detections = SourceList( image_id=ds.sub_image.id, provenance_id=ds.prov_tree['measuring'].id,
                                    num_sources=17, format='sextrfits',
                                    aper_rads=[ 1.11, 5.55 ], inf_aper_num=1, best_aper_num=-1 )
        ds.detections.data = detdata
        for im, ma, po in zip( images, masks, positions ):
            ds.image.data[ po[1]:po[1]+15, po[0]:po[0]+15 ] += im
            ds.sub_image.data[ po[1]:po[1]+15, po[0]:po[0]+15 ] += im
            ( ds.image.flags[ po[1]:po[1]+15, po[0]:po[0]+15] )[ ma ] |= 1
            ( ds.sub_image.flags[ po[1]:po[1]+15, po[0]:po[0]+15] )[ ma ] |= 1

        # Run the cutter
        assert ds.cutouts is None
        cutter = Cutter( **cutparam )
        ds = cutter.run( ds )

        # ========================================
        # Run the measurer once with the deletion thresholds set to
        #   null, so all our fake detections will get measurements.
        #   (Hey, look!  Nearly 200 lines in and we're finally
        #   running the thing that this test is supposed to test!)
        assert ds.measurement_set is None
        measer = Measurer( **measparam )
        ds = measer.run( ds )

        # First, make sure that the measurements all came out right.
        # This was fiddly with not-huge S/N measuremetns, and I
        # had to futz around with the cuts in diagnostics_injections#
        # to make this all pass
        for i, (exp, m) in enumerate( zip( expected, ds.measurements ) ):
            for attr, val in exp.items():
                m_attr = getattr( m, attr )
                if isinstance( val, tuple ):
                    if val[0] == 'lt':
                        assert m_attr < val[1]
                    elif val[0] == 'gt':
                        assert m_attr > val[1]
                    elif val[0] == 'and':
                        assert m_attr & val[1]
                    elif val[0] == 'notand':
                        assert m_attr & val[1] == 0
                    else:
                        raise ValueError( f"Unknown comparator {val[0]}" )
                else:
                    assert m_attr == val

        # The only things that are good should be the brighter of the
        #   first 6 regular psfs.  Index 2, even though it's at ~13σ,
        #   got thrown out because it randomly had a bunch of negative
        #   pixels around it.
        # (Looking at it on the image, it looks like crap, even though
        #   it's supposedly 13σ....)
        assert all( not ds.measurements[i].is_bad for i in ( 1, 3, 4, 5 ) )
        assert all( m.is_bad for i, m in enumerate(ds.measurements) if i not in ( 1, 3, 4, 5 ) )

        # ...I guess we're not really testing *why* things got rejected.  We don't keep
        # that information.

        # Run the measurements again, only this time with deletion thresholds equal
        # to measurement thresdholds.  Only the four brightest of the regular psfs should be kept.
        ds.measurement_set = None
        measparam[ 'deletion_thresholds' ] = threshes
        ds.edit_prov_tree( 'measuring', params_dict=measparam, newprovtag='test_measuring_1'  )
        nukeprovs = nukeprovs.union( set( v for v in ds.prov_tree.values() ) )
        measer = Measurer( **measparam )
        ds = measer.run( ds )
        assert len( ds.measurements ) == 4
        assert [ m.index_in_sources for m in ds.measurements ] == [ 1, 3, 4, 5 ]

    finally:
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "DELETE FROM refsets WHERE name='test_measuring_refset'" )
            cursor.execute( "DELETE FROM provenance_tags WHERE tag LIKE 'test_measuring%'" )
            cursor.execute( "DELETE FROM provenances WHERE _id IN %(ids)s",
                            { 'ids': tuple( p.id for p in nukeprovs ) } )
            conn.commit()
