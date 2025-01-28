import os
import pytest
import uuid

import sqlalchemy as sa

from models.base import SmartSession, Psycopg2Connection
from models.instrument import SensorSection
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.deepscore import DeepScore
from models.provenance import Provenance, ProvenanceTag

from pipeline.data_store import DataStore, ProvenanceTree
from pipeline.top_level import Pipeline


def test_make_prov_tree( decam_exposure, decam_reference ):
    provs_created = set()
    try:
        pars = {}
        for i, step in enumerate( Pipeline.ALL_STEPS ):
            pars[step] = { 'foo': i }
        pars['subtraction']['refset'] = 'test_refset_decam'
        ds = DataStore( decam_exposure, 'S2' )

        # Make sure it spits at us if paramteres are missing for a step
        with pytest.raises( ValueError, match="Step.*not in pars" ):
            ds.make_prov_tree( ['foo', 'bar'], pars )

        # Make sure it spits at us if we include referencing in steps
        with pytest.raises( ValueError, match="Steps must not include referencing" ):
            ds.make_prov_tree( [ 'preprocessing', 'referencing' ], { 'preprocessing': {}, 'referencing': {} } )

        ds.make_prov_tree( Pipeline.ALL_STEPS, pars )
        # Don't delete the exposure or referencing provenances, they were created in a fixture
        provs_created = set( v for k, v in ds.prov_tree.items() if k not in ('referencing','exposure') )

        # Even though 'exposure' and 'referencing' weren't in the list
        #   of steps, they should have been created
        assert set( ds.prov_tree.keys() ) == set( ['exposure', 'referencing'] + Pipeline.ALL_STEPS )
        for i, step in enumerate( Pipeline.ALL_STEPS ):
            assert step in ds.prov_tree
            assert step in ds.prov_tree.upstream_steps
            assert isinstance( ds.prov_tree[step], Provenance )
            assert ds.prov_tree[step].parameters['foo'] == i
            assert ( set( p.id for p in ds.prov_tree[step].upstreams ) ==
                     set( ds.prov_tree[u].id for u in ds.prov_tree.upstream_steps[step] ) )

        # Make sure no provenance tag was created
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "SELECT _id FROM provenance_tags WHERE provenance_id IN %(pid)s",
                            { 'pid': tuple( v.id for k, v in ds.prov_tree.items() if k not in ( 'exposure',
                                                                                                'referencing' ) ) } )
            assert len(cursor.fetchall()) == 0

        # Make sure we can create the provenance tag
        ds.make_prov_tree( Pipeline.ALL_STEPS, pars, provtag='test_data_store_test_make_prov_tree' )
        for i, step in enumerate( Pipeline.ALL_STEPS ):
            # ...quick recheck, we already tested this above
            assert step in ds.prov_tree
            assert step in ds.prov_tree.upstream_steps
            assert isinstance( ds.prov_tree[step], Provenance )
            assert ds.prov_tree[step].parameters['foo'] == i
        with SmartSession() as sess:
            tags = sess.query( ProvenanceTag ).filter( ProvenanceTag.tag=='test_data_store_test_make_prov_tree' ).all()
            assert set( t.provenance_id for t in tags ) == set( p.id for p in ds.prov_tree.values() )

        # Make sure that the prov tree gets replaced and only some steps
        #   created if we give fewer than all steps
        somesteps = [ 'preprocessing', 'backgrounding', 'extraction', 'wcs', 'zp' ]
        ds.make_prov_tree( somesteps, pars )
        assert set( ds.prov_tree.keys() ) == set( ['exposure'] + somesteps )

    finally:
        if len(provs_created) > 0:
            with Psycopg2Connection() as conn:
                cursor = conn.cursor()
                cursor.execute( "DELETE FROM provenances WHERE _id IN %(id)s",
                                { 'id': tuple( [ p.id for p in provs_created ] ) } )
                cursor.execute( "DELETE FROM provenance_tags WHERE tag='test_data_store_test_make_prov_tree'" )
                conn.commit()


def test_make_prov_tree_no_ref_prov( decam_exposure ):
    provs_created = set()
    try:
        pars = {}
        for i, step in enumerate( Pipeline.ALL_STEPS ):
            pars[step] = { 'foo': i }
        pars['subtraction']['refset'] = 'test_refset_decam'
        ds = DataStore( decam_exposure, 'S2' )

        with pytest.raises( ValueError, match="No reference set with name test_refset_decam found in the database!" ):
            ds.make_prov_tree( Pipeline.ALL_STEPS, pars )

        somesteps = [ 'preprocessing', 'backgrounding', 'extraction', 'wcs', 'zp' ]
        ds.make_prov_tree( somesteps, pars )
        assert set( ds.prov_tree.keys() ) == set( ['exposure'] + somesteps )

        ds.make_prov_tree( Pipeline.ALL_STEPS, pars, ok_no_ref_prov=True )
        assert set( ds.prov_tree.keys() ) == { 'exposure', 'preprocessing', 'backgrounding', 'extraction', 'wcs', 'zp' }

    finally:
        if len(provs_created) > 0:
            with Psycopg2Connection() as conn:
                cursor = conn.cursor()
                cursor.execute( "DELETE FROM provenances WHERE _id IN %(id)s",
                                { 'id': tuple( [ p.id for p in provs_created ] ) } )
                conn.commit()


def test_get_provenance( decam_exposure, decam_reference, pipeline_for_tests ):
    ds = DataStore( decam_exposure, 'S2' )

    with pytest.raises( RuntimeError, match="get_provenance requires the DataStore to have a provenance tree" ):
        _ = ds.get_provenance( 'preprocessing' )

    pipeline_for_tests.make_provenance_tree( ds, no_provtag=True, all_steps=True )

    with pytest.raises( ValueError, match="No provenance for foo in provenance tree" ):
        _ = ds.get_provenance( "foo" )

    with pytest.raises( ValueError, match="Passed pars_dict does not match parameters" ):
        _ = ds.get_provenance( "preprocessing", { "cats": [ "Guiseppe", "Antonin" ] } )

    subp = ds.get_provenance( 'subtraction' )
    assert subp.id == ds.prov_tree['subtraction'].id

    subp = ds.get_provenance( 'subtraction', ds.prov_tree['subtraction'].parameters )
    assert subp.id == ds.prov_tree['subtraction'].id



def test_set_prov_tree():
    refimgprov = Provenance( process='preprocessing', parameters={ 'ref': True } )
    refsrcprov = Provenance( process='extraction', parameters={ 'ref': True } )

    provs = { 'exposure': Provenance( process='exposure', parameters={} ) }
    provs['preprocessing'] = Provenance( process='preprocessing',
                                         upstreams=[ provs['exposure'] ],
                                         parameters={ 'a': 4 } )
    provs['extraction'] = Provenance( process='extraction',
                                      upstreams=[ provs['preprocessing'] ],
                                      parameters={ 'b': 8 } )
    provs['referencing'] = Provenance( process='referencing',
                                       upstreams=[ refimgprov, refsrcprov ],
                                       parameters={ 'c': 15 } )
    provs['subtraction'] = Provenance( process='subtraction',
                                       upstreams=[ provs['referencing'],
                                                   provs['preprocessing'], provs['extraction'] ],
                                       parameters={ 'd': 16 } )
    provs['detection'] = Provenance( process='detection',
                                     upstreams=[ provs['subtraction' ] ],
                                     parameters={ 'd': 23 } )
    provs['cutting'] = Provenance( process='cutting',
                                   upstreams=[ provs['detection' ] ],
                                   paramters={ 'e': 42 } )
    provs['measuring'] = Provenance( process='measuring',
                                     upstreams=[ provs['cutting' ] ],
                                     parameters={ 'f': 49152 } )

    upstreams = { 'exposure': [],
                  'preprocessing': ['exposure'],
                  'extraction': ['preprocessing'],
                  'referencing': [],
                  'subtraction': ['referencing', 'preprocessing', 'extraction'],
                  'detection': ['subtraction'],
                  'cutting': ['detection'],
                  'measuring': ['cutting'] }

    refimgprov.insert_if_needed()
    refsrcprov.insert_if_needed()
    for prov in provs.values():
        prov.insert_if_needed()


    ds = DataStore()
    assert ds.prov_tree is None

    with pytest.raises(TypeError, match="When initializing a DataStore's provenance tree, must pass a ProvenanceTree"):
        ds.set_prov_tree( provs )

    provtree = ProvenanceTree( provs, upstreams )
    ds.set_prov_tree( provtree )

    assert all( [ prov.id == provs[process].id for process, prov in ds.prov_tree.items() ] )

    # Verify that downstreams get wiped out if we set an upstream
    for i, process in enumerate( provs.keys() ):
        toset = { list(provs.keys())[j]: provs[process] for j in range(0,i+1) }
        ds.set_prov_tree( toset )
        for dsprocess in list(provs.keys())[:i+1]:
            assert dsprocess in ds.prov_tree
        for dsprocess in list(provs.keys())[i+1:]:
            if dsprocess != 'referencing':
                assert dsprocess not in ds.prov_tree
        # reset
        ds.set_prov_tree( provtree )

    # Clean up
    with SmartSession() as sess:
        idstodel = [ refimgprov.id, refsrcprov.id ]
        idstodel.extend( list( provs.keys() ) )
        sess.execute( sa.delete( Provenance ).where( Provenance._id.in_( idstodel ) ) )
        sess.commit()


# The fixture gets us a datastore with everything saved and committed
# The fixture takes some time to build (even from cache), so glom
# all the tests together in one function.

# (TODO: think about test fixtures, see if we could easily (without too
# much repeated code) have module scope (and even session scope)
# fixtures with decam_datastore alongside the function scope fixture.)

def test_data_store( decam_datastore ):
    ds = decam_datastore

    # ********** Test basic attributes **********

    origexp = ds._exposure_id
    assert ds.exposure_id == origexp

    tmpuuid = uuid.uuid4()
    ds.exposure_id = tmpuuid
    assert ds._exposure_id == tmpuuid
    assert ds.exposure_id == tmpuuid

    with pytest.raises( Exception ):
        ds.exposure_id = 'this is not a valid uuid'

    ds.exposure_id = origexp

    origimg = ds._image_id
    assert ds.image_id == origimg

    tmpuuid = uuid.uuid4()
    ds.image_id = tmpuuid
    assert ds._image_id == tmpuuid
    assert ds.image_id == tmpuuid

    with pytest.raises( Exception ):
        ds.image_id = 'this is not a valud uuid'

    ds.image_id = origimg

    exp = ds.exposure
    assert isinstance( exp, Exposure )
    assert exp.instrument == 'DECam'
    assert exp.format == 'fits'

    # This was not showing up as none; the section attribute must have
    #   been accessed earlier (in a fixture?)
    # assert ds._section is None
    sec = ds.section
    assert isinstance( sec, SensorSection )
    assert sec.identifier == ds.section_id

    assert isinstance( ds.image, Image )
    assert isinstance( ds.sources, SourceList )
    assert isinstance( ds.bg, Background )
    assert isinstance( ds.psf, PSF )
    assert isinstance( ds.wcs, WorldCoordinates )
    assert isinstance( ds.zp, ZeroPoint )
    assert isinstance( ds.sub_image, Image )
    assert isinstance( ds.detections, SourceList )
    assert isinstance( ds.cutouts, Cutouts )
    assert isinstance( ds.measurements, list )
    assert all( [ isinstance( m, Measurements ) for m in ds.measurements ] )
    assert isinstance( ds.aligned_ref_image, Image )
    assert isinstance( ds.aligned_new_image, Image )

    # Test that if we set a property to None, the dependent properties cascade to None

    props = [ 'image', 'sources', 'sub_image', 'detections', 'cutouts', 'measurements' ]
    sourcesiblings = [ 'bg', 'psf', 'wcs', 'zp' ]
    origprops = { prop: getattr( ds, prop ) for prop in props }
    origprops.update( { prop: getattr( ds, prop ) for prop in sourcesiblings } )

    refprov = Provenance.get( ds.reference.provenance_id )

    def resetprops():
        for prop in props:
            if prop == 'ref_image':
                # The way the DataStore was built, it doesn't have a 'referencing'
                #   provenance in its provenance_tree, so we have to
                #   provide one.
                ds.get_reference( provenances=[ refprov ] )
                assert ds.ref_image.id == origprops['ref_image'].id
            else:
                setattr( ds, prop, origprops[ prop ] )
                if prop == 'sources':
                    for sib in sourcesiblings:
                        setattr( ds, sib, origprops[ sib ] )

    for i, prop in enumerate( props ):
        setattr( ds, prop, None )
        for subprop in props[i+1:]:
            assert getattr( ds, subprop ) is None
            if subprop == 'sources':
                assert all( [ getattr( ds, p ) is None for p in sourcesiblings ] )
        resetprops()
        if prop == 'sources':
            for sibling in sourcesiblings:
                setattr( ds, sibling, None )
                for subprop in props[ props.index('sources')+1: ]:
                    assert getattr( ds, subprop ) is None
                resetprops()


    # Test that we can't set a dependent property if the parent property isn't set

    ds.image = None
    for i, prop in enumerate( props ):
        if i == 0:
            continue
        with pytest.raises( RuntimeError, match=f"Can't set DataStore {prop} until it has" ):
            setattr( ds, prop, origprops[ prop ] )
        if props[i-1] == 'sources':
            for subprop in sourcesiblings:
                with pytest.raises( RuntimeError, match=f"Can't set DataStore {subprop} until it has a sources." ):
                    setattr( ds, subprop, origprops[ subprop ] )
        setattr( ds, props[i-1], origprops[ props[i-1] ] )
        if props[i-1] == 'sources':
            for subprop in sourcesiblings:
                setattr( ds, subprop, origprops[ subprop ] )
    setattr( ds, props[-1], origprops[ props[-1] ] )


    # MORE


def test_datastore_delete_everything(decam_datastore):
    im = decam_datastore.image
    im_paths = im.get_fullpath(as_list=True)
    sources = decam_datastore.sources
    sources_path = sources.get_fullpath()
    psf = decam_datastore.psf
    psf_paths = psf.get_fullpath(as_list=True)
    sub = decam_datastore.sub_image
    sub_paths = sub.get_fullpath(as_list=True)
    det = decam_datastore.detections
    det_path = det.get_fullpath()
    cutouts = decam_datastore.cutouts
    cutouts_file_path = cutouts.get_fullpath()
    measurements_list = decam_datastore.measurements
    scores_list = decam_datastore.scores

    # make sure we can delete everything
    decam_datastore.delete_everything()

    # make sure everything is deleted
    for path in im_paths:
        assert not os.path.exists(path)

    assert not os.path.exists(sources_path)

    for path in psf_paths:
        assert not os.path.exists(path)

    for path in sub_paths:
        assert not os.path.exists(path)

    assert not os.path.exists(det_path)

    assert not os.path.exists(cutouts_file_path)

    # check these don't exist on the DB:
    with SmartSession() as session:
        assert session.scalars(sa.select(Image).where(Image._id == im.id)).first() is None
        assert session.scalars(sa.select(SourceList).where(SourceList._id == sources.id)).first() is None
        assert session.scalars(sa.select(PSF).where(PSF._id == psf.id)).first() is None
        assert session.scalars(sa.select(Image).where(Image._id == sub.id)).first() is None
        assert session.scalars(sa.select(SourceList).where(SourceList._id == det.id)).first() is None
        assert session.scalars(sa.select(Cutouts).where(Cutouts._id == cutouts.id)).first() is None
        if len(measurements_list) > 0:
            assert session.scalars(
                sa.select(Measurements).where(Measurements._id == measurements_list[0].id)
            ).first() is None
        if len(scores_list) > 0:
            assert session.scalars( sa.select(DeepScore).where(DeepScore._id == scores_list[0].id) ).first() is None
