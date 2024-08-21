import os
import pytest
import uuid

import sqlalchemy as sa

from models.base import SmartSession
from models.instrument import SensorSection
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.provenance import Provenance

from pipeline.data_store import DataStore


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
                                       upstreams=[ refimgprov, refsrcprov,
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

    refimgprov.insert_if_needed()
    refsrcprov.insert_if_needed()
    for prov in provs.values():
        prov.insert_if_needed()


    ds = DataStore()
    assert ds.prov_tree is None

    # Make sure we get the right error if we assign the wrong thing to the prov_tree attribute
    with pytest.raises( TypeError, match='prov_tree must be a dict of Provenance objects' ):
        ds.prov_tree = 5
    with pytest.raises( TypeError, match='prov_tree must be a dict of Provenance objects' ):
        ds.prov_tree = { 'extraction': provs['extraction'],
                         'preprocesing': 'kittens' }

    # On to actually testing set_prov_tree

    ds.set_prov_tree( provs, wipe_tree=True )
    assert all( [ prov.id == provs[process].id for process, prov in ds.prov_tree.items() ] )

    # Verify that downstreams get wiped out if we set an upstream
    for i, process in enumerate( provs.keys() ):
        toset = { list(provs.keys())[j]: provs[process] for j in range(0,i+1) }
        ds.set_prov_tree( toset, wipe_tree=False )
        for dsprocess in list(provs.keys())[i+1:]:
            if dsprocess != 'referencing':
                assert dsprocess not in ds.prov_tree
        # reset
        ds.set_prov_tree( provs, wipe_tree=True )

    # Verify that wipe_tree=True works as expected
    # (We're making an ill-formed provenance tree here
    # just for test purposes.)
    ds.set_prov_tree( { 'subtraction': provs['subtraction'] }, wipe_tree=True )
    assert all( [ p not in ds.prov_tree for p in provs.keys() if p != 'subtraction' ] )

    # reset and test wipe_tree=False
    ds.set_prov_tree( provs, wipe_tree=True )
    ds.set_prov_tree( {'subtraction': provs['subtraction'] }, wipe_tree=False )
    for shouldbegone in [ 'detection', 'cutting', 'measuring' ]:
        assert shouldbegone not in ds.prov_tree
    for shouldbehere in [ 'exposure', 'preprocessing', 'extraction', 'referencing', 'subtraction' ]:
        assert ds.prov_tree[shouldbehere].id == provs[shouldbehere].id

    # Clean up
    with SmartSession() as sess:
        idstodel = [ refimgprov.id, refsrcprov.id ]
        idstodel.extend( list( provs.keys() ) )
        sess.execute( sa.delete( Provenance ).where( Provenance._id.in_( idstodel ) ) )
        sess.commit()


def test_make_provenance():
    procparams = { 'exposure': {},
                   'preprocessing': { 'a': 1 },
                   'extraction': { 'b': 2 },
                   'referencing': { 'z': 2.7182818 },
                   'subtraction': { 'c': 3 },
                   'detection': { 'd': 4 },
                   'cutting': { 'e': 5 },
                   'measuring': { 'f': 6 }
                  }
    ds = DataStore()
    assert ds.prov_tree is None

    def refresh_tree():
        ds.prov_tree = None
        for process, params in procparams.items():
            ds.get_provenance( process, params, replace_tree=True )

    refresh_tree()

    # Make sure they're all there
    for process, params in procparams.items():
        prov = ds.prov_tree[ process ]
        assert prov.process == process
        assert prov.parameters == params

    # Make sure that if we get one, we get the same one back
    for process, params in procparams.items():
        prov = ds.get_provenance( process, params )
        assert prov.process == process
        assert prov.parameters == params

    # Make sure that the upstreams are consistent
    assert ds.prov_tree['measuring'].upstreams == [ ds.prov_tree['cutting'] ]
    assert ds.prov_tree['cutting'].upstreams == [ ds.prov_tree['detection'] ]
    assert ds.prov_tree['detection'].upstreams == [ ds.prov_tree['subtraction'] ]
    assert set( ds.prov_tree['subtraction'].upstreams ) == { ds.prov_tree['preprocessing'],
                                                             ds.prov_tree['extraction'] }
    assert ds.prov_tree['extraction'].upstreams == [ ds.prov_tree['preprocessing'] ]
    assert ds.prov_tree['preprocessing'].upstreams == [ ds.prov_tree['exposure'] ]

    # Make sure that if we have different parameters, it yells at us
    for process in procparams.keys():
        with pytest.raises( ValueError, match="DataStore getting provenance.*don't match" ):
            prov = ds.get_provenance( process, { 'does_not_exist': 'luminiferous_aether' } )

    # Check that pars_not_match_prov_tree works, but doesn't replace the tree
    for process, params in procparams.items():
        prov = ds.get_provenance( process, { 'does_not_exist': 'luminiferous_aether' },
                                  pars_not_match_prov_tree_pars=True )
        assert prov.process == process
        assert prov.parameters == { 'does_not_exist': 'luminiferous_aether' }

    # Check that if we replace a process, all downstream ones get wiped out
    # (with 'referencing' being a special case exception).
    # NOTE: I'm assuming that the keys in DataStore.UPSTREAM_STEPS are
    # sorted.  Really I should build a tree or something basedon the
    # dependencies.  But, whatevs.
    for i, process in enumerate( procparams.keys() ):
        prov = ds.get_provenance( process, { 'replaced': True }, replace_tree=True )
        assert prov.process == process
        assert prov.parameters == { 'replaced': True }
        for upproc in list( procparams.keys() )[:i]:
            assert upproc in ds.prov_tree
            assert ds.prov_tree[upproc].process == upproc
            assert ds.prov_tree[upproc].parameters == procparams[ upproc ]
        for downproc in list( procparams.keys() )[i+1:]:
            if downproc != 'referencing':
                assert downproc not in ds.prov_tree
        refresh_tree()

    # TODO : test get_provenance when it's pulling upstreams from objects in the
    #   datastore rather than from its own prov_tree.


def test_make_sub_prov_upstreams():
    # The previous test was cavalier about subtraction upstreams.  Explicitly
    #   test that the subtraction provenance doesn't get the referencing
    #   provenance as an upstream but the refrencing provenance's upstreams.
    refimgprov = Provenance( process='preprocessing', parameters={ 'ref': True } )
    refimgprov.insert_if_needed()
    refsrcprov = Provenance( process='extraction', parameters={ 'ref': True } )
    refsrcprov.insert_if_needed()

    provs = { 'exposure': Provenance( process='exposure', parameters={} ) }
    provs['preprocessing'] = Provenance( process='preprocessing',
                                         upstreams=[provs['exposure']],
                                         parameters={ 'a': 1 } )
    provs['extraction'] = Provenance( process='extraction',
                                      upstreams=[provs['preprocessing']],
                                      parameters={ 'a': 1 } )
    provs['referencing'] = Provenance( process='referencing',
                                       upstreams=[ refimgprov, refsrcprov ],
                                       parameters={ 'a': 1 } )
    for prov in provs.values():
        prov.insert_if_needed()

    ds = DataStore()
    ds.set_prov_tree( provs )
    subprov = ds.get_provenance( 'subtraction', {} )
    assert set( subprov.upstreams ) == { refimgprov, refsrcprov, provs['preprocessing'], provs['extraction'] }

    # Clean up
    with SmartSession() as sess:
        idstodel = [ refimgprov.id, refsrcprov.id ]
        idstodel.extend( list( provs.keys() ) )
        idstodel.append( subprov.id )
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

    with pytest.raises( Exception ) as ex:
        ds.exposure_id = 'this is not a valid uuid'

    ds.exposure_id = origexp

    origimg = ds._image_id
    assert ds.image_id == origimg

    tmpuuid = uuid.uuid4()
    ds.image_id = tmpuuid
    assert ds._image_id == tmpuuid
    assert ds.image_id == tmpuuid

    with pytest.raises( Exception ) as ex:
        ds.image_id = 'this is not a valud uuid'

    ds.image_id = origimg

    exp = ds.exposure
    assert isinstance( exp, Exposure )
    assert exp.instrument == 'DECam'
    assert exp.format == 'fits'

    assert ds._section is None
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


