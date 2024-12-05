import time

import pytest
import uuid

import numpy as np

import sqlalchemy as sa

from pipeline.ref_maker import RefMaker

from models.base import SmartSession
from models.provenance import Provenance
from models.image import Image
from models.reference import Reference
from models.refset import RefSet

from util.util import env_as_bool


def add_test_parameters(maker):
    """Utility function to add "test_parameter" to all the underlying objects. """
    for name in ['preprocessor', 'extractor', 'backgrounder', 'astrometor', 'photometor', 'coadder']:
        for pipe in ['pipeline', 'coadd_pipeline']:
            obj = getattr(getattr(maker, pipe), name, None)
            if obj is not None:
                obj.pars._enforce_no_new_attrs = False
                obj.pars.test_parameter = obj.pars.add_par(
                    'test_parameter', 'test_value', str, 'A parameter showing this is part of a test', critical=True,
                )
                obj.pars._enforce_no_new_attrs = True


def test_finding_references( provenance_base, provenance_extra ):
    refstodel = set()
    imgstodel = set()

    try:
        # Create ourselves some fake images and references to use as test fodder

        reuseimgkw = { 'provenance_id': provenance_base.id,
                       'mjd': 60000.,
                       'end_mjd': 60000.000694,
                       'exp_time': 60.,
                       'fwhm_estimate': 1.1,
                       'zero_point_estimate': 24.,
                       'bkg_mean_estimate': 0.,
                       'bkg_rms_estimate': 1.,
                       'md5sum': uuid.uuid4(),
                       'format': 'fits',
                       'telescope': 'testscope',
                       'instrument': 'DECam',
                       'project': 'Mercator'
                      }

        # Make a couple of refsets
        refset_base = RefSet( name="base", description="provenance_base" )
        refset_base.insert()
        refset_extra = RefSet( name="extra", description="provenance_extra" )
        refset_extra.insert()
        refset_both = RefSet( name="both", description="provenance_both" )
        refset_both.insert()
        with SmartSession() as session:
            for pid, rsid in zip( [ provenance_base.id, provenance_extra.id, provenance_base.id, provenance_extra.id ],
                                  [ refset_base.id,     refset_extra.id,     refset_both.id,     refset_both.id   ] ):
                q = "INSERT INTO refset_provenance_association(provenance_id,refset_id) VALUES (:pid,:rsid)"
                session.execute( sa.text(q), { 'pid': pid, 'rsid': rsid } )
            session.commit()

        # Something somewhere, 0.2° on a side, in r and g, and one with a different provenance
        img1 = Image( ra=20., dec=45.,
                      minra=19.8586, maxra=20.1414, mindec=44.9, maxdec=45.1,
                      ra_corner_00=19.8586, ra_corner_01=19.8586, ra_corner_10=20.1414, ra_corner_11=20.1414,
                      dec_corner_00=44.9, dec_corner_10=44.9, dec_corner_01=45.1, dec_corner_11=45.1,
                      target='target1', section_id='1', filter='r', filepath='testimage1.fits', **reuseimgkw )
        img1.calculate_coordinates()
        img1.insert()
        imgstodel.add( img1.id )
        ref1 = Reference( provenance_id=provenance_base.id, image_id=img1.id, target=img1.target,
                          filter=img1.filter, section_id=img1.section_id, instrument=img1.instrument )
        ref1.insert()
        refstodel.add( ref1.id )

        img2 = Image( ra=20., dec=45.,
                      minra=19.8586, maxra=20.1414, mindec=44.9, maxdec=45.1,
                      ra_corner_00=19.8586, ra_corner_01=19.8586, ra_corner_10=20.1414, ra_corner_11=20.1414,
                      dec_corner_00=44.9, dec_corner_10=44.9, dec_corner_01=45.1, dec_corner_11=45.1,
                      target='target1', section_id='1', filter='g', filepath='testimage2.fits', **reuseimgkw )
        img2.calculate_coordinates()
        img2.insert()
        imgstodel.add( img2.id )
        ref2 = Reference( provenance_id=provenance_base.id, image_id=img2.id, target=img2.target,
                          filter=img2.filter, section_id=img2.section_id, instrument=img2.instrument )
        ref2.insert()
        refstodel.add( ref2.id )

        refp = Reference( provenance_id=provenance_extra.id, image_id=img1.id, target=img1.target,
                          filter=img1.filter, section_id=img1.section_id, instrument=img1.instrument )
        refp.insert()
        refstodel.add( refp.id )

        # Offset by 0.15° in both ra and dec
        img3 = Image( ra=20.2121, dec=45.15,
                      minra=20.0707, maxra=20.3536, mindec=45.05, maxdec=45.25,
                      ra_corner_00=20.0707, ra_corner_01=20.0707, ra_corner_10=20.3536, ra_corner_11=20.3536,
                      dec_corner_00=45.05, dec_corner_10=45.05, dec_corner_01=45.25, dec_corner_11=45.25,
                      target='target2', section_id='1', filter='r',  filepath='testimage3.fits', **reuseimgkw )
        img3.calculate_coordinates()
        img3.insert()
        imgstodel.add( img3.id )
        ref3 = Reference( provenance_id=provenance_base.id, image_id=img3.id, target=img3.target,
                          filter=img3.filter, section_id=img3.section_id, instrument=img3.instrument )
        ref3.insert()
        refstodel.add( ref3.id )

        # Offset, but also rotated by 45°
        img4 = Image( ra=20.2121, dec=45.15,
                      minra=20.0121, maxra=20.4121, mindec=45.0086, maxdec=45.2914,
                      ra_corner_00=20.0121, ra_corner_01=20.2121, ra_corner_11=20.4121, ra_corner_10=20.2121,
                      dec_corner_00=45.15, dec_corner_01=45.2914, dec_corner_11=45.15, dec_corner_10=45.0086,
                      target='target2', section_id='1', filter='r',  filepath='testimage4.fits', **reuseimgkw )
        img4.calculate_coordinates()
        img4.insert()
        imgstodel.add( img4.id )
        ref4 = Reference( provenance_id=provenance_base.id, image_id=img4.id, target=img4.target,
                          filter=img4.filter, section_id=img4.section_id, instrument=img4.instrument )
        ref4.insert()
        refstodel.add( ref4.id )

        # At 0 ra
        img5 = Image( ra=0.02, dec=0.,
                      minra=359.92, maxra=0.12, mindec=-0.1, maxdec=0.1,
                      ra_corner_00=359.92, ra_corner_01=359.92, ra_corner_10=0.12, ra_corner_11=0.12,
                      dec_corner_00=-0.1, dec_corner_10=-0.1, dec_corner_01=0.1, dec_corner_11=0.1,
                      target='target3', section_id='1', filter='r', filepath='testimage5.fits', **reuseimgkw )
        img5.calculate_coordinates()
        img5.insert()
        imgstodel.add( img5.id )
        ref5 = Reference( provenance_id=provenance_base.id, image_id=img5.id, target=img5.target,
                          filter=img5.filter, section_id=img5.section_id, instrument=img5.instrument )
        ref5.insert()
        refstodel.add( ref5.id )

        # Test bad parameters
        with pytest.raises( ValueError, match="Must give one of target.*or image" ):
            _, _ = Reference.get_references()
        for kws in [ { 'ra': 20., 'minra': 19. },
                     { 'ra': 20., 'target': 'foo' },
                     { 'minra': 19., 'target': 'foo' },
                     { 'image': img5, 'ra': 20. },
                     { 'image': img5, 'target': 'foo' }
                    ]:
            with pytest.raises( ValueError, match="Specify only one of" ):
                _, _ = Reference.get_references( **kws )
        for kws in [ { 'target': 'foo' }, { 'section_id': '1' } ]:
            with pytest.raises( ValueError, match="Must give both target and section_id" ):
                _, _ = Reference.get_references( **kws )
        for kws in [ { 'ra': 20.}, { 'dec': 45. } ]:
            with pytest.raises( ValueError, match="Must give both ra and dec" ):
                _, _ = Reference.get_references( **kws )
        with pytest.raises( ValueError, match="Specify either image or minra/maxra/mindec/maxdec" ):
            _, _ = Reference.get_references( image=img5, minra=19. )
        # TODO : write clever for loops to test all possibly combinations of minra/maxra/mindec/maxdec
        #   that are missing one or more.  For now, just test a few
        for kws in [ { 'minra': 19.8 },
                     { 'maxra': 20.2, 'mindec': 44.9 },
                     { 'minra': 19.8, 'mindec': 44.9, 'maxdec': 45.1 } ]:
            with pytest.raises( ValueError, match="Must give all of minra, maxra, mindec, maxdec" ):
                _, _ = Reference.get_references( **kws )
        with pytest.raises( ValueError, match="Can't give overlapfrac with target/section_id" ):
            _, _ = Reference.get_references( target='foo', section_id='1', overlapfrac=0.5 )
        with pytest.raises( ValueError, match="Can't give overlapfrac with ra/dec" ):
            _, _ = Reference.get_references( ra=20., dec=45., overlapfrac=0.5 )

        # Get point at center of img1, all filters, all provenances
        refs, imgs = Reference.get_references( ra=20., dec=45. )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 3
        assert set( r.id for r in refs ) == { ref1.id, ref2.id, refp.id }
        assert set( i.id for i in imgs ) == { img1.id, img2.id }

        # Get point at center of img1, all filters, only one provenance
        for provarg in [ provenance_base.id, provenance_base, [ provenance_base.id ], [ provenance_base ] ]:
            refs, imgs = Reference.get_references( ra=20., dec=45., provenance_ids=provarg )
            assert len(imgs) == len(refs)
            assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
            assert len(refs) == 2
            assert set( r.id for r in refs ) == { ref1.id, ref2.id }
            assert set( i.id for i in imgs ) == { img1.id, img2.id }

        # Get point at center of img1, all provenances, only one filter
        refs, imgs = Reference.get_references( ra=20., dec=45., filter='r' )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 2
        assert set( r.id for r in refs ) == { ref1.id, refp.id }
        assert set( i.id for i in imgs ) == { img1.id }

        # Get point at center of img1, one provenance, one filter
        refs, imgs = Reference.get_references( ra=20., dec=45., filter='r', provenance_ids=provenance_base.id )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 1
        assert refs[0].id == ref1.id
        assert imgs[0].id == img1.id

        refs, imgs = Reference.get_references( ra=20., dec=45., filter='g', provenance_ids=provenance_extra.id )
        assert len(refs) == 0
        assert len(imgs) == 0

        # Get point at center of img1, refset base
        refs, imgs = Reference.get_references( ra=20., dec=45., refset='base' )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 2
        assert set( r.id for r in refs ) == { ref1.id, ref2.id }
        assert set( i.id for i in imgs ) == { img1.id, img2.id }

        # Get point at center of img1, refset extra
        refs, imgs = Reference.get_references( ra=20., dec=45., refset='extra' )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 1
        assert refs[0].id == refp.id
        assert imgs[0].id == img1.id

        # Get point at center of img1, refset both
        refs, imgs = Reference.get_references( ra=20., dec=45., refset='both' )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 3
        assert set( r.id for r in refs ) == { ref1.id, ref2.id, refp.id }
        assert set( i.id for i in imgs ) == { img1.id, img2.id }

        # TODO : test limiting on other things like instrument, skip_bad

        # For the rest of the tests, we're going to do filter r and provenance provenance_base
        kwargs = { 'filter': 'r', 'provenance_ids': provenance_base.id }

        # Get point at upper-left of img1
        refs, imgs = Reference.get_references( ra=20.1273, dec=45.09, **kwargs )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 3
        assert set( r.id for r in refs ) == { ref1.id, ref3.id, ref4.id }
        assert set( i.id for i in imgs ) == { img1.id, img3.id, img4.id }

        # Get point included in img3 but not img4
        refs, imgs = Reference.get_references( ra=20.+0.06*np.sqrt(2.), dec=45.06, **kwargs )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 2
        assert set( r.id for r in refs ) == { ref1.id, ref3.id }
        assert set( i.id for i in imgs ) == { img1.id, img3.id }

        # Get point included in img3 and img4 but not img1 (center of img3)
        refs, imgs = Reference.get_references( ra=img3.ra, dec=img3.dec, **kwargs )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 2
        assert set( r.id for r in refs ) == { ref3.id, ref4.id }
        assert set( i.id for i in imgs ) == { img3.id, img4.id }

        # Get points around RA 0
        ras = [ 0., 0.05, 359.95 ]
        decs = [ 0., -0.05, 0.05 ]
        ramess, decmess = np.meshgrid( ras, decs )
        for messdex in range( len(ramess) ):
            for ra, dec in zip( ramess[messdex], decmess[messdex] ):
                refs, imgs = Reference.get_references( ra=ra, dec=dec, **kwargs )
                assert len(imgs) == len(refs)
                assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
                assert len( refs ) == 1
                assert refs[0].id == ref5.id
                assert imgs[0].id == img5.id


        # Overlapping -- overlaps img1 at all
        refs, imgs = Reference.get_references( image=img1, **kwargs )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 3
        assert set( r.id for r in refs ) == { ref1.id, ref3.id, ref4.id }
        assert set( i.id for i in imgs ) == { img1.id, img3.id, img4.id }

        refs, imgs == Reference.get_references( minra=img1.minra, maxra=img1.maxra,
                                                mindec=img1.mindec, maxdec=img1.maxdec,
                                                **kwargs )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 3
        assert set( r.id for r in refs ) == { ref1.id, ref3.id, ref4.id }
        assert set( i.id for i in imgs ) == { img1.id, img3.id, img4.id }

        # Overlapping -- overlaps img1 by at least x%
        # img2 and imgp overlap img1 by 1.0 , but are (respectively) filter g and provenance extra
        # img3 overlaps img1 by 0.063
        # img4 overlaps img1 by 0.021
        # img5 overlaps img1 not at all

        refs, imgs = Reference.get_references( image=img1, overlapfrac=0.5, **kwargs )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 1
        assert refs[0].id == ref1.id
        assert imgs[0].id == img1.id

        refs, imgs = Reference.get_references( image=img1, overlapfrac=0.05, **kwargs )
        assert len(imgs) == len(refs)
        assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
        assert len(refs) == 2
        assert set( r.id for r in refs ) == { ref1.id, ref3.id }
        assert set( i.id for i in imgs ) == { img1.id, img3.id }

        # Overlapping -- overlapping around RA 0
        for ctrra, ctrdec, ovfrac in zip( [ 0.,  0.,    0.08, 0.08, -0.08, -0.08 ],
                                          [ 0.,  0.05,  0.,   0.05,  0.,    0.05 ],
                                          [ 0.9, 0.675, 0.7,  0.525, 0.5,   0.375 ] ):
            minra = ctrra - 0.1
            minra = minra if minra > 0 else 360 + minra
            maxra = ctrra + 0.1
            maxra = maxra if maxra > 0 else 360 + maxra
            refs, imgs = Reference.get_references( minra=minra, maxra=maxra,
                                                   mindec=ctrdec-0.1, maxdec=ctrdec+0.1,
                                                   **kwargs )
            assert len(imgs) == len(refs)
            assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
            assert len(refs) == 1
            assert refs[0].id == ref5.id
            assert imgs[0].id == img5.id

            fiducialfrac = 0.51
            refs, imgs = Reference.get_references( minra=minra, maxra=maxra,
                                                   mindec=ctrdec-0.1, maxdec=ctrdec+0.1,
                                                   overlapfrac=fiducialfrac, **kwargs )
            assert len(imgs) == len(refs)
            assert all( r.image_id == i.id for r, i in zip( refs, imgs ) )
            if ovfrac >= fiducialfrac:
                assert len(refs) == 1
                assert refs[0].id == ref5.id
                assert imgs[0].id == img5.id
            else:
                assert len(refs) == 0

    finally:
        # Clean up images, refs, and refsets we made
        with SmartSession() as session:
            session.execute( sa.delete( Reference ).where( Reference._id.in_( refstodel ) ) )
            session.execute( sa.delete( Image ).where( Image._id.in_( imgstodel ) ) )
            session.execute( sa.delete( RefSet ).where( RefSet.name.in_( ( 'base', 'extra', 'both' ) ) ) )
            session.commit()


def test_make_refset():
    provstodel = set()
    rsname = 'test_making_references.py::test_make_refset'

    try:
        maker = RefMaker( maker={ 'name': rsname, 'instruments': ['PTF'] }, coaddition={ 'method': 'zogy' } )
        assert maker.im_provs is None
        assert maker.ex_provs is None
        assert maker.coadd_im_prov is None
        assert maker.coadd_ex_prov is None
        assert maker.ref_prov is None
        assert maker.refset is None

        # Make sure the refset doesn't pre-exist
        assert RefSet.get_by_name( rsname ) is None

        # Make sure we can create a new refset, and that it sets up the provenances
        maker.make_refset()
        assert maker.ref_prov is not None
        provstodel.add( maker.ref_prov )
        assert len( maker.im_provs ) > 0
        assert len( maker.ex_provs ) > 0
        assert maker.coadd_im_prov is not None
        assert maker.coadd_ex_prov is not None
        rs = RefSet.get_by_name( rsname )
        assert rs is not None
        assert len( rs.provenances ) == 1
        assert rs.provenances[0].id == maker.ref_prov.id

        # Make sure that all is well if we try to make the same RefSet all over again
        newmaker = RefMaker( maker={ 'name': rsname, 'instruments': ['PTF'] }, coaddition={ 'method': 'zogy' } )
        assert newmaker.refset is None
        newmaker.make_refset()
        assert newmaker.refset.id == maker.refset.id
        assert newmaker.ref_prov.id == maker.ref_prov.id
        rs = RefSet.get_by_name( rsname )
        assert len( rs.provenances ) == 1

        # Make sure that all is well if we try to make the same RefSet all over again even if allow_append is false
        donothingmaker = RefMaker( maker={ 'name': rsname, 'instruments': ['PTF'], 'allow_append': False },
                                   coaddition={ 'method': 'zogy' } )
        assert donothingmaker.refset is None
        donothingmaker.make_refset()
        assert donothingmaker.refset.id == maker.refset.id
        assert donothingmaker.ref_prov.id == maker.ref_prov.id
        rs = RefSet.get_by_name( rsname )
        assert len( rs.provenances ) == 1

        # Make sure we can't append a new provenance to an existing RefSet if allow_append is False
        failmaker = RefMaker( maker={ 'name': rsname, 'max_number': 5, 'instruments': ['PTF'], 'allow_append': False },
                              coaddition={ 'method': 'zogy' } )
        assert failmaker.refset is None
        with pytest.raises( RuntimeError, match="RefSet .* exists, allow_append is False, and provenance .* isn't in" ):
            failmaker.make_refset()

        # Make sure that we can append a new provenance to the same RefSet as long
        #   as the upstream thingies are consistent.
        newmaker2 = RefMaker( maker={ 'name': rsname, 'max_number': 5, 'instruments': ['PTF'] },
                              coaddition={ 'method': 'zogy' } )
        newmaker2.make_refset()
        assert newmaker2.refset.id == maker.refset.id
        assert newmaker2.ref_prov.id != maker.ref_prov.id
        provstodel.add( newmaker2.ref_prov )
        assert len( newmaker2.refset.provenances ) == 2
        rs = RefSet.get_by_name( rsname )
        assert len( rs.provenances ) == 2

        # Make sure we can't append a new provenance to the same RefSet
        #   if the upstream thingies are not consistent
        newmaker3 = RefMaker( maker={ 'name': rsname, 'instruments': ['PTF'] },
                              coaddition= { 'coaddition': { 'method': 'naive' } } )
        with pytest.raises( RuntimeError, match="Can't append, reference provenance upstreams are not consistent" ):
            newmaker3.make_refset()
        provstodel.add( newmaker3.ref_prov )

        newmaker4 = RefMaker( maker={ 'name': rsname, 'instruments': ['PTF'] }, coaddition={ 'method': 'zogy' } )
        newmaker4.pipeline.extractor.pars.threshold = maker.pipeline.extractor.pars.threshold + 1.
        with pytest.raises( RuntimeError, match="Can't append, reference provenance upstreams are not consistent" ):
            newmaker4.make_refset()
        provstodel.add( newmaker4.ref_prov )

        # TODO : figure out how to test that the race conditions we work
        #  around in test_make_refset aren't causing problems.  (How to
        #  do that... I really hate to put contitional 'wait here' code
        #  in the actual production code for purposes of tests.  Perhaps
        #  test it repeatedly with multiprocessing to make sure that
        #  that works?)

    finally:
        # Clean up the provenances and refset we made
        with SmartSession() as sess:
            sess.execute( sa.delete( Provenance )
                          .where( Provenance._id.in_( [ p.id for p in provstodel ] ) ) )
            sess.execute( sa.delete( RefSet ).where( RefSet.name==rsname ) )
            sess.commit()


def test_making_refsets_in_run():
    # make a new refset with a new name
    name = uuid.uuid4().hex
    maker = RefMaker(maker={'name': name, 'instruments': ['PTF'], 'corner_distance': None, 'overlap_fraction': None})
    min_number = maker.pars.min_number
    max_number = maker.pars.max_number

    # we still haven't run the maker, so everything is empty
    assert maker.im_provs is None
    assert maker.ex_provs is None
    assert maker.coadd_im_prov is None
    assert maker.coadd_ex_prov is None

    # Make sure we can create a fresh refset
    maker.pars.allow_append = False
    new_ref = maker.run(ra=0, dec=0, filter='R')
    assert new_ref is None  # cannot find a specific reference here
    refset = maker.refset

    assert refset is not None  # can produce a reference set without finding a reference
    assert len( maker.im_provs ) > 0
    assert len( maker.ex_provs ) > 0
    assert all( isinstance(p, Provenance) for p in maker.im_provs.values() )
    assert all( isinstance(p, Provenance) for p in maker.ex_provs.values() )
    assert isinstance(maker.coadd_im_prov, Provenance)
    assert isinstance(maker.coadd_ex_prov, Provenance)

    assert refset.provenances[0].parameters['min_number'] == min_number
    assert refset.provenances[0].parameters['max_number'] == max_number
    assert 'name' not in refset.provenances[0].parameters  # not a critical parameter!
    assert 'description' not in refset.provenances[0].parameters  # not a critical parameter!

    # now make a change to the maker's parameters (not the data production parameters)
    maker.pars.min_number = min_number + 5
    maker.pars.allow_append = False  # this should prevent us from appending to the existing ref-set

    with pytest.raises( RuntimeError,
                        match="RefSet .* exists, allow_append is False, and provenance .* isn't in"
                       ):
        new_ref = maker.run(ra=0, dec=0, filter='R')

    maker.pars.allow_append = True  # now it should be ok
    new_ref = maker.run(ra=0, dec=0, filter='R')
    # Make sure it finds the same refset we're expecting
    assert maker.refset.id == refset.id
    assert new_ref is None  # still can't find images there

    assert len( maker.refset.provenances ) == 2
    assert set( i.parameters['min_number'] for i in maker.refset.provenances ) == { min_number, min_number+5 }
    assert set( i.parameters['max_number'] for i in maker.refset.provenances ) == { max_number }

    refset = maker.refset

    # now try to make a new ref-set with a different name
    name2 = uuid.uuid4().hex
    maker.pars.name = name2
    new_ref = maker.run(ra=0, dec=0, filter='R')
    assert new_ref is None  # still can't find images there

    refset2 = maker.refset
    assert len(refset2.provenances) == 1
    # This refset has a provnenace that was also in th eone we made before
    assert refset2.provenances[0].id in [ i.id for i in refset.provenances ]

    # now try to append with different data parameters:
    maker.pipeline.extractor.pars['threshold'] = 3.14

    with pytest.raises( RuntimeError, match="Can't append, reference provenance upstreams are not consistent" ):
        new_ref = maker.run(ra=0, dec=0, filter='R')

    # Clean up
    with SmartSession() as session:
        session.execute( sa.delete( RefSet ).where( RefSet.name.in_( [ name, name2 ] ) ) )
        session.commit()


def test_identify_images_to_coadd( provenance_base ):
    refmaker = RefMaker( maker={ 'end_time': 60000.,
                                 'corner_distance': 0.8,
                                 'coadd_overlap_fraction': 0.1,
                                 'instruments': ["DemoInstrument"],
                                 'max_seeing': 1.1,
                                 'min_lim_mag': 23. } )
    refmaker.setup_provenances()
    improv = refmaker.im_provs["DemoInstrument"]

    # Make a bunch of images.  All are 0.2° by 0.1°.
    #   img_base
    #   img_rot_45
    #   img_wrongfilter
    #   img_wronginstrument
    #   img_wrongprov
    #   img_toolate
    #   img_badseeing
    #   img_tooshallow
    #   img_shift_up
    #   img_shift_upright

    idstodel = set()

    def imagemaker( filepath, ra, dec, angle=0., **overrides ):
        dra = 0.1
        ddec = 0.05
        _id = uuid.uuid4()
        kwargs = { 'ra': ra,
                   'dec': dec,
                   'provenance_id': improv.id,
                   'mjd': 59000.,
                   'end_mjd': 59000.000694,
                   'exp_time': 60.,
                   'fwhm_estimate': 1.,
                   'lim_mag_estimate': 23.5,
                   'zero_point_estimate': 24.,
                   'bkg_mean_estimate': 0.,
                   'bkg_rms_estimate': 1.,
                   'md5sum': _id,
                   'format': 'fits',
                   'telescope': 'DemoTelescope',
                   'instrument': 'DemoInstrument',
                   'section_id': '1',
                   'filter': 'r',
                   'filepath': filepath,
                   'project': 'Mercator',
                   'target': "The Apple On Top Of William Tell's Head"   # Yes, I know, subject/object
                  }
        corners = np.array( [ [ -dra/2.,  -dra/2.,   dra/2.,  dra/2. ],
                              [ -ddec/2.,  ddec/2., -ddec/2., ddec/2. ] ] )
        rotmat = np.array( [ [ np.cos( angle * np.pi/180. ), np.sin( angle * np.pi/180. ) ],
                             [-np.sin( angle * np.pi/180. ), np.cos( angle * np.pi/180. ) ] ] )
        corners = np.matmul( rotmat, corners )
        corners[0, :] += ra
        corners[1, :] += dec
        # Not going to handle RA near 0...
        minra = min( corners[ 0, : ] )
        maxra = max( corners[ 0, : ] )
        mindec = min( corners[ 1, : ] )
        maxdec = max( corners[ 1, : ] )
        kwargs.update( { 'ra_corner_00': corners[0][0],
                         'ra_corner_01': corners[0][1],
                         'ra_corner_10': corners[0][2],
                         'ra_corner_11': corners[0][3],
                         'minra': minra,
                         'maxra': maxra,
                         'dec_corner_00': corners[1][0],
                         'dec_corner_01': corners[1][1],
                         'dec_corner_10': corners[1][2],
                         'dec_corner_11': corners[1][3],
                         'mindec': mindec,
                         'maxdec': maxdec } )
        kwargs.update( overrides )

        img = Image( **kwargs )
        img.insert()
        idstodel.add( img.id )
        return img

    try:
        img_base = imagemaker( 'img_base.fits', 20., 40. )
        img_rot_45 = imagemaker( 'img_rot_45.fits', 20., 40., 45. )
        _ = imagemaker( 'img_wrongfilter.fits', 20., 40., filter='g' )
        _ = imagemaker( 'img_wronginstrument.fits', 20., 40., instrument='DECam' )
        _ = imagemaker( 'img_wrongprov.fits', 20., 40., provenance_id=provenance_base.id )
        _ = imagemaker( 'img_toolate.fits', 20., 40., mjd=60001, end_mjd=60001.000694 )
        _ = imagemaker( 'img_badseeing.fits', 20., 40., fwhm_estimate=2. )
        _ = imagemaker( 'img_tooshallow.fits', 20., 40., lim_mag_estimate=22. )
        img_shift_up = imagemaker( 'img_shift_up.fits', 20., 40.03 )
        img_shift_upright = imagemaker( 'img_shift_upright.fits', 20 + 0.03 / np.cos( 20. * np.pi/180. ), 40.03 )

        imgs, poses, nums = refmaker.identify_reference_images_to_coadd( image=img_base, filter=['r'] )
        assert poses.shape == (9, 2)
        assert len( nums ) == 9
        # numbers below based on the known order of poses.
        assert nums == [ 2, 1, 2, 2, 1, 1, 3, 4, 3 ]
        assert len(imgs) == 4
        assert set( i.id for i in imgs ) == { img_base.id, img_rot_45.id, img_shift_up.id, img_shift_upright.id }

        # TODO : test the other parameters call (target, section_id ) and test other variations
        #   of refmaker parameters

    finally:
        with SmartSession() as sess:
            sess.execute( sa.delete( Image ).where( Image._id.in_( idstodel ) ) )
            sess.commit()


@pytest.mark.skipif( not env_as_bool('RUN_SLOW_TESTS'), reason="Set RUN_SLOW_TESTS to run this test" )
def test_making_references( ptf_reference_image_datastores ):
    name = uuid.uuid4().hex
    ref = None
    ref5 = None

    refsetstodel = set( name )

    try:
        maker = RefMaker(
            maker={
                'name': name,
                'instruments': ['PTF'],
                'min_number': 4,
                'max_number': 10,
                'end_time': '2010-01-01',
                'corner_distance': None,
                'overlap_fraction': None,
            }
        )
        refsetstodel.add( maker.pars.name )
        add_test_parameters(maker)  # make sure we have a test parameter on everything
        maker.coadd_pipeline.coadder.pars.test_parameter = uuid.uuid4().hex  # do not load an existing image

        t0 = time.perf_counter()
        ref = maker.run(ra=188, dec=4.5, filter='R')
        first_time = time.perf_counter() - t0
        first_refset = maker.refset
        first_image_id = ref.image_id
        assert ref is not None

        # check that this ref is saved to the DB
        with SmartSession() as session:
            loaded_ref = session.scalars(sa.select(Reference).where(Reference._id == ref.id)).first()
            assert loaded_ref is not None

        # now try to make a new ref with the same parameters
        t0 = time.perf_counter()
        ref2 = maker.run(ra=188, dec=4.5, filter='R')
        second_time = time.perf_counter() - t0
        second_refset = maker.refset
        second_image_id = ref2.image_id
        assert second_time < first_time * 0.1  # should be much faster, we are reloading the reference set
        assert ref2.id == ref.id
        assert second_refset.id == first_refset.id
        assert second_image_id == first_image_id

        # now try to make a new ref set with a new name
        maker.pars.name = uuid.uuid4().hex
        refsetstodel.add( maker.pars.name )
        t0 = time.perf_counter()
        ref3 = maker.run(ra=188, dec=4.5, filter='R')
        third_time = time.perf_counter() - t0
        third_refset = maker.refset
        third_image_id = ref3.image_id
        assert third_time < first_time * 0.1  # should be faster, we are loading the same reference
        assert third_refset.id != first_refset.id
        assert ref3.id == ref.id
        assert third_image_id == first_image_id

        # append to the same refset but with different reference parameters (image loading parameters)
        maker.pars.max_number += 1
        t0 = time.perf_counter()
        ref4 = maker.run(ra=188, dec=4.5, filter='R')
        fourth_time = time.perf_counter() - t0
        fourth_refset = maker.refset
        fourth_image_id = ref4.image_id
        assert fourth_time < first_time * 0.1  # should be faster, we can still re-use the underlying coadd image
        assert fourth_refset.id != first_refset.id
        assert ref4.id != ref.id
        assert fourth_image_id == first_image_id

        # now make the coadd image again with a different parameter for the data production
        maker.coadd_pipeline.coadder.pars.flag_fwhm_factor *= 1.2
        maker.pars.name = uuid.uuid4().hex  # MUST give a new name, otherwise it will not allow the new data parameters
        refsetstodel.add( maker.pars.name )
        t0 = time.perf_counter()
        ref5 = maker.run(ra=188, dec=4.5, filter='R')
        fifth_time = time.perf_counter() - t0
        fifth_refset = maker.refset
        fifth_image_id = ref5.image_id
        assert np.log10(fifth_time) == pytest.approx(np.log10(first_time), rel=0.2)  # should take about the same time
        assert ref5.id != ref.id
        assert fifth_refset.id != first_refset.id
        assert fifth_image_id != first_image_id

    finally:  # cleanup
        if ( ref is not None ) and ( ref.image_id is not None ):
            im = Image.get_by_id( ref.image_id )
            im.delete_from_disk_and_database(remove_downstreams=True)

        # we don't have to delete ref2, ref3, ref4, because they depend on the same coadd image, and cascade should
        # destroy them as soon as the coadd is removed

        if ( ref5 is not None ) and ( ref5.image_id is not None ):
            im = Image.get_by_id( ref5.image_id )
            im.delete_from_disk_and_database(remove_downstreams=True)

        # Delete the refsets we made

        with SmartSession() as session:
            session.execute( sa.delete( RefSet ).where( RefSet.name.in_( refsetstodel ) ) )
            session.commit()


def test_datastore_get_reference(ptf_datastore, ptf_ref, ptf_ref_offset):
    with SmartSession() as session:
        refset = session.scalars(sa.select(RefSet).where(RefSet.name == 'test_refset_ptf')).first()

    assert refset is not None
    assert len(refset.provenances) == 1
    assert refset.provenances[0].id == ptf_ref.provenance_id

    refset.append_provenance( Provenance.get( ptf_ref_offset.provenance_id ) )

    ref = ptf_datastore.get_reference(provenances=refset.provenances)

    assert ref is not None
    assert ref.id == ptf_ref.id

    # now offset the image that needs matching
    ptf_datastore.image.ra_corner_00 -= 0.5
    ptf_datastore.image.ra_corner_01 -= 0.5
    ptf_datastore.image.ra_corner_10 -= 0.5
    ptf_datastore.image.ra_corner_11 -= 0.5
    ptf_datastore.image.minra -= 0.5
    ptf_datastore.image.maxra -= 0.5
    ptf_datastore.image.ra -= 0.5

    ref = ptf_datastore.get_reference(provenances=refset.provenances)

    assert ref is not None
    assert ref.id == ptf_ref_offset.id
