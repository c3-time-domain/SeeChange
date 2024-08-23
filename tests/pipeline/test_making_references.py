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


def test_finding_references(ptf_ref):
    with pytest.raises(ValueError, match='Must provide at least ra/dec or target/section_id'):
        ref, img = Reference.get_references(ra=188)
    with pytest.raises(ValueError, match='Must provide at least ra/dec or target/section_id'):
        ref, img = Reference.get_references(dec=4.5)
    with pytest.raises(ValueError, match='Must provide at least ra/dec or target/section_id'):
        ref, img = Reference.get_references(target='foo')
    with pytest.raises(ValueError, match='Must provide at least ra/dec or target/section_id'):
        ref, img = Reference.get_references(section_id='bar')
    with pytest.raises(ValueError, match='Must provide at least ra/dec or target/section_id'):
        ref,img  = Reference.get_references(ra=188, section_id='bar')
    with pytest.raises(ValueError, match='Must provide at least ra/dec or target/section_id'):
        ref, img = Reference.get_references(dec=4.5, target='foo')
    with pytest.raises(ValueError, match='Must provide at least ra/dec or target/section_id'):
        ref, img = Reference.get_references()

    ref, img = Reference.get_references(ra=188, dec=4.5)
    assert len(ref) == 1
    assert ref[0].id == ptf_ref.id

    ref, img = Reference.get_references(ra=188, dec=4.5, provenance_ids=ptf_ref.provenance_id)
    assert len(ref) == 1
    assert ref[0].id == ptf_ref.id

    ref, img = Reference.get_references(ra=0, dec=0)
    assert len(ref) == 0

    ref, img = Reference.get_references(target='foo', section_id='bar')
    assert len(ref) == 0

    ref, img = Reference.get_references(ra=180, dec=4.5, provenance_ids=['foo', 'bar'])
    assert len(ref) == 0

    # TODO : test target/section filter on ra/dec search, test
    # instrument and filter filters, test provenance_ids, test skip_bad


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
    maker = RefMaker(maker={'name': name, 'instruments': ['PTF']})
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
                       ) as e:
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

