import time

import pytest
import uuid

import numpy as np

import sqlalchemy as sa

from pipeline.ref_maker import RefMaker

from models.base import SmartSession
from models.provenance import Provenance
from models.reference import Reference
from models.refset import RefSet


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
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(ra=188)
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(dec=4.5)
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(target='foo')
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(section_id='bar')
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(ra=188, section_id='bar')
    with pytest.raises(ValueError, match='Must provide both'):
        ref = Reference.get_references(dec=4.5, target='foo')
    with pytest.raises(ValueError, match='Must provide either ra and dec, or target and section_id'):
        ref = Reference.get_references()
    with pytest.raises(ValueError, match='Cannot provide target/section_id and also ra/dec! '):
        ref = Reference.get_references(ra=188, dec=4.5, target='foo', section_id='bar')

    ref = Reference.get_references(ra=188, dec=4.5)
    assert len(ref) == 1
    assert ref[0].id == ptf_ref.id

    ref = Reference.get_references(ra=188, dec=4.5, provenance_ids=ptf_ref.provenance_id)
    assert len(ref) == 1
    assert ref[0].id == ptf_ref.id

    ref = Reference.get_references(ra=0, dec=0)
    assert len(ref) == 0

    ref = Reference.get_references(target='foo', section_id='bar')
    assert len(ref) == 0

    ref = Reference.get_references(ra=180, dec=4.5, provenance_ids=['foo', 'bar'])
    assert len(ref) == 0


def test_making_refsets():
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
    assert maker.ref_upstream_hash is None

    new_ref = maker.run(ra=0, dec=0, filter='R')
    assert new_ref is None  # cannot find a specific reference here
    refset = maker.refset

    assert refset is not None  # can produce a reference set without finding a reference
    assert all(isinstance(p, Provenance) for p in maker.im_provs)
    assert all(isinstance(p, Provenance) for p in maker.ex_provs)
    assert isinstance(maker.coadd_im_prov, Provenance)
    assert isinstance(maker.coadd_ex_prov, Provenance)

    up_hash1 = refset.upstream_hash
    assert maker.ref_upstream_hash == up_hash1
    assert isinstance(up_hash1, str)
    assert len(up_hash1) == 20
    assert len(refset.provenances) == 1
    assert refset.provenances[0].parameters['min_number'] == min_number
    assert refset.provenances[0].parameters['max_number'] == max_number
    assert 'name' not in refset.provenances[0].parameters  # not a critical parameter!
    assert 'description' not in refset.provenances[0].parameters  # not a critical parameter!

    # now make a change to the maker's parameters (not the data production parameters)
    maker.pars.min_number = min_number + 5
    maker.pars.allow_append = False  # this should prevent us from appending to the existing ref-set

    with pytest.raises(
            RuntimeError, match='Found a RefSet with the name .*, but it has a different provenance!'
    ):
        new_ref = maker.run(ra=0, dec=0, filter='R')

    maker.pars.allow_append = True  # now it should be ok
    new_ref = maker.run(ra=0, dec=0, filter='R')
    assert new_ref is None  # still can't find images there

    refset = maker.refset
    up_hash2 = refset.upstream_hash
    assert up_hash1 == up_hash2  # the underlying data MUST be the same
    assert len(refset.provenances) == 2
    assert refset.provenances[0].parameters['min_number'] == min_number
    assert refset.provenances[1].parameters['min_number'] == min_number + 5
    assert refset.provenances[0].parameters['max_number'] == max_number
    assert refset.provenances[1].parameters['max_number'] == max_number

    # now try to make a new ref-set with a different name
    name2 = uuid.uuid4().hex
    maker.pars.name = name2
    new_ref = maker.run(ra=0, dec=0, filter='R')
    assert new_ref is None  # still can't find images there

    refset2 = maker.refset
    assert len(refset2.provenances) == 1
    assert refset2.provenances[0].id == refset.provenances[1].id  # these ref-sets share the same provenance!

    # now try to append with different data parameters:
    maker.pipeline.extractor.pars['threshold'] = 3.14

    with pytest.raises(
            RuntimeError, match='Found a RefSet with the name .*, but it has a different upstream_hash!'
    ):
        new_ref = maker.run(ra=0, dec=0, filter='R')


def test_making_references(ptf_reference_images):
    name = uuid.uuid4().hex
    ref = None
    ref5 = None

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
        add_test_parameters(maker)  # make sure we have a test parameter on everything
        maker.coadd_pipeline.coadder.pars.test_parameter = uuid.uuid4().hex  # do not load an existing image

        t0 = time.perf_counter()
        ref = maker.run(ra=188, dec=4.5, filter='R')
        first_time = time.perf_counter() - t0
        first_refset = maker.refset
        first_image = ref.image
        assert ref is not None

        # check that this ref is saved to the DB
        with SmartSession() as session:
            loaded_ref = session.scalars(sa.select(Reference).where(Reference.id == ref.id)).first()
            assert loaded_ref is not None

        # now try to make a new ref with the same parameters
        t0 = time.perf_counter()
        ref2 = maker.run(ra=188, dec=4.5, filter='R')
        second_time = time.perf_counter() - t0
        second_refset = maker.refset
        second_image = ref2.image
        assert second_time < first_time * 0.1  # should be much faster, we are reloading the reference set
        assert ref2.id == ref.id
        assert second_refset.id == first_refset.id
        assert second_image.id == first_image.id

        # now try to make a new ref set with a new name
        maker.pars.name = uuid.uuid4().hex
        t0 = time.perf_counter()
        ref3 = maker.run(ra=188, dec=4.5, filter='R')
        third_time = time.perf_counter() - t0
        third_refset = maker.refset
        third_image = ref3.image
        assert third_time < first_time * 0.1  # should be faster, we are loading the same reference
        assert third_refset.id != first_refset.id
        assert ref3.id == ref.id
        assert third_image.id == first_image.id

        # append to the same refset but with different reference parameters (image loading parameters)
        maker.pars.max_number += 1
        t0 = time.perf_counter()
        ref4 = maker.run(ra=188, dec=4.5, filter='R')
        fourth_time = time.perf_counter() - t0
        fourth_refset = maker.refset
        fourth_image = ref4.image
        assert fourth_time < first_time * 0.1  # should be faster, we can still re-use the underlying coadd image
        assert fourth_refset.id != first_refset.id
        assert ref4.id != ref.id
        assert fourth_image.id == first_image.id

        # now make the coadd image again with a different parameter for the data production
        maker.coadd_pipeline.coadder.pars.flag_fwhm_factor *= 1.2
        maker.pars.name = uuid.uuid4().hex  # MUST give a new name, otherwise it will not allow the new data parameters
        t0 = time.perf_counter()
        ref5 = maker.run(ra=188, dec=4.5, filter='R')
        fifth_time = time.perf_counter() - t0
        fifth_refset = maker.refset
        fifth_image = ref5.image
        assert np.log10(fifth_time) == pytest.approx(np.log10(first_time), rel=0.2)  # should take about the same time
        assert ref5.id != ref.id
        assert fifth_refset.id != first_refset.id
        assert fifth_image.id != first_image.id

    finally:  # cleanup
        if ref is not None and ref.image is not None:
            ref.image.delete_from_disk_and_database(remove_downstreams=True)

        # we don't have to delete ref2, ref3, ref4, because they depend on the same coadd image, and cascade should
        # destroy them as soon as the coadd is removed

        if ref5 is not None and ref5.image is not None:
            ref5.image.delete_from_disk_and_database(remove_downstreams=True)


def test_datastore_get_reference(ptf_datastore, ptf_ref, ptf_ref_offset):
    with SmartSession() as session:
        refset = session.scalars(sa.select(RefSet).where(RefSet.name == 'test_refset_ptf')).first()
        assert refset is not None
        assert len(refset.provenances) == 1
        assert refset.provenances[0].id == ptf_ref.provenance_id

        # append the newer reference to the refset
        ptf_ref_offset = session.merge(ptf_ref_offset)
        refset.provenances.append(ptf_ref_offset.provenance)
        session.commit()

        ref = ptf_datastore.get_reference(provenances=refset.provenances, session=session)

        assert ref is not None
        assert ref.id == ptf_ref.id

        # now offset the image that needs matching
        ptf_datastore.image.ra_corner_00 -= 0.5
        ptf_datastore.image.ra_corner_01 -= 0.5
        ptf_datastore.image.ra_corner_10 -= 0.5
        ptf_datastore.image.ra_corner_11 -= 0.5
        ptf_datastore.image.minra -= 0.5
        ptf_datastore.image.maxra -= 0.5

        ref = ptf_datastore.get_reference(provenances=refset.provenances, session=session)

        assert ref is not None
        assert ref.id == ptf_ref_offset.id

