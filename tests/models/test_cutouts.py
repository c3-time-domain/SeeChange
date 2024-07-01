import os
import uuid
import h5py

import numpy as np
import pytest

import sqlalchemy as sa

from models.base import SmartSession
from models.cutouts import Cutouts


def test_make_save_load_cutouts(decam_detection_list, cutter):
    try:
        cutter.pars.test_parameter = uuid.uuid4().hex
        ds = cutter.run(decam_detection_list)

        assert cutter.has_recalculated
        assert isinstance(ds.cutouts, Cutouts)
        assert len(ds.cutouts.co_dict) == ds.cutouts.sources.num_sources

        subdict_key = "source_index_0"
        co_subdict = ds.cutouts.co_dict[subdict_key]

        assert ds.cutouts.sub_image == decam_detection_list.image
        assert ds.cutouts.ref_image == decam_detection_list.image.ref_aligned_image
        assert ds.cutouts.new_image == decam_detection_list.image.new_aligned_image

        assert isinstance(co_subdict["sub_data"], np.ndarray)
        assert isinstance(co_subdict["sub_weight"], np.ndarray)
        assert isinstance(co_subdict["sub_flags"], np.ndarray)
        assert isinstance(co_subdict["ref_data"], np.ndarray)
        assert isinstance(co_subdict["ref_weight"], np.ndarray)
        assert isinstance(co_subdict["ref_flags"], np.ndarray)
        assert isinstance(co_subdict["new_data"], np.ndarray)
        assert isinstance(co_subdict["new_weight"], np.ndarray)
        assert isinstance(co_subdict["new_flags"], np.ndarray)
        assert ds.cutouts.bitflag is not None

        # set the bitflag just to see if it is loaded or not
        ds.cutouts.bitflag = 2 ** 41  # should be Cosmic Ray

        # save the Cutouts
        ds.cutouts.save()

        # open the file manually and compare
        with h5py.File(ds.cutouts.get_fullpath(), 'r') as file:
            for im in ['sub', 'ref', 'new']:
                for att in ['data', 'weight', 'flags']:
                    assert f'{im}_{att}' in file[subdict_key]
                    assert np.array_equal(co_subdict.get(f'{im}_{att}'),
                                          file[subdict_key][f'{im}_{att}'])

        # load a cutouts from file and compare
        c2 = Cutouts()
        c2.filepath = ds.cutouts.filepath
        c2.sources = ds.cutouts.sources  # necessary for co_dict
        c2.load_all_co_data()  # explicitly load co_dict

        co_subdict2 = c2.co_dict[subdict_key]

        for im in ['sub', 'ref', 'new']:
            for att in ['data', 'weight', 'flags']:
                assert np.array_equal(co_subdict.get(f'{im}_{att}'),
                                        co_subdict2.get(f'{im}_{att}'))

        assert c2.bitflag == 0  # should not load all column data from file

        # change the value of one of the arrays
        ds.cutouts.co_dict[subdict_key]['sub_data'][0, 0] = 100
        co_subdict2['sub_data'][0, 0] = 100  # for comparison later

        # make sure we can re-save
        ds.cutouts.save()

        with h5py.File(ds.cutouts.get_fullpath(), 'r') as file:
            assert np.array_equal(ds.cutouts.co_dict[subdict_key]['sub_data'],
                                file[subdict_key]['sub_data'])
            assert file[subdict_key]['sub_data'][0, 0] == 100  # change has propagated

        # check that we can add the cutouts to the database
        with SmartSession() as session:
            ds.cutouts = session.merge(ds.cutouts)
            session.commit()

        ds.cutouts.load_all_co_data()  # need to re-load after merge
        assert ds.cutouts is not None
        assert len(ds.cutouts.co_dict) > 0

        with SmartSession() as session:
            loaded_cutouts = session.scalars(
                sa.select(Cutouts).where(Cutouts.provenance_id == ds.cutouts.provenance.id)
            ).all()
            assert len(loaded_cutouts) == 1
            loaded_cutouts = loaded_cutouts[0]

            # make sure data is correct
            loaded_cutouts.load_all_co_data()
            co_subdict = loaded_cutouts.co_dict[subdict_key]
            for im in ['sub', 'ref', 'new']:
                for att in ['data', 'weight', 'flags']:
                    assert np.array_equal(co_subdict.get(f'{im}_{att}'),
                                            co_subdict2.get(f'{im}_{att}'))

    finally:
        if 'ds' in locals() and ds.cutouts is not None:
            ds.cutouts.delete_from_disk_and_database()
