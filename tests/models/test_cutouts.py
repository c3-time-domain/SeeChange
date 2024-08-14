import os
import uuid
import h5py

import numpy as np
import pytest

import sqlalchemy as sa

from models.base import SmartSession
from models.cutouts import Cutouts

from pipeline.data_store import DataStore

def test_make_save_load_cutouts( decam_datastore, cutter ):
    try:
        ds = DataStore( decam_datastore )
        # ...this is a little weird; we already made cutouts
        # in the fixture, and now we're going to rerun them.
        # Perhaps we could just look at the ones that
        # were in the fixture?  (Of course, this does let us
        # test has_recalculated.)
        ds.cutouts.delete_from_disk_and_database()
        ds.cutouts = None
        ds = cutter.run( ds )

        assert cutter.has_recalculated
        assert isinstance(ds.cutouts, Cutouts)
        # all_measurements is a test property that isn't really properly
        #   supported in DataStore, so it wasn't set to None when
        #   cutouts was set to None above
        assert len(ds.cutouts.co_dict) == len(ds.all_measurements)

        subdict_key = "source_index_0"
        co_subdict = ds.cutouts.co_dict[subdict_key]

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
        ds.cutouts.set_badness( 'cosmic ray' )

        # save the Cutouts
        ds.cutouts.save( image=ds.sub_image, sources=ds.detections )

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
        c2.sources_id = ds.cutouts.sources_id
        c2.load_all_co_data()

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
        ds.cutouts.save( image=ds.sub_image, sources=ds.detections )

        with h5py.File(ds.cutouts.get_fullpath(), 'r') as file:
            assert np.array_equal(ds.cutouts.co_dict[subdict_key]['sub_data'],
                                file[subdict_key]['sub_data'])
            assert file[subdict_key]['sub_data'][0, 0] == 100  # change has propagated

        # check that we can add the cutouts to the database
        # (First make sure it's not there already, because we deleted it above,
        # and haven't inserted it since re-making it.)
        with SmartSession() as session:
            loaded_cutouts = session.scalars( sa.select(Cutouts)
                                              .where( Cutouts.provenance_id == ds.prov_tree['cutting'].id )
                                             ).all()
            assert len(loaded_cutouts) == 0

        ds.cutouts.insert()

        with SmartSession() as session:
            loaded_cutouts = session.scalars( sa.select(Cutouts)
                                              .where( Cutouts.provenance_id == ds.prov_tree['cutting'].id )
                                             ).all()
            assert len(loaded_cutouts) == 1
            loaded_cutouts = loaded_cutouts[0]

            assert loaded_cutouts.badness == 'cosmic ray'

            # make sure data is correct
            loaded_cutouts.load_all_co_data()
            co_subdict = loaded_cutouts.co_dict[subdict_key]
            for im in ['sub', 'ref', 'new']:
                for att in ['data', 'weight', 'flags']:
                    assert np.array_equal(co_subdict.get(f'{im}_{att}'),
                                            co_subdict2.get(f'{im}_{att}'))

    finally:
        # (This probably shouldn't be necessary, as the fixture cleanup
        # will clean up everything.)
        # if 'ds' in locals() and ds.cutouts is not None:
        #     ds.cutouts.delete_from_disk_and_database()
        pass
