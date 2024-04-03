import os
import h5py
import uuid

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
        assert isinstance(ds.cutouts, list)
        assert len(ds.cutouts) > 1
        assert isinstance(ds.cutouts[0], Cutouts)

        c = ds.cutouts[0]
        assert c.sub_image == decam_detection_list.image
        assert c.ref_image == decam_detection_list.image.ref_aligned_image
        assert c.new_image == decam_detection_list.image.new_aligned_image

        assert isinstance(c.sub_data, np.ndarray)
        assert isinstance(c.sub_weight, np.ndarray)
        assert isinstance(c.sub_flags, np.ndarray)
        assert isinstance(c.ref_data, np.ndarray)
        assert isinstance(c.ref_weight, np.ndarray)
        assert isinstance(c.ref_flags, np.ndarray)
        assert isinstance(c.new_data, np.ndarray)
        assert isinstance(c.new_weight, np.ndarray)
        assert isinstance(c.new_flags, np.ndarray)
        assert isinstance(c.source_row, dict)
        assert c.bitflag is not None

        # set the bitflag just to see if it is loaded or not
        c.bitflag = 41  # should be Cosmic Ray

        # save an individual cutout
        Cutouts.save_list([c])

        # open the file manually and compare
        with h5py.File(c.get_fullpath(), 'r') as file:
            assert 'source_0' in file
            for im in ['sub', 'ref', 'new']:
                for att in ['data', 'weight', 'flags']:
                    assert f'{im}_{att}' in file['source_0']
                    assert np.array_equal(getattr(c, f'{im}_{att}'), file['source_0'][f'{im}_{att}'])
            assert dict(file['source_0'].attrs) == c.source_row

        # load it from file and compare
        c2 = Cutouts.from_file(c.get_fullpath(), source_number=0)
        assert c == c2

        assert c2.bitflag == 0  # should not load all column data from file (e.g., bitflag)

        # save a second cutout to the same file
        Cutouts.save_list(ds.cutouts[1:2])
        assert ds.cutouts[1].filepath == c.filepath

        # change the value of one of the arrays
        c.sub_data[0, 0] = 100

        # make sure we can re-save
        Cutouts.save_list([c])

        with h5py.File(c.get_fullpath(), 'r') as file:
            assert np.array_equal(c.sub_data, file['source_0']['sub_data'])
            assert file['source_0']['sub_data'][0, 0] == 100  # change has been propagated

        # save the whole list of cutouts
        Cutouts.save_list(ds.cutouts)

        # load it from file and compare
        loaded_cutouts = Cutouts.load_list(c.get_fullpath())

        for cut1, cut2 in zip(ds.cutouts, loaded_cutouts):
            assert cut1 == cut2

        # make sure that deleting one cutout does not delete the file
        with pytest.raises(NotImplementedError, match='no support for removing one Cutout at a time'):
            # TODO: fix this if we ever bring back this functionality
            ds.cutouts[1].remove_data_from_disk()
            assert os.path.isfile(ds.cutouts[0].get_fullpath())

            # delete one file from the archive, should still keep the file:
            # TODO: this is not yet implemented! see issue #207
            # ds.cutouts[1].delete_from_archive()
            # TODO: check that the file still exists on the archive

        # check that we can add the cutouts to the database
        with SmartSession() as session:
            ds.cutouts = Cutouts.merge_list(ds.cutouts, session=session)

        assert ds.cutouts is not None
        assert len(ds.cutouts) > 0

        with SmartSession() as session:
            loaded_cutouts = session.scalars(
                sa.select(Cutouts).where(Cutouts.provenance_id == ds.cutouts[0].provenance.id)
            ).all()
            for cut1, cut2 in zip(ds.cutouts, loaded_cutouts):
                assert cut1 == cut2

    finally:
        if 'ds' in locals() and ds.cutouts is not None:
            Cutouts.delete_list(ds.cutouts)

