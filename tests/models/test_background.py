import pytest
import numpy as np
import h5py

from models.provenance import Provenance
from models.background import Background


def test_save_load_backgrounds(decam_raw_image, code_version):
    image = decam_raw_image
    bg_mean = 3.14
    bg_var = 6.28

    try:  # cleanup at the end
        # Create a background object with a scalar model:
        b1 = Background(
            format='scalar',
            method='sep',
            image=image,
            value=bg_mean,
            noise=np.sqrt(bg_var)
        )

        prov = Provenance(
            code_version=code_version,
            process='extraction',
            parameters={'method': 'sep', 'format': 'scalar'},
            upstreams=[image.provenance],
            is_testing=True,
        )

        b1.provenance = prov

        b1.save()

        # check the filename contains the provenance hash
        assert prov.id[:6] in b1.get_fullpath()

        # check that the file contains what we expect:
        with h5py.File(b1.get_fullpath(), 'r') as f:
            # check there's a "background" group:
            assert 'background' in f
            bg = f['background']

            # check the attributes:
            assert bg.attrs['format'] == 'scalar'
            assert bg.attrs['method'] == 'sep'
            assert bg.attrs['value'] == bg_mean
            assert bg.attrs['noise'] == np.sqrt(bg_var)

        # make a new background with some data:
        b2 = Background(
            format='map',
            method='sep',
            image=image,
            value=bg_mean,
            noise=np.sqrt(bg_var),
            counts=np.random.normal(bg_mean, 1, size=(10, 10)),
            variance=np.random.normal(bg_var, 1, size=(10, 10)),
        )

        prov = Provenance(
            code_version=code_version,
            process='extraction',
            parameters={'method': 'sep', 'format': 'map'},
            upstreams=[image.provenance],
            is_testing=True,
        )

        b2.provenance = prov

        with pytest.raises(RuntimeError, match='Counts shape .* does not match image shape .*'):
            b2.save()

        b2.counts = np.random.normal(bg_mean, 1, size=image.data.shape)
        b2.variance = np.random.normal(bg_var, 1, size=image.data.shape)
        b2.save()

        # check the filename contains the provenance hash
        assert prov.id[:6] in b2.get_fullpath()

        # check that the file contains what we expect:
        with h5py.File(b2.get_fullpath(), 'r') as f:
            # check there's a "background" group:
            assert 'background' in f
            bg = f['background']

            # check the attributes:
            assert bg.attrs['format'] == 'map'
            assert bg.attrs['method'] == 'sep'
            assert bg.attrs['value'] == bg_mean
            assert bg.attrs['noise'] == np.sqrt(bg_var)

            # check the data:
            assert np.allclose(bg['counts'], b2.counts)
            assert np.allclose(bg['variance'], b2.variance)

    finally:
        if 'b1' in locals():
            b1.delete_from_disk_and_database()
        if 'b2' in locals():
            b2.delete_from_disk_and_database()
