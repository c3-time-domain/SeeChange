import time
import uuid

import pytest
import numpy as np
import h5py
import sep

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import Provenance
from models.background import Background
from models.source_list import SourceList


def test_save_load_backgrounds(decam_raw_image, decam_raw_image_provenance, code_version):
    image = decam_raw_image
    sources = None
    prov = None
    bg_mean = 3.14
    bg_var = 6.28

    try:  # cleanup at the end
        image.insert()

        prov = Provenance(
            code_version_id=code_version.id,
            process='extraction',
            parameters={'method': 'sep', 'format': 'scalar'},
            upstreams=[ decam_raw_image_provenance ],
            is_testing=True,
        )
        prov.insert()

        # Spoof sources with no actual file so we can point the
        #  background to the image.
        sources = SourceList(
            image_id=image.id,
            md5sum=uuid.uuid4(),    # Spoofed, we're not really saving a file
            format='sepnpy',
            num_sources=42,
            provenance_id=prov.id,
        )
        sources.filepath = sources.invent_filepath( image=image )

        # Create a background object with a scalar model:
        b1 = Background(
            format='scalar',
            method='sep',
            sources_id=sources.id,
            value=bg_mean,
            noise=np.sqrt(bg_var),
            image_shape=image.data.shape
        )
        b1.save( image=image, sources=sources )

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

        rng = np.random.default_rng()
        # make a new background with some data:
        b2 = Background(
            format='map',
            method='sep',
            sources_id=sources.id,
            value=bg_mean,
            noise=np.sqrt(bg_var),
            counts=rng.normal(bg_mean, 1, size=(10, 10)),
            variance=rng.normal(bg_var, 1, size=(10, 10)),
            image_shape=image.data.shape
        )

        with pytest.raises(RuntimeError, match='Counts shape .* does not match image shape .*'):
            b2.save( image=image, sources=sources )

        # use actual background measurements so we can get a realistic estimate of the compression
        back = sep.Background(image.data)
        b2.counts = back.back()
        b2.variance = back.rms() ** 2

        t0 = time.perf_counter()
        b2.save( image=image, sources=sources )
        # print(f'Background save time: {time.perf_counter() - t0:.3f} s')
        # print(f'Background file size: {os.path.getsize(b2.get_fullpath()) / 1024 ** 2:.3f} MB')

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

        # Check that we can get the right image_shape from a SourceList and Image saved in the
        #   database
        sources.insert()
        b3 = Background(
            format='scalar',
            method='sep',
            sources_id=sources.id,
            value=bg_mean,
            noise=np.sqrt(bg_var)
        )
        assert b3._image_shape == image.data.shape

    finally:
        if ( sources is not None ) or ( prov is not None ):
            with SmartSession() as session:
                session.execute( sa.text( "DELETE FROM source_lists WHERE _id=:id" ), { 'id': sources.id } )
                session.execute( sa.text( "DELETE FROM provenances WHERE _id=:id" ), { 'id': prov.id } )
                session.commit()

        if 'b1' in locals():
            b1.delete_from_disk_and_database()
        if 'b2' in locals():
            b2.delete_from_disk_and_database()
