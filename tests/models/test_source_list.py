import pytest
import numpy as np

import sqlalchemy as sa

from models.base import SmartSession
from models.source_list import SourceList


def test_source_list_bitflag(sources, demo_image, provenance_base, provenance_extra):
    filenames = []
    with SmartSession() as session:
        sources.provenance = provenance_extra
        demo_image.provenance = provenance_base
        demo_image.data = np.float32(demo_image.raw_data)
        demo_image.save(no_archive=True)
        filenames.append(demo_image.get_fullpath(as_list=True)[0])
        sources.save(no_archive=True)
        filenames.append(sources.get_fullpath(as_list=True)[0])
        session.add(sources)
        session.commit()

        assert demo_image.id is not None  # was added along with sources
        assert sources.id is not None
        assert sources.image_id == demo_image.id

        # all these data products should have bitflag zero
        assert sources.bitflag == 0
        assert sources.badness == ''

        # try to find this using the bitflag hybrid property
        sources2 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sources.id in [s.id for s in sources2]
        sources2x = session.scalars(sa.select(SourceList).where(SourceList.bitflag > 0)).all()
        assert sources.id not in [s.id for s in sources2x]

        # now add a badness to the image and exposure
        demo_image.badness = 'Saturation'
        demo_image.exposure.badness = 'Banding'

        session.add(demo_image)
        session.commit()

        assert demo_image.bitflag == 2 ** 1 + 2 ** 3
        assert demo_image.badness == 'Banding, Saturation'

        assert sources.bitflag == 2 ** 1 + 2 ** 3
        assert sources.badness == 'Banding, Saturation'

        # try to find this using the bitflag hybrid property
        sources3 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 1 + 2 ** 3)).all()
        assert sources.id in [s.id for s in sources3]
        sources3x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sources.id not in [s.id for s in sources3x]

        # now add some badness to the source list itself

        # cannot add an image badness to a source list
        with pytest.raises(ValueError, match='Keyword "Banding" not recognized in dictionary'):
            sources.badness = 'Banding'

        # add badness that works with source lists (e.g., cross-match failures)
        sources.badness = 'few sources'
        session.add(sources)
        session.commit()

        assert sources.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 43
        assert sources.badness == 'Banding, Saturation, Few Sources'

        # try to find this using the bitflag hybrid property
        sources4 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 43)).all()
        assert sources.id in [s.id for s in sources4]
        sources4x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sources.id not in [s.id for s in sources4x]

        # removing the badness from the exposure is updated directly to the source list
        demo_image.exposure.bitflag = 0
        session.add(demo_image)
        session.commit()

        assert demo_image.badness == 'Saturation'
        assert sources.badness == 'Saturation, Few Sources'

        # check the database queries still work
        sources5 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 3 + 2 ** 43)).all()
        assert sources.id in [s.id for s in sources5]
        sources5x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sources.id not in [s.id for s in sources5x]
