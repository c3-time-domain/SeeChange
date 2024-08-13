import pytest

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession
from models.provenance import Provenance
from models.zero_point import ZeroPoint


def test_zeropoint_get_aper_cor():
    zp = ZeroPoint()
    with pytest.raises( ValueError, match="No aperture corrections tabulated." ):
        _ = zp.get_aper_cor( 1.0 )

    zp = ZeroPoint( aper_cor_radii=[ 1.0097234, 2.4968394 ],
                    aper_cors=[-0.25, -0.125] )
    assert zp.get_aper_cor( 1.0 ) == -0.25
    assert zp.get_aper_cor( 2.5 ) == -0.125

    with pytest.raises( ValueError, match="No aperture correction tabulated for.*within 0.01 pixels of" ):
        _ = zp.get_aper_cor( 1.02 )
    with pytest.raises( ValueError, match="No aperture correction tabulated for.*within 0.01 pixels of" ):
        _ = zp.get_aper_cor( 0.99 )


def test_zeropoint_committing(ztf_datastore_uncommitted, provenance_base, provenance_extra):
    try:
        ds = ztf_datastore_uncommitted
        ds.image.provenance_id = provenance_base.id
        ds.image.save()
        ds.image.insert()
        ds.sources.provenance_id = provenance_extra.id
        ds.sources.save()
        ds.sources.insert()

        zp = ZeroPoint(zp=20.1, dzp=0.1)
        zp.sources_id = ds.sources.id
        zp.insert()

        # add a second ZeroPoint object and make sure we cannot accidentally commit it, too
        zp2 = ZeroPoint(zp=20.1, dzp=0.1)
        zp2.sources_id = ds.sources.id

        with pytest.raises( IntegrityError,
                            match='duplicate key value violates unique constraint "ix_zero_points_sources_id"' ):
            zp2.insert()

    finally:

        with SmartSession() as session:
            session.execute( sa.delete( ZeroPoint ).where( ZeroPoint._id.in_( [ zp.id, zp2.id ] ) ) )
            session.commit()

        ds.image.delete_from_disk_and_database( remove_downstreams=True )





