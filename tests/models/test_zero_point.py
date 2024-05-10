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
    # save the WCS to file and DB
    with SmartSession() as session:
        try:
            provenance_base = session.merge(provenance_base)
            provenance_extra = session.merge(provenance_extra)
            image = ztf_datastore_uncommitted.image
            image.sources = ztf_datastore_uncommitted.sources
            image.sources.provenance = provenance_extra
            image.sources.save()
            image.psf.provenance = provenance_extra
            image.psf.save()
            image.provenance = provenance_base
            image.save()
            image = image.merge_all(session)

            zp = ZeroPoint(zp=20.1, dzp=0.1)
            zp.sources = image.sources
            zp.provenance = Provenance(
                process='test_zero_point',
                code_version=provenance_base.code_version,
                parameters={'test_parameter': 'test_value'},
                upstreams=[provenance_extra],
                is_testing=True,
            )

            session.add(zp)
            session.commit()

            # add a second WCS object and make sure we cannot accidentally commit it, too
            zp2 = ZeroPoint(zp=20.1, dzp=0.1)
            zp2.sources = image.sources
            zp2.provenance = zp.provenance

            with pytest.raises(
                    IntegrityError,
                    match='duplicate key value violates unique constraint "_zp_sources_provenance_uc"'
            ):
                session.add(zp2)
                session.commit()
            session.rollback()

            # if we change any of the provenance parameters we should be able to save it
            zp2.provenance = Provenance(
                process='test_zero_point',
                code_version=provenance_base.code_version,
                parameters={'test_parameter': 'new_test_value'},  # notice we've put another value here
                upstreams=[provenance_extra],
                is_testing=True,
            )
            session.add(zp2)
            session.commit()

        finally:

            if 'zp' in locals():
                if sa.inspect(zp).persistent:
                    session.delete(zp)
                    image.zp = None
                    image.sources.zp = None
            if 'zp2' in locals():
                if sa.inspect(zp2).persistent:
                    session.delete(zp2)
                    image.zp = None
                    image.sources.zp = None

            if 'image' in locals():
                image.delete_from_disk_and_database(session=session, commit=False, remove_downstreams=True)

            session.commit()

    
                    
    
    
