import pytest
import uuid

import psycopg2.errors

from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from pipeline.data_store import DataStore


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


def test_zeropoint_committing(provenance_base, provenance_extra):
    ds = None
    try:
        ds = DataStore()
        ds.image = Image( format='fits',
                          provenance_id=provenance_base.id,
                          mjd=60000.,
                          end_mjd=60000.1,
                          exp_time=42.,
                          instrument='DemoInstrument',
                          telescope='DemoTelescope',
                          filter='R',
                          section_id='31',
                          project='test',
                          target='wallmart',
                          filepath='foo',
                          md5sum=uuid.uuid4(),
                          ra=42.,
                          dec=23.,
                          ra_corner_00=41.9,
                          ra_corner_01=41.9,
                          ra_corner_10=42.1,
                          ra_corner_11=42.1,
                          dec_corner_00=22.9,
                          dec_corner_10=22.9,
                          dec_corner_01=23.1,
                          dec_corner_11=23.1,
                          minra=41.9,
                          maxra=42.1,
                          mindec=22.9,
                          maxdec=23.1
                         )
        ds.image.insert()
        ds.sources = SourceList( format='sepnpy', image_id=ds.image.id, num_sources=0,
                                 provenance_id=provenance_base.id, filepath='foosl', md5sum=uuid.uuid4() )
        ds.sources.insert()
        ds.bg = Background( format='scalar', method='zero', value=0., noise=1., provenance_id=provenance_base.id,
                            sources_id=ds.sources.id, filepath='foobg', md5sum=uuid.uuid4(),
                            image_shape=(256,256) )
        ds.bg.insert()
        ds.wcs = WorldCoordinates( sources_id=ds.sources.id, provenance_id=provenance_base.id,
                                   filepath='foowcs', md5sum=uuid.uuid4() )
        ds.wcs.insert()

        zp = ZeroPoint(zp=20.1, dzp=0.1)
        zp.wcs_id = ds.wcs.id
        zp.background_id = ds.bg.id
        zp.provenance_id = provenance_extra.id
        zp.insert()

        # add a second ZeroPoint object and make sure we cannot accidentally commit it, too
        zp2 = ZeroPoint(zp=20.1, dzp=0.1)
        zp2.wcs_id = ds.wcs.id
        zp2.background_id = ds.bg.id
        zp2.provenance_id = provenance_extra.id

        with pytest.raises( psycopg2.errors.UniqueViolation,
                            match='duplicate key value violates unique constraint "_zp_wcs_bg_provenance_uc"' ):
            zp2.insert()

    finally:

        # with SmartSession() as session:
        #     session.execute( sa.delete( ZeroPoint ).where( ZeroPoint._id.in_( [ zp.id, zp2.id ] ) ) )
        #     session.commit()
        if ds is not None:
            ds.image.delete_from_disk_and_database( remove_downstreams=True )
