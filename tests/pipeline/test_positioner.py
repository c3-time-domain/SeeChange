import pytest
import uuid

import numpy as np

from models.object import Object, ObjectPosition
from models.image import Image
from models.source_list import SourceList
from models.cutouts import Cutouts
from models.measurements import MeasurementSet, Measurements
from models.base import SmartSession, Psycopg2Connection
from pipeline.positioner import Positioner


@pytest.fixture
def fake_data_for_position_tests( provenance_base ):
    # Create fake difference images, source lists, measurements, etc. necessary for the positioner to do its thing.

    # Might be worth thinking about the right way to scale position
    #   scatter with S/N, but for now, here's a cheesy one that will
    #   give us a pos sigma of 0.1" at s/n 20 of higher, scaling up
    #   linearly to a pos sigma of 1.5" at s/n 3 (and more for worse)
    possigma = lambda sn: min( 0.1, 1.5 - ( 1.4 * (sn-3)/17 ) )

    rng = np.random.default_rng( seed=42 )

    bands = [ 'g', 'r', 'i', 'z' ]
    nimages = 40
    images = []
    sourceses = []
    cutoutses = []
    measurementsets = []
    measurementses = []
    obj0 = None
    obj1 = None

    try:
        ra0 = 42.12345
        dec0 = -13.98765

        ra1 = 42.12345
        dec1 = -13.98834

        with SmartSession() as sess:
            obj0 = Object( _id=uuid.uuid4(),
                           name="HelloWorld",
                           ra=ra0 + rng.normal( scale=0.2/3600. ) / np.cos( dec0 * np.pi / 180. ),
                           dec=dec0 + rng.normal( scale=0.2/3600. ),
                           is_test=True,
                           is_bad=False
                          )
            obj0.calculate_coordinates()
            obj0.insert()

            obj1 = Object( _id=uuid.uuid4(),
                           name="GoodbyeWorld",
                           ra=ra1 + rng.normal( scale=0.2/3600. ) / np.cos( dec1 * np.pi / 180. ),
                           dec=dec1 + rng.normal( scale=0.2/3600. ),
                           is_test=True,
                           is_bad=False
                          )
            obj1.calculate_coordinates()
            obj1.insert()

            for i in range(nimages):
                img = Image( _id=uuid.uuid4(),
                             provenance_id=provenance_base.id,
                             format="fits",
                             type="Diff",
                             mjd=60000.,
                             end_mjd=60000.000694,
                             exp_time=60.,
                             instrument='DemoInstrument',
                             telescope='DemoTelescope',
                             section_id='1',
                             project='test',
                             target='test',
                             filter=bands[ rng.integers( 0, len(bands) ) ],
                             ra=ra0,
                             dec=dec0,
                             ra_corner_00=ra0-0.1,
                             ra_corner_01=ra0-0.1,
                             ra_corner_10=ra0+0.1,
                             ra_corner_11=ra0+0.1,
                             dec_corner_00=dec0-0.1,
                             dec_corner_10=dec0-0.1,
                             dec_corner_01=dec0+0.1,
                             dec_corner_11=dec0+0.1,
                             minra=ra0-0.1,
                             maxra=ra0+0.1,
                             mindec=dec0-0.1,
                             maxdec=dec0+0.1,
                             filepath=f'foo{i}.fits',
                             md5sum=uuid.uuid4() )
                img.calculate_coordinates()
                img.insert( session=sess )
                images.append( img )

                src = SourceList( _id=uuid.uuid4(),
                                  provenance_id=provenance_base.id,
                                  format='sextrfits',
                                  image_id=img.id,
                                  best_aper_num=-1,
                                  num_sources=666,
                                  filepath=f'foo{i}.sources.fits',
                                  md5sum=uuid.uuid4() )
                src.insert( session=sess )
                sourceses.append( src )

                cout = Cutouts( _id=uuid.uuid4(),
                                provenance_id=provenance_base.id,
                                sources_id=src.id,
                                format='hdf5',
                                filepath=f'foo{i}.cutouts.hdf5',
                                md5sum=uuid.uuid4() )
                cout.insert( session=sess )
                cutoutses.append( cout )

                mset = MeasurementSet( _id=uuid.uuid4(),
                                       provenance_id=provenance_base.id,
                                       cutouts_id=cout.id )
                mset.insert()
                measurementsets.append( mset )

                dex1 = rng.integers( 0, 128 )
                for whichmeas in range(2):
                    dex = dex1
                    if whichmeas == 1:
                        while dex == dex1:
                            dex = rng.integers( 0, 128 )
                    ra = ra0 if whichmeas==0 else ra1
                    dec = dec0 if whichmeas==0 else dec1
                    obj = obj0 if whichmeas==0 else obj1
                    dflux = rng.normal( 100., 10. )
                    sn = rng.exponential( 10. )
                    flux = dflux * sn

                ra += rng.normal( scale=possigma( sn ) )
                dec += rng.norma( scale=possigma( sn ) )

                meas = Measurements( _id=uuid.uuid4(),
                                     measurementset_id=mset.id,
                                     index_in_sources=dex1,
                                     flux_psf=flux,
                                     flux_psf_err=dflux,
                                     flux_apertures=[],
                                     flux_apertures_err=[],
                                     aper_radii=[],
                                     ra=ra,
                                     dec=dec,
                                     object_id=obj.id,
                                     # positioner doesn't use x/y, or the
                                     #  other measurements, but they're
                                     #  non-nullable, so just put stuff
                                     #  there.
                                     center_x_pixel=1024.,
                                     center_y_pixel=1024.,
                                     x=1024.,
                                     y=1024.,
                                     gfit_x=1024.,
                                     gfit_y=1024.,
                                     major_width=1.,
                                     minor_width=1.,
                                     position_angle=0.,
                                     is_bad=False
                                    )
                meas.insert( session=sess )
                measurementses.append( meas )

            yield True
    finally:
        with Psycopg2Connection() as conn:
            cursor=conn.cursor()
            cursor.execute( "DELETE FROM measurements WHERE _id=ANY(%(id)s)",
                            { 'id': [ m._id for m in measurementses ] } )
            cursor.execute( "DELETE FROM measurement_sets WHERE _id=ANY(%(id)s)",
                            { 'id': [ m._id for m in measurementsets ] } )
            cursor.execute( "DELETE FROM cutouts WHERE _id=ANY(%(id)s)",
                            { 'id': [ c._id for c in cutoutses ] } )
            cursor.execute( "DELETE FROM source_lists WHERE _id=ANY(%(id)s)",
                            { 'id': [ s._id for s in sourceses ] } )
            cursor.execute( "DELETE FROM images WHERE _id=ANY(%(id)s)",
                            { 'id': [ i._id for i in images ] } )
            if obj0 is not None:
                cursor.execute( "DELETE FROM objects WHERE _id=%(id)s", { 'id': obj0.id } )
            if obj1 is not None:
                cursor.execute( "DELETE FROM objects WHERE _id=%(id)s", { 'id': obj1.id } )

            conn.commit()


def test_positioner( fake_data_for_position_tests, provenance_base ):
    with SmartSession() as sess:
        obj = sess.query( Object ).filter( Object.name=="HelloWorld" ).first()
        assert obj is not None
        objpos = sess.query( ObjectPosition ).filter( ObjectPosition.object_id==obj.id ).all()
        assert len(objpos) == 0

    poser = Positioner( measuring_provenance_id=provenance_base.id )
    poser.run( obj.id )

    with SmartSession() as sess:
        objpos = sess.query( ObjectPosition ).filter( ObjectPosition.object_id==obj.id ).all()
        assert len(objpos) == 1
        objpos = objpos[0]

    import pdb; pdb.set_trace()
    pass
