import pytest
import uuid
import datetime

import numpy as np
import astropy.time

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
    #   give us a pos sigma of 0.1" at s/n 20 or higher, scaling up
    #   linearly to a pos sigma of 1.5" at s/n 3 (and more for worse)
    possigma = lambda sn: max( 0.1, 1.5 - ( 1.4 * (sn-3)/17 ) ) / 3600.

    rng = np.random.default_rng( seed=42 )

    bands = [ 'g', 'r', 'i', 'z' ]
    nimages = 200
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
                             mjd=60000. + i,
                             end_mjd=60000.000694 + i,
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

                    ra += rng.normal( scale=possigma( sn ) / np.cos( dec * np.pi / 180. ) )
                    dec += rng.normal( scale=possigma( sn ) )

                    meas = Measurements( _id=uuid.uuid4(),
                                         measurementset_id=mset.id,
                                         index_in_sources=dex,
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

            yield ra0, dec0
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


def test_positioner( fake_data_for_position_tests, provenance_base, provenance_extra ):
    ra0, dec0 = fake_data_for_position_tests

    with SmartSession() as sess:
        obj = sess.query( Object ).filter( Object.name=="HelloWorld" ).first()
        assert obj is not None
        objpos = sess.query( ObjectPosition ).filter( ObjectPosition.object_id==obj.id ).all()
        assert len(objpos) == 0

    t1 = astropy.time.Time( 60100, format='mjd' ).to_datetime( datetime.UTC ).isoformat()
    poser = Positioner( measuring_provenance_id=provenance_base.id, datetime=t1 )
    retval = poser.run( obj.id )
    assert poser.has_recalculated

    # Make sure the database got loaded with the same ObjectPosition we got returned
    with SmartSession() as sess:
        objpos = sess.query( ObjectPosition ).filter( ObjectPosition.object_id==obj.id ).all()
        assert len(objpos) == 1
        objpos = objpos[0]
        assert objpos.id == retval.id
        assert objpos.ra == pytest.approx( retval.ra, rel=1e-9 )
        assert objpos.dec == pytest.approx( retval.dec, rel=1e-9 )

    # Make sure the position is within uncertainty of what's expected
    assert objpos.ra == pytest.approx( ra0, abs=3. * objpos.dra )
    assert objpos.dec == pytest.approx( dec0, abs=3. * objpos.ddec )
    # TODO: rationalize these, right now they're just "what I got"
    assert objpos.dra < 0.16
    assert objpos.ddec < 0.16

    # If we use an later time that includes more measurements, we should get a better position
    t2 = astropy.time.Time( 69999, format='mjd' ).to_datetime( datetime.UTC ).isoformat()
    poser = Positioner( measuring_provenance_id=provenance_base.id, datetime=t2 )
    objpos2 = poser.run( obj.id )
    assert poser.has_recalculated
    assert objpos2.ra == pytest.approx( ra0, abs=3. * objpos2.dra )
    assert objpos2.dec == pytest.approx( dec0, abs=3. * objpos2.ddec )
    assert objpos2.dra < objpos.ra
    assert objpos2.ddec < objpos.ddec

    # Make sure it finds a pre-existing position
    objpos3 = poser.run( obj.id )
    assert not poser.has_recalculated
    assert objpos3.id == objpos2.id

    # This previous things used the ra and dec in the object table as a
    # starting point.  Now try using a previous ObjectPosition as a
    # starting point and see if we get a different answer.  Make the search
    # radius small to increase the chance that we won't catch all the same stuff
    # even when the search positions are not the same.
    poser = Positioner( measuring_provenance_id=provenance_base.id, datetime=t1, radius=0.25 )
    objpos3 = poser.run( obj.id )
    assert poser.has_recalculated
    assert objpos3.ra == pytest.approx( ra0, abs=3. * objpos3.dra )
    assert objpos3.dec == pytest.approx( dec0, abs=3. * objpos3.ddec )
    poser = Positioner( measuring_provenance_id=provenance_base.id, datetime=t1, radius=0.25,
                        current_position_provenance_id=objpos2.provenance_id )
    objpos4 = poser.run( obj.id )
    assert poser.has_recalculated
    assert objpos4.ra == pytest.approx( ra0, abs=3. * objpos3.dra )
    assert objpos4.dec == pytest.approx( dec0, abs=3. * objpos3.ddec )
    assert objpos4.ra != pytest.approx( objpos3.ra, abs=1e-5 )
    assert objpos4.dec != pytest.approx( objpos3.dec, abs=1e-5 )

    # Make sure that if we try to base a position on a previous position,
    #  but no previous position with that provenance exists, it either
    #  fails or falls back, based on configuration.
    with pytest.raises( RuntimeError, match="Cannot find current position for object" ):
        poser = Positioner( measuring_provenance_id=provenance_base.id, datetime=t1,
                            current_position_provenance_id=provenance_extra.id,
                            fall_back_object_position=False )
        poser.run( obj.id )

    poser = Positioner( measuring_provenance_id=provenance_base.id, datetime=t1,
                        current_position_provenance_id=provenance_extra.id,
                        fall_back_object_position=True )
    objpos5 = poser.run( obj.id )
    assert poser.has_recalculated
    assert objpos5.ra == pytest.approx( ra0, abs=3. * objpos5.dra )
    assert objpos5.dec == pytest.approx( dec0, abs=3. * objpos5.ddec )
    # In fact, these positions should be the same as the origina objpos;
    #   that was based off of object position, here we should have
    #   fallen back to object position
    assert objpos5.ra == pytest.approx( objpos.ra, rel=1e-9 )
    assert objpos5.dec == pytest.approx( objpos.dec, rel=1e-9 )

    # Give an absurd S/N to make sure it throws everything out
    with pytest.raises( RuntimeError, match="No matching measurements with S/N>" ):
        poser = Positioner( measuring_provenance_id=provenance_base.id, datetime=t2, sncut=1e10 )
        poser.run( obj.id )

    # Give an absurd sigma clipping cutoff to make sure it throws everything out
    with pytest.raises( RuntimeError, match="For object .*, nothing passed the sigma clipping!" ):
        poser = Positioner( measuring_provenance_id=provenance_base.id, datetime=t2, sigma_clip=0.01 )
        poser.run( obj.id )

    # I'd love to test that our handling of the race condition described
    # in the comments in positioner.py::Positioner.run works, but to do
    # that I'd have to put in various options for sleeps and such into
    # that code, and I hate to mung up the code just for purposes of
    # testing something that probably won't come up very often....
