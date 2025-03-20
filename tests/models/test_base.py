import pytest

import sys
import os
import hashlib
import pathlib
import random
import uuid
import json
import logging

import numpy as np

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from psycopg2.errors import UniqueViolation

import util.config as config
from util.logger import SCLogger
import models.base
from models.base import Base, SmartSession, Psycopg2Connection, UUIDMixin, FileOnDiskMixin, FourCorners
from models.image import Image
from models.datafile import DataFile
from models.object import Object


def test_to_dict(data_dir):
    target = uuid.uuid4().hex
    rng = np.random.default_rng()
    filter = rng.choice( ['g', 'r', 'i', 'z', 'y'] )
    mjd = rng.uniform() * 10000
    ra = rng.uniform() * 360
    dec = rng.uniform() * 180 - 90
    fwhm_estimate = rng.uniform() * 10

    im1 = Image( target=target, filter=filter, mjd=mjd, ra=ra, dec=dec, fwhm_estimate=fwhm_estimate )
    output_dict = im1.to_dict()
    im2 = Image( **output_dict )
    assert im1.target == im2.target
    assert im1.filter == im2.filter
    assert im1.mjd == im2.mjd
    assert im1.ra == im2.ra
    assert im1.dec == im2.dec
    assert im1.fwhm_estimate == im2.fwhm_estimate

    # now try to save it to a YAML file
    try:
        filename = os.path.join(data_dir, 'test_to_dict.json')
        im1.to_json(filename)

        with open(filename, 'r') as fp:
            output_dict = json.load(fp)

        im3 = Image( **output_dict )
        assert im1.target == im3.target
        assert im1.filter == im3.filter
        assert im1.mjd == im3.mjd
        assert im1.ra == im3.ra
        assert im1.dec == im3.dec
        assert im1.fwhm_estimate == im3.fwhm_estimate

    finally:
        os.remove(filename)


# ====================
# Test basic database operations
#
# Using the DataFile model here because it's a relatively lightweight
#   model with a minimum of relationships.  Will spoof md5sum so
#   we don't have to actually save any data to disk.

def test_insert( provenance_base ):

    def make_sure_its_there( _id, filepath ):
        df = DataFile.get_by_id( _id )
        assert df is not None
        assert df.filepath == filepath

    def make_sure_its_not_there( _id, filepath ):
        df = DataFile.get_by_id( _id )
        assert df is None

    uuidstodel = []
    try:
        curid = uuid.uuid4()
        uuidstodel.append( curid )
        df = DataFile( _id=curid, filepath="foo", md5sum=uuid.uuid4(), provenance_id=provenance_base.id )

        # Make sure we get an error if we don't pass the right kind of thing
        with pytest.raises( TypeError, match="session must be a sa Session or psycopg2.extensions.connection or None" ):
            df.insert( 2 )

        # Make sure we can insert
        df.insert()
        make_sure_its_there( curid, "foo" )

        founddf = DataFile.get_by_id( df.id )
        assert founddf is not None
        assert founddf.filepath == df.filepath
        assert founddf.md5sum == df.md5sum
        # We could check that these times are less than datetime.datetime.now(tz=datetime.timezone.utc),
        # but they might fail of the database server and host server clocks aren't exactly in sync.
        assert founddf.created_at is not None
        assert founddf.modified is not None

        # Make sure we get an error if we try to insert something that already exists
        newdf = DataFile( _id=df.id, filepath='bar', md5sum=uuid.uuid4(), provenance_id=provenance_base.id )
        with pytest.raises( UniqueViolation, match='duplicate key value violates unique constraint "data_files_pkey"' ):
            newdf.insert()

        # Make sure we can insert using a session
        curid = uuid.uuid4()
        uuidstodel.append( curid )
        df = DataFile( _id=curid, filepath="foo2", md5sum=uuid.uuid4(), provenance_id=provenance_base.id )
        with SmartSession() as session:
            df.insert( session )
        make_sure_its_there( curid, "foo2" )

        # Make sure nocommit with session works as expected
        curid = uuid.uuid4()
        uuidstodel.append( curid )
        df = DataFile( _id=curid, filepath="foo3", md5sum=uuid.uuid4(), provenance_id=provenance_base.id )
        with SmartSession() as sess:
            df.insert( sess, nocommit=True )
            sess.rollback()
        make_sure_its_not_there( curid, "foo3" )

        # Make sure we can insert passing a psycopg2 connection
        curid = uuid.uuid4()
        uuidstodel.append( curid )
        df = DataFile( _id=curid, filepath="foo4", md5sum=uuid.uuid4(), provenance_id=provenance_base.id )
        with Psycopg2Connection() as conn:
            df.insert( conn )
        make_sure_its_there( curid, "foo4" )

        # Make sure nocommit with connection works as expected
        curid = uuid.uuid4()
        uuidstodel.append( curid )
        df = DataFile( _id=curid, filepath="foo5", md5sum=uuid.uuid4(), provenance_id=provenance_base.id )
        with Psycopg2Connection() as conn:
            df.insert( conn, nocommit=True )
        make_sure_its_not_there( curid, "foo5" )


    finally:
        # Clean up
        with SmartSession() as sess:
            sess.execute( sa.delete( DataFile ).where( DataFile._id.in_( uuidstodel ) ) )
            sess.commit()


def test_upsert( provenance_base ):

    uuidstodel = [ uuid.uuid4() ]
    try:
        assert Image.get_by_id( uuidstodel[0] ) is None

        image = Image( _id = uuidstodel[0],
                       provenance_id = provenance_base.id,
                       mjd = 60575.474664,
                       end_mjd = 60575.4750116,
                       exp_time = 30.,
                       # instrument = 'DemoInstrument',
                       telescope = 'DemoTelescope',
                       project = 'test',
                       target = 'nothing',
                       filepath = 'foo/bar.fits',
                       ra = '23.',
                       dec = '42.',
                       ra_corner_00 = 22.5,
                       ra_corner_01 = 22.5,
                       ra_corner_10 = 23.5,
                       ra_corner_11 = 23.5,
                       dec_corner_00 = 41.5,
                       dec_corner_01 = 42.5,
                       dec_corner_10 = 41.5,
                       dec_corner_11 = 42.5,
                       minra = 22.5,
                       maxra = 23.5,
                       mindec = 41.5,
                       maxdec = 42.5,
                       md5sum = uuid.uuid4()       # spoof since we didn't save a file
                      )

        # Make sure the database yells at us if a required column is missing

        with pytest.raises( IntegrityError, match='null value in column "instrument".*violates not-null' ):
            image.upsert()

        # == Make sure we can insert a thing == a
        image.instrument = 'DemoInstrument'
        image.upsert()

        # Object didn't get updated
        assert image._format is None
        assert image.preproc_bitflag is None
        assert image.created_at is None
        assert image.modified is None

        found = Image.get_by_id( image.id )
        assert found is not None

        # Check the server side defaults
        assert found._format == 1
        assert found.preproc_bitflag == 0
        assert found.created_at is not None
        assert found.modified == found.created_at

        # Change something, do an update
        found.project = 'another_test'
        found.upsert()
        refound = Image.get_by_id( image.id )
        for col in sa.inspect( Image ).c:
            if col.name == 'modified':
                assert refound.modified > found.modified
            elif col.name == 'project':
                assert refound.project == 'another_test'
            else:
                assert getattr( found, col.name ) == getattr( refound, col.name )

        # Verify that we get a new image and the id is generated if the id starts undefined
        refound._id = None
        refound.filepath = 'foo/bar_none.fits'

        refound.upsert()
        assert refound._id is not None
        uuidstodel.append( refound._id )

        with SmartSession() as session:
            multifound = session.query( Image ).filter( Image._id.in_( uuidstodel ) ).all()
            assert len(multifound) == 2
            assert set( [ i.id for i in multifound ] ) == set( uuidstodel )

        # Now verify that server-side values *do* get updated if we ask for it

        image.upsert( load_defaults=True )
        assert image.created_at is not None
        assert image.modified is not None
        assert image.created_at < image.modified
        assert image._format == 1
        assert image.preproc_bitflag == 0

        # Make sure they don't always revert to defaults
        image._format = 2
        image.upsert( load_defaults=True )
        assert image._format == 2
        found = Image.get_by_id( image.id )
        assert found._format == 2

    finally:
        # Clean up
        with SmartSession() as sess:
            sess.execute( sa.delete( Image ).where( Image._id.in_( uuidstodel ) ) )
            sess.commit()

# TODO : test test_upsert_list when one of the object properties is a SQL array.


# This test also implicitly tests UUIDMixin.get_by_id and UUIDMixin.get_back_by_ids
def test_upsert_list( code_version, provenance_base, provenance_extra ):
    # Set the logger to show the SQL emitted by SQLAlchemy for this test.
    # (See comments in models/base.py UUIDMixin.upsert_list.)
    # See this with pytest --capture=tee-sys
    curloglevel = logging.getLogger( 'sqlalchemy.engine' ).level
    logging.getLogger( 'sqlalchemy.engine' ).setLevel( logging.INFO )
    loghandler = logging.StreamHandler( sys.stderr )
    logging.getLogger( 'sqlalchemy.engine' ).addHandler( loghandler )

    uuidstodel = []
    try:
        df1 = DataFile( filepath="foo", md5sum=uuid.uuid4(), provenance_id=provenance_base.id )
        df2 = DataFile( filepath="bar", md5sum=uuid.uuid4(), provenance_id=provenance_base.id )
        df3 = DataFile( filepath="cat", md5sum=uuid.uuid4(), provenance_id=provenance_base.id )
        df4 = DataFile( filepath="dog", md5sum=uuid.uuid4(), provenance_id=provenance_base.id )
        df5 = DataFile( filepath="mouse", md5sum=uuid.uuid4(), provenance_id=provenance_base.id )
        uuidstodel.extend( [ df1.id, df2.id, df3.id, df4.id, df5.id ] )

        # Make sure it yells at us if all the objects aren't the right thing,
        #   and that it doesn't actually insert anything
        SCLogger.debug( "Trying to fail" )
        gratuitous = Object( name='nothing', ra=0., dec=0. )
        with pytest.raises( TypeError, match="passed objects weren't all of this class!" ):
            DataFile.upsert_list( [ df1, df2, gratuitous ] )

        SCLogger.debug( "Making sure nothing got inserted" )
        them = DataFile.get_batch_by_ids( [ df1.id, df2.id ] )
        assert len(them) == 0

        # Make sure we can insert
        SCLogger.debug( "Upserting df1, df2" )
        DataFile.upsert_list( [ df1, df2 ] )
        SCLogger.debug( "Getting df1, df2, df3 by id one at a time" )
        founddf1 = DataFile.get_by_id( df1.id )
        founddf2 = DataFile.get_by_id( df2.id )
        founddf3 = DataFile.get_by_id( df3.id )
        assert founddf1 is not None
        assert founddf2 is not None
        assert founddf3 is None

        df3.insert()

        # Test updating and inserting at the same time (Doing extra
        # files here so that we can see the generated SQL when lots of
        # things happen in upsert_list.)
        df1.filepath = "wombat"
        df1.md5sum = uuid.uuid4()
        df2.filepath = "mongoose"
        df2.md5sum = uuid.uuid4()
        SCLogger.debug( "Upserting df1, df2, df4, df5" )
        DataFile.upsert_list( [ df1, df2, df4, df5 ] )

        SCLogger.debug( "Getting df1 through df5 in a batch" )
        objs = DataFile.get_batch_by_ids( [ df1.id, df2.id, df3.id, df4.id, df5.id ], return_dict=True )
        assert objs[df1.id].filepath == "wombat"
        assert objs[df1.id].md5sum == df1.md5sum
        assert objs[df2.id].filepath == "mongoose"
        assert objs[df2.id].md5sum == df2.md5sum
        assert objs[df3.id].filepath == "cat"
        assert objs[df4.id].filepath == df4.filepath
        assert objs[df5.id].filepath == df5.filepath
        assert objs[df1.id].modified > objs[df1.id].created_at
        assert objs[df2.id].modified > objs[df2.id].created_at
        assert objs[df3.id].modified == objs[df3.id].created_at
        assert objs[df4.id].modified == objs[df4.id].created_at
        assert objs[df5.id].modified == objs[df5.id].created_at

    finally:
        # Clean up
        SCLogger.debug( "Cleaning up" )
        with SmartSession() as sess:
            sess.execute( sa.delete( DataFile ).where( DataFile._id.in_( uuidstodel ) ) )
            sess.commit()
        logging.getLogger( 'sqlalchemy.engine' ).setLevel( curloglevel )
        logging.getLogger( 'sqlalchemy.engine' ).removeHandler( loghandler )


# ======================================================================
# FileOnDiskMixin test
#
# This set of tests isn't complete, because some of the actual saving
# functionality will be tested as a side effect of the save testing in
# test_image.py


class DiskFile(Base, UUIDMixin, FileOnDiskMixin):
    """A temporary database table for testing FileOnDiskMixin"""
    hexbarf = ''.join( [ random.choice( '0123456789abcdef' ) for i in range(8) ] )
    __tablename__ = f"test_diskfiles_{hexbarf}"
    nofile = True

    def get_downstreams( self, session=None ):
        return []


@pytest.fixture(scope='session')
def diskfiletable():
    with SmartSession() as session:
        DiskFile.__table__.create( session.bind )
        session.commit()
    yield True
    with SmartSession() as session:
        DiskFile.__table__.drop( session.bind )
        session.commit()


@pytest.fixture
def diskfile( diskfiletable ):
    df = DiskFile()
    yield df

    df.delete_from_disk_and_database()


def test_fileondisk_save_failuremodes( diskfile ):
    rng = np.random.default_rng()
    data1 = rng.uniform( size=32 ).tobytes()
    # md5sum1 = hashlib.md5( data1 ).hexdigest()
    data2 = rng.uniform( size=32 ).tobytes()
    # md5sum2 = hashlib.md5( data2 ).hexdigest()
    fname = "test_diskfile.dat"
    diskfile.filepath = fname

    # Verify failure of components, md5sum_components, and component= are inconsistent
    diskfile.components = [ '1', '2' ]
    with pytest.raises( RuntimeError, match='Tried to save a non-component file, but this file has components' ):
        diskfile.save( data1 )
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[0], 'rb' )
        ifp.close()
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[1], 'rb' )
        ifp.close()
    diskfile.components = None
    diskfile.md5sum_components = [ uuid.uuid4(), uuid.uuid4() ]
    with pytest.raises( RuntimeError, match='Data integrity error; components is null.*' ):
        diskfile.save( data1 )
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath(), 'rb' )
        ifp.close()
    diskfile.components = [ '1', '2', '3' ]
    with pytest.raises( RuntimeError, match=r'Data integrity error; len\(md5sum_components\).*' ):
        diskfile.save( data1, component='1' )
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[0], 'rb' )
        ifp.close()
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[1], 'rb' )
        ifp.close()
    diskfile.md5sum_components = None
    diskfile.components = [ '1', '2' ]
    with pytest.raises( RuntimeError, match='Data integrity error; components is not null.*' ):
        diskfile.save( data1, component='1' )
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[0], 'rb' )
        ifp.close()
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[1], 'rb' )
        ifp.close()
    diskfile.components = None

    # Should fail if we pass something that's not bytes, string, or Path
    with pytest.raises( TypeError, match='data must be bytes, str, or Path.*' ):
        diskfile.save( 42 )
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath(), 'rb' )
        ifp.close()

    # Should fail if the path exists but is a directory
    filepath = pathlib.Path( diskfile.get_fullpath( nofile=True ) )
    filepath.mkdir( parents=True )
    with pytest.raises( RuntimeError, match ='.*exists but is not a file!.*' ):
        diskfile.save( data1 )
    filepath.rmdir()

    # Make sure it won't overwrite if overwrite = False and exists_ok = False
    with open( filepath, 'wb' ) as ofp:
        ofp.write( data1 )
    with pytest.raises( FileExistsError ):
        diskfile.save( data1, overwrite=False, exists_ok=False )
    # Make sure it won't overwrite if overwrite=False and md5sum doesn't match
    with pytest.raises( ValueError, match='.*its md5sum.*does not match md5sum.*' ):
        diskfile.save( data2, overwrite=False, exists_ok=True, verify_md5=True )
    filepath.unlink()


def test_fileondisk_save_singlefile( diskfile, archive, test_config, data_dir ):
    archive_dir = archive.test_folder_path
    diskfile.filepath = 'test_fileondisk_save.dat'
    rng = np.random.default_rng()
    data1 = rng.uniform( size=32 ).tobytes()
    md5sum1 = hashlib.md5( data1 ).hexdigest()
    data2 = rng.uniform( size=32 ).tobytes()
    md5sum2 = hashlib.md5( data2 ).hexdigest()
    data3 = rng.uniform( size=32 ).tobytes()
    md5sum3 = hashlib.md5( data3 ).hexdigest()
    assert md5sum1 != md5sum2
    assert md5sum2 != md5sum3
    assert md5sum1 != md5sum3

    # Save to the disk, make sure it doesn't go to the archive
    if os.path.isfile(os.path.join(archive_dir, diskfile.filepath)):
        os.remove(os.path.join(archive_dir, diskfile.filepath))
    diskfile.save( data1, no_archive=True )
    assert diskfile.components is None
    assert diskfile.md5sum_components is None
    assert diskfile.md5sum is None
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
    with pytest.raises( FileNotFoundError ):
        ifp = open( os.path.join(archive_dir, diskfile.filepath), 'rb' )
        ifp.close()

    # Make sure we can delete it
    diskfile.remove_data_from_disk()
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath(), 'rb' )
        ifp.close()
    with pytest.raises( FileNotFoundError ):
        ifp = open( os.path.join(archive_dir, diskfile.filepath), 'rb' )
        ifp.close()

    # First save, verify it goes to the disk and to the archive
    diskfile.save( data1 )
    assert diskfile.md5sum.hex == md5sum1
    assert diskfile.md5sum_components is None
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
    with open( os.path.join(archive_dir, diskfile.filepath), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that we can delete from disk without deleting from the archive
    diskfile.remove_data_from_disk()
    assert diskfile.md5sum.hex == md5sum1
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath(), 'rb' )
        ifp.close()
    with open( os.path.join(archive_dir, diskfile.filepath), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that get_fullpath gets the file from the archive
    with open( diskfile.get_fullpath( nofile=False ), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that if the wrong file is on disk, it yells at us if we told it to verify md5
    with open( diskfile.get_fullpath(), 'wb' ) as ofp:
        ofp.write( data2 )
    with pytest.raises( ValueError, match=".*has md5sum.*on disk, which doesn't match the database value.*" ):
        diskfile.get_fullpath( nofile=False, always_verify_md5=True )

    # Clean up for further tests
    filename = diskfile.get_fullpath()
    filepath = diskfile.filepath
    diskfile.delete_from_disk_and_database()
    with pytest.raises( FileNotFoundError ):
        ifp = open( filename, 'rb' )
        ifp.close()
    with pytest.raises( FileNotFoundError ):
        ifp = open( os.path.join(archive_dir, filepath), 'rb' )
        ifp.close()

    # Make sure that saving is happy if the file is already on disk in place
    with open( filename, 'wb' ) as ofp:
        ofp.write( data1 )
    diskfile.filepath = filepath  # this would usually be calculated by the subclass using invent_filepath
    diskfile.save( data1, overwrite=False, exists_ok=True, verify_md5=True )
    assert diskfile.md5sum.hex == md5sum1
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
    with open( os.path.join(archive_dir, diskfile.filepath), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that we can overwrite
    diskfile.save( data2, overwrite=True )
    assert diskfile.md5sum.hex == md5sum2
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()
    with open( os.path.join(archive_dir, diskfile.filepath), 'rb' ) as ifp:
        assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that exists_ok=True and verify_md5sum=False behaves as expected
    diskfile.save( data3, overwrite=False, exists_ok=True, verify_md5=False )
    assert diskfile.md5sum.hex == md5sum2
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()
    with open( os.path.join(archive_dir, diskfile.filepath), 'rb' ) as ifp:
        assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that exists_ok=True and verify_md5sum=True behaves as expected
    diskfile.save( data3, overwrite=True, exists_ok=True, verify_md5=True )
    assert diskfile.md5sum.hex == md5sum3
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()
    with open( os.path.join(archive_dir, diskfile.filepath), 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that overwrite=False, exists_ok=True, verify_md5sum=True and right file on disk works
    diskfile.save( data3, overwrite=False, exists_ok=True, verify_md5=True )
    assert diskfile.md5sum.hex == md5sum3
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()
    with open( os.path.join(archive_dir, diskfile.filepath), 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that overwrite=False, exist_ok=True, verify_md5sum=True and wrong file on disk fails
    # and doesn't overwrite file on disk or archive
    with pytest.raises( ValueError, match='.*exists, but its md5sum.*does not match.*' ):
        diskfile.save( data2, overwrite=False, exists_ok=True, verify_md5=True )
    assert diskfile.md5sum.hex == md5sum3
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()
    with open( os.path.join(archive_dir, diskfile.filepath), 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()


def test_fileondisk_save_singlefile_noarchive( diskfile ):
    # Verify that the md5sum of a file gets set when saving to a disk
    # when the archive is null.

    diskfile.filepath = 'test_fileondisk_save.dat'
    rng = np.random.default_rng()
    data1 = rng.uniform( size=32 ).tobytes()
    md5sum1 = hashlib.md5( data1 ).hexdigest()

    origcfgobj = config.Config._configs[ config.Config._default ]
    origarchive = models.base.ARCHIVE
    try:
        cfg = config.Config.get( static=False )
        # NEVER DO THIS.  For this test, I want to change the value of
        #   'archive' in the default config, so I'm screwing around
        #   with the internal structure of the Config class.  If you
        #   find yourself doing this anywhere outside of a test like this,
        #   then either rethink what you need, or submit issues suggesting
        #   that the config system needs to be modified.  (This change
        #   is undone in the finaly block below.)
        config.Config._configs[ config.Config._default ] = cfg
        cfg.set_value( 'archive', None )
        models.base.ARCHIVE = None
        diskfile.save( data1, overwrite=False, exists_ok=False )
        assert diskfile.md5sum.hex == md5sum1
    finally:
        models.base.ARCHIVE = origarchive
        config.Config._configs[ config.Config._default ] = origcfgobj


def test_fileondisk_save_multifile( diskfile, archive, test_config):
    archive_dir = archive.test_folder_path
    try:
        diskfile.filepath = 'test_fileondisk_save'
        rng = np.random.default_rng()
        data1 = rng.uniform( size=32 ).tobytes()
        md5sum1 = hashlib.md5( data1 ).hexdigest()
        data2 = rng.uniform( size=32 ).tobytes()
        md5sum2 = hashlib.md5( data2 ).hexdigest()
        data3 = rng.uniform( size=32 ).tobytes()
        md5sum3 = hashlib.md5( data3 ).hexdigest()
        assert md5sum1 != md5sum2
        assert md5sum2 != md5sum3
        assert md5sum1 != md5sum3

        # Save an component but not to archive
        diskfile.save( data1, component='_1.dat', no_archive=True )
        assert diskfile.components == [ '_1.dat' ]
        assert diskfile.md5sum_components == [ None ]
        paths = diskfile.get_fullpath()
        assert paths == [ f'{diskfile.local_path}/{diskfile.filepath}._1.dat' ]
        with open( paths[0], 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with pytest.raises( FileNotFoundError ):
            ifp = open( f'{archive_dir}{diskfile.filepath}._1.dat', 'rb')
            ifp.close()

        # Save a second component, this time to the archive
        diskfile.save( data2, component='_2.dat' )
        assert diskfile.components == [ '_1.dat', '_2.dat' ]
        assert diskfile.md5sum_components == [ None, uuid.UUID(md5sum2) ]
        assert diskfile.md5sum is None
        paths = diskfile.get_fullpath()
        assert paths == [ f'{diskfile.local_path}/{diskfile.filepath}._1.dat',
                          f'{diskfile.local_path}/{diskfile.filepath}._2.dat' ]

        archive_path1 = os.path.join(archive_dir, diskfile.filepath) + '._1.dat'
        archive_path2 = os.path.join(archive_dir, diskfile.filepath) + '._2.dat'

        with open( paths[0], 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with open( paths[1], 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()
        with pytest.raises( FileNotFoundError ):
            ifp = open( archive_path1, 'rb')
            ifp.close()
        with open( archive_path2, 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

        # Verify that if we save the first component again, but without noarchive,
        # that it goes up to the archive
        diskfile.save( data1, component='_1.dat' )
        assert diskfile.components == [ '_1.dat', '_2.dat' ]
        assert diskfile.md5sum_components == [ uuid.UUID(md5sum1), uuid.UUID(md5sum2) ]
        assert diskfile.md5sum is None
        paths = diskfile.get_fullpath()
        assert paths == [
            os.path.join(diskfile.local_path, diskfile.filepath) + '._1.dat',
            os.path.join(diskfile.local_path, diskfile.filepath) + '._2.dat',
        ]
        with open( paths[0], 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with open( paths[1], 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()
        with open( archive_path1, 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with open( archive_path2, 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

        # Make sure we can delete without purging the archive
        diskfile.remove_data_from_disk()
        assert diskfile.components == [ '_1.dat', '_2.dat' ]
        assert diskfile.md5sum_components == [ uuid.UUID(md5sum1), uuid.UUID(md5sum2) ]
        assert diskfile.md5sum is None
        paths = diskfile.get_fullpath()
        assert paths == [
            os.path.join(diskfile.local_path, diskfile.filepath) + '._1.dat',
            os.path.join(diskfile.local_path, diskfile.filepath) + '._2.dat',
        ]

        with pytest.raises( FileNotFoundError ):
            ifp = open( paths[0], 'rb' )
            ifp.close()
        with pytest.raises( FileNotFoundError ):
            ifp = open( paths[1], 'rb' )
            ifp.close()
        with open( archive_path1, 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with open( archive_path2, 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

        # Make sure we can get the file back from the archive
        paths = diskfile.get_fullpath( nofile=False )
        assert paths == [
            os.path.join(diskfile.local_path, diskfile.filepath) + '._1.dat',
            os.path.join(diskfile.local_path, diskfile.filepath) + '._2.dat',
        ]
        with open( paths[0], 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with open( paths[1], 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

    finally:
        # Delete it all
        paths = diskfile.get_fullpath()
        diskfile.delete_from_disk_and_database()
        assert diskfile.components is None
        assert diskfile.md5sum_components is None
        assert diskfile.md5sum is None
        assert diskfile.get_fullpath() is None
        with pytest.raises( FileNotFoundError ):
            ifp = open( paths[0], 'rb' )
            ifp.close()
        with pytest.raises( FileNotFoundError ):
            ifp = open( paths[1], 'rb' )
            ifp.close()
        with pytest.raises( FileNotFoundError ):
            ifp = open( archive_path1, 'rb' )
            ifp.close()
        with pytest.raises( FileNotFoundError ):
            ifp = open( archive_path2, 'rb' )
            ifp.close()

    # TODO : test various combinations of overwrite, exists_ok, and verify_md5
    # (Many of those code paths were already tested in the previous test,
    # but it would be good to verify that they work as expected with
    # multi-file saves.)


def test_fileondisk_save_multifile_noarchive( diskfile ):
    # Verify that the md5sum of a file gets set when saving to a disk
    # when the archive is null.

    diskfile.filepath = 'test_fileondisk_save.dat'
    rng = np.random.default_rng()
    data1 = rng.uniform( size=32 ).tobytes()
    md5sum1 = hashlib.md5( data1 ).hexdigest()
    data2 = rng.uniform( size=32 ).tobytes()
    md5sum2 = hashlib.md5( data2 ).hexdigest()
    assert md5sum1 != md5sum2

    origcfgobj = config.Config._configs[ config.Config._default ]
    origarchive = models.base.ARCHIVE
    try:
        cfg = config.Config.get( static=False )
        # NEVER DO THIS.  For this test, I want to change the value of
        #   'archive' in the default config, so I'm screwing around
        #   with the internal structure of the Config class.  If you
        #   find yourself doing this anywhere outside of a test like this,
        #   then either rethink what you need, or submit issues suggesting
        #   that the config system needs to be modified.  (This change
        #   is undone in the finaly block below.)
        config.Config._configs[ config.Config._default ] = cfg
        cfg.set_value( 'archive', None )
        models.base.ARCHIVE = None
        diskfile.save( data1, component="_1.dat", overwrite=False, exists_ok=False )
        diskfile.save( data2, component="_2.dat", overwrite=False, exists_ok=False )
        assert diskfile.md5sum is None
        assert diskfile.md5sum_components == [ uuid.UUID(md5sum1), uuid.UUID(md5sum2) ]
    finally:
        models.base.ARCHIVE = origarchive
        config.Config._configs[ config.Config._default ] = origcfgobj


def test_fourcorners_sort_radec():
    ras = [ 0, 1, 0, 1 ]
    decs = [ 1, 1, 0, 0 ]
    ras, decs = FourCorners.sort_radec( ras, decs )
    assert ras ==  [ 0, 0, 1, 1 ]
    assert decs == [ 0, 1, 0, 1 ]

    # Rotate 30 degrees
    ras =  [  1.366, -1.366,  0.366, -0.366 ]
    decs = [  0.366, -0.366, -1.366,  1.366 ]
    ras, decs = FourCorners.sort_radec( ras, decs )
    assert ras ==  [ -1.366, -0.366,  0.366, 1.366 ]
    assert decs == [ -0.366,  1.366, -1.366, 0.366 ]

    # Make sure ra/dec spanning 0 works right
    ras = [ 0.19, 0.21, 359.79, 359.81 ]
    decs = [ -0.21, 0.19, -0.19, 0.21 ]
    ras, decs = FourCorners.sort_radec( ras, decs )
    assert ras == [ 359.79, 359.81, 0.19, 0.21 ]
    assert decs == [ -0.19, 0.21, -0.21, 0.19 ]

    ras = [ 0.39, 0.41, 359.99, 0.01 ]
    decs = [ -0.21, 0.19, -0.19, 0.21 ]
    ras, decs = FourCorners.sort_radec( ras, decs )
    assert ras == [ 359.99, 0.01, 0.39, 0.41 ]
    assert decs == [ -0.19, 0.21, -0.21, 0.19 ]


def test_fourcorners_contains():
    dra = 0.75
    ddec = 0.375
    rawcorners = np.array( [ [ -dra/2.,  -dra/2.,   dra/2.,   dra/2. ],
                             [ -ddec/2.,  ddec/2., -ddec/2.,  ddec/2. ] ] )
    ras = [ 0., 10. ]
    decs = [ 0., -45, 80. ]
    angles = [ 0., 15., 30., 45. ]
    # (I drew a rectangle in LibreOffice Draw and rotated it to decide the points and truth values below visually.)
    offsets = { 0.: { ( 0,    0  ): True,
                      ( 0.9,  0.9): True,
                      ( 0.,   0.9): True,
                      (-0.9,  0.9): True },
                15.: { ( 0,   0): True,
                       (-0.8,-0.8): False,
                       (-0.8, 0.8): True,
                       (-0.8,-0.4): True,
                       (-0.8, 0.4): True,
                       (-0.8, 0. ): True,
                       ( 0.,  0.9 ) : True,
                       ( 0., -0.9 ) : True },
                30.: { ( 0,   0): True,
                       (-0.8,-0.8): False,
                       (-0.8, 0.8): True,
                       (-0.8,-0.4): False,
                       (-0.8, 0.4): True,
                       (-0.8, 0. ): True,
                       ( 0.,  0.9 ) : True,
                       ( 0., -0.9 ) : True },
                45.: { ( 0,   0): True,
                       (-0.8,-0.8): False,
                       (-0.8, 0.8): True,
                       (-0.8,-0.4): False,
                       (-0.8, 0.4): True,
                       (-0.8, 0. ): False,
                       ( 0.,  0.9 ) : True,
                       ( 0., -0.9 ) : True },
               }

    for ra in ras:
        for dec in decs:
            for angle in angles:
                rotmat = np.array( [ [  np.cos( angle * np.pi/180. ), np.sin( angle * np.pi/180. ) ],
                                     [ -np.sin( angle * np.pi/180. ), np.cos( angle * np.pi/180. ) ] ] )
                corners = np.matmul( rotmat, rawcorners )
                corners[0, :] /= np.cos( dec * np.pi/180. )
                corners[0, :] += ra
                corners[1, :] += dec
                minra = min( corners[ 0, : ] )
                maxra = max( corners[ 0, : ] )
                mindec = min( corners[ 1, : ] )
                maxdec = max( corners[ 1, : ] )
                minra = minra if minra > 0 else minra + 360.
                maxra = maxra if maxra > 0 else maxra
                corners[ 0, corners[0,:]<0. ] += 360.
                obj = Image( ra=ra, dec=dec,
                             ra_corner_00=corners[0][0],
                             ra_corner_01=corners[0][1],
                             ra_corner_10=corners[0][2],
                             ra_corner_11=corners[0][3],
                             minra=minra, maxra=maxra,
                             dec_corner_00=corners[1][0],
                             dec_corner_01=corners[1][1],
                             dec_corner_10=corners[1][2],
                             dec_corner_11=corners[1][3],
                             mindec=mindec, maxdec=maxdec )
                for offset, included in offsets[angle].items():
                    checkra = ra + dra/np.cos( dec*np.pi/180 ) * offset[0]/2.
                    checkra = checkra if checkra > 0. else checkra + 360.
                    checkdec = dec + ddec * offset[1]/2.
                    assert obj.contains( checkra, checkdec ) == included


def test_fourcorners_overlap_frac():
    dra = 0.75
    ddec = 0.375
    radec1 = [(10., -3.), (10., -45.), (10., -80.), ( 0., 0. ), ( 0., 80 ), ( 359.9, 20. ), ( 0.2, -20 ) ]

    for ra, dec in radec1:
        cd = np.cos(dec * np.pi / 180.)
        minra = ra - dra / 2. / cd
        maxra = ra + dra / 2. / cd
        minra = minra if minra >= 0. else minra + 360.
        minra = minra if minra < 360. else minra - 360.
        maxra = maxra if maxra >= 0. else maxra + 360.
        maxra = maxra if maxra < 360. else maxra - 360.
        mindec = dec - ddec / 2.
        maxdec = dec + ddec / 2.
        ras, decs = FourCorners.sort_radec( [ minra, minra, maxra, maxra ], [ mindec, maxdec, mindec, maxdec ] )
        i1 = Image( ra=ra, dec=dec,
                    ra_corner_00=ras[0],
                    ra_corner_01=ras[1],
                    ra_corner_10=ras[2],
                    ra_corner_11=ras[3],
                    minra=minra,
                    maxra=maxra,
                    dec_corner_00=decs[0],
                    dec_corner_01=decs[1],
                    dec_corner_10=decs[2],
                    dec_corner_11=decs[3],
                    mindec=mindec,
                    maxdec=maxdec )
        for frac, offx, offy in [(1., 0., 0.),
                                 (0.5, 0.5, 0.),
                                 (0.5, -0.5, 0.),
                                 (0.5, 0., 0.5),
                                 (0.5, 0., -0.5),
                                 (0.25, 0.5, 0.5),
                                 (0.25, -0.5, 0.5),
                                 (0.25, 0.5, -0.5),
                                 (0.25, -0.5, -0.5),
                                 (0., 1., 0.),
                                 (0., -1., 0.),
                                 (0., 0., 1.),
                                 (0., 0., -1.),
                                 (0., -1., -1.),
                                 (0., 1., -1.),
                                 (0., -1., 1.),
                                 (0., 1., -1.)]:
            ra2 = ra + offx * dra / cd
            ra2 = ra2 if ra2 >= 0. else ra2 + 360.
            ra2 = ra2 if ra2 < 360. else ra2 - 360.
            dec2 = dec + offy * ddec
            cd = np.cos( dec2 * np.pi / 180. )
            minra = ra2 - dra / 2. / cd
            maxra = ra2 + dra / 2. / cd
            minra = minra if minra >= 0. else minra + 360.
            minra = minra if minra < 360. else minra - 360.
            maxra = maxra if maxra >= 0. else maxra + 360.
            maxra = maxra if maxra < 360. else maxra - 360.
            mindec= dec2 - ddec / 2.
            maxdec = dec2 + ddec / 2.
            ras, decs = FourCorners.sort_radec( [ minra, minra, maxra, maxra ], [ mindec, maxdec, mindec, maxdec ] )
            i2 = Image( ra=ra2, dec=dec2,
                        ra_corner_00=ras[0],
                        ra_corner_01=ras[1],
                        ra_corner_10=ras[2],
                        ra_corner_11=ras[3],
                        minra=minra,
                        maxra=maxra,
                        dec_corner_00=decs[0],
                        dec_corner_01=decs[1],
                        dec_corner_10=decs[2],
                        dec_corner_11=decs[3],
                        mindec=mindec,
                        maxdec=maxdec )
            calcfrac = FourCorners.get_overlap_frac(i1, i2)
            # More leeway for high dec
            if np.fabs( dec ) > 70.:
                assert calcfrac == pytest.approx( frac, abs=0.02 )
            else:
                assert calcfrac == pytest.approx(frac, abs=0.01)

    # Test an absurd case to make sure that we don't have problems with
    # the assumptions made for ra around 0. (Such problems existed
    # before we added this test....)

    i1 = Image( ra=0., dec=0.,
                ra_corner_00=359.9,
                ra_corner_01=359.9,
                ra_corner_10=0.1,
                ra_corner_11=0.1,
                minra=359.9,
                maxra=0.1,
                dec_corner_00=-0.1,
                dec_corner_10=-0.1,
                dec_corner_01=1.0,
                dec_corner_11=1.0,
                mindec=-0.1,
                maxdec=0.1 )
    i2 = Image( ra=20., dec=-45.,
                ra_corner_00=19.8586,
                ra_corner_01=19.8586,
                ra_corner_10=20.1414,
                ra_corner_11=20.1414,
                minra=19.8586,
                maxra=20.1414,
                dec_corner_00=-45.1,
                dec_corner_10=-45.1,
                dec_corner_01=-44.9,
                dec_corner_11=-44.9,
                mindec=-45.1,
                maxdec=-44.9 )
    assert FourCorners.get_overlap_frac( i1, i2 ) == 0


    # Not-square-to-the-sky tests. Start with a reference square image.
    # Do this at a few different RAs and decs so that we can test our
    # spherical trig assumptions and ra near 0.
    # (Note: the spherical trig assumption tests are kind of circular,
    # since I use the same cos(dec) factor to calculate the coordinates
    # as I then use later in the overlap calculation.  Maybe somebody
    # should try putting real spherical trig in here.)

    dra = 0.1
    ddec = 0.1
    for ctrra, ctrdec in zip( [ 0., 0.,  0.,  10., 10., 10. ],
                              [ 0., 25., 80., 0.,  25., 80. ]  ):
        # Make a reference image that's square on the sky
        minra = ctrra - dra / np.cos( ctrdec * np.pi / 180. )
        minra = minra if minra > 0 else minra + 360.
        maxra = ctrra + dra / np.cos( ctrdec * np.pi / 180. )
        mindec = ctrdec - ddec
        maxdec = ctrdec + ddec
        i1 = Image( ra=ctrra, dec=ctrdec,
                    ra_corner_00=minra, ra_corner_01=minra, minra=minra,
                    ra_corner_10=maxra, ra_corner_11=maxra, maxra=maxra,
                    dec_corner_00=mindec, dec_corner_10=mindec, mindec=mindec,
                    dec_corner_01=maxdec, dec_corner_11=maxdec, maxdec=maxdec )

        # Rotate in place by 45°.  Overlap should be 0.828 (from geometry)
        minra = ctrra - np.sqrt( dra**2 + ddec**2 ) / np.cos( ctrdec * np.pi / 180. )
        minra = minra if minra > 0 else minra + 360.
        maxra = ctrra + np.sqrt( dra**2 + ddec**2 ) / np.cos( ctrdec * np.pi / 180. )
        mindec = ctrdec - np.sqrt( dra**2 + ddec**2 )
        maxdec = ctrdec + np.sqrt( dra**2 + ddec**2 )
        i2 = Image( ra=ctrra, dec=ctrdec,
                    ra_corner_00=ctrra, ra_corner_01=minra, ra_corner_11=ctrra, ra_corner_10=maxra,
                    dec_corner_00=mindec, dec_corner_01=ctrdec, dec_corner_11=maxdec, dec_corner_10=ctrdec,
                    minra=minra, maxra=maxra, mindec=mindec, maxdec=maxdec )
        assert FourCorners.get_overlap_frac( i1, i2 ) == pytest.approx( 0.828, abs=0.01 )

        # Rotate in place by a few angles <45°. (For expected overlap
        # fraction, I didn't do geometry, I trusted the 0,0 result).
        # Numbers look plausible, though.
        for ang, ovfrac in zip( [ 5.,    10.,   20.,    30. ],
                                [ 0.960, 0.927, 0.877,  0.845 ] ):
            rotmat = np.array( [ [ np.cos( ang*np.pi/180. ), -np.sin( ang*np.pi/180. ) ],
                                 [ np.sin( ang*np.pi/180. ),  np.cos( ang*np.pi/180. ) ] ] )
            # (I should probably made a 4d array and do things less verbosely.  Oh well.)
            corner00 = np.matmul( rotmat, np.array( [ [ -dra ], [ -ddec ] ] ) )
            corner10 = np.matmul( rotmat, np.array( [ [  dra ], [ -ddec ] ] ) )
            corner01 = np.matmul( rotmat, np.array( [ [ -dra ], [  ddec ] ] ) )
            corner11 = np.matmul( rotmat, np.array( [ [  dra ], [  ddec ] ] ) )
            corner00[0][0] = ctrra + corner00[0][0] / np.cos( ctrdec * np.pi / 180. )
            corner01[0][0] = ctrra + corner01[0][0] / np.cos( ctrdec * np.pi / 180. )
            corner10[0][0] = ctrra + corner10[0][0] / np.cos( ctrdec * np.pi / 180. )
            corner11[0][0] = ctrra + corner11[0][0] / np.cos( ctrdec * np.pi / 180. )
            corner00[1][0] = ctrdec + corner00[1][0]
            corner01[1][0] = ctrdec + corner01[1][0]
            corner10[1][0] = ctrdec + corner10[1][0]
            corner11[1][0] = ctrdec + corner11[1][0]
            minra = min( corner00[0][0], corner01[0][0], corner10[0][0], corner11[0][0] )
            maxra = max( corner00[0][0], corner01[0][0], corner10[0][0], corner11[0][0] )
            mindec = min( corner00[1][0], corner01[1][0], corner10[1][0], corner11[1][0] )
            maxdec = max( corner00[1][0], corner01[1][0], corner10[1][0], corner11[1][0] )
            i2 = Image( ra=ctrra, dec=ctrdec,
                        ra_corner_00=corner00[0][0], ra_corner_10=corner10[0][0], ra_corner_01=corner01[0][0],
                        ra_corner_11=corner11[0][0], minra=minra, maxra=maxra,
                        dec_corner_00=corner00[1][0], dec_corner_10=corner10[1][0], dec_corner_01=corner01[1][0],
                        dec_corner_11=corner11[1][0], mindec=mindec, maxdec=maxdec )
            assert FourCorners.get_overlap_frac( i1, i2 ) == pytest.approx( ovfrac, abs=0.01 )

    # TODO : not-square-to-the-sky tests where the centers don't overlap
    # TODO : not-square-to-the-sky tests where dra does not equal ddec
