import os
import hashlib
import pathlib
import random
import uuid
import json

import sqlalchemy as sa

import numpy as np

import pytest

import util.config as config
import models.base
from models.base import Base, SmartSession, AutoIDMixin, FileOnDiskMixin, FourCorners
from models.image import Image


def test_to_dict(data_dir):
    target = uuid.uuid4().hex
    filter = np.random.choice( ['g', 'r', 'i', 'z', 'y'] )
    mjd = np.random.rand() * 10000
    ra = np.random.rand() * 360
    dec = np.random.rand() * 180 - 90
    fwhm_estimate = np.random.rand() * 10

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


# ======================================================================
# FileOnDiskMixin test
#
# This set of tests isn't complete, because some of the actual saving
# functionality will be tested as a side effect of the save testing in
# test_image.py


class DiskFile(Base, AutoIDMixin, FileOnDiskMixin):
    """A temporary database table for testing FileOnDiskMixin

    """
    hexbarf = ''.join( [ random.choice( '0123456789abcdef' ) for i in range(8) ] )
    __tablename__ = f"test_diskfiles_{hexbarf}"
    nofile = True


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

    with SmartSession() as session:
        df = session.merge( df )
        df.remove_data_from_disk( remove_folders=True )
        if sa.inspect( df ).persistent:
            session.delete( df )
        session.expunge( df )
        session.commit()


def test_fileondisk_save_failuremodes( diskfile ):
    data1 = np.random.rand( 32 ).tobytes()
    md5sum1 = hashlib.md5( data1 ).hexdigest()
    data2 = np.random.rand( 32 ).tobytes()
    md5sum2 = hashlib.md5( data2 ).hexdigest()
    fname = "test_diskfile.dat"
    diskfile.filepath = fname

    # Verify failure of filepath_extensions, md5sum_extensions, and extension= are inconsistent
    diskfile.filepath_extensions = [ '1', '2' ]
    with pytest.raises( RuntimeError, match='Tried to save a non-extension file, but this file has extensions' ): 
        diskfile.save( data1 )
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[0], 'rb' )
        ifp.close()
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[1], 'rb' )
        ifp.close()
    diskfile.filepath_extensions = None
    diskfile.md5sum_extensions = [ uuid.uuid4(), uuid.uuid4() ]
    with pytest.raises( RuntimeError, match='Data integrity error; filepath_extensions is null.*' ):
        diskfile.save( data1 )
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath(), 'rb' )
        ifp.close()
    diskfile.filepath_extensions = [ '1', '2', '3' ]
    with pytest.raises( RuntimeError, match=r'Data integrity error; len\(md5sum_extensions\).*' ):
        diskfile.save( data1, extension='1' )
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[0], 'rb' )
        ifp.close()
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[1], 'rb' )
        ifp.close()
    diskfile.md5sum_extensions = None
    diskfile.filepath_extensions = [ '1', '2' ]
    with pytest.raises( RuntimeError, match='Data integrity error; filepath_extensions is not null.*' ):
        diskfile.save( data1, extension='1' )
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[0], 'rb' )
        ifp.close()
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath()[1], 'rb' )
        ifp.close()
    diskfile.filepath_extensions = None

    # Should fail if we pass something that's not bytes, string, or Path
    with pytest.raises( TypeError, match='data must be bytes, str, or Path.*' ):
        diskfile.save( int(42) )
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


def test_fileondisk_save_singlefile( diskfile, archive, config_test ):
    archivebase = f"{config_test.value('archive.local_read_dir')}/{config_test.value('archive.path_base')}"

    diskfile.filepath = 'test_fileondisk_save.dat'
    data1 = np.random.rand( 32 ).tobytes()
    md5sum1 = hashlib.md5( data1 ).hexdigest()
    data2 = np.random.rand( 32 ).tobytes()
    md5sum2 = hashlib.md5( data2 ).hexdigest()
    data3 = np.random.rand( 32 ).tobytes()
    md5sum3 = hashlib.md5( data3 ).hexdigest()
    assert md5sum1 != md5sum2
    assert md5sum2 != md5sum3
    assert md5sum1 != md5sum3

    # Save to the disk, make sure it doesn't go to the archive
    diskfile.save( data1, no_archive=True )
    assert diskfile.filepath_extensions == None
    assert diskfile.md5sum_extensions == None
    assert diskfile.md5sum == None
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
    with pytest.raises( FileNotFoundError ):
        ifp = open( f'{archivebase}{diskfile.filepath}', 'rb' )
        ifp.close()

    # Make sure we can delete it
    diskfile.remove_data_from_disk()
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath(), 'rb' )
        ifp.close()
    with pytest.raises( FileNotFoundError ):
        ifp = open( f'{archivebase}{diskfile.filepath}', 'rb' )
        ifp.close()

    # First save, verify it goes to the disk and to the archive
    diskfile.save( data1 )
    assert diskfile.md5sum.hex == md5sum1
    assert diskfile.md5sum_extensions == None
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
    with open( f'{archivebase}{diskfile.filepath}', 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that we can delete from disk without deleting from the archive
    diskfile.remove_data_from_disk()
    assert diskfile.md5sum.hex == md5sum1
    with pytest.raises( FileNotFoundError ):
        ifp = open( diskfile.get_fullpath(), 'rb' )
        ifp.close()
    with open( f'{archivebase}{diskfile.filepath}', 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that get_fullpath gets the file from the archive
    with open( diskfile.get_fullpath( nofile=False ), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that if the wrong file is on disk, it yells at us if we told it to verify md5
    with open( diskfile.get_fullpath(), 'wb' ) as ofp:
        ofp.write( data2 )
    with pytest.raises( ValueError, match=".*has md5sum.*on disk, which doesn't match the database value.*" ):
        path = diskfile.get_fullpath( nofile=False, always_verify_md5=True )

    # Clean up for further tests
    filename = diskfile.get_fullpath()
    filepath = diskfile.filepath
    diskfile.delete_from_disk_and_database()
    with pytest.raises( FileNotFoundError ):
        ifp = open( filename, 'rb' )
        ifp.close()
    with pytest.raises( FileNotFoundError ):
        ifp = open( f'{archivebase}{diskfile.filepath}', 'rb' )
        ifp.close()

    # Make sure that saving is happy if the file is already on disk in place
    with open( filename, 'wb' ) as ofp:
        ofp.write( data1 )
    diskfile.filepath = filepath  # this would usually be calculated by the subclass using invent_filepath
    diskfile.save( data1, overwrite=False, exists_ok=True, verify_md5=True )
    assert diskfile.md5sum.hex == md5sum1
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
    with open( f'{archivebase}{diskfile.filepath}', 'rb' ) as ifp:
        assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that we can overwrite
    diskfile.save( data2, overwrite=True )
    assert diskfile.md5sum.hex == md5sum2
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()
    with open( f'{archivebase}{diskfile.filepath}', 'rb' ) as ifp:
        assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that exists_ok=True and verify_md5sum=False behaves as expected
    diskfile.save( data3, overwrite=False, exists_ok=True, verify_md5=False )
    assert diskfile.md5sum.hex == md5sum2
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()
    with open( f'{archivebase}{diskfile.filepath}', 'rb' ) as ifp:
        assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that exists_ok=True and verify_md5sum=True behaves as expected
    diskfile.save( data3, overwrite=True, exists_ok=True, verify_md5=True )
    assert diskfile.md5sum.hex == md5sum3
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()
    with open( f'{archivebase}{diskfile.filepath}', 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that overwrite=False, exists_ok=True, verify_md5sum=True and right file on disk works
    diskfile.save( data3, overwrite=False, exists_ok=True, verify_md5=True )
    assert diskfile.md5sum.hex == md5sum3
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()
    with open( f'{archivebase}{diskfile.filepath}', 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()

    # Verify that overwrite=False, exist_ok=True, verify_md5sum=True and wrong file on disk fails
    # and doesn't overwrite file on disk or archive
    with pytest.raises( ValueError, match='.*exists, but its md5sum.*does not match.*' ):
        diskfile.save( data2, overwrite=False, exists_ok=True, verify_md5=True )
    assert diskfile.md5sum.hex == md5sum3
    with open( diskfile.get_fullpath(), 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()
    with open( f'{archivebase}{diskfile.filepath}', 'rb' ) as ifp:
        assert md5sum3 == hashlib.md5( ifp.read() ).hexdigest()


def test_fileondisk_save_singlefile_noarchive( diskfile ):
    # Verify that the md5sum of a file gets set when saving to a disk
    # when the archive is null.

    diskfile.filepath = 'test_fileondisk_save.dat'
    data1 = np.random.rand( 32 ).tobytes()
    md5sum1 = hashlib.md5( data1 ).hexdigest()

    cfg = config.Config.get()
    origcfgarchive = cfg.value( 'archive' )
    origarchive = models.base.ARCHIVE
    try:
        cfg.set_value( 'archive', None )
        models.base.ARCHIVE = None
        diskfile.save( data1, overwrite=False, exists_ok=False )
        assert diskfile.md5sum.hex == md5sum1
    finally:
        cfg.set_value( 'archive', origcfgarchive )
        models.base.ARCHIVE = origarchive


def test_fileondisk_save_multifile( diskfile, archive, config_test ):
    try:
        archivebase = f"{config_test.value('archive.local_read_dir')}/{config_test.value('archive.path_base')}"

        diskfile.filepath = 'test_fileondisk_save'
        data1 = np.random.rand( 32 ).tobytes()
        md5sum1 = hashlib.md5( data1 ).hexdigest()
        data2 = np.random.rand( 32 ).tobytes()
        md5sum2 = hashlib.md5( data2 ).hexdigest()
        data3 = np.random.rand( 32 ).tobytes()
        md5sum3 = hashlib.md5( data3 ).hexdigest()
        assert md5sum1 != md5sum2
        assert md5sum2 != md5sum3
        assert md5sum1 != md5sum3

        # Save an extension but not to archive
        diskfile.save( data1, extension='_1.dat', no_archive=True )
        assert diskfile.filepath_extensions == [ '_1.dat' ]
        assert diskfile.md5sum_extensions == [ None ]
        paths = diskfile.get_fullpath()
        assert paths == [ f'{diskfile.local_path}/{diskfile.filepath}_1.dat' ]
        with open( paths[0], 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with pytest.raises( FileNotFoundError ):
            ifp = open( f'{archivebase}{diskfile.filepath}_1.dat', 'rb')
            ifp.close()

        # Save a second extensions, this time to the archive
        diskfile.save( data2, extension='_2.dat' )
        assert diskfile.filepath_extensions == [ '_1.dat', '_2.dat' ]
        assert diskfile.md5sum_extensions == [ None, uuid.UUID(md5sum2) ]
        assert diskfile.md5sum == None
        paths = diskfile.get_fullpath()
        assert paths == [ f'{diskfile.local_path}/{diskfile.filepath}_1.dat',
                          f'{diskfile.local_path}/{diskfile.filepath}_2.dat' ]
        with open( paths[0], 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with open( paths[1], 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()
        with pytest.raises( FileNotFoundError ):
            ifp = open( f'{archivebase}{diskfile.filepath}_1.dat', 'rb')
            ifp.close()
        with open( f'{archivebase}{diskfile.filepath}_2.dat', 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

        # Verify that if we save the first extension again, but without noarchive,
        # that it goes up to the arhive
        diskfile.save( data1, extension='_1.dat' )
        assert diskfile.filepath_extensions == [ '_1.dat', '_2.dat' ]
        assert diskfile.md5sum_extensions == [ uuid.UUID(md5sum1), uuid.UUID(md5sum2) ]
        assert diskfile.md5sum == None
        paths = diskfile.get_fullpath()
        assert paths == [ f'{diskfile.local_path}/{diskfile.filepath}_1.dat',
                          f'{diskfile.local_path}/{diskfile.filepath}_2.dat' ]
        with open( paths[0], 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with open( paths[1], 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()
        with open( f'{archivebase}{diskfile.filepath}_1.dat', 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with open( f'{archivebase}{diskfile.filepath}_2.dat', 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

        # Make sure we can delete without purging the archive
        diskfile.remove_data_from_disk()
        assert diskfile.filepath_extensions == [ '_1.dat', '_2.dat' ]
        assert diskfile.md5sum_extensions == [ uuid.UUID(md5sum1), uuid.UUID(md5sum2) ]
        assert diskfile.md5sum == None
        paths = diskfile.get_fullpath()
        assert paths == [ f'{diskfile.local_path}/{diskfile.filepath}_1.dat',
                          f'{diskfile.local_path}/{diskfile.filepath}_2.dat' ]
        with pytest.raises( FileNotFoundError ):
            ifp = open( paths[0], 'rb' )
            ifp.close()
        with pytest.raises( FileNotFoundError ):
            ifp = open( paths[1], 'rb' )
            ifp.close()
        with open( f'{archivebase}{diskfile.filepath}_1.dat', 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with open( f'{archivebase}{diskfile.filepath}_2.dat', 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

        # Make sure we can get the file back from the archive
        paths = diskfile.get_fullpath( nofile=False )
        assert paths == [ f'{diskfile.local_path}/{diskfile.filepath}_1.dat',
                          f'{diskfile.local_path}/{diskfile.filepath}_2.dat' ]
        with open( paths[0], 'rb' ) as ifp:
            assert md5sum1 == hashlib.md5( ifp.read() ).hexdigest()
        with open( paths[1], 'rb' ) as ifp:
            assert md5sum2 == hashlib.md5( ifp.read() ).hexdigest()

    finally:
        # Delete it all
        paths = diskfile.get_fullpath()
        diskfile.delete_from_disk_and_database()
        assert diskfile.filepath_extensions is None
        assert diskfile.md5sum_extensions is None
        assert diskfile.md5sum is None
        assert diskfile.get_fullpath() is None
        with pytest.raises( FileNotFoundError ):
            ifp = open( paths[0], 'rb' )
            ifp.close()
        with pytest.raises( FileNotFoundError ):
            ifp = open( paths[1], 'rb' )
            ifp.close()
        with pytest.raises( FileNotFoundError ):
            ifp = open( f'{archivebase}{diskfile.filepath}_1.dat', 'rb' )
            ifp.close()
        with pytest.raises( FileNotFoundError ):
            ifp = open( f'{archivebase}{diskfile.filepath}_2.dat', 'rb' )
            ifp.close()

    # TODO : test various combinations of overwrite, exists_ok, and verify_md5
    # (Many of those code paths were already tested in the previous test,
    # but it would be good to verify that they work as expected with
    # multi-file saves.)


def test_fileondisk_save_multifile_noarchive( diskfile ):
    # Verify that the md5sum of a file gets set when saving to a disk
    # when the archive is null.

    diskfile.filepath = 'test_fileondisk_save.dat'
    data1 = np.random.rand( 32 ).tobytes()
    md5sum1 = hashlib.md5( data1 ).hexdigest()
    data2 = np.random.rand( 32 ).tobytes()
    md5sum2 = hashlib.md5( data2 ).hexdigest()
    assert md5sum1 != md5sum2

    cfg = config.Config.get()
    origcfgarchive = cfg.value( 'archive' )
    origarchive = models.base.ARCHIVE
    try:
        cfg.set_value( 'archive', None )
        models.base.ARCHIVE = None
        diskfile.save( data1, extension="_1.dat", overwrite=False, exists_ok=False )
        diskfile.save( data2, extension="_2.dat", overwrite=False, exists_ok=False )
        assert diskfile.md5sum is None
        assert diskfile.md5sum_extensions == [ uuid.UUID(md5sum1), uuid.UUID(md5sum2) ]
    finally:
        cfg.set_value( 'archive', origcfgarchive )
        models.base.ARCHIVE = origarchive


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

