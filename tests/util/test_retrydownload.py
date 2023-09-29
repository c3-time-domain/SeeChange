import pytest
import hashlib
import pathlib
import random
import string

from util.retrydownload import retry_download

# Ideally, when we set up the test environment, it should also set up an
# http server that the test environment controls; we then put files
# there that we download and verify in the tests.
#
# For now, I've put some static 1MB files at
#   https://portal.nersc.gov/cfs/m2218/fiducial_test_files
# If those files go away, these tests will fail, but hopefully
# they will be at leasdt somewhat stable.
#
# (found on nersc in /global/cfs/cdirs/m2218/www/fiducial_test_files )

md5sum1 = '3f6217f6e68efa71a711ed7083ed1348'
nonexistent = '3f1d598431502e796c3852981d97a576'
url1 = f'https://portal.nersc.gov/cfs/m2218/fiducial_test_files/{md5sum1}.dat'
url_nonexistent = f'https://portal.nersc.gov/cfs/m2218/fiducial_test_files/{nonexistent}.dat'

@pytest.fixture( scope='module' )
def testfile1():
    p = pathlib.Path( f'{md5sum1}.dat' )
    if p.exists():
        p.unlink()
    retry_download( url1, p )

    yield p, url1, md5sum1

    p.unlink()

def checkmd5( path, md5sum ):
    md5 = hashlib.md5()
    with open( path, "rb" ) as ifp:
        md5.update( ifp.read() )
    return md5.hexdigest() == md5sum

def test_no_overwrite_dir():
    fname = "".join( random.choices( string.ascii_lowercase, k=10 ) )
    fpath = pathlib.Path( fname )
    assert not fpath.exists()
    try:
        fpath.mkdir()
        with pytest.raises( FileExistsError, match='.*exists and is not a file' ):
            retry_download( url1, fpath, exists_ok=True, clobber=False )
        with pytest.raises( FileExistsError, match='.*exists and is not a file' ):
            retry_download( url1, fpath, exists_ok=False, clobber=False )
        with pytest.raises( FileExistsError, match='.*exists and is not a file' ):
            retry_download( url1, fpath, exists_ok=False, clobber=True )
        with pytest.raises( FileExistsError, match='.*exists and is not a file' ):
            retry_download( url1, fpath, exists_ok=True, clobber=True )
    finally:
        fpath.rmdir()
        assert not fpath.exists()

def test_basic_download( testfile1 ):
    path, url, md5sum = testfile1
    assert checkmd5( path, md5sum )

def test_404():
    with pytest.raises( RuntimeError, match='3 exceptions trying to download.*, failing.' ):
        retry_download( url_nonexistent, f'{nonexistent}.dat', retries=3, sleeptime=1, exists_ok=False )

def test_overwrite_misc_file():
    fname = "".join( random.choices( string.ascii_lowercase, k=10 ) )
    fpath = pathlib.Path( fname )
    assert not fpath.exists()
    try:
        with open( fpath, "wb" ) as ofp:
            ofp.write( b'abc' )
        md5sumabc = '900150983cd24fb0d6963f7d28e17f72'
        assert checkmd5( fpath, md5sumabc )

        # Make sure it fails when it's supposed to
        with pytest.raises( FileExistsError, match='.*exists and exists_ok is false' ):
            retry_download( url1, fpath, exists_ok=False )
        with pytest.raises( FileExistsError,
                            match=".*exists but md5sum.*doesn't match expected.*and clobber is False" ):
            retry_download( url1, fpath, exists_ok=True, md5sum=md5sum1 )

        # Make sure it "succeeds" when not given an md5sum to verify
        retry_download( url1, fpath, exists_ok=True, clobber=False )
        assert checkmd5( fpath, md5sumabc )
        retry_download( url1, fpath, exists_ok=True, clobber=True )
        assert checkmd5( fpath, md5sumabc )

        # Make sure it redownloads the file when given an md5sum
        retry_download( url1, fpath, md5sum=md5sum1, exists_ok=True, clobber=True )
        assert fpath.exists()
        assert checkmd5( fpath, md5sum1 )

        # Make sure it doesn't error out now that the md5sum  of the file is right even when clobber is False
        retry_download( url1, fpath, md5sum=md5sum1, exists_ok=True, clobber=False )
        assert fpath.exists()
        assert checkmd5( fpath, md5sum1 )

        # Make sure it fails out if exists_ok is False even though the right file is there
        with pytest.raises( FileExistsError, match='.*already exists and exists_ok is false' ):
            retry_download( url1, fpath, md5sum=md5sum1, exists_ok=False )

    finally:
        fpath.unlink()
        assert not fpath.exists()

