import os

import numpy as np

import pytest
import pathlib
import random

from astropy.io import fits

from util.util import read_fits_image, save_fits_image_file

from models.base import FileOnDiskMixin


def test_read_fits_image(decam_fits_image_filename, cache_dir):
    # by default only get the data
    filename = os.path.join(cache_dir, 'DECam', decam_fits_image_filename)
    data = read_fits_image(filename)

    assert data.shape == (4094, 2046)
    assert np.array_equal(data[0, 0:10], np.array([303.96603, 304.1212, 291.13953, 301.72168, 306.76215, 314.1882,
                                                   323.1495, 324.92514, 336.54843, 369.187], dtype=np.float32))

    # get only the header
    header = read_fits_image(filename, output='header')

    assert header['LMT_MG'] == 25.37038556706342

    # get both as a tuple
    data, header = read_fits_image(filename, output='both')

    assert data.shape == (4094, 2046)
    assert np.array_equal(data[0, 0:10], np.array([303.96603, 304.1212, 291.13953, 301.72168, 306.76215, 314.1882,
                                                   323.1495, 324.92514, 336.54843, 369.187], dtype=np.float32))

    assert header['LMT_MG'] == 25.37038556706342


@pytest.fixture
def fits_file():
    filename = ( ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) ) )
    filepath = pathlib.Path( FileOnDiskMixin.temp_path ) / filename

    data = np.zeros( (64, 32), dtype=np.float32 )
    hdr = fits.Header()
    hdr[ 'TEST1' ] = 'testing 1'
    hdr[ 'TEST2' ] = 'testing 2'

    savedpath = pathlib.Path( save_fits_image_file( str(filepath), data, hdr ) )

    yield filepath, savedpath

    savedpath.unlink()


@pytest.fixture
def fits_single_file():
    filename = ( ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) ) )
    filepath = pathlib.Path( FileOnDiskMixin.temp_path ) / filename

    data = np.zeros( (64, 32), dtype=np.float32 )
    hdr = fits.Header()
    hdr[ 'TEST1' ] = 'testing 1'
    hdr[ 'TEST2' ] = 'testing 2'

    savedpath = pathlib.Path( save_fits_image_file( str(filepath), data, hdr,
                                                                   extname='image', single_file=True ) )

    yield filepath, savedpath

    savedpath.unlink()


@pytest.fixture
def two_extension_fits_file():
    filename = ( ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) ) )
    filepath = pathlib.Path( FileOnDiskMixin.temp_path ) / filename

    data = np.full( (64, 32), 3.141, dtype=np.float32 )
    hdr = fits.Header()
    hdr[ 'TEST1' ] = 'testing 64'
    hdr[ 'TEST2' ] = 'testing 128'

    savedpath1 = save_fits_image_file( str(filepath), data, hdr, extname='image', single_file=True )

    data = np.full( (64, 32), 2.718, dtype=np.float32 )
    hdr[ 'TEST1' ] = 'Rosencrantz'
    hdr[ 'TEST2' ] = 'Guildenstern'

    savedpath2 = save_fits_image_file( str(filepath), data, hdr, extname='weight', single_file=True )

    assert savedpath1 == savedpath2

    savedpath = pathlib.Path( savedpath1 )
    yield str(savedpath)

    savedpath.unlink()


def test_basic_save_fits_file( fits_file ):
    _, fullpath = fits_file
    with fits.open( fullpath ) as ifp:
        assert ifp[0].header['BITPIX'] == -32
        assert ifp[0].header['NAXIS'] == 2
        assert ifp[0].header['NAXIS1'] == 32
        assert ifp[0].header['NAXIS2'] == 64
        assert 'BSCALE' not in ifp[0].header
        assert 'BZERO' not in ifp[0].header
        assert ifp[0].header['TEST1'] == 'testing 1'
        assert ifp[0].header['TEST2'] == 'testing 2'
        assert ifp[0].data.dtype == np.dtype('>f4')
        assert ( ifp[0].data == np.zeros( ( 64, 32 ) ) ).all()


def test_save_separate_extension( fits_file ):
    filepath, fullpath = fits_file
    nextpath = filepath.parent / f'{filepath.name}.next.fits'

    data = np.full( (64, 32), 1., dtype=np.float32 )
    hdr = fits.Header()
    hdr[ 'EXTTEST1' ] = 'extension testing 1'
    hdr[ 'EXTTEST2' ] = 'extension testing 2'

    try:
        save_fits_image_file( str(filepath), data, hdr, extname='next' )

        with fits.open( fullpath ) as ifp:
            assert ifp[0].header['TEST1'] == 'testing 1'
            assert ifp[0].header['TEST2'] == 'testing 2'
            assert ( ifp[0].data == np.zeros( ( 64, 32 ) ) ).all()

        with fits.open( nextpath ) as ifp:
            assert ifp[0].header['EXTTEST1'] == 'extension testing 1'
            assert ifp[0].header['EXTTEST2'] == 'extension testing 2'
            assert ( ifp[0].data == np.full( ( 64, 32 ), 1., dtype=np.float32 ) ).all()
    finally:
        nextpath.unlink( missing_ok=True )


def test_save_extension( fits_single_file ):
    filepath, fullpath = fits_single_file

    data = np.full( (64, 32), 1., dtype=np.float32 )
    hdr = fits.Header()
    hdr[ 'EXTTEST1' ] = 'extension testing 1'
    hdr[ 'EXTTEST2' ] = 'extension testing 2'

    save_fits_image_file( str(filepath), data, hdr, extname='next', single_file=True )

    with fits.open( fullpath ) as ifp:
        assert ifp[1].header['TEST1'] == 'testing 1'
        assert ifp[1].header['TEST2'] == 'testing 2'
        assert ( ifp[1].data == np.zeros( ( 64, 32 ) ) ).all()
        assert ifp['image'].header['TEST1'] == 'testing 1'
        assert ifp['image'].header['TEST2'] == 'testing 2'
        assert ( ifp['image'].data == np.zeros( ( 64, 32 ) ) ).all()
        assert ifp[2].header['EXTTEST1'] == 'extension testing 1'
        assert ifp[2].header['EXTTEST2'] == 'extension testing 2'
        assert ( ifp[2].data == np.full( ( 64, 32 ), 1., dtype=np.float32 ) ).all()
        assert ifp['next'].header['EXTTEST1'] == 'extension testing 1'
        assert ifp['next'].header['EXTTEST2'] == 'extension testing 2'
        assert ( ifp['next'].data == np.full( ( 64, 32 ), 1., dtype=np.float32 ) ).all()


def test_no_overwrite( fits_file ):
    filepath, fullpath = fits_file

    data = np.full( (64, 32), 3., dtype=np.float32 )
    hdr = fits.Header()
    hdr[ 'TEST1' ] = 'testing 42'
    hdr[ 'TEST2' ] = 'testing 64738'

    with pytest.raises( OSError, match='File.*already exists' ):
        _ = pathlib.Path( save_fits_image_file( str(filepath), data, hdr, overwrite=False ) )
    with fits.open( fullpath ) as ifp:
        assert ifp[0].header['TEST1'] == 'testing 1'
        assert ifp[0].header['TEST2'] == 'testing 2'
        assert ( ifp[0].data == np.zeros( ( 64, 32 ) ) ).all()


def test_overwrite( fits_file ):
    filepath, fullpath = fits_file

    data = np.full( (64, 32), 3., dtype=np.float32 )
    hdr = fits.Header()
    hdr[ 'TEST1' ] = 'testing 42'
    hdr[ 'TEST2' ] = 'testing 64738'

    savedpath = pathlib.Path( save_fits_image_file( str(filepath), data, hdr, overwrite=True ) )
    assert savedpath == fullpath
    with fits.open( fullpath ) as ifp:
        assert ifp[0].header['TEST1'] == 'testing 42'
        assert ifp[0].header['TEST2'] == 'testing 64738'
        assert ( ifp[0].data == np.full( (64, 32), 3., dtype=np.float32 ) ).all()


def test_basic_read( fits_file ):
    _, fullpath = fits_file

    hdr = read_fits_image( fullpath, output='header' )
    assert isinstance( hdr, fits.Header )
    assert hdr['TEST1'] == 'testing 1'

    data = read_fits_image( fullpath, output='data' )
    assert data.dtype == np.float32
    assert ( data == np.zeros( ( 64, 32 ) ) ).all()

    data, hdr = read_fits_image( fullpath, output='both' )
    assert isinstance( hdr, fits.Header )
    assert hdr['TEST1'] == 'testing 1'
    assert ( data == np.zeros( ( 64, 32 ) ) ).all()


def test_read_extension( two_extension_fits_file ):
    hdr = read_fits_image( two_extension_fits_file, ext='image', output='header' )
    assert isinstance( hdr, fits.Header )
    assert hdr['TEST1'] == 'testing 64'
    assert hdr['TEST2'] == 'testing 128'

    hdr = read_fits_image( two_extension_fits_file, ext='weight', output='header' )
    assert isinstance( hdr, fits.Header )
    assert hdr['TEST1'] == 'Rosencrantz'
    assert hdr['TEST2'] == 'Guildenstern'

    data = read_fits_image( two_extension_fits_file, ext='image', output='data' )
    assert data.dtype == np.float32
    assert ( data == np.full( (64, 32), 3.141, dtype=np.float32 ) ).all()

    data = read_fits_image( two_extension_fits_file, ext='weight', output='data' )
    assert data.dtype == np.float32
    assert ( data == np.full( (64, 32), 2.718, dtype=np.float32 ) ).all()

    data, hdr = read_fits_image( two_extension_fits_file, ext='image', output='both' )
    assert isinstance( hdr, fits.Header )
    assert hdr['TEST1'] == 'testing 64'
    assert hdr['TEST2'] == 'testing 128'
    assert ( data == np.full( (64, 32), 3.141,dtype=np.float32 ) ).all()

    data, hdr = read_fits_image( two_extension_fits_file, ext='weight', output='both' )
    assert isinstance( hdr, fits.Header )
    assert hdr['TEST1'] == 'Rosencrantz'
    assert hdr['TEST2'] == 'Guildenstern'
    assert ( data == np.full( (64, 32), 2.718, dtype=np.float32 ) ).all()


def test_just_update_header( fits_file ):
    filepath, fullpath = fits_file

    with fits.open( fullpath ) as ifp:
        header = ifp[0].header
        data = ifp[0].data

    header['TEST3'] = 'added'
    data = np.full( (64, 32), 1.414, dtype=np.float32 )

    savedpath = save_fits_image_file( str(filepath), data, header, just_update_header=True )
    assert pathlib.Path( savedpath ) == fullpath

    with fits.open( fullpath) as ifp:
        assert ifp[0].header['TEST3'] == 'added'
        assert ( ifp[0].data == np.zeros( (64, 32), dtype=np.float32 ) ).all()
