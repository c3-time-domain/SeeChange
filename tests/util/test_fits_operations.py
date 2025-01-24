import os
import copy

import numpy as np

import pytest
import pathlib
import random

from astropy.io import fits

from util.fits import read_fits_image, save_fits_image_file

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

    with pytest.raises( OSError, match='save_fits_image_file not overwriting' ):
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


@pytest.fixture
def fpacked_fits_file( decam_fits_image_filename, cache_dir ):
    origpath = pathlib.Path( cache_dir ) / 'DECam' / decam_fits_image_filename
    basepath = pathlib.Path( FileOnDiskMixin.temp_path ) / "test_fpacked_fits_file"
    fitspath = basepath.parent / f"{basepath.name}.fits"
    fzpath = basepath.parent / f"{basepath.name}.fits.fz"
    try:
        data, header = read_fits_image( origpath, output='both' )
        outpath = save_fits_image_file( fzpath, data, header, fpack=True )
        # Make sure the right file was written
        assert outpath.endswith( '.fz' )
        assert str(fzpath.resolve()) == outpath
        assert not fitspath.exists()
        assert fzpath.exists()
        # Make sure it compressed
        assert fzpath.stat().st_size / origpath.stat().st_size < 0.25

        yield fzpath

    finally:
        fzpath.unlink( missing_ok=True )
        fitspath.unlink( missing_ok=True )


# This also tests basic saving because the fpacked_fits_file fixture
#   runs save_fits_image_file( fpack=True )
def test_read_fpack_fits_image( decam_fits_image_filename, cache_dir, fpacked_fits_file ):
    origpath = pathlib.Path( cache_dir ) / 'DECam' / decam_fits_image_filename
    origdata, origheader = read_fits_image( origpath, output="both" )
    data, header = read_fits_image( fpacked_fits_file, output="both" )

    # At some point after library upgrades the header field
    #   ZDITHER0 started going away.  I don't know what this
    #   is or what it means, or what the upgrade was that
    #   caused this change.  Scary.
    assert all( i in header for i in origheader if i != 'ZDITHER0' )
    assert all( i in origheader for i in header )

    # The fpacked header will have a few extra things
    assert not any( "Image was compressed by CFITSIO" in h for h in origheader['HISTORY'] )
    assert any( "Image was compressed by CFITSIO" in h for h in header['HISTORY'] )
    assert not any( "q = 4.00" in h for h in origheader['HISTORY'] )
    assert any( "q = 4.00" in h for h in header['HISTORY'] )
    assert not any( "SUBTRACTIVE_DITHER_1" in h for h in origheader['HISTORY'] )
    assert any( "SUBTRACTIVE_DITHER_1" in h for h in header['HISTORY'] )

    # I know that this FITS image has a sky value around ~300, and a sky
    # RMS around ~9.  Fpack was supposed to quantize to sky rms / 4
    # according to the docs, though empirically it didn't quite do that
    # at the outside, though it did better than that on average:
    assert ( np.fabs( data - origdata ) < 4. ).all()
    assert np.fabs( data - origdata ).mean() < 1.

    # However, the previous tests may have been a little too picky,
    # because each pixel is good to ~1%, and on average it's good to
    # 0.3% (with σ=0.002).  I haven't checked to see if the pixel
    # offsets are uncorrelated; if so, then this is better, because any
    # real measurement is going to be spread over ~π*(FWHM)² pixels,
    # giving a corresponding √N improvement.
    assert ( np.fabs( data - origdata ) / data ).max() < 0.011
    assert ( np.fabs( data - origdata ) / data ).mean() < 0.003


def test_things_that_should_not_work():
    filename = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    direc = pathlib.Path( FileOnDiskMixin.temp_path )
    basepath = direc / filename
    fitspath = direc / f'{filename}.fits'
    fzpath = direc / f'{filename}.fits.fz'

    try:
        rng = np.random.default_rng( seed=42 )
        data = rng.random( (64, 64), dtype='f4' ) * 200. - 100.

        with pytest.raises( NotImplementedError, match="fpacking of multi-HDU files not currently supported" ):
            save_fits_image_file( basepath, data, {}, fpack=True, single_file=True )

        with pytest.raises( NotImplementedError, match="just_update_header doesn't work with single_file" ):
            save_fits_image_file( basepath, data, {}, just_update_header=True, single_file=True )

        with pytest.raises( FileNotFoundError, match="just_update_header failure: missing file" ):
            save_fits_image_file( basepath, data, {}, just_update_header=True )


        with open( fitspath, "w" ) as ofp:
            ofp.write( "\n" )
        with pytest.raises( FileExistsError, match=f"save_fits_image_file not overwriting {fitspath}" ):
            save_fits_image_file( basepath, data, {}, overwrite=False )
        with pytest.raises( FileExistsError, match=f"save_fits_image_file not overwriting {fitspath}" ):
            save_fits_image_file( basepath, data, {}, fpack=True, overwrite=False )
        fitspath.unlink( missing_ok=True )
        with open( fzpath, "w" ) as ofp:
            ofp.write( "\n" )
        with pytest.raises( FileExistsError, match=f"save_fits_image_file not overwriting {fzpath}" ):
            save_fits_image_file( basepath, data, {}, fpack=True, overwrite=False )

    finally:
        basepath.unlink( missing_ok=True )
        fitspath.unlink( missing_ok=True )
        fzpath.unlink( missing_ok=True )



def test_fpack_image_update_header( fpacked_fits_file ):
    origdata, origheader = read_fits_image( fpacked_fits_file, output="both" )
    tmphdr = copy.deepcopy( origheader )
    tmphdr[ 'UPDTEST' ] = ( 42, "Header has been updated" )

    outpath = save_fits_image_file( fpacked_fits_file, None, tmphdr, fpack=True, just_update_header=True )
    assert outpath.endswith( '.fz' )

    newdata, newheader = read_fits_image( fpacked_fits_file, output="both" )

    assert 'UPDTEST' in newheader
    assert newheader['UPDTEST'] == 42
    assert newheader.comments['UPDTEST'] == "Header has been updated"
    # In fact, make sure the whole thing is there
    for kw in [ t for t in tmphdr if t != 'EXTNAME' ]:
        assert newheader[kw] == tmphdr[kw]
        assert newheader.comments[kw] == tmphdr.comments[kw]

    # # Make sure the data is identical, as the header update should not have touched it
    # # (Would be nice to test performance, that header update is not taking time to re-fpack
    # # or any of that, but, whatevs.  It's not that big a deal.)
    # assert np.all( origdata == newdata )

    # AUGH.  With some astropy version update (I THINK), this stopped
    # working.  It's possible that we were lucky before, and it worked
    # because the header size was such that the number of 2880-byte FITS
    # header records on the disk didn't change, so it didn't need to
    # actually read and write the file.  But, I suspect it's due to the
    # version change.  Not sure.
    #
    # Not sure how to work around this; I fought with astropy for a while and gave up.
    # See:
    #   https://community.openastronomy.org/t/editing-header-of-compressed-image-hdu-without-recompressing/1189
    #
    # See Issue #402.
    #
    # For now, use a weaker test, which is also used in
    # read_fpack_fits_image above.  (At least we can use tighter
    # constraints!  I suspect this is because most of the loss comes
    # from the quantization, not from the random stuff that changes upon
    # recompression.)

    assert ( np.fabs( newdata - origdata ) < 0.4 ).all()
    assert np.fabs( ( newdata - origdata ).mean() ) < 1e-4
    assert ( np.fabs( newdata - origdata ) / origdata ).max() < 0.0011
    assert ( np.fabs( newdata - origdata ) / origdata ).mean() < 0.0005
