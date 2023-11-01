import pytest
import pathlib

import numpy as np

from astropy.io import fits
from astropy.table import Table

import util.ldac

def test_expected_failures():
    with pytest.raises( AttributeError, match="object has no attribute 'data'" ):
        util.ldac.convert_hdu_to_ldac( "string" )
    with pytest.raises( TypeError, match="imghdr must be.*not a <class 'int'>" ):
        util.ldac.convert_hdu_to_ldac( fits.BinTableHDU( None ), 42 )
    with pytest.raises( TypeError, match="imghdr must be.*not a <class 'int'>" ):
        util.ldac.convert_table_to_ldac( Table(), 42 )
    with pytest.raises( TypeError, match="imghdr must be.*not a <class 'int'>" ):
        util.ldac.save_table_as_ldac( Table(), "deleteme.fits", 42 )
    with pytest.raises( FileNotFoundError ):
        util.ldac.get_table_from_ldac( "this_file_does_not_exist" )

def test_read_ldac( example_source_list_filename ):
    fullpath = example_source_list_filename

    tbl1, tbl2 = util.ldac.get_table_from_ldac( fullpath )
    assert isinstance( tbl1, fits.BinTableHDU )
    assert isinstance( tbl2, Table )

    hdr, tbl = util.ldac.get_table_from_ldac( fullpath, imghdr_as_header=True )
    assert isinstance( hdr, fits.Header )
    assert isinstance( tbl2, Table )
    # Spot check
    assert hdr['SEXNNWF'] == 'default.nnw'
    assert hdr['SEXAPED1'] == pytest.approx( 2, abs=0.01 )
    assert hdr['SEXAPED2'] == pytest.approx( 5, abs=0.01 )
    assert hdr['SEXAPED3'] == 0.
    assert hdr['SEXAPED4'] == 0.
    assert len(tbl2) == 112
    assert tbl['X_IMAGE'].min() == pytest.approx( 4.97, abs=0.01 )
    assert tbl['X_IMAGE'].max() == pytest.approx( 1023.65, abs=0.01 )
    assert tbl['Y_IMAGE'].min() == pytest.approx( 1.49, abs=0.01 )
    assert tbl['Y_IMAGE'].max() == pytest.approx( 993.23, abs=0.01 )
    aper0 = tbl['FLUX_APER'][:, 0]
    aper1 = tbl['FLUX_APER'][:, 1]
    assert aper0.min() == pytest.approx( 56.979435, rel=1e-5 )
    assert aper0.max() == pytest.approx( 98965.32, rel=1e-5 )
    assert aper0.mean() == pytest.approx( 2991.1528, rel=1e-5 )
    assert aper0.std() == pytest.approx( 12142.842, rel=1e-4 )
    assert aper1.min() == pytest.approx( 241.75589, rel=1e-5 )
    assert aper1.max() == pytest.approx( 335792.22, rel=1e-5 )
    assert aper1.mean() == pytest.approx( 9863.518, rel=1e-5 )
    assert aper1.std() == pytest.approx( 39851.023, rel=1e-4 )

def test_save_ldac():
    fname = pathlib.Path( ''.join( np.random.choice( list('abcdefghijklmnopqrstuvwxyz'), 16 ) ) + '.fits' )
    tab = Table( { 'a': np.array(  [ 1, 2, 3 ], dtype=np.float32 ),
                   'b': np.array( [4., 5., 6.] ) } )
    hdr = fits.Header( [ ( 'HELLO', 'WORLD', 'Comment' ),
                         ( 'ANSWER', 42 ) ] )
    try:
        util.ldac.save_table_as_ldac( tab, fname, hdr )

        with fits.open( fname ) as f:
            assert isinstance( f[0], fits.PrimaryHDU )
            assert isinstance( f[1], fits.BinTableHDU )
            assert isinstance( f[2], fits.BinTableHDU )
            assert f[1].header['TTYPE1'] == 'Field Header Card'
            assert f[1].header['TDIM1'] == '(80, 3)'
            hdr = fits.Header.fromstring( f[1].data.tobytes().decode('latin-1') )
            assert hdr['HELLO'] == 'WORLD'
            assert hdr['ANSWER'] == 42
            assert hdr.cards[0].comment == 'Comment'
            assert f[2].header['TTYPE1'].strip() == 'a'
            assert f[2].header['TFORM1'].strip() == 'E'
            assert f[2].header['TTYPE2'].strip() == 'b'
            assert f[2].header['TFORM2'].strip() == 'D'
            assert f[2].header['EXTNAME'] == 'LDAC_OBJECTS'
            assert f[2].data[1][0] == 2.0
            assert f[2].data[1][1] == 5.0

        hdr, tbl = util.ldac.get_table_from_ldac( fname, imghdr_as_header=True )
        assert hdr['HELLO'] == 'WORLD'
        assert hdr['ANSWER'] == 42
        assert hdr.cards[0].comment == 'Comment'
        assert len(tbl) == 3
        assert ( tbl['a'].value == np.array( [1., 2., 3.] ) ).all()
        assert ( tbl['b'].value == np.array( [4., 5., 6.] ) ).all()

    finally:
        fname.unlink( missing_ok=True )
