import pytest
import pathlib
import random

from util.util import listify, ensure_file_does_not_exist

# TODO : tests of most of the stuff in util...!  (Issue #384)


def test_listify():
    assert listify( None ) is None
    assert listify( ( None, ) ) == [ None ]
    assert listify( "test" ) == [ "test" ]
    assert listify( 1 ) == [ 1  ]
    assert listify( [ "a", "b", "c" ] ) == [ "a", "b", "c" ]
    assert listify( [ 1, 2, 3 ] ) == [ 1, 2, 3 ]
    assert listify( ( 1, 2, 3 ) ) == [ 1, 2, 3 ]

    # Make sure require_string works right
    assert listify( "test", require_string=True ) == [ "test" ]
    assert listify( ( "a", "b", "c" ), require_string=True ) == [ "a", "b", "c" ]

    with pytest.raises( TypeError ):
        _ = listify( 1, require_string=True )
    with pytest.raises( TypeError ):
        _ = listify( [ 1, 2, 3], require_string=True )
    with pytest.raises( TypeError ):
        _ = listify( [ "a", 1 ], require_string=True )


def test_ensure_file_does_not_exist():
    fname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    fpath = pathlib.Path( fname )
    assert not fpath.exists()

    try:
        ensure_file_does_not_exist( fname )
        ensure_file_does_not_exist( fpath )

        fpath.mkdir()
        with pytest.raises( FileExistsError, match='.*exists but is not a regular file' ):
            ensure_file_does_not_exist( fname )
        with pytest.raises( FileExistsError, match='.*exists but is not a regular file' ):
            ensure_file_does_not_exist( fpath )
        fpath.rmdir()

        with open( fpath, "w" ) as ofp:
            ofp.write( "Hello, world\n" )

        with pytest.raises( FileExistsError, match='.*exists and delete is False' ):
            ensure_file_does_not_exist( fname )
        with pytest.raises( FileExistsError, match='.*exists and delete is False' ):
            ensure_file_does_not_exist( fpath )

        ensure_file_does_not_exist( fname, delete=True )
        assert not fpath.exists()
        with open( fpath, "w" ) as ofp:
            ofp.write( "Hello, world\n" )
        ensure_file_does_not_exist( fpath, delete=True )
        assert not fpath.exists()

    finally:
        if fpath.exists():
            if fpath.is_file():
                fpath.unlink()
            else:
                fpath.rmdir()
