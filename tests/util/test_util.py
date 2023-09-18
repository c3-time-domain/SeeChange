import pytest

from util.util import listify

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
        l = listify( 1, require_string=True )
    with pytest.raises( TypeError ):
        l = listify( [ 1, 2, 3], require_string=True )
    with pytest.raises( TypeError ):
        l = listify( [ "a", 1 ], require_string=True )
        
