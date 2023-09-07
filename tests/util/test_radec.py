import pytest

from util import radec

def test_parse_sexigesimal_degrees():
    deg = radec.parse_sexigesimal_degrees( '15:32:25' )
    assert deg == pytest.approx( 15.54027778, abs=1e-8 )
    deg = radec.parse_sexigesimal_degrees( '-15:32:25' )
    assert deg == pytest.approx( -15.54027778, abs=1e-8 )
    deg = radec.parse_sexigesimal_degrees( '  -5: 3: 2   ' )
    assert deg == pytest.approx( -5.05055556, abs=1e-8 )

    # Test the postive flag
    deg = radec.parse_sexigesimal_degrees( '-15:32:25', positive=True )
    assert deg == pytest.approx( 344.45972222, abs=1e-8 )

    # Make sure that the "negative dec bug" isn't present
    deg = radec.parse_sexigesimal_degrees( "-00:30:00" )
    assert deg == -0.5

    # Make sure we can parse hours
    deg = radec.parse_sexigesimal_degrees( ' 12:30:36', hours=True )
    assert deg == 187.65
    deg = radec.parse_sexigesimal_degrees( '-00:30:00', hours=True )
    assert deg == 352.5
    deg = radec.parse_sexigesimal_degrees( '-00:30:00', hours=True, positive=False )
    assert deg == -7.5

def test_radec_to_gal_and_eclip():
    gal_l, gal_b, ecl_lon, ecl_lat = radec.radec_to_gal_and_eclip( 210.53, -32.3 )
    assert gal_l == pytest.approx( 319.86357776, abs=1e-8 )
    assert gal_b == pytest.approx( 28.23720981, abs=1e-8 )
    assert ecl_lon == pytest.approx( 219.79120568, abs=1e-8 )
    assert ecl_lat == pytest.approx( -18.63024354, abs=1e-8 )
