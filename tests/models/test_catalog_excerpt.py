import pytest


def test_create_from_file( catexp ):
    assert catexp.origin == 'gaia_dr3'
    assert catexp.format == 'fitsldac'
    assert catexp.filters == [ 'G', 'BP', 'RP' ]
    assert catexp.num_items == 59
    assert len( catexp.data ) == catexp.num_items
    assert catexp.minmag == pytest.approx( 17.04, abs=0.02 )
    assert catexp.maxmag == pytest.approx( 18.98, abs=0.02 )
    assert catexp.ra_corner_00 == pytest.approx( 150.92838, abs=1./3600. )
    assert catexp.ra_corner_10 == pytest.approx( 151.26913, abs=1./3600. )
    assert catexp.dec_corner_00 == pytest.approx( 1.74388, abs=1./3600. )
    assert catexp.dec_corner_01 == pytest.approx( 1.92036, abs=1./3600. )
    assert catexp.ra_corner_01 == catexp.ra_corner_00
    assert catexp.ra_corner_11 == catexp.ra_corner_10
    assert catexp.dec_corner_10 == catexp.dec_corner_00
    assert catexp.dec_corner_11 == catexp.dec_corner_01

    assert catexp.data['MAG_G'][0] == pytest.approx( 17.73, abs=0.01 )
    assert catexp.data['MAG_G'][21] == pytest.approx( 17.48, abs=0.01 )
    assert catexp.data['MAG_RP'][0] == pytest.approx( 16.85, abs=0.01 )
    assert catexp.data['MAG_RP'][21] == pytest.approx( 16.59, abs=0.01 )
    assert catexp.data['MAG_BP'][0] == pytest.approx( 18.57, abs=0.01 )
    assert catexp.data['MAG_BP'][21] == pytest.approx( 18.30, abs=0.01 )


def test_object_ras_decs( catexp ):
    assert len( catexp.object_ras) == catexp.num_items
    assert len( catexp.object_decs) == catexp.num_items
    assert ( catexp.object_ras == catexp.data['X_WORLD'] ).all()
    assert ( catexp.object_decs == catexp.data['Y_WORLD'] ).all()
    assert catexp.object_ras[0] == pytest.approx( 150.978073, abs=0.1/3600. )
    assert catexp.object_decs[0] == pytest.approx( 1.751641, abs=0.1/3600. )
    assert catexp.object_ras[21] == pytest.approx( 151.066776, abs=0.1/3600. )
    assert catexp.object_decs[21] == pytest.approx( 1.793652, abs=0.1/3600. )


