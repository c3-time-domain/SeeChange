import pytest

from models.zero_point import ZeroPoint

def test_get_aper_cor():
    zp = ZeroPoint()
    with pytest.raises( ValueError, match="No aperture corrections tabulated." ):
        _ = zp.get_aper_cor( 1.0 )

    zp = ZeroPoint( aper_cor_radii=[ 1.0097234, 2.4968394 ],
                    aper_cors=[-0.25, -0.125] )
    assert zp.get_aper_cor( 1.0 ) == -0.25
    assert zp.get_aper_cor( 2.5 ) == -0.125

    with pytest.raises( ValueError, match="No aperture correction tabulated for.*within 0\.01 pixels of" ):
        _ = zp.get_aper_cor( 1.02 )
    with pytest.raises( ValueError, match="No aperture correction tabulated for.*within 0\.01 pixels of" ):
        _ = zp.get_aper_cor( 0.99 )

    
                    
    
    
