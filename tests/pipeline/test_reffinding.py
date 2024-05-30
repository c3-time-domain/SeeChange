import pytest
import math
from models.image import Image
from pipeline.data_store import DataStore

def test_data_store_overlap_frac():
    dra = 0.75
    ddec = 0.375
    radec1 = [ ( 10., -3. ), ( 10., -45. ) , (10., -80.) ]

    # TODO : add tests where things aren't perfectly square
    for ra, dec in radec1:
        cd = math.cos( dec * math.pi / 180. )
        i1 = Image( ra = ra, dec = dec,
                    ra_corner_00 = ra - dra/2. / cd,
                    ra_corner_01 = ra - dra/2. / cd,
                    ra_corner_10 = ra + dra/2. / cd,
                    ra_corner_11 = ra + dra/2. / cd,
                    dec_corner_00 = dec - ddec/2.,
                    dec_corner_10 = dec - ddec/2.,
                    dec_corner_01 = dec + ddec/2.,
                    dec_corner_11 = dec + ddec/2. )
        for frac, offx, offy in [ ( 1.  ,  0. ,  0.  ),
                                  ( 0.5 ,  0.5,  0.  ),
                                  ( 0.5 , -0.5,  0.  ),
                                  ( 0.5 ,  0. ,  0.5 ),
                                  ( 0.5 ,  0. , -0.5 ),
                                  ( 0.25,  0.5,  0.5 ),
                                  ( 0.25, -0.5,  0.5 ),
                                  ( 0.25,  0.5, -0.5 ),
                                  ( 0.25, -0.5, -0.5 ),
                                  ( 0.,    1.,   0.  ),
                                  ( 0.,   -1.,   0.  ),
                                  ( 0.,    1.,   0.  ),
                                  ( 0.,   -1.,   0.  ),
                                  ( 0.,   -1.,  -1.  ),
                                  ( 0.,    1.,  -1.  ) ]:
            ra2 = ra + offx * dra / cd
            dec2 = dec + offy * ddec
            i2 = Image( ra = ra2, dec = dec2,
                        ra_corner_00 = ra2 - dra/2. / cd,
                        ra_corner_01 = ra2 - dra/2. / cd,
                        ra_corner_10 = ra2 + dra/2. / cd,
                        ra_corner_11 = ra2 + dra/2. / cd,
                        dec_corner_00 = dec2 - ddec/2.,
                        dec_corner_10 = dec2 - ddec/2.,
                        dec_corner_01 = dec2 + ddec/2.,
                        dec_corner_11 = dec2 + ddec/2. )
            assert DataStore._overlap_frac( i1, i2 ) == pytest.approx( frac, abs=0.01 )
                                    
    
