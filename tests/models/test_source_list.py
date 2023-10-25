import pytest
import pathlib
import numpy as np

import sqlalchemy as sa

import astropy.table
import astropy.io.fits

from models.base import SmartSession, FileOnDiskMixin
from models.source_list import SourceList

from tests.conftest import ImageCleanup

# @pytest.mark.skip( reason="slow" )
def test_source_list_bitflag(sources, demo_image, provenance_base, provenance_extra):
    filenames = []
    with SmartSession() as session:
        sources.provenance = provenance_extra
        demo_image.provenance = provenance_base
        _ = ImageCleanup.save_image(demo_image, archive=True)

        filenames.append(demo_image.get_fullpath(as_list=True)[0])
        sources.save(no_archive=False)
        filenames.append(sources.get_fullpath(as_list=True)[0])
        sources = sources.recursive_merge( session )
        session.add(sources)
        session.commit()

        assert demo_image.id is not None  # was added along with sources
        assert sources.id is not None
        assert sources.image_id == demo_image.id

        # all these data products should have bitflag zero
        assert sources.bitflag == 0
        assert sources.badness == ''

        # try to find this using the bitflag hybrid property
        sources2 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sources.id in [s.id for s in sources2]
        sources2x = session.scalars(sa.select(SourceList).where(SourceList.bitflag > 0)).all()
        assert sources.id not in [s.id for s in sources2x]

        # now add a badness to the image and exposure
        demo_image.badness = 'Saturation'
        demo_image.exposure.badness = 'Banding'

        session.add(demo_image)
        session.commit()

        assert demo_image.bitflag == 2 ** 1 + 2 ** 3
        assert demo_image.badness == 'Banding, Saturation'

        assert sources.bitflag == 2 ** 1 + 2 ** 3
        assert sources.badness == 'Banding, Saturation'

        # try to find this using the bitflag hybrid property
        sources3 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 1 + 2 ** 3)).all()
        assert sources.id in [s.id for s in sources3]
        sources3x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sources.id not in [s.id for s in sources3x]

        # now add some badness to the source list itself

        # cannot add an image badness to a source list
        with pytest.raises(ValueError, match='Keyword "Banding" not recognized in dictionary'):
            sources.badness = 'Banding'

        # add badness that works with source lists (e.g., cross-match failures)
        sources.badness = 'few sources'
        session.add(sources)
        session.commit()

        assert sources.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 43
        assert sources.badness == 'Banding, Saturation, Few Sources'

        # try to find this using the bitflag hybrid property
        sources4 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 43)).all()
        assert sources.id in [s.id for s in sources4]
        sources4x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sources.id not in [s.id for s in sources4x]

        # removing the badness from the exposure is updated directly to the source list
        demo_image.exposure.bitflag = 0
        session.add(demo_image)
        session.commit()

        assert demo_image.badness == 'Saturation'
        assert sources.badness == 'Saturation, Few Sources'

        # check the database queries still work
        sources5 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 3 + 2 ** 43)).all()
        assert sources.id in [s.id for s in sources5]
        sources5x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sources.id not in [s.id for s in sources5x]

def test_read_sextractor( example_source_list ):
    filepath, fullpath = example_source_list

    # Make sure things go haywire when we try to load data with inconsistent
    # num_sources or aper_rads
    with pytest.raises( ValueError, match='self.num_sources=10 but the sextractor file had' ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=10 )
        data = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match sextractor file" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=1069, aper_rads=[1., 2.] )
        data = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match sextractor file" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=1069,
                              aper_rads=[ 4.04170132 , 15.01375008 , 32. ] )
        data = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match sextractor file" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=1069,
                              aper_rads=[ 4.04170132 ] )
        data = sources.data

    # Make sure those fields get properly auto-set
    sources = SourceList( format='sextrfits', filepath=filepath )
    data = sources.data
    assert sources.num_sources == 1069
    assert sources.aper_rads == pytest.approx( [ 4.04, 15.01 ], abs=0.01 )

    # Make sure we can read the file with the right things in place in those fields
    sources = SourceList( format='sextrfits', filepath=filepath, num_sources=1069,
                          aper_rads=[ 4.04170132, 15.01375008 ] )
    assert len(sources.data) == 1069
    assert sources.num_sources == 1069
    assert sources.aper_rads == pytest.approx( [ 4.04, 15.01 ], abs=0.01 )
    assert sources.x[0] == pytest.approx( 1501.45, abs=0.01 )
    assert sources.y[0] == pytest.approx( 60.43, abs=0.01 )
    assert sources.x[50] == pytest.approx( 2890.70, abs=0.01 )
    assert sources.y[50] == pytest.approx( 113.02, abs=0.01 )
    assert sources.ra[0] == pytest.approx( 153.62379146881202, abs=0.1/3600. )
    assert sources.dec[0] == pytest.approx( 39.50160906900172, abs=0.1/3600. )
    assert sources.ra[50] == pytest.approx( 153.11931811345397, abs=0.1/3600. )
    assert sources.dec[50] == pytest.approx( 39.46629724049348, abs=0.1/3600. )
    assert sources.apfluxadu()[0][0] == pytest.approx( 6490.0864, rel=1e-5 )
    assert sources.apfluxadu()[0][50] == pytest.approx( 269.38647, rel=1e-5 )
    assert sources.apfluxadu(apnum=0)[0][0] == pytest.approx( 6490.0864, rel=1e-5 )
    assert sources.apfluxadu(apnum=0)[0][50] == pytest.approx( 269.38647, rel=1e-5 )
    assert sources.apfluxadu(ap=4.04)[0][0] == pytest.approx( 6490.0864, rel=1e-5 )
    assert sources.apfluxadu(ap=4.04)[0][50] == pytest.approx( 269.38647, rel=1e-5 )
    assert sources.apfluxadu(apnum=1)[0][0] == pytest.approx( 20104.086, rel=1e-5 )
    assert sources.apfluxadu(apnum=1)[0][50] == pytest.approx( 162.24628, rel=1e-5 )
    assert sources.apfluxadu(ap=15.01)[0][0] == pytest.approx( 20104.086, rel=1e-5 )
    assert sources.apfluxadu(ap=15.01)[0][50] == pytest.approx( 162.24628, rel=1e-5 )
    assert sources.apfluxadu()[1][0] == pytest.approx( 51.80475, rel=1e-5 )
    assert sources.apfluxadu()[1][50] == pytest.approx( 40.86762, rel=1e-5 )
    assert sources.apfluxadu(apnum=0)[1][0] == pytest.approx( 51.80475, rel=1e-5 )
    assert sources.apfluxadu(apnum=0)[1][50] == pytest.approx( 40.86762, rel=1e-5 )
    assert sources.apfluxadu(ap=4.04)[1][0] == pytest.approx( 51.80475, rel=1e-5 )
    assert sources.apfluxadu(ap=4.04)[1][50] == pytest.approx( 40.86762, rel=1e-5 )
    assert sources.apfluxadu(apnum=1)[1][0] == pytest.approx( 160.91551, rel=1e-5 )
    assert sources.apfluxadu(apnum=1)[1][50] == pytest.approx( 150.53914, rel=1e-5 )
    assert sources.apfluxadu(ap=15.01)[1][0] == pytest.approx( 160.91551, rel=1e-5 )
    assert sources.apfluxadu(ap=15.01)[1][50] == pytest.approx( 150.53914, rel=1e-5 )

    # Check some apfluxadu failure modes
    with pytest.raises( ValueError, match="Aperture radius number 2 doesn't exist." ):
        _ = sources.apfluxadu( apnum = 2 )
    with pytest.raises( ValueError, match="Can't find an aperture of .* pixels" ):
        _ = sources.apfluxadu( ap=6. )

    # Poke directly into the data array as well

    assert sources.data['X_IMAGE'][0] == pytest.approx( 1501.45, abs=0.01 )
    assert sources.data['Y_IMAGE'][0] == pytest.approx( 60.43, abs=0.01 )
    assert sources.data['X_WORLD'][0] == pytest.approx( 153.62379146881202, abs=0.1/3600. )
    assert sources.data['Y_WORLD'][0] == pytest.approx( 39.50160906900172, abs=0.1/3600. )
    assert sources.data['XWIN_IMAGE'][0] == pytest.approx( 1501.39, abs=0.01 )
    assert sources.data['YWIN_IMAGE'][0] == pytest.approx( 60.51, abs=0.01 )
    assert sources.data['FLUX_APER'][0] == pytest.approx( np.array( [ 6490.0864, 20104.086 ] ), rel=1e-5 )
    assert sources.data['FLUXERR_APER'][0] == pytest.approx( np.array( [ 51.80475, 160.91551] ), rel=1e-5 )
    assert sources.data['X_IMAGE'][50] == pytest.approx( 2890.70, abs=0.01 )
    assert sources.data['Y_IMAGE'][50] == pytest.approx( 113.02, abs=0.01 )
    assert sources.data['X_WORLD'][50] == pytest.approx( 153.11931811345397, abs=0.1/3600. )
    assert sources.data['Y_WORLD'][50] == pytest.approx( 39.46629724049348, abs=0.1/3600. )
    assert sources.data['XWIN_IMAGE'][50] == pytest.approx( 2890.56, abs=0.01 )
    assert sources.data['YWIN_IMAGE'][50] == pytest.approx( 113.03, abs=0.01 )
    assert sources.data['FLUX_APER'][50] == pytest.approx( np.array( [269.38647, 162.24628] ), rel=1e-5 )
    assert sources.data['FLUXERR_APER'][50] == pytest.approx( np.array( [ 40.86762, 150.53914] ), rel=1e-5 )


def test_write_sextractor():
    fname = ''.join( np.random.choice( list('abcdefghijklmnopqrstuvwxyz'), 16 ) )
    sources = SourceList( format='sextrfits', filepath=f"{fname}_sources.fits" )
    assert sources.aper_rads is None
    if pathlib.Path( sources.get_fullpath() ).is_file():
        raise RuntimeError( f"{fname}_sources.fits exists when it shouldn't" )
    sources.info = astropy.io.fits.Header( [ ( 'HELLO', 'WORLD', 'Comment' ),
                                             ( 'ANSWER', 42 ) ] )
    sources.data = np.array( [ ( 100.7, 100.2, 42. ),
                               ( 238.1, 1.9, 23. ) ],
                             dtype=[ ('X_IMAGE', '<f4'),
                                     ('Y_IMAGE', '<f4'),
                                     ('FLUX_AUTO', '<f4') ]
                            )
    try:
        sources.save()
        tab = astropy.table.Table.read( sources.get_fullpath(), hdu=2 )
        assert len(tab) == 2
        assert tab['X_IMAGE'].tolist() == pytest.approx( [101.7, 239.1 ], abs=0.01 )
        assert tab['Y_IMAGE'].tolist() == pytest.approx( [101.2, 2.9 ], abs=0.01 )
        assert tab['FLUX_AUTO'].tolist() == pytest.approx( [ 42., 23. ], rel=1e-5 )
        with astropy.io.fits.open( sources.get_fullpath() ) as hdul:
            assert len(hdul) == 3
            assert isinstance( hdul[0], astropy.io.fits.PrimaryHDU )
            assert isinstance( hdul[1], astropy.io.fits.BinTableHDU )
            assert isinstance( hdul[2], astropy.io.fits.BinTableHDU )
            hdr = astropy.io.fits.Header.fromstring( hdul[1].data.tobytes().decode('latin-1') )
            assert hdr['HELLO'] == 'WORLD'
            assert hdr['ANSWER'] == 42
            assert hdr.cards[0].comment == 'Comment'
    finally:
        pathlib.Path( sources.get_fullpath() ).unlink( missing_ok=True )

