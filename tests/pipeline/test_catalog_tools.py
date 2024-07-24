import os
import pathlib
import pytest

from util.exceptions import CatalogNotFoundError
from util import ldac
from pipeline.catalog_tools import download_gaia_dr3, fetch_gaia_dr3_excerpt
from models.base import FourCorners
from models.image import Image

def test_download_gaia_dr3(data_dir):
    firstfilepath = None
    secondfilepath = None
    try:
        catexp, firstfilepath, dbfile = download_gaia_dr3( 150.9427, 151.2425, 1.75582, 1.90649,
                                                                     padding=0.1, minmag=18., maxmag=22. )
        assert firstfilepath == os.path.join(data_dir, 'gaia_dr3_excerpt/94/Gaia_DR3_151.0926_1.8312_18.0_22.0.fits')
        assert dbfile == firstfilepath
        assert catexp.num_items == 178
        assert catexp.format == 'fitsldac'
        assert catexp.origin == 'gaia_dr3'
        assert catexp.minmag == 18.
        assert catexp.maxmag == 22.
        assert ( catexp.dec_corner_11 - catexp.dec_corner_00 ) == pytest.approx( 1.2 * (1.90649-1.75582), abs=1e-4 )
        catexp, secondfilepath, dbfile = download_gaia_dr3( 150.9427, 151.2425, 1.75582, 1.90649,
                                                                      padding=0.1, minmag=17., maxmag=19. )
        assert secondfilepath == os.path.join(data_dir, 'gaia_dr3_excerpt/94/Gaia_DR3_151.0926_1.8312_17.0_19.0.fits')
        assert dbfile == secondfilepath
        assert catexp.num_items == 59
        assert catexp.minmag == 17.
        assert catexp.maxmag == 19.

        hdr, tbl = ldac.get_table_from_ldac( secondfilepath, imghdr_as_header=True )
        for col in [ 'X_WORLD', 'Y_WORLD', 'ERRA_WORLD', 'ERRB_WORLD', 'PM', 'PMRA', 'PMDEC',
                     'MAG_G', 'MAGERR_G', 'MAG_BP', 'MAGERR_BP', 'MAG_RP', 'MAGERR_RP', 'STARPROB',
                     'OBSDATE', 'FLAGS' ]:
            assert col in tbl.columns
        assert ( tbl['STARPROB'] > 0.95 ).sum() == 59

    finally:
        if firstfilepath is not None:
            pathlib.Path( firstfilepath ).unlink( missing_ok=True )
        if secondfilepath is not None:
            pathlib.Path( secondfilepath ).unlink( missing_ok=True )


def test_gaia_dr3_excerpt_failures( ztf_datastore_uncommitted, ztf_gaia_dr3_excerpt ):
    ds = ztf_datastore_uncommitted
    try:
        # Make sure it fails if we give it a ridiculous max mag
        with pytest.raises( CatalogNotFoundError, match="Failed to fetch Gaia DR3 stars at" ):
            catexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=5.0, magrange=4, minstars=50 )

        # ...but make sure it succeeds if we also give it a reasonable max mag
        catexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=[5.0, 20.0], magrange=4.0, minstars=50 )
        assert catexp.id == ztf_gaia_dr3_excerpt.id

        # Make sure it fails if we ask for too many stars
        with pytest.raises( CatalogNotFoundError, match="Failed to fetch Gaia DR3 stars at" ):
            catexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=[20.0], magrange=4.0, minstars=50000 )

        # Make sure it fails if mag range is too small
        with pytest.raises( CatalogNotFoundError, match="Failed to fetch Gaia DR3 stars at" ):
            catexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=[20.0], magrange=0.01, minstars=50 )

    finally:
        catexp.delete_from_disk_and_database()


def test_gaia_dr3_excerpt( ztf_datastore_uncommitted, ztf_gaia_dr3_excerpt ):
    catexp = ztf_gaia_dr3_excerpt
    ds = ztf_datastore_uncommitted

    assert catexp.num_items == 172
    assert catexp.num_items == len( catexp.data )
    assert catexp.filepath == 'gaia_dr3_excerpt/30/Gaia_DR3_153.6459_39.0937_16.0_20.0.fits'
    assert pathlib.Path( catexp.get_fullpath() ).is_file()
    assert catexp.object_ras.min() == pytest.approx( 153.413563, abs=0.1/3600. )
    assert catexp.object_ras.max() == pytest.approx( 153.877110, abs=0.1/3600. )
    assert catexp.object_decs.min() == pytest.approx( 38.914110, abs=0.1/3600. )
    assert catexp.object_decs.max() == pytest.approx( 39.274596, abs=0.1/3600. )
    assert ( catexp.data['X_WORLD'] == catexp.object_ras ).all()
    assert ( catexp.data['Y_WORLD'] == catexp.object_decs ).all()
    assert catexp.data['MAG_G'].min() == pytest.approx( 16.076, abs=0.001 )
    assert catexp.data['MAG_G'].max() == pytest.approx( 19.994, abs=0.001 )
    assert catexp.data['MAGERR_G'].min() == pytest.approx( 0.0004, abs=0.0001 )
    assert catexp.data['MAGERR_G'].max() == pytest.approx( 0.018, abs=0.001 )

    # Test reading of cache
    newcatexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=[20.0], magrange=4.0, minstars=50, onlycached=True )
    assert newcatexp.id == catexp.id

    # Make sure we can't read the cache for something that doesn't exist
    with pytest.raises( CatalogNotFoundError, match='Failed to fetch Gaia DR3 stars' ):
        newcatexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=[20.5], magrange=4.0, minstars=50, onlycached=True )

def test_gaia_dr3_excerpt_ra_span_zero():

    stars = None
    firstcat = None
    try:
        ra = 0.
        dec = 0.
        ras = [ 359.79, 359.81, 0.19, 0.21 ]
        decs = [ -0.19, 0.21, -0.21, 0.19 ]
        ras, decs = FourCorners.sort_radec( ras, decs )
        minra = min( ras )
        maxra = max( ras )
        mindec = min( decs )
        maxdec = max( decs )
        img = Image( ra=ra, dec=dec,
                     ra_corner_00=ras[0],
                     ra_corner_01=ras[1],
                     ra_corner_10=ras[2],
                     ra_corner_11=ras[3],
                     minra=minra,
                     maxra=maxra,
                     dec_corner_00=decs[0],
                     dec_corner_01=decs[1],
                     dec_corner_10=decs[2],
                     dec_corner_11=decs[3],
                     mindec=mindec,
                     maxdec=maxdec )
        stars = fetch_gaia_dr3_excerpt( img )
        assert stars.num_items == 776
        lt0ras = stars.object_ras[ stars.object_ras > 180. ]
        gt0ras = stars.object_ras[ stars.object_ras < 180. ]
        assert len(lt0ras) == 399
        assert len(gt0ras) == 377
        # The numbers below take into account the 5% padding on each side that fetch_gaia_dr3_excerpt uses
        assert min(lt0ras) == pytest.approx( 359.748, abs=0.003 )
        assert max(lt0ras) == pytest.approx( 360., abs=0.001 )
        assert min(gt0ras) == pytest.approx( 0., abs=0.001 )
        assert max(gt0ras) == pytest.approx( 0.252, abs=0.003 )
        assert min(stars.object_decs) == pytest.approx( -0.252, abs=0.003 )
        assert max(stars.object_decs) == pytest.approx( 0.252, abs=0.003 )

        # Do it again, make sure we get the same thing
        firstcat = stars
        stars = fetch_gaia_dr3_excerpt( img )
        assert stars.id == firstcat.id

    finally:
        # Clean up
        if stars is not None:
            stars.delete_from_disk_and_database()
        if firstcat is not None:
            firstcat.delete_from_disk_and_database()
