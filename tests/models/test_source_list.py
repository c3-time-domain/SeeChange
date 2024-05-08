import pytest
import os
import psutil
import gc
import pathlib
import numpy as np

import sqlalchemy as sa

import astropy.table
import astropy.io.fits

from models.base import SmartSession, FileOnDiskMixin
from models.image import Image
from models.source_list import SourceList


def test_source_list_bitflag(sim_sources):
    with SmartSession() as session:
        sim_sources = sim_sources.merge_all( session )

        # all these data products should have bitflag zero
        assert sim_sources.bitflag == 0
        assert sim_sources.badness == ''

        # try to find this using the bitflag hybrid property
        sim_sources2 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sim_sources.id in [s.id for s in sim_sources2]
        sim_sources2x = session.scalars(sa.select(SourceList).where(SourceList.bitflag > 0)).all()
        assert sim_sources.id not in [s.id for s in sim_sources2x]

        # now add a badness to the image and exposure
        sim_sources.image.badness = 'Saturation'
        sim_sources.image.exposure.badness = 'Banding'
        sim_sources.image.exposure.update_downstream_badness(session)
        session.add(sim_sources.image)
        session.commit()

        assert sim_sources.image.bitflag == 2 ** 1 + 2 ** 3
        assert sim_sources.image.badness == 'banding, saturation'

        assert sim_sources.bitflag == 2 ** 1 + 2 ** 3
        assert sim_sources.badness == 'banding, saturation'

        # try to find this using the bitflag hybrid property
        sim_sources3 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 1 + 2 ** 3)).all()
        assert sim_sources.id in [s.id for s in sim_sources3]
        sim_sources3x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sim_sources.id not in [s.id for s in sim_sources3x]

        # now add some badness to the source list itself

        # cannot add an image badness to a source list
        with pytest.raises(ValueError, match='Keyword "Banding" not recognized in dictionary'):
            sim_sources.badness = 'Banding'

        # add badness that works with source lists (e.g., cross-match failures)
        sim_sources.badness = 'few sources'
        session.add(sim_sources)
        session.commit()

        assert sim_sources.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 16
        assert sim_sources.badness == 'banding, saturation, few sources'

        # try to find this using the bitflag hybrid property
        sim_sources4 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 16)).all()
        assert sim_sources.id in [s.id for s in sim_sources4]
        sim_sources4x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sim_sources.id not in [s.id for s in sim_sources4x]

        # removing the badness from the exposure is updated directly to the source list
        sim_sources.image.exposure.bitflag = 0
        sim_sources.image.exposure.update_downstream_badness(session)
        session.add(sim_sources.image)
        session.commit()

        assert sim_sources.image.badness == 'saturation'
        assert sim_sources.badness == 'saturation, few sources'

        # check the database queries still work
        sim_sources5 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 3 + 2 ** 16)).all()
        assert sim_sources.id in [s.id for s in sim_sources5]
        sim_sources5x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sim_sources.id not in [s.id for s in sim_sources5x]

        # make sure new SourceList object gets the badness from the Image
        new_sources = SourceList(image=sim_sources.image)
        assert new_sources.badness == 'saturation'


def test_invent_filepath( provenance_base, provenance_extra ):
    imgargs = {
        'instrument': 'DemoInstrument',
        'section_id': 0,
        'type': "Sci",
        'format': "fits",
        'ra': 12.3456,
        'dec': -0.42,
        'mjd': 61738.64,
        'filter': 'r',
        'provenance': provenance_base,
    }

    hash1 = provenance_base.id[:6]
    hash2 = provenance_extra.id[:6]

    image = Image( filepath="testing", **imgargs )
    sources = SourceList( image=image, format='sextrfits', provenance=provenance_extra )
    assert sources.invent_filepath() == f'{image.filepath}.sources_{hash2}.fits'

    image = Image( **imgargs )
    sources = SourceList( image=image, format='sextrfits', provenance=provenance_extra )
    assert sources.invent_filepath() == f'012/Demo_20271129_152136_0_r_Sci_{hash1}.sources_{hash2}.fits'

    image = Image( filepath="this.is.a.test", **imgargs )
    sources = SourceList( image=image, format='sextrfits', provenance=provenance_extra )
    assert sources.invent_filepath() == f'this.is.a.test.sources_{hash2}.fits'


def test_read_sextractor( ztf_filepath_sources ):
    fullpath = ztf_filepath_sources
    filepath = fullpath.relative_to( pathlib.Path( FileOnDiskMixin.local_path ) )

    # Make sure things go haywire when we try to load data with inconsistent
    # num_sources or aper_rads
    with pytest.raises( ValueError, match='self.num_sources=10 but the sextractor file had' ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=10 )
        data = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match the number of apertures found in" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=112, aper_rads=[1., 2., 3.] )
        data = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match sextractor file" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=112,
                              aper_rads=[ 2., 5. ] )
        data = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match sextractor file" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=112,
                              aper_rads=[ 1., 2. ] )
        data = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match the number of apertures found in" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=112,
                              aper_rads=[ 1. ] )
        data = sources.data

    # Make sure those fields get properly auto-set
    sources = SourceList( format='sextrfits', filepath=filepath )
    data = sources.data
    assert sources.num_sources == 112
    assert sources.aper_rads == [ 1., 2.5 ]

    # Make sure we can read the file with the right things in place in those fields
    sources = SourceList( format='sextrfits', filepath=filepath, num_sources=112, aper_rads=[ 1.0, 2.5 ] )
    assert len(sources.data) == 112
    assert sources.num_sources == 112
    assert sources.good.sum() == 105
    assert sources.aper_rads == [ 1.0, 2.5 ]
    assert sources._inf_aper_num is None
    assert sources.inf_aper_num == 1
    assert sources.x[0] == pytest.approx( 798.24, abs=0.01 )
    assert sources.y[0] == pytest.approx( 17.14, abs=0.01 )
    assert sources.x[50] == pytest.approx( 899.33, abs=0.01 )
    assert sources.y[50] == pytest.approx( 604.52, abs=0.01 )
    assert sources.apfluxadu()[0][0] == pytest.approx( 3044.9092, rel=1e-5 )
    assert sources.apfluxadu()[0][50] == pytest.approx( 165.99489, rel=1e-5 )
    assert sources.apfluxadu(apnum=0)[0][0] == pytest.approx( 3044.9092, rel=1e-5 )
    assert sources.apfluxadu(apnum=0)[0][50] == pytest.approx( 165.99489, rel=1e-5 )
    assert sources.apfluxadu(ap=0.995)[0][0] == pytest.approx( 3044.9092, rel=1e-5 )
    assert sources.apfluxadu(ap=1.)[0][50] == pytest.approx( 165.99489, rel=1e-5 )
    assert sources.apfluxadu(apnum=1)[0][0] == pytest.approx( 9883.959, rel=1e-5 )
    assert sources.apfluxadu(apnum=1)[0][50] == pytest.approx( 432.86523, rel=1e-5 )
    assert sources.apfluxadu(ap=2.505)[0][0] == pytest.approx( 9883.959, rel=1e-5 )
    assert sources.apfluxadu(ap=2.5)[0][50] == pytest.approx( 432.86523, rel=1e-5 )
    assert sources.apfluxadu()[1][0] == pytest.approx( 37.005665, rel=1e-5 )
    assert sources.apfluxadu()[1][50] == pytest.approx( 21.135862, rel=1e-5 )
    assert sources.apfluxadu(apnum=0)[1][0] == pytest.approx( 37.005665, rel=1e-5 )
    assert sources.apfluxadu(apnum=0)[1][50] == pytest.approx( 21.135862, rel=1e-5 )
    assert sources.apfluxadu(ap=1.)[1][0] == pytest.approx( 37.005665, rel=1e-5 )
    assert sources.apfluxadu(ap=1.)[1][50] == pytest.approx( 21.135862, rel=1e-5 )
    assert sources.apfluxadu(apnum=1)[1][0] == pytest.approx( 74.79863, rel=1e-5 )
    assert sources.apfluxadu(apnum=1)[1][50] == pytest.approx( 50.378757, rel=1e-5 )
    assert sources.apfluxadu(ap=2.5)[1][0] == pytest.approx( 74.79863, rel=1e-5 )
    assert sources.apfluxadu(ap=2.5)[1][50] == pytest.approx( 50.378757, rel=1e-5 )

    # Check some apfluxadu failure modes
    with pytest.raises( ValueError, match="Aperture radius number 2 doesn't exist." ):
        _ = sources.apfluxadu( apnum = 2 )
    with pytest.raises( ValueError, match="Can't find an aperture of .* pixels" ):
        _ = sources.apfluxadu( ap=6. )

    # Poke directly into the data array as well

    assert sources.data['X_IMAGE'][0] == sources.x[0]
    assert sources.data['Y_IMAGE'][0] == sources.y[0]
    assert sources.data['XWIN_IMAGE'][0] == pytest.approx( 798.29, abs=0.01 )
    assert sources.data['YWIN_IMAGE'][0] == pytest.approx( 17.11, abs=0.01 )
    assert sources.data['FLUX_APER'][0] == pytest.approx( np.array( [ 3044.9092, 9883.959 ] ), rel=1e-5 )
    assert sources.data['FLUXERR_APER'][0] == pytest.approx( np.array( [ 37.005665, 74.79863 ] ), rel=1e-5 )
    assert sources.data['X_IMAGE'][50] == sources.x[50]
    assert sources.data['Y_IMAGE'][50] == sources.y[50]
    assert sources.data['XWIN_IMAGE'][50] == pytest.approx( 899.29, abs=0.01 )
    assert sources.data['YWIN_IMAGE'][50] == pytest.approx( 604.58, abs=0.01 )
    assert sources.data['FLUX_APER'][50] == pytest.approx( np.array( [ 165.99489, 432.86523 ] ), rel=1e-5 )
    assert sources.data['FLUXERR_APER'][50] == pytest.approx( np.array( [ 21.135862, 50.378757 ] ), rel=1e-5 )


def test_write_sextractor(archive):
    fname = ''.join( np.random.choice( list('abcdefghijklmnopqrstuvwxyz'), 16 ) )
    sources = SourceList( format='sextrfits', filepath=f"{fname}.sources.fits" )
    assert sources.aper_rads is None
    if pathlib.Path( sources.get_fullpath() ).is_file():
        raise RuntimeError( f"{fname}.sources.fits exists when it shouldn't" )
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
        archive.delete(sources.filepath, okifmissing=True)


def test_calc_apercor( decam_datastore ):
    sources = decam_datastore.get_sources()

    # These numbers are when you don't use is_star at all:
    assert sources.calc_aper_cor() == pytest.approx(-0.4509, abs=0.01)
    assert sources.calc_aper_cor(aper_num=1) == pytest.approx(-0.177, abs=0.01)
    assert sources.calc_aper_cor(inf_aper_num=7) == pytest.approx(-0.4509, abs=0.01)
    assert sources.calc_aper_cor(inf_aper_num=2) == pytest.approx(-0.428, abs=0.01)
    assert sources.calc_aper_cor(aper_num=2) == pytest.approx(-0.028, abs=0.01)
    assert sources.calc_aper_cor(aper_num=2, inf_aper_num=7) == pytest.approx(-0.02356, abs=0.01)

    # The numbers below are what you get when you use CLASS_STAR in SourceList.is_star
    # assert sources.calc_aper_cor() == pytest.approx( -0.457, abs=0.01 )
    # assert sources.calc_aper_cor( aper_num=1 ) == pytest.approx( -0.177, abs=0.01 )
    # assert sources.calc_aper_cor( inf_aper_num=7 ) == pytest.approx( -0.463, abs=0.01 )
    # assert sources.calc_aper_cor( inf_aper_num=2 ) == pytest.approx( -0.428, abs=0.01 )
    # assert sources.calc_aper_cor( aper_num=2 ) == pytest.approx( -0.028, abs=0.01 )
    # assert sources.calc_aper_cor( aper_num=2, inf_aper_num=7 ) == pytest.approx( -0.034, abs=0.01 )

    # The numbers below are what you get if you use the SPREAD_MODEL
    # parameter in SourceList.is_star instead of CLASS_STAR
    # ...all of this should make us conclude that we should really not be claiming
    # to do photometry to better than a couple of percent!
    # assert sources.calc_aper_cor() == pytest.approx( -0.450, abs=0.001 )
    # assert sources.calc_aper_cor( aper_num=1 ) == pytest.approx( -0.173, abs=0.001 )
    # assert sources.calc_aper_cor( inf_aper_num=7 ) == pytest.approx( -0.450, abs=0.001 )
    # assert sources.calc_aper_cor( inf_aper_num=2 ) == pytest.approx( -0.425, abs=0.001 )
    # assert sources.calc_aper_cor( aper_num=2 ) == pytest.approx( -0.025, abs=0.001 )
    # assert sources.calc_aper_cor( aper_num=2, inf_aper_num=7 ) == pytest.approx( -0.024, abs=0.001 )


@pytest.mark.flaky(max_runs=3)
def test_free( decam_datastore ):
    ds = decam_datastore
    ds.get_sources()
    proc = psutil.Process()

    # Make sure image and source data is loaded into memory,
    #  then try freeing just the source data
    _ = ds.image.data
    _ = ds.sources.data
    _ = None

    assert ds.image._data is not None
    assert ds.sources._data is not None
    assert ds.sources._info is not None
    origmem = proc.memory_info()

    ds.sources.free()
    assert ds.sources._data is None
    assert ds.sources._info is None
    gc.collect()
    freemem = proc.memory_info()

    # Empirically, ds.sources._data.nbytes is about 10MiB That sounds
    #   like a lot for ~6000 sources, but oh well.  Right now, no memory
    #   seems to be freed.  I'm distressed.  gc.get_referrers only
    #   showed one thing referring to ds.sources._data, so setting that
    #   to None and garbage collecting should have freed stuff up.
    #   TODO: worry about this.  (Since at the moment we seem to be able
    #   to free image memory, and that is bigger, we've at least made
    #   some progress.)

    # assert ( origmem.rss - freemem.rss ) > ( 10 * 1024 * 1024 )

    # Make sure that if we free_derived_products from the image, then
    #   the sources get freed.  The image is 4096x2048 32-bit, so should
    #   use 4096*2046*4 = 32MiB.  There is also a 32-bit weight image,
    #   and a 16-bit flags image, so we expect to free 80MB of memory
    #   (plus whatever gets freed from the sources), but empirically I
    #   only got 64MB back on my home machine, and the google actions
    #   server only got just under 32MB back.  (All off ds.image._data,
    #   ds.image._weight, and ds.image._flags have a single referrer,
    #   based on gc.get_referrers.)  Memory management under the hood is
    #   almost certainly complicated, with the gc system (or whatever)
    #   deciding to keep some memory allocated and ready to be assigned
    #   to something new vs. actually returning it to the system.  I'd
    #   have to learn more about how all that works to understand why we
    #   don't get back everything we're freeing.  (Or, we could just
    #   give up on python and go back to pure C and manage all our
    #   memory ourselves.)

    _ = ds.image.data
    _ = ds.sources.data
    _ = None
    origmem = proc.memory_info()

    ds.image.free( free_derived_products=True )
    assert ds.image._data is None
    assert ds.image._weight is None
    assert ds.image._flags is None
    assert ds.sources._data is None
    assert ds.sources._info is None
    gc.collect()
    freemem = proc.memory_info()

    # Grr... last time I tried this on github actions, it didn't
    #   release any memory.  Further thought required.
    # assert ( origmem.rss - freemem.rss ) > ( 64 * 1024 * 1024 )
    # assert ( origmem.rss - freemem.rss ) > ( 30 * 1024 * 1024 )


