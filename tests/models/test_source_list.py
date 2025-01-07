import pytest
import os
# import psutil
import gc
import pathlib
import numpy as np
import time
import uuid

import sqlalchemy as sa

import astropy.table
import astropy.io.fits

from models.base import SmartSession, FileOnDiskMixin, CODE_ROOT
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from util.util import env_as_bool


def test_source_list_bitflag(sim_sources):
    # all these data products should have bitflag zero
    assert sim_sources.bitflag == 0
    assert sim_sources.badness == ''

    image = Image.get_by_id( sim_sources.image_id )
    exposure = Exposure.get_by_id( image.exposure_id )

    with SmartSession() as session:
        # try to find this using the bitflag hybrid property
        sim_sources2 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sim_sources.id in [s.id for s in sim_sources2]
        sim_sources2x = session.scalars(sa.select(SourceList).where(SourceList.bitflag > 0)).all()
        assert sim_sources.id not in [s.id for s in sim_sources2x]

    # now add a badness to the image and exposure
    image.set_badness( 'Saturation' )
    exposure.set_badness( 'Banding' )
    exposure.update_downstream_badness()

    # Reload from database, make sure stuff got updated
    image = Image.get_by_id( sim_sources.image_id )
    exposure = Exposure.get_by_id( image.exposure_id )
    sources = SourceList.get_by_id( sim_sources.id )

    assert image.bitflag == 2**1 + 2**3
    assert image.badness == 'banding, saturation'

    assert sources.bitflag == 2**1 + 2**3
    assert sources.badness == 'banding, saturation'

    # try to find this using the bitflag hybrid property
    with SmartSession() as session:
        sim_sources3 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 1 + 2 ** 3)).all()
        assert sim_sources.id in [s.id for s in sim_sources3]
        sim_sources3x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sim_sources.id not in [s.id for s in sim_sources3x]

    # now add some badness to the source list itself

    # cannot add an image badness to a source list
    with pytest.raises(ValueError, match='Keyword "Banding" not recognized in dictionary'):
        sources.set_badness( 'Banding' )

    # add badness that works with source lists (e.g., cross-match failures)
    sources.set_badness( 'few sources' )

    # Reload sources from database
    sources = SourceList.get_by_id( sources.id )

    assert sources.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 16
    assert sources.badness == 'banding, saturation, few sources'

    # try to find this using the bitflag hybrid property
    with SmartSession() as session:
        sim_sources4 = session.scalars( sa.select(SourceList)
                                        .where(SourceList.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 16)
                                       ).all()
        assert sim_sources.id in [s.id for s in sim_sources4]
        sim_sources4x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sim_sources.id not in [s.id for s in sim_sources4x]

    # removing the badness from the exposure is updated directly to the source list
    exposure.set_badness( '' )
    exposure.update_downstream_badness()

    # Reload image and sources from database
    image = Image.get_by_id( image.id )
    sources = SourceList.get_by_id( sources.id )

    assert image.badness == 'saturation'
    assert sources.badness == 'saturation, few sources'

    # check the database queries still work
    with SmartSession() as session:
        sim_sources5 = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 2 ** 3 + 2 ** 16)).all()
        assert sources.id in [s.id for s in sim_sources5]
        sim_sources5x = session.scalars(sa.select(SourceList).where(SourceList.bitflag == 0)).all()
        assert sources.id not in [s.id for s in sim_sources5x]

    # make sure new SourceList object gets the badness from the Image
    #
    # It won't -- this will only happen after you commit new_sources to
    # the datbase and call image.update_downstream_badness.
    # new_sources = SourceList( image_id=image.id )
    # assert new_sources.badness == 'saturation'


def test_invent_filepath( provenance_base, provenance_extra ):
    # Most of these fields aren't really needed for this test, but have
    #   to be there to commit to the database because of non-NULL constraints.
    imgargs = {
        'telescope': 'DemoTelescope',
        'instrument': 'DemoInstrument',
        'project': 'tests',
        'target': 'nothing',
        'section_id': 0,
        'type': "Sci",
        'format': "fits",
        'ra': 12.3456,
        'dec': -0.42,
        'mjd': 61738.64,
        'end_mjd': 61738.6407,
        'exp_time': 60.,
        'filter': 'r',
        'provenance_id': provenance_base.id,
        'md5sum': uuid.uuid4()                # Spoof since we don't really save a file
    }
    dra = 0.2
    ddec = 0.2
    imgargs['ra_corner_00'] = imgargs['ra'] - dra/2.
    imgargs['ra_corner_01'] = imgargs['ra'] - dra/2.
    imgargs['ra_corner_10'] = imgargs['ra'] + dra/2.
    imgargs['ra_corner_11'] = imgargs['ra'] + dra/2.
    imgargs['minra'] = imgargs['ra'] - dra/2.
    imgargs['maxra'] = imgargs['ra'] + dra/2.
    imgargs['dec_corner_00'] = imgargs['dec'] - ddec/2.
    imgargs['dec_corner_01'] = imgargs['dec'] + ddec/2.
    imgargs['dec_corner_10'] = imgargs['dec'] - ddec/2.
    imgargs['dec_corner_11'] = imgargs['dec'] + ddec/2.
    imgargs['mindec'] = imgargs['dec'] - ddec/2.
    imgargs['maxdec'] = imgargs['dec'] + ddec/2.

    hash1 = provenance_base.id[:6]
    hash2 = provenance_extra.id[:6]

    # Make sure it screams if we have no image_id
    sources = SourceList( format='sextrfits', provenance_id=provenance_extra.id )
    with pytest.raises( RuntimeError, match="Can't invent a filepath for sources without an image" ):
        sources.invent_filepath()

    # Make sure it screams if we point to an image that doesn't exist
    sources = SourceList( image_id=uuid.uuid4(), format='sextrfits', provenance_id=provenance_extra.id )
    with pytest.raises( RuntimeError, match='Could not find image for sourcelist' ):
        sources.invent_filepath()

    # Make sure it works if we explicitly pass an image
    image = Image( filepath="testing", **imgargs )
    assert sources.invent_filepath( image ) == f'{image.filepath}.sources_{hash2}.fits'

    # Make sure it get an image filepath from an image saved in the database with automatically generated filepath
    try:
        image = Image( **imgargs )
        image.filepath = image.invent_filepath()
        image.insert()
        sources = SourceList( image_id=image.id, format='sextrfits', provenance_id=provenance_extra.id )
        assert sources.invent_filepath() == f'012/Demo_20271129_152136_0_r_Sci_{hash1}.sources_{hash2}.fits'
    finally:
        with SmartSession() as session:
            session.execute( sa.text( "DELETE FROM images WHERE _id=:id" ), { 'id': image.id } )
            session.commit()

    # Make sure it can get an image filepath from an imaged saved in the database with a manual filepath
    try:
        image = Image( filepath="this.is.a.test", **imgargs )
        image.insert()
        sources = SourceList( image_id=image.id, format='sextrfits', provenance_id=provenance_extra.id )
        assert sources.invent_filepath() == f'this.is.a.test.sources_{hash2}.fits'
    finally:
        with SmartSession() as session:
            session.execute( sa.text( "DELETE FROM images WHERE _id=:id" ), { 'id': image.id } )
            session.commit()


def test_read_sextractor( ztf_filepath_sources ):
    fullpath = ztf_filepath_sources
    filepath = fullpath.relative_to( pathlib.Path( FileOnDiskMixin.local_path ) )

    # Make sure things go haywire when we try to load data with inconsistent
    # num_sources or aper_rads
    with pytest.raises( ValueError, match='self.num_sources=10 but the sextractor file had' ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=10 )
        _ = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match the number of apertures found in" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=112, aper_rads=[1., 2., 3.] )
        _ = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match sextractor file" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=112,
                              aper_rads=[ 2., 5. ] )
        _ = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match sextractor file" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=112,
                              aper_rads=[ 1., 2. ] )
        _ = sources.data
    with pytest.raises( ValueError, match="self.aper_rads.*doesn't match the number of apertures found in" ):
        sources = SourceList( format='sextrfits', filepath=filepath, num_sources=112,
                              aper_rads=[ 1. ] )
        _ = sources.data

    # Make sure those fields get properly auto-set
    sources = SourceList( format='sextrfits', filepath=filepath )
    # Access the data property to get the data loaded
    _ = sources.data
    assert sources.num_sources == 112
    assert sources.aper_rads == [ 1., 2.5 ]

    # Make sure we can read the file with the right things in place in those fields
    sources = SourceList( format='sextrfits', filepath=filepath, num_sources=112, aper_rads=[ 1.0, 2.5 ] )
    # Access the data property to get the data loaded
    _ = sources.data
    assert len(sources.data) == 112
    assert sources.num_sources == 112
    assert sources.good.sum() == 105
    assert sources.aper_rads == [ 1.0, 2.5 ]
    assert sources.inf_aper_num is None
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
    rng = np.random.default_rng()
    fname = ''.join( rng.choice( list('abcdefghijklmnopqrstuvwxyz'), 16 ) )
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
    # We should think again about using CLASS_STAR once we address Issue #381.
    assert sources.calc_aper_cor() == pytest.approx(-0.2642, abs=0.01)
    assert sources.calc_aper_cor(aper_num=1) == pytest.approx(-0.0686, abs=0.01)
    assert sources.calc_aper_cor(inf_aper_num=3) == pytest.approx(-0.2642, abs=0.01)
    assert sources.calc_aper_cor(inf_aper_num=1) == pytest.approx(-0.2032, abs=0.01)
    assert sources.calc_aper_cor(aper_num=2) == pytest.approx(-0.0255, abs=0.01)
    assert sources.calc_aper_cor(aper_num=2, inf_aper_num=3) == pytest.approx(-0.0255, abs=0.01)


def test_lim_mag_estimate( ptf_datastore_through_zp ):
    ds = ptf_datastore_through_zp

    # make and save a Magnitude vs SNR (limiting mag) plot
    if env_as_bool('INTERACTIVE'):
        limMagEst = ds.sources.estimate_lim_mag( aperture=1, zp=ds.zp,
                                                 savePlot=os.path.join(CODE_ROOT, 'tests/plots/snr_mag_plot.png' ) )
    else:
        limMagEst = ds.sources.estimate_lim_mag( aperture=1, zp=ds.zp )

    # check the limiting magnitude is consistent with previous runs
    assert limMagEst == pytest.approx(20.04, abs=0.05)

    # Make sure that it can auto-get the zp if you don't pass one
    redo = ds.sources.estimate_lim_mag( aperture=1 )
    assert redo == limMagEst


# ROB TODO : check this test once you've updated DataStore and the associated fixtures
@pytest.mark.skip(reason="This test regularly fails, even when flaky is used. See Issue #263")
def test_free( decam_datastore ):
    ds = decam_datastore
    ds.get_sources()
    # proc = psutil.Process()

    sleeptime = 0.5 # in seconds

    # Make sure image and source data is loaded into memory,
    #  then try freeing just the source data
    _ = ds.image.data
    _ = ds.sources.data
    _ = None

    assert ds.image._data is not None
    assert ds.sources._data is not None
    assert ds.sources._info is not None
    # origmem = proc.memory_info()

    ds.sources.free()
    time.sleep(sleeptime)
    assert ds.sources._data is None
    assert ds.sources._info is None
    gc.collect()
    # freemem = proc.memory_info()

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
    # origmem = proc.memory_info()

    ds.image.free( free_derived_products=True )
    time.sleep(sleeptime)
    assert ds.image._data is None
    assert ds.image._weight is None
    assert ds.image._flags is None
    assert ds.sources._data is None
    assert ds.sources._info is None
    gc.collect()
    # freemem = proc.memory_info()

    # Grr... last time I tried this on github actions, it didn't
    #   release any memory.  Further thought required.
    # assert ( origmem.rss - freemem.rss ) > ( 64 * 1024 * 1024 )
    # assert ( origmem.rss - freemem.rss ) > ( 30 * 1024 * 1024 )
