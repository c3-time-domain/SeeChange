import pytest
import io
import psutil
import time
import uuid
import random
import pathlib
import subprocess

import numpy as np
import scipy.ndimage

import sqlalchemy as sa
import psycopg2.errors
import psycopg2.extras
import astropy
from astropy.io import fits

from util.config import Config
from util.util import env_as_bool, asUUID
from util.logger import SCLogger
from models.base import SmartSession, Psycopg2Connection, FileOnDiskMixin, CODE_ROOT, get_archive_object
from models.provenance import Provenance
from models.psf import PSF, PSFExPSF, DeltaPSF, GaussianPSF, ImagePSF
from models.enums_and_bitflags import PSFFormatConverter


def test_psf_polymorphism( bogus_image_factory, bogus_sources_factory ):
    # Make sure that SQLAlchmey writes things properly and
    #   loads the right classes.

    imgdel = []
    srcdel = []
    psfdel = []

    try:
        pimg = bogus_image_factory( uuid.uuid4(), "test_psf_poly_psfex" )
        imgdel.append( pimg )
        psrc = bogus_sources_factory( uuid.uuid4(), "test_psf_poly_psfex_sources.fits", pimg )
        srcdel.append( psrc )
        ppsf = PSFExPSF( fwhm_pixels=2., sources_id=psrc.id )
        ppsf.filepath = 'test_psf_poly_psfex_psf.fits'
        ppsf.md5sum_components = [ uuid.uuid4(), uuid.uuid4() ]
        ppsf.insert()
        psfdel.append( ppsf )

        dimg = bogus_image_factory( uuid.uuid4(), "test_psf_poly_delta" )
        imgdel.append( dimg )
        dsrc = bogus_sources_factory( uuid.uuid4(), "test_psf_poly_delta_sources.fits", dimg )
        srcdel.append( dsrc )
        dpsf = DeltaPSF( fwhm_pixels=0., sources_id=dsrc.id )
        dpsf.save()  # Doesn't really save any file
        dpsf.insert()
        psfdel.append( dpsf )

        gimg = bogus_image_factory( uuid.uuid4(), "test_psf_poly_gaussian" )
        imgdel.append( gimg )
        gsrc = bogus_sources_factory( uuid.uuid4(), "test_psf_poly_gaussian_sources.fits", gimg )
        srcdel.append( gsrc )
        gpsf = GaussianPSF( fwhm_pixels=2.1, sources_id=gsrc.id )
        gpsf.save()   # Doesn't really save any file
        gpsf.insert()
        psfdel.append( gpsf )

        iimg = bogus_image_factory( uuid.uuid4(), "test_psf_poly_image" )
        imgdel.append( iimg )
        isrc = bogus_sources_factory( uuid.uuid4(), "test_psf_poly_image_sources.fits", iimg )
        srcdel.append( isrc )
        ipsf = ImagePSF( fwhm_pixels=0., sources_id=isrc.id )
        ipsf.filepath = "test_psf_poly_image_psf.hdf5"
        ipsf.md5sum = uuid.uuid4()
        ipsf.insert()
        psfdel.append( ipsf )

        # Make sure that they all got stuck in the database
        with Psycopg2Connection() as con:
            cursor = con.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( "SELECT * FROM psfs WHERE _id IN %(ids)s",
                            { 'ids': ( ppsf.id, dpsf.id, gpsf.id, ipsf.id ) } )
            psfs = cursor.fetchall()
            assert len(psfs) == 4
            for psf in psfs:
                if asUUID(psf['_id']) == ppsf.id:
                    assert PSFFormatConverter.to_string( psf['_format'] ) == 'psfex'
                elif asUUID(psf['_id']) == dpsf.id:
                    assert PSFFormatConverter.to_string( psf['_format'] ) == 'delta'
                elif asUUID(psf['_id']) == gpsf.id:
                    assert PSFFormatConverter.to_string( psf['_format'] ) == 'gaussian'
                elif asUUID(psf['_id']) == ipsf.id:
                    assert PSFFormatConverter.to_string( psf['_format'] ) == 'image'
                else:
                    raise RuntimeError( "This should never happen." )

        # Make sure SQLAlchemy loads the right things
        with SmartSession() as sess:
            psfs = sess.query( PSF ).filter( PSF._id.in_( i.id for i in [ ppsf, dpsf, gpsf, ipsf ] ) ).all()
            assert len( psfs ) == 4
            for psf in psfs:
                if psf.id == ppsf.id:
                    assert isinstance( psf, PSFExPSF )
                elif psf.id == dpsf.id:
                    assert isinstance( psf, DeltaPSF )
                elif psf.id == gpsf.id:
                    assert isinstance( psf, GaussianPSF )
                elif psf.id == ipsf.id:
                    assert isinstance( psf, ImagePSF )
                else:
                    raise RuntimeError( "This should never happen." )

    finally:
        with Psycopg2Connection() as con:
            cursor = con.cursor()
            cursor.execute( "DELETE FROM psfs WHERE _id=ANY(%(id)s)", { 'id': ([i.id for i in psfdel],) } )
            con.commit()
        for src in srcdel:
            src.delete_from_disk_and_database()
        for img in imgdel:
            img.delete_from_disk_and_database()


def check_example_psfex_psf_values( psf ):
    assert psf.header[ 'TTYPE1' ] == 'PSF_MASK'
    assert psf.header[ 'POLDEG1' ] == 2
    assert psf.header[ 'PSFNAXIS' ] == 3
    assert psf.header[ 'PSFAXIS1' ] == 25
    assert psf.header[ 'PSFAXIS2' ] == 25
    assert psf.header[ 'POLNAME1' ] == 'X_IMAGE'
    assert psf.header[ 'POLZERO1' ] == pytest.approx( 514.31, abs=0.01 )
    assert psf.header[ 'POLSCAL1' ] == pytest.approx( 1018.67, abs=0.01 )
    assert psf.header[ 'POLNAME2' ] == 'Y_IMAGE'
    assert psf.header[ 'POLZERO2' ] == pytest.approx( 497.36, abs=0.01 )
    assert psf.header[ 'POLSCAL2' ] == pytest.approx( 991.75, abs=0.01 )
    assert psf.data.shape == ( 6, 25, 25, )

    bytio = io.BytesIO( psf.info.encode( 'utf-8' ) )
    psfstats = astropy.io.votable.parse( bytio ).get_table_by_index(1)
    assert psfstats.array[ 'NStars_Loaded_Mean' ] == 43
    assert psfstats.array[ 'NStars_Accepted_Mean' ] ==  41
    assert psfstats.array[ 'FWHM_FromFluxRadius_Mean' ] == pytest.approx( 3.13, abs=0.01 )


def test_read_psfex_psf( ztf_filepaths_image_sources_psf ):
    _, _, _, _, psfpath, psfxmlpath = ztf_filepaths_image_sources_psf
    psf = PSFExPSF()
    psf.load( psfpath=psfpath, psfxmlpath=psfxmlpath )
    check_example_psfex_psf_values( psf )


def test_write_psfex_psf( ztf_filepaths_image_sources_psf ):
    image, weight, flags, _, psfpath, psfxmlpath = ztf_filepaths_image_sources_psf
    psf = PSFExPSF()
    psf.load( psfpath=psfpath, psfxmlpath=psfxmlpath )

    tempname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    psfpath = f'{tempname}.psf.fits'
    psffullpath = pathlib.Path( FileOnDiskMixin.local_path ) / psfpath
    psfxmlpath = f'{tempname}.psf.xml'
    psfxmlfullpath = pathlib.Path( FileOnDiskMixin.local_path ) / psfxmlpath
    sourcesfullpath = pathlib.Path( FileOnDiskMixin.local_path ) / f'{tempname}.cat'

    try:
        # Write it out, make sure the expected files get created
        psf.save( tempname )
        assert psffullpath.is_file()
        assert psfxmlfullpath.is_file()
        archive = get_archive_object()
        assert archive.get_info( psfpath ) is not None
        assert archive.get_info( psfxmlpath ) is not None

        # See if we can read the psf we wrote back in
        psf = PSFExPSF()
        psf.load( psfpath=psffullpath, psfxmlpath=psfxmlfullpath )
        check_example_psfex_psf_values( psf )

        # Make sure SEXtractor can read this psf file

        # Figure out where astromatic config files are:
        astromatic_dir = None
        cfg = Config.get()
        if cfg.value( 'astromatic.config_dir' ) is not None:
            astromatic_dir = pathlib.Path( cfg.value( 'astromatic.config_dir' ) )
        elif cfg.value( 'astromatic.config_subdir' ) is not None:
            astromatic_dir = pathlib.Path( CODE_ROOT ) / cfg.value( 'astromatic.config_subdir' )
        assert astromatic_dir is not None
        assert astromatic_dir.is_dir()
        conv = astromatic_dir / "default.conv"
        nnw = astromatic_dir / "default.nnw"
        param = astromatic_dir / "sourcelist_sextractor_with_psf.param"

        command = [ 'source-extractor',
                    '-CATALOG_NAME', sourcesfullpath,
                    '-CATALOG_TYPE', 'FITS_LDAC',
                    '-PARAMETERS_NAME', param,
                    '-FILTER', 'Y',
                    '-FILTER_NAME', conv,
                    '-WEIGHT_TYPE', 'MAP_WEIGHT',
                    '-RESCALE_WEIGHTS', 'N',
                    '-WEIGHT_IMAGE', weight,
                    '-FLAG_IMAGE', flags,
                    '-FLAG_TYPE', 'OR',
                    '-PHOT_APERTURES', '2,5',
                    '-SATUR_LEVEL', '54000',
                    '-STARNNW_NAME', nnw,
                    '-BACK_TYPE', 'AUTO',
                    '-BACK_SIZE', '128',
                    '-BACK_FILTERSIZE', '3',
                    '-PSF_NAME', psffullpath,
                    image ]
        res = subprocess.run( command, capture_output=True, timeout=60 )
        assert res.returncode == 0

    finally:
        psffullpath.unlink( missing_ok=True )
        psfxmlfullpath.unlink( missing_ok=True )
        sourcesfullpath.unlink( missing_ok=True )
        archive.delete(psfpath, okifmissing=True)
        archive.delete(psfxmlpath, okifmissing=True)


def test_psfex_psf_positioning( ztf_filepaths_image_sources_psf ):
    _, _, _, _, psfpath, psfxmlpath = ztf_filepaths_image_sources_psf
    psf = PSFExPSF()
    psf.load( psfpath=psfpath, psfxmlpath=psfxmlpath )

    # In pratice the ZTF PSF seems to be slightly lopsided, so centroids
    #   aren't showing up exactly where expected.  But, the general
    #   principle will work here -- making sure that things at
    #   (x.49,y.49) are up and to the right, and things at (x.50,y.50)
    #   are down and to the left.

    # A PSF at a pixel position should be centered on the return image
    img = psf.get_clip( 512, 512 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 8.0, abs=0.07 )
    assert cy == pytest.approx( 8.0, abs=0.07 )

    img = psf.get_clip( 5, 5 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 8.0, abs=0.07 )
    assert cy == pytest.approx( 8.0, abs=0.08 )  #...close...

    img = psf.get_clip( 999, 12 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 8.0, abs=0.07 )
    assert cy == pytest.approx( 8.0, abs=0.07 )

    # A PSF at a (integer+(0,0.5)) position should be centered up and to the right on the return image
    img = psf.get_clip( 512.2, 512.2 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 8.2, abs=0.07 )
    assert cy == pytest.approx( 8.2, abs=0.07 )

    # A PSF at a (integer+[0.5,1.0)) position should be centered down and to the left on the return image
    img = psf.get_clip( 512.5, 512.5 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 7.5, abs=0.07 )
    assert cy == pytest.approx( 7.5, abs=0.07 )

    img = psf.get_clip( 512.0, 512.5 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 8.0, abs=0.07 )
    assert cy == pytest.approx( 7.5, abs=0.07 )

    img = psf.get_clip( 512.9, 512.9 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 7.9, abs=0.07 )
    assert cy == pytest.approx( 7.9, abs=0.07 )

    img = psf.get_clip( 512.49, 512.49 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 8.49, abs=0.07 )
    assert cy == pytest.approx( 8.49, abs=0.07 )

    img = psf.get_clip( 512.51, 512.51 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 7.51, abs=0.07 )
    assert cy == pytest.approx( 7.51, abs=0.07 )

    # Make sure that even/odd rounding conventions don't give inconsistent results at x.5
    img = psf.get_clip( 513.5, 513.5 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 7.5, abs=0.07 )
    assert cy == pytest.approx( 7.5, abs=0.07 )

    img = psf.get_clip( 513.0, 513.5 )
    assert img.shape == ( 17, 17 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 8.0, abs=0.07 )
    assert cy == pytest.approx( 7.5, abs=0.07 )

    # ...the rest of this may be a bit gratuitous given
    # test_get_centered_offset below

    # Make sure that get_centered_psf does the right thing
    # If the return image has an odd size, then the psf should be
    #   centered on the center pixel

    img = psf.get_centered_psf( 999, 999 )
    assert img.shape == ( 999, 999 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 499, abs=0.07 )
    assert cy == pytest.approx( 499, abs=0.07 )

    # If there return image has an even size, then the psf should
    #   be centered on the corner of the four pixels at the center

    img = psf.get_centered_psf( 1000, 1000 )
    assert img.shape == ( 1000, 1000 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 499.5, abs=0.07 )
    assert cy == pytest.approx( 499.5, abs=0.07 )

    # Also make sure we didn't mix up x and y in our calculations
    # (Especially since ndarrays are indexed [y, x])

    img = psf.get_centered_psf( 1000, 999 )
    assert img.shape == ( 999, 1000 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 499.5, abs=0.07 )
    assert cy == pytest.approx( 499, abs=0.07 )

    img = psf.get_centered_psf( 999, 1000 )
    assert img.shape == ( 1000, 999 )
    cy, cx = scipy.ndimage.center_of_mass( img )
    assert cx == pytest.approx( 499, abs=0.07 )
    assert cy == pytest.approx( 499.5, abs=0.07 )



def test_save_psfex_psf( ztf_datastore_uncommitted, provenance_base, provenance_extra ):
    try:
        im = ztf_datastore_uncommitted.image
        src = ztf_datastore_uncommitted.sources
        psf = ztf_datastore_uncommitted.psf

        improv = Provenance( process='gratuitous image' )
        srcprov = Provenance( process='gratuitous sources' )
        improv.insert()
        srcprov.insert()

        im.provenance_id = improv.id
        im.save()
        src.provenance_id = srcprov.id
        src.save()
        psf.save( image=im, sources=src )
        im.insert()
        src.insert()
        psf.insert()

        # TODO : make sure we can load the one we just saved

        # make a copy of the PSF (we will not be able to save it, with the same image_id and provenance)
        psf2 = PSFExPSF()
        psf2._data = psf.data
        psf2._header = psf.header
        psf2._info = psf.info
        psf2.sources_id = psf.sources_id
        psf2.fwhm_pixels = psf.fwhm_pixels * 2  # make it a little different
        psf2.save( filename=uuid.uuid4().hex[:10], image=im, sources=src )

        with pytest.raises( psycopg2.errors.UniqueViolation,
                            match='duplicate key value violates unique constraint "ix_psfs_sources_id"' ):
            psf2.insert()


    finally:
        if 'psf2' in locals():
            psf2.delete_from_disk_and_database()
        if 'im' in locals():
            im.delete_from_disk_and_database()
        # Should cascade down to delelete sources and psf

        with SmartSession() as session:
            session.execute( sa.delete( Provenance ).filter( Provenance._id.in_( [ improv.id, srcprov.id ] ) ) )


@pytest.mark.skip(reason="This test regularly fails, even when flaky is used. See Issue #263")
def test_free( decam_datastore ):
    ds = decam_datastore
    ds.get_psf()
    proc = psutil.Process()

    sleeptime = 0.5 # in seconds

    # Make sure memory is loaded
    _ = ds.image.data
    _ = ds.psf.data
    _ = None

    assert ds.image._data is not None
    assert ds.psf._data is not None
    assert ds.psf._info is not None
    assert ds.psf._header is not None

    origmem = proc.memory_info()
    ds.psf.free()
    time.sleep(sleeptime)
    assert ds.psf._data is None
    assert ds.psf._info is None
    assert ds.psf._header is None
    freemem = proc.memory_info()

    # psf._data.nbytes was 15k, so it's going to be in the noise of free
    #  memory.  (High-school me with his commodore 64 is facepalming at
    #  that statement.)  Empirically, origmem.rss and freemem.rss are
    #  the same right now.

    # Make sure it reloads
    _ = ds.psf.data
    assert ds.psf._data is not None
    assert ds.psf._info is not None
    assert ds.psf._header is not None
    ds.psf.free()

    origmem = proc.memory_info()
    ds.psf.free()
    ds.sources.free()
    ds.image.free()
    time.sleep(sleeptime)
    assert ds.psf._data is None
    assert ds.psf._info is None
    assert ds.psf._header is None
    freemem = proc.memory_info()

    assert origmem.rss - freemem.rss > 60 * 1024 * 1024


@pytest.mark.skipif( not env_as_bool('RUN_SLOW_TESTS'), reason="Set RUN_SLOW_TESTS to run this test" )
def test_psfex_rendering( psf_palette ): # round_psf_palette ):
    # psf_palette = round_psf_palette
    psf = psf_palette.psf

    resamp = psf.get_resampled_psf( 512., 512., dtype=np.float32 )
    # ****
    # Uncomment this for by-eye debugging
    # fits.writeto( '/seechange/data/resamp.fits', resamp, overwrite=True )
    # ****
    assert resamp.shape == ( 31, 31 )

    clip = psf.get_clip( 512., 512., 1., dtype=np.float64 )
    assert clip.shape == ( 25, 25 )
    assert clip.sum() == pytest.approx( 1., abs=1e-5 )

    with fits.open( psf_palette.imagename ) as ifp:
        data = ifp[0].data

    model = np.zeros( data.shape, dtype=np.float32 )
    for x in psf_palette.xpos:
        for y in psf_palette.ypos:
            psf.add_psf_to_image( model, x, y, psf_palette.flux )

    # ****
    # Uncomment these for by-eye debugging
    # fits.writeto( '/seechange/data/model.fits', model, overwrite=True )
    # fits.writeto( '/seechange/data/resid.fits', data-model, overwrite=True )
    # with fits.open( psf_palette.imagename ) as ifp:
    #     fits.writeto( '/seechange/data/data.fits', ifp[0].data, overwrite=True )
    # ****

    with fits.open( psf_palette.weightname, memmap=False ) as whdu:
        weight = whdu[0].data

    resid = data - model
    chisq = 0.
    halfwid = clip.shape[1] // 2
    n = 0
    for x in psf_palette.xpos:
        for y in psf_palette.ypos:
            # ix = int( np.floor( x + 0.5 ) )
            iy = int( np.floor( y + 0.5 ) )
            chisq += np.square( resid[ iy - halfwid : iy + halfwid + 1 ]
                                * weight[ iy - halfwid : iy + halfwid + 1 ] ).sum()
            n += clip.size

    # Yes, ideally chisq / n is supposed to be close to 1., but in
    # practice it does not seem to be so.  So, the psf model isn't a
    # perfect model.  This warrants further thought.  (Other
    # investigations do suggest that it is good enough for psf
    # photometry, at least as compared to aperture photometry.)  Suffice
    # to say that the chisq would be a *lot* worse if (for instance) I
    # mixed up the x and y terms on the polynomial in
    # PSF.get_resampled_psf (the **i and **j).

    assert chisq / n < 105.


def test_delta_psf():
    psf = DeltaPSF()

    # Remember that numpy arrays are indexed [y, x]

    # Centered
    clip = psf.get_clip( 5, 39 )
    assert clip.shape == ( 5, 5 )
    assert np.all( clip >= 0. )
    assert clip.sum() == pytest.approx( 1., rel=1e-6 )
    assert clip[2, 2] == pytest.approx( 1., rel=1e-6 )

    # Offset up and right
    clip = psf.get_clip( 1024.25, 3.25 )
    assert np.all( clip >= 0. )
    assert clip.sum() == pytest.approx( 1., rel=1e-6 )
    assert clip[2, 2] == pytest.approx( 0.75 * 0.75, rel=1e-6 )
    assert clip[3, 2] == pytest.approx( 0.75 * 0.25, rel=1e-6 )
    assert clip[2, 3] == pytest.approx( 0.25 * 0.75, rel=1e-6 )
    assert clip[3, 3] == pytest.approx( 0.25 * 0.25, rel=1e-6 )

    clip = psf.get_clip( 1024.4, 3.1 )
    assert np.all( clip >= 0. )
    assert clip.sum() == pytest.approx( 1., rel=1e-6 )
    assert clip[2, 2] == pytest.approx( 0.6 * 0.9, rel=1e-6 )
    assert clip[3, 2] == pytest.approx( 0.6 * 0.1, rel=1e-6 )
    assert clip[2, 3] == pytest.approx( 0.4 * 0.9, rel=1e-6 )
    assert clip[3, 3] == pytest.approx( 0.4 * 0.1, rel=1e-6 )

    # Offset down and right
    clip = psf.get_clip( 512.4, 99.9 )
    assert np.all( clip >= 0. )
    assert clip.sum() == pytest.approx( 1., rel=1e-6 )
    assert clip[2, 2] == pytest.approx( 0.6 * 0.9, rel=1e-6 )
    assert clip[1, 2] == pytest.approx( 0.6 * 0.1, rel=1e-6 )
    assert clip[2, 3] == pytest.approx( 0.4 * 0.9, rel=1e-6 )
    assert clip[1, 3] == pytest.approx( 0.4 * 0.1, rel=1e-6 )

    # Offset up and left
    clip = psf.get_clip( 66.6, 300.1 )
    assert np.all( clip >= 0. )
    assert clip.sum() == pytest.approx( 1., rel=1e-6 )
    assert clip[2, 2] == pytest.approx( 0.6 * 0.9, rel=1e-6 )
    assert clip[3, 2] == pytest.approx( 0.6 * 0.1, rel=1e-6 )
    assert clip[2, 1] == pytest.approx( 0.4 * 0.9, rel=1e-6 )
    assert clip[3, 1] == pytest.approx( 0.4 * 0.1, rel=1e-6 )

    # Offset down and left
    clip = psf.get_clip( 1.6, 3.9 )
    assert np.all( clip >= 0. )
    assert clip.sum() == pytest.approx( 1., rel=1e-6 )
    assert clip[2, 2] == pytest.approx( 0.6 * 0.9, rel=1e-6 )
    assert clip[1, 2] == pytest.approx( 0.6 * 0.1, rel=1e-6 )
    assert clip[2, 1] == pytest.approx( 0.4 * 0.9, rel=1e-6 )
    assert clip[1, 1] == pytest.approx( 0.4 * 0.1, rel=1e-6 )


    # Not tested, but maybe should be: flux, noisy, dtype, etc.  (Use
    # case for the delta psf for all of those is dubous, though.


def test_gaussian_psf():
    psf = GaussianPSF( fwhm_pixels=3.2 )

     # Mostly I'm just testing that I got my centering right...!

    clip = psf.get_clip( 512., 512. )
    mid = clip.shape[0] // 2
    assert clip.sum() == pytest.approx( 1., rel=1e-6 )
    # Should be centered on the center pixel
    cy, cx = scipy.ndimage.center_of_mass( clip )
    assert cx == pytest.approx( mid, abs=0.01 )
    assert cy == pytest.approx( mid, abs=0.01 )
    assert clip[ mid, mid ] > clip[ mid-1, mid-1 ]
    assert clip[ mid+1, mid+1 ] == pytest.approx( clip[ mid-1, mid-1 ], rel=1e-6 )
    assert clip[ mid+1, mid-1 ] == pytest.approx( clip[ mid-1, mid-1 ], rel=1e-6 )
    assert clip[ mid-1, mid+1 ] == pytest.approx( clip[ mid-1, mid-1 ], rel=1e-6 )

    clip = psf.get_clip( 512.5, 512.5 )
    # Should be centered on the corner
    cy, cx = scipy.ndimage.center_of_mass( clip )
    assert cx == pytest.approx( mid-0.5, abs=0.01 )
    assert cy == pytest.approx( mid-0.5, abs=0.01 )
    assert clip[ mid, mid-1 ] == pytest.approx( clip[ mid, mid ], rel=1e-6 )
    assert clip[ mid-1, mid ] == pytest.approx( clip[ mid, mid ], rel=1e-6 )
    assert clip[ mid-1, mid-1 ] == pytest.approx( clip[ mid, mid ], rel=1e-6 )

    clip = psf.get_clip( 512.5, 512.6 )
    # Should be centered on the half-pixel in x, lean down
    cy, cx = scipy.ndimage.center_of_mass( clip )
    assert cx == pytest.approx( mid-0.5, abs=0.01 )
    assert cy == pytest.approx( mid-0.4, abs=0.01 )
    assert clip[ mid, mid-1 ] == pytest.approx( clip[ mid, mid ], rel=1e-6 )
    assert clip[ mid-1, mid-1 ] == pytest.approx( clip[ mid-1, mid], rel=1e-6 )
    assert clip[ mid-1, mid ] < clip[ mid, mid ]
    assert clip[ mid-1, mid ] > clip[ mid+1, mid ]

    clip = psf.get_clip( 512.6, 512.5 )
    # Should be centered on the half-pixel in y, lean left
    cy, cx = scipy.ndimage.center_of_mass( clip )
    assert cx == pytest.approx( mid-0.4, abs=0.01 )
    assert cy == pytest.approx( mid-0.5, abs=0.01 )
    assert clip[ mid-1, mid ] == pytest.approx( clip[ mid, mid ], rel=1e-6 )
    assert clip[ mid-1, mid-1 ] == pytest.approx( clip[ mid, mid-1], rel=1e-6 )
    assert clip[ mid, mid-1 ] < clip[ mid, mid ]
    assert clip[ mid, mid-1 ] > clip[ mid, mid+1 ]

    clip = psf.get_clip( 512.2, 512.8 )
    # Should lean right, down
    cy, cx = scipy.ndimage.center_of_mass( clip )
    assert cx == pytest.approx( mid+0.2, abs=0.01 )
    assert cy == pytest.approx( mid-0.2, abs=0.01 )
    assert clip[ mid, mid+1 ] < clip[ mid, mid ]
    assert clip[ mid, mid+1 ] > clip[ mid, mid-1 ]
    assert clip[ mid-1, mid ] < clip[ mid, mid ]
    assert clip[ mid-1, mid ] > clip[ mid+1, mid ]
    assert clip[ mid-1, mid ] == pytest.approx( clip[ mid, mid+1 ], rel=1e-6 )
    assert clip[ mid+1, mid ] == pytest.approx( clip[ mid, mid-1 ], rel=1e-6 )

    clip = psf.get_clip( 512.8, 512.2 )
    # Should lean left, up
    cy, cx = scipy.ndimage.center_of_mass( clip )
    assert cx == pytest.approx( mid-0.2, abs=0.01 )
    assert cy == pytest.approx( mid+0.2, abs=0.01 )
    assert clip[ mid, mid-1 ] < clip[ mid, mid ]
    assert clip[ mid, mid-1 ] > clip[ mid, mid+1 ]
    assert clip[ mid+1, mid ] < clip[ mid, mid ]
    assert clip[ mid+1, mid ] > clip[ mid-1, mid ]
    assert clip[ mid+1, mid ] == pytest.approx( clip[ mid, mid-1 ], rel=1e-6 )
    assert clip[ mid-1, mid ] == pytest.approx( clip[ mid, mid+1 ], rel=1e-6 )

    clip = psf.get_clip( 512.2, 512.2 )
    # Should lean right, up
    cy, cx = scipy.ndimage.center_of_mass( clip )
    assert cx == pytest.approx( mid+0.2, abs=0.01 )
    assert cy == pytest.approx( mid+0.2, abs=0.01 )
    assert clip[ mid, mid+1 ] < clip[ mid, mid ]
    assert clip[ mid, mid+1 ] > clip[ mid, mid-1 ]
    assert clip[ mid+1, mid ] < clip[ mid, mid ]
    assert clip[ mid+1, mid ] > clip[ mid-1, mid ]
    assert clip[ mid+1, mid ] == pytest.approx( clip[ mid, mid+1 ], rel=1e-6 )
    assert clip[ mid-1, mid ] == pytest.approx( clip[ mid, mid-1 ], rel=1e-6 )

    clip = psf.get_clip( 512.8, 512.8 )
    # Should lean left, down
    cy, cx = scipy.ndimage.center_of_mass( clip )
    assert cx == pytest.approx( mid-0.2, abs=0.01 )
    assert cy == pytest.approx( mid-0.2, abs=0.01 )
    assert clip[ mid, mid-1 ] < clip[ mid, mid ]
    assert clip[ mid, mid-1 ] > clip[ mid, mid+1 ]
    assert clip[ mid-1, mid ] < clip[ mid, mid ]
    assert clip[ mid-1, mid ] > clip[ mid+1, mid ]
    assert clip[ mid-1, mid ] == pytest.approx( clip[ mid, mid-1 ], rel=1e-6 )
    assert clip[ mid+1, mid ] == pytest.approx( clip[ mid, mid+1 ], rel=1e-6 )


def test_get_centered_psf_offset():
    psf = GaussianPSF( fwhm_pixels=1.5 )

    # Try both even an odd side lengths
    # Try lengths that will have even and odd nx//2 to test 0.5 rounding edge cases
    # Try offsets <1 pixel and >1 pixel
    # Try offsets in both directions (except negative offsets aren't implemented)
    # Try offsets with fractional parts below, at and above 0.5

    # For centered, if there are 8 pixels, then the center is
    #      here
    #        |
    # X X X X X X X X
    # That position is 3.5, because the first pixel is numbered 0, but
    # (perversely) astronomers have adopted the convention that .0 is
    # the middle of a pixel.  (What makes this perverse is that an image
    # is not *sampling* the sky, it's integrating the sky in pixel-sized
    # boxes.  This convention means that the left side of the image is
    # at position -0.5, and the right side is at position 7.5.  If
    # astronomers had been sane and chosen .5 to be the middle of the
    # pixel, then the array would have covered the range [0,8), which
    # is just more intuitive.  Alas, the convention is what it
    # is.)
    #
    # If there are 9 pixels, then the center is
    #       here
    #         |
    # X X X X X X X X X
    # Which is the middle of pixel 4, for 0-indexed pixels, and
    # again, perversely, astronomers have decided that .0 is the
    # middle of the pixel, so this is 4.0.
    #
    # So, to generalize, for even-length sides, the center is at (n//2 -
    # 0.5), and for odd-length sides, the center is at (n//2) (both for
    # 0-offset arrays).

    pixes = [ 511, 512, 513, 514 ]
    ctrs = [ 255., 255.5, 256., 256.5 ]
    intparts = [ 0, 1, -1, 2, -2 ]
    fracparts = [ 0, 0.1, -0.1, 0.5, -0.5, 0.7, -0.7 ]

    # 6 nested for loops FTW!  N⁶ baby!
    # Anally test all kinds of combinations.  It's fast enough.  (Takes ~14 seconds.)
    for ( ypix, yctr ) in zip( pixes, ctrs ):
        for yintpart in intparts:
            for yfracpart in fracparts:
                for ( xpix, xctr ) in zip( pixes, ctrs ):
                    for xintpart in intparts:
                        for xfracpart in fracparts:
                            if ( ( xintpart == 0 ) and ( xfracpart == 0. )
                                 and ( yintpart == 0 ) and ( yfracpart == 0. )
                                ):
                                img = psf.get_centered_psf( xpix, ypix )
                                cy, cx = scipy.ndimage.center_of_mass( img )
                                assert cx == pytest.approx( xctr, abs=0.01 )
                                assert cy == pytest.approx( yctr, abs=0.01 )

                            offx = xintpart + xfracpart
                            offy = yintpart + yfracpart

                            SCLogger.debug( f"xpix={xpix}, xctr={xctr}, offx={offx} ; "
                                            f"ypix={ypix}, yctr={yctr}, offy={offy}" )

                            img = psf.get_centered_psf( xpix, ypix, offx=offx, offy=offy )
                            cy, cx = scipy.ndimage.center_of_mass( img )
                            assert cx == pytest.approx( xctr + offx, abs=0.01 )
                            assert cy == pytest.approx( yctr + offy, abs=0.01 )


# This test writes out some fits files for looking at actual rendered PSFs,
#   so one can understand where it's centering them.  It doesn't test anything,
#   so it should only be run if somebody wants to run interactive tests.
@pytest.mark.skipif( not env_as_bool("INTERACTIVE"), reason='Set INTERACTIVE=1 to run this "test"' )
def test_write_some_fits_clips( ztf_filepaths_image_sources_psf ):

    for i in range( 2 ):
        if i == 0:
            image, _weight, _flags, _, psfpath, psfxmlpath = ztf_filepaths_image_sources_psf
            hdul = fits.open( image )
            wid = hdul[0].header['NAXIS1']
            hei = hdul[0].header['NAXIS2']
            psf = PSFExPSF()
            psf.image_shape = ( hei, wid )
            psf.load( psfpath=psfpath, psfxmlpath=psfxmlpath )
            prefix = "psfex"
        else:
            psf = GaussianPSF( fwhm_pixels=2.1 )
            psf.image_shape = ( 1024, 1024 )
            prefix = "gauss"

        # Pass in both even and odd integral parts to make sure
        #   that rounding conventions aren't leading to
        #   inconsistent results.  (The zp5 and op5  clips
        #   should look ~identical.)

        zp4 = psf.get_clip( 100.4, 100.4 )
        zp5 = psf.get_clip( 100.5, 100.5 )
        zp6 = psf.get_clip( 100.6, 100.6 )
        op4 = psf.get_clip( 101.4, 101.4 )
        op5 = psf.get_clip( 101.5, 101.5 )
        op6 = psf.get_clip( 101.6, 101.6 )

        fits.writeto( f"{prefix}_zp4.fits", zp4, overwrite=True )
        fits.writeto( f"{prefix}_zp5.fits", zp5, overwrite=True )
        fits.writeto( f"{prefix}_zp6.fits", zp6, overwrite=True )
        fits.writeto( f"{prefix}_op4.fits", op4, overwrite=True )
        fits.writeto( f"{prefix}_op5.fits", op5, overwrite=True )
        fits.writeto( f"{prefix}_op6.fits", op6, overwrite=True )
