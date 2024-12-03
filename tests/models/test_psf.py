import pytest
import io
import psutil
import time
import uuid
import random
import math
import pathlib
import subprocess

import sqlalchemy as sa
import psycopg2.errors

import numpy as np
from scipy.integrate import dblquad

import astropy.io
from astropy.io import fits

from util.config import Config
from util.logger import SCLogger
from models.base import SmartSession, FileOnDiskMixin, CODE_ROOT, get_archive_object
from models.provenance import Provenance
from models.psf import PSF

from util.util import env_as_bool


class PSFPaletteMaker:
    def __init__( self, round=False ):
        tempname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
        self.imagename = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.fits'
        self.weightname = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.weight.fits'
        self.flagsname = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.flags.fits'
        self.catname = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.cat'
        self.psfname = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.psf'
        self.psfxmlname = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.psf.xml'

        self.nx = 1024
        self.ny = 1024

        self.clipwid = 17

        self.flux = 200000.
        self.noiselevel = 5.

        self.x0 = self.nx/2.
        self.sigx0 = 1.
        if round:
            self.sigxx = 0.
            self.sigxy = 0.
        else:
            self.sigxx = 0.25 / self.nx
            self.sigxy = 0.

        self.y0 = self.ny/2.
        if round:
            self.sigy0 = 1.
            self.sigyx = 0.
            self.sigyy = 0.
        else:
            self.sigy0 = 1.5
            self.sigyx = -0.25 / self.ny
            self.sigyy = 0.

        self.theta0 = 0.
        if round:
            self.thetax = 0.
            self.thetay = 0.
        else:
            self.thetax = 0.
            self.thetay = math.pi / 4. / self.nx

        # Positions where we're going to put the PSFs.  Want to have
        # about 10 0 of them, but also don't want them all to fall right
        # at the center of the pixel (hence the nonintegral spacing)
        self.xpos = np.arange( 25., 1000., 102.327 )
        self.ypos = np.arange( 25., 1000., 102.327 )

    @staticmethod
    def psffunc ( yr, xr, sigx, sigy, theta ):
        xrot =  xr * math.cos(theta) + yr * math.sin(theta)
        yrot = -xr * math.sin(theta) + yr * math.cos(theta)
        return 1/(2*math.pi*sigx*sigy) * math.exp( -( xrot**2/(2.*sigx**2) + yrot**2/(2.*sigy**2) ) )

    def psfpixel( self, x, y, xi, yi ):
        sigx = self.sigx0 + (x - self.x0) * self.sigxx + (y - self.y0) * self.sigxy
        sigy = self.sigy0 + (x - self.x0) * self.sigyx + (y - self.y0) * self.sigyy
        theta = self.theta0 + (x - self.x0) * self.thetax + (y - self.y0) * self.thetay

        res = dblquad( PSFPaletteMaker.psffunc, xi-x-0.5, xi-x+0.5, yi-y-0.5, yi-y+0.5, args=( sigx, sigy, theta ) )
        return res[0]

    def make_psf_palette( self ):
        self.img = np.zeros( ( self.nx, self.ny ) )

        for i, xc in enumerate( self.xpos ):
            xi0 = int( math.floor(xc)+0.5 )
            SCLogger.info( f"Making psf palette, on x {i} of {len(self.xpos)}" )
            for yc in self.ypos:
                yi0 = int( math.floor(yc)+0.5 )
                for xi in range(xi0 - (self.clipwid//2), xi0 + self.clipwid//2 + 1):
                    for yi in range(yi0 - (self.clipwid//2), yi0 + self.clipwid//2 + 1):
                        self.img[yi, xi] = self.flux * self.psfpixel( xc, yc, xi, yi )

        # Have to have some noise in there, or sextractor will choke on the image
        self.img += np.random.normal( 0., self.noiselevel, self.img.shape )

        hdu = fits.PrimaryHDU( data=self.img )
        hdu.writeto( self.imagename, overwrite=True )
        hdu = fits.PrimaryHDU( data=np.zeros_like( self.img, dtype=np.uint8 ) )
        hdu.writeto( self.flagsname, overwrite=True )
        hdu = fits.PrimaryHDU( data=np.full( self.img.shape, 1. / ( self.noiselevel**2 ) ) )
        hdu.writeto( self.weightname, overwrite=True )

    def extract_and_psfex( self ):
        astromatic_dir = None
        cfg = Config.get()
        if cfg.value( 'astromatic.config_dir' ) is not None:
            astromatic_dir = pathlib.Path( cfg.value( 'astromatic.config_dir' ) )
        elif cfg.value( 'astromatic.config_subdir' ) is not None:
            astromatic_dir = pathlib.Path( CODE_ROOT ) / cfg.value( 'astromatic.config_subdir' )
        if astromatic_dir is None:
            raise FileNotFoundError( "Can't figure out where astromatic config directory is" )
        if not astromatic_dir.is_dir():
            raise FileNotFoundError( f"Astromatic config dir {str(astromatic_dir)} doesn't exist "
                                     f"or isn't a directory." )

        conv = astromatic_dir / "default.conv"
        nnw = astromatic_dir / "default.nnw"
        paramfile = astromatic_dir / "sourcelist_sextractor.param"

        SCLogger.info( "Running sextractor..." )
        # Run sextractor to give psfex something to do
        command = [ 'source-extractor',
                    '-CATALOG_NAME', self.catname,
                    '-CATALOG_TYPE', 'FITS_LDAC',
                    '-PARAMETERS_NAME', paramfile,
                    '-FILTER', 'Y',
                    '-FILTER_NAME', str(conv),
                    '-WEIGHT_TYPE', 'MAP_WEIGHT',
                    '-RESCALE_WEIGHTS', 'N',
                    '-WEIGHT_IMAGE', self.weightname,
                    '-FLAG_IMAGE', self.flagsname,
                    '-FLAG_TYPE', 'OR',
                    '-PHOT_APERTURES', '4.7',
                    '-SATUR_LEVEL', '1000000',
                    '-STARNNW_NAME', nnw,
                    '-BACK_TYPE', 'MANUAL',
                    '-BACK_VALUE', '0.0',
                    self.imagename
                   ]
        res = subprocess.run( command, capture_output=True, timeout=60 )
        assert res.returncode == 0

        SCLogger.info( "Runing psfex..." )
        # Run psfex to get the psf and psfxml files
        command = [ 'psfex',
                    '-PSF_SIZE', '31',
                    '-SAMPLE_FWHMRANGE', f'1.0,10.0',
                    '-SAMPLE_VARIABILITY', '0.5',
                    '-CHECKPLOT_DEV', 'NULL',
                    '-CHECKPLOT_TYPE', 'NONE',
                    '-CHECKIMAGE_TYPE', 'NONE',
                    '-WRITE_XML', 'Y',
                    '-XML_NAME', self.psfxmlname,
                    '-XML_URL', 'file:///usr/share/psfex/psfex.xsl',
                    self.catname
                   ]
        res = subprocess.run( command, capture_output=True, timeout=60 )
        assert res.returncode == 0

        self.psf = PSF( format='psfex' )
        self.psf.load( psfpath=self.psfname, psfxmlpath=self.psfxmlname )
        self.psf.fwhm_pixels = float( self.psf.header['PSF_FWHM'] )

    def cleanup( self ):
        self.imagename.unlink( missing_ok=True )
        self.weightname.unlink( missing_ok=True )
        self.flagsname.unlink( missing_ok=True )
        self.catname.unlink( missing_ok=True )
        self.psfname.unlink( missing_ok=True )
        self.psfxmlname.unlink( missing_ok=True )


@pytest.fixture(scope="module")
def round_psf_palette():
    palette = PSFPaletteMaker( round=True )
    palette.make_psf_palette()
    palette.extract_and_psfex()

    yield palette

    palette.cleanup()


@pytest.fixture(scope="module")
def psf_palette():
    palette = PSFPaletteMaker( round=False )
    palette.make_psf_palette()
    palette.extract_and_psfex()

    yield palette

    palette.cleanup()


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
    im, wt, fl, sr, psfpath, psfxmlpath = ztf_filepaths_image_sources_psf
    psf = PSF( format='psfex' )
    psf.load( psfpath=psfpath, psfxmlpath=psfxmlpath )
    check_example_psfex_psf_values( psf )


def test_write_psfex_psf( ztf_filepaths_image_sources_psf ):
    image, weight, flags, sourcepath, psfpath, psfxmlpath = ztf_filepaths_image_sources_psf
    psf = PSF( format='psfex' )
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
        psf = PSF( format='psfex' )
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

def test_save_psf( ztf_datastore_uncommitted, provenance_base, provenance_extra ):
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
        psf2 = PSF(format='psfex')
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
    assert clip.shape == ( 19, 19 )                        # (15, 15) for round psf (resampling ends up different)
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
            ix = int( np.floor( x + 0.5 ) )
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

    assert chisq / n < 25.0
