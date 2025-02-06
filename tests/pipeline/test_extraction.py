# TODO -- write a test to make sure that passing a wcs to extract_sources_sextractor
#   really uses the updated wcs!!!

import pytest
import io
import os
import re
import uuid
import pathlib
import shutil
import random

import numpy as np

from astropy.io import votable, fits

from models.base import FileOnDiskMixin, CODE_ROOT
from models.provenance import Provenance
from models.image import Image

from tests.conftest import SKIP_WARNING_TESTS

# from util.logger import SCLogger


def test_sep_find_sources_in_small_image(decam_small_image, extractor, blocking_plots):
    det = extractor
    det.pars.method = 'sep'
    det.pars.subtraction = False
    det.pars.threshold = 3.0
    det.pars.test_parameter = uuid.uuid4().hex
    sources, _, _, _ = det.extract_sources(decam_small_image)

    assert sources.num_sources == 46
    assert max(sources.data['flux']) == 52382.15625
    assert abs(np.mean(sources.data['x']) - 256) < 10
    assert abs(np.mean(sources.data['y']) - 256) < 25
    assert 2.0 < np.median(sources.data['rhalf']) < 2.5

    if blocking_plots:  # use this for debugging / visualization only!
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        data = decam_small_image.data
        m, s = np.mean(data), np.std(data)

        obj = sources.data

        _, ax = plt.subplots()
        ax.imshow(data, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
        for i in range(len(obj)):
            e = Ellipse(xy=(obj['x'][i], obj['y'][i]), width=6 * obj['a'][i], height=6 * obj['b'][i],
                        angle=obj['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)

        plt.show(block=True)

    # increasing the threshold should find fewer sources
    det.pars.threshold = 7.5
    sources2, _, _, _ = det.extract_sources(decam_small_image)
    assert sources2.num_sources < sources.num_sources

    # flux will change with new threshold, but not by more than 10%
    assert abs( max(sources2.data['flux']) - max(sources.data['flux'])) / max(sources.data['flux']) < 0.1

    # fewer sources also means the mean position will be further from center
    assert abs(np.mean(sources2.data['x']) - 256) < 35.
    assert abs(np.mean(sources2.data['y']) - 256) < 50.

    assert 3.0 < np.median(sources2.data['rhalf']) < 3.5


def test_sep_save_source_list(decam_small_image, provenance_base, extractor):
    decam_small_image.provenance_id = provenance_base.id

    extractor.pars.method = 'sep'
    extractor.pars.subtraction = False
    extractor.pars.threshold = 3.0
    extractor.pars.test_parameter = uuid.uuid4().hex
    sources, _, _, _ = extractor.extract_sources(decam_small_image)
    prov = Provenance(
        process='extraction',
        code_version_id=provenance_base.code_version_id,
        parameters=extractor.pars.get_critical_pars(),
        upstreams=[ Provenance.get( decam_small_image.provenance_id ) ],
        is_testing=True,
    )
    prov.insert_if_needed()
    sources.provenance_id = prov.id

    try:  # cleanup file / DB at the end
        sources.save( image=decam_small_image )
        filename = sources.get_fullpath()

        assert os.path.isfile(filename)

        # check the naming convention
        assert re.search(r'.*/\d{3}/c4d_\d{8}_\d{6}_.+_.+_.+_.{6}.sources_.{6}\.npy', filename)

        # check the file contents can be loaded successfully
        data = np.load(filename)
        assert np.array_equal(data, sources.data)

        decam_small_image.save()  # pretend to save this file
        decam_small_image.insert()
        sources.insert()

    finally:
        if 'sources' in locals():
            sources.delete_from_disk_and_database()


# This is running sextractor in one particular way that is used by more than one test
def run_sextractor( image, extractor ):
    tempname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    sourcelist, bkg, bkgsig = extractor._run_sextractor_once( image, tempname=tempname )
    sourcefile = pathlib.Path( FileOnDiskMixin.temp_path ) / f"{tempname}.sources.fits"
    imagefile =  pathlib.Path( FileOnDiskMixin.temp_path ) / f"{tempname}.fits"
    assert not imagefile.exists()
    assert sourcefile.exists()

    return sourcelist, sourcefile, bkg, bkgsig


def test_sextractor_extract_once( decam_datastore_through_preprocessing, extractor ):
    ds = decam_datastore_through_preprocessing
    try:
        extractor.pars.method = 'sextractor'
        extractor.pars.subtraction = False
        extractor.pars.apers = [ 5. ]
        extractor.pars.threshold = 4.5
        extractor.pars.test_parameter = uuid.uuid4().hex
        sourcelist, sourcefile, bkg, bkgsig = run_sextractor(ds.image, extractor)

        assert bkg == pytest.approx( 650.0, abs=0.1 )
        assert bkgsig == pytest.approx( 13.2673, abs=0.01 )

        assert sourcelist.num_sources == 1311
        assert len(sourcelist.data) == sourcelist.num_sources
        assert sourcelist.aper_rads == [ 5. ]

        assert sourcelist.info['SEXAPED1'] == 10.0
        assert sourcelist.info['SEXAPED2'] == 0.
        assert sourcelist.info['SEXBKGND'] == pytest.approx( 650.0, abs=0.1 )

        snr = sourcelist.apfluxadu()[0] / sourcelist.apfluxadu()[1]
        # SCLogger.info(
        #     f'\nsourcelist.x.min()= {sourcelist.x.min()}'
        #     f'\nsourcelist.x.max()= {sourcelist.x.max()}'
        #     f'\nsourcelist.y.min()= {sourcelist.y.min()}'
        #     f'\nsourcelist.y.max()= {sourcelist.y.max()}'
        #     f'\nsourcelist.errx.min()= {sourcelist.errx.min()}'
        #     f'\nsourcelist.errx.max()= {sourcelist.errx.max()}'
        #     f'\nsourcelist.erry.min()= {sourcelist.erry.min()}'
        #     f'\nsourcelist.erry.max()= {sourcelist.erry.max()}'
        #     f'\nsourcelist.apfluxadu()[0].min()= {sourcelist.apfluxadu()[0].min()}'
        #     f'\nsourcelist.apfluxadu()[0].max()= {sourcelist.apfluxadu()[0].max()}'
        #     f'\nsnr.min()= {snr.min()}'
        #     f'\nsnr.max()= {snr.max()}'
        #     f'\nsnr.mean()= {snr.mean()}'
        #     f'\nsnr.std()= {snr.std()}'
        # )
        assert sourcelist.x.min() == pytest.approx( 15.44, abs=0.1 )
        assert sourcelist.x.max() == pytest.approx( 2039.97, abs=0.1 )
        assert sourcelist.y.min() == pytest.approx( 25.18, abs=0.1 )
        assert sourcelist.y.max() == pytest.approx( 4087.88, abs=0.1 )
        assert sourcelist.errx.min() == pytest.approx( 0.00182, abs=1e-4 )
        assert sourcelist.errx.max() == pytest.approx( 1.298, abs=0.01 )
        assert sourcelist.erry.min() == pytest.approx( 0.00096, abs=1e-4 )
        assert sourcelist.erry.max() == pytest.approx( 1.306, abs=0.01 )
        assert ( np.sqrt( sourcelist.varx ) == sourcelist.errx ).all()
        assert ( np.sqrt( sourcelist.vary ) == sourcelist.erry ).all()
        assert sourcelist.apfluxadu()[0].min() == pytest.approx( -93.90485, rel=1e-5 )
        assert sourcelist.apfluxadu()[0].max() == pytest.approx( 2624645.2, rel=1e-5 )
        assert snr.min() == pytest.approx( -0.806, abs=0.1 )
        assert snr.max() == pytest.approx( 2222.799, abs=1. )
        assert snr.mean() == pytest.approx( 56.63, abs=0.1 )
        assert snr.std() == pytest.approx( 181.7, abs=1. )

        # Test multiple apertures
        sourcelist, _, _ = extractor._run_sextractor_once( ds.image, apers=[ 2., 5. ])

        assert sourcelist.num_sources == 1311    # It *finds* the same things
        assert len(sourcelist.data) == sourcelist.num_sources
        assert sourcelist.aper_rads == [ 2., 5. ]

        assert sourcelist.info['SEXAPED1'] == 4.0
        assert sourcelist.info['SEXAPED2'] == 10.0
        assert sourcelist.info['SEXBKGND'] == pytest.approx( 650.0, abs=0.1 )

        # SCLogger.info(
        #     f'\nsourcelist.x.min()= {sourcelist.x.min()}'
        #     f'\nsourcelist.x.max()= {sourcelist.x.max()}'
        #     f'\nsourcelist.y.min()= {sourcelist.y.min()}'
        #     f'\nsourcelist.y.max()= {sourcelist.y.max()}'
        #     f'\nsourcelist.apfluxadu(apnum=1)[0].min()= {sourcelist.apfluxadu(apnum=1)[0].min()}'
        #     f'\nsourcelist.apfluxadu(apnum=1)[0].max()= {sourcelist.apfluxadu(apnum=1)[0].max()}'
        #     f'\nsourcelist.apfluxadu(apnum=0)[0].min()= {sourcelist.apfluxadu(apnum=0)[0].min()}'
        #     f'\nsourcelist.apfluxadu(apnum=0)[0].max()= {sourcelist.apfluxadu(apnum=0)[0].max()}'
        # )
        assert sourcelist.x.min() == pytest.approx( 15.44, abs=0.1 )
        assert sourcelist.x.max() == pytest.approx( 2039.97, abs=0.1 )
        assert sourcelist.y.min() == pytest.approx( 25.18, abs=0.1 )
        assert sourcelist.y.max() == pytest.approx( 4087.88, abs=0.1 )
        assert sourcelist.apfluxadu(apnum=1)[0].min() == pytest.approx( -93.9048, rel=1e-5 )
        assert sourcelist.apfluxadu(apnum=1)[0].max() == pytest.approx( 2624645.25, rel=1e-5 )
        assert sourcelist.apfluxadu(apnum=0)[0].min() == pytest.approx( 155.472946, rel=1e-5 )
        assert sourcelist.apfluxadu(apnum=0)[0].max() == pytest.approx( 513817.81, rel=1e-5 )

    finally:  # cleanup temporary file
        if 'sourcefile' in locals():
            sourcefile.unlink( missing_ok=True )


# Egg, meet chicken.  decam_datastore_through_extraction will have
#   run psfex already, because the extraction step does both
#   source extraction and psf estimation.  But, we need the
#   source extraction done to be able to test psfex, so, whatever.
def test_run_psfex( decam_datastore_through_extraction, extractor ):
    ds = decam_datastore_through_extraction
    sourcelist = ds.sources
    tempname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    temp_path = pathlib.Path( FileOnDiskMixin.temp_path )
    tmpsourcefile =  temp_path / f'{tempname}.sources.fits'
    tmppsffile = temp_path / f"{tempname}.sources.psf"
    tmppsfxmlfile = temp_path / f"{tempname}.sources.psf.xml"
    shutil.copy2( sourcelist.get_fullpath(), tmpsourcefile )

    try:
        extractor.pars.method = 'sextractor'
        extractor.pars.subtraction = False
        extractor.pars.threshold = 4.5
        psf = extractor._run_psfex( tempname, ds.image )
        assert psf._header['PSFAXIS1'] == 27
        assert psf._header['PSFAXIS2'] == 27
        assert psf._header['PSFAXIS3'] == 6
        assert psf._header['PSF_SAMP'] == pytest.approx( 0.90, abs=0.01 )
        assert psf._header['CHI2'] == pytest.approx( 0.97, abs=0.1 )
        bio = io.BytesIO( psf._info.encode( 'utf-8' ) )
        psfstats = votable.parse( bio ).get_table_by_index(1)
        assert psfstats.array['FWHM_FromFluxRadius_Max'] == pytest.approx( 4.221, abs=0.01 )
        assert not tmppsffile.exists()
        assert not tmppsfxmlfile.exists()

        psf = extractor._run_psfex( tempname, ds.image, do_not_cleanup=True )
        assert tmppsffile.exists()
        assert tmppsfxmlfile.exists()
        tmppsffile.unlink()
        tmppsfxmlfile.unlink()

        psf = extractor._run_psfex( tempname, ds.image, psf_size=26 )
        assert psf._header['PSFAXIS1'] == 31
        assert psf._header['PSFAXIS1'] == 31

    finally:
        tmpsourcefile.unlink( missing_ok=True )
        tmppsffile.unlink( missing_ok=True )
        tmppsfxmlfile.unlink( missing_ok=True )


def test_extract_sources_sextractor( decam_datastore_through_preprocessing,
                                     extractor, provenance_base, data_dir, blocking_plots ):
    ds = decam_datastore_through_preprocessing

    extractor.pars.method = 'sextractor'
    extractor.measure_psf = True
    extractor.pars.threshold = 5.0
    sources, psf, bkg, bkgsig = extractor.extract_sources( ds.image )

    assert bkg == pytest.approx( 650.0, abs=0.1 )
    assert bkgsig == pytest.approx( 13.2673, abs=0.01 )

    # Make True to write some ds9 regions
    if os.getenv('INTERACTIVE', False):
        basepath = os.path.join(CODE_ROOT, 'tests/plots/test_sources')
        sources.ds9_regfile( basepath + '_stars.reg', color='green', radius=4, whichsources='stars' )
        sources.ds9_regfile( basepath + '_nonstars.reg', color='red', radius=4, whichsources='nonstars' )

        # Manually write one that uses CLASS_STAR instead of SPREAD_MODEL
        use = sources.data['CLASS_STAR'] > 0.5
        with open( basepath + '_class_star.reg', 'w' ) as ofp:
            for x, y, use in zip( sources.x, sources.y, use ):
                if use:
                    ofp.write( f"image;circle({x+1},{y+1},6) # color=blue width=2\n" )

    assert sources.num_sources > 1400
    assert sources.num_sources == len(sources.data)
    expected_radii = np.array([1.0, 2.0, 3.0, 5.0]) * psf.fwhm_pixels
    assert sources.aper_rads == pytest.approx(expected_radii, abs=0.01 )
    assert sources.inf_aper_num == -1
    assert psf.fwhm_pixels == pytest.approx( 4.126, abs=0.01 )
    assert psf.fwhm_pixels == pytest.approx( psf.header['PSF_FWHM'], rel=1e-5 )
    assert psf.data.shape == ( 6, 25, 25 )
    assert psf.sources_id == sources.id

    assert sources.apfluxadu()[0].min() == pytest.approx( 490.52, rel=0.01 )
    assert sources.apfluxadu()[0].max() == pytest.approx( 1876754., rel=0.01 )
    assert sources.apfluxadu()[0].mean() == pytest.approx( 17757., rel=0.01 )
    assert sources.apfluxadu()[0].std() == pytest.approx( 117502., rel=0.01 )

    assert sources.good.sum() == pytest.approx(975, rel=0.01)
    # This is what you get with CLASS_STAR; you'll get different values with SPREAD_MODEL
    assert sources.is_star.sum() == pytest.approx(479, rel=0.01)
    assert ( sources.good & sources.is_star ).sum() == pytest.approx(337, abs=5)

    try:  # make sure saving the PSF and source list goes as expected, and cleanup at the end
        sources.provenance_id = provenance_base.id
        sources.save()
        psf.save( image=ds.image, sources=sources )
        assert re.match(r'\d{3}/c4d_\d{8}_\d{6}_S2_r_Sci_.{6}.psf_.{6}', psf.filepath)
        assert os.path.isfile( os.path.join(data_dir, psf.filepath + '.fits') )

        assert re.match(r'\d{3}/c4d_\d{8}_\d{6}_S2_r_Sci_.{6}.sources_.{6}.fits', sources.filepath)
        assert os.path.isfile(os.path.join(data_dir, sources.filepath))

        # TODO: add background object here

    finally:  # cleanup
        psf.delete_from_disk_and_database()
        sources.delete_from_disk_and_database()


def test_compare_sep_sextractor( extractor ):
    # This test was created to address Issue #341

    # Create a small image with some Gaussians on it to see how sep and sextractor compare
    skynoise = 20.
    sig = 1.75
    rng = np.random.default_rng( 42 )
    data = rng.normal( scale=skynoise, size=(256, 256) )
    var = np.full( (256, 256), skynoise**2 )

    # Use non-integral spacing so that gaussians end up centered differently relative to pixels
    xs, ys = np.meshgrid( np.arange( 25., 225., 21.173 ), np.arange( 25., 225., 21.173 ) )
    xs = xs.flatten()
    ys = ys.flatten()
    clipxs, clipys = np.meshgrid( np.arange(25)-12, np.arange(25)-12 )
    clipxs = clipxs.flatten()
    clipys = clipys.flatten()
    fluxen = np.arange( len(xs) ) * ( 200.*skynoise / len(xs) ) + 5.*skynoise
    for x, y, flux in zip( xs, ys, fluxen ):
        xc = int( round( x ) )
        yc = int( round( y ) )
        star = ( flux / np.sqrt(2.*np.pi*sig**2) *
                   np.exp( -( clipxs**2/(2.*sig**2) + clipys**2/(2.*sig**2) ) ) )
        star += rng.normal( scale=np.sqrt(star) )
        starvar = star
        starvar[ starvar <= 0. ] = 0.
        data[ clipys+yc, clipxs+xc ] += star
        var[ clipys+yc, clipxs+xc ] += starvar

    image = Image( instrument='DECam' )
    image.data = data
    image.header = fits.Header( { 'SATURATE': 40000. } )
    image.weight = 1./var
    image.flags = np.zeros_like( image.data, dtype=np.uint16 )

    sexsrc, _bkg, _bkgsig = extractor._run_sextractor_once( image )
    sepsrc = extractor.extract_sources_sep( image )

    # They won't necessarily find all the same things, because
    # thresholds and such.  And, anyway, we don't expect the order of
    # sources to be the same.  So, do a match.
    dist = np.sqrt( ( sexsrc.x[:, np.newaxis] - sepsrc.x[np.newaxis, :] )**2 +
                    ( sexsrc.y[:, np.newaxis] - sepsrc.y[np.newaxis, :] )**2 )
    # Do a very conservative matching.
    wsex, wsep = np.where( dist < 5. )

    # Make sure positions are consistent.  Among other things,
    #   this tells us that our handling of 1-offsets, x and y ordering,
    #   are consistent.
    assert np.all( np.fabs( sepsrc.x[wsep] - sexsrc.x[wsex] ) < 0.2 )
    assert np.all( np.fabs( sepsrc.y[wsep] - sexsrc.y[wsex] ) < 0.2 )
    assert np.all( dist[ wsex, wsep ] < 0.2 )

    # Make sure fluxes are consistent
    sepflux = sepsrc.data['flux']
    # dsepflux = sepsrc.data['dflux'] ... sep doesn't give a flux uncertainty???
    sexflux, dsexflux = sexsrc.apfluxadu()
    sepflux = sepflux[wsep]
    # dsepflux = dsepflux[wsep]
    sexflux = sexflux[wsex]
    dsexflux = dsexflux[wsex]

    # ... really I was looking for relative consistency, but I'm
    # surprsied that they're this close, given that the sextrator
    # aperture was kinda random and is not aperture corrected, and I
    # don't know what sep is doing.
    assert ( sexflux / sepflux ).mean() == pytest.approx( 1.0, abs=0.02 )
    assert ( sexflux / sepflux ).std() < 0.04
    assert ( sexflux / sepflux ).min() > 0.9
    assert ( sexflux / sepflux ).max() < 1.15


def test_warnings_and_exceptions( decam_datastore_through_preprocessing ):
    ds = decam_datastore_through_preprocessing
    extractor = ds._pipeline.extractor

    if not SKIP_WARNING_TESTS:
        extractor.pars.inject_warnings = 1
        ds._pipeline.make_provenance_tree(ds)

        with pytest.warns(UserWarning) as record:
            extractor.run( ds )
        assert len(record) > 0
        assert any("Warning injected by pipeline parameters in process 'detection'." in str(w.message) for w in record)

    ds.sources = None
    extractor.pars.inject_warnings = 0
    extractor.pars.inject_exceptions = 1
    ds._pipeline.make_provenance_tree(ds)
    with pytest.raises(Exception, match="Exception injected by pipeline parameters in process 'detection'."):
        ds = extractor.run( ds )
