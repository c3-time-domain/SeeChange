import pytest
import io
import os
import re
import uuid
import pathlib
import shutil
import random

import numpy as np
import sqlalchemy as sa

from astropy.io import votable

from models.base import SmartSession, FileOnDiskMixin, get_archive_object, CODE_ROOT
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList


def test_sep_find_sources_in_small_image(decam_small_image, extractor, blocking_plots):
    det = extractor
    det.pars.method = 'sep'
    det.pars.subtraction = False
    det.pars.threshold = 3.0
    det.pars.test_parameter = uuid.uuid4().hex
    sources, _, _, _ = det.extract_sources(decam_small_image)

    assert sources.num_sources == 158
    assert max(sources.data['flux']) == 3670450.0
    assert abs(np.mean(sources.data['x']) - 256) < 10
    assert abs(np.mean(sources.data['y']) - 256) < 10
    assert 2.0 < np.median(sources.data['rhalf']) < 2.5

    if blocking_plots:  # use this for debugging / visualization only!
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        data = decam_small_image.data
        m, s = np.mean(data), np.std(data)

        obj = sources.data

        fig, ax = plt.subplots()
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
    assert abs(np.mean(sources2.data['x']) - 256) < 25
    assert abs(np.mean(sources2.data['y']) - 256) < 25

    assert 2.0 < np.median(sources2.data['rhalf']) < 2.5


def test_sep_save_source_list(decam_small_image, provenance_base, extractor):
    decam_small_image.provenance = provenance_base

    extractor.pars.method = 'sep'
    extractor.pars.subtraction = False
    extractor.pars.threshold = 3.0
    extractor.pars.test_parameter = uuid.uuid4().hex
    sources, _, _, _ = extractor.extract_sources(decam_small_image)
    prov = Provenance(
        process='extraction',
        code_version=provenance_base.code_version,
        parameters=extractor.pars.get_critical_pars(),
        upstreams=[decam_small_image.provenance],
        is_testing=True,
    )
    sources.provenance = prov

    try:  # cleanup file / DB at the end
        sources.save()
        filename = sources.get_fullpath()

        assert os.path.isfile(filename)

        # check the naming convention
        assert re.search(r'.*/\d{3}/c4d_\d{8}_\d{6}_.+_.+_.+_.{6}.sources_.{6}\.npy', filename)

        # check the file contents can be loaded successfully
        data = np.load(filename)
        assert np.array_equal(data, sources.data)

        with SmartSession() as session:
            sources = session.merge( sources )
            decam_small_image.save()  # pretend to save this file
            decam_small_image.exposure.save()
            session.commit()
            image_id = decam_small_image.id
            sources_id = sources.id

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


def test_sextractor_extract_once( decam_datastore, extractor ):
    try:
        extractor.pars.method = 'sextractor'
        extractor.pars.subtraction = False
        extractor.pars.apers = [ 5. ]
        extractor.pars.threshold = 4.5
        extractor.pars.test_parameter = uuid.uuid4().hex
        sourcelist, sourcefile, bkg, bkgsig = run_sextractor(decam_datastore.image, extractor)

        assert bkg == pytest.approx( 179.82, abs=0.1 )
        assert bkgsig == pytest.approx( 7.533, abs=0.01 )

        assert sourcelist.num_sources == 5611
        assert len(sourcelist.data) == sourcelist.num_sources
        assert sourcelist.aper_rads == [ 5. ]
        assert sourcelist._inf_aper_num is None
        assert sourcelist.inf_aper_num == 0

        assert sourcelist.info['SEXAPED1'] == 10.0
        assert sourcelist.info['SEXAPED2'] == 0.
        assert sourcelist.info['SEXBKGND'] == pytest.approx( 179.8, abs=0.1 )

        assert sourcelist.x.min() == pytest.approx( 16.0, abs=0.1 )
        assert sourcelist.x.max() == pytest.approx( 2039.6, abs=0.1 )
        assert sourcelist.y.min() == pytest.approx( 16.264, abs=0.1 )
        assert sourcelist.y.max() == pytest.approx( 4087.9, abs=0.1 )
        assert sourcelist.errx.min() == pytest.approx( 0.0005, abs=1e-4 )
        assert sourcelist.errx.max() == pytest.approx( 1.0532, abs=0.01 )
        assert sourcelist.erry.min() == pytest.approx( 0.001, abs=1e-3 )
        assert sourcelist.erry.max() == pytest.approx( 0.62, abs=0.01 )
        assert ( np.sqrt( sourcelist.varx ) == sourcelist.errx ).all()
        assert ( np.sqrt( sourcelist.vary ) == sourcelist.erry ).all()
        assert sourcelist.apfluxadu()[0].min() == pytest.approx( -656.8731, rel=1e-5 )
        assert sourcelist.apfluxadu()[0].max() == pytest.approx( 2850920.0, rel=1e-5 )
        snr = sourcelist.apfluxadu()[0] / sourcelist.apfluxadu()[1]
        assert snr.min() == pytest.approx( -9.91, abs=0.1 )
        assert snr.max() == pytest.approx( 2348.2166, abs=1. )
        assert snr.mean() == pytest.approx( 146.80, abs=0.1 )
        assert snr.std() == pytest.approx( 285.4, abs=1. )

        # Test multiple apertures
        sourcelist, _, _ = extractor._run_sextractor_once( decam_datastore.image, apers=[2, 5] )

        assert sourcelist.num_sources == 5611    # It *finds* the same things
        assert len(sourcelist.data) == sourcelist.num_sources
        assert sourcelist.aper_rads == [ 2., 5. ]
        assert sourcelist._inf_aper_num is None
        assert sourcelist.inf_aper_num == 1

        assert sourcelist.info['SEXAPED1'] == 4.0
        assert sourcelist.info['SEXAPED2'] == 10.0
        assert sourcelist.info['SEXBKGND'] == pytest.approx( 179.8, abs=0.1 )
        assert sourcelist.x.min() == pytest.approx( 16.0, abs=0.1 )
        assert sourcelist.x.max() == pytest.approx( 2039.6, abs=0.1 )
        assert sourcelist.y.min() == pytest.approx( 16.264, abs=0.1 )
        assert sourcelist.y.max() == pytest.approx( 4087.9, abs=0.1 )
        assert sourcelist.apfluxadu(apnum=1)[0].min() == pytest.approx( -656.8731, rel=1e-5 )
        assert sourcelist.apfluxadu(apnum=1)[0].max() == pytest.approx( 2850920.0, rel=1e-5 )
        assert sourcelist.apfluxadu(apnum=0)[0].min() == pytest.approx( 89.445114, rel=1e-5 )
        assert sourcelist.apfluxadu(apnum=0)[0].max() == pytest.approx( 557651.8, rel=1e-5 )

    finally:  # cleanup temporary file
        if 'sourcefile' in locals():
            sourcefile.unlink( missing_ok=True )


def test_run_psfex( decam_datastore, extractor ):
    sourcelist = decam_datastore.sources
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
        psf = extractor._run_psfex( tempname, sourcelist.image )
        assert psf._header['PSFAXIS1'] == 25
        assert psf._header['PSFAXIS2'] == 25
        assert psf._header['PSFAXIS3'] == 6
        assert psf._header['PSF_SAMP'] == pytest.approx( 0.92, abs=0.01 )
        assert psf._header['CHI2'] == pytest.approx( 0.9, abs=0.1 )
        bio = io.BytesIO( psf._info.encode( 'utf-8' ) )
        psfstats = votable.parse( bio ).get_table_by_index(1)
        assert psfstats.array['FWHM_FromFluxRadius_Max'] == pytest.approx( 4.31, abs=0.01 )
        assert not tmppsffile.exists()
        assert not tmppsfxmlfile.exists()

        psf = extractor._run_psfex( tempname, sourcelist.image, do_not_cleanup=True )
        assert tmppsffile.exists()
        assert tmppsfxmlfile.exists()
        tmppsffile.unlink()
        tmppsfxmlfile.unlink()

        psf = extractor._run_psfex( tempname, sourcelist.image, psf_size=26 )
        assert psf._header['PSFAXIS1'] == 29
        assert psf._header['PSFAXIS1'] == 29

    finally:
        tmpsourcefile.unlink( missing_ok=True )
        tmppsffile.unlink( missing_ok=True )
        tmppsfxmlfile.unlink( missing_ok=True )


def test_extract_sources_sextractor( decam_datastore, extractor, provenance_base, data_dir, blocking_plots ):
    ds = decam_datastore

    extractor.pars.method = 'sextractor'
    extractor.measure_psf = True
    extractor.pars.threshold = 5.0
    sources, psf, bkg, bkgsig = extractor.extract_sources( ds.image )

    assert bkg == pytest.approx( 179.82, abs=0.1 )
    assert bkgsig == pytest.approx( 7.533, abs=0.01 )

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

    assert sources.num_sources == 5500
    assert sources.num_sources == len(sources.data)
    assert sources.aper_rads == pytest.approx( [ 2.885, 4.286, 8.572, 12.858,
                                                 17.145, 21.431, 30.003, 42.862 ], abs=0.01 )
    assert sources._inf_aper_num == 5
    assert sources.inf_aper_num == 5
    assert psf.fwhm_pixels == pytest.approx( 4.286, abs=0.01 )
    assert psf.fwhm_pixels == pytest.approx( psf.header['PSF_FWHM'], rel=1e-5 )
    assert psf.data.shape == ( 6, 25, 25 )
    assert psf.image_id == ds.image.id

    assert sources.apfluxadu()[0].min() == pytest.approx( 200.34559, rel=1e-5 )
    assert sources.apfluxadu()[0].max() == pytest.approx( 1105999.625, rel=1e-5 )
    assert sources.apfluxadu()[0].mean() == pytest.approx( 36779.797 , rel=1e-5 )
    assert sources.apfluxadu()[0].std() == pytest.approx(  121950.04 , rel=1e-5 )

    assert sources.good.sum() == 3638
    # This value is what you get using the SPREAD_MODEL parameter
    # assert sources.is_star.sum() == 4870
    # assert ( sources.good & sources.is_star ).sum() == 3593
    # This is what you get with CLASS_STAR
    assert sources.is_star.sum() == 337
    assert ( sources.good & sources.is_star ).sum() == 61

    try:  # make sure saving the PSF and source list goes as expected, and cleanup at the end
        psf.provenance = provenance_base
        psf.save()
        assert re.match(r'\d{3}/c4d_\d{8}_\d{6}_N1_g_Sci_.{6}.psf_.{6}', psf.filepath)
        assert os.path.isfile( os.path.join(data_dir, psf.filepath + '.fits') )

        sources.provenance = provenance_base
        sources.save()
        assert re.match(r'\d{3}/c4d_\d{8}_\d{6}_N1_g_Sci_.{6}.sources_.{6}.fits', sources.filepath)
        assert os.path.isfile(os.path.join(data_dir, sources.filepath))

    finally:  # cleanup
        psf.delete_from_disk_and_database()
        sources.delete_from_disk_and_database()

# TODO : add tests that handle different combinations
#  of measure_psf and psf being passed to the Detector constructor


# TODO: is this test really the same as the one above?
def test_run_detection_sextractor( decam_datastore, extractor ):
    ds = decam_datastore

    # det = Detector( method='sextractor', measure_psf=True, threshold=5.0 )
    extractor.pars.method = 'sextractor'
    extractor.measure_psf = True
    extractor.pars.threshold = 5.0
    extractor.pars.test_parameter = uuid.uuid4().hex
    ds = extractor.run( ds )

    assert extractor.has_recalculated
    assert ds.sources.num_sources == 5500
    assert ds.sources.num_sources == len(ds.sources.data)
    assert ds.sources.aper_rads == pytest.approx( [ 2.88551706,  4.28627014,  8.57254028, 12.85881042, 17.14508057,
                                                    21.43135071, 30.00389099, 42.86270142], abs=0.01 )
    assert ds.sources._inf_aper_num == 5
    assert ds.sources.inf_aper_num == 5
    assert ds.psf.fwhm_pixels == pytest.approx( 4.286, abs=0.01 )
    assert ds.psf.fwhm_pixels == pytest.approx( ds.psf.header['PSF_FWHM'], rel=1e-5 )
    assert ds.psf.data.shape == ( 6, 25, 25 )
    assert ds.psf.image_id == ds.image.id

    assert ds.sources.apfluxadu()[0].min() == pytest.approx( 200.3456, rel=1e-5 )
    assert ds.sources.apfluxadu()[0].max() == pytest.approx( 1105999.6, rel=1e-5 )
    assert ds.sources.apfluxadu()[0].mean() == pytest.approx( 36779.797, rel=1e-5 )
    assert ds.sources.apfluxadu()[0].std() == pytest.approx(  121950.04 , rel=1e-5 )

    assert ds.sources.good.sum() == 3638
    # This value is what you get using the SPREAD_MODEL parameter
    # assert ds.sources.is_star.sum() == 4870
    # assert ( ds.sources.good & ds.sources.is_star ).sum() == 3593
    # This value is what you get using the CLASS_STAR parameter
    assert ds.sources.is_star.sum() == 337
    assert ( ds.sources.good & ds.sources.is_star ).sum() == 61

    # TODO : actually think about these psf fluxes and how they compare
    #  to the aperture fluxes (esp. the large-aperture fluxes).  Try to
    #  understand what SExtractor psf weighted photometry actually
    #  does....  Preliminary investigations suggest that something may be
    #  wrong.

    assert ds.sources.psffluxadu()[0].min() == 0.0
    assert ds.sources.psffluxadu()[0].max() == pytest.approx( 1725000.0, rel=1e-2 )
    assert ds.sources.psffluxadu()[0].mean() == pytest.approx( 48000.0, rel=1e-2 )
    assert ds.sources.psffluxadu()[0].std() == pytest.approx( 170000.0, rel=1e-2 )

    assert ds.sources.provenance is not None
    assert ds.sources.provenance == ds.psf.provenance
    assert ds.sources.provenance.process == 'extraction'

    assert ds.image.bkg_mean_estimate == pytest.approx( 179.82, abs=0.1 )
    assert ds.image.bkg_rms_estimate == pytest.approx( 7.533, abs=0.01 )

    from sqlalchemy.exc import IntegrityError

    try:
        ds.save_and_commit()

        # Make sure all the files exist
        archive = get_archive_object()
        imdir = pathlib.Path( FileOnDiskMixin.local_path )
        relpaths = []
        relpaths += [ds.image.filepath + ext for ext in ds.image.filepath_extensions]
        relpaths += [ds.sources.filepath]
        relpaths += [ds.psf.filepath + ext for ext in ds.psf.filepath_extensions]
        for relp in relpaths:
            assert ( imdir / relp ).is_file()
            assert archive.get_info( relp ) is not None

        # Make sure the bkg fields in the image database table aren't empty

        with SmartSession() as sess:
            imgs = sess.query( Image ).filter( Image.id == ds.image.id ).all()
            assert len(imgs) == 1
            assert imgs[0].bkg_mean_estimate == pytest.approx( 179.82, abs=0.1 )
            assert imgs[0].bkg_rms_estimate == pytest.approx( 7.533, abs=0.01 )

    finally:
        ds.delete_everything()


def test_warnings_and_exceptions(decam_datastore, extractor):
    extractor.pars.inject_warnings = 1

    with pytest.warns(UserWarning) as record:
        extractor.run(decam_datastore)
    assert len(record) > 0
    assert any("Warning injected by pipeline parameters in process 'detection'." in str(w.message) for w in record)

    extractor.pars.inject_warnings = 0
    extractor.pars.inject_exceptions = 1
    with pytest.raises(Exception) as excinfo:
        ds = extractor.run(decam_datastore)
        ds.reraise()
    assert "Exception injected by pipeline parameters in process 'detection'." in str(excinfo.value)
    ds.read_exception()