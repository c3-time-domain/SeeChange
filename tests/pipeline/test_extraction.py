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

from models.base import SmartSession, FileOnDiskMixin, get_archive_object
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList

from pipeline.detection import Detector


def test_sep_find_sources_in_small_image(decam_small_image):
    det = Detector(method='sep', subtraction=False, threshold=3.0)

    sources, _ = det.extract_sources(decam_small_image)

    assert sources.num_sources == 158
    assert max(sources.data['flux']) == 3670450.0
    assert abs(np.mean(sources.data['x']) - 256) < 10
    assert abs(np.mean(sources.data['y']) - 256) < 10
    assert 2.0 < np.median(sources.data['rhalf']) < 2.5

    if False:  # use this for debugging / visualization only!
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        matplotlib.use('Qt5Agg')

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
    sources2, _ = det.extract_sources(decam_small_image)
    assert sources2.num_sources < sources.num_sources

    # flux will change with new threshold, but not by more than 10%
    assert abs( max(sources2.data['flux']) - max(sources.data['flux'])) / max(sources.data['flux']) < 0.1

    # fewer sources also means the mean position will be further from center
    assert abs(np.mean(sources2.data['x']) - 256) < 25
    assert abs(np.mean(sources2.data['y']) - 256) < 25

    assert 2.0 < np.median(sources2.data['rhalf']) < 2.5


def test_sep_save_source_list(decam_small_image, provenance_base, code_version):
    decam_small_image.provenance = provenance_base
    det = Detector(method='sep', subtraction=False, threshold=3.0)
    sources, _ = det.extract_sources(decam_small_image)
    prov = Provenance(
        process='extraction',
        code_version=code_version,
        parameters=det.pars.get_critical_pars(),
        upstreams=[provenance_base],
        is_testing=True
    )
    prov.update_id()
    sources.provenance = prov

    filename = None
    image_id = None
    sources_id = None

    try:  # cleanup file / DB at the end
        sources.save()
        filename = sources.get_fullpath()

        assert os.path.isfile(filename)

        # check the naming convention
        assert re.search(r'.*/\d{3}/c4d_\d{8}_\d{6}_.+_.+_.+_.{6}.sources\.npy', filename)

        # check the file contents can be loaded successfully
        data = np.load(filename)
        assert np.array_equal(data, sources.data)

        with SmartSession() as session:
            decam_small_image.recursive_merge(session)
            sources.provenance = session.merge( sources.provenance )
            decam_small_image.save()  # pretend to save this file
            decam_small_image.exposure.save()
            session.add(sources)
            session.commit()
            image_id = decam_small_image.id
            sources_id = sources.id

    finally:
        if filename is not None and os.path.isfile(filename):
            os.remove(filename)
            folder = filename
            for i in range(10):
                folder = os.path.dirname(folder)
                if len(os.listdir(folder)) == 0:
                    os.rmdir(folder)
                else:
                    break
        with SmartSession() as session:
            if sources_id is not None:
                session.execute(sa.delete(SourceList).where(SourceList.id == sources_id))
            if image_id is not None:
                session.execute(sa.delete(Image).where(Image.id == image_id))
            session.commit()

# This is running sextractor in one particular way that is used by more than one test
@pytest.fixture
def run_sextractor( decam_example_reduced_image_ds ):
    ds = decam_example_reduced_image_ds
    ds.sources = None
    ds.psf = None

    detector = Detector( method='sextractor', subtraction=False, apers=[5.], threshold=4.5 )

    tempname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    sourcelist = detector._run_sextractor_once( ds.image, tempname=tempname )
    sourcefile = pathlib.Path( FileOnDiskMixin.temp_path ) / f"{tempname}.sources.fits"
    imagefile =  pathlib.Path( FileOnDiskMixin.temp_path ) / f"{tempname}.fits"
    assert not imagefile.exists()
    assert sourcefile.exists()

    yield sourcelist, sourcefile

def test_sextractor_extract_once( decam_example_reduced_image_ds, run_sextractor ):
    sourcelist, sourcefile = run_sextractor

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
    assert sourcelist.y.min() == pytest.approx( 16.3, abs=0.1 )
    assert sourcelist.y.max() == pytest.approx( 4087.9, abs=0.1 )
    assert sourcelist.errx.min() == pytest.approx( 0.0005, abs=1e-4 )
    assert sourcelist.errx.max() == pytest.approx( 1.05, abs=0.01 )
    assert sourcelist.erry.min() == pytest.approx( 0.001, abs=1e-3 )
    assert sourcelist.erry.max() == pytest.approx( 0.62, abs=0.01 )
    assert ( np.sqrt( sourcelist.varx ) == sourcelist.errx ).all()
    assert ( np.sqrt( sourcelist.vary ) == sourcelist.erry ).all()
    assert sourcelist.apfluxadu()[0].min() == pytest.approx( -656.8731, rel=1e-5 )
    assert sourcelist.apfluxadu()[0].max() == pytest.approx( 2850920.0, rel=1e-5 )
    snr = sourcelist.apfluxadu()[0] / sourcelist.apfluxadu()[1]
    assert snr.min() == pytest.approx( -9.91, abs=0.1 )
    assert snr.max() == pytest.approx( 2348.2, abs=1. )
    assert snr.mean() == pytest.approx( 146.80, abs=0.1 )
    assert snr.std() == pytest.approx( 285.4, abs=1. )

    # Test multiple apertures
    detector = Detector( method='sextractor', subtraction=False, threshold=4.5 )

    sourcelist = detector._run_sextractor_once( decam_example_reduced_image_ds.image, apers=[2,5] )

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
    assert sourcelist.y.min() == pytest.approx( 16.3, abs=0.1 )
    assert sourcelist.y.max() == pytest.approx( 4087.9, abs=0.1 )
    assert sourcelist.apfluxadu(apnum=1)[0].min() == pytest.approx( -656.8731, rel=1e-5 )
    assert sourcelist.apfluxadu(apnum=1)[0].max() == pytest.approx( 2850920.0, rel=1e-5 )
    assert sourcelist.apfluxadu(apnum=0)[0].min() == pytest.approx( 89.445114, rel=1e-5 )
    assert sourcelist.apfluxadu(apnum=0)[0].max() == pytest.approx( 557651.8, rel=1e-5 )

def test_run_psfex( decam_example_reduced_image_ds ):
    sourcelist = decam_example_reduced_image_ds.sources
    tempname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    temp_path = pathlib.Path( FileOnDiskMixin.temp_path )
    tmpsourcefile =  temp_path / f'{tempname}.sources.fits'
    tmppsffile = temp_path / f"{tempname}.sources.psf"
    tmppsfxmlfile = temp_path / f"{tempname}.sources.psf.xml"
    shutil.copy2( sourcelist.get_fullpath(), tmpsourcefile )

    try:
        detector = Detector( method='sextractor', subtraction=False, threshold=4.5 )
        psf = detector._run_psfex( tempname, sourcelist.image_id )
        assert psf._header['PSFAXIS1'] == 25
        assert psf._header['PSFAXIS2'] == 25
        assert psf._header['PSFAXIS3'] == 6
        assert psf._header['PSF_SAMP'] == pytest.approx( 0.92, abs=0.01 )
        assert psf._header['CHI2'] == pytest.approx( 0.9, abs=0.1 )
        bio = io.BytesIO( psf._info.encode( 'utf-8' ) )
        psfstats = votable.parse( bio ).get_table_by_index(1)
        assert psfstats.array['FWHM_FromFluxRadius_Max'] == pytest.approx( 4.33, abs=0.01 )
        assert not tmppsffile.exists()
        assert not tmppsfxmlfile.exists()

        psf = detector._run_psfex( tempname, sourcelist.image_id, do_not_cleanup=True )
        assert tmppsffile.exists()
        assert tmppsfxmlfile.exists()
        tmppsffile.unlink()
        tmppsfxmlfile.unlink()

        psf = detector._run_psfex( tempname, sourcelist.image_id, psf_size=26 )
        assert psf._header['PSFAXIS1'] == 29
        assert psf._header['PSFAXIS1'] == 29

    finally:
        tmpsourcefile.unlink( missing_ok=True )
        tmppsffile.unlink( missing_ok=True )
        tmppsfxmlfile.unlink( missing_ok=True )


def test_extract_sources_sextractor( decam_example_reduced_image_ds ):
    ds = decam_example_reduced_image_ds
    ds.sources = None
    ds.psf = None

    det = Detector( method='sextractor', measure_psf=True, threshold=5.0 )
    sources, psf = det.extract_sources( ds.image )

    # Make True to write some ds9 regions
    if False:
        sources.ds9_regfile( 'test_sources_stars.reg', color='green', radius=4, whichsources='stars' )
        sources.ds9_regfile( 'test_sources_nonstars.reg', color='red', radius=4, whichsources='nonstars' )

        # Manually write one that uses CLASS_STAR instead of SPREAD_MODEL
        use = sources.data['CLASS_STAR'] > 0.5
        with open( 'test_sources_class_star.reg', 'w' ) as ofp:
            for x, y, use in zip( sources.x, sources.y, use ):
                if use:
                    ofp.write( f"image;circle({x+1},{y+1},6) # color=blue width=2\n" )

    assert sources.num_sources == 5499
    assert sources.num_sources == len(sources.data)
    assert sources.aper_rads == pytest.approx( [ 2.914, 4.328, 8.656, 12.984,
                                                 17.312, 21.640, 30.296, 43.280 ], abs=0.01 )
    assert sources._inf_aper_num == 5
    assert sources.inf_aper_num == 5
    assert psf.fwhm_pixels == pytest.approx( 4.328, abs=0.01 )
    assert psf.fwhm_pixels == pytest.approx( psf.header['PSF_FWHM'], rel=1e-5 )
    assert psf.data.shape == ( 6, 25, 25 )
    assert psf.image_id == ds.image.id

    assert sources.apfluxadu()[0].min() == pytest.approx( 204.55038, rel=1e-5 )
    assert sources.apfluxadu()[0].max() == pytest.approx( 1131884.6, rel=1e-5 )
    assert sources.apfluxadu()[0].mean() == pytest.approx( 37183.51, rel=1e-5 )
    assert sources.apfluxadu()[0].std() == pytest.approx( 123518.94, rel=1e-5 )

    assert sources.good.sum() == 3642
    # This value is what you get using the SPREAD_MODEL parameter
    # assert sources.is_star.sum() == 4870
    # assert ( sources.good & sources.is_star ).sum() == 3593
    # This is what you get with CLASS_STAR
    assert sources.is_star.sum() == 337
    assert ( sources.good & sources.is_star ).sum() == 63


# TODO : add tests that handle different
# combinations of measure_psf and psf being passed to the Detector constructor

def test_run_detection_sextractor( decam_example_reduced_image_ds ):
    ds = decam_example_reduced_image_ds
    ds.sources = None
    ds.psf = None

    det = Detector( method='sextractor', measure_psf=True, threshold=5.0 )
    ds = det.run( ds )

    assert ds.sources.num_sources == 5499
    assert ds.sources.num_sources == len(ds.sources.data)
    assert ds.sources.aper_rads == pytest.approx( [ 2.914, 4.328, 8.656, 12.984,
                                                    17.312, 21.640, 30.296, 43.280 ], abs=0.01 )
    assert ds.sources._inf_aper_num == 5
    assert ds.sources.inf_aper_num == 5
    assert ds.psf.fwhm_pixels == pytest.approx( 4.328, abs=0.01 )
    assert ds.psf.fwhm_pixels == pytest.approx( ds.psf.header['PSF_FWHM'], rel=1e-5 )
    assert ds.psf.data.shape == ( 6, 25, 25 )
    assert ds.psf.image_id == ds.image.id

    assert ds.sources.apfluxadu()[0].min() == pytest.approx( 204.55038, rel=1e-5 )
    assert ds.sources.apfluxadu()[0].max() == pytest.approx( 1131884.6, rel=1e-5 )
    assert ds.sources.apfluxadu()[0].mean() == pytest.approx( 37183.51, rel=1e-5 )
    assert ds.sources.apfluxadu()[0].std() == pytest.approx( 123518.94, rel=1e-5 )

    assert ds.sources.good.sum() == 3642
    # This value is what you get using the SPREAD_MODEL parameter
    # assert ds.sources.is_star.sum() == 4870
    # assert ( ds.sources.good & ds.sources.is_star ).sum() == 3593
    # This value is what you get using the CLASS_STAR parameter
    assert ds.sources.is_star.sum() == 337
    assert ( ds.sources.good & ds.sources.is_star ).sum() == 63

    # TODO : actually think about these psf fluxes and how they compare
    # to the aperture fluxes (esp. the large-aperture fluxes).  Try to
    # understand what SExtractor psf weighted photometry actually
    # does....  Preliminary investigations suggest that something may be
    # wrong.

    assert ds.sources.psffluxadu()[0].min() == 0.0
    assert ds.sources.psffluxadu()[0].max() == pytest.approx( 1726249.0, rel=1e-5 )
    assert ds.sources.psffluxadu()[0].mean() == pytest.approx( 48067.805, rel=1e-5 )
    assert ds.sources.psffluxadu()[0].std() == pytest.approx( 169444.77, rel=1e-5 )

    assert ds.sources.provenance is not None
    assert ds.sources.provenance == ds.psf.provenance
    assert ds.sources.provenance.process == 'extraction'

    try:
        ds.save_and_commit()

        # Make sure all the files exist
        archive = get_archive_object()
        imdir = pathlib.Path( FileOnDiskMixin.local_path )
        base = ds.image.filepath
        relpaths = [ f'{base}{i}' for i in [ '.image.fits', '.weight.fits', '.flags.fits',
                                             '.sources.fits', '.psf', '.psf.xml' ] ]
        for relp in relpaths:
            assert ( imdir / relp ).is_file()
            assert archive.get_info( relp ) is not None

    finally:
        ds.delete_everything()
