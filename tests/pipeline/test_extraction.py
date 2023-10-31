import pytest
import io
import os
import re
import uuid
import pathlib
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
# It deliberately does not have a module scope.
@pytest.fixture
def run_sextractor( decam_example_reduced_image_ds ):
    detector = Detector( method='sextractor', subtraction=False, apers=[5.], threshold=4.5 )

    tempname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    sourcelist = detector._run_sextractor_once( decam_example_reduced_image_ds.image, tempname=tempname )
    sourcefile = pathlib.Path( FileOnDiskMixin.temp_path ) / f"{tempname}.sources.fits"
    imagefile =  pathlib.Path( FileOnDiskMixin.temp_path ) / f"{tempname}.fits"
    assert not imagefile.exists()
    assert sourcefile.exists()

    yield sourcelist, sourcefile

    sourcefile.unlink( missing_ok=True )

def test_sextractor_extract_once( decam_example_reduced_image_ds, run_sextractor ):
    sourcelist, sourcefile = run_sextractor

    assert sourcelist.num_sources == 5611
    assert len(sourcelist.data) == sourcelist.num_sources
    assert sourcelist.aper_rads == [ 5. ]

    assert sourcelist.info['SEXAPED1'] == 5.0
    assert sourcelist.info['SEXAPED2'] == 0.
    assert sourcelist.info['SEXBKGND'] == pytest.approx( 179.8, abs=0.1 )

    assert sourcelist.x.min() == pytest.approx( 16.0, abs=0.1 )
    assert sourcelist.x.max() == pytest.approx( 2039.6, abs=0.1 )
    assert sourcelist.y.min() == pytest.approx( 16.3, abs=0.1 )
    assert sourcelist.y.max() == pytest.approx( 4087.9, abs=0.1 )
    assert sourcelist.apfluxadu()[0].min() == pytest.approx( 79.2300, rel=1e-5 )
    assert sourcelist.apfluxadu()[0].max() == pytest.approx( 852137.56, rel=1e-5 )
    snr = sourcelist.apfluxadu()[0] / sourcelist.apfluxadu()[1]
    assert snr.min() == pytest.approx( 2.33, abs=0.01 )
    assert snr.max() == pytest.approx( 1285, abs=1. )
    assert snr.mean() == pytest.approx( 120.85, abs=0.1 )
    assert snr.std() == pytest.approx( 205, abs=1. )

    # Test multiple apertures
    detector = Detector( method='sextractor', subtraction=False, threshold=4.5 )

    sourcelist = detector._run_sextractor_once( decam_example_reduced_image_ds.image, apers=[2,5] )

    assert sourcelist.num_sources == 5611    # It *finds* the same things
    assert len(sourcelist.data) == sourcelist.num_sources
    assert sourcelist.aper_rads == [ 2., 5. ]

    assert sourcelist.info['SEXAPED1'] == 2.0
    assert sourcelist.info['SEXAPED2'] == 5.0
    assert sourcelist.info['SEXBKGND'] == pytest.approx( 179.8, abs=0.1 )
    assert sourcelist.x.min() == pytest.approx( 16.0, abs=0.1 )
    assert sourcelist.x.max() == pytest.approx( 2039.6, abs=0.1 )
    assert sourcelist.y.min() == pytest.approx( 16.3, abs=0.1 )
    assert sourcelist.y.max() == pytest.approx( 4087.9, abs=0.1 )
    assert sourcelist.apfluxadu(apnum=1)[0].min() == pytest.approx( 79.2300, rel=1e-5 )
    assert sourcelist.apfluxadu(apnum=1)[0].max() == pytest.approx( 852137.56, rel=1e-5 )
    assert sourcelist.apfluxadu(apnum=0)[0].min() == pytest.approx( 35.02905, rel=1e-5 )
    assert sourcelist.apfluxadu(apnum=0)[0].max() == pytest.approx( 152206.1, rel=1e-5 )

def test_run_psfex( run_sextractor ):
    sourcelist, sourcefile = run_sextractor
    match = re.search( '^(.*).sources.fits', sourcefile.name )
    assert match is not None
    tempname = match.group(1)
    tmppsffile = pathlib.Path( FileOnDiskMixin.temp_path ) / f"{tempname}.sources.psf"
    tmppsfxmlfile = pathlib.Path( FileOnDiskMixin.temp_path ) / f"{tempname}.sources.psf.xml"

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
        tmppsffile.unlink( missing_ok=True )
        tmppsfxmlfile.unlink( missing_ok=True )


def test_extract_sources_sextractor( decam_example_reduced_image_ds ):
    det = Detector( method='sextractor', measure_psf=True, threshold=5.0 )

    sources, psf = det.extract_sources( decam_example_reduced_image_ds.image )

    assert sources.num_sources == 5506
    assert sources.num_sources == len(sources.data)
    assert sources.aper_rads == pytest.approx( [ 4.328, 8.656, 12.984, 17.312, 21.640, 30.296, 43.280 ], abs=0.01 )
    assert psf.fwhm_pixels == pytest.approx( 4.328, abs=0.01 )
    assert psf.fwhm_pixels == pytest.approx( psf.header['PSF_FWHM'], rel=1e-5 )
    assert psf.data.shape == ( 6, 25, 25 )
    assert psf.image_id == decam_example_reduced_image_ds.image.id

    assert sources.apfluxadu()[0].min() == pytest.approx( 146.98804, rel=1e-5 )
    assert sources.apfluxadu()[0].max() == pytest.approx( 645737.44, rel=1e-5 )
    assert sources.apfluxadu()[0].mean() == pytest.approx( 25484.562, rel=1e-5 )
    assert sources.apfluxadu()[0].std() == pytest.approx( 80211.62, rel=1e-5 )

def test_run_detection_sextractor( decam_example_reduced_image_ds ):
    det = Detector( method='sextractor', measure_psf=True, threshold=5.0 )
    ds = det.run( decam_example_reduced_image_ds )

    assert ds.sources.num_sources == 5506
    assert ds.sources.num_sources == len(ds.sources.data)
    assert ds.sources.aper_rads == pytest.approx( [ 4.328, 8.656, 12.984, 17.312, 21.640, 30.296, 43.280 ], abs=0.01 )
    assert ds.psf.fwhm_pixels == pytest.approx( 4.328, abs=0.01 )
    assert ds.psf.fwhm_pixels == pytest.approx( ds.psf.header['PSF_FWHM'], rel=1e-5 )
    assert ds.psf.data.shape == ( 6, 25, 25 )
    assert ds.psf.image_id == decam_example_reduced_image_ds.image.id

    assert ds.sources.apfluxadu()[0].min() == pytest.approx( 146.98804, rel=1e-5 )
    assert ds.sources.apfluxadu()[0].max() == pytest.approx( 645737.44, rel=1e-5 )
    assert ds.sources.apfluxadu()[0].mean() == pytest.approx( 25484.562, rel=1e-5 )
    assert ds.sources.apfluxadu()[0].std() == pytest.approx( 80211.62, rel=1e-5 )

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
