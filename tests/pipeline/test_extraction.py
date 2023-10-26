import pytest
import os
import re
import uuid
import numpy as np

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList

from pipeline.detection import Detector


def test_sep_find_sources_in_small_image(decam_small_image):
    det = Detector(method='sep', subtraction=False, threshold=3.0)

    sources = det.extract_sources(decam_small_image)

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
    sources2 = det.extract_sources(decam_small_image)
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
    sources = det.extract_sources(decam_small_image)
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
        assert re.search(r'.*/\d{3}/c4d_\d{8}_\d{6}_.+_.+_.+_.{6}_sources\.npy', filename)

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


def test_sextractor_extract_once( decam_example_reduced_image_ds ):
    detector = Detector( method='sextractor', subtraction=False, threshold=3.0 )

    sourcelist = detector._run_sextractor_once( decam_example_reduced_image_ds.image )

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
