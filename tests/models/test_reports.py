import time
import uuid

from pprint import pprint

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import Provenance
from models.report import Report

from util.util import env_as_bool


def test_report_bitflags(decam_exposure, decam_reference, decam_default_calibrators):
    report = Report(exposure_id=decam_exposure.id, section_id='S3')

    # test that the progress steps flag is working
    assert report.progress_steps_bitflag == 0
    assert report.progress_steps == ''

    report.progress_steps = 'preprocessing'
    assert report.progress_steps_bitflag == 2 ** 1
    assert report.progress_steps == 'preprocessing'

    report.progress_steps = 'preprocessing, Extraction'
    assert report.progress_steps_bitflag == 2 ** 1 + 2 ** 2
    assert report.progress_steps == 'preprocessing, extraction'

    report.append_progress('preprocessing')  # appending it again makes no difference
    assert report.progress_steps_bitflag == 2 ** 1 + 2 ** 2
    assert report.progress_steps == 'preprocessing, extraction'

    report.append_progress('subtraction, cutting')  # append two at a time
    assert report.progress_steps_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 6 + 2 ** 8
    assert report.progress_steps == 'preprocessing, extraction, subtraction, cutting'

    # test that the products exist flag is working
    assert report.products_exist_bitflag == 0
    assert report.products_exist == ''

    report.products_exist = 'image'
    assert report.products_exist_bitflag == 2 ** 1
    assert report.products_exist == 'image'

    report.products_exist = 'image, sources'
    assert report.products_exist_bitflag == 2 ** 1 + 2 ** 2
    assert report.products_exist == 'image, sources'

    report.append_products_exist('psf')
    assert report.products_exist_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 3
    assert report.products_exist == 'image, sources, psf'

    report.append_products_exist('image')  # appending it again makes no difference
    assert report.products_exist_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 3
    assert report.products_exist == 'image, sources, psf'

    report.append_products_exist('sub_image, detections')  # append two at a time
    assert report.products_exist_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 3 + 2 ** 7 + 2 ** 8
    assert report.products_exist == 'image, sources, psf, sub_image, detections'

    # test that the products committed flag is working
    assert report.products_committed_bitflag == 0
    assert report.products_committed == ''

    report.products_committed = 'sources'
    assert report.products_committed_bitflag == 2 ** 2
    assert report.products_committed == 'sources'

    report.products_committed = 'sources, zp'
    assert report.products_committed_bitflag == 2 ** 2 + 2 ** 6
    assert report.products_committed == 'sources, zp'

    report.append_products_committed('sub_image')
    assert report.products_committed_bitflag == 2 ** 2 + 2 ** 6 + 2 ** 7
    assert report.products_committed == 'sources, zp, sub_image'

    report.append_products_committed('sub_image, detections')  # append two at a time
    assert report.products_committed_bitflag == 2 ** 2 + 2 ** 6 + 2 ** 7 + 2 ** 8
    assert report.products_committed == 'sources, zp, sub_image, detections'

    report.append_products_committed('sub_image')  # appending it again makes no difference
    assert report.products_committed_bitflag == 2 ** 2 + 2 ** 6 + 2 ** 7 + 2 ** 8
    assert report.products_committed == 'sources, zp, sub_image, detections'


def test_measure_runtime_memory(decam_exposure, decam_reference, pipeline_for_tests, decam_default_calibrators):
    p = pipeline_for_tests
    p.subtractor.pars.refset = 'test_refset_decam'
    p.pars.save_before_subtraction = True
    p.pars.save_at_finish = False
    # make sure we get a random new provenance, not reuse any of the existing data
    p.preprocessor.pars.test_parameter = uuid.uuid4().hex

    try:
        t0 = time.perf_counter()
        ds = p.run(decam_exposure, 'S2')
        total_time = time.perf_counter() - t0

        assert p.preprocessor.has_recalculated
        assert p.extractor.has_recalculated
        assert p.backgrounder.has_recalculated
        assert p.astrometor.has_recalculated
        assert p.photometor.has_recalculated
        assert p.subtractor.has_recalculated
        assert p.detector.has_recalculated
        assert p.cutter.has_recalculated
        assert p.measurer.has_recalculated
        assert p.scorer.has_recalculated

        measured_time = 0
        peak_memory = 0
        for step in ds.runtimes.keys():  # also make sure all the keys are present in both dictionaries
            measured_time += ds.runtimes[step]
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                peak_memory = max(peak_memory, ds.memory_usages[step])

        print(f'total_time: {total_time:.1f}s')
        print(f'measured_time: {measured_time:.1f}s')
        pprint(ds.report.process_runtime, sort_dicts=False)
        assert measured_time > 0.98 * total_time  # at least 98% of the time is accounted for

        if env_as_bool('SEECHANGE_TRACEMALLOC'):
            print(f'peak_memory: {peak_memory:.1f}MB')
            pprint(ds.memory_usages, sort_dicts=False)
            assert 1000.0 < peak_memory < 10000.0  # memory usage is in MB, takes between 1 and 10 GB

        with SmartSession() as session:
            rep = session.scalars(sa.select(Report).where(Report.exposure_id == decam_exposure.id)).one()
        assert rep is not None
        assert rep.success
        runtimes = rep.process_runtime.copy()
        runtimes.pop('reporting')
        assert runtimes == ds.runtimes
        assert rep.process_memory == ds.memory_usages
        assert rep.progress_steps == ( 'preprocessing, extraction, backgrounding, astrocal, photocal, '
                                       'subtraction, detection, cutting, measuring, scoring, finalize' )
        assert rep.products_exist == ('image, sources, psf, bg, wcs, zp, '
                                      'sub_image, detections, cutouts, measurement_set, deepscore_set')
        assert rep.products_committed == 'image, sources, psf, bg, wcs, zp'  # we use intermediate save
        repprov = Provenance.get( rep.provenance_id )
        assert repprov.upstreams[0].id == ds.deepscore_set.provenance_id
        assert rep.num_prev_reports == 0
        ds.save_and_commit()
        rep.scan_datastore(ds)
        assert rep.products_committed == ('image, sources, psf, bg, wcs, zp, '
                                          'sub_image, detections, cutouts, measurement_set, deepscore_set')
    finally:
        if 'ds' in locals():
            ds.delete_everything()


# Commented out the fixtures because they take a noticable amount of time to run....
# (Leaving the test here, though, because it's aspirational.)
# def test_inject_warnings(decam_datastore, decam_reference, pipeline_for_tests, decam_default_calibrators):
def test_inject_warnings():
    pass


# def test_inject_exceptions(decam_datastore, decam_reference, pipeline_for_tests):
def test_inject_exceptions():
    pass
