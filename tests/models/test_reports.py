import os
import time
import uuid

from pprint import pprint

import sqlalchemy as sa

from pipeline.top_level import PROCESS_OBJECTS

from models.base import SmartSession
from models.report import Report

from util.util import env_as_bool


def test_report_bitflags(decam_exposure, decam_reference, decam_default_calibrators):
    report = Report(exposure=decam_exposure, section_id='S3')

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
    assert report.progress_steps_bitflag == 2 ** 1 + 2 ** 2 + 2 ** 5 + 2 ** 7
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
    # make sure we get a random new provenance, not reuse any of the existing data
    p = pipeline_for_tests
    p.subtractor.pars.refset = 'test_refset_decam'
    p.pars.save_before_subtraction = True
    p.pars.save_at_finish = False
    p.preprocessor.pars.test_parameter = uuid.uuid4().hex

    try:
        t0 = time.perf_counter()
        ds = p.run(decam_exposure, 'S3')
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

        measured_time = 0
        peak_memory = 0
        for step in ds.runtimes.keys():  # also make sure all the keys are present in both dictionaries
            measured_time += ds.runtimes[step]
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                peak_memory = max(peak_memory, ds.memory_usages[step])

        print(f'total_time: {total_time:.1f}s')
        print(f'measured_time: {measured_time:.1f}s')
        pprint(ds.report.process_runtime, sort_dicts=False)
        assert measured_time > 0.99 * total_time  # at least 99% of the time is accounted for

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
            # should contain: 'preprocessing, extraction, subtraction, detection, cutting, measuring'
            assert rep.progress_steps == ', '.join(PROCESS_OBJECTS.keys())
            assert rep.products_exist == ('image, sources, psf, bg, wcs, zp, '
                                          'sub_image, detections, cutouts, measurements, scores')
            assert rep.products_committed == 'image, sources, psf, bg, wcs, zp'  # we use intermediate save
            assert rep.provenance.upstreams[0].id == ds.measurements[0].provenance.id
            assert rep.num_prev_reports == 0
            ds.save_and_commit(session=session)
            rep.scan_datastore(ds, session=session)
            assert rep.products_committed == ('image, sources, psf, bg, wcs, zp, '
                                              'sub_image, detections, cutouts, measurements, scores')
    finally:
        if 'ds' in locals():
            ds.delete_everything()


def test_inject_warnings(decam_datastore, decam_reference, pipeline_for_tests, decam_default_calibrators):
    pass


def test_inject_exceptions(decam_datastore, decam_reference, pipeline_for_tests):
    pass


