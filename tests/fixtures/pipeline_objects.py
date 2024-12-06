import pytest

import sqlalchemy as sa

from models.base import SmartSession

from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.backgrounding import Backgrounder
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.coaddition import Coadder, CoaddPipeline
from pipeline.subtraction import Subtractor
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer
from pipeline.scoring import Scorer
from pipeline.top_level import Pipeline
from pipeline.ref_maker import RefMaker


@pytest.fixture(scope='session')
def preprocessor_factory(test_config):

    def make_preprocessor():
        prep = Preprocessor(**test_config.value('preprocessing'))
        prep.pars._enforce_no_new_attrs = False
        prep.pars.test_parameter = prep.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        prep.pars._enforce_no_new_attrs = True

        return prep

    return make_preprocessor


@pytest.fixture
def preprocessor(preprocessor_factory):
    return preprocessor_factory()


@pytest.fixture(scope='session')
def extractor_factory(test_config):

    def make_extractor():
        extr = Detector(**test_config.value('extraction.sources'))
        extr.pars._enforce_no_new_attrs = False
        extr.pars.test_parameter = extr.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        extr.pars._enforce_no_new_attrs = True

        return extr

    return make_extractor


@pytest.fixture
def extractor(extractor_factory):
    return extractor_factory()


@pytest.fixture(scope='session')
def backgrounder_factory(test_config):

    def make_backgrounder():
        bg = Backgrounder(**test_config.value('extraction.bg'))
        bg.pars._enforce_no_new_attrs = False
        bg.pars.test_parameter = bg.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        bg.pars._enforce_no_new_attrs = True

        return bg

    return make_backgrounder


@pytest.fixture(scope='session')
def astrometor_factory(test_config):

    def make_astrometor():
        astrom = AstroCalibrator(**test_config.value('extraction.wcs'))
        astrom.pars._enforce_no_new_attrs = False
        astrom.pars.test_parameter = astrom.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        astrom.pars._enforce_no_new_attrs = True

        return astrom

    return make_astrometor


@pytest.fixture
def astrometor(astrometor_factory):
    return astrometor_factory()


@pytest.fixture(scope='session')
def photometor_factory(test_config):

    def make_photometor():
        photom = PhotCalibrator(**test_config.value('extraction.zp'))
        photom.pars._enforce_no_new_attrs = False
        photom.pars.test_parameter = photom.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        photom.pars._enforce_no_new_attrs = True

        return photom

    return make_photometor


@pytest.fixture(scope='session')
def coadder_factory(test_config):

    def make_coadder():

        coadd = Coadder(**test_config.value('coaddition.coaddition'))
        coadd.pars._enforce_no_new_attrs = False
        coadd.pars.test_parameter = coadd.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        coadd.pars._enforce_no_new_attrs = True

        return coadd

    return make_coadder


@pytest.fixture
def coadder(coadder_factory):
    return coadder_factory()


@pytest.fixture(scope='session')
def subtractor_factory(test_config):

    def make_subtractor():
        sub = Subtractor(**test_config.value('subtraction'))
        sub.pars._enforce_no_new_attrs = False
        sub.pars.test_parameter = sub.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        sub.pars._enforce_no_new_attrs = True

        return sub

    return make_subtractor


@pytest.fixture
def subtractor(subtractor_factory):
    return subtractor_factory()


@pytest.fixture(scope='session')
def detector_factory(test_config):

    def make_detector():
        det = Detector(**test_config.value('detection'))
        det.pars._enforce_no_new_attrs = False
        det.pars.test_parameter = det.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        det.pars._enforce_no_new_attrs = True

        return det

    return make_detector


@pytest.fixture
def detector(detector_factory):
    return detector_factory()


@pytest.fixture(scope='session')
def cutter_factory(test_config):

    def make_cutter():
        cut = Cutter(**test_config.value('cutting'))
        cut.pars._enforce_no_new_attrs = False
        cut.pars.test_parameter = cut.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        cut.pars._enforce_no_new_attrs = True

        return cut

    return make_cutter


@pytest.fixture
def cutter(cutter_factory):
    return cutter_factory()


@pytest.fixture(scope='session')
def measurer_factory(test_config):

    def make_measurer():
        meas = Measurer(**test_config.value('measuring'))
        meas.pars._enforce_no_new_attrs = False
        meas.pars.test_parameter = meas.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        meas.pars._enforce_no_new_attrs = True

        return meas

    return make_measurer


@pytest.fixture
def measurer(measurer_factory):
    return measurer_factory()


@pytest.fixture(scope='session')
def scorer_factory(test_config):

    def make_scorer():
        scor = Scorer(**test_config.value('scoring'))
        scor.pars._enforce_no_new_attrs = False
        scor.pars.test_parameter = scor.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        scor.pars._enforce_no_new_attrs = True

        return scor

    return make_scorer


@pytest.fixture
def scorer(scorer_factory):
    return scorer_factory()


@pytest.fixture(scope='session')
def pipeline_factory(
        preprocessor_factory,
        extractor_factory,
        backgrounder_factory,
        astrometor_factory,
        photometor_factory,
        subtractor_factory,
        detector_factory,
        cutter_factory,
        measurer_factory,
        scorer_factory,
        test_config,
):
    def make_pipeline( provtag=None ):
        kwargs = {}
        if provtag is not None:
            kwargs['pipeline'] = { 'provenance_tag': provtag }
        p = Pipeline(**kwargs)
        p.pars.save_before_subtraction = False
        p.pars.save_at_finish = False
        p.preprocessor = preprocessor_factory()
        p.extractor = extractor_factory()
        p.backgrounder = backgrounder_factory()
        p.astrometor = astrometor_factory()
        p.photometor = photometor_factory()

        # make sure when calling get_critical_pars() these objects will produce the full, nested dictionary
        siblings = {
            'sources': p.extractor.pars,
            'bg': p.backgrounder.pars,
            'wcs': p.astrometor.pars,
            'zp': p.photometor.pars
        }
        p.extractor.pars.add_siblings(siblings)
        p.backgrounder.pars.add_siblings(siblings)
        p.astrometor.pars.add_siblings(siblings)
        p.photometor.pars.add_siblings(siblings)

        p.subtractor = subtractor_factory()
        p.detector = detector_factory()
        p.cutter = cutter_factory()
        p.measurer = measurer_factory()
        p.scorer = scorer_factory()

        return p

    return make_pipeline


@pytest.fixture
def pipeline_for_tests(pipeline_factory):
    p = pipeline_factory( 'pipeline_for_tests' )
    yield p

    # Clean up the provenance tag potentially created by the pipeline
    with SmartSession() as session:
        session.execute( sa.text( "DELETE FROM provenance_tags WHERE tag=:tag" ), {'tag': 'pipeline_for_tests' } )
        session.commit()



@pytest.fixture(scope='session')
def coadd_pipeline_factory(
        coadder_factory,
        extractor_factory,
        backgrounder_factory,
        astrometor_factory,
        photometor_factory,
        test_config,
):
    def make_pipeline():
        p = CoaddPipeline(**test_config.value('pipeline'))
        p.coadder = coadder_factory()
        p.extractor = extractor_factory()
        p.backgrounder = backgrounder_factory()
        p.astrometor = astrometor_factory()
        p.photometor = photometor_factory()

        # make sure when calling get_critical_pars() these objects will produce the full, nested dictionary
        siblings = {
            'sources': p.extractor.pars,
            'bg': p.backgrounder.pars,
            'wcs': p.astrometor.pars,
            'zp': p.photometor.pars,
        }
        p.extractor.pars.add_siblings(siblings)
        p.backgrounder.pars.add_siblings(siblings)
        p.astrometor.pars.add_siblings(siblings)
        p.photometor.pars.add_siblings(siblings)

        return p

    return make_pipeline


@pytest.fixture
def coadd_pipeline_for_tests(coadd_pipeline_factory):
    return coadd_pipeline_factory()


@pytest.fixture(scope='session')
def refmaker_factory(test_config, pipeline_factory, coadd_pipeline_factory):

    def make_refmaker(name, instrument, provtag='refmaker_factory'):
        maker = RefMaker(maker={'name': name, 'instruments': [instrument]})
        maker.pars._enforce_no_new_attrs = False
        maker.pars.test_parameter = maker.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        maker.pars._enforce_no_new_attrs = True
        maker.pipeline = pipeline_factory( provtag )
        maker.pipeline.override_parameters(**test_config.value('referencing.pipeline'))
        maker.coadd_pipeline = coadd_pipeline_factory()
        maker.coadd_pipeline.override_parameters(**test_config.value('referencing.coaddition'))

        return maker

    return make_refmaker
