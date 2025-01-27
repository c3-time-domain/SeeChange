import datetime
import time
import warnings

import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.preprocessing import Preprocessor
from pipeline.backgrounding import Backgrounder
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.subtraction import Subtractor
from pipeline.detection import Detector
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer
from pipeline.scoring import Scorer

from models.base import SmartSession
from models.exposure import Exposure
from models.report import Report

from util.config import Config
from util.logger import SCLogger
from util.util import env_as_bool

# describes the pipeline objects that are used to produce each step of the pipeline
# if multiple objects are used in one step, replace the string with a sub-dictionary,
# where the sub-dictionary keys are the keywords inside the expected critical parameters
# that come from all the different objects.
PROCESS_OBJECTS = {
    'preprocessing': 'preprocessor',
    'extraction': {
        'sources': 'extractor',
        'psf': 'extractor',
        'bg': 'backgrounder',
        'wcs': 'astrometor',
        'zp': 'photometor',
    },
    'subtraction': 'subtractor',
    'detection': 'detector',
    'cutting': 'cutter',
    'measuring': 'measurer',
    'scoring': 'scorer',
}


# put all the top-level pipeline parameters in the init of this class:
class ParsPipeline(Parameters):

    def __init__(self, **kwargs):
        super().__init__()

        self.example_pipeline_parameter = self.add_par(
            'example_pipeline_parameter', 1, int, 'an example pipeline parameter', critical=False
        )

        self.save_before_subtraction = self.add_par(
            'save_before_subtraction',
            True,
            bool,
            'Save intermediate images to the database, '
            'after doing extraction, background, and astro/photo calibration, '
            'if there is no reference, will not continue to doing subtraction'
            'but will still save the products up to that point. ',
            critical=False,
        )

        self.save_on_exception = self.add_par(
            'save_on_exception',
            False,
            bool,
            "If there's an exception, normally data products won't be saved "
            "(unless the exception is subtraction or later and save_before_subtraction "
            "is set, in which case the pre-subtraction ones will be saved).  If this is "
            "true, then the save_and_commit() method of the DataStore will be called, "
            "saving everything that it has.  WARNING: it could be that the thing that "
            "caused the exception will end up saved and committed!  Generally you only "
            "want to set this when testing, developing, or debugging.",
            critical=False,
        )

        self.save_at_finish = self.add_par(
            'save_at_finish',
            True,
            bool,
            'Save the final products to the database and disk',
            critical=False,
        )

        self.provenance_tag = self.add_par(
            'provenance_tag',
            'current',
            ( None, str ),
            "The ProvenanceTag that data products should be associated with.  Will be "
            "created it doesn't exist;  if it does exist, will verify that all the "
            "provenances we're running with are properly tagged there.",
            critical=False
        )

        self.through_step = self.add_par(
            'through_step',
            None,
            ( None, str ),
            "Stop after this step.  None = run the whole pipeline.  String values can be "
            "any of preprocessing, backgrounding, extraction, wcs, zp, subtraction, detection, "
            "cutting, measuring, scoring",
            critical=False
        )

        self.generate_report = self.add_par(
            'generate_report',
            True,
            bool,
            "If True, generate a report object if the pipeline starts from an Exposure.  "
            "(Reports are linked to exposures, so it's not possible to generate a report "
            "when starting from an image.)  If False, don't generate a report or a report "
            "provenance.",
            critical=False
        )

        self._enforce_no_new_attrs = True  # lock against new parameters

        self.override(kwargs)


class Pipeline:
    ALL_STEPS = [ 'preprocessing', 'backgrounding', 'extraction', 'wcs', 'zp', 'subtraction',
                  'detection', 'cutting', 'measuring', 'scoring', ]

    def __init__(self, **kwargs):
        config = Config.get()

        # top level parameters
        self.pars = ParsPipeline(**(config.value('pipeline', {})))
        self.pars.augment(kwargs.get('pipeline', {}))

        # dark/flat and sky subtraction tools
        preprocessing_config = config.value('preprocessing', {})
        preprocessing_config.update(kwargs.get('preprocessing', {}))
        self.pars.add_defaults_to_dict(preprocessing_config)
        self.preprocessor = Preprocessor(**preprocessing_config)

        # source detection ("extraction" for the regular image!)
        extraction_config = config.value('extraction.sources', {})
        extraction_config.update(kwargs.get('extraction', {}).get('sources', {}))
        extraction_config.update({'measure_psf': True})
        self.pars.add_defaults_to_dict(extraction_config)
        self.extractor = Detector(**extraction_config)

        # background estimation using either sep or other methods
        background_config = config.value('extraction.bg', {})
        background_config.update(kwargs.get('extraction', {}).get('bg', {}))
        self.pars.add_defaults_to_dict(background_config)
        self.backgrounder = Backgrounder(**background_config)

        # astrometric fit using a first pass of sextractor and then astrometric fit to Gaia
        astrometor_config = config.value('extraction.wcs', {})
        astrometor_config.update(kwargs.get('extraction', {}).get('wcs', {}))
        self.pars.add_defaults_to_dict(astrometor_config)
        self.astrometor = AstroCalibrator(**astrometor_config)

        # photometric calibration:
        photometor_config = config.value('extraction.zp', {})
        photometor_config.update(kwargs.get('extraction', {}).get('zp', {}))
        self.pars.add_defaults_to_dict(photometor_config)
        self.photometor = PhotCalibrator(**photometor_config)

        # make sure when calling get_critical_pars() these objects will produce the full, nested dictionary
        siblings = {
            'sources': self.extractor.pars,
            'bg': self.backgrounder.pars,
            'wcs': self.astrometor.pars,
            'zp': self.photometor.pars,
        }
        self.extractor.pars.add_siblings(siblings)
        self.backgrounder.pars.add_siblings(siblings)
        self.astrometor.pars.add_siblings(siblings)
        self.photometor.pars.add_siblings(siblings)

        # reference fetching and image subtraction
        subtraction_config = config.value('subtraction', {})
        subtraction_config.update(kwargs.get('subtraction', {}))
        self.pars.add_defaults_to_dict(subtraction_config)
        self.subtractor = Subtractor(**subtraction_config)

        # source detection ("detection" for the subtracted image!)
        detection_config = config.value('detection', {})
        detection_config.update(kwargs.get('detection', {}))
        self.pars.add_defaults_to_dict(detection_config)
        self.detector = Detector(**detection_config)
        self.detector.pars.subtraction = True

        # produce cutouts for detected sources:
        cutting_config = config.value('cutting', {})
        cutting_config.update(kwargs.get('cutting', {}))
        self.pars.add_defaults_to_dict(cutting_config)
        self.cutter = Cutter(**cutting_config)

        # measure photometry, analytical cuts, and deep learning models on the Cutouts:
        measuring_config = config.value('measuring', {})
        measuring_config.update(kwargs.get('measuring', {}))
        self.pars.add_defaults_to_dict(measuring_config)
        self.measurer = Measurer(**measuring_config)

        # assign r/b and ml/dl scores
        scoring_config = config.value('scoring', {})
        scoring_config.update(kwargs.get('scoring', {}))
        self.pars.add_defaults_to_dict(scoring_config)
        self.scorer = Scorer(**scoring_config)

        # Other initializationj
        self._generate_report = self.pars.generate_report

    def override_parameters(self, **kwargs):
        """Override some of the parameters for this object and its sub-objects, using Parameters.override(). """
        for key, value in kwargs.items():
            if key in PROCESS_OBJECTS:
                if isinstance(PROCESS_OBJECTS[key], dict):
                    for sub_key, sub_value in PROCESS_OBJECTS[key].items():
                        if sub_key in value:
                            getattr(self, sub_value).pars.override(value[sub_key])
                elif isinstance(PROCESS_OBJECTS[key], str):
                    getattr(self, PROCESS_OBJECTS[key]).pars.override(value)
            else:
                self.pars.override({key: value})

    def augment_parameters(self, **kwargs):
        """Add some parameters to this object and its sub-objects, using Parameters.augment(). """
        for key, value in kwargs.items():
            if key in PROCESS_OBJECTS:
                getattr(self, PROCESS_OBJECTS[key]).pars.augment(value)
            else:
                self.pars.augment({key: value})

    def setup_datastore(self, *args, no_provtag=False, ok_no_ref_provs=False, **kwargs):
        """Initialize a datastore, including an exposure and a report, to use in the pipeline run.

        Will raise an exception if there is no valid Exposure,
        if there's no reference available, or if the report cannot
        be posted to the database.

        After these objects are instantiated, the pipeline will proceed
        and record any exceptions into the report object before raising them.

        Parameters
        ----------
          Positional parameters:
            Inputs should include the exposure and section_id, or a
            datastore with these things already loaded.

            If a session is passed in as one of the arguments, it will
            be used as a single session for running the entire pipeline
            (instead of opening and closing sessions where needed).
            Usually you don't want to do this.

          no_provtag: bool, default False
            If True, won't create a provenance tag, and won't ensure
            that the provenances created match the provenance_tag
            parameter to the pipeline.  If False, will create the
            provenance tag if it doesn't exist.  If it does exist, will
            verify that all the provenances in the created provenance
            tree are what's tagged.

          ok_no_ref_provs: bool, default False
            Normally, if a refeset can't be found, or no image
            provenances associated with that refset can be found, an
            execption will be raised.  Set this to True to indicate that
            that's OK; in that case, the returned prov_tree will not
            have any provenances for steps other than preprocessing and
            extraction.

          All other keyword arguments are passed to DataStore.from_args

        Returns
        -------
        ds : DataStore
            The DataStore object that was created or loaded.

        session: sqlalchemy.orm.session.Session
            An optional session. If not given, this will be None.  You usually don't want to give this.

        """
        ds, session = DataStore.from_args(*args, **kwargs)

        if ds.exposure is None:
            raise RuntimeError('Cannot run this pipeline method without an exposure!')

        # Make sure exposure is in DB
        if Exposure.get_by_id( ds.exposure.id ) is None:
            raise RuntimeError( "Exposure must be loaded into the database." )

        try:  # create (and commit, if not existing) all provenances for the products
            provs = self.make_provenance_tree( ds,
                                               ds.exposure,
                                               no_provtag=no_provtag,
                                               all_steps=self.pars.generate_report )
            ds.prov_tree = provs
        except Exception as e:
            raise RuntimeError( f'Failed to create the provenance tree: {str(e)}' ) from e


        if self._generate_report:
            try:  # must make sure the report is on the DB
                report = Report( exposure_id=ds.exposure.id, section_id=ds.section_id )
                report.start_time = datetime.datetime.now( tz=datetime.UTC )
                report.provenance_id = provs['report'].id
                with SmartSession(session) as dbsession:
                    # check how many times this report was generated before
                    prev_rep = dbsession.scalars(
                        sa.select(Report).where(
                            Report.exposure_id == ds.exposure.id,
                            Report.section_id == ds.section_id,
                            Report.provenance_id == provs['report'].id,
                        )
                    ).all()
                    report.num_prev_reports = len(prev_rep)
                    report.insert( session=dbsession )

                if report.exposure_id is None:
                    raise RuntimeError('Report did not get a valid exposure_id!')
            except Exception as e:
                raise RuntimeError('Failed to create or merge a report for the exposure!') from e

            ds.report = report
        else:
            ds.report = None

        return ds, session


    def _get_stepstodo( self ):
        stepstodo = self.ALL_STEPS
        if self.pars.through_step is not None:
            if self.pars.through_step not in stepstodo:
                raise ValueError( f"Unknown through_step: \"{self.pars.through_step}\"" )
            stepstodo = stepstodo[ :stepstodo.index(self.pars.through_step)+1 ]
        return stepstodo


    def get_critical_pars_dicts( self ):
        return { 'preprocessing': self.preprocessor.get_critical.pars(),
                 'backgrounding': self.backgrounder.get_critical_pars(),
                 'extraction': self.extractor.get_critical_pars(),
                 'wcs': self.astrometor.get_critical_pars(),
                 'zp': self.photometor.get_critical_pars(),
                 'subtraction': self.subtractor.get_critical_pars(),
                 'detection': self.detector.get_critical_pars(),
                 'cutting': self.cutter.get_critical_pars(),
                 'measuring': self.measuring.get_critical_pars(),
                 'scoring': self.scorer.get_critical_pars(),
                 'report': {}
                }

    def make_provenance_tree( self,
                              ds,
                              exposure,
                              no_provtag=False,
                              all_steps=False ):
        """Create provenances for all steps in the pipeline.

        Use the current configuration of the pipeline and all the
        objects it has to generate the provenances for all the
        processing steps by calling ds.make_prov_tree()

        Start from either an Exposure or an Image; the provenance for
        the starting object must already be in the database.

        (Note that if starting from an Image, we just use that Image's
        provenance without verifying that it's consistent with the
        parameters of the preprocessing step of the pipeline.  Most of
        the time, you want to start with an exposure (hence the name of
        the parameter), as that's how the pipeline is designed.
        However, at least in some tests we use this starting with an
        Image.)

        Parameters
        ----------
        ds : DataStore
            The DataStore to make the provenance tree in.

        exposure : Exposure or Image
            The exposure to use to get the initial provenance.
            Alternatively, can be a preprocessed Image.  In either case,
            the object's provenance must already be in the database.  If
            Image, then no report will be generated even if the
            generate_report parameter is True, because reports are
            linked to exposures.

        no_provtag: bool, default False
            If True, won't create a provenance tag, and won't ensure
            that the provenances created match the provenance_tag
            parameter to the pipeline.  If False, will create the
            provenance tag if it doesn't exist.  If it does exist, will
            verify that all the provenances in the created provenance
            tree are what's tagged.

        all_steps: bool, default False
            Normally, will only generate provenances up to the
            through_step parameter of this Pipeline.  If this is True,
            it will try to generate provenances for all steps.  (If this
            is True and the DataStore can't find a reference, then an
            exception will be raised.)  As a side effect, if all_steps
            is not True and the through_step parameter isn't the
            last parameter in Pipeline.ALL_STEPS, then no report
            will be generated (because we don't know the provenance
            for it!).

        Returns
        -------
        ProvenanceTree
            A map of all the provenances that were created in this
            function, keyed according to the different steps in the
            pipeline.  (ds.prov_tree will be set to this same value.)

        """

        if not isinstance( ds, DataStore ):
            raise TypeError( "First argument to make_provenance_tree must be a DataStore." )

        if all_steps:
            stepstogenerateprov = self.ALL_STEPS
        else:
            stepstogenerateprov = self._get_stepstodo()

        self._generate_report = ( stepstogenerateprov == self.ALL_STEPS ) and ( isinstance( exposure, Exposure ) )
        if self.pars.generate_report:
            if self._generate_report:
                stepstogenerateprov.append( 'report' )
            else:
                SCLogger.warning( "generate_report is true but no report will be generated!" )

        parsdict = self.get_critical_pars_dicts()
        ds.make_prov_tree( exposure, stepstogenerateprov, parsdict,
                           provtag=None if no_provtag else self.pars.provenance_tag )
        return ds.prov_tree


    def run(self, *args, **kwargs):
        """Run the entire pipeline on a specific CCD in a specific exposure.

        Will open a database session and grab any existing data,
        and calculate and commit any new data that did not exist.

        Parameters
        ----------
        Inputs should include the exposure and section_id, or a datastore
        with these things already loaded. If a session is passed in as
        one of the arguments, it will be used as a single session for
        running the entire pipeline (instead of opening and closing
        sessions where needed).

        Returns
        -------
        ds : DataStore
            The DataStore object that includes all the data products.

        """

        ds = None
        try:
            ds, session = self.setup_datastore(*args, **kwargs)
            if session is not None:
                raise RuntimeError( "You have a persistent session in Pipeline.run; don't do that." )

            stepstodo = self._get_stepstodo()

            if ds.image is not None:
                SCLogger.info(f"Pipeline starting for image {ds.image.id} ({ds.image.filepath}), "
                              f"running through step {stepstodo[-1]}" )
            elif ds.exposure is not None:
                SCLogger.info(f"Pipeline starting for exposure {ds.exposure.id} "
                              f"({ds.exposure}) section {ds.section_id}, "
                              f"running through step {stepstodo[-1]}" )
            else:
                SCLogger.info(f"Pipeline starting with args {args}, kwargs {kwargs}, "
                              f"running through step {stepstodo[-1]}" )

            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                # ref: https://docs.python.org/3/library/tracemalloc.html#record-the-current-and-peak-size-of-all-traced-memory-blocks
                import tracemalloc
                tracemalloc.start()  # trace the size of memory that is being used

            with warnings.catch_warnings(record=True) as w:
                ds.warnings_list = w  # appends warning to this list as it goes along
                # run dark/flat preprocessing, cut out a specific section of the sensor

                if 'preprocessing' in stepstodo:
                    SCLogger.info("preprocessor")
                    ds = self.preprocessor.run(ds, session)
                    ds.update_report('preprocessing', session=None)
                    SCLogger.info(f"preprocessing complete: image id = {ds.image.id}, filepath={ds.image.filepath}")

                # extract sources and make a SourceList and PSF from the image
                if 'extraction' in stepstodo:
                    SCLogger.info(f"extractor for image id {ds.image.id}")
                    ds = self.extractor.run(ds, session)
                    ds.update_report('extraction', session=None)

                # find the background for this image
                if 'backgrounding' in stepstodo:
                    SCLogger.info(f"backgrounder for image id {ds.image.id}")
                    ds = self.backgrounder.run(ds, session)
                    ds.update_report('backgrounding', session=None)

                # find astrometric solution, save WCS into Image object and FITS headers
                if 'wcs' in stepstodo:
                    SCLogger.info(f"astrometor for image id {ds.image.id}")
                    ds = self.astrometor.run(ds, session)
                    ds.update_report('astrocal', session=None)

                # cross-match against photometric catalogs and get zero point, save into Image object and FITS headers
                if 'zp' in stepstodo:
                    SCLogger.info(f"photometor for image id {ds.image.id}")
                    ds = self.photometor.run(ds, session)
                    ds.update_report('photocal', session=None)

                if self.pars.save_before_subtraction:
                    t_start = time.perf_counter()
                    try:
                        SCLogger.info(f"Saving intermediate image for image id {ds.image.id}")
                        ds.save_and_commit(session=session)
                    except Exception as e:
                        ds.update_report('save intermediate', session=None)
                        SCLogger.error(f"Failed to save intermediate image for image id {ds.image.id}")
                        SCLogger.error(e)
                        raise e

                    ds.runtimes['save_intermediate'] = time.perf_counter() - t_start

                # fetch reference images and subtract them, save subtracted Image objects to DB and disk
                if 'subtraction' in stepstodo:
                    SCLogger.info(f"subtractor for image id {ds.image.id}")
                    ds = self.subtractor.run(ds, session)
                    ds.update_report('subtraction', session=None)

                # find sources, generate a source list for detections
                if 'detection' in stepstodo:
                    SCLogger.info(f"detector for image id {ds.image.id}")
                    ds = self.detector.run(ds, session)
                    ds.update_report('detection', session=None)

                # make cutouts of all the sources in the "detections" source list
                if 'cutting' in stepstodo:
                    SCLogger.info(f"cutter for image id {ds.image.id}")
                    ds = self.cutter.run(ds, session)
                    ds.update_report('cutting', session=None)

                # extract photometry and analytical cuts
                if 'measuring' in stepstodo:
                    SCLogger.info(f"measurer for image id {ds.image.id}")
                    ds = self.measurer.run(ds, session)
                    ds.update_report('measuring', session=None)

                # measure deep learning models on the cutouts/measurements
                if 'scoring' in stepstodo:
                    SCLogger.info(f"scorer for image id {ds.image.id}")
                    ds = self.scorer.run(ds, session)
                    ds.update_report('scoring', session=None)

                if self.pars.save_at_finish and ( 'subtraction' in stepstodo ):
                    t_start = time.perf_counter()
                    try:
                        SCLogger.info(f"Saving final products for image id {ds.image.id}")
                        ds.save_and_commit(session=session)
                    except Exception as e:
                        ds.update_report('save final', session)
                        SCLogger.error(f"Failed to save final products for image id {ds.image.id}")
                        SCLogger.error(e)
                        raise e

                    ds.runtimes['save_final'] = time.perf_counter() - t_start

                ds.finalize_report()

                return ds

        except Exception as e:
            if self.pars.save_on_exception and ( ds is not None ):
                SCLogger.error( "DataStore saving data products on pipeline exception" )
                ds.save_and_commit()
            SCLogger.exception( f"Exception in Pipeline.run: {e}" )
            if ds is not None:
                ds.exceptions.append( e )
            raise
