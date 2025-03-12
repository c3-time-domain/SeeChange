import datetime
import time
import warnings

import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.preprocessing import Preprocessor
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.subtraction import Subtractor
from pipeline.detection import Detector
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer
from pipeline.scoring import Scorer
from pipeline.fakeinjection import FakeInjector

from models.base import SmartSession
from models.exposure import Exposure
from models.report import Report

from util.config import Config
from util.logger import SCLogger

# describes the pipeline objects that are used to produce each step of the pipeline
# if multiple objects are used in one step, replace the string with a sub-dictionary,
# where the sub-dictionary keys are the keywords inside the expected critical parameters
# that come from all the different objects.
_PROCESS_OBJECTS = {
    'preprocessing': 'preprocessor',
    'extraction': 'extractor',
    'wcs': 'astrometor',
    'zp': 'photometor',
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
            "Save intermediate images to the database, "
            "after doing extraction and astro/photo calibration, "
            "if there is no reference, will not continue to doing subtraction "
            "but will still save the products up to that point. "
            "(It's possible the pipeline won't work if this is False...)",
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
            "any of preprocessing, extraction, wcs, zp, subtraction, detection, "
            "cutting, measuring, scoring",
            critical=False
        )

        self.inject_fakes = self.add_par(
            'inject_fakes',
            False,
            bool,
            "Inject fake transients on the image?  If true, fake injection is controlled by "
            "the configuration of the pipeline/fakeinjection.py/FakeInjector object which is "
            "configured by default from the fakeinjection dictionary in the config yaml file. "
            "Fake injection will only happen if the set of steps to do goes through at least "
            "scoring.",
            critical=False             # Non-critical because it doesn't affect any of the non-fake downstreams!
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
    ALL_STEPS = [ 'preprocessing', 'extraction', 'wcs', 'zp', 'subtraction',
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
        extraction_config = config.value('extraction', {})
        extraction_config.update(kwargs.get('extraction', {}))
        extraction_config.update({'measure_psf': True})
        self.pars.add_defaults_to_dict(extraction_config)
        self.extractor = Detector(**extraction_config)

        # astrometric fit using a first pass of sextractor and then astrometric fit to Gaia
        astrometor_config = config.value('wcs', {})
        astrometor_config.update(kwargs.get('wcs', {}))
        self.pars.add_defaults_to_dict(astrometor_config)
        self.astrometor = AstroCalibrator(**astrometor_config)

        # photometric calibration:
        photometor_config = config.value('zp', {})
        photometor_config.update(kwargs.get('zp', {}))
        self.pars.add_defaults_to_dict(photometor_config)
        self.photometor = PhotCalibrator(**photometor_config)

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

        # fake injection
        # (Make the object even if self.pars.inject_fakes is false
        # because one of our tests wants to use it.)
        fakeinjection_config = config.value( 'fakeinjection', {} )
        fakeinjection_config.update( kwargs.get( 'fakeinjection', {} ) )
        self.pars.add_defaults_to_dict( fakeinjection_config )
        self.fakeinjector = FakeInjector( **fakeinjection_config )

        # Other initialization
        self._generate_report = self.pars.generate_report

    def override_parameters(self, **kwargs):
        """Override some of the parameters for this object and its sub-objects, using Parameters.override(). """
        for key, value in kwargs.items():
            if key in _PROCESS_OBJECTS:
                if isinstance(_PROCESS_OBJECTS[key], dict):
                    for sub_key, sub_value in _PROCESS_OBJECTS[key].items():
                        if sub_key in value:
                            getattr(self, sub_value).pars.override(value[sub_key])
                elif isinstance(_PROCESS_OBJECTS[key], str):
                    getattr(self, _PROCESS_OBJECTS[key]).pars.override(value)
            else:
                self.pars.override({key: value})

    def augment_parameters(self, **kwargs):
        """Add some parameters to this object and its sub-objects, using Parameters.augment(). """
        for key, value in kwargs.items():
            if key in _PROCESS_OBJECTS:
                getattr(self, _PROCESS_OBJECTS[key]).pars.augment(value)
            else:
                self.pars.augment({key: value})

    def setup_datastore(self, *args, no_provtag=False, ok_no_ref_prov=False, **kwargs):
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

          ok_no_ref_prov: bool, default False
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

        """
        ds = DataStore.from_args(*args, **kwargs)

        if ds.exposure is not None:
            if Exposure.get_by_id( ds.exposure.id ) is None:
                raise RuntimeError( "Exposure must be loaded into the database." )
        elif ds.image is not None:
            # ...I think it's OK if image isn't in the database?  Maybe?
            # if Image.get_by_id( ds.image.id ) is None:
            #     raise RuntimeError( "Image must be loaded into the database." )
            pass
        else:
            raise RuntimeError( "Datastore must have either an image or an exposure" )

        try:  # create (and commit, if not existing) all provenances for the products
            provs = self.make_provenance_tree( ds,
                                               no_provtag=no_provtag,
                                               ok_no_ref_prov=ok_no_ref_prov,
                                               all_steps=self.pars.generate_report )
        except Exception as e:
            raise RuntimeError( f'Failed to create the provenance tree: {str(e)}' ) from e


        if self._generate_report:
            try:  # must make sure the report is on the DB
                report = Report( exposure_id=ds.exposure.id, section_id=ds.section_id )
                report.start_time = datetime.datetime.now( tz=datetime.UTC )
                report.provenance_id = provs['report'].id
                with SmartSession() as dbsession:
                    # check how many times this report was generated before
                    prev_rep = dbsession.scalars(
                        sa.select(Report).where(
                            Report.exposure_id == ds.exposure.id,
                            Report.section_id == str( ds.section_id ),
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

        return ds


    def _get_stepstodo( self ):
        stepstodo = self.ALL_STEPS.copy()
        if self.pars.through_step is not None:
            if self.pars.through_step not in stepstodo:
                raise ValueError( f"Unknown through_step: \"{self.pars.through_step}\"" )
            stepstodo = stepstodo[ :stepstodo.index(self.pars.through_step)+1 ]
        return stepstodo


    def get_critical_pars_dicts( self ):
        # The contents of this dictionary must be synced with _PROCESS_OBJECTS above.
        return { 'preprocessing': self.preprocessor.pars.get_critical_pars(),
                 'extraction': self.extractor.pars.get_critical_pars(),
                 'wcs': self.astrometor.pars.get_critical_pars(),
                 'zp': self.photometor.pars.get_critical_pars(),
                 'subtraction': self.subtractor.pars.get_critical_pars(),
                 'detection': self.detector.pars.get_critical_pars(),
                 'cutting': self.cutter.pars.get_critical_pars(),
                 'measuring': self.measurer.pars.get_critical_pars(),
                 'scoring': self.scorer.pars.get_critical_pars(),
                 'report': {}
                }

    def make_provenance_tree( self,
                              ds,
                              no_provtag=False,
                              ok_no_ref_prov=False,
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
            The DataStore to make the provenance tree in.  Will use
            either the exposure or the image that's in this DataStore.
            In either case, the object's provenance must already be in
            the database.  If there is no exposure in the DataStore,
            only an image, then no report will be generated even if the
            generate_report parameter is True, because reports are
            linked to exposures.

        no_provtag: bool, default False
            If True, won't create a provenance tag, and won't ensure
            that the provenances created match the provenance_tag
            parameter to the pipeline.  If False, will create the
            provenance tag if it doesn't exist.  If it does exist, will
            verify that all the provenances in the created provenance
            tree are what's tagged.

        ok_no_ref_prov: bool, default False
            Normally, if a refeset can't be found, or no image
            provenances associated with that refset can be found, an
            execption will be raised.  Set this to True to indicate that
            that's OK; in that case, the returned prov_tree will not
            have any provenances for steps other than preprocessing and
            extraction.

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
            stepstogenerateprov = self.ALL_STEPS.copy()
        else:
            stepstogenerateprov = self._get_stepstodo()

        if ( stepstogenerateprov == self.ALL_STEPS ) and ( self.pars.generate_report ):
            stepstogenerateprov.append( 'report' )

        parsdict = self.get_critical_pars_dicts()
        ds.make_prov_tree( stepstogenerateprov, parsdict,
                           provtag=None if no_provtag else self.pars.provenance_tag,
                           ok_no_ref_prov=ok_no_ref_prov )

        if 'report' not in ds.prov_tree:
            if self.pars.generate_report:
                SCLogger.warning( "generate_report is true but no report will be generated!" )
                self._generate_report = False
            else:
                self._generate_report = True

        return ds.prov_tree


    def run(self, *args, **kwargs):
        """Run the entire pipeline on a specific CCD in a specific exposure.

        Will open a database session and grab any existing data,
        and calculate and commit any new data that did not exist.

        Parameters
        ----------
        Inputs should include the exposure and section_id, or a datastore
        with these things already loaded.

        Returns
        -------
        ds : DataStore
            The DataStore object that includes all the data products.

        """

        ds = None
        try:
            ds = self.setup_datastore(*args, **kwargs)
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

            if ds.update_memory_usages:
                # ref: https://docs.python.org/3/library/tracemalloc.html#record-the-current-and-peak-size-of-all-traced-memory-blocks
                import tracemalloc
                tracemalloc.start()  # trace the size of memory that is being used

            with warnings.catch_warnings(record=True) as w:
                ds.warnings_list = w  # appends warning to this list as it goes along
                # run dark/flat preprocessing, cut out a specific section of the sensor

                if 'preprocessing' in stepstodo:
                    SCLogger.info("preprocessor")
                    ds = self.preprocessor.run(ds)
                    ds.update_report('preprocessing')
                    SCLogger.info(f"preprocessing complete: image id = {ds.image.id}, filepath={ds.image.filepath}")

                # extract sources and make a SourceList, PSF, and Background from the image
                if 'extraction' in stepstodo:
                    SCLogger.info(f"extractor for image id {ds.image.id}")
                    ds = self.extractor.run(ds)
                    ds.update_report('extraction')

                # find astrometric solution, save WCS into Image object and FITS headers
                if 'wcs' in stepstodo:
                    SCLogger.info(f"astrometor for image id {ds.image.id}")
                    ds = self.astrometor.run(ds)
                    ds.update_report('astrocal')

                # cross-match against photometric catalogs and get zero point, save into Image object and FITS headers
                if 'zp' in stepstodo:
                    SCLogger.info(f"photometor for image id {ds.image.id}")
                    ds = self.photometor.run(ds)
                    ds.update_report('photocal')

                if self.pars.save_before_subtraction:
                    t_start = time.perf_counter()
                    try:
                        SCLogger.info(f"Saving intermediate image for image id {ds.image.id}")
                        ds.save_and_commit()
                    except Exception as e:
                        ds.update_report('save intermediate')
                        SCLogger.error(f"Failed to save intermediate image for image id {ds.image.id}")
                        SCLogger.error(e)
                        raise e

                    if ds.update_runtimes:
                        ds.runtimes['save_intermediate'] = time.perf_counter() - t_start

                # fetch reference images and subtract them, save subtracted Image objects to DB and disk
                if 'subtraction' in stepstodo:
                    SCLogger.info(f"subtractor for image id {ds.image.id}")
                    ds = self.subtractor.run(ds)
                    ds.update_report('subtraction')

                # find sources, generate a source list for detections
                if 'detection' in stepstodo:
                    SCLogger.info(f"detector for image id {ds.image.id}")
                    ds = self.detector.run(ds)
                    ds.update_report('detection')

                # make cutouts of all the sources in the "detections" source list
                if 'cutting' in stepstodo:
                    SCLogger.info(f"cutter for image id {ds.image.id}")
                    ds = self.cutter.run(ds)
                    ds.update_report('cutting')

                # extract photometry and analytical cuts
                if 'measuring' in stepstodo:
                    SCLogger.info(f"measurer for image id {ds.image.id}")
                    ds = self.measurer.run(ds)
                    ds.update_report('measuring')

                # measure deep learning models on the cutouts/measurements
                if 'scoring' in stepstodo:
                    SCLogger.info(f"scorer for image id {ds.image.id}")
                    ds = self.scorer.run(ds)
                    ds.update_report('scoring')

                if self.pars.save_at_finish and ( 'subtraction' in stepstodo ):
                    t_start = time.perf_counter()
                    try:
                        SCLogger.info(f"Saving final products for image id {ds.image.id}")
                        ds.save_and_commit()
                    except Exception as e:
                        ds.update_report('save final')
                        SCLogger.error(f"Failed to save final products for image id {ds.image.id}")
                        SCLogger.error(e)
                        raise e

                    if ds.update_runtimes:
                        ds.runtimes['save_final'] = time.perf_counter() - t_start

                # Parallel pipeline path for fake injection
                if ( all( s in stepstodo
                          for s in [ 'subtraction', 'detection', 'cutting', 'measuring', 'scoring' ] )
                     and ( self.pars.inject_fakes )
                    ):
                    t_start = time.perf_counter()
                    if ds.update_memory_usages:
                        import tracemalloc
                        tracemalloc.reset_peak()   # start accounting for peak memory usage from here

                    # Turn off runtime and memory usage tracking inside
                    #  pipeline objects, because we want to do it as
                    #  a block for fake analysis.
                    orig_update_runtimes = ds.update_runtimes
                    orig_update_memory_usages = ds.update_memory_usages
                    try:
                        ds.update_runtimes = False
                        ds.update_memory_usages = False

                        SCLogger.info( f"Injecting fakes on to image id {ds.image.id}" )
                        fakeds = self.fakeinjector.run( ds )
                        ds.fakes = fakeds.fakes
                        if self.pars.save_at_finish:
                            ds.fakes.save()
                            ds.fakes.insert()

                        SCLogger.info( f"Running subtraction with fake-injected image id {ds.image.id}" )
                        fakeds = self.subtractor.run( fakeds )

                        SCLogger.info( f"Running detection on fake-injected subtraction of image id {ds.image.id}" )
                        fakeds = self.detector.run( fakeds )

                        SCLogger.info( f"Running cutting on fake-injected subtraction of image id {ds.image.id}" )
                        fakeds = self.cutter.run( fakeds )

                        SCLogger.info( f"Running measuring on fake-injected subtraction of image id {ds.image.id}" )
                        fakeds = self.measurer.run( fakeds )

                        SCLogger.info( f"Running scoring of detections on fake-injected subtraction "
                                       f"of image id {ds.image.id}" )
                        fakeds = self.scorer.run( fakeds )

                        SCLogger.info( f"Looking to see which fakes are detected on fake-injected subtraction "
                                       f"of image id {ds.image.id}" )
                        ds.fakeanal = self.fakeinjector.analyze_fakes( fakeds, ds )
                        ds.fakeanal.orig_deepscore_set_id = ds.deepscore_set.id
                        if self.pars.save_at_finish:
                            ds.fakeanal.save()
                            ds.fakeanal.insert()

                        if orig_update_runtimes:
                            ds.runtimes[ 'fakeanalysis' ] = time.perf_counter() - t_start
                        if orig_update_memory_usages:
                            import tracemalloc
                            ds.memory_usages[ 'fakeanalysis' ] = tracemalloc.get_traced_memory()[1] / 1024**2  # in MiB

                        ds.update_report( 'fakeanalysis' )
                    finally:
                        ds.update_runtimes = orig_update_runtimes
                        ds.update_memory_usages = orig_update_memory_usages

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
