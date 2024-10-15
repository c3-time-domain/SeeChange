import io
import datetime
import time
import warnings

import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore, UPSTREAM_STEPS
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
from models.provenance import CodeVersion, Provenance, ProvenanceTag, ProvenanceTagExistsError
from models.refset import RefSet
from models.exposure import Exposure
from models.image import Image
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

        self._enforce_no_new_attrs = True  # lock against new parameters

        self.override(kwargs)


class Pipeline:
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
            provs = self.make_provenance_tree( ds.exposure,
                                               no_provtag=no_provtag,
                                               ok_no_ref_provs=ok_no_ref_provs )
            ds.prov_tree = provs
        except Exception as e:
            raise RuntimeError( f'Failed to create the provenance tree: {str(e)}' ) from e


        try:  # must make sure the report is on the DB
            report = Report( exposure_id=ds.exposure.id, section_id=ds.section_id )
            report.start_time = datetime.datetime.now( tz=datetime.timezone.utc )
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

        return ds, session

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
        try:  # first make sure we get back a datastore, even an empty one
            ds, session = self.setup_datastore(*args, **kwargs)
        except Exception as e:
            return DataStore.catch_failure_to_parse(e, *args)

        if session is not None:
            raise RuntimeError( "You have a persistent session in Pipeline.run; don't do that." )

        try:
            stepstodo = [ 'preprocessing', 'backgrounding', 'extraction', 'wcs', 'zp', 'subtraction',
                          'detection', 'cutting', 'measuring', 'scoring', ]
            if self.pars.through_step is not None:
                if self.pars.through_step not in stepstodo:
                    raise ValueError( f"Unknown through_step: \"{self.parse.through_step}\"" )
                stepstodo = stepstodo[ :stepstodo.index(self.pars.through_step)+1 ]

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
                    SCLogger.info(f"preprocessor")
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
                    ds.update_report('extraction', session=None)

                # find astrometric solution, save WCS into Image object and FITS headers
                if 'wcs' in stepstodo:
                    SCLogger.info(f"astrometor for image id {ds.image.id}")
                    ds = self.astrometor.run(ds, session)
                    ds.update_report('extraction', session=None)

                # cross-match against photometric catalogs and get zero point, save into Image object and FITS headers
                if 'zp' in stepstodo:
                    SCLogger.info(f"photometor for image id {ds.image.id}")
                    ds = self.photometor.run(ds, session)
                    ds.update_report('extraction', session=None)

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
            if self.pars.save_on_exception:
                SCLogger.error( "DataStore saving data products on pipeline exception" )
                ds.save_and_commit()
            ds.catch_exception(e)
        finally:
            # make sure the DataStore is returned in case the calling scope want to debug the pipeline run
            return ds

    def run_with_session(self):
        """
        Run the entire pipeline using one session that is opened
        at the beginning and closed at the end of the session,
        just to see if that causes any problems with too many open sessions.
        """
        with SmartSession() as session:
            self.run(session=session)

    def make_provenance_tree( self, exposure, overrides=None, no_provtag=False, ok_no_ref_provs=False ):
        """Create provenances for all steps in the pipeline.

        Use the current configuration of the pipeline and all the
        objects it has to generate the provenances for all the
        processing steps.

        This will conclude with the reporting step, which simply has an
        upstreams list of provenances to the measuring provenance and to
        the machine learning score provenances. From those, a user can
        recreate the entire tree of provenances.  (Note: if
        ok_no_ref_provs is True, and no referencing provenances are
        found, then the report provenance will have the extraction
        provenance as its upstream, as there will be no measuring
        provenance.)

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
        exposure : Exposure or Image
            The exposure to use to get the initial provenance.
            Alternatively, can be a preprocessed Image.  In either case,
            the object's provenance must already be in the database.

        overrides: dict, optional
            A dictionary of provenances to override any of the steps in
            the pipeline.  For example, set overrides={'preprocessing':
            prov} to use a specific provenance for the basic Image
            provenance.

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

        Returns
        -------
        dict
            A dictionary of all the provenances that were created in this function,
            keyed according to the different steps in the pipeline.
            The provenance will all be inserted into the database if necessary.

        """
        if overrides is None:
            overrides = {}

        provs = {}

        code_version = None
        is_testing = None

        # Get started with the passed Exposure (usual case) or Image
        if isinstance( exposure, Exposure ):
            with SmartSession() as session:
                exp_prov = Provenance.get( exposure.provenance_id, session=session )
                code_version = CodeVersion.get_by_id( exp_prov.code_version_id, session=session )
            provs['exposure'] = exp_prov
            is_testing  = exp_prov.is_testing
        elif isinstance( exposure, Image ):
            exp_prov = None
            with SmartSession() as session:
                passed_image_provenance = Provenance.get( exposure.provenance_id, session=session )
                code_version = CodeVersion.get_by_id( passed_image_provenance.code_version_id, session=session )
            is_testing = passed_image_provenance.is_testing
        else:
            raise TypeError( f"The first parameter to make_provenance_tree must be an Exposure or Image, "
                             f"not a {exposure.__class__.__name__}" )

        # Get the reference
        ref_provs = None  # allow multiple reference provenances for each refset
        refset_name = self.subtractor.pars.refset
        # If refset is None, we will just fail to produce a subtraction, but everything else works...
        # Note that the upstreams for the subtraction provenance will be wrong, because we don't have
        # any reference provenances to link to. But this is what you get when putting refset=None.
        # Just know that the "output provenance" (e.g., of the Measurements) will never actually exist,
        # even though you can use it to make the Report provenance (just so you have something to refer to).
        if refset_name is not None:
            refset = RefSet.get_by_name( refset_name )
            if refset is None:
                if not ok_no_ref_provs:
                    raise ValueError(f'No reference set with name {refset_name} found in the database!')
                else:
                    ref_provs = None
            else:
                ref_provs = refset.provenances
                if ref_provs is None or len(ref_provs) == 0:
                    if not ok_no_ref_provs:
                        raise ValueError(f'No provenances found for reference set {refset_name}!')
                    ref_provs = None

        if ref_provs is not None:
            provs['referencing'] = ref_provs  # notice that this is a list, not a single provenance!

        for step in PROCESS_OBJECTS:
            if ( ref_provs is None ) and ( step == 'subtraction' ):
                # If we don't have reference provenances, we can't build the rest
                break

            if step in overrides:
                # accept explicit provenances specified by the user as overrides
                # TODO: worry if step is preprocessing and an Image was passed;
                #   there could be an inconsistency!
                provs[step] = overrides[step]
            else:
                # special case handling for 'preprocessing' if we don't have an exposure
                if ( step == 'preprocessing' ) and exp_prov is None:
                    provs[step] = passed_image_provenance
                else:
                    # load the parameters from the objects on the pipeline
                    obj_name = PROCESS_OBJECTS[step]  # translate the step to the object name
                    if isinstance(obj_name, dict):
                        # sub-objects, e.g., extraction.sources, extraction.wcs, etc.
                        # get the first item of the dictionary and hope its pars object has siblings defined correctly:
                        obj_name = obj_name.get(list(obj_name.keys())[0])
                    parameters = getattr(self, obj_name).pars.get_critical_pars()

                    # figure out which provenances go into the upstreams for this step
                    up_steps = UPSTREAM_STEPS[step]
                    if isinstance(up_steps, str):
                        up_steps = [up_steps]
                    upstream_provs = []
                    for upstream in up_steps:
                        if upstream == 'referencing':  # this is an externally supplied provenance upstream
                            if ref_provs is not None:
                                # We never put the Reference object's provenance into the upstreams of the subtraction
                                # instead, put the provenances of the coadd image and its extraction products
                                # this is so the subtraction provenance has the (preprocessing+extraction) provenance
                                # for each one of its upstream_images (in this case, ref+new).
                                # By construction all references on the refset SHOULD have the same upstreams.
                                upstream_provs += ref_provs[0].upstreams
                        else:  # just grab the provenance of what is upstream of this step from the existing tree
                            upstream_provs.append(provs[upstream])

                    provs[step] = Provenance(
                        code_version_id=code_version.id,
                        process=step,
                        parameters=parameters,
                        upstreams=upstream_provs,
                        is_testing=is_testing,
                    )
                    provs[step].insert_if_needed()

        # Make the report provenance
        if ( 'measuring' not in provs ) and ( not ok_no_ref_provs ):
            raise RuntimeError( "Something has gone wrong; we didn't create a measuring provenance, but "
                                "ok_no_ref_provs is False.  We should have errored out before this message." )
        rptupstr = provs['measuring'] if 'measuring' in provs else provs['extraction']
        provs['report'] = Provenance(
            process='report',
            code_version_id=code_version.id,
            parameters={},
            upstreams=[rptupstr],
            is_testing=is_testing
        )
        provs['report'].insert_if_needed()

        # Ensure that the provenance tag is right, creating it if it doesn't exist
        if not no_provtag:
            provtag = self.pars.provenance_tag
            try:
                provids = []
                for prov in provs.values():
                    if isinstance( prov, list ):
                        provids.extend( [ i.id for i in prov ] )
                    else:
                        provids.append( prov.id )
                ProvenanceTag.newtag( provtag, provids )
            except ProvenanceTagExistsError as ex:
                pass

            # The rest of this could be inside the except block,
            #   but leaving it outside verifies that the
            #   ProvenanceTag.newtag worked properly.
            missing = []
            with SmartSession() as sess:
                ptags = sess.query( ProvenanceTag ).filter( ProvenanceTag.tag==provtag ).all()
                ptag_pids = [ pt.provenance_id for pt in ptags ]
            for step, prov in provs.items():
                if isinstance( prov, list ):
                    missing.extend( [ i for i in prov if i.id not in ptag_pids ] )
                elif prov.id not in ptag_pids:
                    missing.append( prov )
            if len( missing ) != 0:
                strio = io.StringIO()
                strio.write( f"The following provenances are not associated with provenance tag {provtag}:\n " )
                for prov in missing:
                    strio.write( f"   {prov.process}: {prov.id}\n" )
                SCLogger.error( strio.getvalue() )
                raise RuntimeError( strio.getvalue() )

        return provs


