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

from models.base import SmartSession, merge_concurrent
from models.provenance import Provenance, ProvenanceTag, ProvenanceTagExistsError
from models.refset import RefSet
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
    # TODO: add one more for R/B deep learning scores
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

    def setup_datastore(self, *args, **kwargs):
        """Initialize a datastore, including an exposure and a report, to use in the pipeline run.

        Will raise an exception if there is no valid Exposure,
        if there's no reference available, or if the report cannot
        be posted to the database.

        After these objects are instantiated, the pipeline will proceed
        and record any exceptions into the report object before raising them.

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
            The DataStore object that was created or loaded.
        session: sqlalchemy.orm.session.Session
            An optional session. If not given, this will be None
        """
        ds, session = DataStore.from_args(*args, **kwargs)

        if ds.exposure is None:
            raise RuntimeError('Cannot run this pipeline method without an exposure!')

        try:  # must make sure the exposure is on the DB
            ds.exposure = ds.exposure.merge_concurrent(session=session)
        except Exception as e:
            raise RuntimeError('Failed to merge the exposure into the session!') from e

        try:  # create (and commit, if not existing) all provenances for the products
            with SmartSession(session) as dbsession:
                provs = self.make_provenance_tree(ds.exposure, session=dbsession, commit=True)
        except Exception as e:
            raise RuntimeError('Failed to create the provenance tree!') from e

        try:  # must make sure the report is on the DB
            report = Report(exposure=ds.exposure, section_id=ds.section_id)
            report.start_time = datetime.datetime.utcnow()
            report.provenance = provs['report']
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
                report = merge_concurrent( report, dbsession, True )

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

        try:
            if ds.image is not None:
                SCLogger.info(f"Pipeline starting for image {ds.image.id} ({ds.image.filepath})")
            elif ds.exposure is not None:
                SCLogger.info(f"Pipeline starting for exposure {ds.exposure.id} ({ds.exposure}) section {ds.section_id}")
            else:
                SCLogger.info(f"Pipeline starting with args {args}, kwargs {kwargs}")

            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                # ref: https://docs.python.org/3/library/tracemalloc.html#record-the-current-and-peak-size-of-all-traced-memory-blocks
                import tracemalloc
                tracemalloc.start()  # trace the size of memory that is being used

            with warnings.catch_warnings(record=True) as w:
                ds.warnings_list = w  # appends warning to this list as it goes along
                # run dark/flat preprocessing, cut out a specific section of the sensor

                SCLogger.info(f"preprocessor")
                ds = self.preprocessor.run(ds, session)
                ds.update_report('preprocessing', session=None)
                SCLogger.info(f"preprocessing complete: image id = {ds.image.id}, filepath={ds.image.filepath}")

                # extract sources and make a SourceList and PSF from the image
                SCLogger.info(f"extractor for image id {ds.image.id}")
                ds = self.extractor.run(ds, session)
                ds.update_report('extraction', session=None)

                # find the background for this image
                SCLogger.info(f"backgrounder for image id {ds.image.id}")
                ds = self.backgrounder.run(ds, session)
                ds.update_report('extraction', session=None)

                # find astrometric solution, save WCS into Image object and FITS headers
                SCLogger.info(f"astrometor for image id {ds.image.id}")
                ds = self.astrometor.run(ds, session)
                ds.update_report('extraction', session=None)

                # cross-match against photometric catalogs and get zero point, save into Image object and FITS headers
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
                SCLogger.info(f"subtractor for image id {ds.image.id}")
                ds = self.subtractor.run(ds, session)
                ds.update_report('subtraction', session=None)

                # find sources, generate a source list for detections
                SCLogger.info(f"detector for image id {ds.image.id}")
                ds = self.detector.run(ds, session)
                ds.update_report('detection', session=None)

                # make cutouts of all the sources in the "detections" source list
                SCLogger.info(f"cutter for image id {ds.image.id}")
                ds = self.cutter.run(ds, session)
                ds.update_report('cutting', session=None)

                # extract photometry and analytical cuts
                SCLogger.info(f"measurer for image id {ds.image.id}")
                ds = self.measurer.run(ds, session)
                ds.update_report('measuring', session=None)

                # measure deep learning models on the cutouts/measurements
                # TODO: add this...

                if self.pars.save_at_finish:
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

                ds.finalize_report(session)

                return ds

        except Exception as e:
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

    def make_provenance_tree( self, exposure, overrides=None, session=None, no_provtag=False, commit=True ):
        """Use the current configuration of the pipeline and all the objects it has
        to generate the provenances for all the processing steps.
        This will conclude with the reporting step, which simply has an upstreams
        list of provenances to the measuring provenance and to the machine learning score
        provenances. From those, a user can recreate the entire tree of provenances.

        Parameters
        ----------
        exposure : Exposure
            The exposure to use to get the initial provenance.
            This provenance should be automatically created by the exposure.

        overrides: dict, optional
            A dictionary of provenances to override any of the steps in
            the pipeline.  For example, set overrides={'preprocessing':
            prov} to use a specific provenance for the basic Image
            provenance.

        session : SmartSession, optional
            The function needs to work with the database to merge
            existing provenances.  If a session is given, it will use
            that, otherwise it will open a new session, which will also
            close automatically at the end of the function.

        no_provtag: bool, default False
            If True, won't create a provenance tag, and won't ensure
            that the provenances created match the provenance_tag
            parameter to the pipeline.  If False, will create the
            provenance tag if it doesn't exist.  If it does exist, will
            verify that all the provenances in the created provenance
            tree are what's tagged

        commit: bool, optional, default True
            By default, the provenances are merged and committed inside
            this function.  To disable this, set commit=False. This may
            leave the provenances in a transient state, and is most
            likely not what you want.

        Returns
        -------
        dict
            A dictionary of all the provenances that were created in this function,
            keyed according to the different steps in the pipeline.
            The provenances are all merged to the session.

        """
        if overrides is None:
            overrides = {}

        if ( not no_provtag ) and ( not commit ):
            raise RuntimeError( "Commit required when no_provtag is not set" )

        with SmartSession(session) as sess:
            # start by getting the exposure and reference
            exp_prov = sess.merge(exposure.provenance)  # also merges the code_version
            provs = {'exposure': exp_prov}
            code_version = exp_prov.code_version
            is_testing = exp_prov.is_testing

            ref_provs = None  # allow multiple reference provenances for each refset
            refset_name = self.subtractor.pars.refset
            # If refset is None, we will just fail to produce a subtraction, but everything else works...
            # Note that the upstreams for the subtraction provenance will be wrong, because we don't have
            # any reference provenances to link to. But this is what you get when putting refset=None.
            # Just know that the "output provenance" (e.g., of the Measurements) will never actually exist,
            # even though you can use it to make the Report provenance (just so you have something to refer to).
            if refset_name is not None:

                refset = sess.scalars(sa.select(RefSet).where(RefSet.name == refset_name)).first()
                if refset is None:
                    raise ValueError(f'No reference set with name {refset_name} found in the database!')

                ref_provs = refset.provenances
                if ref_provs is None or len(ref_provs) == 0:
                    raise ValueError(f'No provenances found for reference set {refset_name}!')

            provs['referencing'] = ref_provs  # notice that this is a list, not a single provenance!

            for step in PROCESS_OBJECTS:  # produce the provenance for this step
                if step in overrides:  # accept override from user input
                    provs[step] = overrides[step]
                else:  # load the parameters from the objects on the pipeline
                    obj_name = PROCESS_OBJECTS[step]  # translate the step to the object name
                    if isinstance(obj_name, dict):  # sub-objects, e.g., extraction.sources, extraction.wcs, etc.
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
                                # we never put the Reference object's provenance into the upstreams of the subtraction
                                # instead, put the provenances of the coadd image and its extraction products
                                # this is so the subtraction provenance has the (preprocessing+extraction) provenance
                                # for each one of its upstream_images (in this case, ref+new).
                                # by construction all references on the refset SHOULD have the same upstreams
                                upstream_provs += ref_provs[0].upstreams
                        else:  # just grab the provenance of what is upstream of this step from the existing tree
                            upstream_provs.append(provs[upstream])

                    provs[step] = Provenance(
                        code_version=code_version,
                        process=step,
                        parameters=parameters,
                        upstreams=upstream_provs,
                        is_testing=is_testing,
                    )

                provs[step] = provs[step].merge_concurrent(session=sess, commit=commit)

            # Make the report provenance
            prov = Provenance(
                process='report',
                code_version=exposure.provenance.code_version,
                parameters={},
                upstreams=[provs['measuring']],
                is_testing=exposure.provenance.is_testing,
            )
            provs['report'] = prov.merge_concurrent( session=sess, commit=commit )

            if commit:
                sess.commit()

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
                    ProvenanceTag.newtag( provtag, provids, session=session )
                except ProvenanceTagExistsError as ex:
                    pass

                # The rest of this could be inside the except block,
                #   but leaving it outside verifies that the
                #   ProvenanceTag.newtag worked properly.
                missing = []
                with SmartSession( session ) as sess:
                    ptags = sess.query( ProvenanceTag ).filter( ProvenanceTag.tag==provtag ).all()
                    ptag_pids = [ pt.provenance_id for pt in ptags ]
                for step, prov in provs.items():
                    if isinstance( prov, list ):
                        missing.extend( [ i.id for i in prov if i.id not in ptag_pids ] )
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


