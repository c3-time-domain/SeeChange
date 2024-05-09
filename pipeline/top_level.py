
from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.preprocessing import Preprocessor
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.subtraction import Subtractor
from pipeline.detection import Detector
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer

from util.config import Config

# should this come from db.py instead?
from models.base import SmartSession

# describes the pipeline objects that are used to produce each step of the pipeline
# if multiple objects are used in one step, replace the string with a sub-dictionary,
# where the sub-dictionary keys are the keywords inside the expected critical parameters
# that come from all the different objects.
PROCESS_OBJECTS = {
    'preprocessing': 'preprocessor',
    'extraction': 'extractor',  # the same object also makes the PSF (and background?)
    # TODO: when joining the astro/photo cal into extraction, use this format:
    # 'extraction': {
    #     'sources': 'extractor',
    #     'astro_cal': 'astro_cal',
    #     'photo_cal': 'photo_cal',
    # }
    'astro_cal': 'astro_cal',
    'photo_cal': 'photo_cal',
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
            'example_pipeline_parameter', 1, int, 'an example pipeline parameter'
        )

        self._enforce_no_new_attrs = True  # lock against new parameters

        self.override(kwargs)


class Pipeline:
    def __init__(self, **kwargs):
        self.config = Config.get()

        # top level parameters
        self.pars = ParsPipeline(**(self.config.value('pipeline', {})))
        self.pars.augment(kwargs.get('pipeline', {}))

        # dark/flat and sky subtraction tools
        preprocessing_config = self.config.value('preprocessing', {})
        preprocessing_config.update(kwargs.get('preprocessing', {}))
        self.pars.add_defaults_to_dict(preprocessing_config)
        self.preprocessor = Preprocessor(**preprocessing_config)

        # source detection ("extraction" for the regular image!)
        extraction_config = self.config.value('extraction', {})
        extraction_config.update(kwargs.get('extraction', {'measure_psf': True}))
        self.pars.add_defaults_to_dict(extraction_config)
        self.extractor = Detector(**extraction_config)

        # astrometric fit using a first pass of sextractor and then astrometric fit to Gaia
        astro_cal_config = self.config.value('astro_cal', {})
        astro_cal_config.update(kwargs.get('astro_cal', {}))
        self.pars.add_defaults_to_dict(astro_cal_config)
        self.astro_cal = AstroCalibrator(**astro_cal_config)

        # photometric calibration:
        photo_cal_config = self.config.value('photo_cal', {})
        photo_cal_config.update(kwargs.get('photo_cal', {}))
        self.pars.add_defaults_to_dict(photo_cal_config)
        self.photo_cal = PhotCalibrator(**photo_cal_config)

        # reference fetching and image subtraction
        subtraction_config = self.config.value('subtraction', {})
        subtraction_config.update(kwargs.get('subtraction', {}))
        self.pars.add_defaults_to_dict(subtraction_config)
        self.subtractor = Subtractor(**subtraction_config)

        # source detection ("detection" for the subtracted image!)
        detection_config = self.config.value('detection', {})
        detection_config.update(kwargs.get('detection', {}))
        self.pars.add_defaults_to_dict(detection_config)
        self.detector = Detector(**detection_config)
        self.detector.pars.subtraction = True

        # produce cutouts for detected sources:
        cutting_config = self.config.value('cutting', {})
        cutting_config.update(kwargs.get('cutting', {}))
        self.pars.add_defaults_to_dict(cutting_config)
        self.cutter = Cutter(**cutting_config)

        # measure photometry, analytical cuts, and deep learning models on the Cutouts:
        measurement_config = self.config.value('measuring', {})
        measurement_config.update(kwargs.get('measuring', {}))
        self.pars.add_defaults_to_dict(measurement_config)
        self.measurer = Measurer(**measurement_config)

    def run(self, *args, **kwargs):
        """
        Run the entire pipeline on a specific CCD in a specific exposure.
        Will open a database session and grab any existing data,
        and calculate and commit any new data that did not exist.
        """

        ds, session = DataStore.from_args(*args, **kwargs)

        # run dark/flat and sky subtraction tools, save the results as Image objects to DB and disk
        ds = self.preprocessor.run(ds, session)

        # extract sources and make a SourceList from the regular image
        ds = self.extractor.run(ds, session)

        # find astrometric solution, save WCS into Image object and FITS headers
        ds = self.astro_cal.run(ds, session)

        # cross-match against photometric catalogs and get zero point, save into Image object and FITS headers
        ds = self.photo_cal.run(ds, session)

        # fetch reference images and subtract them, save SubtractedImage objects to DB and disk
        ds = self.subtractor.run(ds, session)

        # find sources, generate a source list for detections
        ds = self.detector.run(ds, session)

        # make cutouts of all the sources in the "detections" source list
        ds = self.cutter.run(ds, session)

        # extract photometry, analytical cuts, and deep learning models on the Cutouts:
        ds = self.measurer.run(ds, session)

        return ds

    def run_with_session(self):
        """
        Run the entire pipeline using one session that is opened
        at the beginning and closed at the end of the session,
        just to see if that causes any problems with too many open sessions.
        """
        with SmartSession() as session:
            self.run(session=session)

    def make_provenance_tree(self, exposure, session=None, commit=True):
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
        session : SmartSession, optional
            The function needs to work with the database to merge existing provenances.
            If a session is given, it will use that, otherwise it will open a new session,
            which will also close automatically at the end of the function.
        commit: bool, optional, default True
            By default, the provenances are merged and committed inside this function.
            To disable this, set commit=False. This may leave the provenances in a
            transient state, and is most likely not what you want.

        """
        provs = {'exposure': exposure.provenance}  # TODO: does this always work on any exposure?

        for step in

