from models.exposure import Exposure
from models.image import Image
from models.instrument import get_instrument_instance, Instrument, SensorSection

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.preprocessing import Preprocessor
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.subtraction import Subtractor
from pipeline.detection import Detector
from pipeline.cutting import Cutter
from pipeline.measurement import Measurer

from util.config import Config

# should this come from db.py instead?
from models.base import SmartSession

# put all the top-level pipeline parameters in the init of this class:
class ParsPipeline(Parameters):

    def __init__(self, **kwargs):
            super().__init__()

            self.add_par('instrument', None, (str, None), 'The instrument the exposure/image came from' )
            self.add_par('section', None, (str, int, None), 'The SensorSection of the image' )
            self.add_par('exposure', None, (Exposure, None),
                         'The Exposure to work on; only pass one of Exposure or Image',
                         False )
            self.add_par('image', None, (Image, None),
                         'The Image to work on; only pass one of Exposure or Image',
                         False)

            self._enforce_no_new_attrs = True  # lock against new parameters

            self.override(kwargs)


class Pipeline:
    def __init__(self, **kwargs):
        """Initialize a pipeline for a single exposure and instrument setion.

        Parameters
        ----------
        pipeline : dict
          Parameters to initilize the pipeline.  Options include:
            instrument : str (required)
              The name of the instrument
            section : str (required)
              The name of the sensor section
            exposure: Exposure or None
              The exposure to work on
            image: Image or None
              The image to work on

        instrument : str or Instrument
          The instrument; will be used to set other default parameters
        section : str or SensorSection
          The sensorsection
        exposure : Exposure or None
          The exposure to work on
        image : Image or None
          The image to work on.  Only one of image or exposure can be
          non-null.  (Currently, image is not implemented.)

        preprocessing : dict or None
          Parameters to pass to the Preprocessor initialization
        extraction : dict or None
          Parameters to pass to the Detector initialization used to extract sources from the image
        astro_cal : dict or None
          Parameters to pass to the AstroCalibrator initialization
        photo_cal : dict or None
          Parameters to pass to the PhotCalibrator ininitialization
        subtraction : dict or None
          Parmeters to pass to the Subtractor initialization
        detection : dict or None
          Parameters to pass to the Detector initialization used to find candidates on a subtraction
        cutting : dict or None
          Parameters to pass to the Cutter initialization
        measurment : dict or None
          Parameters to pass to the Measurer initialization

        """
        self.config = Config.get()

        # top level parameters
        self.pars = ParsPipeline(**(self.config.value('pipeline', {})))
        self.pars.augment(kwargs.get('pipeline', {}))

        # Make sure we don't have both an exposure and an image
        if ( self.pars.exposure is not None ) and ( self.pars.image is not None ):
            raise ValueError( f'Only one of exposure and image can be non-None' )

        # Intialize required things
        if self.pars.instrument is None:
            raise RuntimeError( 'instrument is required to initialize a Pipeline' )
        if not isinstance( self.pars.instrument, Instrument ):
            instrument = get_instrument_instance( self.pars.instrument )
        self.instrument = instrument

        if self.pars.section is None:
            raise RuntimeError( 'section is required to initialize a Pipeline' )
        if not isinstance( self.pars.section, SensorSection ):
            instrument.fetch_sections()
            section = instrument.get_section( self.pars.section )
        self.section = section

        self.exposure = self.pars.exposure
        self.image = self.pars.image

        # TODO : think about implementing things when passed an image, not an exposure
        if self.exposure is None:
            raise NotImplementedError( "Pipeline currently only implemented being passed an exposure." )

        # dark/flat and sky subtraction tools
        preprocessing_config = self.config.value('preprocessing', {})
        preprocessing_config.update(kwargs.get('preprocessing', {}))
        self.pars.add_defaults_to_dict(preprocessing_config)
        self.preprocessor = Preprocessor( self.pars.instrument, self.pars.section, self.pars.exposure,
                                          **preprocessing_config)

        # source detection ("extraction" for the regular image!)
        extraction_config = self.config.value('extraction', {})
        extraction_config.update(kwargs.get('extraction', {}))
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
        measurement_config = self.config.value('measurement', {})
        measurement_config.update(kwargs.get('measurement', {}))
        self.pars.add_defaults_to_dict(measurement_config)
        self.measurer = Measurer(**measurement_config)

    def run(self, ds=None, session=None):
        """Run the entire pipeline.

        Will run on the image or exposure passed to the object
        constructor.  Will open a database session and grab any existing
        data, and calculate and commit any new data that did not exist.

        Parameters
        ----------
          ds : DataStore
            The DataStore object.  If passed, it should be consistent with
            all of the arguments passed to the object constructor.  (Usually,
            you would only pass this if you got it as a return value from
            a previous call to a run().)
          session : Session
            A database session; optional.  Illegal if ds is not None; in that case,
            this method will use the session inside the datastore.

        Returns
        -------
          DataStore 

        """

        if ds is not None:
            if session is not None:
                raise RuntimeError( "Pipeline.run: can't pass a Session when you pass a pre-existing DataStore" )
            session = ds.session
        else:
            if ( self.pars.exposure is not None ) and ( self.pars.section is not None ):
                if session is not None:
                    ds, session = DataStore.from_args( self.pars.exposure, self.pars.section, session=session )
                else:
                    ds, session = DataStore.from_args( self.pars.exposure, self.pars.section )
            else:
                raise NotImplementedError( "Pipeline currently only works if you initialize it with "
                                           "instrument, section, and exposure" )


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

