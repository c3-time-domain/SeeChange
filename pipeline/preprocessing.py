from types import SimpleNamespace

import numpy as np
import sqlalchemy as sa

from models.base import SmartSession, _logger
from models.exposure import Exposure, ExposureImageIterator
from models.image import Image
from models.datafile import DataFile
from models.instrument import get_instrument_instance, Instrument, SensorSection
from models.enums_and_bitflags import image_preprocessing_inverse, string_to_bitflag

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from util.config import Config

class ParsPreprocessor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.use_sky_subtraction = self.add_par('use_sky_subtraction', False, bool, 'Apply sky subtraction. ',
                                                critical=True)
        self.add_par( 'steps', None, ( list, None ), "Steps to do; don't specify, or pass None, to do all." )
        self.add_par( 'calibset', None, ( str, None ),
                      ( "One of the CalibratorSetConverter enum; "
                        "the calibrator set to use.  Defaults to the instrument default" ),
                      critical = True )
        self.add_alias( 'calibrator_set', 'calibset' )
        self.add_par( 'flattype', None, ( str, None ),
                      ( "One of the FlatTypeConverter enum; defaults to the instrument default" ),
                      critical = True )

        # A note about provenance.
        #
        # Ideally, we want the flats/biases/etc. that go into image
        # preprocessing be part of the provenance.  However, we also
        # would like for all of the chips on one instrument to have the
        # same provenance (when appropriate).  If the fileid of the
        # calibrator were in the provenance, it would be different for
        # every chip.  Hence, the defintion of "flat_set", "dark_set",
        # etc.  We can tag things in the CalibratorFiles models in the
        # database with a calbrator_set, and *that* is what will go into
        # the provenance.  It's then up to the users to make sure that
        # the calibrator_set field is used properly.

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'preprocessing'


class Preprocessor:
    def __init__(self, **kwargs):
        """Create a preprocessor.

        Preprocessing is instrument-defined, but usually includes a subset of:
          * overscan subtraction
          * bias (zero) subtraction
          * dark current subtraction
          * linearity correction
          * flatfielding
          * fringe correction
          * illumination correction

        After initialization, just call run() to perform the
        preprocessing.  This will return a DataStore with the
        preprocessed image.

        Parameters
        ----------
        instrument: Instrument or str
          The instrument we're working on
        section: SensorSectin or str
          The SensorSection of the image
        exposure: Exposure
          The exposure we're working on.

        """

        self.pars = ParsPreprocessor( **kwargs )

        # Things that get cached
        self.instrument = None
        self.stepfilesids = {}
        self.stepfiles = {}

        # TODO : remove this if/when we actually put sky subtraction in run()
        if self.pars.use_sky_subtraction:
            raise NotImplementedError( "Sky subtraction in preprocessing isn't implemented." )

    def run( self, *args, **kwargs ):
        """Run preprocessing for a given exposure and section_identifier.

        Parameters are passed to the data_store constructor (see
        DataStore.parse_args).  For preprocessing, an exposure and a
        sensorsection is required, so args must be one of:
          - DataStore (which has an exposure and a section)
          - exposure_id, seciton_identifier
          - Exposure, section_identifier
        Passing just an image won't work.

        kwargs can also include things that override the preprocessing
        behavior.  (TODO: document this)

        Returns
        -------
        DataStore
          contains the products of the processing.

        """

        ds, session = DataStore.from_args( *args, **kwargs )

        # This is here just for testing purposes
        self._ds = ds

        if ( ds.exposure is None ) or ( ds.section_id is None ):
            raise RuntimeError( "Preprocessing requires an exposure and a sensor section" )

        cfg = Config.get()

        if ( self.instrument is None ) or ( self.instrument.name != ds.exposure.instrument ):
            self.instrument = ds.exposure.instrument_object

        # The only reason these are saved in self, rather than being
        # local variables, is so that tests can probe them
        self._calibset = None
        self._flattype = None
        self._stepstodo = None

        if 'calibset' in kwargs:
            self._calibset = kwargs['calibset']
        elif 'calibratorset' in kwargs:
            self._calibset = kwargs['calibrator_set']
        elif self.pars.calibset is not None:
            self._calibset = self.pars.calibset
        else:
            self._calibset = cfg.value( f'{self.instrument.name}.calibratorset',
                                        default=cfg.value( 'instrument_default.calibratorset' ) )

        if 'flattype' in kwargs:
            self._flattype = kwargs['flattype']
        elif self.pars.flattype is not None:
            self._flattype = self.pars.flattype
        else:
            self._flattype = cfg.value( f'{self.instrument.name}.flattype',
                                        default=cfg.value( 'instrument_default.flattype' ) )

        if 'steps' in kwargs:
            self._stepstodo = [ s for s in self.instrument.preprocessing_steps if s in kwargs['steps'] ]
        elif self.pars.steps is not None:
            self._stepstodo = [ s for s in self.instrument.preprocessing_steps if s in self.pars.steps ]
        else:
            self._stepstodo = self.instrument.preprocessing_steps

        # Get the calibrator files

        preprocparam = self.instrument.preprocessing_calibrator_params( self._calibset,
                                                                        self._flattype,
                                                                        ds.section_id,
                                                                        ds.exposure.filter_short,
                                                                        ds.exposure.mjd,
                                                                        session = session )


        # get the provenance for this step, using the current parameters:
        # Provenance includes not just self.pars.get_critical_pars(),
        # but also the steps that were performed.  Reason: we may well
        # load non-flatfielded images in the database for purposes of
        # collecting images used for later building flats.  We will then
        # flatfield those images.  The two images in the database must have
        # different provenances.
        # We also include any overrides to calibrator files, as that indicates
        # that something individual happened here that's different from
        # normal processing of the image.
        provdict = dict( self.pars.get_critical_pars() )
        provdict['preprocessing_steps' ] = self._stepstodo
        prov = ds.get_provenance(self.pars.get_process_name(), provdict, session=session)

        # check if the image already exists in memory or in the database:
        image = ds.get_image(prov, session=session)

        if image is None:  # need to make new image
            # get the single-chip image from the exposure
            image = Image.from_exposure( ds.exposure, ds.section_id )

        if image is None:
            raise ValueError('Image cannot be None at this point!')

        if image.preproc_bitflag is None:
            image.preproc_bitflag = 0

        # Overscan is always first (as it reshapes the image)
        if 'overscan' in self._stepstodo:
            image.data = self.instrument.overscan_and_trim( image )
            image.preproc_bitflag |= string_to_bitflag( 'overscan', image_preprocessing_inverse )
        else:
            image.data = image.raw_data

        # Apply steps in the order expected by the instrument
        for step in self._stepstodo:
            if step == 'overscan':
                continue

            stepfileid = None
            # Acquire the calibration file
            if f'{step}_fileid' in kwargs:
                stepfileid = kwargs[ f'{step}_fileid' ]
            elif f'{step}_fileid' in preprocparam:
                stepfileid = preprocparam[ f'{step}_fileid' ]
            else:
                raise RuntimeError( f"Can't find calibration file for preprocessing step {step}" )

            if stepfileid is None:
                _logger.warning( f"Skipping step {step} for filter {ds.exposure.filter_short} "
                                 f"because there is no calibration file (this may be normal)" )
                continue

            # Use the cached calibrator file for this step if it's the right one; otherwise, grab it
            if ( stepfileid in self.stepfilesids ) and ( self.stepfilesids[step] == stepfileid ):
                calibfile = self.stepfiles[ calibfile ]
            else:

                with SmartSession( session ) as session:
                    if step in [ 'zero', 'dark', 'flat', 'illumination', 'fringe' ]:
                        calibfile = session.get( Image, stepfileid )
                        if calibfile is None:
                            raise RuntimeError( f"Unable to load image id {stepfileid} for preproc step {step}" )
                    elif step == 'linearity':
                        calibfile = session.get( DataFile, stepfileid )
                        if calibfile is None:
                            raise RuntimeError( f"Unable to load datafile id {stepfileid} for preproc step {step}" )
                    else:
                        raise ValueError( f"Preprocessing step {step} has an unknown file type (image vs. datafile)" )
                self.stepfilesids[ step ] = stepfileid
                self.stepfiles[ step ] = calibfile
            if step in [ 'zero', 'dark' ]:
                # Subtract zeros and darks
                image.data -= calibfile.data

            elif step in [ 'flat', 'illumination' ]:
                # Divide flats and illuminations
                image.data /= calibfile.data

            elif step == 'fringe':
                # TODO FRINGE CORRECTION
                _logger.warning( "Fringe correction not implemented" )

            elif step == 'linearity':
                # Linearity is instrument-specific
                self.instrument.linearity_correct( image, linearitydata=calibfile )

            else:
                # TODO: Replace this with a call into an instrument method?
                # In that case, the logic above about acquiring step files
                # will need to be updated.
                raise ValueError( f"Unknown preprocessing step {step}" )

            image.preproc_bitflag |= string_to_bitflag( step, image_preprocessing_inverse )

        # TODO : Issue #95
        _logger.warning( "Weight and dataquality flag file creation not yet implemented." )

        if image.provenance is None:
            image.provenance = prov
        else:
            if image.provenance.id != prov.id:
                # Logically, this should never happen
                raise ValueError('Provenance mismatch for image and provenance!')

        ds.image = image
        ds.image.filepath = ds.image.invent_filepath()

        return ds
