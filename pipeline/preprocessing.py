import os
import pathlib
import time

import numpy as np

import sep

from models.base import SmartSession
from models.image import Image
from models.datafile import DataFile
from models.enums_and_bitflags import image_preprocessing_inverse, string_to_bitflag, flag_image_bits_inverse

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from util.config import Config
from util.logger import SCLogger
from util.util import parse_bool


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

        Parameters are parsed by ParsPreprocessor

        """

        self.pars = ParsPreprocessor( **kwargs )

        # Things that get cached
        self.instrument = None
        self.stepfilesids = {}
        self.stepfiles = {}

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

        # TODO : remove this if/when we actually put sky subtraction in run()
        if self.pars.use_sky_subtraction:
            raise NotImplementedError( "Sky subtraction in preprocessing isn't implemented." )

    def run( self, *args, **kwargs ):
        """Run preprocessing for a given exposure and section_identifier.

        Parameters are passed to the data_store constructor (see
        DataStore.parse_args).  For preprocessing, an exposure and a
        sensorsection is required, so args must be one of:
          - DataStore (which has an exposure and a section)
          - exposure_id, section_identifier
          - Exposure, section_identifier
        Passing just an image won't work.

        kwargs can also include things that override the preprocessing
        behavior.  (TODO: document this)

        Returns
        -------
        DataStore
          contains the products of the processing.

        """
        self.has_recalculated = False
        try:  # first make sure we get back a datastore, even an empty one
            ds, session = DataStore.from_args( *args, **kwargs )
        except Exception as e:
            return DataStore.catch_failure_to_parse(e, *args)

        # This is here just for testing purposes
        self._ds = ds  # TODO: is there a reason not to just use the output datastore?

        try:  # catch any exceptions and save them in the datastore
            t_start = time.perf_counter()
            if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            if ( ds.exposure is None ) or ( ds.section_id is None ):
                raise RuntimeError( "Preprocessing requires an exposure and a sensor section" )

            self.pars.do_warning_exception_hangup_injection_here()

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
            SCLogger.debug("preprocessing: getting calibrator files")
            preprocparam = self.instrument.preprocessing_calibrator_files( self._calibset,
                                                                           self._flattype,
                                                                           ds.section_id,
                                                                           ds.exposure.filter_short,
                                                                           ds.exposure.mjd,
                                                                           session=session )

            SCLogger.debug("preprocessing: got calibrator files")

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

            required_bitflag = 0
            for step in self._stepstodo:
                required_bitflag |= string_to_bitflag( step, image_preprocessing_inverse )

            if image._data is None:  # in case we are skipping all preprocessing steps
                image.data = image.raw_data

            if image.preproc_bitflag != required_bitflag:
                self.has_recalculated = True
                # Overscan is always first (as it reshapes the image)
                if 'overscan' in self._stepstodo:
                    SCLogger.debug('preprocessing: overscan and trim')
                    image.data = self.instrument.overscan_and_trim( image )
                    # Update the header ra/dec calculations now that we know the real width/height
                    image.set_corners_from_header_wcs(setradec=True)
                    image.preproc_bitflag |= string_to_bitflag( 'overscan', image_preprocessing_inverse )

                # Apply steps in the order expected by the instrument
                for step in self._stepstodo:
                    if step == 'overscan':
                        continue
                    SCLogger.debug(f"preprocessing: {step}")
                    stepfileid = None
                    # Acquire the calibration file
                    if f'{step}_fileid' in kwargs:
                        stepfileid = kwargs[ f'{step}_fileid' ]
                    elif f'{step}_fileid' in preprocparam:
                        stepfileid = preprocparam[ f'{step}_fileid' ]
                    else:
                        raise RuntimeError( f"Can't find calibration file for preprocessing step {step}" )

                    if stepfileid is None:
                        SCLogger.info(f"Skipping step {step} for filter {ds.exposure.filter_short} "
                                         f"because there is no calibration file (this may be normal)")
                        # should we also mark it as having "done" this step? otherwise it will not know it's done
                        image.preproc_bitflag |= string_to_bitflag( step, image_preprocessing_inverse )
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
                        SCLogger.info( "Fringe correction not implemented" )

                    elif step == 'linearity':
                        # Linearity is instrument-specific
                        self.instrument.linearity_correct( image, linearitydata=calibfile )

                    else:
                        # TODO: Replace this with a call into an instrument method?
                        # In that case, the logic above about acquiring step files
                        # will need to be updated.
                        raise ValueError( f"Unknown preprocessing step {step}" )

                    image.preproc_bitflag |= string_to_bitflag( step, image_preprocessing_inverse )

                # Get the Instrument standard bad pixel mask for this image
            if image._flags is None or image._weight is None:
                image._flags = self.instrument.get_standard_flags_image( ds.section_id )

                # Estimate the background rms with sep
                boxsize = self.instrument.background_box_size
                filtsize = self.instrument.background_filt_size
                SCLogger.debug( "Subtracting sky and estimating sky RMS" )
                # Dysfunctionality alert: sep requires a *float* image for the mask
                # IEEE 32-bit floats have 23 bits in the mantissa, so they should
                # be able to precisely represent a 16-bit integer mask image
                # In any event, sep.Background uses >0 as "bad"
                fmask = np.array( image._flags, dtype=np.float32 )
                backgrounder = sep.Background( image.data, mask=fmask,
                                               bw=boxsize, bh=boxsize, fw=filtsize, fh=filtsize )
                fmask = None
                rms = backgrounder.rms()
                sky = backgrounder.back()
                subim = image.data - sky
                SCLogger.debug( "Building weight image and augmenting flags image" )

                wbad = np.where( rms <= 0 )
                wgood = np.where( rms > 0 )
                rms = rms ** 2
                subim[ subim < 0 ] = 0
                gain = self.instrument.average_gain( image )
                gain = gain if gain is not None else 1.
                # Shot noise from image above background
                rms += subim / gain
                image._weight = np.zeros( image.data.shape, dtype=np.float32 )
                image._weight[ wgood ] = 1. / rms[ wgood ]
                image._flags[ wbad ] |= string_to_bitflag( "zero weight", flag_image_bits_inverse )
                # Now make the weight zero on the bad pixels too
                image._weight[ image._flags != 0 ] = 0.
                # Figure out saturated pixels
                satlevel = self.instrument.average_saturation_limit( image )
                if satlevel is not None:
                    wsat = image.data >= satlevel
                    image._flags[ wsat ] |= string_to_bitflag( "saturated", flag_image_bits_inverse )
                    image._weight[ wsat ] = 0.

            if image.provenance is None:
                image.provenance = prov
            else:
                if image.provenance.id != prov.id:
                    # Logically, this should never happen
                    raise ValueError('Provenance mismatch for image and provenance!')

            image.filepath = image.invent_filepath()
            SCLogger.debug( f"Done with {pathlib.Path(image.filepath).name}" )

            if image._upstream_bitflag is None:
                image._upstream_bitflag = 0
            image._upstream_bitflag |= ds.exposure.bitflag

            ds.image = image

            ds.runtimes['preprocessing'] = time.perf_counter() - t_start
            if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                import tracemalloc
                ds.memory_usages['preprocessing'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:
            return ds
