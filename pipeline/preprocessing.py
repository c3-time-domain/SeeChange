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

from util.logger import SCLogger
from util.util import env_as_bool


class ParsPreprocessor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.add_par( 'steps_required', [], list, "Steps that need to be done to each exposure" )

        self.add_par( 'preprocessing',
                      'internal',
                      str,
                      "Where was the preprocessing done?  'internal' means using SeeChange preprocessor starting "
                      "from raw images.  'noirlab_instcal' means image was loaded alredy preprocessed by the NOIRLab "
                      "pipeline.  Set this parameter manually to something other than 'internal' when loading "
                      "in already-preprocessed images.",
                      critical=True )

        self.add_par( 'calibset', 'externally_supplied', str,
                      "The calibrator set to use.  Choose one of the CalibratorSetConverter enum. ",
                      critical=True )
        self.add_alias( 'calibrator_set', 'calibset' )

        self.add_par( 'flattype', 'externally_supplied', str,
                      "One of the FlatTypeConverter enum. ",
                      critical=True )

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

    def run( self, *args, **kwargs ):
        """Run preprocessing for a given exposure and section_identifier.

        Parameters are passed to the data_store constructor (see
        DataStore.parse_args).  For preprocessing, an exposure and a
        sensorsection is required, so args must be one of:
          - DataStore (which has an exposure and a section)
          - exposure_id, section_identifier
          - Exposure, section_identifier
        Passing just an image won't work.

        Returns
        -------
        DataStore
          contains the products of the processing.

        """
        self.has_recalculated = False

        try:
            ds, session = DataStore.from_args( *args, **kwargs )
            t_start = time.perf_counter()
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            if ( ds.exposure is None ) or ( ds.section_id is None ):
                raise RuntimeError( "Preprocessing requires an exposure and a sensor section" )

            self.pars.do_warning_exception_hangup_injection_here()

            if ( self.instrument is None ) or ( self.instrument.name != ds.exposure.instrument ):
                self.instrument = ds.exposure.instrument_object

            # check that all required steps can be done (or have been done) by the instrument:
            known_steps = self.instrument.preprocessing_steps_available
            known_steps += self.instrument.preprocessing_steps_done
            known_steps = set(known_steps)
            needed_steps = set(self.pars.steps_required)
            if not needed_steps.issubset(known_steps):
                raise ValueError(
                    f'Missing some preprocessing steps {needed_steps - known_steps} '
                    f'for instrument {self.instrument.name}'
                )

            # Get the calibrator files
            SCLogger.debug("preprocessing: getting calibrator files")
            preprocparam = self.instrument.preprocessing_calibrator_files( self.pars.calibset,
                                                                           self.pars.flattype,
                                                                           ds.section_id,
                                                                           ds.exposure.filter_short,
                                                                           ds.exposure.mjd,
                                                                           session=session )

            SCLogger.debug("preprocessing: got calibrator files")

            # get the provenance for this step, using the current parameters:
            prov = ds.get_provenance('preprocessing', self.pars.get_critical_pars(), session=session)

            # check if the image already exists in memory or in the database:
            image = ds.get_image(prov, session=session)

            if image is None:  # need to make new image
                # get the single-chip image from the exposure
                image = Image.from_exposure( ds.exposure, ds.section_id )

            if image is None:
                raise ValueError('Image cannot be None at this point!')

            if image.preproc_bitflag is None:
                image.preproc_bitflag = 0

            # Figure out what steps are generally needed.
            # Needed steps started above at set(self.pars.steps_required).
            # Subtract off what is *always* done by this instrument.
            needed_steps -= set(self.instrument.preprocessing_steps_done)
            filter_skips = self.instrument.preprocessing_step_skip_by_filter.get(ds.exposure.filter_short, [])
            if not isinstance(filter_skips, list):
                raise ValueError(f'Filter skips parameter for {ds.exposure.filter_short} must be a list')
            filter_skips = set(filter_skips)
            needed_steps -= filter_skips

            # in case we skip all preprocessing steps
            if image._data is None:
                image.data = image.raw_data

            # The image keeps track of the steps already done to it in
            #   image.preproc_bitflag, which is translated into a string
            #   of comma-separated keywords in image.preprocessing_done.
            #   This includes the things that already were applied in
            #   the exposure, and should have been set when the image
            #   was extracted from the exposure, but does not include
            #   the things that were skipped for this filter (i.e., the
            #   instrument's preprocessing_step_skip_by_filter).
            already_done = set( image.preprocessing_done.split(', ') if image.preprocessing_done else [] )

            # If self.pars.preprocessing is anything other than
            #   'internal', there should be nothing left to do except
            #   set the image provenance, and maybe calculate the weight
            #   and flags.  Verify that.
            if self.pars.preprocessing != 'internal':
                if len( needed_steps - already_done ) > 0:
                    raise ValueError( f"Preprocessing error: self.pars.preprocessing is {self.pars.preprocessing}, "
                                      f"but we still need to do steps {needed_steps-already_done} "
                                      f"(needed_steps={needed_steps}, preproc_bitflag={image.preproc_bitflag})" )

            if not needed_steps.issubset(already_done):  # still things to do here
                self.has_recalculated = True
                # Overscan is always first (as it reshapes the image)
                if 'overscan' in needed_steps:
                    SCLogger.debug('preprocessing: overscan and trim')
                    image.data = self.instrument.overscan_and_trim( image )
                    # Update the header ra/dec calculations now that we know the real width/height
                    image.set_corners_from_header_wcs(setradec=True)
                    image.preproc_bitflag |= string_to_bitflag( 'overscan', image_preprocessing_inverse )

                # Apply steps in the order expected by the instrument
                for step in self.pars.steps_required:
                    if step not in needed_steps:
                        continue
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
                        calibfile = self.stepfiles[ stepfileid ]
                    else:

                        with SmartSession( session ) as session:
                            if step in [ 'zero', 'dark', 'flat', 'illumination', 'fringe' ]:
                                calibfile = session.get( Image, stepfileid )
                                if calibfile is None:
                                    raise RuntimeError( f"Unable to load image id {stepfileid} "
                                                        f"for preproc step {step}" )
                            elif step == 'linearity':
                                calibfile = session.get( DataFile, stepfileid )
                                if calibfile is None:
                                    raise RuntimeError( f"Unable to load datafile id {stepfileid} "
                                                        f"for preproc step {step}" )
                            else:
                                raise ValueError( f"Preprocessing step {step} has an unknown "
                                                  f"file type (image vs. datafile)" )
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

            # Build the weight and flags images (if necessary)
            if image._flags is None or image._weight is None:
                # Start with the Instrument standard bad pixel mask for this image
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
                del fmask
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

            if image.provenance_id is None:
                image.provenance_id = prov.id
            else:
                if image.provenance_id != prov.id:
                    # Logically, this should never happen
                    raise ValueError('Provenance mismatch for image and provenance!')

            image.filepath = image.invent_filepath()
            SCLogger.debug( f"Done with {pathlib.Path(image.filepath).name}" )

            if image._upstream_bitflag is None:
                image._upstream_bitflag = 0
            image._upstream_bitflag |= ds.exposure.bitflag

            ds.image = image

            ds.runtimes['preprocessing'] = time.perf_counter() - t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['preprocessing'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            return ds

        except Exception as e:
            SCLogger.exception( f"Exception in Preprocessor.run: {e}" )
            ds.exceptions.append( e )
            raise
