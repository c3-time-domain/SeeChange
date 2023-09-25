import numpy as np
import sqlalchemy as sa

from models.base import SmartSession
from models.exposure import Exposure, ExposureImageIterator
from models.image import Image
from models.instrument import get_instrument_instance, Instrument, SensorSection

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore


class ParsPreprocessor(Parameters):
    def __init__(self, instrument, **kwargs):
        super().__init__()

        self.add_par( 'from_exposure', None, int, 'ID of exposure this image came from', critical=True )
        for calib in instrument.preprocesing_files:
            self.add_par( f'{calib}_set', None, str, 'Calibrator set for {calib}', critical=True )
            self.add_par( f'{calib}_isimage', False, bool, 'Is the calibrator an Image?' )
            self.add_par( f'{calib}_fileid', None, int, 'image_id or datafile_id for the calibrator file' )

        # A note about provenance.
        #
        # Ideally, we want the flats/biases/etc. that go into image
        # preprocessing be part of the provenance.  However, we also
        # would like for all of the chips on one instrument to have the
        # same provenance (when appropriate).  Hence, the defintion of
        # "flat_set", "dark_set", etc.  We can tag things in the
        # CalibratorFiles models in the database with a calbrator_set,
        # and *that* is what will go into the provenance.  It's then up
        # to the users to make sure that the calibrator_set field is
        # used properly.

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'preprocessing'


class Preprocessor:
    def __init__(self, instrument, section, exposure, **kwargs):
        """Create a preprocessor for a given exposure / section.

        Preprocessing is instrument-defined, but usually includes a subset of:
          * overscan subtraction
          * bias (zero) subtraction
          * dark current subtraction
          * linearity correction
          * flatfielding
          * fringe correction
          * illumination correction

        Parameters
        ----------
        instrument: Instrument or str
          The instrument we're working on
        section: SensorSectin or str
          The SensorSection of the image
        exposure: Exposure
          The exposure we're working on.

        """

        if not isinstance( instrument, Instrument ):
            instrument =  get_instrument_instance( instrument )
        self.instrument = instrument

        if not isinstance( section, SensorSection ):
            section = instrument.get_section( section )
        self.section = section

        self.exposure = exposure

        # Get the preprocessing parameters (i.e. flats, biases, etc.) for this instrument / mjd
        preprocparam = self.instrument.preprocessing_calibrator_params( exposure.mjd, section.identifier )

        # Update with anything passed
        preprocparam.update( **kwargs )

        self.pars = ParsPreprocessor( preprocparam )

    def run( self ):
        """Run preprocessing.

        Returns a DataStore object with the products of the processing.
        """
        # TODO: implement the actual code to do this.
        #  Save the dark/flat to attributes on "self"
        #  Apply the dark/flat/sky subtraction
        #  If not, create an Image for the specific CCD and add it to the cache and database.
        ds, session = DataStore.from_args( self.exposure, self.section.id )

        # get the provenance for this step, using the current parameters:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # check if the image already exists in memory or in the database:
        image = ds.get_image(prov, session=session)

        if image is None:  # need to make new image
            # get the CCD image from the exposure
            image = Image.from_exposure( self.exposure, ds.section_id )

        if image is None:
            raise ValueError('Image cannot be None at this point!')


        # Overscan
        image.data = self.instrument.overscan_and_trim_image( image.header, image.raw_data )

        # Apply subsequent steps in the order expected by the instrument
        for step in self.instrument.preprocessing_files:
            if getattr( self.pars, f'{step}_fileid' ) is not None:

                # Subtract zeros and darks ; divide flats and illumiation
                if step in [ 'zero', 'dark', 'flat', 'illumination' ]:
                    if not getattr( self.pars, f'{step}_isimage' ):
                        raise RuntimeError( f"Expected {step} file to be an image!" )
                    calim = ( session.query( Image )
                              .filter( Image.id==getattr( self.pars, f'{step}_fileid' ) )
                              .first() )
                    if step in [ 'zero', 'dark' ]:
                        image.data -= calim.data
                    else:
                        image.data /= calim.data

                # TODO FRINGE CORRECTION
                elif step == 'fringe':
                    _logger.warning( "Fringe correction not implemented" )

                # Linearity is instrument-specific
                elif step == 'linearity':
                    self.instrument.linearity_correct( image.data, self.section.id )

                else:
                    # TODO: Replace this with a call into an instrument method
                    raise ValueError( f"Unknown preprocessing step {step}" )



        if image.provenance is None:
            image.provenance = prov
        else:
            if image.provenance.id != prov.id:
                raise ValueError('Provenance mismatch for image and provenance!')

        ds.image = image

        # make sure this is returned to be used in the next step
        return ds
