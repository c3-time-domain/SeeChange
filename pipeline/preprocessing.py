import numpy as np
import sqlalchemy as sa

from models.base import SmartSession
from models.exposure import Exposure
from models.image import Image

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore


class ParsPreprocessor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.use_sky_subtraction = self.add_par('use_sky_subtraction', True, bool, 'Apply sky subtraction. ')

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'preprocessing'


class Preprocessor:
    def __init__(self, **kwargs):
        self.pars = ParsPreprocessor(**kwargs)

    def run(self, *args, **kwargs):
        """
        Run dark and flat processing, and apply sky subtraction.
        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        # TODO: implement the actual code to do this.
        #  Save the dark/flat to attributes on "self"
        #  Apply the dark/flat/sky subtraction
        #  If not, create an Image for the specific CCD and add it to the cache and database.
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step, using the current parameters:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # check if the image already exists in memory or in the database:
        image = ds.get_image(prov, session=session)

        if image is None:  # need to make new image
            exposure = ds.get_raw_exposure(session=session)

            # get the CCD image from the exposure
            image = Image.from_exposure(exposure, ds.section_id)
            image.data = image.raw_data - np.median(image.raw_data)  # TODO: replace this!

        if image is None:
            raise ValueError('Image cannot be None at this point!')

            # TODO: apply dark/flat/sky subtraction
            #  right now this is just a placeholder:


        if image.provenance is None:
            image.provenance = prov
        else:
            if image.provenance.unique_hash != prov.unique_hash:
                raise ValueError('Provenance mismatch for image and provenance!')

        ds.image = image

        # make sure this is returned to be used in the next step
        return ds
