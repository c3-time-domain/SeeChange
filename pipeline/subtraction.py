
import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.image import Image
from models.base import SmartSession


class ParsSubtractor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.algorithm = self.add_par(
            'algorithm',
            'hotpants',
            str,
            'Which algorithm to use. Possible values are: "hotpants", "zogy". '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'subtraction'


class Subtractor:
    def __init__(self, **kwargs):
        self.pars = ParsSubtractor()

        # TODO: add a reference cache here.

    def run(self, *args, **kwargs):
        """
        Get a reference image and subtract it from the new image.
        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)
        sub_image = ds.get_subtraction(prov, session=session)

        if sub_image is None:
            # use the latest image in the data store,
            # or load using the provenance given in the
            # data store's upstream_provs, or just use
            # the most recent provenance for "preprocessing"
            image = ds.get_image(session=session)
            if image is None:
                raise ValueError(f'Cannot find an image corresponding to the datastore inputs: {ds.get_inputs()}')

            # look for a reference that has to do with the current image
            ref = ds.get_reference_image(session=session)
            if ref is None:
                raise ValueError(
                    f'Cannot find a reference image corresponding to the datastore inputs: {ds.get_inputs()}'
                )

            sub_image = Image.from_ref_and_new(ref, image)
            sub_image.data = image.data - ref.data  # TODO: implement the subtraction algorithm here
            if sub_image.provenance is None:
                sub_image.provenance = prov
            else:
                if sub_image.provenance.unique_hash != prov.unique_hash:
                    raise ValueError('Provenance mismatch for sub_image and provenance!')

        ds.sub_image = sub_image

        # make sure this is returned to be used in the next step
        return ds
