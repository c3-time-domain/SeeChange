
import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.base import SmartSession
from models.cutouts import Cutouts


class ParsCutter(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.cutout_size = self.add_par(
            'cutout_size',
            20,
            int,
            'Size of the cutout in pixels. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'cutting'


class Cutter:
    def __init__(self, **kwargs):
        self.pars = ParsCutter(**kwargs)

    def run(self, *args, **kwargs):
        """
        Go over a list of sources and for each source position,
        cut out a postage stamp image from the new,
        reference and subtraction images.
        The results are saved in a Cutouts object for each source.

        Returns a DataStore object with the products of the processing.
        """
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find some measurements in memory or in the database:
        cutout_list = ds.get_cutouts(prov, session=session)

        if cutout_list is None:  # must create a new list of Cutouts

            # use the latest source list in the data store,
            # or load using the provenance given in the
            # data store's upstream_provs, or just use
            # the most recent provenance for "detection"
            detections = ds.get_detections(session=session)

            if detections is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')

            # TODO: implement the actual code to do this.
            #  For each source in the SourceList make a Cutouts object.
            #  For each Cutouts calculate the photometry (flux, centroids).
            #  Apply analytic cuts to each stamp image, to rule out artefacts.
            #  Apply deep learning (real/bogus) to each stamp image, to rule out artefacts.
            #  Save the results as Measurement objects, append them to the Cutouts objects.
            #  Commit the results to the database.

            # add the resulting list to the data store
            if cutout_list.provenance is None:
                cutout_list.provenance = prov
            else:
                if cutout_list.provenance.unique_hash != prov.unique_hash:
                    raise ValueError('Provenance mismatch for cutout_list and provenance!')

            ds.cutouts = cutout_list

        # make sure this is returned to be used in the next step
        return ds

