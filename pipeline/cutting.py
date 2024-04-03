
from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.utils import parse_session

from models.source_list import SourceList
from models.cutouts import Cutouts

from improc.tools import make_cutouts


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

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

    def run(self, *args, **kwargs):
        """
        Go over a list of sources and for each source position,
        cut out a postage stamp image from the new,
        reference and subtraction images.
        The results are saved in a Cutouts object for each source.

        Returns a DataStore object with the products of the processing.
        """
        self.has_recalculated = False
        if isinstance(args[0], SourceList) and args[0].is_sub:  # most likely to get a SourceList detections object
            args, kwargs, session = parse_session(*args, **kwargs)
            ds = DataStore()
            ds.detections = args[0]
            ds.sub_image = args[0].image
            ds.image = args[0].image.new_image
        else:
            ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find some measurements in memory or in the database:
        cutout_list = ds.get_cutouts(prov, session=session)

        if cutout_list is None or len(cutout_list) == 0:  # must create a new list of Cutouts
            self.has_recalculated = True
            # use the latest source list in the data store,
            # or load using the provenance given in the
            # data store's upstream_provs, or just use
            # the most recent provenance for "detection"
            detections = ds.get_detections(session=session)

            if detections is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')

            cutout_list = []
            x = detections.x
            y = detections.y
            sz = self.pars.cutout_size
            sub_stamps_data = make_cutouts(ds.sub_image.data, x, y, sz)
            sub_stamps_weight = make_cutouts(ds.sub_image.weight, x, y, sz)
            sub_stamps_flags = make_cutouts(ds.sub_image.flags, x, y, sz)
            ref_stamps_data = make_cutouts(ds.sub_image.ref_aligned_image.data, x, y, sz)
            ref_stamps_weight = make_cutouts(ds.sub_image.ref_aligned_image.weight, x, y, sz)
            ref_stamps_flags = make_cutouts(ds.sub_image.ref_aligned_image.flags, x, y, sz)
            new_stamps_data = make_cutouts(ds.sub_image.new_aligned_image.data, x, y, sz)
            new_stamps_weight = make_cutouts(ds.sub_image.new_aligned_image.weight, x, y, sz)
            new_stamps_flags = make_cutouts(ds.sub_image.new_aligned_image.flags, x, y, sz)

            for i, source in enumerate(detections.data):
                # get the cutouts
                cutout = Cutouts.from_detections(detections, i, provenance=prov)
                cutout.sub_data = sub_stamps_data[i]
                cutout.sub_weight = sub_stamps_weight[i]
                cutout.sub_flags = sub_stamps_flags[i]
                cutout.ref_data = ref_stamps_data[i]
                cutout.ref_weight = ref_stamps_weight[i]
                cutout.ref_flags = ref_stamps_flags[i]
                cutout.new_data = new_stamps_data[i]
                cutout.new_weight = new_stamps_weight[i]
                cutout.new_flags = new_stamps_flags[i]
                cutout_list.append(cutout)

        # add the resulting list to the data store
        for cutout in cutout_list:
            if cutout.provenance is None:
                cutout.provenance = prov
            else:
                if cutout.provenance.id != prov.id:
                    raise ValueError(
                        f'Provenance mismatch for cutout {cutout.provenance.id[:6]} '
                        f'and preset provenance {prov.id[:6]}!'
                    )

        ds.cutouts = cutout_list

        # make sure this is returned to be used in the next step
        return ds

