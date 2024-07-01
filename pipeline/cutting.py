import time

from improc.tools import make_cutouts

from models.source_list import SourceList
from models.cutouts import Cutouts

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from util.util import parse_session, env_as_bool


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
        try:  # first make sure we get back a datastore, even an empty one
            if isinstance(args[0], SourceList) and args[0].is_sub:  # most likely to get a SourceList detections object
                args, kwargs, session = parse_session(*args, **kwargs)
                ds = DataStore()
                ds.detections = args[0]
                ds.sub_image = args[0].image
                ds.image = args[0].image.new_image
            else:
                ds, session = DataStore.from_args(*args, **kwargs)
        except Exception as e:
            return DataStore.catch_failure_to_parse(e, *args)

        try:
            t_start = time.perf_counter()
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            prov = ds.get_provenance('cutting', self.pars.get_critical_pars(), session=session)

            detections = ds.get_detections(session=session)
            if detections is None:
                raise ValueError(
                    f'Cannot find a detections source list corresponding to the datastore inputs: {ds.get_inputs()}'
                )

            # try to find some cutouts in memory or in the database:
            cutouts = ds.get_cutouts(prov, session=session)

            if cutouts is not None:
                cutouts.load_all_co_data()

            if cutouts is None or len(cutouts.co_dict) == 0:
                self.has_recalculated = True
                # find detections in order to get the cutouts

                x = detections.x
                y = detections.y
                sz = self.pars.cutout_size
                sub_stamps_data = make_cutouts(ds.sub_image.data, x, y, sz)
                sub_stamps_weight = make_cutouts(ds.sub_image.weight, x, y, sz, fillvalue=0)
                sub_stamps_flags = make_cutouts(ds.sub_image.flags, x, y, sz, fillvalue=0)

                # TODO: figure out if we can actually use this flux (maybe renormalize it)
                # if ds.sub_image.psfflux is not None and ds.sub_image.psffluxerr is not None:
                #     sub_stamps_psfflux = make_cutouts(ds.sub_image.psfflux, x, y, sz, fillvalue=0)
                #     sub_stamps_psffluxerr = make_cutouts(ds.sub_image.psffluxerr, x, y, sz, fillvalue=0)
                # else:
                #     sub_stamps_psfflux = None
                #     sub_stamps_psffluxerr = None

                ref_stamps_data = make_cutouts(ds.sub_image.ref_aligned_image.data, x, y, sz)
                ref_stamps_weight = make_cutouts(ds.sub_image.ref_aligned_image.weight, x, y, sz, fillvalue=0)
                ref_stamps_flags = make_cutouts(ds.sub_image.ref_aligned_image.flags, x, y, sz, fillvalue=0)

                new_stamps_data = make_cutouts(ds.sub_image.new_aligned_image.data, x, y, sz)
                new_stamps_weight = make_cutouts(ds.sub_image.new_aligned_image.weight, x, y, sz, fillvalue=0)
                new_stamps_flags = make_cutouts(ds.sub_image.new_aligned_image.flags, x, y, sz, fillvalue=0)

                cutouts = Cutouts.from_detections(detections, provenance=prov)

                for i, source in enumerate(detections.data):
                    data_dict = {}
                    data_dict["sub_data"] = sub_stamps_data[i]
                    data_dict["sub_weight"] = sub_stamps_weight[i]
                    data_dict["sub_flags"] = sub_stamps_flags[i]
                    # TODO: figure out if we can actually use this flux (maybe renormalize it)
                    # if sub_stamps_psfflux is not None and sub_stamps_psffluxerr is not None:
                    #     data_dict['sub_psfflux'] = sub_stamps_psfflux[i]
                    #     data_dict['sub_psffluxerr'] = sub_stamps_psffluxerr[i]

                    data_dict["ref_data"] = ref_stamps_data[i]
                    data_dict["ref_weight"] = ref_stamps_weight[i]
                    data_dict["ref_flags"] = ref_stamps_flags[i]

                    data_dict["new_data"] = new_stamps_data[i]
                    data_dict["new_weight"] = new_stamps_weight[i]
                    data_dict["new_flags"] = new_stamps_flags[i]
                    cutouts.co_dict[f"source_index_{i}"] = data_dict

            # regardless of whether we loaded or calculated the cutouts, we need to update the bitflag
            cutouts._upstream_bitflag = 0
            cutouts._upstream_bitflag |= detections.bitflag

            # add the resulting Cutouts to the data store
            if cutouts.provenance is None:
                cutouts.provenance = prov
            else:
                if cutouts.provenance.id != prov.id:
                    raise ValueError(
                            f'Provenance mismatch for cutout {cutouts.provenance.id[:6]} '
                            f'and preset provenance {prov.id[:6]}!'
                        )

            ds.cutouts = cutouts

            ds.runtimes['cutting'] = time.perf_counter() - t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['cutting'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
            return ds
