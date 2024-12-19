import time

from improc.tools import make_cutouts

from models.image import Image  # noqa: F401
from models.source_list import SourceList
from models.cutouts import Cutouts

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from util.util import env_as_bool
from util.logger import SCLogger


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
        """Create a Cutouts objects for a list of sources.

        Go over a list of sources and for each source position, cut out
        a postage stamp image from the new, reference and subtraction
        images.  Background subtract new and ref before extracting
        the cutout. The results are saved in a Cutouts object for each
        source.

        Returns a DataStore object with the products of the processing.

        """
        self.has_recalculated = False

        try:
            # if isinstance(args[0], SourceList) and args[0].is_sub:  # most likely gets a SourceList detections object
            if isinstance( args[0], SourceList ):
                raise RuntimeError( "Need to update the code for creating a Cutter from a detections list" )
                # args, kwargs, session = parse_session(*args, **kwargs)
                # ds = DataStore()
                # ds.detections = args[0]
                # ds.sub_image = args[0].image
                # ds.image = args[0].image.new_image
            else:
                ds, session = DataStore.from_args(*args, **kwargs)

            t_start = time.perf_counter()
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            prov = ds.get_provenance('cutting', self.pars.get_critical_pars(), session=session)

            detections = ds.get_detections(session=session)
            if detections is None:
                raise ValueError( f'Cannot find a detections source list corresponding to '
                                  f'the datastore inputs: {ds.inputs_str}' )

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
                sub_stamps_data = make_cutouts(ds.sub_image.data, x, y, sz, dtype='>f4')
                sub_stamps_weight = make_cutouts(ds.sub_image.weight, x, y, sz, fillvalue=0, dtype='>f4')
                sub_stamps_flags = make_cutouts(ds.sub_image.flags, x, y, sz, fillvalue=0, dtype='>u2')

                # TODO: figure out if we can actually use this flux (maybe renormalize it)
                # if ds.sub_image.psfflux is not None and ds.sub_image.psffluxerr is not None:
                #     sub_stamps_psfflux = make_cutouts(ds.sub_image.psfflux, x, y, sz, fillvalue=0, dtype='>f4')
                #     sub_stamps_psffluxerr = make_cutouts(ds.sub_image.psffluxerr, x, y, sz, fillvalue=0, dtype='>f4')
                # else:
                #     sub_stamps_psfflux = None
                #     sub_stamps_psffluxerr = None

                # For both ref and new, we want to do cutouts on sky-subtracted images.
                # (The ref is almost certainly already over a zero sky, so for the ref this
                # should approximately be a null operation.)

                rdata = ds.aligned_ref_bg.subtract_me( ds.aligned_ref_image.data )
                ref_stamps_data = make_cutouts(rdata, x, y, sz, dtype='>f4')
                ref_stamps_weight = make_cutouts(ds.aligned_ref_image.weight, x, y, sz, fillvalue=0, dtype='>f4')
                ref_stamps_flags = make_cutouts(ds.aligned_ref_image.flags, x, y, sz, fillvalue=0, dtype='>u2')
                del rdata

                # Rescale the reference cutouts to have the same zeropoint as the
                #   new cutouts:
                ref_stamps_data *= 10 ** ( ( ds.aligned_ref_zp.zp - ds.aligned_new_zp.zp ) / -2.5 )
                ref_stamps_weight *= 10 ** ( ( ds.aligned_ref_zp.zp - ds.aligned_new_zp.zp ) / 5. )

                ndata = ds.aligned_new_bg.subtract_me( ds.aligned_new_image.data )
                new_stamps_data = make_cutouts(ndata, x, y, sz, dtype='>f4')
                new_stamps_weight = make_cutouts(ds.aligned_new_image.weight, x, y, sz, fillvalue=0, dtype='>f4')
                new_stamps_flags = make_cutouts(ds.aligned_new_image.flags, x, y, sz, fillvalue=0, dtype='>u2')
                del ndata

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
                    data_dict["new_x"] = x[i]
                    data_dict["new_y"] = y[i]

                    cutouts.co_dict[f"source_index_{i}"] = data_dict

            # regardless of whether we loaded or calculated the cutouts, we need to update the bitflag
            cutouts._upstream_bitflag = 0
            cutouts._upstream_bitflag |= detections.bitflag

            # add the resulting Cutouts to the data store
            if cutouts.provenance_id is None:
                cutouts.provenance_id = prov.id
            else:
                if cutouts.provenance_id != prov.id:
                    raise ValueError(
                            f'Provenance mismatch for cutout {cutouts.provenance.id[:6]} '
                            f'and preset provenance {prov.id[:6]}!'
                        )

            ds.cutouts = cutouts

            ds.runtimes['cutting'] = time.perf_counter() - t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['cutting'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            return ds

        except Exception as e:
            SCLogger.exception( f"Exception in Cutter.run: {e}" )
            ds.exceptions.append( e )
            raise
