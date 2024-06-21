import os
import time

import numpy as np

import sep

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.background import Background

from util.logger import SCLogger
from util.util import parse_bool


class ParsBackgrounder(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.format = self.add_par(
            'format',
            'map',
            str,
            'Format of the background image. Choose: "map", "scalar", or "polynomial". ',
            critical=True
        )

        self.method = self.add_par(
            'method',
            'sep',
            str,
            'Method to use to estimate the background. Choose: "sep" or "zero". ',
            critical=True
        )

        self.poly_order = self.add_par(
            'poly_order',
            1,
            int,
            'Order of the polynomial to fit to the background. ',
            critical=True
        )

        self.sep_box_size = self.add_par(
            'sep_box_size',
            128,
            int,
            'Size of the box in pixels to use for the background estimation using sep. ',
            critical=True
        )

        self.sep_filt_size = self.add_par(
            'sep_filt_size',
            3,
            int,
            'Size of the filter to use for the background estimation using sep. ',
            critical=True
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'backgrounding'


class Backgrounder:
    def __init__(self, **kwargs):
        self.pars = ParsBackgrounder(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

    def run(self, *args, **kwargs):
        """Calculate the background for the given image.

        Arguments are parsed by the DataStore.parse_args() method.
        Returns a DataStore object with the products of the processing.
        """
        self.has_recalculated = False

        try:  # first make sure we get back a datastore, even an empty one
            ds, session = DataStore.from_args(*args, **kwargs)
        except Exception as e:
            return DataStore.catch_failure_to_parse(e, *args)

        try:
            t_start = time.perf_counter()
            if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            prov = ds.get_provenance('extraction', self.pars.get_critical_pars(), session=session)

            # try to find the background object in memory or in the database:
            bg = ds.get_background(prov, session=session)

            if bg is None:  # need to produce a background object
                self.has_recalculated = True
                image = ds.get_image(session=session)

                if self.pars.method == 'sep':
                    # Estimate the background mean and RMS with sep
                    boxsize = self.pars.sep_box_size
                    filtsize = self.pars.sep_filt_size
                    SCLogger.debug("Subtracting sky and estimating sky RMS")
                    # Dysfunctionality alert: sep requires a *float* image for the mask
                    # IEEE 32-bit floats have 23 bits in the mantissa, so they should
                    # be able to precisely represent a 16-bit integer mask image
                    # In any event, sep.Background uses >0 as "bad"
                    fmask = np.array(image._flags, dtype=np.float32)
                    sep_bg_obj = sep.Background(image.data.copy(), mask=fmask,
                                                  bw=boxsize, bh=boxsize, fw=filtsize, fh=filtsize)
                    fmask = None
                    bg = Background(
                        value=float(np.nanmedian(sep_bg_obj.back())),
                        noise=float(np.nanmedian(sep_bg_obj.rms())),
                        counts=sep_bg_obj.back(),
                        rms=sep_bg_obj.rms(),
                        format='map',
                        method='sep'
                    )
                elif self.pars.method == 'zero':  # don't measure the b/g
                    bg = Background(value=0, noise=0, format='scalar', method='zero')
                else:
                    raise ValueError(f'Unknown background method "{self.pars.method}"')

                bg.image_id = image.id
                bg.image = image

            if bg.provenance is None:
                bg.provenance = prov
            else:
                if bg.provenance.id != prov.id:
                    raise ValueError('Provenance mismatch for background and extraction provenance!')

            # since these are "first look estimates" we don't update them if they are already set
            if ds.image.bkg_mean_estimate is None and ds.image.bkg_rms_estimate is None:
                ds.image.bkg_mean_estimate = float( bg.value )
                ds.image.bkg_rms_estimate = float( bg.noise )

            bg._upstream_bitflag = 0
            bg._upstream_bitflag |= ds.image.bitflag

            sources = ds.get_sources(session=session)
            if sources is not None:
                bg._upstream_bitflag |= sources.bitflag

            psf = ds.get_psf(session=session)
            if psf is not None:
                bg._upstream_bitflag |= psf.bitflag

            ds.bg = bg

            ds.runtimes['backgrounding'] = time.perf_counter() - t_start
            if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                import tracemalloc
                ds.memory_usages['backgrounding'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
            return ds
