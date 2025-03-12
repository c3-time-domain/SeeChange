import time

import numpy as np

import sep_pjw as sep

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.background import Background

from util.logger import SCLogger


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
            None,
            ( int, None ),
            ( "Size of the box in pixels to use for the background estimation using sep; "
              "None = use instrument's background_box_size" ),
            critical=True
        )

        self.sep_filt_size = self.add_par(
            'sep_filt_size',
            None,
            ( int, None ),
            ( "Size of the filter to use for the background estimation using sep; "
              "None = use instruments' background_filt_size" ),
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

        Returns a Background object.

        """
        self.has_recalculated = False

        try:
            ds = DataStore.from_args(*args, **kwargs)
            t_start = time.perf_counter()
            if ds.update_memory_usages:
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # try to find the background object in memory or in the database:
            bg = ds.get_background()

            if bg is None:  # need to produce a background object
                self.has_recalculated = True
                image = ds.get_image()
                # Maybe todo in the future: if we ever actually implement a background algorithm
                #   that looks at sources for masking, check that sources exist too
                if image is None:
                    raise RuntimeError( "Backgrounding can't proceed unless the DataStore already has image" )

                if self.pars.method == 'sep':
                    # Estimate the background mean and RMS with sep
                    boxsize = self.pars.sep_box_size
                    if boxsize is None:
                        boxsize = image.instrument_object.background_box_size
                    filtsize = self.pars.sep_filt_size
                    if filtsize is None:
                        filtsize = image.instrument_object.background_filt_size
                    SCLogger.debug("Backgrounder estimating sky level and RMS")
                    # Dysfunctionality alert: sep requires a *float* image for the mask
                    # IEEE 32-bit floats have 23 bits in the mantissa, so they should
                    # be able to precisely represent a 16-bit integer mask image
                    # In any event, sep.Background uses >0 as "bad"
                    fmask = np.array(image._flags, dtype=np.float32)
                    # Further issue: sep requires native byte order.  image.data may
                    # well not be in native byteorder.
                    tmpimagedata = image.data.copy()
                    if not tmpimagedata.dtype.isnative:
                        tmpimagedata= tmpimagedata.byteswap().newbyteorder()
                    sep_bg_obj = sep.Background(tmpimagedata, mask=fmask,
                                                bw=boxsize, bh=boxsize, fw=filtsize, fh=filtsize)
                    del fmask
                    del tmpimagedata
                    bg = Background(
                        value=float(np.nanmedian(sep_bg_obj.back())),
                        noise=float(np.nanmedian(sep_bg_obj.rms())),
                        counts=sep_bg_obj.back(),
                        rms=sep_bg_obj.rms(),
                        format='map',
                        method='sep',
                        image_shape=image.data.shape
                    )
                elif self.pars.method == 'zero':  # don't measure the b/g
                    bg = Background(value=0, noise=0, format='scalar', method='zero', image_shape=image.data.shape)
                else:
                    raise ValueError(f'Unknown background method "{self.pars.method}"')

            # since these are "first look estimates" we don't update them if they are already set
            if ds.image.bkg_mean_estimate is None and ds.image.bkg_rms_estimate is None:
                ds.image.bkg_mean_estimate = float( bg.value )
                ds.image.bkg_rms_estimate = float( bg.noise )

            # Not updating the background upstream bitflags here; that will be done
            #   in detection.py, which is probably what this was called from anyway.

            if ds.update_runtimes:
                ds.runtimes['backgrounding'] = time.perf_counter() - t_start
            if ds.update_memory_usages:
                import tracemalloc
                ds.memory_usages['backgrounding'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            return bg

        except Exception as e:
            SCLogger.exception( f"Exception in Backgrounder.run: {e}" )
            ds.exceptions.append( e )
            raise
