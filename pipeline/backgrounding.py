import time

import numpy as np

from astropy.io import fits
import sep

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.background import Background
from improc.sextrsky import sextrsky
from improc.sextractor import run_sextractor

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
            'Method to use to estimate the background. Choose: "iter_sextr", "sextr", "sep" or "zero". ',
            critical=True
        )

        self.poly_order = self.add_par(
            'poly_order',
            1,
            int,
            'Order of the polynomial to fit to the background. ',
            critical=True
        )

        self.box_size = self.add_par(
            'box_size',
            None,
            ( int, None ),
            ( "Size of the box in pixels to use for the background estimation using sep and sextractor-based methods; "
              "None = use instrument's background_box_size" ),
            critical=True
        )

        self.filt_size = self.add_par(
            'filt_size',
            None,
            ( int, None ),
            ( "Size of the filter to use for the background estimation using sep and sextractor-based methods; "
              "None = use instruments' background_filt_size" ),
            critical=True
        )

        self.iter_sextr_iterations = self.add_par(
            'iter_sextr_iterations',
            5,
            int,
            'Number of times to iterate object finding / object-rejected sky measurement for iter_sextr',
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


    def iterative_sextractor( self, image, boxsize, filtsize ):
        """Iterate running sextractor to find objects and running sextrsky to measure sky.

        Each iteration (except for the first) uses the previous
        iteration's sky map as the sky background.  Each iteration uses
        the sextractor segmentation map as a mask to sextrsky.

        """
        sky = 0
        for iter in range( self.pars.iter_sextr_iterations ):
            # We don't know the seeing size when we're doing this, but since we
            #   don't really care about star/galaxy separation anyway, I don't
            #   think it matters
            # Using a HUGE mem_pixstack here because this bg method is probably only
            #   to be used in crowded fields, and we need a big mem_pixstack
            #   to be able to find all the stars in such a field.
            if iter == 0:
                sextr_res = run_sextractor( image.header, image.data, image.weight, maskdata=image.flags,
                                            back_type='AUTO', back_size=boxsize, back_filtersize=filtsize,
                                            writeseg=True, timeout=300, mem_pixstack=10000000 )
            else:
                sextr_res = run_sextractor( image.header, image.data - sky, image.weight, maskdata=image.flags,
                                            back_type='MANUAL', back_value=0.,
                                            writeseg=True, timeout=300, mem_pixstack=10000000 )
            with fits.open( sextr_res['segmentation'] ) as ifp:
                objmask = ifp[0].data

            sky, skysig = sextrsky( image.data, maskdata=objmask, boxsize=boxsize, filtsize=filtsize )

        return sky, skysig


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

                if self.pars.method in ( 'sep', 'sextr', 'iter_sextr' ):
                    boxsize = self.pars.box_size
                    if boxsize is None:
                        boxsize = image.instrument_object.background_box_size
                    filtsize = self.pars.filt_size
                    if filtsize is None:
                        filtsize = image.instrument_object.background_filt_size

                    if self.pars.method == 'sep':
                        SCLogger.debug("Backgrounder estimating sky level and RMS with sep")
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

                    elif self.pars.method in ( 'sextr', 'iter_sextr' ):
                        if self.pars.method == 'sextr':
                            SCLogger.debug( "Backgrounder estimating sky level and RMS with sextrsky" )
                            sky, skysig = sextrsky( image.data, image.flags, boxsize=boxsize, filtsize=filtsize )
                        else:
                            SCLogger.debug( "Backgrounder estimating sky level and RSM with iterative "
                                            "sextrsky and sextractor to mask objects" )
                            sky, skysig = self.iterative_sextractor( image, boxsize, filtsize )
                        bg = Background( value = float( np.nanmedian( sky ) ),
                                         noise = skysig,
                                         counts = sky,
                                         rms = np.full_like( sky, skysig, dtype=np.float32 ),
                                         format='map',
                                         method='sextr',
                                         image_shape=image.data.shape )

                    else:
                        raise RuntimeError( "This should never happen." )

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
