import pathlib
import time
import random
import shutil
import subprocess

import numpy as np
from numpy.fft import fft2, ifft2, fftshift

import sep

from models.base import SmartSession, FileOnDiskMixin
from models.enums_and_bitflags import BitFlagConverter
from models.provenance import Provenance, CodeVersion
from models.image import Image

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator

from improc.bitmask_tools import dilate_bitflag
from improc.inpainting import Inpainter
from improc.alignment import ImageAligner
from improc.tools import sigma_clipping
import improc.tools

from util.config import Config
from util.fits import save_fits_image_file, read_fits_image
from util.exceptions import SubprocessFailure
from util.logger import SCLogger


class ParsCoadd(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.method = self.add_par(
            'method',
            'zogy',
            str,
            'Coaddition method.  Currently only "naive" and "zogy" are supported. ',
            critical=True
        )

        self.alignment = self.add_par(
            'alignment',
            {},
            dict,
            'Alignment parameters. ',
            critical=True
        )

        self.alignment_index = self.add_par(
            'alignment_index',
            'last',
            str,
            ( 'How to choose the index of image to align to.  Can be "first", "last", "other", or an integer; '
              '"other" is currently only supported by coadd method swarp' ),
            critical=True
        )

        self.inpainting = self.add_par(
            'inpainting',
            {},
            dict,
            'Inpainting parameters. ',
            critical=True
        )

        self.noise_estimator = self.add_par(
            'noise_estimator',
            'sep',
            str,
            'Method to estimate noise (sigma) in the image.  '
            'Use "sep" or "sigma" for sigma clipping. ',
            critical=True,
        )

        self.flag_fwhm_factor = self.add_par(
            'flag_fwhm_factor',
            1.0,
            float,
            ( 'Multiplicative factor for the PSF FWHM (in pixels) to use for dilating the flag maps. '
              '(Currently only used by zogy.)' ),
            critical=True,
        )

        self.cleanup_alignment = self.add_par(
            'cleanup_alignemnt',
            True,
            bool,
            ( 'Try to clean up aligned images from the Coadder object after running the coadd. '
              'This should save memory, but you might want to set this to False for testing purposes.' ),
            critical=False
        )

        self.swarp_timeout = self.add_par(
            'swarp_timeout',
            600,
            int,
            'Timeout for swarp in seconds, if method is swarp',
            critical=False
        )

        self._enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name(self):
        return 'coaddition'


class Coadder:
    """Use this class to coadd (stack) images together to make a deeper image.

    Each image should have a PSF and a ZeroPoint associated with it (and loaded!) when running coaddition.

    Images are expected to be aligned (use the Aligner) and should generally be of the same sky region.
    If not already aligned, they need to have the SourceList and WorldCoordinates loaded so that
    alignment can be done on the fly.

    Input images should also have a valid Provenance associated with them. Not that for any set of images
    that share a provenance hash, their respective downstreams (e.g., PSF) should also have a single provenance
    hash for each type of downstream.  This makes it possible to identify the upstream images' associated products
    based solely on the provenance's upstream hashes.

    Areas on the edges where the images are not overlapping (or bad pixels, etc.) are coadded but will
    contribute zero weight, so the total weight of that pixel will be zero (if all input images have a bad pixel)
    or they would have lower weight if only some images had bad pixels there.

    Remember that some coaddition methods use convolution with the PSF, so any effects of individual pixels
    could affect nearby pixels, depending on the size of the PSF.
    """

    def __init__( self, **kwargs ):
        self.pars = ParsCoadd(**kwargs)
        self.inpainter = Inpainter(**self.pars.inpainting)
        self.pars.inpainting = self.inpainter.pars.get_critical_pars()  # add Inpainter defaults into this dictionary
        self.aligner = ImageAligner(**self.pars.alignment)
        self.pars.alignment = self.aligner.pars.get_critical_pars()  # add ImageAligner defaults into this dictionary

    def _estimate_background(self, data):
        """Get the mean and noise RMS of the background of the given image.

        Parameters
        ----------
        data: ndarray
            The image for which background should be estimated.

        Returns
        -------
        bkg: float
            The mean background in the image.
        sigma: float
            The RMS of the background in the image.
        """
        if self.pars.noise_estimator == 'sep':
            b = sep.Background(data)
            bkg = b.globalback
            sigma = b.globalrms
        elif self.pars.noise_estimator.startswith('sigma'):
            bkg, sigma = sigma_clipping(data)
        else:
            raise ValueError(
                f'Unknown noise estimator: {self.pars.noise_estimator}.  Use "sep" or "sigma_clipping" or "bkg_rms". '
            )

        return bkg, sigma

    # ======================================================================

    def _coadd_naive(self, images, weights=None, flags=None):
        """Simply sum the values in each image on top of each other.

        Parameters
        ----------
        images: list of Image or list of 2D ndarrays
            Images that have been aligned to each other.
        weights: list of 2D ndarrays
            The weights to use for each image.
            If images is given as Image objects, can be left as None.
        flags: list of 2D ndarrays
            The bit flags to use for each image.
            If images is given as Image objects, can be left as None.

        Returns
        -------
        outim: ndarray
            The image data after coaddition.
        outwt: ndarray
            The weight image after coaddition.
        outfl: ndarray
            The bit flags array after coaddition.
        """
        if not all( isinstance(image, type(images[0]) ) for image in images):
            raise ValueError('Not all images are of the same type. ')
        if isinstance(images[0], Image):
            data = [image.data for image in images]
            weights = [image.weight for image in images]
            flags = [image.flags for image in images]
        elif isinstance(images[0], np.ndarray):
            data = images
        else:
            raise ValueError('images must be a list of Image objects or 2D arrays. ')

        imcube = np.array(data)
        outim = np.sum(imcube, axis=0)

        wtcube = np.array(weights)
        varflag = wtcube == 0
        wtcube2 = wtcube ** 2
        wtcube2[varflag] = np.nan
        varmap = 1 / wtcube2

        outwt = 1 / np.sqrt(np.sum(varmap, axis=0))

        outfl = np.zeros(outim.shape, dtype='uint16')
        for f in flags:
            outfl |= f

        return outim, outwt, outfl

    # ======================================================================

    def _zogy_core(self, datacube, psfcube, sigmas, flux_zps):
        """Perform the core Zackay & Ofek proper image coaddition on the input data cube.

        Parameters
        ----------
        datacube: ndarray
            The data cube to coadd. Can be images or weight maps (or
            anything else... SORT OF.  It will produce a sum if you give
            it something other than images, but the normalization will
            not be consistent if e.g. you send in images and variance
            maps; the summed variance will have been normalized
            incorrectly as compared to how the summed image was
            normalized.)  (...in fact, I think it doesn't even do quite
            the right coadd (up to a normalization factor) for something
            other than an image, unless all of the flux_zps are the same
            as each other.  So, don't use this for anything other than
            images.)  Must already be aligned.

        psfcube: ndarray
            The PSF cube to use for coaddition.  3D numpy array, each layer must
            have a normalized (i.e. sum=1) PSF image at the same pixel scale
            as the images in datacube.

        sigmas: ndarray
            The background noise estimate for each image in the data cube.
            Must be a 1D array with a length equal to the first axis of the data cube.
            It could have additional dimensions, but it will be reshaped to be multiplied
            with the data cube and psf cube.

        flux_zps: ndarray
            The flux zero points for each image in the data cube.
            (Images are divided by this value to turn them into flux
            values that have a zeropoint of 0, i.e. m_true =
            -2.5*log(data/flux_zp) = -2.5*log(data) +
            2.5*log(flux_zp), so 2.5*log(flux_zp) = the traditional
            magnitude zeropoint.)

            Must be a 1D array with a length equal to the first axis of the data cube.
            It could have additional dimensions, but it will be reshaped to be multiplied
            with the data cube and psf cube.

        Returns
        -------
        outdata: ndarray
            The coadded 2D data array.
        outpsf: ndarray
            The coadded 2D PSF cube.
        score: ndarray
            The matched-filter result of cross correlating outdata with outpsf.

        """
        # data verification:
        if datacube.shape != psfcube.shape:
            raise ValueError('The data cube and PSF cube must have the same shape. ')
        if len(datacube.shape) != 3:
            raise ValueError('The data cube and PSF cube must have 3 dimensions. ')

        sigmas = np.reshape(np.array(sigmas), (len(sigmas), 1, 1))
        if sigmas.size != datacube.shape[0]:
            raise ValueError('The sigmas array must have the same length as the first axis of the data cube. ')

        flux_zps = np.reshape(np.array(flux_zps), (len(flux_zps), 1, 1))
        if flux_zps.size != datacube.shape[0]:
            raise ValueError('The flux_zps array must have the same length as the first axis of the data cube. ')

        if np.sum(np.isnan(datacube)) > 0:
            raise ValueError('There are NaNs values in the data cube! Use inpainting to remove them... ')

        # calculations:
        datacube_f = fft2(datacube)
        psfcube_f = fft2(psfcube)

        # paper ref: https://ui.adsabs.harvard.edu/abs/2017ApJ...836..188Z/abstract
        score_f = np.sum(flux_zps / sigmas ** 2 * np.conj(psfcube_f) * datacube_f, axis=0)  # eq 7
        psf_f = np.sqrt(np.sum(flux_zps ** 2 / sigmas ** 2 * np.abs(psfcube_f) ** 2, axis=0))  # eq 10
        outdata_f = score_f / psf_f  # eq 8

        outdata = fftshift(ifft2(outdata_f).real)
        score = fftshift(ifft2(score_f).real)
        psf = fftshift(ifft2(psf_f).real)
        psf = psf / np.sum(psf)

        return outdata, psf, score

    def _coadd_zogy(
            self,
            images,
            bgs=None,
            impsfs=None,
            zps=None,
            weights=None,
            flags=None,
            psf_clips=None,
            psf_fwhms=None,
            flux_zps=None,
            bkg_means=None,
            bkg_sigmas=None,
    ):
        """Use Zackay & Ofek proper image coaddition to add the images together.

        This method uses the PSF of each image to coadd images with proper weight
        given to each frequency in Fourier space, such that it preserves information
        even when using images with different PSFs.

        There are two different calling semantics:

        (1) images is a list of Image objects

        In this case, you must also pass bgs, impsfs, and zps, but you
        do not pass weights, flags, psf_clips, psf_fwhms, flux_zps,
        bkg_means, or bkg_sigmas.

        (2) images is a list of 2D ndnarrays

        In this case, you do not pass bgs, impsfs or zps, but you must
        pass weights, flags, psfs_clips, psf_fwhms, flux_zps, bkg_means,
        and bkg_sigmas.

        TODO QUESTION : does this implicitly assume that all the images have a lot of
        overlap?  (It must, since it does inpainting.  What about images that don't
        have a lot of overlap?  That's a legitimate thing to want to coadd sometimes.)

        Parameters
        ----------
        images: list of Image or list of 2D ndarrays
            Images that have been aligned to each other.

        bgs: list of Background objects, or None

        impsfs: list of PSF objects, or None

        zps: list of ZeroPoint objects, or None

        weights: list of 2D ndarrays, or None
            The weights to use for each image.

        flags: list of 2D ndarrays, or None
            The bit flags to use for each image.

        psf_clips: list of 2D ndarrays or None
            The PSF images to use for each image.

        psf_fwhms: list of floats, or None
            The FWHM of the PSF for each image.

        flux_zps: list of floats, or None
            The flux zero points for each image.  Defined so that
            mag = -2.5*log(data/flux_zp)

        bkg_means: list of floats, or None
            The mean background for each image.
            If images are already background subtracted, set these to zeros.

        bkg_sigmas: list of floats
            The RMS of the background for each image.

        Returns
        -------
        outim: ndarray
            The image data after coaddition.
        outwt: ndarray
            The weight image after coaddition.
        outfl: ndarray
            The bit flags array after coaddition.
        psf: ndarray
            An array with the PSF of the output image.
        score: ndarray
            A matched-filtered score image of the coadded image.

        """
        if not all( isinstance( image, type(images[0]) ) for image in images):
            raise ValueError('Not all images are of the same type. ')

        if isinstance(images[0], Image):
            data = []
            flags = []
            weights = []
            psf_clips = []
            psf_fwhms = []
            flux_zps = []
            bkg_means = []
            bkg_sigmas = []

            for image, bg, psf, zp in zip( images, bgs, impsfs, zps ):
                data.append(image.data)
                flags.append(image.flags)
                weights.append(image.weight)
                psf_clips.append(psf.get_clip())
                psf_fwhms.append(psf.fwhm_pixels)
                flux_zps.append(10 ** (0.4 * zp.zp))
                bkg_means.append(bg.value)
                bkg_sigmas.append(bg.noise)

        elif isinstance(images[0], np.ndarray):
            data = images
        else:
            raise ValueError('images must be a list of Image objects or 2D arrays. ')

        # pad the PSFs to the same size as the image data
        psfs = []
        for array, psf in zip(data, psf_clips):
            padsize_x1 = int(np.ceil((array.shape[1] - psf.shape[1]) / 2))
            padsize_x2 = int(np.floor((array.shape[1] - psf.shape[1]) / 2))
            padsize_y1 = int(np.ceil((array.shape[0] - psf.shape[0]) / 2))
            padsize_y2 = int(np.floor((array.shape[0] - psf.shape[0]) / 2))
            psf_pad = np.pad(psf, ((padsize_y1, padsize_y2), (padsize_x1, padsize_x2)))
            psf_pad /= np.sum(psf_pad)
            psfs.append(psf_pad)

        # estimate the background if not given
        if bkg_means is None or bkg_sigmas is None:
            raise ValueError('Background must be given if images are not Image objects. ')

        # NOTE -- this will use a LOT of memory.
        # All of the numpy arrays will now be stored twice -- once in
        #   the list, and once in the datacube created.  Suggest
        #   refactoring so that the for loop above just builds the
        #   datacubes directly instead of going through the lists.
        imcube = np.array(data)
        flcube = np.array(flags)
        wtcube = np.array(weights)
        psfcube = np.array(psfs)
        bkg_means = np.reshape(np.array(bkg_means), (len(bkg_means), 1, 1))
        bkg_sigmas = np.reshape(np.array(bkg_sigmas), (len(bkg_sigmas), 1, 1))
        flux_zps = np.reshape(np.array(flux_zps), (len(flux_zps), 1, 1))

        # subtract the background
        # NOTE -- this isn't necessarily *really* subtracting the
        #   backgrounds.  It's subtracting the mean background value from
        #   every pixel.  (bkg_means above was built by looking at
        #   bg.value.  If there's a background image, to capture varying
        #   background, that is lost here.)
        imcube -= bkg_means

        # make sure to inpaint missing data
        # TODO: make sure images are scaled before inpainting, or add that in the inpainting code
        imcube = self.inpainter.run(imcube, flcube, wtcube)

        if np.any(np.isnan(imcube)):
            raise ValueError('There are still NaNs in the image data after inpainting!')

        # This is where the magic happens
        outim, psf, score = self._zogy_core(imcube, psfcube, bkg_sigmas, flux_zps)

        # coadd the variance as well
        #  ---> THIS DOESNT WORK RIGHT.
        # Reason: the coadd image is normalized in _zogy_core to its standard
        #   deviation.  The same normalization is not right for the variance image;
        #   the stdev image is what should be normalized the same way.
        # varflag = ( wtcube <= 0 ) | ( flcube != 0 )
        # wtcube[varflag] = np.nan
        # varmap = 1. / wtcube
        # varmap = self.inpainter.run(varmap, varflag, wtcube)  # wtcube doesn't do anything, maybe put something else?
        # outvarmap, _, _ = self._zogy_core(varmap, psfcube, bkg_sigmas, flux_zps)
        # outwt = 1. / np.abs(outvarmap)

        # Because the whole zogy algorithm implicitly assumes that it's entirely sky-noise
        # dominated, let's just flow with it and ignore poisson noise in objects.  That's
        # not great, but, well, we're kind of stuck right now.  Additionally, the zogy
        # algorithm normalizes the summed image so that its variance is 1, so we
        # can just set the weight to 1 everywhere, and that should be right for sky
        # noise.
        outwt = np.ones_like( outim, dtype=np.float32 )

        outfl = np.zeros(outim.shape, dtype='uint16')
        for f, p in zip(flags, psf_fwhms):
            splash_pixels = int(np.ceil(p * self.pars.flag_fwhm_factor))
            outfl = outfl | dilate_bitflag(f, iterations=splash_pixels)
        outwt[ outfl != 0 ] = 0.

        return outim, outwt, outfl, psf, score


    # ======================================================================

    def _coadd_swarp( self,
                      data_store_list,
                      alignment_target_datastore,
                      try_to_reduce_memory_usage=True,
                      leave_behind_temp_files=False ):
        """Perform alignment and coadding using a single call to swarp.

        Parameters
        ----------
          data_store_list: list of DataStore
             Data stores holding the images to be coadded.  Each must
             have products through zeropoint.

          alignment_target_datastore: DataStore
             Data store holding the image to which the images in
             data_store_list will be aligned before they are coadded.
             This may be (but does not have to be) one of the members of
             data_store_list.

          try_to_reduce_memory_usage: bool, default True
             Call each datastore's free() method after we're done with
             it.

          leave_behind_temp_files: bool, default False
             For testing purposes.

        """

        # For subtraction, or one-by-one alignment for coadd methods
        #   other than swarp, we scamp a new WCS for the target image
        #   using the source image's source list as a RA/Dec catalog;
        #   see the massive comment in
        #   alignment.py::ImageAligner._align_swarp for an explanation
        #   of the reason.
        #
        # Here, we do it differently.  We need to align a whole bunch of
        #   images to a single target all at the same time.  As such we
        #   will use the target's source list as the RA/Dec catalog, and
        #   make a new WCS for each source image based on that.  We'll
        #   then put the target image's WCS as the output, but since the
        #   temporary source image WCSes were made using the target
        #   image's sources as a catalog, the alignment should be better
        #   than if we'd gone Source->Gaia->Target.
        #
        # For this reason, "source" and "target" are backwards in the
        #   call to ImageAligner.get_swarp_fodder_wcs

        # FileOnDiskMixin.temp_path is a temp directory
        tmpdir = ( pathlib.Path( FileOnDiskMixin.temp_path )
                   / ( ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) ) ) )
        SCLogger.debug( f"_coadd_swarp working in directory {tmpdir}" )
        tmpdir.mkdir( exist_ok=True, parents=True )
        try:
            swarp_vmem_dir = tmpdir / 'vmem'
            swarp_vmem_dir.mkdir( exist_ok=True, parents=True )
            swarp_resample_dir = tmpdir / 'resample'
            swarp_resample_dir.mkdir( exist_ok=True, parents=True )

            targds = alignment_target_datastore
            sumimgs = []
            sumwts = []

            # Write out an output header file using the correct target wcs
            hdr = targds.image.header.copy()
            improc.tools.strip_wcs_keywords( hdr )
            hdr.update( targds.wcs.wcs.to_header() )
            hdr.tofile( tmpdir / 'coadd.head' )

            # Write out a bunch of temporary images that are the source images with an
            #   updated wcs in the header, and all scaled to the same zeropoint.
            for imgdex, ds in enumerate(data_store_list):
                # image_paths = ds.image.get_fullpath( as_list=True )
                # imdex = ds.image.components.index( 'image' )
                # wtdex = ds.image.components.index( 'weight' )
                # fldex = ds.image.components.index( 'flags' )

                # Get a wcs for the source image using target image's source list as the RA/Dec catalog
                dswcs = self.aligner.get_swarp_fodder_wcs( targds.image, targds.sources, targds.wcs, targds.zp,
                                                           ds.sources )

                # Need to write out a temporary image whose header has this new wcs
                hdr = ds.image.header.copy()
                improc.tools.strip_wcs_keywords( hdr )
                hdr.update( dswcs.to_header() )

                # Background subtract
                data = ds.bg.subtract_me( ds.image.data )

                # Scale to the target's zeropoint.  This is ultimately arbitrary,
                #  but we want all the images scaled the same way for the sum
                data *= 10 ** ( ( targds.zp.zp - ds.zp.zp ) / 2.5 )
                wtdata = ds.image.weight * 10 ** ( ( ds.zp.zp - targds.zp.zp ) / 1.25 )
                # Make sure weight is 0 for all bad pixels
                # (This is what swarp expects.)
                wtdata[ ds.image.flags != 0 ] = 0.

                tmpim = tmpdir / f'in{imgdex:03d}_image.fits'
                tmpwt = tmpdir / f'in{imgdex:03d}_weight.fits'
                save_fits_image_file( tmpim, data, hdr, extname=None, single_file=False )
                save_fits_image_file( tmpwt, wtdata, hdr, extname=None, single_file=False )
                sumimgs.append( tmpim )
                sumwts.append( tmpwt )

                if try_to_reduce_memory_usage:
                    data = None
                    hdr = None
                    ds.free()

            if try_to_reduce_memory_usage:
                targds.free()

            # Use swarp to coadd all the source images, aligning with the target image.
            #
            # We don't want to mess with gain multiplication, so make
            #   sure that swarp doesn't try to do anything funny by
            #   setting an unlikely keyword.  Because our weights
            #   already include noise from objects, not just from the
            #   sky background, we want gain=0 for swarp to do the
            #   right thing.  Likewise, saturated pixels should already
            #   be marked as such from our preprocessing (I really
            #   hope), so try to avoid letting swarp doing things there
            #   too.  Also make sure swarp doesn't try to do fscaling,
            #   since we already scaled the images to zeorpoints.
            #   (Never know what's in the header!)
            #
            # The swarp manual is incomplete.  I don't know what the
            #   CLIP_SIGMA et al. parameters do.  Are they only used
            #   with COMBINE_TYPE=CLIPPED?  We really want to do a
            #   weighted combination, but also we want to do some
            #   clipping to reject CRs and the like.
            #   ...Looking at the swarp source code, it looks like CLIPPED
            #   is doing a weighted mean of the things it doesn't
            #   throw out, so CLIPPED should be good to just use.
            #   https://github.com/astromatic/swarp/blob/3d8ddf1e61718a2ba402473990c6483862671806/src/coadd.c#L1418

            command = [ 'swarp' ]
            command.extend( sumimgs )
            command.extend( [ '-IMAGEOUT_NAME', str( tmpdir / 'coadd.fits' ),
                              '-WEIGHTOUT_NAME', str( tmpdir / 'coadd.weight.fits' ),
                              '-RESCALE_WEIGHTS', 'N',
                              '-SUBTRACT_BACK', 'N',
                              '-RESAMPLE_DIR', swarp_resample_dir,
                              '-VMEM_DIR', swarp_vmem_dir,
                              '-VMEM_MAX', '1024',
                              '-MEM_MAX', '1024',
                              '-WRITE_XML', 'N',
                              '-INTERPOLATE', 'Y',
                              '-FSCALE_KEYWORD', 'THIS_KEYWORD_WILL_NEVER_EXIST',
                              '-FSCALE_DEFAULT', '1.0',
                              '-GAIN_KEYWORD', 'THIS_KEYWORD_WILL_NEVER_EXIT',
                              '-GAIN_DEFAULT', '0.0',
                              '-SATLEV_KEYWORD', 'THIS_KEYWORD_WILL_NEVER_EXIST',
                              '-SATLEV_DEFAULT', '1e10',
                              '-COMBINE', 'Y',
                              '-COMBINE_TYPE', 'CLIPPED',
                              '-WEIGHT_TYPE', 'MAP_WEIGHT',
                              '-WEIGHT_IMAGE', ','.join([ str(s) for s in sumwts ])
                             ] )


            SCLogger.debug( f"Running swarp to coadd {len(sumimgs)} images; swarp command is {command}" )
            t0 = time.perf_counter()
            res = subprocess.run( command, capture_output=True, timeout=self.pars.swarp_timeout )
            t1 = time.perf_counter()
            SCLogger.debug( f"Swarp to sum {len(sumimgs)} images took {t1-t0:.2f} seconds" )
            SCLogger.debug( f"Swarp stdout:\n{res.stdout}" )
            SCLogger.debug( f"Swarp stderr:\n{res.stderr}" )
            if res.returncode != 0:
                raise SubprocessFailure( res )

            data, hdr = read_fits_image( tmpdir / 'coadd.fits', output='both' )
            weight = read_fits_image( tmpdir / 'coadd.weight.fits' )
            flags = np.zeros( weight.shape, dtype=np.int16 )
            # TODO : should probably use BitFlagConverter 'out of bounds' if we
            #   can figure out a way.  Look into swarp, see if it gives us this
            #   information somewhere.  (In reality, everywhere in the pipeline
            #   we're probably just using flags!=0 as "bad", so it doesn't matter
            #   that much.)
            flags[ weight<=0 ] = 2 ** BitFlagConverter.to_int( 'bad pixel' )

            return hdr, data, weight, flags

        finally:
            if not leave_behind_temp_files:
                if tmpdir.is_dir():
                    shutil.rmtree( tmpdir )

    # ======================================================================

    def run_alignment( self, data_store_list, index ):

        """Run the alignment.

        Creates self.aligned_datastores with the aligned images, sources, bgs, wcses, and zps.

        Parameters
        ----------
        data_store_list: list of DataStore
            data stores holding the images to be coadded.  Each
            DataStore should have its image field filled, and the
            databse should hold enough information that sources, bg,
            psf, wcs, and zp will all return something.

        index: int
            Index into data_store_list that is the alignment
            target. TODO: we need a way to specify an alignment image
            that may not be one of the images being summed!

        """

        aligner = ImageAligner( **self.pars.alignment )
        self.aligned_datastores = []
        parentwcs = data_store_list[index].wcs.copy()
        parentwcs.load()
        parentwcs.filepath = None
        parentwcs.sources_id = None
        parentwcs.md5sum = None
        for ds in data_store_list:
            wrpim, wrpsrc, wrpbg, wrppsf = aligner.run( ds.image, ds.sources, ds.bg, ds.psf, ds.wcs, ds.zp,
                                                        data_store_list[index].image,
                                                        data_store_list[index].sources )
            alds = DataStore( wrpim )
            alds.sources = wrpsrc
            alds.sources.image_id= alds.image.id
            alds.bg = wrpbg
            alds.bg.sources_id = alds.sources.id
            alds.psf = wrppsf
            alds.psf.sources_id = alds.sources.id

            alds.wcs = parentwcs.copy()
            alds.wcs.wcs = parentwcs.wcs           # reference not copy... should not be changed in practice, so OK
            alds.wcs.sources_id = alds.sources.id

            # Alignment doesn't change the zeropoint -- BUT WAIT, it could,
            #  because it could change the aperture corrections!  Issue #353.
            alds.zp = ds.zp.copy()
            alds.sources_id = alds.sources.id

            self.aligned_datastores.append( alds )

        ImageAligner.cleanup_temp_images()

    def get_coadd_prov( self, data_store_list, upstream_provs=None, code_version_id=None ):
        """Figure out the Provenance and CodeVersion of the coadded image.

        Also adds the coadd provenance to the database if necessary.

        Parameters
        ----------
          data_store_list: list of DataStore or None
            DataStore objects for all the images to be summed.  Must
            have be able to get the zp property for each DataStore.
            Ignored if upstream_provs is not None.

          upstream_provs: list of Provenance or None
            upstream provenances for the coadd provenance.  Can specify
            this instead of data_store_list, if you really know what
            you're doing.

          code_version_id: str or None
            If None, the code version will be dtermined automatically
            using Provenance.get_code_version()

        """

        # Figure out all upstream provenances
        if upstream_provs is None:
            provids = set( d.get_zp().provenance_id for d in data_store_list )
            upstream_provs = Provenance.get_batch( provids )
            if len( upstream_provs ) != len( provids ):
                raise RuntimeError( "Coadder didn't find all the expected upstream provenances!" )

        if code_version_id is None:
            code_version = Provenance.get_code_version( process='coaddition' )
        else:
            code_version = CodeVersion.get_by_id( code_version_id )

        coadd_provenance = Provenance(
            code_version_id=code_version.id,
            parameters=self.pars.get_critical_pars(),
            upstreams=upstream_provs,
            process='coaddition',
        )
        coadd_provenance.insert_if_needed()

        return coadd_provenance, code_version


    def run( self, data_store_list, aligned_datastores=None, coadd_provenance=None,
             alignment_target_datastore=None ):
        """Run coaddition on the given list of images, and return the coadded image.

        The images should have at least a set of SourceList and WorldCoordinates loaded, so they can be aligned.
        The images must also have a PSF and ZeroPoint loaded for the coaddition process.

        Parameters
        ----------
        data_store_list: list of DataStore
            data stores holding the images to be coadded.  Each
            DataStore should have its image field filled, and the
            databse should hold enough information that sources, bg,
            psf, wcs, and zp will all return something.

        aligned_datastores: list of DataStore (optional)
            Usually you don't want to give this.  If you don't, all
            images will be aligned according to the parameters.  This is
            here for efficiency (e.g. it's used in tests, where the
            results of alignment are cached).  If for some reason you
            already have the aligned images, pass in DataStores here
            with the images, source lists, backgrounds, psfs, wcses, and
            zeropoints all loaded.  The code will assume that they're
            right, i.e. that they correspond to the list of images in
            data_store_list (in the same order), and that they were
            created with the proper alignment parameters.

        alignment_target_datastore: DataStore or None
            If self.pars.alignment_index is 'other', then this needs to
            be a DataStore with loaded image and sources for the target
            image of the alignment.  This is only supported (currently)
            with the 'swarp' coaddition method.  For any other value
            of self.pars.alignment_index, this must be None.

        coadd_provenance: Provenance (optional)
            (for efficiency)

        Returns
        -------
        output: Image object
            The coadded image.

        """

        # Sort images by mjd
        dexen = list( range( 0, len(data_store_list) ) )
        dexen.sort( key=lambda i: data_store_list[i].image.mjd )
        data_store_list = [ data_store_list[i] for i in dexen ]

        # Figure out the index; index=-1 means we're using
        #   an external alignment target
        if self.pars['alignment_index'] == 'last':
            index = len(data_store_list) - 1
        elif self.pars['alignment_index'] == 'first':
            index = 0
        elif self.pars['alignment_index'] == 'other':
            if alignment_target_datastore is None:
                raise ValueError( "alignment_index 'other' requires alignment_target_datastore" )
            index = -1
        else:
            try:
                index = int( self.pars['alignment_index'] )
            except Exception:
                raise ValueError( f"alignment_index must be 'first', 'last', 'other', or an integer, not "
                                  f"\"{self.pars['alignment_index']}\"" )
            if ( index < 0 ) or ( index >= len( data_store_list ) ):
                raise ValueError( f"alignment_index {index} is outside of the range [0,{len(data_store_list)-1}]" )

        # Provenance
        if coadd_provenance is None:
            coadd_provenance, _ = self.get_coadd_prov( data_store_list )

        # Actually coadd

        if self.pars.method == 'swarp':
            # 'Swarp' method does alignment and coaddition all in one go
            if aligned_datastores is not None:
                raise RuntimeError( "Passing aligned_datastores currently not compatible with swarp coadd method" )

            if index >= 0:
                alignment_target_datastore = data_store_list[ index ]
            outhdr, outim, outwt, outfl = self._coadd_swarp( data_store_list, alignment_target_datastore )

        else:
            # Other methods require alignment first
            if index < 0:
                raise ValueError( "Only alignment method swarp supports alignment_index=other" )

            if aligned_datastores is not None:
                SCLogger.debug( "Coadder using passed aligned datastores" )
                aligned_datastores = [ aligned_datastores[i] for i in dexen ]
                self.aligned_datastores = aligned_datastores
            else:
                SCLogger.debug( "Coadder aligning all images" )
                self.run_alignment( data_store_list, index )

            # actually coadd

            aligned_images = [ d.image for d in self.aligned_datastores ]
            aligned_bgs = [ d.bg for d in self.aligned_datastores ]
            aligned_psfs = [ d.psf for d in self.aligned_datastores ]
            aligned_zps = [ d.zp for d in self.aligned_datastores ]

            if self.pars.method == 'naive':
                SCLogger.debug( "Coadder doing naive addition" )
                outim, outwt, outfl = self._coadd_naive( aligned_images )
            elif self.pars.method == 'zogy':
                SCLogger.debug( "Coadder doing zogy addition" )
                outim, outwt, outfl, outpsf, outscore = self._coadd_zogy( aligned_images,
                                                                          aligned_bgs,
                                                                          aligned_psfs,
                                                                          aligned_zps )
            else:
                raise ValueError(f'Unknown coaddition method: {self.pars.method}. Use "naive", "swarp", or "zogy".')

        output = Image.from_image_zps( [ d.zp for d in data_store_list ],
                                       index=index if index>=0 else 0,
                                       alignment_target=None if index>=0 else alignment_target_datastore.image
                                      )
        output.provenance_id = coadd_provenance.id
        output.is_coadd = True
        output.data = outim
        output.weight = outwt
        output.flags = outfl
        if 'outhdr' in locals():
            output.header = outhdr

        # Issue #350 -- where to put these?  Look at how subtraction or other things use them!!!
        # (See also comment in test_coaddition.py::test_coaddition_pipeline_outputs)
        if 'outpsf' in locals():
            output.zogy_psf = outpsf  # TODO: do we have a better place to put this?
        if 'outscore' in locals():
            output.zogy_score = outscore


        if self.pars.cleanup_alignment:
            self.aligned_datastores = None

        return output


class ParsCoaddPipeline(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.date_range = self.add_par(
            'date_range', 7.0, float, 'Number of days before end date to set start date, if start date is not given. '
        )

        self._enforce_no_new_attrs = True  # lock against new parameters

        self.override(kwargs)


class CoaddPipeline:
    """A pipeline that runs coaddition and other tasks like source extraction on the coadd image. """

    def __init__(self, **kwargs):
        self.config = Config.get()

        # top level parameters
        self.pars = ParsCoaddPipeline(**(self.config.value('coaddition.pipeline', {})))
        self.pars.augment(kwargs.get('pipeline', {}))

        # coaddition process
        coadd_config = self.config.value('coaddition.coaddition', {})
        coadd_config.update(kwargs.get('coaddition', {}))
        self.pars.add_defaults_to_dict(coadd_config)
        self.coadder = Coadder(**coadd_config)

        # source detection ("extraction" for the regular image!)
        extraction_config = self.config.value('extraction', {})
        extraction_config.update(self.config.value('coaddition.extraction', {}))  # override coadd specific pars
        extraction_config.update(kwargs.get('extraction', {}))
        extraction_config.update({'measure_psf': True})
        self.pars.add_defaults_to_dict(extraction_config)
        self.extractor = Detector(**extraction_config)

        # astrometric fit using a first pass of sextractor and then astrometric fit to Gaia
        astrometor_config = self.config.value('astrocal', {})
        astrometor_config.update(self.config.value('coaddition.astrocal', {}))  # override coadd specific pars
        astrometor_config.update(kwargs.get('astrocal', {}))
        self.pars.add_defaults_to_dict(astrometor_config)
        self.astrometor = AstroCalibrator(**astrometor_config)

        # photometric calibration:
        photometor_config = self.config.value('photocal', {})
        photometor_config.update(self.config.value('coaddition.photocal', {}))  # override coadd specific pars
        photometor_config.update(kwargs.get('photocal', {}))
        self.pars.add_defaults_to_dict(photometor_config)
        self.photometor = PhotCalibrator(**photometor_config)

        self.datastore = None  # use this datastore to save the coadd image and all the products


    def run( self, data_store_list, aligned_datastores=None ):
        """Run the CoaddPipeline

        Parameters
        ----------
          data_store_list: list of DataStore
            data stores holding the images to be coadded.  Each
            DataStore should have its image field filled, and the
            databse should hold enough information that sources, bg,
            psf, wcs, and zp will all return something.

         aligned_datastores: list of DataStore (optional)
            Usually you don't want to give this.  If you don't, all
            images will be aligned according to the parameters.  This is
            here for efficiency (e.g. it's used in tests, where the
            results of alignment are cached).  If for some reason you
            already have the aligned images, pass in DataStores here
            with the images, source lists, backgrounds, psfs, wcses, and
            zeropoints all loaded.  The code will assume that they're
            right, i.e. that they correspond to the list of images in
            data_store_list (in the same order), and that they were
            created with the proper alignment parameters.

        Returns
        -------
          A DataStore with the coadded image and other data products.

        """

        if ( ( not isinstance( data_store_list, list ) ) or
             ( not all( [ isinstance( d, DataStore ) for d in data_store_list ] ) )
            ):
            raise TypeError( "Must pass a list of DataStore objects to CoaddPipeline.run" )

        self.datastore = DataStore()
        self.make_provenance_tree( data_store_list )

        # check if this exact coadd image already exists in the DB
        with SmartSession() as dbsession:
            coadd_prov = self.datastore.prov_tree['starting_point']
            coadd_image = Image.get_coadd_from_components( [ d.zp for d in data_store_list ],
                                                           coadd_prov, session=dbsession)

        if coadd_image is not None:
            self.datastore.image = coadd_image
            self.aligned_datastores = aligned_datastores
        else:
            # the self.aligned_datastores is None unless you explicitly pass in the pre-aligned images to save time
            self.datastore.image = self.coadder.run( data_store_list,
                                                     aligned_datastores=aligned_datastores,
                                                     coadd_provenance=self.datastore.prov_tree['starting_point'] )
            self.aligned_datastores = self.coadder.aligned_datastores


        # Get sources, background, wcs, and zp of the coadded image

        # TODO: add the warnings/exception capturing, runtime/memory tracking (and Report making) as in top_level.py

        self.datastore = self.extractor.run(self.datastore)
        if self.datastore.sources is None:
            raise RuntimeError( "CoaddPipeline failed to extract sources from coadded image." )
        self.datastore = self.astrometor.run(self.datastore)
        if self.datastore.wcs is None:
            raise RuntimeError( "CoaddPipline failed to solve for WCS of coadded image." )
        self.datastore = self.photometor.run(self.datastore)
        if self.datastore.zp is None:
            raise RuntimeError( "CoaddPipeline failed to solve for zeropoint of coadded image." )


        return self.datastore

    def make_provenance_tree( self, data_store_list, upstream_provs=None, code_version_id=None ):
        """Make a provenance tree in self.datastore for all coadded data products."""

        # NOTE I'm not handling the "test_parameter" thing here, may need to.
        # (But see Issue #408)
        coadd_prov, _code_version = self.coadder.get_coadd_prov( data_store_list, upstream_provs=upstream_provs,
                                                                 code_version_id=code_version_id )
        coadd_prov.insert_if_needed()

        steps = [ 'extraction', 'astrocal', 'photocal' ]
        upstream_steps = { 'extraction': [],
                           'astrocal': [ 'extraction' ],
                           'photocal': [ 'astrocal' ]
                          }
        parses = { 'extraction': self.extractor.pars.get_critical_pars(),
                   'astrocal': self.astrometor.pars.get_critical_pars(),
                   'photocal': self.photometor.pars.get_critical_pars() }
        self.datastore.make_prov_tree( steps, parses, upstream_steps=upstream_steps, starting_point=coadd_prov )

        return self.datastore.prov_tree

    def override_parameters(self, **kwargs):
        """Override the parameters of this pipeline and its sub objects. """
        from pipeline.top_level import _PROCESS_OBJECTS

        for key, value in kwargs.items():
            if key in _PROCESS_OBJECTS:
                if isinstance(_PROCESS_OBJECTS[key], dict):
                    for sub_key, sub_value in _PROCESS_OBJECTS[key].items():
                        if sub_key in value:
                            getattr(self, sub_value).pars.override(value[sub_key])
                elif isinstance(_PROCESS_OBJECTS[key], str):
                    getattr(self, _PROCESS_OBJECTS[key]).pars.override(value)
            elif key == 'coaddition':
                self.coadder.pars.override(value)
            else:
                self.pars.override({key: value})
