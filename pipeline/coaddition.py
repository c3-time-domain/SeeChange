import numpy as np
from numpy.fft import fft2, ifft2, fftshift

from astropy.time import Time

from sep import Background

from models.base import SmartSession
from models.provenance import Provenance
from models.image import Image

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.detection import Detector
from pipeline.backgrounding import Backgrounder
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from util.util import get_latest_provenance, parse_session

from improc.bitmask_tools import dilate_bitflag
from improc.inpainting import Inpainter
from improc.alignment import ImageAligner
from improc.tools import sigma_clipping

from util.config import Config
from util.util import listify


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
            'Multiplicative factor for the PSF FWHM (in pixels) to use for dilating the flag maps. ',
            critical=True,
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
            b = Background(data)
            bkg = b.globalback
            sigma = b.globalrms
        elif self.pars.noise_estimator.startswith('sigma'):
            bkg, sigma = sigma_clipping(data)
        else:
            raise ValueError(
                f'Unknown noise estimator: {self.pars.noise_estimator}.  Use "sep" or "sigma_clipping" or "bkg_rms". '
            )

        return bkg, sigma

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
        if not all(type(image) == type(images[0]) for image in images):
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

    def _zogy_core(self, datacube, psfcube, sigmas, flux_zps):
        """Perform the core Zackay & Ofek proper image coaddition on the input data cube.

        Parameters
        ----------
        datacube: ndarray
            The data cube to coadd. Can be images or weight maps (or anything else).
        psfcube: ndarray
            The PSF cube to use for coaddition.
        sigmas: ndarray
            The background noise estimate for each image in the data cube.
            Must be a 1D array with a length equal to the first axis of the data cube.
            It could have additional dimensions, but it will be reshaped to be multiplied
            with the data cube and psf cube.
        flux_zps: ndarray
            The flux zero points for each image in the data cube.
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

        Parameters
        ----------
        images: list of Image or list of 2D ndarrays
            Images that have been aligned to each other.
            Each image must also have a PSF and a background object attached.
        weights: list of 2D ndarrays
            The weights to use for each image.
            If images is given as Image objects, can be left as None.
        flags: list of 2D ndarrays
            The bit flags to use for each image.
            If images is given as Image objects, can be left as None.
        psf_clips: list of 2D ndarrays
            The PSF images to use for each image.
            If images is given as Image objects, can be left as None.
        psf_fwhms: list of floats
            The FWHM of the PSF for each image.
            If images is given as Image objects, can be left as None.
        flux_zps: list of floats
            The flux zero points for each image.
            If images is given as Image objects, can be left as None.
        bkg_means: list of floats
            The mean background for each image.
            If images is given as Image objects, can be left as None.
            If images are already background subtracted, set these to zeros.
        bkg_sigmas: list of floats
            The RMS of the background for each image.
            If images is given as Image objects, can be left as None.

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
        if not all(type(image) == type(images[0]) for image in images):
            raise ValueError('Not all images are of the same type. ')

        if isinstance(images[0], Image):
            data = []
            flags = []
            weights = []
            psf_clips = []
            psf_fwhms = []
            flux_zps = []
            
            for image in images:
                data.append(image.data)
                flags.append(image.flags)
                weights.append(image.weight)
                psf_clips.append(image.psf.get_clip())
                psf_fwhms.append(image.psf.fwhm_pixels)
                flux_zps.append(10 ** (0.4 * image.zp.zp))

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
            if not isinstance(images[0], Image):
                raise ValueError('Background must be given if images are not Image objects. ')
            bkg_means = [im.bg.value for im in images]
            bkg_sigmas = [im.bg.noise for im in images]

        imcube = np.array(data)
        flcube = np.array(flags)
        wtcube = np.array(weights)
        psfcube = np.array(psfs)
        bkg_means = np.reshape(np.array(bkg_means), (len(bkg_means), 1, 1))
        bkg_sigmas = np.reshape(np.array(bkg_sigmas), (len(bkg_sigmas), 1, 1))
        flux_zps = np.reshape(np.array(flux_zps), (len(flux_zps), 1, 1))

        # subtract the background
        imcube -= bkg_means

        # make sure to inpaint missing data
        # TODO: make sure images are scaled before inpainting, or add that in the inpainting code
        imcube = self.inpainter.run(imcube, flcube, wtcube)

        if np.any(np.isnan(imcube)):
            raise ValueError('There are still NaNs in the image data after inpainting!')

        # This is where the magic happens
        outim, psf, score = self._zogy_core(imcube, psfcube, bkg_sigmas, flux_zps)

        # coadd the variance as well
        varflag = wtcube == 0
        wtcube2 = wtcube ** 2
        wtcube2[varflag] = np.nan
        varmap = 1 / wtcube2
        varmap = self.inpainter.run(varmap, varflag, wtcube)  # wtcube doesn't do anything, maybe put something else?
        outvarmap, _, _ = self._zogy_core(varmap, psfcube, bkg_sigmas, flux_zps)
        outwt = 1 / np.sqrt(np.abs(outvarmap))

        outfl = np.zeros(outim.shape, dtype='uint16')
        for f, p in zip(flags, psf_fwhms):
            splash_pixels = int(np.ceil(p * self.pars.flag_fwhm_factor))
            outfl = outfl | dilate_bitflag(f, iterations=splash_pixels)

        return outim, outwt, outfl, psf, score

    def run(self, images, aligned_images=None):
        """Run coaddition on the given list of images, and return the coadded image.

        The images should have at least a set of SourceList and WorldCoordinates loaded, so they can be aligned.
        The images must also have a PSF and ZeroPoint loaded for the coaddition process.

        Parameters
        ----------
        images: list of Image objects
            The input Image objects that will be used as the upstream_images for the new, coadded image.
        aligned_images: list of Image objects (optional)
            A list of images that correspond to the images list,
            but already aligned to each other, so it can be put into the output image's aligned_images attribute.
            The aligned images must have the same alignment parameters as in the output image's provenance
            (i.e., the "alignment" dictionary should be the same as in the coadder object's pars).
            If not given, the output Image object will generate the aligned images by itself,
            using the input images and its provenance's alignment parameters.

        Returns
        -------
        output: Image object
            The coadded image.
        """
        images.sort(key=lambda image: image.mjd)
        if self.pars.alignment['to_index'] == 'last':
            index = len(images) - 1
        elif self.pars.alignment['to_index'] == 'first':
            index = 0
        else:  # TODO: consider allowing a specific index as integer?
            raise ValueError(f"Unknown alignment reference index: {self.pars.alignment['to_index']}")
        output = Image.from_images(images, index=index)
        output.provenance = Provenance(
            code_version=images[0].provenance.code_version,
            parameters=self.pars.get_critical_pars(),
            upstreams=output.get_upstream_provenances(),
            process='coaddition',
        )
        output.provenance_id = output.provenance.id

        output.is_coadd = True

        # note: output is a newly formed image, that has upstream_images
        # and also a Provenance that contains "alignment" parameters...
        # it can create its own aligned_images, but if you already have them,
        # you can pass them in to save time re-calculating them here:
        if aligned_images is not None:
            output.aligned_images = aligned_images
            output.info['alignment_parameters'] = self.pars.alignment

        if self.pars.method == 'naive':
            outim, outwt, outfl = self._coadd_naive(output.aligned_images)
        elif self.pars.method == 'zogy':
            outim, outwt, outfl, outpsf, outscore = self._coadd_zogy(output.aligned_images)
        else:
            raise ValueError(f'Unknown coaddition method: {self.pars.method}. Use "naive" or "zogy".')

        output.data = outim
        output.weight = outwt
        output.flags = outfl

        if 'outpsf' in locals():
            output.zogy_psf = outpsf  # TODO: do we have a better place to put this?
        if 'outscore' in locals():
            output.zogy_score = outscore

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
        extraction_config = self.config.value('extraction.sources', {})
        extraction_config.update(self.config.value('coaddition.extraction.sources', {}))  # override coadd specific pars
        extraction_config.update(kwargs.get('extraction', {}).get('sources', {}))
        extraction_config.update({'measure_psf': True})
        self.pars.add_defaults_to_dict(extraction_config)
        self.extractor = Detector(**extraction_config)

        # background estimation
        backgrounder_config = self.config.value('extraction.bg', {})
        backgrounder_config.update(self.config.value('coaddition.extraction.bg', {}))  # override coadd specific pars
        backgrounder_config.update(kwargs.get('extraction', {}).get('bg', {}))
        self.pars.add_defaults_to_dict(backgrounder_config)
        self.backgrounder = Backgrounder(**backgrounder_config)

        # astrometric fit using a first pass of sextractor and then astrometric fit to Gaia
        astrometor_config = self.config.value('extraction.wcs', {})
        astrometor_config.update(self.config.value('coaddition.extraction.wcs', {}))  # override coadd specific pars
        astrometor_config.update(kwargs.get('extraction', {}).get('wcs', {}))
        self.pars.add_defaults_to_dict(astrometor_config)
        self.astrometor = AstroCalibrator(**astrometor_config)

        # photometric calibration:
        photometor_config = self.config.value('extraction.zp', {})
        photometor_config.update(self.config.value('coaddition.extraction.zp', {}))  # override coadd specific pars
        photometor_config.update(kwargs.get('extraction', {}).get('zp', {}))
        self.pars.add_defaults_to_dict(photometor_config)
        self.photometor = PhotCalibrator(**photometor_config)

        # make sure when calling get_critical_pars() these objects will produce the full, nested dictionary
        siblings = {
            'sources': self.extractor.pars,
            'bg': self.backgrounder.pars,
            'wcs': self.astrometor.pars,
            'zp': self.photometor.pars
        }
        self.extractor.pars.add_siblings(siblings)
        self.backgrounder.pars.add_siblings(siblings)
        self.astrometor.pars.add_siblings(siblings)
        self.photometor.pars.add_siblings(siblings)

        self.datastore = None  # use this datastore to save the coadd image and all the products

        self.images = None  # use this to store the input images
        self.aligned_images = None  # use this to pass in already aligned images

    def parse_inputs(self, *args, **kwargs):
        """Parse the possible inputs to the run method.

        The possible input types are:
        - unnamed arguments that are all Image objects, to be treated as self.images
        - a list of Image objects, assigned into self.images
        - two lists of Image objects, the second one is a list of aligned images matching the first list,
          such that the two lists are assigned to self.images and self.aligned_images
        - start_time + end_time + instrument + filter + section_id + provenance_id + RA + Dec (or target)

        To pass the latter option, must use named parameters.
        An optional session can be given, either as one of the named
        or unnamed args, and it will be used throughout the pipeline
        (and left open at the end).

        The start_time and end_time can be floats (interpreted as MJD)
        or strings (interpreted by astropy.time.Time(...)), or None.
        If end_time is not given, will use current time.
        If start_time is not given, will use end_time minus the
        coaddition pipeline parameter date_range.
        The provenance_ids can be None, which will use the most recent "preprocessing" provenance.
        Can also provide a list of provenance_ids or a single string.
        The coordinates can be given as either float (decimal degrees) or strings
        (sexagesimal hours for RA and degrees for Dec).
        Can leave coordinates empty and provide a "target" instead (i.e., target
        will be used as the "field identifier" in the survey).

        In either case, the output is a list of Image objects.
        Each image is checked to see if it has the related products
        (SourceList, PSF, WorldCoordinates, ZeroPoint).
        If not, it will raise an exception. If giving these images directly
        (i.e., not letting the pipeline load them from DB) the calling scope
        must make sure to load those products first.
        """
        # first parse the session from args and kwargs
        args, kwargs, session = parse_session(*args, **kwargs)
        self.images = None
        self.aligned_images = None
        if len(args) == 0:
            pass  # there are not args, we can skip them quietly
        elif len(args) == 1 and isinstance(args[0], list):
            if not all([isinstance(a, Image) for a in args[0]]):
                raise TypeError('When supplying a list, all elements must be Image objects. ')
            self.images = args[0]  # in case we are given a list of images
        elif len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list):
            if not all([isinstance(im, Image) for im in args[0] + args[1]]):
                raise TypeError('When supplying two lists, both must be lists of Image objects. ')
            self.images = args[0]
            self.aligned_images = args[1]
        elif all([isinstance(a, Image) for a in args]):
            self.images = args
        else:
            raise ValueError('All unnamed arguments must be Image objects. ')

        if self.images is None:  # get the images from the DB
            # if no images were given, parse the named parameters
            ra = kwargs.get('ra', None)
            dec = kwargs.get('dec', None)
            target = kwargs.get('target', None)
            if target is None and (ra is None or dec is None):
                raise ValueError('Must give either target or RA and Dec. ')

            start_time = kwargs.get('start_time', None)
            end_time = kwargs.get('end_time', None)
            if end_time is None:
                end_time = Time.now().mjd
            if start_time is None:
                start_time = end_time - self.pars.date_range

            instrument = kwargs.get('instrument', None)
            filter = kwargs.get('filter', None)
            section_id = str(kwargs.get('section_id', None))

            provenance_ids = kwargs.get('provenance_ids', None)
            if provenance_ids is None:
                prov = get_latest_provenance('preprocessing', session=session)
                provenance_ids = [prov.id]
            provenance_ids = listify(provenance_ids)

            with SmartSession(session) as dbsession:
                stmt = Image.query_images(
                    ra=ra,
                    dec=dec,
                    target=target,
                    section_id=section_id,
                    instrument=instrument,
                    filter=filter,
                    min_dateobs=start_time,
                    max_dateobs=end_time,
                    provenance_ids=provenance_ids
                )
                self.images = dbsession.scalars(stmt.order_by(Image.mjd.asc())).all()

        return session

    def run(self, *args, **kwargs):
        session = self.parse_inputs(*args, **kwargs)
        if self.images is None or len(self.images) == 0:
            raise ValueError('No images found matching the given parameters. ')

        # use the images and their source lists to get a list of provenances and code versions
        coadd_upstreams = set()
        code_versions = set()
        # assumes each image given to the coaddition pipline has sources loaded
        for im in self.images:
            coadd_upstreams.add(im.provenance)
            coadd_upstreams.add(im.sources.provenance)
            code_versions.add(im.provenance.code_version)
            code_versions.add(im.sources.provenance.code_version)

        code_versions = list(code_versions)
        code_versions.sort(key=lambda x: x.id)
        code_version = code_versions[-1]  # choose the most recent ID if there are multiple code versions
        coadd_upstreams = list(coadd_upstreams)

        self.datastore = DataStore()
        self.datastore.prov_tree = self.make_provenance_tree(coadd_upstreams, code_version, session=session)

        # check if this exact coadd image already exists in the DB
        with SmartSession(session) as dbsession:
            coadd_prov = self.datastore.prov_tree['coaddition']
            coadd_image = Image.get_image_from_upstreams(self.images, coadd_prov, session=dbsession)

        if coadd_image is not None:
            self.datastore.image = coadd_image
        else:
            # the self.aligned_images is None unless you explicitly pass in the pre-aligned images to save time
            self.datastore.image = self.coadder.run(self.images, self.aligned_images)

        # TODO: add the warnings/exception capturing, runtime/memory tracking (and Report making) as in top_level.py
        self.datastore = self.extractor.run(self.datastore)
        self.datastore = self.backgrounder.run(self.datastore)
        self.datastore = self.astrometor.run(self.datastore)
        self.datastore = self.photometor.run(self.datastore)

        return self.datastore.image

    def make_provenance_tree(self, coadd_upstreams, code_version, session=None):
        """Make a (short) provenance tree to use when fetching the provenances of upstreams. """
        with SmartSession(session) as session:

            pars_dict = self.coadder.pars.get_critical_pars()
            coadd_prov = Provenance(
                code_version=code_version,
                process='coaddition',
                upstreams=coadd_upstreams,
                parameters=pars_dict,
                is_testing="test_parameter" in pars_dict,  # this is a flag for testing purposes
            )
            coadd_prov = coadd_prov.merge_concurrent(session=session, commit=True)

            # the extraction pipeline
            pars_dict = self.extractor.pars.get_critical_pars()
            extract_prov = Provenance(
                code_version=code_version,
                process='extraction',
                upstreams=[coadd_prov],
                parameters=pars_dict,
                is_testing="test_parameter" in pars_dict['sources'],  # this is a flag for testing purposes
            )
            extract_prov = extract_prov.merge_concurrent(session=session, commit=True)

        return {'coaddition': coadd_prov, 'extraction': extract_prov}

    def override_parameters(self, **kwargs):
        """Override the parameters of this pipeline and its sub objects. """
        from pipeline.top_level import PROCESS_OBJECTS

        for key, value in kwargs.items():
            if key in PROCESS_OBJECTS:
                if isinstance(PROCESS_OBJECTS[key], dict):
                    for sub_key, sub_value in PROCESS_OBJECTS[key].items():
                        if sub_key in value:
                            getattr(self, sub_value).pars.override(value[sub_key])
                elif isinstance(PROCESS_OBJECTS[key], str):
                    getattr(self, PROCESS_OBJECTS[key]).pars.override(value)
            elif key == 'coaddition':
                self.coadder.pars.override(value)
            else:
                self.pars.override({key: value})
