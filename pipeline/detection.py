import pathlib
import random
import subprocess
import time
import warnings

import numpy as np
import numpy.lib.recfunctions as rfn
from scipy import ndimage

import astropy.table
import sep

from astropy.io import fits, votable

from util.logger import SCLogger

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.backgrounding import Backgrounder

from models.base import FileOnDiskMixin
from models.image import Image  # noqa: F401
from models.source_list import SourceList
from models.psf import PSF

from improc.sextractor import run_sextractor
from improc.tools import sigma_clipping


class ParsDetector(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.method = self.add_par(
            'method', 'sextractor', str, 'Method to use (sextractor, sep, filter)', critical=True
        )

        self.measure_psf = self.add_par(
            'measure_psf',
            False,
            bool,
            ( 'Measure PSF?  If false, will use existing image PSF.  If true, '
              'will measure PSF and put it in image object; will also iterate '
              'on source extraction to get PSF photometry with the returned PSF.' ),
            critical=True
        )

        self.apers = self.add_par(
            'apers',
            [1.0, 2.0, 3.0, 5.0],
            list,
            'Apertures in which to measure photometry; a list of floats. ',
            critical=True
        )
        self.add_alias( 'apertures', 'apers' )

        self.inf_aper_num = self.add_par(
            'inf_aper_num',
            -1,
            int,
            'Which of apers is the one to use as the "infinite" aperture for aperture corrections. '
            'If -1, will use the last aperture, not the PSF flux! ',
            critical=True
        )

        self.best_aper_num = self.add_par(
            'best_aper_num',
            0,
            int,
            'Which of apers is the one to use as the "best" aperture, for things like plotting or calculating'
            'the limiting magnitude. Note that -1 will use the PSF flux, not the last aperture on the list. '
        )

        self.aperunit = self.add_par(
            'aperunit',
            'fwhm',
            str,
            'Units of the apertures in the apers parameters; one of "fwhm" or "pixel"',
            critical=True
        )
        self.add_alias( 'aperture_unit', 'aperunit' )

        self.separation_fwhms = self.add_par(
            'separation_fwhms',
            1.0,
            float,
            'Minimum separation between sources in units of FWHM',
            critical=True
        )

        self.threshold = self.add_par(
            'threshold',
            3.0,
            [float, int],
            'The number of standard deviations above the background '
            'to use as the threshold for detecting a source. ',
            critical=True
        )

        self.subtraction = self.add_par(
            'subtraction',
            False,
            bool,
            'Whether this is expected to run on a subtraction image or a regular image. ',
            critical=True
        )

        self.sextractor_timeout = self.add_par(
            'sextractor_timeout',
            120,
            int,
            'Timeout for SExtractor, in seconds. ',
            critical=False,
        )

        self.sextractor_back_type = self.add_par(
            'sextractor_back_type',
            'MANUAL',
            str,
            ( "-BACK_TYPE parameter for sextractor: AUTO or MANUAL.  You usually want this to be MANUAL ",
              "(with sextractor_back_value=0) because background subtraction is run separately from sextractor" ),
            critical=True
        )

        self.sextractor_back_value = self.add_par(
            'sextractor_back_value',
            0,
            float,
            "-BACK_VALUE parameter for sextractor.  Ignored if sextractor_back_type is AUTO",
            critical=True
        )

        self.sextractor_back_size = self.add_par(
            'sextractor_back_size',
            None,
            ( int, None ),
            ( "-BACK_SIZE parameter for sextractor.  Ignored if sextractor_back_type is MANUAL.  "
              "Defaults to the Instrument's background_box_size" ),
            critical=True
        )

        self.sextractor_back_filtersize = self.add_par(
            'sextractor_back_filtersize',
            None,
            ( int, None ),
            ( "-BACK_FILTERSIZE parameter for sextractor.  Ignored if sextractor_back_type is MANUAL.  "
              "Defaults to the Instrument's background_filt_size" ),
            critical=True
        )

        self.backgrounding = self.add_par(
            'backgrounding',
            { 'format': 'scalar', 'method': 'zero' },
            dict,
            ( "Parameters for background subtraction; see backgrounding.py.  If subtraction is True, "
              "then backgrounding.method must be zero" ),
            critical=True
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'detection'


class Detector:
    """Extract sources (and possibly a psf) from images or subtraction images.

    There are a few different ways to get the sources from an image.
    The most common is to use SExtractor, which is the default we use for direct images.
    There is also "sep", which is a python-based equivalent (sort of) of SExtractor,
    but it is not fully compatible and not fully supported.
    Finally, you can use a matched-filter approach, meaning that you cross-correlate
    the image with the PSF and find everything above some threshold.
    There are some subtleties involved, like what to do if multiple pixels
    (connected or just close to each other) are above threshold.
    In some cases, like after running ZOGY, the image already contains a "score" image,
    which is just the normalized result of running a matched-filter.
    In that case, the "filter" method of this class will just use the score image
    instead of re-running PSF matched-filtering.

    It should also be noted that using the "filter" method does not find the PSF
    from the image (whereas the SExtractor method does). This is one of the main reasons
    SExtractor is used for detecting sources on the direct images.
    On the subtraction image, on the other hand, you can use the "filter" method
    and either use the PSF from ZOGY or just assume the subtraction image's PSF is
    the same as the direct image's PSF.

    This is also why when you use this class on the subtraction image, it will
    be initialized with pars.measure_psf=False.

    We distinguish between finding sources in a regular image, calling it "extraction"
    (we are extracting many sources of static objects, mostly) and finding sources
    in difference images, calling it "detection" (identifying transient sources that
    may or may not exist in an image). The pars.subtraction parameter is used to
    define a Detection object to use in either of these cases, with a completely
    different set of parameters for each object.

    """

    def __init__(self, **kwargs):
        """Initialize Detector.

        NOTE : if you change self.pars.backgrounding, call make_backgrounder to
        get an updated backgrounding object!

        Parmameters
        -----------
          method: str, default sextractor
            sextractor or sep.  sep is not fully supported

          measure_psf: bool, default False
            Measure the psf?  Does not make sense to set this True for
            subtraction images; for subtraction images, you must pass a
            psf using the psf parameter.

          psf: PSF, default None
            A PSF object.  Ignored if measure_psf is True.  If passed,
            will be used to determine the image FWHM to set aperture
            sizes, and will be used for PSF photometry.

          apers: list of float, default None
            Apertures in which to do aperture photometry.  If None, will
            use a default set of apertures that is (1, 2, 3, 4, 5, 7,
            10) times the FWHM.  (If this is None and no psf is measured
            or passed, then the apertures will be fixed at (2, 4, 6, 8,
            10, 14, 20) pixels.)  The "primary" aperture should be the
            first one on the list.

          aperunit: str, default fwhm
            The unit (fwhm or pixel) apers is given in; ignored if apers
            is None.

          threshold: float, default 5.0
            Threshold for finding sources in units of sigma.

          subtraction: bool, default False
            Is this Detector intended to find sources on a subtraction?
            If False, it's for finding sources on a regular image.

        """

        self.pars = ParsDetector(**kwargs)
        self.make_backgrounder()

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False


    def make_backgrounder( self ):
        self.backgrounder = Backgrounder( **(self.pars.backgrounding) )


    def run(self, *args, **kwargs):
        """Extract sources (and possibly a psf) from a regular image or a subtraction image.

        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        self.has_recalculated = False

        if self.pars.subtraction:
            if  self.backgrounder.pars.method != 'zero':
                raise ValueError( "Running detection on a subtraction requires backgrounding.method=zero" )
            try:
                ds = DataStore.from_args(*args, **kwargs)
                t_start = time.perf_counter()
                if ds.update_memory_usages:
                    import tracemalloc
                    tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

                self.pars.do_warning_exception_hangup_injection_here()

                if ds.sub_image is None and ds.image is not None and ds.image.is_sub:
                    # ...I think this should be an exception, it's an ill-constructed DataStore
                    raise RuntimeError( "You have a DataStore whose image is a subtraction." )
                    # ds.sub_image = ds.image
                    # # back-fill the image from the sub image
                    # ds.image = None
                    # ds.image_id = ds.sub_image.new_image_id
                    # ds.get_image()

                if ds.sub_image is None:
                    raise RuntimeError( "detection.py: self.pars.subtraction is true, but "
                                        "DataStore has no sub_image" )

                prov = ds.get_provenance('detection', self.pars.get_critical_pars())

                # try to find the sources/detections in memory or in the database:
                detections = ds.get_detections(prov)

                if detections is None:
                    self.has_recalculated = True

                    psf = ds.get_psf()
                    if ( psf is None ) and ( self.pars.method == 'sextractor' ):
                        raise RuntimeError( "detection on subtraction cannot proceed; datastore does "
                                            "not have a psf, and the sextractor method requires one." )

                    # NOTE -- we're assuming that the sub image is
                    #  aligned with the new image here!  That assumption
                    #  is also implicitly built into measurements.py,
                    #  and in subtraction.py there is a RuntimeError if
                    #  you try to align to ref instead of new.
                    detections, _, _, _ = self.extract_sources( ds.sub_image,
                                                                None,
                                                                psf=psf,
                                                                wcs=ds.wcs,
                                                                score=getattr( ds, 'zogy_score', None  ),
                                                                zogy_alpha=getattr( ds, 'zogy_alpha', None ) )
                    detections.image_id = ds.sub_image.id
                    if detections.provenance_id is None:
                        detections.provenance_id = prov.id
                    else:
                        if detections.provenance_id != prov.id:
                            raise ValueError('Provenance mismatch for detections!')

                detections._upstream_bitflag |= ds.sub_image.bitflag
                ds.detections = detections

                if ds.update_runtimes:
                    ds.runtimes['detection'] = time.perf_counter() - t_start
                if ds.update_memory_usages:
                    import tracemalloc
                    ds.memory_usages['detection'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

                return ds

            except Exception as e:
                SCLogger.exception( f"Exception in Detector.run: {e}" )
                ds.exceptions.append( e )
                raise

        else:  # regular image
            try:
                ds = DataStore.from_args(*args, **kwargs)
                prov = ds.get_provenance('extraction', self.pars.get_critical_pars())

                sources = ds.get_sources(provenance=prov)
                psf = ds.get_psf(provenance=prov)
                bg = ds.get_background()

                bg = self.backgrounder.run( ds )

                t_start = time.perf_counter()
                if ds.update_memory_usages:
                    import tracemalloc
                    tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

                self.pars.do_warning_exception_hangup_injection_here()

                if sources is None or psf is None:
                    # TODO: when only one of these is not found (which is a strange situation)
                    #  we may end up with a new version of the existing object
                    #  (if sources is missing, we will end up with one sources and two psfs).
                    #  This could get us in trouble when saving (the object will have the same provenance)
                    #  Right now this is taken care of using "safe_merge" but I don't know if that's the right thing.
                    self.has_recalculated = True
                    # use the latest image in the data store,
                    # or load using the provenance given in the
                    # data store's upstream_provs, or just use
                    # the most recent provenance for "preprocessing"
                    image = ds.get_image()

                    if image is None:
                        raise ValueError(f'Cannot find an image corresponding to the datastore inputs: '
                                         f'{ds.inputs_str}')

                    sources, psf, _, _ = self.extract_sources( image, bg, wcs=ds.wcs )

                    sources.image_id = image.id
                    psf.sources_id = sources.id
                    if sources.provenance_id is None:
                        sources.provenance_id = prov.id
                    else:
                        if sources.provenance_id != prov.id:
                            raise ValueError('Provenance mismatch for sources and provenance!')

                    psf.sources_id = sources.id

                ds.sources = sources
                ds.psf = psf
                if ds.image.fwhm_estimate is None:
                    ds.image.fwhm_estimate = psf.fwhm_pixels * ds.image.instrument_object.pixel_scale

                if ds.update_runtimes:
                    ds.runtimes['extraction'] = time.perf_counter() - t_start
                if ds.update_memory_usages:
                    import tracemalloc
                    ds.memory_usages['extraction'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

                bg.sources_id = sources.id
                # See Issue #440
                bg._upstream_bitflag = 0
                bg._upstream_bitflag |= ds.image.bitflag
                bg._upstream_bitflag |= sources.bitflag
                bg._upstream_bitflag |= psf.bitflag

                ds.bg = bg

                return ds

            except Exception as e:
                SCLogger.exception( f"Exception in Detector.run: {e}" )
                ds.exceptions.append(e)
                raise

    def extract_sources(self, image, bg, psf=None, wcs=None, score=None, zogy_alpha=None):
        """Calls one of the extraction methods, based on self.pars.method.

        Parameters
        ----------
        image: Image
          The Image object from which to extract sources.

        bg : Background or None
          The Background object.  If not None, will be subtracted from the image before extraction.

        psf : PSF or None
          The PSF to use for PSF photometry, aperture size
          determinations, and other things.  For methods other than
          sextractor, this can (should?) be None.  For sextractor, if
          self.pars.measure_psf is True, this should be None.  But, if
          self.pars.measure_psf is False, then for method=sextractor
          this is needed.

        wcs: WorldCoordiantes or None
          Needed if self.pars.method is 'filter'.  If self.pars.method
          is 'sextractor', this will be used in place of the one in the
          image header to get RA and Dec.

        score: numpy array
          ZOGY score image.  Needed if self.pars.method is 'filter'.

        zogy_alpha: numpy array
          ZOGY alpha.  Needed if self.pars.method is 'filter'

        Returns
        -------
        sources: SourceList object
            A list of sources with their positions and fluxes.
        psf: PSF object
            An estimate for the point spread function of the image.
        bkg: float
            An estimate for the mean value of the background of the image.
        bkgsig: float
            An estimate for the standard deviation of the background of the image.

        """
        sources = None
        input_psf = psf
        psf = None
        bkg = None
        bkgsig = None
        if self.pars.method == 'sep':
            if input_psf is not None:
                SCLogger.warning( "Passed an input_psf to extract_sources that won't be used." )
            sources = self.extract_sources_sep(image)
        elif self.pars.method == 'sextractor':
            if self.pars.subtraction:
                sources, _, _, _ = self.extract_sources_sextractor(image, bg, psf=input_psf )
            else:
                if input_psf is not None:
                    SCLogger.warning( "Passed an input_psf to extract_sources that won't be used." )
                sources, psf, bkg, bkgsig = self.extract_sources_sextractor(image, bg)
        elif self.pars.method == 'filter':
            if input_psf is not None:
                SCLogger.warning( "Passed an input_psf to extract_sources that won't be used." )
            if self.pars.subtraction:
                if ( wcs is None ) or ( score is None ) or ( zogy_alpha is None ):
                    raise RuntimeError( '"filter" extraction requires wcs, score, and zogy_alpha' )
                sources = self.extract_sources_filter( image, score, zogy_alpha, wcs )
            else:
                raise ValueError('Cannot use "filter" method on regular image!')
        else:
            raise ValueError(f'Unknown extraction method "{self.pars.method}"')

        if sources is not None:
            sources._upstream_bitflag |= image.bitflag
        if psf is not None:
            psf._upstream_bitflag |= image.bitflag

        return sources, psf, bkg, bkgsig

    def extract_sources_sextractor( self, image, bg, psf=None, wcs=None ):
        tempnamebase = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
        sourcepath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.fits'
        delfiles = [ sourcepath ]

        try:  # cleanup at the end
            apers = np.array(self.pars.apers)

            if self.pars.measure_psf:
                if psf is not None:
                    raise ValueError( "psf is not None when self.pars.measure_psf is True" )
                if self.pars.subtraction:
                    raise ValueError( "measure_psf is True when running on a subtraction; you don't want this." )
                # Run sextractor once without a psf to get objects from
                # which to build the psf.
                #
                # As for the aperture, if the units are FWHM, then we don't
                # know one yet, so guess that it's 2".  This doesn't really
                # matter that much, because we're not going to save these
                # values.
                aperrad = apers[0]
                if self.pars.aperunit == 'fwhm':
                    if image.instrument_object.pixel_scale is not None:
                        aperrad *= 2. / image.instrument_object.pixel_scale
                SCLogger.debug( "detection: running sextractor once without PSF to get sources" )
                sources, _, _ = self._run_sextractor_once( image, bg, apers=[aperrad],
                                                           psffile=None, wcs=wcs, tempname=tempnamebase )

                # Get the PSF
                SCLogger.debug( "detection: determining psf" )
                psf = self._run_psfex( tempnamebase, image, do_not_cleanup=True )
                psfpath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.psf'
                _psfxmlpath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.psf.xml'

            elif psf is not None:
                psfpaths = psf.get_fullpath()
                if psfpaths is None:
                    # PSF not saved to disk yet, so write it to a temp file we can pass to sextractor
                    barf = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
                    saveto = str( pathlib.path( FileOnDiskMixin.temp_path ) / f'{barf}.psf' )
                    psf.save( filename=saveto, filename_is_absolute=True, no_archive=True )
                    psfpaths = [ f'{saveto}.fits', f'{saveto}.xml' ]
                    delfiles = [ pathlib.Path(i) for i in psfpaths ]
                psfpath = psfpaths[0]
                _psfxmlpath = psfpaths[1]

            else:
                raise ValueError( "Must either have self.pars.measure_psf True, or must pass a psf" )

            if self.pars.aperunit == 'fwhm':
                apers *= psf.fwhm_pixels

            # Now that we have a psf, run sextractor (maybe a second time)
            # to get the actual measurements.
            SCLogger.debug( "detection: running sextractor with psf to get final source list" )

            psf_clip = psf.get_clip()
            psf_norm = 1 / np.sqrt(np.sum(psf_clip ** 2))  # normalization factor for the sextractor thresholds

            sources, bkg, bkgsig = self._run_sextractor_once(
                image,
                bg,
                apers=apers,
                psffile=psfpath,
                psfnorm=psf_norm,
                wcs=wcs,
                tempname=tempnamebase,
                seeing_fwhm=psf.fwhm_pixels * image.instrument_object.pixel_scale
            )
            SCLogger.debug( f"detection: sextractor found {len(sources.data)} sources on image {image.filepath}" )

            snr = sources.apfluxadu()[0] / sources.apfluxadu()[1]
            if snr.min() > self.pars.threshold:
                warnings.warn( "SExtractor may not have detected everything down to your threshold." )
            w = np.where( snr >= self.pars.threshold )
            sources.data = sources.data[w]
            sources.num_sources = len( sources.data )
            sources.inf_aper_num = self.pars.inf_aper_num
            sources.best_aper_num = self.pars.best_aper_num
            psf.sources_id = sources.id

        finally:
            # Clean up the temporary files created (that weren't already cleaned up by _run_sextractor_once)
            for p in delfiles:
                p.unlink( missing_ok=True )

        return sources, psf, bkg, bkgsig


    def _run_sextractor_once(self, image, bg, apers=[5, ], psffile=None, psfnorm=3.0, wcs=None,
                             tempname=None, seeing_fwhm=1.2, do_not_cleanup=False):
        """Extract a SourceList from a FITS image using SExtractor.

        This function should not be called from outside this class.  If
        you really want to run sextractor yourself, see
        improc/sextractor.py::run_sextractor

        Parameters
        ----------
          image: Image
            The Image object from which to extract.  This routine will
            use all of image, weight, and flags data.

          bg: Background or None
            If not None, a background to subtract from the image

          apers: list of float
            Aperture radii in pixels in which to do aperture photometry.

          psffile: Path or str, or None
            File that has the PSF to use for PSF photometry.  If None,
            won't do psf photometry.

          psfnorm: float
            The normalization of the PSF image (i.e., the sqrt of the
            sum of squares of the psf values).  This is used to set the
            threshold for sextractor.  When the PSF is not known, we
            will use a rough approximation and set this value to 3.0.

          wcs: WorldCoordinates or None
            If passed, will replace the WCS in the image header with the
            WCS from this object before passing it to SExtractor.

          tempname: str
            If not None, a filename base for where the catalog will be
            written.  The source file will be written to
            "{FileOnDiskMixin.temp_path}/{tempname}.sources.fits".  It
            is the responsibility of the calling routine to delete this
            temporary file.  In any event, a number of temporary files
            will be created and deleted inside this routine (image,
            weight, mask), all of which will also be deleted by this
            routine unless do_not_cleanup is True.  If tempname is None,
            the sources file will also be automatically deleted; only if
            it is not None is the calling routine always responsible for
            deleting it.

          seeing_fwhm: 1.2
            An estimate of the image seeing in arcseconds.  This is used
            in Sextractor's CLASS_STAR determination.  Ideally, it should
            be within 20% of reality, but the first time you run this
            you will probably have no idea.  1.2 is the Sextractor
            default.

          do_not_cleanup: bool, default False
            This routine writes some temp files with the image, weight,
            mask, and sourcelist data in it, much of which is probably
            redundant with what's already written somewhere.  Normally,
            they're deleted at the end of the routine.  Set this to True
            to keep the files for debugging purposes.  (If tempname is
            not None, then the sourcelist file will not be deleted even
            if this is False; the reasoning is, if you passed a
            tempname, it's because you needed that file, and indeed
            extract_sources_sextractor does.)

        Returns
        -------
          sources: SourceList, bkg: float, bkgsig: float
            sources has data already loaded
            bkg and bkgsig are the sky background estimates sextractor calculates

        """

        sextr_res = None

        try:
            if image.data is None or image.weight is None or image.flags is None:
                raise RuntimeError( "Must have all of image data, weight, and flags" )

            if tempname is not None:
                tempname = pathlib.Path( FileOnDiskMixin.temp_path ) / tempname

            imgdata = image.data if bg is None else bg.subtract_me( image.data )
            sextr_res = run_sextractor(
                image.header,
                imgdata,
                image.weight,
                maskdata = image.flags,
                outbase = tempname,
                detect_thresh = self.pars.threshold / psfnorm,
                analysis_thresh = self.pars.threshold / psfnorm,
                apers = apers,
                psffile = psffile,
                wcs = wcs,
                satur_level = image.instrument_object.average_saturation_limit( image ),
                seeing_fwhm = seeing_fwhm,
                pixel_scale = image.instrument_object.pixel_scale,
                gain = image.instrument_object.get_gain_at_pixel( image,
                                                                  image.data.shape[1] // 2,
                                                                  image.data.shape[0] // 2,
                                                                  section_id=image.section_id ),
                back_type = self.pars.sextractor_back_type,
                back_size = ( self.pars.sextractor_back_size
                              if self.pars.sextractor_back_size is not None
                              else image.instrument_object.background_box_size ),
                back_filtersize = ( self.pars.sextractor_back_filtersize
                                    if self.pars.sextractor_back_filtersize is not None
                                    else image.instrument_object.background_filt_size ),
                back_value = self.pars.sextractor_back_value,
                do_not_cleanup = do_not_cleanup,
                timeout = self.pars.sextractor_timeout
            )
            del imgdata

            bkg = sextr_res[ 'bkg_mean' ]
            bkgsig = sextr_res[ 'bkg_sig' ]

            sourcelist = SourceList( image_id=image.id, format="sextrfits", aper_rads=apers )
            # Since we don't set the filepath to the temp file, manually load
            # the _data and _info fields
            sourcelist.load( sextr_res['sources'] )
            sourcelist.num_sources = len( sourcelist.data )

            return sourcelist, bkg, bkgsig

        finally:
            # This is a little weird, but historically it was how we did it, so
            #   for compatiblity it still does it this way.  If the caller
            #   passed a tempname, it means that they want to see the
            #   sources file, so we shouldn't delete it.  (See docs
            #   on the tempname parameter above.)
            if ( not do_not_cleanup ) and ( sextr_res is not None ) and ( tempname is None ):
                sextr_res[ 'sources' ].unlink( missing_ok=True )


    def _run_psfex( self, tempname, image, psf_size=None, do_not_cleanup=False ):
        """Create a PSF from a SExtractor catalog file.

        Will run psfex twice, to make sure it has the right data size.
        The first pass, it will use a resampled PSF data array size of
        psf_size in x and y (or 25, if psf_size is None).  The second
        pass, it will use a resampled PSF data array size
        psf_size/psfsamp, where psfsamp is the psf sampling determined
        in the first pass.  In the second pass, psf_size will be what
        was passed; if None was passed, then it will be 5 times the
        measured FWHM (using the "FWHM" determined from the half-light
        radius) in the first pass.

        Parameters
        ----------
          tempname: str (required)
            The catalog file is found in
            {FileOnDiskMixin.temp_path}/{tempname}.sources.fits.

          image: Image
            The Image that the sources were extracted from.

          psf_size: int or None
            The size of one side of the thumbnail of the PSF, in pixels.
            Should be odd; if it's not, 1 will be added to it.
            If None, will be determined automatically.

          do_not_cleanup: bool, default False
            If True, don't delete the psf and psfxml files that will be
            created on the way to building the PSF that's returned.
            (Normally, these temporary files are deleted.)  The psf FITS
            file will be in
            {FileOnDiskMixin.temp_path}/{tempname}.sources.psf and the
            psfxml file will be in
            {FileOnDiskMixin.temp_path}/{tempname}.sources.psf.xml


        Returns
        -------
          A PSF object.

        """
        sourcefile = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.sources.fits'
        psffile = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.sources.psf'
        psfxmlfile = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.sources.psf.xml'

        if psf_size is not None:
            psf_size = int( psf_size )
            if psf_size % 2 == 0:
                psf_size += 1
        psf_sampling = 1.

        try:
            usepsfsize = psf_size if psf_size is not None else 25
            for i in range(2):
                psfdatasize = int( usepsfsize / psf_sampling + 0.5 )
                if psfdatasize % 2 == 0:
                    psfdatasize += 1

                # TODO: make the fwhmmax tried configurable
                #  (This is just a range of things to try to see if we can
                #  get psfex to succeed; it will stop after the first one that does.)
                fwhmmaxtotry = [ 10.0, 15.0, 20.0, 25.0 ]
                #
                # TODO: make -XML_URL configurable.  (The default there is what
                #  is installed if you install the psfex package on a
                #  debian-based distro, which is what the Dockerfile is built from.)
                for fwhmmaxdex, fwhmmax in enumerate( fwhmmaxtotry ):
                    command = [ 'psfex',
                                '-PSF_SIZE', f'{psfdatasize},{psfdatasize}',
                                '-SAMPLE_FWHMRANGE', f'0.5,{fwhmmax}',
                                '-SAMPLE_VARIABILITY', "0.2",   # Allowed FWHM variability (1.0 = 100%)
                                '-SAMPLE_IMAFLAGMASK', "0xff",
                                '-SAMPLE_MINSN', '5',  # Minimum S/N for sampling
                                '-CHECKPLOT_DEV', 'NULL',
                                '-CHECKPLOT_TYPE', 'NONE',
                                '-CHECKIMAGE_TYPE', 'NONE',
                                '-WRITE_XML', 'Y',
                                '-XML_NAME', psfxmlfile,
                                '-XML_URL', 'file:///usr/share/psfex/psfex.xsl',
                                # '-PSFVAR_DEGREES', '4',  # polynomial order for PSF fitting across image
                                sourcefile ]
                    res = subprocess.run(
                        command,
                        cwd=sourcefile.parent,
                        capture_output=True,
                        timeout=self.pars.sextractor_timeout
                    )
                    if res.returncode == 0:
                        success = True
                        psfxml = votable.parse( psfxmlfile )
                        psfstats = psfxml.get_table_by_index( 1 )
                        last_psf_sampling = psf_sampling
                        psf_sampling = psfstats.array['Sampling_Mean'][0]
                        if psf_sampling <= 0:
                            psf_sampling = last_psf_sampling
                            success = False
                        if success and ( psf_size is None ):
                            last_usepsfsize = usepsfsize
                            usepsfsize = int( np.ceil( psfstats.array['FWHM_FromFluxRadius_Mean'][0] * 5. ) )
                            if usepsfsize <= 0:
                                success = False
                                usepsfsize = last_usepsfsize
                            elif usepsfsize % 2 == 0:
                                usepsfsize += 1
                        if success:
                            fwhmmaxtotry = [ fwhmmax ]
                    if success:
                        break
                    else:
                        if fwhmmaxdex == len(fwhmmaxtotry) - 1:
                            SCLogger.error( f"psfex failed with all attempted fwhmmax.\n"
                                            f"stdout:\n------\n{res.stdout.decode('utf-8')}\n"
                                            f"stderr:\n------\n{res.stderr.decode('utf-8')}" )
                            raise RuntimeError( "Repeated failures from psfex call" )
                        SCLogger.warning( f"psfex failed with fwhmmax={fwhmmax}, trying {fwhmmaxtotry[fwhmmaxdex+1]}" )


            psf = PSF( format="psfex", fwhm_pixels=float(psfstats.array['FWHM_FromFluxRadius_Mean'][0]) )
            psf.load( psfpath=psffile, psfxmlpath=psfxmlfile )
            psf.header['IMAXIS1'] = image.data.shape[1]
            psf.header['IMAXIS2'] = image.data.shape[0]
            with fits.open(psffile) as hdul:
                hdul[1].header['IMAXIS1'] = image.data.shape[1]
                hdul[1].header['IMAXIS2'] = image.data.shape[0]
                # TODO: any more information about the Image or SourceList we want to save here?

            return psf

        finally:
            if not do_not_cleanup:
                psffile.unlink( missing_ok=True )
                psfxmlfile.unlink( missing_ok=True )

    def extract_sources_sep(self, image):
        """Run source-extraction (using SExtractor) on the given image.

        Parameters
        ----------
        image: Image
            The image to extract sources from.

        Returns
        -------
        sources: SourceList
            The list of sources detected in the image.
            This contains a table where each row represents
            one source that was detected, along with all its properties.

        """

        SCLogger.warning( "The sep detecton method isn't fully compatible with the rest of the pipeline." )

        # TODO: finish this
        # TODO: this should also generate an estimate of the PSF?
        # TODO: handle background subtraction or lack thereof!  Needs
        #       to be configurable as in the case of sextractor.

        data = image.data

        # see the note in https://sep.readthedocs.io/en/v1.0.x/tutorial.html#Finally-a-brief-word-on-byte-order
        if ( data.dtype == '>f8' ) or ( data.dtype == '>f4' ):  # TODO: what about other datatypes besides f4, f8?
            data = data.byteswap().newbyteorder()
        b = sep.Background(data)

        data_sub = data - b.back()

        objects = sep.extract(data_sub, self.pars.threshold, err=b.rms())

        # get the radius containing half the flux for each source
        r, _ = sep.flux_radius(data_sub, objects['x'], objects['y'], 6.0 * objects['a'], 0.5, subpix=5)
        r = np.array(r, dtype=[('rhalf', '<f4')])
        objects = rfn.merge_arrays((objects, r), flatten=True)
        sources = SourceList(image_id=image.id, data=objects, format='sepnpy')

        return sources

    def extract_sources_filter( self, image, score, zogy_alpha, wcs, fwhm=None ):
        """Find sources in an image using the matched-filter method.

        If the image has a "score" array, will use that.
        If not, will apply the PSF of the image to make a score array.

        Sources are detected as points in the score image that are
        above the given threshold. If points are too close, they
        will be merged into a single source.
        The output SourceList will include the aperture and PSF photometry,
        and the x,y positions will be given using the source centroid.

        # TODO: should we do iterative PSF tapered centroiding?

        Parameters
        ----------
        image: Image
            The image to extract sources from.

        score: numpy array
            The score image from zogy

        zogy_alpha: SOMETHING

        wcs: WorldCoordiantes

        fwhm: float or None
            FWHM of the image in arcseconds; if None, will be read from image.fwhm_estimate

        Returns
        -------
        sources: SourceList
            The list of sources detected in the image.
            This contains a table where each row represents
            one source that was detected, along with all its properties.

        """
        if score is None:
            raise NotImplementedError('Still need to add the matched-filter cross correlation! ')
        score = score.copy()

        # psf_image = None
        # if image.psf is not None:
        #     psf_image = image.psf.get_clip()
        # elif getattr(image, 'zogy_psf', None) is not None:
        #     psf_image = image.zogy_psf

        if fwhm is None:
            fwhm = image.fwhm_estimate

        if fwhm is None:
            raise RuntimeError("Cannot find a FWHM estimate from the given image or its new_image attribute.")

        # typical scale of PSF is X times the FWHM
        fwhm_pixels = max(int(np.ceil(fwhm * self.pars.separation_fwhms / image.instrument_object.pixel_scale)), 1)
        # remove flagged pixels
        score[image.flags > 0] = np.nan

        # remove the edges
        border = fwhm_pixels * 2
        score[:border, :] = np.nan
        score[-border:, :] = np.nan
        score[:, :border] = np.nan
        score[:, -border:] = np.nan

        # normalize the score based on sigma_clipping
        # TODO: we should check if we still need this after b/g subtraction on the input images
        mu, sigma = sigma_clipping(score)
        score = (score - mu) / sigma
        det_map = abs(score) > self.pars.threshold  # catch negative peaks too (can get rid of them later)

        # dilate the map to merge nearby peaks
        struc = np.zeros((3, 3), dtype=bool)
        struc[1, :] = True
        struc[:, 1] = True
        det_map = ndimage.binary_dilation(det_map, iterations=fwhm_pixels, structure=struc).astype(det_map.dtype)

        # label the map to get the number of sources
        labels, num_sources = ndimage.label(det_map)
        all_idx = np.arange(1, num_sources + 1)
        # get the x,y positions of the sources (rough estimate)
        xys = ndimage.center_of_mass(abs(image.data), labels, all_idx)
        x = np.array([xy[1] for xy in xys])
        y = np.array([xy[0] for xy in xys])
        coords = wcs.wcs.pixel_to_world(x, y)
        ra = [c.ra.value for c in coords]
        dec = [c.dec.value for c in coords]

        label_fluxes = ndimage.sum(image.data, labels, all_idx)  # sum image values where labeled

        # run aperture and iterative PSF photometry
        fluxes = ndimage.labeled_comprehension(zogy_alpha, labels, all_idx, np.max, float, np.nan)

        region_sizes = [np.sum(labels == i) for i in all_idx]

        def peak_score(arr):
            """Find the best score, whether positive or negative. """
            return arr[np.unravel_index(np.nanargmax(abs(arr)), arr.shape)]

        scores = ndimage.labeled_comprehension(score, labels, all_idx, peak_score, float, np.nan)

        def count_nans(arr):
            return np.sum(np.isnan(arr))

        num_flagged = ndimage.labeled_comprehension(score, labels, all_idx, count_nans, int, 0)

        # TODO: add a correct aperture photometry, instead of the label_fluxes which only sums the labeled pixels

        tab = astropy.table.Table(
            [ra, dec, x, y, label_fluxes, fluxes, region_sizes, num_flagged, scores],
            names=('ra', 'dec', 'x', 'y', 'flux', 'psf_flux', 'num_pixels', 'num_flagged', 'score'),
            meta={'fwhm': fwhm, 'threshold': self.pars.threshold}
        )

        sources = SourceList(data=tab, format='filter', num_sources=num_sources)
        sources.labelled_regions = labels  # this is not saved with the SourceList, but added for debugging/testing!

        return sources
