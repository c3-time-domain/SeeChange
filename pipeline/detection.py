import pathlib
import random
import subprocess
import time
import warnings

import numpy as np
import numpy.lib.recfunctions as rfn
from scipy import ndimage

import astropy.table
import sep_pjw as sep

from astropy.io import fits, votable

from util.config import Config
from util.logger import SCLogger

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.base import FileOnDiskMixin, CODE_ROOT
from models.image import Image  # noqa: F401
from models.source_list import SourceList
from models.psf import PSF

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

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

    def run(self, *args, **kwargs):
        """Extract sources (and possibly a psf) from a regular image or a subtraction image.

        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        self.has_recalculated = False

        if self.pars.subtraction:
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

                    # NOTE -- we're assuming that the sub image is
                    #  aligned with the new image here!  That assumption
                    #  is also implicitly built into measurements.py,
                    #  and in subtraction.py there is a RuntimeError if
                    #  you try to align to ref instead of new.
                    detections, _, _, _ = self.extract_sources( ds.sub_image,
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

                t_start = time.perf_counter()
                if ds.update_memory_usages:
                    import tracemalloc
                    tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

                self.pars.do_warning_exception_hangup_injection_here()

                sources = ds.get_sources(provenance=prov)
                psf = ds.get_psf(provenance=prov)

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

                    sources, psf, _, _ = self.extract_sources( image, wcs=ds.wcs )

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

                return ds

            except Exception as e:
                SCLogger.exception( f"Exception in Detector.run: {e}" )
                ds.exceptions.append(e)
                raise

    def extract_sources(self, image, wcs=None, score=None, zogy_alpha=None):
        """Calls one of the extraction methods, based on self.pars.method.

        Parameters
        ----------
        image: Image
          The Image object from which to extract sources.

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
        psf = None
        bkg = None
        bkgsig = None
        if self.pars.method == 'sep':
            sources = self.extract_sources_sep(image)
        elif self.pars.method == 'sextractor':
            if self.pars.subtraction:
                sources, _, _, _ = self.extract_sources_sextractor(image, psffile=None)
            else:
                sources, psf, bkg, bkgsig = self.extract_sources_sextractor(image)
        elif self.pars.method == 'filter':
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

    def extract_sources_sextractor( self, image, psffile=None, wcs=None ):
        tempnamebase = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
        sourcepath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.fits'
        psfpath = pathlib.Path( psffile ) if psffile is not None else None
        psfxmlpath = None

        try:  # cleanup at the end
            apers = np.array(self.pars.apers)

            if self.pars.measure_psf:
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
                sources, _, _ = self._run_sextractor_once( image, apers=[aperrad],
                                                           psffile=None, wcs=wcs, tempname=tempnamebase )

                # Get the PSF
                SCLogger.debug( "detection: determining psf" )
                psf = self._run_psfex( tempnamebase, image, do_not_cleanup=True )
                psfpath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.psf'
                psfxmlpath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.psf.xml'
            else:
                psf = None

            if self.pars.aperunit == 'fwhm':
                if psf is None:
                    raise RuntimeError( "No psf measured or passed to extract_sources_sextractor, so apertures can "
                                        "not be based on the FWHM." )
                else:
                    apers *= psf.fwhm_pixels

            # Now that we have a psf, run sextractor (maybe a second time)
            # to get the actual measurements.
            SCLogger.debug( "detection: running sextractor with psf to get final source list" )

            if psf is not None:
                psf_clip = psf.get_clip()
                psf_norm = 1 / np.sqrt(np.sum(psf_clip ** 2))  # normalization factor for the sextractor thresholds
            else:  # we don't have a psf for some reason, use the "good enough" approximation
                psf_norm = 3.0

            sources, bkg, bkgsig = self._run_sextractor_once(
                image,
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
            sourcepath.unlink( missing_ok=True )
            if psffile is None:
                if psfpath is not None:
                    psfpath.unlink( missing_ok=True )
                if psfxmlpath is not None:
                    psfxmlpath.unlink( missing_ok=True )

        return sources, psf, bkg, bkgsig

    def _run_sextractor_once(self, image, apers=[5, ], psffile=None, psfnorm=3.0, wcs=None,
                             tempname=None, seeing_fwhm=1.2, do_not_cleanup=False):
        """Extract a SourceList from a FITS image using SExtractor.

        This function should not be called from outside this class.

        Parameters
        ----------
          image: Image
            The Image object from which to extract.  This routine will
            use all of image, weight, and flags data.

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
        tmpnamebase = tempname
        if tmpnamebase is None: tmpnamebase = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
        tmpimage = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tmpnamebase}.fits'
        tmpweight = tmpimage.parent / f'{tmpnamebase}.weight.fits'
        tmpflags = tmpimage.parent / f'{tmpnamebase}.flags.fits'
        tmpsources = tmpimage.parent / f'{tmpnamebase}.sources.fits'
        tmpxml = tmpimage.parent / f'{tmpnamebase}.sources.xml'
        tmpparams = tmpimage.parent / f'{tmpnamebase}.param'

        # For debugging purposes
        self._tmpimage = tmpimage
        self._tmpweight = tmpweight
        self._tmpflags = tmpflags
        self._tmpsources = tmpsources
        self._tmpxml = tmpxml

        if image.data is None or image.weight is None or image.flags is None:
            raise RuntimeError( "Must have all of image data, weight, and flags" )

        # Figure out where astromatic config files are:
        astromatic_dir = None
        cfg = Config.get()
        if cfg.value( 'astromatic.config_dir' ) is not None:
            astromatic_dir = pathlib.Path( cfg.value( 'astromatic.config_dir' ) )
        else:
            astromatic_dir = pathlib.Path( CODE_ROOT )
        astromatic_dir = astromatic_dir / cfg.value( 'astromatic.config_subdir' )
        if not astromatic_dir.is_dir():
            raise FileNotFoundError( f"Astromatic config dir {str(astromatic_dir)} doesn't exist "
                                     f"or isn't a directory." )

        # TODO : make these configurable by instrument (at least!)
        # For now we're using the astromatic defaults that everybody
        # just uses....
        conv = astromatic_dir / "default.conv"
        nnw = astromatic_dir / "default.nnw"

        if not ( conv.is_file() and nnw.is_file() ):
            raise FileNotFoundError( f"Can't find SExtractor conv and/or nnw file: {conv} , {nnw}" )

        if psffile is not None:
            if not pathlib.Path(psffile).is_file():
                raise FileNotFoundError( f"Can't read PSF file {psffile}" )
            psfargs = [ '-PSF_NAME', psffile ]
            paramfilebase = astromatic_dir / "sourcelist_sextractor_with_psf.param"
        else:
            psfargs = []
            paramfilebase = astromatic_dir / "sourcelist_sextractor.param"

        # SExtractor reads the measurements it produces from a parameters
        # file.  We need to edit it, though, so that the number of
        # apertures we have matches the apertures we ask for.
        # TODO : review the default param file and make sure we have the
        #  things we want, and don't have too much.
        #
        # (Note that adding the SPREAD_MODEL parameter seems to add
        # substantially to sextractor runtime-- on the decam test image
        # on my desktop, it goes from a several seconds to a minute and
        # a half.  SPREAD_MODEL is supposed to be a much better
        # star/galaxy separator than the older CLASS_STAR parameter.
        # So, for now, don't use SPREAD_MODEL, even though it would be
        # nice.  It's just too slow.  Investigate whether there's a way
        # to parallelize this (using -NTHREADS 8 doesn't lead sextractor
        # to using more than one CPU), or even GPUize it....  (I.e.,
        # rewrite sextractor....)

        if len(apers) == 1:
            paramfile = paramfilebase
        else:
            paramfile = tmpparams
            with open( paramfilebase ) as ifp:
                params = [ line.strip() for line in ifp.readlines() ]
            for i in range(len(params)):
                if params[i] in [ "FLUX_APER", "FLUXERR_APER" ]:
                    params[i] = f"{params[i]}({len(apers)})"
            with open( paramfile, "w") as ofp:
                for param in params:
                    ofp.write( f"{param}\n" )

        try:
            if not isinstance( image.header, fits.Header ):
                raise TypeError( f"Expected image.header to be an astropy.io.fits.Header, but it's a "
                                 f"{type(image.header)}" )
            hdr = image.header.copy()
            if wcs is not None:
                hdr.update( wcs.wcs.to_header() )

            fits.writeto( tmpimage, image.data, header=hdr )
            fits.writeto( tmpweight, image.weight )
            fits.writeto( tmpflags, image.flags )

            # TODO : right now, we're assuming that the default background
            #  subtraction is fine.  Experience shows that in crowded fields
            #  (e.g. star-choked galactic fields), a much slower algorithm
            #  can do a lot better.

            # TODO: Understand RESCALE_WEIGHTS and WEIGHT_GAIN.
            #  Since we believe our weight image is right, we don't
            #  want to be doing any rescaling of it, but it's possible
            #  that I don't fully understand what these parameters
            #  really are.  (Documentation is lacking; see
            #  https://www.astromatic.net/2009/06/02/playing-the-weighting-game-i/
            #  and notice the "...to be continued".)

            # The sextractor DETECT_THRESH and ANALYSIS_THRESH don't
            # exactly correspond to what we want when we set a
            # threshold.  (There's also the question as to : which
            # measurement (i.e. which one of the apertures, or the psf
            # photometry) are we using to determine threshold?  We'll
            # use the primary aperture.)  Empirically, we need to set
            # the sextractor thresholds lower than the threshold we want
            # in order to get everything that's at least the threshold
            # we want.  Outside of this function, we'll have to crop it.
            # (Dividing by 3 may not be enough...)

            # NOTE -- looking at the SExtractor documentation, it needs
            # SEEING_FWHM to be ±20% right for CLASS_STAR to be reasonable for
            # bright sources, ±5% for dim.  We have a hardcore chicken-and-egg
            # problem here.  The default SEEING_FWHM is 1.2; try just going with that,
            # and give it PIXEL_SCALE that's right.  We may need to move to
            # doing this iteratively (i.e. go from two to three runs of SExtractor;
            # first time around, do whatever, but look at the distribution of FWHMs
            # in an attempt to figure out what the actual seeing FWHM is.)

            args = [ "source-extractor",
                     "-CATALOG_NAME", tmpsources,
                     "-CATALOG_TYPE", "FITS_LDAC",
                     "-WRITE_XML", "Y",
                     "-XML_NAME", tmpxml,
                     "-PARAMETERS_NAME", paramfile,
                     "-THRESH_TYPE", "RELATIVE",
                     "-DETECT_THRESH", str( self.pars.threshold / psfnorm ),
                     "-ANALYSIS_THRESH", str( self.pars.threshold / psfnorm ),
                     "-FILTER", "Y",
                     "-FILTER_NAME", str(conv),
                     "-WEIGHT_TYPE", "MAP_WEIGHT",
                     "-RESCALE_WEIGHTS", "N",
                     "-WEIGHT_IMAGE", str(tmpweight),
                     "-WEIGHT_GAIN", "N",
                     "-FLAG_IMAGE", str(tmpflags),
                     "-FLAG_TYPE", "OR",
                     "-PHOT_APERTURES", ",".join( [ str(a*2.) for a in apers ] ),
                     "-SATUR_LEVEL", str( image.instrument_object.average_saturation_limit( image ) ),
                     "-GAIN", "1.0",  # TODO: we should probably put the instrument gain here
                     "-STARNNW_NAME", nnw,
                     "-SEEING_FWHM", str( seeing_fwhm ),
                     "-PIXEL_SCALE", str( image.instrument_object.pixel_scale ),
                     "-BACK_TYPE", "AUTO",
                     "-BACK_SIZE", str( image.instrument_object.background_box_size ),
                     "-BACK_FILTERSIZE", str( image.instrument_object.background_filt_size ),
                     "-MEMORY_OBJSTACK", str( 20000 ),  # TODO: make these configurable?
                     "-MEMORY_PIXSTACK", str( 1000000 ),
                     "-MEMORY_BUFSIZE", str( 4096 ),
                    ]
            args.extend( psfargs )
            args.append( tmpimage )
            res = subprocess.run(args, cwd=tmpimage.parent, capture_output=True, timeout=self.pars.sextractor_timeout)
            if res.returncode != 0:
                SCLogger.error( f"Got return {res.returncode} from sextractor call; stderr:\n{res.stderr}\n"
                                f"-------\nstdout:\n{res.stdout}" )
                raise RuntimeError( "Error return from source-extractor call" )

            # Get the background from the xml file that sextractor wrote
            sextrstat = votable.parse( tmpxml ).get_table_by_index( 1 )
            bkg = sextrstat.array['Background_Mean'][0][0]
            bkgsig = sextrstat.array['Background_StDev'][0][0]

            sourcelist = SourceList( image_id=image.id, format="sextrfits", aper_rads=apers )
            # Since we don't set the filepath to the temp file, manually load
            # the _data and _info fields
            sourcelist.load( tmpsources )
            sourcelist.num_sources = len( sourcelist.data )

            return sourcelist, bkg, bkgsig

        finally:
            if not do_not_cleanup:
                tmpimage.unlink( missing_ok=True )
                tmpweight.unlink( missing_ok=True )
                tmpflags.unlink( missing_ok=True )
                tmpparams.unlink( missing_ok=True )
                tmpxml.unlink( missing_ok=True )
                if tempname is None: tmpsources.unlink( missing_ok=True )

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
