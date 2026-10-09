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
from util.config import Config

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.backgrounding import Backgrounder

from models.base import FileOnDiskMixin
from models.image import Image  # noqa: F401
from models.source_list import SourceList
from models.psf import PSFExPSF

from improc.sextractor import run_sextractor
from improc.tools import sigma_clipping


class ParsDetector(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        # IMPORTANT
        # Notice that many of the parameters critical=False that really probably ought
        #   to be critical=True.  Reason: the subconfig_update system in parameters.py.
        # The massive subconfigs parameter, which IS critical, holds values that will
        #   be set into the parameters at runtime.
        # This *does* break the whole "critical" system of parameters a bit.  However,
        #   we were facing the problem of wanting to have a *single* provenacne that
        #   had *different* configs for galactic and extragalactic fields.
        # This also means that the defaults configured in most of the "add_par" commands below
        #   are never seen; they will be overwritten by the values from the default
        #   in subconfigs and subconfigs_noncritical

        self.method = self.add_par(
            name = 'method',
            default = 'sextractor',
            par_types = str,
            docstring = 'Method to use (sextractor, sep, filter)',
            critical = False
        )

        self.measure_psf = self.add_par(
            name = 'measure_psf',
            default = False,
            par_types = bool,
            docstring = ( 'Measure PSF?  If false, will use existing image PSF.  If true, '
                          'will measure PSF and put it in image object; will also iterate '
                          'on source extraction to get PSF photometry with the returned PSF.' ),
            critical = False
        )

        self.apers = self.add_par(
            name = 'apers',
            default = [1.0, 2.0, 3.0, 5.0],
            par_types = list,
            docstring = 'Apertures in which to measure photometry; a list of floats. ',
            critical = False
        )
        self.add_alias( 'apertures', 'apers' )

        self.inf_aper_num = self.add_par(
            name = 'inf_aper_num',
            default = -1,
            par_types = int,
            docstring = ( 'Which of apers is the one to use as the "infinite" aperture for aperture corrections. '
                          'If -1, will use the last aperture, not the PSF flux! ' ),
            critical = False
        )

        self.best_aper_num = self.add_par(
            name = 'best_aper_num',
            default = 0,
            par_types = int,
            docstring = ( 'Which of apers is the one to use as the "best" aperture, for things like plotting '
                          'or calculating the limiting magnitude. Note that -1 will use the PSF flux, not the '
                          'last aperture on the list. ' ),
            critical = False
        )

        self.aperunit = self.add_par(
            name = 'aperunit',
            default = 'fwhm',
            par_types = str,
            docstring = 'Units of the apertures in the apers parameters; one of "fwhm" or "pixel"',
            critical = False
        )
        self.add_alias( 'aperture_unit', 'aperunit' )

        self.separation_fwhms = self.add_par(
            name = 'separation_fwhms',
            default = 1.0,
            par_types = float,
            docstring = 'Minimum separation between sources in units of FWHM',
            critical = False
        )

        self.initial_threshold = self.add_par(
            name = 'initial_threshold',
            default = 1.5,
            par_types = [float, int],
            docstring = ( 'Like sextr_threshold, but used for the initial sextractor run before psf determination. '
                          'You want this to be higher so that only decent detections are passed on to psfex.' ),
            critical = False
        )

        self.sextr_threshold = self.add_par(
            name = 'sextr_threshold',
            default = 1.5,
            par_types = [float, int],
            docstring = ( 'The number of standard deviations above the background to use as the threshold '
                        'for detecting a source.  Passed to sextractor DETECT_THRESH and ANALYSIS_THRDSH '
                        'on the second pass when the psf is known.' ),
            critical = False
        )


        self.snr_threshold = self.add_par(
            name = 'snr_threshold',
            default = 3.0,
            par_types = [float, int],
            docstring = 'Only keep things whose S/N is at least this high.',
            critical = False
        )

        self.subtraction = self.add_par(
            name = 'subtraction',
            default = False,
            par_types = bool,
            docstring = 'Whether this is expected to run on a subtraction image or a regular image. ',
            critical = False
        )

        self.sextractor_timeout = self.add_par(
            name = 'sextractor_timeout',
            default = 120,
            par_types = int,
            docstring = 'Timeout for SExtractor, in seconds. ',
            critical = False
        )

        self.sextractor_back_type = self.add_par(
            name = 'sextractor_back_type',
            default = 'MANUAL',
            par_types = str,
            docstring = ( "-BACK_TYPE parameter for sextractor: AUTO or MANUAL.  You usually want this to be "
                          "MANUAL (with sextractor_back_value=0) because background subtraction is run separately "
                          "from sextractor" ),
            critical = False
        )

        self.sextractor_back_value = self.add_par(
            name = 'sextractor_back_value',
            default = 0,
            par_types = float,
            docstring = "-BACK_VALUE parameter for sextractor.  Ignored if sextractor_back_type is AUTO",
            critical = False
        )

        self.sextractor_back_size = self.add_par(
            name = 'sextractor_back_size',
            default = None,
            par_types = ( int, None ),
            docstring = ( "-BACK_SIZE parameter for sextractor.  Ignored if sextractor_back_type is MANUAL.  "
                          "Defaults to the Instrument's background_box_size" ),
            critical = False
        )

        self.sextractor_back_filtersize = self.add_par(
            name = 'sextractor_back_filtersize',
            default = None,
            par_types = ( int, None ),
            docstring = ( "-BACK_FILTERSIZE parameter for sextractor.  Ignored if sextractor_back_type is MANUAL.  "
                          "Defaults to the Instrument's background_filt_size" ),
            critical = False
        )

        self.backgrounding = self.add_par(
            name = 'backgrounding',
            default = { 'format': 'scalar', 'method': 'zero' },
            par_types = dict,
            docstring = ( "Parameters for background subtraction; see backgrounding.py.  If subtraction is True, "
                          "then backgrounding.method must be zero" ),
            critical = False
        )

        self.psf_method = self.add_par(
            name = 'psf_method',
            default = 'psfex',
            par_types = str,
            docstring = ( "Which PSF method to use.  (Currently only psfex is supported.)  If subtraction is "
                          "True, no PSF fitting is done, so all psf_* parameters are ignored" ),
            critical = False
        )

        self.psf_timeout = self.add_par(
            name = 'psf_timeout',
            default = 240,
            par_types = ( float, int ),
            docstring = "How many seconds to try fitting the PSF before assuming the process is hung",
            critical = False
        )

        self.psf_params = self.add_par(
            name = 'psf_params',
            default = { 'fwhm_min': 0.5,
                        'fwhm_max_to_try': [ 10.0, 15.0, 20.0, 25.0 ],
                        'psf_init_size': 25,
                        'psf_final_size': None
                       },
            par_types = dict,
            docstring = "Parameters for PSF extraction; details depend on psf_method.",
            critical = False
        )

        self.subconfigs = self.add_par(
            name = 'subconfigs',
            default = {
                'choice_algorithm': 'star_density',
                'choice_params': {
                    # Magnitude cutoff to look at star density
                    'star_mag_cutoff': 20,
                    # For healpix(32,nest=True), each healpix is about 100' (1.8°) on a side,
                    #   so each healpix is roughly 3.4 square degrees
                    'star_density_cutoff': 1e5,
                },
                'default_subconfig': 'extragalactic',
                'subconfigs': {
                    'extragalactic': {
                        'method': 'sextractor',
                        'measure_psf': False,
                        'apers': [ 1.0, 2.0, 3.0, 5.0 ],
                        'inf_aper_num': -1,
                        'best_aper_num': 0,
                        'aperunit': 'fwhm',
                        'separation_fwhms': 1.0,
                        'initial_threshold': 1.5,
                        'sextr_threshold': 1.5,
                        'snr_threshold': 3.0,
                        'subtraction': False,
                        'sextractor_back_type': 'MANUAL',
                        'sextractor_back_value': 0.,
                        'sextractor_back_size': None,
                        'sextractor_back_filtersize': None,
                        'backgrounding': { 'format': 'scalar', 'method': 'zero' },
                        # So, yeah, you're wondering why there are values for psf_method and
                        #   psf_params when measure_psf is False.
                        # We need measure_psf to default to False because there are places in the
                        #   pipeline that depend on that.  However, we do want to have some defaults
                        #   so that if somebody sets measure_psf to True, there will be defaults
                        #   there for them.
                        'psf_method': 'psfex',
                        'psf_params': { 'fwhm_min': 0.5,
                                        'fwhm_max_to_try': [ 10.0, 15.0, 20.0, 25.0 ],
                                        'psf_init_size': 25,
                                        'psf_final_size': None }
                    },
                    'galactic': {
                        'snr_threshold': 10.0,
                    }
                }
            },
            par_types = dict,
            docstring = ( "Algoirthm for choosing all the config parameters, and replacments for all the config "
                          "parameters based on that choice." ),
            critical = True
        )

        self.subconfigs_noncritical = self.add_par(
            name = 'subconfigs_noncritical',
            default = {
                'choice_params': {
                    # Parent directory to look for config files and tables. If None, uses CODE_ROOT
                    'config_dir': None,
                    'gaia_density_catalog': 'share/gaia_density/gaia_healpix_density.pq'
                },
                'subconfigs': {
                    'extragalactic': {
                        'sextractor_timeout': 240,
                        'psf_timeout': 300.,
                    },
                    'galactic': {
                        'sextractor_timeout': 120,
                        'psf_timeout': 180.,
                    }
                }
            },
            par_types = dict,
            docstring = "Substitutions for self.subconfigs that shouldn't go in the provenance",
            critical = False
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
        """Initialize Detector."""

        self.pars = ParsDetector(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False


    def make_backgrounder( self ):
        self.backgrounder = Backgrounder( **(self.pars.backgrounding) )


    def run(self, *args, input_psf=None, do_not_load=False, **kwargs):
        """Extract sources (and possibly a psf) from a regular image or a subtraction image.

        Paramters
        ---------
        Most arguments are parsed by the DataStore.parse_args() method.

        input_psf : PSF, default None
           Only pass this if self.pars.measure_psf is False.  In that case,
           you can pass this in as the PSF that will be used to measure
           stuff.

           On a subtraction image, if input_psf is None, It will use the
           psf object in the datastore.

           On a regular image— you probably shouldn't as there you want
           to measure the PSF.  However, for some tests, and efficiency,
           we support this with a regular image.  In that case, the psf
           that is "created" will be a copy of input_psf.  For a regular
           image, if selfpars.measure_psf is False, then input_psf is
           required.


        Returns
        -------
        Returns a DataStore object with the products of the processing.

        """
        self.has_recalculated = False

        if self.pars.subtraction:
            try:
                ds = DataStore.from_args( *args, **kwargs )
                self.pars.subconfig_update( ds )
                self.make_backgrounder()
                if  self.backgrounder.pars.method != 'zero':
                    raise ValueError( "Running detection on a subtraction requires backgrounding.method=zero" )

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
                detections = None if do_not_load else ds.get_detections( prov )

                if detections is None:
                    self.has_recalculated = True

                    psf = ds.get_psf() if input_psf is None else input_psf
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
                    # ****** OMG ROB LOOK AT THIS ******
                    # _imname = pathlib.Path( ds.image.filepath ).name
                    # wcshdr = ds.wcs.wcs.to_header()
                    # fits.writeto( f'temp-{_imname}.fits', ds.image.data, wcshdr, overwrite=True )
                    # fits.writeto( f'temp-{_imname}_sub.fits', ds.sub_image.data, wcshdr, overwrite=True )
                    # fits.writeto( f'temp-{_imname}_zogyscore.fits', ds.zogy_score, wcshdr, overwrite=True )
                    # fits.writeto( f'temp-{_imname}_zogyalpha.fits', ds.zogy_alpha, wcshdr, overwrite=True )
                    # detections.ds9_regfile( f'temp-{_imname}.reg', radcolor=None )
                    # ****** OMG ROB LOOK AT THIS ******
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
                ds = DataStore.from_args( *args, **kwargs )
                self.pars.subconfig_update( ds )
                self.make_backgrounder()
                prov = ds.get_provenance('extraction', self.pars.get_critical_pars())

                if do_not_load:
                    sources = None
                    psf = None
                    bg = None
                else:
                    sources = ds.get_sources(provenance=prov)
                    psf = ds.get_psf(provenance=prov)
                    bg = ds.get_background()

                if bg is None:
                    bg = self.backgrounder.run( ds )

                t_start = time.perf_counter()
                if ds.update_memory_usages:
                    import tracemalloc
                    tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

                self.pars.do_warning_exception_hangup_injection_here()

                if sources is None or psf is None:
                    # TODO: think about what it means when only one of these is set.
                    #   That probably shouldn't happen, since both are determined
                    #   at the same time!  It probably means database corruption.
                    self.has_recalculated = True
                    # use the latest image in the data store,
                    # or load using the provenance given in the
                    # data store's upstream_provs, or just use
                    # the most recent provenance for "preprocessing"
                    image = ds.get_image()
                    if image is None:
                        raise ValueError(f'Cannot find an image corresponding to the datastore inputs: '
                                         f'{ds.inputs_str}')

                    if ( not self.pars.measure_psf ) and ( input_psf is None ):
                        raise ValueError( "self.pars.measure_psf is False you didn't give an input_psf." )

                    sources, psf, _, _ = self.extract_sources( image, bg, psf=input_psf, wcs=ds.wcs )

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
                    SCLogger.debug( f"FWHM estimate = {ds.image.fwhm_estimate:.02f} arcsec" )

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

        wcs: WorldCoordinates or None
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
                sources, psf, bkg, bkgsig = self.extract_sources_sextractor(image, bg, psf=input_psf)
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

                # ****
                # sources.ds9_regfile( 'them.reg', radcolor={'star': (2,'yellow'), 'nonstar': (2,'blue'),
                #                                            'bad': (2.4, 'red'), 'flagged': (2.8,'orange'),
                #                                            'highsn': (3.0, 'green') } )
                # import pdb; pdb.set_trace()
                # ****

                # Get the PSF
                SCLogger.debug( "detection: determining psf..." )
                cfg = Config.get()
                if self.pars.psf_method == 'psfex':
                    psf = self._run_psfex( tempnamebase, image, do_not_cleanup=True )
                    SCLogger.debug( f"...psf done, got FWHM = "
                                    f"{psf.fwhm_pixels*image.instrument_object.pixel_scale:.03f} arcsec" )
                    psfpath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.psf'
                    _psfxmlpath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.psf.xml'
                    delfiles.extend( [ psfpath, _psfxmlpath ] )
                else:
                    raise ValueError( f"Unknown psf method {cfg.value('psf_method')}; only psfex is supported." )

            elif psf is not None:
                # Make a copy of the psf and wipe out the id and the sources id so that
                #   this will be saved to the database as a new psf object.
                #   (Note: still pointing to the same data blocks, but that should be OK.)
                psf = psf.copy()
                psf.sources_id = None
                psf._id = None
                psfpaths = psf.get_fullpath()
                if psfpaths is None:
                    # PSF not saved to disk yet, so write it to a temp file we can pass to sextractor
                    barf = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
                    saveto = str( pathlib.Path( FileOnDiskMixin.temp_path ) / f'{barf}.psf' )
                    psf.save( filename=saveto, filename_is_absolute=True, no_archive=True )
                    psfpaths = [ f'{saveto}.fits', f'{saveto}.xml' ]
                    delfiles.extend( [ pathlib.Path(i) for i in psfpaths ] )
                psfpath = psfpaths[0]
                _psfxmlpath = psfpaths[1]
                # Now wipe out the filepath, so that a new one will be generated when saving
                #   this psf
                psf.filepath = None

            else:
                raise ValueError( "Must either have self.pars.measure_psf True, or must pass a psf" )

            if self.pars.aperunit == 'fwhm':
                apers *= psf.fwhm_pixels

            # Now that we have a psf, run sextractor (maybe a second time)
            # to get the actual measurements.
            SCLogger.debug( "detection: running sextractor with psf to get final source list" )

            psf_clip = psf.get_clip()
            # NOTE -- at the moment, _run_sextractor_once doesnt' use this next value
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
            if snr.min() > 0.75 * self.pars.snr_threshold:
                warnings.warn( "SExtractor may not have detected everything down to your threshold." )
            w = np.where( snr >= self.pars.snr_threshold )
            SCLogger.debug( f"{len(w[0])} out of {len(sources.data)} sources above the SNR cut of "
                            f"{self.pars.snr_threshold}" )
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

          psfnorm: float, default 3.0
            This is complicated. SExtractor detect_thresh and
            analysis_thresh are the thresholds, respectively, for
            finding something in the first place to look at, and for
            including in an object's isophotal area.

            OMG I'M SO CONFUSED.  See Issue #536.

            FOR NOW THIS IS NOT USED AT ALL.

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
            thresh = self.pars.initial_threshold if psffile is None else self.pars.sextr_threshold
            # thresh /= psfnorm
            sextr_res = run_sextractor(
                image.header,
                imgdata,
                image.weight,
                maskdata = image.flags,
                outbase = tempname,
                detect_thresh = thresh,
                analysis_thresh = thresh,
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


    def _run_psfex( self, tempname, image, do_not_cleanup=False ):
        """Create a PSF from a SExtractor catalog file.

        Will run psfex twice, to make sure it has the right data size.
        The first pass, it will use a resampled PSF data array size of
        psf__init_size in x and y.  The second pass, it will use a
        resampled PSF data array size psf_final_size/psfsamp, where
        psfsamp is the psf sampling determined in the first pass.  In
        the second pass, if psf_final_size is None, then it will be 5
        times the measured FWHM (using the "FWHM" determined from the
        half-light radius) in the first pass.

        Parameters
        ----------
          tempname: str (required)
            The catalog file is found in
            {FileOnDiskMixin.temp_path}/{tempname}.sources.fits.

          image: Image
            The Image that the sources were extracted from.

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

        psf_init_size = self.pars.psf_params['psf_init_size']
        psf_final_size = self.pars.psf_params['psf_final_size']

        psf_init_size = int( psf_init_size )
        if psf_init_size % 2 == 0:
            psf_init_size += 1

        if psf_final_size is not None:
            psf_final_size = int( psf_final_size )
            if psf_final_size % 2 == 0:
                psf_final_size += 1

        psf_sampling = 1.

        try:
            found_fwhm = None
            for i in range(2):
                if i == 0:
                    usepsfsize = psf_init_size
                    minfwhm = self.pars.psf_params['fwhm_min']
                    #  (This is just a range of things to try to see if we can
                    #  get psfex to succeed; it will stop after the first one that does.)
                    fwhmmaxtotry = self.pars.psf_params['fwhm_max_to_try']
                else:
                    if psf_final_size is None:
                        usepsfsize = int( np.ceil( 5.0 * found_fwhm ) )
                    else:
                        usepsfsize = psf_final_size
                    minfwhm = 0.75 * found_fwhm
                    fwhmmaxtotry = [ 1.5 * found_fwhm ]

                psfdatasize = int( usepsfsize / psf_sampling + 0.5 )
                if psfdatasize % 2 == 0:
                    psfdatasize += 1

                # TODO: make -XML_URL configurable.  (The default there is what
                #  is installed if you install the psfex package on a
                #  debian-based distro, which is what the Dockerfile is built from.)
                for fwhmmaxdex, fwhmmax in enumerate( fwhmmaxtotry ):
                    command = [ 'psfex',
                                '-PSF_SIZE', f'{psfdatasize},{psfdatasize}',
                                '-SAMPLE_FWHMRANGE', f'{minfwhm},{fwhmmax}',
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
                    _psfcom = " ".join( f'"{str(i)}"' if ' ' in str(i) else str(i) for i in command )
                    SCLogger.debug( f"Running command {_psfcom}" )
                    res = subprocess.run(
                        command,
                        cwd=sourcefile.parent,
                        capture_output=True,
                        timeout=self.pars.psf_timeout
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
                        if success:
                            found_fwhm = psfstats.array['FWHM_FromFluxRadius_Mean'][0]
                            if found_fwhm <= 0:
                                success = False
                                found_fwhm = None
                    if success:
                        break
                    else:
                        if fwhmmaxdex == len(fwhmmaxtotry) - 1:
                            SCLogger.error( f"psfex failed with all attempted fwhmmax.\n"
                                            f"stdout:\n------\n{res.stdout.decode('utf-8')}\n"
                                            f"stderr:\n------\n{res.stderr.decode('utf-8')}" )
                            raise RuntimeError( "Repeated failures from psfex call" )
                        SCLogger.warning( f"psfex failed with fwhmmax={fwhmmax}, trying {fwhmmaxtotry[fwhmmaxdex+1]}" )


            psf = PSFExPSF( fwhm_pixels=float(found_fwhm) )
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

        objects = sep.extract(data_sub, self.pars.snr_threshold, err=b.rms())

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
        det_map = abs(score) > self.pars.snr_threshold  # catch negative peaks too (can get rid of them later)

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
            meta={'fwhm': fwhm, 'threshold': self.pars.snr_threshold}
        )

        sources = SourceList(data=tab, format='filter', num_sources=num_sources)
        sources.labelled_regions = labels  # this is not saved with the SourceList, but added for debugging/testing!

        return sources
