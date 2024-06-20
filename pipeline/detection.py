import os
import pathlib
import random
import subprocess
import time

import numpy as np
import numpy.lib.recfunctions as rfn
from scipy import ndimage
import sqlalchemy as sa

import astropy.table
import sep

from astropy.io import fits, votable

from util.config import Config
from util.logger import SCLogger
from util.util import parse_bool

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.base import FileOnDiskMixin, CODE_ROOT
from models.image import Image
from models.psf import PSF
from models.source_list import SourceList

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

        self.psf = self.add_par(
            'psf',
            None,
            ( PSF, int, None ),
            'Use this PSF; pass the PSF object, or its integer id. '
            'If None, will not do PSF photometry.  Ignored if measure_psf is True.' ,
            critical=True
        )

        self.apers = self.add_par(
            'apers',
            None,
            ( None, list ),
            'Apertures in which to measure photometry; a list of floats or None',
            critical=True
        )
        self.add_alias( 'apertures', 'apers' )

        self.inf_aper_num = self.add_par(
            'inf_aper_num',
            None,
            ( None, int ),
            ( 'Which of apers is the one to use as the "infinite" aperture for aperture corrections; '
              'default is to use the last one.  Ignored if self.apers is None.' ),
            critical=True
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
        try:  # first make sure we get back a datastore, even an empty one
            ds, session = DataStore.from_args(*args, **kwargs)
        except Exception as e:
            return DataStore.catch_failure_to_parse(e, *args)

        # try to find the sources/detections in memory or in the database:
        if self.pars.subtraction:
            try:
                t_start = time.perf_counter()
                if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                    import tracemalloc
                    tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

                self.pars.do_warning_exception_hangup_injection_here()

                if ds.sub_image is None and ds.image is not None and ds.image.is_sub:
                    ds.sub_image = ds.image
                    ds.image = ds.sub_image.new_image  # back-fill the image from the sub_image

                prov = ds.get_provenance('detection', self.pars.get_critical_pars(), session=session)

                detections = ds.get_detections(prov, session=session)

                if detections is None:
                    self.has_recalculated = True

                    # load the subtraction image from memory
                    # or load using the provenance given in the
                    # data store's upstream_provs, or just use
                    # the most recent provenance for "subtraction"
                    image = ds.get_subtraction(session=session)

                    if image is None:
                        raise ValueError(
                            f'Cannot find a subtraction image corresponding to the datastore inputs: {ds.get_inputs()}'
                        )

                    # TODO -- should probably pass **kwargs along to extract_sources
                    #  in any event, need a way of passing parameters
                    #  Question: why is it not enough to just define what you need in the Parameters object?
                    #  Related to issue #50
                    detections, _, _, _ = self.extract_sources( image )

                    detections.image = image

                    if detections.provenance is None:
                        detections.provenance = prov
                    else:
                        if detections.provenance.id != prov.id:
                            raise ValueError('Provenance mismatch for detections and provenance!')

                detections._upstream_bitflag |= ds.sub_image.bitflag
                ds.sub_image.sources = detections
                ds.detections = detections

                ds.runtimes['detection'] = time.perf_counter() - t_start
                if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                    import tracemalloc
                    ds.memory_usages['detection'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            except Exception as e:
                ds.catch_exception(e)
            finally:  # make sure datastore is returned to be used in the next step
                return ds

        else:  # regular image
            prov = ds.get_provenance('extraction', self.pars.get_critical_pars(), session=session)
            try:
                t_start = time.perf_counter()
                if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                    import tracemalloc
                    tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

                self.pars.do_warning_exception_hangup_injection_here()

                sources = ds.get_sources(prov, session=session)
                psf = ds.get_psf(prov, session=session)

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
                    image = ds.get_image(session=session)

                    if image is None:
                        raise ValueError(f'Cannot find an image corresponding to the datastore inputs: {ds.get_inputs()}')

                    sources, psf, bkg, bkgsig = self.extract_sources( image )
                    sources.image = image
                    if sources.provenance is None:
                        sources.provenance = prov
                    else:
                        if sources.provenance.id != prov.id:
                            raise ValueError('Provenance mismatch for sources and provenance!')

                    psf.image_id = image.id
                    if psf.provenance is None:
                        psf.provenance = prov
                    else:
                        if psf.provenance.id != prov.id:
                            raise ValueError('Provenance mismatch for pfs and provenance!')

                ds.sources = sources
                ds.psf = psf
                ds.image.fwhm_estimate = psf.fwhm_pixels  # TODO: should we only write if the property is None?
                if self.has_recalculated:
                    ds.image.bkg_mean_estimate = float( bkg )
                    ds.image.bkg_rms_estimate = float( bkgsig )

                ds.runtimes['extraction'] = time.perf_counter() - t_start
                if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                    import tracemalloc
                    ds.memory_usages['extraction'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            except Exception as e:
                ds.catch_exception(e)
            finally:  # make sure datastore is returned to be used in the next step
                return ds

    def extract_sources(self, image):
        """Calls one of the extraction methods, based on self.pars.method. """
        sources = None
        psf = None
        bkg = None
        bkgsig = None
        if self.pars.method == 'sep':
            sources = self.extract_sources_sep(image)
        elif self.pars.method == 'sextractor':
            if self.pars.subtraction:
                psffile = None if self.pars.psf is None else self.pars.psf.get_fullpath()
                sources, _, _, _ = self.extract_sources_sextractor(image, psffile=psffile)
            else:
                sources, psf, bkg, bkgsig = self.extract_sources_sextractor(image)
        elif self.pars.method == 'filter':
            if self.pars.subtraction:
                sources = self.extract_sources_filter(image)
            else:
                raise ValueError('Cannot use "filter" method on regular image!')
        else:
            raise ValueError(f'Unknown extraction method "{self.pars.method}"')

        if psf is not None:
            if psf._upstream_bitflag is None:
                psf._upstream_bitflag = 0
            psf._upstream_bitflag |= image.bitflag
        if sources is not None:
            if sources._upstream_bitflag is None:
                sources._upstream_bitflag = 0
            sources._upstream_bitflag |= image.bitflag

        return sources, psf, bkg, bkgsig

    def extract_sources_sextractor( self, image, psffile=None ):
        tempnamebase = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
        sourcepath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.fits'
        psfpath = pathlib.Path( psffile ) if psffile is not None else None
        psfxmlpath = None

        try:  # cleanup at the end
            if self.pars.apers is None:
                apers = np.array( image.instrument_object.standard_apertures() )
                inf_aper_num = image.instrument_object.fiducial_aperture()
            else:
                apers = self.pars.apers
                inf_aper_num = self.pars.inf_aper_num
                if inf_aper_num is None:
                    inf_aper_num = len( apers ) - 1

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
                                                           psffile=None, tempname=tempnamebase )

                # Get the PSF
                SCLogger.debug( "detection: determining psf" )
                psf = self._run_psfex( tempnamebase, image, do_not_cleanup=True )
                psfpath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.psf'
                psfxmlpath = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempnamebase}.sources.psf.xml'
            elif self.pars.psf is not None:
                psf = self.pars.psf
                psfpath, psfxmlpath = psf.get_fullpath()
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
            sources, bkg, bkgsig = self._run_sextractor_once( image, apers=apers,
                                                              psffile=psfpath, tempname=tempnamebase )
            SCLogger.debug( f"detection: sextractor found {len(sources.data)} sources" )

            snr = sources.apfluxadu()[0] / sources.apfluxadu()[1]
            if snr.min() > self.pars.threshold:
                SCLogger.warning( "SExtractor may not have detected everything down to your threshold." )
            w = np.where( snr >= self.pars.threshold )
            sources.data = sources.data[w]
            sources.num_sources = len( sources.data )
            sources._inf_aper_num = inf_aper_num

        finally:
            # Clean up the temporary files created (that weren't already cleaned up by _run_sextractor_once)
            sourcepath.unlink( missing_ok=True )
            if ( psffile is None ) and ( self.pars.psf is None ):
                if psfpath is not None: psfpath.unlink( missing_ok=True )
                if psfxmlpath is not None: psfxmlpath.unlink( missing_ok=True )

        return sources, psf, bkg, bkgsig

    def _run_sextractor_once( self, image, apers=[5, ], psffile=None, tempname=None, do_not_cleanup=False ):
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
            raise RuntimeError( f"Must have all of image data, weight, and flags" )

        # Figure out where astromatic config files are:
        astromatic_dir = None
        cfg = Config.get()
        if cfg.value( 'astromatic.config_dir' ) is not None:
            astromatic_dir = pathlib.Path( cfg.value( 'astromatic.config_dir' ) )
        elif cfg.value( 'astromatic.config_subdir' ) is not None:
            astromatic_dir = pathlib.Path( CODE_ROOT ) / cfg.value( 'astromatic.config_subdir' )
        if astromatic_dir is None:
            raise FileNotFoundError( "Can't figure out where astromatic config directory is" )
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
        # Right now, it's in there, as source_list uses it in the
        # is_star property.  Investigate whether there's a way to
        # parallelize this (using -NTHREADS 8 doesn't lead sextractor to
        # using more than one CPU), or even GPUize it....  (I.e.,
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

            # TODO : if the image is already on disk, then we may not need
            #  to do the writing here.  In that case, we do have to think
            #  about whether the extensions are different HDUs in the same
            #  file, or if they are separate files (which I think can just be
            #  handled by changing the sextractor arguments a bit).  If
            #  they're non-FITS files, we'll have to write them.  For
            #  simplicity, right now, just write the temp files, even though
            #  it might be redundant.
            fits.writeto( tmpimage, image.data, header=image.header )
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

            args = [ "source-extractor",
                     "-CATALOG_NAME", tmpsources,
                     "-CATALOG_TYPE", "FITS_LDAC",
                     "-WRITE_XML", "Y",
                     "-XML_NAME", tmpxml,
                     "-PARAMETERS_NAME", paramfile,
                     "-THRESH_TYPE", "RELATIVE",
                     "-DETECT_THRESH", str( self.pars.threshold / 3. ),
                     "-ANALYSIS_THRESH", str( self.pars.threshold / 3. ),
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
                     "-GAIN", "1.0",
                     "-STARNNW_NAME", nnw,
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
                raise RuntimeError( f"Error return from source-extractor call" )

            # Get the background from the xml file that sextractor wrote
            sextrstat = votable.parse( tmpxml ).get_table_by_index( 1 )
            bkg = sextrstat.array['Background_Mean'][0][0]
            bkgsig = sextrstat.array['Background_StDev'][0][0]

            sourcelist = SourceList( image=image, format="sextrfits", aper_rads=apers )
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
                        fwhmmaxtotry = [ fwhmmax ]
                        break
                    else:
                        if fwhmmaxdex == len(fwhmmaxtotry) - 1:
                            SCLogger.error( f"psfex failed with all attempted fwhmmax.\n"
                                            f"stdout:\n------\n{res.stdout.decode('utf-8')}\n"
                                            f"stderr:\n------\n{res.stderr.decode('utf-8')}" )
                            raise RuntimeError( "Repeated failures from psfex call" )
                        SCLogger.warning( f"psfex failed with fwhmmax={fwhmmax}, trying {fwhmmaxtotry[fwhmmaxdex+1]}" )

                psfxml = votable.parse( psfxmlfile )
                psfstats = psfxml.get_table_by_index( 1 )
                psf_sampling = psfstats.array['Sampling_Mean'][0]
                if psf_size is None:
                    usepsfsize = int( np.ceil( psfstats.array['FWHM_FromFluxRadius_Mean'][0] * 5. ) )
                    if usepsfsize % 2 == 0:
                        usepsfsize += 1

            psf = PSF(
                format="psfex", image=image, image_id=image.id,
                fwhm_pixels=float(psfstats.array['FWHM_FromFluxRadius_Mean'][0])
            )
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
        """
        Run source-extraction (using SExtractor) on the given image.

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
        r, flags = sep.flux_radius(data_sub, objects['x'], objects['y'], 6.0 * objects['a'], 0.5, subpix=5)
        r = np.array(r, dtype=[('rhalf', '<f4')])
        objects = rfn.merge_arrays((objects, r), flatten=True)
        sources = SourceList(image=image, data=objects, format='sepnpy')

        return sources

    def extract_sources_filter(self, image):
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
            The image to extract sources from. Must have a score, or a PSF,
            that can be gotten from one of these places (in order):
            - a PSF object loaded on the Image object itself.
            - a zogy_psf attribute attached to the Image object.
            - a PSF object of the new_image attribute of the Image object.
            If the image also has a zogy_alpha attribute, it will be used
            to extract the PSF photometry.

        Returns
        -------
        sources: SourceList
            The list of sources detected in the image.
            This contains a table where each row represents
            one source that was detected, along with all its properties.

        """
        score = image.score.copy()

        if score is None:
            raise NotImplementedError('Still need to add the matched-filter cross correlation! ')

        # psf_image = None
        # if image.psf is not None:
        #     psf_image = image.psf.get_clip()
        # elif getattr(image, 'zogy_psf', None) is not None:
        #     psf_image = image.zogy_psf

        fwhm = image.fwhm_estimate
        if fwhm is None:
            fwhm = image.new_image.fwhm_estimate
        if fwhm is None and image.psf is not None:
            fwhm = image.psf.fwhm_pixels
        if fwhm is None and image.new_image.psf is not None:
            fwhm = image.new_image.psf.fwhm_pixels

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
        coords = image.wcs.wcs.pixel_to_world(x, y)
        ra = [c.ra.value for c in coords]
        dec = [c.dec.value for c in coords]

        label_fluxes = ndimage.sum(image.data, labels, all_idx)  # sum image values where labeled

        # run aperture and iterative PSF photometry
        fluxes = ndimage.labeled_comprehension(image.psfflux, labels, all_idx, np.max, float, np.nan)

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


if __name__ == '__main__':
    from models.base import Session
    from models.provenance import Provenance
    session = Session()
    source_lists = session.scalars(sa.select(SourceList)).all()
    prov = session.scalars(sa.select(Provenance)).all()
