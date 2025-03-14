import random
import pathlib
import subprocess

import numpy as np
from astropy.io import fits, votable

from models.base import FileOnDiskMixin, CODE_ROOT
from util.config import Config
from util.logger import SCLogger


def run_sextractor( imghdr, imagedata, weightdata, maskdata=None, outbase=None,
                    detect_thresh=3.0, analysis_thresh=3.0, apers=[5,], psffile=None,
                    wcs=None, satur_level=50000, seeing_fwhm=1.2, pixel_scale=1.0, gain=1.0,
                    back_type='AUTO', back_value=0., back_size=256, back_filtersize=3,
                    writebg=False, writebgrms=False, writeseg=False, do_not_cleanup=False,
                    mem_objstack=20000, mem_pixstack=1000000, mem_bufsize=4096,
                    timeout=120 ):
    """Extract a SourceList and maybe other things from Sextractor.

    Parameters
    ----------
      imghdr : fits.Header
         Header from the image we're extracting sources from

      imagedata : 2d numpy array
         Image data

      weightdata : 2d numpy array
         Weight data for the image (weight = 1/σ²)

      maskdata : 2d numpy integer array (optional)
         Mask data for the image.  Passes it on to sextractor, which
         will interpet it as it does.  If not passed, a mask will be
         built with all pixels with weight <= 0 having a mask value of
         1, all other pixels a mask value of 0.

      outbase : Path or None
         Where to write output files.  Various extensions will be
         appended; see "Returns" below.  If not passed, a random name in
         the temp directory will be generated.

      detect_thresh : float, default 3.0
         Passed to sextractor DETECT_THRESH.  NOTE : sextractor's
         DETECT_THRESH and ANALYSIS_THRESH are a little weird.  They are
         (very roughly) kinda what you think *if* your psf is normalized
         to 1.0.  Otherwise, you want to divide the value you want by
         the PSFs normalization.  I'm really not sure what you want to
         use when you don't have a PSF; it probably depends on the
         aperture size.  TODO, figure this out; spend some quality time
         with the sextractor manual and/or source code.

      analysis_thresh : float, default 3.0
         Passed to sextractor ANALYSIS_THRESH

      apers : list of float, default [5,]
         Apertur radii in pixels in which to do photometry

      psffile : Path or str, or None
         File that has the psfex PSF to use for PSF photometry.  If
         None, then sextractor won't do PSF photometry.

      wcs : astropy.WCS or None
         WCS to write to the image header before running sextractor.
         Use this if the WCS in the image header isn't exactly right,
         and you want to replace it with this.  This parameter is
         unneeded if you're going to ignore any WORLD parameters
         produced by sextractor.

      satur_level : float, default 50000
         Saturation level in ADU to pass to sextractor SATUR_LEVEL

      seeing_fwhmn : float, default 1.2
         An estimate of the image seeing in arcseconds.  This is used in
         Sextractor's CLASS_STAR determination.  Ideally, it should be
         within 20% of reality, but the first time you run this you will
         probably have no idea.  1.2 is the Sextractor default.

      pixel_scale : float, default 1.0
         Image pixel scale in arcseconds / pixel

      gain : float, default 1.0
         Image gain in e-/adu

      back_type : str, default 'AUTO'
         The background subtraction sextractor should do; must be 'AUTO'
         or 'MANUAL'

      back_value : float, default 0.
         The background level in adu.  Ignored if back_type is not
         'MANUAL'.  Passed to sextractor BACK_VALUE.

      back_size : int, default 256
         The box size in which to do background estimations before
         fitting a smooth background to the estimations within each box.
         Ignored if back_type is not 'AUTO'.  Passed to sextractor
         BACK_SIZE.

      back_filtersize : int, default 3
         The size of the median filter to run over the background values
         found in the back_size boxes.  Ignored if back_type is not
         'AUTO'.  Passed to sextractor BACK_FILTERSIZE.

      writebg : bool, default False
         If true, write out a FITS file with the background sextractor
         subtracted.

      writebgrms : bool, default False
         If true, write out a FITS file with the background noise values
         that sextractor determined.

      writeseg : bool, dfeault False
         If true, write out a FITS file with sextractor's segmentation
         map.  This is map where each pixel holds the object number that
         that pixel on the image is a part of.

      do_not_cleanup : bool, default False
         Leave behind all temporary files that this function and
         sextractor create.  This includes files written for the image,
         weight, and mask.

      mem_objstack : int, default 20000
         Passed to sextractor MEMORY_OBJSTACK

      mem_pixstack : int, default 1000000
         Passed to sextractor MEMORY_PIXSTACK

      mem_bufsize : int, default 4095
         Passed to sextractor MEMORY_BUFSIZE

      timeout : int, default 120
         Tell python's subprocess to give up if sextractor runs for more
         than this many seconds.  The default value is too small for
         crowded filds, expecially if you're doing PSF photometry.

    Returns
    -------
      dict

      Will at least include keys, with values;
         'bkg_mean' : float, The mean background value
         'bkg_sig'  : float, The mean background noise
         'sources'  : Path, the output FITS_LDAC file with sources sextractor found
                      (Suitable for directly passing to SourceList.load )

      There may be additional information.  if do_not_cleanup is true,
      then the dictionary will also include:
         'image'   : Path, the image file sextractor read (holds imghdr and imagedata, with
                     imghdr modified by wcs if given)
         'weight'  : Path, the weight file sextractor read (holds weightdata)
         'mask'    : Path, the mask file sextractor read (holds maskdata)

      If writebg, writebgrms, or writeseg are set, then the dictionary
      will also include, respectively:
         'bkg'     : Path, a FITS file with the background map sextractor subtracted
         'bkgrms'  : Path, a FITS file with the background noise map sextractor determined
         'segmentation' : Path, a FITS file with sextractor's segmentatin map

    """

    if not isinstance( imghdr, fits.Header ):
        raise TypeError( f"Expected image.header to be an astropy.io.fits.Header, but it's a {type(imghdr)}" )
    if ( not isinstance( imagedata, np.ndarray ) ) or ( len( imagedata.shape ) != 2 ):
        raise TypeError( "imagedata must be a 2d numpy array" )
    if ( not isinstance( weightdata, np.ndarray ) ) or ( weightdata.shape != imagedata.shape ):
        raise TypeError( "weightdata must be a 2d numpy array of the same size as imagedata" )
    if maskdata is None:
        maskdata = np.zeros_like( weightdata, dtype=np.int16 )
        maskdata[ weightdata <= 0 ] = 1
    else:
        if ( not isinstance( maskdata, np.ndarray ) ) or ( maskdata.shape != imagedata.shape ):
            raise TypeError( "maskdata must be a 2d numpy array of the same size as imagedata" )
        if not np.issubdtype( maskdata.dtype, np.integer ):
            raise TypeError( "maskdata must be an integer array" )

    if outbase is None:
        outbase = ( pathlib.Path( FileOnDiskMixin.temp_path )
                    / ( ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) ) ) )
    else:
        outbase = pathlib.Path( outbase )
    tmpimage = outbase.parent / f"{outbase.name}.image.fits"
    tmpweight = outbase.parent / f"{outbase.name}.weight.fits"
    tmpflags = outbase.parent / f"{outbase.name}.mask.fits"
    tmpsources = outbase.parent / f"{outbase.name}.sources.fits"
    tmpxml = outbase.parent / f'{outbase.name}.sources.xml'
    tmpbkg = outbase.parent / f"{outbase.name}.bkg.fits"
    tmpbkgrms = outbase.parent / f"{outbase.name}.bkgrms.fits"
    tmpseg = outbase.parent / f"{outbase.name}.seg.fits"
    tmpparams = outbase.parent / f"{outbase.name}.param"

    retval = { 'sources': tmpsources }
    if do_not_cleanup:
        retval['image'] = tmpimage
        retval['weight'] = tmpweight
        retval['mask'] = tmpflags
        retval['votable'] = tmpxml

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

    # Background params
    bgargs = []
    if back_type == 'AUTO':
        bgargs.extend( [ '-BACK_TYPE', 'AUTO',
                         '-BACK_SIZE', str( back_size ),
                         '-BACK_FILTERSIZE', str( back_filtersize ),
                        ] )
    elif back_type == 'MANUAL':
        bgargs.extend( [ '-BACK_TYPE', 'MANUAL',
                         '-BACK_VALUE', str( back_value )
                        ] )
    else:
        raise ValueError( f"Unknown sextractor_back_type {back_type}, "
                          f"must be AUTO or MANUAL" )

    # Check images
    checktypes = ""
    checkimages = ""
    comma = ""
    if writebg:
        checktypes += comma + "BACKGROUND"
        checkimages += comma + str( tmpbkg )
        comma = ","
        retval['bkg'] = tmpbkg
    if writebgrms:
        checktypes += comma + "BACKGROUND_RMS"
        checkimages += comma + str( tmpbkgrms )
        comma = ","
        retval['bkgrms'] = tmpbkgrms
    if writeseg:
        checktypes += comma + "SEGMENTATION"
        checkimages += comma + str( tmpseg )
        comma = ","
        retval['segmentation'] = tmpseg
    if checktypes == "":
        checkimages = [ '-CHECKIMAGE_TYPE', 'NONE' ]
    else:
        checkimages = [ '-CHECKIMAGE_TYPE', checktypes, '-CHECKIMAGE_NAME', checkimages ]


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
        hdr = imghdr.copy()
        if wcs is not None:
            hdr.update( wcs.wcs.to_header() )

        # This likely to be redundant disk I/O, because for many uses of
        #   this function somebody will have *just read* that data from
        #   disk files that we could just point to directly.  The reason
        #   to do this is twofold.  First, flexibility : it allows us to
        #   use this function even if there isn't already a file on
        #   disk.  Second, the whole wcs thing.  We know in actual
        #   operation of SeeChange that the WCS in the image headers
        #   often aren't the ones we want to be using here, so we need
        #   the ability to update the header before passing it to
        #   sextractor.  And, we wouldn't want to update an original
        #   images's header!
        # We could update the interface to this function so that you could
        #   pass either files or data.  If you do that, make sure
        #   to fix everything everywhere that calls this function!
        fits.writeto( tmpimage, imagedata, header=hdr )
        fits.writeto( tmpweight, weightdata )
        fits.writeto( tmpflags, maskdata )

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
        # and give it PIXEL_SCALE that's right.  When this is called in actual
        # use, a later call will have run psfex and should have a good seeing_fwhm
        # estimate.  The only issue is if the first try was good enough to
        # give psfex the right things to fit.

        args = [ "source-extractor",
                 "-CATALOG_NAME", tmpsources,
                 "-CATALOG_TYPE", "FITS_LDAC",
                 "-WRITE_XML", "Y",
                 "-XML_NAME", tmpxml,
                 "-PARAMETERS_NAME", paramfile,
                 "-THRESH_TYPE", "RELATIVE",
                 "-DETECT_THRESH", str( detect_thresh ),
                 "-ANALYSIS_THRESH", str( analysis_thresh ),
                 "-FILTER", "Y",
                 "-FILTER_NAME", str(conv),
                 "-WEIGHT_TYPE", "MAP_WEIGHT",
                 "-RESCALE_WEIGHTS", "N",
                 "-WEIGHT_IMAGE", str(tmpweight),
                 "-WEIGHT_GAIN", "N",
                 "-FLAG_IMAGE", str(tmpflags),
                 "-FLAG_TYPE", "OR",
                 "-PHOT_APERTURES", ",".join( [ str(a*2.) for a in apers ] ),
                 "-SATUR_LEVEL", str( satur_level ),
                 "-GAIN", str( gain ),
                 "-STARNNW_NAME", nnw,
                 "-SEEING_FWHM", str( seeing_fwhm ),
                 "-PIXEL_SCALE", str( pixel_scale ),
                 "-MEMORY_OBJSTACK", str( mem_objstack ),
                 "-MEMORY_PIXSTACK", str( mem_pixstack ),
                 "-MEMORY_BUFSIZE", str( mem_bufsize ),
                ]
        args.extend( bgargs )
        args.extend( psfargs )
        args.extend( checkimages )
        args.append( tmpimage )
        SCLogger.debug( "Running sextractor..." )
        res = subprocess.run(args, cwd=tmpimage.parent, capture_output=True, timeout=timeout)
        SCLogger.debug( f"...sextractor run finished with return code {res.returncode}" )
        if res.returncode != 0:
            SCLogger.error( f"Got return {res.returncode} from sextractor call; stderr:\n{res.stderr}\n"
                            f"-------\nstdout:\n{res.stdout}" )
            raise RuntimeError( "Error return from source-extractor call" )

        # Look for pixel stack overflow warnings in the sextractor output
        err = res.stderr.decode( "utf-8" )
        if "WARNING: Pixel stack overflow" in err:
            raise RuntimeError( "Pixel stack overflow in sextractor" )

        # Get the background from the xml file that sextractor wrote
        sextrstat = votable.parse( tmpxml ).get_table_by_index( 1 )
        retval['bkg_mean'] = sextrstat.array['Background_Mean'][0][0]
        retval['bkg_sig'] = sextrstat.array['Background_StDev'][0][0]

        return retval

    except Exception:
        # If there's an exception, clean up everything, because the calling function won't.
        if not do_not_cleanup:
            SCLogger.error( "Cleaning up sextractor files on Exception" )
            tmpsources.unlink( missing_ok=True )
            tmpbkg.unlink( missing_ok=True )
            tmpbkgrms.unlink( missing_ok=True )
            tmpseg.unlink( missing_ok=True )
        raise

    finally:
        # These are the things that we clean up on either failed or successful execution
        if not do_not_cleanup:
            tmpimage.unlink( missing_ok=True )
            tmpweight.unlink( missing_ok=True )
            tmpflags.unlink( missing_ok=True )
            tmpparams.unlink( missing_ok=True )
            tmpxml.unlink( missing_ok=True )
