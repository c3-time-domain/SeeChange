import os
import pathlib
import random
import time
import subprocess

import numpy as np

import astropy.table
import astropy.wcs.utils

from util import ldac
from util.exceptions import SubprocessFailure
from util.fits import read_fits_image, save_fits_image_file
from util.logger import SCLogger
from util.exceptions import BadMatchException
import improc.scamp
import improc.tools

from models.base import FileOnDiskMixin
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.enums_and_bitflags import string_to_bitflag, flag_image_bits_inverse

from pipeline.parameters import Parameters
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from improc.bitmask_tools import dilate_bitflag


class ParsImageAligner(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.method = self.add_par(
            'method',
            'swarp',
            str,
            'Alignment method.  Currently only swarp is supported',
            critical=True,
        )

        self.max_arcsec_residual = self.add_par(
            'max_arcsec_residual',
            0.2,
            float,
            'Maximum residual in arcseconds, along both RA and Dec (i.e. not a radial residual),  '
            'for the WCS solution to be considered successful.  Used in SCAMP only. ',
            critical=True,
        )

        self.crossid_radius = self.add_par(
            'crossid_radius',
            2.0,
            float,
            'The radius in arcseconds for the initial SCAMP match (not the final solution).  Used in SCAMP only. ',
            critical=True,
        )

        self.max_sources_to_use = self.add_par(
            'max_sources_to_use',
            2000,
            int,
            'If specified, if the number of objects in sources is larger than this number, '
            'tell SCAMP to only use this many sources for the initial match.  '
            '(This makes the initial match go faster.)  Used in SCAMP only. ',
            critical=True,
        )

        self.min_frac_matched = self.add_par(
            'min_frac_matched',
            0.1,
            float,
            'SCAMP must be able to match at least this fraction of min(number of sources, number of catalog objects) '
            'for the match to be considered good.  Used in SCAMP only. ',
            critical=True,
        )

        self.min_matched = self.add_par(
            'min_matched',
            10,
            int,
            'SCAMP must be able to match at least this many objects for the match to be considered good.  '
            'Used in SCAMP only. ',
            critical=True,
        )

        self.scamp_timeout = self.add_par(
            'scamp_timeout',
            60,
            int,
            'Timeout in seconds for SCAMP to run.  Used in SCAMP only. ',
            critical=False,
        )

        self.swarp_timeout = self.add_par(
            'swarp_timeout',
            60,
            int,
            'Timeout in seconds for SWARP to run.  Used in SWARP only. ',
            critical=False,
        )

        self._enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name(self):
        return 'alignment'


class ImageAligner:
    """Align images.

    NOTE: Aligned images should not be saved to the database!

    If we ever decide we want to do that, we have to deal with the image upstreams properly,
    and indicating what is the alignment target vs. what is the thing that got warped.
    Right now, the database doesn't have the structure for this.

    """
    temp_images = []

    @classmethod
    def cleanup_temp_images( cls ):
        for im in cls.temp_images:
            im.remove_data_from_disk()
            for file in im.get_fullpath(as_list=True):
                if file is not None and os.path.isfile( file ):
                    raise RuntimeError( f'Failed to clean up {file}' )

        cls.temp_images = []

    def __init__( self, **kwargs ):
        self.pars = ParsImageAligner( **kwargs )

    @staticmethod
    def image_source_warped_to_target(image, target):
        """Create a new Image object from the source and target images.

        Most image attributes are from the source image, but the coordinates
        (and corners) are taken from the target image.

        The image type is Warped and the bitflag is 0, with the upstream bitflag
        set to the bitflags of the source and target.

        Parameters
        ----------
        image: Image
            The source image to be warped.
        target: Image
            The target image to which the source image will be warped.

        Returns
        -------
        warpedim: Image
            A new Image object.  NOTE: the image has not yet actually
            been warped!  This method doesn't do the warping, it just
            sets up the structure.  The image data in warpedim is just a
            copy of what was in image.  The ra, dec, and four corner
            attributes are copied from target (to which the image will
            be warped when the image is actually warped).

        """
        warpedim = Image.copy_image(image)
        for att in ['ra', 'dec']:
            setattr(warpedim, att, getattr(target, att))
            for corner in ['00', '01', '10', '11']:
                setattr(warpedim, f'{att}_corner_{corner}', getattr(target, f'{att}_corner_{corner}'))

        warpedim.calculate_coordinates()

        # TODO: are the WorldCoordinates also included? Are they valid for the warped image?
        # --> warpedim should get a copy of target.wcs

        warpedim.type = 'Warped'
        warpedim._set_bitflag( 0 )
        warpedim._upstream_bitflag = 0
        warpedim._upstream_bitflag |= image.bitflag
        warpedim._upstream_bitflag |= target.bitflag

        return warpedim

    def get_swarp_fodder_wcs( self, source_image, source_sources, source_wcs, source_zp, target_sources,
                               fall_back_wcs=None ):
        """Get a WCS for an image-to-image alignment.

        Get a WCS for target_sources that uses source_sources as a
        catalog reference.  If the original WCSes were very good, then
        this new WCS should be very close to the target's WCS we already
        have (somewhere in the database).  However, this WCS should
        usually be better for image alignment, as it provides (as much
        as possible given the WCS framework) a direct transformation
        between the two images, rather than relying on a transformation
        from one image to world, and world to the other image.  (In
        practice, the transformation will always be calculated in that
        two-step manner.  However, from this method, the Gaia (or
        whatever) catalog used to determine image WCSes aren't an
        intermediary.  See the massive comment in _align_swarp.)

        Parameters
        ----------
          source_image : Image
             The image that we will eventually want to align to target_image.

          source_sources : SourceList
             A SourceList from source_image.

          source_wcs : WorldCoordinates
             A WorldCoordinates from source_image.

          source_zp : ZeroPoint
             A ZeroPoint from source_image.

          target_sources: SourceList
             A SourceList from target_image.  target_image isn't
             actually a parameter of this method, because it's not
             needed, but it's the image to which source_image is going
             to be aligned.  If it seems perverse that we're returning a
             new WCS for target_image and not source_image, see the
             massive comment in the _align_swarp method.

          fall_back_wcs : WorldCoordinates or None
             If not None, and the scamp fails (e.g. because the two
             images didn't have enough stars in the region where they
             overlap), and if this WCS is given, just return this WCS.
             Practically speaking, this means that you'll be missing out
             on the benefits described for this method, and assuming
             that the WCS solutions to Gaia are good enough for the
             two-step transformation to provide sub-pixel alignments.

        Returns
        -------
           astropy.wcs.WCS

        """
        tmppath = pathlib.Path( source_image.temp_path )
        tmpname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwzyz', k=10 ) )
        tmpimagecat = tmppath / f'{tmpname}_image.sources.fits'
        tmptargetcat = tmppath / f'{tmpname}_target.sources.fits'

        try:
            # For everything to work as in the massive comment below in
            # _align_swarp, we need the "sources" object to have current
            # ra and dec (X_WORLD and Y_WORLD) fields based on image's
            # current wcs, as this will be serving as the
            # faux-astrometric-reference catalog for scamp.

            imskyco = source_wcs.wcs.pixel_to_world( source_sources.x, source_sources.y )
            # ...the choice of a numpy recarray is inconvenient here, since
            # adding a column requires making a new datatype, copying data, etc.
            # Take the shortcut of using astropy.table.Table.  (Could also use Pandas.)
            datatab = astropy.table.Table( source_sources.data )
            datatab['X_WORLD'] = imskyco.ra.deg
            datatab['Y_WORLD'] = imskyco.dec.deg
            # TODO: the astropy doc says this returns the pixel scale along
            # each axis in the same units as the WCS yields.  Can we assume
            # that the WCS is always yielding degrees?
            pixsc = astropy.wcs.utils.proj_plane_pixel_scales( source_wcs.wcs ).mean()
            datatab['ERRA_WORLD'] = source_sources.errx * pixsc
            datatab['ERRB_WORLD'] = source_sources.erry * pixsc
            flux, dflux = source_sources.apfluxadu()
            datatab['MAG'] = -2.5 * np.log10( flux ) + source_zp.zp
            # TODO: Issue #251
            datatab['MAG'] += source_zp.get_aper_cor( source_sources.aper_rads[0] )
            datatab['MAGERR'] = 1.0857 * dflux / flux

            # Convert from numpy convention to FITS convention and write
            # out LDAC files for scamp to chew on.
            datatab = SourceList._convert_to_sextractor_for_saving( datatab )
            targetdat = astropy.table.Table( SourceList._convert_to_sextractor_for_saving( target_sources.data ) )
            ldac.save_table_as_ldac( datatab, tmpimagecat, imghdr=source_sources.info, overwrite=True )
            ldac.save_table_as_ldac( targetdat, tmptargetcat, imghdr=target_sources.info, overwrite=True )

            # Scamp it up
            try:
                swarp_fodder_wcs = improc.scamp.solve_wcs_scamp(
                    tmptargetcat,
                    tmpimagecat,
                    magkey='MAG',
                    magerrkey='MAGERR',
                    crossid_radius=self.pars.crossid_radius,
                    max_sources_to_use=self.pars.max_sources_to_use,
                    min_frac_matched=self.pars.min_frac_matched,
                    min_matched=self.pars.min_matched,
                    max_arcsec_residual=self.pars.max_arcsec_residual,
                    timeout=self.pars.scamp_timeout,
                )
            except ( BadMatchException, subprocess.TimeoutExpired ) as ex:
                if fall_back_wcs is not None:
                    return fall_back_wcs
                raise ex

            return swarp_fodder_wcs

        finally:
            tmpimagecat.unlink( missing_ok=True )
            tmptargetcat.unlink( missing_ok=True )

    def _align_swarp( self, source_image, source_sources, source_bg, source_psf, source_wcs, source_zp,
                      target_image, target_sources, warped_prov, warped_sources_prov ):
        """Use scamp and swarp to align image to target.

        Parameters
        ---------
          source_image: Image
            The image to be warped.  Must be saved on disk (and perhaps
            to the database?) so that image.get_fullpath() will work.
            Assumes that the weight image will be 0 everywhere flags is
            non-0.  (This is the case for a weight image created by
            pipeline/preprocessing.)

          source_sources: SourceList
            A SourceList from the image.  (RA/DEC values will not be
            used direclty, but recalculated from sourcewcs).  Assumed to
            be in sextrfits format.

          source_bg: Background
            Background for source_image.  It will be subtracted before
            warping. (Is that really what we want to do?)

          source_psf: PSF
            PSF for source_image.

          source_wcs: WorldCoordinates
            wcs for source_image.  This WCS must correspond to the x/y
            and ra/dec values in source_sources.

          source_zp: ZeroPoint
            ZeroPoint for source_image

          target_image: Image
            The image to which source_image will be aligned once
            source_image has been warped.  Profligate; only uses this to
            get a shape, but will load the full target_image data into
            memory (if it's not there already) in so doing.

          target_sources: SourceList
            A SourceList from the other image to which this image should
            be aligned, with "good enough" RA/Dec values.  (Scamp will
            use these for initial matching, but really it's using the
            x/y values here for its solution; see massive comment in the
            bdy of the function.)  Assumed to be in sextrfits format.


          warped_prov: Provenance
            The provenance to assign to the warped image

          warped_sources_prov: Provenance
            The provenance to assign to the sources extracted from the warped image.

        Returns
        -------
          Image, Sources, Background, PSF
            An Image with the warped image data.  image, header, weight, and flags are all populated.

        """
        tmppath = pathlib.Path( source_image.temp_path )
        tmpname = ''.join( random.choices( 'abcdefghijlkmnopqrstuvwxyz', k=10 ) )

        tmpim = tmppath / f'{tmpname}_image.fits'
        tmpwt = tmppath / f'{tmpname}_weight.fits'
        tmpflags = tmppath / f'{tmpname}_flags.fits'
        tmpbg = tmppath / f'{tmpname}_bg.fits'

        outim = tmppath / f'{tmpname}_warped.image.fits'
        outwt = tmppath / f'{tmpname}_warped.weight.fits'
        outfl = tmppath / f'{tmpname}_warped.flags.fits'
        outflwt = tmppath / f'{tmpname}_warped.flags.weight.fits'
        outbg = tmppath / f'{tmpname}_warped.bg.fits'
        outbgwt = tmppath / f'{tmpname}_warped.bg.weight.fits'
        outimhead = tmppath / f'{tmpname}_warped.image.head'
        outflhead = tmppath / f'{tmpname}_warped.flags.head'
        outbghead = tmppath / f'{tmpname}_warped.bg.head'
        swarp_vmem_dir = tmppath /f'{tmpname}_vmem'

        # Writing this all out because several times I've looked at code
        # like this elsewhere and wondered why the heck it was doing what
        # it did, and had to think about it and dig through SWarp
        # documentation to figure out why it was doing what I wanted.
        #
        # SWarp will normally align a bunch of images to the first
        # image, cropping them, and summing them, which is more (and
        # less) than what we want.  In order to get it to align a single
        # image to another single image, we rely on behavior that's
        # mentioned almost as an afterthought in the SWarp manual: "To
        # implement the unusual output features required, one must write
        # a coadd.head ASCII file that contains a custom anisotropic
        # scaling matrix."  Practically speaking, we put the WCS that we
        # want the output image to have into <outim>.head.  SWarp will
        # then warp the input image so that the requested WCS will be
        # the right one for the warped image.
        #
        # We *could* just put the target image's WCS in the coadd.head
        # file, and then swarp the source image.  However, that spatial
        # transformation is the combination of two transformation
        # functions (from the source image to RA/Dec via its WCS, and
        # then from RA/Dec to the target image WCS).  It may be that the
        # Gaia WCSes we use nowadays are good enough for this, but we
        # can do better by making a *direct* transformation between the
        # two images.  To this end, we use the *source image's* source
        # list as the catalog, and use Scamp to calculate the
        # transformation necessary from the target image to the source
        # image.  The resultant WCS is now a WCS for the target image,
        # only it's using the RA/Decs calculated from the pixel
        # positions in the image's source list.  We then tell SWarp to
        # warp the source image so that it has this new WCS, and that
        # will warp the pixels so that they will give all the objects on
        # the source image the same RA/Dec that they have right now,
        # only using the WCS that was derived for the target
        # image... meaning the source image has been aligned with the
        # target image.  There is one more wrinkle: we have to tell
        # SWarp the shape of the output image, since it will default
        # to the shape of the input image.  But, we want it to have
        # the shape of the target image.
        #
        # (This mode of operation is not well-documented in the SWarp
        # manual, which assumes most people want to coadd with cropping,
        # or align to some sort of absolute RA/Dec projection, but it
        # works!)

        try:

            swarp_fodder_wcs = self.get_swarp_fodder_wcs( source_image, source_sources, source_wcs, source_zp,
                                                          target_sources )

            # Write out the .head file that swarp will use to figure out what to do
            hdr = swarp_fodder_wcs.to_header()
            hdr['NAXIS'] = 2
            hdr['NAXIS1'] = target_image.data.shape[1]
            hdr['NAXIS2'] = target_image.data.shape[0]
            hdr.tofile( outimhead )
            hdr.tofile( outflhead )
            hdr.tofile( outbghead )

            # Warp the image
            # TODO : support single image with image, weight, flags in
            #  different HDUs.  Right now, the code below assumes that
            #  image, weight, and flags are each single-HDU FITS files.
            #  (I hope swarp is smart enough that you could do
            #  imagepat[1] to get HDU 1, but I don't know if that's the
            #  case.)
            if source_image.components is None:
                raise NotImplementedError( "Only separate image/weight/flags images currently supported." )
            # impaths = source_image.get_fullpath( as_list=True )
            # imdex = source_image.components.index( 'image' )
            # wtdex = source_image.components.index( 'weight' )
            # fldex = source_image.components.index( 'flags' )

            # For swarp to work right, the header of image must have the
            # WCS we assumed it had when calculating the transformation
            # with scamp above.
            # (TODO: I think I can get away with writing a head file and
            # putting in a symbolic link for the full FITS, instead of
            # copying the FITS data as here.  Look into that.)
            # Also, swarp doesn't seem to be able to handle .fits.fz
            # files, so just to amke sure we can cope, write out the
            # weights to a temp file too.

            hdr = source_image.header.copy()
            improc.tools.strip_wcs_keywords(hdr)
            hdr.update(source_wcs.wcs.to_header())
            data = source_bg.subtract_me( source_image.data )

            save_fits_image_file(tmpim, data, hdr, extname=None, single_file=False)
            save_fits_image_file(tmpwt, source_image.weight, hdr, extname=None, single_file=False)
            save_fits_image_file(tmpflags, source_image.flags, hdr, extname=None, single_file=False)

            swarp_vmem_dir.mkdir( exist_ok=True, parents=True )

            command = [ 'swarp', tmpim,
                        '-IMAGEOUT_NAME', outim,
                        '-WEIGHTOUT_NAME', outwt,
                        '-SUBTRACT_BACK', 'N',
                        '-RESAMPLE_DIR', FileOnDiskMixin.temp_path,
                        '-VMEM_DIR', swarp_vmem_dir,
                        '-WEIGHT_TYPE', 'MAP_WEIGHT',
                        '-WEIGHT_IMAGE', tmpwt,
                        '-RESCALE_WEIGHTS', 'N',
                        '-VMEM_MAX', '1024',
                        '-MEM_MAX', '1024',
                        '-WRITE_XML', 'N' ]

            t0 = time.perf_counter()
            res = subprocess.run(command, capture_output=True, timeout=self.pars.swarp_timeout)
            t1 = time.perf_counter()
            SCLogger.debug( f"swarp of image took {t1-t0:.2f} seconds" )
            if res.returncode != 0:
                raise SubprocessFailure( res )

            # do the same for flags
            command = ['swarp', tmpflags,
                       '-IMAGEOUT_NAME', outfl,
                       '-WEIGHTOUT_NAME', outflwt,
                       '-RESAMPLING_TYPE', 'NEAREST',
                       '-SUBTRACT_BACK', 'N',
                       '-RESAMPLE_DIR', FileOnDiskMixin.temp_path,
                       '-VMEM_DIR', swarp_vmem_dir,
                       '-VMEM_MAX', '1024',
                       '-MEM_MAX', '1024',
                       '-WRITE_XML', 'N']

            t0 = time.perf_counter()
            res = subprocess.run(command, capture_output=True, timeout=self.pars.swarp_timeout)
            t1 = time.perf_counter()
            SCLogger.debug(f"swarp of flags took {t1 - t0:.2f} seconds")
            if res.returncode != 0:
                raise SubprocessFailure(res)

            warpedim = self.image_source_warped_to_target( source_image, target_image )
            warpedim.provenance_id = warped_prov.id

            warpedim.data, warpedim.header = read_fits_image( outim, output="both" )
            # TODO: either make this not a hardcoded header value, or verify
            #  that we've constructed these images to have these hardcoded values
            #  (which would probably be a mistake, since it a priori assumes two amps).
            #  Issue #216
            for att in ['SATURATA', 'SATURATB']:
                if att in source_image.header:
                    warpedim.header[att] = source_image.header[att]

            warpedim.weight = read_fits_image(outwt)
            warpedim.flags = read_fits_image(outfl)
            warpedim.flags = np.rint(warpedim.flags).astype(np.uint16)  # convert back to integers

            warpedim.md5sum = None
            # warpedim.md5sum_components = [ None, None, None ]
            warpedim.md5sum_components = None

            # warp the background noise image:
            # (There is an assumption here that the warped image has the
            #  same noise as the unwarped image.  Correlated pixels make
            #  that fraught.  Also, if the pixel scale isn't the same,
            #  that really won't be true....)
            warpedbg = Background(
                value=0,
                noise=source_bg.noise,
                format=source_bg.format,
                method=source_bg.method,
                _bitflag=source_bg._bitflag,
                sources_id=None,
                image_shape=warpedim.data.shape,
            )
            # TODO: what about polynomial model backgrounds?
            if source_bg.format == 'map':
                save_fits_image_file(tmpbg, source_bg.variance, hdr, extname=None, single_file=False)
                command = ['swarp', tmpbg,
                           '-IMAGEOUT_NAME', outbg,
                           '-WEIGHTOUT_NAME', outbgwt,
                           '-SUBTRACT_BACK', 'N',
                           '-RESAMPLE_DIR', FileOnDiskMixin.temp_path,
                           '-VMEM_DIR', swarp_vmem_dir,
                           '-VMEM_MAX', '1024',
                           '-MEM_MAX', '1024',
                           '-WRITE_XML', 'N']

                t0 = time.perf_counter()
                res = subprocess.run(command, capture_output=True, timeout=self.pars.swarp_timeout)
                t1 = time.perf_counter()
                SCLogger.debug(f"swarp of background took {t1 - t0:.2f} seconds")
                if res.returncode != 0:
                    raise SubprocessFailure(res)

                warpedbg.variance = read_fits_image(outbg, output='data')
                warpedbg.counts = np.zeros_like(warpedbg.variance)
            elif source_bg.format == 'polynomial':
                raise RuntimeError( "polynomial backgrounds not supported" )

            # re-calculate the source list and PSF for the warped image
            source_sources_prov = Provenance.get( source_sources.provenance_id )
            extractor = Detector()
            extractor.pars.override(source_sources_prov.parameters, ignore_addons=True)
            warpedsources, warpedpsf, _, _ = extractor.extract_sources( warpedim, warpedbg )

            prov = Provenance(
                code_version_id=Provenance.get_code_version().id,
                process='extraction',
                parameters=extractor.pars.get_critical_pars(),
                upstreams=[ warped_prov ],
            )
            warpedsources.provenance_id = prov.id
            warpedpsf.sources_id = warpedsources.id
            warpedbg.sources_id = warpedsources.id

            # expand bad pixel mask to allow for warping that smears the badness
            warpedim.flags = dilate_bitflag(warpedim.flags, iterations=1)  # use the default structure

            # warpedim.flags = np.zeros( warpedim.weight.shape, dtype=np.uint16 )  # Do I want int16 or uint16?
            # TODO : a good cutoff for this weight
            #  For most images I've seen, no image
            #  will have a pixel with noise above 100000,
            #  hence the 1e-10.

            oob_bitflag = string_to_bitflag( 'out of bounds', flag_image_bits_inverse)
            warpedim.flags[ np.logical_and(warpedim.flags == 0, warpedim.weight < 1e-10)] = oob_bitflag

            # Try to save some memory by getting rid of big stuff that got automatically loaded.
            # (It's possible that whoever called this will be all annoyed as they have to reload it,
            # but it's more likely that they will keep the objects around without looking at the data
            # and it will just be wasted memory.  If you think about something like a coadd, that could
            # get significant.)
            source_image.data = None
            source_image.weight = None
            source_image.flags = None
            source_sources.data = None
            source_bg.counts = None
            source_bg.variance = None

            return warpedim, warpedsources, warpedbg, warpedpsf

        finally:
            tmpim.unlink( missing_ok=True )
            tmpwt.unlink( missing_ok=True )
            tmpflags.unlink( missing_ok=True )
            tmpbg.unlink( missing_ok=True )
            outim.unlink( missing_ok=True )
            outwt.unlink( missing_ok=True )
            outfl.unlink( missing_ok=True )
            outflwt.unlink( missing_ok=True )
            outbg.unlink( missing_ok=True )
            outbgwt.unlink( missing_ok=True )
            outimhead.unlink( missing_ok=True )
            outflhead.unlink( missing_ok=True )
            outbghead.unlink( missing_ok=True )
            if swarp_vmem_dir.is_dir():
                for f in swarp_vmem_dir.iterdir():
                    f.unlink()
                swarp_vmem_dir.rmdir()


    def get_provenances( self, upstrprovs, source_sources_prov ):
        """Get provenances for warped images and data products

        This is a little screwed up, because we don't get the parameters
        right.  But, also, warped images aren't supposed to be saved to
        the database, so that shouldn't matter.  We probably ought to
        think about setting these provenances at all.

        """

        code_version = Provenance.get_code_version()
        warped_prov = Provenance( code_version_id=code_version.id,
                                  process='alignment',
                                  parameters=self.pars.get_critical_pars(),
                                  upstreams=upstrprovs
                                 )
        tmp_extractor = Detector()
        tmp_extractor.pars.override( source_sources_prov.parameters, ignore_addons=True )
        warped_sources_prov = Provenance( code_version_id=code_version.id,
                                          process='extraction',
                                          parameters=tmp_extractor.pars.get_critical_pars(),
                                          upstreams=[ warped_prov ]
                                         )
        tmp_astrometor = AstroCalibrator()
        warped_wcs_prov = Provenance( code_version_id=code_version.id,
                                      process='astrocal',
                                      parameters=tmp_astrometor.pars.get_critical_pars(),
                                      upstreams=[ warped_sources_prov ]
                                     )
        tmp_photometor = PhotCalibrator()
        warped_zp_prov = Provenance( code_version_id=code_version.id,
                                     process='photocal',
                                     paramters=tmp_photometor.pars.get_critical_pars(),
                                     upstreams=[ warped_wcs_prov ]
                                    )

        return warped_prov, warped_sources_prov, warped_wcs_prov, warped_zp_prov

    # TODO : pass a DataStore for source and target instead of all these parameters
    def run( self,
             source_image, source_sources, source_bg, source_psf, source_wcs, source_zp,
             target_image, target_sources ):
        """Warp source image so that it is aligned with target image.

        If the source_image and target_image are the same, will just create
        a copy of the same image data in a new Image object.

        Parameters
        ----------
          source_image: Image
             An Image that will get warped.

          source_sources: SourceList
            correponding to source_image

          source_bg: Background
            corresponding to source_sources

          source_psf: PSF
            corresponding to source_sources

          source_wcs: WorldCoordinates
            correponding to source_sources

          source_zp: ZeroPoint
            correponding to source_sources

          target_image: Image
             An image to which the source_image will be aligned.

          target_sources: SourceList
             corresponding to target_image

        Returns
        -------
          Image, Sources, Background, PSF
            Versions of all of these, warped from source to target

            There are some implicit assumptions that these will never
            get saved to the database.

        """
        SCLogger.debug( f"ImageAligner.run: aligning image {source_image.id} ({source_image.filepath}) "
                        f"to {target_image.id} ({target_image.filepath})" )

        upstrprovs = Provenance.get_batch( [ source_image.provenance_id, source_sources.provenance_id,
                                             target_image.provenance_id, target_sources.provenance_id ] )
        source_sources_prov = Provenance.get( source_sources.provenance_id )
        ( warped_prov, warped_sources_prov,
          _warped_wcs_prov, _warped_zp_prov ) = self.get_provenances( upstrprovs, source_sources_prov )

        if target_image == source_image:
            SCLogger.debug( "...target and source are the same, not warping " )
            warped_image = Image.copy_image( source_image )
            warped_image.type = 'Warped'
            warped_image.data = source_bg.subtract_me( source_image.data )
            if ( warped_image.weight is None or warped_image.flags is None ):
                raise RuntimeError( "ImageAligner.run: source image weight and flags missing!  I can't cope!" )
            warped_image.filepath = None
            warped_image.md5sum = None
            warped_image.md5sum_components = None             # Which is the right thing to do with extensions???
            # warped_image.md5sum_components = [ None, None, None ]

            warped_sources = source_sources.copy()
            warped_sources.provenance_id = warped_sources_prov.id
            warped_sources.image_id = warped_image.id
            warped_sources.data = source_sources.data
            warped_sources.info = source_sources.info
            warped_sources.filepath = None
            warped_sources.md5sum = None

            warped_bg = Background(
                format = source_bg.format,
                method = source_bg.method,
                value = 0,                  # since we subtracted above
                noise = source_bg.noise,
                sources_id = warped_sources.id,
                image_shape = warped_image.data.shape,
                filepath = None,
            )
            if warped_bg.format == 'map':
                warped_bg.counts = np.zeros_like( source_bg.counts )
                warped_bg.variance = source_bg.variance              # note: is a reference, not a copy...

            warped_psf = source_psf.copy()
            warped_psf.data = source_psf.data
            warped_psf.info = source_psf.info
            warped_psf.header = source_psf.header
            warped_psf.sources_id = warped_sources.id
            warped_psf.filepath = None
            warped_psf.md5sum = None

            # warped_wcs = source_wcs.copy()
            # warped_wcs.sources_id = warped_sources.id
            # warped_wcs.filepath = None
            # warped_wcs.md5sum = None
            # warped_wcs.provenance_id = warped_wcs_prov.id

            # warped_zp = source_zp.copy()
            # warped_zp.sources_id = warped_sources.id
            # warped_zp.provenance_id = warped_zp_prov.id

        else:  # Do the warp
            if self.pars.method == 'swarp':
                SCLogger.debug( '...aligning with swarp' )
                if ( source_sources.format != 'sextrfits' ) or ( target_sources.format != 'sextrfits' ):
                    raise RuntimeError( 'swarp ImageAligner requires sextrfits sources' )
                ( warped_image, warped_sources,
                  warped_bg, warped_psf ) = self._align_swarp( source_image,
                                                               source_sources,
                                                               source_bg,
                                                               source_psf,
                                                               source_wcs,
                                                               source_zp,
                                                               target_image,
                                                               target_sources,
                                                               warped_prov,
                                                               warped_sources_prov )
            else:
                raise ValueError( f'alignment method {self.pars.method} is unknown' )

        # Right now we don't save any warped images to the database, so being
        #  careful about provenances probably isn't necessary.  (I'm not sure
        #  we're being careful enough....)
        warped_image.provenance_id = warped_prov.id
        warped_image.info['original_image_id'] = source_image.id
        warped_image.info['original_image_filepath'] = source_image.filepath  # verification of aligned images
        warped_image.info['alignment_parameters'] = self.pars.get_critical_pars()

        upstream_bitflag = source_image.bitflag
        upstream_bitflag |= target_image.bitflag
        upstream_bitflag |= source_sources.bitflag
        upstream_bitflag |= target_sources.bitflag
        upstream_bitflag |= source_wcs.bitflag
        upstream_bitflag |= source_zp.bitflag

        warped_image._upstream_bitflag = upstream_bitflag
        # TODO, upstream_bitflags should updated for
        #   other things too!!!!!  (For instance, target wcs, since if
        #   that's bad, the alignment will be bad.)  (This is one of
        #   several things that motivates the note
        #   in the docstring about assuming things
        #   aren't saved to the database.)

        return warped_image, warped_sources, warped_bg, warped_psf
