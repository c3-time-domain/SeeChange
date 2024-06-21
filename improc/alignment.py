import os
import pathlib
import random
import time
import subprocess
import warnings

import numpy as np

import astropy.table
import astropy.wcs.utils

from util import ldac
from util.exceptions import SubprocessFailure
from util.util import read_fits_image, save_fits_image_file
from util.logger import SCLogger
import improc.scamp
import improc.tools

from models.base import FileOnDiskMixin
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.enums_and_bitflags import string_to_bitflag, flag_image_bits_inverse

from pipeline.data_store import DataStore
from pipeline.parameters import Parameters
from pipeline.detection import Detector
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

        self.to_index = self.add_par(
            'to_index',
            'last',
            str,
            'How to choose the index of image to align to. Can choose "first" or "last" (default). ',
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

        self.enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name(self):
        return 'alignment'


class ImageAligner:
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
        warpedim.zp = image.zp  # zp not available when loading from DB (zp.image_id doesn't point to warpedim)
        # TODO: are the WorldCoordinates also included? Are they valid for the warped image?
        # --> warpedim should get a copy of target.wcs

        warpedim.type = 'Warped'
        warpedim.bitflag = 0
        warpedim._upstream_bitflag = 0
        warpedim._upstream_bitflag |= image.bitflag
        warpedim._upstream_bitflag |= target.bitflag

        return warpedim

    def _align_swarp( self, image, target, sources, target_sources ):
        """Use scamp and swarp to align image to target.

        Parameters
        ---------
          image: Image
            The image to be warped.  Must be saved on disk (and perhaps
            to the database?) so that image.get_fullpath() will work.
            Assumes that the weight image will be 0 everywhere flags is
            non-0.  (This is the case for a weight image created by
            pipeline/preprocessing.)

          target: Image
            The target image we're aligning with.

          sources: SourceList
            A SourceList from the image, with good RA/Dec values.
            Assumed to be in sextrfits format.

          target_sources: SourceList
            A SourceList from the other image to which this image should
            be aligned, with good RA/Dec values.  Assumed to be in
            sextrfits format.

        Returns
        -------
          Image
            An Image with the warped image data.  image, header, weight, and flags are all populated.

        """
        tmppath = pathlib.Path( image.temp_path )
        tmpname = ''.join( random.choices( 'abcdefghijlkmnopqrstuvwxyz', k=10 ) )

        tmpimagecat = tmppath / f'{tmpname}_image.sources.fits'
        tmptargetcat = tmppath / f'{tmpname}_target.sources.fits'
        tmpim = tmppath / f'{tmpname}_image.fits'
        tmpflags = tmppath / f'{tmpname}_flags.fits'
        tmpbg = tmppath / f'{tmpname}_bg.fits'

        outim = tmppath / f'{tmpname}_warped.image.fits'
        outwt = tmppath / f'{tmpname}_warped.weight.fits'
        outfl = tmppath / f'{tmpname}_warped.flags.fits'
        outbg = tmppath / f'{tmpname}_warped.bg.fits'
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
            # For everything to work as described above, we need the
            # "sources" object to have current ra and dec (X_WORLD and
            # Y_WORLD) fields based on image's current wcs, as this will
            # be serving as the faux-astrometric-reference catalog for
            # scamp.

            imagewcs = image.wcs
            imskyco = imagewcs.wcs.pixel_to_world( sources.x, sources.y )
            # ...the choice of a numpy recarray is inconvenient here, since
            # adding a column requires making a new datatype, copying data, etc.
            # Take the shortcut of using astropy.table.Table.  (Could also use Pandas.)
            datatab = astropy.table.Table( sources.data )
            datatab['X_WORLD'] = imskyco.ra.deg
            datatab['Y_WORLD'] = imskyco.dec.deg
            # TODO: the astropy doc says this returns the pixel scale along
            # each axis in the same units as the WCS yields.  Can we assume
            # that the WCS is always yielding degrees?
            pixsc = astropy.wcs.utils.proj_plane_pixel_scales( imagewcs.wcs ).mean()
            datatab['ERRA_WORLD'] = sources.errx * pixsc
            datatab['ERRB_WORLD'] = sources.erry * pixsc
            flux, dflux = sources.apfluxadu()
            datatab['MAG'] = -2.5 * np.log10( flux ) + image.zp.zp
            # TODO: Issue #251
            datatab['MAG'] += image.zp.get_aper_cor( sources.aper_rads[0] )
            datatab['MAGERR'] = 1.0857 * dflux / flux

            # Convert from numpy convention to FITS convention and write
            # out LDAC files for scamp to chew on.
            datatab = SourceList._convert_to_sextractor_for_saving( datatab )
            targetdat = astropy.table.Table( SourceList._convert_to_sextractor_for_saving( target_sources.data ) )
            ldac.save_table_as_ldac( datatab, tmpimagecat, imghdr=sources.info, overwrite=True )
            ldac.save_table_as_ldac( targetdat, tmptargetcat, imghdr=target_sources.info, overwrite=True )

            # Scamp it up
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
                timeout=self.pars.swarp_timeout,
            )

            # Write out the .head file that swarp will use to figure out what to do
            hdr = swarp_fodder_wcs.to_header()
            hdr['NAXIS'] = 2
            hdr['NAXIS1'] = target.data.shape[1]
            hdr['NAXIS2'] = target.data.shape[0]
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
            if image.filepath_extensions is None:
                raise NotImplementedError( "Only separate image/weight/flags images currently supported." )
            impaths = image.get_fullpath( as_list=True )
            imdex = image.filepath_extensions.index( '.image.fits' )
            wtdex = image.filepath_extensions.index( '.weight.fits' )
            fldex = image.filepath_extensions.index( '.flags.fits' )

            # For swarp to work right, the header of image must have the
            # WCS we assumed it had when calculating the transformation
            # with scamp above.
            # (TODO: I think I can get away with writing a head file and
            # putting in a symbolic link for the full FITS, instead of
            # copying the FITS data as here.  Look into that.)

            hdr = image.header.copy()
            improc.tools.strip_wcs_keywords(hdr)
            hdr.update(imagewcs.wcs.to_header())
            if image.bg is None:
                # to avoid this warning, consider adding a "zero" background object to the image
                warnings.warn("No background image found. Using original image data.")
                data = image.data
            else:
                data = image.data_bgsub

            save_fits_image_file(tmpim, data, hdr, extname=None, single_file=False)
            save_fits_image_file(tmpflags, image.flags, hdr, extname=None, single_file=False)

            swarp_vmem_dir.mkdir( exist_ok=True, parents=True )

            command = [ 'swarp', tmpim,
                        '-IMAGEOUT_NAME', outim,
                        '-WEIGHTOUT_NAME', outwt,
                        '-SUBTRACT_BACK', 'N',
                        '-RESAMPLE_DIR', FileOnDiskMixin.temp_path,
                        '-VMEM_DIR', swarp_vmem_dir,
                        # '-VMEM_DIR', '/tmp',
                        '-WEIGHT_TYPE', 'MAP_WEIGHT',
                        '-WEIGHT_IMAGE', impaths[wtdex],
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
                       '-RESAMPLING_TYPE', 'NEAREST',
                       '-SUBTRACT_BACK', 'N',
                       '-RESAMPLE_DIR', FileOnDiskMixin.temp_path,
                       '-VMEM_DIR', swarp_vmem_dir,
                       # '-VMEM_DIR', '/tmp',
                       '-VMEM_MAX', '1024',
                       '-MEM_MAX', '1024',
                       '-WRITE_XML', 'N']

            t0 = time.perf_counter()
            res = subprocess.run(command, capture_output=True, timeout=self.pars.swarp_timeout)
            t1 = time.perf_counter()
            SCLogger.debug(f"swarp of flags took {t1 - t0:.2f} seconds")
            if res.returncode != 0:
                raise SubprocessFailure(res)

            warpedim = self.image_source_warped_to_target(image, target)

            warpedim.data, warpedim.header = read_fits_image( outim, output="both" )
            # TODO: either make this not a hardcoded header value, or verify
            #  that we've constructed these images to have these hardcoded values
            #  (which would probably be a mistake, since it a priori assumes two amps).
            #  Issue #216
            for att in ['SATURATA', 'SATURATB']:
                if att in image.header:
                    warpedim.header[att] = image.header[att]

            warpedim.weight = read_fits_image(outwt)
            warpedim.flags = read_fits_image(outfl)
            warpedim.flags = np.rint(warpedim.flags).astype(np.uint16)  # convert back to integers

            # warp the background noise image:
            if image.bg is not None:
                bg = Background(
                    value=0,
                    noise=image.bg.noise,
                    format=image.bg.format,
                    method=image.bg.method,
                    _bitflag=image.bg._bitflag,
                    image=warpedim,
                    provenance=image.bg.provenance,
                    provenance_id=image.bg.provenance_id,
                )
                # TODO: what about polynomial model backgrounds?
                if image.bg.format == 'map':
                    save_fits_image_file(tmpbg, image.bg.variance, hdr, extname=None, single_file=False)
                    command = ['swarp', tmpbg,
                               '-IMAGEOUT_NAME', outbg,
                               '-SUBTRACT_BACK', 'N',
                               '-RESAMPLE_DIR', FileOnDiskMixin.temp_path,
                               '-VMEM_DIR', swarp_vmem_dir,
                               # '-VMEM_DIR', '/tmp',
                               '-VMEM_MAX', '1024',
                               '-MEM_MAX', '1024',
                               '-WRITE_XML', 'N']

                    t0 = time.perf_counter()
                    res = subprocess.run(command, capture_output=True, timeout=self.pars.swarp_timeout)
                    t1 = time.perf_counter()
                    SCLogger.debug(f"swarp of background took {t1 - t0:.2f} seconds")
                    if res.returncode != 0:
                        raise SubprocessFailure(res)

                    bg.variance = read_fits_image(outbg, output='data')
                    bg.counts = np.zeros_like(bg.variance)

                warpedim.bg = bg

            # re-calculate the source list and PSF for the warped image
            extractor = Detector()
            extractor.pars.override(sources.provenance.parameters['sources'], ignore_addons=True)
            warpedsrc, warpedpsf, _, _ = extractor.extract_sources(warpedim)
            warpedim.sources = warpedsrc
            warpedim.psf = warpedpsf

            prov = Provenance(
                code_version=image.provenance.code_version,
                process='extraction',
                parameters=extractor.pars.get_critical_pars(),
                upstreams=[image.provenance],
            )
            warpedim.sources.provenance = prov
            warpedim.sources.provenance_id = prov.id
            warpedim.psf.provenance = prov
            warpedim.psf.provenance_id = prov.id

            # expand bad pixel mask to allow for warping that smears the badness
            warpedim.flags = dilate_bitflag(warpedim.flags, iterations=1)  # use the default structure

            # warpedim.flags = np.zeros( warpedim.weight.shape, dtype=np.uint16 )  # Do I want int16 or uint16?
            # TODO : a good cutoff for this weight
            #  For most images I've seen, no image
            #  will have a pixel with noise above 100000,
            #  hence the 1e-10.

            oob_bitflag = string_to_bitflag( 'out of bounds', flag_image_bits_inverse)
            warpedim.flags[ np.logical_and(warpedim.flags == 0, warpedim.weight < 1e-10)] = oob_bitflag

            return warpedim

        finally:
            tmpimagecat.unlink( missing_ok=True )
            tmptargetcat.unlink( missing_ok=True )
            tmpim.unlink( missing_ok=True )
            tmpflags.unlink( missing_ok=True )
            tmpbg.unlink( missing_ok=True )
            outim.unlink( missing_ok=True )
            outwt.unlink( missing_ok=True )
            outfl.unlink( missing_ok=True )
            outbg.unlink( missing_ok=True )
            outimhead.unlink( missing_ok=True )
            outflhead.unlink( missing_ok=True )
            outbghead.unlink( missing_ok=True )
            for f in swarp_vmem_dir.iterdir():
                f.unlink()
            swarp_vmem_dir.rmdir()

    def run( self, source_image, target_image ):
        """Warp source image so that it is aligned with target image.

        If the source_image and target_image are the same, will just create
        a copy of the same image data in a new Image object.

        Parameters
        ----------
          source_image: Image
             An Image that will get warped.  Image must have
             already been through astrometric and photometric calibration.
             Will use the sources, wcs, and zp attributes attached to
             the Image object.

          target_image: Image
             An image to which the source_image will be aligned.
             Will use the sources and wcs fields attributes attached to
             the Image object.

        Returns
        -------
          DataStore
            A new DataStore (that is not either of the input DataStores)
            whose image field holds the aligned image.  Extraction, etc.
            has not been run.

        """
        # Make sure we have what we need
        source_sources = source_image.sources
        if source_sources is None:
            raise RuntimeError( f'Image {source_image.id} has no sources' )
        source_wcs = source_image.wcs
        if source_wcs is None:
            raise RuntimeError( f'Image {source_image.id} has no wcs' )
        source_zp = source_image.zp
        if source_zp is None:
            raise RuntimeError( f'Image {source_image.id} has no zp' )

        target_sources = target_image.sources
        if target_sources is None:
            raise RuntimeError( f'Image {target_image.id} has no sources' )
        target_wcs = target_image.wcs
        if target_wcs is None:
            raise RuntimeError( f'Image {target_image.id} has no wcs' )

        if target_image == source_image:
            warped_image = Image.copy_image( source_image )
            warped_image.type = 'Warped'
            if source_image.bg is None:
                warnings.warn("No background image found. Using original image data.")
                warped_image.data = source_image.data
                warped_image.bg = None  # this will be a problem later if you need to coadd the images!
            else:
                warped_image.data = source_image.data_bgsub
                # make a copy of the background object but with zero mean
                bg = Background(
                    value=0,
                    noise=source_image.bg.noise,
                    format=source_image.bg.format,
                    method=source_image.bg.method,
                    _bitflag=source_image.bg._bitflag,
                    image=warped_image,
                    provenance=source_image.bg.provenance,
                    provenance_id=source_image.bg.provenance_id,
                )
                if bg.format == 'map':
                    bg.counts = np.zeros_like(warped_image.data)
                    bg.variance = source_image.bg.variance
                warped_image.bg = bg

            warped_image.psf = source_image.psf
            warped_image.zp = source_image.zp
            warped_image.wcs = source_image.wcs
            # TODO: what about SourceList?
            # TODO: should these objects be copies of the products, or references to the same objects?
        else:  # Do the warp
            if self.pars.method == 'swarp':
                SCLogger.debug( 'Aligning with swarp' )
                if ( source_sources.format != 'sextrfits' ) or ( target_sources.format != 'sextrfits' ):
                    raise RuntimeError( f'swarp ImageAligner requires sextrfits sources' )
                warped_image = self._align_swarp(source_image, target_image, source_sources, target_sources)
            else:
                raise ValueError( f'alignment method {self.pars.method} is unknown' )

        warped_image.provenance = Provenance(
            code_version=source_image.provenance.code_version,
            process='alignment',
            parameters=self.pars.get_critical_pars(),
            upstreams=[
                source_image.provenance,
                source_sources.provenance,
                source_wcs.provenance,
                source_zp.provenance,
                target_image.provenance,
                target_sources.provenance,
                target_wcs.provenance,
            ],  # this does not really matter since we are not going to save this to DB!
        )
        warped_image.provenance_id = warped_image.provenance.id  # make sure this is filled even if not saved to DB
        warped_image.info['original_image_id'] = source_image.id
        warped_image.info['original_image_filepath'] = source_image.filepath  # verification of aligned images
        warped_image.info['alignment_parameters'] = self.pars.get_critical_pars()

        upstream_bitflag = source_image.bitflag
        upstream_bitflag |= target_image.bitflag
        upstream_bitflag |= source_sources.bitflag
        upstream_bitflag |= target_sources.bitflag
        upstream_bitflag |= source_wcs.bitflag
        upstream_bitflag |= target_wcs.bitflag
        upstream_bitflag |= source_zp.bitflag

        warped_image._upstream_bitflag = upstream_bitflag

        return warped_image

