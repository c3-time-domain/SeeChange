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
import improc.scamp


from models.base import FileOnDiskMixin, _logger
from models.provenance import Provenance
from models.image import Image

from pipeline.data_store import DataStore
from pipeline.parameters import Parameters
from pipeline.utils import read_fits_image
from improc.bitmask_tools import dilate_bitmask


class ParsImageAligner(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.method = self.add_par(
            'method',
            'swarp',
            str,
            'Alignment method.  Currently only swarp is supported',
            critical=True
        )

        self.to_index = self.add_par(
            'to_index',
            'last',
            str,
            'How to choose the index of image to align to. Can choose "first" or "last" (default). ',
            critical=True
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
    def image_from_source_and_target(image, target):
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
            A new Image object with the warped image data.
        """
        warpedim = Image.copy_image(image)
        for att in ['ra', 'dec']:
            setattr(warpedim, att, getattr(target, att))
            for corner in ['00', '01', '10', '11']:
                setattr(warpedim, f'{att}_corner_{corner}', getattr(target, f'{att}_corner_{corner}'))

        warpedim.calculate_coordinates()
        warpedim.psf = image.psf  # psf not available when loading from DB (psf.image_id doesn't point to warpedim)

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
            A SourceList from the image, with good RA/Dec values

          target_sources: SourceList
            A SourceList from the other image to which this image should
            be aligned, with good RA/Dec values

        Returns
        -------
          Image
            An image with the warped image.  image, raw_header, weight, and flags are all populated.

        """

        tmppath = pathlib.Path( image.temp_path )
        tmpname = ''.join( random.choices( 'abcdefghijlkmnopqrstuvwxyz', k=10 ) )
        imagecat = tmppath / f'{tmpname}_image.sources.fits'
        targetcat = tmppath / f'{tmpname}_target.sources.fits'
        outim = tmppath / f'{tmpname}_warped.image.fits'
        outwt = tmppath / f'{tmpname}_warped.weight.fits'
        outfl = tmppath / f'{tmpname}_warped.flags.fits'
        outimhead = tmppath / f'{tmpname}_warped.image.head'
        outflhead = tmppath / f'{tmpname}_warped.flags.head'

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
            # Write out the source lists for scamp to chew on
            sourcedat = sources._convert_to_sextractor_for_saving( sources.data )
            targetdat = target_sources._convert_to_sextractor_for_saving( target_sources.data )
            ldac.save_table_as_ldac( astropy.table.Table( sourcedat), imagecat,
                                     imghdr=sources.info, overwrite=True )
            ldac.save_table_as_ldac( astropy.table.Table( targetdat ), targetcat,
                                     imghdr=target_sources.info, overwrite=True )

            # Scamp it up
            wcs = improc.scamp._solve_wcs_scamp( targetcat, imagecat, magkey='MAG', magerrkey='MAGERR' )

            # Write out the .head file that swarp will use to figure out what to do
            hdr = wcs.to_header()
            hdr['NAXIS'] = 2
            hdr['NAXIS1'] = target.data.shape[1]
            hdr['NAXIS2'] = target.data.shape[0]
            hdr.tofile( outimhead )
            hdr.tofile( outflhead )
            # Warp the image
            # TODO : support single image.  (I hope swarp is smart
            #  enough that you could do imagepat[1] to get HDU 1, but
            #  I don't know if that's the case.)
            if image.filepath_extensions is None:
                raise NotImplementedError( "Only separate image/weight/flags images currently supported." )
            impaths = image.get_fullpath( as_list=True )
            imdex = image.filepath_extensions.index( '.image.fits' )
            wtdex = image.filepath_extensions.index( '.weight.fits' )
            fldex = image.filepath_extensions.index( '.flags.fits' )

            command = [ 'swarp', impaths[imdex],
                        '-IMAGEOUT_NAME', outim,
                        '-WEIGHTOUT_NAME', outwt,
                        '-SUBTRACT_BACK', 'N',
                        '-RESAMPLE_DIR', FileOnDiskMixin.temp_path,
                        '-VMEM_DIR', FileOnDiskMixin.temp_path,
                        '-MAP_TYPE', 'MAP_WEIGHT',
                        '-WEIGHT_IMAGE', impaths[wtdex],
                        '-RESCALE_WEIGHTS', 'N',
                        '-VMEM_MAX', '16384',
                        '-MEM_MAX', '1024',
                        '-WRITE_XML', 'N' ]

            t0 = time.perf_counter()
            res = subprocess.run( command, capture_output=True )
            t1 = time.perf_counter()
            _logger.debug( f"swarp took {t1-t0:.2f} seconds" )
            if res.returncode != 0:
                raise SubprocessFailure( res )

            # do the same for flags
            command = ['swarp', impaths[fldex],
                       '-IMAGEOUT_NAME', outfl,
                       '-RESAMPLING_TYPE', 'NEAREST',
                       '-SUBTRACT_BACK', 'N',
                       '-RESAMPLE_DIR', FileOnDiskMixin.temp_path,
                       '-VMEM_DIR', FileOnDiskMixin.temp_path,
                       '-VMEM_MAX', '16384',
                       '-MEM_MAX', '1024',
                       '-WRITE_XML', 'N']

            t0 = time.perf_counter()
            res = subprocess.run(command, capture_output=True)
            t1 = time.perf_counter()
            _logger.debug(f"swarp took {t1 - t0:.2f} seconds")
            if res.returncode != 0:
                raise SubprocessFailure(res)

            warpedim = self.image_from_source_and_target(image, target)

            warpedim.data, warpedim.raw_header = read_fits_image( outim, output="both" )
            warpedim.weight = read_fits_image(outwt)
            warpedim.flags = read_fits_image(outfl)
            warpedim.flags = np.rint(warpedim.flags).astype(np.uint16)  # convert back to integers

            # expand bad pixel mask to allow for warping that smears the badness
            warpedim.flags = dilate_bitmask(warpedim.flags, iterations=1)  # use the default structure

            # warpedim.flags = np.zeros( warpedim.weight.shape, dtype=np.uint16 )  # Do I want int16 or uint16?
            # TODO : a good cutoff for this weight
            #  For most images I've seen, no image
            #  will have a pixel with noise above 100000,
            #  hence the 1e-10.
            warpedim.flags[ warpedim.weight < 1e-10 ] = 1

            return warpedim

        finally:
            imagecat.unlink( missing_ok=True )
            targetcat.unlink( missing_ok=True )
            outim.unlink( missing_ok=True )
            outwt.unlink( missing_ok=True )
            outfl.unlink( missing_ok=True )
            outimhead.unlink( missing_ok=True )
            outflhead.unlink( missing_ok=True )

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
            warped_image.psf = source_image.psf
        else:  # Do the warp

            if self.pars.method == 'swarp':
                if ( source_sources.format != 'sextrfits' ) or ( target_sources.format != 'sextrfits' ):
                    raise RuntimeError( f'swarp ImageAligner requires sextrfits sources' )

                # We need good RA/Dec values, and a magnitude, in the object list
                # that's serving as the catalog to scamp
                imskyco = source_wcs.wcs.pixel_to_world( source_sources.x, source_sources.y )
                # ...the choice of a numpy recarray is inconvenient here, since
                # adding a column requires making a new datatype, copying data, etc.
                # Take the shortcut of using astropy.Table.  (Could also use Pandas.)
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
                datatab['MAG'] += source_zp.get_aper_cor( source_sources.aper_rads[0] )
                datatab['MAGERR'] = 1.0857 * dflux / flux
                source_sources.data = datatab.as_array()

                # targskyco = target_wcs.wcs.pixel_to_world( target_sources.x, target_sources.y )
                # datatab = astropy.table.Table( target_sources.data )
                # datatab['X_WORLD'] = targskyco.ra.deg
                # datatab['Y_WORLD'] = targskyco.dec.deg
                # pixsc = astropy.wcs.utils.proj_plane_pixel_scales( target_wcs.wcs ).mean()
                # datatab['ERRA_IMAGE'] = source_sources.
                # target_sources.data = datatab.as_array()

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
        warped_image.provenance.update_id()

        upstream_bitflag = source_image.bitflag
        upstream_bitflag |= target_image.bitflag
        upstream_bitflag |= source_sources.bitflag
        upstream_bitflag |= target_sources.bitflag
        upstream_bitflag |= source_wcs.bitflag
        upstream_bitflag |= target_wcs.bitflag
        upstream_bitflag |= source_zp.bitflag

        warped_image._upstream_bitflag = upstream_bitflag

        return warped_image

