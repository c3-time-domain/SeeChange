import pathlib
import random
import time
import subprocess

import numpy as np

import astropy.table
import astropy.wcs.utils

from util import ldac
from util.exceptions import SubprocessFailure, BadMatchException
import improc.scamp


from models.base import FileOnDiskMixin, _logger
from models.provenance import Provenance
from models.image import Image

from pipeline.data_store import DataStore
from pipeline.parameters import Parameters
from pipeline.utils import read_fits_image


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

        self.enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name(self):
        return 'alignment'


class ImageAligner:
    def __init__( self, **kwargs ):
        self.pars = ParsImageAligner( **kwargs )

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
        outimhead = tmppath / f'{tmpname}_warped.image.head'

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

            warpedim = Image( format='fits', upstream_images=[image, target],
                              type='Warped', mjd=target.mjd,
                              end_mjd=image.end_mjd, exp_time=image.exp_time,
                              instrument=image.instrument, telescope=image.telescope,
                              filter=image.filter, section_id=image.section_id,
                              project=image.project, target=image.target,
                              preproc_bitflag=image.preproc_bitflag,
                              astro_cal_done=True, _bitflag=0,  # external scope will set _upstream_bitflag
                              ra=target.ra, dec=target.dec,
                              gallat=target.gallat, gallon=target.gallon,
                              ecllat=target.ecllat, ecllon=target.ecllon,
                              ra_corner_00=target.ra_corner_00, ra_corner_01=target.ra_corner_01,
                              ra_corner_10=target.ra_corner_10, ra_corner_11=target.ra_corner_11,
                              dec_corner_00=target.dec_corner_00, dec_corner_01=target.dec_corner_01,
                              dec_corner_10=target.dec_corner_10, dec_corner_11=target.dec_corner_11
                             )
            warpedim.data, warpedim.raw_header = read_fits_image( outim, output="both" )
            warpedim.weight = read_fits_image( outwt )
            warpedim.flags = np.zeros( warpedim.weight.shape, dtype=np.uint16 )  # Do I want int16 or uint16?
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
            outimhead.unlink( missing_ok=True )

    def run( self, source_image, target_image ):
        """Warp source image so that it is aligned with target image.

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

        # Do the warp

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
        upstream_bitflag = source_image.bitflag
        upstream_bitflag |= target_image.bitflag
        upstream_bitflag |= source_sources.bitflag
        upstream_bitflag |= target_sources.bitflag
        upstream_bitflag |= source_wcs.bitflag
        upstream_bitflag |= target_wcs.bitflag
        upstream_bitflag |= source_zp.bitflag

        warped_image._upstream_bitflag = upstream_bitflag

        return warped_image

