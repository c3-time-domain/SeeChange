import io
import pathlib
import random
import subprocess

import numpy as np
import numpy.lib.recfunctions as rfn
import sqlalchemy as sa

import sep

from astropy.io import fits, votable

from util.config import Config

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.base import SmartSession, FileOnDiskMixin, CODE_ROOT, _logger
from models.image import Image
from models.psf import PSF
from models.source_list import SourceList


class ParsDetector(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.method = self.add_par( 'method', 'sextractor', str, 'Method to use (sextractor, sep)', critical=True )

        self.measurepsf = self.add_par(
            'measurepsf',
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
            ( 'Use this PSF; pass the PSF object, or its integer id. '
              'If None, will not do PSF photometry (if measurepsf is False)' )
        )

        self.apers = self.add_par(
            'apers',
            [ 1. ],
            ( None, list ),
            'Apertures in which to measure photometry; a list of 1-4 floats',
            critical=True
        )
        self.add_alias( 'apertures', 'apers' )

        self.aperunit = self.add_par(
            'aperunit',
            'fwhm',
            str,
            'Units of the apertures in the apers parameters; one of "fwhm" or "pixel"',
            critical=True
        )
        self.add_alias( 'aperture_unit', 'aperunit' )

        self.subtraction = self.add_par(
            'subtraction',
            False,
            bool,
            'Whether this is expected to run on a subtraction image or a regular image. '
        )

        self.threshold = self.add_par(
            'threshold',
            5.0,
            [float, int],
            'The number of standard deviations above the background '
            'to use as the threshold for detecting a source. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        if self.subtraction:
            return 'detection'
        else:
            return 'extraction'


class Detector:
    def __init__(self, **kwargs):
        self.pars = ParsDetector(**kwargs)

    def run(self, *args, **kwargs):
        """
        Search a regular or subtraction image for new sources, and generate a SourceList.
        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find the sources/detections in memory or in the database:
        if self.pars.subtraction:
            detections = ds.get_detections(prov, session=session)

            if detections is None:
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
                if self.pars.method == 'sep':
                    detections = self.extract_sources_sep(image)
                elif self.pars.method == 'sextractor':
                    psffile = None if self.pars.psf is None else self.pars.psf.get_fullpath()
                    detections = self.extract_sources_sextractor( image, measurepsf=self.pars.measurepsf,
                                                                  psffile=psffile )
                else:
                    raise ValueError( f"Unknown source extraction method: {self.pars.method}" )

                detections.image = image

                if detections.provenance is None:
                    detections.provenance = prov
                else:
                    if detections.provenance.id != prov.id:
                        raise ValueError('Provenance mismatch for detections and provenance!')

            ds.detections = detections

        else:  # regular image
            sources = ds.get_sources(prov, session=session)

            if sources is None:
                # use the latest image in the data store,
                # or load using the provenance given in the
                # data store's upstream_provs, or just use
                # the most recent provenance for "preprocessing"
                image = ds.get_image(session=session)

                if image is None:
                    raise ValueError(f'Cannot find an image corresponding to the datastore inputs: {ds.get_inputs()}')

                sources = self.extract_sources(image)
                sources.image = image
                if sources.provenance is None:
                    sources.provenance = prov
                else:
                    if sources.provenance.id != prov.id:
                        raise ValueError('Provenance mismatch for sources and provenance!')

            ds.sources = sources

        # make sure this is returned to be used in the next step
        return ds

    def extract_sources( self, *args, **kwargs ):
        if self.pars.method == 'sep':
            return self.extract_sources_sep( *args, **kwargs )
        elif self.pars.method == 'sextractor':
            return self.extract_sources_sextractor( *args, **kwargs )
        else:
            raise ValueError( "Unknown extraction method {self.pars.method}" )

    def extract_sources_sextractor( self, *args, **kwargs ):
        return self._run_sextractor_once( *args, **kwargs )

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

          psffile: Path or str
            File that has the PSF to use for PSF photometry

          tempname: str
            A filename base for where the catalog will be written.  The
            source file will be written to
            "{FileOnDiskMixin.temp_path}/{tempname}_sources.fits".  (A
            temporary image, weight, and mask file will be written with
            the same name base to the same directory, but will be
            deleted by this routine unless do_not_cleanup is False.)  It
            is the responsibility of the calling routine to delete this
            temporary file.

          do_not_cleanup: bool, default False
            This routine writes some temp files with the image, weight,
            mask, and sourcelist data in it, muchof which is probably
            redundant with what's already written somewhere.  Normally,
            they're deleted at the end of the routine.  Set this to True
            to keep the files for debugging purposes.  (If tempname is
            not None, then the sourcelist fill will not be deleted even
            if this is False.)

        Returns
        -------
          SourceList
            Has data and info already loaded

        """
        tmpnamebase = tempname
        if tmpnamebase is None: tmpnamebase = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
        tmpimage = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tmpnamebase}.fits'
        tmpweight = tmpimage.parent / f'{tmpnamebase}_weight.fits'
        tmpflags = tmpimage.parent / f'{tmpnamebase}_flags.fits'
        tmpsources = tmpimage.parent / f'{tmpnamebase}_sources.fits'
        tmpparams = tmpimage.parent / f'{tmpnamebase}.param'

        # For debugging purposes
        self._tmpimage = tmpimage
        self._tmpweight = tmpweight
        self._tmpflags = tmpflags
        self._tmpsources = tmpsources

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

        # The parameters we want SExtractor to write go in a file.
        # We need to edit it, though, so that the number of apertures
        # we have matches the parameters we ask for.
        # TODO : review the default param file and make sure we
        # have the things we want, and don't have too much.
        if len(apers) == 1:
            paramfile = astromatic_dir / "detection_sextractor.param"
        else:
            if len(apers) > 4:
                # This is evidently a constraint of what gets
                # written into the info HDU; it only has header
                # keywords for 4 apertures.
                raise ValueError( "Supports at most 4 apertures." )
            paramfile = tmpparams
            with open( astromatic_dir / "detection_sextractor.param" ) as ifp:
                params = [ line.strip() for line in ifp.readlines() ]
            for i in range(len(params)):
                if params[i] in [ "FLUX_APER", "FLUXERR_APER" ]:
                    params[i] = f"{params[i]}({len(apers)})"
            with open( paramfile, "w") as ofp:
                for param in params:
                    ofp.write( f"{param}\n" )

        if psffile is not None:
            if not pathlib.Path(psffile).is_file():
                raise FileNotFoundError( f"Can't read PSF file {psffile}" )

        try:

            # TODO : if the image is already on disk, then we may not need
            # to do the writing here.  In that case, we do have to think
            # about whether the extensions are different HDUs in the same
            # file, or if they are separate files (which I think can just be
            # handled by changing the sextractor arguments a bit).  If
            # they're non-FITS files, we'll have to write them.  For
            # simplicity, right now, just write the temp files, even though
            # it might be redundant.
            fits.writeto( tmpimage, image.data, header=image.raw_header )
            fits.writeto( tmpweight, image.weight )
            fits.writeto( tmpflags, image.flags )

            # TODO : right now, we're assuming that the default background
            # subtraction is fine.  Experience shows that in crowded fields
            # (e.g. star-choked galactic fields), a much slower algorithm
            # can do a lot better.

            # TODO: Understand RESCALE_WEIGHTS and WEIGHT_GAIN.
            # Since we believe our weight image is right, we don't
            # want to be doing any rescaling of it, but it's possible
            # that I don't fully understand what these parameters
            # really are.  (Documentation is lacking; see
            # https://www.astromatic.net/2009/06/02/playing-the-weighting-game-i/
            # and notice the "...to be continued".)

            res = subprocess.run( [ "source-extractor",
                                    "-CATALOG_NAME", tmpsources,
                                    "-CATALOG_TYPE", "FITS_LDAC",
                                    "-PARAMETERS_NAME", paramfile,
                                    "-FILTER", "Y",
                                    "-FILTER_NAME", str(conv),
                                    "-WEIGHT_TYPE", "MAP_WEIGHT",
                                    "-RESCALE_WEIGHTS", "N",
                                    "-WEIGHT_IMAGE", str(tmpweight),
                                    "-WEIGHT_GAIN", "N",
                                    "-FLAG_IMAGE", str(tmpflags),
                                    "-FLAG_TYPE", "OR",
                                    "-PHOT_APERTURES", ",".join( [ str(a) for a in apers ] ),
                                    "-SATUR_LEVEL", str( image.instrument_object.average_saturation_limit( image ) ),
                                    "-STARNNW_NAME", nnw,
                                    "-BACK_TYPE", "AUTO",
                                    "-BACK_SIZE", str( image.instrument_object.background_box_size ),
                                    "-BACK_FILTERSIZE", str( image.instrument_object.background_filt_size ),
                                    "-MEMORY_OBJSTACK", str( 20000 ),  # TODO: make these configurable?
                                    "-MEMORY_PIXSTACK", str( 1000000 ),
                                    "-MEMORY_BUFSIZE", str( 4096 ),
                                    tmpimage
                                   ],
                                  cwd=tmpimage.parent,
                                  capture_output=True
                                 )
            if res.returncode != 0:
                _logger.error( f"Got return {res.returncode} from sextractor call; stderr:\n{res.stderr}\n"
                               f"-------\nstdout:\n{res.stdout}" )
                raise RuntimeError( f"Error return from source-extractor call" )

            sourcelist = SourceList( image=image, format="sextrfits" )
            # Since we don't set the filepath to the temp file, manually load
            # the _data and _info fields
            sourcelist.load( tmpsources )
            sourcelist.num_sources = len( sourcelist.data )
            sourcelist.aper_rads = apers

            return sourcelist

        finally:
            if not do_not_cleanup:
                tmpimage.unlink( missing_ok=True )
                tmpweight.unlink( missing_ok=True )
                tmpflags.unlink( missing_ok=True )
                tmpparams.unlink( missing_ok=True )
                if tempname is None: tmpsources.unlink( missing_ok=True )


    def _run_psfex( self, tempname, image_id, psf_size=None, do_not_cleanup=False ):
        """Create a PSF from a SExtractor catalog file.

        Will run psfex twice, to make sure it has the right data size.
        The first pass, it will use a resampled PSF data array size of
        psf_size in x and y (or 25, of psf_size is None).  The second
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
            {FileOnDiskMixin.temp_path}/{tempname}_sources.fits.

          image_id: int
            The id of the Image the sources were extracted from

          psf_size: int or None
            The size of one side of the thumbnail of the PSF, in pixels.
            Should be odd; if it's not, 1 will be added to it.
            If None, will be determined automatically.

          do_not_cleanup: bool, default False
            If True, don't delete the psf and psfxml files that will be
            created on the way to building the PSF that's returned.
            (Normally, these temporary files are deleted.)  The psf FITS
            file will be in
            {FileOnDiskMixin.temp_path}/{tempname}_sources.psf and the
            psfxml file will be in
            {FileOnDiskMixin.temp_path}/{tempname}_sources.psf.xml


        Returns
        -------
          A PSF object.

        """
        sourcefile = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}_sources.fits'
        psffile = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}_sources.psf'
        psfxmlfile = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}_sources.psf.xml'

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
                    psfdatsize += 1

                # TODO: make the fwhmmax tried configurable
                # (This is just a range of things to try to see if we can
                # get psfex to succeed; it will stop after the first one that does.)
                fwhmmaxtotry = [ 10.0, 15.0, 20.0, 25.0 ]
                #
                # TODO: make -XML_URL configurable.  (The default there is what
                # is installed if you install the psfex package on a
                # debian-based distro, which is what the Dockerfile is built from.)
                for fwhmmaxdex, fwhmmax in enumerate( fwhmmaxtotry ):
                    command = [ 'psfex',
                                '-PSF_SIZE', f'{psfdatasize},{psfdatasize}',
                                '-SAMPLE_FWHMRANGE', f'0.5,{fwhmmax}',
                                '-SAMPLE_VARIABILITY', "0.2",   # Allowed FWHM variability (1.0 = 100%)
                                '-SAMPLE_IMAFLAGMASK', "0xff",
                                '-CHECKPLOT_DEV', 'NULL',
                                '-CHECKPLOT_TYPE', 'NONE',
                                '-CHECKIMAGE_TYPE', 'NONE',
                                '-WRITE_XML', 'Y',
                                '-XML_NAME', psfxmlfile,
                                '-XML_URL', 'file:///usr/share/psfex/psfex.xsl',
                                sourcefile ]
                    res = subprocess.run( command, cwd=sourcefile.parent, capture_output=True )
                    if res.returncode == 0:
                        fwhmmaxtotry = [ fwhmmax ]
                        break
                    else:
                        if fwhmmaxdex == len(fwhmmaxtotry) - 1:
                            _logger.error( f"psfex failed with all attempted fwhmmax.\n"
                                           f"stdout:\n------\n{res.stdout.decode('utf-8')}\n"
                                           f"stderr:\n------\n{res.stderr.decode('utf-8')}" )
                            raise RuntimeError( "Repeated failures from psfex call" )
                        _logger.warning( f"psfex failed with fwhmmax={fwhmmax}, trying {fwhmmaxtotry[fwhmmaxdex+1]}" )

                psfxml = votable.parse( psfxmlfile )
                psfstats = psfxml.get_table_by_index( 1 )
                psf_sampling = psfstats.array['Sampling_Mean'][0]
                if psf_size is None:
                    usepsfsize = int( np.ceil( psfstats.array['FWHM_FromFluxRadius_Mean'][0] * 5. ) )
                    if usepsfsize % 2 == 0:
                        usepsfsize += 1

            psf = PSF( format="psfex", image_id=image_id, fwhm_pixels=psfstats.array['FWHM_FromFluxRadius_Mean'][0] )
            psf.load( psfpath=psffile, psfxmlpath=psfxmlfile )

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

        _logger.warning( "The sep detecton method isn't fully compatible with the rest of the pipeline." )

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


if __name__ == '__main__':
    from models.base import Session
    from models.provenance import Provenance
    session = Session()
    source_lists = session.scalars(sa.select(SourceList)).all()
    prov = session.scalars(sa.select(Provenance)).all()
