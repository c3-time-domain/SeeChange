import time
import random
import pathlib
import shutil
import subprocess

import numpy as np
import pandas
import sqlalchemy as sa

from astropy.io import fits
import astropy.coordinates
import astropy.units as units

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.base import SmartSession, FileOnDiskMixin
from models.image import Image
from models.refset import RefSet

from improc.zogy import zogy_subtract, zogy_add_weights_flags
from improc.inpainting import Inpainter
from improc.alignment import ImageAligner
from improc.tools import sigma_clipping

from util.util import env_as_bool
from util.fits import save_fits_image_file
from util.logger import SCLogger
from util.exceptions import SubprocessFailure


class ParsSubtractor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.method = self.add_par(
            'method',
            'hotpants',
            str,
            'Which subtraction method to use. Possible values are: "hotpants", "zogy". '
        )

        self.refset = self.add_par(
            'refset',
            None,
            (None, str),
            'The name of the reference set to use for getting a reference image. '
        )

        self.alignment = self.add_par(
            'alignment',
            {'method': 'swarp'},
            dict,
            'How to align the reference image to the new image. This will be ingested by ImageAligner. '
        )

        self.alignment_index = self.add_par(
            'alignment_index',
            'new',
            str,
            'How to choose the index of image to align to.  Can be "new" or "ref"',
            critical=True
        )

        self.reference = self.add_par(
            'reference',
            {'minovfrac': 0.85,
             'must_match_instrument': True,
             'must_match_filter': True,
             'must_match_section': False,
             'must_match_target': False },
            dict,
            'Parameters passed to DataStore.get_reference for identifying references'
        )

        self.inpainting = self.add_par(
            'inpainting',
            {},
            dict,
            'Inpainting parameters. ',
            critical=True
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'subtraction'


class Subtractor:
    def __init__(self, **kwargs):
        self.pars = ParsSubtractor(**kwargs)
        inpainter = Inpainter(**self.pars.inpainting)
        self.pars.inpainting = inpainter.pars.get_critical_pars()  # add Inpainter defaults into this dictionary
        del inpainter
        aligner = ImageAligner(**self.pars.alignment)
        self.pars.alignment = aligner.pars.get_critical_pars()  # add ImageAligner defaults into this dictionary

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

        # TODO: add a reference cache here.

    @staticmethod
    def _subtract_naive(new_image, ref_image):
        """Subtract the reference from the image directly, assuming they are aligned and same shape.

        Doesn't do any fancy PSF matching or anything, just takes the difference of the data arrays.

        Parameters
        ----------
        new_image : Image
            The Image containing the new data, including the data array, weight, and flags
        ref_image : Image
            The Image containing the reference data, including the data array, weight, and flags

        Returns
        -------
        dictionary with the following keys:
            outim : np.ndarray
                The difference between the new and reference images
            outwt: np.ndarray
                The weight image for the difference
            outfl: np.ndarray
                The flag image for the difference
        """
        outim = new_image.data - ref_image.data

        # must add the variance to make a new weight image
        new_mask = new_image.weight <= 0
        new_weight = new_image.weight.copy()
        new_weight[new_mask] = np.nan
        new_var = 1 / new_weight ** 2

        ref_mask = ref_image.weight <= 0
        ref_weight = ref_image.weight.copy()
        ref_weight[ref_mask] = np.nan
        ref_var = 1 / ref_weight ** 2

        outwt = 1 / np.sqrt(new_var + ref_var)
        outwt[new_mask] = 0  # make sure to make zero weight the pixels that started out at zero weight

        outfl = new_image.flags.copy()
        outfl |= ref_image.flags

        return dict(outim=outim, outwt=outwt, outfl=outfl)

    def _subtract_zogy(self,
                       new_image, new_bg, new_psf, new_zp,
                       ref_image, ref_bg, ref_psf, ref_zp ):
        """Use ZOGY to subtract the two images.

        This applies PSF matching and uses the ZOGY algorithm to subtract the two images.
        reference: https://ui.adsabs.harvard.edu/abs/2016ApJ...830...27Z/abstract

        Parameters
        ----------
        new_image : Image
            The Image containing the new data, including the data array, weight, and flags.

        new_bg : Background
            Sky background for new_image

        new_psf : PSF
            PSF for new_image

        new_zp: ZeroPoint
            ZeroPoint for new_image

        ref_image : Image
            The Image containing the reference data, including the data array, weight, and flags
            The reference image must already be aligned to the new image!

        ref_bg : Background
            Sky background for ref_image

        ref_psf: PSF
            PSF for the aligned ref image

        ref_zp: ZeroPoint
            ZeroPoint for the aligned ref image

        Returns
        -------
        dictionary with the following keys:
            outim : np.ndarray
                The difference between the new and reference images
            outwt: np.ndarray
                The weight image for the difference
            outfl: np.ndarray
                The flag image for the difference
            zogy_score: np.ndarray
                The ZOGY score image (the matched-filter result)
            zogy_psf: np.ndarray
                The ZOGY PSF image (the matched-filter PSF)
            zogy_alpha: np.ndarray
                The ZOGY alpha image (the PSF flux image)
            zogy_alpha_err: np.ndarray
                The ZOGY alpha error image (the PSF flux error image)
            translient: numpy.ndarray
                The "translational transient" score for moving
                objects or slightly misaligned images.
                See the paper: ... TODO: add reference once paper is out!
            translient_sigma: numpy.ndarray
                The translient score, converted to S/N units assuming a chi2 distribution.
            translient_corr: numpy.ndarray
                The source-noise-corrected translient score.
            translient_corr_sigma: numpy.ndarray
                The corrected translient score, converted to S/N units assuming a chi2 distribution.
        """
        new_image_data = new_image.data - new_bg.counts
        ref_image_data = ref_image.data - ref_bg.counts
        new_image_psf = new_psf.get_clip()
        ref_image_psf = ref_psf.get_clip()
        new_image_noise = new_image.bkg_rms_estimate
        ref_image_noise = ref_image.bkg_rms_estimate
        new_image_flux_zp = 10 ** (0.4 * new_zp.zp)
        ref_image_flux_zp = 10 ** (0.4 * ref_zp.zp)
        # TODO: consider adding an estimate for the astrometric uncertainty dx, dy

        inpainter = Inpainter(**self.pars.inpainting)
        new_image_data = inpainter.run(new_image_data, new_image.flags, new_image.weight)
        del inpainter

        output = zogy_subtract(
            ref_image_data,
            new_image_data,
            ref_image_psf,
            new_image_psf,
            ref_image_noise,
            new_image_noise,
            ref_image_flux_zp,
            new_image_flux_zp,
        )
        # rename for compatibility
        output['outim'] = output.pop('sub_image')
        output['zogy_score_uncorrected'] = output.pop('score')
        output['score'] = output.pop('score_corr')
        output['alpha_err'] = output.pop('alpha_std')
        # convert flux based into magnitude based zero point
        output['zero_point'] = 2.5 * np.log10(output['zero_point'])

        outwt, outfl = zogy_add_weights_flags(
            ref_image.weight,
            new_image.weight,
            ref_image.flags,
            new_image.flags,
            ref_zp.zp,
            new_zp.zp,
            output['zero_point'],
            ref_psf.fwhm_pixels,
            new_psf.fwhm_pixels
        )
        output['outwt'] = outwt
        output['outfl'] = outfl


        return output

    def _subtract_hotpants(self,
                           new_image, new_bg, new_sources, new_wcs, new_psf,
                           ref_image, ref_bg, ref_sources, ref_wcs, ref_psf ):
        """Alard/Lupton subtraction.

        This applies PSF matching and uses the Alard/Lupton algorithm,
        as implemented by "hotpants", to subtract the two images.

           https://github.com/acbecker/hotpants
           https://ui.adsabs.harvard.edu/abs/1998ApJ...503..325A/abstract

        Parameters
        ----------
        new_image : Image
            The Image containing the new data, including the data array,
            weight, and flags.  ref_image and new_image must be aligned
            to each other.

        new_bg : Background
            Sky background for new_image

        new_sources : SourceList
            Source List for new_image.  Must have the is_star parameter
            (which, as of this writing, means that the source list must
            be a sextrfits source list.)

        new_wcs : WorldCoordinates
            WCS  for new_image

        new_psf : PSF
            PSF for new_image

        ref_image : Image
            The Image containing the reference data, including the data
            array, weight, and flags.  ref_image and new_image must be
            aligned to each other.

        ref_sources: SourceList
            Source List for ref_image.  Must have the is_star parameter
            (which, as of this writing, means that the source list must
            be a sextrfits source list.)

        ref_bg : Background
            Sky background for ref_image

        ref_wcs : WorldCoordinates
            WCS for ref_image

        ref_psf : PSF
            PSF for ref_image

        Returns
        -------
        dictionary with the following keys:
            outimhdr : fits.Header
                The FITS header that hotpants wrote
            outim : np.ndarray
                The difference between the new and reference images
            outwt: np.ndarray
                The weight image for the difference
            outfl: np.ndarray
                The flag image for the difference

        """

        tmpdir = ( pathlib.Path( FileOnDiskMixin.temp_path )
                   / ( ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) ) ) )
        SCLogger.debug( f'_subtract_hotpants working in directory {tmpdir}' )
        try:
            tmpdir.mkdir( exist_ok=True, parents=True )
            newim = tmpdir / "new.fits"
            newflags = tmpdir / "new.flags.fits"
            newnoise = tmpdir / "new.noise.fits"
            refim = tmpdir / "ref.fits"
            refflags = tmpdir / "ref.flags.fits"
            refnoise = tmpdir / "ref.noise.fits"
            refconv = tmpdir / "ref.convolved.fits"
            subim = tmpdir / "sub.fits"
            subflags = tmpdir / "sub.flags.fits"
            subnoise = tmpdir / "sub.noise.fits"
            substamp_file = tmpdir / "tmp.ssf"

            pixscale = new_image.instrument_object.pixel_scale

            # It's not immediately obvious what to do with the hotpants iu/tu keywords, since we haven't
            #   been carefully tracking saturation levels.  (Maybe we should?)  So, use the
            #   maximum not-flagged pixel.  (Look into how these are actually used.  Since we're
            #   explicitly giving hotpants stars to use for PSF matching, and we throw out stars
            #   that got flagged as bad, then maybe this doesn't matter so much.  Dunno.)
            il = new_bg.value - 10.0 * new_bg.noise
            iu = np.max( new_image.data[ new_image.flags == 0 ] )
            tl = ref_bg.value - 10.0 * ref_bg.noise
            tu = np.max( ref_image.data[ new_image.flags == 0 ] )

            kernelrad = int( 2.5 * new_psf.fwhm_pixels )
            rss = int( 3.75 * new_psf.fwhm_pixels )
            trtlt = 2. * np.sqrt( 2. * np.log(2.,) )

            # One could probably spend ten years futzing around and deciding
            #  what the right parmaeters to use here are.  What I've got here
            #  right now is what I used in curveball, which seemed to work
            #  for ZTF and DECam.
            hotgaussorders = [ 2, 0, 0 ]
            hotgaussfactors = [ 0.5, 1.0, 1.5 ]
            if new_psf.fwhm_pixels < ref_psf.fwhm_pixels:
                SCLogger.warning( f"New has better setting {new_psf.fwhm_pixels:.2f} "
                                  f"than ref {ref_psf.fwhm_pixels:.2f}! "
                                  f"Going to use higher orders so there can be sharpneing scariness (brrr....)" )
                hotgaussorders = [ 6, 4, 2 ]
                hotgaussfactors = [ 0.5, 1., 2. ]
                nominalfwhm = trtlt * pixscale
            else:
                nominalfwhm = np.sqrt( new_psf.fwhm_pixels**2 - ref_psf.fwhm_pixels**2 ) * pixscale
            SCLogger.debug( f'Hotpants nominal convolution kernel fwhm: {nominalfwhm: .3f}"' )

            # hotpants seems to have issues if the gaussian sigmas are too
            # small.  This could also be a case where the seeings are too
            # close.  I've seen cases where the ref was *slightly* lower
            # seeing than the new (as measured), but it was actually the new
            # that probably needed to be convolved.  Convolving the ref was
            # a disaster, until we used higher orders.  So, in this case,
            # use higher orders, and worry and fret a lot.
            nominalsigpix = nominalfwhm / trtlt / pixscale
            if nominalsigpix < 0.6:
                self.logger.warning( f'Seeings were close enough '
                                     f'(new={new_psf.fwhm_pixels:.2f}, ref={ref_psf.fwhm_pixels:.2f}) '
                                     f'that nominalsigpix was < 0.6.  Going to use higher orders in case there needs '
                                     f'to be sharpening scariness (brrr...)' )
                nominalsigpix = 1.0
                hotgaussorders = [6, 4, 2]
                hotgaussfactors = [0.5, 1., 2]

            gaussparam = [ '-ng', str(len(hotgaussorders)) ]
            tmptxt = 'Hotpants convolution σ and orders: '
            for order, factor in zip( hotgaussorders, hotgaussfactors ):
                gaussparam.extend( [ str(order), str(factor * nominalsigpix) ] )
                tmptxt += f'{factor*nominalsigpix:.3f} {order}, '
            SCLogger.debug( tmptxt )

            # Write out the aligned image data to a place hotpants can find it
            # (We *could* look to see if it's already on disk, and that would
            # save us extra writes, but right now I'm being lazy and just sticking
            # everything in the temp directory.  We don't in general expect
            # the aligned ref to be on disk, so we will have to do some writes.)
            save_fits_image_file( newim, new_image.data, new_image.header )
            save_fits_image_file( newflags, new_image.flags, new_image.header )
            save_fits_image_file( refim, ref_image.data, ref_image.header )
            save_fits_image_file( refflags, ref_image.flags, ref_image.header )

            # hotpants needs noise images not 1/σ² weight images, so do that:
            SCLogger.debug( 'Making noise images for hotpants' )
            for img, outfile in zip( [ new_image, ref_image ], [ newnoise, refnoise ] ):
                noisedata = img.weight.copy()
                # Set the weight to something tiny (i.e. ~infinite
                #   noise) where things are masked, or where noise≤0
                #   (which doesn't make sense, and is hopefully already
                #   masked).  Do this so that we don't divide by 0 when
                #   we do sqrt(1/weight).  Typical images are going to
                #   have values up to 10⁴-10⁵, so typical noises are
                #   going to be of order 10¹-10³, so typical weights are
                #   going to be of order 10⁻⁶ or bigger.  If we set
                #   "something tiny" to be 10⁻¹², that should have the
                #   same practical effect as making weight 0.
                noisedata[ ( noisedata <= 0. ) | ( img.flags != 0 ) ] = 1e-12
                noisedata = ( 1. / np.sqrt( noisedata ) )
                fits.writeto( outfile, noisedata )

            # Match _stars_ on the two images, to give the hotpants PSF
            #   matching the best chance to succeed.  Of course,
            #   this depends on sextractors CLASS_STAR being good,
            #   which is dubious, but it's what we can do.
            # To do this, we need to figure out the ra and dec of the
            #   objects on both the new and the ref images using the
            #   current WCSes.

            # wgood = new_sources.is_star & new_sources.good
            # ...sextractor's CLASS_STAR is doing so terribly that we
            #   sometimes aren't left with anything.  So, do a much
            #   cheesier star/galaxy separator right here to make sure
            #   we at least have something to work with: anything whose
            #   FWHM is within 10% of the nominal image FWHM is, we shall
            #   pretend, a star.  (Really, ought to be good enough for
            #   what hotpants really needs.)  Hopefully the FWHM won't
            #   vary too much across the image, or this could end up
            #   throwing away all the stars in (say) one corner of the
            #   image.
            # This is going to crash if there isn't a FWHM_IMAGE
            #   field in ref_sources and new_sources.  If that happens,
            #   hopefully you came here and saw this comment.  Those
            #   fields really should be there if sources were extracted
            #   with sextractor, and right now there are lots of things
            #   in the code that depend on stuff sextractor does that
            #   sep doesn't.
            width_ratio = new_sources.data['FWHM_IMAGE'] / new_psf.fwhm_pixels
            wgood = ( width_ratio > 0.9 ) & ( width_ratio < 1.1 ) & ( new_sources.good )

            newstars = pandas.DataFrame( { 'x': new_sources.x[wgood],
                                           'y': new_sources.y[wgood],
                                           'flux': new_sources.psffluxadu()[0][wgood] } )
            newsc = new_wcs.wcs.pixel_to_world( newstars.x, newstars.y )

            # wgood = ref_sources.is_star & ref_sources.good
            width_ratio = ref_sources.data['FWHM_IMAGE'] / ref_psf.fwhm_pixels
            wgood = ( width_ratio > 0.9 ) & ( width_ratio < 1.1 ) & ( ref_sources.good )
            refstars = pandas.DataFrame( { 'x': ref_sources.x[wgood],
                                           'y': ref_sources.y[wgood] } )
            refsc = ref_wcs.wcs.pixel_to_world( refstars.x, refstars.y )

            matchdex, sep2d, _ = astropy.coordinates.match_coordinates_sky( refsc, newsc )
            wmatch = np.where( sep2d < 1 * units.arcsecond )[0]

            refstars = refstars.iloc[ wmatch ].reset_index()
            newstars = newstars.iloc[ matchdex[wmatch] ].reset_index()

            SCLogger.debug( f"_subtract_hotpants matched {len(newstars)} stars between new and ref" )

            # How many stars do we want?  Imagine dividing the image
            # into regions of size 256; we want at least one star per
            # region.  (This is kinda arbitrary, but whatevs.)
            nstars_ideal = ( new_image.data.shape[0] // 256 ) * ( new_image.data.shape[1] // 256 )

            if len(newstars) < nstars_ideal // 8:
                SCLogger.error( f"_subtract_hotpants: {len(newstars)} stars for PSF matching isn't enough" )
                raise RuntimeError( "Not enough stars for _subtract_hotpants PSF matching" )
            if len(newstars) < nstars_ideal // 4:
                SCLogger.warning( f"_subtract_hotpants: {len(newstars)} stars for PSF matching is low, "
                                  f"reducing kernel orders" )
                hotgaussorders = [ 1, 0 ]
                hotgaussfactors = [ 0.6, 1.2 ]
            if len(newstars) > nstars_ideal * 2:
                # embarssment of riches, crop the list to make hotpants run faster
                sortdex = np.argsort( newstars.flux )
                sortdex = sortdex[ -(nstars_ideal*2): ]
                newstars = newstars.iloc[ sortdex ].reset_index()
                refstars = refstars.iloc[ sortdex ].reset_index()

            SCLogger.debug( f"_subtract_hotpants using {len(newstars)} stars for kernel determination" )
            with open( substamp_file, "w" ) as ofp:
                for i in range(len(newstars)):
                    ofp.write( f'{newstars.x[i]} {newstars.y[i]}\n' )

            com = [ 'hotpants',
                    '-inim', newim,
                    '-tmplim', refim,
                    '-outim', subim,
                    '-tni', refnoise,
                    '-ini', newnoise,
                    '-tmi', refflags,
                    '-imi', newflags,
                    '-oni', subnoise,
                    '-omi', subflags,
                    '-oci', refconv,
                    '-hki',
                    '-n', 'i',
                    '-c', 't',
                    '-tl', str(tl),
                    '-tu', str(tu),
                    '-il', str(il),
                    '-iu', str(iu),
                    '-r', str(kernelrad),
                    '-rss', str(rss),
                    '-ssf', substamp_file,
                    '-v', "0",
                    '-nrx', "1",
                    '-nry', "1",
                    '-ko', "1",    # Maybe make this configurable?  Order of kernel spatial variation
                    '-bgo', "1"    # Order of background variation.  Since we are not doing bgsubbed news, this matters
                    ]
            com.extend( gaussparam )
            SCLogger.debug( f"Running hotpants with command: {com}" )
            res = subprocess.run( com, shell=False, timeout=600, check=True, capture_output=True )
            SCLogger.debug( "...done running hotpants." )
            if res.returncode != 0:
                raise SubprocessFailure( res )

            # Read the results and return
            retval = {}
            with fits.open( subim ) as hdul:
                retval['outimhdr'] = hdul[0].header
                retval['outim'] = hdul[0].data
            with fits.open( subflags ) as hdul:
                retval['outfl'] = hdul[0].data
            with fits.open( subnoise ) as hdul:
                noise = hdul[0].data

            # Have to turn noise back into weight
            wt = np.empty_like( noise, dtype=np.float32 )
            wbad = ( noise <= 0 )
            wgood = ( ~wbad )
            wt[ wgood ] = 1. / ( noise[wgood] **2 )
            wt[ wbad ] = 0.
            wt[ retval['outfl'] != 0 ] = 0.
            retval['outwt'] = wt

            SCLogger.debug( "_subtract_hotpants returning." )
            return retval

        # except Exception as ex:
        #     # This is just here for debugging so I can see
        #     #   what the exception was before it jumps to
        #     #   the finally block.
        #     import pdb; pdb.set_trace()
        #     raise

        finally:
            # Clean up the temp directory
            if tmpdir.is_dir():
                shutil.rmtree( tmpdir )


    def run(self, *args, **kwargs):
        """Get a reference image and subtract it from the new image.

        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        self.has_recalculated = False

        try:
            ds, session = DataStore.from_args(*args, **kwargs)
            t_start = time.perf_counter()
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            with SmartSession(session) as session:
                # look for a reference that has to do with the current image and refset
                if self.pars.refset is None:
                    raise ValueError('No reference set given for subtraction')
                refset = session.scalars(sa.select(RefSet).where(RefSet.name == self.pars.refset)).first()
                if refset is None:
                    raise ValueError(f'Cannot find a reference set with name {self.pars.refset}')

                # TODO: we can add additional parameters of get_reference() that come from
                #  the subtraction config, such as skip_bad, match_filter, ignore_target_and_section, min_overlap
                ref = ds.get_reference(provenances=refset.provenance, session=session)
                if ref is None:
                    raise ValueError(
                        f'Cannot find a reference image corresponding to the datastore inputs: {ds.inputs_str}; '
                        f'referencing prov = {ds.prov_tree["referencing"]}'
                    )

                prov = ds.get_provenance('subtraction', self.pars.get_critical_pars(), session=session)

                if ds.get_subtraction( prov, session=session ) is None:
                    self.has_recalculated = True
                    # use the latest image in the data store,
                    # or load using the provenance given in the
                    # data store's upstream_provs, or just use
                    # the most recent provenance for "preprocessing"
                    image = ds.get_image(session=session)
                    if image is None:
                        raise ValueError(f'Cannot find an image corresponding to the datastore inputs: '
                                         f'{ds.inputs_str}')

                    SCLogger.debug( f"Making new subtraction from image {image.id} path {image.filepath} , "
                                    f"reference {ds.ref_image.id} path {ds.ref_image.filepath}" )
                    sub_image = Image.from_ref_and_new(ds.reference, image)
                    sub_image.is_sub = True
                    sub_image.provenance_id = prov.id
                    sub_image.set_coordinates_to_match_target( image )

            if self.has_recalculated:

                # Align the images
                to_index = self.pars.alignment_index
                aligner = ImageAligner(**self.pars.alignment)
                if to_index == 'ref':
                    SCLogger.error( "Aligning new to ref will violate assumptions in detection.py and measuring.py" )
                    raise RuntimeError( "Aligning new to ref not supported; align ref to new instead" )

                    for needed in [ ds.image, ds.sources, ds.bg, ds.wcs, ds.zp, ds.ref_image, ds.ref_sources ]:
                        if needed is None:
                            raise RuntimeError( "Not all data products needed for alignment to ref "
                                                "are present in the DataStore" )

                    ( aligned_image, aligned_sources,
                      aligned_bg, aligned_psf ) = aligner.run( ds.image, ds.sources, ds.bg, ds.psf, ds.wcs, ds.zp,
                                                               ds.ref_image, ds.ref_sources )
                    ds.aligned_new_image = aligned_image
                    ds.aligned_new_sources = aligned_sources
                    ds.aligned_new_bg = aligned_bg
                    ds.aligned_new_psf = aligned_psf
                    ds.aligned_new_zp = ds.zp
                    ds.aligned_ref_image = ds.ref_image
                    ds.aligned_ref_sources = ds.ref_sources
                    ds.aligned_ref_bg = ds.ref_bg
                    ds.aligned_ref_psf = ds.ref_psf
                    ds.aligned_ref_zp = ds.ref_zp
                    ds.aligned_wcs = ds.ref_wcs

                elif to_index == 'new':
                    SCLogger.debug( "Aligning ref to new" )

                    for needed in [ ds.ref_image, ds.ref_sources, ds.ref_bg, ds.ref_wcs, ds.ref_zp,
                                    ds.image, ds.sources ]:
                        if needed is None:
                            raise RuntimeError( "Not all data products needed for alignment to new "
                                                "are present in the DataStore" )

                    ( aligned_image, aligned_sources,
                      aligned_bg, aligned_psf ) = aligner.run( ds.ref_image, ds.ref_sources, ds.ref_bg,
                                                               ds.ref_psf, ds.ref_wcs, ds.ref_zp,
                                                               ds.image, ds.sources )
                    ds.aligned_new_image = ds.image
                    ds.aligned_new_sources = ds.sources
                    ds.aligned_new_bg = ds.bg
                    ds.aligned_new_psf = ds.psf
                    ds.aligned_new_zp = ds.zp
                    ds.aligned_ref_image = aligned_image
                    ds.aligned_ref_sources = aligned_sources
                    ds.aligned_ref_bg = aligned_bg
                    ds.aligned_ref_psf = aligned_psf
                    ds.aligned_ref_zp = ds.ref_zp
                    ds.aligned_wcs = ds.wcs

                    # We are going to make the aligned ref image as *not* a coadd, because
                    #   it's not a direct coadd, it's a warp of another image.  Scary.  But,
                    #   these don't generally get saved to the database, so it shouldn't matter.
                    #   (The test fixtures do care about this, as it affects filename munging.)
                    ds.aligned_ref_image.is_coadd = False

                else:
                    raise ValueError( f"alignment_index must be ref or new, not {to_index}" )

                del aligner
                ImageAligner.cleanup_temp_images()

                SCLogger.debug( "Alignment complete" )

                if self.pars.method == 'naive':
                    SCLogger.debug( "Subtracting with naive" )
                    outdict = self._subtract_naive( ds.aligned_new_image, ds.aligned_ref_image )

                elif self.pars.method == 'hotpants':
                    SCLogger.debug( "Subtracting with hotpants" )
                    outdict = self._subtract_hotpants( ds.aligned_new_image, ds.aligned_new_bg,
                                                       ds.aligned_new_sources, ds.aligned_wcs, ds.aligned_new_psf,
                                                       ds.aligned_ref_image, ds.aligned_ref_bg,
                                                       ds.aligned_ref_sources, ds.aligned_wcs, ds.aligned_ref_psf )

                    # The hotpants call in the code above ensures the sub image is normalized to the new image

                elif self.pars.method == 'zogy':
                    SCLogger.debug( "Subtracting with zogy" )
                    outdict = self._subtract_zogy( ds.aligned_new_image, ds.aligned_new_bg,
                                                   ds.aligned_new_psf, ds.aligned_new_zp,
                                                   ds.aligned_ref_image, ds.aligned_ref_bg,
                                                   ds.aligned_ref_psf, ds.aligned_ref_zp )

                    # Renormalize the difference image back to the zeropoint of the new image.
                    # Not going to renormalize score; I'd have to think harder
                    #   about whether that's the right thing to do, and it
                    #   gets renormalized to its σ in detection.py anyway.

                    normfac = 10 ** ( 0.4 * ( ds.aligned_new_zp.zp - outdict['zero_point'] ) )
                    outdict['outim'] *= normfac
                    outdict['outwt'] /= normfac*normfac
                    outdict['alpha'] *= normfac
                    outdict['alpha_err'] *= normfac
                    if 'bkg_mean' in outdict:
                        outdict['bkg_mean'] *= normfac
                    if 'bkg_rms' in outdict:
                        outdict['bkg_rms'] *= normfac

                else:
                    raise ValueError(f'Unknown subtraction method {self.pars.method}')

                SCLogger.debug( "Subtraction complete" )

                sub_image.data = outdict['outim']
                sub_image.weight = outdict['outwt']
                sub_image.flags = outdict['outfl']
                if 'outimhdr' in outdict:
                    sub_image.header = outdict['outimhdr']
                if 'score' in outdict:
                    ds.zogy_score = outdict['score']
                if 'alpha' in outdict:
                    ds.zogy_alpha = outdict['alpha']
                if 'alpha_err' in outdict:
                    ds.zogy_alpha_err = outdict['alpha_err']
                if 'psf' in outdict:
                    ds.zogy_psf = outdict['psf']

                ds.subtraction_output = outdict  # save the full output for debugging

                # TODO: can we get better estimates from our subtraction outdict? Issue #312
                # (Rough estimate might just be the larger of new/ref seeing.  That's what it would
                # be using the old 1990s SCP subtraction algorithm, and I think also with Alard/Luption.)
                sub_image.fwhm_estimate = ds.image.fwhm_estimate
                # We (I THINK) renormalized the sub_image to new_image above, so its zeropoint is the new's zeropoint
                sub_image.zero_point_estimate = ds.zp.zp
                # TODO: this implicitly assumes that the ref is much deeper than the new.
                #  If it's not, this is going to be too generous. → Issue #364
                sub_image.lim_mag_estimate = ds.image.lim_mag_estimate

                # if the subtraction does not provide an estimate of the background, use sigma clipping
                if 'bkg_mean' not in outdict or 'bkg_rms' not in outdict:
                    mu, sig = sigma_clipping(sub_image.data)
                else:
                    mu = 0.
                    sig = 0.
                sub_image.bkg_mean_estimate = outdict.get('bkg_mean', mu)
                sub_image.bkg_rms_estimate = outdict.get('bkg_rms', sig)

                sub_image._upstream_bitflag = 0
                if ( ds.exposure is not None ):
                    sub_image._upstream_bitflag |= ds.exposure.bitflag
                sub_image._upstream_bitflag |= ds.image.bitflag
                sub_image._upstream_bitflag |= ds.sources.bitflag
                sub_image._upstream_bitflag |= ds.psf.bitflag
                sub_image._upstream_bitflag |= ds.bg.bitflag
                sub_image._upstream_bitflag |= ds.wcs.bitflag
                sub_image._upstream_bitflag |= ds.zp.bitflag
                sub_image._upstream_bitflag |= ds.ref_image.bitflag

                ds.sub_image = sub_image

            ds.runtimes['subtraction'] = time.perf_counter() - t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['subtraction'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            return ds

        except Exception as e:
            # ds.catch_exception(e)
            # TODO: remove the try block above and just let exceptions be exceptions.
            # This is here as a temporary measure so that we don't have lots of
            # gratuitous diffs in a PR that's about other things simply as a result
            # of indentation changes.
            SCLogger.exception( f"Exception in Subtractor.run: {e}" )
            raise
        # finally:  # make sure datastore is returned to be used in the next step
        #     return ds
