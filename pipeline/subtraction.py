import time
import numpy as np

import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.base import SmartSession
from models.image import Image
from models.refset import RefSet

from improc.zogy import zogy_subtract, zogy_add_weights_flags
from improc.inpainting import Inpainter
from improc.alignment import ImageAligner
from improc.tools import sigma_clipping

from util.util import env_as_bool
from util.logger import SCLogger

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
        self.inpainter = Inpainter(**self.pars.inpainting)
        self.pars.inpainting = self.inpainter.pars.get_critical_pars()  # add Inpainter defaults into this dictionary
        self.aligner = ImageAligner(**self.pars.alignment)
        self.pars.alignment = self.aligner.pars.get_critical_pars()  # add ImageAligner defaults into this dictionary

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

        new_image_data = self.inpainter.run(new_image_data, new_image.flags, new_image.weight)

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

        outwt, outfl = zogy_add_weights_flags(
            ref_image.weight,
            new_image.weight,
            ref_image.flags,
            new_image.flags,
            ref_psf.fwhm_pixels,
            new_psf.fwhm_pixels
        )
        output['outwt'] = outwt
        output['outfl'] = outfl

        # convert flux based into magnitude based zero point
        output['zero_point'] = 2.5 * np.log10(output['zero_point'])

        return output

    def _subtract_hotpants(self, new_image, ref_image):
        """Use Hotpants to subtract the two images.

        This applies PSF matching and uses the Hotpants algorithm to subtract the two images.
        reference: ...

        Parameters
        ----------
        new_image : Image
            The Image containing the new data, including the data array, weight, and flags.
            Image must also have the PSF and ZeroPoint objects loaded.
        ref_image : Image
            The Image containing the reference data, including the data array, weight, and flags
            Image must also have the PSF and ZeroPoint objects loaded.

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
        raise NotImplementedError('Not implemented Hotpants subtraction yet')

    def run(self, *args, **kwargs):
        """Get a reference image and subtract it from the new image.

        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        self.has_recalculated = False
        try:  # first make sure we get back a datastore, even an empty one
            ds, session = DataStore.from_args(*args, **kwargs)
        except Exception as e:
            return DataStore.catch_failure_to_parse(e, *args)

        try:
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
                ref = ds.get_reference(provenances=refset.provenances, session=session)
                if ref is None:
                    raise ValueError(
                        f'Cannot find a reference image corresponding to the datastore inputs: {ds.get_inputs()}'
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
                                         f'{ds.get_inputs()}')

                    SCLogger.debug( f"Making new subtraction from image {image.id} path {image.filepath} , "
                                    f"reference {ds.ref_image.id} path {ds.ref_image.filepath}" )
                    sub_image = Image.from_ref_and_new(ds.ref_image, image)
                    sub_image.is_sub = True
                    sub_image.provenance_id = prov.id
                    sub_image.set_coordinates_to_match_target( image )

            if self.has_recalculated:

                # Align the images
                to_index = self.pars.alignment_index
                if to_index == 'ref':
                    SCLogger.error( "Aligning new to ref will violate assumptions in detection.py and measuring.py" )
                    raise RuntimeError( "Aligning new to ref not supported; align ref to new instead" )

                    for needed in [ ds.image, ds.sources, ds.bg, ds.wcs, ds.zp, ds.ref_image, ds.ref_sources ]:
                        if needed is None:
                            raise RuntimeError( "Not all data products needed for alignment to ref "
                                                "are present in the DataStore" )
                    ( aligned_image, aligned_sources,
                      aligned_bg, aligned_psf ) = self.aligner.run( ds.image, ds.sources, ds.bg, ds.psf, ds.wcs, ds.zp,
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
                      aligned_bg, aligned_psf ) = self.aligner.run( ds.ref_image, ds.ref_sources, ds.ref_bg,
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

                else:
                    raise ValueError( f"alignment_index must be ref or new, not {to_index}" )

                ImageAligner.cleanup_temp_images()

                SCLogger.debug( "Alignment complete" )

                if self.pars.method == 'naive':
                    SCLogger.debug( "Subtracting with naive" )
                    outdict = self._subtract_naive( ds.aligned_new_image, ds.aligned_ref_image )

                elif self.pars.method == 'hotpants':
                    SCLogger.debug( "Subtracting with hotpants" )
                    outdict = self._subtract_hotpants() # FIGURE OUT ARGUMENTS

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
                if 'score' in outdict:
                    ds.zogy_score = outdict['score']
                    # sub_image.score = outdict['score']
                if 'alpha' in outdict:
                    # sub_image.psfflux = outdict['alpha']
                    ds.zogy_alpha = outdict['alpha']
                if 'alpha_err' in outdict:
                    # sub_image.psffluxerr = outdict['alpha_err']
                    ds.zogy_alpha_err = outdict['alpha_err']
                if 'psf' in outdict:
                    ds.zogy_psf = outdict['psf']

                ds.subtraction_output = outdict  # save the full output for debugging

                # TODO: can we get better estimates from our subtraction outdict? Issue #312
                sub_image.fwhm_estimate = ds.image.fwhm_estimate
                # We (I THINK) renormalized the sub_image to new_image above, so its zeropoint is the new's zeropoint
                sub_image.zero_point_estimate = ds.zp.zp
                # TODO: this implicitly assumes that the ref is much deeper than the new.
                #  If it's not, this is going to be too generous.
                sub_image.lim_mag_estimate = ds.image.lim_mag_estimate

                # if the subtraction does not provide an estimate of the background, use sigma clipping
                if 'bkg_mean' not in outdict or 'bkg_rms' not in outdict:
                    mu, sig = sigma_clipping(sub_image.data)
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

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
            return ds
