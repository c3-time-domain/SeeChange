import os
import time
import numpy as np

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.base import SmartSession
from models.image import Image

from improc.zogy import zogy_subtract, zogy_add_weights_flags
from improc.inpainting import Inpainter
from improc.alignment import ImageAligner
from improc.tools import sigma_clipping

from util.util import parse_bool


class ParsSubtractor(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.method = self.add_par(
            'method',
            'hotpants',
            str,
            'Which subtraction method to use. Possible values are: "hotpants", "zogy". '
        )

        self.alignment = self.add_par(
            'alignment',
            {'method:': 'swarp', 'to_index': 'new'},
            dict,
            'How to align the reference image to the new image. This will be ingested by ImageAligner. '
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

    def _subtract_zogy(self, new_image, ref_image):
        """Use ZOGY to subtract the two images.

        This applies PSF matching and uses the ZOGY algorithm to subtract the two images.
        reference: https://ui.adsabs.harvard.edu/abs/2016ApJ...830...27Z/abstract

        Parameters
        ----------
        new_image : Image
            The Image containing the new data, including the data array, weight, and flags.
            Image must also have the PSF and ZeroPoint objects loaded.
        ref_image : Image
            The Image containing the reference data, including the data array, weight, and flags
            Image must also have the PSF and ZeroPoint objects loaded.
            The reference image must already be aligned to the new image!

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
        new_image_data = new_image.data
        ref_image_data = ref_image.data
        new_image_psf = new_image.psf.get_clip()
        ref_image_psf = ref_image.psf.get_clip()
        new_image_noise = new_image.bkg_rms_estimate  # TOOD: improve this by using a Background object?
        ref_image_noise = 1.0  # proper coaddition images have noise=1.0 by construction
        new_image_flux_zp = 10 ** (0.4 * new_image.zp.zp)
        ref_image_flux_zp = 10 ** (0.4 * ref_image.zp.zp)
        # TODO: consider adding an estimate for the astrometric uncertainty dx, dy

        # do some additional processing of the new image
        mu, sigma = sigma_clipping(new_image_data)
        new_image_data = (new_image_data - mu) / sigma  # TODO: skip this if we already background subtracted
        if new_image_noise is not None:
            new_image_noise = new_image_noise / sigma
        else:
            new_image_noise = 1.0  # TODO: this can go away after we verify images always have background estimates!
        new_image_flux_zp = new_image_flux_zp / sigma

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
            ref_image.psf.fwhm_pixels,
            new_image.psf.fwhm_pixels
        )
        output['outwt'] = outwt
        output['outfl'] = outfl

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
        """
        Get a reference image and subtract it from the new image.
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
            if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            with SmartSession(session) as session:
                prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

                # look for a reference that has to do with the current image
                ref = ds.get_reference(session=session)
                if ref is None:
                    raise ValueError(
                        f'Cannot find a reference image corresponding to the datastore inputs: {ds.get_inputs()}'
                    )

                # manually replace the "reference" provenances with the reference image and its products
                upstreams = prov.upstreams
                upstreams = [x for x in upstreams if x.process != 'reference']  # remove reference provenance
                upstreams.append(ref.image.provenance)
                upstreams.append(ref.sources.provenance)
                upstreams.append(ref.psf.provenance)
                upstreams.append(ref.wcs.provenance)
                upstreams.append(ref.zp.provenance)
                prov.upstreams = upstreams  # must re-assign to make sure list items are unique
                prov.update_id()
                prov = session.merge(prov)
                sub_image = ds.get_subtraction(prov, session=session)

                if sub_image is None:
                    self.has_recalculated = True
                    # use the latest image in the data store,
                    # or load using the provenance given in the
                    # data store's upstream_provs, or just use
                    # the most recent provenance for "preprocessing"
                    image = ds.get_image(session=session)
                    if image is None:
                        raise ValueError(f'Cannot find an image corresponding to the datastore inputs: {ds.get_inputs()}')

                    sub_image = Image.from_ref_and_new(ref.image, image)
                    sub_image.is_sub = True
                    sub_image.provenance = prov
                    sub_image.provenance_id = prov.id
                    sub_image.coordinates_to_alignment_target()  # make sure the WCS is aligned to the correct image

                    # make sure to grab the correct aligned images
                    new_image = [im for im in sub_image.aligned_images if im.mjd == sub_image.new_image.mjd]
                    if len(new_image) != 1:
                        raise ValueError('Cannot find the new image in the aligned images')
                    new_image = new_image[0]

                    ref_image = [im for im in sub_image.aligned_images if im.mjd == sub_image.ref_image.mjd]
                    if len(ref_image) != 1:
                        raise ValueError('Cannot find the reference image in the aligned images')
                    ref_image = ref_image[0]

                    if self.pars.method == 'naive':
                        outdict = self._subtract_naive(new_image, ref_image)
                    elif self.pars.method == 'hotpants':
                        outdict = self._subtract_hotpants(new_image, ref_image)
                    elif self.pars.method == 'zogy':
                        outdict = self._subtract_zogy(new_image, ref_image)
                    else:
                        raise ValueError(f'Unknown subtraction method {self.pars.method}')

                    sub_image.data = outdict['outim']
                    sub_image.weight = outdict['outwt']
                    sub_image.flags = outdict['outfl']
                    if 'score' in outdict:
                        sub_image.score = outdict['score']
                    if 'alpha' in outdict:
                        sub_image.psfflux = outdict['alpha']
                    if 'alpha_err' in outdict:
                        sub_image.psffluxerr = outdict['alpha_err']
                    if 'psf' in outdict:
                        # TODO: clip the array to be a cutout around the PSF, right now it is same shape as image!
                        sub_image.zogy_psf = outdict['psf']  # not saved, can be useful for testing / source detection
                        if 'alpha' in outdict and 'alpha_err' in outdict:
                            sub_image.psfflux = outdict['alpha']
                            sub_image.psffluxerr = outdict['alpha_err']

                    sub_image.subtraction_output = outdict  # save the full output for debugging

            if sub_image._upstream_bitflag is None:
                sub_image._upstream_bitflag = 0
            sub_image._upstream_bitflag |= ds.sources.bitflag
            sub_image._upstream_bitflag |= ds.image.bitflag
            sub_image._upstream_bitflag |= ds.wcs.bitflag
            sub_image._upstream_bitflag |= ds.zp.bitflag
            if 'ref_image' in locals():
                sub_image._upstream_bitflag |= ref_image.bitflag

            ds.sub_image = sub_image

            ds.runtimes['subtraction'] = time.perf_counter() - t_start
            if parse_bool(os.getenv('SEECHANGE_TRACEMALLOC')):
                import tracemalloc
                ds.memory_usages['subtraction'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
            return ds
