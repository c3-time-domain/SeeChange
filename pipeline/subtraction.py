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
from improc import jigsaw
from improc.tools import sigma_clipping

from util.util import env_as_bool


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
            {'method': 'swarp', 'to_index': 'new'},
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

        self.subtiling = self.add_par(
            'subtiling',
            {},
            dict,
            'Subtiling parameters. '
            'Set the dictionary keys: "cut_size", "overlap" and "pad_value" '
            'as the inputs to the jigsaw.cut() function. ',
            critical=True
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'subtraction'

    def __setattr__(self, key, value):
        # make sure the pad_value in this dictionary is always the same value (as a critical parameter it should be)
        if key == 'subtiling':
            if value is None:
                value = {}
            if 'pad_value' in value:
                pad_value = value['pad_value']
                if (
                        pad_value is None or
                        (isinstance(pad_value, str) and pad_value.lower() == 'nan') or
                        (isinstance(pad_value, np.number) and np.isnan(pad_value))
                ):
                    value['pad_value'] = 'nan'

        super().__setattr__(key, value)


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
        if new_image.bg is not None:
            new_image_data = new_image.data_bgsub

        ref_image_data = ref_image.data
        if ref_image.bg is not None:
            ref_image_data = ref_image.data_bgsub

        new_image_psf = new_image.psf.get_clip()
        ref_image_psf = ref_image.psf.get_clip()
        new_image_noise = new_image.bkg_rms_estimate
        ref_image_noise = ref_image.bkg_rms_estimate
        new_image_flux_zp = 10 ** (0.4 * new_image.zp.zp)
        ref_image_flux_zp = 10 ** (0.4 * ref_image.zp.zp)
        # TODO: consider adding an estimate for the astrometric uncertainty dx, dy

        new_image_data = self.inpainter.run(new_image_data, new_image.flags, new_image.weight)
        ref_image_data = self.inpainter.run(ref_image_data, ref_image.flags, ref_image.weight)

        if self.pars.subtiling:
            new_tile_data, corners = jigsaw.cut(new_image_data, **self.pars.subtiling)
            ref_tile_data, _ = jigsaw.cut(ref_image_data, **self.pars.subtiling)  # corners should be the same!

            tile_shape = ref_tile_data.shape[1:]
            centers = [(c[0] + tile_shape[0] // 2, c[1] + tile_shape[1] // 2) for c in corners]  # x,y in flipped order!

            # find the PSF clip in the center of each tile
            new_tile_psfs = [new_image.psf.get_clip(*c) for c in centers]
            ref_tile_psfs = [ref_image.psf.get_clip(*c) for c in centers]

            # get noise arrays, find their median internally in zogy_subtract
            if isinstance(new_image_noise, np.ndarray):
                new_tile_noises, _ = jigsaw.cut(new_image_noise, **self.pars.subtiling)
            else:
                new_tile_noises = [new_image_noise] * len(corners)

            if isinstance(ref_image_noise, np.ndarray):
                ref_tile_noises, _ = jigsaw.cut(ref_image_noise, **self.pars.subtiling)
            else:
                ref_tile_noises = [ref_image_noise] * len(corners)

            # flux ZPs are assumed uniform across the image...

            # loop over the tiles and make the subtractions
            outputs = []
            for i in range(len(corners)):
                out = zogy_subtract(
                    ref_tile_data[i],
                    new_tile_data[i],
                    ref_tile_psfs[i],
                    new_tile_psfs[i],
                    ref_tile_noises[i],
                    new_tile_noises[i],
                    ref_image_flux_zp,
                    new_image_flux_zp,
                )
                outputs.append(out)

            output = {}
            tiled_output = {}
            mid_index = len(outputs) // 2
            for i, out in enumerate(outputs):
                for key, value in out.items():
                    if key == 'zero_point' and i == mid_index:
                        output[key] = value
                    if key == 'sub_psf' and i == mid_index:  # grab the central PSF as representative
                        output[key] = value  # the output PSF is smaller than if we don't tile!

                    if key not in tiled_output:
                        tiled_output[key] = np.zeros((len(corners), *tile_shape))
                    tiled_output[key][i] = value

            for key, value in tiled_output.items():  # for each type of array, e.g., sub_image, score, etc.
                output[key] = jigsaw.stitch(
                    value,
                    new_image.data.shape,
                    corners,
                    overlap=self.pars.subtiling.get('overlap')
                )

        else:  # do not use subtiling
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

        # expand the weight and flag images by the larger PSF width
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
                ref = ds.get_reference(refset.provenances, session=session)
                if ref is None:
                    raise ValueError(
                        f'Cannot find a reference image corresponding to the datastore inputs: {ds.get_inputs()}'
                    )

                prov = ds.get_provenance('subtraction', self.pars.get_critical_pars(), session=session)
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

                    # Need to make sure the upstream images are loaded into this session before
                    # we disconnect it from the database.  (We don't want to hold the database
                    # connection open through all the slow processes below.)
                    upstream_images = sub_image.upstream_images

            if self.has_recalculated:
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

                # TODO: can we get better estimates from our subtraction outdict? Issue #312
                sub_image.fwhm_estimate = new_image.fwhm_estimate
                # if the subtraction does not provide an estimate of the ZP, use the one from the new image
                sub_image.zero_point_estimate = outdict.get('zero_point', new_image.zp.zp)
                sub_image.lim_mag_estimate = new_image.lim_mag_estimate

                # if the subtraction does not provide an estimate of the background, use sigma clipping
                if 'bkg_mean' not in outdict or 'bkg_rms' not in outdict:
                    mu, sig = sigma_clipping(sub_image.data)
                    sub_image.bkg_mean_estimate = outdict.get('bkg_mean', mu)
                    sub_image.bkg_rms_estimate = outdict.get('bkg_rms', sig)

            sub_image._upstream_bitflag = 0
            sub_image._upstream_bitflag |= ds.image.bitflag
            sub_image._upstream_bitflag |= ds.sources.bitflag
            sub_image._upstream_bitflag |= ds.psf.bitflag
            sub_image._upstream_bitflag |= ds.bg.bitflag
            sub_image._upstream_bitflag |= ds.wcs.bitflag
            sub_image._upstream_bitflag |= ds.zp.bitflag

            if 'ref_image' in locals():
                sub_image._upstream_bitflag |= ref_image.bitflag

            ds.sub_image = sub_image

            ds.runtimes['subtraction'] = time.perf_counter() - t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['subtraction'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
            return ds
