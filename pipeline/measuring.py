import warnings
import numpy as np

from scipy import signal

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from util.util import parse_session

from models.base import _logger
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.objects import Object
from models.enums_and_bitflags import BitFlagConverter

from improc.photometry import iterative_cutouts_photometry
from improc.tools import make_gaussian


class ParsMeasurer(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.annulus_radii = self.add_par(
            'annulus_radii',
            [7.5, 10.0],
            list,
            'Inner and outer radii of the annulus. '
        )

        self.annulus_units = self.add_par(
            'annulus_units',
            'pixels',
            str,
            'Units for the annulus radii. Can be pixels or fwhm. '
            'This describes the units of the annulus_radii input. '
            'Use "pixels" to make a constant annulus, or "fwhm" to '
            'adjust the annulus size for each image based on the PSF width. '
        )

        # TODO: should we choose the "best aperture" using the config, or should each Image have its own aperture?
        self.chosen_aperture = self.add_par(
            'chosen_aperture',
            0,
            [str, int],
            'The aperture radius that is used for photometry. '
            'Choose either the index in the aperture_radii list, '
            'the string "psf", or the string "auto" to choose '
            'the best aperture in each image separately. '
        )

        self.analytical_cuts = self.add_par(
            'analytical_cuts',
            ['negatives', 'bad pixels', 'offsets', 'filter bank'],
            [list],
            'Which kinds of analytic cuts are used to give scores to this measurement. '
        )

        self.outlier_sigma = self.add_par(
            'outlier_sigma',
            3.0,
            float,
            'How many times the local background RMS for each pixel counts '
            'as being a negative or positive outlier pixel. '
        )

        self.bad_pixel_radius = self.add_par(
            'bad_pixel_radius',
            3.0,
            float,
            'Radius in pixels for the bad pixel cut. '
        )

        self.bad_pixel_exclude = self.add_par(
            'bad_pixel_exclude',
            [],
            list,
            'List of strings of the bad pixel types to exclude from the bad pixel cut. '
            'The same types are ignored when running photometry. '
        )

        self.streak_filter_angle_step = self.add_par(
            'streak_filter_angle_step',
            5.0,
            float,
            'Step in degrees for the streaks filter bank. '
        )

        self.width_filter_multipliers = self.add_par(
            'width_filter_multipliers',
            [0.25, 2.0, 5.0, 10.0],
            list,
            'Multipliers of the PSF width to use as matched filter templates'
            'to compare against the real width (x1.0) when running psf width filter. '
        )

        self.thresholds = self.add_par(
            'thresholds',
            {
                'negatives': 0.3,
                'bad pixels': 1,
                'offsets': 5.0,
                'filter bank': 1,
            },
            dict,
            'Thresholds for the disqualifier scores. '
            'If the score is higher than (or equal to) the threshold, the measurement is disqualified. '
        )

        self.association_radius = self.add_par(
            'association_radius',
            2.0,
            float,
            'Radius in arcseconds to associate measurements with an object. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'measuring'


class Measurer:
    def __init__(self, **kwargs):
        self.pars = ParsMeasurer(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

        self._filter_bank = None  # store an array of filters to match against the cutouts
        self._filter_psf_fwhm = None  # recall the FWHM used to produce this filter bank, recalculate if it changes

    def run(self, *args, **kwargs):
        """Go over the cutouts from an image and measure all sorts of things
        for each cutout: photometry (flux, centroids), etc.

        Returns a DataStore object with the products of the processing.
        """
        # most likely to get a Cutouts object or list of Cutouts
        if isinstance(args[0], Cutouts):
            new_args = [args[0]]  # make it a list if we got a single Cutouts object for some reason
            new_args += list(args[1:])
            args = tuple(new_args)

        if isinstance(args[0], list) and all([isinstance(c, Cutouts) for c in args[0]]):
            args, kwargs, session = parse_session(*args, **kwargs)
            ds = DataStore()
            ds.cutouts = args[0]
            ds.detections = ds.cutouts[0].sources
            ds.sub_image = ds.detections.image
            ds.image = ds.sub_image.new_image
        else:
            ds, session = DataStore.from_args(*args, **kwargs)
        self.has_recalculated = False

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find some measurements in memory or in the database:
        measurements_list = ds.get_measurements(prov, session=session)

        # note that if measurements_list is found, there will not be an all_measurements appended to datastore!
        if measurements_list is None or len(measurements_list) == 0:  # must create a new list of Measurements
            self.has_recalculated = True
            # use the latest source list in the data store,
            # or load using the provenance given in the
            # data store's upstream_provs, or just use
            # the most recent provenance for "detection"
            detections = ds.get_detections(session=session)

            if detections is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')

            cutouts = ds.get_cutouts(session=session)

            # prepare the filter bank for this batch of cutouts
            if self._filter_psf_fwhm is None or self._filter_psf_fwhm != cutouts[0].sources.image.get_psf().fwhm_pixels:
                self.make_filter_bank(cutouts[0].sub_data.shape[0], cutouts[0].sources.image.get_psf().fwhm_pixels)

            # go over each cutouts object and produce a measurements object
            measurements_list = []
            for i, c in enumerate(cutouts):
                m = Measurements(cutouts=c)
                # make sure to remember which cutout belongs to this measurement,
                # before either of them is in the DB and then use the cutouts_id instead
                m._cutouts_list_index = i

                m.aper_radii = c.sources.image.new_image.zp.aper_cor_radii  # zero point corrected aperture radii

                ignore_bits = 0
                for badness in self.pars.bad_pixel_exclude:
                    ignore_bits |= 2 ** BitFlagConverter.convert(badness)

                # remove the bad pixels that we want to ignore
                flags = c.sub_flags.astype('uint16') & ~np.array(ignore_bits).astype('uint16')

                annulus_radii_pixels = self.pars.annulus_radii
                if self.pars.annulus_units == 'fwhm':
                    annulus_radii_pixels = [rad * c.source.image.get_psf().fwhm_pixels for rad in annulus_radii_pixels]

                # TODO: consider if there are any additional parameters that photometry needs
                output = iterative_cutouts_photometry(
                    c.sub_data,
                    c.sub_weight,
                    flags,
                    m.psf,
                    radii=m.aper_radii,
                    annulus=annulus_radii_pixels,
                )

                m.flux_psf = output['psf_flux']
                m.flux_psf_err = output['psf_err']
                m.area_psf = output['psf_area']
                m.flux_apertures = output['fluxes']
                m.flux_apertures_err = [np.sqrt(output['variance'] * a) for a in output['areas']]  # TODO: add source noise??
                m.aper_radii = output['radii']
                m.area_apertures = output['areas']
                m.background = output['background']
                m.background_err = np.sqrt(output['variance'])
                m.offset_x = output['offset_x']
                m.offset_y = output['offset_y']
                m.width = (output['major'] + output['minor']) / 2
                m.elongation = output['elongation']
                m.position_angle = output['angle']

                if self.pars.chosen_aperture == 'auto':
                    raise NotImplementedError('Automatic aperture selection is not yet implemented.')
                if self.pars.chosen_aperture == 'psf':
                    ap_index = -1
                elif isinstance(self.pars.chosen_aperture, int):
                    ap_index = self.pars.chosen_aperture
                else:
                    raise ValueError(
                        f'Invalid value "{self.pars.chosen_aperture}" for chosen_aperture in the measuring parameters.'
                    )
                m.best_aperture = ap_index

                m.provenance = prov
                m.provenance_id = prov.id

                # Apply analytic cuts to each stamp image, to rule out artefacts.
                m.disqualifier_scores = {}
                if m.background != 0 and m.background_err > 0.1:
                    norm_data = (c.sub_nandata - m.background) / m.background_err  # normalize
                else:
                    warnings.warn(f'Background mean= {m.background}, std= {m.background_err}, normalization skipped!')
                    norm_data = c.sub_nandata  # no good background measurement, do not normalize!

                positives = np.sum(norm_data > self.pars.outlier_sigma)
                negatives = np.sum(norm_data < -self.pars.outlier_sigma)
                if negatives == 0:
                    m.disqualifier_scores['negatives'] = 0.0
                elif positives == 0:
                    m.disqualifier_scores['negatives'] = 1.0
                else:
                    m.disqualifier_scores['negatives'] = negatives / positives

                x, y = np.meshgrid(range(c.sub_data.shape[0]), range(c.sub_data.shape[1]))
                x = x - c.sub_data.shape[1] // 2 - m.offset_x
                y = y - c.sub_data.shape[0] // 2 - m.offset_y
                r = np.sqrt(x ** 2 + y ** 2)
                bad_pixel_inclusion = r <= self.pars.bad_pixel_radius + 0.5
                m.disqualifier_scores['bad pixels'] = np.sum(flags[bad_pixel_inclusion] > 0)

                norm_data_no_nans = norm_data.copy()
                norm_data_no_nans[np.isnan(norm_data)] = 0

                filter_scores = []
                for template in self._filter_bank:
                    filter_scores.append(np.max(signal.correlate(abs(norm_data_no_nans), template, mode='same')))

                m.disqualifier_scores['filter bank'] = np.argmax(filter_scores)

                offset = np.sqrt(m.offset_x ** 2 + m.offset_y ** 2)
                m.disqualifier_scores['offsets'] = offset

                # TODO: add additional disqualifiers

                # make sure disqualifier scores don't have any numpy types
                for k, v in m.disqualifier_scores.items():
                    if isinstance(v, np.number):
                        m.disqualifier_scores[k] = v.item()

                measurements_list.append(m)

            saved_measurements = []
            for m in measurements_list:
                if m.passes():  # all disqualifiers are below threshold
                    saved_measurements.append(m)

            # add the resulting measurements to the data store
            ds.all_measurements = measurements_list  # debugging only
            ds.failed_measurements = [m for m in measurements_list if m not in saved_measurements]  # debugging only
            ds.measurements = saved_measurements  # only keep measurements that passed the disqualifiers cuts.
            ds.sub_image.measurements = saved_measurements

        # make sure this is returned to be used in the next step
        return ds

    def make_filter_bank(self, imsize, psf_fwhm):
        """Make a filter bank matching the PSF width.

        Parameters
        ----------
        imsize: int
            The size of the image cutouts, which is also the size of the filters. # TODO: allow smaller filters?
        psf_fwhm : float
            The FWHM of the PSF in pixels.
        """
        psf_sigma = psf_fwhm / 2.355  # convert FWHM to sigma
        templates = []
        templates.append(make_gaussian(imsize=imsize, sigma_x=psf_sigma, sigma_y=psf_sigma, norm=2))

        # narrow gaussian to trigger on cosmic rays, wider templates for extended sources
        for multiplier in self.pars.width_filter_multipliers:
            templates.append(
                make_gaussian(imsize=imsize, sigma_x=psf_sigma * multiplier, sigma_y=psf_sigma * multiplier, norm=2)
            )

        # add some streaks:
        (y, x) = np.meshgrid(range(-(imsize // 2), imsize // 2 + 1), range(-(imsize // 2), imsize // 2 + 1))
        for angle in np.arange(-90.0, 90.0, self.pars.streak_filter_angle_step):

            if abs(angle) == 90:
                d = np.abs(x)  # distance from line
            else:
                a = np.tan(np.radians(angle))
                b = 0  # impact parameter is zero for centered streak
                d = np.abs(a * x - y + b) / np.sqrt(1 + a ** 2)  # distance from line
            streak = (1 / np.sqrt(2.0 * np.pi) / psf_sigma) * np.exp(
                -0.5 * d ** 2 / psf_sigma ** 2
            )
            streak /= np.sqrt(np.sum(streak ** 2))  # verify that the template is normalized

            templates.append(streak)

        self._filter_bank = templates
        self._filter_psf_fwhm = psf_fwhm

