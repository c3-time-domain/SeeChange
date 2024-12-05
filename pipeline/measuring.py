import time
import warnings
import numpy as np

from scipy import signal

from improc.photometry import iterative_cutouts_photometry
from improc.tools import make_gaussian

from models.base import SmartSession
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.enums_and_bitflags import BitFlagConverter, BadnessConverter

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from util.util import env_as_bool
from util.logger import SCLogger


class ParsMeasurer(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.annulus_radii = self.add_par(
            'annulus_radii',
            [7.5, 10.0],
            list,
            'Inner and outer radii of the background annulus. '
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

        self.use_annulus_for_centroids = self.add_par(
            'use_annulus_for_centroids',
            True,
            bool,
            'Use the local background measurements via an annulus to adjust the centroids and second moments. '
        )

        self.analytical_cuts = self.add_par(
            'analytical_cuts',
            ['negatives', 'bad pixels', 'offsets', 'filter bank', 'bad_flag'],
            [list],
            ( 'Which kinds of analytic cuts are used to give scores to this measurement. '
              'TODO: remove these in favor of the thresholds dict (and put None to not threshold '
              'on any one of them)?  Issue #319.' ),
        )

        self.outlier_sigma = self.add_par(
            'outlier_sigma',
            3.0,
            float,
            'How many times the local background RMS the pixel flux must be '
            'to be considered a negative or positive outlier pixel. '
        )

        self.bad_pixel_radius = self.add_par(
            'bad_pixel_radius',
            3.0,
            float,
            ( 'Any bad pixels within this pixel radius of the center of a cutout will be counted '
              'towards the "bad pixels" disqualifier score' )
        )

        self.bad_pixel_exclude = self.add_par(
            'bad_pixel_exclude',
            [],
            list,
            'List of strings of the bad pixel types to exclude from the bad pixel cut. '
            'The same types are ignored when running photometry. '
        )

        self.bad_flag_exclude = self.add_par(
            'bad_flag_exclude',
            [],
            list,
            'List of strings of the bad flag types (i.e., bitflag) to exclude from the bad flag cut. '
            'This includes things like image saturation, too many sources, etc. '
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
                'bad_flag': 1,
            },
            dict,
            'Failure thresholds for the disqualifier scores. '
            'If the score is higher than (or equal to) the threshold, the measurement is marked as bad. '
        )

        self.deletion_thresholds = self.add_par(
            'deletion_thresholds',
            None,
            (dict, None),
            'Deletion thresholds for the disqualifier scores. '
            'If the score is higher than (or equal to) the threshold, the measurement is not saved. ',
            critical=False
        )

        self.association_radius = self.add_par(
            'association_radius',
            2.0,
            float,
            'Radius in arcseconds to associate measurements with an object. '
        )

        self.do_not_associate = self.add_par(
            'do_not_associate',
            False,
            bool,
            'By default, associate_object is called for each measurement, which will commit new '
            'Object rowss to the database.  Set this flag to skip this step.'
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

    def run(self, *args, sub_psf=None, **kwargs):
        """Measure all sorts of things on the cutous from an image.

        Go over the cutouts from an image and measure all sorts of things
        for each cutout: photometry (flux, centroids), etc.

        Returns a DataStore object with the products of the processing.

        Parameters
        ----------
          sub_psf: PSF or None
             The PSF object from the sub image.  If None, will use the
             psf object from the data store, which is probably not the
             right thing for zogy, but is hopefully close enough.  (It
             would be the right thing in an Alard/Lupton subtraction
             where the ref was convolved to the new.)

        """
        self.has_recalculated = False
        try:  # first make sure we get back a datastore, even an empty one
            if isinstance(args[0], Cutouts):
                raise RuntimeError( "Need to update the code for creating a Measurer given a Cutouts" )
                # args, kwargs, session = parse_session(*args, **kwargs)
                # ds = DataStore()
                # ds.cutouts = args[0]
                # ds.detections = ds.cutouts.sources
                # ds.sub_image = ds.detections.image
                # ds.image = ds.sub_image.new_image
            else:
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
            prov = ds.get_provenance('measuring', self.pars.get_critical_pars(), session=session)

            # sub_image = ds.get_subtraction( session=session )
            # if sub_image is None:
            #     raise ValueError(f"Can't find a subtraction image corresponding to "
            #                      f"datastore inputs: {ds.get_inputs()}" )
            new_zp = ds.get_zp( session=session )
            if new_zp is None:
                raise ValueError(f"Can't find a zp corresponding to the datastore inputs: {ds.get_inputs()}")

            new_wcs = ds.get_wcs( session=session )
            if new_wcs is None:
                raise ValueError(f"Can't find a wcs corresponding to the datastore inputs: {ds.get_inputs()}")

            detections = ds.get_detections(session=session)
            if detections is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')

            cutouts = ds.get_cutouts(session=session)
            if cutouts is None:
                raise ValueError(f'Cannot find cutouts corresponding to the datastore inputs: {ds.get_inputs()}')
            else:
                cutouts.load_all_co_data( sources=detections )

            if sub_psf is None:
                sub_psf = ds.psf

            # try to find some measurements in memory or in the database:
            measurements_list = ds.get_measurements(prov, session=session)

            # note that if measurements_list is found, there will not be an all_measurements appended to datastore!
            if measurements_list is None or len(measurements_list) == 0:  # must create a new list of Measurements
                self.has_recalculated = True

                SCLogger.debug( f"Measurer performing measurements on {len(cutouts.co_dict)} cutouts" )

                # prepare the filter bank for this batch of cutouts

                if self._filter_psf_fwhm is None or self._filter_psf_fwhm != sub_psf.fwhm_pixels:
                    self.make_filter_bank( cutouts.co_dict["source_index_0"]["sub_data"].shape[0], sub_psf.fwhm_pixels )

                # go over each cutouts object and produce a measurements object
                measurements_list = []
                for key, co_subdict in cutouts.co_dict.items():
                    m = Measurements( cutouts_id=cutouts.id )
                    m.index_in_sources = int(key[13:]) # grab just the number from "source_index_xxx"
                    m.get_data_from_cutouts( cutouts=cutouts, detections=detections )

                    m.best_aperture = -1 if detections.best_aper_num is None else detections.best_aper_num

                    # These will be rounded by Measurements.__setattr__
                    m.center_x_pixel = detections.x[m.index_in_sources]
                    m.center_y_pixel = detections.y[m.index_in_sources]

                    # zero point corrected aperture radii
                    m.aper_radii = new_zp.aper_cor_radii

                    ignore_bits = 0
                    for badness in self.pars.bad_pixel_exclude:
                        ignore_bits |= 2 ** BitFlagConverter.convert(badness)

                    # remove the bad pixels that we want to ignore
                    flags = co_subdict['sub_flags'].astype('uint16') & ~np.array(ignore_bits).astype('uint16')

                    annulus_radii_pixels = self.pars.annulus_radii
                    if self.pars.annulus_units == 'fwhm':
                        fwhm = sub_psf.fwhm_pixels
                        annulus_radii_pixels = [rad * fwhm for rad in annulus_radii_pixels]

                    # TODO: consider if there are any additional parameters that photometry needs
                    # TODO : I'm distressed by how long this next one takes.
                    output = iterative_cutouts_photometry(
                        co_subdict['sub_data'],
                        co_subdict['sub_weight'],
                        flags,
                        radii=m.aper_radii,
                        annulus=annulus_radii_pixels,
                        local_bg=self.pars.use_annulus_for_centroids,
                    )

                    m.flux_apertures = output['fluxes']
                    m.flux_apertures_err = [np.sqrt(output['variance']) * norm for norm in output['normalizations']]
                    m.aper_radii = output['radii']
                    m.area_apertures = output['areas']
                    m.bkg_mean = output['background']
                    m.bkg_std = np.sqrt(output['variance'])
                    m.bkg_pix = output['n_pix_bg']
                    m.offset_x = output['offset_x']
                    m.offset_y = output['offset_y']
                    m.width = (output['major'] + output['minor']) / 2
                    m.elongation = output['elongation']
                    m.position_angle = output['angle']

                    # update the coordinates using the centroid offsets
                    x = m.center_x_pixel + m.offset_x
                    y = m.center_y_pixel + m.offset_y
                    ra, dec = new_wcs.wcs.pixel_to_world_values(x, y)
                    m.ra = float(ra)
                    m.dec = float(dec)
                    m.calculate_coordinates()

                    # PSF photometry:
                    # Two options: use the PSF flux from ZOGY, or use the new image PSF to measure the flux.
                    # TODO: this is currently commented out since I don't know how to normalize this flux
                    # if c.sub_psfflux is not None and c.sub_psffluxerr is not None:
                    #     ix = int(np.round(m.offset_x + c.sub_data.shape[1] // 2))
                    #     iy = int(np.round(m.offset_y + c.sub_data.shape[0] // 2))
                    #
                    #     # when offsets are so big it really doesn't matter what we put here, it will fail the cuts
                    #     if ix < 0 or ix >= c.sub_psfflux.shape[1] or iy < 0 or iy >= c.sub_psfflux.shape[0]:
                    #         m.flux_psf = np.nan
                    #         m.flux_psf_err = np.nan
                    #         m.area_psf = np.nan
                    #     else:
                    #         m.flux_psf = c.sub_psfflux[iy, ix]
                    #         m.flux_psf_err = c.sub_psffluxerr[iy, ix]
                    #         psf = c.sources.image.get_psf()
                    #         m.area_psf = np.nansum(psf.get_clip(c.x, c.y))
                    # else:
                    if np.isnan(ra) or np.isnan(dec):
                        flux = np.nan
                        fluxerr = np.nan
                        area = np.nan
                    else:
                        flux, fluxerr, area = m.get_flux_at_point( ra, dec, aperture='psf', wcs=new_wcs, psf=sub_psf )
                    m.flux_psf = flux
                    m.flux_psf_err = fluxerr
                    m.area_psf = area

                    # update the provenance
                    m.provenance_id = prov.id

                    # Apply analytic cuts to each stamp image, to rule out artefacts.
                    m.disqualifier_scores = {}
                    if m.bkg_mean != 0 and m.bkg_std > 0.1:
                        norm_data = (m.sub_nandata - m.bkg_mean) / m.bkg_std  # normalize
                    else:
                        # only provide this warning if the offset is within the image
                        # otherwise, this measurement is never going to pass any cuts
                        # and we don't want to spam the logs with this warning
                        if (
                                abs(m.offset_x) < m.sub_data.shape[1] and
                                abs(m.offset_y) < m.sub_data.shape[0]
                        ):
                            warnings.warn(f'Background mean= {m.bkg_mean}, std= {m.bkg_std}, normalization skipped!')
                        norm_data = m.sub_nandata  # no good background measurement, do not normalize!

                    positives = np.sum(norm_data > self.pars.outlier_sigma)
                    negatives = np.sum(norm_data < -self.pars.outlier_sigma)
                    if negatives == 0:
                        m.disqualifier_scores['negatives'] = 0.0
                    elif positives == 0:
                        m.disqualifier_scores['negatives'] = 1.0
                    else:
                        m.disqualifier_scores['negatives'] = negatives / positives

                    x, y = np.meshgrid(range(m.sub_data.shape[0]), range(m.sub_data.shape[1]))
                    x = x - m.sub_data.shape[1] // 2 - m.offset_x
                    y = y - m.sub_data.shape[0] // 2 - m.offset_y
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

                SCLogger.debug( f"Running threshold cuts on {len(measurements_list)} measurements" )

                saved_measurements = []
                for m in measurements_list:
                    # regardless of wether we created these now, or loaded from DB,
                    # the bitflag should be updated based on the most recent data
                    m._upstream_bitflag = 0
                    m._upstream_bitflag |= cutouts.bitflag

                    ignore_bits = 0
                    for badness in self.pars.bad_flag_exclude:
                        ignore_bits |= 2 ** BadnessConverter.convert(badness)

                    m.disqualifier_scores['bad_flag'] = int(np.bitwise_and(
                        np.array(m.bitflag).astype('uint64'),
                        ~np.array(ignore_bits).astype('uint64'),
                    ))

                    threshold_comparison = self.compare_measurement_to_thresholds(m)
                    if threshold_comparison != "delete":  # all disqualifiers are below threshold
                        m.is_bad = threshold_comparison == "bad"
                        saved_measurements.append(m)

                # add the resulting measurements to the data store
                ds.all_measurements = measurements_list  # debugging only
                ds.failed_measurements = [m for m in measurements_list if m not in saved_measurements]  # debugging only
                ds.measurements = saved_measurements  # only keep measurements that passed the disqualifiers cuts.

                # Associate objects
                if not self.pars.do_not_associate:
                    SCLogger.debug( f"Associating objects to {len(ds.measurements)} Measurementses" )

                    with SmartSession( session ) as sess:
                        for m in ds.measurements:
                            m.associate_object( self.pars.association_radius,
                                                is_testing=prov.is_testing,
                                                session=sess )
                else:
                    SCLogger.debug( "Skipping association of measurements with objects." )

            else:
                ds.measurements = measurements_list

            ds.runtimes['measuring'] = time.perf_counter() - t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['measuring'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
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

    def compare_measurement_to_thresholds(self, m):
        """Compare measurement disqualifiers of a Measurements object to the thresholds set for
        this measurer object.

        Inputs:
          - m : a Measurements object to be compared

        returns one of three strings to indicate the result
          - "ok"     : All disqualifiers below both thresholds
          - "bad"    : Some disqualifiers above mark_thresh but all
                       below deletion_thresh
          - "delete" : Some disqualifiers above deletion_thresh

        """
        passing_status = "ok"

        mark_thresh = self.pars.thresholds
        deletion_thresh = ( mark_thresh if self.pars.deletion_thresholds is None
                           else self.pars.deletion_thresholds )

        combined_keys = np.unique(list(mark_thresh.keys()) + list(deletion_thresh.keys())) # unique keys from both
        for key in combined_keys:
            if deletion_thresh.get(key) is not None and m.disqualifier_scores[key] >= deletion_thresh[key]:
                passing_status =  "delete"
                break
            if mark_thresh.get(key) is not None and m.disqualifier_scores[key] >= mark_thresh[key]:
                passing_status = "bad" # no break because another key could trigger "delete"

        return passing_status
