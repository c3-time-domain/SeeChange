import time

import numpy as np
import astropy.time

from improc.photometry import photometry_and_diagnostics

from models.cutouts import Cutouts
from models.measurements import MeasurementSet
from models.object import Object

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from util.logger import SCLogger


class ParsMeasurer(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.annulus_radii = self.add_par(
            'annulus_radii',
            [4., 5.],
            list,
            'Inner and outer radii of the background annulus. '
        )

        self.annulus_units = self.add_par(
            'annulus_units',
            'fwhm',
            str,
            'Units for the annulus radii. Can be pixels or fwhm. '
            'This describes the units of the annulus_radii input. '
            'Use "pixels" to make a constant annulus, or "fwhm" to '
            'adjust the annulus size for each image based on the PSF width. '
        )

        self.use_annulus_bg_on_sub = self.add_par(
            'annulus_bg_on_sub',
            False,
            bool,
            ( 'Use an annulus background for measurements on the sub image.  Defaults to '
              'False, because sub images should already have a 0 background.' )
        )

        self.diag_box_halfsize = self.add_par(
            'diag_box_halfsize',
            2.,
            float,
            ( 'The diagnostic box used for counting bad pixels and negative pixels will be '
              '2 * diag_box_halfsize + 1 on a side.  (Think of this as a radius.)' )
        )

        self.diag_box_halfsize_unit = self.add_par(
            'diag_box_halfsize_unit',
            'fwhm',
            str,
            ( 'fwhm or pixel' )
        )

        self.negatives_n_sigma_outlier = self.add_par(
            'negatives_n_sigma_outlier',
            2.,
            float,
            ( 'A pixel this many times the 1σ noise away from 0 will be considered an outlier when '
              'counting the number of negative and positive pixels within the diag_box.' )
        )

        self.bad_thresholds = self.add_par(
            'bad_thresholds',
            { 'psf_fit_flags_bitmask': 0x2e,
              'detection_dist': 5.,
              'gaussfit_dist': 5.,
              'elongation': 3.,
              'width_ratio': 2.,
              'nbadpix': 1,
              'negfrac': 0.5,
              'negfluxfrac': 0.5,
             },
            dict,
            ( 'A dictionary of thresholds. If a Measurements has a property whose value is ≥ the '
              'threshold, it will be marked bad (is_bad set to True).  Set a value to None to not '
              'use that cut.' )
        )

        self.deletion_thresholds = self.add_par(
            'deletion_thresholds',
            { 'psf_fit_flags_bitmask': 0x2e,
              'detection_dist': 5.,
              'gaussfit_dist': 5.,
              'elongation': 3.,
              'width_ratio': 2.,
              'nbadpix': 1,
              'negfrac': 0.5,
              'negfluxfrac': 0.5,
             },
            dict,
            ( 'Like bad_thresholds, but Measurements with values ≥ the threshold will not even '
              'be saved to the database.' )
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
            'By default, Object.associate_measurements is called for measurements, which will commit new '
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


    def run(self, *args, sub_psf=None, **kwargs):
        """Measure sources found on subtraction images.

        Measure the psf flux and aperture fluxes (depending on config)
        on the sub image.  Measure morphological parameters, perform
        threshold cuts.

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

        try:
            if isinstance(args[0], Cutouts):
                raise RuntimeError( "Need to update the code for creating a Measurer given a Cutouts" )
                # args, kwargs, session = parse_session(*args, **kwargs)
                # ds = DataStore()
                # ds.cutouts = args[0]
                # ds.detections = ds.cutouts.sources
                # ds.sub_image = ds.detections.image
                # ds.image = ds.sub_image.new_image
            else:
                ds = DataStore.from_args(*args, **kwargs)

            t_start = time.perf_counter()
            if ds.update_memory_usages:
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            prov = ds.get_provenance('measuring', self.pars.get_critical_pars())

            sub_image = ds.get_sub_image()
            if sub_image is None:
                raise ValueError( "Can't perform measurements, DataStore is missing sub_image" )

            new_zp = ds.get_zp()
            if new_zp is None:
                raise ValueError(f"Can't find a zp corresponding to the datastore inputs: {ds.inputs_str}")

            # We'll be assuming that the sub was aligned with the new
            new_wcs = ds.get_wcs()
            if new_wcs is None:
                raise ValueError(f"Can't find a wcs corresponding to the datastore inputs: {ds.inputs_str}")

            detections = ds.get_detections()
            if detections is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.inputs_str}')

            cutouts = ds.get_cutouts()
            if cutouts is None:
                raise ValueError(f'Cannot find cutouts corresponding to the datastore inputs: {ds.inputs_str}')
            else:
                cutouts.load_all_co_data( sources=detections )

            if sub_psf is None:
                sub_psf = ds.psf

            # try to find some measurements in memory or in the database:
            measurement_set = ds.get_measurement_set( prov )

            # note that if measurement_set is already in the datastore,
            #  there will not be an all_measurements appended to datastore!
            if measurement_set is None:
                self.has_recalculated = True

                measurement_set = MeasurementSet( cutouts_id=ds.cutouts.id, provenance_id=prov.id )

                SCLogger.debug( f"Measurer performing measurements on {len(cutouts.co_dict)} cutouts" )

                inner_annulus_px = self.pars.annulus_radii[0]
                outer_annulus_px = self.pars.annulus_radii[1]
                if self.pars.annulus_units == 'fwhm':
                    inner_annulus_px *= sub_psf.fwhm_pixels
                    outer_annulus_px *= sub_psf.fwhm_pixels

                aper_radii = new_zp.aper_cor_radii
                if len( aper_radii ) == 0:
                    raise RuntimeError( "I don't know how to cope with no apertures." )

                # Photutils requires 1σ errors, not weights
                # It also requires True/False masks
                sub_mask = np.full_like( sub_image.flags, False, dtype=bool )
                sub_mask[ sub_image.flags != 0 ] = True
                sub_mask[ sub_image.weight <= 0. ] = True
                sub_noise = 1. / np.sqrt( sub_image.weight )
                sub_noise[ sub_mask ] = np.nan

                cutouts.load_all_co_data( sources=detections )
                rangecutouts = range( len(detections.x) )
                sub_cutouts = [ cutouts.co_dict[f"source_index_{i}"]["sub_data"] for i in rangecutouts ]
                sub_weight_cutouts = [ cutouts.co_dict[f"source_index_{i}"]["sub_weight"] for i in rangecutouts ]
                sub_flags_cutouts = [ cutouts.co_dict[f"source_index_{i}"]["sub_flags"] for i in rangecutouts ]
                # Make the weight and mask cutouts that photometry needs
                sub_mask_cutouts = [ np.full_like( c, False, dtype=bool ) for c in sub_flags_cutouts ]
                for m, c, w in zip( sub_mask_cutouts, sub_flags_cutouts, sub_weight_cutouts ):
                    m[ c != 0 ] = True
                    m[ w <= 0. ] = True
                sub_noise_cutouts = [ 1. / np.sqrt( c ) for c in sub_weight_cutouts ]
                for m, n in zip( sub_mask_cutouts, sub_noise_cutouts ):
                    n[ m ] = np.nan

                positions = [ ( detections.x[i], detections.y[i] ) for i in range(len(detections.x)) ]
                all_measurements = photometry_and_diagnostics( sub_image.data, sub_noise, sub_mask,
                                                               positions, aper_radii, psfobj=sub_psf,
                                                               dobgsub=self.pars.use_annulus_bg_on_sub,
                                                               innerrad=inner_annulus_px,
                                                               outerrad=outer_annulus_px,
                                                               cutouts=sub_cutouts,
                                                               noise_cutouts=sub_noise_cutouts,
                                                               mask_cutouts=sub_mask_cutouts,
                                                               diagdist=self.pars.diag_box_halfsize,
                                                               distunit=self.pars.diag_box_halfsize_unit )
                # Fill in some basic fields of the measurements
                for i, m in enumerate( all_measurements ):
                    m.measurementset_id = measurement_set.id
                    m.index_in_sources = i
                    sc = new_wcs.wcs.pixel_to_world( m.x, m.y )
                    m.ra = sc.ra.deg
                    m.dec = sc.dec.deg
                    m.calculate_coordinates()

                ds.all_measurements = all_measurements

                # Threshold cutting
                SCLogger.debug( f"Doing threshold cuts on {len(all_measurements)} measurements..." )
                measurements = []
                badthresh = self.pars.bad_thresholds
                delthresh = self.pars.deletion_thresholds
                _2sqrt2ln2 = 2.35482
                for m in all_measurements:
                    is_bad = False
                    keep = True

                    # Chuck it if the psf fit failed
                    if ( ( badthresh['psf_fit_flags_bitmask'] is not None )
                         and ( m.psf_fit_flags & badthresh['psf_fit_flags_bitmask'] )
                        ):
                        is_bad = True
                    if ( ( delthresh['psf_fit_flags_bitmask'] is not None )
                         and ( m.psf_fit_flags & delthresh['psf_fit_flags_bitmask'] )
                        ):
                        keep = False

                    # detection to center of fit psf distance
                    if ( badthresh['detection_dist'] is not None ) or ( delthresh['detection_dist'] is not None ):
                        dist = np.sqrt( ( m.x - m.center_x_pixel ) ** 2 + ( m.y - m.center_y_pixel ) ** 2 )
                        if ( badthresh['detection_dist'] is not None ) and ( dist >= badthresh['detection_dist'] ):
                            is_bad = True
                        if ( delthresh['detection_dist'] is not None ) and ( dist >= delthresh['detection_dist'] ):
                            keep = False

                    # Gaussian fit position to center of fit psf distance
                    if ( badthresh['gaussfit_dist'] is not None ) or ( delthresh['gaussfit_dist'] is not None ):
                        dist = np.sqrt( ( m.x - m.gfit_x ) ** 2 + ( m.y - m.gfit_y ) ** 2 )
                        if ( badthresh['gaussfit_dist'] is not None ) and ( dist >= badthresh['gaussfit_dist'] ):
                            is_bad = True
                        if ( delthresh['gaussfit_dist'] is not None ) and ( dist >= delthresh['gaussfit_dist'] ):
                            keep = False

                    # Ratio of width to psf
                    if ( badthresh['width_ratio'] is not None ) or ( delthresh['width_ratio'] is not None ):
                        width = ( m.major_width + m.minor_width ) / 2.
                        rat = width / sub_psf.fwhm_pixels
                        if ( badthresh['width_ratio'] is not None ) and ( rat >= badthresh['width_ratio'] ):
                            is_bad = True
                        if ( delthresh['width_ratio'] is not None ) and ( rat >= delthresh['width_ratio'] ):
                            keep = False

                    # Elongation
                    if ( badthresh['elongation'] is not None ) or ( delthresh['elongation'] is not None ):
                        elongation = 1e32 if m.minor_width <= 0. else m.major_width / m.minor_width
                        if ( badthresh['elongation'] is not None ) and ( elongation >= badthresh['elongation'] ):
                            is_bad = True
                        if ( delthresh['elongation'] is not None ) and ( elongation >= delthresh['elongation'] ):
                            keep = False

                    # The rest can be done in a simple loop:
                    for prop in [ 'nbadpix','negfrac', 'negfluxfrac' ]:
                        if ( badthresh[prop] is not None ) and ( getattr( m, prop ) >= badthresh[prop] ):
                            is_bad = True
                        if ( delthresh[prop] is not None ) and ( getattr( m, prop ) >= delthresh[prop] ):
                            keep = False

                    m.is_bad = is_bad
                    if keep:
                        measurements.append( m )

                # Associate objects with measurements that passed deletion thresholds
                if not self.pars.do_not_associate:
                    year = int( np.floor( astropy.time.Time( sub_image.mjd, format='mjd' ).jyear ) )
                    Object.associate_measurements( measurements, self.pars.association_radius, year=year )

                # Make sure the upstream bitflag is set for all measurements
                measurement_set._upstream_bitflag = ds.cutouts.bitflag
                for m in measurements:
                    m._upstream_bitflag = ds.cutouts.bitflag

                ds.measurement_set = measurement_set
                measurement_set.measurements = measurements

                SCLogger.debug( f"...done doing threshold cuts, {len(ds.measurements)} survived." )


            if ds.update_runtimes:
                ds.runtimes['measuring'] = time.perf_counter() - t_start
            if ds.update_memory_usages:
                import tracemalloc
                ds.memory_usages['measuring'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            return ds

        except Exception as e:
            SCLogger.exception( f"Exception in Measurer.run: {e}" )
            ds.exceptions.append( e )
            raise
