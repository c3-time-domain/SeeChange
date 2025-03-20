import time
import uuid

import numpy as np

from util.logger import SCLogger

from models.provenance import Provenance
from models.fakeset import FakeSet, FakeAnalysis
from models.base import SmartSession
from models.image import Image

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore, ProvenanceTree


class ParsFakeInjector(Parameters):
    def __init__( self, **kwargs ):
        super().__init__()

        self.min_fake_mag = self.add_par(
            'min_fake_mag',
            -2.,
            float,
            'Minimum (brightest) magnitude of fake to inject.  Relative to the image magnitude limit if '
            'mag_rel_limmag is True, otherwise just the apparent magnitude.',
            critical=True
        )

        self.max_fake_mag = self.add_par(
            'max_fake_mag',
            1.,
            float,
            'Maximum (dimmest) magnitude of fake to inject.  Relative to the image magnitlude limit if '
            'mag_rel_limmag is True, otherwise just the apparent magnitude.',
            critical=True
        )

        self.mag_rel_limmag = self.add_par(
            'mag_rel_limmag',
            True,
            bool,
            'Is min/max_fake_mag relative to the image lim_mag_estimate, or straight-up magnitude?',
            critical=True
        )

        self.num_fakes = self.add_par(
            'num_fakes',
            100,
            int,
            'Total number of fakes to inject',
            critical=True
        )

        self.mag_prob_ratio = self.add_par(
            'mag_prob_ratio',
            1.,
            float,
            'Probability density of max mag divided by probability density of min mag.  If 1., '
            'fakes are chosen from a uniform magnitude distribution.  If <1., there will be more '
            'brighter fakes than dimmer fakes.  If >1., there will be more dimmer fakes than '
            'brighter fakes.',
            critical=True
        )

        self.random_seed = self.add_par(
            'random_seed',
            0,
            int,
            "Random seed to use.  0 = pull a random seed from system entropy.  This is usually what you "
            "want, but you can set this to something else if you want detailed reproducibility.  "
            "Note that the provenance generated will depend on the actual random seed, not 0, so "
            "you won't get reproducible provenances either if you keep this at 0!",
            critical=True
        )

        self.hostless_frac = self.add_par(
            'hostless_frac',
            0.,
            float,
            "Fraction of fakes that aren't (necessdarily) near a host galaxy.  Hostless fakes will be "
            "randomly placed in the image without regard to the distribution of sources in the image.  "
            "Set this to 0. for all fakes to have a host, to 1. for all fakes to be purely randomly placed.",
            critical=True
        )

        self.host_minmag = self.add_par(
            'host_minmag',
            -3.,
            float,
            "Place fakes on hosts that are no brighter than this relative to the fake magnitude. ",
            critical=True
        )

        self.host_maxmag = self.add_par(
            'host_maxmag',
            0.5,
            float,
            "Place fakes on hosts that are no dimmer this relative to the fake magnitude. ",
            critical=True
        )

        self.host_distscale = self.add_par(
            'host_distscale',
            1.,
            float,
            "Fakes placed on hosts will be placed in an exponential distribution with this length scale "
            "away from the host from the center in units of the 1σ moment (Sextractor AWIN_IMAGE/BWIN_IMAGE).",
            critical=True
        )

        self.detection_pixel_range = self.add_par(
            'detection_pixel_range',
            2.,
            float,
            "When determining if a fake was detected, look for detections within this many pixels "
            "of the fake's injected position",
            critical=True
        )

        self._enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name(self):
        return 'fakeinjection'


class FakeInjector:
    """Come up with a list of fake point sources to inject on to an image.

    Despite the name, doesn't actually do the injection.  It generates a
    FakeSet object; call the inject_on_image() method of that object to
    generate an image with the injected fakes.

    """

    def __init__( self, **kwargs ):
        self.pars = ParsFakeInjector( **kwargs )
        self.has_recalculated = False


    def place_fake_on_host( self, ds, mag, rng ):
        sources = ds.get_sources()
        zp = ds.get_zp()
        if sources.format != 'sextrfits':
            raise RuntimeError( "Can only place fakes on hosts if source_list is from sextractor" )

        if self._hosts_used is None:
            # Initialize stuff based on the source list
            self._hostmag = -2.5 * np.log10( sources.data['FLUX_AUTO'] ) + zp.zp
            self._hostcand = np.where( ( ~np.isnan(self._hostmag) ) & ( ~sources.is_star ) & sources.good )[0]
            self._hosts_used = []

        possibilities = [ dex for dex in self._hostcand
                          if ( ( dex not in self._hosts_used ) and
                               ( self._hostmag[dex] >= ( mag + self.pars.host_minmag ) ) and
                               ( self._hostmag[dex] <= ( mag + self.pars.host_maxmag ) ) ) ]
        if len( possibilities ) == 0:
            raise RuntimeError( "Failed to find suitable host for fake with magnitude {m}." )

        dexdex = rng.integers( 0, len(possibilities) )
        dex = possibilities[dexdex]
        self._hosts_used.append( dex )

        reldist = rng.exponential( scale=self.pars.host_distscale )
        # This angle is relative to the host major axis
        angle = rng.uniform( 0, 2*np.pi )
        if sources.data['AWIN_IMAGE'][dex] <= 0.:
            # punt
            ecc = 0.
            a = sources.data['FWHM_IMAGE'][dex] / 2.355
            if a <= 0.:
                raise ValueError( "Sextractor source has non-positive AWIN_IMAGE and FWHM_IMAGE" )
        else:
            a = sources.data['AWIN_IMAGE'][dex]
            if sources.data['BWIN_IMAGE'][dex] >= a:
                # punt
                ecc = 0.
            else:
                ecc = np.sqrt( 1. - ( sources.data['BWIN_IMAGE'][dex] / a ) ** 2 )
        hostsize = a * np.sqrt( 1. - ( ecc * np.sin( angle ) )**2. )
        dist = reldist * hostsize
        xp = dist * np.cos( angle )
        yp = dist * np.sin( angle )
        # Rotate to image plane
        theta = sources.data['THETAWIN_IMAGE'][dex]
        x = xp * np.cos( theta ) - yp * np.sin( theta )
        y = xp * np.sin( theta ) + yp * np.cos( theta )

        x += sources.x[dex]
        y += sources.y[dex]

        return x, y, dex


    def create_new_datastore( self, ds, fakeprov ):
        fakeds = DataStore()

        # Copy the reporting conditions
        fakeds.update_runtimes = ds.update_runtimes
        fakeds.update_memory_usages = ds.update_memory_usages

        # Make the provenance tree right

        fakeds.edit_prov_tree( ProvenanceTree( ds.prov_tree, ds.prov_tree.upstream_steps ) )
        fakeds.prov_tree.upstream_steps['fakeinjection'] = [ 'photocal' ]
        if 'subtraction' in fakeds.prov_tree.upstream_steps:
            fakeds.prov_tree.upstream_steps['subtraction'] = [ 'referencing', 'fakeinjection' ]
        fakeds.edit_prov_tree( 'fakeinjection', prov=fakeprov, new_step=True )

        # Copy of all data products.  For Image, we need to make a new
        #   one, because we're going to inject fakes.  Make sure it
        #   doesn't have the same uuid as ds.image so we don't
        #   accidentally overwrite it.  For the other data products,
        #   we're just going to use them as-is, so we don't have to
        #   waste memory doing a deep copy, just point back to the data
        #   product in ds.  (fakeds.image and following data products
        #   are *not* saved to the disk or database, except for
        #   ds.fakeanal, so we don't need to worry about the upstreams
        #   of things like fakeds.sources being wrong.)

        fakeds.image = Image.copy_image( ds.image, no_copy_data=True )
        fakeds.image._id = uuid.uuid4()
        fakeds.sources = ds.sources
        fakeds.psf = ds.psf
        fakeds.bg = ds.bg
        fakeds.wcs = ds.wcs
        fakeds.zp = ds.zp
        fakeds.reference = ds.reference

        # Copy aligned images over to fakeds
        # Assuming that we aligned new to ref here
        fakeds.aligned_new_image = fakeds.image
        fakeds.aligned_new_sources = ds.sources
        fakeds.aligned_new_psf = ds.psf
        fakeds.aligned_new_bg = ds.bg
        fakeds.aligned_new_zp = ds.zp
        fakeds.aligned_ref_image = ds.aligned_ref_image
        fakeds.aligned_ref_sources = ds.aligned_ref_sources
        fakeds.aligned_ref_psf = ds.aligned_ref_psf
        fakeds.aligned_ref_bg = ds.aligned_ref_bg
        fakeds.aligned_ref_zp = ds.aligned_ref_zp

        return fakeds


    def run( self, *args, **kwargs ):
        """Inject fakes on to an image.

        Unlike most pipeline objects, this will not return the passed
        DataStore.  It will always make a new datastore, but then point
        the sources, bg, wcs, and zp properties to the same objects as
        the passed DataStore.  It will make a new image object that's a
        clone of the image object in the data store, but with fakes
        injected on to the image and weight data, and with its id and
        provenance updated so as not to accidentally ovefwrite the real
        image.  It will also adjust the ProvenanceTree so the upstream
        of the subtraction includes the fake process, with downstreams
        updated accordingly.

        To work, the passed data store must have all of its image,
        sources, bg, wcs, zp, and reference fields set (i.e. not be
        None).

        The returned data store should *not* be saved anywhere.  We
        don't save fakes to the database.

        WARNING : this code (in particular in create_new_datastore)
        assumes that the provenance tree upstreams of the DataStore
        matches the default that's created in DataStore.make_prov_tree.

        """

        self.has_recalculated = False

        ds = None
        try:
            t_start = time.perf_counter()
            ds = DataStore.from_args( *args, **kwargs )
            if ds.update_memory_usages:
                import tracemalloc
                tracemalloc.reset_peak()

            image = ds.get_image()
            zp = ds.get_zp()
            if zp is None:
                raise ValueError( "Need to be able to get ds.zp for fake definition" )

            if self.pars.mag_prob_ratio < 0.:
                raise ValueError( f'mag_prob_ratio must be positive, but is {self.pars.mag_prob_ratio}' )
            if self.pars.min_fake_mag >= self.pars.max_fake_mag:
                raise ValueError( f'min_fake_mag must be < max_fake_mag, but got (min, max) = '
                                  f'({self.pars.min_fake_mag}, {self.pars.max_fake_mag})' )

            # Figure out our random seed
            random_seed = self.pars.random_seed
            if random_seed == 0:
                rng = np.random.default_rng()
                random_seed = rng.integers( 2147483647 )

            # get provenance for this step.  It's not in the DataStore's
            #   provenance tree because of the whole handling of
            #   random_seed.
            # NOTE: we're going to be creating lots of new provenances
            #   if we use a random random seed (i.e. self.pars.ranadom_seed=0).
            #   This might create performace issues; see Issue #416.
            params = self.pars.get_critical_pars()
            params['random_seed'] = random_seed
            zpprov = Provenance.get( zp.provenance_id )
            prov = Provenance( code_version_id=Provenance.get_code_version().id,
                               process=self.pars.get_process_name(),
                               parameters=params,
                               upstreams=[zpprov] )
            prov.insert_if_needed()

            origds = ds
            ds = self.create_new_datastore( origds, prov )

            # Look for an existing FakeSet

            with SmartSession() as session:
                fakes = ( session.query( FakeSet )
                          .filter( FakeSet.zp_id==zp.id )
                          .filter( FakeSet.provenance_id==prov.id )
                         ).first()
                if fakes is not None:
                    ds.fakes = fakes
                    return ds

            # Didn't find an existing fake set, so make one

            self.has_recalculated = True
            fakes = FakeSet( zp_id=zp.id, provenance_id=prov.id )
            fakes.random_seed = random_seed
            fakes.fake_x = np.zeros( self.pars.num_fakes )
            fakes.fake_y = np.zeros( self.pars.num_fakes )
            fakes.fake_mag = np.zeros( self.pars.num_fakes )
            fakes.host_dex = np.zeros( self.pars.num_fakes, dtype=int )

            # Probability distribution is:
            #    f(m) = b + s * m   for m0 ≤ m ≤ m1, 0 otherwise
            # Cumulative distribution is:
            #    F(m) = b * (m - m0) + s/2 * (m² - m0²)
            # Inverse CDF:
            #    m = -b/s ± sqrt( b² - 4 * s/2 * ( -b*m0 - s/2 * m0² - F(m) ) ) / s
            #    m = -b/s ± sqrt( b² + 4 * s/2 * ( s/2 * m0² + b*m0 + F(m) ) ) / s
            #    m = -b/s ± sqrt( b² + (s m0)² + 2 s ( b m0 + F(m) ) ) / s
            #    ...pick the + of ± to get positive magnitudes.

            # Probability distribution is defined by m0, m1, r, such that:
            #    f(m1) = r f(m0)
            # It is normalized between m0 and m1:
            #    b (m1 - m0) + s/2 (m1² - m0²) = 1
            # solve, get:
            r = self.pars.mag_prob_ratio
            m0 = self.pars.min_fake_mag
            m1 = self.pars.max_fake_mag
            if self.pars.mag_rel_limmag:
                m0 += image.lim_mag_estimate
                m1 += image.lim_mag_estimate
            if r != 1:
                b = 2 * (m1 - r*m0) / ( (r+1) * (m1-m0)**2 )
                s = 2 * (r-1) / ( (r+1) * (m1-m0)**2 )

            def m_of_F( F ):
                if r == 1.:
                    return m0 + F * ( m1 - m0 )
                else:
                    return ( -b + np.sqrt( b**2 + (s * m0)**2 + 2*s*(b*m0 + F) ) ) / s

            rng = np.random.default_rng( random_seed )
            if self.pars.hostless_frac > 0.:
                nx = fakes.image.data.shape[1]
                ny = fakes.image.data.shape[0]

            self._hosts_used = None
            for i in range( self.pars.num_fakes ):
                m = m_of_F( rng.uniform() )

                if rng.uniform() < self.pars.hostless_frac:
                    x = rng.uniform( 0., float(nx) )
                    y = rng.uniform( 0., float(ny) )
                    hostdex = -1
                else:
                    x, y, hostdex = self.place_fake_on_host( ds, m, rng )

                fakes.fake_x[i] = x
                fakes.fake_y[i] = y
                fakes.fake_mag[i] = m
                fakes.host_dex[i] = hostdex

            ds.fakes = fakes

            ds.image.data, ds.image.weight = ds.fakes.inject_on_to_image()
            ds.image.flags = origds.image.flags

            if ds.update_runtimes:
                ds.runtimes['fakeinjection'] = time.perf_counter() + t_start
            if ds.update_memory_usages:
                import tracemalloc
                ds.memory_usages['fakeinjection'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            return ds

        except Exception as e:
            SCLogger.exception( f"Exception in FakeInjector.run: {e}" )
            if ds is not None:
                ds.exceptions.append( e )
            raise


    def analyze_fakes( self, ds, origds ):
        """Determine which fakes that were detected, and get measurements attributes of them.

        Parameters
        ----------
          ds : DataStore
            A DataStore fully processed through scoring.  Must have a
            property "fakes" that is the FakeSet we're analyzing.  The
            subtraction should have been done on an image with that
            fakeset injected on to the image.

          origds : DataStore
            A DataStore fully processed through scoring.  This is the
            DataStore of the subtraction (and following) on the original
            image, *not* the image with injected fakes.  The parameters of
            all the pipeline steps in ds and origds should be the same.

        Returns
        -------
          FakeAnalysis

        """

        fakesetprov = Provenance.get( ds.fakes.provenance_id )
        origdeepscoresetprov = Provenance.get( origds.get_deepscore_set().provenance_id )
        prov = Provenance(
            code_version_id = fakesetprov.code_version_id,
            process = 'fakeanalysis',
            params={},
            upstreams=[ fakesetprov, origdeepscoresetprov ]
        )
        prov.insert_if_needed()
        fakeanal = FakeAnalysis( fakeset_id=ds.fakes.id,
                                 orig_deepscore_set_id=origds.deepscore_set.id,
                                 provenance_id=prov.id )
        fakeanal.is_detected = np.full( ds.fakes.fake_x.shape, False, dtype=bool )
        fakeanal.is_kept = np.full( ds.fakes.fake_x.shape, False, dtype=bool )
        fakeanal.is_bad = np.full( ds.fakes.fake_x.shape, False, dtype=bool )
        fakeanal.nbadpix = np.full( ds.fakes.fake_x.shape, -32767, dtype=int )
        fakeanal.psf_fit_flags = np.full( ds.fakes.fake_x.shape, 0, dtype=int )
        fakeanal.center_x_pixel = np.full( ds.fakes.fake_x.shape, -32767, dtype=int )
        fakeanal.center_y_pixel = np.full( ds.fakes.fake_x.shape, -32767, dtype=int )
        fakeanal.deepscore_algorithm = np.full( ds.fakes.fake_x.shape, 0, dtype=int )
        for prop in [ 'flux_psf', 'flux_psf_err', 'bkg_per_pix', 'best_aperture',
                      'x', 'y', 'gfit_x', 'gfit_y', 'major_width', 'minor_width',
                      'position_angle', 'negfrac', 'negfluxfrac', 'score' ]:
            setattr( fakeanal, prop, np.full( ds.fakes.fake_x.shape, np.nan, dtype=np.float32 ) )


        # Find the index into measurements and into all_measurements that correspond
        # to each fake
        allmeasx = np.array( [ m.x for m in ds.all_measurements ] )
        allmeasy = np.array( [ m.y for m in ds.all_measurements ] )
        allmeasdist = np.sqrt( ( ds.fakes.fake_x[ :, np.newaxis ] - allmeasx[ np.newaxis, : ] ) ** 2  +
                               ( ds.fakes.fake_y[ :, np.newaxis ] - allmeasy[ np.newaxis, : ] ) ** 2 )
        measindex = np.array( [ m.index_in_sources for m in ds.measurements ] )

        for n, (fake_x, fake_y), in enumerate( zip( ds.fakes.fake_x, ds.fakes.fake_y ) ):
            allmeasmatch = np.where( allmeasdist[ n, : ] < self.pars.detection_pixel_range )[0]
            if len(allmeasmatch) == 0:
                continue
            if len( allmeasmatch ) > 1:
                # Pick the closest one
                dex = np.argmin( allmeasdist[ n, : ][ allmeasmatch ] )
                allmeasmatch = allmeasmatch[ dex ]
            else:
                allmeasmatch = allmeasmatch[ 0 ]

            fakeanal.is_detected[ n ] = True
            for prop in [ 'is_bad', 'nbadpix', 'psf_fit_flags', 'center_x_pixel', 'center_y_pixel',
                          'flux_psf', 'flux_psf_err', 'bkg_per_pix', 'best_aperture',
                          'x', 'y', 'gfit_x', 'gfit_y', 'major_width', 'minor_width',
                          'position_angle', 'negfrac', 'negfluxfrac' ]:
                getattr( fakeanal, prop )[n] = getattr( ds.all_measurements[ allmeasmatch ], prop )

            measmatch = np.where( measindex == ds.all_measurements[allmeasmatch].index_in_sources )[0]
            if len( measmatch ) == 0 :
                continue
            if len( measmatch ) > 1:
                raise RuntimeError( "This should never happen." )
            measmatch = measmatch[0]

            fakeanal.is_kept[ n ] = True
            fakeanal.deepscore_algorithm[ n ] = ds.deepscore_set._algorithm
            fakeanal.score[ n ] = ds.deepscores[ measmatch ].score

        return fakeanal
