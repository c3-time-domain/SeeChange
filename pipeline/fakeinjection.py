import time

import numpy as np

from util.util import env_as_bool
from util.logger import SCLogger

from models.provenance import Provenance
from models.fakeset import FakeSet
from models.base import SmartSession

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore


class ParsFakeInjector(Parameters):
    def __init__( self, **kwargs ):
        super().__init__()

        self.min_fake_mag = self.add_par(
            'min_fake_mag',
            -2.,
            float,
            'Minimum (brightest) magnitude of fake to inject.  Relative to the image zeropoint if '
            'mag_rel_zp is True, otherwise just the apparent magnitude.',
            critical=True
        )

        self.max_fake_mag = self.add_par(
            'max_fake_mag',
            2.,
            float,
            'Maximum (dimmest) magnitude of fake to inject.  Relative to the image zeropoint if '
            'mag_rel_zp is True, otherwise just the apparent magnitude.',
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

        self.host_dmag = self.add_par(
            'host_dmag',
            2.,
            float,
            "Place fakes on hosts that are within ± this many magnitudes of the fake's magnitude.  Make this -1. "
            "to consider all host galaxies without regard to the magnitude of the galaxy.",
            critical=True
        )

        # MORE PARAMETERS ABOUT DISTANCE FROM CENTER ETC. (Issue #410)

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


    def run( self, *args, **kwargs ):
        """Figure out the fakes to inject on to an image.

        Sets ds.fakes to a FakeSet.  Doesn't actually do injection; to
        do that, call ds.fakes.inject_on_to_image()

        """

        self.has_recalculated = False

        ds = None
        try:
            t_start = time.perf_counter()
            if env_as_bool( 'SEECHANGE_TRACEMALLOC' ):
                import tracemalloc
                tracemalloc.reset_peak()

            ds = DataStore.from_args( *args, **kwargs )
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
            params = self.pars.get_critical_pars()
            params['random_seed'] = random_seed
            zpprov = Provenance.get( zp.provenance_id )
            prov = Provenance( code_version_id=Provenance.get_code_version().id,
                               process=self.pars.get_process_name(),
                               params=params,
                               upstreams=[zpprov] )

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

            for i in range( self.pars.num_fakes ):
                m = m_of_F( rng.uniform() )

                if rng.uniform() < self.pars.hostless_frac:
                    x = rng.uniform( 0., float(nx) )
                    y = rng.uniform( 0., float(ny) )
                else:
                    raise NotImplementedError( "Fakes near hosts not implemented yet.  (Issue #410.)" )

                fakes.fake_x[i] = x
                fakes.fake_y[i] = y
                fakes.fake_mag[i] = m

            ds.fakes = fakes

            ds.runtimes['fakeinjection'] = time.perf_counter() + t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['fakeinjection'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            return ds

        except Exception as e:
            SCLogger.exception( f"Exception in FakeInjector.run: {e}" )
            if ds is not None:
                ds.exceptions.append( e )
            raise
