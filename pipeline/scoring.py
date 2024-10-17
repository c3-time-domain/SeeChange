
import time

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

import numpy as np
import torch

import RBbot_inference

from models.measurements import Measurements
from models.deepscore import DeepScore
from models.provenance import Provenance
from models.enums_and_bitflags import DeepscoreAlgorithmConverter

from util.config import Config
from util.util import env_as_bool
from util.logger import SCLogger


class ParsScorer(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.algorithm = self.add_par(
            'algorithm',
            'random',
            str,
            'Name of the algorithm used to generate a score for this object.'
            'Valid names can be found in enums_and_bitflags.py.'
        )

        self.rbbot_model_dir = self.add_par(
            'rbbot_model_dir',
            '/seechange/share/RBbot_models',
            str,
            'Directory where RBbot models may be found.',
            critical=False
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'scoring'

class Scorer:
    def __init__(self, **kwargs):
        self.config = Config.get()

        self.pars = ParsScorer( **(self.config.value('scoring')) )
        self.pars.augment( kwargs )

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

    def score_rbbot( self, ds, deepmodel ):
        sources = ds.get_sources( session=session )
        cutouts = ds.get_cutouts( session=session )
        cutouts.load_all_co_data( sources=sources )

        # Construct the numpy array
        # Current RBbot models assume 41×41 cutouts

        import pdb; pdb.set_trace()

        RBbot_inference.load_model( deepmodel, model_root=self.pars.rbbot_model_dir )



    def run(self, *args, **kwargs):
        """
        Look at the measurements and assign scores based
        on the chosen ML/DL model. Potentially will include an R/B
        score in addition to other scores.
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
            prov = ds.get_provenance('scoring', self.pars.get_critical_pars(), session=session)

            # find the list of measurements
            measurements = ds.get_measurements(session=session)
            if measurements is None:
                raise ValueError(
                    f'Cannot find a measurements corresponding to '
                    f'the datastore inputs: {ds.get_inputs()}'
                )

            # find if these deepscores have already been made
            scores = ds.get_scores( prov, session = session, reload=True )

            if scores is None or len(scores) == 0:
                self.has_recalculated = True
                algo = Provenance.get( d.provenance_id ).parameters['algorithm']

                if ( algo == 'random' ) or ( algo =='allperfect' ):
                    scorelist = []
                    for m in measurements:
                        d = DeepSCore.from_measurements( m, provenance=prov )

                        if algo == 'random':
                            d.score = np.random.default_rng().random()
                            d.algorithm = algo

                        elif algo == 'allperfect':
                            d.score = 1.0
                            d.algorithm = algo

                        # add it to the list
                        scorelist.append( d )

                elif algo[0:5] == 'RBbot':
                    scorelist = self.score_rbbot( ds, algo )

                elif algo in DeepscoreAlgorithmConverter.dict_inverse:
                    raise NotImplementedError(f"algorithm {algo} isn't yet implemented")

                else:
                    raise ValueError(f"{algo} is not a valid ML algorithm.")


                scores = scorelist

            #   regardless of whether we loaded or calculated the scores, we need
            # to update the bitflag

            # NOTE: zip only works since get_scores ensures score are sorted to measurements
            for score, measurement in zip( scores, measurements ):
                score._upstream_bitflag = 0
                score._upstream_bitflag |= measurement.bitflag

            # add the resulting scores to the ds

            for score in scores:
                if score.provenance_id is None:
                    score.provenance_id = prov.id
                else:
                    if score.provenance_id != prov.id:
                        raise ValueError(
                                f'Provenance mismatch for cutout {score.provenance.id[:6]} '
                                f'and preset provenance {prov.id[:6]}!'
                            )

            ds.scores = scores

            ds.runtimes['scoring'] = time.perf_counter() - t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['scoring'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2 # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
            return ds
