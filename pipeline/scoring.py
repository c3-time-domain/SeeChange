import time

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.measurements import Measurements
from models.deepscore import DeepScore

from util.util import parse_session, env_as_bool

class ParsScorer(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        # would we rather this be the integer in the enum list instead?
        self.algorithm = self.add_par(
            'algorithm',
            'random',
            str,
            'The name of the algorithm used to generate a score for this object.'
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'scoring'

class Scorer:
    def __init__(self, **kwargs):
        self.pars = ParsScorer(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

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

        # run the process
        try:
            #do a thing
            t_start = time.perf_counter()
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            # NOTE: as in issue, need capability to use multiple provenances
            # For now this is solved by creating multiple scorer objects in top_level, since
            # if we pass all that info down into Scorer, it could cause some provenance trouble.
            # (A 'random' algorithm deepscore created in a pipeline run using only 'random' should
            # have the same provenance as a 'random' algorithm deepscore created in a pipeline run
            # which scored based on 'random' and 'allperfect')
            prov = ds.get_provenance('scoring', self.pars.get_critical_pars(), session=session)

            # find the list of measurements
            measurements = ds.get_measurements(session=session)
            if measurements is None:
                raise ValueError(
                    f'Cannot find a measurements corresponding to the datastore inputs: {ds.get_inputs()}'
                )

            # find if these deepscores have already been made
            scores = ds.get_deepscores( session=session )

            if scores is not None:
                # TODO: consider if comparing something other than the algorithm is better
                # each scorer uniquely uses an algorithm, so any scores using this algo in this ds
                # should only come from this scorer.
                same_algo_scores = [s for s in scores if s.provenance.parameters['algorithm'] == self.pars.algorithm]


            if scores is None or len(same_algo_scores) == 0:

                self.has_recalculated = True
                if hasattr(ds, 'scores') and ds.scores is not None:
                    badscores = [s for s in ds.scores if s.provenance.parameters['algorithm'] == self.pars.algorithm]
                    ds.scores = [s for s in ds.scores if s not in badscores] # take out deepscores matching this algo
                else:
                    ds.scores = []

                newscores = []
                # iterate over the measurements, creating an appropriate DeepScore object for each.
                for m in measurements:

                    # make a deepscore object for a specific measurement
                    newscore = DeepScore.from_measurements(m, provenance=prov)

                    newscore.evaluate_scores() # calculate the rb and ml scores

                    # add it to the list
                    newscores.append(newscore)

                ds.scores.extend(newscores)

            ds.runtimes['scoring'] = time.perf_counter() - t_start
            if env_as_bool('SEECHANGE_TRACEMALLOC'):
                import tracemalloc
                ds.memory_usages['scoring'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

        except Exception as e:
            ds.catch_exception(e)
        finally:  # make sure datastore is returned to be used in the next step
            return ds
