
import time

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

import numpy as np
import torch

import RBbot_inference

from models.deepscore import DeepScore, DeepScoreSet
from models.enums_and_bitflags import DeepscoreAlgorithmConverter

from util.config import Config
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

    def score_rbbot( self, ds, deepmodel, session=None ):
        if len(ds.measurements) == 0:
            SCLogger.debug( "No measurements, returning empty score list" )
            return []

        SCLogger.debug( "score_rbbot starting, loading cutouts data" )

        detections = ds.get_detections( session=session )
        cutouts = ds.get_cutouts( session=session )
        cutouts.load_all_co_data( sources=detections )

        # Construct the numpy array
        # Current RBbot models assume 41×41 cutouts
        if ( cutouts.co_dict[f'source_index_{ds.measurements[0].index_in_sources}']['sub_data'].shape
             != (41,41) ):
            raise ValueError( "RBbot currently requires cutouts to be 41×41" )
        data = np.empty( ( len(ds.measurements), 3, 41, 41 ) )
        tmpdata = np.empty( ( 3, 41, 41 ) )
        tmpmask = np.empty( ( 3, 41, 41 ), dtype=bool )
        for i, meas in enumerate( ds.measurements ):
            cdex = f'source_index_{meas.index_in_sources}'
            tmpdata[ 0, :, : ] = cutouts.co_dict[cdex]['new_data']
            tmpdata[ 1, :, : ] = cutouts.co_dict[cdex]['ref_data']
            tmpdata[ 2, :, : ] = cutouts.co_dict[cdex]['sub_data']

            # Zero out any masked pixels.  (TODO: thought requred.  Is this the right
            # thing to do?  One might expect a pipeline to NaNify masked pixels,
            # and setting NaNs to zero is explicitly required for RBbot....)
            tmpmask[ 0, :, : ] = ( cutouts.co_dict[cdex]['new_flags'] != 0 )
            tmpmask[ 1, :, : ] = ( cutouts.co_dict[cdex]['ref_flags'] != 0 )
            tmpmask[ 2, :, : ] = ( cutouts.co_dict[cdex]['sub_flags'] != 0 )
            tmpdata[ tmpmask ] = 0.

            data[ i, :, :, : ] = tmpdata

        # Make sure there are no nans in data.  (RBbot expects them to have
        # been set to 0.)
        data[ np.isnan( data ) ] = 0.

        # rbbot expects each cutout to be normalized by sqrt(Σf²)
        norm = np.sqrt( ( data*data ).sum( axis=(2,3) ) )
        data /= norm[ :, :, np.newaxis, np.newaxis ]

        # Load the rbbot model
        # TODO : cache this so we don't have to reload it?  Maybe not a big deal,
        #  since in a single run of the pipeline we expet this function to
        #  only be called once.
        SCLogger.debug( f"Loading RBbot model; model={deepmodel}, model_root={self.pars.rbbot_model_dir}" )
        model = RBbot_inference.load_model.load_model( deepmodel, model_root=self.pars.rbbot_model_dir )
        SCLogger.debug( "Mode loaded, running inference." )

        # Run the inference
        trips_tensor = torch.Tensor(data).to('cpu')
        output = model( trips_tensor )
        scores = output[0].detach().cpu().numpy()

        SCLogger.debug( "Building DeepsScore objects" )
        scorelist = []
        for score, meas in zip( scores, ds.measurements ):
            d = DeepScore( index_in_sources=meas.index_in_sources, score=score )
            scorelist.append( d )

        SCLogger.debug( "score_rbbot done" )
        return scorelist


    def run(self, *args, **kwargs):
        """Assign deepscores to measurements.

        Look at the measurements and assign scores based
        on the chosen ML/DL model. Potentially will include an R/B
        score in addition to other scores.
        """
        self.has_recalculated = False

        try:
            ds = DataStore.from_args(*args, **kwargs)
            t_start = time.perf_counter()
            if ds.update_memory_usages:
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            prov = ds.get_provenance('scoring', self.pars.get_critical_pars())

            # find the list of measurements
            measurement_set = ds.get_measurement_set()
            if measurement_set is None:
                raise ValueError( f'Cannot find a measurement set corresponding to '
                                  f'the datastore inputs: {ds.inputs_str}' )
            measurements = measurement_set.measurements

            # find if these deepscores have already been made
            deepscore_set = ds.get_deepscore_set( prov, reload=True )

            if deepscore_set is not None:
                scores = deepscore_set.deepscores
            else:
                self.has_recalculated = True
                algo = self.pars.algorithm
                deepscore_set = DeepScoreSet( measurementset_id=measurement_set.id,
                                              provenance_id=prov.id,
                                              algorithm=algo )

                if ( algo == 'random' ) or ( algo =='allperfect' ):
                    scorelist = []
                    for m in measurements:
                        d = DeepScore( deepscoreset_id=deepscore_set.id, index_in_sources=m.index_in_sources )
                        if algo == 'random':
                            d.score = np.random.default_rng().random()

                        elif algo == 'allperfect':
                            d.score = 1.0

                        # add it to the list
                        scorelist.append( d )

                elif algo[0:5] == 'RBbot':
                    scorelist = self.score_rbbot( ds, algo )
                    for score in scorelist:
                        score.deepscoreset_id = deepscore_set.id

                elif algo in DeepscoreAlgorithmConverter.dict_inverse:
                    raise NotImplementedError(f"algorithm {algo} isn't yet implemented")

                else:
                    raise ValueError(f"{algo} is not a valid ML algorithm.")


                scores = scorelist

            # NOTE: zip only works since deepscore and measurements both order themselves by index_in_sources
            for score, measurement in zip( scores, measurements ):
                score._upstream_bitflag = 0
                score._upstream_bitflag |= measurement.bitflag

            deepscore_set._upstream_bitflag = measurement_set.bitflag
            for d in scores:
                d._upstream_bitflag = measurement_set.bitflag

            deepscore_set.deepscores = scores
            ds.deepscore_set = deepscore_set

            if ds.update_runtimes:
                ds.runtimes['scoring'] = time.perf_counter() - t_start
            if ds.update_memory_usages:
                import tracemalloc
                ds.memory_usages['scoring'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2 # in MB

            return ds

        except Exception as e:
            SCLogger.exception( f"Exception in Scorer.run: {e}" )
            ds.exceptions.append(e)
            raise
