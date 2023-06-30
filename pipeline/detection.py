
import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.base import SmartSession
from models.image import Image
from models.source_list import SourceList


class ParsDetector(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.subtraction = self.add_par(
            'subtraction',
            False,
            bool,
            'Whether this is expected to run on a subtraction image or a regular image. '
        )

        self.threshold = self.add_par(
            'threshold',
            5.0,
            [float, int],
            'The number of standard deviations above the background '
            'to use as the threshold for detecting a source. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        if self.subtraction:
            return 'detection'
        else:
            return 'extraction'


class Detector:
    def __init__(self, **kwargs):
        self.pars = ParsDetector(**kwargs)

    def run(self, *args, **kwargs):
        """
        Search a regular or subtraction image for new sources, and generate a SourceList.
        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find the sources/detections in memory or in the database:
        if self.pars.subtraction:
            detections = ds.get_detections(prov, session=session)

            if detections is None:
                # load the subtraction image from memory
                # or load using the provenance given in the
                # data store's upstream_provs, or just use
                # the most recent provenance for "subtraction"
                image = ds.get_subtraction(session=session)

                if image is None:
                    raise ValueError(
                        f'Cannot find a subtraction image corresponding to the datastore inputs: {ds.get_inputs()}'
                    )

                detections = self.extract_sources(image)
                detections.image = image

                if detections.provenance is None:
                    detections.provenance = prov
                else:
                    if detections.provenance.unique_hash != prov.unique_hash:
                        raise ValueError('Provenance mismatch for detections and provenance!')

            detections.is_sub = True
            ds.detections = detections

        else:  # regular image
            sources = ds.get_sources(prov, session=session)

            if sources is None:
                # use the latest image in the data store,
                # or load using the provenance given in the
                # data store's upstream_provs, or just use
                # the most recent provenance for "preprocessing"
                image = ds.get_image(session=session)

                if image is None:
                    raise ValueError(f'Cannot find an image corresponding to the datastore inputs: {ds.get_inputs()}')

                sources = self.extract_sources(image)
                sources.image = image
                if sources.provenance is None:
                    sources.provenance = prov
                else:
                    if sources.provenance.unique_hash != prov.unique_hash:
                        raise ValueError('Provenance mismatch for sources and provenance!')

            ds.sources = sources

        # make sure this is returned to be used in the next step
        return ds

    def extract_sources(self, image):
        # TODO: finish this
        # TODO: this should also generate an estimate of the PSF

        sources = SourceList()

        return sources


if __name__ == '__main__':
    from models.base import Session
    from models.provenance import Provenance
    session = Session()
    source_lists = session.scalars(sa.select(SourceList)).all()
    prov = session.scalars(sa.select(Provenance)).all()
