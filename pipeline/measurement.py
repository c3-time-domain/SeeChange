
import sqlalchemy as sa

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.base import SmartSession
from models.cutouts import Cutouts
from models.measurements import Measurements


class ParsMeasurer(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.photometry_method = self.add_par(
            'photometry_method',
            'aperture',
            str,
            'Type of photometry used. Possible values are "psf" and "aperture". '
        )
        self.aperture_radius = self.add_par(
            'aperture_radius',
            3.0,
            [float, list],
            'Radius of the aperture in pixels. Can give a list of values. '
        )  # TODO: should this be in pixels or in arcsec?

        self.real_bogus_version = self.add_par(
            'real_bogus_version',
            None,
            [str, None],
            'Which version of Real/Bogus deep learning code was used. '
            'If None, then deep learning will not be used. '
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'measurement'


class Measurer:
    def __init__(self, **kwargs):
        self.pars = ParsMeasurer(**kwargs)

    def run(self, *args, **kwargs):
        """
        Go over the cutouts from an image and measure all sorts of things
        for each cutout: photemetry (flux, centroids), real/bogus, etc.

        Returns a DataStore object with the products of the processing.
        """
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find some measurements in memory or in the database:
        measurements = ds.get_measurements(prov, session=session)

        if measurements is None:  # must create a new list of Measurements

            # use the latest source list in the data store,
            # or load using the provenance given in the
            # data store's upstream_provs, or just use
            # the most recent provenance for "detection"
            detections = ds.get_detections(session=session)

            if detections is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')

            # TODO: implement the actual code to do this.
            #  For each source in the SourceList make a Cutouts object.
            #  For each Cutouts calculate the photometry (flux, centroids).
            #  Apply analytic cuts to each stamp image, to rule out artefacts.
            #  Apply deep learning (real/bogus) to each stamp image, to rule out artefacts.
            #  Save the results as Measurement objects, append them to the Cutouts objects.
            #  Commit the results to the database.

            # add the resulting list to the data store
            if measurements.provenance is None:
                measurements.provenance = prov
            else:
                if measurements.provenance.unique_hash != prov.unique_hash:
                    raise ValueError('Provenance mismatch for measurements and provenance!')

            ds.measurements = measurements

        # make sure this is returned to be used in the next step
        return ds

