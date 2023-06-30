
from pipeline.parameters import Parameters
from pipeline.data_store import DataStore

from models.zero_point import ZeroPoint


class ParsPhotCalibrator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_match_catalog = self.add_par(
            'cross_match_catalog',
            'Gaia',
            str,
            'Which catalog should be used for cross matching for photometric calibration. '
        )
        self.add_alias('catalog', 'cross_match_catalog')

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'photo_cal'


class PhotCalibrator:
    def __init__(self, **kwargs):
        self.pars = ParsPhotCalibrator()

    def run(self, *args, **kwargs):
        """
        Extract sources and use their magnitudes to calculate the photometric solution.
        Arguments are parsed by the DataStore.parse_args() method.

        Returns a DataStore object with the products of the processing.
        """
        ds, session = DataStore.from_args(*args, **kwargs)

        # get the provenance for this step:
        prov = ds.get_provenance(self.pars.get_process_name(), self.pars.get_critical_pars(), session=session)

        # try to find the world coordinates in memory or in the database:
        zp = ds.get_zp(prov, session=session)

        if zp is None:  # must create a new ZeroPoint object

            # use the latest source list in the data store,
            # or load using the provenance given in the
            # data store's upstream_provs, or just use
            # the most recent provenance for "extraction"
            sources = ds.get_sources(session=session)

            if sources is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.get_inputs()}')

            wcs = ds.get_wcs(session=session)

            if wcs is None:
                raise ValueError(
                    f'Cannot find an astrometric solution corresponding to the datastore inputs: {ds.get_inputs()}'
                )

            # TODO: get the reference catalog and save it in "self"
            # TODO: cross-match the sources with the catalog
            # TODO: save a ZeroPoint object to database
            # TODO: update the image's FITS header with the zp

            zp = ZeroPoint()
            zp.source_list = sources
            if zp.provenance is None:
                zp.provenance = prov
            else:
                if zp.provenance.unique_hash != prov.unique_hash:
                    raise ValueError('Provenance mismatch for zp and provenance!')

            # update the data store with the new ZeroPoint
            ds.zp = zp

        # make sure this is returned to be used in the next step
        return ds
