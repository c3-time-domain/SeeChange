import pytest
import re
import uuid

import sqlalchemy as sa

from astropy.time import Time

from models.base import SmartSession
from models.provenance import Provenance
from models.measurements import Measurements
from models.object import Object


def test_object_creation():
    obj = Object(ra=1.0, dec=2.0, is_test=True, is_bad=False)
    with SmartSession() as session:
        session.add(obj)
        session.commit()
        assert obj.id is not None
        assert obj.name is not None
        assert re.match(r'\w+\d{4}\w+', obj.name)

    with SmartSession() as session:
        obj2 = session.scalars(sa.select(Object).where(Object.id == obj.id)).first()
        assert obj2.ra == 1.0
        assert obj2.dec == 2.0
        assert obj2.name is not None
        assert re.match(r'\w+\d{4}\w+', obj2.name)


@pytest.mark.flaky(max_runs=5)
def test_lightcurves_from_measurements(sim_lightcurves):
    for lc in sim_lightcurves:
        expected_flux = []
        expected_error = []
        measured_flux = []

        for m in lc:
            measured_flux.append(m.flux_apertures[3] - m.background * m.area_apertures[3])
            expected_flux.append(m.sources.data['flux'][m.cutouts.index_in_sources])
            expected_error.append(m.sources.data['flux_err'][m.cutouts.index_in_sources])

        assert len(expected_flux) == len(measured_flux)
        for i in range(len(measured_flux)):
            assert measured_flux[i] == pytest.approx(expected_flux[i], abs=expected_error[i] * 3)


@pytest.mark.flaky(max_runs=5)
def test_filtering_measurements_on_object(sim_lightcurves):
    assert len(sim_lightcurves) > 0
    assert len(sim_lightcurves[0]) > 3

    idx = 0
    mjds = [m.mjd for m in sim_lightcurves[idx]]
    mjds.sort()

    with SmartSession() as session:
        # make a new provenance for the measurements
        prov = Provenance(
            process=sim_lightcurves[idx][0].provenance.process,
            upstreams=sim_lightcurves[idx][0].provenance.upstreams,
            code_version=sim_lightcurves[idx][0].provenance.code_version,
            parameters=sim_lightcurves[idx][0].provenance.parameters.copy(),
            is_testing=True,
        )
        prov.parameters['test_parameter'] = uuid.uuid4().hex
        prov.update_id()
        obj = session.merge(sim_lightcurves[idx][0].object)

        measurements = sim_lightcurves[idx]
        new_measurements = []
        for i, m in enumerate(measurements):
            m2 = Measurements()
            for key, value in m.__dict__.items():
                if key not in [
                    '_sa_instance_state',
                    'id',
                    'created_at',
                    'modified',
                    'from_db',
                    'provenance',
                    'provenance_id',
                    'object',
                    'object_id',
                ]:
                    setattr(m2, key, value)
            m2.provenance = prov
            m2.provenance_id = prov.id
            m2.ra += 0.05 * i / 3600.0  # move the RA by less than one arcsec
            m2.ra = m2.ra % 360.0  # make sure RA is in range
            m2.associate_object(session)
            m2 = session.merge(m2)
            new_measurements.append(m2)

        session.commit()
        session.refresh(obj)
        all_ids = [m.id for m in new_measurements + measurements]
        all_ids.sort()
        assert set([m.id for m in obj.measurements]) == set(all_ids)

    assert all([m.id is not None for m in new_measurements])
    assert all([m.id != m2.id for m, m2 in zip(new_measurements, measurements)])

    with SmartSession() as session:
        obj = session.merge(obj)
        # by default will load the most recently created Measurements objects
        found_ids = [m.id for m in obj.get_measurements_list()]
        assert set(found_ids) == set([m.id for m in new_measurements])

        # explicitly ask for the measurements with the provenance hash
        found_ids = [m.id for m in obj.get_measurements_list(prov_hash_list=[prov.id])]
        assert set(found_ids) == set([m.id for m in new_measurements])

        # ask for the old measurements
        found_ids = [m.id for m in obj.get_measurements_list(prov_hash_list=[measurements[0].provenance.id])]
        assert set(found_ids) == set([m.id for m in measurements])

        # get measurements up to a date:
        found = obj.get_measurements_list(mjd_end=mjds[1])
        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(m.mjd <= mjds[1] for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        # get measurements after a date:
        found = obj.get_measurements_list(mjd_start=mjds[1])
        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(m.mjd >= mjds[1] for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        # check this works with datetimes too
        time_start = Time(mjds[1], format='mjd').isot
        found = obj.get_measurements_list(time_start=time_start)
        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(m.mjd >= mjds[1] for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        time_end = Time(mjds[1], format='mjd').isot
        found = obj.get_measurements_list(time_end=time_end)
        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(m.mjd <= mjds[1] for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        # get measurements between two dates
        t1 = Time(mjds[1] - 0.01, format='mjd').isot
        t2 = Time(mjds[2] + 0.01, format='mjd').isot
        found = obj.get_measurements_list(time_start=t1, time_end=t2)

        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(mjds[1] <= m.mjd <= mjds[2] for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        # get measurements that are very close to the source
        found = obj.get_measurements_list(radius=0.2)  # should include only 1-3 measurements
        assert len(found) > 0
        assert len(found) < len(new_measurements)
        assert all(m.distance_to(obj) <= 0.2 for m in found)
        assert set([m.id for m in found]).issubset(set([m.id for m in new_measurements]))

        # filter on all the offsets disqualifier score
        offsets = [m.disqualifier_scores['offsets'] for m in measurements]  # should be the same as the new measurements
        offsets.sort()
        found = obj.get_measurements_list(thresholds={'offsets': offsets[1]})  # should be only one lower than this
        assert len(found) == 1
        assert found[0].disqualifier_scores['offsets'] < offsets[1]
        assert found[0].id in [m.id for m in new_measurements]

        offsets = [m.disqualifier_scores['offsets'] for m in measurements]  # should be the same as the new measurements
        offsets.sort()
        found = obj.get_measurements_list(thresholds={'offsets': offsets[-1]})  # all but one will pass
        assert len(found) == len(new_measurements) - 1
        assert found[0].disqualifier_scores['offsets'] < offsets[-1]
        for m2 in found:
            assert m2.id in [m.id for m in new_measurements]

        # now give different thresholds for different provenances
        thresholds = {
            measurements[0].provenance.id: {'offsets': offsets[1]},  # will only get one from old list
            prov.id: {'offsets': offsets[-1]},  # should get all but one from the new list, minus one from the old list
        }
        found = obj.get_measurements_list(
            thresholds=thresholds,
            prov_hash_list=[measurements[0].provenance.id, prov.id],  # prefer old measurements
        )
        assert len([m.id for m in found if m.id in [m2.id for m2 in measurements]]) == 1
        assert len([m.id for m in found if m.id in [m2.id for m2 in new_measurements]]) == len(new_measurements) - 2

        # now delete some measurements from each provenance and see if we can get the subsets
        delete_old_id = measurements[0].id
        delete_new_id = new_measurements[-1].id
        old_id_list = [m.id for m in measurements if not m.id == delete_old_id] + [new_measurements[0].id]
        old_id_list.sort()

        new_id_list = [m.id for m in new_measurements if not m.id == delete_new_id] + [measurements[-1].id]
        new_id_list.sort()

        # removing the deleted measurements from the object should remove them from DB as well
        obj.measurements = [m for m in obj.measurements if m.id != delete_old_id and m.id != delete_new_id]
        m = session.scalars(sa.select(Measurements).where(Measurements.id == delete_old_id)).first()
        assert m is None
        m = session.scalars(sa.select(Measurements).where(Measurements.id == delete_new_id)).first()
        assert m is None

        # get the old and only if not found go to the new
        found = obj.get_measurements_list(prov_hash_list=[measurements[0].provenance.id, prov.id])
        assert set([m.id for m in found]) == set(old_id_list)

        # get the new and only if not found go to the old
        found = obj.get_measurements_list(prov_hash_list=[prov.id, measurements[0].provenance.id])
        assert set([m.id for m in found]) == set(new_id_list)


def test_separate_good_and_bad_objects(measurer, ptf_datastore):
    measurements = ptf_datastore.measurements
    m = measurements[0]  # grab the first one as an example

    with SmartSession() as session:
        m = session.merge(m)

        prov=Provenance(
            process=m.provenance.process,
            upstreams=m.provenance.upstreams,
            code_version=m.provenance.code_version,
            parameters=m.provenance.parameters.copy(),
            is_testing=True,
        )
        prov.parameters['test_parameter'] = uuid.uuid4().hex
        prov.update_id()
        obj1 = session.merge(m.object)

        m2 = Measurements()
        for key, value in m.__dict__.items():
            if key not in [
                '_sa_instance_state',
                'id',
                'created_at',
                'modified',
                'from_db',
                'provenance',
                'provenance_id',
                'object',
                'object_id',
            ]:
                setattr(m2, key, value)
        m2.provenance = prov
        m2.provenance_id = prov.id
        m2.is_bad = not m.is_bad # flip the is_bad tag
        m2.associate_object(session)
        m2 = session.merge(m2)
        obj2 = session.merge(m2.object)

        # check we got a new obj, proper badness on each, one of each badness
        assert obj1 is not obj2
        assert obj1.is_bad == m.is_bad
        assert obj2.is_bad == m2.is_bad
        assert not obj1.is_bad == obj2.is_bad
