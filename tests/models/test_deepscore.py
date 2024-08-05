import uuid
import pytest

import sqlalchemy as sa

from models.base import SmartSession
from models.deepscore import DeepScore

def test_deepscore_saving(ptf_datastore, scorers):

    scorer1 = scorers[0]  # 'random'
    scorer2 = scorers[1]  # 'allperfect'

    scorer1.pars.test_parameter = uuid.uuid4().hex
    scorer2.pars.test_parameter = uuid.uuid4().hex

    ds = ptf_datastore

    m_count = len(ds.measurements)

    try:

        # run with the scorer, using algorithm 'random'
        with SmartSession() as session:

            ds = scorer1.run(ds, session)
            ds = scorer2.run(ds, session)

            ds.save_and_commit(session=session)

        # check that we can query the database for the deepscores
        m_ids = [m.id for m in ds.measurements]
        with SmartSession() as session:
            scores1 = session.scalars(
                sa.select(DeepScore)
                .filter(DeepScore._algorithm == 0)
                .filter(DeepScore.measurements_id.in_(m_ids))
            ).all()

            scores2 = session.scalars(
                sa.select(DeepScore)
                .filter(DeepScore._algorithm == 1)
                .filter(DeepScore.measurements_id.in_(m_ids))
            ).all()
        assert scorer1.has_recalculated
        assert scorer2.has_recalculated
        assert len(scores1) == m_count
        assert len(scores2) == m_count

        # check the logic for running when the scores are already in db

        with SmartSession() as session:
            ds = scorer1.run(ds, session)
            ds.save_and_commit()

        assert not scorer1.has_recalculated
        assert len(ds.scores) == m_count * 2

        with SmartSession() as session:
            scores1 = session.scalars(
                sa.select(DeepScore)
                .filter(DeepScore._algorithm == 0)
                .filter(DeepScore.measurements_id.in_(m_ids))
            ).all()

            allscores = session.scalars(
                sa.select(DeepScore)
                .filter(DeepScore.measurements_id.in_(m_ids))
            ).all()

        assert len(scores1) == m_count
        assert len(allscores) == m_count * 2


    finally:
        return None
    # cleanup happens in ptf_datastore fixture
        if 'ds' in locals():
            ds.delete_everything()
