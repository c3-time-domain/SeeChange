import pytest
import uuid

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import CodeHash, CodeVersion, Provenance


@pytest.mark.xfail
def test_code_versions():
    cv = CodeVersion(id="test_v0.0.1")
    cv.update()

    assert cv.code_hashes is not None
    assert len(cv.code_hashes) == 1
    assert cv.code_hashes[0] is not None
    assert isinstance(cv.code_hashes[0].hash, str)
    assert len(cv.code_hashes[0].hash) == 40

    try:
        with SmartSession() as session:
            session.add(cv)
            session.flush()
            cv_id = cv.id
            git_hash = cv.code_hashes[0].hash
            assert cv_id is not None

        with SmartSession() as session:
            ch = session.scalars(sa.select(CodeHash).where(CodeHash.hash == git_hash)).first()
            cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.version == 'test_v0.0.1')).first()
            assert cv is not None
            assert cv.id == cv_id
            assert cv.code_hashes[0].id == ch.id

        # add old hash
        old_hash = '696093387df591b9253973253756447079cea61d'
        ch2 = session.scalars(sa.select(CodeHash).where(CodeHash.hash == old_hash)).first()
        if ch2 is None:
            ch2 = CodeHash(old_hash)
        cv.code_hashes.append(ch2)

        with SmartSession() as session:
            session.add(cv)
            session.flush()

            assert len(cv.code_hashes) == 2
            assert cv.code_hashes[0].hash == git_hash
            assert cv.code_hashes[1].hash == old_hash
            assert cv.code_hashes[0].code_version_id == cv.id
            assert cv.code_hashes[1].code_version_id == cv.id

        # check that we can remove commits and have that cascaded
        with SmartSession() as session:
            session.add(cv)  # add it back into the new session
            session.delete(ch2)
            session.flush()
            # This assertion fails with expire_on_commit=False in session creation; have to manually refresh
            session.refresh(cv)
            assert len(cv.code_hashes) == 1
            assert cv.code_hashes[0].hash == git_hash

            # now check the delete orphan
            cv.code_hashes = []
            session.flush()
            assert len(cv.code_hashes) == 0
            orphan_hash = session.scalars(sa.select(CodeHash).where(CodeHash.hash == git_hash)).first()
            assert orphan_hash is None

    finally:
        with SmartSession() as session:
            session.execute(sa.delete(CodeVersion).where(CodeVersion.version == 'test_v0.0.1'))
            session.flush()


def test_provenances(code_version):
    # cannot create a provenance without a process name
    with pytest.raises(ValueError) as e:
        Provenance()
    assert "must have a process name" in str(e)

    # cannot create a provenance without a code version
    with pytest.raises(ValueError) as e:
        Provenance(process='foo')
    assert "Provenance must have a code_version. " in str(e)

    # cannot create a provenance with a code_version of wrong type
    with pytest.raises(ValueError) as e:
        Provenance(process='foo', code_version=123)
    assert "Code version must be a models.CodeVersion" in str(e)

    pid1 = pid2 = None

    try:

        with SmartSession() as session:
            ninitprovs = session.query( Provenance ).count()

            p = Provenance(
                process="test_process",
                code_version=code_version,
                parameters={"test_parameter": "test_value1"},
                upstreams=[],
                is_testing=True,
            )

            # adding the provenance also calculates the hash
            p = session.merge(p)
            session.flush()
            pid1 = p.id
            assert pid1 is not None
            assert isinstance(p.id, str)
            assert len(p.id) == 20
            hash = p.id

            p2 = Provenance(
                code_version=code_version,
                parameters={"test_parameter": "test_value2"},
                process="test_process",
                upstreams=[],
                is_testing=True,
            )

            # adding the provenance also calculates the hash
            p2 = session.merge(p2)
            session.flush()
            pid2 = p2.id
            assert pid2 is not None
            assert isinstance(p2.id, str)
            assert len(p2.id) == 20
            assert p2.id != hash
    finally:
        with SmartSession() as session:
            session.execute(sa.delete(Provenance).where(Provenance.id.in_([pid1, pid2])))


def test_unique_provenance_hash(code_version):
    parameter = uuid.uuid4().hex
    p = Provenance(
        process='test_process',
        code_version=code_version,
        parameters={'test_parameter': parameter},
        upstreams=[],
        is_testing=True,
    )

    try:  # cleanup
        with SmartSession() as session:
            p = session.merge(p)
            session.flush()
            pid = p.id
            assert pid is not None
            assert len(p.id) == 20
            hash = p.id
            session.expunge(p)

            p2 = Provenance(
                process='test_process',
                code_version=code_version,
                parameters={'test_parameter': parameter},
                upstreams=[],
                is_testing=True,
            )
            assert p2.id == hash

            with pytest.raises(sa.exc.IntegrityError) as e:
                session.add(p2)
                session.commit()

            session.rollback()
            session.begin()  # after rollback, we need to start a new transaction
            assert 'duplicate key value violates unique constraint "pk_provenances"' in str(e)

    finally:
        if 'pid' in locals():
            with SmartSession() as session:
                session.execute(sa.delete(Provenance).where(Provenance.id == pid))


def test_upstream_relationship( provenance_base, provenance_extra ):
    new_ids = []
    fixture_ids = []

    with SmartSession() as session:
        try:
            # provenance_base = session.merge(provenance_base)
            # provenance_extra = session.merge(provenance_extra)
            fixture_ids = [provenance_base.id, provenance_extra.id]
            p1 = Provenance(
                process="test_downstream_process",
                code_version=provenance_base.code_version,
                parameters={"test_parameter": "test_value1"},
                upstreams=[provenance_base],
                is_testing=True,
            )

            session.add(p1)
            session.flush()
            pid1 = p1.id
            new_ids.append(pid1)
            assert pid1 is not None
            assert isinstance(p1.id, str)
            assert len(p1.id) == 20
            hash = p1.id

            p2 = Provenance(
                process="test_downstream_process",
                code_version=provenance_base.code_version,
                parameters={"test_parameter": "test_value1"},
                upstreams=[provenance_base, provenance_extra],
                is_testing=True,
            )

            session.add(p2)
            session.flush()
            pid2 = p2.id
            assert pid2 is not None
            new_ids.append(pid2)
            assert isinstance(p2.id, str)
            assert len(p2.id) == 20
            # added a new upstream, so the hash should be different
            assert p2.id != hash

            # check that new provenances get added via relationship cascade
            p3 = Provenance(
                code_version=provenance_base.code_version,
                parameters={"test_parameter": "test_value1"},
                process="test_downstream_process",
                upstreams=[],
                is_testing=True,
            )
            p2.upstreams.append(p3)
            session.flush()

            pid3 = p3.id
            assert pid3 is not None
            new_ids.append(pid3)

            p3_recovered = session.scalars(sa.select(Provenance).where(Provenance.id == pid3)).first()
            assert p3_recovered is not None
            assert p3_recovered is p3

            # check that the downstreams of our fixture provenances have been updated too
            base_downstream_ids = [p.id for p in provenance_base.downstreams]
            assert all([pid in base_downstream_ids for pid in [pid1, pid2]])
            assert pid2 in [p.id for p in provenance_extra.downstreams]

        finally:
            # session.execute(sa.delete(Provenance).where(Provenance.id.in_(new_ids)))
            session.delete(p1)
            session.delete(p2)
            session.delete(p3)
            session.flush()
            # must refresh these because they do not get expired after commit (since expire_on_commit=False)
            # and then their downstream relationships are not automatically updated to show the deletions
            session.refresh(provenance_base)
            session.refresh(provenance_extra)

            fixture_provenances = session.scalars(sa.select(Provenance).where(Provenance.id.in_(fixture_ids))).all()
            assert len(fixture_provenances) == 2
            cv = session.scalars(
                sa.select(CodeVersion).where(CodeVersion.id == provenance_base.code_version.id)
            ).first()
            assert cv is not None


def test_cascade_merge( provenance_base ):
    try:
        with SmartSession() as session:
            session.add( provenance_base )
            p1 = Provenance( process="test_secondary_process_1",
                             code_version=provenance_base.code_version,
                             parameters={'test_parameter': 'test_value'},
                             upstreams=[ provenance_base ],
                             is_testing=True )

            p2 = Provenance( process="test_secondary_process_2",
                             code_version=provenance_base.code_version,
                             parmeters={'test_parameter': 'test_value'},
                             upstreams=[ p1 ],
                             is_testing=True )

            p3 = Provenance( process="test_tertiary_process",
                             code_version=provenance_base.code_version,
                             paremeters={'test_parameter': 'test_value'},
                             upstreams=[ p2, p1 ],
                             is_testing=True )

            p4 = Provenance( process="test_final_process",
                             code_version=provenance_base.code_version,
                             parmeters={'test_parameter': 'test_value'},
                             upstreams=[ p3 ],
                             is_testing=True )

            # this would not actually be a different session
            with SmartSession() as different_session:
                merged_p4 = different_session.merge(p4)

                found = set()
                for obj in different_session:
                    if isinstance( obj, Provenance ):
                        found.add( obj.id )

                for p in [ p1, p2, p3, p4, provenance_base ]:
                    assert p.id in found

                def check_in_session( sess, obj ):
                    assert obj in sess
                    for upstr in obj.upstreams:
                        check_in_session( sess, upstr )

                check_in_session( different_session, merged_p4 )

    finally:
        with SmartSession() as session:
            if 'p1' in locals():
                session.execute(sa.delete(Provenance).where(Provenance.id == p1.id))
            if 'p2' in locals():
                session.execute(sa.delete(Provenance).where(Provenance.id == p2.id))
            if 'p3' in locals():
                session.execute(sa.delete(Provenance).where(Provenance.id == p3.id))
            if 'p4' in locals():
                session.execute(sa.delete(Provenance).where(Provenance.id == p4.id))
