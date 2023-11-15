import pytest
import uuid

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import CodeHash, CodeVersion, Provenance


@pytest.mark.xfail
def test_code_versions():
    cv = CodeVersion(version="test_v0.0.1")
    cv.update()

    assert cv.code_hashes is not None
    assert len(cv.code_hashes) == 1
    assert cv.code_hashes[0] is not None
    assert isinstance(cv.code_hashes[0].hash, str)
    assert len(cv.code_hashes[0].hash) == 40

    try:
        with SmartSession() as session:
            session.add(cv)
            session.commit()
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
            session.commit()

            assert len(cv.code_hashes) == 2
            assert cv.code_hashes[0].hash == git_hash
            assert cv.code_hashes[1].hash == old_hash
            assert cv.code_hashes[0].code_version_id == cv.id
            assert cv.code_hashes[1].code_version_id == cv.id

        # check that we can remove commits and have that cascaded
        with SmartSession() as session:
            session.add(cv)  # add it back into the new session
            session.delete(ch2)
            session.commit()
            # This assertion failes with expire_on_commit=False in session creation; have to manually refresh
            session.refresh(cv)
            assert len(cv.code_hashes) == 1
            assert cv.code_hashes[0].hash == git_hash

            # now check the delete orphan
            cv.code_hashes = []
            session.commit()
            assert len(cv.code_hashes) == 0
            orphan_hash = session.scalars(sa.select(CodeHash).where(CodeHash.hash == git_hash)).first()
            assert orphan_hash is None

    finally:
        with SmartSession() as session:
            session.execute(sa.delete(CodeVersion).where(CodeVersion.version == 'test_v0.0.1'))
            session.commit()


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
                parameters={"test_key": "test_value1"},
                upstreams=[],
            )

            # adding the provenance also calculates the hash
            session.add(p)
            session.commit()
            pid1 = p.id
            assert pid1 is not None
            assert isinstance(p.id, str)
            assert len(p.id) == 20
            hash = p.id

            p2 = Provenance(
                code_version=code_version,
                parameters={"test_key": "test_value2"},
                process="test_process",
                upstreams=[],
            )

            # adding the provenance also calculates the hash
            session.add(p2)
            session.commit()
            pid2 = p2.id
            assert pid2 is not None
            assert isinstance(p2.id, str)
            assert len(p2.id) == 20
            assert p2.id != hash
    finally:
        with SmartSession() as session:
            session.execute(sa.delete(Provenance).where(Provenance.id.in_([pid1, pid2])))
            session.commit()

    # deleting the Provenance does not delete the CodeVersion!
    with SmartSession() as session:
        cv = session.scalars(sa.select(CodeVersion).where(CodeVersion.id == code_version.id)).first()
        assert cv is not None


def test_unique_provenance_hash(code_version):
    parameter = uuid.uuid4().hex
    p = Provenance(
        process='test_process',
        code_version=code_version,
        parameters={'test_key': parameter},
        upstreams=[]
    )

    pid = None
    try:  # cleanup
        with SmartSession() as session:
            session.add(p)
            session.commit()
            pid = p.id
            assert pid is not None
            assert len(p.id) == 20
            hash = p.id

            p2 = Provenance(
                process='test_process',
                code_version=code_version,
                parameters={'test_key': parameter},
                upstreams=[]
            )
            p2.update_id()
            assert p2.id == hash

            with pytest.raises(sa.exc.IntegrityError) as e:
                session.add(p2)
                session.commit()
            assert 'duplicate key value violates unique constraint "pk_provenances"' in str(e)

    finally:
        if pid is not None:
            with SmartSession() as session:
                session.execute(sa.delete(Provenance).where(Provenance.id == pid))
                session.commit()


def test_upstream_relationship( provenance_base, provenance_extra ):
    new_ids = []
    fixture_ids = []

    with SmartSession() as session:
        try:
            session.add(provenance_base)
            session.add(provenance_extra)
            fixture_ids = [provenance_base.id, provenance_extra.id]
            p1 = Provenance(
                process="test_downstream_process",
                code_version=provenance_base.code_version,
                parameters={"test_key": "test_value1"},
                upstreams=[provenance_base],
            )

            session.add(p1)
            session.commit()
            pid1 = p1.id
            new_ids.append(pid1)
            assert pid1 is not None
            assert isinstance(p1.id, str)
            assert len(p1.id) == 20
            hash = p1.id

            p2 = Provenance(
                process="test_downstream_process",
                code_version=provenance_base.code_version,
                parameters={"test_key": "test_value1"},
                upstreams=[provenance_base, provenance_extra],
            )

            session.add(p2)
            session.commit()
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
                parameters={"test_key": "test_value1"},
                process="test_downstream_process",
                upstreams=[],
            )
            p2.upstreams.append(p3)
            session.commit()

            pid3 = p3.id
            assert pid3 is not None
            new_ids.append(pid3)

            p3_recovered = session.scalars(sa.select(Provenance).where(Provenance.id == pid3)).first()
            assert p3_recovered is not None

            # check that the downstreams of our fixture provenances have been updated too
            # base_downstream_ids = [p.id for p in provenance_base.downstreams]
            # assert all([pid in base_downstream_ids for pid in [pid1, pid2]])
            # assert pid2 in [p.id for p in provenance_extra.downstreams]

        finally:
            session.execute(sa.delete(Provenance).where(Provenance.id.in_(new_ids)))
            session.commit()

            fixture_provenances = session.scalars(sa.select(Provenance).where(Provenance.id.in_(fixture_ids))).all()
            assert len(fixture_provenances) == 2
            cv = session.scalars(sa.select(CodeVersion)
                                 .where(CodeVersion.id == provenance_base.code_version.id)).first()
            assert cv is not None

        # # the deletion of the new provenances should have cascaded to the downstreams
        # session.refresh(provenance_base)
        # base_downstream_ids = [p.id for p in provenance_base.downstreams]
        # assert all([pid not in base_downstream_ids for pid in new_ids])
        #
        # session.refresh(provenance_extra)
        # extra_downstream_ids = [p.id for p in provenance_extra.downstreams]
        # assert all([pid not in extra_downstream_ids for pid in new_ids])

def test_recursive_merge( provenance_base ):
    with SmartSession() as session:
        session.add( provenance_base )
        p1 = Provenance( process="test_secondary_process_1",
                         code_version=provenance_base.code_version,
                         parameters={},
                         upstreams=[ provenance_base ],
                         is_testing=True )
        p1.update_id()
        # session.add( p1 )
        p2 = Provenance( process="test_secondary_process_2",
                         code_version=provenance_base.code_version,
                         parmeters={},
                         upstreams=[ p1 ],
                         is_testing=True )
        p2.update_id()
        # session.add( p2 )
        p3 = Provenance( process="test_tertiary_process",
                         code_version=provenance_base.code_version,
                         paremeters={},
                         upstreams=[ p2, p1 ],
                         is_testing=True )
        p3.update_id()
        # session.add( p3 )
        p4 = Provenance( process="test_final_process",
                         code_version=provenance_base.code_version,
                         parmeters={},
                         upstreams=[ p3 ],
                         is_testing=True )
        p4.update_id()
        # session.add( p4 )
        # session.commit()

        # Now, in another session....
        with SmartSession() as different_session:
            merged_p4 = p4.recursive_merge( different_session )
            different_session.add( merged_p4 )
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
