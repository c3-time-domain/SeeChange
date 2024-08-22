import pytest
import uuid

import sqlalchemy as sa
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession
from models.provenance import CodeHash, CodeVersion, Provenance

from util.util import get_git_hash

# Note: ProvenanceTag.newtag is tested as part of pipeline/test_pipeline.py::test_provenance_tree

def test_code_versions( code_version ):
    cv = code_version
    git_hash = get_git_hash()

    # These things won't work if get_git_hash() returns None, because it won't
    #   have a hash to try to add.  So, only run these tests where they might actually pass.
    if git_hash is not None:
        # Make sure we can't update a cv that's not yet in the database
        newcv = CodeVersion( id="this_code_version_does_not_exist_v0.0.1" )
        with pytest.raises( IntegrityError, match='insert or update on table "code_hashes" violates foreign key' ):
            newcv.update()

        # Make sure we have a code hash associated with code_version
        cv.update()
        with SmartSession() as sess:
            n1 = sess.query( CodeHash ).count()
        # Make sure that we can run it again
        cv.update()
        with SmartSession() as sess:
            n2 = sess.query( CodeHash ).count()
        assert n2 == n1

    hashes = cv.get_code_hashes()
    assert set( [ i.id for i in cv.code_hashes ] ) == set( [ i.id for i in hashes ] )
    assert hashes is not None
    if git_hash is not None:
        # There probably won't be a code_hash at all if get_git_hash didn't work.
        #   (Certainly not if the tests started with a clean database as they were supposed to.)
        assert len(hashes) == 1
        assert hashes[0] is not None
        assert isinstance(hashes[0].id, str)
        assert len(cv.code_hashes[0].id) == 40

    # add old hash
    old_hash = '696093387df591b9253973253756447079cea61d'
    try:
        with SmartSession() as session:
            ch = session.scalars(sa.select(CodeHash).where(CodeHash._id == git_hash)).first()
            cv = session.scalars(sa.select(CodeVersion).where(CodeVersion._id == 'test_v1.0.0')).first()
            assert cv is not None
            assert cv.id == code_version.id
            if git_hash is not None:
                assert cv.code_hashes[0].id == ch.id

            ch2 = session.scalars(sa.select(CodeHash).where(CodeHash._id == old_hash)).first()
            assert ch2 is None
            ch2 = CodeHash( id=old_hash, code_version_id=code_version.id )
            session.add( ch2 )
            session.commit()
        with SmartSession() as session:
            cv = session.scalars(sa.select(CodeVersion).where(CodeVersion._id == 'test_v1.0.0')).first()
            assert ch2.id in [ h.id for h in cv.code_hashes ]

    finally:
        # Remove the code hash we added
        with SmartSession() as session:
            session.execute( sa.text( "DELETE FROM code_hashes WHERE _id=:hash" ), { 'hash': old_hash } )


def test_provenances(code_version):
    # cannot create a provenance without a process name
    with pytest.raises( ValueError, match="must have a process name" ):
        Provenance()

    # cannot create a provenance with a code_version of wrong type
    with pytest.raises( ValueError, match="Code version must be a str" ):
        Provenance(process='foo', code_version_id=123)

    pid1 = pid2 = None

    try:

        with SmartSession() as session:
            ninitprovs = session.query( Provenance ).count()

            p = Provenance(
                process="test_process",
                code_version_id=code_version.id,
                parameters={"test_parameter": "test_value1"},
                upstreams=[],
                is_testing=True,
            )
            # hash is calculated in init

            pid1 = p.id
            assert pid1 is not None
            assert isinstance(pid1, str)
            assert len(pid1) == 20

            p2 = Provenance(
                code_version_id=code_version.id,
                parameters={"test_parameter": "test_value2"},
                process="test_process",
                upstreams=[],
                is_testing=True,
            )

            pid2 = p2.id
            assert pid2 is not None
            assert isinstance(p2.id, str)
            assert len(p2.id) == 20
            assert pid2 != pid1

            # Check automatic code version getting
            p3 = Provenance(
                parameters={ "test_parameter": "test_value2" },
                process="test_process",
                upstreams=[],
                is_testing=True
            )

            assert p3.id == p2.id
            assert p3.code_version_id == code_version.id
    finally:
        with SmartSession() as session:
            session.execute(sa.delete(Provenance).where(Provenance._id.in_([pid1, pid2])))
            session.commit()


def test_unique_provenance_hash(code_version):
    parameter = uuid.uuid4().hex
    p = Provenance(
        process='test_process',
        code_version_id=code_version.id,
        parameters={'test_parameter': parameter},
        upstreams=[],
        is_testing=True,
    )

    try:  # cleanup
        p.insert()
        pid = p.id
        assert pid is not None
        assert len(p.id) == 20
        hash = p.id

        p2 = Provenance(
            process='test_process',
            code_version_id=code_version.id,
            parameters={'test_parameter': parameter},
            upstreams=[],
            is_testing=True,
        )
        assert p2.id == hash

        with pytest.raises(sa.exc.IntegrityError) as e:
            p2.insert()
        assert 'duplicate key value violates unique constraint "provenances_pkey"' in str(e)

        p2.insert( _exists_ok=True )

    finally:
        if 'pid' in locals():
            with SmartSession() as session:
                session.execute(sa.delete(Provenance).where(Provenance._id == pid))
                session.commit()


def test_upstream_relationship( provenance_base, provenance_extra ):
    new_ids = []
    fixture_ids = []
    fixture_ids = [provenance_base.id, provenance_extra.id]

    with SmartSession() as session:
        try:
            provenance_base = session.merge(provenance_base)
            provenance_extra = session.merge(provenance_extra)

            assert provenance_extra.id in [ i.id for i in provenance_base.get_downstreams() ]

            p1 = Provenance(
                process="test_downstream_process",
                code_version_id=provenance_base.code_version_id,
                parameters={"test_parameter": "test_value1"},
                upstreams=[provenance_base],
                is_testing=True,
            )
            p1.insert()

            pid1 = p1.id
            new_ids.append(pid1)
            assert pid1 is not None
            assert isinstance(p1.id, str)
            assert len(p1.id) == 20
            hash = p1.id

            p2 = Provenance(
                process="test_downstream_process",
                code_version_id=provenance_base.code_version_id,
                parameters={"test_parameter": "test_value1"},
                upstreams=[provenance_base, provenance_extra],
                is_testing=True,
            )
            p2.insert()

            pid2 = p2.id
            assert pid2 is not None
            new_ids.append(pid2)
            assert isinstance(p2.id, str)
            assert len(p2.id) == 20
            # added a new upstream, so the hash should be different
            assert p2.id != hash

            # check that the downstreams of our fixture provenances have been updated too
            base_downstream_ids = [ p.id for p in provenance_base.get_downstreams() ]
            assert all( [ pid in base_downstream_ids for pid in [pid1, pid2] ] )
            assert pid2 in [ p.id for p in provenance_extra.get_downstreams() ]

        finally:
            session.execute(sa.delete(Provenance).where(Provenance._id.in_(new_ids)))
            session.commit()

            fixture_provenances = session.scalars(sa.select(Provenance).where(Provenance._id.in_(fixture_ids))).all()
            assert len(fixture_provenances) == 2
            cv = session.scalars(sa.select(CodeVersion)
                                 .where(CodeVersion._id == provenance_base.code_version_id)).first()
            assert cv is not None


