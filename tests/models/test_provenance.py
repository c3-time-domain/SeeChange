import pytest
import uuid
import psycopg2

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import CodeVersion, Provenance, ProvenanceTag


def test_codeversion_uniqueness():

    with SmartSession() as session:
        cv_count_initial = len(session.scalars(sa.select( CodeVersion )).all())

    try:
        cv = CodeVersion( process='cv_test', version_major=-1, version_minor=-1, version_patch=-1)
        cv.insert()
        cv2 = CodeVersion( process='cv_test', version_major=-2, version_minor=-1, version_patch=-1)
        cv2.insert()
        cv3 = CodeVersion( process='cv_test', version_major=-1, version_minor=-2, version_patch=-1)
        cv3.insert()
        cv4 = CodeVersion( process='cv_test', version_major=-1, version_minor=-1, version_patch=-2)
        cv4.insert()

        with SmartSession() as session:
            cv_count_final = len(session.scalars(sa.select( CodeVersion )).all())
        assert cv_count_final - cv_count_initial == 4

        with pytest.raises( psycopg2.errors.UniqueViolation,
                           match='violates unique constraint "_codeversion_process_versions_uc"'):
            cv5 = CodeVersion( process='cv_test', version_major=-1, version_minor=-1, version_patch=-1)
            cv5.insert()

    finally:
        with SmartSession() as session:
            session.execute( sa.text( "DELETE FROM code_versions WHERE process=:process" ), { 'process': 'cv_test' } )
            session.commit()


def test_provenances():
    # cannot create a provenance without a process name
    with pytest.raises( ValueError, match="must have a process name" ):
        Provenance()

    # cannot create a provenance with a code_version of wrong type
    with pytest.raises( ValueError, match="Code version must be a uuid" ):
        Provenance(process='foo', code_version_id="testvalue")

    pid1 = pid2 = None

    try:

        with SmartSession() as session:
            p = Provenance(
                process="test_process",
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

    finally:
        with SmartSession() as session:
            session.execute(sa.delete(Provenance).where(Provenance._id.in_([pid1, pid2])))
            session.commit()


def test_unique_provenance_hash():
    parameter = uuid.uuid4().hex
    p = Provenance(
        process='test_process',
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


def test_provenance_tag():
    delprovids = set()

    try:
        # These are not valid parameter lists for the various processes,
        # but ProvenanceTag doesn't know anyting about that, so this is
        # all good for the test.
        allprovs = [ Provenance( process='preprocessing', parameters={'a': 1} ),
                     Provenance( process='extraction', parameters={'a': 1} ),
                     Provenance( process='subtraction', parameters={'a': 1} ),
                     Provenance( process='detection', parameters={'a': 1} ),
                     Provenance( process='cutting', parameters={'a': 1} ),
                     Provenance( process='measuring', parameters={'a': 1} ),
                     Provenance( process='scoring', parameters={'a': 1} ),
                     Provenance( process='referencing', parameters={'a': 1} ),
                     Provenance( process='referencing', parameters={'a': 2} )
                    ]

        for p in allprovs:
            p.insert_if_needed()
            delprovids.add( p.id )

        # Make sure we get yelled at if there is a duplicate process
        tmpprovs = allprovs.copy()
        tmpprovs.append( Provenance( process='preprocessing', parameters={'a': 2} ) )
        tmpprovs[-1].insert_if_needed()
        delprovids.add( tmpprovs[-1].id )
        with pytest.raises( ValueError, match='Process preprocessing is in the list of provenances more than once!' ):
            ProvenanceTag.addtag( 'tagtest', tmpprovs )

        # Insert the first few
        provs = allprovs[0:3]
        ProvenanceTag.addtag( 'tagtest', provs )
        with SmartSession() as sess:
            existing = sess.query( ProvenanceTag ).filter( ProvenanceTag.tag=='tagtest' ).all()
            assert len(existing) == 3
            assert set( e.provenance_id for e in existing ) == set( p.id for p in provs )

        # Make sure we get yelled at if we try to add more things but we said to yell at us for doing that
        with pytest.raises( RuntimeError,
                            match="The following provenances are not associated with provenance tag tagtest:" ):
            ProvenanceTag.addtag( 'tagtest', allprovs, add_missing_processes_to_provtag=False )
        with SmartSession() as sess:
            assert sess.query( ProvenanceTag ).filter( ProvenanceTag.tag=='tagtest' ).count() == 3

        # Make sure that if we try to add something that's inconsistent, nothing gets added
        tmpprovs = allprovs.copy()
        tmpprovs[0] = Provenance( process='preprocessing', parameters={'a': 2} )
        tmpprovs[0].insert_if_needed()
        delprovids.add( tmpprovs[0].id )
        with pytest.raises( RuntimeError,
                            match=( "The following provenances do not match the existing provenance for tag tagtest:\n"
                                    ".*preprocessing" ) ):
            ProvenanceTag.addtag( 'tagtest', tmpprovs )
        with SmartSession() as sess:
            assert sess.query( ProvenanceTag ).filter( ProvenanceTag.tag=='tagtest' ).count() == 3

        # Finally make sure that if we add a happy list where some already exist, it all works
        ProvenanceTag.addtag( 'tagtest', allprovs )
        with SmartSession() as sess:
            existing = sess.query( ProvenanceTag ).filter( ProvenanceTag.tag=='tagtest' ).all()
            assert len(existing) == len(allprovs)
            assert set( e.provenance_id for e in existing ) == set( p.id for p in allprovs )

    finally:
        # Clean up
        with SmartSession() as session:
            session.execute( sa.delete( ProvenanceTag ).where( ProvenanceTag.tag=='tagtest' ) )
            session.execute( sa.delete( Provenance ).where( Provenance._id.in_( delprovids ) ) )
            session.commit()
