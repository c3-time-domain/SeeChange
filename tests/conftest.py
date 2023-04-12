import pytest
import uuid

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import CodeVersion, Provenance
from models.exposure import Exposure


@pytest.fixture(scope="session", autouse=True)
def code_version():
    cv = CodeVersion(version="test_v1.0.0")
    cv.update()

    yield cv

    with SmartSession() as session:
        session.execute(sa.delete(CodeVersion).where(CodeVersion.version == 'test_v1.0.0'))
        session.commit()


@pytest.fixture
def provenance_base(code_version):
    p = Provenance(
        process="test_base_process",
        code_version=code_version,
        parameters={"test_key": uuid.uuid4().hex},
        upstreams=[],
    )

    with SmartSession() as session:
        session.add(p)
        session.commit()
        pid = p.id

    yield p

    with SmartSession() as session:
        session.execute(sa.delete(Provenance).where(Provenance.id == pid))
        session.commit()


@pytest.fixture
def provenance_extra(code_version, provenance_base):
    p = Provenance(
        process="test_base_process",
        code_version=code_version,
        parameters={"test_key": uuid.uuid4().hex},
        upstreams=[provenance_base],
    )

    with SmartSession() as session:
        session.add(p)
        session.commit()
        pid = p.id

    yield p

    with SmartSession() as session:
        session.execute(sa.delete(Provenance).where(Provenance.id == pid))
        session.commit()


@pytest.fixture
def exposure():

    e = Exposure()

    yield e

    if e.id is not None:
        with SmartSession() as session:
            session.execute(sa.delete(Exposure).where(Exposure.id == e.id))
            session.commit()