import os

from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy import func

from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import database_exists, create_database

utcnow = func.timezone("UTC", func.current_timestamp())


# this is the root SeeChange folder
CODE_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

# this is where the data lives
# (could be changed for, e.g., new external drive)
DATA_ROOT = os.getenv("SEECHANGE_DATA")
if DATA_ROOT is None:  # TODO: should also check if folder exists?
    DATA_ROOT = os.path.join(CODE_ROOT, "output")

DATA_TEMP = os.path.join(CODE_ROOT, "DATA_TEMP")

# to drop the database use: sudo -u postgres psql -c "DROP DATABASE seechange WITH(force)"

# TODO: check with Rob if he has a preferred way of doing this
# create database here:
# url = "postgresql://postgres:postgres@postgres:5432/seechange"
url = "postgresql://postgres:fragile@seechange_postgres:5432/seechange"
engine = sa.create_engine(url, future=True)
if not database_exists(engine.url):
    create_database(engine.url)

# TODO: we need to decide if we want to make expire_on_commit=False the default
Session = sessionmaker(bind=engine, expire_on_commit=True)


@contextmanager
def SmartSession(input_session=None):
    """
    Retrun a Session() instance that may or may not
    be inside a context manager.

    If the input is already a session, just return that.
    If the input is None, create a session that would
    close at the end of the life of the calling scope.
    """
    # open a new session and close it when outer scope is done
    if input_session is None:
        with Session() as session:
            yield session

    # return the input session with the same scope as given
    elif isinstance(input_session, sa.orm.session.Session):
        yield input_session

    # wrong input type
    else:
        raise TypeError(
            "input_session must be a sqlalchemy session or None"
        )


def safe_mkdir(path):

    allowed_dirs = [
        DATA_ROOT,
        os.path.join(CODE_ROOT, "results"),
        os.path.join(CODE_ROOT, "catalogs"),
        DATA_TEMP,
    ]

    ok = False

    for d in allowed_dirs:
        parent = os.path.realpath(os.path.abspath(d))
        child = os.path.realpath(os.path.abspath(path))

        if os.path.commonpath([parent]) == os.path.commonpath([parent, child]):
            ok = True
            break

    if not ok:
        err_str = "Cannot make a new folder not inside the following folders: "
        err_str += "\n".join(allowed_dirs)
        err_str += f"\n\nAttempted folder: {path}"
        raise ValueError(err_str)

    # if the path is ok, also make the subfolders
    os.makedirs(path, exist_ok=True)


def clear_tables():
    from models.provenance import CodeVersion, Provenance

    try:
        Provenance.metadata.drop_all(engine)
    except:
        pass

    try:
        CodeVersion.metadata.drop_all(engine)
    except:
        pass


class SeeChangeBase:
    """Base class for all SeeChange classes."""

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Unique identifier for this dataset",
    )

    created_at = sa.Column(
        sa.DateTime,
        nullable=False,
        default=utcnow,
        index=True,
        doc="UTC time of insertion of object's row into the database.",
    )

    modified = sa.Column(
        sa.DateTime,
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
        doc="UTC time the object's row was last modified in the database.",
    )


Base = declarative_base(cls=SeeChangeBase)


if __name__ == "__main__":
    pass
