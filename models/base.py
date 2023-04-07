import os

from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy import func

from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import database_exists, create_database

import util.config as config

utcnow = func.timezone("UTC", func.current_timestamp())


# this is the root SeeChange folder
CODE_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

_engine = None
_Session = None
    
        
@contextmanager
def SmartSession(input_session=None):
    """
    Return a Session() instance that may or may not
    be inside a context manager.

    If the input is already a session, just return that.
    If the input is None, create a session that would
    close at the end of the life of the calling scope.
    """
    global _Session, _engine

    # open a new session and close it when outer scope is done
    if input_session is None:
        if _Session is None:
            cfg = config.Config.get()
            url = ( f'{cfg.value("db.engine")}://{cfg.value("db.user")}:{cfg.value("db.password")}'
                    f'@{cfg.value("db.host")}:{cfg.value("db.port")}/{cfg.value("db.database")}' )
            engine = sa.create_engine( url, future=True, poolclass=sa.pool.NullPool )
            
            _Session = sessionmaker(bind=engine, expire_on_commit=True)
        with _Session() as session:
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

    cfg = config.Config.get()
    allowed_dirs = [
        cfg.value('path.data_root'),
        cfg.value('path.data_temp'),
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
