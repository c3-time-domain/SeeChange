import json
import hashlib
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from pipeline.utils import get_git_hash

from models.base import Base, SmartSession


class CodeHash(Base):
    __tablename__ = "code_hashes"

    def __init__(self, git_hash):
        self.hash = git_hash

    hash = sa.Column(sa.String, index=True, unique=True)

    code_version_id = sa.Column(sa.Integer, sa.ForeignKey("code_versions.id", ondelete="CASCADE"))

    code_version = relationship("CodeVersion", back_populates="code_hashes", lazy='selectin')


class CodeVersion(Base):
    __tablename__ = 'code_versions'

    version = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        unique=True,
        doc='Version of the code. Can use semantic versioning or date/time, etc. '
    )

    code_hashes = sa.orm.relationship(
        CodeHash,
        back_populates='code_version',
        cascade='all, delete-orphan',
        passive_deletes=True,
        doc='List of commit hashes for this version of the code',
    )

    def update(self, session=None):
        git_hash = get_git_hash()

        if git_hash is None:
            return  # quietly fail if we can't get the git hash
        with SmartSession(session) as session:
            hash_obj = session.scalars(sa.select(CodeHash).where(CodeHash.hash == git_hash)).first()
            if hash_obj is None:
                hash_obj = CodeHash(git_hash)

            self.code_hashes.append(hash_obj)


provenance_self_association_table = sa.Table(
    'provenance_upstreams',
    Base.metadata,
    sa.Column('upstream_id', sa.Integer, sa.ForeignKey('provenances.id', ondelete="CASCADE"), primary_key=True),
    sa.Column('downstream_id', sa.Integer, sa.ForeignKey('provenances.id', ondelete="CASCADE"), primary_key=True),
)


class Provenance(Base):
    __tablename__ = "provenances"

    def __init__(self, process=None, code_version=None, parameters=None, upstreams=None):
        """
        Create a provenance object.

        Parameters
        ----------
        process: str
            Name of the process that created this provenance object.
            Examples can include: "calibration", "subtraction", "source extraction" or just "level1".
        code_version: CodeVersion
            Version of the code used to create this provenance object.
        parameters: dict
            Dictionary of parameters used in the process.
            Include only the critical parameters that affect the final products.
        upstreams: list of Provenance
            List of provenance objects that this provenance object is dependent on.
        """
        if process is None:
            raise ValueError('Provenance must have a process name. ')
        else:
            self.process = process
        if not isinstance(code_version, CodeVersion):
            raise ValueError(f'Code version must be a models.CodeVersion. Got {type(code_version)}.')
        else:
            self.code_version = code_version

        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters

        if upstreams is None:
            self.upstreams = []
        else:
            if not isinstance(upstreams, list):
                self.upstreams = [upstreams]
            if len(upstreams) > 0:
                if isinstance(upstreams[0], Provenance):
                    self.upstreams = upstreams
                else:
                    raise ValueError('upstreams must be a list of Provenance objects')

    process = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Name of the process (pipe line step) that produced these results. "
    )

    code_version_id = sa.Column(
        sa.ForeignKey("code_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of the code version the provenance is associated with. ",
    )

    code_version = relationship(
        "CodeVersion",
        back_populates="provenances",
        cascade="save-update, merge, expunge, refresh-expire",
        lazy='selectin',
    )

    parameters = sa.Column(
        JSONB,
        nullable=False,
        default={},
        doc="Critical parameters used to generate the underlying data. ",
    )

    upstreams = relationship(
        "Provenance",
        secondary=provenance_self_association_table,
        primaryjoin='provenances.c.id == provenance_upstreams.c.downstream_id',
        secondaryjoin='provenances.c.id == provenance_upstreams.c.upstream_id',
        back_populates="downstreams",
        passive_deletes=True,
        lazy='selectin',  # should be able to get upstream_ids without a session!
    )

    downstreams = relationship(
        "Provenance",
        secondary=provenance_self_association_table,
        primaryjoin='provenances.c.id == provenance_upstreams.c.upstream_id',
        secondaryjoin='provenances.c.id == provenance_upstreams.c.downstream_id',
        back_populates="upstreams",
        passive_deletes=True,
        # can add lazy='selectin' here, but probably not need it
    )

    CodeVersion.provenances = relationship(
        "Provenance",
        back_populates="code_version",
        cascade="save-update, merge, expunge, refresh-expire",
        foreign_keys="Provenance.code_version_id",
        passive_deletes=True,
    )

    unique_hash = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        unique=True,
        doc="Unique hash of the code version, parameters and upstream provenances used to generate this dataset. ",
    )

    @property
    def upstream_ids(self):
        if self.upstreams is None:
            return []
        else:
            ids = set([u.id for u in self.upstreams])
            ids = list(ids)
            ids.sort()
            return ids

    def update_hash(self):
        """
        Update the unique_hash using the code_version, parameters and upstream_ids.
        """
        if self.process is None or self.parameters is None or self.upstream_ids is None or self.code_version is None:
            raise ValueError('Provenance must have process, code_version, parameters and upstream_ids defined. ')

        superdict = dict(
            process=self.process,
            parameters=self.parameters,
            upstream_ids=self.upstream_ids,
            code_version=self.code_version.version
        )
        json_string = json.dumps(superdict, sort_keys=True)
        self.unique_hash = hashlib.sha256(json_string.encode("utf-8")).hexdigest()

    @classmethod
    def get_code_version(cls, session=None):
        """
        Get the most relevant or latest code version.
        Tries to match the current git hash with a CodeHash
        instance, but if that doesn't work (e.g., if the
        code is running on a machine without git) then
        the latest CodeVersion is returned.

        Parameters
        ----------
        session: SmartSession
            SQLAlchemy session object. If None, a new session is created,
            and closed as soon as the function finishes.

        Returns
        -------
        code_version: CodeVersion
            CodeVersion object
        """
        code_hash = session.scalars(sa.select(CodeHash).where(CodeHash.hash == get_git_hash())).first()
        if code_hash is not None:
            code_version = code_hash.code_version
        else:
            code_version = session.scalars(sa.select(CodeVersion).order_by(CodeVersion.version.desc())).first()

        return code_version


@event.listens_for(Provenance, "before_insert")
def insert_new_dataset(mapper, connection, target):
    """
    This function is called before a new provenance is inserted into the database.
    It will check all the required fields are populated and update the unique_hash.
    """
    target.update_hash()


