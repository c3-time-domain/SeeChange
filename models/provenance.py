import json
import base64
import hashlib
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from pipeline.utils import get_git_hash

from models.base import Base, SeeChangeBase, SmartSession, safe_merge


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
        # back_populates="downstreams",
        passive_deletes=True,
        lazy='selectin',  # should be able to get upstream_hashes without a session!
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

    is_bad = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Flag to indicate if the provenance is bad and should not be used. ",
    )

    bad_comment = sa.Column(
        sa.String,
        nullable=True,
        doc="Comment on why the provenance is bad. ",
    )

    is_outdated = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Flag to indicate if the provenance is outdated and should not be used. ",
    )

    replaced_by = sa.Column(
        sa.Integer,
        sa.ForeignKey("provenances.id", ondelete="SET NULL"),
        nullable=True,
        doc="ID of the provenance that replaces this one. ",
    )

    is_testing = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Flag to indicate if the provenance is for testing purposes only. ",
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

    @property
    def upstream_hashes(self):
        if self.upstreams is None:
            return []
        else:
            hashes = set([u.unique_hash for u in self.upstreams])
            hashes = list(hashes)
            hashes.sort()
            return hashes

    @classmethod
    def create_or_load( cls, **kwargs ):
        """Return a Provenance object, adding it to the database if it's not there already.

        Paramteres : same as Provenance.__init__

        """
        prov = Provenance( **kwargs )
        prov.update_hash()
        passedsession = None if 'session' not in kwargs.keys() else kwargs['session']
        with SmartSession( passedsession ) as session:
            q = session.query( Provenance ).filter( Provenance.unique_hash==prov.unique_hash )
            existingprov = q.first()
            if existingprov is None:
                session.add( prov )
                if passedsession is None:
                    session.commit()
            else:
                prov = existingprov
        return prov

    def __init__(self, **kwargs):
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
        is_bad: bool
            Flag to indicate if the provenance is bad and should not be used.
        bad_comment: str
            Comment on why the provenance is bad.
        is_testing: bool
            Flag to indicate if the provenance is for testing purposes only.
        is_outdated: bool
            Flag to indicate if the provenance is outdated and should not be used.
        replaced_by: int
            ID of the Provenance object that replaces this one.
        """
        SeeChangeBase.__init__(self)

        if kwargs.get('process') is None:
            raise ValueError('Provenance must have a process name. ')
        else:
            self.process = kwargs.get('process')

        if 'code_version' not in kwargs:
            raise ValueError('Provenance must have a code_version. ')

        code_version = kwargs.get('code_version')
        if not isinstance(code_version, CodeVersion):
            raise ValueError(f'Code version must be a models.CodeVersion. Got {type(code_version)}.')
        else:
            self.code_version = code_version

        self.parameters = kwargs.get('parameters', {})
        upstreams = kwargs.get('upstreams', [])
        if not isinstance(upstreams, list):
            self.upstreams = [upstreams]
        if len(upstreams) > 0:
            if isinstance(upstreams[0], Provenance):
                self.upstreams = upstreams
            else:
                raise ValueError('upstreams must be a list of Provenance objects')

    def __repr__(self):
        return (
            '<Provenance('
            f'id= {self.id}, '
            f'process="{self.process}", '
            f'code_version="{self.code_version.version}", '
            f'parameters={self.parameters}, '
            f'upstreams={[h[:6] for h in self.upstream_hashes]}, '
            f'hash= {self.unique_hash[:6] if self.unique_hash else ""})>'
        )

    def update_hash(self):
        """
        Update the unique_hash using the code_version, parameters and upstream_hashes.
        """
        if self.process is None or self.parameters is None or self.code_version is None:
            raise ValueError('Provenance must have process, code_version, and parameters defined. ')

        superdict = dict(
            process=self.process,
            parameters=self.parameters,
            upstream_hashes=self.upstream_hashes,
            code_version=self.code_version.version
        )
        json_string = json.dumps(superdict, sort_keys=True)

        self.unique_hash = base64.urlsafe_b64encode(hashlib.sha256(json_string.encode("utf-8")).digest()).decode()[:20]

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
        with SmartSession( session ) as session:
            code_hash = session.scalars(sa.select(CodeHash).where(CodeHash.hash == get_git_hash())).first()
            if code_hash is not None:
                code_version = code_hash.code_version
            else:
                code_version = session.scalars(sa.select(CodeVersion).order_by(CodeVersion.version.desc())).first()

        return code_version

    def recursive_merge(self, session, done_list=None):
        """
        Recursively merge this object, its CodeVersion,
        and any upstream/downstream provenances into
        the given session.

        Parameters
        ----------
        session: SmartSession
            SQLAlchemy session object to merge into.

        Returns
        -------
        merged_provenance: Provenance
            The merged provenance object.
        """
        if done_list is None:
            done_list = set()

        if self in done_list:
            return self

        merged_self = safe_merge(session, self)
        done_list.add(merged_self)

        merged_self.code_version = safe_merge(session, merged_self.code_version)

        merged_self.upstreams = [
            u.recursive_merge(session, done_list=done_list) for u in merged_self.upstreams if u is not None
        ]

        return merged_self


@event.listens_for(Provenance, "before_insert")
def insert_new_dataset(mapper, connection, target):
    """
    This function is called before a new provenance is inserted into the database.
    It will check all the required fields are populated and update the unique_hash.
    """
    target.update_hash()


