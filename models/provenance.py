import json
import base64
import hashlib
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from util.util import get_git_hash

from models.base import Base, SeeChangeBase, SmartSession, safe_merge


class CodeHash(Base):
    __tablename__ = "code_hashes"

    def __init__(self, git_hash):
        self.id = git_hash

    id = sa.Column(sa.String, primary_key=True)

    code_version_id = sa.Column(sa.String, sa.ForeignKey("code_versions.id",
                                                         ondelete="CASCADE",
                                                         name='code_hashes_code_version_id_fkey'),
                                index=True )

    code_version = relationship("CodeVersion", back_populates="code_hashes", lazy='selectin')


class CodeVersion(Base):
    __tablename__ = 'code_versions'

    id = sa.Column(
        sa.String,
        primary_key=True,
        nullable=False,
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
            hash_obj = session.scalars(sa.select(CodeHash).where(CodeHash.id == git_hash)).first()
            if hash_obj is None:
                hash_obj = CodeHash(git_hash)

            self.code_hashes.append(hash_obj)


provenance_self_association_table = sa.Table(
    'provenance_upstreams',
    Base.metadata,
    sa.Column('upstream_id',
              sa.String,
              sa.ForeignKey('provenances.id', ondelete="CASCADE", name='provenance_upstreams_upstream_id_fkey'),
              primary_key=True),
    sa.Column('downstream_id',
              sa.String,
              sa.ForeignKey('provenances.id', ondelete="CASCADE", name='provenance_upstreams_downstream_id_fkey'),
              primary_key=True),
)


class Provenance(Base):
    __tablename__ = "provenances"

    __mapper_args__ = {
        "confirm_deleted_rows": False,
    }

    id = sa.Column(
        sa.String,
        primary_key=True,
        nullable=False,
        doc="Unique hash of the code version, parameters and upstream provenances used to generate this dataset. ",
    )

    process = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Name of the process (pipe line step) that produced these results. "
    )

    code_version_id = sa.Column(
        sa.ForeignKey("code_versions.id", ondelete="CASCADE", name='provenances_code_version_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the code version the provenance is associated with. ",
    )

    code_version = relationship(
        "CodeVersion",
        back_populates="provenances",
        cascade="save-update, merge, expunge, refresh-expire",
        passive_deletes=True,
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
        passive_deletes=True,
        cascade="save-update, merge, expunge, refresh-expire",
        lazy='selectin',  # should be able to get upstream_hashes without a session!
    )

    downstreams = relationship(
        "Provenance",
        secondary=provenance_self_association_table,
        primaryjoin='provenances.c.id == provenance_upstreams.c.upstream_id',
        secondaryjoin='provenances.c.id == provenance_upstreams.c.downstream_id',
        passive_deletes=True,
        cascade="delete",
        overlaps="upstreams",
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
        sa.String,
        sa.ForeignKey("provenances.id", ondelete="SET NULL", name='provenances_replaced_by_fkey'),
        nullable=True,
        index=True,
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
        return self.upstream_ids  # hash and ID are the same now

    @property
    def downstream_ids(self):
        if self.downstreams is None:
            return []
        else:
            ids = set([u.id for u in self.downstreams])
            ids = list(ids)
            ids.sort()
            return ids

    @property
    def downstream_hashes(self):
        return self.downstream_ids  # hash and ID are the same now

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
        if upstreams is None:
            self.upstreams = []
        elif not isinstance(upstreams, list):
            self.upstreams = [upstreams]
        else:
            self.upstreams = upstreams

        self.is_bad = kwargs.get('is_bad', False)
        self.bad_comment = kwargs.get('bad_comment', None)
        self.is_testing = kwargs.get('is_testing', False)

        self.update_id()  # too many times I've forgotten to do this!

    def __repr__(self):
        try:
            upstream_hashes = [h[:6] for h in self.upstream_hashes]
        except:
            upstream_hashes = '[...]'

        return (
            '<Provenance('
            f'id= {self.id[:6] if self.id else "<None>"}, '
            f'process="{self.process}", '
            f'code_version="{self.code_version.id}", '
            f'parameters={self.parameters}, '
            f'upstreams={upstream_hashes})>'
        )

    def __setattr__(self, key, value):
        if key in ['upstreams', 'downstreams']:
            if value is None:
                super().__setattr__(key, [])
            elif isinstance(value, list):
                if not all([isinstance(u, Provenance) for u in value]):
                    raise ValueError(f'{key} must be a list of Provenance objects')

                # make sure no duplicate upstreams are added
                hashes = set([u.id for u in value])
                new_list = []
                for p in value:
                    if p.id in hashes:
                        new_list.append(p)
                        hashes.remove(p.id)

                super().__setattr__(key, new_list)
            else:
                raise ValueError(f'{key} must be a list of Provenance objects')
        else:
            super().__setattr__(key, value)

    def update_id(self):
        """
        Update the id using the code_version, parameters and upstream_hashes.
        """
        if self.process is None or self.parameters is None or self.code_version is None:
            raise ValueError('Provenance must have process, code_version, and parameters defined. ')

        superdict = dict(
            process=self.process,
            parameters=self.parameters,
            upstream_hashes=self.upstream_hashes,  # this list is ordered by upstream ID
            code_version=self.code_version.id
        )
        json_string = json.dumps(superdict, sort_keys=True)

        self.id = base64.b32encode(hashlib.sha256(json_string.encode("utf-8")).digest()).decode()[:20]


    # This next thing wasn't working... it was getting detached session
    #  errors when trying to lazy load sqlalchemy relationship things.
    # We need to get rid of all of those.  They're just not consistent
    #  with not holding open sessions.
    # def update_id_and_sync_to_database( self, session=None ):
    #     """Does a merge to the database, trying to mitigate race conditions

    #     If multiple processes are running at once, there will be race
    #     conditions in database merges when new provenances are created.
    #     If more than one process create the same provenance, inside
    #     SQLAlchemy's merge() a second process could create the
    #     provenance in between when this process detects it's not there
    #     and then creates the provenance.  This routine tries to work
    #     around that by manually doing some of the merge itself.  It
    #     looks for a provenance; if not there, it tries to make a new one
    #     and merge it.  If there's an exception in that process, it
    #     iterates; hopefully, next time around, it will just find the one
    #     that was already created by another process.  After five retries
    #     (which should be 4 more than needed...), it gives up and throws
    #     the exception.

    #     Includes a call to update_id().

    #     Parmeters
    #     ---------
    #     session : Sesssion (optional)
    #         Database session

    #     Returns
    #     -------
    #     newprov : Provenance
    #         A version of self that is (hopefully!) guaranteed to be
    #         saved into the database, and that is merged into the session
    #         if one was passed.

    #     """

    #     self.update_id()

    #     # Have to do this in a different session; committing
    #     #   the passed session might cause trouble, as it may
    #     #   have things that aren't yet in a state to really
    #     #   be saved to the databse (e.g. images without md5sums).
    #     # A little sad, since we often have too many database
    #     #   connections as it is, but if you're going to work
    #     #   with SQLAlchemy, you're going to have nightmares.
    #     with SmartSession() as othersession:
    #         newprov = None
    #         retries = 5
    #         while newprov is None and retries > 0:
    #             try:
    #                 searchprov = othersession.query( Provenance ).filter( Provenance.id==self.id )
    #                 if searchprov.count() > 0:
    #                     newprov = searchprov.first()
    #                 else:
    #                     othersession.merge( self )
    #                     othersession.commit()
    #             except Exception as ex:
    #                 newprov = None
    #                 othersession.rollback()
    #                 # This may have been a race condition where the same
    #                 #  provenance was created by multiple processes at once,
    #                 #  and a creation by another process fell into the gap
    #                 #  inside session.merge between checking for an
    #                 #  existing db row and adding the row.
    #                 if retries == 0:
    #                     _logger.error( "Repeated failures trying to merge a provenance" )
    #                     raise
    #                 else:
    #                     retries -= 1

    #     with SmartSession( session ) as session:
    #         if self in session:
    #             # Make sure we're going to get the database object in case it was just created!
    #             session.expunge( self )
    #         # We're now very sure it will be there
    #         newprov = session.query( Provenance ).filter( Provenance.id==self.id ).first()
    #     return newprov

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
            code_hash = session.scalars(sa.select(CodeHash).where(CodeHash.id == get_git_hash())).first()
            if code_hash is not None:
                code_version = code_hash.code_version
            else:
                code_version = session.scalars(sa.select(CodeVersion).order_by(CodeVersion.id.desc())).first()
        return code_version


@event.listens_for(Provenance, "before_insert")
def insert_new_dataset(mapper, connection, target):
    """
    This function is called before a new provenance is inserted into the database.
    It will check all the required fields are populated and update the id.
    """
    target.update_id()


CodeVersion.provenances = relationship(
    "Provenance",
    back_populates="code_version",
    cascade="save-update, merge, expunge, refresh-expire, delete, delete-orphan",
    foreign_keys="Provenance.code_version_id",
    passive_deletes=True,
)
