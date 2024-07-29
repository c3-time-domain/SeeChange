import time
import json
import base64
import hashlib
from collections import defaultdict
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import event
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import UniqueConstraint

from util.util import get_git_hash

import models.base
from models.base import Base, AutoIDMixin, SeeChangeBase, SmartSession, safe_merge


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
        join_depth=3,  # how many generations up the upstream chain to load
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
        """Update the id using the code_version, process, parameters and upstream_hashes.
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

    def get_combined_upstream_hash(self):
        """Make a single hash from the hashes of the upstreams.
        This is useful for identifying RefSets.
        """
        json_string = json.dumps(self.upstream_hashes, sort_keys=True)
        return base64.b32encode(hashlib.sha256(json_string.encode("utf-8")).digest()).decode()[:20]

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

    def merge_concurrent(self, session=None, commit=True):
        """Merge the provenance but make sure it doesn't exist before adding it to the database.

        If between the time we check if the provenance exists and the time it is merged,
        another process has added the same provenance, we will get an integrity error.
        This is expected under the assumptions of "optimistic concurrency".
        If that happens, we simply begin again, checking for the provenance and merging it.
        """
        return models.base.merge_concurrent( self, session=session, commit=commit )


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


class ProvenanceTagExistsError(Exception):
    pass

class ProvenanceTag(Base, AutoIDMixin):
    """A human-readable tag to associate with provenances.

    A well-defined provenane tag will have a provenance defined for every step, but there will
    only be a *single* provenance for each step (except for refrenceing, where there could be
    multiple provenances defined).  The class method validate can check this for duplicates.

    """

    __tablename__ = "provenance_tags"

    __table_args__ = ( UniqueConstraint( 'tag', 'provenance_id', name='_provenancetag_prov_tag_uc' ), )

    tag = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc='Human-readable tag name; one tag has many provenances associated with it.'
    )

    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances.id', ondelete="CASCADE", name='provenance_tags_provenance_id_fkey' ),
        index=True,
        doc='Provenance ID.  Each tag/process should only have one provenance.'
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=( "Provenance" )
    )

    def __repr__( self ):
        return ( '<ProvenanceTag('
                 f'tag={self.tag}, '
                 f'provenance_id={self.provenance_id}>' )

    @classmethod
    def newtag( cls, tag, provs, session=None ):
        """Add a new ProvenanceTag.  Will thrown an error if it already exists.

        Usually, this is called from pipeline.top_level.make_provenance_tree, not directly.

        Always commits.

        Parameters
        ----------
          tag: str
            The human-readable provenance tag.  For cleanliness, should be ASCII, no spaces.

          provs: list of str or Provenance
            The provenances to include in this tag.  Usually, you want to make sure to include
            a provenance for every process in the pipeline: exposure, referencing, preprocessing,
            extraction, subtraction, detection, cutting, measuring, [TODO MORE: deepscore, alert]

            -oo- load_exposure, download, import_image, alignment or aligning, coaddition

        """

        with SmartSession( session ) as sess:
            # Get all the provenance IDs we're going to insert
            provids = set()
            for prov in provs:
                if isinstance( prov, Provenance ):
                    provids.add( prov.id )
                elif isinstance( prov, str ):
                    provobj = sess.get( Provenance, prov )
                    if provobj is None:
                        raise ValueError( f"Unknown Provenance ID {prov}" )
                    provids.add( provobj.id )
                else:
                    raise TypeError( f"Everything in the provs list must be Provenance or str, not {type(prov)}" )

            try:
                # Make sure that this tag doesn't already exist.  To avoid race
                #  conditions of two processes creating it at once (which,
                #  given how we expect the code to be used, should probably
                #  not happen in practice), lock the table before searching
                #  and only unlock after inserting.
                sess.connection().execute( sa.text( "LOCK TABLE provenance_tags" ) )
                current = sess.query( ProvenanceTag ).filter( ProvenanceTag.tag == tag )
                if current.count() != 0:
                    sess.rollback()
                    raise ProvenanceTagExistsError( f"ProvenanceTag {tag} already exists." )

                for provid in provids:
                    sess.add( ProvenanceTag( tag=tag, provenance_id=provid ) )

                sess.commit()
            finally:
                # Make sure no lock is left behind; exiting the with block
                #   ought to do this, but be paranoid.
                sess.rollback()

    @classmethod
    def validate( cls, tag, processes=None, session=None ):
        """Verify that a given tag doesn't have multiply defined processes.

        One exception: referenceing can have multiply defined processes.

        Raises an exception if things don't work.

        Parameters
        ----------
          tag: str
            The tag to validate

          processes: list of str
            The processes to make sure are present.  If None, won't make sure
            that any processes are present, will just make sure there are no
            duplicates.

        """

        repeatok = { 'referencing' }

        with SmartSession( session ) as sess:
            ptags = ( sess.query( (ProvenanceTag.id,Provenance.process) )
                      .filter( ProvenanceTag.provenance_id==Provenance.id )
                      .filter( ProvenanceTag.tag==tag )
                     ).all()

        count = defaultdict( lambda: 0 )
        for ptagid, process in ptags:
            count[ process ] += 1

        multiples = [ i for i in count.keys() if count[i] > 1 and i not in repeatok ]
        if len(multiples) > 0:
            raise ValueError( f"Database integrity error: ProcessTag {tag} has more than one "
                              f"provenance for processes {multiples}" )

        if processes is not None:
            missing = [ i for i in processes if i not in count.keys() ]
            if len( missing ) > 0:
                raise ValueError( f"Some processes missing from ProcessTag {tag}: {missing}" )
