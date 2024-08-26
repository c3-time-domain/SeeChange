import time
import re
import json
import base64
import hashlib
import uuid
from collections import defaultdict
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.schema import UniqueConstraint

from util.util import get_git_hash
from util.logger import SCLogger

import models.base
from models.base import Base, UUIDMixin, SeeChangeBase, SmartSession


class CodeHash(Base):
    __tablename__ = "code_hashes"

    _id = sa.Column(sa.String, primary_key=True)

    @property
    def id( self ):
        return self._id

    @id.setter
    def id( self, val ):
        self._id = val

    code_version_id = sa.Column(sa.String, sa.ForeignKey("code_versions._id",
                                                         ondelete="CASCADE",
                                                         name='code_hashes_code_version_id_fkey'),
                                index=True )


    @property
    def code_version( self ):
        raise RuntimeError( f"CodeHash.code_version is deprecated, don't use it" )

    @code_version.setter
    def code_version( self, val ):
        raise RuntimeError( f"CodeHash.code_version is deprecated, don't use it" )




class CodeVersion(Base):
    __tablename__ = 'code_versions'

    _id = sa.Column(
        sa.String,
        primary_key=True,
        nullable=False,
        doc='Version of the code. Can use semantic versioning or date/time, etc. '
    )

    @property
    def id( self ):
        return self._id

    @id.setter
    def id( self, val ):
        self._id = val


    # There is a kind of race condition in making this property the way we do, that in practice
    # is not going to matter.  Somebody else could add a new hash to this code version, and we
    # wouldn't get that new hash if we'd called code_hashes before on this code_version object.
    # Not worth worrying about.
    @property
    def code_hashes( self ):
        if self._code_hashes is None:
            self._code_hashes = self.get_code_hashes()
        return self._code_hashes

    def update(self, session=None):
        """Create a new CodeHash object associated with this CodeVersion using the current git hash.

        Will do nothing if it already exists, or if the current git hash can't be determined.

        """
        git_hash = get_git_hash()

        if git_hash is None:
            return  # quietly fail if we can't get the git hash

        hash_obj = CodeHash( _id=git_hash, code_version_id=self.id )
        try:
            hash_obj.insert( session=session )
        except IntegrityError as ex:
            if 'duplicate key value violates unique constraint "code_hashes_pkey"' in str(ex):
                # It's already there, so we don't care.
                pass
            else:
                raise

    def get_code_hashes( self, session=None ):
        """Return all CodeHash objects associated with this codeversion"""
        with SmartSession( session ) as sess:
            hashes = sess.query( CodeHash ).filter( CodeHash.code_version_id==self.id ).all()
        return hashes

    @classmethod
    def get_by_id( cls, cvid, session=None ):
        with SmartSession( session ) as sess:
            cv = sess.query( CodeVersion ).filter( CodeVersion._id == cvid ).first()
        return cv

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self._code_hashes = None

    @orm.reconstructor
    def init_on_load( self ):
        self._code_hashes = None

    def __repr__( self ):
        return f"<CodeVersion {self.id}>"

    # ======================================================================
    # The fields below are things that we've deprecated; these definitions
    #   are here to catch cases in the code where they're still used

    # @property
    # def code_hashes( self ):
    #     raise RuntimeError( f"CodeVersion.code_hashes is deprecated, don't use it" )

    @code_hashes.setter
    def code_hashes( self, val ):
        raise RuntimeError( f"CodeVersion.code_hashes setter is deprecated, don't use it" )

    @property
    def provenances( self ):
        raise RuntimeError( f"CodeVersion.provenances is deprecated, don't use it" )

    @provenances.setter
    def provenances( self, val ):
        raise RuntimeError( f"CodeVersion.provenances is deprecated, don't use it" )


provenance_self_association_table = sa.Table(
    'provenance_upstreams',
    Base.metadata,
    sa.Column('upstream_id',
              sa.String,
              sa.ForeignKey('provenances._id', ondelete="CASCADE", name='provenance_upstreams_upstream_id_fkey'),
              primary_key=True),
    sa.Column('downstream_id',
              sa.String,
              sa.ForeignKey('provenances._id', ondelete="CASCADE", name='provenance_upstreams_downstream_id_fkey'),
              primary_key=True),
)


class Provenance(Base):
    __tablename__ = "provenances"

    __mapper_args__ = {
        "confirm_deleted_rows": False,
    }

    _id = sa.Column(
        sa.String,
        primary_key=True,
        nullable=False,
        doc="Unique hash of the code version, parameters and upstream provenances used to generate this dataset. ",
    )

    @property
    def id( self ):
        if self._id is None:
            self.update_id()
        return self._id

    @id.setter
    def id( self, val ):
        raise RuntimeError( "Don't set Provenance.id directly, use update_id()" )

    process = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Name of the process (pipe line step) that produced these results. "
    )

    code_version_id = sa.Column(
        sa.ForeignKey("code_versions._id", ondelete="CASCADE", name='provenances_code_version_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the code version the provenance is associated with. ",
    )

    parameters = sa.Column(
        JSONB,
        nullable=False,
        server_default='{}',
        doc="Critical parameters used to generate the underlying data. ",
    )

    is_bad = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default='false',
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
        server_default='false',
        doc="Flag to indicate if the provenance is outdated and should not be used. ",
    )

    replaced_by = sa.Column(
        sa.String,
        sa.ForeignKey("provenances._id", ondelete="SET NULL", name='provenances_replaced_by_fkey'),
        nullable=True,
        index=True,
        doc="ID of the provenance that replaces this one. ",
    )

    is_testing = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default='false',
        doc="Flag to indicate if the provenance is for testing purposes only. ",
    )

    @property
    def upstreams( self ):
        if self._upstreams is None:
            self._upstreams = self.get_upstreams()
        return self._upstreams


    def __init__(self, **kwargs):
        """Create a provenance object.

        Parameters
        ----------
        process: str
            Name of the process that created this provenance object.
            Examples can include: "calibration", "subtraction", "source
            extraction" or just "level1".

        code_version_id: str
            Version of the code used to create this provenance object.
            If None, will use Provenance.get_code_version()

        parameters: dict
            Dictionary of parameters used in the process.  Include only
            the critical parameters that affect the final products.

        upstreams: list of Provenance
            List of provenance objects that this provenance object is
            dependent on.

        is_bad: bool
            Flag to indicate if the provenance is bad and should not be
            used.

        bad_comment: str
            Comment on why the provenance is bad.

        is_testing: bool
            Flag to indicate if the provenance is for testing purposes
            only.

        is_outdated: bool
            Flag to indicate if the provenance is outdated and should
            not be used.

        replaced_by: int
            ID of the Provenance object that replaces this one.

        """
        SeeChangeBase.__init__(self)

        if kwargs.get('process') is None:
            raise ValueError('Provenance must have a process name. ')
        else:
            self.process = kwargs.get('process')

        # The dark side of **kwargs when refactoring code...
        #   have to catch problems like this manually.
        if 'code_version' in kwargs:
            raise RuntimeError( 'code_version is not a valid argument to Provenance.__init__; '
                                'use code_version_id' )

        if 'code_version_id' in kwargs:
            code_version_id = kwargs.get('code_version_id')
            if not isinstance(code_version_id, str ):
                raise ValueError(f'Code version must be a str. Got {type(code_version_id)}.')
            else:
                self.code_version_id = code_version_id
        else:
            cv = Provenance.get_code_version()
            self.code_version_id = cv.id

        self.parameters = kwargs.get('parameters', {})
        upstreams = kwargs.get('upstreams', [])
        if upstreams is None:
            self._upstreams = []
        elif not isinstance(upstreams, list):
            self._upstreams = [upstreams]
        else:
            self._upstreams = upstreams
        self._upstreams.sort( key=lambda x: x.id )

        self.is_bad = kwargs.get('is_bad', False)
        self.bad_comment = kwargs.get('bad_comment', None)
        self.is_testing = kwargs.get('is_testing', False)

        self.update_id()  # too many times I've forgotten to do this!

    @orm.reconstructor
    def init_on_load( self ):
        SeeChangeBase.init_on_load( self )
        self._upstreams = None


    def __repr__(self):
        # try:
        #     upstream_hashes = [h[:6] for h in self.upstream_hashes]
        # except:
        #     upstream_hashes = '[...]'

        return (
            '<Provenance('
            f'id= {self.id[:6] if self.id else "<None>"}, '
            f'process="{self.process}", '
            f'code_version="{self.code_version_id}", '
            f'parameters={self.parameters}'
            # f', upstreams={upstream_hashes}'
            f')>'
        )


    @classmethod
    def get( cls, provid, session=None ):
        """Get a provenace given an id, or None if it doesn't exist."""
        with SmartSession( session ) as sess:
            return sess.query( Provenance ).filter( Provenance._id==provid ).first()

    @classmethod
    def get_batch( cls, provids, session=None ):
        """Get a list of provenances given a list of ids."""
        with SmartSession( session ) as sess:
            return sess.query( Provenance ).filter( Provenance._id.in_( provids ) ).all()

    def update_id(self):
        """Update the id using the code_version, process, parameters and upstream_hashes.
        """
        if self.process is None or self.parameters is None or self.code_version_id is None:
            raise ValueError('Provenance must have process, code_version_id, and parameters defined. ')

        superdict = dict(
            process=self.process,
            parameters=self.parameters,
            upstream_hashes=[ u.id for u in self._upstreams ],  # this list is ordered by upstream ID
            code_version=self.code_version_id
        )
        json_string = json.dumps(superdict, sort_keys=True)

        self._id = base64.b32encode(hashlib.sha256(json_string.encode("utf-8")).digest()).decode()[:20]

    @classmethod
    def combined_upstream_hash( self, upstreams ):
        json_string = json.dumps( [ u.id for u in upstreams ], sort_keys=True)
        return base64.b32encode(hashlib.sha256(json_string.encode("utf-8")).digest()).decode()[:20]


    def get_combined_upstream_hash(self):
        """Make a single hash from the hashes of the upstreams.
        This is useful for identifying RefSets.
        """
        return self.__class__.combined_upstream_hash( self.upsterams )


    # This is a cache.  It won't change in one run, so we can save
    #  querying the database repeatedly in get_code_version by saving
    #  the result.
    _current_code_version = None

    @classmethod
    def get_code_version(cls, session=None):
        """Get the most relevant or latest code version.

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

        if Provenance._current_code_version is None:
            code_version = None
            with SmartSession( session ) as session:
                code_hash = session.scalars(sa.select(CodeHash).where(CodeHash._id == get_git_hash())).first()
                if code_hash is not None:
                    code_version = session.scalars( sa.select(CodeVersion)
                                                    .where( CodeVersion._id == code_hash.code_version_id ) ).first()
                if code_version is None:
                    code_version = session.scalars(sa.select(CodeVersion).order_by(CodeVersion._id.desc())).first()
            if code_version is None:
                raise RuntimeError( "There is no code_version in the database.  Put one there." )
            Provenance._current_code_version = code_version

        return Provenance._current_code_version


    def insert( self, session=None, _exists_ok=False ):
        """Insert the provenance into the database.

        Will raise a constraint violation if the provenance ID already exists in the database.

        Parameters
        ----------
          session : SQLAlchmey sesion or None
            Usually you don't want to use this.

        """

        with SmartSession( session ) as sess:
            try:
                SeeChangeBase.insert( self, sess )

                # Should be safe to go ahead and insert into the association table
                # If the provenance already existed, we will have raised an exceptipn.
                # If not, somebody else who might try to insert this provenance
                # will get an exception on the insert() statement above, and so won't
                # try the following association table inserts.

                upstreams = self._upstreams if self._upstreams is not None else self.get_upstreams( session=sess )
                if len(upstreams) > 0:
                    for upstream in upstreams:
                        sess.execute( sa.text( "INSERT INTO provenance_upstreams(upstream_id,downstream_id) "
                                               "VALUES (:upstream,:me)" ),
                                      { 'me': self.id, 'upstream': upstream.id } )
                    sess.commit()
            except IntegrityError as ex:
                if _exists_ok and ( 'duplicate key value violates unique constraint "provenances_pkey"' in str(ex) ):
                    sess.rollback()
                else:
                    raise


    def insert_if_needed( self, session=None ):
        """Insert the provenance into the database if it's not already there.

        Parameters
        ----------
          session : SQLAlchemy session or None
            Usually you don't want to use this

        """

        self.insert( session=session, _exists_ok=True )


    def get_upstreams( self, session=None ):
        with SmartSession( session ) as sess:
            upstreams = ( sess.query( Provenance )
                          .join( provenance_self_association_table,
                                 provenance_self_association_table.c.upstream_id==Provenance._id )
                          .where( provenance_self_association_table.c.downstream_id==self.id )
                          .order_by( Provenance._id )
                         ).all()
            return upstreams

    def get_downstreams( self, session=None ):
        with SmartSession( session ) as sess:
            downstreams = ( sess.query( Provenance )
                            .join( provenance_self_association_table,
                                   provenance_self_association_table.c.downstream_id==Provenance._id )
                            .where( provenance_self_association_table.c.upstream_id==self.id )
                            .order_by( Provenance._id )
                           ).all()
        return downstreams


    # ======================================================================
    # The fields below are things that we've deprecated; these definitions
    #   are here to catch cases in the code where they're still used

    @property
    def code_Version( self ):
        raise RuntimeError( f"Don't use Provenance.code_Version, use code_Version_id" )

    @code_Version.setter
    def code_Version( self, val ):
        raise RuntimeError( f"Don't use Provenance.code_Version, use code_Version_id" )

    @upstreams.setter
    def upstreams( self, val ):
        raise RuntimeError( f"Provenance.upstreams is deprecated, only set it on creation." )

    @property
    def downstreams( self ):
        raise RuntimeError( f"Provenance.downstreams is deprecated, use get_downstreams" )

    @downstreams.setter
    def downstreams( self, val ):
        raise RuntimeError( f"Provenance.downstreams is deprecated, can't be set" )

    @property
    def upstream_ids( self ):
        raise RuntimeError( f"Provenance.upstream_ids is deprecated, use upsterams" )

    @upstream_ids.setter
    def upstream_ids( self, val ):
        raise RuntimeError( f"Provenance.upstream_ids is deprecated, use upstreams" )

    @property
    def downstream_ids( self ):
        raise RuntimeError( f"Provenance.downstream_ids is deprecated, use get_downstreams" )

    @downstream_ids.setter
    def downstream_ids( self, val ):
        raise RuntimeError( f"Provenance.downstream_ids is deprecated, use get_downstreams" )

    @property
    def upstream_hashes( self ):
        raise RuntimeError( f"Provenance.upstream_hashes is deprecated, use upstreams" )

    @upstream_hashes.setter
    def upstream_hashes( self, val ):
        raise RuntimeError( f"Provenance.upstream_hashes is deprecated, use upstreams" )

    @property
    def downstream_hashes( self ):
        raise RuntimeError( f"Provenance.downstream_hashes is deprecated, use get_downstreams" )

    @downstream_hashes.setter
    def downstream_hashes( self, val ):
        raise RuntimeError( f"Provenance.downstream_hashes is deprecated, use get_downstreams" )


class ProvenanceTagExistsError(Exception):
    pass

class ProvenanceTag(Base, UUIDMixin):
    """A human-readable tag to associate with provenances.

    A well-defined provenane tag will have a provenance defined for every step, but there will
    only be a *single* provenance for each step (except for refrenceing, where there could be
    multiple provenances defined).  The class method validate can check this for duplicates.

    """

    __tablename__ = "provenance_tags"

    @declared_attr
    def __table_args__(cls):
        return ( UniqueConstraint( 'tag', 'provenance_id', name='_provenancetag_prov_tag_uc' ), )

    tag = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc='Human-readable tag name; one tag has many provenances associated with it.'
    )

    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances._id', ondelete="CASCADE", name='provenance_tags_provenance_id_fkey' ),
        index=True,
        doc='Provenance ID.  Each tag/process should only have one provenance.'
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

          provs: list of str, UUID, or Provenance
            The provenances to include in this tag.  Usually, you want to make sure to include
            a provenance for every process in the pipeline: exposure, referencing, preprocessing,
            extraction, subtraction, detection, cutting, measuring, [TODO MORE: deepscore, alert]

            -oo- load_exposure, download, import_image, alignment or aligning, coaddition

        """

        # The debug comments in this function are for debugging database
        #   deadlocks.  Uncomment them if you're trying to deal with
        #   that.  Normally they're commented out because they make the
        #   debug output much more verbose.

        with SmartSession( session ) as sess:
            # Get all the provenance IDs we're going to insert
            provids = set()
            for prov in provs:
                if isinstance( prov, Provenance ):
                    provids.add( prov.id )
                elif isinstance( prov, str ) or isinstance( prov, uuid.UUID ):
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
                cls._get_table_lock( sess )
                current = sess.query( ProvenanceTag ).filter( ProvenanceTag.tag == tag )
                if current.count() != 0:
                    # SCLogger.debug( "ProvenanceTag rolling back" )
                    sess.rollback()
                    raise ProvenanceTagExistsError( f"ProvenanceTag {tag} already exists." )

                for provid in provids:
                    sess.add( ProvenanceTag( tag=tag, provenance_id=provid ) )

                # SCLogger.debug( "ProvenanceTag comitting" )
                sess.commit()
            finally:
                # Make sure no lock is left behind; exiting the with block
                #   ought to do this, but be paranoid.
                # SCLogger.debug( "ProvenanceTag rolling back" )
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
            ptags = ( sess.query( (ProvenanceTag._id,Provenance.process) )
                      .filter( ProvenanceTag.provenance_id==Provenance._id )
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
