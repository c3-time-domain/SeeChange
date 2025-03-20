import io
import json
import base64
import hashlib
import uuid

import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.schema import UniqueConstraint
import psycopg2.extras
import psycopg2.errors

from util.util import get_git_hash, NumpyAndUUIDJsonEncoder
from util.logger import SCLogger

from models.base import Base, UUIDMixin, SeeChangeBase, SmartSession, Psycopg2Connection


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
        raise RuntimeError( "CodeHash.code_version is deprecated, don't use it" )

    @code_version.setter
    def code_version( self, val ):
        raise RuntimeError( "CodeHash.code_version is deprecated, don't use it" )




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
        except psycopg2.errors.UniqueViolation as ex:
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
        """Update the id using the code_version, process, parameters and upstream_hashes."""
        if self.process is None or self.parameters is None or self.code_version_id is None:
            raise ValueError('Provenance must have process, code_version_id, and parameters defined. ')

        superdict = dict(
            process=self.process,
            parameters=self.parameters,
            upstream_hashes=[ u.id for u in self._upstreams ],  # this list is ordered by upstream ID
            code_version=self.code_version_id
        )
        json_string = json.dumps(superdict, sort_keys=True, cls=NumpyAndUUIDJsonEncoder)

        self._id = base64.b32encode(hashlib.sha256(json_string.encode("utf-8")).digest()).decode()[:20]

    @classmethod
    def combined_upstream_hash( cls, upstreams ):
        json_string = json.dumps( [ u.id for u in upstreams ], sort_keys=True, cls=NumpyAndUUIDJsonEncoder)
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
                    SCLogger.debug( f"Inserting upstreams of {self.id}: {[p.id for p in upstreams]}" )
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
    def __table_args__(cls):  # noqa: N805
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
    def addtag( cls, tag, provs, add_missing_processes_to_provtag=True ):
        """Tag provenances with a given (string) tag.

        If the provenance tag does not exist at all, create it, tagging
        provs.

        Ensures that there are no conflicts.  If a provenance already
        exists for a given process tagged with tag, and that provenance
        doesn't match the provenance for that process in provs, raise an
        exception.

        If the provenance tag exists, and there is no currently-tagged
        provenance for a given process in provs, then do one of two
        things.  If add_missing_proceses_to_provtag is False, raise an
        Exception.  If it's true, add that provenance to the provenance
        tag as a process.

        Locks the provenance_tags table to avoid race conditions of
        multiple different instances of the pipeline trying to tag
        provenances all at the same time.

        Parameters
        ----------
          tag: str
            The provenance tag

          provs: list of Provenance
            The provenances to tag

          add_missing_processes_to_provtag: bool, default True
            See above.

        """

        # First, make sure that provs doesn't have multiple entries for
        #   processes other than 'referencing'
        seen = set()
        missing = []
        conflict = []
        for p in provs:
            if ( p.process != 'referencing' ) and ( p.process in seen ):
                raise ValueError( f"Process {p.process} is in the list of provenances more than once!" )
            seen.add( p.process )

        # Use direct postgres connection rather than SQLAlchemy so that we can
        # lock tables without a world of hurt.  (See massive comment in
        # base.SmartSession.)
        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( "LOCK TABLE provenance_tags" )
            cursor.execute( "SELECT t.tag,p._id,p.process FROM provenance_tags t "
                            "INNER JOIN provenances p ON t.provenance_id=p._id "
                            "WHERE t.tag=%(tag)s",
                            { 'tag': tag } )
            known = {}
            for row in cursor.fetchall():
                if row['process'] == 'referencing':
                    if 'referencing' not in known:
                        known['referencing'] = [ row['_id'] ]
                    else:
                        known['referencing'].append( row['_id'] )
                else:
                    if row['process'] in known:
                        raise RuntimeError( f"Database corruption error!  The process {row['process']} "
                                            f"has more than one entry for provenance tag {tag}." )
                    known[row['process']] = row['_id']
            if len(known) == 0:
                # If the provenance tag didn't exist at all, then create it even
                # if add_missing_process_to_provtag is False
                add_missing_processes_to_provtag = True

            addedsome = False
            for prov in provs:
                # Special case handling for 'referencing', because there we do allow
                #   multiple provenances tagged with the same tag.
                if prov.process == 'referencing':
                    if 'referencing' not in known:
                        known['referencing'] = []
                    if prov.id not in known['referencing']:
                        if not add_missing_processes_to_provtag:
                            missing.append( prov )
                        else:
                            cursor.execute( "INSERT INTO provenance_tags(tag,provenance_id,_id) "
                                            "VALUES (%(tag)s,%(provid)s,%(uuid)s)",
                                            { 'tag': tag, 'provid': prov.id, 'uuid': uuid.uuid4() } )
                            known['referencing'].append( prov.id )
                            addedsome = True
                else:
                    if prov.process not in known:
                        if not add_missing_processes_to_provtag:
                            missing.append( prov )
                        else:
                            cursor.execute( "INSERT INTO provenance_tags(tag,provenance_id,_id) "
                                            "VALUES (%(tag)s,%(provid)s,%(uuid)s)",
                                            { 'tag': tag, 'provid': prov.id, 'uuid': uuid.uuid4() } )
                            known[prov.process] = prov.id
                            addedsome = True
                    elif known[prov.process] != prov.id:
                        conflict.append( prov )
            if ( addedsome ) and ( len(missing) == 0 ) and ( len(conflict) == 0 ):
                conn.commit()

        if len( conflict ) != 0:
            strio = io.StringIO()
            strio.write( f"The following provenances do not match the existing provenance for tag {tag}:\n " )
            for prov in conflict:
                strio.write( f"   {prov.process}: {prov.id}  (existing: {known[prov.process]})\n" )
            SCLogger.error( strio.getvalue() )
            raise RuntimeError( strio.getvalue() )

        if len( missing ) != 0:
            strio = io.StringIO()
            strio.write( f"The following provenances are not associated with provenance tag {tag}:\n " )
            for prov in missing:
                strio.write( f"   {prov.process}: {prov.id}\n" )
            SCLogger.error( strio.getvalue() )
            raise RuntimeError( strio.getvalue() )
