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

from util.util import NumpyAndUUIDJsonEncoder
from util.logger import SCLogger

from models.base import Base, UUIDMixin, SeeChangeBase, SmartSession, Psycopg2Connection



class CodeVersion(Base, UUIDMixin):
    __tablename__ = 'code_versions'

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            UniqueConstraint('process',
                             'version_minor',
                             'version_major',
                             'version_patch',
                             name='_codeversion_process_versions_uc'),
        )

    version_major = sa.Column(
        sa.Integer,
        nullable=False,
        doc='As per Semantic Versioning, the MAJOR category of MAJOR.MINOR.PATCH'
    )

    version_minor = sa.Column(
        sa.Integer,
        nullable=False,
        doc='As per Semantic Versioning, the MINOR category of MAJOR.MINOR.PATCH'
    )

    version_patch = sa.Column(
        sa.Integer,
        nullable=False,
        doc='As per Semantic Versioning, the PATCH category of MAJOR.MINOR.PATCH'
    )

    process = sa.Column(
        sa.String,
        nullable=False,
        doc='Process for this CodeVersion'
    )

    # represents the versions of each process in the current repository
    #     sometimes when changing certain values here, hardcoded provenances in ptf and decam fixtures
    # will need to be updated or tests will fail (Check warnings for base path not matching)
    #
    #     NOTE: PATCH changes should never result in a change in any data produced, and can be changed without
    #     affecting provenances. MINOR changes will result in some change in the data products, and MAJOR will
    #     represent a major change in how they interact with other parts of the pipeline.
    CODE_VERSION_DICT = {
        'preprocessing': (0,1,0),
        'extraction': (0,1,0),
        'astrocal' : (0,1,0),
        'photocal' : (0,1,0),
        'subtraction': (0,1,0),
        'detection': (0,1,0),
        'cutting': (0,1,0),
        'measuring': (0,1,0),
        'scoring': (0,1,0),
        'bg': (0,1,0),
        'wcs': (0,1,0),
        'zp': (0,1,0),
        'test_process' : (0,1,0),
        'referencing' : (0,1,0),
        'download': (0,1,0),
        'DECam Default Calibrator' : (0,1,0),
        'import_external_reference' : (0,1,0),
        'no_process' : (0,1,0),
        'alignment' : (0,1,0),
        'coaddition' : (0,1,0),
        'manual_reference' : (0,1,0),
        'gratuitous image' : (0,1,0),
        'gratuitous sources' : (0,1,0),
        "acquired" : (0,1,0),
        'fakeinjection' : (0,1,0),
        'exposure' : (0,1,0),
    }

    _code_version_cache = {
        'preprocessing': None,
        'extraction': None,
        'astrocal' : None,
        'photocal' : None,
        'subtraction': None,
        'detection': None,
        'cutting': None,
        'measuring': None,
        'scoring': None,
        'bg': None,
        'wcs': None,
        'zp': None,
        'test_process' : None,
        'referencing' : None,
        'download': None,
        'DECam Default Calibrator' : None,
        'import_external_reference' : None,
        'no_process' : None,
        'alignment' : None,
        'coaddition' : None,
        'manual_reference' : None,
        'gratuitous image' : None,
        'gratuitous sources' : None,
        "acquired" : None,
        'fakeinjection' : None,
        'exposure' : None,
    }

    @classmethod
    def get_by_id( cls, cvid, session=None ):
        with SmartSession( session ) as sess:
            cv = sess.query( CodeVersion ).filter( CodeVersion._id == cvid ).first()
        return cv

    @classmethod
    def is_cv_newer( cls, cv1, cv2 ):
        """Returns True if cv1 is newer than cv2"""
        # check if it is strictly older
        if cv1.version_major > cv2.version_major:
            return True
        if (cv1.version_major == cv2.version_major
            and cv1.version_minor > cv2.version_minor):
            return True
        if (cv1.version_major == cv2.version_major
            and cv1.version_minor == cv2.version_minor
            and cv1.version_patch > cv2.version_patch):
            return True
        return False

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )

    def __repr__( self ):
        return (f"<CodeVersion process: {self.process}, version: {self.version_major}" +
                f".{self.version_minor}.{self.version_patch}, id: {self.id}>")


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
            if not isinstance(code_version_id, uuid.UUID ):
                raise ValueError(f'Code version must be a uuid. Got {type(code_version_id)}.')
            else:
                self.code_version_id = code_version_id
        else:
            cv = Provenance.get_code_version( process=self.process )
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
        """Get a provenance given an id, or None if it doesn't exist."""
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

        # for hash get the static versions from codeversion rather than UUID which changes each run of tests
        cv_string = None
        if self.code_version_id is not None:
            with SmartSession() as sess:
                cv = sess.query( CodeVersion ).filter( CodeVersion._id == self.code_version_id ).first()
                # Don't use patch because patch shouldn't change data thus require a provenance change
                cv_string = f"{cv.version_major}.{cv.version_minor}"

        superdict = dict(
            process=self.process,
            parameters=self.parameters,
            upstream_hashes=[ u.id for u in self._upstreams ],  # this list is ordered by upstream ID
            code_version=cv_string
        )
        json_string = json.dumps(superdict, sort_keys=True, cls=NumpyAndUUIDJsonEncoder)

        self._id = base64.b32encode(hashlib.sha256(json_string.encode("utf-8")).digest()).decode()[:20]


    @classmethod
    def get_code_version(cls, process, session=None):
        """Get the most relevant or latest code version.

        Searches the DB to check if the codeversion matching the current
        codebase already exists, then returns or creates-and-returns it.

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

        if CodeVersion._code_version_cache[process] is None:

            # down the line may want to perform a comparison with the most recent using a search like this
            # with SmartSession( session ) as session:
            #     if code_version is None:
            #         code_version = session.scalars(sa.select(CodeVersion)
            #                                        .where( CodeVersion.process == process )
            #                                        .order_by(CodeVersion.version_major.desc())
            #                                        .order_by(CodeVersion.version_minor.desc())
            #                                        .order_by(CodeVersion.version_patch.desc())).first()

            # ISSUE consider raising exception if there exists a more up-to-date version than the hardcoded

            codebase_semver = CodeVersion.CODE_VERSION_DICT[process]  # (major, minor, patch) eg. (2,0,1)
            with SmartSession() as sess:
                code_version = sess.scalars(sa.select(CodeVersion)
                                               .where( CodeVersion.process == process )
                                               .where( CodeVersion.version_major == codebase_semver[0] )
                                               .where( CodeVersion.version_minor == codebase_semver[1] )
                                               .where( CodeVersion.version_patch == codebase_semver[2] )).first()

            if code_version is not None:
                CodeVersion._code_version_cache[process] = code_version
            else:
                cv = CodeVersion( process=process,
                                  version_major=codebase_semver[0],
                                  version_minor=codebase_semver[1],
                                  version_patch=codebase_semver[2]
                )
                cv.insert()  # should never collide here and need upsert due to above
                CodeVersion._code_version_cache[process] = cv

        return CodeVersion._code_version_cache[process]


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
