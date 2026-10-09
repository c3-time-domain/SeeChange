import io
import json
import base64
import hashlib
import textwrap
import uuid

from psycopg import sql
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.schema import UniqueConstraint

from util.util import NumpyAndUUIDJsonEncoder, asUUID, listify
from util.logger import SCLogger

from models.base import Base, UUIDMixin, SeeChangeBase, PGDB



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
    #
    #     sometimes when changing certain values here, hardcoded provenances in ptf and decam
    #     fixtures will need to be updated or tests will fail, or at the very least be slower as
    #     they won't use the cache right (Check warnings for base path not matching)
    #
    #     NOTE: PATCH changes should never result in a change in any data produced, and can be changed without
    #     affecting provenances. MINOR changes will result in some change in the data products, and MAJOR will
    #     represent a major change in how they interact with other parts of the pipeline.
    #     (Technically, you're allowed to violate this as long as MAJOR is 0.)
    CODE_VERSION_DICT = {
        # The core processes of the pipeline
        'acquire_exposure': (0,2,0),
        'preprocessing': (0,11,0),
        'extraction': (0,9,0),
        'astrocal' : (0,6,0),
        'photocal' : (0,3,0),
        'subtraction': (0,5,0),
        'detection': (0,4,0),
        'cutting': (0,2,0),
        'measuring': (0,3,0),
        'scoring': (0,2,0),
        'alerting': (0,1,0),
        'fakeinjection' : (0,2,0),

        # Other processes of associated pipelines
        'referencing' : (0,9,0),
        'coaddition' : (0,9,0),
        'positioning': (0,2,0),
        'flat_bias_builder': (0,5,0),
        'Image.trim': (0,1,0),
        'Image.trim.sources': (0,1,0),
        'Image.trim.wcs': (0,1,0),
        'Image.trim.zp': (0,1,0),
        'diaforcedphot': (0,2,0),

        # The next couple are processes whose direct data products
        #   are not saved to the database.  If their versions change,
        #   then probably the subtraction, and maybe the coadd,
        #   version should change too, as changes in these processes
        #   will affect both subtraction and coadd.
        'alignment' : (0,5,0),
        'inpainting' : (0,2,0),

        # These are processes for downloading stuff
        #   that we need either to do specific surveys,
        #   or for specific instruments.,
        'DECam Default Calibrator' : (0,1,0),
        'manual_reference' : (0,1,0),
        'LS4Cam Default Calibrator': (0, 1, 0),

        # The remaining processes are just used in tests (I think)
        'no_process' : (0,1,0),
        'test_image' : (0,1,0),
        'test_process' : (0,1,0),
        'test_downstream_process': (0,1,0),
        'test_extra_process': (0,1,0),
    }

    _code_version_cache = None


    @classmethod
    def get_by_id( cls, cvid, pgdb=None, session=None ):
        with PGDB( pgdb if pgdb is not None else session, dictcursor=True ) as pgdb:
            rows = pgdb.execute(
                sql.SQL( "SELECT * FROM code_versions WHERE _id={cvid}" )
                .format( cvid=cvid )
            )
            if len(rows) == 0:
                return None
            elif len(rows) > 1:
                raise RuntimeError( "This should never happen." )
            else:
                return CodeVersion( **(rows[0]) )


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
            self.get_upstreams( save_to_object=True )
        return self._upstreams


    def __init__(self, dont_update_id=False, _id=None, pgdb=None, **kwargs):
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

        dont_update_id: bool, default False
            Usually you want this to be False.  If this is False, at the
            end of the constructor, the update_id() method is called to
            make sure the id field is correct.  However, if you are
            playing games like creating a Provenance but planning to
            manually set the _upstreams field later (don't do that
            unless you really know what you're doing!!!), then you may
            need to set this to True.  In that case, you must either have
            passed an _id argument to the constructor, or you must be sure
            to call update_id() yourself.  Setting this to True has other
            side effects that are needed internally, and you really shouldn't
            be using this unless you really know what you're doing.

        _id: string, default None
            You usually do NOT want to send this, you want to let
            provenance automatically set it.  Has other side effects
            that are used internally. Only use this if you really know
            what you're doing.

        """
        SeeChangeBase.__init__(self)
        self._id = _id

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
            cv = Provenance.get_code_version( process=self.process, pgdb=pgdb )
            self.code_version_id = cv.id

        self.parameters = kwargs.get('parameters', {})
        upstreams = kwargs.get('upstreams', None)
        if upstreams is None:
            if dont_update_id:
                self._upstreams = None
            else:
                self._upstreams = []
        elif not isinstance(upstreams, list):
            self._upstreams = [upstreams]
        else:
            self._upstreams = upstreams
        if self._upstreams is not None:
            self._upstreams.sort( key=lambda x: x.id )

        self.is_bad = kwargs.get('is_bad', False)
        self.bad_comment = kwargs.get('bad_comment', None)
        self.is_testing = kwargs.get('is_testing', False)

        if not dont_update_id:
            self.update_id()

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
    def get( cls, provid, must_exist=False, pgdb=None, session=None ):
        """Get a provenance given an id, or None if it doesn't exist.

        If pvid is a Provenance, it is just returned.  (This may be used
        as a convenience function to allow somebody to pass either a
        Provenance or its id; just call this to make sure you have the
        Provenance object.)

        """
        if provid is None:
            if must_exist:
                raise ValueError( "Dude.  You passed None as provid, and then set must_exist.  What did you expect?" )
            return None

        if isinstance( provid, Provenance ):
            return provid

        with PGDB( pgdb if pgdb is not None else session, dictcursor=True ) as pgdb:
            rows = pgdb.execute( sql.SQL( "SELECT * FROM provenances WHERE _id={provid}" )
                                 .format( provid=provid ) )
            if len(rows) == 0:
                if must_exist:
                    return RuntimeError( "Failed to find provenance with id {provid}" )
                else:
                    return None
            elif len(rows) > 1:
                raise RuntimeError( "This should never happen." )
            else:
                return Provenance( dont_update_id=True, **(rows[0]) )


    @classmethod
    def get_batch( cls, provids, session=None ):
        """Get a list of provenances given a list of ids."""
        with PGDB( session, dictcursor=True ) as pgdb:
            provids = listify( provids, require_string=True )
            rows = pgdb.execute( sql.SQL( "SELECT * FROM provenances WHERE _id=ANY(ARRAY[{provids}])" )
                                 .format( provids=sql.SQL(",").join(provids) ) )
            return [ Provenance(dont_update_id=True, **r) for r in rows ]


    @classmethod
    def get_for_tag( cls, tag, process=None, pgdb=None ):
        """Return either the provenance for a tag and process, or all provenances for a tag.

        Parameters
        ----------
          tag : string
            The provenance-tag you want provenances for.

          process : string, default None
             The process you want the provenance for.  If omitted, you
             will get the provenances for all processes defined for the
             tag.

          conn : PGDB, psycopg.Connection, psycopg.Cursor, or (shudder) sa Session, default None
             If given, use this connection to the database.  Otherwise,
             a new connection will be created and closed in this
             function.

        Returns
        -------
          Provenance, None, or dict of Provenance

          If process is not None, you will get back a single Provenance
          object, or None if there isn't one that matches the given tag
          and process.  If process is none, you will get a dictionary of
          process(string): Provenance (which will be empty if the
          provenance tag is not defined).

        """
        with PGDB( pgdb, dictcursor=True ) as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT p.* FROM provenances p
                INNER JOIN provenance_tags t ON p._id=t.provenance_id
                WHERE t.tag={tag}
                """
            ) ).format( tag=tag )
            if process is not None:
                q += sql.SQL( "  AND p.process={process}" ).format( process=process )
            rows = pgdb.execute( q )

        if process is not None:
            if len(rows) > 1:
                raise ValueError( f"Database corruption!  There are multiple provenances for "
                                  f"tag {tag} and process {process}." )
            elif len(rows) == 0:
                return None
            else:
                return Provenance( dont_update_id=True, **rows[0] )

        else:
            return { r['process']: Provenance( dont_update_id=True, **r ) for r in rows }


    def update_id(self):

        """Update the id using the code_version, process, parameters and upstream_hashes."""
        if self.process is None or self.parameters is None or self.code_version_id is None:
            raise ValueError('Provenance must have process, code_version_id, and parameters defined. ')

        # for hash get the static versions from codeversion rather than UUID which changes each run of tests
        cv_string = None
        if self.code_version_id is not None:
            with PGDB() as pgdb:
                rows, _cols = pgdb.execute( sql.SQL( "SELECT version_major, version_minor FROM code_versions "
                                                     "WHERE _id={cvid}").format( cvid=self.code_version_id ) )
                # Don't use patch because patch shouldn't change data thus require a provenance change
                cv_string = f"{rows[0][0]}.{rows[0][1]}"

        superdict = dict(
            process=self.process,
            parameters=self.parameters,
            upstream_hashes=[ u.id for u in self._upstreams ],  # this list is ordered by upstream ID
            code_version=cv_string
        )
        json_string = json.dumps(superdict, sort_keys=True, cls=NumpyAndUUIDJsonEncoder)

        self._id = base64.b32encode(hashlib.sha256(json_string.encode("utf-8")).digest()).decode()[:20]


    @classmethod
    def get_code_version(cls, process, pgdb=None, session=None, nocommit=False):
        """Get the most relevant or latest code version.

        Searches the DB to check if the codeversion matching the current
        codebase already exists, then returns or creates-and-returns it.

        Parameters
        ----------
        process : str
           Process for code version

        pgdb, session: PGDB, psycopg.Connection, psycopg.Cursor, or (shudder) sa Session, default None
            Databse connection.  If None, a new session is created, and
            closed as soon as the function finishes.  WARNING : will
            commit or rollback unless nocommit=True.  Use both this and nocomit with care;
            safer to just not use either.

        Returns
        -------
        code_version: CodeVersion
            CodeVersion object

        """

        if CodeVersion._code_version_cache is None:
            CodeVersion._code_version_cache = { i: None for i in CodeVersion.CODE_VERSION_DICT }

        if CodeVersion._code_version_cache[process] is None:
            pgdb = pgdb if pgdb is not None else session

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

            with PGDB( pgdb, dictcursor=True ) as pgdb:
                # To minimize use of locks, first see if the code version exists, and if it does, be happy.
                # If not, lock the table, check *again* to see if it exists (because another process
                # might have created it in the mean time!), and create it if it doesn't.
                cvsearchq = sql.SQL( textwrap.dedent(
                    """\
                    SELECT * FROM code_versions
                    WHERE process={process}
                      AND version_major={major}
                      AND version_minor={minor}
                      AND version_patch={patch}
                    """
                ) ).format( process=process,
                            major=codebase_semver[0],
                            minor=codebase_semver[1],
                            patch=codebase_semver[2] )
                rows = pgdb.execute( cvsearchq )
                code_version = CodeVersion()
                if len(rows) > 0:
                    rows[0]['_id'] = asUUID( rows[0]['_id'] )
                    code_version.set_attributes_from_dict( rows[0] )
                else:
                    # Rollback the database connection to free up any shared
                    #   locks it has incidentally acquired.
                    if not nocommit:
                        pgdb.rollback()
                    try:
                        pgdb.execute_nofetch( "LOCK TABLE code_versions" )
                        rows = pgdb.execute( cvsearchq )
                        if len(rows) > 0:
                            rows[0]['_id'] = asUUID( rows[0]['_id'] )
                            code_version.set_attributes_from_dict( rows[0] )
                        else:
                            subdict = { '_id': uuid.uuid4(),
                                        'process': process,
                                        'version_major': codebase_semver[0],
                                        'version_minor': codebase_semver[1],
                                        'version_patch': codebase_semver[2] }
                            pgdb.execute_nofetch( sql.SQL( textwrap.dedent(
                                """\
                                INSERT INTO code_versions(_id,process,version_major,version_minor,version_patch)
                                VALUES ({_id},{process},{version_major},{version_minor},{version_patch})
                                """
                            ) ).format( **subdict ) )
                            if not nocommit:
                                pgdb.commit()
                            code_version.set_attributes_from_dict( subdict )
                    finally:
                        if not nocommit:
                            # Make sure lock is freed
                            pgdb.rollback()

                CodeVersion._code_version_cache[process] = code_version

        return CodeVersion._code_version_cache[process]


    def insert( self, pgdb=None, session=None, _exists_ok=False, nocommit=False ):
        """Insert the provenance into the database.

        Will raise a constraint violation if the provenance ID already exists in the database.

        Parameters
        ----------
          pgdb, session : PGDB, psycpog.Connection, psycopg.Cursor, or SQLAlchmey sesion, or None
            Usually you don't want to use this.  Warning: commits or
            rollbacks unless nocommit is True.  Two arguments are
            synonyms; if both are given, pgdb is used.

        """

        pgdb = pgdb if pgdb is not None else session

        with PGDB( pgdb ) as pgdb:
            # Lock the table so we don't have a disaster of two different processes inserting the
            #  provenance and the upstreams all at the same time.  But, because any use of database
            #  locks is just asking for a deadlock, first search without locking, and if it exists,
            #  be happy.  If it doesn't, only then lock, search *again* (just in case somebody else
            #  created it in the mean time), and create it if it doesn't exist.  This doesn't guarantee
            #  no database deadlocks (have to audit the entire SeeChange codebase to make sure that
            #  won't happen...), but it should reduce the risk.
            rows, _cols = pgdb.execute( sql.SQL( "SELECT * FROM provenances WHERE _id={me}" ).format( me=self.id ) )
            if len(rows) > 0:
                if len(rows) > 1:
                    raise RuntimeError( "This should never happen." )
                if not _exists_ok:
                    raise RuntimeError( f"Provenance {self.id} already exists in the database." )
                # Should we verity that it's actually the same?  If we trust the hashing that goes
                #   into the id, that shouldn't be necessary
                return

            # Didn't exist, so we have to make it, and have to lock the table to avoid race conditions
            # First rollback to free up any shared locks we've incidentally acquired
            if not nocommit:
                pgdb.rollback()
            try:
                pgdb.execute_nofetch( "LOCK TABLE provenances" )
                rows, _cols = pgdb.execute( sql.SQL( "SELECT * FROM provenances WHERE _id={me}" ).format( me=self.id ) )
                if len(rows) > 0:
                    if len(rows) > 1:
                        raise RuntimeError( "This should never happen." )
                    if not _exists_ok:
                        raise RuntimeError( f"Provenance {self.id} already exists in the database." )
                    # Should we verity that it's actually the same?  If we trust the hashing that goes
                    #   into the id, that shouldn't be necessary
                    return

                SCLogger.debug( f"Adding provenance {self.id} to database" )
                SeeChangeBase.insert( self, pgdb, nocommit=True )
                upstreams = self._upstreams if self._upstreams is not None else self.get_upstreams( pgdb=pgdb )
                if len(upstreams) > 0:
                    SCLogger.debug( f"Inserting upstreams of {self.id}: {[p.id for p in upstreams]}" )
                    for upstream in upstreams:
                        q = sql.SQL( "INSERT INTO provenance_upstreams(upstream_id, downstream_id) "
                                     "VALUES ({upstream},{me})" ).format( upstream=upstream.id, me=self.id )
                        pgdb.execute( q )

                if not nocommit:
                    pgdb.commit()
            finally:
                # Make sure to free database lock unless we were explicitly told not to
                if not nocommit:
                    pgdb.rollback()


    def insert_if_needed( self, pgdb=None, session=None, nocommit=False ):
        """Insert the provenance into the database if it's not already there.

        Parameters
        ----------
          pgdb, session : PGDB, psycopg.Connection, psycopg.Cursor, sa Session, or None
            Usually you don't want to use this.  Warning: commits unless nocommit=True.

        """

        pgdb = pgdb if pgdb is not None else session
        self.insert( pgdb=pgdb, _exists_ok=True, nocommit=nocommit )


    def get_upstreams( self, pgdb=None, save_to_object=False, always_reload=True ):
        """Get the upstream provenances of this provenance.

        Parameters
        ----------
           pgdb: PGDB, psycopg.connection, or psycopg.cursor, default None
             Database connection.  If not given, will open and close
             connections as necessary.  (This can also be a
             base.PsycopgConnection or base.Session, but those are
             deprecated and should not be used in new code.)

           save_to_object: bool, default False
             If True, then update self._upstreams with what is found.
             (This defaults to False for backwards compatibility.)

           always_reload: bool, default True
             If False and self._upstreams is not None, then just return
             that.  If True, always go to the database.  (This defalts
             to True for backwards compatiblity.)

        Returns
        -------
           list of Provenance

        """

        if ( not always_reload ) and ( self._upstreams is not None ):
            return self._upstreams
        else:
            with PGDB( pgdb, dictcursor=True ) as pgdb:
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT p.* FROM provenances p
                    INNER JOIN provenance_upstreams pu ON p._id=pu.upstream_id
                    WHERE pu.downstream_id={me}
                    ORDER BY p._id
                    """
                ) ).format( me=self.id )
                rows = pgdb.execute( q )
                upstreams = [ Provenance(dont_update_id=True, **row) for row in rows ]
                if save_to_object:
                    self._upstreams = upstreams
                return upstreams

    def get_downstreams( self, pgdb=None ):
        with PGDB( pgdb, dictcursor=True ) as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT p.* FROM provenances p
                INNER JOIN provenance_upstreams pu ON p._id=pu.downstream_id
                WHERE pu.upstream_id={me}
                ORDER BY p._id
                """
            ) ).format( me=self.id )
            rows = pgdb.execute( q )
            return [ Provenance(dont_update_id=True, **row) for row in rows ]


    # def get_upstreams( self, session=None ):
    #     with SmartSession( session ) as sess:
    #         upstreams = ( sess.query( Provenance )
    #                       .join( provenance_self_association_table,
    #                              provenance_self_association_table.c.upstream_id==Provenance._id )
    #                       .where( provenance_self_association_table.c.downstream_id==self.id )
    #                       .order_by( Provenance._id )
    #                      ).all()
    #         return upstreams

    # def get_downstreams( self, session=None ):
    #     with SmartSession( session ) as sess:
    #         downstreams = ( sess.query( Provenance )
    #                         .join( provenance_self_association_table,
    #                                provenance_self_association_table.c.downstream_id==Provenance._id )
    #                         .where( provenance_self_association_table.c.upstream_id==self.id )
    #                         .order_by( Provenance._id )
    #                        ).all()
    #     return downstreams


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
    def addtag( cls, tag, provs, add_missing_processes_to_provtag=True, pgdb=None ):
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

          pgdb : PGDB, psycopg.Connection, psycopg.Cursor, or sa Session, default None
            Database connection.  If not given, will open and close a
            new one.  Warning: may commit or rollback!

        """

        # First, make sure that provs doesn't have multiple entries for
        #   processes other than 'referencing'
        seen = set()
        for p in provs:
            if ( p.process != 'referencing' ) and ( p.process in seen ):
                raise ValueError( f"Process {p.process} is in the list of provenances more than once!" )
            seen.add( p.process )

        # Use direct postgres connection rather than SQLAlchemy so that we can
        #   lock tables without a world of hurt.  (See massive comment in
        #   base.SmartSession.)
        # Because database locks are scary, do the search first to see if everything
        #   is in place and correct.  If so, then just be happy, no need to lock anything.
        #   If not, then we have to lock, do the search *again* to avoid race conditions,
        #   and create stuff if needed.

        def _search_for_provtag( tag, provs, pgdb ):
            nonlocal add_missing_processes_to_provtag

            rows = pgdb.execute( sql.SQL( textwrap.dedent(
                """\
                SELECT t.tag,p._id,p.process FROM provenance_tags t
                INNER JOIN provenances p ON t.provenance_id=p._id
                WHERE t.tag={tag}
                """
            ) ).format( tag=tag  ) )
            known = {}
            mustadd = set()
            missing = set()
            conflict = set()
            for row in rows:
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

            for prov in provs:
                # Special case handling for 'referencing', because there we do allow
                #   multiple provenances tagged with the same tag.
                # ...OR DO WE?  I don't think we do any more?
                if prov.process == 'referencing':
                    if 'referencing' not in known:
                        known['referencing'] = []
                    if prov.id not in known['referencing']:
                        if add_missing_processes_to_provtag:
                            mustadd.add( prov )
                            known['referencing'].append( prov.id )
                        else:
                            missing.add( prov )
                else:
                    if prov.process not in known:
                        if add_missing_processes_to_provtag:
                            mustadd.add( prov )
                        else:
                            missing.add( prov )
                            known[prov.process] = prov.id
                    elif known[prov.process] != prov.id:
                        conflict.add( prov )

            return known, mustadd, missing, conflict

        with PGDB( pgdb, dictcursor=True) as pgdb:
            known, mustadd, missing, conflict = _search_for_provtag( tag, provs, pgdb )

            # See if we need to add stuff; if so, lock the table, search again, and add
            if ( len(missing) == 0 ) and ( len(conflict) == 0 ) and ( len(mustadd) > 0 ):
                # Rollback the databse to free up any shared locks we may have incidentally
                #   acquired, so that multiple processes doing this at once won't deadlock here.
                pgdb.rollback()
                try:
                    pgdb.execute_nofetch( "LOCK TABLE provenance_tags" )
                    known, mustadd, missing, conflict = _search_for_provtag( tag, provs, pgdb )
                    if ( len(missing) == 0 ) and ( len(conflict) == 0 ) and ( len(mustadd) > 0 ):
                        for prov in mustadd:
                            pgdb.execute_nofetch( sql.SQL( textwrap.dedent(
                                """\
                                INSERT INTO provenance_tags(tag,provenance_id,_id)
                                VALUES ({tag},{provid},{uuid})
                                """
                            ) ).format( tag=tag, provid=prov.id, uuid=uuid.uuid4() ) )
                        pgdb.commit()
                finally:
                    # Free that lock!
                    pgdb.rollback()

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
