import warnings
import os
import time
import math
import copy
import types
import hashlib
import pathlib
import json
import datetime
import uuid
from uuid import UUID
from contextlib import contextmanager

import numpy as np
import shapely

from astropy.coordinates import SkyCoord

import psycopg2
from psycopg2.errors import UniqueViolation

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql
from sqlalchemy import func, orm
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.dialects.postgresql import array as sqlarray
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import IntegrityError, OperationalError

from sqlalchemy.schema import CheckConstraint

from models.enums_and_bitflags import (
    data_badness_dict,
    data_badness_inverse,
    string_to_bitflag,
    bitflag_to_string,
)

import util.config as config
from util.archive import Archive
from util.logger import SCLogger
from util.radec import radec_to_gal_ecl
from util.util import asUUID, UUIDJsonEncoder

# Postgres adapters to allow insertion of some numpy types
import psycopg2.extensions
def _adapt_numpy_float64_psycopg2( val ):
    return psycopg2.extensions.AsIs( val )
def _adapt_numpy_float32_psycopg2( val ):
    return psycopg2.extensions.AsIs( val )
psycopg2.extensions.register_adapter( np.float64, _adapt_numpy_float64_psycopg2 )
psycopg2.extensions.register_adapter( np.float32, _adapt_numpy_float32_psycopg2 )


# this is the root SeeChange folder
CODE_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
#
# # printout the list of relevant environmental variables:
# print("SeeChange environment variables:")
# for key in [
#     'INTERACTIVE',
#     'LIMIT_CACHE_USAGE',
#     'SKIP_NOIRLAB_DOWNLOADS',
#     'RUN_SLOW_TESTS',
#     'SEECHANGE_TRACEMALLOC',
# ]:
#     print(f'{key}: {os.getenv(key)}')


# This is a list of warnings that are categorically ignored in the pipeline. Beware:
def setup_warning_filters():
    # ignore FITS file warnings
    warnings.filterwarnings('ignore', message=r'.*Removed redundant SIP distortion parameters.*')
    warnings.filterwarnings('ignore', message=r".*'datfix' made the change 'Set MJD-OBS to.*")
    warnings.filterwarnings('ignore', message=r"(?s).*the RADECSYS keyword is deprecated, use RADESYSa.*")

    # if you want to add the provenance, you should do it explicitly, not by adding it to a CodeVersion
    warnings.filterwarnings(
        'ignore',
        message=r".*Object of type <Provenance> not in session, "
                r"add operation along 'CodeVersion\.provenances' will not proceed.*"
    )

    # if the object is not in the session, why do I care that we removed some related object from it?
    warnings.filterwarnings(
        'ignore',
        message=r".*Object of type .* not in session, delete operation along .* won't proceed.*"
    )

    # this happens when loading/merging something that refers to another thing that refers back to the original thing
    warnings.filterwarnings(
        'ignore',
        message=r".*Loader depth for query is excessively deep; caching will be disabled for additional loaders.*"
    )

    warnings.filterwarnings(
        'ignore',
        "Can't emit change event for attribute 'Image.md5sum' "
        "- parent object of type <Image> has been garbage collected",
    )


setup_warning_filters()  # need to call this here and also call it explicitly when setting up tests

_engine = None
_Session = None
_psycopg2params = None

def Session():
    """Make a session if it doesn't already exist.

    Use this in interactive sessions where you don't want to open the
    session as a context manager.  Don't use this anywhere in the code
    base.  Instead, always use a context manager, getting your
    connection using "with SmartSession(...) as ...".

    Returns
    -------
    sqlalchemy.orm.session.Session
        A session object that doesn't automatically close.

    """
    global _Session, _engine

    if _Session is None:
        cfg = config.Config.get()

        if cfg.value("db.engine") != "postgresql":
            raise ValueError( "This pipeline only supports PostgreSQL as a database engine" )

        password = cfg.value( "db.password" )
        if password is None:
            if cfg.value( "db.password_file" ) is None:
                raise RuntimeError( "Must specify either db.password or db.password_file in config" )
            with open( cfg.value( "db.password_file" ) ) as ifp:
                password = ifp.readline().strip()

        url = (f'{cfg.value("db.engine")}://{cfg.value("db.user")}:{password}'
               f'@{cfg.value("db.host")}:{cfg.value("db.port")}/{cfg.value("db.database")}')
        _engine = sa.create_engine( url,
                                    future=True,
                                    poolclass=sa.pool.NullPool,
                                    connect_args={ "options": "-c timezone=utc" }
                                   )

        _Session = sessionmaker(bind=_engine, expire_on_commit=False)

    session = _Session()

    return session


@contextmanager
def SmartSession(*args):
    """Return a Session() instance that may or may not be inside a context manager.

    If a given input is already a session, just return that.
    If all inputs are None, create a session that would
    close at the end of the life of the calling scope.

    """
    global _Session, _engine

    for arg in args:
        if isinstance(arg, sa.orm.session.Session):
            yield arg
            return
        if arg is None:
            continue
        else:
            raise TypeError(
                "All inputs must be sqlalchemy sessions or None. "
                f"Instead, got {args}"
            )

    # none of the given inputs managed to satisfy any of the conditions...
    # open a new session and close it when outer scope is done
    with Session() as session:
        try:
            yield session
        finally:
            # Ideally the sesson just closes itself when it goes out of
            # scope, and the database connection is dropped (since we're
            # using NullPool), but that didn't always seem to be working;
            # intermittently (and unpredictably) we'd be left with a
            # dangling session that was idle in transaction, that would
            # later cause database deadlocks because of the table locks we
            # use.  It's probably depending on garbage collection, and
            # sometimes the garbage doesn't get collected in time.  So,
            # explicitly close and invalidate the session.
            #
            # NOTE -- this doesn't seem to have actually fixed the problem. :(
            # I've tried to hack around it by putting a timeout on the locks
            # with a retry loop.  Sigh.
            #
            # Even *that* doesn't seem to have fully fixed it.
            # *Sometimes*, not reproducibly, there's a session that
            # hangs around that is idle in transaction.  There must be
            # some reference to it *somewhere* that's stopping it from
            # getting garbage collected.  I really wish SQLA just closed
            # the connection when I told it to.  I tried adding
            # "session.rollback()" here, but then got all kinds of
            # deatched instance errors trying to access objects later.
            # It seems that rollback() subverts the session's
            # expire_on_commit=False setting.
            #
            # OOO, ooo, here's an idea: just use SQL to rollback.  Hopefully
            # SQLAlchemy won't realize what we're doing and won't totally
            # undermine us for doing it.  (My god I hate SQLA.)
            # (What I'm really trying to accomplish here is given that we
            # seem to rarely have an idle session sitting around, make sure
            # it's not in a transaction that will prevent table locks.)
            #
            # session.execute( sa.text( "ROLLBACK" ) )
            #
            # NOPE!  That didn't work.  If there was a previous
            # exception, sqlalchemy catches that before it lets me run
            # session.execute, saying I gotta rollback before doing
            # anything else.  (There is irony here.)
            #
            # OK, lets try grabbing the connection from the session and
            # manually rolling back with psycopg2 or whatever is
            # underneath.  I'm not sure this will do what I want either,
            # because I don't know if session.bind.raw_connection() gets
            # me the connection that session is using, or if it gets
            # another connection.  (If the latter, than this code is
            # wholly gratuitous.)
            #
            # dbcon = session.bind.raw_connection()
            # cursor = dbcon.cursor()
            # cursor.execute( "ROLLBACK" )

            # ...even that doesn't seem to be solving the problem.
            # The solution may end up being moving totally away from
            # SQLAlchemy and using something that lets us actually
            # control our database connections.

            # OK, another thing to try.  See if expunging all objects
            # lets me rollback.
            session.expunge_all()
            session.rollback()

            session.close()
            session.invalidate()

@contextmanager
def Psycopg2Connection( current=None ):
    """Get a direct psycopg2 connection to the database; use this in a with statement.

    Useful if you don't want to fight with SQLAlchemy, e.g. if you
    want to use table locks (see comment above in SmartSession).

    Parameters
    ----------
      current : psycopg2.extensions.connection or None (default None)
         Pass an existing connection, get it back.  Useful if you are in
         nested functions that might want to be working within the same
         transaction.

    Returns
    -------
       psycopg2.extensions.connection

       After the with block, the connection will be rolled back and
       closed.  So, if you want what you've done committed, make sure to
       call the commit() method on the return value before the with
       block exits.

    """
    global _psycopg2params

    if current is not None:
        if not isinstance( current, psycopg2.extensions.connection ):
            raise TypeError( f"Must pass a psycopg2.extensions.connection or None to Pyscopg2Conection" )
        yield current
        # Don't roll back or close, because whoever created it in the
        #   first place is responsible for that.
        return

    # If a connection wasn't passed, make one, and then be sure to roll it back and close it when we're done

    if _psycopg2params is None:
        cfg = config.Config.get()
        if cfg.value( "db.engine" ) != "postgresql":
            raise ValueError( "This pipeline only supports PostgreSQL as a database engine" )

        password = cfg.value( 'db.password' )
        if password is None:
            if cfg.value( "db.password_file" ) is None:
                raise RuntimeError( "Must specify either db.password or db.password_file in config" )
            with open( cfg.value( "db.password_file" ) ) as ifp:
                password = ifp.readline().strip()

        _psycopg2params = { 'host': cfg.value('db.host'),
                            'port': cfg.value('db.port'),
                            'dbname': cfg.value('db.database'),
                            'user': cfg.value('db.user'),
                            'password': password }

    try:
        conn = psycopg2.connect( **_psycopg2params )
        yield conn

    finally:
        # Just in case things were done, roll back.  Often, the caller
        #   will have done a conn.commit() (which it must if it wants to
        #   keep things that were done) or conn.rollback(), in which
        #   case this rollback is gratuitous.  However, we can't count
        #   on the caller having done that.  (E.g., if there's an
        #   exception, the caller may have short-circuited, which is why
        #   the yield is in a try and this cleaup is in a finally.)
        conn.rollback()
        conn.close()


def db_stat(obj):
    """Check the status of an object. It can be one of: transient, pending, persistent, deleted, detached."""
    for word in ['transient', 'pending', 'persistent', 'deleted', 'detached']:
        if getattr(sa.inspect(obj), word):
            return word


def get_all_database_objects(display=False, session=None):
    """Find all the objects and their associated IDs in the database.

    WARNING: this is only meant to be used on test databases.
    Calling this on a production database would be very slow!

    Parameters
    ----------
    display: bool (optional)
        If True, print the results to stdout.
    session: sqlalchemy.orm.session.Session (optional)
        The session to use. If None, a new session will be created.

    Returns
    -------
    dict
        A dictionary with the object class names as keys and the IDs list as values.

    """
    from models.provenance import Provenance, ProvenanceTag, CodeVersion, CodeHash
    from models.datafile import DataFile
    from models.knownexposure import KnownExposure, PipelineWorker
    from models.exposure import Exposure
    from models.image import Image
    from models.source_list import SourceList
    from models.psf import PSF
    from models.world_coordinates import WorldCoordinates
    from models.zero_point import ZeroPoint
    from models.cutouts import Cutouts
    from models.measurements import Measurements
    from models.deepscore import DeepScore
    from models.object import Object
    from models.calibratorfile import CalibratorFile, CalibratorFileDownloadLock
    from models.catalog_excerpt import CatalogExcerpt
    from models.reference import Reference
    from models.refset import RefSet
    from models.instrument import SensorSection
    from models.user import AuthUser, PasswordLink

    models = [
        CodeHash, CodeVersion, Provenance, ProvenanceTag, DataFile, Exposure, Image,
        SourceList, PSF, WorldCoordinates, ZeroPoint, Cutouts, Measurements, DeepScore,
        Object, CalibratorFile, CalibratorFileDownloadLock, CatalogExcerpt, Reference,
        RefSet, SensorSection, AuthUser, PasswordLink, KnownExposure, PipelineWorker
    ]

    output = {}
    with SmartSession(session) as session:
        for model in models:
            # Note: AuthUser and PasswordLink have id instead of id_, because
            #  they need to be compatible with rkwebutil rkauth
            if ( model == AuthUser ) or ( model == PasswordLink ):
                object_ids = session.scalars(sa.select(model.id)).all()
            else:
                object_ids = session.scalars(sa.select(model._id)).all()
            output[model] = object_ids

            if display:
                SCLogger.debug(f"{model.__name__:16s}: ", end='')
                for obj_id in object_ids:
                    SCLogger.debug(obj_id, end=', ')
                SCLogger.debug()

    return output


class SeeChangeBase:
    """Base class for all SeeChange classes."""

    created_at = sa.Column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
        doc="UTC time of insertion of object's row into the database.",
    )

    modified = sa.Column(
        sa.DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="UTC time the object's row was last modified in the database.",
    )

    type_annotation_map = { UUID: sqlUUID }

    def __init__(self, **kwargs):
        self.from_db = False  # let users know this object was newly created

        if hasattr(self, '_bitflag'):
            self._bitflag = 0
        if hasattr(self, 'upstream_bitflag'):
            self._upstream_bitflag = 0

        for k, v in kwargs.items():
            setattr(self, k, v)

    @orm.reconstructor
    def init_on_load(self):
        self.from_db = True  # let users know this object was loaded from the database

    def get_attribute_list(self):
        """
        Get a list of all attributes of this object,
        not including internal SQLAlchemy attributes,
        and database level attributes like id, created_at, etc.
        """
        attrs = [
            a for a in self.__dict__.keys()
            if (
                a not in ['_sa_instance_state', 'id', 'created_at', 'modified', 'from_db']
                and not callable(getattr(self, a))
                and not isinstance(getattr(self, a), (
                    orm.collections.InstrumentedList, orm.collections.InstrumentedDict
                ))
            )
        ]

        return attrs

    def set_attributes_from_dict( self, dictionary ):
        """Set all atributes of self from a dictionary, excepting existing attributes that are methods.

        Parameters
        ----------
        dictionary: dict
          A dictionary of attributes to set in self

        """
        for key, value in dictionary.items():
            if hasattr(self, key):
                if type( getattr( self, key ) ) != types.MethodType:
                    setattr(self, key, value)


    @classmethod
    def _get_table_lock( cls, session, tablename=None ):
        """Never use this.  The code that uses this is already written.  Use it and get Bobby Tablesed."""

        # This is kind of irritating.  I got the point where I was sure
        # there were no deadlocks written into the code.  However,
        # sometimes, unreproducibly, we'd get a deadlock when trying to
        # LOCK TABLE because there was a dangling database session that
        # was idle in transaction.  I can't figure out what was doing
        # it, and my best hypothesis is that SQLAlchemy is relying on
        # garbage collection to close database connections, even after a
        # call to .invalidate() (which I added to
        # SeeChangeBase.SmartSession).  Sometimes those connections didn't
        # get garbaged collected before the process got to creating a lock.
        #
        # Probably can't figure it out without totally removing SQLAlchemy
        # session management from the code base (and we've already done
        # a big chunk of that, but the last bit would be painful), so work
        # around it with gratuitous retries.
        #
        # ...and this still doesn't seem to be working.  I'm still getting
        # timeouts after 16s of waiting.  But, after the thing dies
        # (drops into the debugger with pytest --pdb), there are no
        # locks in the database.  Somehow, somewhere, something is not
        # releasing a database connection that has an idle transaction.
        # The solution may be to move completely away from SQLAlchemy,
        # which will mean rewriting even more code.

        if tablename is None:
            tablename = cls.__tablename__

        # Uncomment this next debug statement if debugging table locks
        # SCLogger.debug( f"SeeChangeBase.upsert ({cls.__name__}) LOCK TABLE on {tablename}" )
        sleeptime = 0.25
        failed = False
        while sleeptime < 16:
            try:
                session.connection().execute( sa.text( "SET lock_timeout TO '1s'" ) )
                session.connection().execute( sa.text( f'LOCK TABLE {tablename}' ) )
                break
            except OperationalError as e:
                sleeptime *= 2
                if sleeptime >= 16:
                    failed = True
                    break
                else:
                    SCLogger.warning( f"Timeout waiting for lock on {tablename}, sleeping {sleeptime}s and retrying." )
                    session.rollback()
                    time.sleep( sleeptime )
        if failed:
            # import pdb; pdb.set_trace()
            session.rollback()
            SCLogger.error( f"Repeated failures getting lock on {tablename}." )
            raise RuntimeError( f"Repeated failures getting lock on {tablename}." )


    def _get_cols_and_vals_for_insert( self ):
        cols = []
        values = []
        for col in sa.inspect( self.__class__ ).c:
            val = getattr( self, col.name )
            if col.name == 'created_at':
                continue
            elif col.name == 'modified':
                val = datetime.datetime.now( tz=datetime.timezone.utc )

            if isinstance( col.type, sqlalchemy.dialects.postgresql.json.JSONB ) and ( val is not None ):
                val = json.dumps( val )
            elif isinstance( val, np.ndarray ):
                val = list( val )

            # In our case, everything nullable has a default of NULL.  So,
            #   if a nullable column has val at None, it means that we
            #   know we want it to be None, not that we want the server
            #   default to overwrite the None.
            if col.server_default is not None:
                if ( val is not None ) or ( col.nullable and ( val is None ) ):
                    cols.append( col.name )
                    values.append( val )
            else:
                cols.append( col.name )
                values.append( val )

        return cols, values


    def insert( self, session=None, nocommit=False ):
        """Insert the object into the database.

        Does not do any saving to disk, only saves the database record.

        In any event, if there are no exceptions, self.id will be set upon return.

        Will *not* set any unfilled fileds with their defaults.  If you
        want that, reload the row from the database.

        Depends on the subclass of SeeChangeBase having a column _id in
        the database, and a property id that accesses that column,
        autogenerating it if it doesn't exist.

        Parameters
        ----------
          session: SQLALchemy Session, or psycopg2.extensions.connection, or None
            Usually you do not want to pass this; it's mostly for other
            upsert etc. methods that cascade to this.

          nocommit: bool, default False
            If True, run the statement to insert the object, but don't
            actually commit the database.  Do this if you want the
            insert to be inside a transaction you've started on session.
            It doesn't make sense to set nocommit=True unless you've
            passed either a Session or a psycopg2 connection.

        """

        myid = self.id    # Make sure id is generated

        # Doing this manually for a few reasons.  First, doing a
        #  Session.add wasn't always just doing an insert, but was doing
        #  other things like going to the database and checking if it
        #  was there and merging, whereas here we want an exception to
        #  be raised if the row already exists in the database.  Second,
        #  to work around that, we did orm.make_transient( self ), but
        #  that wiped out the _id field, and I'm nervous about what
        #  other unintended consequences calling that SQLA function
        #  might have.  Third, now that we've moved defaults to be
        #  database-side defaults, we'll get errors from SQLA if those
        #  fields aren't filled by trying to do an add, whereas we
        #  should be find with that as the database will just load
        #  the defaults.
        #
        # In any event, doing this manually dodges any weirdness associated
        #  with objects attached, or not attached, to sessions.
        #
        # (Even better, unless a sa Session is passed, bypass sqlalchemy
        # altogether.)

        cols, values = self._get_cols_and_vals_for_insert()
        notmod = [ c for c in cols if c != 'modified' ]

        if ( session is not None ) and ( isinstance( session, sa.orm.session.Session ) ):
            q = f'INSERT INTO {self.__tablename__}({",".join(notmod)}) VALUES (:{",:".join(notmod)}) '
            subdict = { c: v for c, v in zip( cols, values ) if c != 'modified' }
            with SmartSession( session ) as sess:
                sess.execute( sa.text( q ), subdict )
                if not nocommit:
                    sess.commit()
            return

        if ( session is not None ) and ( not isinstance( session, psycopg2.extensions.connection ) ):
            raise TypeError( f"session must be a sa Session or psycopg2.extensions.connection or None, "
                             f"not a {type(session)}" )

        q = f'INSERT INTO {self.__tablename__}({",".join(notmod)}) VALUES (%({")s,%(".join(notmod)})s) '
        subdict = { c: v for c, v in zip( cols, values ) if c != 'modified' }
        with Psycopg2Connection( session ) as conn:
            cursor = conn.cursor()
            cursor.execute( q, subdict )
            if not nocommit:
                conn.commit()


    def upsert( self, session=None, load_defaults=False ):
        """Insert an object into the database, or update it if it's already there (using _id as the primary key).

        Will *not* update self's fields with server default values!
        Re-get the database row if you want that.

        Will not attach the object to session if you pass it.

        Will assign the object an id if it doesn't alrady have one (in self.id).

        If the object is already there, will NOT update any association
        tables (e.g. the image_upstreams_association table), because we
        do not define any SQLAlchemy relationships.  Those must have
        been set when the object was first loaded.

        Be careful with this.  There are some cases where we do want to
        update database records (e.g. the images table once we know
        fwhm, depth, etc), but most of the time we don't want to update
        the database after the first save.

        Parameters
        ----------
          session: SQLAlchemy Session, default None
            Usually you don't want to pass this.

          load_defaults: bool, default False
            Normally, will *not* update self's fields with server
            default values.  Set this to True for that to happen.  (This
            will trigger an additional read from the database.)

        """

        # Doing this manually because I don't think SQLAlchemy has a
        #   clean and direct upsert statement.
        #
        # Used to do this with a lock table followed by search followed
        #   by either an insert or an update.  However, SQLAlchemy
        #   wasn't always closing connections when we told it to.
        #   Sometimes, rarely and unreproducably, there was a lingering
        #   connection in a transaction that caused lock tables to fail.
        #   My hypothesis is that SQLAlchemy is relying on garbage
        #   collection to *actually* close database connections, and I
        #   have not found a way to say "no, really, close the
        #   connection for this session right now".  So, as long as we
        #   still use SQLAlchemy at all, locking tables is likely to
        #   cause intermittent problems.
        #
        # (Doing this manually also has the added advantage of avoiding
        #   sqlalchemy "add" and "merge" statements, so we don't have to
        #   worry about whatever other side effects those things have.)

        # Make sure that self._id is generated
        myid = self.id
        cols, values = self._get_cols_and_vals_for_insert()
        notmod = [ c for c in cols if c != 'modified' ]
        q = ( f'INSERT INTO {self.__tablename__}({",".join(notmod)}) VALUES (:{",:".join(notmod)}) '
              f'ON CONFLICT (_id) DO UPDATE SET '
              f'{",".join( [ f"{c}=:{c}" for c in cols if c!="id" ] )} ')
        subdict = { c: v for c, v in zip( cols, values ) }
        with SmartSession( session ) as sess:
            sess.execute( sa.text( q ), subdict )
            sess.commit()

            if load_defaults:
                dbobj = self.__class__.get_by_id( self.id, session=sess )
                for col in sa.inspect( self.__class__ ).c:
                    if ( ( col.name == 'modified' ) or
                         ( ( col.server_default is not None ) and ( getattr( self, col.name ) is None ) )
                        ):
                        setattr( self, col.name, getattr( dbobj, col.name ) )


    @classmethod
    def upsert_list( cls, objects, session=None, load_defaults=False ):
        """Like upsert, but for a bunch of objects in a list, and tries to be efficient about it.

        Do *not* use this with classes that have things like association
        tables that need to get updated (i.e. with Image, maybe
        eventually some others).

        All reference fields (ids of other objects) of the objects must
        be up to date.  If the referenced objects don't exist in the
        database already, you'll get integrity errors.

        Will update object id fields, but will not update any other
        object fields with database defaults.  Reload the rows from the
        table if that's what you need.

        """

        # Doing this manually for the same reasons as in upset()

        if not all( [ isinstance( o, cls ) for o in objects ] ):
            raise TypeError( f"{cls.__name__}.upsert_list: passed objects weren't all of this class!" )

        with SmartSession( session ) as sess:
            for obj in objects:
                myid = obj.id                 #  Make sure _id is generated
                cols, values = obj._get_cols_and_vals_for_insert()
                notmod = [ c for c in cols if c != 'modified' ]
                q = ( f'INSERT INTO {cls.__tablename__}({",".join(notmod)}) VALUES (:{",:".join(notmod)}) '
                      f'ON CONFLICT (_id) DO UPDATE SET '
                      f'{",".join( [ f"{c}=:{c}" for c in cols if c!="id" ] )} ')
                subdict = { c: v for c, v in zip( cols, values ) }
                sess.execute( sa.text( q ), subdict )
            sess.commit()

            if load_defaults:
                for obj in objects:
                    dbobj = obj.__class__.get_by_id( obj.id, session=sess )
                    for col in sa.inspect( obj.__class__).c:
                        if ( ( col.name == 'modified' ) or
                             ( ( col.server_default is not None ) and ( getattr( obj, col.name ) is None ) )
                            ):
                            setattr( obj, col.name, getattr( dbobj, col.name ) )


    def _delete_from_database( self ):
        """Remove the object from the database.  Don't call this, call delete_from_disk_and_database.

        This does not remove any associated files (if this is a
        FileOnDiskMixin) and does not remove the object from the archive.

        Note that if you call this, cascading relationships in the database
        may well delete other objects.  This shouldn't be a problem if this is
        called from within SeeChangeBase.delete_from_disk_and_database (the
        only place it should be called!), because that recurses itself and
        makes sure to clean up all files and archive files before the database
        records get deleted.

        """

        with SmartSession() as session:
            session.execute( sa.text( f"DELETE FROM {self.__tablename__} WHERE _id=:id" ), { 'id': self.id } )
            session.commit()

        # Look how much easier this is when you don't have to spend a whole bunch of time
        #  deciding if the object needs to be merged, expunged, etc. to a session


    def get_upstreams(self, session=None):
        """Get all data products that were directly used to create this object (non-recursive)."""
        raise NotImplementedError( f'get_upstreams not implemented for this {self.__class__.__name__}' )

    def get_downstreams(self, session=None, siblings=True):
        """Get all data products that were created directly from this object (non-recursive).

        This optionally includes siblings: data products that are co-created in the same pipeline step
        and depend on one another. E.g., a source list and psf have an image upstream and a (subtraction?) image
        as a downstream, but they are each other's siblings.
        """
        raise NotImplementedError( f'get_downstreams not implemented for {self.__class__.__name__}' )


    def delete_from_disk_and_database( self, remove_folders=True, remove_downstreams=True, archive=True ):
        """Delete any data from disk, archive and the database.

        Use this to clean up an entry from all locations, as relevant
        for the particular class.  Will delete the object from the DB
        using the given session (or using an internal session).  If
        using an internal session, commit must be True, to allow the
        change to be committed before closing it.

        This will silently continue if the file does not exist
        (locally or on the archive), or if it isn't on the database,
        and will attempt to delete from any locations regardless
        of if it existed elsewhere or not.

        Parameters
        ----------
        remove_folders: bool
            If True, will remove any folders on the path to the files
            associated to this object, if they are empty.

        remove_downstreams: bool
            If True, will also remove any downstream data.
            Will recursively call get_downstreams() and find any objects
            that can have their data deleted from disk, archive and database.
            Default is True.  Setting this to False is probably a bad idea;
            because of the database structure, some downstream objects may
            get deleted through a cascade, but then the files on disk and
            in the archive will be left behind.  In any event, it violates
            database integrity to remove something and not remove everything
            downstream of it.

        archive: bool
            If True, will also delete the file from the archive.
            Default is True.

        """

        if not remove_downstreams:
            warnings.warn( "Setting remove_downstreams to False in delete_from_disk_and_database "
                           "is probably a bad idea; see docstring." )

        # Recursively remove downstreams first

        if remove_downstreams:
            downstreams = self.get_downstreams()
            if downstreams is not None:
                for d in downstreams:
                    if hasattr( d, 'delete_from_disk_and_database' ):
                        d.delete_from_disk_and_database( remove_folders=remove_folders, archive=archive,
                                                         remove_downstreams=True )

        # Remove files from archive

        if archive and hasattr( self, "filepath" ):
            if self.filepath is not None:
                if self.filepath_extensions is None:
                    self.archive.delete( self.filepath, okifmissing=True )
                else:
                    for ext in self.filepath_extensions:
                        self.archive.delete( f"{self.filepath}{ext}", okifmissing=True )

            # make sure these are set to null just in case we fail
            # to commit later on, we will at least know something is wrong
            self.md5sum = None
            self.md5sum_extensions = None

        # Remove data from disk

        if hasattr( self, "remove_data_from_disk" ):
            self.remove_data_from_disk( remove_folders=remove_folders )
            # make sure these are set to null just in case we fail
            # to commit later on, we will at least know something is wrong
            self.filepath_extensions = None
            self.filepath = None

        # Finally, after everything is cleaned up, remove the database record

        self._delete_from_database()


    def to_dict(self):
        """Translate all the SQLAlchemy columns into a dictionary.

        This can be used, e.g., to cache a row from DB to a file.
        This will include foreign keys, which are not guaranteed
        to remain the same when loading into a new database,
        so all the relationships the object has should be
        reconstructed manually when loading it from the dictionary.

        This will not include any of the attributes of the object
        that are not saved into the database, but those have to
        be lazy loaded anyway, as they are not persisted.

        Will convert non-standard data types:
        md5sum UUIDS will be converted to string (using .hex)
        _id UUIDS will be converted to string (using str())
        Numpy arrays are replaced by lists.

        To reload, use the from_dict() method:
        reloaded_object = MyClass.from_dict( output_dict )
        This will reconstruct the object, including the non-standard
        data types like the UUID.
        """
        output = {}
        for key in sa.inspect(self).mapper.columns.keys():
            value = getattr(self, key)
            # get rid of numpy types
            if isinstance(value, np.number):
                value = value.item()  # convert numpy number to python primitive
            if isinstance(value, list):
                value = [v.item() if isinstance(v, np.number) else v for v in value]
            if isinstance(value, dict):
                value = {k: v.item() if isinstance(v, np.number) else v for k, v in value.items()}

            if key == 'md5sum' and value is not None:
                if isinstance(value, UUID):
                    value = value.hex
            if key == 'md5sum_extensions' and value is not None:
                if isinstance(value, list):
                    value = [v.hex if isinstance(v, UUID) else v for v in value]

            if key == '_id' and value is not None:
                if isinstance(value, UUID):
                    value = str(value)

            if isinstance(value, np.ndarray) and key in [
                'aper_rads', 'aper_radii', 'aper_cors', 'aper_cor_radii',
                'flux_apertures', 'flux_apertures_err', 'area_apertures',
                'ra', 'dec',
            ]:
                if len(value.shape) > 0:
                    value = list(value)
                else:
                    value = float(value)

            if isinstance(value, np.number):
                value = value.item()

            # 'claim_time' is from KnownExposure, lastheartbeat is from PipelineWorker
            # 'start_time' and 'finish_time' are from Report
            # We should probably define a class-level variable "_datetimecolumns" and list them
            #   there, other than adding to what's hardcoded here.  (Likewise for the ndarray aper stuff
            #   above.)
            if (   ( key in [ 'modified', 'created_at', 'claim_time', 'lastheartbeat',
                              'start_time', 'finish_time' ] ) and
                   isinstance(value, datetime.datetime) ):
                value = value.isoformat()

            if isinstance(value, (datetime.datetime, np.ndarray)):
                raise TypeError('Found some columns with non-standard types. Please parse all columns! ')

            output[key] = value

        return output

    @classmethod
    def from_dict(cls, dictionary):
        """Convert a dictionary into a new object. """
        dictionary.pop('modified', None)  # we do not want to recreate the object with an old "modified" time

        obj_id = dictionary.get('_id')
        if obj_id is not None:
            dictionary['_id'] = UUID(obj_id)

        md5sum = dictionary.get('md5sum', None)
        if md5sum is not None:
            dictionary['md5sum'] = UUID(md5sum)

        md5sum_extensions = dictionary.get('md5sum_extensions', None)
        if md5sum_extensions is not None:
            new_extensions = [UUID(md5) for md5 in md5sum_extensions if md5 is not None]
            dictionary['md5sum_extensions'] = new_extensions

        aper_rads = dictionary.get('aper_rads', None)
        if aper_rads is not None:
            dictionary['aper_rads'] = np.array(aper_rads)

        aper_cors = dictionary.get('aper_cors', None)
        if aper_cors is not None:
            dictionary['aper_cors'] = np.array(aper_cors)

        aper_cor_radii = dictionary.get('aper_cor_radii', None)
        if aper_cor_radii is not None:
            dictionary['aper_cor_radii'] = np.array(aper_cor_radii)

        created_at = dictionary.get('created_at', None)
        if created_at is not None:
            dictionary['created_at'] = datetime.datetime.fromisoformat(created_at)

        return cls(**dictionary)

    def to_json(self, filename):
        """Translate a row object's column values to a JSON file.

        See the description of to_dict() for more details.

        Parameters
        ----------
        filename: str or path
            The path to the output JSON file.
        """
        with open(filename, 'w') as fp:
            try:
                json.dump(self.to_dict(), fp, indent=2, cls=UUIDJsonEncoder)
            except:
                raise

    def copy(self):
        """Make a new instance of this object, with all column-based attributed (shallow) copied. """
        new = self.__class__()
        for key in sa.inspect(self).mapper.columns.keys():
            value = getattr( self, key )
            setattr( new, key, value )
        return new


Base = declarative_base(cls=SeeChangeBase)

ARCHIVE = None


def get_archive_object():
    """Return a global archive object. If it doesn't exist, create it based on the current config. """
    global ARCHIVE
    if ARCHIVE is None:
        cfg = config.Config.get()
        archive_specs = cfg.value('archive', None)
        if archive_specs is not None:
            archive_specs[ 'logger' ] = SCLogger
            ARCHIVE = Archive(**archive_specs)
    return ARCHIVE


class FileOnDiskMixin:
    """Mixin for objects that refer to files on disk.

    Files are assumed to live on the local disk (underneath the
    configured path.data_root), and optionally on a archive server
    (configured through the subproperties of "archive" in the yaml
    config file).  The property filepath is the path relative to the
    root in both cases.

    If there is a single file associated with this entry, then filepath
    has the name of that file.  md5sum holds a checksum for the file,
    *if* it has been correctly saved to the archive.  If the file has
    not been saved to the archive, md5sum is null.  In this case,
    filepath_extensions and md5sum_extensions will be null.  (Exception:
    if you are configured with a null archive (config parameter archive
    in the yaml config file is null), then md5sum will be set when the
    image is saved to disk, instead of when it's saved to the archive.)

    If there are multiple files associated with this entry, then
    filepath is the beginning of the names of all the files.  Each entry
    in filepath_extensions is then appended to filepath to get the
    actual path of the file.  For example, if an image file has the
    image itself, an associated weight, and an associated mask, then
    filepath might be "image" and filepath_extensions might be
    [".fits.fz", ".mask.fits.fz", ".weight.fits.fz"] to indicate that
    the three files image.fits.fz, image.mask.fits.fz, and
    image.weight.fz are all associated with this entry.  When
    filepath_extensions is non-null, md5sum should be null, and
    md5sum_extensions is an array with the same length as
    filepath_extensions.  (For extension files that have not yet been
    saved to the archive, that element of the md5sum_etensions array is
    null.)

    Saving data:

    Any object that implements this mixin must call this class' "save"
    method in order to save the data to disk.  (This may be through
    super() if the subclass has to do custom things.)  The save method
    of this class will save to the local filestore (underneath
    path.data_root), and also save it to the archive.  Once a file is
    saved on the archive, the md5sum (or md5sum_extensions) field in the
    database record is updated.  (If the file has not been saved to the
    archive, then the md5sum and md5sum_extensions fields will be null.)

    Loading data:

    When calling get_fullpath(), the object will first check if the file
    exists locally, and then it will import it from archive if missing
    (and if archive is defined).  If you want to avoid downloading, use
    get_fullpath(download=False) or get_fullpath(nofile=True).  (The
    latter case won't even try to find the file on the local disk, it
    will just tell you what the path should be.)  If you want to always
    get a list of filepaths (even if filepath_extensions=None) use
    get_fullpath(as_list=True).  If the file is missing locally, and
    downloading cannot proceed (because no archive is defined, or
    because the download=False flag is used, or because the file is
    missing from server), then the call to get_fullpath() will raise an
    exception (unless you use download=False or nofile=True).

    After all the pulling from the archive is done and the file(s) exist
    locally, the full (absolute) path to the local file is returned.  It
    is then up to the inheriting object (e.g., the Exposure or Image) to
    actually load the file from disk and figure out what to do with the
    data.

    The path to the local file store and the archive object are saved in
    class variables "local_path" and "archive" that are initialized from
    the config system the first time the class is loaded.

    """
    local_path = None
    temp_path = None

    # ref: https://docs.sqlalchemy.org/en/20/orm/declarative_mixins.html#creating-indexes-with-mixins
    # ...but I have not succeded in finding a way for it to work with multiple mixins and having
    # cls.__tablename__ be the subclass tablename, not the mixin tablename.  So, for now, the solution
    # is the manual stuff below
    # @declared_attr
    # def __table_args__( cls ):
    #     return (
    #         CheckConstraint(
    #             sqltext='NOT(md5sum IS NULL AND '
    #                     '(md5sum_extensions IS NULL OR array_position(md5sum_extensions, NULL) IS NOT NULL))',
    #             name=f'{cls.__tablename__}_md5sum_check'
    #         ),
    #     )

    # Subclasses of this class must include the following in __table_args__:
    #   CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
    #                    '(md5sum_extensions IS NULL OR array_position(md5sum_extensions, NULL) IS NOT NULL))',
    #                    name=f'{cls.__tablename__}_md5sum_check' )


    @classmethod
    def configure_paths(cls):
        cfg = config.Config.get()
        cls.local_path = cfg.value('path.data_root', None)

        if cls.local_path is None:
            cls.local_path = cfg.value('path.data_temp', None)
        if cls.local_path is None:
            cls.local_path = os.path.join(CODE_ROOT, 'data')

        if not os.path.isabs(cls.local_path):
            cls.local_path = os.path.join(CODE_ROOT, cls.local_path)
        if not os.path.isdir(cls.local_path):
            os.makedirs(cls.local_path, exist_ok=True)

        # use this to store temporary files (scratch files)
        cls.temp_path = cfg.value('path.data_temp', None)
        if cls.temp_path is None:
            cls.temp_path = os.path.join(CODE_ROOT, 'data')

        if not os.path.isabs(cls.temp_path):
            cls.temp_path = os.path.join(CODE_ROOT, cls.temp_path)
        if not os.path.isdir(cls.temp_path):
            os.makedirs(cls.temp_path, exist_ok=True)

    @classmethod
    def safe_mkdir(cls, path):
        if path is None or path == '':
            return  # ignore empty paths, we don't need to make them!
        cfg = config.Config.get()

        allowed_dirs = []
        if cls.local_path is not None:
            allowed_dirs.append(cls.local_path)
        temp_path = cfg.value('path.data_temp', None)
        if temp_path is not None:
            allowed_dirs.append(temp_path)

        allowed_dirs = list(set(allowed_dirs))

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

    @declared_attr
    def filepath(cls):
        return sa.Column(
            sa.Text,
            nullable=False,
            index=True,
            unique=True,
            doc="Base path (relative to the data root) for a stored file"
        )

    filepath_extensions = sa.Column(
        ARRAY(sa.Text, zero_indexes=True),
        nullable=True,
        doc="If non-null, array of text appended to filepath to get actual saved filenames."
    )

    md5sum = sa.Column(
        sqlUUID(as_uuid=True),
        nullable=True,
        server_default=None,
        doc="md5sum of the file, provided by the archive server"
    )

    md5sum_extensions = sa.Column(
        ARRAY(sqlUUID(as_uuid=True), zero_indexes=True),
        nullable=True,
        server_default=None,
        doc="md5sum of extension files; must have same number of elements as filepath_extensions"
    )

    def __init__(self, *args, **kwargs):
        """Initialize an object that is associated with a file on disk.

        If giving a single unnamed argument, will assume that is the filepath.
        Note that the filepath should not include the global data path,
        but only a path relative to that. # TODO: remove the global path if filepath starts with it?

        Parameters
        ----------
        args: list
            List of arguments, should only contain one string as the filepath.

        kwargs: dict
            Dictionary of keyword arguments.
            These include:
            - filepath: str
                Use instead of the unnamed argument.
            - nofile: bool
                If True, will not require the file to exist on disk.
                That means it will not try to download it from archive, either.
                This should be used only when creating a new object that will
                later be associated with a file on disk (or for tests).
                This property is NOT SAVED TO DB!
                Saving to DB should only be done when a file exists
                This is True by default, except for subclasses that
                override the _do_not_require_file_to_exist() method.
                # TODO: add the check that file exists before committing?
        """
        if len(args) == 1 and isinstance(args[0], str):
            self.filepath = args[0]

        self.filepath = kwargs.pop('filepath', self.filepath)
        self.nofile = kwargs.pop('nofile', self._do_not_require_file_to_exist())

        self._archive = None

    @orm.reconstructor
    def init_on_load(self):
        self.nofile = self._do_not_require_file_to_exist()
        self._archive = None

    @property
    def archive(self):
        if getattr(self, '_archive', None) is None:
            self._archive = get_archive_object()
        return self._archive

    @archive.setter
    def archive(self, value):
        self._archive = value

    @staticmethod
    def _do_not_require_file_to_exist():
        """
        The default value for the nofile property of new objects.
        Generally it is ok to make new FileOnDiskMixin derived objects
        without first having a file (the file is created by the app and
        saved to disk before the object is committed).
        Some subclasses (e.g., Exposure) will override this method
        so that the default is that a file MUST exist upon creation.
        In either case the caller to the __init__ method can specify
        the value of nofile explicitly.
        """
        return True

    def __setattr__(self, key, value):
        if key == 'filepath' and isinstance(value, str):
            value = self._validate_filepath(value)

        super().__setattr__(key, value)

    def _validate_filepath(self, filepath):
        """
        Make sure the filepath is legitimate.
        If the filepath starts with the local path
        (i.e., an absolute path is given) then
        the local path is removed from the filepath,
        forcing it to be a relative path.

        Parameters
        ----------
        filepath: str
            The filepath to validate.

        Returns
        -------
        filepath: str
            The validated filepath.
        """
        if filepath.startswith(self.local_path):
            filepath = filepath[len(self.local_path) + 1:]

        return filepath

    def get_fullpath(self, download=True, as_list=False, nofile=None, always_verify_md5=False):
        """Get the full path of the file, or list of full paths of files if filepath_extensions is not None.

        If the archive is defined, and download=True (default),
        the file will be downloaded from the server if missing.
        If the file is not found on server or locally, will
        raise a FileNotFoundError.
        When setting self.nofile=True, will not check if the file exists,
        or try to download it from server. The assumption is that an
        object with self.nofile=True will be associated with a file later on.

        If the file is found on the local drive, under the local_path,
        (either it was there or after it was downloaded)
        the full path is returned.
        The application is then responsible for loading the content
        of the file.

        When the filepath_extensions is None, will return a single string.
        When the filepath_extensions is an array, will return a list of strings.
        If as_list=False, will always return a list of strings,
        even if filepath_extensions is None.

        Parameters
        ----------
        download: bool
            Whether to download the file from server if missing.
            Must have archive defined. Default is True.
        as_list: bool
            Whether to return a list of filepaths, even if filepath_extensions=None.
            Default is False.
        nofile: bool
            Whether to check if the file exists on local disk.
            Default is None, which means use the value of self.nofile.
        always_verify_md5: bool
            Set True to verify that the file's md5sum matches what's
            in the database (if there is one in the database), and
            raise an exception if it doesn't.  Ignored if nofile=True.

        Returns
        -------
        str or list of str
            Full path to the file(s) on local disk.
        """
        if self.filepath_extensions is None:
            if as_list:
                return [self._get_fullpath_single(download=download, nofile=nofile,
                                                  always_verify_md5=always_verify_md5)]
            else:
                return self._get_fullpath_single(download=download, nofile=nofile,
                                                 always_verify_md5=always_verify_md5)
        else:
            return [
                self._get_fullpath_single(download=download, ext=ext, nofile=nofile,
                                          always_verify_md5=always_verify_md5)
                for ext in self.filepath_extensions
            ]

    def _get_fullpath_single(self, download=True, ext=None, nofile=None, always_verify_md5=False):
        """Get the full path of a single file.
        Will follow the same logic as get_fullpath(),
        of checking and downloading the file from the server
        if it is not on local disk.

        Parameters
        ----------
        download: bool
            Whether to download the file from server if missing.
            Must have archive defined. Default is True.
        ext: str
            Extension to add to the filepath. Default is None
            (indicating that this object is stored in a single file).
        nofile: bool
            Whether to check if the file exists on local disk.
            Default is None, which means use the value of self.nofile.
        always_verify_md5: bool
            Set True to verify that the file's md5sum matches what's
            in the database (if there is one in the database), and
            raise an exception if it doesn't.  Ignored if nofile=True.

        Returns
        -------
        str
            Full path to the file on local disk.

        """
        if self.filepath is None:
            return None

        if nofile is None:
            nofile = self.nofile

        if not nofile and self.local_path is None:
            raise ValueError("Local path not defined!")

        fname = self.filepath
        md5sum = None
        if ext is None:
            md5sum = self.md5sum.hex if self.md5sum is not None else None
        else:
            found = False
            try:
                extdex = self.filepath_extensions.index( ext )
            except ValueError:
                raise ValueError(f"Unknown extension {ext} for {fname}" )
            if (self.md5sum_extensions is None ) or ( extdex >= len(self.md5sum_extensions) ):
                md5sum = None
            else:
                md5sum = self.md5sum_extensions[extdex]
                md5sum = None if md5sum is None else md5sum.hex
            fname += ext

        downloaded = False
        fullname = os.path.join(self.local_path, fname)
        if ( not nofile ) and ( not os.path.exists(fullname) ) and download and ( self.archive is not None ):
            if md5sum is None:
                raise RuntimeError(f"Don't have md5sum in the database for {fname}, can't download")
            self.archive.download( fname, fullname, verifymd5=True, clobbermismatch=False, mkdir=True )
            downloaded = True

        if not nofile:
            if not os.path.exists(fullname):
                raise FileNotFoundError(f"File {fullname} not found!")
            elif always_verify_md5 and not downloaded and md5sum is not None:
                # self.archive.download will have already verified the md5sum
                filemd5 = hashlib.md5()
                with open(fullname, "rb") as ifp:
                    filemd5.update(ifp.read())
                localmd5 = filemd5.hexdigest()
                if localmd5 != md5sum:
                    raise ValueError( f"{fname} has md5sum {localmd5} on disk, which doesn't match the "
                                      f"database value of {md5sum}" )

        return fullname


    def save(self, data, extension=None, overwrite=True, exists_ok=True, verify_md5=True, no_archive=False ):
        """Save a file to disk, and to the archive.

        Does not write anything to the database.  (At least, it's not supposed to....)

        Parameters
        ---------
        data: bytes, string, or Path
          The data to be saved

        extension: string or None
          The file extension

        overwrite: bool
          True to overwrite existing files (locally and on the archive).

        exists_ok: bool
          Ignored if overwrite is True.  Otherwise: if the file exists
          on disk, and this is False, raise an exception.

        verify_md5: bool
          Used to modify both overwrite and exists_ok
            LOCAL STORAGE
               verify_md5 = True
                  if overwrite = True, check file md5 before actually overwriting
                  if overwrite = False
                      if exists_ok = True, verify existing file
               verify_md5 = False
                  if overwrite = True, always overwrite the file
                  if overwrite = False
                      if exists_ok = True, assume existing file is right
            ARCHIVE
               If self.md5sum (or the appropriate entry in
               md5sum_extensions) is null, then always upload to the
               archive as long as no_archive is False). Otherwise,
               verify_md5 modifies the behavior;
               verify_md5 = True
                 If self.md5sum (or the appropriate entry in
                 md5sum_extensions) matches the md5sum of the passed data,
                 do not upload to the archive.  Otherwise, if overwrite
                 is true, upload to the archive; if overwrite is false,
                 raise an exception.
               verify_md5 = False
                 If overwrite is True, upload to the archive,
                 overwriting what's there.  Otherwise, assume that
                 what's on the archive is right.

        no_archive: bool
          If True, do *not* save to the archive, only to the local filesystem.

        If data is a "bytes" type, then it represents the relevant
        binary data.  It will be written to the right place in the local
        filestore (underneath path.data_root).

        If data is a pathlib.Path or a string, then it is the
        (resolvable) path to a file on disk.  In this case, if the file
        is already in the right place underneath path.data_root, it is
        just left in place (modulo verify_md5).  Otherwise, it is copied
        there.

        Then, in either case, the file is uploaded to the archive (if
        the class property archive is not None, modul verify_md5).  Once
        it's uploaded to the archive, the object's md5sum is set (or
        updated, if overwrite is True and it wasn't null to start with).

        If extension is not None, and it isn't already in the list of
        filepath_extensions, it will be added.

        Performance notes: if you call this with anything other than
        overwrite=False, exists_ok=True, verify_md5=False, you may well
        have redundant I/O.  You may have saved an image before, and
        then (for instance) called
        pipeline.data_store.save_and_commit(), which will at the very
        least read things to verify that md5sums match.

        Of course, not either using overwrite=True or verify_md5=True
        could lead to incorrect files in either the local filestore on
        the server not being detected.

        """

        # (This one is an ugly mess of if statements, done to avoid reading
        # or writing files when not necessary since I/O tends to be much
        # more expensive than processing.)

        # First : figure out if this is an extension or not,
        #   and make sure that's consistent with the object.
        # If it is:
        #   Find the index into the extensions array for
        #   this extension, or append to the array if
        #   it's a new extension that doesn't already exist.
        #   Set the variables curextensions and extmd5s to lists with
        #   extensions and md5sums of extension files,
        #   initially copied from self.filepath_extensions and
        #   self.md5sum_extensions, and modified if necessary
        #   with the saved file.  extensiondex holds the index
        #   into both of these arrays for the current extension.
        # else:
        #   Set curextions, extmd5s, and extensiondex to None.

        # We will either replace these two variables with empty lists,
        #  or make a copy (using list()).  The reason for this: we don't
        #  want to directly modify the lists in self until the saving is
        #  done.  That way, self doesn't get mucked up if this function
        #  exceptions out.
        curextensions = self.filepath_extensions
        extmd5s = self.md5sum_extensions

        extensiondex = None
        if extension is None:
            if curextensions is not None:
                raise RuntimeError( "Tried to save a non-extension file, but this file has extensions" )
            if extmd5s is not None:
                raise RuntimeError( "Data integrity error; filepath_extensions is null, "
                                    "but md5sum_extensions isn't." )
        else:
            if curextensions is None:
                if extmd5s is not None:
                    raise RuntimeError( "Data integrity error; filepath_extensions is null, "
                                        "but md5sum_extensions isn't." )
                curextensions = []
                extmd5s = []
            else:
                if extmd5s is None:
                    raise RuntimeError( "Data integrity error; filepath_extensions is not null, "
                                        "but md5sum_extensions is" )
                curextensions = list(curextensions)
                extmd5s = list(extmd5s)
            if len(curextensions) != len(extmd5s):
                raise RuntimeError( f"Data integrity error; len(md5sum_extensions)={len(extmd5s)}, "
                                    f"but len(filepath_extensions)={len(curextensions)}" )
            try:
                extensiondex = curextensions.index( extension )
            except ValueError:
                curextensions.append( extension )
                extmd5s.append( None )
                extensiondex = len(curextensions) - 1

        # relpath holds the path of the file relative to the data store root
        # origmd5 holds the md5sum (hashlib.hash object) of the original file,
        #   *unless* the original file is already the right file in the local file store
        #   (in which case it's None)
        # localpath holds the absolute path of where the file should be written in the local file store
        relpath = pathlib.Path( self.filepath if extension is None else self.filepath + extension )
        localpath = pathlib.Path( self.local_path ) / relpath
        if isinstance( data, bytes ):
            path = "passed data"
        else:
            if isinstance( data, str ):
                path = pathlib.Path( data )
            elif isinstance( data, pathlib.Path ):
                path = data
            else:
                raise TypeError( f"data must be bytes, str, or Path, not {type(data)}" )
            path = path.absolute()
            data = None

        alreadyinplace = False
        mustwrite = False
        origmd5 = None
        if not localpath.exists():
            mustwrite = True
        else:
            if not localpath.is_file():
                raise RuntimeError( f"{localpath} exists but is not a file!  Can't save." )
            if localpath == path:
                alreadyinplace = True
                # SCLogger.debug( f"FileOnDiskMixin.save: local file store path and original path are the same: {path}" )
            else:
                if ( not overwrite ) and ( not exists_ok ):
                    raise FileExistsError( f"{localpath} already exists, cannot save." )
                if verify_md5:
                    origmd5 = hashlib.md5()
                    if data is None:
                        with open( path, "rb" ) as ifp:
                            data = ifp.read()
                    origmd5.update( data )
                    localmd5 = hashlib.md5()
                    with open( localpath, "rb" ) as ifp:
                        localmd5.update( ifp.read() )
                    if localmd5.hexdigest() != origmd5.hexdigest():
                        if overwrite:
                            SCLogger.debug( f"Existing {localpath} md5sum mismatch; overwriting." )
                            mustwrite = True
                        else:
                            raise ValueError( f"{localpath} exists, but its md5sum {localmd5.hexdigest()} does not "
                                              f"match md5sum of {path} {origmd5.hexdigest()}" )
                else:
                    if overwrite:
                        SCLogger.debug( f"Overwriting {localpath}" )
                        mustwrite = True
                    elif exists_ok:
                        SCLogger.debug( f"{localpath} already exists, not verifying md5 nor overwriting" )
                    else:
                        # raise FileExistsError( f"{localpath} already exists, not saving" )
                        # Logically, should not be able to get here
                        raise RuntimeError( "This should never happen" )

        if mustwrite and not alreadyinplace:
            if data is None:
                with open( path, "rb" ) as ifp:
                    data = ifp.read()
            if origmd5 is None:
                origmd5 = hashlib.md5()
                origmd5.update( data )
            localpath.parent.mkdir( exist_ok=True, parents=True )
            with open( localpath, "wb" ) as ofp:
                ofp.write( data )
            # Verify written file
            with open( localpath, "rb" ) as ifp:
                writtenmd5 = hashlib.md5()
                writtenmd5.update( ifp.read() )
                if writtenmd5.hexdigest() != origmd5.hexdigest():
                    raise RuntimeError( f"Error writing {localpath}; written file md5sum mismatches expected!" )

        # If there is no archive, update the md5sum now
        if self.archive is None:
            if origmd5 is None:
                origmd5 = hashlib.md5()
                with open( localpath, "rb" ) as ifp:
                    origmd5.update( ifp.read() )
            if curextensions is not None:
                extmd5s[ extensiondex ] = UUID( origmd5.hexdigest() )
                self.filepath_extensions = curextensions
                self.md5sum_extensions = extmd5s
            else:
                self.md5sum = UUID( origmd5.hexdigest() )
            return

        # This is the case where there *is* an archive, but the no_archive option was passed
        if no_archive:
            if curextensions is not None:
                self.filepath_extensions = curextensions
                self.md5sum_extensions = extmd5s
            return

        # The rest of this deals with the archive

        archivemd5 = self.md5sum if extension is None else extmd5s[extensiondex]
        logfilepath = self.filepath if extension is None else f'{self.filepath}{extension}'

        mustupload = False
        if archivemd5 is None:
            mustupload = True
        else:
            if not verify_md5:
                if overwrite:
                    SCLogger.debug( f"Uploading {logfilepath} to archive, overwriting existing file" )
                    mustupload = True
                else:
                    SCLogger.debug( f"Assuming existing {logfilepath} on archive is correct" )
            else:
                if origmd5 is None:
                    origmd5 = hashlib.md5()
                    if data is None:
                        with open( localpath, "rb" ) as ifp:
                            data = ifp.read()
                    origmd5.update( data )
                if origmd5.hexdigest() == archivemd5.hex:
                    SCLogger.debug( f"Archive md5sum for {logfilepath} matches saved data, not reuploading." )
                else:
                    if overwrite:
                        SCLogger.debug( f"Archive md5sum for {logfilepath} doesn't match saved data, "
                                              f"overwriting on archive." )
                        mustupload = True
                    else:
                        raise ValueError( f"Archive md5sum for {logfilepath} does not match saved data!" )

        if mustupload:
            remmd5 = self.archive.upload(
                localpath=localpath,
                remotedir=relpath.parent,
                remotename=relpath.name,
                overwrite=overwrite,
                md5=origmd5
            )
            remmd5 = UUID( remmd5 )
            if curextensions is not None:
                extmd5s[extensiondex] = remmd5
                self.md5sum = None
                self.filepath_extensions = curextensions
                self.md5sum_extensions = extmd5s
            else:
                self.md5sum = remmd5

    def remove_data_from_disk(self, remove_folders=True):
        """Delete the data from local disk, if it exists.
        If remove_folders=True, will also remove any folders
        if they are empty after the deletion.

        To remove both the files and the database entry, use
        delete_from_disk_and_database() instead.  That one
        also supports removing downstreams.

        Parameters
        ----------
        remove_folders: bool
            If True, will remove any folders on the path to the files
            associated to this object, if they are empty.
        """
        if self.filepath is not None:
            # get the filepath, but don't check if the file exists!
            for f in self.get_fullpath(as_list=True, nofile=True):
                if os.path.exists(f):
                    os.remove(f)
                    if remove_folders:
                        folder = f
                        for i in range(10):
                            folder = os.path.dirname(folder)
                            if len(os.listdir(folder)) == 0:
                                os.rmdir(folder)
                            else:
                                break


# load the default paths from the config
FileOnDiskMixin.configure_paths()


def safe_mkdir(path):
    FileOnDiskMixin.safe_mkdir(path)


class UUIDMixin:
    # We use UUIDs rather than auto-incrementing SQL sequences for
    # unique object primary keys so that we can generate unique ids
    # without having to contact the database.  This allows us, for
    # example, to build up a collection of objects including foreign
    # keys to each other, and save them to the database at the end.
    # With auto-generating primary keys, we wouldn't be able to set the
    # foreign keys until we'd saved the referenced object to the
    # databse, so that its id was generated.  (SQLAlchemy gets around
    # this with object relationships, but object relationships in SA
    # caused us so many headaches that we stopped using them.)  It also
    # allows us to do things like cache objects that we later load into
    # the database, without worrying that the cached object's id (and
    # references amongst multiple cached objects) will be inconsistent
    # with the state of the database counters.

    # Note that even though the default is uuid.uuid4(), this is set by SQLAlchemy
    #   when the object is saved to the database, not when the object is created.
    #   It will be None when a new object is created if not explicitly set.
    #   (In practice, often this id will get set by our code when we access the
    #   id property of a created object before it's saved to the datbase, or it will
    #   be set in our insert/upsert methods, as we only very rarely let SQLAlchemy
    #   itself actually save anything to the database.)
    _id = sa.Column(
        sqlUUID,
        primary_key=True,
        index=True,
        default=uuid.uuid4,            # This is the one exception to always using server_default
        doc="Unique identifier for this row",
    )

    @property
    def id( self ):
        """If the id is None, make one."""

        if self._id is None:
            self._id=uuid.uuid4()
        return self._id

    @id.setter
    def id( self, val ):
        self._id = asUUID( val )

    @classmethod
    def get_by_id( cls, uuid, session=None ):
        """Get an object of the current class that matches the given uuid.

        Returns None if not found.
        """
        with SmartSession( session ) as sess:
            return sess.query( cls ).filter( cls._id==uuid ).first()

    @classmethod
    def get_batch_by_ids( cls, uuids, session=None, return_dict=False ):
        """Get objects whose ids are in the list uuids.

        Parameters
        ----------
          uuids: list of UUID
            The object IDs whose corresponding objects you want.

          session: SQLAlchmey session or None

          return_dict: bool, default False
            If False, just return a list of objects.  If True, return a
            dict of { id: object }.

        """

        with SmartSession( session ) as sess:
            objs = sess.query( cls ).filter( cls._id.in_( uuids ) ).all()
        return { o.id: o for o in objs } if return_dict else objs



class SpatiallyIndexed:
    """A mixin for tables that have ra and dec fields indexed via q3c."""

    # Subclasses of this class must include the following in __table_args__:
    #   sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec))

    # @declared_attr
    # def __table_args__( cls ):
    #     # ...this doesn't seem to work the way I want.  What I want is for subclasses to
    #     # inherit and run all the __table_args__ from all of their superclasses, but
    #     # in practice it doesn't seem to really work that way.  So, we fall back to
    #     # the manual solution in the comment above.
    #     return (
    #         sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
    #     )

    ra = sa.Column(sa.Double, nullable=False, doc='Right ascension in degrees')

    dec = sa.Column(sa.Double, nullable=False, doc='Declination in degrees')

    gallat = sa.Column(sa.Double, index=True, doc="Galactic latitude of the target. ")

    gallon = sa.Column(sa.Double, index=False, doc="Galactic longitude of the target. ")

    ecllat = sa.Column(sa.Double, index=True, doc="Ecliptic latitude of the target. ")

    ecllon = sa.Column(sa.Double, index=False, doc="Ecliptic longitude of the target. ")

    def calculate_coordinates(self):
        """Fill self.gallat, self.gallon, self.ecllat, and self.ecllong based on self.ra and self.dec."""

        if self.ra is None or self.dec is None:
            return

        self.gallat, self.gallon, self.ecllat, self.ecllon = radec_to_gal_ecl( self.ra, self.dec )


    @hybrid_method
    def within( self, fourcorn ):
        """An SQLAlchemy filter to find all things within a FourCorners object

        Parameters
        ----------
          fourcorn: FourCorners
            A FourCorners object

        Returns
        -------
          An expression usable in a sqlalchemy filter

        """

        return func.q3c_poly_query( self.ra, self.dec,
                                    sqlarray( [ fourcorn.ra_corner_00, fourcorn.dec_corner_00,
                                                fourcorn.ra_corner_01, fourcorn.dec_corner_01,
                                                fourcorn.ra_corner_11, fourcorn.dec_corner_11,
                                                fourcorn.ra_corner_10, fourcorn.dec_corner_10 ] ) )

    @classmethod
    def cone_search( cls, ra, dec, rad, radunit='arcsec', ra_col='ra', dec_col='dec' ):
        """Find all objects of this class that are within a cone.

        Parameters
        ----------
          ra: float
            The central right ascension in decimal degrees
          dec: float
            The central declination in decimal degrees
          rad: float
            The radius of the circle on the sky
          radunit: str
            The units of rad.  One of 'arcsec', 'arcmin', 'degrees', or
            'radians'.  Defaults to 'arcsec'.
          ra_col: str
            The name of the ra column in the table.  Defaults to 'ra'.
          dec_col: str
            The name of the dec column in the table.  Defaults to 'dec'.

        Returns
        -------
          A query with the cone search.

        """
        if radunit == 'arcmin':
            rad /= 60.
        elif radunit == 'arcsec':
            rad /= 3600.
        elif radunit == 'radians':
            rad *= 180. / math.pi
        elif radunit != 'degrees':
            raise ValueError( f'SpatiallyIndexed.cone_search: unknown radius unit {radunit}' )

        return func.q3c_radial_query( getattr(cls, ra_col), getattr(cls, dec_col), ra, dec, rad )

    def distance_to(self, other, units='arcsec'):
        """Calculate the angular distance between this object and another object."""
        if not isinstance(other, (SpatiallyIndexed, SkyCoord)):
            raise ValueError(f'Cannot calculate distance between {type(self)} and {type(other)}')

        coord1 = SkyCoord(self.ra, self.dec, unit='deg')
        coord2 = SkyCoord(other.ra, other.dec, unit='deg')

        return coord1.separation(coord2).to(units).value


class FourCorners:
    """A mixin for tables that have four RA/Dec corners"""

    ra_corner_00 = sa.Column( sa.REAL, nullable=False, index=True,
                              doc="RA of the low-RA, low-Dec corner (degrees)" )
    ra_corner_01 = sa.Column( sa.REAL, nullable=False, index=True,
                              doc="RA of the low-RA, high-Dec corner (degrees)" )
    ra_corner_10 = sa.Column( sa.REAL, nullable=False, index=True,
                              doc="RA of the high-RA, low-Dec corner (degrees)" )
    ra_corner_11 = sa.Column( sa.REAL, nullable=False, index=True,
                              doc="RA of the high-RA, high-Dec corner (degrees)" )
    dec_corner_00 = sa.Column( sa.REAL, nullable=False, index=True,
                               doc="Dec of the low-RA, low-Dec corner (degrees)" )
    dec_corner_01 = sa.Column( sa.REAL, nullable=False, index=True,
                               doc="Dec of the low-RA, high-Dec corner (degrees)" )
    dec_corner_10 = sa.Column( sa.REAL, nullable=False, index=True,
                               doc="Dec of the high-RA, low-Dec corner (degrees)" )
    dec_corner_11 = sa.Column( sa.REAL, nullable=False, index=True,
                               doc="Dec of the high-RA, high-Dec corner (degrees)" )

    # These next four can be calcualted from the columns above, but are here to speed up
    #   searches.  They are filled assuming that no RA/Dec goes outside the corners,
    #   which isn't strictly true on a sphere, but damn close for the sizes of
    #   things we're going to be dealing with.
    # ra is cyclic in the range [0,360), so maxra may be less than
    #   minra, e.g. maxra=1, minra=359 is a 2° ra range cenetered on 0.
    minra = sa.Column( sa.REAL, nullable=False, index=True, doc="Min RA of image (degrees)" )
    maxra = sa.Column( sa.REAL, nullable=False, index=True, doc="Max RA of image (degrees)" )
    mindec = sa.Column( sa.REAL, nullable=False, index=True, doc="Min Dec of image (degrees)" )
    maxdec = sa.Column( sa.REAL, nullable=False, index=True, doc="Max Dec of image (degrees)" )


    @classmethod
    def sort_radec( cls, ras, decs ):
        """Sort ra and dec lists so they're each in the order in models.base.FourCorners

        Parameters
        ----------
          ras: list of float
             Four ra values in a list.
          decs: list of float
             Four dec values in a list.

        Returns
        -------
          Two new lists (ra, dec) sorted so they're in the order:
          (lowRA,lowDec), (lowRA,highDec), (highRA,lowDec), (highRA,highDec)

        """

        if len(ras) != 4:
            raise ValueError(f'ras must be a list/array with exactly four elements. Got {ras}')
            raise ValueError(f'decs must be a list/array with exactly four elements. Got {decs}')

        raorder = list( range(4) )
        raorder.sort( key=lambda i: ras[i] )

        # Try to detect an RA that spans 0.
        if ras[raorder[3]] - ras[raorder[0]] > 180.:
            newras = []
            for ra in ras:
                if ra > 180.:
                    newras.append( ra - 360. )
                else:
                    newras.append( ra )
            raorder.sort( key=lambda i: newras[i] )

        # Of two lowest ras, of those, pick the one with the lower dec;
        #   that's lowRA,lowDec; the other one is lowRA, highDec

        dex00 = raorder[0] if decs[raorder[0]] < decs[raorder[1]] else raorder[1]
        dex01 = raorder[1] if decs[raorder[0]] < decs[raorder[1]] else raorder[0]

        # Same thing, only now high ra

        dex10 = raorder[2] if decs[raorder[2]] < decs[raorder[3]] else raorder[3]
        dex11 = raorder[3] if decs[raorder[2]] < decs[raorder[3]] else raorder[2]

        return ( [  ras[dex00],  ras[dex01],  ras[dex10],  ras[dex11] ],
                 [ decs[dex00], decs[dex01], decs[dex10], decs[dex11] ] )


    @classmethod
    def find_containing_siobj( cls, siobj, session=None ):
        """Return all images (or whatever) that contain the given SpatiallyIndexed thing

        Parameters
        ----------
          siobj: SpatiallyIndexed
            A single object that is spatially indexed

        Returns
        -------
           An sql query result thingy.

        """

        # Overabundance of caution to avoid Bobby Tables.
        # (Because python is not strongly typed, siobj.ra and
        # siobj.dec could be set to anything.)
        ra = float( siobj.ra )
        dec = float( siobj.dec )
        return cls.find_containing( ra, dec, session=session )

    @classmethod
    def _find_possibly_containing_temptable( cls, ra, dec, session, prov_id=None ):
        """Internal.

        Looks for all cls objects where ra, dec is between minra:maxra,
        mindec:maxdec.  This will be a superset of the images that
        contain ra, dec.

        Lots of special case code for images that cross RA 0.

        Loads up the temp table temp_find_containing

        Parameters
        ----------
          ra, dec : float
             Coordinates to search for; deciam degrees.

          session : Session
             Required here, otherwise the temp table would be useless.

          prov_id : str, list of str, or None
             If not None, search for objects with this provenance, or any of these provenances if a list.

        """
        session.execute( sa.text( "DROP TABLE IF EXISTS temp_find_containing" ) )

        # Shouldn't need this, but just in case somebody gave us a wrapped RA:
        while ( ra < 0 ): ra += 360.
        while ( ra >= 360.): ra -= 360.

        query = ( "SELECT i._id, i.ra_corner_00, i.ra_corner_01, i.ra_corner_10, i.ra_corner_11, "
                  "       i.dec_corner_00, i.dec_corner_01, i.dec_corner_10, i.dec_corner_11 "
                  "INTO TEMP TABLE temp_find_containing "
                  f"FROM {cls.__tablename__} i "
                  "WHERE ( "
                  "  ( maxdec >= :dec AND mindec <= :dec ) "
                  "  AND ( "
                  "    ( (maxra > minra ) AND "
                  "      ( maxra >= :ra AND minra <= :ra ) )"
                  "    OR "
                  "    ( ( maxra < minra ) AND "
                  "      ( ( maxra >= :ra OR :ra > 180. ) AND ( minra <= :ra OR :ra <= 180. ) ) )"
                  "  )"
                  ")"
                 )
        subdict = { "ra": ra, "dec": dec }
        if prov_id is not None:
            if isinstance( prov_id, str ):
                query += " AND provenance_id=:prov"
                subdict['prov'] = prov_id
            elif isinstance( prov_id, list ):
                query += " AND provenance_id IN :prov"
                subdict['prov'] = tuple( prov_id )
            else:
                raise TypeError( "prov_id must be a a str or a list of str" )

        session.execute( sa.text( query ), subdict )


    @classmethod
    def find_containing( cls, ra, dec, prov_id=None, session=None ):
        """Return all objects in this class that contain the given RA and Dec

        Parameters
        ----------
          ra, dec: float, decimal degrees

          prov_id : str, list of str, or None
             If not None, search for objects with this provenance, or any of these provenances if a list.

        Returns
        -------
          A list of objects of cls.

        """
        # This should protect against SQL injection
        ra = float(ra) if isinstance(ra, int) else ra
        dec = float(dec) if isinstance(dec, int) else dec
        if ( not isinstance( ra, float ) ) or ( not isinstance( dec, float ) ):
            raise TypeError( f"(ra,dec) must be floats, got ({type(ra)},{type(dec)})" )

        # Becaue q3c_poly_query uses an index on ra, dec, just using
        # that directly wouldn't use any index here, meaning every row
        # of the table would have to be scanned and passed through the
        # polygon check.  To make the query faster, we first call
        # _find_possibly_containing_temptable that does a
        # square-to-the-sky search using minra, maxra, mindec, maxdec
        # (which *are* indexed) to greatly reduce the number of things
        # we'll q3c_poly_query.

        with SmartSession( session ) as sess:
            cls._find_possibly_containing_temptable( ra, dec, sess, prov_id=prov_id )
            query = sa.text( f"SELECT i.* FROM {cls.__tablename__} i "
                             f"INNER JOIN temp_find_containing t ON t._id=i._id "
                             f"WHERE q3c_poly_query( {ra}, {dec}, ARRAY[ t.ra_corner_00, t.dec_corner_00, "
                             f"                                          t.ra_corner_01, t.dec_corner_01, "
                             f"                                          t.ra_corner_11, t.dec_corner_11, "
                             f"                                          t.ra_corner_10, t.dec_corner_10 ])" )
            objs = sess.scalars( sa.select( cls ).from_statement( query ) ).all()
            sess.execute( sa.text( "DROP TABLE temp_find_containing" ) )
            return objs

    @classmethod
    def _find_potential_overlapping_temptable( cls, fcobj, session, prov_id=None ):
        """Internal.

        Given a FourCorners object fcobj, will return all objects of
        this class that *might* overlap that object.  It does this by
        making sure that each object's min(ra,dec) is less than the
        other object's max(ra,dec).  If all four of those criteria are
        true, then we have a potential overlap.

        (...except for the special case of one or both images including
        RA=0°, when things are a bit more complicated.)

        Parameters
        ----------
          fcobj : FourCorners

          session : Session
             required here; otherwise, the temp table wouldn't be useful

          prov_id: str, list of str, or None
             id or ids of the provenance of cls objects to search; if
             None, won't filter on provenance

        """

        session.execute( sa.text( "DROP TABLE IF EXISTS temp_find_overlapping" ) )

        # All kinds of special cases (everything from the first OR
        # onwards) below to deal with the the case where RA crosses 0
        # TODO : speed tests once we have a big enough database for that
        # to matter to see how much this hurts us.

        query = ( "SELECT i._id, i.ra_corner_00, i.ra_corner_01, i.ra_corner_10, i.ra_corner_11, "
                  "       i.dec_corner_00, i.dec_corner_01, i.dec_corner_10, i.dec_corner_11 "
                  "INTO TEMP TABLE temp_find_overlapping "
                  f"FROM {cls.__tablename__} i "
                  "WHERE ( "
                  "  ( i.maxdec >= :mindec AND i.mindec <= :maxdec ) "
                  "  AND "
                  "  ( ( ( i.maxra >= i.minra AND :maxra >= :minra ) AND "
                  "      i.maxra >= :minra AND i.minra <= :maxra ) "
                  "    OR "
                  "    ( i.maxra < i.minra AND :maxra < :minra ) "   # both include RA=0, will overlap in RA
                  "    OR "
                  "    ( ( i.maxra < i.minra AND :maxra >= :minra AND :minra <= 180. ) AND "
                  "      i.maxra >= :minra ) "
                  "    OR "
                  "    ( ( i.maxra < i.minra AND :maxra >= :minra AND :minra > 180. ) AND "
                  "      i.minra <= :maxra ) "
                  "    OR "
                  "    ( ( i.maxra >= i.minra AND :maxra < :minra AND i.maxra <= 180. ) AND "
                  "      i.minra <= :maxra ) "
                  "    OR "
                  "    ( ( i.maxra >= i.minra AND :maxra < :minra AND i.maxra > 180. ) AND "
                  "      i.maxra >= :minra ) "
                  "  )"
                  ") " )
        subdict = { 'minra': fcobj.minra, 'maxra': fcobj.maxra,
                    'mindec': fcobj.mindec, 'maxdec': fcobj.maxdec }
        if prov_id is not None:
            if isinstance( prov_id, str ):
                query += " AND provenance_id=:prov"
                subdict['prov'] = prov_id
            elif isinstance( prov_id, list ):
                query += " AND provenance_id IN :prov"
                subdict['prov'] = tuple( prov_id )
            else:
                raise TypeError( "prov_id must be a a str or a list of str" )

        session.execute( sa.text( query ), subdict )

    @classmethod
    def find_potential_overlapping( cls, fcobj, prov_id=None, session=None ):
        """Return all objects of this class that *might* overlap FourCorners object fcobj.

        This will in general be a superset of things that actually do
        overlap.  To do this, it defines NS-EW bounding rectangles for
        cls objects.  (We're assuming that the spherical trig isn't
        going to kill us here, so this may get wonky with big Δra/Δdec
        or right near the poles.)  This box is defined by the least/greatest
        RA/dec of all four corners.  (Below: the actual image is tilted
        rectangle (modulo your font aspect ratio), the bounding box is the
        one square to the screen.)

                  __
                 │╱╲│
                 │╲╱│
                  ‾‾

        Parameters
        ----------
          fcobj: FourCorners object
             The FourCorners object to look for overlaps with.

          prov_id: str
             The ide of the provenance of objects in this class to search for

          session: Session
             (Optional) SA session.

        Returns
        -------
          The result of a sess.scalars(...).all() with members of this class.

        """
        with SmartSession( session ) as sess:
            cls._find_potential_overlapping_temptable( fcobj, sess, prov_id=prov_id )
            objs = sess.scalars( sa.select( cls )
                                 .from_statement( sa.text( "SELECT _id FROM temp_find_overlapping" ) )
                                ).all()
            sess.execute( sa.text( "DROP TABLE temp_find_overlapping" ) )
            return objs

    @classmethod
    def get_overlap_frac(cls, obj1, obj2):
        """Calculate the overlap fraction between two objects that have four corners.

        Returns
        -------
        overlap_frac: float
            The fraction of obj1's area that is covered by the intersection of the objects

        Assumes that the images are small enough that a simple cos(dec)
        correction for RA is enough that we can assume that the sky is
        flat.  This assumption will break down near the poles.

        """

        o1ra = np.array( [ [ obj1.ra_corner_00, obj1.ra_corner_01 ], [ obj1.ra_corner_10, obj1.ra_corner_11 ] ] )
        o2ra = np.array( [ [ obj2.ra_corner_00, obj2.ra_corner_01 ], [ obj2.ra_corner_10, obj2.ra_corner_11 ] ] )
        o1dec = np.array( [ [ obj1.dec_corner_00, obj1.dec_corner_01 ], [ obj1.dec_corner_10, obj1.dec_corner_11 ] ] )
        o2dec = np.array( [ [ obj2.dec_corner_00, obj2.dec_corner_01 ], [ obj2.dec_corner_10, obj2.dec_corner_11 ] ] )

        # Have to handle the case of ra spanning 0.  This happens when
        # maxra < minra.  In that case, take all ras > 180 and subtract
        # 360 to make them negative.  Subsequent computations will then
        # work.  This will break horribly if the size of the image
        # approaches 180°, but that's an absurd case that should never
        # happen.  (If you're using this pipeline with some sort of
        # fisheye all-sky camera, then... well, sorry.  All kinds of things
        # are probably going to break having to do with coordinates.)
        if ( obj1.maxra < obj1.minra ) or ( obj2.maxra < obj2.minra ):
            o1ra[ o1ra > 180. ] -= 360.
            o2ra[ o2ra > 180. ] -= 360.

        # Really cheesy spherical trig.  Multiply all RAs by cos(dec).
        #   This will move them on the sky, but it will move them all
        #   together so that doesn't matter for area computations.  More
        #   importantly, it will make all the relative positions
        #   approximately correct in units of linear degrees under the
        #   assumption that the surface of a sphere is "flat enough"
        #   within the area covered by the images.  Use dec1 as our dec
        #   because that's the reference.  (For things where dec is far
        #   from each other, this isn't really right for obj2, but in
        #   that case, they won't overlap anyway, so we'll still get
        #   intersection area 0 and it won't matter.)
        o1ra *= np.cos( obj1.dec * np.pi / 180. )
        o2ra *= np.cos( obj1.dec * np.pi / 180. )

        obj1 = shapely.Polygon( ( ( o1ra[0,0], o1dec[0,0] ),
                                  ( o1ra[1,0], o1dec[1,0] ),
                                  ( o1ra[1,1], o1dec[1,1] ),
                                  ( o1ra[0,1], o1dec[0,1] ),
                                  ( o1ra[0,0], o1dec[0,0] ) )
                               )
        obj2 = shapely.Polygon( ( ( o2ra[0,0], o2dec[0,0] ),
                                  ( o2ra[1,0], o2dec[1,0] ),
                                  ( o2ra[1,1], o2dec[1,1] ),
                                  ( o2ra[0,1], o2dec[0,1] ),
                                  ( o2ra[0,0], o2dec[0,0] ) )
                               )

        return obj1.intersection( obj2 ).area / obj1.area


    def contains( self, ra, dec ):
        """Return True if ra, dec is contained within the four corners."""

        corners = np.array( [ [ self.ra_corner_00, self.dec_corner_00 ],
                              [ self.ra_corner_01, self.dec_corner_01 ],
                              [ self.ra_corner_11, self.dec_corner_11 ],
                              [ self.ra_corner_10, self.dec_corner_10 ],
                              [ self.ra_corner_00, self.dec_corner_00 ] ] )
        if self.maxra < self.minra:
            corners[ corners[:,0]>180, 0 ] -= 360.
            if ra > 180.:
                ra -= 360.

        obj = shapely.Polygon( corners )
        return obj.contains( shapely.Point( ra, dec ) )


class HasBitFlagBadness:
    """A mixin class that adds a bitflag marking why this object is bad. """
    _bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '0' ),
        index=True,
        doc='Bitflag for this object. Good objects have a bitflag of 0. '
            'Bad objects are each bad in their own way (i.e., have different bits set). '
            'The bitflag will include this value, bit-wise-or-ed with the bitflags of the '
            'upstream object that were used to make this one. '
    )

    @declared_attr
    def _upstream_bitflag(cls):
        if cls.__name__ != 'Exposure':
            return sa.Column(
                sa.BIGINT,
                nullable=False,
                server_default=sa.sql.elements.TextClause( '0' ),
                index=True,
                doc='Bitflag of objects used to generate this object. '
            )
        else:
            return None

    @hybrid_property
    def bitflag(self):
        if self._bitflag is None:
            self._bitflag = 0
        if self._upstream_bitflag is None:
            self._upstream_bitflag = 0
        return self._bitflag | self._upstream_bitflag

    @bitflag.inplace.expression
    @classmethod
    def bitflag(cls):
        return cls._bitflag.op('|')(cls._upstream_bitflag)

    @bitflag.inplace.setter
    def bitflag(self, value):
        raise RuntimeError( "Don't use this, use set_badness" )
        # allowed_bits = 0
        # for i in self._get_inverse_badness().values():
        #     allowed_bits += 2 ** i
        # if value & ~allowed_bits != 0:
        #     raise ValueError(f'Bitflag value {bin(value)} has bits set that are not allowed.')
        # self._bitflag = value

    @property
    def own_bitflag( self ):
        return self._bitflag

    @own_bitflag.setter
    def own_bitflag( self, val ):
        raise RuntimeError( "Don't use this ,use set_badness" )

    @property
    def own_badness( self ):
        """A comma separated string of keywords describing why this data is bad.

        Does not include badness inherited from upstream objects; use badness
        for that.

        """
        return bitflag_to_string( self._bitflag, data_badness_dict )

    @own_badness.setter
    def own_badness( self, value ):
        raise RuntimeError( "Don't use this, use set_badness()" )

    @property
    def badness(self):
        """A comma separated string of keywords describing why this data is bad, including upstreams.

        Based on the bitflag.  This includes all the reasons this data is bad,
        including the parent data models that were used to create this data
        (e.g., the Exposure underlying the Image).

        """
        return bitflag_to_string (self.bitflag, data_badness_dict )

    @badness.setter
    def badness( self, value ):
        raise RuntimeError( "Don't set badness, use set_badness." )

    def _set_bitflag( self, value=None, commit=True ):
        """Set the objects bitflag to the integer value.

        See set_badness

        """
        if value is not None:
            self._bitflag = value
        if commit and ( self.id is not None ):
            with SmartSession() as sess:
                sess.execute( sa.text( f"UPDATE {self.__tablename__} SET _bitflag=:bad WHERE _id=:id" ),
                              { "bad": self._bitflag, "id": self.id } )
                sess.commit()

    def set_badness( self, value=None, commit=True ):
        """Set the badness for this image using a comma separated string.

        In general, you should *not* set the bits that are bad only because an
        upstream is bad, but just the ones that are bade specifically from
        this image.

        DEVELOPER NOTE: any object that inherits from HasBitFlagBadness must
        have an id property.  This will be the case for objects that inherit
        from UUIDMixin, as most of ours do.

        Parameters
        ----------
          value: str or None
            If str, a comma-separated string indicating the badnesses to set.
            If None, it means save this object's own bitflag as is to the
            database.  It doesn't make sense to use value=None and
            commit=False.

          commit: bool, default True
            If True, and the object is already in the database, will save the
            bitflag changes to the database.  If False, then it's the
            responsibility of the calling function to make sure they get saved
            if necessary.  (That can be accomplished with a subsequent call to
            obj.set_badness( None, commit=True ).)

            (If the object isn't already in the database, then nothing gets
            saved.  However, in that case, when the object is later saved, it
            will get saved with its value of _bitflag then, so things will all
            work out in the end.)

        """

        if value is not None:
            value = string_to_bitflag( value, self._get_inverse_badness() )
        self._set_bitflag( value, commit=commit )


    def append_badness( self, value, commit=True ):
        """Add badness (comma-separated string of keywords) to the object.

        Parameters
        ----------
          value: str

          commit: bool, default True
            If false, won't commit to the database.  (See set_badness.)

        """

        self._set_bitflag( self._bitflag | string_to_bitflag( value, self._get_inverse_badness() ), commit=commit )

    description = sa.Column(
        sa.Text,
        nullable=True,
        doc='Free text comment about this data product, e.g., why it is bad. '
    )

    def __init__(self):
        self._bitflag = 0
        self._upstream_bitflag = 0

    def update_downstream_badness(self, session=None, commit=True, siblings=True, objbank=None):
        """Send a recursive command to update all downstream objects that have bitflags.

        Since this function is called recursively, it always updates the
        current object's _upstream_bitflag to reflect the state of this
        object's immediate upstreams, before calling the same function on all
        downstream objects.

        Note that this function will session.merge() this object and all its
        recursive downstreams (to update the changes in bitflag) and will
        commit the new changes on its own (unless given commit=False)
        but only at the end of the recursion.

        If session=None and commit=False an exception is raised.

        Parameters
        ----------
        session: sqlalchemy session
            The session to use for the update. If None, will open a new session,
            which will also close at the end of the call. In that case, must
            provide a commit=True to commit the changes.

        commit: bool (default True)
            Whether to commit the changes to the database.

        siblings: bool (default True)
            Whether to also update the siblings of this object.
            Default is True. This is usually what you want, but
            anytime this function calls itself, it uses siblings=False,
            to avoid infinite recursion.

        objbank: dict
            Don't pass this, it's only used internally.

        """

        if objbank is None:
            objbank = {}

        with SmartSession(session) as session:
            # Before the database refactor, this was done with
            # SQLAlchemy, and worked.  Afterwards, even though in this
            # one place I tried to keep them all in one session, it
            # didn't work.  What was happening was that when an object,
            # merged into the session, was changed here, that same
            # object (i.e. same memory location) was *not* being pulled
            # out from the queries in image.get_upstreams(), even though
            # session was passed on to get_upstreams().  So, things
            # weren't propagating right.  Something about session
            # querying and merging wasn't working right.  (WHAT?
            # Confusion with SQLAlchemy merging?  Never!)
            #
            # So, rather than fully trusting the mysteriousness of
            # sqlalchemy sessions, use an object bank that we pass
            # recursively, to make sure that every time we want to refer
            # an object of a given id, we refer to the same object in
            # memory.  That way, we can be sure that changes we make
            # during the recursion will stick.  (We're still trusting SA
            # that when we commit, because we merged all of those
            # objects, the changes to them will get sent in to the
            # databse.  Fingers crossed.  merge is always scary.)

            if self.id not in objbank.keys():
                merged_self = session.merge(self)
                objbank[ merged_self.id ] = merged_self
            merged_self = objbank[ self.id ]

            new_bitflag = 0  # start from scratch, in case some upstreams have lost badness
            for upstream in merged_self.get_upstreams( session=session ):
                if upstream.id in objbank.keys():
                    upstream = objbank[ upstream.id ]
                if hasattr(upstream, '_bitflag'):
                    new_bitflag |= upstream.bitflag

            if hasattr(merged_self, '_upstream_bitflag'):
                merged_self._upstream_bitflag = new_bitflag
                self._upstream_bitflag = merged_self._upstream_bitflag

            # recursively do this for all downstream objects
            for downstream in merged_self.get_downstreams(session=session, siblings=siblings):
                if hasattr(downstream, 'update_downstream_badness') and callable(downstream.update_downstream_badness):
                    downstream.update_downstream_badness(session=session, siblings=False, commit=False, objbank=objbank)

            if commit:
                session.commit()

    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object

        For the base class this is the most inclusive inverse (allows all badness).
        """
        return data_badness_inverse


if __name__ == "__main__":
    pass
