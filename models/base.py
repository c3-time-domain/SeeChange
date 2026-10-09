import warnings
import os
import time
import math
import types
import hashlib
import pathlib
import json
import datetime
import uuid
import socket
import threading
import functools
import logging
import textwrap
from uuid import UUID
from contextlib import contextmanager

import numpy as np
import shapely

import astropy.wcs
import astropy.units as u
from astropy.coordinates import SkyCoord

import psycopg
from psycopg import sql
# import psycopg.adapt

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql
from sqlalchemy import func, orm
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.dialects.postgresql import array as sqlarray
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import OperationalError

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
from util.util import asUUID, NumpyAndUUIDJsonEncoder, listify, retry_with_sleep

# Postgres adapters to allow insertion of some numpy types
# ...let's see if we can get by without these in psycopg3

# def _adapt_numpy_float_psycopg2( val ):
#     if np.isnan( val ):
#         return psycopg2.extensions.AsIs( "'NaN'::float" )
#     else:
#         return psycopg2.extensions.AsIs( val )


# def _adapt_numpy_int_psycopg2( val ):
#     return psycopg2.extensions.AsIs( val )


# psycopg2.extensions.register_adapter( np.float64, _adapt_numpy_float_psycopg2 )
# psycopg2.extensions.register_adapter( np.float32, _adapt_numpy_float_psycopg2 )
# psycopg2.extensions.register_adapter( np.int64, _adapt_numpy_int_psycopg2 )
# psycopg2.extensions.register_adapter( np.int32, _adapt_numpy_int_psycopg2 )


# this is the root SeeChange folder
CODE_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
#
# # printout the list of relevant environmental variables:
# import io
# strio = io.StringIO()
# strio.write( "SeeChange environment variables:\n" )
# for key in [
#     'INTERACTIVE',
#     'LIMIT_CACHE_USAGE',
#     'SKIP_NOIRLAB_DOWNLOADS',
#     'RUN_SLOW_TESTS',
#     'SEECHANGE_TRACEMALLOC',
# ]:
#     strio.write( f'   {key}: {os.getenv(key)}\n' )
# SCLogger.info( strio.getvalue() )


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
# _Session isn't actually a SQLAlchemy session, it's a sessionmaker
_Session = None
_psycopg_params = None


def _get_psycopg_params():
    global _psycopg_params

    if _psycopg_params is None:
        cfg = config.Config.get()
        if cfg.value( "db.engine" ) != "postgresql+psycopg":
            raise ValueError( "This pipeline only supports PostgreSQL as a database engine" )
        if psycopg.__version__[0] != '3':
            raise ValueError( "This pipeline requires psycopg version 3." )

        password = cfg.value( 'db.password' )
        if password is None:
            if cfg.value( "db.password_file" ) is None:
                raise RuntimeError( "Must specify either db.password or db.password_file in config" )
            with open( cfg.value( "db.password_file" ) ) as ifp:
                password = ifp.readline().strip()

        # psycopg docs seems to suggest that the client_encoding parameter isn't necessary,
        #   but empirically it is.
        _psycopg_params = { 'engine': cfg.value('db.engine'),
                            'host': cfg.value('db.host'),
                            'port': cfg.value('db.port'),
                            'dbname': cfg.value('db.database'),
                            'user': cfg.value('db.user'),
                            'password': password,
                            'client_encoding': 'UTF8' }

    return _psycopg_params


def Session():
    """Make a SQLAlchemy session.

    This is primarily intended for interactive sessions where you're
    developing or testing.  In real code, you should use SmartSession in
    a context manager ("with SmartSession(...) as sess: ... ").  In
    fact, in real code, you should move towards using the PGDB context
    manager and stop using SQLAlchemy at all.

    Returns
    -------
    sqlalchemy.orm.session.Session
        A session object that doesn't automatically close.

    """
    global _Session, _engine

    if _Session is None:
        params = _get_psycopg_params()
        url = ( f'{params["engine"]}://{params["user"]}:{params["password"]}@{params["host"]}:{params["port"]}/'
                f'{params["dbname"]}?client_encoding=utf8')
        cfg = config.Config.get()
        _engine = sa.create_engine( url,
                                    future=True,
                                    poolclass=sa.pool.NullPool,
                                    connect_args={ "options": "-c timezone=UTC",
                                                   "connect_timeout": cfg.value("db.sa_connect_timeout")
                                                  }
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

    For new code, use the PGDB() context manager, and start writing SQL
    instead of using SQLAlchemy constructs.  (Issue #516.)

    """
    global _engine

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
            # manually rolling back with psycopg or whatever is
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

            # ...I found myself still left with a dangling session.  Not
            # a case of an explicit table lock, but where I wanted to do
            # something (truncate tables in test cleanup) that the
            # dangling session wasn't letting me do.  OMG I hate
            # sqlalchemy with a burning passion.


@contextmanager
def PsycopgConnection( current=None ):
    """Get a direct psycopg3 connection to the database; use this in a with statement.

    Useful if you don't want to fight with SQLAlchemy, e.g. if you
    want to use table locks (see comment above in SmartSession).

    For new code, use the PGDB class (in a with block) instead of using
    this function.

    Parameters
    ----------
      current : psycopg.Connection or None (default None)
         Pass an existing connection, get it back.  Useful if you are in
         nested functions that might want to be working within the same
         transaction.

    Returns
    -------
       psycopg.Connection

       After the with block, the connection will be rolled back and
       closed.  So, if you want what you've done committed, make sure to
       call the commit() method on the return value before the with
       block exits.

    """

    if current is not None:
        if not isinstance( current, psycopg.Connection ):
            raise TypeError( "Must pass a psycopg.Connection or None to PyscopgConection" )
        yield current
        # Don't roll back or close, because whoever created it in the
        #   first place is responsible for that.
        return

    # If a connection wasn't passed, make one, and then be sure to roll it back and close it when we're done

    conn = None
    try:
        params = _get_psycopg_params().copy()
        del params['engine']
        conn = psycopg.connect( **params )
        yield conn

    finally:
        # Just in case things were done, roll back.  Often, the caller
        #   will have done a conn.commit() (which it must if it wants to
        #   keep things that were done) or conn.rollback(), in which
        #   case this rollback is gratuitous.  However, we can't count
        #   on the caller having done that.  (E.g., if there's an
        #   exception, the caller may have short-circuited, which is why
        #   the yield is in a try and this cleaup is in a finally.)
        if conn is not None:
            conn.rollback()
            conn.close()


class PGDBTimings:
    def __init__( self ):
        self.reset()

    def reset( self ):
        self.last_query_time = None
        self.last_commit_time = None
        self.last_fetch_time = None
        self.tot_query_time = 0.
        self.tot_commit_time = 0.
        self.tot_fetch_time = 0.


class PGDB:
    """A class that encapsulates a psycopg connection to the databsae.

    Use this in a context manager:

       with PGDB() as dbcon:
           rows, cols = dbcon.execute( query, subdict )
           # do other things

    It will automatically close the connection when the with block ends.
    You can pass either a psycopg.connection or a PGDB object to the
    PGDB constructor, and it will reuse that connection; in that case,
    it *won't* close the connection, trusting wherever the connection
    came from to close it.  Do this if you call functions within a block
    where you already have an open connection; pass the PGDB object to
    the function, have the function use that in the constructor.

    Send queries using DBCon.execute() and DBCon.execute_nofetch()

    If for some reason you need access to the underlying cursor, you can
    get it from the cursor property.

    """

    _sleept = None
    _sleepmin = None
    _sleepfac = None
    _sleepfuzz = None
    _sleepmax = None

    def __init__( self, con=None, dictcursor=False,
                  sleept=0.25, sleepmin=0.125, sleepfac=2, sleepfuzz=0.1, sleepmax=4.0
                 ):
        """Instantiate.

        If you use this, either use it in a with block that doesn't last
        too long, or call close(), and soon.

        It will use the connection it is passed, or, if not, make a new
        connection and hold on to it.  It will relinquish the connection
        when you call the object's close method.  (This may or may not
        actually close the connection to the database, based on how the
        class was instantiated.)  Better, use this inside a context
        manager; then the connection is relinquished when the connection
        is released.  Example:

           with PGDB( oldconnection ) as pgdb:
               # do things

        Where oldconnection can be, ideally, a PGDB, but can also be a
        psycopg.Connection or a psycopg.Cursor.  (A sqlalchemy Session
        will also work, but we're trying to phase that out.)  The passed
        connection will be wrapped by this PGDB, and will *not* be
        closed when the with block (as whoever opened it and started
        this with block will still be expecting it to be there).  If
        oldconnectoin is None, then a new connection is made, and closed
        when the with block ends.

        In the event that you're opening a new connection, it will retry
        the connection if it fails with a psycopg.OperationalError.
        *Hopefully* this will work around temporary spates of there
        being too many connections to the database.  (If everybody is
        obeying the request to not hold database connections open too
        long.)  If you don't want it to do this, pass the optional
        parameter maxsleep=0. to the constructor.

        Parameters
        ----------
          con : psycopg.Connection, psycopg.Cursor, PGDB, or (shudder) sa.orm.session.Session
            If None (the default), will make a new connection, and will
            roll back and close it when done.  If not None, then will
            instead wrap this connection; when close() is called, or
            when the context manager that created this object ends, will
            roll back and close the connection.  However, if con is not
            None, then the assumption is that somebody else is managing
            the connection, so will not rollback or close.

          dictcursor : bool, default False
            If True, then the cursor uses psycopg.rows.dict_row as its
            row factory.  execite() will return a list of dictionaries,
            with each element of the list being one row of the result.
            If False, then execute returns two lists: a list of tuples
            (the rows) and a list of strings (the column names).

          sleept, sleepmin, sleepfac, sleepfuzz, sleepmax : passed to util.util.retry_with_sleep
            If it needs to make a new connection to the database, and
            the connection fails because of a psycopg.OperationalError,
            it will retry, configured by these values.  They default to
            values from config.db.*.

        """

        made_a_new_PGDB = True
        if con is not None:
            if isinstance( con, PGDB ):
                self.con = con.con
                self.timings = con.timings
                self.echoqueries = con.echoqueries
                self.alwaysexplain = con.alwaysexplain
                self.alwaysanalyze = con.alwaysanalyze
                made_a_new_PGDB = False
            elif isinstance( con, psycopg.Connection ):
                self.con = con
            elif isinstance( con, psycopg.Cursor ):
                self.con = con.connection
            elif isinstance( con, sa.orm.session.Session ):
                SCLogger.debug( "FEAR/LOATHING: You're using a SQLAlchemy Session, still trying "
                                "to make a PGDB from it (Issue #516)" )
                self.con = con.connection().connection.driver_connection
            else:
                raise TypeError( f"con must be None, a PGDB, a psycopg.Connection, a psycopg.Cursor, or a "
                                 f"sa.orm.session.Session (shudder), not a {type(con)}" )
            self._con_is_mine = False
        else:
            params = _get_psycopg_params().copy()
            del params['engine']
            connector = functools.partial( psycopg.connect, **params )
            if PGDB._sleept is None:
                cfg = config.Config.get()
                PGDB._sleept = cfg.value( 'db.sleept' )
                PGDB._sleepmin = cfg.value( 'db.sleepmin' )
                PGDB._sleepfac = cfg.value( 'db.sleepfac' )
                PGDB._sleepfuzz = cfg.value( 'db.sleepfuzz' )
                PGDB._sleepmax = cfg.value( 'db.sleepmax' )
            sleept = sleept if sleept is not None else PGDB._sleept
            sleepmin = sleepmin if sleepmin is not None else PGDB._sleepmin
            sleepfac = sleepfac if sleepfac is not None else PGDB._sleepfac
            sleepfuzz = sleepfuzz if sleepfuzz is not None else PGDB._sleepfuzz
            sleepmax = sleepmax if sleepmax is not None else PGDB._sleepmax
            self.con = retry_with_sleep( connector, sleepmin=sleepmin, sleept=sleept, sleepfac=sleepfac,
                                         sleepfuzz=sleepfuzz, sleepmax=sleepmax,
                                         failmessage=f"to connect to database {params['dbname']} on {params['host']}",
                                         accept_exceptions=psycopg.OperationalError )
            self._con_is_mine = True

        if made_a_new_PGDB:
            self.timings = PGDBTimings()
            cfg = config.Config.get()
            self.echoqueries = cfg.value( 'db.echoqueries' )
            self.alwaysexplain = cfg.value( 'db.alwaysexplain' )
            self.alwaysanalyze = cfg.value( 'db.alwaysanalyze' )

        self.dictcursor = dictcursor
        self.remake_cursor()


    def __enter__( self ):
        return self

    def __exit__( self, type, value, traceback ):
        self.close()

    def remake_cursor( self, dictcursor=None ):
        """Recreate the cursor used for database communication.

        Parameters
        ----------
          dictcursor : bool, default None
            If None, will make a cursor that returns dictionaries
            (vs. tuples) for rows based on what was passed to the
            dictcursor argument of the DBCon constructor.  If True,
            makes a cursor that will cause execute() to return a list of
            dictionaries.  If False, makes a cursor that will cause
            execute() to return two lists; the first is a list of tuples
            (the rows), the second is a list of strings (the column
            names).

        """
        self.curcursorisdict = self.dictcursor if dictcursor is None else dictcursor
        if self.curcursorisdict:
            self.cursor = self.con.cursor( row_factory=psycopg.rows.dict_row )
        else:
            self.cursor = self.con.cursor()


    def close( self ):
        """Rolls back and closes the connection if appropriate.

        If you did stuff you want kept, make sure to call commit.

        If the constructor was called with con=None, then the connection
        will be rolled back.  If the constructor was callled with a
        non-None none, then this method does nothing.  (In the latter case,
        this PGDB object is wrapping a connection that was made externally,
        so whoever made it is responsible for rolling back and closing.)

        """
        if self._con_is_mine:
            self.con.rollback()
            self.con.close()


    def rollback( self ):
        """Rollback any ongoing transaction.

        Also be all cavalier about python function calling overhead.

        """
        self.con.rollback()


    def commit( self ):
        """Commit changes to the database.

        Call this if you've done any INSERT or UPDATE or similar
        commands that change the database, and you want your commands to
        stick.

        """
        t0 = time.perf_counter()
        self.con.commit()
        t1 = time.perf_counter()
        self.timings.last_commit_time = t1 - t0
        self.timings.tot_commit_time += t1 - t0
        self.remake_cursor( self.curcursorisdict )  # ...is this necessary?


    def execute_nofetch( self, q, subdict={}, echo=None, explain=None, analyze=None ):
        """Runs a query where you don't expect to fetch results.

        Parameters are the same as execute(), except for analyze, which
        works just like explain, only it does "EXPLAIN ANALYZE" rather
        than just "EXPLAIN".  Returns nothing.

        """

        t0 = time.perf_counter()

        alreadydid = False
        if not isinstance( q, ( sql.SQL, sql.Composed ) ):
            q = sql.SQL( q )

        if SCLogger.instance().get().level <= logging.DEBUG:
            echo = echo if echo is not None else self.echoqueries
            explain = explain if explain is not None else self.alwaysexplain
            analyze = analyze if analyze is not None else self.alwaysanalyze
            if echo:
                SCLogger.debug( f"Sending query\n{q.as_string()}\nwith substitutions: {subdict}" )

            nl = '\n'
            if explain:
                SCLogger.debug( "Explaining..." )
                self.cursor.execute( sql.SQL("EXPLAIN ") + q, subdict )
                rows = self.cursor.fetchall()
                dex = 'QUERY PLAN' if self.curcursorisdict else 0
                SCLogger.debug( f"Query plan:\n{nl.join([r[dex] for r in rows])}" )
            if analyze:
                SCLogger.debug( "Doing EXPLAIN ANALYZE..." )
                self.cursor.execute( sql.SQL("EXPLAIN ANALYZE ") + q, subdict )
                alreadydid = True
                rows = self.cursor.fetchall()
                dex = 'QUERY PLAN' if self.curcursorisdict else 0
                SCLogger.debug( f"Query plan:\n{nl.join([r[dex] for r in rows])}" )

        # NOTE: if we ran with EXPLAIN ANALYZE, any side effects of the query happened!
        # So don't run the query again.  This is why execute() forces analyze to
        # be false, because you can't EXPLAIN ANALYZE the query and get the results
        # all in one call.
        if not alreadydid:
            self.cursor.execute( q, subdict )

        if ( SCLogger.instance().get().level <= logging.DEBUG ) and ( echo or explain ):
            SCLogger.debug( "Query complete." )

        t1 = time.perf_counter()
        self.timings.last_query_time = t1 - t0
        self.timings.tot_query_time += t1 - t0

    def execute( self, q, subdict={}, silent=False, echo=None, explain=None ):
        """Runs a query, and returns either (rows, columns) or just rows.

        Parmaeters
        ----------
          q : str or sql.SQL (or a subclass thereof)
            The query.  Use %(var)s in the string for a substitution, if
            necessary.  The key "var" must then show up in subdict.

          subdict : dict
            Substitution dictionary. For every %(var)s that shows up in
            q, there must be a key "var" in this dictionary with the
            value to be substituted.  Extra keys are ignored.  Do not
            pass this if q is a Composed; in that case, you've already
            built in the substitutions.

          echo : bool, default None
            If True, echo queries before sending them.  If False, don't.
            If None, use the default (self.echoqueries, initialized from
            the db.echoqueries config value).

          explain : bool, default None
            If True, before running the query run an EXPLAIN on it and
            send the output to debug logging.  If False, don't.  If
            None, use the default (self.alwaysexplain, initialized from
            the db.alwaysexplain config value).

            WARNING: use of this makes you susceptible to SQL injection
            attacks if you aren't completely and totally confident about
            where your SQL came from.  Do not get bobby tablesed!

        Returns
        -------
          If the current cursor is a dict cursor, returns a list of dictionaries.

          If the current cursor is not a dict cursor, returns two lists.
          The first is a list of lists, with the rows pulled from the
          dictionary.  The second is a list of column names.

        """
        self.execute_nofetch( q, subdict, echo=echo, explain=explain, analyze=False )
        if self.curcursorisdict:
            if self.cursor.description is None:
                return None
            else:
                if ( echo or explain ):
                    SCLogger.debug( "Fetching..." )
                t0 = time.perf_counter()
                rval = self.cursor.fetchall()
                t1 = time.perf_counter()
                self.timings.last_fetch_time = t1 - t0
                self.timings.tot_fetch_time += t1 - t0
                if ( echo or explain ):
                    SCLogger.debug( "...fetched" )
                return rval
        else:
            if self.cursor.description is None:
                return None, None
            if ( echo or explain ):
                SCLogger.debug( "Fetching..." )
            t0 = time.perf_counter()
            rows = self.cursor.fetchall()
            cols = [ desc[0] for desc in self.cursor.description ]
            t1 = time.perf_counter()
            self.timings.last_fetch_time = t1 - t0
            self.timings.tot_fetch_time += t1 - t0
            if ( echo or explain ):
                SCLogger.debug( "...fetched" )
            return rows, cols




def db_stat(obj):

    """Check the status of an object. It can be one of: transient, pending, persistent, deleted, detached."""
    for word in ['transient', 'pending', 'persistent', 'deleted', 'detached']:
        if getattr(sa.inspect(obj), word):
            return word


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

    @classmethod
    def create( cls, create_uuidify=True, **kwargs ):
        """Create an object.  Usually use this instead of the object constructor directly.

        For most objects, it just calls the object constructor.  Some classes may subclass this
        so that they can create objects of different classes based on what's passed.

        """
        if create_uuidify:
            mess = sa.inspect( cls )
            knowncols = { c.name: c for c in mess.columns }
            for k in kwargs:
                if ( k in knowncols ) and ( isinstance( knowncols[k].type, sqlUUID ) ):
                    kwargs[k] = None if kwargs[k] is None else asUUID( kwargs[k] )
        return cls( **kwargs )

    def __init__(self, **kwargs):
        # THIS NEXT ONE IS ILL-CONSIDERED.
        # For it to really be right, we need to go everywhere through the code
        #    and make sure we're setting it.  Perhaps this worked if you only
        #    ever used SQLAlchemy to pull things from the database (in which case
        #    __init__ isn't called), but that would be such a nightmare world to
        #    live in that I don't even want to think about it.
        # Should be removed as part if Issue #516.
        self.from_db = False  # let users know this object was newly created

        if hasattr(self, '_bitflag'):
            self._bitflag = 0
        if hasattr(self, '_upstream_bitflag'):
            self._upstream_bitflag = 0

        for k, v in kwargs.items():
            setattr(self, k, v)

    @orm.reconstructor
    def init_on_load(self):
        self.from_db = True  # let users know this object was loaded from the database

    def get_attribute_list(self):
        """Get a list of all attributes of this object.

        Does not including internal SQLAlchemy attributes, and database
        level attributes like id, created_at, etc.

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
                if not isinstance( getattr( self, key ), types.MethodType ):
                    setattr(self, key, value)


    @classmethod
    def _get_table_lock( cls, session, tablename=None ):
        """Never use this.  The code that uses this is already written.  Use it and get Bobby Tablesed."""

        raise RuntimeError( "_get_table_lock is deprecated.  Issue #516." )

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
            except OperationalError:
                sleeptime *= 2
                if sleeptime >= 16:
                    failed = True
                    break
                else:
                    SCLogger.warning( f"Timeout waiting for lock on {tablename}, sleeping {sleeptime}s and retrying." )
                    session.rollback()
                    time.sleep( sleeptime )
        if failed:
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
                val = datetime.datetime.now( tz=datetime.UTC )

            if isinstance( col.type, sqlalchemy.dialects.postgresql.json.JSONB ) and ( val is not None ):
                val = json.dumps( val, cls=NumpyAndUUIDJsonEncoder )
            elif isinstance( val, np.ndarray ):
                val = list( val )

            # if isinstance( val, list ):
            #     # Gotta handle nans manually; it looks like psycopg
            #     #   doesn't do this right in sql arrays :(
            #     # TODO : look into doing this with an adapter
            #     val = [ v if not( isinstance(v, float) and np.isnan(v) ) else None for v in val ]

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


    def insert( self, pgdb=None, session=None, nocommit=False, load_defaults=False ):
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
          pgdb, session: PGDB, psycopg.Connection, psycogp.Cursor, or sqlalchemy Session, or None
            Usually you do not want to pass this; it's mostly for other
            upsert etc. methods that cascade to this.  The two things
            are synoyms; if both are given, pgdb takes precedence.

          nocommit: bool, default False
            If True, run the statement to insert the object, but don't
            actually commit the database.  Do this if you want the
            insert to be inside a transaction you've started on pgdb.
            It doesn't make sense to set nocommit=True unless you've
            passed something in pgdb.

          load_defaults: bool, default False
            Normally, will *not* update self's fields with server
            default values.  Set this to True for that to happen.  (This
            will trigger an additional read from the database.)

        """

        pgdb = pgdb if pgdb is not None else session

        _ = self.id    # Make sure id is generated

        # Do this manually.  SQLAlchemy's Session.add was doing all
        # kinds of stuff behind the scenes that made it impossible to
        # really know what was happening to the database, was raising
        # errors, and was generally doing all the things that make me
        # wish that I had never even heard of SQLAlchemy in the first
        # place.  We *want* an exception if we try to insert something
        # that just already exists.

        cols, values = self._get_cols_and_vals_for_insert()
        subdict = { c: v for c,v in zip( cols, values ) if c != 'modified' }

        with PGDB( pgdb ) as pgdb:
            q = sql.SQL( "INSERT INTO {tab}({fields}) VALUES ({vals})"
                        ).format( tab=sql.Identifier(self.__tablename__),
                                  fields=sql.SQL(",").join( sql.Identifier(c) for c in subdict.keys() ),
                                  vals=sql.SQL(",").join( sql.SQL(f'%({c})s') for c in subdict.keys() )
                                 )
            pgdb.execute_nofetch( q, subdict )

            if load_defaults:
                dbobj = self.__class__.get_by_id( self.id, pgdb=pgdb )
                for col in sa.inspect( self.__class__ ).c:
                    if ( ( col.name == 'modified' ) or
                         ( ( col.server_default is not None ) and ( getattr( self, col.name ) is None ) )
                        ):
                        setattr( self, col.name, getattr( dbobj, col.name ) )

            if not nocommit:
                pgdb.commit()


    def upsert( self, pgdb=None, session=None, load_defaults=False, nocommit=False ):
        """Insert an object into the database, or update it if it's already there (using _id as the primary key).

        Will *not* update self's fields with server default values!
        Re-get the database row if you want that.

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
          pgdb, session: PGDB, psycopg.Connect, psycopg.Cursor, sa.orm.session.Session, or None
            Usually you don't want to pass this.  The two arguments are
            synonyms; if both are given, will use pgdb.

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

        _ = self.id   # Make sure that self._id is generated
        cols, values = self._get_cols_and_vals_for_insert()
        subdict = { c: v for c, v in zip( cols, values ) }
        subdict['modified'] = datetime.datetime.now( tz=datetime.UTC )
        basicdict = subdict.copy()
        del basicdict['modified']
        conflictdict = subdict.copy()
        if '_id' in conflictdict:
            del conflictdict['_id']

        q = sql.SQL( textwrap.dedent(
            """\
            INSERT INTO {tab}({fields})
            VALUES ({vals})
            ON CONFLICT( _id) DO UPDATE SET {conflict}
            """
        ) ).format( tab=sql.Identifier(self.__tablename__),
                    fields=sql.SQL(",").join( sql.Identifier(c) for c in basicdict.keys() ),
                    vals=sql.SQL(",").join( sql.SQL(f'%({c})s') for c in basicdict.keys() ),
                    conflict=sql.SQL(",").join( sql.SQL(f"{{c}}=%({c})s").format( c=sql.Identifier(c) )
                                                for c in conflictdict )
                   )
        with PGDB( pgdb if pgdb is not None else session ) as pgdb:
            pgdb.execute_nofetch( q, subdict )
            if not nocommit:
                pgdb.commit()

            if load_defaults:
                dbobj = self.__class__.get_by_id( self.id, pgdb=pgdb )
                for col in sa.inspect( self.__class__ ).c:
                    if ( ( col.name == 'modified' ) or
                         ( ( col.server_default is not None ) and ( getattr( self, col.name ) is None ) )
                        ):
                        setattr( self, col.name, getattr( dbobj, col.name ) )


    @classmethod
    def insert_list( cls, objects, pgdb=None, session=None, load_defaults=False, upsert=False, nocommit=False ):
        """Like upsert, but for a bunch of objects in a list, and tries to be sorta efficient about it.

        Do *not* use this with classes that have things like association
        tables that need to get updated (i.e. with Image, maybe
        eventually some others).

        All reference fields (ids of other objects) of the objects must
        be up to date.  If the referenced objects don't exist in the
        database already, you'll get integrity errors.

        Will update object id fields, but will not update any other
        object fields with database defaults.  Reload the rows from the
        table if that's what you need.

        "sorta efficient": it does it all in one transaction, but it
        issues a separate "INSERT" for each object in the list.  Use
        this for small numbers of things to insert (no more than of
        order 10¹, say).  For larger numbers, you really want to be
        using a PostgreSQL COPY.

        """

        if not all( [ isinstance( o, cls ) for o in objects ] ):
            raise TypeError( f"{cls.__name__}.upsert_list: passed objects weren't all of this class!" )

        if nocommit and ( pgdb is None ) and ( session is None ):
            raise ValueError( "If you set nocommit=True, you must pass a pgdb" )

        with PGDB( pgdb if pgdb is not None else session ) as pgdb:
            for obj in objects:
                _ = obj.id                 #  Make sure _id is generated
                cols, values = obj._get_cols_and_vals_for_insert()
                subdict = { c: v for c, v in zip( cols, values ) }
                subdict['modified'] = datetime.datetime.now( tz=datetime.UTC )
                basicdict = subdict.copy()
                del basicdict['modified']
                if upsert:
                    conflictclause = ( sql.SQL( "ON CONFLICT(_id) DO UPDATE SET {conflict}" )
                                       .format(
                                           conflict=sql.SQL(",").join(
                                               sql.SQL(f"{{c}}=%({c})s").format( c=sql.Identifier(c) )
                                               for c in subdict.keys() if c != '_id' )
                                       )
                                      )
                else:
                    conflictclause = sql.SQL( "" )


                q = sql.SQL( textwrap.dedent(
                    """\
                    INSERT INTO {tab}({fields})
                    VALUES ({vals})
                    {conflictclause}
                    """
                ) ).format( tab=sql.Identifier(cls.__tablename__),
                            fields=sql.SQL(",").join( sql.Identifier(c) for c in basicdict.keys() ),
                            vals=sql.SQL(",").join( sql.SQL(f'%({c})s') for c in basicdict.keys() ),
                            conflictclause=conflictclause )
                pgdb.execute_nofetch( q, subdict )
            if not nocommit:
                pgdb.commit()

            if load_defaults:
                for obj in objects:
                    dbobj = obj.__class__.get_by_id( obj.id, pgdb=pgdb )
                    for col in sa.inspect( obj.__class__).c:
                        if ( ( col.name == 'modified' ) or
                             ( ( col.server_default is not None ) and ( getattr( obj, col.name ) is None ) )
                            ):
                            setattr( obj, col.name, getattr( dbobj, col.name ) )


    @classmethod
    def upsert_list( cls, objects, pgdb=None, session=None, load_defaults=False ):
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

        cls.insert_list( objects, pgdb=pgdb, session=session, load_defaults=load_defaults, upsert=True )


    def _delete_from_database( self, pgdb=None ):
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

        with PGDB( pgdb ) as pgdb:
            pgdb.execute_nofetch( sql.SQL( "DELETE FROM {tab} WHERE _id={myid}" )
                                  .format( tab=sql.Identifier(self.__class__.__tablename__),
                                           myid=self.id ) )

            pgdb.commit()

        # Look how much easier this is when you don't have to spend a whole bunch of time
        #  deciding if the object needs to be merged, expunged, etc. to a session


    def get_upstream_ids(self, pgdb=None):
        """Get a list of tuples of (type, id) for all direct upstreams of this object (non-recursive)."""
        raise NotImplementedError( f'get_upstream_ids not implemented for this {self.__class__.__name__}' )

    def get_upstreams(self, nofile=False, pgdb=None, session=None):
        """Get all data products that were directly used to create this object (non-recursive)."""
        upstreams = []
        with PGDB( pgdb if pgdb is not None else session, dictcursor=True ) as pgdb:
            upstream_info = self.get_upstream_ids( pgdb=pgdb )
            for cls, upid in upstream_info:
                q = sql.SQL( "SELECT * FROM {tab} WHERE _id={objid}" ).format( tab=sql.Identifier(cls.__tablename__),
                                                                               objid=upid )
                rows = pgdb.execute( q )
                if len(rows) != 1:
                    raise RuntimeError( "This should never happen." )
                upstreams.append( cls.create( pgdb=pgdb, nofile=nofile, **(rows[0]) ) )
        return upstreams

    def get_downstream_ids(self, pgdb=None):
        """Get a list of tuples of (type, id) for all direct downstreams of this object (non-recursive)."""
        raise NotImplementedError( f'get_downstream_ids not implemented for this {self.__class__.__name__}' )

    def get_downstreams(self, pgdb=None, session=None):
        """Get all data products that were created directly from this object (non-recursive)."""
        downstreams = []
        with PGDB( pgdb if pgdb is not None else session, dictcursor=True ) as pgdb:
            downstream_info = self.get_downstream_ids( pgdb )
            for cls, dwnid in downstream_info:
                q = sql.SQL( "SELECT * FROM {tab} WHERE _id={objid}" ).format( tab=sql.Identifier(cls.__tablename__),
                                                                               objid=dwnid )
                rows = pgdb.execute( q )
                if len(rows) != 1:
                    raise RuntimeError( "This should never happen." )
                downstreams.append( cls.create( pgdb=pgdb, **(rows[0]) ) )
        return downstreams

    def delete_everything_in_provtag( self, tag, models=[], remove_folders=True,
                                      remove_downstreams=True, archive=True ):
        raise NotImplementedError( "In progress" )
        with PGDB( dictcursor=True ) as pgdb:
            # Find all the provenances associated with this provence tag
            rows = pgdb.execute( sql.SQL( "SELECT provenance_id FROM provenance_tags WHERE tag={tag}" )
                                 .format( tag=tag ) )
            chopping_block = set( r['provenance_id'] for r in rows )

            # Remove any provenances that are in another provenance tag
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT provenacne_id, tag FROM provenance_tags
                WHERE tag!={tag}
                AND provenance_id=ANY(ARRAY[{provids}])
                """
            ) ).format( tag=tag, provids=sql.SQL(",").join( chopping_block ) )
            rows = pgdb.execute( q )
            for row in rows:
                SCLogger.warning( f"Not deleting things from provenance {row['provenance_id']} because "
                                  f"it also exists in tag {row['tag']}" )
                chopping_block.remove( row['provenance_id'] )

            # OMG delete
            for model in models:
                rows = pgdb.execute( sql.SQL( "SELECT * FROM {tab} WHERE provenance_id=ANY(ARRAY[{provids}])" )
                                     .format( tab=sql.Identifier(model.__tablenme__),
                                              provids=sql.SQL(",").join(chopping_block) ) )
                objs = [ model(**row) for row in rows ]
                SCLogger.warning( f"Deleteing {len(objs)} rows from {model.__tablename__}, plus associated "
                                  f"data, plus (probably) all downstreams." )
                for i, obj in enumerate(objs):
                    if i % 100 == 0:
                        SCLogger.debug( f"...deleted {i} of {len(objs)}..." )
                    obj.delete_from_disk_and_database( remove_folders=remove_folders,
                                                       remove_downstreams=remove_downstreams,
                                                       archive=archive )
                SCLogger.debug( f"...done deleting {len(objs)} rows from {model.__tablename__}." )



    def delete_from_disk_and_database( self, remove_folders=True, remove_downstreams=True, archive=True, pgdb=None ):
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
                                                         remove_downstreams=True, pgdb=pgdb )

        # Remove files from archive

        if archive and hasattr( self, "filepath" ):
            if self.filepath is not None:
                if self.components is None:
                    self.archive.delete( self.filepath, okifmissing=True )
                else:
                    for comp in self.components:
                        self.archive.delete( f"{self.filepath}.{comp}{self._file_suffix(comp)}", okifmissing=True )

            # make sure these are set to null just in case we fail
            # to commit later on, we will at least know something is wrong
            self.md5sum = None
            self.md5sum_components = None

        # Remove data from disk

        if hasattr( self, "remove_data_from_disk" ):
            self.remove_data_from_disk( remove_folders=remove_folders )
            # make sure these are set to null just in case we fail
            # to commit later on, we will at least know something is wrong
            self.components = None
            self.filepath = None

        # Finally, after everything is cleaned up, remove the database record

        self._delete_from_database( pgdb=pgdb )


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
            if isinstance( value, datetime.datetime ):
                value = value.isoformat()

            if key == 'md5sum' and value is not None:
                if isinstance(value, UUID):
                    value = value.hex
            if key == 'md5sum_components' and value is not None:
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
                raise TypeError( f"Column {key} has type {type(value)} which I don't know how to parse." )

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

        md5sum_components = dictionary.get('md5sum_components', None)
        if md5sum_components is not None:
            new_components = [UUID(md5) for md5 in md5sum_components if md5 is not None]
            dictionary['md5sum_components'] = new_components

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

        return cls.create(**dictionary)

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
                json.dump(self.to_dict(), fp, indent=2, cls=NumpyAndUUIDJsonEncoder)
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


def table_class_map():
    # Imports here.  All these things import base.py, so we can't do them at the
    #   top of the file or we'd have conflits.
    # TODO: think if there's a reflection way we can get these imports
    #   without having to remember to add a file here every time we
    #   add something to models.
    import models.background        # noqa: F401
    import models.calibratorfile    # noqa: F401
    import models.catalog_excerpt   # noqa: F401
    import models.cutouts           # noqa: F401
    import models.datafile          # noqa: F401
    import models.deepscore         # noqa: F401
    import models.exposure          # noqa: F401
    import models.fakeset           # noqa: F401
    import models.image             # noqa: F401
    import models.knownexposure     # noqa: F401
    import models.measurements      # noqa: F401
    import models.object            # noqa: F401
    import models.provenance        # noqa: F401
    import models.psf               # noqa: F401
    import models.reference         # noqa: F401
    import models.refset            # noqa: F401
    import models.report            # noqa: F401
    import models.source_list       # noqa: F401
    import models.user              # noqa: F401
    import models.world_coordinates # noqa: F401
    import models.zero_point        # noqa: F401

    tabclsmap = {}
    for cls in Base.__subclasses__():
        if hasattr( cls, '__tablename__' ):
            tabclsmap[cls.__tablename__] = cls

    return tabclsmap


ARCHIVE = None


def get_archive_object():
    """Return a global archive object. If it doesn't exist, create it based on the current config. """
    global ARCHIVE
    if ARCHIVE is None:
        cfg = config.Config.get()
        archive_specs = cfg.value('archive', None)
        if archive_specs is not None:
            archive_specs[ 'logger' ] = SCLogger
            archive_specs[ 'lockfunc' ] = ArchiveLock.lockfunc
            if ( 'token' not in archive_specs ) or  ( archive_specs[ 'token' ] is None ):
                if ( 'token_file' not in archive_specs ) or ( archive_specs[ 'token_file' ] is None ):
                    raise RuntimeError( "Archive specs don't include a token or token_file" )
                with open( archive_specs[ 'token_file' ] ) as ifp:
                    archive_specs[ 'token' ] = ifp.readline().strip()
                del archive_specs[ 'token_file' ]
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
    components and md5sum_components will be null.  (Exception: if you
    are configured with a null archive (config parameter archive in the
    yaml config file is null), then md5sum will be set when the image is
    saved to disk, instead of when it's saved to the archive.)

    If there are multiple files associated with this entry, then
    filepath is the beginning of the names of all the files.  The full
    filenames are constrcuted by appending ".", the component name
    (stored in self.components), and the file suffix (returned by
    self._file_suffix(comp)) to filepath.  For example, if an image file
    has the image itself, an associated weight, and an associated mask,
    then filepath might be "image" and components might be ["image",
    "mask", "weight"] to indicate that the three files image.image.fits,
    image.mask.fits, and image.weight.fits are all associated with this
    entry.  When components is non-null, md5sum should be null, and
    md5sum_components is an array with the same length as components.
    (For extension files that have not yet been saved to the archive,
    that element of the md5sum_components array is null.)

    Saving data:

    Any object that implements this mixin must call this class' "save"
    method in order to save the data to disk.  (This may be through
    super() if the subclass has to do custom things.)  The save method
    of this class will save to the local filestore (underneath
    path.data_root), and also save it to the archive.  Once a file is
    saved on the archive, the md5sum (or md5sum_copmonents) field in the
    database record is updated.  (If the file has not been saved to the
    archive, then the md5sum and md5sum_components fields will be null.)

    Loading data:

    When calling get_fullpath(), the object will first check if the file
    exists locally, and then it will import it from archive if missing
    (and if archive is defined).  If you want to avoid downloading, use
    get_fullpath(download=False) or get_fullpath(nofile=True).  (The
    latter case won't even try to find the file on the local disk, it
    will just tell you what the path should be.)  If you want to always
    get a list of filepaths (even if components=None) use
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
    #                     '(md5sum_components IS NULL OR array_position(md5sum_components, NULL) IS NOT NULL))',
    #             name=f'{cls.__tablename__}_md5sum_check'
    #         ),
    #     )

    # Subclasses of this class must include the following in __table_args__:
    #   CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
    #                    '(md5sum_components IS NULL OR array_position(md5sum_components, NULL) IS NOT NULL))',
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
    def filepath(cls):  # noqa: N805
        return sa.Column(
            sa.Text,
            nullable=False,
            index=True,
            unique=True,
            doc="Base path (relative to the data root) for a stored file"
        )

    components = sa.Column(
        ARRAY(sa.Text, zero_indexes=True),
        nullable=True,
        doc=( "If non-null, an array of strings identifying components that are saved to "
              "separate files on disk." )
    )

    md5sum = sa.Column(
        sqlUUID(as_uuid=True),
        nullable=True,
        server_default=None,
        doc="md5sum of the file, provided by the archive server"
    )

    md5sum_components = sa.Column(
        ARRAY(sqlUUID(as_uuid=True), zero_indexes=True),
        nullable=True,
        server_default=None,
        doc="md5sum of components files; must have same number of elements as components"
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
        """The default value for the nofile property of new objects.

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
        """Make sure the filepath is legitimate.

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

    def _file_suffix( self, comp=None ):
        """Returns the suffix on saved files.  Used by get_relpath and get_fullpath.

        Subclasses should override this if they are going to use
        multiple components.  See Image._file_suffix for an example.

        Parameters
        ----------
          comp: str or None
            The component whose suffix we want.

        """

        return ""

    def get_relpath( self, as_list=False ):
        """Get path of the file, or list of paths of files, relative to the local data storage root.

        Does not do any downloading or verification; for that, call get_fullpath.

        Parameters
        ----------
          as_list: bool, default False
             Return a (single-element) list even if there is only a
             single file (i.e. there are no components).

        Returns
        -------
          str or list of str
             Will return the filepath if there are no components, or a
             list of filepaths if there are copmonents.

        """

        if self.components is None:
            return [ self.filepath ] if as_list else self.filepath
        return [ f'{self.filepath}.{comp}{self._file_suffix(comp)}' for comp in self.components ]


    def get_fullpath( self, download=True, as_list=False, components=None,
                      nofile=None, always_verify_md5=False ):
        """Get the full path of the file, or list of full paths of files if components is not None.

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

        When the components is None, will return a single string.
        When the components is an array, will return a list of strings.
        If as_list=False, will always return a list of strings,
        even if components is None.

        Parameters
        ----------
        download: bool
            Whether to download the file from server if missing.
            Must have archive defined. Default is True.

        as_list: bool
            Whether to return a list of filepaths, even if self.components=None.
            Default is False.

        components: list, str, or None
            Which components to get.  Must be None if self.components is
            None None.  If not given, defaults to self.components.

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
            Absolute path to the file(s) on local disk.

        """

        components = self.components if components is None else listify( components )

        if components is None:
            if as_list:
                return [self._get_fullpath_single(download=download, nofile=nofile,
                                                  always_verify_md5=always_verify_md5)]
            else:
                return self._get_fullpath_single(download=download, nofile=nofile,
                                                 always_verify_md5=always_verify_md5)
        else:
            if self.components is None:
                raise ValueError( "Can't give components for an object that doesn't have components." )
            unknown = set(components) - set(self.components)
            if unknown:
                raise ValueError( f"Unknown components: {unknown}" )

            return [
                self._get_fullpath_single(download=download, comp=comp, nofile=nofile,
                                          always_verify_md5=always_verify_md5)
                for comp in components
            ]


    def _get_fullpath_single(self, download=True, comp=None, nofile=None, always_verify_md5=False):
        """Get the full path of a single file.

        Will follow the same logic as get_fullpath(),
        of checking and downloading the file from the server
        if it is not on local disk.

        Parameters
        ----------
        download: bool
            Whether to download the file from server if missing.
            Must have archive defined. Default is True.

        comp: str or None
            The component file whose path we want, or None if
            the object is stored in a single file.

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
        if comp is None:
            md5sum = self.md5sum.hex if self.md5sum is not None else None
        else:
            try:
                compdex = self.components.index( comp )
            except ValueError:
                raise ValueError(f"Unknown component {comp} for {fname}" )
            if (self.md5sum_components is None ) or ( compdex >= len(self.md5sum_components) ):
                md5sum = None
            else:
                md5sum = self.md5sum_components[compdex]
                md5sum = None if md5sum is None else md5sum.hex
            fname += f'.{comp}{self._file_suffix(comp)}'

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


    def save(self, data, component=None, overwrite=True, exists_ok=True, verify_md5=True, no_archive=False):
        """Save a file to disk, and to the archive.

        Does not write anything to the database.  (At least, it's not supposed to....)

        Parameters
        ---------
        data: bytes, string, or Path
          The data to be saved

        component: string or None
          The file component.  SHould be None if this object is saved to a single file.

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
               md5sum_components) is null, then always upload to the
               archive as long as no_archive is False). Otherwise,
               verify_md5 modifies the behavior;
               verify_md5 = True
                 If self.md5sum (or the appropriate entry in
                 md5sum_components) matches the md5sum of the passed data,
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

        If component is not None, and it isn't already in the list of
        components, it will be added.

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

        # First : figure out if this is a component or not,
        #   and make sure that's consistent with the object.
        # If it is:
        #   Find the index into the components array for
        #   this component, or append to the array if
        #   it's a new component that doesn't already exist.
        #   Set the variables curcomponents and compmd5s to lists with
        #   components and md5sums of component files,
        #   initially copied from self.components and
        #   self.md5sum_components, and modified if necessary
        #   with the saved file.  compdex holds the index
        #   into both of these arrays for the current components.
        # else:
        #   Set curcomponents, compmd5s, and compdex to None

        # We will either replace these two variables with empty lists,
        #  or make a copy (using list()).  The reason for this: we don't
        #  want to directly modify the lists in self until the saving is
        #  done.  That way, self doesn't get mucked up if this function
        #  exceptions out.
        curcomponents = self.components
        compmd5s = self.md5sum_components

        compdex = None
        if component is None:
            if curcomponents is not None:
                raise RuntimeError( "Tried to save a non-component file, but this file has components." )
            if compmd5s is not None:
                raise RuntimeError( "Data integrity error; components is null, "
                                    "but md5sum_components isn't." )
        else:
            if curcomponents is None:
                if compmd5s is not None:
                    raise RuntimeError( "Data integrity error; components is null, "
                                        "but md5sum_components isn't." )
                curcomponents = []
                compmd5s = []
            else:
                if compmd5s is None:
                    raise RuntimeError( "Data integrity error; components is not null, "
                                        "but md5sum_components is" )
                curcomponents = list( curcomponents )
                compmd5s = list( compmd5s )
            if len(curcomponents) != len(compmd5s):
                raise RuntimeError( f"Data integrity error; len(md5sum_components)={len(compmd5s)}, "
                                    f"but len(components)={len(curcomponents)}" )
            try:
                compdex = curcomponents.index( component )
            except ValueError:
                curcomponents.append( component )
                compmd5s.append( None )
                compdex = len(curcomponents) - 1

        # relpath holds the path of the file relative to the data store root
        # origmd5 holds the md5sum (hashlib.hash object) of the original file,
        #   *unless* the original file is already the right file in the local file store
        #   (in which case it's None)
        # localpath holds the absolute path of where the file should be written in the local file store
        relpath = pathlib.Path( self.filepath if component is None else
                                self.filepath + '.' + component + self._file_suffix(component) )
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
                # SCLogger.debug( f"FileOnDiskMixin.save: local file store path & original path are the same: {path}" )
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
            if curcomponents is not None:
                compmd5s[ compdex ] = UUID( origmd5.hexdigest() )
                self.components = curcomponents
                self.md5sum_components = compmd5s
            else:
                self.md5sum = UUID( origmd5.hexdigest() )
            return

        # This is the case where there *is* an archive, but the no_archive option was passed
        if no_archive:
            if curcomponents is not None:
                self.components = curcomponents
                self.md5sum_components = compmd5s
            return

        # The rest of this deals with the archive

        archivemd5 = self.md5sum if component is None else compmd5s[compdex]
        logfilepath = ( self.filepath if component is None
                        else f'{self.filepath}.{component}{self._file_suffix(component)}' )

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
            if curcomponents is not None:
                compmd5s[compdex] = remmd5
                self.md5sum = None
                self.components = curcomponents
                self.md5sum_components = compmd5s
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
    # ...and that was really annoying, because as I wrote more code that didn't
    #   use SQLAlchemy, having SQLAlchmey handle the default was troublesome.
    #   However, I'm afraid of removing the SQLAlchmey default, because it will
    #   probably break lots of code in lots of places, so just try putting in
    #   both here.  Cf. Issue #516.
    _id = sa.Column(
        sqlUUID,
        primary_key=True,
        index=True,
        default=uuid.uuid4,            # This is the one exception to always using server_default
        server_default=func.gen_random_uuid(),
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
    def get_by_id( cls, uuid, session=None, pgdb=None, **kwargs ):
        """Get an object of the current class that matches the given uuid.

        Returns None if not found.

        Parameters
        ----------
          uuid : UUID
            The id of the object you want

          session, pgdb: PGDB, psycopg.Connection, psycopg.Cursor, or sqlalchmey session
            Will use pgdb if it's not None, else session.  If both are None,
            makes and closes a new connection to the database.

          **kwargs: further keywords are passed on to the object constructor

        Returns
        -------
          object of type cls

        """

        with PGDB( (pgdb if pgdb is not None else session), dictcursor=True ) as pgdb:
            q = sql.SQL( "SELECT * FROM {table} WHERE _id=%(id)s" ).format( table=sql.Identifier(cls.__tablename__) )
            rows = pgdb.execute( q, { 'id': uuid } )
            if len(rows) == 0:
                return None
            elif len(rows) > 1:
                raise RuntimeError( "This should never happen." )
            else:
                kwargs = kwargs.copy()
                kwargs.update( rows[0] )
                obj = cls.create( pgdb=pgdb, **kwargs )
                obj.from_db = True
                return obj


    @classmethod
    def get_batch_by_ids( cls, uuids, session=None, pgdb=None, return_dict=False ):
        """Get objects whose ids are in the list uuids.

        Parameters
        ----------
          uuids: UUID or list of UUID
            The object IDs whose corresponding objects you want.

          session, pgdb: PGDB, psycopg.Connection, psycopg.Cursor, or sqlalchemy session
            Will use pgdb if it's not None, else session.  If both are None,
            makes and closes a new connection to the database.

          return_dict: bool, default False
            If False, just return a list of objects.  If True, return a
            dict of { id: object }.

        Returns
        -------
          either list of cls, or dict of { UUID: cls }

        """

        uuids = listify( uuids )
        with PGDB( (pgdb if pgdb is not None else session), dictcursor=True ) as pgdb:
            q = sql.SQL( "SELECT * FROM {tab} WHERE _id=ANY(ARRAY[{ids}])"
                        ).format( tab=sql.Identifier( cls.__tablename__ ),
                                  ids=sql.SQL(",").join(uuids) )
            rows = pgdb.execute( q )

            if return_dict:
                return { r['_id']: cls.create(pgdb=pgdb, **r) for r in rows }
            else:
                return [ cls.create(pgdb=pgdb, **r) for r in rows ]


    @classmethod
    def get_by_field_value( cls, field, values, pgdb=None ):
        """Get a list of objects of a class whose field have a certain value or are in a list of values.

        Parameters
        ----------
          field : str
            The name of the field as defined in the database (so, use _id, not id, etc.).

          values : *
            Either a sequence of values to match, or a single value to match.

          pgdb : PGDB, psycopg.Connection, psycopg.Cursor, sqlalchemy session, or None
            Use this database connection; if None, makes and closes a new one.

        """

        values = listify( values )
        with PGDB( pgdb, dictcursor=True ) as pgdb:
            rows = pgdb.execute( sql.SQL( "SELECT * FROM {tab} WHERE {field}=ANY(ARRAY[{vals}])" )
                                 .format( tab=sql.Identifier(cls.__tablename__),
                                          field=sql.Identifier(field),
                                          vals=sql.SQL(",").join(values) ) )
        return [ cls.create(pgdb=pgdb, **r) for r in rows ]



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


    @classmethod
    def at_ra_dec( cls, ra, dec, radius=1.0, session=None ):
        """Return all objects that are within radius arcsecnds of (ra,dec).

        Parmaeters
        ----------
          ra : float
            RA of cone search.

          dec : float
            Dec of cone search

          radius : float, default 1.0
            Radius in arcseconds of cone search

          session : PGDB, psycopg.Connection, psycopg.Cursor, or Session

        Returns
        -------
          list of Object

        """

        with PGDB( session, dictcursor=True ) as pgdb:
            q = sql.SQL( "SELECT * FROM {tab} WHERE q3c_radial_query( ra, dec, {ra}, {dec}, {rad} )" )
            q = q.format( tab=sql.Identifier(cls.__tablename__), ra=ra, dec=dec, rad=radius/3600. )
            rows = pgdb.execute( q )

        return [ cls.create(pgdb=pgdb, **row) for row in rows ]


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
        """An SQLalchemy clause to find all objects of this class that are within a cone.

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

    ra_corner_00 = sa.Column( sa.REAL, nullable=False, index=False,
                              doc="RA of the low-RA, low-Dec corner (degrees)" )
    ra_corner_01 = sa.Column( sa.REAL, nullable=False, index=False,
                              doc="RA of the low-RA, high-Dec corner (degrees)" )
    ra_corner_10 = sa.Column( sa.REAL, nullable=False, index=False,
                              doc="RA of the high-RA, low-Dec corner (degrees)" )
    ra_corner_11 = sa.Column( sa.REAL, nullable=False, index=False,
                              doc="RA of the high-RA, high-Dec corner (degrees)" )
    dec_corner_00 = sa.Column( sa.REAL, nullable=False, index=False,
                               doc="Dec of the low-RA, low-Dec corner (degrees)" )
    dec_corner_01 = sa.Column( sa.REAL, nullable=False, index=False,
                               doc="Dec of the low-RA, high-Dec corner (degrees)" )
    dec_corner_10 = sa.Column( sa.REAL, nullable=False, index=False,
                               doc="Dec of the high-RA, low-Dec corner (degrees)" )
    dec_corner_11 = sa.Column( sa.REAL, nullable=False, index=False,
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
    def _fromclause( cls, fromclause=None ):
        return ( sql.SQL(fromclause) if fromclause is not None
                 else ( sql.SQL( "FROM {tab} i" )
                        .format( tab=sql.Identifier(cls.__tablename__) ) )
                )

    @classmethod
    def _provclause( cls, prov_id, provtable=None ):
        provtable = cls.__tablename__ if provtable is None else provtable

        if isinstance( prov_id, str ):
            provclause = sql.SQL( "AND {provtable}.provenance_id={prov_id}"
                                  ).format( provtable=sql.Identifier(provtable),
                                            prov_id=prov_id )
        elif isinstance( prov_id, list ):
            provclause = sql.SQL( "AND {provtable}.provenance_id=ANY(ARRAY[{provs}])"
                                 ).format( provtable=sql.Identifier(provtable),
                                           provs=sql.SQL(",").join( sql.SQL("{p}").format(p=p) for p in prov_id ) )
        elif prov_id is not None:
            raise TypeError( "prov_id must be string, list, or None" )
        else:
            provclause = sql.SQL("")

        return provclause


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
          racorners, deccorners, minra, maxra, mindec, maxdec

            racorners and deccorners are lists, sorted so that they're in the order:
              (lowRA,lowDec), (lowRA,highDec), (highRA,lowDec), (highRA,highDec)

            min/maxra is the min/max of all the RAs, trying to properly deal with ra spanning 0
            min/maxdec is the min/max of all the decs

        """

        if len(ras) != 4:
            raise ValueError(f'ras must be a list/array with exactly four elements. Got {ras}')
            raise ValueError(f'decs must be a list/array with exactly four elements. Got {decs}')
        if any( ( r < 0. ) or ( r >= 360. ) for r in ras ):
            raise ValueError( f"ras must be in the range [0,360); got {ras}" )
        if any( ( d < -90. ) or ( d > 90. ) for d in decs ):
            raise ValueError( f"decs must be in the range [-90, 90]; got {decs}" )

        raorder = list( range(4) )
        raorder.sort( key=lambda i: ras[i] )

        # Try to detect an RA that spans 0.  Assume that no FourCorners is ever going to span more than 180° in RA
        if ras[raorder[3]] - ras[raorder[0]] > 180.:
            # Deal with this by just subtracting 360 from RAS between 180 and 360 and then fixing it later
            ras = [ r - 360. if r > 180. else r for r in ras ]
            raorder.sort( key=lambda i: ras[i] )

        # Of two lowest ras, of those, pick the one with the lower dec;
        #   that's lowRA,lowDec; the other one is lowRA, highDec

        dex00 = raorder[0] if decs[raorder[0]] < decs[raorder[1]] else raorder[1]
        dex01 = raorder[1] if decs[raorder[0]] < decs[raorder[1]] else raorder[0]

        # Same thing, only now high ra

        dex10 = raorder[2] if decs[raorder[2]] < decs[raorder[3]] else raorder[3]
        dex11 = raorder[3] if decs[raorder[2]] < decs[raorder[3]] else raorder[2]

        # Min/max

        minra = min( ras )
        maxra = max( ras )
        mindec = min( decs )
        maxdec = max( decs )

        # Fix the ra-crossing-0 detection stuff we did above

        minra = minra + 360. if minra < 0. else minra
        maxra = maxra + 360. if maxra < 0. else maxra
        ras = [ r + 360. if r < 0. else r for r in ras ]

        return ( [  ras[dex00],  ras[dex01],  ras[dex10],  ras[dex11] ],
                 [ decs[dex00], decs[dex01], decs[dex10], decs[dex11] ],
                 minra, maxra, mindec, maxdec )


    def set_corners_minmax( self, ras, decs ):
        ras, decs, minra, maxra, mindec, maxdec = FourCorners.sort_radec( ras, decs )
        self.ra_corner_00 = ras[0]
        self.ra_corner_01 = ras[1]
        self.ra_corner_10 = ras[2]
        self.ra_corner_11 = ras[3]
        self.dec_corner_00 = decs[0]
        self.dec_corner_01 = decs[1]
        self.dec_corner_10 = decs[2]
        self.dec_corner_11 = decs[3]
        self.minra = minra
        self.maxra = maxra
        self.mindec = mindec
        self.maxdec = maxdec


    @classmethod
    def find_containing_siobj( cls, siobj, **kwargs ):
        """Return all images (or whatever) that contain the given SpatiallyIndexed thing

        Parameters
        ----------
          siobj: SpatiallyIndexed
            A single object that is spatially indexed

          **kwargs : further arguments passed to find_containing()

        Returns
        -------
           An sql query result thingy.

        """

        # Overabundance of caution to avoid Bobby Tables.
        # (Because python is not strongly typed, siobj.ra and
        # siobj.dec could be set to anything.)
        ra = float( siobj.ra )
        dec = float( siobj.dec )
        return cls.find_containing( ra, dec, **kwargs )

    @classmethod
    def _find_possibly_containing_temptable( cls, ra, dec, session, prov_id=None,
                                             fromclause=None, provtable='i',
                                             corner="corner", limprefix="",
                                             temptable="temp_find_containing" ):
        """Internal.

        Looks for all cls objects where ra, dec is between minra:maxra,
        mindec:maxdec.  This will be a superset of the images that
        contain ra, dec.

        Lots of special case code for images that cross RA 0.

        Loads up the temp table specified by argument temptable.

        Parameters
        ----------
          ra, dec : float
             Coordinates to search for; decimal degrees.

          session : sa.orm.session.Session or PGDB or psycopg.Connection or psycopg.Cursor
             Required here, otherwise the temp table would be useless.

          prov_id : str, list of str, or None
             If not None, search for objects with this provenance, or
             any of these provenances if a list.

          fromclause : str, default None
             Complicated.  Used in Image.find_images.  WARNING.  Misuse
             of this can totally Bobby Tables the database.  Be good.

          corner : str, default "corner"
             ...used so that subclasses don't have to reimplement this
             method.  If you misuse this, you can totally Bobby Tables
             the databse, but if you're calling this function, you have
             access anyway, so you may as well just "DROP TABLE..."
             directly.  But, don't expose this to anything outside, and
             don't use it unless you really know what you're doing.

          limprefix: str, default ""
             ...used so that subclasses don't have to reimplement this
             method.  See warnings on corner re: Bobby Tables.

          provtable : str, default 'i'
             Complicated.  Used in Image.find_images.  WARNING.  Misuse
             of this can totally Bobby Tables the database.  Be good.

          temptable : str, default "temp_find_containing"
             Name of the temptable to write to.

        """
        if not isinstance( session, ( sa.orm.session.Session, psycopg.Connection, psycopg.Cursor, PGDB ) ):
            raise TypeError( f"session must be a sa.orm.session.Session, psycopg.Connection, psycopg.Cursor, or PGDB, "
                             f"not a {type(session)}" )

        # Shouldn't need this, but just in case somebody gave us a wrapped RA:
        while ( ra < 0 ): ra += 360.
        while ( ra >= 360.): ra -= 360.

        fromclause = cls._fromclause( fromclause )
        provclause = cls._provclause( prov_id, provtable )

        q = sql.SQL( textwrap.dedent(
            """\
            SELECT i._id,
                   i.ra_{corner}_00 AS ra_corner_00,
                   i.ra_{corner}_01 AS ra_corner_01,
                   i.ra_{corner}_10 AS ra_corner_10,
                   i.ra_{corner}_11 AS ra_corner_11,
                   i.dec_{corner}_00 AS dec_corner_00,
                   i.dec_{corner}_01 AS dec_corner_01,
                   i.dec_{corner}_10 AS dec_corner_10,
                   i.dec_{corner}_11 AS dec_corner_11
            INTO TEMP TABLE {temptable}
            {fromclause}
            WHERE (
              ( i.{limprefix}maxdec >= {dec} AND i.{limprefix}mindec <= {dec} )
              AND (
                ( (i.{limprefix}maxra > i.{limprefix}minra ) AND
                  ( i.{limprefix}maxra >= {ra} AND i.{limprefix}minra <= {ra} ) )
                OR
                ( ( i.{limprefix}maxra < i.{limprefix}minra ) AND
                  ( ( i.{limprefix}maxra >= {ra} OR {ra} > 180. ) AND ( i.{limprefix}minra <= {ra} OR {ra} <= 180. ) ) )
              )
              {provclause}
            )
            """
        ) ).format( ra=ra, dec=dec, fromclause=fromclause, provclause=provclause,
                    corner=sql.SQL(corner), temptable=sql.Identifier(temptable), limprefix=sql.SQL(limprefix) )

        with PGDB( session ) as pgdb:
            pgdb.execute_nofetch( sql.SQL( "DROP TABLE IF EXISTS {temptable}" )
                                  .format( temptable=sql.Identifier(temptable) ) )
            pgdb.execute_nofetch( q )


    @classmethod
    def find_containing( cls, ra, dec, corner="corner", limprefix="", prov_id=None, session=None,
                         temptable="temp_find_containing" ):
        """Return all objects in this class that contain the given RA and Dec

        Parameters
        ----------
          ra, dec: float, decimal degrees

          corner: str, default "corner"
             ...used so that subclasses don't have to reimplement this
             method.  If you misuse this, you can totally Bobby Tables
             the databse, but if you're calling this function, you have
             access anyway, so you may as well just "DROP TABLE..."
             directly.  But, don't expose this to anything outside, and
             don't use it unless you really know what you're doing.

          limperfix: str, default ""
             ...used by subclasses.  See warnings on corner.

          prov_id : str, list of str, or None
             If not None, search for objects with this provenance, or any of these provenances if a list.

          session: sa.orm.session.Session, PGDB, psycopg.Connection, psycopg.Cursor, or None

          temptable: str, default "temp_find_containing"
             Name of an internally used temporary table.

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

        with PGDB( session, dictcursor=True ) as pgdb:
            cls._find_possibly_containing_temptable( ra, dec, pgdb, prov_id=prov_id,
                                                     corner=corner, limprefix=limprefix,
                                                     temptable=temptable )

            q = sql.SQL( textwrap.dedent(
                """
                SELECT i.* FROM {tab} i
                INNER JOIN {temptable} t ON t._id=i._id
                WHERE q3c_poly_query( {ra}, {dec}, ARRAY[ t.ra_corner_00, t.dec_corner_00,
                                                          t.ra_corner_01, t.dec_corner_01,
                                                          t.ra_corner_11, t.dec_corner_11,
                                                          t.ra_corner_10, t.dec_corner_10 ])
                """
            ) ).format( tab=sql.Identifier(cls.__tablename__),
                        temptable=sql.Identifier(temptable),
                        ra=ra, dec=dec )

            rows = pgdb.execute( q )
            objs = [ cls.create(pgdb=pgdb, **r) for r in rows ]
            pgdb.execute_nofetch( sql.SQL( "DROP TABLE {temptable}" ).format( temptable=sql.Identifier(temptable) ) )
            return objs


    @classmethod
    def _find_potential_overlapping_temptable( cls, fcobj, session, prov_id=None,
                                               fromclause=None, provtable='i',
                                               corner="corner", limprefix="",
                                               temptable="temp_find_overlapping" ):
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

          session : sa.orm.session.Session or PGDB or psycopg.Connection or psycopg.Cursor
             Required here, otherwise the temp table would be useless.

          prov_id: str, list of str, or None
             id or ids of the provenance of cls objects to search; if
             None, won't filter on provenance

          fromclause : str, default None
             Complicated.  Used in Image.find_images.  WARNING.  Misuse
             of this can totally Bobby Tables the database.  Be good.

          provtable : str, default 'i'
             Complicated.  Used in Image.find_images.  WARNING.  Misuse
             of this can totally Bobby Tables the database.  Be good.

          corner, limprefix : See _find_possibly_containing_temptable ; same thing

          temptable : str, default "temp_find_overlapping"
             Name of temp table to create.

        """

        fromclause = cls._fromclause( fromclause )
        provclause = cls._provclause( prov_id, provtable )

        if not isinstance( session, ( sa.orm.session.Session, psycopg.Connection, psycopg.Cursor, PGDB ) ):
            raise TypeError( f"session must be a sa.orm.session.Session, psycopg.Connection, psycopg.Cursor, or PGDB, "
                             f"not a {type(session)}" )

        # All kinds of special cases (everything from the first OR
        # onwards) below to deal with the the case where RA crosses 0
        # TODO : speed tests once we have a big enough database for that
        # to matter to see how much this hurts us.

        q = sql.SQL( textwrap.dedent(
            """
            SELECT i._id,
                   i.ra_{corner}_00 AS ra_corner_00,
                   i.ra_{corner}_01 AS ra_corner_01,
                   i.ra_{corner}_10 AS ra_corner_10,
                   i.ra_{corner}_11 AS ra_corner_11,
                   i.dec_{corner}_00 AS dec_corner_00,
                   i.dec_{corner}_01 AS dec_corner_01,
                   i.dec_{corner}_10 AS dec_corner_10,
                   i.dec_{corner}_11 AS dec_corner_11
            INTO TEMP TABLE {temptable}
            {fromclause}
            WHERE (
              ( i.{limprefix}maxdec >= {mindec} AND i.{limprefix}mindec <= {maxdec} )
              AND
              ( ( ( i.{limprefix}maxra >= i.{limprefix}minra AND {maxra} >= {minra} ) AND
                  i.{limprefix}maxra >= {minra} AND i.{limprefix}minra <= {maxra} )
                OR
                ( i.{limprefix}maxra < i.{limprefix}minra AND {maxra} < {minra} )
                OR
                ( ( i.{limprefix}maxra < i.{limprefix}minra AND {maxra} >= {minra} AND {minra} <= 180. ) AND
                  i.{limprefix}maxra >= {minra} )
                OR
                ( ( i.{limprefix}maxra < i.{limprefix}minra AND {maxra} >= {minra} AND {minra} > 180. ) AND
                  i.{limprefix}minra <= {maxra} )
                OR
                ( ( i.{limprefix}maxra >= i.{limprefix}minra AND {maxra} < {minra} AND i.{limprefix}maxra <= 180. ) AND
                  i.{limprefix}minra <= {maxra} )
                OR
                ( ( i.{limprefix}maxra >= i.{limprefix}minra AND {maxra} < {minra} AND i.{limprefix}maxra > 180. ) AND
                 i.{limprefix}maxra >= {minra} )
              )
              {provclause}
            )
            """
        ) ).format( mindec=fcobj.mindec, maxdec=fcobj.maxdec, minra=fcobj.minra, maxra=fcobj.maxra,
                    temptable=sql.Identifier(temptable), provclause=provclause, fromclause=fromclause,
                    corner=sql.SQL(corner), limprefix=sql.SQL(limprefix) )

        with PGDB( session ) as pgdb:
            pgdb.execute_nofetch( sql.SQL( "DROP TABLE IF EXISTS {temptable}" )
                                  .format( temptable=sql.Identifier(temptable) ) )
            pgdb.execute_nofetch( q )


    @classmethod
    def find_potential_overlapping( cls, fcobj, prov_id=None, session=None,
                                    corner="corner", limprefix="", temptable="temp_find_overlapping" ):
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

          corner, limprefix: str
             Don't use this.  Used internally by subclasses. Passed on
             to _find_potential_overlapping_temptable.

          session: sa.orm.session.Session, PGDB, psycopg,Connection, or psycopg.Cursor

        Returns
        -------
          The result of a sess.scalars(...).all() with members of this class.

        """
        with PGDB( session, dictcursor=True ) as pgdb:
            cls._find_potential_overlapping_temptable( fcobj, pgdb, prov_id=prov_id,
                                                       corner=corner, limprefix=limprefix )
            rows = pgdb.execute( sql.SQL( "SELECT i.* FROM {tab} i INNER JOIN {temptable} t ON i._id=t._id" )
                                 .format( tab=sql.Identifier(cls.__tablename__),
                                          temptable=sql.Identifier(temptable) ) )
            objs = [ cls.create(pgdb=pgdb, **r) for r in rows ]
            pgdb.execute_nofetch( sql.SQL( "DROP TABLE {temptable}" ).format( temptable=sql.Identifier(temptable) ) )
            return objs


    @classmethod
    def get_overlap_frac(cls, obj1, obj2, corner="corner", limprefix="" ):
        """Calculate the overlap fraction between two objects that have four corners.

        Returns
        -------
        overlap_frac: float
            The fraction of obj1's area that is covered by the intersection of the objects

        corner: str, default "corner"
            Used by subclasses

        limprefix: str, default ""
            Used by subclasses

        Assumes that the images are small enough that a simple cos(dec)
        correction for RA is enough that we can assume that the sky is
        flat.  This assumption will break down near the poles.

        """

        o1ra = np.array( [ [ getattr( obj1,  f"ra_{corner}_00" ), getattr( obj1,  f"ra_{corner}_01" ) ],
                           [ getattr( obj1,  f"ra_{corner}_10" ), getattr( obj1,  f"ra_{corner}_11" ) ] ])
        o2ra = np.array( [ [ getattr( obj2,  f"ra_{corner}_00" ), getattr( obj2,  f"ra_{corner}_01" ) ],
                           [ getattr( obj2,  f"ra_{corner}_10" ), getattr( obj2,  f"ra_{corner}_11" ) ] ] )
        o1dec = np.array( [ [ getattr( obj1,  f"dec_{corner}_00" ), getattr( obj1,  f"dec_{corner}_01" ) ],
                            [ getattr( obj1,  f"dec_{corner}_10" ), getattr( obj1,  f"dec_{corner}_11" ) ] ] )
        o2dec = np.array( [ [ getattr( obj2,  f"dec_{corner}_00" ), getattr( obj2,  f"dec_{corner}_01" ) ],
                            [ getattr( obj2,  f"dec_{corner}_10" ), getattr( obj2,  f"dec_{corner}_11" ) ] ] )

        # Have to handle the case of ra spanning 0.  This happens when
        # maxra < minra.  In that case, take all ras > 180 and subtract
        # 360 to make them negative.  Subsequent computations will then
        # work.  This will break horribly if the size of the image
        # approaches 180°, but that's an absurd case that should never
        # happen.  (If you're using this pipeline with some sort of
        # fisheye all-sky camera, then... well, sorry.  All kinds of things
        # are probably going to break having to do with coordinates.)
        if ( ( getattr( obj1, f"{limprefix}maxra" ) < getattr( obj1, f"{limprefix}minra" ) ) or
             ( getattr( obj2, f"{limprefix}maxra" ) < getattr( obj2, f"{limprefix}minra" ) )
            ):
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


    def contains( self, ra, dec, corner="corner", limprefix="" ):
        """Return True if ra, dec is contained within the four corners."""

        corners = np.array( [ [ getattr( self, f"ra_{corner}_00" ), getattr( self, f"dec_{corner}_00" ) ],
                              [ getattr( self, f"ra_{corner}_01" ), getattr( self, f"dec_{corner}_01" ) ],
                              [ getattr( self, f"ra_{corner}_11" ), getattr( self, f"dec_{corner}_11" ) ],
                              [ getattr( self, f"ra_{corner}_10" ), getattr( self, f"dec_{corner}_10" ) ],
                              [ getattr( self, f"ra_{corner}_00" ), getattr( self, f"dec_{corner}_00" ) ] ] )
        if getattr( self, f"{limprefix}maxra" ) < getattr( self, f"{limprefix}minra" ):
            corners[ corners[:,0]>180, 0 ] -= 360.
            if ra > 180.:
                ra -= 360.

        obj = shapely.Polygon( corners )
        return obj.contains( shapely.Point( ra, dec ) )


    def set_corners_from_wcs( self, wcs=None, width=None, height=None, setradec=False ):
        """Update the object's four corners (and, optionally, RA/Dec) from a WCS.

        Subclasses may have alternate sets of arguments they can supply.

        Parameters
        ----------
        wcs : astropy.wcs.WCS or WorldCoordinates default None
           The WCS to use.  If nothing is here in this positional
           parameter, then the first two positional parameters are
           actually width and height, and this method will use self.wcs.
           This will only work if self.wcs exists and is the right kind
           of thing.

        width : int, default None
            Width (x-size) of image.  Either (width, height) or image is required.

        height : int, default None
            Height (y-size) of image.  Either (width, height) or image is required

        image : Image, default None

        setradec : bool, default False
           If True, also update the image's ra and dec fields, as well
           as the things calculated from it (galactic, ecliptic
           coordinates).

        """

        # avoid circular imports
        from models.world_coordinates import WorldCoordinates

        if any( x is None for x in ( wcs, width, height ) ):
            raise ValueError( "Must provide all of wcs, width, height" )

        if isinstance( wcs, WorldCoordinates ):
            wcs = wcs.wcs
        elif not isinstance( wcs, astropy.wcs.WCS ):
            raise TypeError( f"Error, wcs must be a WorldCoordinates or a WCS, not a {type(wcs)}" )

        ras = []
        decs = []
        xs = [ 0., width-1., 0., width-1. ]
        ys = [ 0., height-1., height-1., 0. ]
        scs = wcs.pixel_to_world( xs, ys )
        if isinstance( scs[0].ra, astropy.coordinates.Longitude ):
            ras = [ i.ra.to_value() for i in scs ]
            decs = [ i.dec.to_value() for i in scs ]
        else:
            ras = [ i.ra.value_in(u.deg).value for i in scs ]
            decs = [ i.dec.value_in(u.deg).value for i in scs ]
        self.set_corners_minmax( ras, decs )

        if setradec:
            sc = wcs.pixel_to_world( width / 2., height / 2. )
            self.ra = sc.ra.to(u.deg).value
            self.dec = sc.dec.to(u.deg).value
            self.gallat = sc.galactic.b.deg
            self.gallon = sc.galactic.l.deg
            self.ecllat = sc.barycentrictrueecliptic.lat.deg
            self.ecllon = sc.barycentrictrueecliptic.lon.deg



class FourCornersWithGood( FourCorners ):
    """FourCorners, plus another set of fields indicating what's actually good.

    This is for images where a substantial fraction of the image is
    masked out, e.g. an image where one of two chips is bad.

    """

    ra_good_00 = sa.Column( sa.REAL, nullable=False, index=False )
    ra_good_01 = sa.Column( sa.REAL, nullable=False, index=False )
    ra_good_10 = sa.Column( sa.REAL, nullable=False, index=False )
    ra_good_11 = sa.Column( sa.REAL, nullable=False, index=False )
    dec_good_00 = sa.Column( sa.REAL, nullable=False, index=False )
    dec_good_01 = sa.Column( sa.REAL, nullable=False, index=False )
    dec_good_10 = sa.Column( sa.REAL, nullable=False, index=False )
    dec_good_11 = sa.Column( sa.REAL, nullable=False, index=False )
    good_minra = sa.Column( sa.REAL, nullable=False, index=True )
    good_maxra = sa.Column( sa.REAL, nullable=False, index=True )
    good_mindec = sa.Column( sa.REAL, nullable=False, index=True )
    good_maxdec = sa.Column( sa.REAL, nullable=False, index=True )


    @classmethod
    def find_containing_siobj( cls, siobj, session=None, corner="good", limprefix="good_" ):
        return FourCorners.find_containing_siobj( cls, siobj, session=session, corner=corner, limprefix=limprefix )

    @classmethod
    def find_containing( cls, ra, dec, corner="good", limprefix="good_", prov_id=None, session=None ):
        return FourCorners.find_containing( cls, ra, dec, corner=corner, limprefix=limprefix,
                                            prov_id=prov_id, session=session )


    @classmethod
    def find_potentialy_overlapping( cls, fcobj, prov_id=None, session=None, corner="good", limprefix="good_" ):
        return FourCorners.find_potentially_overlapping( cls, fcobj, prov_id=prov_id, session=session,
                                                         corner=corner, limprefix=limprefix )

    @classmethod
    def get_overlap_frac( cls, obj1, obj2, corner="good", limprefix="good_" ):
        return FourCorners.get_overlap_frac( obj1, obj2, corner=corner, limprefix=limprefix )


    def contains( self, ra, dec, corner="good", limprefix="good_" ):
        return FourCorners.contains( self, ra, dec, corner=corner, limprefix=limprefix )


    def set_corners_minmax( self, ras, decs, goodras=None, gooddecs=None ):
        FourCorners.set_corners_minmax( self, ras, decs )
        if goodras is not None:
            ras, decs, minra, maxra, mindec, maxdec = FourCorners.sort_radec( goodras, gooddecs )
            self.ra_good_00 = ras[0]
            self.ra_good_01 = ras[1]
            self.ra_good_10 = ras[2]
            self.ra_good_11 = ras[3]
            self.good_minra = minra
            self.good_maxra = maxra
            self.dec_good_00 = decs[0]
            self.dec_good_01 = decs[1]
            self.dec_good_10 = decs[2]
            self.dec_good_11 = decs[3]
            self.good_mindec = mindec
            self.good_maxdec = maxdec


    def set_corners_from_wcs( self, wcs, width=None, height=None, setradec=False, mask=None ):
        """Update four corners.

        If mask is given, use that to set the "good" corners and limits.
        Otherwise, they will be direct copies of the regular ones.

        """

        if ( width is None ) or ( height is None ):
            if mask is None:
                raise ValueError( "Must give width/height, or mask" )
            width = width if width is not None else mask.shape[1]
            height = height if height is not None else mask.shape[0]

        xs = [ 0., width-1., 0., width-1. ]
        ys = [ 0., 0., height-1., height-1. ]
        ras, decs = wcs.pixel_to_world_values( xs, ys )

        if mask is not None:
            # Figure out what is the outer "bad region" that we want to throw out
            # or along y to get the xs that have any good pixels
            xok = np.where( np.any( mask==0, axis=0 ) )[0]
            xgood0 = np.min( xok )
            xgood1 = np.max( xok )
            # or along x to get the ys that have any good pixels
            yok = np.where( np.any( mask==0, axis=1 ) )[0]
            ygood0 = np.min( yok )
            ygood1 = np.max( yok )

            xs = [ xgood0, xgood0, xgood1, xgood1 ]
            ys = [ ygood0, ygood1, ygood0, ygood1 ]
            goodras, gooddecs = wcs.pixel_to_world_values( xs, ys )
        else:
            goodras = ras
            gooddecs = decs

        self.set_corners_minmax( ras, decs, goodras, gooddecs )

        if setradec:
            # So... do we want it at the center of the whole image, or the center of the
            #   good part?  Let's do whole image.  Answer not obvious.
            sc = wcs.pixel_to_world( width / 2., height / 2. )
            self.ra = sc.ra.to(u.deg).value
            self.dec = sc.dec.to(u.deg).value
            self.gallat = sc.galactic.b.deg
            self.gallon = sc.galactic.l.deg
            self.ecllat = sc.barycentrictrueecliptic.lat.deg
            self.ecllon = sc.barycentrictrueecliptic.lon.deg



class HasBitFlagBadnessButNoUpstream:
    """A mixin class that adds a bitflag marking why this object is bad.

    Most classes actually use HasBitFlagBadness, but Exposure uses this one.

    """

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

    @hybrid_property
    def bitflag(self):
        if self._bitflag is None:
            self._bitflag = 0
        if hasattr( self, '_upstream_bitflag' ):
            if self._upstream_bitflag is None:
                self._upstream_bitflag = 0
            return self._bitflag | self._upstream_bitflag
        else:
            return self._bitflag

    @bitflag.inplace.expression
    @classmethod
    def bitflag(cls):
        if hasattr( cls, '_bitflag' ):
            return cls._bitflag.op('|')(cls._upstream_bitflag)
        else:
            return cls._bitflag

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
            with PGDB() as pgdb:
                q = sql.SQL( "UPDATE {tab} SET _bitflag={bad} WHERE _id={objid}" )
                q = q.format( tab=sql.Identifier(self.__tablename__), bad=self._bitflag, objid=self.id )
                pgdb.execute( q )
                pgdb.commit()


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
        if hasattr( self, '_upstream_bitflag' ):
            self._upstream_bitflag = 0

    def update_downstream_badness(self, session=None, commit=True, _objbank=None):
        """Send a recursive command to update all downstream objects that have bitflags.

        Since this function is called recursively, it always updates the
        current object's _upstream_bitflag to reflect the state of this
        object's immediate upstreams, before calling the same function on all
        downstream objects.

        If session=None and commit=False an exception is raised.

        Parameters
        ----------
        session: PGDB, psycopg.Connection, psycopg.Cursor, or sqlalchemy Session (default None)
            The session to use for the update. If None, will open a new session,
            which will also close at the end of the call. In that case, must
            provide commit=True to commit the changes,

        commit: bool (default True)
            Whether to commit the changes to the database.

        _objbank: dict
            Don't pass this, it's only used internally.

        """

        if ( session is None ) and ( not commit ):
            raise ValueError( "Must either pass a session, or set commit to True." )

        # Keep an object bank so we don't have to keep regetting stuff from the database in recursive calls.
        # (...though would that happne?  Not sure, would have to think about the possible upstream/downstream
        # trees.)  (Pretty sure it could happen.  Consider, for instance, a reference that has more than one
        # downstream subtraction.  It's possible that more than one of those subtractions will share the
        # same upstream new zp, if the subtraction provenance was changed.  In that case, object_bank
        # saves us from grabbing the upstream zeropoints repeatedly.)
        if _objbank is None:
            _objbank = {}

        with PGDB(session) as pgdb:
            if self.id not in _objbank.keys():
                _objbank[ self.id ] = self
            elif self is not _objbank[ self.id ]:
                raise RuntimeError( "This should never happen" )

            # Start from scratch; we're updating recursively, and it's possible
            #  some bits will have been cleared, so just bitwise anding with the
            #  existing might be the wrong thing.
            new_bitflag = 0

            for upstream_model, upstream_id in self.get_upstream_ids( pgdb=pgdb ):
                if upstream_id in _objbank.keys():
                    upstream = _objbank[ upstream_id ]
                else:
                    # HACK ALERT.... remove this ugly hack when Issue #542 is solved
                    if upstream_model.__name__ == 'Exposure':
                        upstream = upstream_model.get_by_id( upstream_id, pgdb=pgdb, nofile=True )
                    else:
                        upstream = upstream_model.get_by_id( upstream_id, pgdb=pgdb )
                    _objbank[ upstream_id ] = upstream
                if hasattr(upstream, '_bitflag'):
                    new_bitflag |= upstream.bitflag

            if hasattr( self, '_upstream_bitflag' ):
                self._upstream_bitflag = new_bitflag
                pgdb.execute( sql.SQL( "UPDATE {tab} SET _upstream_bitflag={val} WHERE _id={me}" )
                              .format( tab=sql.Identifier(self.__tablename__),
                                       val=self._upstream_bitflag,
                                       me=self.id ) )

            # recursively do this for all downstream objects
            for downstream_model, downstream_id in self.get_downstream_ids( pgdb=pgdb ):
                if ( hasattr( downstream_model, 'update_downstream_badness' ) and
                     callable( downstream_model.update_downstream_badness )
                    ):
                    if downstream_id not in _objbank:
                        _objbank[ downstream_id ] = downstream_model.get_by_id( downstream_id, pgdb=pgdb )
                    _objbank[ downstream_id ].update_downstream_badness( session=pgdb, commit=False, _objbank=_objbank )


            if commit:
                pgdb.commit()


    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object

        For the base class this is the most inclusive inverse (allows all badness).
        """
        return data_badness_inverse


class HasBitFlagBadness( HasBitFlagBadnessButNoUpstream ):
    _upstream_bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( '0' ),
        index=True,
        doc='Bitflag of objects used to generate this object. '
    )


class ArchiveLock( Base, UUIDMixin ):
    __tablename__ = 'archive_locks'

    serverpath = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        unique=True,
        doc="Path on the archive server that we want to lock"
    )

    hostname = sa.Column(
        sa.Text,
        nullable=True,
        server_default=None,
        doc="hostname that holds the lock"
    )

    pid = sa.Column(
        sa.Integer,
        nullable=True,
        server_default=None,
        doc="PID of the process that holds the lock"
    )

    identifier = sa.Column(
        sa.Text,
        nullable=True,
        server_default=None,
        doc=( "Some sort of identifier of the thread that holds the lock so it can be sure not to delete "
              "locks owned by other threads." )
    )


    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )


    @staticmethod
    def lockfunc( serverpath,
                  unlock=False,
                  sleep_min=0.5,
                  sleep_init=2,
                  sleep_max=32,
                  sleep_fac=2,
                  sleep_fuzz=0.1 ):
        if unlock:
            with PGDB() as con:
                q = sql.SQL( "DELETE FROM archive_locks "
                             "WHERE serverpath={path} "
                             "  AND hostname={host} "
                             "  AND pid={pid} "
                             "  AND identifier={id}",
                            ).format( path=serverpath,
                                      host=socket.gethostname(),
                                      pid=os.getpid(),
                                      id=str(threading.get_ident()) )
                con.execute_nofetch( q )
                con.commit()
                return

        rng = np.random.default_rng()
        sleept = sleep_init
        t0 = time.perf_counter()
        ok = False
        while not ok:
            with PsycopgConnection() as con:
                try:
                    cursor = con.cursor()
                    cursor.execute( "LOCK TABLE archive_locks" )
                    cursor.execute( "SELECT serverpath, hostname, pid, identifier, created_at "
                                    "FROM archive_locks "
                                    "WHERE serverpath=%(path)s",
                                    { 'path': serverpath } )
                    rows = cursor.fetchall()
                    if len(rows) == 0:
                        cursor.execute( "INSERT INTO archive_locks(serverpath, hostname, pid, identifier) "
                                        "VALUES (%(path)s, %(host)s, %(pid)s, %(id)s)",
                                        { 'path': serverpath,
                                          'host': socket.gethostname(),
                                          'pid': os.getpid(),
                                          'id': str(threading.get_ident()) }
                                       )
                        con.commit()
                        ok = True
                finally:
                    con.rollback()

            if not ok:
                nextsleept = sleept * sleep_fac
                if nextsleept > sleep_max:
                    raise RuntimeError( f"Failed to get archive lock on {serverpath} after "
                                        f"{time.perf_counter()-t0:.1f}s" )
                actualsleept = max( sleep_min, sleept + rng.normal( scale=sleep_fuzz * sleept ) )
                SCLogger.debug( f"PID {os.getpid()} thread {threading.get_ident()} didn't get "
                                f"archive lock on {serverpath}" )
                SCLogger.debug( f"Archive lock held by {rows[0][1]} PID {rows[0][2]} thread "
                                f"{rows[0][3]} at {rows[0][4]}" )
                SCLogger.info( f"Archive lock exists on {serverpath}; sleeping {actualsleept:.1f}s and trying again." )
                if len(rows) > 1:
                    SCLogger.error( f"{len(rows)} locks held on {serverpath}; that's not supposed to happen!" )
                time.sleep( actualsleept )
                sleept = nextsleept


if __name__ == "__main__":
    pass
