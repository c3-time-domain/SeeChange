import os
import time
import math
import datetime
import contextlib
import random

import sqlalchemy as sa
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.exc import IntegrityError

from models.base import Base, UUIDMixin, SmartSession
from models.enums_and_bitflags import CalibratorTypeConverter, CalibratorSetConverter, FlatTypeConverter

# from util.logger import SCLogger


class CalibratorFile(Base, UUIDMixin):
    __tablename__ = 'calibrator_files'

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        server_default=sa.sql.elements.TextClause(str(CalibratorTypeConverter.convert( 'unknown' )) ) ,
        doc="Type of calibrator (Dark, Flat, Linearity, etc.)"
    )

    @hybrid_property
    def type( self ):
        return CalibratorTypeConverter.convert( self._type )

    @type.expression
    def type( cls ):  # noqa: N805
        return sa.case( CalibratorTypeConverter.dict, value=cls._type )

    @type.setter
    def type( self, value ):
        self._type = CalibratorTypeConverter.convert( value )

    _calibrator_set = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        server_default=sa.sql.elements.TextClause( str(CalibratorTypeConverter.convert('unknown')) ),
        doc="Calibrator set for instrument (unknown, externally_supplied, general, nightly)"
    )

    @hybrid_property
    def calibrator_set( self ):
        return CalibratorSetConverter.convert( self._calibrator_set )

    @calibrator_set.expression
    def calibrator_set( cls ):  # noqa: N805
        return sa.case( CalibratorSetConverter.dict, value=cls._calibrator_set )

    @calibrator_set.setter
    def calibrator_set( self, value ):
        self._calibrator_set = CalibratorSetConverter.convert( value )

    _flat_type = sa.Column(
        sa.SMALLINT,
        nullable=True,
        index=True,
        doc="Type of flat (unknown, observatory_supplied, sky, twilight, dome), or None if not a flat"
    )

    @hybrid_property
    def flat_type( self ):
        return FlatTypeConverter.convert( self._flat_type )

    @flat_type.inplace.expression
    @classmethod
    def flat_type( cls ):
        return sa.case( FlatTypeConverter.dict, value=cls._flat_type )

    @flat_type.setter
    def flat_type( self, value ):
        self._flat_type = FlatTypeConverter.convert( value )

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Instrument this calibrator image is for"
    )

    sensor_section = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Sensor Section of the Instrument this calibrator image is for"
    )

    image_id = sa.Column(
        sa.ForeignKey( 'images._id', ondelete='CASCADE', name='calibrator_files_image_id_fkey' ),
        nullable=True,
        index=True,
        doc='ID of the image (if any) that is this calibrator'
    )

    datafile_id = sa.Column(
        sa.ForeignKey( 'data_files._id', ondelete='CASCADE', name='calibrator_files_data_file_id_fkey' ),
        nullable=True,
        index=True,
        doc='ID of the miscellaneous data file (if any) that is this calibrator'
    )

    validity_start = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=( 'The time (of exposure acquisition) for exposures for '
              ' which this calibrator file becomes valid.  If None, this '
              ' calibrator is valid from the beginning of time.' )
    )

    validity_end = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=( 'The time (of exposure acquisition) for exposures for '
              ' which this calibrator file is no longer.  If None, this '
              ' calibrator is valid to the end of time.' )
    )

    def __repr__(self):
        return (
            f'<CalibratorFile('
            f'id={self.id}, '
            f'set={self.calibrator_set}, '
            f'type={self.type}, '
            f'image_id={self.image_id}, '
            f'datafile_id={self.datafile_id}, '
            f'for {self.instrument} section {self.sensor_section}'
            f'>'
        )


# This next table is kind of an ugly hack put in place
#   to deal with race conditions; see Instrument.preprocessing_calibrator_files

class CalibratorFileDownloadLock(Base, UUIDMixin):
    __tablename__ = 'calibfile_downloadlock'

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            UniqueConstraint( '_type', '_calibrator_set', '_flat_type', 'instrument', 'sensor_section',
                              name='calibfile_downloadlock_unique',
                              postgresql_nulls_not_distinct=True ),
        )

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=False,
        server_default=sa.sql.elements.TextClause( str(CalibratorTypeConverter.convert( 'unknown' )) ),
        doc="Type of calibrator (Dark, Flat, Linearity, etc.)"
    )

    _locks = {}

    @hybrid_property
    def type( self ):
        return CalibratorTypeConverter.convert( self._type )

    @type.expression
    def type( cls ):  # noqa: N805
        return sa.case( CalibratorTypeConverter.dict, value=cls._type )

    @type.setter
    def type( self, value ):
        self._type = CalibratorTypeConverter.convert( value )

    _calibrator_set = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=False,
        server_default=sa.sql.elements.TextClause( str(CalibratorTypeConverter.convert('unknown')) ),
        doc="Calibrator set for instrument (unknown, externally_supplied, general, nightly)"
    )

    @hybrid_property
    def calibrator_set( self ):
        return CalibratorSetConverter.convert( self._calibrator_set )

    @calibrator_set.expression
    def calibrator_set( cls ):  # noqa: N805
        return sa.case( CalibratorSetConverter.dict, value=cls._calibrator_set )

    @calibrator_set.setter
    def calibrator_set( self, value ):
        self._calibrator_set = CalibratorSetConverter.convert( value )

    _flat_type = sa.Column(
        sa.SMALLINT,
        nullable=True,
        index=False,
        doc="Type of flat (unknown, observatory_supplied, sky, twilight, dome), or None if not a flat"
    )

    @hybrid_property
    def flat_type( self ):
        return FlatTypeConverter.convert( self._flat_type )

    @flat_type.inplace.expression
    @classmethod
    def flat_type( cls ):
        return sa.case( FlatTypeConverter.dict, value=cls._flat_type )

    @flat_type.setter
    def flat_type( self, value ):
        self._flat_type = FlatTypeConverter.convert( value )

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=False,
        doc="Instrument this calibrator image is for"
    )

    sensor_section = sa.Column(
        sa.Text,
        nullable=True,
        index=False,
        doc="Sensor Section of the Instrument this calibrator image is for"
    )

    def __repr__( self ) :
        return ( f"CalibratorFileDownloadLock("
                 f"id={self.id}, "
                 f"calibrator_set={self.calibrator_set}, "
                 f"type={self.type}, "
                 f"flat_type={self.flat_type}, "
                 f"for {self.instrument} section {self.sensor_section})" )

    @classmethod
    @contextlib.contextmanager
    def acquire_lock( cls, instrument, section, calibset, calibtype, flattype=None, maxsleep=40, session=None ):
        """Get a lock on updating/adding Calibrators of a given type for a given instrument/section.

        This class method should *only* be called as a context manager ("with").

        Parameters
        ----------
        instrument: str
           The instrument

        section: str
           The sensor section, or None (meaning an instrument-global file)

        calibset: str
           The calibrator set

        calibtype: str
           The calibrator type (e.g. 'linearity', 'flat', etc.)

        flattype: str, default None
           The flat type if calibtype is 'flat'

        maxsleep: int, default 40
           Keep trying for at most this many seconds before finally
           giving up and failing to get the lock.

        session: Session

        We need to avoid a race condition where two processes both look
        for a calibrator file, don't find it, and both try to download
        it at the same time.  Just using database locks doesn't work
        here, because the process of downloading and committing the
        images takes long enough that the database server starts whining
        about deadlocks.  So, we manually invent our own database lock
        mechanism here and use that.  (One advantage is that we can just
        lock the specific thing being downloaded.)

        This has a danger : if the code fully crashes with a lock
        checked out, it will leave behind this lock (i.e. the row in the
        database).  That should be rare, because of the use of yield
        below, but a badly-time crash could leave these rows behind.

        """

        # First, see if the session already has the lock, and if so, just
        #  return (not yield) it.  (In this case, this function was called
        #  earlier with the same session.)
        if ( session is not None ) and ( session in cls._locks ):
            return cls._locks[ session ]

        lockid = None
        fail = False
        sleepmin = 0.25
        sleepsigma = 0.25
        totsleep = 0.
        # Use os.urandom() to see the rng because if we just use the
        #  random module stuff, it will use the system clock.  A bunch
        #  of processes may well hit this line at the same time and get
        #  the same random seed.
        random.seed( os.urandom(4) )
        try:
            while ( lockid is None ) and ( not fail ):
                # Try to create the lock
                with SmartSession(session) as sess:
                    try:
                        caliblock = CalibratorFileDownloadLock( calibrator_set=calibset,
                                                                instrument=instrument,
                                                                type=calibtype,
                                                                sensor_section=section,
                                                                flat_type=flattype )
                        sess.add( caliblock )
                        # SCLogger.debug( "CalibratorFileDownloadLock comitting" )
                        sess.commit()
                        sess.refresh( caliblock )
                        lockid = caliblock.id
                    except IntegrityError:
                        sess.rollback()
                        lockid = None
                if lockid is None:
                    # Lock already existed, so wait a bit and try again
                    if totsleep > maxsleep:
                        fail = True
                    else:
                        # We used to keep exponentially expanding the sleep time by a factor of 2
                        # each time, but that had a race condition of its own.  When launching a
                        # bunch of processes with a multiprocessing pool, they'd all be synchronized
                        # enough that multiple processes would get to a long sleep at the same time,
                        # and then all pool for the lock at close enough to the same time that only
                        # one would get it.  The rest would all wait a very long time (while, for
                        # most of it, no lock was being held) before trying again.  They'd only have
                        # a few tries left, and ultimately several would fail.  So, instead, wait a
                        # random amount of time, to prevent synchronization.
                        tsleep = sleepmin + math.fabs( random.normalvariate( mu=0., sigma=sleepsigma ) )
                        time.sleep( tsleep )
                        totsleep += tsleep

            if fail:
                raise RuntimeError( f"Couldn't get CalibratorFileDownloadLock for "
                                    f"{instrument} {section} {calibset} {calibtype} after many tries." )

            # Assign the lock to the passed session, if any
            if session is not None:
                cls._locks[session] = lockid

            yield lockid

        finally:
            if lockid is not None:
                with SmartSession(session) as sess:
                    # SCLogger.debug( f"Deleting calibfile_downloadlock {lockid}" )
                    sess.connection().execute( sa.text( 'DELETE FROM calibfile_downloadlock WHERE _id=:id' ),
                                               { 'id': lockid } )
                    sess.commit()

            if session is not None:
                try:
                    del cls._locks[ session ]
                except KeyError:
                    pass


    @classmethod
    def lock_reaper( cls, secondsold=120 ):
        """Utility function for cleaning out rows that are older than a certain cutoff."""

        cutoff = datetime.datetime.now() - datetime.timestamp( seconds=secondsold )
        with SmartSession() as sess:
            oldlocks = sess.query( cls ).filter( cls.created_at > cutoff ).all()
            for oldlock in oldlocks:
                sess.delete( oldlock )
            sess.commit()
