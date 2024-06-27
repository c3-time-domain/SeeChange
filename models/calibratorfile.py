import time
import datetime
import contextlib

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property

from models.base import Base, AutoIDMixin, SmartSession
from models.image import Image
from models.datafile import DataFile
from models.enums_and_bitflags import CalibratorTypeConverter, CalibratorSetConverter, FlatTypeConverter

from util.logger import SCLogger

class CalibratorFile(Base, AutoIDMixin):
    __tablename__ = 'calibrator_files'

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        default=CalibratorTypeConverter.convert( 'unknown' ),
        doc="Type of calibrator (Dark, Flat, Linearity, etc.)"
    )

    @hybrid_property
    def type( self ):
        return CalibratorTypeConverter.convert( self._type )

    @type.expression
    def type( cls ):
        return sa.case( CalibratorTypeConverter.dict, value=cls._type )

    @type.setter
    def type( self, value ):
        self._type = CalibratorTypeConverter.convert( value )

    _calibrator_set = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        default=CalibratorTypeConverter.convert('unknown'),
        doc="Calibrator set for instrument (unknown, externally_supplied, general, nightly)"
    )

    @hybrid_property
    def calibrator_set( self ):
        return CalibratorSetConverter.convert( self._calibrator_set )

    @calibrator_set.expression
    def calibrator_set( cls ):
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
        sa.ForeignKey( 'images.id', ondelete='CASCADE', name='calibrator_files_image_id_fkey' ),
        nullable=True,
        index=True,
        doc='ID of the image (if any) that is this calibrator'
    )

    image = orm.relationship(
        'Image',
        cascade='save-update, merge, refresh-expire, expunge',  # ROB REVIEW THIS
        doc='Image for this CalibratorImage (if any)'
    )

    datafile_id = sa.Column(
        sa.ForeignKey( 'data_files.id', ondelete='CASCADE', name='calibrator_files_data_file_id_fkey' ),
        nullable=True,
        index=True,
        doc='ID of the miscellaneous data file (if any) that is this calibrator'
    )

    datafile = orm.relationship(
        'DataFile',
        cascade='save-update, merge, refresh-expire, expunge', # ROB REVIEW THIS
        doc='DataFile for this CalibratorFile (if any)'
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

class CalibratorFileDownloadLock(Base, AutoIDMixin):
    __tablename__ = 'calibfile_downloadlock'

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        default=CalibratorTypeConverter.convert( 'unknown' ),
        doc="Type of calibrator (Dark, Flat, Linearity, etc.)"
    )

    _locks = {}

    @hybrid_property
    def type( self ):
        return CalibratorTypeConverter.convert( self._type )

    @type.expression
    def type( cls ):
        return sa.case( CalibratorTypeConverter.dict, value=cls._type )

    @type.setter
    def type( self, value ):
        self._type = CalibratorTypeConverter.convert( value )

    _calibrator_set = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        default=CalibratorTypeConverter.convert('unknown'),
        doc="Calibrator set for instrument (unknown, externally_supplied, general, nightly)"
    )

    @hybrid_property
    def calibrator_set( self ):
        return CalibratorSetConverter.convert( self._calibrator_set )

    @calibrator_set.expression
    def calibrator_set( cls ):
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
        nullable=True,
        index=True,
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
    def acquire_lock( cls, instrument, section, calibset, calibtype, flattype=None, maxsleep=20, session=None ):
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

        maxsleep: int, default 25
           When trying to get a lock, this routine will sleep after
           failing to get it.  It start sleeping at 0.1 seconds, and
           doubles the sleep time each time it fails.  Once sleeptime is
           this value or greater, it will raise an exception.

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

        lockid = None
        sleeptime = 0.1
        while lockid is None:
            with SmartSession(session) as sess:
                # Lock the calibfile_downloadlock table to avoid a race condition
                sess.connection().execute( sa.text( 'LOCK TABLE calibfile_downloadlock' ) )

                # Check to see if there's a lock now
                lockq = ( sess.query( CalibratorFileDownloadLock )
                          .filter( CalibratorFileDownloadLock.calibrator_set == calibset )
                          .filter( CalibratorFileDownloadLock.instrument == instrument )
                          .filter( CalibratorFileDownloadLock.type == calibtype )
                          .filter( CalibratorFileDownloadLock.sensor_section == section ) )
                if calibtype == 'flat':
                    lockq = lockq.filter( CalibratorFileDownloadLock.flat_type == flattype )
                if lockq.count() == 0:
                    # There isn't, so create the lock
                    caliblock = CalibratorFileDownloadLock( calibrator_set=calibset,
                                                            instrument=instrument,
                                                            type=calibtype,
                                                            sensor_section=section,
                                                            flat_type=flattype )
                    sess.add( caliblock )
                    sess.commit()
                    sess.refresh( caliblock )   # is this necessary?
                    lockid = caliblock.id
                    # SCLogger.debug( f"Created calibfile_downloadlock {lockid}" )
                else:
                    if lockq.count() > 1:
                        raise RuntimeError( f"Database corruption: multiple CalibratorFileDownloadLock for "
                                            f"{instrument} {section} {calibset} {calibtype} {flattype}" )
                    lockid = lockq.first().id
                    sess.rollback()
                    if ( ( lockid in cls._locks.keys() ) and ( cls._locks[lockid] == sess ) ):
                        # The lock already exists, and is owned by this
                        # session, so just return it.  Return not yield;
                        # if the lock already exists, then there should
                        # be an outer with block that grabbed the lock,
                        # and we don't want to delete it prematurely.
                        # (Note that above, we compare
                        # cls._locks[lockid] to sess, not to session.
                        # if cls._locks[lockid] is None, it means that
                        # it's a global lock owned by nobody; if session
                        # is None, it means no session was passed.  A
                        # lack of a sesson doesn't own a lock owned by
                        # nobody.)
                        return lockid
                    else:
                        # Either the lock doesn't exist, or belongs to another session,
                        # so wait a bit and try again.
                        lockid = None
                        if sleeptime > maxsleep:
                            lockid = -1
                        else:
                            time.sleep( sleeptime )
                            sleeptime *= 2
        if lockid == -1:
            raise RuntimeError( f"Couldn't get CalibratorFileDownloadLock for "
                                f"{instrument} {section} {calibset} {calibtype} after many tries." )

        # Assign the lock to the passed session.  (If no session was passed, it will be assigned
        # to None, which is OK.)
        cls._locks[lockid] = session
        yield lockid

        with SmartSession(session) as sess:
            # SCLogger.debug( f"Deleting calibfile_downloadlock {lockid}" )
            sess.connection().execute( sa.text( 'DELETE FROM calibfile_downloadlock WHERE id=:id' ),
                                       { 'id': lockid } )
            sess.commit()
            try:
                del cls._locks[ lockid ]
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
