import pytest

from models.base import SmartSession
from models.calibratorfile import CalibratorFileDownloadLock

def test_acquire_lock():
    # Should be able to do this twice in a row because the lock gets
    #  released when the with exits
    for i in range(2):
        with CalibratorFileDownloadLock.acquire_lock( 'DECam', 'N1', 'externally_supplied', 'flat', 'dome' ) as l1:
            with SmartSession() as sess:
                lfound = ( sess.query( CalibratorFileDownloadLock )
                           .filter( CalibratorFileDownloadLock.type == 'flat' )
                           .filter( CalibratorFileDownloadLock.calibrator_set == 'externally_supplied' )
                           .filter( CalibratorFileDownloadLock.flat_type == 'dome' )
                           .filter( CalibratorFileDownloadLock.instrument == 'DECam' )
                           .filter( CalibratorFileDownloadLock.sensor_section == 'N1' )
                          ).first()
                assert lfound is not None
                assert lfound.id == l1

    # Should be no locks leftover
    with SmartSession() as sess:
        assert sess.query( CalibratorFileDownloadLock ).count() == 0


def test_acquire_lock_failure():
    # Create the lock in one session
    with CalibratorFileDownloadLock.acquire_lock( 'DECam', 'N1', 'externally_supplied', 'flat', 'dome' ) as l1:
        # Now try to create the lock in another session:
        with pytest.raises( RuntimeError, match="Couldn't get CalibratorFileDownloadLock for.*after many tries" ):
            with CalibratorFileDownloadLock.acquire_lock( 'DECam', 'N1', 'externally_supplied', 'flat', 'dome',
                                                          maxsleep=1 ) as l2:
                pass

    # Should be no locks leftover
    with SmartSession() as sess:
        assert sess.query( CalibratorFileDownloadLock ).count() == 0



