import pytest
import datetime
import logging

import psycopg2
import psycopg2.extras

from models.base import Psycopg2Connection, SmartSession
from models.enums_and_bitflags import KnownExposureStateConverter
from models.exposure import Exposure
from models.knownexposure import KnownExposure
from pipeline.exposure_processor import ExposureProcessor


# This test won't the multiprocessing part of exposure_processor.
# That happens in test_pipeline_exposure_launcher
def test_exposure_processor( decam_default_calibrators,
                             conductor_config_decam_pull_all_held,
                             decam_elais_e1_two_references ):
    decam_exposure_name = 'c4d_230702_080904_ori.fits.fz'

    exposureid = None
    zpid = None
    try:
        # Make sure that the exposure is currently held
        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( "SELECT * FROM knownexposures WHERE instrument='DECam' AND identifier=%(iden)s",
                            { 'iden': decam_exposure_name } )
            ke = cursor.fetchone()
        assert ke['_state'] == KnownExposureStateConverter.to_int( 'held' )
        assert ke['claim_time'] is None
        assert ke['start_time'] is None
        assert ke['release_time'] is None
        assert ke['cluster_id'] is None
        assert ke['node_id'] is None
        assert ke['machine_name'] is None
        assert ke['exposure_id'] is None

        # Set up an exposure processor that will run through photocal
        processor = ExposureProcessor( 'DECam', decam_exposure_name, 1, 'test', 'test', machine_name='test',
                                       onlychips=['S2'], through_step='photocal', worker_log_level=logging.DEBUG )

        # Make sure it yells at us if we don't assume_claimed and the exposure is claimed by somebody else
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            for state in [ 'claimed', 'running' ]:
                # for clust, node, mach in zip ( ['nottest', 'test', 'test'], ['test', 'nottest', 'test'],
                #                                ['test', 'test', 'nottest'] ):
                for clust, node, mach in zip( [ 'nottest' ], [ 'test' ], [ 'test' ] ):
                    cursor.execute( "UPDATE knownexposures SET _state=%(state)s, cluster_id=%(clus)s, "
                                    "node_id=%(nod)s, machine_name=%(mach)s "
                                    "WHERE instrument='DECam' AND identifier=%(iden)s",
                                    { 'state': KnownExposureStateConverter.to_int( state ),
                                      'clus': clust,
                                      'nod': node,
                                      'mach': mach,
                                      'iden': decam_exposure_name } )
                    conn.commit()

                    with pytest.raises( RuntimeError, match=f"Exposure is in state {state}, isn't claimed by me" ):
                        processor.secure_exposure()

                    cursor.execute( "UPDATE knownexposures SET _state=%(state)s, cluster_id=NULL, "
                                    "node_id=NULL, machine_name=NULL "
                                    "WHERE instrument='DECam' and identifier=%(iden)s",
                                    { 'state': KnownExposureStateConverter.to_int( 'held' ),
                                      'iden': decam_exposure_name } )
                    conn.commit()

        # Make sure we can download the exposure
        t0 = datetime.datetime.now( tz=datetime.UTC )
        processor.secure_exposure()
        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory = psycopg2.extras.RealDictCursor )
            cursor.execute( "SELECT * FROM knownexposures WHERE instrument='DECam' AND identifier=%(iden)s",
                            { 'iden': decam_exposure_name } )
            ke = cursor.fetchone()
            t1 = datetime.datetime.now( tz=datetime.UTC )
            assert ke['exposure_id'] is not None
            exposureid = ke['exposure_id']
            assert ke['_state'] == KnownExposureStateConverter.to_int( 'running' )
            assert ke['cluster_id'] == 'test'
            assert ke['node_id'] == 'test'
            assert ke['machine_name'] == 'test'
            assert ke['claim_time'] is not None
            assert ke['claim_time'] > t0
            assert ke['claim_time'] < t1
            assert ke['start_time'] is None
            assert ke['release_time'] is None

            cursor.execute( "SELECT * FROM exposures WHERE _id=%(id)s", { 'id': exposureid } )
            exp = cursor.fetchone()
            assert exp is not None
            assert exp['instrument'] == "DECam"
            assert exp['origin_identifier'] == decam_exposure_name

        # Run that puppy, make sure it only goes through photocal
        processor()

        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( "SELECT * FROM knownexposures WHERE instrument='DECam' AND identifier=%(iden)s",
                            { 'iden': decam_exposure_name } )
            ke = cursor.fetchone()
            assert ke['_state'] == KnownExposureStateConverter.to_int( 'done' )
            assert ke['cluster_id'] == 'test'
            assert ke['node_id'] == 'test'
            assert ke['machine_name'] == 'test'
            assert ke['claim_time'] is not None
            assert ke['claim_time'] > t0
            assert ke['claim_time'] < t1
            assert ke['release_time'] is not None
            assert ke['start_time'] > t1
            assert ke['release_time'] > ke['start_time']
            assert ke['release_time'] < datetime.datetime.now( tz=datetime.UTC )

            cursor.execute( "SELECT i.section_id, z._id, z.created_at FROM zero_points z "
                            "INNER JOIN world_coordinates w ON z.wcs_id=w._id "
                            "INNER JOIN source_lists s ON w.sources_id=s._id "
                            "INNER JOIN images i ON s.image_id=i._id "
                            "INNER JOIN exposures e ON i.exposure_id=e._id "
                            "WHERE e._id=%(expid)s",
                            { 'expid': exposureid } )
            rows = cursor.fetchall()
            assert len(rows) == 1
            assert rows[0]['section_id'] == 'S2'
            zpid = rows[0]['_id']
            assert rows[0]['created_at'] > t1

            # Make sure no subtraction
            cursor.execute( "SELECT sub._id FROM images sub "
                            "INNER JOIN image_subtraction_components isc ON sub._id=isc.image_id "
                            "INNER JOIN zero_points z ON isc.new_zp_id=z._id "
                            "INNER JOIN world_coordinates w ON z.wcs_id=w._id "
                            "INNER JOIN source_lists s ON w.sources_id=s._id "
                            "INNER JOIN images i ON s.image_id=i._id "
                            "INNER JOIN exposures e ON i.exposure_id=e._id "
                            "WHERE e._id=%(expid)s",
                            { 'expid': exposureid } )
            rows = cursor.fetchall()
            assert len(rows) == 0

        # Make a new processor that will pick up where this one left off
        processor = ExposureProcessor( 'DECam', decam_exposure_name, 1, 'test', 'test', machine_name='test',
                                       onlychips=['S2'], worker_log_level=logging.DEBUG )
        with pytest.raises( ValueError, match="There's already an exposure associated.*but cont is False" ):
            processor.secure_exposure( cont=False )

        # cont defaults to True
        t0 = datetime.datetime.now( tz=datetime.UTC )
        processor.secure_exposure()
        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( "SELECT * FROM knownexposures WHERE instrument='DECam' AND identifier=%(iden)s",
                            { 'iden': decam_exposure_name } )
            ke = cursor.fetchone()
            t1 = datetime.datetime.now( tz=datetime.UTC )
            assert ke['exposure_id'] is not None and ke['exposure_id'] == exposureid
            assert ke['_state'] == KnownExposureStateConverter.to_int( 'running' )
            assert ke['cluster_id'] == 'test'
            assert ke['node_id'] == 'test'
            assert ke['machine_name'] == 'test'
            assert ke['claim_time'] is not None
            # assert ke['claim_time'] > t0   # This won't be true; claimed already by us, won't update claim time
            assert ke['claim_time'] < t1
            assert ke['start_time'] is None
            assert ke['release_time'] is None

        # Run the processor, make sure it gets all the way through deep scoring
        t2 = datetime.datetime.now( tz=datetime.UTC )
        processor()

        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( "SELECT * FROM knownexposures WHERE instrument='DECam' AND identifier=%(iden)s",
                            { 'iden': decam_exposure_name } )
            ke = cursor.fetchone()
            assert ke['exposure_id'] is not None and ke['exposure_id'] == exposureid
            assert ke['_state'] == KnownExposureStateConverter.to_int( 'done' )
            assert ke['cluster_id'] == 'test'
            assert ke['node_id'] == 'test'
            assert ke['machine_name'] == 'test'
            assert ke['claim_time'] is not None
            # assert ke['claim_time'] > t0
            assert ke['claim_time'] < t1
            assert ke['start_time'] is not None
            assert ke['start_time'] > t2
            assert ke['release_time'] is not None
            assert ke['release_time'] > ke['start_time']
            assert ke['release_time'] < datetime.datetime.now( tz=datetime.UTC )

            cursor.execute( "SELECT ds._id, ds.created_at, z._id AS zpid, z.created_at as zpcreated, e._id AS expid "
                            "FROM deepscore_sets ds "
                            "INNER JOIN measurement_sets ms ON ds.measurementset_id=ms._id "
                            "INNER JOIN cutouts cu ON ms.cutouts_id=cu._id "
                            "INNER JOIN source_lists subsrc ON cu.sources_id=subsrc._id "
                            "INNER JOIN images sub ON subsrc.image_id=sub._id "
                            "INNER JOIN image_subtraction_components isc ON isc.image_id=sub._id "
                            "INNER JOIN zero_points z ON isc.new_zp_id=z._id "
                            "INNER JOIN world_coordinates w ON z.wcs_id=w._id "
                            "INNER JOIN source_lists s ON w.sources_id=s._id "
                            "INNER JOIN images i ON s.image_id=i._id "
                            "INNER JOIN exposures e ON e._id=%(expid)s",
                            { 'expid': exposureid } )
            rows = cursor.fetchall()
            assert len(rows) == 1
            assert rows[0]['created_at'] > t2
            assert rows[0]['zpcreated'] < t0
            assert rows[0]['zpid'] == zpid
            assert rows[0]['expid'] == exposureid

        # NOTE -- we haven't tested "delete"

    finally:
        if exposureid is not None:
            with SmartSession() as sess:
                ke = sess.query( KnownExposure ).filter( KnownExposure.exposure_id==exposureid ).first()
                ke.exposure_id = None
                sess.commit()
                exp = Exposure.get_by_id( exposureid, session=sess )
            exp.delete_from_disk_and_database()
