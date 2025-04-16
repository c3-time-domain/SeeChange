import io
import uuid
import time
import random
import datetime

import numpy as np
import sqlalchemy as sa

import fastavro
import confluent_kafka
import hop
import hop.auth
import hop.models

from models.base import SmartSession
from models.image import Image
from models.reference import image_subtraction_components
from models.source_list import SourceList
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import MeasurementSet, Measurements
from models.deepscore import DeepScoreSet, DeepScore
from models.object import Object
from util.config import Config
from util.logger import SCLogger


# Alerting doesn't work with the Parameters system because there's no Provenance associated with it,
#  so there's no need to get critical parameters.  It will just read the config directly.

class Alerting:

    known_methods = [ 'kafka', 'hopskotch' ]

    def __init__( self, send_alerts=None, methods=None ):
        """Initialize an Alerting object.

        Parameters
        ----------
          send_alerts: bool, default None
             Set to False to globally disable alert sending, or True to
             send alerts based on the methods list and the 'enabled'
             element for each method.  If None, will read
             alerts.send_alerts from config.

          methods: dict, default None
             Configuration of alert sending methods.  If None, will read
             alerts.methods from config.

        """
        cfg = Config.get()
        self.send_alerts = cfg.value( 'alerts.send_alerts' ) if send_alerts is None else send_alerts
        self.methods = cfg.value( 'alerts.methods' ) if methods is None else methods

        if ( not isinstance( self.methods, dict ) ):
            raise TypeError( "Alerting: methods must be a dict." )

        for methodname, methoddef in self.methods.items():
            methoddef['name'] = methodname

            if 'method' not in methoddef:
                raise ValueError( f"'method' is not defined for (at least) alert method {methodname}" )

            if methoddef['method'] not in self.known_methods:
                raise ValueError( f"Unknown alert method {methoddef['method']}; known are {self.known_methods}" )

            if 'enabled' not in methoddef:
                raise ValueError( f"Each method must have a value for 'enabled'; (at least) "
                                  f"{methodname} does not." )

            if methoddef['method'] in ( 'kafka', 'hopskotch' ):
                methoddef['schema'] = fastavro.schema.parse_schema(
                    fastavro.schema.load_schema( methoddef['avro_schema'] ) )

                if 'topic' not in methoddef:
                    if 'topic_pattern' not in methoddef:
                        raise ValueError( f"Each method must have either at topic or a topic_pattern; "
                                          f"{methodname} does not." )
                    now = datetime.datetime.now()
                    barf = "".join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=6 ) )
                    methoddef['topic'] = methoddef['topic_pattern'].format( year = now.year,
                                                                            month = now.month,
                                                                            day = now.day,
                                                                            barf = barf )
                else:
                    if 'topic_pattern' in methoddef:
                        SCLogger.warning( f"topic_pattern for alert method {methodname} ignored because "
                                          f"topic {methoddef['topic']} is given" )


    def run( self, *args, **kwargs ):
        """Calls send.  (Need this method to work with pipeline/top_level.py.)"""
        return self.send( *args, **kwargs )


    def send( self, ds, skip_bad=True ):
        """Send alerts from a fully processed DataStore.

        The DataStore must have run all the way through scoring.

        Will read the 'alerts' section of the config file, and send
        alerts through every method defined underneath alerts.method.

        Parmameters
        -----------
          ds: DataStore

          skip_bad : bool
            If True, do not send alerts for measurements whose is_bad
            field is set to True.  If False, send alerts for all of
            ds.measurements.

        Returns
        -------
          ds

          The only changes that will have been made to the passed
          DataStore are that a key will have been added to ds.runtimes
          and ds.memory_usages if (respectively) ds.update_runtimes and
          ds.update_memory_usages are true.  If there is an exception,
          that will also be appended to ds.exceptions.

        """
        if not self.send_alerts:
            return

        t_start = time.perf_counter()
        try:
            if ds.update_memory_usages:
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            # TODO: verify that datastore has measurements and scores

            # If any of the methods are kafka or hopskotch, we will need to build avro alerts
            avroalerts = None
            if any( method['enabled'] and
                    ( ( method['method'] == 'kafka' ) or ( method['method'] == 'hopskotch' ) )
                    for method in self.methods.values()
                   ):
                avroalerts = self.build_avro_alert_structures( ds, skip_bad=skip_bad )

            # Issue alerts for all methods
            for methodname, method in self.methods.items():
                if not method['enabled']:
                    continue

                if method['method'] == 'kafka':
                    self.send_kafka_alerts( avroalerts, method )

                elif method['method'] == 'hopskotch':
                    self.send_hopskotch_alerts( avroalerts, method )

                else:
                    raise RuntimeError( "This should never happen." )

            return ds

        except Exception as e:
            SCLogger.exception( f"Exception in Alering.send: {e}" )
            ds.exceptions.append( e )
            raise

        finally:
            if ds.update_runtimes:
                ds.runtimes[ 'alerting' ] = time.perf_counter() - t_start
            if ds.update_memory_usages:
                import tracemalloc
                ds.memory_usages['alerting'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2 # in MB


    def dia_source_alert( self, meas, score, img, deepscore_set, zp=None, aperdex=None, fluxscale=None ):
        # For snr, we're going to assume that the detection was approximately
        #   detection in a 1-FWHM aperture.  This isn't really right, but
        #   it should be approximately right.
        #
        # Don't do background subtractions for the fluxes here, because they're
        #   off of difference images.  (Note that the "flux" property of
        #   measurements *does* do background subtraction, but flux_psf and
        #   flux_apertures does not.)

        if aperdex is None:
            radfwhm = np.array(meas.aper_radii) / ( img.fwhm_estimate / img.instrument_object.pixel_scale )
            w = np.where( np.fabs( radfwhm - 1. ) < 0.01 )
            if len(w) == 0.:
                raise RuntimeError( f"No 1FWHM aperture (have {radfwhm})" )
            aperdex = w[0]
        if fluxscale is None:
            if zp is None:
                raise ValueError( "Must pass one of fluxscale or zp" )
            fluxscale = 10 ** ( ( zp.zp - 27.5 ) / -2.5 )

        return { 'diaSourceId': str( meas.id ),
                 'diaObjectId': str( meas.object_id ),
                 'MJD': img.mid_mjd,
                 'ra': meas.ra,
                 'raErr': None,
                 'dec': meas.dec,
                 'decErr': None,
                 'ra_dec_Cov': None,
                 'band': img.filter,
                 'fluxZeroPoint': 27.5,
                 'apFlux': meas.flux_apertures[ aperdex ] * fluxscale,
                 'apFluxErr': meas.flux_apertures_err[ aperdex ] * fluxscale,
                 'psfFlux': meas.flux_psf * fluxscale,
                 'psfFluxErr':meas.flux_psf_err * fluxscale,
                 'rb': None if score is None else score.score,
                 'rbcut': None if score is None else DeepScoreSet.get_rb_cut( deepscore_set.algorithm ),
                 'rbtype': None if score is None else deepscore_set.algorithm
                }

    def dia_object_alert( self, obj ):
        return { 'diaObjectId': str( obj.id ),
                 'name': obj.name,
                 'ra': obj.ra,
                 'raErr': None,
                 'dec': obj.dec,
                 'decErr': None,
                 'ra_dec_Cov': None,
                }


    def build_avro_alert_structures( self, ds, skip_bad=True ):
        sub_image = ds.get_sub_image()
        image = ds.get_image()
        zp = ds.get_zp()
        detections = ds.get_detections()
        measurement_set = ds.get_measurement_set()
        measurements = measurement_set.measurements
        cutouts = ds.get_cutouts()
        cutouts.load_all_co_data( sources=detections )
        deepscore_set = ds.get_deepscore_set()
        scores = deepscore_set.deepscores

        if len(scores) == 0:
            # Nothing to do!
            return

        if len(scores) != len(measurements):
            raise ValueError( f"Alert sending error: DataStore has different number of "
                              f"measurements ({len(measurements)}) and scores ({len(scores)}).  "
                              f"This should never happen." )

        # Figure out which aperture is radius of 1 FWHM
        radfwhm = np.array(measurements[0].aper_radii) / ( image.fwhm_estimate / image.instrument_object.pixel_scale )
        w = np.where( np.fabs( radfwhm - 1. ) < 0.01 )
        if len(w[0]) == 0.:
            raise RuntimeError( f"No 1FWHM aperture (have {radfwhm})" )
        aperdex = w[0][0]

        # We're going to use the standard SNANA zeropoint for no adequately explained reason
        # (We *could* just store the image zeropoint in fluxZeroPoint.  However, it will be
        # more convenient for people if all of the flux values from all of the sources, previous
        # sources, and previous forced sources are on the same scale.  We have to pick something,
        # and SNANA is something.  So, there's an explanation; I don't know if it's an
        # adequate explanation.)
        fluxscale = 10 ** ( ( zp.zp - 27.5 ) / -2.5 )

        alerts = []

        for meas, scr in zip( measurements, scores ):
            # By default, no alerts for bad measurements
            if skip_bad and meas.is_bad:
                continue

            # Make sure cutout data is big-endian floats and that
            #  masked pixels are NaN
            cdex = f'source_index_{meas.index_in_sources}'
            newdata = cutouts.co_dict[cdex]['new_data'].astype(">f4", copy=True)
            refdata = cutouts.co_dict[cdex]['ref_data'].astype(">f4", copy=True)
            subdata = cutouts.co_dict[cdex]['sub_data'].astype(">f4", copy=True)
            newdata[ cutouts.co_dict[cdex]['new_flags'] != 0 ] = np.nan
            refdata[ cutouts.co_dict[cdex]['ref_flags'] != 0 ] = np.nan
            subdata[ cutouts.co_dict[cdex]['sub_flags'] != 0 ] = np.nan

            alert = { 'alertId': str(uuid.uuid4()),
                      'diaSource': {},
                      'prvDiaSources': [],
                      'prvDiaForcedSources': None,
                      'prvDiaNonDetectionLimits': [],
                      'diaObject': {},
                      'cutoutDifference': subdata.tobytes(),
                      'cutoutScience': newdata.tobytes(),
                      'cutoutTemplate': refdata.tobytes() }

            alert['diaSource'] = self.dia_source_alert( meas, scr, image, deepscore_set,
                                                        zp=zp, aperdex=aperdex, fluxscale=fluxscale )
            alert['diaObject'] = self.dia_object_alert( Object.get_by_id( meas.object_id ) )

            # In Image.from_new_and_ref, we set a lot of the sub image's properties (crucially,
            #   filter and mjd) to be the same as the new image.  So, for what we need for
            #   alerts, we can just use the sub image.

            with SmartSession() as sess:
                # Get all previous sources with the same provenance.
                # The cutouts parent is a SourceList that is
                #   detections on the sub image, so it's parent
                #   is the sub image.  We need that for mjd and filter.
                # But, also, we need to get the sub image's parent
                #   zeropoint, which is the zeropoint of the new
                #   image that went into the sub image.  In subtraction,
                #   we normalize the sub image so it has the same
                #   zeropoint as the new image, so that's also the
                #   right zeropoint to use with the Measurements
                #   that we pull out.
                # And, finally, we have to get the DeepScore objects
                #   associated with the previous measurements.
                #   That's not upstream of anything, so we have to
                #   include the DeepScore provenance in the join condition.

                # TODO -- handle previous_sources_days

                prvimgids = {}
                q = ( sess.query( Measurements, DeepScore, Image, ZeroPoint )
                      .join( MeasurementSet, Measurements.measurementset_id==Measurements._id )
                      .join( Cutouts, MeasurementSet.cutouts_id==Cutouts._id )
                      .join( SourceList, Cutouts.sources_id==SourceList._id )
                      .join( Image, SourceList.image_id==Image._id )
                      .join( image_subtraction_components, image_subtraction_components.c.image_id==Image._id )
                      .join( ZeroPoint, ZeroPoint._id==image_subtraction_components.c.new_zp_id )
                      .join( DeepScoreSet, sa.and_( DeepScoreSet.measurementset_id==MeasurementSet._id,
                                                    DeepScoreSet.provenance_id==deepscore_set.provenance_id ),
                             isouter=True )
                      .join( DeepScore, sa.and_( DeepScore.deepscoreset_id==DeepScoreSet._id,
                                                 DeepScore.index_in_sources==Measurements.index_in_sources ),
                             isouter=True )
                      .filter( Measurements.object_id==meas.object_id )
                      .filter( MeasurementSet.provenance_id==measurement_set.provenance_id )
                      .filter( Measurements._id!=meas.id )
                      .order_by( Image.mjd ) )
                for prvmeas, prvscr, prvimg, prvzp in q.all():
                    alert.prvDiaSources.append( self.dia_source_alert( prvmeas, prvscr, prvimg, zp=prvzp ) )
                    prvimgids.add( prvimg.id )

            # Get all previous nondetections on subtractions of the same provenance.
            #   Note that in the subtraction code that exists right now, we set the
            #   sub image's lim_mag_estimate to be the same as the new image's.  This
            #   implicitly assumes that (a) the ref image is a lot deeper than the new
            #   image, and (b) the ref image has seeing ≤ that of the new image.  If
            #   either of those is wrong, the limiting magnitude estimate will be worse.
            #   We should probably worry about that. --> Issue #364
            # IN ANY EVENT.  For alert purposes, we are going to treat
            #   the filter, mjd, end_mjd, and lim_mag_estimate of the
            #   subtraction images as reliable.
            imgs = Image.find_images( ra=meas.ra, dec=meas.dec,
                                      provenance_ids=sub_image.provenance_id, type='Diff',
                                      order_by='earliest' )

            for img in imgs:
                if img.id not in prvimgids:
                    alert['prvDiaNondetectionLimits'] = { 'MJD': img.mid_mjd,
                                                          'band': img.filter,
                                                          'limitingMag': img.lim_mag_estimate }

            alerts.append( alert )

        return alerts


    def send_kafka_alerts( self, avroalerts, method ):
        # TODO when appropriate : deal with login information etc.

        if avroalerts is None:
            return

        SCLogger.info( f"Sending ≤{len(avroalerts)} alerts to kafka server "
                       f"{method['kafka_server']}, topic {method['topic']}" )

        # TODO : put in a timeout for the server connection to fail, and
        #   test that it does in fact time out rather than hang forever
        #   if it tries to connect to a non-existent server.
        producer = confluent_kafka.Producer( { 'bootstrap.servers': method['kafka_server'],
                                               'batch.size': 131072,
                                               'linger.ms': 50 } )

        nsent = 0
        for alert in avroalerts:
            rb = alert[ 'diaSource' ][ 'rb' ]
            rbmethod = alert[ 'diaSource' ][ 'rbtype' ]
            cut = method['deepcut'] if method['deepcut'] is not None else DeepScoreSet.get_rb_cut( rbmethod )
            if ( rb is not None ) and ( rb >= cut ):
                msgio = io.BytesIO()
                fastavro.write.schemaless_writer( msgio, method['schema'], alert )
                producer.produce( method['topic'], msgio.getvalue() )
                nsent +=1

        producer.flush()
        SCLogger.info( f"Sent {nsent} alerts to kafka server {method['kafka_server']}, topic {method['topic']}" )
        return nsent


    def send_hopskotch_alerts( self, avroalerts, method ):
        # WARNING : this function isn't tested in our tests; we'd have to
        #   spin up a hopskotch server in order to do that, and that
        #   sounds painful.  (Issue #445)
        #
        # In practice, as of this writing, we're using it with LS4 to
        #   send to SCiMMA.  So, for now, make sure in production that
        #   things are hooked up right (esp. auth and such) by sending
        #   to a test topic before actually starting mass production.
        #
        # NOTES ON ACTUALLY SETTING IT UP FOR LS4.
        # (Todo: put this in our documentation somewhere?  Or, better,
        # put the genera procedure in our documentation somewhere, and
        # create a "LS4 pipeline manager documentation" file!)
        #
        # * Log into SCiMMA, and go to https://www.scimma.org/hopauth/
        # * Scroll down, find "Groups".  Find the "ls4" row, and
        #   then click on the "tools" icon in the "Manage Topics" column.
        # * Create the topic; make it public if appropriate.
        # * Put the topic in the "topic" field of the
        #   alert method config, remembering it sould begin with "ls4.".
        # * Go back to https://www.scimma.org/hopauth/
        # * Click on the "Manage" tools icon for the relevant credentials.
        # * Scroll down under "Available Permissions", and find the
        #   permissions for ls4.<topic> for the topic you just created.
        # * Give the credentials both read and write permission
        # * Put the credential name in the "auth_user" field
        #   of the alert method config, and the password (which
        #   you need to have saved when you made the credential)
        #   in the "auth_password" field of the alert method config.

        if avroalerts is None:
            return

        server_url = method['server_url']
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        server_url = f"{server_url}/{method['topic']}"

        SCLogger.info( f"Sending ≤{len(avroalerts)} alerts to hopskotch server {server_url}" )

        auth = hop.auth.Auth( method['auth_user'], method['auth_password'] )
        stream = hop.Stream( auth=auth )
        nsent = 0
        with stream.open( server_url, "w" ) as stream_writer:
            for alert in avroalerts:
                rb = alert[ 'diaSource' ][ 'rb' ]
                rbmethod = alert[ 'diaSource' ][ 'rbtype' ]
                cut = method['deepcut'] if method['deepcut'] is not None else DeepScoreSet.get_rb_cut( rbmethod )
                if ( rb is not None ) and ( rb >= cut ):
                    msgio = io.BytesIO()
                    fastavro.write.schemaless_writer( msgio, method['schema'], alert )
                    stream_writer.write( hop.models.Blob( msgio.getvalue() ) )
                    nsent += 1

        SCLogger.info( f"Sent {nsent} alerts to hopskotch server {server_url}" )
        return nsent
