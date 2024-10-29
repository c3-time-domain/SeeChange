import io
import uuid
import random
import datetime

import fastavro
import confluent_kafka

import numpy as np

from models.base import SmartSession
from models.image import Image, image_upstreams_association_table
from models.source_list import SourceList
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.object import Object
from util.config import Config


# Alerting doesn't work with the Parameters system because there's no Provenance associated with it,
#  so there's no need to get critical parameters.  It will just read the config directly.

class Alerting:

    known_methods = [ 'kafka' ]

    def __init__( self, send_alerts=None, methods=None ):
        """Initialize an Alerting object.

        Parameters
        ----------
          send_alerts: bool, default None
             Set to False to globally disable alert sending, or True to
             send alerts based on the methods list and the 'enabled'
             element for each method.  If None, will read
             alerts.send_alerts from config.

          methods: list of dicts, default None
             Configuration of alert sending methods.  If None, will read
             alerts.methods from config.

        """
        cfg = Config.get()
        self.send_alerts = cfg.value( 'alerts.send_alerts' ) if send_alerts is None else send_alerts
        self.methods = cfg.value( 'alerts.methods' ) if methods is None else methods

        if ( not isinstance( self.methods, list ) ) or ( not all( isinstance(m, dict) for m in self.methods ) ):
            raise TypeError( "Alerting: methods must be a list of dicts." )

        if not len( set( m['name'] for m in self.methods ) ) == len( self.methods ):
            raise ValueError( "Alerting: all alert methods must have unique names" )
        
        for method in self.methods:
            if 'method' not in method:
                raise ValueError( f"'method' is not defined for (at least) alert method {method['name']}" )

            if method['method'] not in self.known_methods:
                raise ValueError( f"Unknown alert method {method['method']}; known are {self.known_methods}" )
            
            if 'enabled' not in method:
                raise ValueError( f"Each method must have a value for 'enabled'; (at least) "
                                  f"{method['name']} does not." )

            if method['method'] == 'kafka':
                method['schema'] = fastavro.schema.load_schema( method['avro_schema'] )

                now = datetime.datetime.now()
                barf = "".join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=6 ) )
                method['topic'] = method['kafka_topic_pattern'].format( year = now.year,
                                                                        month = now.month,
                                                                        day = now.day,
                                                                        barf = barf )



    def send( self, ds ):
        """Send alerts from a fully processed DataStore.

        The DataStore must have run all the way through scoring.

        Parmameters
        -----------
          ds: DataStore

        """
        if not self.send_alerts:
            return

        # TODO: verify that datastore has measurements and scores

        # If any of the methods are kafka, we will need to build avro alerts
        avroalerts = None
        if any( method['enabled'] and ( method['method'] == 'kafka' ) for method in self.methods ):
            avroalerts = self.build_avro_alert_structures( ds )

        # Issue alerts for all methods
        for method in self.methods:
            if not method['enabled']:
                continue

            if method['method'] == 'kafka':
                self.send_kafka_alerts( avroalerts, method )

            else:
                raise RuntimeError( "This should never happen." )
        

    def dia_source_alert( self, meas, img, aperdex=None, fluxscale=None ):
        # For snr, we're going to assume that the detection was approximately
        #   detection in a 1-FWHM aperture.  This isn't really right, but
        #   it should be approximately right.
        #
        # Don't do background subtractions for the fluxes here, because they're
        #   off of difference images.  (Note that the "flux" property of
        #   measurements *does* do background subtraction, but flux_psf and
        #   flux_apertures does not.)
            
        if aperdex is None:
            radfwhm = meas.aper_radii / ( img.fwhm_estimate / img.instrument_object.pixel_scale )
            w = np.where( np.fabs( radfwhm - 1. ) < 0.01 )
            if len(w) == 0.:
                raise RuntimeError( f"No 1FWHM aperture (have {radfwhm})" )
            aperdex = w[0]
        if fluxscale is None:
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
                 'snr': meas.flux_apertures[aperdex] / meas.flux_apertures_err[aperdex],
                 'psfFlux': meas.flux_psf * fluxscale,
                 'psfFluxErr':meas.flux_psf_err * fluxscale,
                 'ixx': None,
                 'ixxErr': None,
                 'iyy': None,
                 'iyyErr': None,
                 'ixy': None,
                 'ixyErr': None,
                 'ixx_iyy_Cov': None,
                 'ixx_ixy_Cov': None,
                 'iyy_iyy_Cov': None,
                 'ixxPSF': None,
                 'iyyPSF': None,
                 'ixyPSF': None
                }        

    def dia_object_alert( self, obj ):
        return { 'diaObjectId': str( obj.id ),
                 'name': obj.name,
                 'ra': obj.ra,
                 'raErr': None,
                 'dec': obj.dec,
                 'decErr': None,
                 'ra_dec_Cov': None,
                 'nearbyObj1': None,
                 'nearbyObj1Dist': None,
                 'nearbyObj2': None,
                 'nearbyObj2Dist': None,
                 'nearbyObj3': None,
                 'nearbyObj3Dist': None,
                }
                 
    
    def build_avro_alert_structures( self, ds ):
        sub_image = ds.get_subtraction()
        image = ds.get_image()
        zp = ds.get_zp()
        detections = ds.get_detections()
        measurements = ds.get_measurements()
        cutouts = ds.get_cutouts()
        cutouts.load_all_co_data( sources=detections )
        scores = ds.get_scores()

        if len(scores) == 0:
            # Nothing to do!
            return

        if len(scores) != len(measurements):
            raise ValueError( f"Alert sending error: DataStore has different number of "
                              f"measurements ({len(measurements)}) and scores ({len(scores)}).  "
                              f"This should never happen." )

        # Figure out which aperture is radius of 1 FWHM
        radfwhm = measurements[0].aper_radii / ( image.fwhm_estimate / image.instrument_object.pixel_scale )
        w = np.where( np.fabs( radfwhm - 1. ) < 0.01 )
        if len(w[0]) == 0.:
            raise RuntimeError( f"No 1FWHM aperture (have {radfwhm})" )
        aperdex = w[0][0]

        # We're going to use the standard SNANA zeropoint for no adequately explained reason
        # (We *could* just store the image zeropoint in fluxZeroPoint.  However, it will be
        # more convenient for people of all of the flux values from all of the sources, previous
        # sources, and previous forced sources are on the same scale.  We have to pick something,
        # and SNANA is something.  So, there's an explanation; I don't know if it's an
        # adequate explanation.)
        fluxscale = 10 ** ( ( zp.zp - 27.5 ) / -2.5 )

        alerts = []
        
        for meas, scr in zip( measurements, scores ):
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

            alert['diaSource'] = self.dia_source_alert( meas, image, aperdex=aperdex, fluxscale=fluxscale )
            alert['diaObject'] = self.dia_object_alert( Object.get_by_id( meas.object_id ) )

            # In Image.from_new_and_ref, we set a lot of the sub image's properties (crucially,
            #   filter and mjd) to be the same as the new image.  So, for what we need for
            #   alerts, we can just use the sub image.
            
            with SmartSession() as sess:
                # Get all previous sources with the same provenance.  Need to
                # join to Image (we'll be joining to the sub image,
                # because the cutouts source list is the detections, and
                # the detections image is the sub image) to get
                # mjd and filter.

                # TODO -- handle previous_sources_days
                 
                prvimgids = {}
                q = ( sess.query( Measurements, Image )
                      .join( Cutouts, Measurements.cutouts_id==Cutouts._id )
                      .join( SourceList, Cutouts.sources_id==SourceList._id )
                      .join( Image, SourceList.image_id==Image._id )
                      .filter( Measurements.object_id==meas.object_id )
                      .filter( Measurements.provenance_id==meas.provenance_id )
                      .filter( Measurements._id!=meas.id )
                      .order_by( Image.mjd ) )
                for pvrmeas, prvimg in q.all():
                    alert.prvDiaSources.append( self.dia_source_alert( prvmeas, prvimg ) )
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

        # TODO : put in a timeout for the server connection to fail, and
        #   test that it does in fact time out rather than hang forever
        #   if it tries to connect to a non-existent server.
        producer = confluent_kafka.Producer( { 'bootstrap.servers': method['kafka_server'],
                                               'batch.size': 131072,
                                               'linger.ms': 50 } )

        # TODO : filter on scoring
        
        for alert in avroalerts:
            msgio = io.BytesIO()
            fastavro.write.schemaless_writer( msgio, method['schema'], alert )
            producer.produce( method['topic'], msgio.getvalue() )

        producer.flush()
