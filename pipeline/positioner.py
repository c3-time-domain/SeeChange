import datetime
import pytz

import numpy as np

from models.provenance import Provenance
from models.object import Object
from models.base import Psycopg2Connection, SmartSession

class ParsPositioner(Parameters):
    def __init__( self, **kwargs ):
        super().__init__()

        self.datetime = sef.add_par(
            'datetime',
            '1970-01-01 00:00:00',
            datetime.datetime,
            'Only images from times before this will be included in the positioner run.'
        )

        self.sigma_clip = self.add_par(
            'sigma_clip',
            3.,
            float,
            'Do iterative sigma clipping of outliers at this sigma.'
        )

        self.sncut = self.add_par(
            'sncut',
            3.,
            float,
            "Throw out measurements with PSF S/N less than this cut.  Don't make this negative!"
        )
        
        self.filter = self.add_par(
            'filter',
            'i',
            str,
            'Do object positioning on measurements of images in this filter.'
        )

        self.measuring_provenance_id = self.add_par(
            'measuring_provenance_id',
            '',
            str,
            'The ID of the measuring provenance to use for finding measurements to calculate the position.'
        )
        
        self.use_obj_association = self.add_par(
            'use_obj_association',
            False,
            bool,
            ( 'If False (default), ignore pre-existing object associations when finding measurements '
              'for this object, and instead use sources that are within radius of the object\'s current '
              'position (see current_position_provenance_id).' )
        )

        self.current_position_provenance_id = self.add_par(
            'current_position_provenance_id',
            None,
            [ str, None ],
            ( 'If None, then the object\'s current position is assumed to be the ra and dec in the object '
              'table, which is whatever happened to be the position of the first source associated with '
              'this object.  If not None, then find the object in the object_positions table with this '
              'provenance and use that position; if the object is not found, fall back to the position in '
              'the object\'s table if fall_back_object_position is True, otherwise raise an exception.  '
              'Ignored if use_obj_association is True.' )
        )

        self.fall_back_object_position = self.add_par(
            'fall_back_object_position',
            False,
            bool,
            'See doc on current_position_provenance_id'
        )
        
        self.radius = self.add_par(
            'radius',
            2.0,
            float,
            ( 'Radius in arcseconds to identify sources to associate with this object.  This means '
              'that the object centering may well include sources that were not originally associated '
              'with this object!  Ignored if use_obj_association=True' )
        )

        self._enforce_no_new_attrs( True )
        self.override( kwargs )

    def get_process_name( self ):
        'positioning'


class Positioner:
    def __init__( self, **kwargs ):
        self.pars = ParsPositioner( **kwargs )
        # TODO : override from config

    def run( self, object_id, **kwargs ):
        """Run the positioner, updating the database if necessary.

        Parameters
        ----------
          object_id - Object or UUID
            The Object, or the UUID of the object, on which to run the
            positioning.

        Returns
        -------
          ObjectPosition

        """

        self.has_recalculated = False

        if not isinstance( measuring_provenance, Provenance ):
            measuring_provenance = Provenance.get( measuring_provenance )
        if isinstance( object_id, Object ):
            obj = object_id
        else:
            obj = Object.get_by_id( object_id )
        if obj is None:
            raise ValueError( "Unknown object {object_id}" )

        # First, search the database to see if a position for this object
        #   with this provenance already exists.  (As a side effect, load
        #   up the variable prov with the provenance we're working with.
        #   And, in so doing, remind yourself that Python's scoping rules
        #   are perhaps a little cavalier.)
        with SmartSession() as sess:
            # Figure out the provenance we're working with
            prov = Provenance( process = self.get_process_name(),
                               # THIS NEXT ONE WILL NEED TO BE FIXED WITH THE NEW CODE VERSION SYSTEM
                               code_version_id = measuring_provenance.code_version_id,
                               parameters = self.pars.get_critical_parametrs(),
                               upstreams = [ measuring_provenance ] )
            prov.insert_if_needed( session=sess )

            exsiting = ( sess.query( ObjectPosition )
                         .filter( ObjectPosition.object_id==obj._id )
                         .filter( ObjectPosition.provenance_id==prov._id )
                        ).all()
            if len(existing) > 0:
                return existing[0]

        # There's actually (sort of) a race condition here.  The rest of
        #   this code assumes that the object position doesn't already
        #   exist.  It's possible that another process will insert it
        #   while this code is running.  If that happens, at the end
        #   we're going to bump up against a unique constraint
        #   violation, and we'll just shrug and move on (thereby solving
        #   the race condition, which is why I called it sort of above).
            
        # Get the cutoff mjd from the self.pars.datetime parameter.
        #   First, Make sure we have a timezone-aware datetime.  If a timezone isn't
        #   given, we assume UTC.
        dt = self.pars.datetime
        if dt.tzinfo is None:
            dt = pytz.utc.localize( t )
        mjdcut = astropy.time.time( dt, format='datetime' ).mjd

        with Psycopg2Connection() as con:
            cursor = con.cursor()

            # Find all measurements in the appropriate band that go with this object

            if self.pars.use_obj_association:
                # Find all measurements in the current band already associated with the object
                
                q = ( "SELECT m.ra, m.dec, m.flux_psf, m.flux_psf_err FROM measurements m "
                      "INNER JOIN cutouts cu ON ms.cutouts_id=cu._id "
                      "INNER JOIN source_lists s ON c.sources_id=s._id "
                      "INNER JOIN images i ON s.image_id=i._id "
                      "WHERE i.filter=%(filt)s "
                      "  AND m.object_id=%(objid)s" )
                cursor.execute( q, { 'filt': self.pars.filter, 'objid': obj._id } )
                rows = cursor.fetchall()
                srcra = np.array( [ rows[0] for r in rows ] )
                srcdec = np.array( [ rows[1] for r in rows ] )
                srcflux = np.array( [ rows[2] for r in rows ] )
                srcdflux = np.array( [ rows[3] for r in rows ] )

            else:
                # Get the object's current position
                curra = obj.ra
                curdec = obj.dec
                if self.pars.current_position_provenance_id is not None:
                    q = ( "SELECT ra, dec FROM object_positions "
                          "WHERE object_id=%(objid)s AND provenance_id=%(curposprov)s" )
                    cursor.execute( q, { 'objid': obj._id, 'curpposprov': self.pars.current_position_provenance_id } )
                    row = cursor.fetchone()
                    if ( row is None ) and ( not self.pars.fall_back_object_position ):
                        raise RuntimeError( f"Cannot find current position for object {obj._id} with position "
                                            f"provenance {self.pars.current_position_provenance_id}" )
                    else:
                        curra = row[0]
                        curdec = row[1]

                # Find all measurements in the current band within radius of curra, curdec

                q = ( "SELECT m.ra, m.dec, m.flux_psf, m.flux_psf_err FROM measurements m "
                      "INNER JOIN measurement_sets ms ON m.mesurementset_id=ms._id "
                      "INNER JOIN cutouts cu ON ms.cutouts_id=cu._id "
                      "INNER JOIN source_lists s ON c.sources_id=s._id "
                      "INNER JOIN images i ON s.image_id=i._id "
                      "WHERE i.filter=%(filt)s "
                      "  AND i.mjd<=%(mjdcut)s "
                      "  AND ms.provenance_id=%(measprov)s "
                      "  AND q3c_radial_query( m.ra, m.dec, %(ra)s, %(dec)s, %(rad)s ) " )
                cursor.execute( q, { 'filt': self.pars.filter, 'mjdcut': mjdcut,
                                     'measprov': self.pars.measuring_provenance_id,
                                     'ra': curra, 'dec': curdec, 'rad': self.pars.radius/3600. } )
                rows = cursor.fetchall()
                srcra = np.array( [ rows[0] for r in rows ] )
                srcdec = np.array( [ rows[1] for r in rows ] )
                srcflux = np.array( [ rows[2] for r in rows ] )
                srcdflux = np.array( [ rows[3] for r in rows ] )

        # Filter out measurements with S/N < 3
        w = srcflux / srcdflux > self.pars.sncut
        srcra = srcra[ w ]
        srcdec = srcdec[ w ]
        srcflux = srcflux[ w ]
        srcdflux = srcdflux[ w ]
        
        if len( srcra ) == 0:
            raise RuntimeError( f"No matching measurements with S/N>{self.pars.sncut} found for object {obj._id}" )

        if len( srcra ) == 1:
            raise NotImplementedError( "Rob, do the thing." )

        # Sigma cutting : do an unweighted mean position and throw out measurements that are too many σ
        #   away from that mean
        lastpass = len(srcra) + 1
        while len(srcra) < lastpass:
            lastpass = len( srcra )
            meanra = srcra.mean()
            mandec = srcdec.mean()
            sigra = srcra.std()
            sigdec = srcdec.std()
            w = ( ( np.fabs( srcra - meanra ) < self.sigma_clip * sigra ) &
                  ( np.fabs( srdec - meandec ) < self.sigma_clip * sigdec ) )
            srcra = srcra[ w ]
            srcdec = srcdec[ w ]
            srcflux = srcflux[ w ]
            srcdflux = srcdflux[ w ]
            if len(srcra) == 0:
                # ... this may not be formally possible unless somebody sets sigma_clip to 0 or negative.
                raise RuntimeError( f"For object {obj._id}, noting passed the sigma clipping!" )
            if len(srcra) == 1:
                # ... I think this formally possible if somebody sets
                #   sigma_clip low enough (like 1 or 2), but hopefully
                #   nobody will set it that absurdly low.
                raise RuntimeError( f"For object {obj._id}, sigma clipping reduced things to a single measurement!" )
            

        # Do a S/N weighted mean of the things that passed the sigma cutting.
        # (Is S/N what we want?  Or should we do (S/N)² in analogy to doing vartiance-weighted stuff?)
        weights = srcflux / srcdflux
        weightsum = weights.sum()
        meanra = ( weights * srcra ).sum() / weightsum
        meandec = ( weights * srcdec ).sum() / weightsum
        ravar = ( weights * ( srcra - meanra )**2 ).sum() / weightsum
        decvar = ( weights * ( srcdec - meandec )**2 ).sum() / weightsum
        covar = ( weights * ( srcra - meanra ) * (srcdec - meandec ) ) / weightsum

        objpos = ObjectPosition( object_id=obj._id,
                                 provenance_id=prov._id,
                                 ra=meanra,
                                 dec=meandec,
                                 dra=np.sqrt(ravar),
                                 ddec=np.sqrt(decvar),
                                 ra_dec_cov=covar )
        objpos.calculate_coordinates()
        objpos.insert()
        # TODO, catch existing error

        return objpos
