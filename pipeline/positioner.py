import datetime
import pytz

import numpy as np
import astropy.time
import psycopg2.errors

from models.provenance import Provenance
from models.object import Object, ObjectPosition
from models.base import Psycopg2Connection, SmartSession
from pipeline.parameters import Parameters

from util.logger import SCLogger


class ParsPositioner(Parameters):
    def __init__( self, **kwargs ):
        super().__init__()

        self.datetime = self.add_par(
            'datetime',
            '1970-01-01 00:00:00',
            str,
            ( 'Only images from times before this will be included in the positioner run. '
              'Must be in ISO 8601 format (with a space in place of the T allowed).' )
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

        self._enforce_no_new_attrs = True
        self.override( kwargs )

    def get_process_name( self ):
        return 'positioning'


class Positioner:
    def __init__( self, **kwargs ):
        self.pars = ParsPositioner( **kwargs )
        # TODO : override from config (Issue #475)

    def run( self, object_id, **kwargs ):
        """Run the positioner, adding a row to the database if necessary.

        If an ObjectPosition for this object with the right provenance
        alrady exists, just return that.  Otherwise, calculate the
        position, save it to the database, and return it.

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

        measuring_provenance = Provenance.get( self.pars.measuring_provenance_id )
        obj = object_id if isinstance( object_id, Object ) else Object.get_by_id( object_id )
        if obj is None:
            raise ValueError( "Unknown object {object_id}" )

        # First, search the database to see if a position for this object
        #   with this provenance already exists.  (As a side effect, load
        #   up the variable prov with the provenance we're working with.
        #   And, in so doing, remind yourself that Python's scoping rules
        #   are perhaps a little cavalier.)
        with SmartSession() as sess:
            # Figure out the provenance we're working with
            prov = Provenance( process = self.pars.get_process_name(),
                               code_version_id = Provenance.get_code_version(self.pars.get_process_name()).id,
                               parameters = self.pars.get_critical_pars(),
                               upstreams = [ measuring_provenance ] )
            prov.insert_if_needed( session=sess )

            existing = ( sess.query( ObjectPosition )
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

        self.has_recalculated = True

        # Get the cutoff mjd from the self.pars.datetime parameter.
        #   First, Make sure we have a timezone-aware datetime.  If a timezone isn't
        #   given, we assume UTC.
        dt = datetime.datetime.fromisoformat( self.pars.datetime )
        if dt.tzinfo is None:
            dt = pytz.utc.localize( dt )
        mjdcut = astropy.time.Time( dt, format='datetime' ).mjd

        with Psycopg2Connection() as con:
            cursor = con.cursor()

            # Find all measurements in the appropriate band that go with this object

            if self.pars.use_obj_association:
                # Find all measurements in the current band already associated with the object

                q = ( "SELECT m.ra, m.dec, m.flux_psf, m.flux_psf_err FROM measurements m "
                      "INNER JOIN cutouts c ON ms.cutouts_id=c._id "
                      "INNER JOIN source_lists s ON c.sources_id=s._id "
                      "INNER JOIN images i ON s.image_id=i._id "
                      "WHERE i.filter=%(filt)s "
                      "  AND m.object_id=%(objid)s" )
                cursor.execute( q, { 'filt': self.pars.filter, 'objid': obj._id } )
                rows = cursor.fetchall()
                srcra = np.array( [ r[0] for r in rows ] )
                srcdec = np.array( [ r[1] for r in rows ] )
                srcflux = np.array( [ r[2] for r in rows ] )
                srcdflux = np.array( [ r[3] for r in rows ] )

            else:
                # Get the object's current position
                curra = obj.ra
                curdec = obj.dec
                if self.pars.current_position_provenance_id is not None:
                    q = ( "SELECT ra, dec FROM object_positions "
                          "WHERE object_id=%(objid)s AND provenance_id=%(curposprov)s" )
                    cursor.execute( q, { 'objid': obj._id, 'curposprov': self.pars.current_position_provenance_id } )
                    row = cursor.fetchone()
                    if row is None:
                        if not self.pars.fall_back_object_position:
                            raise RuntimeError( f"Cannot find current position for object {obj._id} with position "
                                                f"provenance {self.pars.current_position_provenance_id}" )
                    else:
                        curra = row[0]
                        curdec = row[1]

                # Find all measurements in the current band within radius of curra, curdec
                q = ( "SELECT m.ra, m.dec, m.flux_psf, m.flux_psf_err FROM measurements m "
                      "INNER JOIN measurement_sets ms ON m.measurementset_id=ms._id "
                      "INNER JOIN cutouts c ON ms.cutouts_id=c._id "
                      "INNER JOIN source_lists s ON c.sources_id=s._id "
                      "INNER JOIN images i ON s.image_id=i._id "
                      "WHERE i.filter=%(filt)s "
                      "  AND i.mjd<=%(mjdcut)s "
                      "  AND ms.provenance_id=%(measprov)s "
                      "  AND q3c_radial_query( m.ra, m.dec, %(ra)s, %(dec)s, %(rad)s ) " )
                subdict = { 'filt': self.pars.filter, 'mjdcut': mjdcut,
                            'measprov': self.pars.measuring_provenance_id,
                            'ra': curra, 'dec': curdec, 'rad': self.pars.radius/3600. }
                SCLogger.debug( f"Running query: {cursor.mogrify(q, subdict)}" )
                cursor.execute( q, subdict )
                rows = cursor.fetchall()
                srcra = np.array( [ r[0] for r in rows ] )
                srcdec = np.array( [ r[1] for r in rows ] )
                srcflux = np.array( [ r[2] for r in rows ] )
                srcdflux = np.array( [ r[3] for r in rows ] )

        nmatch = len(srcra )

        # Filter out measurements with S/N < 3
        w = srcflux / srcdflux > self.pars.sncut
        srcra = srcra[ w ]
        srcdec = srcdec[ w ]
        srcflux = srcflux[ w ]
        srcdflux = srcdflux[ w ]
        nhighsn = len(srcra)

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
            meandec = srcdec.mean()
            sigra = srcra.std()
            sigdec = srcdec.std()
            w = ( ( np.fabs( srcra - meanra ) < self.pars.sigma_clip * sigra ) &
                  ( np.fabs( srcdec - meandec ) < self.pars.sigma_clip * sigdec ) )
            srcra = srcra[ w ]
            srcdec = srcdec[ w ]
            srcflux = srcflux[ w ]
            srcdflux = srcdflux[ w ]
            if len(srcra) == 0:
                # This should only happen if somebody set sigma_clip to something absurdly small.
                # (Or if, somehow, the measurements saved to the database all came out exactly the same,
                # so the stdev is 0.)
                raise RuntimeError( f"For object {obj._id}, nothing passed the sigma clipping!" )
            if len(srcra) == 1:
                # ... I think this formally possible if somebody sets
                #   sigma_clip low enough (like 1 or 2), but hopefully
                #   nobody will set it that absurdly low.
                raise RuntimeError( f"For object {obj._id}, sigma clipping reduced things to a single measurement!" )

        SCLogger.debug( f"{len(srcra)} passed the sigma clipping out of {nhighsn} S/N>3 sources, "
                        f"out of {nmatch} sources close enough to the previous position." )


        # Do a S/N weighted mean of the things that passed the sigma cutting.
        # (Is S/N what we want?  Or should we do (S/N)² in analogy to doing vartiance-weighted stuff?)
        weights = srcflux / srcdflux
        weightsum = weights.sum()
        meanra = ( weights * srcra ).sum() / weightsum
        meandec = ( weights * srcdec ).sum() / weightsum
        ravar = ( weights**2 * ( srcra - meanra )**2 ).sum() / ( weightsum**2 )
        decvar = ( weights**2 * ( srcdec - meandec )**2 ).sum() / ( weightsum**2 )
        covar = ( weights**2 * ( srcra - meanra ) * (srcdec - meandec ) ).sum() / ( weightsum**2 )

        objpos = ObjectPosition( object_id=obj._id,
                                 provenance_id=prov._id,
                                 ra=meanra,
                                 dec=meandec,
                                 dra=np.sqrt(ravar),
                                 ddec=np.sqrt(decvar),
                                 ra_dec_cov=covar )
        objpos.calculate_coordinates()

        try:
            objpos.insert()
        except psycopg2.errors.UniqueViolation():
            # This means that somebody else calculated and saved this
            # ObjectPosition between back when we made sure it didn't
            # exist and now.  In that case, all is well, we just wasted
            # a bit of effort.  Pull down the existing object and
            # return that
            existing = ( sess.query( ObjectPosition )
                         .filter( ObjectPosition.object_id==obj._id )
                         .filter( ObjectPosition.provenance_id==prov._id )
                        ).all()
            if len(existing) > 0:
                return existing[0]
            else:
                raise RuntimeError( "This should never happen." )

        return objpos
