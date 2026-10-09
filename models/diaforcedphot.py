import numbers
import textwrap

import numpy as np
import psycopg.sql as sql

import sqlalchemy as sa
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declared_attr

from util.logger import SCLogger
from util.util import parse_dateobs, listify, asUUID

from improc.tools import make_cutouts

from models.base import PGDB, Base, SeeChangeBase, UUIDMixin, HasBitFlagBadness
from models.provenance import Provenance
import models.object # OMG why did we name one of our classes Object????
from models.object import ObjectPosition
from models.image import Image
from models.background import Background
from models.enums_and_bitflags import measurements_badness_inverse


class DiaForcedPhot( Base, UUIDMixin, HasBitFlagBadness ):
    __tablename__ = 'dia_forced_photometry'

    @declared_attr
    def __table_args__( cls ):    # noqa: N805
        return (
            UniqueConstraint( 'object_id', 'subtraction_id', 'provenance_id', name='dia_forcedphot_unique' ),
        )

    object_id = sa.Column(
        sa.ForeignKey( 'objects._id', ondelete='RESTRICT', name='dia_forcedphot_object_id_fkey' ),
        nullable = False,
        index = True,
        doc = "ID of the object this is forced photometry for"
    )

    object_position_id = sa.Column(
        sa.ForeignKey( 'object_positions._id', ondelete='RESTRICT', name='dia_forcedphot_object_position_id_fkey' ),
        nullable = True,
        index = True,
        doc = "ID (if any) of the object position used for this forced photometry."
    )

    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances._id', ondelete='CASCADE', name='dia_forcedphot_provenance_id_fkey' ),
        nullable = False,
        index = True,
        doc = ( "ID of the Provenance of this forced photometry point" )
    )

    subtraction_id = sa.Column(
        sa.ForeignKey( 'images._id', ondelete='RESTRICT', name='dia_forcedphot_subtraction_id_fkey' ),
        nullable = False,
        index = True,
        doc = ( "ID of the subtraction this forced phot was performed on." )
    )

    # Note: the NaN default here is just because I had an existing database to which
    #   these columns were going to be added, so I didn't want the not null to kill
    #   it.  Yeah, I will need to manually go back and fix those columns later (if
    #   I decide I care).
    x = sa.Column(
        sa.REAL,
        nullable = False,
        index = False,
        default = np.nan,
        server_default = sa.sql.elements.TextClause( "'NaN'" ),
        doc = ( "x position (0-indexed, n.0=center of pixel) on image given by subtraction_id" )
    )

    y = sa.Column(
        sa.REAL,
        nullable = False,
        index = False,
        default = np.nan,
        server_default = sa.sql.elements.TextClause( "'NaN'" ),
        doc = ( "y position (0-indexed, n.0=center of pixel) on image given by subtraction_id" )
    )

    flux_psf = sa.Column(
        sa.REAL,
        nullable = False,
        index = False,
        doc = ( "PSF flux, in dn.  Need to use the zeropoint to turn this into something standard. "
                "WARNING: right now for zogy we don't use the right PSF for phtometry!" )
    )

    flux_psf_err = sa.Column(
        sa.REAL,
        nullable = False,
        index = False,
        doc = ( "Uncertainty on psf_flux." ),
    )

    flux_apertures = sa.Column(
        ARRAY( sa.REAL, zero_indexes=True ),
        nullable = False,
        index = False,
        doc = ( "Aperture fluxes, in dn.  Does not include aperture corrections.  They are in the apertures "
                "defined in the zero_point record.  WARNING: these aperture correctoins are not right for zogy." )
    )

    flux_apertures_err = sa.Column(
        ARRAY( sa.REAL, zero_indexes=True ),
        nullable = False,
        index = False,
        doc = ( "Uncertainties on flux_apertures." )
    )

    def __init__( self, *args, **kwargs ):
        SeeChangeBase.__init__( self )
        HasBitFlagBadness.__init__( self )
        self.set_attributes_from_dict( kwargs )

    def _get_inverse_badness( self ):
        return measurements_badness_inverse

    def get_upstream_ids( self, pgdb=None ):
        upstrs = [ ( models.object.Object, self.object_id ),
                   ( Image, self.subtraction_id ) ]
        if self.object_position_id is not None:
            upstrs.append( [ ( ObjectPosition, self.object_position_id ) ] )
        return upstrs

    def get_downstream_ids( self, pgdb=None ):
        return []


    def get_clips( self, clip_size=None, bgsub=True, scale_ref=True, pgdb=None ):
        """Get 3d numpy array that are new, warped ref, sub cutouts for this forced phot point.

        Parameters
        ----------
          clip_size : int
             The size of the clip.  Will be rounded up to the next odd
             number.  If not given, will use 10 times the seeing FWHM of
             the new.

          bgsub : bool, default True
             Subtract backgrounds from new and ref?

          scale_ref : bool, default True
             The new and sub already have the same zeropoint.  If this
             is True, scale the warped ref to have the same zeropoint,
             so you can compare directly.

          pgdb : PGDB or psycopg.Connection
             Database connection.  If not given, will open and close a
             new one.

        Returns
        -------
          np.array with shape (3, clip_size, clip_size) and dtype np.float32

              clips[0, :, :] is new
              clips[1, :, :] is ref (all NaN if warped ref not found)
              clips[2, :, :] is sub

            If the clip goes off of the edge of the image, it will be
            padded with NaN.

            (With the caveat that clip_size will be rounded up to the
            next odd integer.)

        """

        with PGDB( pgdb, dictcursor=True ) as pgdb:
            # LEFT JOIN for the warped ref because right now
            #   it might not exist.
            # Need to refactor so that alignment is a legitimate
            #   upstream of subtraction.
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT isc.image_id AS subid, i._id AS newid, ri._id AS warprefid,
                       b._id AS newbgid, rb._id AS refbgid,
                       z.zp AS newzp, p.fwhm_pixels, rz.zp AS refzp
                FROM image_subtraction_components isc
                INNER JOIN zero_points z ON z._id=isc.new_zp_id
                INNER JOIN world_coordinates w ON w._id=z.wcs_id
                INNER JOIN source_lists s ON s._id=w.sources_id
                INNER JOIN backgrounds b ON s._id=b.sources_id
                INNER JOIN psfs p ON s._id=p.sources_id
                INNER JOIN images i ON i._id=s.image_id
                INNER JOIN refs r ON isc.ref_id=r._id
                INNER JOIN zero_points rz ON r.zp_id=rz._id
                LEFT JOIN source_lists rs ON isc.warped_ref_sources_id=rs._id
                LEFT JOIN images ri ON rs.image_id=ri._id
                LEFT JOIN backgrounds rb ON rb.sources_id=rs._id
                WHERE isc.image_id={subid}
                """
            ) ).format( subid=self.subtraction_id )
            rows = pgdb.excute( q )
            if len(rows) == 0:
                raise RuntimeError( f"Failed to find images for dia_forced_phot {self.id}; "
                                    f"this shouldn't happen." )
            elif len(rows) > 1:
                raise RuntimeError( f"Found multiple subs/news for dia_forced_phot {self.id}; "
                                    "this shouldn't happen." )
            subim = Image.get_by_id( rows[0]['subid'], pgdb=pgdb )
            # No need to background subtract a subtraction!
            subdata = subim.data
            newim = Image.get_by_id( rows[0]['newid'], pgdb=pgdb )
            if bgsub:
                newbg = Background.get_by_id( rows[0]['newbgid'], pgdb=pgdb )
                newdata = newbg.subtract_me( newim.data )
            else:
                newdata = newim.data
            newzp = rows[0]['zp']
            newseeing = rows[0]['fwhm_pixels']
            if rows[0]['warprefid'] is None:
                SCLogger.warning( f"Failed to find warped ref for subtraction {self.subtracttion_id}" )
                warprefdata = None
                warprefzp = None
            else:
                warpref = Image.get_by_id( rows[0]['warprefid'], pgdb=pgdb )
                if bgsub:
                    warprefbg = Background.get_by_id( rows[0]['refbgid'], pgdb=pgdb )
                    warprefdata = warprefbg.subtract_me( warpref.data )
                else:
                    warprefdata = warpref.data
                warprefzp = rows[0]['refzp']

        # Figure out the clip size
        if clip_size is None:
            clip_size = 10. * newseeing
        if not isinstance( clip_size, numbers.Integral ):
            clip_size = int( clip_size + 0.5 )
        clip_size += 1 if (clip_size % 2 == 0) else 0

        # Do
        ix = int( np.round( self.x ) )
        iy = int( np.round( self.y ) )
        clips = np.empty( (3, clip_size, clip_size), dtype=np.float32 )
        clips[0] = make_cutouts( newdata, ix, iy, size=clip_size, fillvalue=np.nan, dtype=np.float32 )[0]
        clips[2] = make_cutouts( subdata, ix, iy, size=clip_size, fillvalue=np.nan, dtype=np.float32 )[0]
        if warprefdata is not None:
            clips[1] = make_cutouts( warprefdata, ix, iy, size=clip_size, fillvalue=np.nan, dtype=np.float32 )[0]
            if scale_ref:
                clips[1] *= 10 ** ( ( newzp - warprefzp) / 2.5 )
        else:
            clips[1] = np.nan

        return clips

    @classmethod
    def get_lightcurve_for_object( cls, obj, prov, filters=None, include_apertures=False,
                                   include_subim_ids=False, include_pos=False, diaforcedphotobjs=False,
                                   return_format="listofdicts", pgdb=None ):
        """Return all dia forced photometry for a given object in a given provenance.

        Parmaeters
        ----------
          obj : Object, uuid, or str
            An Object object, the id of an object, or the name of an object

          prov : Provenance or str
            The diaforcedphot provenance for the lightcurve

          filters : str or list of str, default None
            If not None, only include data from these filters.  By
            default, all dia forced phot for this object will be returned.

          include_apertures : bool, default False
            Normally, just returns psf photometry.  IF true, also return aperture photometry.
            This will be a list, for multiple apertures.  WARNING : I don't think aperture
            corrections are done, so the meaning of this is fraught.

          include_subim_ids : bool, default False
            If True, there will be an additional colum subim_id with the
            id of the subtraction image each forced photometry point was
            performed on.

          include_pos : bool, default False
            If True, there will be two additional columns x,y giving the
            position used for the object for this point.

          diaforcedhotobjs : bool, default False
            Changes what's returned; see Results below.

          return_format : str, default listofdicts
            Either listofdicts or dict of lists.

          pgdb : PGDB, psycopg.Connection, or psycopg.Curosr, default None
            Database connection.  If not given, a new connection will be
            opened and closed.

        Returns
        -------
          Either a dict of lists, or a list of dicts.  *If*
          diaforcedphotobjs was False, you could feed this into
          pandas.DataFrame, for instance.

          If a dict of lists, the columns are the keys and each list has
          the same number rows (the number of lightcurve points).

          If a list of dicts, the length of the list is the number of
          lightcurve points, and each element of the list is a
          dictionary of column: value.

          By default, the keys are the column names:
            mjd
            filter
            flux_psf
            flux_psf_err
            [ flux_apertures - ** ]
            [ flux_apertures_err - ** ]
            [ subimid -- uuid, id of the subtraction image, only present if include_subim_id is True ]
            [ x -- position on subtraction image, only if include_pos is True ]
            [ y -- position on subtraction image, only if include_pos is True ]

          If diaforcedphots is True, then the colums are:
            mjd
            filter
            diaforcedphot - a DiaForcedPhot object (or list thereof)

        """

        filters = listify( filters )
        if len(filters) == 1:
            filtwhere = sql.SQL( "  AND i.filter={filt}" ).format( filt=filters[0] )
        elif len(filters) > 1:
            filtwhere = sql.SQL( "  AND i.filter=ANY(ARRAY[{filts}])" ).format( filts=sql.SQL(",").join(filters) )
        else:
            filtwhere = sql.SQL( "" )

        with PGDB( pgdb ) as pgdb:
            prov = Provenance.get( prov, must_exist=True, pgdb=pgdb )
            if not isinstance( obj, models.object.Object ):
                try:
                    obj = asUUID( obj )
                    objobj = models.object.Object.get_by_id( obj, pgdb=pgdb )
                except Exception:
                    objobj = models.object.Object.get_by_field_value( 'name', obj )
                    objobj = None if len(objobj) == 0 else objobj[0]
                if objobj is None:
                    raise RuntimeError( f"Failed to find object {obj}" )
                obj = objobj

            if diaforcedphotobjs:
                photcols = sql.SQL( "p.*" )
            else:
                photcols = sql.SQL( "p.flux_psf, p.flux_psf_err{apcols}{subimcols}{poscols}"
                                    ).format( apcols=sql.SQL( ", flux_apertures, flux_apertures_err"
                                                              if include_apertures else "" ),
                                              subimidcol=sql.SQL( ", i._id AS subimid" if include_subim_ids else "" ),
                                              poscols=sql.SQL( ", x, y" if include_pos else "" ) )

            q = sql.SQL( textwrap.dedent(
                """\
                SELECT i.mjd, i.filter, {photcols}
                FROM dia_forced_photometry p
                INNER JOIN images i ON p.subtraction_id=i._id
                WHERE p.object_id={objid} AND p.provenance_id={provid}
                {filtwhere}
                ORDER BY i.mjd
                """ ) ).format( objid=obj.id, provid=prov.id, photcols=photcols, filtwhere=filtwhere )

            rows, cols = pgdb.execute( q )

        coldex = { c: i for i, c in enumerate(cols) }
        if diaforcedphotobjs:
            # I set create_uuidify=False below because I know given how I did it above
            #   that I already have UUIDs (or None).  May as well save all that
            #   function calling overhead.
            if return_format == 'listofdicts':
                rval = []
                for row in rows:
                    rows.append( { 'mjd': row[coldex['mjd']],
                                   'filter': row[coldex['filter']] } )
                    mess = { c: row[coldex[c]] for c in cols if c not in ( 'mjd', 'filter' ) }
                    rval[-1]['diaforcedphot'] = DiaForcedPhot.create( create_uuidify=False, **mess )
            else:
                rval = { 'mjd': [ row[coldex['mjd']] for row in rows ],
                         'filter': [ row[coldex['filter']] for row in rows ],
                         'diaforcedphot': []
                        }
                for row in rows:
                    mess = { c: row[coldex[c]] for c in cols if c not in ( 'mjd', 'filter' ) }
                    rval[-1]['diaforcedphot'].append( DiaForcedPhot.create( create_uuidify=False, **mess ) )

        else:
            if return_format == 'listofdicts':
                # I bet there's some python tangle just like the one in else to accomplish this next row
                rval = [ { c: row[coldex[c]] for c in cols } for row in rows ]
            else:
                rval = dict( zip( cols, map(list, zip(*rows)) ) )

        return rval


    @classmethod
    def find_objects_with_diaforcedphot( cls, prov, include_filters=False, pgdb=None, **kwargs ):

        """Return object names, ids, and counts of dia_forced_photometry.

        Parameters
        ----------
          prov : Provenance or str
            The forced photometry provenance to search.

          include_filters : bool, default False
            If True, then there will be (potentially) multiple entries
            in the return for each object, as the counts will be
            separated out by filters.  If False, then counts will be
            combining all filters.

          pgdb: PGDB or psycopg.Connection, default None
            Database connection.  If not given, a new one will be opened
            and closed.

        Returns
        -------
          dict of column: list

          Each value will be a list of the same length; this is
          something you could (for instance) feed directly into
          pandas.DataFrame().

          columns will be:
            id : object id (a uuid)
            name : object name
            ra : object ra
            dec : object dec
            numphot : number of forced photometry points within the specified limits
            [ filter : filter; only if include_filters is True ]

          If include_filters is True, then numphot is the number in that
          filter for that object; otherwise, it combines all filters.

        """

        # Validate and parse arguments

        ra = None
        if ( 'ra' in kwargs ) or ( 'dec' in kwargs ):
            if not ( ('ra' in kwargs) and ('dec' in kwargs) ):
                raise ValueError( "Must pass both or neither of ra and dec, not just one." )
            if any( k in kwargs for k in [ 'minra', 'maxra', 'mindec', 'maxdec' ] ):
                raise ValueError( "Can't pass any of min/max ra/dec with (ra, dec, [radius])" )

            ra = kwargs['ra']
            dec = kwargs['dec']
            del kwargs['ra']
            del kwargs['dec']
            if 'radius' in kwargs:
                radius = kwargs['radius'] / 3600.
                del kwargs['radius']
            else:
                radius = 1. / 3600.

        coordlim = { f"{w}{c}": None for w in ( 'min', 'max' ) for c in ( 'ra', 'dec' ) }
        for kw in coordlim.keys():
            if kw in kwargs:
                coordlim[kw] = kwargs[kw]
                del kwargs[kw]
        minmjd = None
        maxmjd = None
        if 'minmjd' in kwargs:
            minmjd = parse_dateobs( kwargs['minmjd'] )
            del kwargs['minmjd']
        if 'maxmjd' in kwargs:
            maxmjd = parse_dateobs( kwargs['maxmjd'] )

        if len(kwargs) > 0:
            raise ValueError( f"Unknown arguments: {set(kwargs.keys())}" )

        # Do things

        with PGDB( pgdb ) as pgdb:
            prov = Provenance.get( prov, pgdb=pgdb )
            posprov = None
            if prov.parameters['object_position_prov'] is not None:
                posprov = Provenance.get( prov.parameters['object_position_prov'], pgdb=pgdb )
                if posprov is None:
                    raise RuntimeError( f"Failed to find position provenance "
                                        f"{prov.parameters['object_position_prov']}" )

            # Build the SQL conditionals we'll use

            if posprov is None:
                postab = sql.Identifier( "o" )
                posjoin = sql.SQL( "" )
            else:
                postab = sql.Identifier( "p" )
                posjoin = sql.SQL( textwrap.dedent(
                    """
                    INNER JOIN object_positions p ON p.object_id=o.id
                                                 AND p.provenance_id={posprov}
                    """
                ) ).format( posprov=posprov.id )

            where = "WHERE"
            mjdwhere = sql.SQL("")
            if minmjd is not None:
                mjdwhere += sql.SQL( "{where} i.mjd>={mjd}\n" ).format( mjd=minmjd )
                where = "  AND"
            if maxmjd is not None:
                mjdwhere += sql.SQL( "{where} i.mjd<={mjd}\n" ).format( mjd=maxmjd )

            poswhere = sql.SQL("")
            if ra is not None:
                poswhere += sql.SQL( "WHERE q3c_radial_query({postab}.ra, {postab}.dec, {ra}, {dec}, {radius})\n"
                                    ).format( postab=postab, ra=ra, dec=dec, radius=radius )
            else:
                where = "WHERE"
                for minmax in ( 'min', 'max' ):
                    for coord in ( 'ra', 'dec' ):
                        if coordlim[ f'{minmax}{coord}' ] is not None:
                            poswhere += sql.SQL( "{where} {postab}{coord}{direc}{val}\n"
                                                ).format( where=where,
                                                          postab=postab, coord=sql.Identifier(coord),
                                                          direc=sql.SQL( "<=" if minmax=='max' else ">=" ),
                                                          val=coordlim[ f'{minmax}{coord}' ] )
                            where = "  AND"

            if include_filters:
                filtagg = sql.SQL( "jsonb_object_agg(q1.filter, q1.numphot) AS numphot" )
                commafilt = sql.SQL( ", i.filter" )
            else:
                filtagg = sql.SQL( "q1.numphot" )
                commafilt = sql.SQL( "" )

            q = sql.SQL( textwrap.dedent(
                """
                SELECT o._id AS id, name, {postab}.ra AS ra, {postab}.dec AS dec, {filtagg}
                FROM (
                  SELECT f.object_id{commafilt}, COUNT(f._id) AS numphot FROM dia_forced_photometry f
                  INNER JOIN images i ON f.subtraction_id=i._id
                  {mjdwhere}
                  GROUP BY f.object_id{commafilt}
                  ORDER BY f.object_id{commafilt}
                ) q1
                INNER JOIN objects o ON o._id=q1.object_id
                {posjoin}
                {poswhere}
                """ ) ).format( postab=postab, filtagg=filtagg, commafilt=commafilt,
                                mjdwhere=mjdwhere, posjoin=posjoin, poswhere=poswhere )
            rows, cols = pgdb.execute( q )

        # This next line is byzantine, but seems to work
        return dict( zip( cols, map(list, zip(*rows)) ) )
