import numbers
import pandas
import nested_pandas
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

    psfflags = sa.Column(
        sa.Integer,
        nullable = False,
        index = False,
        default = 0,
        server_default = '0',
        doc = ( "photutils flags from psf photometry." )
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

    aperflags = sa.Column(
        ARRAY( sa.Integer, zero_indexes=True ),
        nullable = False,
        index = False,
        doc = ( "photutils flags from aperture photometry" )
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
            rows = pgdb.execute( q )
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
            newzp = rows[0]['newzp']
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
        clips[0] = make_cutouts( newdata, [ix], [iy], size=clip_size, fillvalue=np.nan, dtype=np.float32 )[0]
        clips[2] = make_cutouts( subdata, [ix], [iy], size=clip_size, fillvalue=np.nan, dtype=np.float32 )[0]
        if warprefdata is not None:
            clips[1] = make_cutouts( warprefdata, [ix], [iy], size=clip_size, fillvalue=np.nan, dtype=np.float32 )[0]
            if scale_ref:
                clips[1] *= 10 ** ( ( newzp - warprefzp) / 2.5 )
        else:
            clips[1] = np.nan

        return clips

    @classmethod
    def get_lightcurve_for_object( cls, obj, prov, filters=None, include_apertures=False, include_raw=False,
                                   include_im_ids=False, include_pos=False, include_phot_ids=False,
                                   diaforcedphotobjs=False, return_format="listofdicts", return_obj=False,
                                   pgdb=None ):
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
            Normally, just returns psf photometry.  If True, also return
            aperture photometry; see Returns below.  Ignored if
            diaforcedphotojbs is True.

          include_raw : bool, default False
            Normally, just returns photometry in units of nJy (so all
            zeropoints will have been applied).  If this is True, then
            there are additional columns with the raw photometry in
            counts.  Ignored if diaforcedphotobjs is True.

          include_im_ids : bool, default False
            If True, there will be three additional columns newid,
            refid, subid that have the ids of the image, warped
            reference, and difference image, where that last one is the
            the forced photometry point was performed on.  Ignored if
            diaforcedphotobjs is True.

          include_pos : bool, default False
            If True, there will be two additional columns x,y giving the
            position used for the object for this point.  Ignored if
            diaforcedphotojbs is True.

          include_phot_ids : bool, default False
            If True, there will be an additional column diaforcedphotid
            giving the UUIDs of the DiaForcedPhot objects that go with
            the lightcurve points.

          diaforcedphotobjs : bool, default False
            Changes what's returned; see Results below.

          return_format : str, default listofdicts
            Either listofdicts, dictoflists, or dataframe.  You really
            don't want to use dataframe if diaforcedphotojbs is True,
            but I won't stop you.

          pgdb : PGDB, psycopg.Connection, or psycopg.Curosr, default None
            Database connection.  If not given, a new connection will be
            opened and closed.

        Returns
        -------
          Either a dict of lists, a list of dicts, a pandas.DataFrame,
          or a nested_pandas.NestedFrame

          If diaforcedphots is True

            You either get back a list (if retrurn_format is
            "listofdicts") where each element has keys:
              mjd -- value is a double
              filter -- value is a string
              diaforcedphot -- value is a DiaForcedPhotObj

            or you get back a dict of lists, where the keys are as above
            and each element is a list of the same length with the values.

          If diaforcedphots is False:

            If a return_format is "dictoflists", you get a dict of lists
            where the keys are the column names, and the values are
            lists all with the same length (the number of lightcurve
            points).

            If return_format is "dictoflists", the length of the list is the number of
            lightcurve points, and each element of the list is a
            dictionary of column: value.

            If return_format is "dataframe", then if include_apertures
            is False, it will be a pandas dataframe.  If
            include_apertures is True, it will be a nested_pandas
            dataframe, and the 'flux_apertures_raw',
            'flux_apertures_raw_err', and 'aper_cors' columns are in the
            nested 'apphot' subframe of each row.  (So, if you put the
            return into df, you'd access it via
            df['apphot]['flux_apertures_raw'].)

            The columns are:
                mjd
                filter
                flux_psf           -- in nJy
                flux_psf_err       -- in nJy
                psfflags           -- 32-bit bitfield; see enums_and_bitflags.py::psf_phot_flags_dict
                [ flux_psf_raw     -- counts on image, only present if include_raw is True ]
                [ flux_psf_raw_err -- counts on image, only present if include_raw is True ]
                [ flux_aper_0      -- in nJy ** ]
                [ flux_aper_0_err  -- in nJy ** ]
                [ flux_aper_n      -- in nJy ** ]
                [ flux_aper_n_err  -- in nJy ** ]
                [ aperflags_0      -- 32-bit bitfield; only present if include_apertures is True ]
                [ aperflags_n      -- 32-bit bitfield; only present if include_apertures is True ]
                [ flux_apertures_raw      -- list, counts on image ** ]
                [ flux_apertures_raw_err  -- list, counts on image ** ]
                [ aperflags_raw           -- ** ]
                [ aper_pix                -- list, aperture radii in pixels ** ]
                [ aper_cors               -- list, aperture corrections (magnitudes) ** ]
                [ newid -- uuid, id of the new image, only present if include_im_ids is True ]
                [ newzpid -- uuid, id of the zp of the new image, only present if include_im_ids is True ]
                [ refid -- uuid, id of the warped ref image, only present if include_im_ids is True ]
                [ refzpid -- uuid, id of the zp of the ref image, only present if include_im_ids is True ]
                [ subid -- uuid, id of the subtraction image, only present if include_im_ids is True ]
                [ diaforcedphotid-- uuid, id of the diaforcedphot table row, only present if include_phot_ids is True ]
                [ x -- position on subtraction image, only if include_pos is True ]
                [ y -- position on subtraction image, only if include_pos is True ]

                ** aperture photometry is only included if include_apertures
                   is True.  Aperture photometry is in the apertures that
                   are defined by the ZeroPoint object, and are returned in
                   aper_pix; corresponding aperture corrections are in
                   aper_cors.  The flux_apertures_raw and
                   flux_apertures_raw_err have photometry in counts on the
                   image, with NO APERTURE CORRECTIONS APPLIED; aperture
                   corrections can be found in aper_cors.  These two columns
                   will only be included if both include_apertures and
                   include_raw is true.  flux_aper_0 and flux_aper_n have
                   photometry in nJy in apertures of radii aper_pix[0] and
                   aper_pix[-1]; these DO have aperture corrections (from
                   aper_cors[0] and aper_cors[-1]) applied.

            It's possible that some fields will be None, and if float_aper_0
            or float_aper_n would come out None (because of aperture
            photometry failure for some reason or another), they will be
            NaN.

        """

        if return_format not in ( "dictoflists", "listofdicts", "dataframe" ):
            raise ValueError( f"Unknown return_format {return_format}; must be one of dictoflists, "
                              f"listofdicts, or dataframe." )

        filters = listify( filters )
        if ( filters is None ) or ( len(filters) <= 0 ):
            filtwhere = sql.SQL( "" )
        elif len(filters) == 1:
            filtwhere = sql.SQL( "  AND i.filter={filt}" ).format( filt=filters[0] )
        else:
            filtwhere = sql.SQL( "  AND i.filter=ANY(ARRAY[{filts}])" ).format( filts=sql.SQL(",").join(filters) )

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
                zpjoin = sql.SQL( "" )
                idjoin = sql.SQL( "" )
                photcols = sql.SQL( "i.mjd, i.filter, p.*" )
                photmetacols = sql.SQL( "*" )
            else:
                zpjoin = sql.SQL( "INNER JOIN zero_points z ON z._id=isc.new_zp_id" )
                idjoin = sql.SQL( "" )
                photcols = sql.SQL( "i.mjd, i.filter, z.zp{photidcol}{poscols}, p.psfflags,\n"
                                    "    p.flux_psf * 10^((31.4-z.zp)/2.5) AS flux_psf,\n"
                                    "    p.flux_psf_err * 10^((31.4-z.zp)/2.5) AS flux_psf_err"
                                   ).format( photidcol=sql.SQL( ", p._id as diaforcedphotid"
                                                                if include_phot_ids else "" ),
                                             poscols=sql.SQL( ", x, y" if include_pos else "" ) )
                photmetacols = sql.SQL(
                    "mjd, filter, zp{photidcol}{poscols}, psfflags,\n"
                    "  CASE WHEN flux_psf IS NULL THEN 'NaN' ELSE flux_psf END AS flux_psf,\n"
                    "  CASE WHEN flux_psf_err IS NULL THEN 'NaN' ELSE flux_psf_err END AS flux_psf_err"
                ).format( photidcol=sql.SQL( ", diaforcedphotid" if include_phot_ids else "" ),
                          poscols=sql.SQL( ", x, y" if include_pos else "" ) )
                if include_raw:
                    photcols += sql.SQL( ",\n    p.flux_psf AS flux_psf_raw, p.flux_psf_err AS flux_psf_raw_err" )
                    photmetacols += sql.SQL( ",\n  flux_psf_raw, flux_psf_raw_err" )
                if include_apertures:
                    photcols += sql.SQL( ",\n" ) + sql.SQL( textwrap.indent( textwrap.dedent(
                        """\
                        z.aper_cors, z.aper_cor_radii AS aper_pix,
                        p.flux_apertures[1] * 10^((31.4-z.zp-z.aper_cors[1])/2.5) AS flux_aper_0,
                        p.flux_apertures_err[1] * 10^((31.4-z.zp-z.aper_cors[1])/2.5) AS flux_aper_0_err,
                        p.flux_apertures[array_upper(p.flux_apertures, 1)] *
                          10^((31.4-z.zp-z.aper_cors[array_upper(z.aper_cors, 1)])/2.5) AS flux_aper_n,
                        p.flux_apertures_err[array_upper(p.flux_apertures_err, 1)] *
                          10^((31.4-z.zp-z.aper_cors[array_upper(z.aper_cors, 1)])/2.5) AS flux_aper_n_err,
                        p.aperflags[1] AS aperflags_0, p.aperflags[array_upper(p.aperflags, 1)] AS aperflags_n
                        """
                    ), "    " ) )
                    photmetacols += sql.SQL( ",\n") + sql.SQL( textwrap.indent( textwrap.dedent(
                        """\
                        aper_cors, aper_pix,
                        CASE WHEN flux_aper_0 IS NULL THEN 'NaN' ELSE flux_aper_0 END AS flux_aper_0,
                        CASE WHEN flux_aper_0_err IS NULL THEN 'NaN' ELSE flux_aper_0_err END AS flux_aper_0_err,
                        CASE WHEN flux_aper_n IS NULL THEN 'NaN' ELSE flux_aper_n END AS flux_aper_n,
                        CASE WHEN flux_aper_n_err IS NULL THEN 'NaN' ELSE flux_aper_n_err END AS flux_aper_n_err,
                        aperflags_0, aperflags_n\
                        """
                    ), "  " ) )
                    if include_raw:
                        photcols += sql.SQL( ",\n    p.flux_apertures AS flux_apertures_raw"
                                             ",\n    p.flux_apertures_err AS flux_apertures_raw_err"
                                             ",\n    p.aperflags AS aperflags_raw" )
                        photmetacols += sql.SQL( ",\n  flux_apertures_raw, flux_apertures_raw_err, aperflags_raw" )
                if include_im_ids:
                    # LEFT JOIN on the ref stuff because we weren't originally saving warped refs
                    idjoin = sql.SQL( textwrap.indent( textwrap.dedent(
                        """\
                        INNER JOIN world_coordinates w ON z.wcs_id=w._id
                        INNER JOIN source_lists s ON w.sources_id=s._id
                        INNER JOIN refs r ON isc.ref_id=r._id
                        INNER JOIN zero_points rz ON r.zp_id=rz._id
                        LEFT JOIN source_lists wrs ON isc.warped_ref_sources_id=wrs._id
                        """
                    ), "  " ) )
                    photcols += sql.SQL( ",\n" ) + sql.SQL( textwrap.indent( textwrap.dedent(
                        """\
                        s.image_id AS newid, z._id AS newzpid,
                        wrs.image_id AS refid, rz._id AS refzpid,
                        i._id AS subid\
                        """
                    ), "    " ) )
                    photmetacols += sql.SQL( ",\n  newid, newzpid, refid, refzpid, subid" )

            q = sql.SQL( textwrap.dedent(
                """\
                SELECT {photmetacols}
                FROM (
                  SELECT {photcols}
                  FROM dia_forced_photometry p
                  INNER JOIN images i ON p.subtraction_id=i._id
                  INNER JOIN image_subtraction_components isc ON isc.image_id=i._id
                  {zpjoin}
                  {idjoin}
                  WHERE p.object_id={objid} AND p.provenance_id={provid}
                  {filtwhere}
                ) subq
                ORDER BY mjd
                """ ) ).format( objid=obj.id, provid=prov.id, photcols=photcols,
                                photmetacols=photmetacols, zpjoin=zpjoin, idjoin=idjoin,
                                filtwhere=filtwhere )

            rows, cols = pgdb.execute( q )
            coldex = { c: i for i, c in enumerate(cols) }

        if diaforcedphotobjs:
            # I set create_uuidify=False below because I know given how I did it above
            #   that I already have UUIDs (or None).  May as well save all that
            #   function calling overhead.
            if return_format == 'listofdicts':
                rval = []
                for row in rows:
                    rval.append( { 'mjd': row[coldex['mjd']],
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
                    rval['diaforcedphot'].append( DiaForcedPhot.create( create_uuidify=False, **mess ) )

                if return_format == "dataframe":
                    # ...this can't be a good idea
                    rval = pandas.DataFrame( rval )

        else:
            if return_format == 'dictoflists':
                # Python magic
                rval = dict( zip( cols, map(list, zip(*rows)) ) )

            elif return_format == 'listofdicts':
                rval = [ { c: r[coldex[c]] for c in cols } for r in rows ]

            elif return_format == 'dataframe':
                if include_apertures:
                    rval = nested_pandas.NestedFrame( rows, columns=cols )
                    # This magic was suggested by Kostya
                    tonest = [ 'aper_cors', 'aper_pix' ]
                    if include_raw:
                        tonest.extend( [ 'flux_apertures_raw', 'flux_apertures_raw_err', 'aperflags_raw' ] )
                    rval = rval.nest_lists( tonest, 'apphot' )
                else:
                    rval = pandas.DataFrame( rows, columns=cols )

            else:
                raise RuntimeError( "This should never happen." )

        if return_obj:
            return rval, obj
        else:
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
