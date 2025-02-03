from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import UUID as sqlUUID

from models.base import Base, SeeChangeBase, HasBitFlagBadness, FourCorners, UUIDMixin, SmartSession
from models.enums_and_bitflags import reference_badness_inverse
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

# It's a little bit excessive to have this table, since there is a 1:1
# correspondence between a sub image and it's parent reference, and
# between a sub image and it's parent new zp.  We could just have had a
# "ref_id" and "new_zp_id" field as part of Image.  However, that would
# lead to circular table definitions, as Image includes Reference which
# includes Image.  This may be OK -- it's not really circular, since the
# Image of a Reference is not the same as the Image that the Reference
# is a Reference of.  However, alembic was spazzing out about the
# definition.  This also saves us from storing a new_zp_id and ref_id
# from images that aren't subtractions.
image_subtraction_components = sa.Table(
    'image_subtraction_components',
    Base.metadata,
    sa.Column('image_id',
              sqlUUID,
              sa.ForeignKey('images._id', ondelete='CASCADE', name='image_subtraction_sub_image_fkey' ),
              primary_key=True),
    sa.Column('new_zp_id',
              sqlUUID,
              sa.ForeignKey('zero_points._id', ondelete='RESTRICT', name='image_subtraction_new_zp_fkey' ),
              nullable=False,
              index=True),
    sa.Column('ref_id',
              sqlUUID,
              sa.ForeignKey('refs._id', ondelete='RESTRICT', name='image_subtraction_ref_fkey' ),
              nullable=False,
              index=True)
)


class Reference(Base, UUIDMixin, HasBitFlagBadness):
    """A table that keeps track of subtraction references.

    A reference is defined by a zeropoint.  While this seems
    counter-intuitive, to actually use references, we need the image to
    be fully reduced, which means it needs to have an image, source
    list, background, wcs, and zp.  The zp is all that's necessary to
    find the self-consistent set of the rest.

    """

    __tablename__ = 'refs'   # 'references' is a reserved postgres word

    zp_id = sa.Column(
        sa.ForeignKey('zero_points._id', ondelete='CASCADE', name='references_zp_id_fkey' ),
        nullable=False,
        index=True,
        doc="ID of the zeropoint (and, hence, wcs, bg, source_list, and image) that defines this reference."
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances._id', ondelete="CASCADE", name='references_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this reference. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this reference. "
        )
    )


    @property
    def image( self ):
        if self._image is None:
            self._load_ref_data_products()
        return self._image

    @property
    def sources( self ):
        if self._sources is None:
            self._load_ref_data_products()
        return self._sources

    @property
    def psf( self ):
        if self._psf is None:
            self._load_ref_data_products()
        return self._psf

    @property
    def bg( self ):
        if self._bg is None:
            self._load_ref_data_products()
        return self._bg

    @property
    def wcs( self ):
        if self._wcs is None:
            self._load_ref_data_products()
        return self._wcs

    @property
    def zp( self ):
        if self._zp is None:
            self._load_zp_data_products()
        return self._zp


    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object"""
        return reference_badness_inverse


    def __init__( self, *args, **kwargs ):
        HasBitFlagBadness.__init__( self )
        SeeChangeBase.__init__( self, *args, **kwargs )
        self._image = None
        self._sources = None
        self._bg = None
        self._wcs = None
        self._zp = None

    @orm.reconstructor
    def init_on_load( self ):
        SeeChangeBase.init_on_load( self )
        self._image = None
        self._sources = None
        self._bg = None
        self._wcs = None
        self._zp = None


    def _load_ref_data_products(self, session=None):
        """Load the (SourceList, Background, PSF, WorldCoordiantes, Zeropoint) assocated with self.image_id

        Only works if the all of the upstream dataproducts (image,
        sources, bg, wcs, zp) have been committed ot the database (or
        are in the session, but, brrrr.).

        """

        with SmartSession( session ) as sess:
            self._zp = ZeroPoint.get_by_id( self.zp_id, session=sess )
            self._wcs = WorldCoordinates.get_by_id( self._zp.wcs_id, session=sess )
            self._bg = Background.get_by_id( self._zp.background_id, session=sess )
            if self._bg.sources_id != self._wcs.sources_id:
                raise RuntimeError( f"Database corruption.  Zeropoint {self._zp.id} has wcs {self._wcs.id} and "
                                    f"background {self._bg.id}, but the wcs and bg don't have the same sources_id" )
            self._sources = SourceList.get_by_id( self._wcs.sources_id, session=sess )
            self._image = Image.get_by_id( self._sources.image_id, session=sess)
            self._psf = sess.query( PSF ).filter( PSF.sources_id==self._sources.id ).first()


    def get_upstreams( self, session=None ):
        """Get upstreams of this Reference.  That is the ZeroPoint of the reference."""
        with SmartSession( session ) as sess:
            return [ ZeroPoint.get_by_id( self.zp_id, session=sess ) ]

    def get_downstreams( self, session=None ):
        """Get downstreams of this Reference.  That is all subtraction images that use this as a reference."""
        with SmartSession( session ) as sess:
            return list( sess.query( Image )
                         .join( image_subtraction_components, image_subtraction_components.c.image_id==Image._id )
                         .filter( image_subtraction_components.c.ref_id==self.id )
                         .all() )


    @classmethod
    def get_references(
            cls,
            ra=None,
            dec=None,
            minra=None,
            maxra=None,
            mindec=None,
            maxdec=None,
            image=None,
            overlapfrac=None,
            target=None,
            section_id=None,
            instrument=None,
            filter=None,
            skip_bad=True,
            refset=None,
            provenance_ids=None,
            session=None
    ):
        """Find all references in the specified part of the sky, with the given filter.
        Can also match specific provenances and will (by default) not return bad references.

        Operates in three modes:

        * References tagged for a given target and section_id

        * References that include a given point on the sky.  Specify ra and dec, do not
          pass any of minra, maxra, mindec, maxdec, or target.

        * References that overlap an area on the sky.  Specify either
          minra/maxra/mindec/maxdec or image.  If overlapfrac is None,
          will return references that overlap the area at all; this is
          usually not what you want.

        Parameters
        ----------
        ra: float, optional
            Right ascension in degrees.  If given, must also give the declination.

        dec: float, optional
            Declination in degrees. If given, must also give the right ascension.

        minra, maxra, mindec, maxdec: float, optional
           Rectangle on the sky, in degrees.  Will find references whose
           bounding rectangle overlaps this rectangle on the sky.
           minra is the W edge, maxra is the E edge, so if the center of
           a 2° wide retange is at RA=0, then minra=359 and maxra=1.

        image: Image, optional
           If specified, minra/maxra/mindec/maxdec will be pulled from
           this Image (or any other type of object that inherits from
           FourCorners).

        overlapfrac: float, default None
           If minra/maxra/mindec/maxdec or image is not None, then only
           return references whose bounding rectangle overlaps the
           passed bounding rectangle by at least this much.  Ignored if
           ra/dec or target/section_id is specified.

        target: string, optional
            Name of the target object or field id.  Will only match
            references of this target.  If ra/dec is not given, then
            this and section_id must be given, and that will be used to
            match the reference.

        section_id: string, optional
            Section ID of the reference image.  If given, will only
            match images with this section.

        instrument: string. optional
            Instrument of the reference image.  If given, will only
            match references from this image.

        filter: string, optional
            Filter of the reference image.
            If not given, will return references with any filter.

        refset: string, list of String, or None
            If not None, will only find references that have a
            provenance included in this refset, or in these refsets.

        provenance_ids: list of strings or Provenance objects, optional
            List of provenance IDs to match.  The references must have a
            provenance with one of these IDs.  If neither refset nor
            provenance_ids are given, will load all matching references
            with any provenance.

        skip_bad: bool
            Whether to skip bad references. Default is True.

        session: Session, optional
            The database session to use.
            If not given, will open a session and close it at end of function.

        Returns
        -------
          list of Reference, list of Image

        """

        radecgiven = ( ra is not None ) or ( dec is not None )
        areagiven = ( image is not None ) or any( i is not None for i in [ minra, maxra, mindec, maxdec ] )
        targetgiven = ( target is not None ) or ( section_id is not None )
        if ( ( radecgiven and ( areagiven or targetgiven ) ) or
             ( areagiven and ( radecgiven or targetgiven ) ) or
             ( targetgiven and ( radecgiven or areagiven ) )
            ):
            raise ValueError( "Specify only one of ( target/section_id, "
                              "ra/dec, minra/maxra/mindec/maxdec, or image )" )

        if ( provenance_ids is not None ) and ( refset is not None ):
            raise ValueError( "Specify at most one of provenance_ids or refset" )

        fcobj = None

        with SmartSession( session ) as sess:
            # Mode 1 : target / section_id

            if ( ( target is not None ) or ( section_id is not None ) ):
                if ( target is None ) or (section_id is None ):
                    raise ValueError( "Must give both target and section_id" )
                if overlapfrac is not None:
                    raise ValueError( "Can't give overlapfrac with target/section_id" )

                q = ( "SELECT r.* FROM refs r "
                      "INNER JOIN zero_points z ON r.zp_id=z._id "
                      "INNER JOIN world_coordinates w ON z.wcs_id=w._id "
                      "INNER JOIN source_lists s ON w.sources_id=s._id "
                      "INNER JOIN images i ON S.image_id=i._id "
                      "WHERE i.target=:target AND i.section_id=:section_id " )
                subdict = { 'target': target, 'section_id': section_id }

            # Mode 2 : ra/dec

            elif ( ( ra is not None ) or ( dec is not None ) ):
                if ( ra is None ) or ( dec is None ):
                    raise ValueError( "Must give both ra and dec" )
                if overlapfrac is not None:
                    raise ValueError( "Can't give overlapfrac with ra/dec" )

                # Bobby Tables
                ra = float(ra) if isinstance( ra, int ) else ra
                dec = float(dec) if isinstance( dec, int ) else dec
                if ( not isinstance( ra, float ) ) or ( not isinstance( dec, float ) ):
                    raise TypeError( f"(ra, dec) must be floats, got ({type(ra)}, {type(dec)})" )

                # This code is kind of redundant with the code in
                #   FourCorners._find_possibly_containing_temptable and
                #   FourCorners.find_containing, but we can't just use
                #   that because Reference isn't a FourCorners, and we
                #   have to join Reference to Image

                q = ( "CREATE TEMPORARY TABLE temp_find_containing_ref AS "
                      "( SELECT r.*, i.ra_corner_00, i.ra_corner_01, i.ra_corner_10, i.ra_corner_11, "
                      "         i.dec_corner_00, i.dec_corner_01, i.dec_corner_10, i.dec_corner_11 "
                      "  FROM refs r "
                      "  INNER JOIN zero_points z ON r.zp_id=z._id "
                      "  INNER JOIN world_coordinates w ON z.wcs_id=w._id "
                      "  INNER JOIN source_lists s ON w.sources_id=s._id "
                      "  INNER JOIN images i ON s.image_id=i._id "
                      "  WHERE ( "
                      "    ( i.maxdec >= :dec AND i.mindec <= :dec ) "
                      "    AND ( "
                      "      ( ( i.maxra > i.minra ) AND "
                      "        ( i.maxra >= :ra AND i.minra <= :ra ) )"
                      "      OR "
                      "      ( ( i.maxra < i.minra ) AND "
                      "        ( ( i.maxra >= :ra OR :ra > 180. ) AND ( i.minra <= :ra OR :ra <= 180. ) ) )"
                      "    )"
                      "  )"
                      ")"
                     )
                subdict = { "ra": ra, "dec": dec }
                sess.execute( sa.text(q), subdict )

                q = ( "SELECT r.* FROM refs r INNER JOIN temp_find_containing_ref t ON r._id=t._id "
                      "INNER JOIN zero_points z ON r.zp_id=z._id "
                      "INNER JOIN world_coordinates w ON z.wcs_id=w._id "
                      "INNER JOIN source_lists s ON w.sources_id=s._id "
                      "INNER JOIN images i ON s.image_id=i._id "
                      "WHERE q3c_poly_query( :ra, :dec, ARRAY[ t.ra_corner_00, t.dec_corner_00, "
                      "                                        t.ra_corner_01, t.dec_corner_01, "
                      "                                        t.ra_corner_11, t.dec_corner_11, "
                      "                                        t.ra_corner_10, t.dec_corner_10 ] ) " )

            # Mode 3 : overlapping area

            elif ( image is not None ) or any( i is not None for i in [ minra, maxra, mindec, maxdec ] ):
                if image is not None:
                    if any( i is not None for i in [ minra, maxra, mindec, maxdec ] ):
                        raise ValueError( "Specify either image or minra/maxra/mindec/maxdec, not both" )
                    minra = image.minra
                    maxra = image.maxra
                    mindec = image.mindec
                    maxdec = image.maxdec
                    fcobj = image
                else:
                    if any( i is None for i in [ minra, maxra, mindec, maxdec ] ):
                        raise ValueError( "Must give all of minra, maxra, mindec, maxdec" )
                    fcobj = FourCorners()
                    fcobj.ra = (minra + maxra) / 2.
                    fcobj.dec = (mindec + maxdec) / 2.
                    fcobj.ra_corner_00 = minra
                    fcobj.ra_corner_01 = minra
                    fcobj.minra = minra
                    fcobj.ra_corner_10 = maxra
                    fcobj.ra_corner_11 = maxra
                    fcobj.maxra = maxra
                    fcobj.dec_corner_00 = mindec
                    fcobj.dec_corner_10 = mindec
                    fcobj.mindec = mindec
                    fcobj.dec_corner_01 = maxdec
                    fcobj.dec_corner_11 = maxdec
                    fcobj.maxdec = maxdec

                # Sort of redundant code from FourCorners._find_potential_overlapping_temptable,
                #  but we can't just use that because Reference isn't a FourCorners and
                #  we have to do the refs/images join.

                q = ( "SELECT r.* FROM refs r "
                      "INNER JOIN zero_points z ON r.zp_id=z._id "
                      "INNER JOIN world_coordinates w ON z.wcs_id=w._id "
                      "INNER JOIN source_lists s ON w.sources_id=s._id "
                      "INNER JOIN images i ON s.image_id=i._id "
                      "WHERE ( "
                      "  ( i.maxdec >= :mindec AND i.mindec <= :maxdec ) "
                      "  AND "
                      "  ( ( ( i.maxra >= i.minra AND :maxra >= :minra ) AND "
                      "      i.maxra >= :minra AND i.minra <= :maxra ) "
                      "    OR "
                      "    ( i.maxra < i.minra AND :maxra < :minra ) "   # both include RA=0, will overlap in RA
                      "    OR "
                      "    ( ( i.maxra < i.minra AND :maxra >= :minra AND :minra <= 180. ) AND "
                      "      i.maxra >= :minra ) "
                      "    OR "
                      "    ( ( i.maxra < i.minra AND :maxra >= :minra AND :minra > 180. ) AND "
                      "      i.minra <= :maxra ) "
                      "    OR "
                      "    ( ( i.maxra >= i.minra AND :maxra < :minra AND i.maxra <= 180. ) AND "
                      "      i.minra <= :maxra ) "
                      "    OR "
                      "    ( ( i.maxra >= i.minra AND :maxra < :minra AND i.maxra > 180. ) AND "
                      "      i.maxra >= :minra ) "
                      "  )"
                      ") " )
                subdict = { 'minra': minra, 'maxra': maxra, 'mindec': mindec, 'maxdec': maxdec }

            else:
                raise ValueError( "Must give one of target/section_id, ra/dec, or minra/maxra/mindec/maxdec or image" )

            # Additional criteria

            if refset is not None:
                q += " AND r.provenance_id IN "
                q += " ( SELECT rs.provenance_id FROM refsets rs "
                # TODO : be fancier with collections.abc.Sequence or something
                if isinstance( refset, list ):
                    q += "  WHERE rs.name IN :refsets ) "
                    subdict['refsets'] = tuple( refset )
                else:
                    q += "  WHERE rs.name=:refset ) "
                    subdict['refset'] = refset

            elif provenance_ids is not None:
                if isinstance( provenance_ids, str ) or isinstance( provenance_ids, UUID ):
                    q += " AND r.provenance_id=:prov"
                    subdict['prov'] = provenance_ids
                elif isinstance( provenance_ids, Provenance ):
                    q += " AND r.provenance_id=:prov"
                    subdict['prov'] = provenance_ids.id
                elif isinstance( provenance_ids, list ):
                    q += " AND r.provenance_id IN :provs"
                    subdict['provs'] = []
                    for pid in provenance_ids:
                        subdict['provs'].append( pid if isinstance( pid, str ) or isinstance( pid, UUID )
                                                 else pid.id )
                    subdict['provs'] = tuple( subdict['provs'] )

            if instrument is not None:
                q += " AND i.instrument=:instrument "
                subdict['instrument'] = instrument

            if filter is not None:
                q += " AND i.filter=:filter "
                subdict['filter'] = filter

            if skip_bad:
                q += " AND r._bitflag=0 AND r._upstream_bitflag=0 "

            # Get the Reference objects
            references = list( sess.scalars( sa.select( Reference )
                                             .from_statement( sa.text(q).bindparams(**subdict) )
                                            ).all() )

            # Get the image and zeropoint objects
            things = list( sess.query( Image, ZeroPoint )
                           .join( SourceList, Image._id==SourceList.image_id )
                           .join( WorldCoordinates, SourceList._id==WorldCoordinates.sources_id )
                           .join( ZeroPoint, WorldCoordinates._id==ZeroPoint.wcs_id )
                           .filter( ZeroPoint._id.in_( r.zp_id for r in references ) )
                           .all() )
            images = [ t[0] for t in things ]
            zeropoints = [ t[1] for t in things ]

        # Make sure the images are sorted right
        imdict = { z._id : i for i, z in zip(images, zeropoints) }
        if not all( r.zp_id in imdict.keys() for r in references ):
            raise RuntimeError( "Didn't get back the images expected; this should not happen!" )
        images = [ imdict[r.zp_id] for r in references ]

        # Deal with overlapfrac if relevant

        if overlapfrac is not None:
            retref = []
            retim = []
            for r, i in zip( references, images ):
                if FourCorners.get_overlap_frac( fcobj, i ) >= overlapfrac:
                    retref.append( r )
                    retim.append( i )
            references = retref
            images = retim

        # Done!

        return references, images

    def free(self):
        for prop in [ self._image, self._sources, self._bg, self._wcs ]:
            if prop is not None:
                prop.free()

        self._image = None
        self._sources = None
        self._bg = None
        self._wcs = None
        self._zp = None
