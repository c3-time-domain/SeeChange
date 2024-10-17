from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, FourCorners, UUIDMixin, SmartSession
from models.provenance import Provenance
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from util.util import listify


class Reference(Base, UUIDMixin):
    """
    A table that refers to each reference Image object,
    based on the object/field it is targeting.
    The provenance of this table (tagged with the "reference" process)
    will have as its upstream IDs the provenance IDs of the image,
    the source list, the PSF, the WCS, and the zero point.

    This means that the reference should always come loaded
    with the image and all its associated products,
    based on the provenance given when it was created.
    """

    __tablename__ = 'refs'   # 'references' is a reserved postgres word

    image_id = sa.Column(
        sa.ForeignKey('images._id', ondelete='CASCADE', name='references_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the reference image that this object is referring to. "
    )

    # TODO: some of the targets below are redundant with image, and searches should probably
    #   be replaced with joins to image.
    target = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc=(
            'Name of the target object or field id. '
            'This string is used to match the reference to new images, '
            'e.g., by matching the field ID on a pre-defined grid of fields. '
        )
    )

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Name of the instrument used to make the images for this reference image. "
    )

    filter = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Filter used to make the images for this reference image. "
    )

    section_id = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Section ID of the reference image. "
    )

    # this badness is in addition to the regular bitflag of the underlying products
    # it can be used to manually kill a reference and replace it with another one
    is_bad = sa.Column(
        sa.Boolean,
        nullable=False,
        server_default='false',
        doc="Whether this reference image is bad. "
    )

    bad_reason = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "The reason why this reference image is bad. "
            "Should be a single pharse or a comma-separated list of reasons. "
        )
    )

    bad_comment = sa.Column(
        sa.Text,
        nullable=True,
        doc="Any additional comments about why this reference image is bad. "
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

    def get_upstream_provenances(self):
        """Collect the provenances for all upstream objects.
        Assumes all the objects are already committed to the DB
        (or that at least they have provenances with IDs).

        Returns
        -------
        list of Provenance objects:
            a list of unique provenances, one for each data type.
        """
        raise RuntimeError( "Deprecated" )


    def get_ref_data_products(self, session=None):
        """Get the (SourceList, Background, PSF, WorldCoordiantes, Zeropoint) assocated with self.image_id

        Only works if the sources (etc.) have already been committed to
        the database with a provenance that's in the upstreams of this
        object's provenance.

        Returns
        -------
          sources, bg, psf, wcs, zp

        """

        with SmartSession( session ) as sess:
            prov = Provenance.get( self.provenance_id, session=sess )
            upstrs = prov.get_upstreams( session=sess )
            upids = [ p.id for p in upstrs ]
            srcs = ( sess.query( SourceList )
                     .filter( SourceList.image_id == self.image_id )
                     .filter( SourceList.provenance_id.in_( upids ) )
                    ).all()

            if len( srcs ) > 1:
                raise RuntimeError( "Reference found more than one matching SourceList; this shouldn't happen" )
            if len( srcs ) == 0:
                raise RuntimeError( f"Sources not in database for Reference {self.id}" )
            sources = srcs[0]

            # For the rest, we're just going to assume that there aren't multiples in the database.
            # By construction, there shouldn't be....
            bg = sess.query( Background ).filter( Background.sources_id == sources.id ).first()
            psf = sess.query( PSF ).filter( PSF.sources_id == sources.id ).first()
            wcs = ( sess.query( WorldCoordinates )
                    .filter( WorldCoordinates.sources_id == sources.id ) ).first()
            zp = sess.query( ZeroPoint ).filter( ZeroPoint.sources_id == sources.id ).first()

        return sources, bg, psf, wcs, zp


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
            raise ValueError( "Specify only one of ( target/section_id, ra/dec, minra/maxra/mindec/maxdec, or image )" )

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

                q = "SELECT r.* FROM refs r WHERE target=:target AND section_id=:section_id "
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
                      "  INNER JOIN images i ON r.image_id=i._id "
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

                q = ( "SELECT r.* FROM refs r INNER JOIN images i ON r.image_id=i._id "
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
                q += " ( SELECT DISTINCT ON(rpa.provenance_id) rpa.provenance_id "
                q += "   FROM refset_provenance_association rpa "
                q +=  "     INNER JOIN refsets rs ON rpa.refset_id=rs._id "
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
                q += " AND r.instrument=:instrument "
                subdict['instrument'] = instrument

            if filter is not None:
                q += " AND r.filter=:filter "
                subdict['filter'] = filter

            if skip_bad:
                q += " AND NOT r.is_bad "

            # Get the Reference objects
            references = list( sess.scalars( sa.select( Reference )
                                             .from_statement( sa.text(q).bindparams(**subdict) )
                                            ).all() )

            # Get the image objects
            images = list( sess.scalars( sa.select( Image )
                                         .where( Image._id.in_( r.image_id for r in references ) )
                                        ).all() )

        # Make sure they're sorted right

        imdict = { i._id : i for i in images }
        if not all( r.image_id in imdict.keys() for r in references ):
            raise RuntimeError( "Didn't get back the images expected; this should not happen!" )
        images = [ imdict[r.image_id] for r in references ]

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


    # ======================================================================
    # The fields below are things that we've deprecated; these definitions
    #   are here to catch cases in the code where they're still used

    @property
    def image( self ):
        raise RuntimeError( f"Don't use Reference.image, use image_id" )

    @image.setter
    def image( self, val ):
        raise RuntimeError( f"Don't use Reference.image, use image_id" )

    @property
    def provenance( self ):
        raise RuntimeError( f"Don't use Reference.provenance, use provenance_id" )

    @provenance.setter
    def provenance( self, val ):
        raise RuntimeError( f"Don't use Reference.provenance, use provenance_id" )

    @property
    def sources( self ):
        raise RuntimeError( f"Reference.sources is deprecated, don't use it" )

    @sources.setter
    def sources( self, val ):
        raise RuntimeError( f"Reference.sources is deprecated, don't use it" )

    @property
    def psf( self ):
        raise RuntimeError( f"Reference.psf is deprecated, don't use it" )

    @psf.setter
    def psf( self, val ):
        raise RuntimeError( f"Reference.psf is deprecated, don't use it" )

    @property
    def bg( self ):
        raise RuntimeError( f"Reference.bg is deprecated, don't use it" )

    @bg.setter
    def bg( self, val ):
        raise RuntimeError( f"Reference.bg is deprecated, don't use it" )

    @property
    def wcs( self ):
        raise RuntimeError( f"Reference.wcs is deprecated, don't use it" )

    @wcs.setter
    def wcs( self, val ):
        raise RuntimeError( f"Reference.wcs is deprecated, don't use it" )

    @property
    def zp( self ):
        raise RuntimeError( f"Reference.zp is deprecated, don't use it" )

    @zp.setter
    def zp( self, val ):
        raise RuntimeError( f"Reference.zp is deprecated, don't use it" )

