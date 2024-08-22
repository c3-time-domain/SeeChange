import shapely.geometry

import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, UUIDMixin, SmartSession
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
            target=None,
            section_id=None,
            instrument=None,
            filter=None,
            skip_bad=True,
            provenance_ids=None,
            session=None
    ):
        """Find all references in the specified part of the sky, with the given filter.
        Can also match specific provenances and will (by default) not return bad references.

        Parameters
        ----------
        ra: float or string, optional
            Right ascension in degrees, or a hexagesimal string (in hours!).
            If given, must also give the declination.

        dec: float or string, optional
            Declination in degrees, or a hexagesimal string (in degrees).
            If given, must also give the right ascension.

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

        provenance_ids: list of strings or Provenance objects, optional
            List of provenance IDs to match.
            The references must have a provenance with one of these IDs.
            If not given, will load all matching references with any provenance.

        skip_bad: bool
            Whether to skip bad references. Default is True.

        session: Session, optional
            The database session to use.
            If not given, will open a session and close it at end of function.

        Returns
        -------
          list of Reference, list of Image

        """
        if ( ( ( ra is None ) or ( dec is None ) ) and
             ( ( target is None ) or ( section_id is None ) )
            ):
            raise ValueError( "Must provide at least ra/dec or target/section_id" )

        if ( ra is None ) != ( dec is None ):
            raise ValueError( "Must provide both or neither of ra/dec" )

        if ra is None:
           stmt = ( sa.select( Reference, Image )
                     .where( Reference.target == target )
                     .where( Reference.section_id == section_id )
                    )
        else:
            # Not using FourCorners.containing here, because
            #   that doesn't actually use the q3c indices,
            #   so will be slow.  minra, maxra, mindec, maxdec
            #   have classic indices, so this is a good first pass.
            #   Below, we'll crop the list down.
            stmt = ( sa.select( Reference, Image )
                     .where( Image._id==Reference.image_id )
                     .where( Image.minra<=ra )
                     .where( Image.maxra>=ra )
                     .where( Image.mindec<=dec )
                     .where( Image.maxdec>=dec )
                    )
            if target is not None:
                stmt = stmt.where( Reference.target==target )
            if section_id is not None:
                stmt = stmt.where( Reference.section_id==str(section_id) )

        if instrument is not None:
            stmt = stmt.where( Reference.instrument==instrument )

        if filter is not None:
            stmt = stmt.where( Reference.filter==filter )

        if skip_bad:
            stmt = stmt.where( Reference.is_bad.is_( False ) )

        provenance_ids = listify(provenance_ids)
        if provenance_ids is not None:
            for i, prov in enumerate(provenance_ids):
                if isinstance(prov, Provenance):
                    provenance_ids[i] = prov.id
                elif not isinstance(prov, str):
                    raise ValueError(f"Provenance ID must be a string or a Provenance object, not {type(prov)}.")

            stmt = stmt.where( Reference.provenance_id.in_(provenance_ids) )

        with SmartSession( session ) as sess:
            refs = sess.execute( stmt ).all()
        imgs = [ r[1] for r in refs ]
        refs = [ r[0] for r in refs ]

        if ra is not None:
            # Have to crop down the things found to things that actually include
            #  the ra/dec
            croprefs = []
            cropimgs = []
            for ref, img in zip( refs, imgs ):
                poly = shapely.geometry.Polygon( [ ( img.ra_corner_00, img.dec_corner_00 ),
                                                   ( img.ra_corner_01, img.dec_corner_01 ),
                                                   ( img.ra_corner_11, img.dec_corner_11 ),
                                                   ( img.ra_corner_10, img.dec_corner_10 ),
                                                   ( img.ra_corner_00, img.dec_corner_00 ) ] )
                if poly.contains( shapely.geometry.Point( ra, dec ) ):
                    croprefs.append( ref )
                    cropimgs.append( img )
            refs = croprefs
            imgs = cropimgs

        return refs, imgs

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

