import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.ext.declarative import declared_attr

from models.base import Base, UUIDMixin, HasBitFlagBadness, SeeChangeBase, SmartSession
from models.enums_and_bitflags import catalog_match_badness_inverse

# many-to-many link of the zeropoints of coadded images to images that went into the coadd
image_coadd_component_table = sa.Table(
    'image_coadd_component',
    Base.metadata,
    sa.Column('zp_id',
              sqlUUID,
              sa.ForeignKey('zero_points._id', ondelete="RESTRICT", name='image_coadd_component_zp_fkey'),
              index=True,
              primary_key=True),
    sa.Column('coadd_image_id',
              sqlUUID,
              sa.ForeignKey('images._id', ondelete="CASCADE", name='image_coadd_component_coadd_fkey'),
              index=True,
              primary_key=True),
)


class ZeroPoint(Base, UUIDMixin, HasBitFlagBadness):
    __tablename__ = 'zero_points'

    @declared_attr
    def __table_args__(cls):   # noqa: N805
        return (
            UniqueConstraint( 'wcs_id', 'background_id', 'provenance_id', name='_zp_wcs_bg_provenance_uc' ),
        )

    # Note that the sources_id of both the upstream wcs and background must be the same
    # The structure of the database does not enforce this.
    wcs_id = sa.Column(
        sa.ForeignKey('world_coordinates._id', ondelete='CASCADE', name='zp_wcs_id_fkey' ),
        nullable=False,
        index=True,
        doc="ID of the wcs this zero point is based on."
    )

    background_id = sa.Column(
        sa.ForeignKey('backgrounds._id', ondelete='CASCADE', name='zp_background_id_fkey' ),
        nullable=False,
        index=True,
        doc="ID of the background this zero point is based on."
    )

    zp = sa.Column(
        sa.REAL,
        nullable=False,
        index=False,
        doc="Zeropoint: -2.5*log10(adu_psf) + zp = mag"
    )

    dzp = sa.Column(
        sa.REAL,
        nullable=False,
        index=False,
        doc="Uncertainty on zp"
    )

    aper_cor_radii = sa.Column(
        ARRAY( sa.REAL, zero_indexes=True ),
        nullable=True,
        server_default=None,
        index=False,
        doc="Pixel radii of apertures whose aperture corrections are in aper_cors."
    )

    aper_cors = sa.Column(
        ARRAY( sa.REAL, zero_indexes=True ),
        nullable=True,
        server_default=None,
        index=False,
        doc=( "Aperture corrections for apertures with radii in aper_cor_radii.  Defined so that "
              "mag = -2.5*log10(adu_aper) + zp + aper_cor, where adu_aper is the number of ADU "
              "in the aperture with the specified radius.  There is a built-in approximation that a "
              "single aperture applies across the entire image, which should be OK given that the "
              "pipeline isn't expected to have photometry to better than a couple of percent." )
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances._id', ondelete="CASCADE", name='zp_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the provenance of this zerpoinht. "
    )

    def __init__(self, *args, **kwargs):
        HasBitFlagBadness.__init__(self)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        # manually set all properties (columns or not)
        self.set_attributes_from_dict(kwargs)

    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)

    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object"""
        return catalog_match_badness_inverse

    def get_aper_cor( self, rad ):
        """Return the aperture correction for a given aperture radius in pixels.

        Requires rad to be within 0.01 pixels of one of the tabulated
        aperture corrections.  If the requested one isn't found, will
        raise a ValueError.

        Parameters
        ------------
          rad: float
            The radius of the aperture in pixels

        Returns
        -------
          aper_cor: float
             Defined so that mag = -2.5*log10(aperflux_adu) + zp + aper_cor

        """

        if self.aper_cor_radii is None:
            raise ValueError( "No aperture corrections tabulated." )

        for aprad, apcor in zip( self.aper_cor_radii, self.aper_cors ):
            if np.fabs( rad - aprad ) <= 0.01:
                return apcor

        raise ValueError( f"No aperture correction tabulated for zeropoint {self.id} "
                          f"for apertures within 0.01 pixels of {rad}; "
                          f"available apertures are {self.aper_cor_radii}" )


    def get_upstreams(self, session=None):
        """Get the WCS and Background that are upstream to this ZeroPoint."""
        from models.background import Background
        from models.world_coordinates import WorldCoordinates
        with SmartSession(session) as session:
            bgs = list( session.scalars( sa.select(Background).where( Background._id==self.background_id ) ).all() )
            wcses = list( session.scalars( sa.select(WorldCoordinates)
                                           .where( WorldCoordinates._id==self.wcs_id ) ).all() )
            return bgs + wcses

    def get_downstreams(self, session=None):
        """Get any downstreams of this ZeroPoint.

        This includes subtraction images, coadded images, and references.

        """
        from models.image import Image
        from models.reference import Reference, image_subtraction_components
        with SmartSession(session) as session:
            coadds = ( session.query( Image )
                       .join( image_coadd_component_table, image_coadd_component_table.c.coadd_image_id==Image._id )
                       .filter( image_coadd_component_table.c.zp_id==self.id ) ).all()
            subs = ( session.query( Image )
                     .join( image_subtraction_components, Image._id==image_subtraction_components.c.image_id )
                     .filter( image_subtraction_components.c.new_zp_id==self.id ) ).all()
            refs = ( session.query( Reference ).filter( Reference.zp_id==self.id ) ).all()
            return list(coadds) + list(subs) + list(refs)
