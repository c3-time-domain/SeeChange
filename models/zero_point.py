import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY

from models.base import Base, SmartSession, AutoIDMixin, HasBitFlagBadness
from models.enums_and_bitflags import catalog_match_badness_inverse
from models.world_coordinates import WorldCoordinates
from models.source_list import SourceList

class ZeroPoint(Base, AutoIDMixin, HasBitFlagBadness):
    __tablename__ = 'zero_points'

    __table_args__ = (
        UniqueConstraint('sources_id', 'provenance_id', name='_zp_sources_provenance_uc'),
    )

    sources_id = sa.Column(
        sa.ForeignKey('source_lists.id', ondelete='CASCADE', name='zero_points_source_list_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the source list this zero point is associated with. ",
    )

    sources = orm.relationship(
        'SourceList',
        lazy='selectin',
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        doc="The source list this zero point is associated with. ",
    )

    image = association_proxy( "sources", "image" )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='zero_points_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this zero point. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this zero point. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this zero point. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this zero point. "
        )
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
        default=None,
        index=False,
        doc="Pixel radii of apertures whose aperture corrections are in aper_cors."
    )

    aper_cors = sa.Column(
        ARRAY( sa.REAL, zero_indexes=True ),
        nullable=True,
        default=None,
        index=False,
        doc=( "Aperture corrections for apertures with radii in aper_cor_radii.  Defined so that "
              "mag = -2.5*log10(adu_aper) + zp + aper_cor, where adu_aper is the number of ADU "
              "in the aperture with the specified radius.  There is a built-in approximation that a "
              "single aperture applies across the entire image, which should be OK given that the "
              "pipeline isn't expected to have photometry to better than a couple of percent." )
    )

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

        raise ValueError( f"No aperture correction tabulated for aperture radius within 0.01 pixels of {rad}" )

    def get_upstreams(self, session=None):
        """Get the extraction SourceList and WorldCoordinates used to make this ZeroPoint"""
        from models.provenance import Provenance
        with SmartSession(session) as session:
            source_list = session.scalars(sa.select(SourceList).where(SourceList.id == self.sources_id)).all()

            wcs_prov_id = None
            for prov in self.provenance.upstreams:
                if prov.process == "astro_cal":
                    wcs_prov_id = prov.id
            wcs = []
            if wcs_prov_id is not None:
                wcs = session.scalars(sa.select(WorldCoordinates) 
                                    .where(WorldCoordinates.provenance 
                                            .has(Provenance.id == wcs_prov_id))).all()

        return source_list + wcs
    
    def get_downstreams(self, session=None):
        """Get the downstreams of this ZeroPoint"""
        from models.image import Image
        from models.provenance import Provenance
        with SmartSession(session) as session:
            subs = session.scalars(sa.select(Image)
                                    .where(Image.provenance
                                            .has(Provenance.upstreams
                                                .any(Provenance.id == self.provenance.id)))).all()
        return subs
