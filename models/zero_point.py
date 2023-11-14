import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.associationproxy import association_proxy

from models.base import Base, AutoIDMixin


class ZeroPoint(Base, AutoIDMixin):
    __tablename__ = 'zero_points'

    source_list_id = sa.Column(
        sa.ForeignKey('source_lists.id', ondelete='CASCADE', name='zero_points_source_list_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the source list this zero point is associated with. "
    )

    source_list = orm.relationship(
        'SourceList',
        lazy='selectin',
        doc="The source list this zero point is associated with. "
    )

    image = association_proxy( "source_list", "image" )

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
        sa.Float,
        nullable=False,
        index=False,
        doc="Zeropoint: -2.5*log10(adu_psf) + zp = mag"
    )

    dzp = sa.Column(
        sa.Float,
        nullable=False,
        index=False,
        doc="Uncertainty on zp"
    )

    aper_cor_radii = sa.Column(
        sa.ARRAY( sa.REAL ),
        nullable=True,
        default=None,
        index=False,
        doc="Pixel radii of apertures whose aperture corrections are in aper_cors."
    )

    aper_cors = sa.Column(
        sa.ARRAY( sa.REAL ),
        nullable=True,
        default=None,
        index=False,
        doc=( "Aperture corrections for apertures with radii in aper_cor_radii.  Defined so that "
              "mag = -2.5*log10(adu_aper) + zp + aper_cor, where adu_aper is the number of ADU "
              "in the aperture with the specfiied radius.  There is a built-in approximation that a "
              "single aperture applies across the entire image, which should be OK given that the "
              "pipeline isn't expected to have photometry to better than a couple of percent." )
    )

    def get_aper_cor( self, rad ):
        """Return the aperture correction for a given aperture radius in pixels.

        Requires rad to be within 0.01 pixels of one of the tabluated
        aperture corrections.  If the requested one isn't found, will
        raise a ValueError.

        Parameteters
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
