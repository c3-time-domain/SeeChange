import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY

from models.base import Base, SmartSession, UUIDMixin, HasBitFlagBadness, FileOnDiskMixin, SeeChangeBase
from models.enums_and_bitflags import catalog_match_badness_inverse
from models.world_coordinates import WorldCoordinates
from models.image import Image
from models.source_list import SourceList, SourceListSibling


class ZeroPoint(SourceListSibling, Base, UUIDMixin, HasBitFlagBadness):
    __tablename__ = 'zero_points'

    sources_id = sa.Column(
        sa.ForeignKey('source_lists._id', ondelete='CASCADE', name='zero_points_source_list_id_fkey'),
        nullable=False,
        index=True,
        unique=True,
        doc="ID of the source list this zero point is associated with. ",
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

        raise ValueError( f"No aperture correction tabulated for sources {self.sources_id} "
                          f"for apertures within 0.01 pixels of {rad}; "
                          f"available apertures are {self.aper_cor_radii}" )

    # ======================================================================
    # The fields below are things that we've deprecated; these definitions
    #   are here to catch cases in the code where they're still used

    @property
    def sources( self ):
        raise RuntimeError( f"Don't use ZeroPoint.sources, use sources_id" )

    @sources.setter
    def sources( self, val ):
        raise RuntimeError( f"Don't use ZeroPoint.sources, use sources_id" )

    @property
    def image( self ):
        raise RuntimeError( f"ZeroPoint.image is deprecated, don't use it" )

    @image.setter
    def image( self, val ):
        raise RuntimeError( f"ZeroPoint.image is deprecated, don't use it" )

    @property
    def provenance_id( self ):
        raise RuntimeError( f"ZeroPoint.provenance_id is deprecated; get provenance from sources" )

    @provenance_id.setter
    def provenance_id( self, val ):
        raise RuntimeError( f"ZeroPoint.provenance_id is deprecated; get provenance from sources" )

    @property
    def provenance( self ):
        raise RuntimeError( f"ZeroPoint.provenance is deprecated; get provenance from sources" )

    @provenance.setter
    def provenance( self, val ):
        raise RuntimeError( f"ZeroPoint.provenance is deprecated; get provenance from sources" )
