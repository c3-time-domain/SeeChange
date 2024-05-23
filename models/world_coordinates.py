import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.schema import UniqueConstraint

from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs import utils

from models.base import Base, SmartSession, AutoIDMixin, HasBitFlagBadness, FileOnDiskMixin, SeeChangeBase
from models.enums_and_bitflags import catalog_match_badness_inverse
from models.source_list import SourceList


class WorldCoordinates(Base, AutoIDMixin, HasBitFlagBadness):
    __tablename__ = 'world_coordinates'

    __table_args__ = (
        UniqueConstraint('sources_id', 'provenance_id', name='_wcs_sources_provenance_uc'),
    )

    # This is a little profligate.  There will eventually be millions of
    # images, which means that there will be gigabytes of header data
    # stored in the relational database.  (One header excerpt is about
    # 4k.)  It's not safe to assume we know exactly what keywords
    # astropy.wcs.WCS will produce, as there may be new FITS standard
    # extensions etc., and astropy doesn't document the keywords.
    #
    # Another option would be to parse all the keywords into a dict of {
    # string: (float or string) } and store them as a JSONB; that would
    # reduce the size pretty substantially, but it would still be
    # roughly a KB for each header, so the consideration is similar.
    # (It's also more work to implement....)
    #
    # Yet another option is to store the WCS in an external file, but
    # now we're talking something awfully small (a few kB) for this HPC
    # filesystems.
    #
    # Even yet another option that we won't do short term because it's
    # WAY too much effort is to have an additional nosql database of
    # some sort that is designed for document storage (which really is
    # what this is here).
    #
    # For now, we'll be profliate with the database, and hope we don't
    # regret it later.
    header_excerpt = sa.Column(
        sa.Text,
        nullable=False,
        index=False,
        doc="Text that containts FITS header cards (ASCII, \n-separated) with the header that defines this WCS"
    )

    sources_id = sa.Column(
        sa.ForeignKey('source_lists.id', ondelete='CASCADE', name='world_coordinates_source_list_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the source list this world coordinate system is associated with. "
    )

    sources = orm.relationship(
        'SourceList',
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        doc="The source list this world coordinate system is associated with. "
    )

    image = association_proxy( "sources", "image" )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='world_coordinates_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this world coordinate system. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this world coordinate system. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this world coordinate system. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this world coordinate system. "
        )
    )

    @property
    def wcs( self ):
        if self._wcs is None:
            self._wcs = WCS( fits.Header.fromstring( self.header_excerpt, sep='\n' ) )
        return self._wcs

    @wcs.setter
    def wcs( self, value ):
        self._wcs = value
        self.header_excerpt = value.to_header().tostring( sep='\n', padding=False )

    def __init__(self, *args, **kwargs):
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values
        self._wcs = None

        # manually set all properties (columns or not)
        self.set_attributes_from_dict(kwargs)

    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)

    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object"""
        return catalog_match_badness_inverse

    @orm.reconstructor
    def init_on_load( self ):
        Base.init_on_load( self )
        self._wcs = None

    def get_pixel_scale(self):
        """Calculate the mean pixel scale using the WCS, in units of arcseconds per pixel."""
        if self.wcs is None:
            return None
        pixel_scales = utils.proj_plane_pixel_scales(self.wcs)  # the scale in x and y direction
        return np.mean(pixel_scales) * 3600.0
    
    def get_upstreams(self, session=None):
        """Get the extraction SourceList that was used to make this WorldCoordinates"""
        with SmartSession(session) as session:
            return session.scalars(sa.select(SourceList).where(SourceList.id == self.sources_id)).all()
        
    def get_downstreams(self, session=None):
        """Get the downstreams of this WorldCoordinates"""
        # get the ZeroPoint that uses the same SourceList as this WCS
        from models.zero_point import ZeroPoint
        from models.image import Image
        from models.provenance import Provenance
        with SmartSession(session) as session:
            zps = session.scalars(sa.select(ZeroPoint) 
                                  .where(ZeroPoint.provenance 
                                         .has(Provenance.upstreams 
                                              .any(Provenance.id == self.provenance.id)))).all()

            subs = session.scalars(sa.select(Image)
                                   .where(Image.provenance
                                          .has(Provenance.upstreams
                                               .any(Provenance.id == self.provenance.id)))).all()

        downstreams = zps + subs
        return downstreams
    
