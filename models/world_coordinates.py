
import sqlalchemy as sa
from sqlalchemy import orm

from astropy.wcs import WCS
from astropy.io import fits

from models.base import Base, AutoIDMixin


class WorldCoordinates(Base, AutoIDMixin):
    __tablename__ = 'world_coordinates'

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

    source_list_id = sa.Column(
        sa.ForeignKey('source_lists.id', ondelete='CASCADE', name='world_coordinates_source_list_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the source list this world coordinate system is associated with. "
    )

    source_list = orm.relationship(
        'SourceList',
        doc="The source list this world coordinate system is associated with. "
    )

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

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self._wcs = None

    @orm.reconstructor
    def init_on_load( self ):
        Base.init_on_load( self )
        self._wcs = None
