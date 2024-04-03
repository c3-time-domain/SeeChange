import numpy

import sqlalchemy as sa
import sqlalchemy.types
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property

import util.ldac
from util.util import ensure_file_does_not_exist
from models.base import Base, SeeChangeBase, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed, FourCorners, _logger
from models.enums_and_bitflags import CatalogExcerptFormatConverter, CatalogExcerptOriginConverter


class CatalogExcerpt(Base, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed, FourCorners):
    """A class for storing catalog excerpts.

    The primary use for this is a cache.  For instance, for astrometry,
    we will go to a server to get stars from the gaia catalog.  If we
    see the same field many times, we can speed things up, and reduce
    the number of connections to the external catalog, by caching the
    result.

    Format assumptions:
      fitsldac : at least the following columns are included:
        X_WORLD  : ra in decimal degrees
        Y_WORLD  : dec in decimal degrees
        ERRA_WORLD : uncertainty on ra
        ERRB_WORLD : uncertainty on dec

    """

    __tablename__ = 'catalog_excerpts'

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=CatalogExcerptFormatConverter.convert('fitsldac'),
        doc="Format of the file on disk.  Currently only fitsldac is supported. "
            "Saved as intetger but is converted to string when loaded."
    )

    @hybrid_property
    def format( self ):
        return CatalogExcerptFormatConverter.convert(self._format)

    @format.expression
    def format( cls ):
        return sa.case( CatalogExcerptFormatConverter.dict, value=cls._format )

    @format.setter
    def format( self, value ):
        self._format = CatalogExcerptFormatConverter.convert( value )

    _origin = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        doc="Where this catalog excerpt came from.  Saved as an integer, converted to string when loaded."
    )

    @hybrid_property
    def origin( self ):
        return CatalogExcerptOriginConverter.convert( self._origin )

    @origin.expression
    def origin( cls ):
        return sa.case( CatalogExcerptOriginConverter.dict, value=cls._origin )

    @origin.setter
    def origin( self, value ):
        self._origin = CatalogExcerptOriginConverter.convert( value )

    @property
    def data( self ):
        """The underlying table.  This is format- and origin-specific."""
        if self._data is None:
            if self.format != 'fitsldac':
                raise ValueError( f"Don't know how to read a CatalogExcerpt of type {self.format}" )
            self._hdr, self._data = util.ldac.get_table_from_ldac( self.get_fullpath() )
        return self._data

    num_items = sa.Column(
        sa.Integer,
        index=True,
        nullable=False,
        doc="Number of items in the catalog"
    )

    minmag = sa.Column(
        sa.REAL,
        nullable=True,
        index=True,
        doc="Minimum magnitude cut used in making the excerpt"
    )

    maxmag = sa.Column(
        sa.REAL,
        nullable=True,
        index=True,
        doc="Maximum magnitude cut used in making the excerpt"
    )

    filters = sa.Column(
        sa.ARRAY(sa.Text),
        nullable=False,
        default=[],
        doc=( "Filters covered by the catalog; names of the filters will be "
              "standard for the catalog source, not globally standard." )
    )

    # Can't name these just "ra" and "dec" because those already exists
    # from SpatiallyIndexed
    @property
    def object_ras( self ):
        """A numpy array: ra in decimal degrees of the catalog excerpt objects."""

        if self.format != 'fitsldac':
            raise ValueError( f"Don't know how to get ra of a CatalogExcerpt of type {self.format}" )
        if self.data is None:
            raise RuntimeError( "Failed to read data file for CatalogExcerpt" )
        return self.data[ 'X_WORLD' ].value

    @property
    def object_decs( self ):
        """A numpy array: dec in decimal degrees of the catalog excerpt objects."""

        if self.format != 'fitsldac':
            raise ValueError( f"Don't know how to get ra of a CatalogExcerpt of type {self.format}" )
        if self.data is None:
            raise RuntimeError( "Failed to read data file for CatalogExcerpt" )
        return self.data[ 'Y_WORLD' ].value

    def __init__(self, *args, **kwargs ):
        FileOnDiskMixin.__init__( self, *args, **kwargs )
        SeeChangeBase.__init__( self )  # don't pass kwargs as they could contain non-column key-values

        self._hdr = None
        self._data = None

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr( self, key ):
                setattr( self, key, value )

        self.calculate_coordinates()

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load( self )
        FileOnDiskMixin.init_on_load( self )

        self._hdr = None
        self._data = None

    @staticmethod
    def create_from_file( filepath, origin, format="fitsldac" ):
        """Create a CatalogExcerpt from a file on disk.  Use with care!

        Reads a file on disk and creates a CatalogExcerpt object from
        it.  Don't use this unless you really know what you're doing;
        instead, use pipeline utilities to create the relevant catalog
        excerpts.

        Parameters
        ----------
          filepath: str or Path
            Path to the file

          origin: str
            The origin of the catalog excerpt.  (See origin property.)

          format: str
            The format of the file to read.  Only fitsldac is currently supported.

        Returns
        -------
          CatalogExcerpt

        """

        if format != 'fitsldac':
            raise ValueError( f"Can only create catalog excerpts from fitsldac files, not {format}" )

        hdr, tbl = util.ldac.get_table_from_ldac( filepath, frame=1, imghdr_as_header=False )
        catexp = CatalogExcerpt( format=format, origin=origin, num_items=len(tbl) )
        catexp._data = tbl
        catexp._hdr = hdr
        catexp.num_items = len( tbl )
        if origin == 'GaiaDR3':
            catexp.filters = [ 'G', 'BP', 'RP' ]
            catexp.minmag = tbl[ 'MAG_G' ].min()
            catexp.maxmag = tbl[ 'MAG_G' ].max()
            catexp.ra_corner_00 = tbl[ 'X_WORLD' ].min()
            catexp.ra_corner_01 = catexp.ra_corner_00
            catexp.ra_corner_10 = tbl[ 'X_WORLD' ].max()
            catexp.ra_corner_11 = catexp.ra_corner_10
            catexp.dec_corner_00 = tbl[ 'Y_WORLD' ].min()
            catexp.dec_corner_10 = catexp.dec_corner_00
            catexp.dec_corner_01 = tbl[ 'Y_WORLD' ].max()
            catexp.dec_corner_11 = catexp.dec_corner_01
            catexp.ra = ( catexp.ra_corner_00 + catexp.ra_corner_10 ) / 2.
            catexp.dec = ( catexp.dec_corner_00 + catexp.dec_corner_01 ) / 2.
            catexp.calculate_coordinates()
        else:
            _logger.warning( f"spatial coordinates and min/max mag not set in CatalogExcerpt with origin {origin}" )

        return catexp

    def ds9_regfile( self, regfile, radius=2, color='red', width=2, clobber=True ):

        """Write a DS9 region file with the contents of this catalog excerpt.

        Parameters
        ----------
        regfile: str or Path
           output DS9 region file
        radius: float
           radius of circles to draw in arcsec
        color: str
           color of circles to draw
        width: float
           linewidth of circles to draw
        clobber: bool, default True
           If True, will overwrite an existing file

        """

        ensure_file_does_not_exist( regfile, delete=clobber )

        ras = self.object_ras
        decs = self.object_decs
        with open( regfile, "w" ) as ofp:
            for ra, dec, in zip( ras, decs ):
                ofp.write( f'icrs;circle({ra}d,{dec}d,{radius}") # color={color} width={width}\n' )
