import uuid
from contextlib import contextmanager

import sqlalchemy as sa
import sqlalchemy.types
from sqlalchemy import orm
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import CheckConstraint

import util.ldac
from util.util import ensure_file_does_not_exist
from util.logger import SCLogger
from models.base import Base, SeeChangeBase, UUIDMixin, FileOnDiskMixin, SpatiallyIndexed, FourCorners
from models.base import Psycopg2Connection
from models.enums_and_bitflags import CatalogExcerptFormatConverter, CatalogExcerptOriginConverter
from sqlalchemy.dialects.postgresql import ARRAY


class CatalogExcerpt(Base, UUIDMixin, FileOnDiskMixin, SpatiallyIndexed, FourCorners):
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

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
                             '(md5sum_components IS NULL OR array_position(md5sum_components, NULL) IS NOT NULL))',
                             name=f'{cls.__tablename__}_md5sum_check' ),
            sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
        )


    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( str(CatalogExcerptFormatConverter.convert('fitsldac')) ),
        doc="Format of the file on disk.  Currently only fitsldac is supported. "
            "Saved as intetger but is converted to string when loaded."
    )

    @hybrid_property
    def format( self ):
        return CatalogExcerptFormatConverter.convert(self._format)

    @format.expression
    def format( cls ):  # noqa: N805
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
    def origin( cls ):  # noqa: N805
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
            self._hdr, self._data = util.ldac.get_table_from_ldac( self.get_fullpath( nofile=False ) )
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
        ARRAY(sa.Text, zero_indexes=True),
        nullable=False,
        server_default='{}',
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

    def get_downstreams( self, session=None, siblings=True ):
        """CatalogExcerpt has no downstreams """
        return []

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
        if origin == 'gaia_dr3':
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
            catexp.minra = catexp.ra_corner_00
            catexp.maxra = catexp.ra_corner_11
            catexp.mindec = catexp.dec_corner_00
            catexp.maxdec = catexp.dec_corner_11
            catexp.ra = ( catexp.ra_corner_00 + catexp.ra_corner_10 ) / 2.
            catexp.dec = ( catexp.dec_corner_00 + catexp.dec_corner_01 ) / 2.
            catexp.calculate_coordinates()
        else:
            SCLogger.warning( f"spatial coordinates and min/max mag not set in CatalogExcerpt with origin {origin}" )

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


class GaiaDR3DownloadLock(Base, UUIDMixin):
    """See comments in catalog_tools.py::fetch_gaia_dr3_excerpt"""

    __tablename__ = 'gaiadr3_downloadlock'

    # Not bothering to index the columns because I don't expect this
    #   table to ever have very many rows at one time.
    minra = sa.Column( sa.REAL, nullable=False, index=False, doc="Min RA" )
    maxra = sa.Column( sa.REAL, nullable=False, index=False, doc="Max RA" )
    mindec = sa.Column( sa.REAL, nullable=False, index=False, doc="Min RA" )
    maxdec = sa.Column( sa.REAL, nullable=False, index=False, doc="Max RA" )
    minmag = sa.Column( sa.REAL, nullable=True, index=False, doc="min mag" )
    maxmag = sa.Column( sa.REAL, nullable=True, index=False, doc="max mag" )

    @classmethod
    @contextmanager
    def lock( cls, minra, maxra, mindec, maxdec, minmag, maxmag ):
        """Acquire a lock on a specific set of parameters for Gaia DR3.

        Here so that we can avoid the race condition of multiple
        processes downloading the same Gaia catalog at the same time.

        Call this in a with statement ("with GaiaDR3DownloadLock(...)").
        You will *not* be holding any database locks inside the with
        block, it just means that there's a row on a specialized table
        that will stop other processes from trying to download the same
        catalog at the same time.

        """

        def sqlconds( minra, maxra, minmag, maxmag ):
            q = ( "WHERE ABS(minra - %(minra)s) < 0.005 "
                  "  AND ABS(maxra - %(maxra)s) < 0.005 "
                  "  AND ABS(mindec - %(mindec)s) < 0.005 "
                  "  AND ABS(maxdec - %(maxdec)s) < 0.005 " )
            subdict = { 'minra': minra, 'maxra': maxra,
                        'mindec': mindec, 'maxdec': maxdec }
            if minmag is None:
                q += "  AND minmag IS NULL "
            else:
                q += "  AND ABS(minmag - %(minmag)s) < 0.1 "
                subdict['minmag'] = minmag
            if maxmag is None:
                q += "  AND maxmag IS NULL "
            else:
                q += "  AND ABS(maxmag - %(maxmag)s) < 0.1 "
                subdict['maxmag'] = maxmag

            return q, subdict

        try:
            with Psycopg2Connection() as conn:
                gotit = False
                cursor = None
                while not gotit:
                    cursor = conn.cursor()
                    cursor.execute( "LOCK TABLE gaiadr3_downloadlock" )
                    q = "SELECT * FROM gaiadr3_downloadlock "
                    conds, subdict = sqlconds( minra, maxra, minmag, maxmag )
                    q += conds
                    cursor.execute( q, subdict )
                    rows = cursor.fetchall()
                    if len(rows) == 0:
                        gotit = True
                    else:
                        # The row existed, so somebody else is doing this.
                        # Release the lock, wait, try again
                        conn.rollback()
                        SCLogger.debug( "...waiting for gaiadr3 downloadlock..." )

                # If we get here, we're holding a lock
                q = ( "INSERT INTO gaiadr3_downloadlock(_id,minra,maxra,mindec,maxdec,minmag,maxmag) "
                      "VALUES (%(_id)s,%(minra)s,%(maxra)s,%(mindec)s,%(maxdec)s,%(minmag)s,%(maxmag)s)" )
                cursor.execute( q, { '_id': uuid.uuid4(),
                                     'minra': minra, 'maxra': maxra,
                                     'mindec': mindec, 'maxdec': maxdec,
                                     'minmag': minmag, 'maxmag': maxmag } )
                # This will add the row to the table and release the lock
                conn.commit()

            yield True

        finally:
            # This is in a "finally" so that if the thing that we
            #  yielded to gets an exception, this stuff still gets
            #  executed.
            with Psycopg2Connection() as conn:
                cursor = conn.cursor()
                q = "DELETE FROM gaiadr3_downloadlock "
                conds, subdict = sqlconds( minra, maxra, minmag, maxmag )
                q += conds
                cursor.execute( q, subdict )
                conn.commit()
