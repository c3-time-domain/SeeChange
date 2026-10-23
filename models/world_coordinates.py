import os
import textwrap
import pathlib

import numpy as np

from psycopg import sql
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import CheckConstraint, UniqueConstraint
from sqlalchemy.ext.declarative import declared_attr

from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs import utils

from models.base import ( Base, SeeChangeBase, SmartSession, PGDB,
                          UUIDMixin, HasBitFlagBadness, FileOnDiskMixin, SpatiallyIndexed, FourCornersWithGood )
from models.enums_and_bitflags import catalog_match_badness_inverse
from models.image import Image
from models.source_list import SourceList
from improc.tools import strip_wcs_keywords
from util.logger import SCLogger


class WorldCoordinates(Base, UUIDMixin, FileOnDiskMixin, HasBitFlagBadness, SpatiallyIndexed, FourCornersWithGood ):
    __tablename__ = 'world_coordinates'

    @declared_attr
    def __table_args__(cls):  # noqa: N805
        return (
            CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
                               '(md5sum_components IS NULL OR array_position(md5sum_components, NULL) IS NOT NULL))',
                               name=f'{cls.__tablename__}_md5sum_check' ),
            UniqueConstraint('sources_id', 'provenance_id', name='_wcs_source_list_provenance_uc' ),
            sa.Index(f"{cls.__tablename__}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
        )

    sources_id = sa.Column(
        sa.ForeignKey('source_lists._id', ondelete='CASCADE', name='world_coordinates_source_list_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the source list this world coordinate system is associated with. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances._id', ondelete="CASCADE", name='wcs_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the provenance of this wcs."
    )


    @property
    def wcs( self ):
        if self._wcs is None and self.filepath is not None:
            self.load()
        return self._wcs

    @wcs.setter
    def wcs( self, value ):
        self._wcs = value

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__( self, **kwargs )
        HasBitFlagBadness.__init__(self)
        SeeChangeBase.__init__( self )
        self._wcs = None

        # manually set all properties (columns or not)
        self.set_attributes_from_dict(kwargs)

    def _fill_bogus_coordinate_fields( self, image=None, ra=-999., dec=-999.,
                                       minra=-999., maxra=-999., mindec=-999., maxdec=-999. ):
        """This is used in tests to make sure some fields aren't NULL."""

        if image is not None:
            self.ra = image.ra
            self.dec = image.dec
            self.calculate_coordinates()
            for radec in [ 'ra', 'dec' ]:
                for good in [ '', 'good_' ]:
                    for minmax in [ 'min', 'max' ]:
                        setattr( self, f'{good}{minmax}{radec}', getattr( image, f'{minmax}{radec}' ) )
                for good in [ 'corner', 'good' ]:
                    for corner in [ '00', '01', '10', '11' ]:
                        setattr( self, f'{radec}_{good}_{corner}', getattr( image, f'{radec}_corner_{corner}' ) )
        else:
            self.ra = ra
            self.dec = dec
            if ( self.ra >= 0. ) and ( self.ra < 360. ) and ( self.dec >= -90. ) and ( self.dec <= 90. ):
                self.calculate_coordinates()
            for good in [ '', 'good_' ]:
                setattr( self, f'{good}minra', minra )
                setattr( self, f'{good}maxra', maxra )
                setattr( self, f'{good}mindec', mindec )
                setattr( self, f'{good}maxdec', maxdec )
            for good in [ 'corner', 'good' ]:
                setattr( self, f'ra_{good}_00', minra )
                setattr( self, f'ra_{good}_01', minra )
                setattr( self, f'ra_{good}_10', maxra )
                setattr( self, f'ra_{good}_11', maxra )
                setattr( self, f'dec_{good}_00', mindec )
                setattr( self, f'dec_{good}_01', maxdec )
                setattr( self, f'dec_{good}_10', mindec )
                setattr( self, f'dec_{good}_11', maxdec )


    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object"""
        return catalog_match_badness_inverse

    @orm.reconstructor
    def init_on_load( self ):
        SeeChangeBase.init_on_load( self )
        FileOnDiskMixin.init_on_load( self )
        self._wcs = None

    def get_pixel_scale(self):
        """Calculate the mean pixel scale using the WCS, in units of arcseconds per pixel."""
        if self.wcs is None:
            return None
        pixel_scales = utils.proj_plane_pixel_scales(self.wcs)  # the scale in x and y direction
        return np.mean(pixel_scales) * 3600.0


    def set_corners_from_wcs( self, image=None, width=None, height=None, setradec=False, mask=None ):
        """Update the WorldCoordinates four corners (and optionally, RA/Dec)

        Parameters
        ----------
          image: Image, default None
             Can use this instead of giving width and height.  Must pass
             one of image, mask, or (width and height).

           width, height: int, default None
             Size of the image, so that we can figure out where the
             corners are.  Can omit this if image is not None or mask is
             not None.

           mask: 2d numpy array, default None
             If given, where values are not 0, those are considered bad
             pixels.  The "good" corner and min/max fields will be set
             such that all pixels on the image outside that region are
             flagged as bad.  Useful on chips where big swaths are bad
             (e.g. if the left 200 pixels are vignetted, or something).

             If this is not given, the "good" corners and min/max are
             not set.

           setradec: bool, default False
             Set this to True to also update ra and dec (plus ecliptic
             and galactic coordinates), not just the corners and
             min/max.

        """

        mskwid = mask.shape[1] if mask is not None else None
        mskhei = mask.shape[0] if mask is not None else None
        imwid = None
        imhei = None
        if image is not None:
            if isinstance( image, Image ):
                imwid = image.width
                imhei = image.height
            else:
                imwid = image.shape[1]
                imhei = image.shape[0]

        if ( mskwid is not None ) and ( imwid is not None ) and ( ( imwid != mskwid ) or ( imhei != mskhei ) ):
            raise ValueError( f"Image is {imwid}×{imhei}, but mask is {mskwid}×{mskhei}, which is inconsistent." )

        if ( width is not None ) and ( ( ( imwid is not None ) and ( imwid != width ) )
                                       or
                                       ( ( mskwid is not None ) and ( mskwid != width ) )
                                      ):
            raise ValueError( f"You passed width {width}, which is not consistent with the image and/or mask" )
        if ( height is not None ) and ( ( ( imhei is not None ) and ( imhei != height ) )
                                        or
                                        ( ( mskhei is not None ) and ( mskhei != height ) )
                                       ):
            raise ValueError( f"You passed height {height}, which is not consistent with the image and/or mask" )

        width = width if width is not None else imwid if imwid is not None else None
        height = height if height is not None else imhei if imhei is not None else None

        super().set_corners_from_wcs( wcs=self.wcs, width=width, height=height, mask=mask, setradec=setradec )


    def trim( self, x0, x1, y0, y1 ):
        newwcs = WorldCoordinates()
        newwcs.wcs = self.wcs[ y0:y1, x0:x1 ]
        return newwcs


    def save( self, filename=None, image=None, **kwargs ):

        """Write the WCS data to disk.

        Updates self.filepath

        Parameters
        ----------
          filename: str or Path, or None
             The path to the file to write, relative to the local store
             root.  Do not include the extension (e.g. '.psf') at the
             end of the name; that will be added automatically.
             If None, will call image.invent_filepath() to get a
             filestore-standard filename and directory.

          image: Image or None
             Ignored if filename is specified.  Otherwise, the Image to
             use in inventing the filepath.  If None, will try to load
             it from the database.  Use this for efficiency, or if you
             know the image isn't yet in the database.

        Additional arguments are passed on to FileOnDiskMixin.save
        """

        # ----- Make sure we have a path ----- #
        # if filename already exists, check it is correct and use

        if filename is not None:
            if not filename.endswith('.txt'):
                filename += '.txt'
            self.filepath = filename

        # if not, generate one
        else:
            if image is None:
                with SmartSession() as session:
                    image = ( session.query( Image )
                              .join( SourceList, SourceList.image_id==Image._id )
                              .filter( SourceList._id==self.sources_id )
                             ).first()
                if image is None:
                    raise RuntimeError( "Can't invent WorldCoordinates filepath; can't find corresponding image." )


            self.filepath = image.filepath if image.filepath is not None else image.invent_filepath()
            self.filepath += f'.wcs_{self.provenance_id[:6]}.txt'

        txtpath = pathlib.Path( self.local_path ) / self.filepath

        # ----- Get the header string to save and save ----- #
        header_txt = self.wcs.to_header( relax=True ).tostring(padding=False, sep='\\n' )

        if txtpath.exists():
            if not kwargs.get('overwrite', True):
                # raise the error if overwrite is explicitly set False
                raise FileExistsError( f"{txtpath} already exists, cannot save." )

        with open( txtpath, "w") as ofp:
            ofp.write( header_txt )

        # ----- Write to the archive ----- #
        FileOnDiskMixin.save( self, txtpath, **kwargs )

    def load( self, download=True, always_verify_md5=False, txtpath=None ):
        """Load this wcs from the file.

        updates self.wcs.

        Parameters
        ----------
        txtpath: str, Path, or None
            File to read. If None, will load the file returned by self.get_fullpath()
        """

        if txtpath is None:
            txtpath = self.get_fullpath( download=download, always_verify_md5=always_verify_md5, nofile=False )

        if not os.path.isfile(txtpath):
            raise OSError(f'WCS file is missing at {txtpath}')

        with open( txtpath ) as ifp:
            headertxt = ifp.read()
            self.wcs = WCS( fits.Header.fromstring( headertxt , sep='\\n' ))


    def export_image( self, ofpath, image=None, which='image', pgdb=None, overwrite=False ):
        """Write the FITS image with the header having this WCS.

        If you don't pass image, then it only works if everything is
        already saved to the database.

        """

        if image is None:
            with PGDB( pgdb, dictcursor=True ) as pgdb:
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT i.* FROM images i
                    INNER JOIN source_lists s ON s.image_id=i._id
                    INNER JOIN world_coordinates w ON w.sources_id=s._id
                    WHERE w._id={me}
                    """
                ) ).format( me=self.id )
                rows = pgdb.execute( q )
                if len(rows) > 1:
                    raise RuntimeError( "This should never happen." )
                elif len(rows) == 0:
                    SCLogger.warning( "Could not find image for wcs; maybe it's not yet saved to the database." )
                    return None

            image = Image( **(rows[0]) )

        data = ( image.data if which == 'image'
                 else image.flags if which == 'flags'
                 else image.weight if which == 'weight'
                 else None )
        if data is None:
            SCLogger.error( f"Couldn't find array {which} for image {image.filepath}" )
            return None

        hdr = fits.Header( image.header )
        strip_wcs_keywords( hdr )
        hdr.extend( self.wcs.to_header( relax=True ) )

        fits.writeto( ofpath, data, hdr, overwrite=overwrite )


    def free(self):
        """Free loaded world coordinates memory.

        Wipe out the _wcs text field, freeing a small amount of memory.
        Depends on python garbage collection, so if there are other
        references to those objects, the memory won't actually be freed.
        """
        self._wcs = None

    def get_upstream_ids(self, pgdb=None):
        """"Get the id of the source list that was used to make this wcs."""
        return [ ( SourceList, self.sources_id ) ]

    def get_downstream_ids(self, pgdb=None):
        """Get downstreams of this wcs.

        This will include zeropoints and (maybe) images (which were trimmed or warped).

        """
        downstreams = []

        from models.zero_point import ZeroPoint
        seen = set()
        with PGDB() as pgdb:
            rows, _cols = pgdb.execute( sql.SQL( "SELECT _id FROM zero_points WHERE wcs_id={wcs}" )
                                 .format( wcs=self.id ) )
            for row in rows:
                if row[0] not in seen:
                    downstreams.append( ( ZeroPoint, row[0] ) )
                    seen.add( row[0] )

            rows, _cols = pgdb.execute(
                sql.SQL( "SELECT image_id FROM image_trim_parent WHERE parent_wcs_id={wcs}" )
                .format( wcs=self.id ) )
            for row in rows:
                if row[0] not in seen:
                    downstreams.append( ( Image, row[0] ) )
                    seen.add( row[0] )

            rows, _cols = pgdb.execute(
                sql.SQL( "SELECT warped_id FROM image_warp_parent WHERE target_wcs_id={wcs}" )
                .format( wcs=self.id ) )
            for row in rows:
                if row[0] not in seen:
                    downstreams.append( ( Image, row[0] ) )
                    seen.add( row[0] )

        return downstreams
