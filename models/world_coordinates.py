import pathlib
import numpy as np
import os

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import CheckConstraint, UniqueConstraint
from sqlalchemy.ext.declarative import declared_attr

from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs import utils

from models.base import Base, SmartSession, UUIDMixin, HasBitFlagBadness, FileOnDiskMixin, SeeChangeBase
from models.enums_and_bitflags import catalog_match_badness_inverse
from models.image import Image
from models.source_list import SourceList


class WorldCoordinates(Base, UUIDMixin, FileOnDiskMixin, HasBitFlagBadness):
    __tablename__ = 'world_coordinates'

    @declared_attr
    def __table_args__(cls):  # noqa: N805
        return (
            CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
                               '(md5sum_components IS NULL OR array_position(md5sum_components, NULL) IS NOT NULL))',
                               name=f'{cls.__tablename__}_md5sum_check' ),
            UniqueConstraint('sources_id', 'provenance_id', name='_wcs_source_list_provenance_uc' )
        )

    sources_id = sa.Column(
        sa.ForeignKey('source_lists._id', ondelete='CASCADE', name='world_coordinates_source_list_id_fkey'),
        nullable=False,
        index=True,
        unique=True,
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
        header_txt = self.wcs.to_header().tostring(padding=False, sep='\\n' )

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

    def free(self):
        """Free loaded world coordinates memory.

        Wipe out the _wcs text field, freeing a small amount of memory.
        Depends on python garbage collection, so if there are other
        references to those objects, the memory won't actually be freed.
        """
        self._wcs = None

    def get_upstreams(self, session=None):
        """Get the source list that was used to make this wcs."""
        with SmartSession(session) as session:
            return session.scalars( sa.select(SourceList).where( SourceList._id==self.sources_id ) ).all()

    def get_downstreams(self, session=None):
        """Get immediate downstreams of this wcs, which are zeropoints."""
        from models.zero_point import ZeroPoint
        with SmartSession(session) as session:
            return session.scalars( sa.select(ZeroPoint).where( ZeroPoint.wcs_id==self._id ) ).all()
