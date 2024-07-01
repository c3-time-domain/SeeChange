import pathlib
import numpy as np
import os

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.schema import UniqueConstraint

from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs import utils

from models.base import Base, SmartSession, AutoIDMixin, HasBitFlagBadness, FileOnDiskMixin, SeeChangeBase
from models.enums_and_bitflags import catalog_match_badness_inverse
from models.image import Image
from models.source_list import SourceList


class WorldCoordinates(Base, AutoIDMixin, FileOnDiskMixin, HasBitFlagBadness):
    __tablename__ = 'world_coordinates'

    __table_args__ = (
        UniqueConstraint('sources_id', 'provenance_id', name='_wcs_sources_provenance_uc'),
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
        lazy='selectin',
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
    
    def get_upstreams(self, session=None):
        """Get the extraction SourceList that was used to make this WorldCoordinates"""
        with SmartSession(session) as session:
            return session.scalars(sa.select(SourceList).where(SourceList.id == self.sources_id)).all()
        
    def get_downstreams(self, session=None, siblings=False):
        """Get the downstreams of this WorldCoordinates.

        If siblings=True  then also include the SourceList, PSF, background object and ZP
        that were created at the same time as this WorldCoordinates.
        """
        from models.source_list import SourceList
        from models.psf import PSF
        from models.background import Background
        from models.zero_point import ZeroPoint
        from models.provenance import Provenance

        with (SmartSession(session) as session):
            output = []
            if self.provenance is not None:
                subs = session.scalars(
                    sa.select(Image).where(
                        Image.provenance.has(Provenance.upstreams.any(Provenance.id == self.provenance.id)),
                        Image.upstream_images.any(Image.id == self.sources.image_id),
                    )
                ).all()
                output += subs

            if siblings:
                sources = session.scalars(sa.select(SourceList).where(SourceList.id == self.sources_id)).all()
                if len(sources) > 1:
                    raise ValueError(
                        f"Expected exactly one SourceList for WorldCoordinates {self.id}, but found {len(sources)}."
                    )

                output.append(sources[0])

                psf = session.scalars(
                    sa.select(PSF).where(
                        PSF.image_id == sources.image_id, PSF.provenance_id == self.provenance_id
                    )
                ).all()

                if len(psf) > 1:
                    raise ValueError(f"Expected exactly one PSF for WorldCoordinates {self.id}, but found {len(psf)}.")

                output.append(psf[0])

                bgs = session.scalars(
                    sa.select(Background).where(
                        Background.image_id == sources.image_id, Background.provenance_id == self.provenance_id
                    )
                ).all()

                if len(bgs) > 1:
                    raise ValueError(
                        f"Expected exactly one Background for WorldCoordinates {self.id}, but found {len(bgs)}."
                    )

                output.append(bgs[0])

                zp = session.scalars(sa.select(ZeroPoint).where(ZeroPoint.sources_id == sources.id)).all()

                if len(zp) > 1:
                    raise ValueError(
                        f"Expected exactly one ZeroPoint for WorldCoordinates {self.id}, but found {len(zp)}."
                    )
                output.append(zp[0])

        return output

    def save( self, filename=None, **kwargs ):
        """Write the WCS data to disk.
        Updates self.filepath
        Parameters
        ----------
          filename: str or path
             The path to the file to write, relative to the local store
             root.  Do not include the extension (e.g. '.psf') at the
             end of the name; that will be added automatically.
             If None, will call image.invent_filepath() to get a
             filestore-standard filename and directory.
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
            if self.provenance is None:
                raise RuntimeError("Can't invent a filepath for the WCS without a provenance")
            
            if self.image.filepath is not None:
                self.filepath = self.image.filepath
            else:
                self.filepath = self.image.invent_filepath()

            self.filepath += f'.wcs_{self.provenance.id[:6]}.txt'

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
            txtpath = self.get_fullpath( download=download, always_verify_md5=always_verify_md5)

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
