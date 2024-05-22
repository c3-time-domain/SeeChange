import pathlib
import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.schema import UniqueConstraint

from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs import utils

from models.base import Base, SmartSession, SeeChangeBase, AutoIDMixin, HasBitFlagBadness, FileOnDiskMixin
from models.enums_and_bitflags import catalog_match_badness_inverse
from models.source_list import SourceList


class WorldCoordinates(Base, AutoIDMixin, FileOnDiskMixin, HasBitFlagBadness):
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
        lazy="selectin",
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

    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object"""
        return catalog_match_badness_inverse

    def __init__( self, *args, **kwargs ):
        FileOnDiskMixin.__init__( self, **kwargs )
        SeeChangeBase.__init__( self , *args, **kwargs)
        self._wcs = None

    @orm.reconstructor
    def init_on_load( self ):
        Base.init_on_load( self )
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
    
    def save( self, filename=None, **kwargs ):
        """Write the PSF to disk.

        May or may not upload to the archive and update the
        FileOnDiskMixin-included fields of this object based on the
        additional arguments that are forwarded to FileOnDiskMixin.save.

        For psfex-format psfs, this saves two files: the .psf file (the
        FITS file with the data), and the .psf.xml file (the XML file
        created by PSFex.)

        Parameters
        ----------
          filename: str or path
             The path to the file to write, relative to the local store
             root.  Do not include the extension (e.g. '.psf') at the
             end of the name; that will be added automatically for all
             extensions.  If None, will call image.invent_filepath() to get a
             filestore-standard filename and directory.

          Additional arguments are passed on to FileOnDiskMixin.save

        steps:
        - make sure we have a path (roughly done)
        - get the data we want to save (a fits header from astropy WCS)
        - write the data we want to save to the proper path
        - save the path to the archive with FODM.save

        """ 


        # ----- Make sure we have a path ----- #
        # check for the values required to save
        if False:
            raise RuntimeError( "must have all required data non-None")

        # if filename already exists, check it is correct and use
        if filename is not None:
            if not filename.endswith('.txt'):
                filename += '.txt'
            self.filepath = filename
        # if not, generate one
        else:
            if self.image.filepath is not None:
                self.filepath = self.image.filepath
            else:
                self.filepath = self.image.invent_filepath()

            if self.provenance is None:
                raise RuntimeError("Can't invent a filepath for the WCS without a provenance")
            self.filepath += f'.wcs_{self.provenance.id[:6]}.txt'
        
        txtpath = pathlib.Path( self.local_path ) / f'{self.filepath}'
        # self.filepath = self.image.invent_filepath()
        # self.filepath += f'.wcs_{self.provenance.id[:6]}'

        # ----- Get the header to save and save ----- #
        # breakpoint()
        
        header_txt = self.wcs.to_header().tostring(padding=False, sep='\\n' )

        with open( txtpath, "w") as ofp:
            ofp.write( header_txt )
        
        # ----- Write to the archive ----- #
        FileOnDiskMixin.save( self, txtpath, **kwargs )

    def load( self, download=True, always_verify_md5=False, txtpath=None ):
        """Load the data from the files into the _data, _header, and _info fields.

        Parameters
        ----------
          download : Bool, default True
            If True, download the files from the archive if they're not
            found in local storage.  Ignored if psfpath is not None.

          always_verify_md5 : Bool, default False
            If the file is found locally, verify the md5 of the file; if
            it doesn't match, re-get the file from the archive.  Ignored
            if psfpath is not None.

          psfpath : str or Path, default None
            If None, files will be read using the get_fullpath() method
            to get the right files form the local store and/or archive
            given the databse fields.  If not None, read _header and
            _data from this file.  (This exists so that this method may
            be used to load the data with a psf that's not yet in the
            database, without having to play games with the filepath
            field.)

          psfxmlpath : str or Path, default None
            Must be non-None if psfpath is non-None; the name of the
            .psf.xml file to read _info from.

        """

        # if self.format != 'psfex': # not applicable to wcs
        #     raise NotImplementedError( "Only know how to load psfex PSF files" )

        # breakpoint()
        if txtpath is None:
            txtpath = self.get_fullpath( download=download, always_verify_md5=always_verify_md5)

        if txtpath is None:
            raise ValueError("WCS object has no filepath locally or in archive.")
        
        # if ( psfpath is None ) != ( psfxmlpath is None ):
        #     raise ValueError( "Either both or neither of psfpath and psfxmlpath must be None" )

        # breakpoint()
        with open( txtpath ) as ifp:
            headertxt = ifp.read()
            self._wcs = WCS( fits.Header.fromstring( headertxt , sep='\\n' ))

        # with fits.open( fitspath, memmap=False ) as hdul:
        #     breakpoint()
        #     wcs_header_string = hdul[0].header.tostring( sep='\n', padding=False )
        #     self._wcs = WCS( fits.Header.fromstring( wcs_header_string, sep='\n' ) )
        