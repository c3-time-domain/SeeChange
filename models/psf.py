import pathlib

import numpy as np

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import CheckConstraint

from astropy.io import fits

from models.base import Base, SmartSession, SeeChangeBase, UUIDMixin, FileOnDiskMixin, HasBitFlagBadness
from models.enums_and_bitflags import PSFFormatConverter, psf_badness_inverse
from models.image import Image
from models.source_list import SourceList, SourceListSibling
from util.logger import SCLogger

# NOTE.  As of this writing, the only format for PSFs we were
# considering was the output of PSFEx.  As such, some stuff here may not
# be as general as we want (in case we ever have another way of
# generating PSFs).  In particular, the structure of "header/data/info"
# is tuned to the output of PSFEx, where it's a FITS file (in HDU 1)
# with a header and data, and there's an XML file that can be read into
# a votable (which goes in info here).
#
# This may take some refactoring if we ever want to support anything
# else.  Hopefully the database structure will still work, but any
# details of looking into header and info will need different versions
# for different formats.


class PSF(SourceListSibling, Base, UUIDMixin, FileOnDiskMixin, HasBitFlagBadness):
    __tablename__ = 'psfs'

    @declared_attr
    def __table_args__(cls):  # noqa: N805
        return (
            CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
                               '(md5sum_extensions IS NULL OR array_position(md5sum_extensions, NULL) IS NOT NULL))',
                               name=f'{cls.__tablename__}_md5sum_check' ),
        )

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( str(PSFFormatConverter.convert('psfex')) ),
        doc='Format of the PSF file.  Currently only supports psfex.'
    )

    @hybrid_property
    def format(self):
        return PSFFormatConverter.convert( self._format )

    @format.inplace.expression
    @classmethod
    def format(cls):
        return sa.case( PSFFormatConverter.dict, value=cls._format )

    @format.inplace.setter
    def format( self, value ):
        self._format = PSFFormatConverter.convert( value )

    sources_id = sa.Column(
        sa.ForeignKey( 'source_lists._id', ondelete='CASCADE', name='psfs_source_lists_id_fkey' ),
        nullable=False,
        index=True,
        unique=True,
        doc="id of the source_list this psf is associated with"
    )

    fwhm_pixels = sa.Column(
        sa.REAL,
        nullable=False,
        index=False,
        doc="Approximate FWHM of seeing in pixels; use for a broad estimate, doesn't capture spatial variation."
    )


    @property
    def data( self ):
        """The data for this PSF.  It's nature will depend on the format of the psf.

        For PSFEx formatted files, this is what's in the HDU 1 data of
        the output of psfex, a 3-dimensional numpy array with the basis
        images used in reconstructing the position-variable PSF at any
        point along the image.  (The code in get_clip performs this
        reconstruction.)

        """
        if self._data is None and self.filepath is not None:
            self.load()
        return self._data

    @data.setter
    def data( self, value ):
        self._data = value

    @property
    def header( self ):
        """Any header information associated with the psf; an astropy.io.fits.header.Header object.

        For PSFEx, this is the header from the .psf FITS file written by PSFex.

        """
        if self._header is None and self.filepath is not None:
            self.load()
        return self._header

    @header.setter
    def header( self, value ):
        self._header = value

    # Right now, this may be PSF specific, in that it assumes
    # there's a header and data from a FITS file, and a votable
    # from the xml file
    @property
    def info( self ):
        """Associated info for this PSF; something opaque.

         For PSFEx, this is the contents of the xml file produced when
         psfex ran.  (It can be parsed into a votable with
         astropy.io.votable.parse.)

        """
        if self._info is None and self.filepath is not None:
            self.load()
        return self._info

    @info.setter
    def info( self, value ):
        self._info = value

    @property
    def image_shape(self):
        """The shape of the image this PSF is for."""
        return self.header['IMAXIS2'], self.header['IMAXIS1']

    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object"""
        return psf_badness_inverse

    def __init__( self, *args, **kwargs ):
        FileOnDiskMixin.__init__( self, **kwargs )
        HasBitFlagBadness.__init__(self)
        SeeChangeBase.__init__( self )
        self._header = None
        self._data = None
        self._table = None
        self._info = None

        # Manually set all properties ( columns or not )
        for key, value in kwargs.items():
            if hasattr( self, key ):
                setattr( self, key, value )

    @sa.orm.reconstructor
    def init_on_load( self ):
        Base.init_on_load( self )
        FileOnDiskMixin.init_on_load( self )
        self._header = None
        self._data = None
        self._table = None
        self._info = None

    def save( self, filename=None, image=None, sources=None, **kwargs ):
        """Write the PSF to disk.

        May or may not upload to the archive and update the
        FileOnDiskMixin-included fields of this object based on the
        additional arguments that are forwarded to FileOnDiskMixin.save.

        For psfex-format psfs, this saves two files: the .psf file (the
        FITS file with the data), and the .psf.xml file (the XML file
        created by PSFex.)

        Parameters
        ----------
          filename: str or Path, or None
             The path to the file to write, relative to the local store
             root.  Do not include the extension (e.g. '.psf') at the
             end of the name; that will be added automatically for all
             extensions.  If None, will call image.invent_filepath() to get a
             filestore-standard filename and directory.

          sources: SourceList or None
             Ignored if filename is specified.  Otherwise, the
             SourceList to use in inventing the filepath (needed to get
             the provenance). If None, will try to load it from the
             database.  Use this for efficiency, or if you know the
             soruce list isn't yet in the databse.

          image: Image or None
             Ignored if filename is specified.  Otherwise, the Image to
             use in inventing the filepath.  If None, will try to load
             it from the database.  Use this for efficiency, or if you
             know the image isn't yet in the database.

          Additional arguments are passed on to FileOnDiskMixin.save

        """
        if self.format != 'psfex':
            raise NotImplementedError( "Only know how to save psfex PSF files" )

        if ( self.data is None ) or ( self.header is None ) or ( self.info is None ):
            raise RuntimeError( "data, header, and info must all be non-None" )

        if filename is not None:
            if not filename.endswith('.psf'):
                filename += '.psf'
            self.filepath = filename
        else:
            if ( sources is None ) or ( image is None ):
                with SmartSession() as session:
                    if sources is None:
                        sources = SourceList.get_by_id( self.sources_id, session=session )
                    if ( sources is not None ) and ( image is None ):
                        image = Image.get_by_id( sources.image_id, session=session )
                if ( sources is None ) or ( image is None ):
                    raise RuntimeError( "Can't invent PSF filepath; can't find either the corresponding "
                                        "SourceList or the corresponding Image." )

            self.filepath = image.filepath if image.filepath is not None else image.invent_filepath()
            self.filepath += f'.psf_{sources.provenance_id[:6]}'

        psfpath = pathlib.Path( self.local_path ) / f'{self.filepath}.fits'
        psfxmlpath = pathlib.Path( self.local_path ) / f'{self.filepath}.xml'

        # header0 = fits.Header( [ fits.Card( 'SIMPLE', 'T', 'This is a FITS file' ),
        #                          fits.Card( 'BITPIX', 8 ),
        #                          fits.Card( 'NAXIS', 0 ),
        #                          fits.Card( 'EXTEND', 'T', 'This file may contain FITS extensions' ),
        #                         ] )
        # hdu0 = fits.PrimaryHDU( header=header0 )
        # The PSFEx format is a bit byzantine
        fitsshape = list( self._data.shape )
        fitsshape.reverse()
        fitsshape = str( tuple( fitsshape ) )
        format = f'{np.prod(self._data.shape)}E'
        fitscol = fits.Column(name='PSF_MASK', format=format, dim=fitsshape, array=[self._data])
        fitsrec = fits.FITS_rec.from_columns( fits.ColDefs( [ fitscol ] ) )
        hdu = fits.BinTableHDU( fitsrec, self._header )
        hdu.writeto( psfpath, overwrite=( 'overwrite' in kwargs and kwargs['overwrite'] ) )

        with open( psfxmlpath, "w" ) as ofp:
            ofp.write( self._info )

        # Save the file to the archive and update the database record
        # (From what we did above, the files are already in the right place in the local filestore.)
        FileOnDiskMixin.save( self, psfpath, '.fits', **kwargs )
        FileOnDiskMixin.save( self, psfxmlpath, '.xml', **kwargs )

    def load( self, download=True, always_verify_md5=False, psfpath=None, psfxmlpath=None ):
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
            given the database fields.  If not None, read _header and
            _data from this file.  (This exists so that this method may
            be used to load the data with a psf that's not yet in the
            database, without having to play games with the filepath
            field.)

          psfxmlpath : str or Path, default None
            Must be non-None if psfpath is non-None; the name of the
            .psf.xml file to read _info from.

        """

        if self.format != 'psfex':
            raise NotImplementedError( "Only know how to load psfex PSF files" )

        if ( psfpath is None ) != ( psfxmlpath is None ):
            raise ValueError( "Either both or neither of psfpath and psfxmlpath must be None" )

        if psfpath is None:
            if self.filepath_extensions != [ '.fits', '.xml' ]:
                raise ValueError( f"Can't load psfex file; filepath_extensions is {self.filepath_extensions}, "
                                  f"but expected ['.fits', '.xml']." )
            psfpath, psfxmlpath = self.get_fullpath( download=download, always_verify_md5=always_verify_md5 )

        with fits.open( psfpath, memmap=False ) as hdul:
            self._header = hdul[1].header
            self._data = hdul[1].data[0][0]
        with open( psfxmlpath ) as ifp:
            self._info = ifp.read()

    def free( self ):
        """Free loaded PSF memory.

        Wipe out the data, info, and header fields, freeing memory.
        Depends on python garbage collection, so if there are other
        references to those objects, the memory won't actually be freed.

        """
        self._data = None
        self._info = None
        self._header = None

    def get_resampled_psf( self, x, y, dtype=np.float64 ):
        """Return an image fragment with the PSF at the underlying sampling of the PSF model.

        This is usually not what you want; usually you want to call get_clip

        Parameters
        ----------
          x: float
            x-coordinate on the image the PSF was extracted for (0-offset)

          y: float
            y-coordinate of the image the PSF was extracted from (0-offset)

          dtype: type, default numpy.float64
            Type of the returned array; usually either numpy.float64 or numpy.float32

        Returns
        -------
           A 2d numpy array

        """

        if self.format != 'psfex':
            raise NotImplementedError( "Only know how to get resampled psf for psfex PSF files" )

        psforder = int( self.header['POLDEG1'] )
        x0 = float( self.header['POLZERO1'] ) - 1
        xsc = float( self.header['POLSCAL1'] )
        y0 = float( self.header['POLZERO2'] ) - 1
        ysc = float( self.header['POLSCAL2'] )

        psfbase = np.zeros_like( self.data[0,:,:], dtype=dtype )
        off = 0
        for j in range( psforder+1 ) :
            for i in range( psforder+1-j ):
                psfbase += self.data[off] * (
                    ( (x - x0) / xsc )**i *
                    ( (y - y0) / ysc )**j
                )
                off += 1

        return psfbase

    def _get_clip_info( self ):
        psfwid = self.data.shape[1]
        if ( psfwid % 2 ) == 0:
            raise ValueError( f"Even psf width {psfwid}; should be odd.  This error should never happen." )
        if self.data.shape[2] != psfwid:
            raise ValueError( f"Non-square psf ({self.psfwid} × {self.psfdata.shape[2]}); it needs to be square." )
        psfsamp = self.header['PSF_SAMP']
        stampwid = int( np.floor( psfsamp * psfwid ) + 0.5 )
        if ( stampwid % 2 ) == 0:
            # SCLogger.warning( f'PSF stamp width came out even ({stampwid}), subtracting 1' )
            stampwid -= 1
        psfdex1d = np.arange( -(psfwid//2), psfwid//2+1, dtype=int )

        return psfwid, psfsamp, stampwid, psfdex1d

    def get_clip( self, x=None, y=None, flux=1.0, norm=True, noisy=False, gain=1., rng=None, dtype=np.float64 ):
        """Get an image clip with the psf.

        The clip will have the same pixel scale as the image.

        Parameters
        ----------
          x: float
            x-coordinate on the image the PSF was extracted for (0-offset)
            If None (default) will use the center of the image.
          y: float
            y-coordinate of the image the PSF was extracted from (0-offset)
            If None (default) will use the center of the image.
          flux: float
            Sum of the psf flux values over all pixels.
          norm: bool, default True
            Normalize the psf to 1.0, before adding noise if any.  (This
            seems to be necessary with PSFEx.)
          noisy: bool, default False
            If True, will also scatter the pixel values using
            Poisson statistics, assuming gain e-/adu.
          gain: float, default 1.
            Assumed e-/adu gain for calculating Poisson statistics if
            noisy is true.
          rng: numpy.random.Generator, default None
            If not None, will use this (already-seeded) random number
            generator (produced, for example, with numpy.default_rng) to
            generate the noise.  Pass this if you want reproducible
            noise for testing purposes.  If None, will use
            numpy.random.default_rng() (i.e. seeded from system entropy).
          dtype: type, default numpy.float64
            Type of the returned array; usually either numpy.float64 or numpy.float32

        Returns
        -------
          2d numpy array

        """

        if self.format != 'psfex':
            raise NotImplementedError( "Only know how to get clip for psfex PSF files" )

        if x is None:
            x = self.image_shape[1] / 2.
        if y is None:
            y = self.image_shape[0] / 2.

        psfbase = self.get_resampled_psf( x, y, dtype=np.float64 )

        _, psfsamp, stampwid, psfdex1d = self._get_clip_info()

        xc = int( np.round(x) )
        yc = int( np.round(y) )

        # See Chapter 5, "How PSFEx Works", of the PSFEx manual
        #   https://psfex.readthedocs.io/en/latest/Working.html

        clip = np.empty( ( stampwid, stampwid ), dtype=dtype )
        xmin = xc - stampwid // 2
        xmax = xc + stampwid // 2 + 1
        ymin = yc - stampwid // 2
        ymax = yc + stampwid // 2 + 1
        for xi in range( xmin, xmax ):
            for yi in range( ymin, ymax ):
                xsincarg = psfdex1d - (xi-x) / psfsamp
                xsincvals = np.sinc( xsincarg ) * np.sinc( xsincarg/4. )
                xsincvals[ ( xsincarg > 4 ) | ( xsincarg < -4 ) ] = 0
                ysincarg = psfdex1d - (yi-y) / psfsamp
                ysincvals = np.sinc( ysincarg ) * np.sinc( ysincarg/4. )
                ysincvals[ ( ysincarg > 4 ) | ( ysincarg < -4 ) ] = 0
                clip[ yi-ymin, xi-xmin ] = ( xsincvals[np.newaxis, :]
                                             * ysincvals[:, np.newaxis]
                                             * psfbase ).sum()

        if norm:
            clip /= clip.sum()

        clip *= flux

        if noisy:
            if rng is None:
                rng = np.random.default_rng()
            sig = np.zeros_like( clip )
            sig[ clip > 0 ] = np.sqrt( clip[ clip > 0 ] / gain )
            clip = rng.normal( clip, sig )

        return clip

    def add_psf_to_image( self, image, x, y, flux, norm=True, noisy=False, weight=None, gain=1., rng=None ):
        """Add a psf with indicated flux to the 2d image.

        image : a 2d numpy array

        x, y : position of the PSF

        flux : flux of the PSF in ADU

        weight : a 2d numpy array with inverse variances.  If
           noisy=True, then shot noise will be added to this image,
           assuming that if adding flux f to one pixel, the uncertainty
           is sqrt(f/gain)

        shape of image should match the shape of image the psf was extracted from

        For documentation on x, y, noisy, gain, and rng see PSFExReader.clip

        """

        if self.format != 'psfex':
            raise NotImplementedError( "Only know how to add psf to image for psfex PSF files" )

        if ( x < 0 ) or ( x >= image.shape[1] ) or ( y < 0 ) or ( y >= image.shape[0] ):
            SCLogger.warn( "Center of psf to be added to image is off of edge of image" )

        xc = int( np.round(x) )
        yc = int( np.round(y) )
        clip = self.get_clip( x, y, flux, norm=norm, noisy=noisy, gain=gain, rng=rng )
        stampwid = clip.shape[1]

        xmin = xc - stampwid // 2
        x0 = 0
        if xmin < 0:
            x0 = -xmin
            xmin = 0
        xmax = xc + stampwid // 2 + 1
        x1 = stampwid
        if xmax > image.shape[1]:
            x1 -= xmax - image.shape[1]
            xmax = image.shape[1]
        ymin = yc - stampwid // 2
        y0 = 0
        if ymin < 0:
            y0 = -ymin
            ymin = 0
        ymax = yc + stampwid // 2 + 1
        y1 = stampwid
        if ymax > image.shape[0]:
            y1 -= ymax - image.shape[0]
            ymax = image.shape[0]

        image[ ymin:ymax, xmin:xmax ] += clip[ y0:y1, x0:x1 ]
        if noisy and weight is not None:
            weight[ ymin:ymax, xmin:xmax ] = ( 1. / ( ( 1. / weight[ ymin:ymax, xmin:xmax  ] ) +
                                                      ( clip[ y0:y1, x0:x1 ] / gain )
                                                     )
                                              )
