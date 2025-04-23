import pathlib
import numbers
import uuid
import random

import numpy as np
import h5py

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import CheckConstraint

from astropy.io import fits

from models.base import Base, SmartSession, SeeChangeBase, UUIDMixin, FileOnDiskMixin, HasBitFlagBadness
from models.enums_and_bitflags import PSFFormatConverter, psf_badness_inverse
from models.image import Image
from models.source_list import SourceList
from util.logger import SCLogger


class PSF(Base, UUIDMixin, FileOnDiskMixin, HasBitFlagBadness):
    """Encapsulates a PSF.

    You should not instantiate this class directly, but rather one of
    its subclasses.

    """
    __tablename__ = 'psfs'

    @declared_attr
    def __table_args__(cls):  # noqa: N805
        return (
            CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
                               '(md5sum_components IS NULL OR array_position(md5sum_components, NULL) IS NOT NULL))',
                               name=f'{cls.__tablename__}_md5sum_check' ),
        )

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( str(PSFFormatConverter.convert('psfex')) ),
        doc=( 'Format of the PSF.  Currently supports psfex, delta, gaussian, and image.  delta and gaussian '
              'psfs are really just for test purposes.  gaussian samples, does not integrate, so is pretty '
              'terrible for low-fwhm psfs.' )
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


    # ****************************************
    # end of schema definition
    # Put in polymorphism so we can use subclasses

    __mapper_args__ = {
        "polymorphic_on": "_format",
        "polymorphic_identity": -1
    }

    # ****************************************


    @property
    def data( self ):
        """The data for this PSF.  It's nature will depend on the format of the psf.

        It's best not to use this outside this class.  Instead, use get_clip().

        For PSFEx formatted files, this is what's in the HDU 1 data of
        the output of psfex, a 3-dimensional numpy array with the basis
        images used in reconstructing the position-variable PSF at any
        point along the image.  (The code in get_clip performs this
        reconstruction.)

        For delta and gaussian PSFs, this is None.

        For image PSFs, this is a 2d numpy array.

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

        This will not be defined for some PSF types, so it's best not to
        use this property outside this class.

        For PSFEx, this is the header from the .psf FITS file written by PSFex.

        For the other classes (as of this writing), this is None.

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
        """Associated info for this PSF; something opaque, so don't use it outside this class.

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
        """The (ny, nx) shape of the image this PSF is for."""
        if self._image_shape is None:
            self._load_image_shape()
        return self._image_shape

    @image_shape.setter
    def image_shape( self, val ):
        # Normally you shouldn't use this, but if you're doing a test you may
        #   need to set this manually
        self._image_shape = val

    def _load_image_shape( self ):
        raise NotImplementedError( f"{self.__class__.__name__} needs to implement _load_image_shape" )

    @property
    def clip_shape(self):
        """The (ny, nx) shape of a standard clip at image resolution for this PSF."""
        if self._clip_shape is None:
            self._load_clip_shape()
        return self._clip_shape

    @clip_shape.setter
    def clip_shape( self, val ):
        # Normally you shouldn't use this, but if you're doing a test you may
        #   need to set this manually
        self._clip_shape = val

    def _load_clip_shape( self ):
        raise NotImplementedError( f"{self.__class__.__name__} needs to implement _load_clip_shape" )

    @property
    def oversampling_factor( self ):
        """The oversampling factor for this PSF, if relevant.

        This is the pixel size of the oversampled PSF divided by the
        pixel size of the image, so an oversampled PSF has an
        oversampling_factor less than 1

        For some PSF classes, this is meaningless.

        """
        if self._oversampling_factor is None:
            self._load_oversampling_factor()
        return self._oversampling_factor

    @oversampling_factor.setter
    def oversampling_factor( self, val ):
        # Normally you shouldn't use this, but if you're doing a test you may
        #   need to set this manually
        self._oversampling_factor = val

    def _load_oversampling_factor( self ):
        raise NotImplementedError( f"{self.__class__.__name__} needs to implement _load_oversampling_factor" )

    @property
    def raw_clip_shape( self ):
        """The (ny, nx) of the PSF as stored oversampled, if relevant

        This is meaningless for some PSF classes.
        """
        if self._raw_clip_shape is None:
            self._load_raw_clip_shape()
        return self._raw_clip_shape

    @raw_clip_shape.setter
    def raw_clip_shape( self, val ):
        # Normally you shouldn't use this, but if you're doing a test you may
        #   need to set this manually
        self._raw_clip_shape = val

    def _load_raw_clip_shape( self ):
        raise NotImplementedError( f"{self.__class__.__name} needs to implement _load_raw_clip_shape" )


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
        self._image_shape = None
        self._raw_clip_shape = None
        self._clip_shape = None
        self._oversampling_factor = None

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
        self._image_shape = None
        self._raw_clip_shape = None
        self._clip_shape = None
        self._oversampling_factor = None


    def _determine_filepath( self, filename=None, image=None, sources=None, filename_is_absolute=False ):
        """Figure out the filepath for this PSF.

        Will set self.filepath to the determned filename iff
        filename_is_absolute is False.

        Will end in ".psf" or ".psf_{sources.provenance_id[:6]".
        Individual subclasses will probably add more extensions when
        actually saving or loading a file.

        Returns a str which is the determiend file path; this will be an
        absolute if filename_is_absolute is True, or relative to the
        local data store if filename_is_absolute is False.

        """

        if filename is not None:
            if not filename.endswith('.psf'):
                filename += '.psf'
            if not filename_is_absolute:
                self.filepath = filename
        else:
            if filename_is_absolute:
                raise ValueError( "filename_is_absolute requires a non-None filename" )
            if ( sources is None ) or ( image is None ):
                with SmartSession() as session:
                    if sources is None:
                        sources = SourceList.get_by_id( self.sources_id, session=session )
                    if ( sources is not None ) and ( image is None ):
                        image = Image.get_by_id( sources.image_id, session=session )
                if ( sources is None ) or ( image is None ):
                    raise RuntimeError( "Can't invent PSF filepath; can't find either the corresponding "
                                        "SourceList or the corresponding Image." )

            filename = image.filepath if image.filepath is not None else image.invent_filepath()
            filename += f'.psf_{sources.provenance_id[:6]}'
            self.filepath = filename

        return filename


    def save( self, filename=None, image=None, sources=None, filename_is_absolute=False, **kwargs ):
        """Write the PSF to disk.

        May or may not upload to the archive and update the
        FileOnDiskMixin-included fields of this object based on the
        additional arguments that are forwarded to FileOnDiskMixin.save.

        For psfex-format psfs, this saves two files: the .psf file (the
        FITS file with the data), and the .psf.xml file (the XML file
        created by PSFex.)

        Parameters
        ----------
          filename: str or None
             The path to the file to write, relative to the local store
             root.  Do not include the extension (e.g. '.psf') at the
             end of the name; that will be added automatically for all
             extensions.  If None, will call image.invent_filepath() to
             get a filestore-standard filename and directory.  The psf
             object's filepath will be updated with the resultant path
             (either from this parameter, or from invent_filepath()),
             unless filename_is_absolute is True.

          sources: SourceList or None
             Ignored if filename is specified.  Otherwise, the
             SourceList to use in inventing the filepath (needed to get
             the provenance). If None, will try to load it from the
             database.  Use this for efficiency, or if you know the
             source list isn't yet in the database.

          image: Image or None
             Ignored if filename is specified.  Otherwise, the Image to
             use in inventing the filepath.  If None, will try to load
             it from the database.  Use this for efficiency, or if you
             know the image isn't yet in the database.

          filename_is_absolute : bool, default False
             If False (default), then filename is relative to the local
             store root.  If True, then filename is an absolute path
             (and must be specified).  In this case, the psf object's
             filepath will _not_ be updated.  You also almost always
             want to include no_archive=True as an argument when doing
             this.

          Additional arguments are passed on to FileOnDiskMixin.save

        """

        raise NotImplementedError( f"{self.__class__.__name__} needs to implement save" )


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
            given the database fields.  If not None, read the PSF from
            this file.  (For PSFExPSF psfs, _header and _data both get
            read from this file.  For ImagePSF, _data gets read from
            this file.  For the others, I'm not sure what happens.)
            (This exists so that this method may be used to load the
            data with a psf that's not yet in the database, without
            having to play games with the filepath field.)

          psfxmlpath : str or Path, default None
            For PSFEx PSFs only, must be non-None if psfpath is
            non-None; the name of the .psf.xml file to read _info from.

        """

        raise NotImplementedError( f"{self.__class__.__name__} needs to imlement load" )

    def copy( self ):
        """Make a shallow copy of a PSF.  WARNING : will point to same data blocks!"""
        newpsf = self.__class__()
        newpsf._format = self._format
        newpsf.sources_id = self.sources_id
        newpsf.fwhm_pixels = self.fwhm_pixels
        newpsf.data = self.data
        newpsf.header = self.header
        newpsf.info = self.info
        newpsf._image_shape = self._image_shape
        newpsf._raw_clip_shape = self._raw_clip_shape
        newpsf._clip_shape = self._clip_shape
        newpsf._oversampling_factor = self._oversampling_factor
        return newpsf

    def free( self ):
        """Free loaded PSF memory.

        Wipe out the data, info, and header fields, freeing memory.
        Depends on python garbage collection, so if there are other
        references to those objects, the memory won't actually be freed.

        """
        self._data = None
        self._info = None
        self._header = None


    def get_centered_psf( self, nx, ny, x=None, y=None, offx=0., offy=0., flux=1.0,
                          norm=True, noisy=False, gain=1., rng=None, dtype=np.float64 ):
        """Get a full-size image with a centered PSF.

        Parameters
        -----------
          nx : int
            The x-size of the output image.  You usually want this to be
            image.shape[1] if you're trying to get the centered psf for
            image.

          ny : int
            The y-size of the output image; use image.shape[0].

          (x, y) : float, float
            The position, sort of, where to evalute the PSF.  If not
            given, this will be at ~(nx/2, ny/2).  The PSF may be
            evaluted at up to half a pixel off from this position, so as
            to really return a centered PSF.  (E.g., if nx is odd, then
            the PSF will be centered along the x-axis at the middle of a
            pixel; if nx is even, it will be centered along the x-axis
            at the edge of a pixel.)  Note that even if you give x and
            y, the psf is still rendered at the center of the returned
            image!  If you want x and y rendered somewhere else, use
            offx, offy.  (So, if what you want is a psf rendered at
            (x,y) that is the shape that the psf would have at that
            position, pass offx=(x-ctrx), offy=(y-ctry), where
            ctrx=(nx//2-0.5) if nx is even and (nx//2) if nx is odd,
            etc. for ctry.)

            (The standard use-case for this routine is generating a psf
            image to use with fourier transforms, and you want that psf
            centered on the image to avoid convolutions in fourier space
            shifting the whole image.  In this use case, if you have a
            spatially-variable psf, then things are dubious anyway.)

          (offx, offy) : float, float
            If for some perverse reason you want the psf offset from the center (e.g.
            zogy seems to need this...!), give that offset here.

          Other parameters are passed on as-is to get_clip

          Returns
          -------
            An nx by ny image (i.e. with shape [ny,nx]) with a centered PSF.

        """

        if ( not isinstance( nx, numbers.Integral ) ) or not ( isinstance( ny, numbers.Integral ) ):
            raise TypeError( f"nx and ny must be integers; got nx as a {type(nx)} and ny as a {type(ny)}" )

        # Figure out where the center is; if the side is even length, then it
        #    needs to be at the edge of a pixel; if it's odd length, then
        #    it needs to be at the center of a pixel.
        ctrx = float( nx // 2 - 0.5 if nx % 2 == 0 else nx // 2 )
        ctry = float( ny // 2 - 0.5 if ny % 2 == 0 else ny // 2 )

        # Add the offset to get the position where we want to render the PSF
        xpos = ctrx + offx
        ypos = ctry + offy

        # This is a necessary but not sufficient test
        if ( xpos < 0 ) or ( ypos < 0 ) or ( xpos >= nx ) or ( ypos >= ny ):
            raise ValueError( f"(xpos,ypos)=({xpos:.2f},{ypos:.2f}) is off the edge of the (nx,ny)=({nx},{ny}) image" )

        # We want to evaluate the PSF at (x,y), but we want it centered relative the pixel
        #  at the apropriate place for xpos, ypos.  So, adjust x and y so the fractional part
        #  is correct.  (If (x,y) is not given, we want to evalute it at (ctrx, ctry).)
        # To avoid edge cases with even/odd numbers and 0.5, we're going
        #  to redefine rounding here to mean floor(x+0.5).  That should
        #  always put x within 0.5 pixel of what was passed or ctrx (if
        #  None was passed).
        x = ctrx if x is None else x
        y = ctry if y is None else y
        x = np.floor(x+0.5) + xpos - np.floor(xpos+0.5)
        y = np.floor(y+0.5) + ypos - np.floor(ypos+0.5)

        psfclip = self.get_clip( x, y, flux=flux, norm=norm, noisy=noisy, gain=gain, rng=rng, dtype=dtype )

        if ( nx < psfclip.shape[1] ) or ( ny < psfclip.shape[0] ):
            raise ValueError( f"Asked for return image size {nx}×{ny} which is smaller than the PSF clip "
                              f"{psfclip.shape[1]}×{psfclip.shape[0]}" )

        # Padding.  More complicated than you'd think.
        #
        # NOTE : below when I say "the fractional part of q" I mean
        # "q-floor(q)".  So, the fractional part of -1.2 is 0.8, the
        # fractional part of 1.5 is .5, and the fractional part of 2.5
        # is .5.  "round", with it's scary conventions about even/odd
        # integers, does not come into it.
        #
        #  * For odd-length images : if there was no offset, then
        #    get_clip will have returned an odd-size clip with the psf
        #    centered on the center pixel of that clip.  So, we simply
        #    pad by the same amount on both sides and we get the psf
        #    centered on the final image.
        #
        #    If there was an offset, then we need to add something like
        #    (xpos-ctrx) to the left padding.  If (xpos-ctrx) is an
        #    integer, then just do that.  If the fractional part of
        #    (xpos-ctrx) is < 0.5, then it's just slide within the same
        #    pixel, so we can just add (floor(xpos-ctrx)) to the left
        #    padding.  But, if (xpos-ctrx) >= 0.5, then psf's get_clip
        #    will have rendered the PSF down and to the left of its
        #    center, so we need to add (floor(xpos-ctrx)+1) to the left
        #    padding.
        #
        # * For even-length images : if there was no offset, then
        #   get_clip will have returned an odd-size clip with the
        #   psf centered down and to the left of the center of the clip.
        #   To push that to the center of the final image, we need to pad
        #   the left one more than we pad the left.  It turns out
        #   That just using nx//2 - psfclip.shape[1]//2, just like we
        #   do with odd images, does the right thing here (since the //
        #   on the odd psfclip.shape will be on the low side).
        #
        #   If there's an offset, we can always just add
        #   floor(xpos-ctrx) to the left padding.

        padlowx = nx // 2 - psfclip.shape[1] // 2 + int( np.floor( xpos-ctrx ) )
        padlowx += 1 if ( nx % 2 == 1) and ( (xpos-ctrx) - np.floor(xpos-ctrx) >= 0.5 ) else 0
        padhighx = nx - padlowx - psfclip.shape[1]

        padlowy = ny // 2 - psfclip.shape[0] // 2 + int( np.floor( ypos-ctry ) )
        padlowy += 1 if ( ny % 2 == 1) and ( (ypos-ctry) - np.floor(ypos-ctry) >= 0.5 ) else 0
        padhighy = ny - padlowy - psfclip.shape[0]

        retimg = np.zeros( (ny, nx), dtype=dtype )
        retimg[ padlowy:-padhighy, padlowx:-padhighx ] = psfclip
        return retimg


    def get_resampled_psf( self, x, y, dtype=np.float64 ):
        """Return an image fragment with the PSF at the underlying sampling of the PSF model.

        May not be implemented for all PSF classes, as for some it's meaningless.

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

        raise NotImplementedError( f"get_resampled_psf is not defined for {self.__class__.__name__}." )


    def get_clip(  self, x=None, y=None, flux=1.0, norm=True, noisy=False, gain=1., rng=None, dtype=np.float64 ):
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
          2d numpy array.
            Will have an odd size, as this class enforces odd-sized
            stamp widths for PSFs.  If x and y are integers, the PSF
            will be centered on the center pixel of the return image.
            If x and y are integers+(0.0,0.5], the PSF will be centered
            up and to the right of the center pixel of the return
            image.  If x and y integers+(0.5,1.0), the PSF will be
            centered down and to the left of the center pixel of the
            return image.
        """

        # This implementation works for psfexpsf and imagepsf, where get_resampled_psf is implemented.
        # Other classes will need to override this method.

        if x is None:
            x = self.image_shape[1] / 2.
        if y is None:
            y = self.image_shape[0] / 2.

        psfbase = self.get_resampled_psf( x, y, dtype=np.float64 )

        # round() isn't the right thing to use here, because it will
        #   behave differently when x - round(x) = 0.5 based on whether
        #   floor(x) is even or odd.  What we *want* is for the psf to
        #   be as close to the center of the clip as possible.  In the
        #   case where the fractional part of x is exactly 0.5, it's
        #   ambiguous what that means-- there are four places you could
        #   stick the PSF to statisfy that criterion.  By using
        #   floor(x+0.5), we will consistently have the psf leaning down
        #   and to the left when the fractional part of x (and y) is
        #   exactly 0.5, whereas using round would give different
        #   results based on the integer part of x (and y).

        xc = int( np.floor( x + 0.5 ) )
        yc = int( np.floor( y + 0.5 ) )

        # See Chapter 5, "How PSFEx Works", of the PSFEx manual
        #     https://psfex.readthedocs.io/en/latest/Working.html
        # We're using this method for both image and psfex PSFs,
        #   as the interpolation is more general than PSFEx:
        #      https://en.wikipedia.org/wiki/Lanczos_resampling
        #   ...though of course, the choice of a=4 comes from PSFEx.

        psfsamp = self.oversampling_factor
        stampwid = self.clip_shape[0]
        psfwid = self.raw_clip_shape[0]
        psfdex1d = np.arange( -( psfwid//2), psfwid//2+1, dtype=int )

        xmin = xc - stampwid // 2
        xmax = xc + stampwid // 2 + 1
        ymin = yc - stampwid // 2
        ymax = yc + stampwid // 2 + 1

        xs = np.array( range( xmin, xmax ) )
        ys = np.array( range( ymin, ymax ) )
        xsincarg = psfdex1d[:, np.newaxis] - ( xs - x ) / psfsamp
        xsincvals = np.sinc( xsincarg ) * np.sinc( xsincarg/4. )
        xsincvals[ ( xsincarg > 4 ) | ( xsincarg < -4 ) ] = 0.
        ysincarg = psfdex1d[:, np.newaxis] - ( ys - y ) / psfsamp
        ysincvals = np.sinc( ysincarg ) * np.sinc( ysincarg/4. )
        ysincvals[ ( ysincarg > 4 ) | ( ysincarg < -4 ) ] = 0.
        tenpro = np.tensordot( ysincvals[:, :, np.newaxis], xsincvals[:, :, np.newaxis], axes=0 )[ :, :, 0, :, :, 0 ]
        clip = ( psfbase[:, np.newaxis, :, np.newaxis ] * tenpro ).sum( axis=0 ).sum( axis=1 )

        # Keeping the code below, because the code above is inpenetrable, and it's trying to
        #   do the same thing as the code below.
        # (I did emprically test it using the PSFs from the test_psf.py::test_psfex_rendering,
        #  and it worked.  In particular, there is not a transposition error in the "tenpro=" line;
        #  if you swap the order of yxincvals and xsincvals in the test, then the values of clip
        #  do not match the code below very well.  As is, they match to within a few times 1e-17,
        #  which is good enough as the minimum non-zero value in either one is of order 1e-12.)
        # clip = np.empty( ( stampwid, stampwid ), dtype=dtype )
        # for xi in range( xmin, xmax ):
        #     for yi in range( ymin, ymax ):
        #         xsincarg = psfdex1d - (xi-x) / psfsamp
        #         xsincvals = np.sinc( xsincarg ) * np.sinc( xsincarg/4. )
        #         xsincvals[ ( xsincarg > 4 ) | ( xsincarg < -4 ) ] = 0
        #         ysincarg = psfdex1d - (yi-y) / psfsamp
        #         ysincvals = np.sinc( ysincarg ) * np.sinc( ysincarg/4. )
        #         ysincvals[ ( ysincarg > 4 ) | ( ysincarg < -4 ) ] = 0
        #         clip[ yi-ymin, xi-xmin ] = ( xsincvals[np.newaxis, :]
        #                                      * ysincvals[:, np.newaxis]
        #                                      * psfbase ).sum()

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

        if ( x < 0 ) or ( x >= image.shape[1] ) or ( y < 0 ) or ( y >= image.shape[0] ):
            SCLogger.warn( "Center of psf to be added to image is off of edge of image" )

        xc = int( np.floor(x + 0.5) )
        yc = int( np.floor(y + 0.5) )
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


    def get_upstreams( self, session=None ):
        """Get the source list that is associated with this PSF."""
        with SmartSession(session) as session:
            return [ SourceList.get_by_id( self.sources_id, session=session ) ]


    def get_downstreams( self, session=None ):
        """PSF has no downstreams.

        (It has no provenance.  There is 1:1 between SourceList and PSF,
        as the process that extracts the SourceList is the same as the
        process that determines the PSF (in pipeline/extraction.py).

        """
        return []


# **********************************************************************

class PSFExPSF(PSF):
    """A PSF as produced by psfex (https://www.astromatic.net/software/psfex/)."""

    __mapper_args__ = {
        # "psfex" in enums_and_bitflags.py::PSFFormatConverter
        "polymorphic_identity": 1
    }

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self.format = 'psfex'

    def _load_image_shape( self ):
        self._image_shape = ( self.header['IMAXIS2'], self.header['IMAXIS1'] )

    def _load_clip_shape( self ):
        psfwid = self.data.shape[1]
        psfsamp = self.header['PSF_SAMP']
        stampwid = int( np.floor( psfsamp * psfwid ) + 0.5 )
        self._clip_shape = ( stampwid, stampwid )

    def _load_oversampling_factor( self ):
        self._oversampling_factor = self.header['PSF_SAMP']

    def _load_raw_clip_shape( self ):
        # Data is a 3d array; the first index is the index over the orders
        self._raw_clip_shape = self.data.shape[1:]


    def get_resampled_psf( self, x, y, dtype=np.float64 ):
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


    def save( self, filename=None, image=None, sources=None, filename_is_absolute=False, **kwargs ):
        """Write a PSFEx PSF to disk.

        For psfex-format psfs, this saves two files: the .psf file (the
        FITS file with the data), and the .psf.xml file (the XML file
        created by PSFex.)

        Parameters are as in PSF::save.

        """
        if ( self._data is None ) or ( self._header is None ) or ( self._info is None ):
            raise RuntimeError( "_data, _header, and _info must all be non-None" )

        filename = self._determine_filepath( filename=filename, image=image, sources=sources,
                                             filename_is_absolute=filename_is_absolute )

        if filename_is_absolute:
            psfpath = pathlib.Path( f'{filename}.fits' )
            psfxmlpath = pathlib.Path( f'{filename}.xml' )
        else:
            psfpath = pathlib.Path( self.local_path ) / f'{filename}.fits'
            psfxmlpath = pathlib.Path( self.local_path ) / f'{filename}.xml'

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
        FileOnDiskMixin.save( self, psfpath, 'fits', **kwargs )
        FileOnDiskMixin.save( self, psfxmlpath, 'xml', **kwargs )


    def load( self, download=True, always_verify_md5=False, psfpath=None, psfxmlpath=None ):
        if ( psfpath is None ) != ( psfxmlpath is None ):
            raise ValueError( "Either both or neither of psfpath and psfxmlpath must be None" )

        if psfpath is None:
            if self.components != [ 'fits', 'xml' ]:
                raise ValueError( f"Can't load psfex file; components is {self.components}, "
                                  f"but expected ['fits', 'xml']." )
            psfpath, psfxmlpath = self.get_fullpath( download=download,
                                                     always_verify_md5=always_verify_md5,
                                                     nofile=False )

        with fits.open( psfpath, memmap=False ) as hdul:
            self._header = hdul[1].header
            self._data = hdul[1].data[0][0]
        with open( psfxmlpath ) as ifp:
            self._info = ifp.read()



# **********************************************************************

class DeltaPSF(PSF):
    """A "delta-function" PSF.  Sort of.

    For a PSF centered at x.0, y.0, this PSF has a value of 1 in the
    center pixel.  If it's offset, then the flux is spread between the
    four pixels around the included corner with linear interpolation.

    """

    __mapper_args__ = {
        # "delta" in enums_and_bitflags.py::PSFFormatConverter
        "polymorphic_identity": 2
    }

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self.format = 'delta'

    def get_clip(  self, x=None, y=None, flux=1.0, norm=True, noisy=False, gain=1., rng=None, dtype=np.float64 ):
        fx = x - np.floor( x + 0.5 )
        fy = y - np.floor( y + 0.5 )

        # Use a 5×5 clip because 3×3 is enough but I wanted more
        clip = np.zeros( (5, 5), dtype=dtype )

        if fx < 0:
            if fy < 0:
                clip[1, 1] = ( -fx ) * ( -fy )
                clip[1, 2] = ( 1+fx ) * ( -fy )
                clip[2, 1] = ( -fx ) * ( 1+fy )
                clip[2, 2] = ( 1+fx ) * ( 1+fy )
            else:
                clip[2, 1] = ( -fx ) * ( 1-fy )
                clip[2, 2] = ( 1+fx ) * ( 1-fy )
                clip[3, 1] = ( -fx ) * ( fy )
                clip[3, 2] = ( 1+fx ) * ( fy )
        else:
            if fy < 0:
                clip[1, 2] = ( 1-fx ) * ( -fy )
                clip[1, 3] = ( fx ) * ( -fy )
                clip[2, 2] = ( 1-fx ) * ( 1+fy )
                clip[2, 3] = ( fx ) * ( 1+fy )
            else:
                clip[2, 2] = ( 1-fx ) * ( 1-fy )
                clip[2, 3] = ( fx ) * ( 1-fy )
                clip[3, 2] = ( 1-fx ) * ( fy )
                clip[3, 3] = ( fx )* ( fy )

        clip *= flux

        if noisy:
            if rng is None:
                rng = np.random.default_rng()
            sig = np.sqrt( clip / gain )
            clip = rng.normal( clip, sig )

        return clip

    def save( self, *args, **kwargs ):
        # A DeltaPSF doesn't actually save anything to disk.  But,
        # because some other PSF classes do, we have to have save()
        # implemented in general for PSFs.  What's more, we have to
        # spoof an md5sum for DeltaPSF rows in the database so that the
        # non-null md5sum constraint won't cause problems.
        barf = "".join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
        self.filepath = f"not_real_{barf}"
        self.copmonents = None
        self.md5sum = uuid.uuid4()

    def load( self, *args, **kwargs ):
        # Nothing to load
        pass


# **********************************************************************

class GaussianPSF(PSF):
    """A sampling Gaussian PSF.

    This is a *sampling* Gaussian, not one that integrates in
    pixel-sized boxes, so it won't be very consistent for small values
    of self.fwhm_pixels (undersampled images).  Ideally,
    self.fwhm_pixels should at least be a few.

    """

    __mapper_args__ = {
        # "gaussian" in enums_and_bitflags.py::PSFFormatConverter
        "polymorphic_identity": 3
    }

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self.format = 'gaussian'

    def get_clip( self, x=None, y=None, flux=1.0, norm=True,
                           noisy=False, gain=1., rng=None, dtype=np.float64 ):
        fx = np.floor( x + 0.5 ) - x
        fy = np.floor( y + 0.5 ) - y
        halfwid = int( 5. * self.fwhm_pixels + 0.5 )
        xvals, yvals = np.meshgrid( np.arange( -halfwid+fx, halfwid+fx+1, 1. ),
                                    np.arange( -halfwid+fy, halfwid+fy+1, 1. ) )
        sig = self.fwhm_pixels / 2.35482
        clip = flux / ( 2. * np.pi * sig**2 ) * np.exp( -( xvals**2 + yvals**2 ) / ( 2. * sig**2 ) )

        if norm:
            clip /= clip.sum()

        if noisy:
            if rng is None:
                rng = np.random.default_rng()
            noise = np.sqrt( clip / gain )
            clip = rng.normal( clip, noise )

        return clip

    def save( self, *args, **kwargs ):
        # A GaussianPSF doesn't actually save anything to disk.  But,
        # because some other PSF classes do, we have to have save()
        # implemented in general for PSFs.  What's more, we have to
        # spoof an md5sum for DeltaPSF rows in the database so that the
        # non-null md5sum constraint won't cause problems.
        barf = "".join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
        self.filepath = f"not_real_{barf}"
        self.copmonents = None
        self.md5sum = uuid.uuid4()

    def load( self, *args, **kwargs ):
        # Nothing to load
        pass


# **********************************************************************

class ImagePSF(PSF):
    """A PSF that is stored as data array, which does not vary over the image the PSF is for.

    The image has shape self.raw_clip_shape.

    The PSF is stored in an HDF5 file which has a single group "/psf".
    It will have attributes "image_shape_0", "image_shape_1",
    "clip_shape_0", "clip_shape_1" and "oversampling_factor", and one
    dataset "/psf/data".

    """

    __mapper_args__ = {
        # "image" in enums_and_bitflags.py::PSFFormatConverter
        "polymorphic_identity": 4
    }

    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
        self.format = 'image'

    def _load_image_shape( self ):
        self.load()

    def _load_raw_clip_shape( self ):
        self.load()

    def _load_clip_shape( self ):
        self.load()

    def _load_oversampling_factor( self ):
        self.load()


    def get_resampled_psf( self, x, y, dtype=np.float64 ):
        return np.array( self.data, dtype=dtype )


    def set_data( self, data, clip_width, oversampling_factor ):
        """Set the data for this ImagePSF.

        Parameters
        ----------
          data : 2d numpy array
             Should be normalized (sum to 1).

          clip_width : int
             The width of a thumbnail on the image (at the pixel scale
             of the image) that holds this psf.

          oversampling_factor : float
             The pixel scale of data divided by the pixel scale of the
             image this PSF is for.  So, if the PSF is oversampled, this
             should be <1.  If data is at the same scale as the image,
             this should be 1.0.  clip_width / data.shape[0] should be
             *approximately* (but not exactly) oversampling_factor.

        """

        self._data = data
        self._clip_shape = ( clip_width, clip_width )
        self._oversampling_factor = oversampling_factor


    def save( self, filename=None, image=None, sources=None, filename_is_absolute=False, **kwargs ):
        """Write a ImagePSF to disk.

        Parameters are as in PSF::save.

        """

        if ( ( self._data is None ) or
             ( self._clip_shape is None ) or
             ( self._oversampling_factor is None ) or
             ( self._image_shape is None )
            ):
            raise RuntimeError( "_data, _clip_shape, and _oversampling_factor must all be non-None" )
        if ( not isinstance( self._data, np.ndarray ) ) or ( len(self._data.shape) != 2 ):
            raise TypeError( "_data must be a 2d numpy array" )
        if ( not isinstance( self._clip_shape, tuple ) ) or ( len(self._clip_shape) != 2 ):
            raise TypeError( "_clip_shape must be a 2-element tuple" )
        if ( not isinstance( self._image_shape, tuple ) ) or ( len(self._image_shape) != 2 ):
            raise TypeError( "_image_shape must be a 2-element tuple" )
        clip_shape = ( int(self._clip_shape[0]), int(self._clip_shape[1]) )
        image_shape = ( int(self._image_shape[0]), int(self._image_shape[1]) )
        oversampling_factor = float( self.oversampling_factor )

        filename = self._determine_filepath( filename=filename, image=image, sources=sources,
                                             filename_is_absolute=filename_is_absolute )
        if filename_is_absolute:
            filepath = pathlib.Path( f'{filename}.hdf5' )
        else:
            filepath = pathlib.Path( self.local_path ) / f'{filename}.hdf5'

        with h5py.File( filepath, 'w' ) as h5f:
            psfgrp = h5f.create_group( 'psf ')
            psfgrp.attrs['image_shape_0'] = image_shape[0]
            psfgrp.attrs['image_shape_1'] = image_shape[1]
            psfgrp.attrs['clip_shape_0'] = clip_shape[0]
            psfgrp.attrs['clip_shape_1'] = clip_shape[1]
            psfgrp.attrs['oversampling_factor'] = oversampling_factor
            psfgrp.create_dataset( 'data', data=self._data )

        FileOnDiskMixin.save( self, filepath, component=None, **kwargs )


    def load( self, download=True, always_verify_md5=False, psfpath=None, psfxmlpath=None ):
        """Load an ImagePSF from disk.

        Parameters are as in PSF::load

        """

        if psfxmlpath is not None:
            raise ValueError( "psfxmlpath is meaningless for ImagePSF" )

        if psfpath is None:
            psfpath = self.get_fullpath( download=download, always_verify_md5=always_verify_md5, nofile=False )

        with h5py.File( psfpath, 'r' ) as h5f:
            if 'psf' not in h5f:
                raise ValueError( "No psf group found in the file" )

            self._image_shape = ( h5f["psf"].attrs["image_shape_0"], h5f["psf"].attrs["image_shape_1"] )
            self._clip_shape = ( h5f["psf"].attrs["clip_shape_0"], h5f["psf"].attrs["clip_shape_1"] )
            self._oversampling_factor = h5f["psf"].attrs["oversampling_factor"]
            self._data = h5f["psf/data"][:]
            self._raw_clip_shape = self._data.shape
