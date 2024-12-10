import os

import numpy as np

import h5py

import sqlalchemy as sa
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.schema import CheckConstraint

from models.base import Base, SeeChangeBase, SmartSession, UUIDMixin, FileOnDiskMixin, HasBitFlagBadness
from models.image import Image
from models.source_list import SourceList, SourceListSibling

from models.enums_and_bitflags import BackgroundFormatConverter, BackgroundMethodConverter, bg_badness_inverse

# from util.logger import SCLogger
import warnings


class Background(SourceListSibling, Base, UUIDMixin, FileOnDiskMixin, HasBitFlagBadness):
    __tablename__ = 'backgrounds'

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
        server_default=sa.sql.elements.TextClause( str(BackgroundFormatConverter.convert('scalar')) ),
        doc='Format of the Background model. Can include scalar, map, or polynomial. '
    )

    @hybrid_property
    def format(self):
        return BackgroundFormatConverter.convert(self._format)

    @format.inplace.expression
    @classmethod
    def format(cls):
        return sa.case(BackgroundFormatConverter.dict, value=cls._format)

    @format.inplace.setter
    def format(self, value):
        self._format = BackgroundFormatConverter.convert(value)

    _method = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( str(BackgroundMethodConverter.convert('zero')) ),
        doc='Method used to calculate the background. '
            'Can be an algorithm like "sep", or "zero" for an image that was already background subtracted. ',
    )

    @hybrid_property
    def method(self):
        return BackgroundMethodConverter.convert(self._method)

    @method.inplace.expression
    @classmethod
    def method(cls):
        return sa.case(BackgroundMethodConverter.dict, value=cls._method)

    @method.inplace.setter
    def method(self, value):
        self._method = BackgroundMethodConverter.convert(value)

    sources_id = sa.Column(
        sa.ForeignKey('source_lists._id', ondelete='CASCADE', name='backgrounds_source_lists_id_fkey'),
        nullable=False,
        index=True,
        unique=True,
        doc="ID of the source list this background is associated with"
    )

    value = sa.Column(
        sa.Float,
        index=True,
        nullable=False,
        doc="Value of the background level (in units of counts), as a best representative value for the entire image."
    )

    noise = sa.Column(
        sa.Float,
        index=True,
        nullable=False,
        doc="Noise RMS of the background (in units of counts), as a best representative value for the entire image."
    )

    @property
    def image_shape(self):
        if self._image_shape is None and self.filepath is not None:
            self.load()
        return self._image_shape

    @image_shape.setter
    def image_shape(self, value):
        self._image_shape = value

    @property
    def counts(self):
        """The background counts data for this object.

        This will either be a map that is loaded directly from file,
        or an interpolated map based on the polynomial or scalar value
        mapped onto the image shape.

        This is a best-estimate (or, best-estimate-we-have-done, anyway)
        of the sky counts, ignoring as best as possible the sources in
        the sky, and looking only at the smoothed background level.

        """
        if self._counts_data is None and self.filepath is not None:
            self.load()
        return self._counts_data

    @counts.setter
    def counts(self, value):
        self._counts_data = value

    @property
    def variance(self):
        """The background variance data for this object.

        This will either be a map that is loaded directly from file,
        or an interpolated map based on the polynomial or scalar value
        mapped onto the image shape.

        This is a best-estimate of the sky noise, ignoring as best as
        possible the sources in the sky, and looking only at the smoothed
        background variability.
        """
        if self._var_data is None and self.filepath is not None:
            self.load()
        return self._var_data

    @variance.setter
    def variance(self, value):
        self._var_data = value

    @property
    def rms(self):
        if self.variance is None:
            return None
        return np.sqrt(self.variance)

    @rms.setter
    def rms(self, value):
        self.variance = value ** 2

    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object"""
        return bg_badness_inverse

    def __init__( self, *args, **kwargs ):
        FileOnDiskMixin.__init__( self, **kwargs )
        HasBitFlagBadness.__init__(self)
        SeeChangeBase.__init__( self )
        self._image_shape = None
        self._counts_data = None
        self._var_data = None

        if 'image_shape' in kwargs:
            self._image_shape = kwargs['image_shape']
        else:
            if ( 'sources_id' not in kwargs ) or ( kwargs['sources_id'] is None ):
                raise RuntimeError( "Error, can't figure out background image_shape.  Either explicitly pass "
                                    "image_shape, or make sure that sources_id is set, and the SourceList and "
                                    "Image are already saved to the database." )
            with SmartSession() as session:
                image = ( session.query( Image )
                          .join( SourceList, Image._id==SourceList.image_id )
                          .filter( SourceList._id==kwargs['sources_id'] )
                         ).first()
                if image is None:
                    raise RuntimeError( "Error, can't figure out background image_shape.  Either explicitly pass "
                                        "image_shape, or make sure that sources_id is set, and the SourceList and "
                                        "Image are already saved to the database." )
                # I don't like this; we're reading the image data just
                # to get its shape.  Perhaps we should add width and
                # height fields to the Image model?
                # (Or, really, when making a background, pass an image_shape!)
                wrnmsg = ( "Getting background shape from associated image.  This is inefficient. "
                           "Pass image_shape when constructing a background." )
                warnings.warn( wrnmsg )
                # SCLogger.warning( wrnmsg )
                self._image_shape = image.data.shape

        # Manually set all properties ( columns or not )
        for key, value in kwargs.items():
            if hasattr( self, key ):
                setattr( self, key, value )

    @sa.orm.reconstructor
    def init_on_load( self ):
        Base.init_on_load( self )
        FileOnDiskMixin.init_on_load( self )
        self._image_shape = None
        self._counts_data = None
        self._var_data = None


    def subtract_me( self, image ):
        """Subtract this background from an image.

        Parameters
        ----------
          image: numpy array
            shape must match self.image_shape (not checked)

        Returns
        -------
           numpy array : background-subtracted image
        """
        if self.format == 'scalar':
            return image - self.value
        elif self.format == 'map':
            return image - self.counts
        else:
            raise RuntimeError( f"Don't know how to subtract background of type {self.format}" )

    def save( self, filename=None, image=None, sources=None, **kwargs ):
        """Write the Background to disk.

        May or may not upload to the archive and update the
        FileOnDiskMixin-included fields of this object based on the
        additional arguments that are forwarded to FileOnDiskMixin.save.

        This saves an HDF5 file that contains a single group called "/background".
        It will have a few attributes, notably: "format", "value", "noise" and "image_shape".

        If the format is "map", there are two datasets under this group: "background/counts" and "background/variance".
        Counts represents the background counts at each location in the image, while the variance represents the noise
        variability that comes from the sky, ignoring the sources (as much as possible).

        If the format is "polynomial", there are three datasets:
        "background/coeffs" and "background/x_degree" and "background/y_degree".
        These will include the coefficients of the polynomial, and the degree of the polynomial in x and y as such:
        Constant term, x term, y term, x^2 term, xy term, y^2 term, x^3 term, x^2y term, xy^2 term, y^3 term, etc.
        Which corresponds to a list of degrees:
        x_degree: [0, 1, 0, 2, 1, 0, 3, 2, 1, 0, ...]
        y_degree: [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, ...]

        Finally, if the format is "scalar", there would not be any datasets.

        Parameters
        ----------
          filename: str or path
             The path to the file to write, relative to the local store
             root.  Do not include the extension (e.g. '.h5') at the
             end of the name; that will be added automatically for all
             extensions.  If None, will call image.invent_filepath() to get a
             filestore-standard filename and directory.

          image: Image (optional)
             Ignored if filename is not None.  If filename is None,
             will use this image's filepath to generate the background's
             filepath.  If both filename and image are None, will try
             to load the background's image from the database, if possible.

          sources: SourceList (optional)
             Ignored if filename is not None.  If filename is None,
             use this SourceList's provenance to genernate the background's
             filepath.  If both filename and soruces are None, will try to
             load the background's SourceList from the database, if possible.

          Additional arguments are passed on to FileOnDiskMixin.save()

        """
        if self.format not in ['scalar', 'map', 'polynomial']:
            raise ValueError(f'Unknown background format "{self.format}".')

        if self.value is None or self.noise is None:
            raise RuntimeError( "Both value and noise must be non-None" )

        if self.format == 'map' and (self.counts is None or self.variance is None):
            raise RuntimeError( "Both counts and variance must be non-None" )

        # TODO: add some checks for the polynomial format

        if filename is not None:
            if not filename.endswith('.h5'):
                filename += '.h5'
            self.filepath = filename
        else:
            if ( sources is None ) or ( image is None ):
                with SmartSession() as session:
                    if sources is None:
                        sources = SourceList.get_by_id( self.sources_id, session=session )
                    if ( sources is not None ) and ( image is None ):
                        image = Image.get_by_id( sources.image_id, session=session )
                if ( sources is None ) or ( image is None ):
                    raise RuntimeError( "Can't invent Background filepath; can't find either the corresponding "
                                        "SourceList or the corresponding Image." )

            self.filepath = image.filepath if image.filepath is not None else image.invent_filepath()
            self.filepath += f'.bg_{sources.provenance_id[:6]}.h5'

        h5path = os.path.join( self.local_path, f'{self.filepath}')

        with h5py.File(h5path, 'w') as h5f:
            bggrp = h5f.create_group('background')
            bggrp.attrs['format'] = self.format
            bggrp.attrs['method'] = self.method
            bggrp.attrs['value'] = self.value
            bggrp.attrs['noise'] = self.noise
            bggrp.attrs['image_shape'] = self.image_shape

            if self.format == 'map':
                if self.counts is None or self.variance is None:
                    raise RuntimeError("Both counts and variance must be non-None")
                if self.counts.shape != self.image_shape:
                    raise RuntimeError(
                        f"Counts shape {self.counts.shape} does not match image shape {self.image_shape}"
                    )
                if self.variance.shape != self.image_shape:
                    raise RuntimeError(
                        f"Variance shape {self.variance.shape} does not match image shape {self.image_shape}"
                    )

                opts = dict(compression='gzip', compression_opts=1, chunks=(128, 128))
                bggrp.create_dataset( 'counts', data=self.counts, **opts )
                bggrp.create_dataset( 'variance', data=self.variance, **opts )
            elif self.format == 'polynomial':
                raise NotImplementedError('Currently we do not support a polynomial background model. ')
                bggrp.create_dataset( 'coeffs', data=self.counts )
                bggrp.create_dataset( 'x_degree', data=self.x_degree )
                bggrp.create_dataset( 'y_degree', data=self.y_degree )
            elif self.format == 'scalar':
                pass  # no datasets to create
            else:
                raise ValueError( f'Unknown background format "{self.format}".' )

        # Save the file to the archive and update the database record
        # (From what we did above, the files are already in the right place in the local filestore.)
        FileOnDiskMixin.save( self, h5path, component=None, **kwargs )

    def load(self, download=True, always_verify_md5=False, filepath=None):
        """Load the data from the files into the _counts_data, _var_data and _image_shape fields.

        Parameters
        ----------
          download : Bool, default True
            If True, download the files from the archive if they're not
            found in local storage.  Ignored if filepath is not None.

          always_verify_md5 : Bool, default False
            If the file is found locally, verify the md5 of the file; if
            it doesn't match, re-get the file from the archive.  Ignored
            if filepath is not None.

        """
        if filepath is None:
            filepath = self.get_fullpath(download=download, always_verify_md5=always_verify_md5)

        with h5py.File(filepath, 'r') as h5f:
            if 'background' not in h5f:
                raise ValueError('No background group found in the file. ')
            loaded_format = h5f['background'].attrs['format']

            if self.format != loaded_format:
                raise ValueError(
                    f'Loaded background format "{loaded_format}" does not match the expected format "{self.format}".'
                )

            self.value = float(h5f['background'].attrs['value'])
            self.noise = float(h5f['background'].attrs['noise'])
            self.image_shape = tuple(h5f['background'].attrs['image_shape'])

            if loaded_format == 'map':
                self._counts_data = h5f['background/counts'][:]
                self._var_data = h5f['background/variance'][:]
            elif loaded_format == 'polynomial':
                raise NotImplementedError('Currently we do not support a polynomial background model. ')
                self._counts_data = h5f['background/coeffs'][:]
                self._x_degree = h5f['background/x_degree'][:]
                self._y_degree = h5f['background/y_degree'][:]
            elif loaded_format == 'scalar':
                pass
            else:
                raise ValueError( f'Unknown background format "{loaded_format}".' )

    def free( self ):
        """Free loaded world coordinates memory.

        Wipe out the _counts_data and _var_data fields, freeing memory.
        Depends on python garbage collection, so if there are other
        references to those objects, the memory won't actually be freed.

        """
        self._counts_data = None
        self._var_data = None

    @classmethod
    def copy_bg( cls, bg ):
        """Make a new Bcakground with the same data as an existing Background object.

        Does *not* set the sources_id field.

        """

        if bg is None:
            return None

        newbg = Background( _format = bg._format,
                            _method = bg._method,
                            _sources_id = None,
                            value = bg.value,
                            noisg = bg.noise,
                           )
        if bg.format == 'map':
            newbg.counts = bg.counts.copy()
            newbg.variance = bg.counts.copy()
        elif bg.format == 'polynomnial':
            newbg.coeffs = bg.coeffs.copy()
            newbg.x_degree = bg.coeffs.copy()
            newbg.y_degree = bg.coeffs.copy()

        return newbg

    def to_dict( self ):
        # Background needs special handling for to_dict, at least for the
        #   testing cache.  Normally, only the fields that would get saved
        #   to the database go into the dict.  However, in Background.__init__,
        #   it needs to be passed image_shape if the image and sources
        #   aren't already in the database (which they aren't in the test
        #   fixtures cache).  But, image_shape isn't saved to the database.
        #   So, add it.

        output = super().to_dict()
        output['image_shape'] = self.image_shape
        return output
