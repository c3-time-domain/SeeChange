import os

import numpy as np

import h5py

import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import UniqueConstraint

from models.base import Base, SeeChangeBase, SmartSession, AutoIDMixin, FileOnDiskMixin, HasBitFlagBadness
from models.image import Image

from models.enums_and_bitflags import BackgroundFormatConverter, BackgroundMethodConverter, bg_badness_inverse


class Background(Base, AutoIDMixin, FileOnDiskMixin, HasBitFlagBadness):
    __tablename__ = 'backgrounds'

    __table_args__ = (
        UniqueConstraint('image_id', 'provenance_id', name='_bg_image_provenance_uc'),
    )

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=BackgroundFormatConverter.convert('scalar'),
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
        default=BackgroundMethodConverter.convert('zero'),
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

    image_id = sa.Column(
        sa.ForeignKey('images.id', ondelete='CASCADE', name='backgrounds_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the image for which this is the background."
    )

    image = orm.relationship(
        'Image',
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        lazy='selectin',
        doc="Image for which this is the background."
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

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='backgrounds_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this Background object. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this Background object."
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this Background object. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this Background object."
        )
    )

    __table_args__ = (
        sa.Index( 'backgrounds_image_id_provenance_index', 'image_id', 'provenance_id', unique=True ),
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

        This is a best-estimate of the sky counts, ignoring as best as
        possible the sources in the sky, and looking only at the smoothed
        background level.
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

    def __setattr__(self, key, value):
        if key == 'image':
            if value is not None and not isinstance(value, Image):
                raise ValueError(f'Background.image must be an Image object. Got {type(value)} instead. ')
            self._image_shape = value.data.shape

        super().__setattr__(key, value)

    def save( self, filename=None, **kwargs ):
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
            if self.image.filepath is not None:
                self.filepath = self.image.filepath
            else:
                self.filepath = self.image.invent_filepath()

            if self.provenance is None:
                raise RuntimeError("Can't invent a filepath for the Background without a provenance")
            self.filepath += f'.bg_{self.provenance.id[:6]}.h5'

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

                bggrp.create_dataset( 'counts', data=self.counts )
                bggrp.create_dataset( 'variance', data=self.variance )
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
        FileOnDiskMixin.save( self, h5path, extension=None, **kwargs )

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

    def get_upstreams(self, session=None):
        """Get the image that was used to make this Background object. """
        with SmartSession(session) as session:
            return session.scalars(sa.select(Image).where(Image.id == self.image_id)).all()

    def get_downstreams(self, session=None, siblings=False):
        """Get the downstreams of this Background object.

        If siblings=True then also include the SourceList, PSF, WCS, and ZP
        that were created at the same time as this PSF.
        """
        from models.source_list import SourceList
        from models.psf import PSF
        from models.world_coordinates import WorldCoordinates
        from models.zero_point import ZeroPoint
        from models.provenance import Provenance

        with SmartSession(session) as session:
            output = []
            if self.image_id is not None and self.provenance is not None:
                subs = session.scalars(
                    sa.select(Image).where(
                        Image.provenance.has(Provenance.upstreams.any(Provenance.id == self.provenance.id)),
                        Image.upstream_images.any(Image.id == self.image_id),
                    )
                ).all()
                output += subs

            if siblings:
                # There should be exactly one source list, wcs, and zp per PSF, with the same provenance
                # as they are created at the same time.
                sources = session.scalars(
                    sa.select(SourceList).where(
                        SourceList.image_id == self.image_id, SourceList.provenance_id == self.provenance_id
                    )
                ).all()
                if len(sources) != 1:
                    raise ValueError(
                        f"Expected exactly one source list for Background {self.id}, but found {len(sources)}"
                    )

                output.append(sources[0])

                psfs = session.scalars(
                    sa.select(PSF).where(PSF.image_id == self.image_id, PSF.provenance_id == self.provenance_id)
                ).all()
                if len(psfs) != 1:
                    raise ValueError(f"Expected exactly one PSF for Background {self.id}, but found {len(psfs)}")

                output.append(psfs[0])

                wcs = session.scalars(
                    sa.select(WorldCoordinates).where(WorldCoordinates.sources_id == sources.id)
                ).all()
                if len(wcs) != 1:
                    raise ValueError(f"Expected exactly one wcs for Background {self.id}, but found {len(wcs)}")

                output.append(wcs[0])

                zp = session.scalars(sa.select(ZeroPoint).where(ZeroPoint.sources_id == sources.id)).all()

                if len(zp) != 1:
                    raise ValueError(f"Expected exactly one zp for Background {self.id}, but found {len(zp)}")

                output.append(zp[0])

        return output
