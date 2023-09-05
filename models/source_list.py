import os

from functools import partial

import numpy as np
import pandas as pd

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property

from models.base import Base, FileOnDiskMixin, SeeChangeBase, file_format_enum
from models.image import Image


class SourceList(Base, FileOnDiskMixin):

    __tablename__ = 'source_lists'

    format = sa.Column(
        file_format_enum,
        nullable=False,
        default='fits',
        doc="Format of the file on disk. Should be fits, hdf5, csv or npy. "
    )

    image_id = sa.Column(
        sa.ForeignKey('images.id'),
        nullable=False,
        index=True,
        doc="ID of the image this source list was generated from. "
    )

    image = orm.relationship(
        'Image',
        lazy='selectin',
        doc="The image this source list was generated from. "
    )

    @hybrid_property
    def is_sub(self):
        """Whether this source list is from a subtraction image (detections),
        or from a regular image (sources, the default).
        """
        if self.image is None:
            return None
        else:
            return self.image.is_sub

    @is_sub.expression
    def is_sub(cls):
        return sa.select(Image.is_sub).where(Image.id == cls.image_id).label('is_sub')

    num_sources = sa.Column(
        sa.Integer,
        nullable=False,
        index=True,
        doc="Number of sources in this source list. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this source list. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this source list. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this source list. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this source list. "
        )
    )

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self._data = None

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)
        self._data = None

    def __repr__(self):
        output = (
            f'<SourceList(id={self.id}, '
            f'image_id={self.image_id}, '
            f'is_sub={self.is_sub}), '
            f'num_sources= {self.num_sources}>'
        )

        return output

    @property
    def data(self):
        """
        The data in this source list. A table of sources and their properties.
        """
        if self._data is None and self.filepath is not None:
            self._data = self.load()
        return self._data

    @data.setter
    def data(self, value):
        # TODO: add astropy table?
        if isinstance(value, pd.DataFrame):
            value = value.to_records(index=False)

        if not isinstance(value, np.ndarray) or value.dtype.names is None:
            raise TypeError("data must be a pandas DataFrame or numpy recarray")

        self._data = value
        self.num_sources = len(value)

    def load(self):
        """
        Load this source list from the database.
        """
        # TODO: should we replace this with FITS and astropy tables?
        return np.load(self.get_fullpath())  # this should always be a single file, right?

    def save(self):
        """
        Save the data table to a file on disk.
        """
        if self.image is None:
            raise ValueError("Cannot save source list without an image")

        if self.data is None:
            raise ValueError("Cannot save source list without data")

        filename = self.image.filepath
        if filename is None:
            filename = self.image.invent_filename()

        if filename.endswith(('.fits', '.h5', '.hdf5')):
            filename = os.path.splitext(filename)[0]

        # TODO: use FITS and astropy tables instead?
        filename += '_sources'
        filename += '.npy'
        fullname = os.path.join(self.local_path, filename)
        self.safe_mkdir(os.path.dirname(fullname))
        np.save(fullname, self.data)

        self.filepath = filename


# add "property" attributes to SourceList referencing the image for convenience
for att in [
    'section_id',
    'mjd',
    'ra',
    'dec',
    'filter',
    'filter_short',
    'telescope',
    'instrument',
    'instrument_object',
]:
    setattr(SourceList, att, property(fget=lambda self, att=att: getattr(self.image, att) if self.image is not None else None))
