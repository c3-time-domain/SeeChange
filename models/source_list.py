import os

from functools import partial

import numpy as np
import pandas as pd

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql.functions import coalesce

from models.base import Base, AutoIDMixin, FileOnDiskMixin, SeeChangeBase
from models.image import Image
from models.enums_and_bitflags import (
    SourceListFormatConverter,
    bitflag_to_string,
    string_to_bitflag,
    data_badness_dict,
    source_list_badness_inverse,
)


class SourceList(Base, AutoIDMixin, FileOnDiskMixin):

    __tablename__ = 'source_lists'

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=SourceListFormatConverter.convert('npy'),
        doc="Format of the file on disk. Should be fits, hdf5, csv or npy. "
            "Saved as integer but is converter to string when loaded. "
    )

    @hybrid_property
    def format(self):
        return SourceListFormatConverter.convert(self._format)

    @format.expression
    def format(cls):
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(SourceListFormatConverter.dict, value=cls._format)

    @format.setter
    def format(self, value):
        self._format = SourceListFormatConverter.convert(value)

    image_id = sa.Column(
        sa.ForeignKey('images.id', name='source_lists_image_id_fkey'),
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
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='source_lists_provenance_id_fkey'),
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

    _bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        default=0,
        index=True,
        doc='Bitflag for this source list. Good source lists have a bitflag of 0. '
            'Bad source list are each bad in their own way (i.e., have different bits set). '
            'Will include all the bits from data used to make this source list '
            '(e.g., the exposure it is based on). '
    )

    @hybrid_property
    def bitflag(self):
        return self._bitflag | self.image.bitflag

    @bitflag.inplace.expression
    @classmethod
    def bitflag(cls):
        stmt = sa.select(coalesce(cls._bitflag, 0).op('|')(Image.bitflag))
        stmt = stmt.where(cls.image_id == Image.id)
        stmt = stmt.scalar_subquery()
        return stmt

    @bitflag.setter
    def bitflag(self, value):
        self._bitflag = value

    @property
    def badness(self):
        """
        A comma separated string of keywords describing
        why this data is not good, based on the bitflag.
        This includes all the reasons this data is bad,
        including the parent data models that were used
        to create this data (e.g., the Exposure underlying
        the Image).
        """
        return bitflag_to_string(self.bitflag, data_badness_dict)

    @badness.setter
    def badness(self, value):
        """Set the badness for this image using a comma separated string. """
        self.bitflag = string_to_bitflag(value, source_list_badness_inverse)

    def append_badness(self, value):
        """Add some keywords (in a comma separated string)
        describing what is bad about this image.
        The keywords will be added to the list "badness"
        and the bitflag for this image will be updated accordingly.
        """
        self.bitflag |= string_to_bitflag(value, source_list_badness_inverse)

    description = sa.Column(
        sa.Text,
        nullable=True,
        doc='Free text comment about this source list, e.g., why it is bad. '
    )

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self._data = None
        self._bitflag = 0

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

    def save(self, **kwargs):
        """
        Save the data table to a file on disk.
        """
        if self.image is None:
            raise ValueError("Cannot save source list without an image")

        if self.data is None:
            raise ValueError("Cannot save source list without data")

        filename = self.image.filepath
        if filename is None:
            filename = self.image.invent_filepath()

        if filename.endswith(('.fits', '.h5', '.hdf5')):
            filename = os.path.splitext(filename)[0]

        # TODO: use FITS and astropy tables instead?
        filename += '_sources'
        filename += '.npy'
        fullname = os.path.join(self.local_path, filename)
        self.safe_mkdir(os.path.dirname(fullname))
        np.save(fullname, self.data)

        self.filepath = filename
        super().save(fullname, **kwargs)

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
