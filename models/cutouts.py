
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property

from models.base import Base, SeeChangeBase, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed
from models.enums_and_bitflags import CutoutsFormatConverter

class Cutouts(Base, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed):

    __tablename__ = 'cutouts'

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=CutoutsFormatConverter.convert('fits'),
        doc="Format of the file on disk. Should be fits, hdf5, csv or npy. "
            "Saved as integer but is converter to string when loaded. "
    )

    @hybrid_property
    def format(self):
        return CutoutsFormatConverter.convert(self._format)

    @format.expression
    def format(cls):
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(CutoutsFormatConverter.dict, value=cls._format)

    @format.setter
    def format(self, value):
        self._format = CutoutsFormatConverter.convert(value)

    source_list_id = sa.Column(
        sa.ForeignKey('source_lists.id', name='cutouts_source_list_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the source list this cutout is associated with. "
    )

    source_list = orm.relationship(
        'SourceList',
        doc="The source list this cutout is associated with. "
    )

    new_image_id = sa.Column(
        sa.ForeignKey('images.id', name='cutouts_new_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the new science image this cutout is associated with. "
    )

    new_image = orm.relationship(
        'Image',
        primaryjoin="Cutouts.new_image_id==Image.id",
        doc="The new science image this cutout is associated with. "
    )

    ref_image_id = sa.Column(
        sa.ForeignKey('images.id', name='cutouts_ref_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the reference image this cutout is associated with. "
    )

    ref_image = orm.relationship(
        'Image',
        primaryjoin="Cutouts.ref_image_id==Image.id",
        doc="The reference image this cutout is associated with. "
    )

    sub_image_id = sa.Column(
        sa.ForeignKey('images.id', name='cutouts_sub_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the subtraction image this cutout is associated with. "
    )

    sub_image = orm.relationship(
        'Image',
        primaryjoin="Cutouts.sub_image_id==Image.id",
        doc="The subtraction image this cutout is associated with. "
    )

    pixel_x = sa.Column(
        sa.Integer,
        nullable=False,
        doc="X pixel coordinate of the center of the cutout. "
    )

    pixel_y = sa.Column(
        sa.Integer,
        nullable=False,
        doc="Y pixel coordinate of the center of the cutout. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='cutouts_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this cutout. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this cutout. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this cutout. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this cutout. "
        )
    )

    _bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        default=0,
        index=True,
        doc='Bitflag for these cutouts. Good cutouts have a bitflag of 0. '
            'Bad cutouts are each bad in their own way (i.e., have different bits set). '
            'Will include all the bits from data used to make these cutouts '
            '(e.g., the exposure it is based on). '
    )

    @hybrid_property
    def bitflag(self):
        return self._bitflag | self.image.bitflag

    @bitflag.expression
    def bitflag(cls):
        sa.select(Cutouts).where(
            Cutouts._bitflag,
            Cutouts.ref_image.bitflag,
            Cutouts.new_image.bitflag,
            Cutouts.sub_image.bitflag,
            Cutouts.source_list.bitflag,
        ).label('bitflag')

    @bitflag.setter
    def bitflag(self, value):
        self._bitflag = value

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
