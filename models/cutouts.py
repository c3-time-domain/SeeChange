
import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, FileOnDiskMixin, SpatiallyIndexed


class Cutouts(Base, FileOnDiskMixin, SpatiallyIndexed):

    __tablename__ = 'cutouts'

    source_list_id = sa.Column(
        sa.ForeignKey('source_lists.id'),
        nullable=False,
        index=True,
        doc="ID of the source list this cutout is associated with. "
    )

    source_list = orm.relationship(
        'SourceList',
        doc="The source list this cutout is associated with. "
    )

    new_image_id = sa.Column(
        sa.ForeignKey('images.id'),
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
        sa.ForeignKey('images.id'),
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
        sa.ForeignKey('images.id'),
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
        sa.ForeignKey('provenances.id', ondelete="CASCADE"),
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

