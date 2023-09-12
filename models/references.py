import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, AutoIDMixin


class ReferenceEntry(Base, AutoIDMixin):
    """
    A table that refers to each reference Image object,
    based on the validity time range, and the object/field it is targeting.
    """

    __tablename__ = 'reference_images'

    image_id = sa.Column(
        sa.ForeignKey('images.id', ondelete='CASCADE', name='reference_images_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the reference image this object is referring to. "
    )

    image = orm.relationship(
        'Image',
        doc="The reference image this entry is referring to. "
    )

    target = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc=(
            'Name of the target object or field id. '
            'This string is used to match the reference to new images, '
            'e.g., by matching the field ID on a pre-defined grid of fields. '
        )
    )

    filter = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Filter used to make the images for this reference image. "
    )

    section_id = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Section ID of the reference image. "
    )

    validity_start = sa.Column(
        sa.DateTime,
        nullable=False,
        index=True,
        doc="The start of the validity time range of this reference image. "
    )

    validity_end = sa.Column(
        sa.DateTime,
        nullable=False,
        index=True,
        doc="The end of the validity time range of this reference image. "
    )

    is_bad = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc="Whether this reference image is bad. "
    )

    bad_reason = sa.Column(
        sa.Text,
        nullable=True,
        doc=(
            "The reason why this reference image is bad. "
            "Should be a single pharse or a comma-separated list of reasons. "
        )
    )

    bad_comment = sa.Column(
        sa.Text,
        nullable=True,
        doc="Any additional comments about why this reference image is bad. "
    )

    # this table doesn't have provenance.
    # The underlying image will have its own provenance for the "coaddition" process.

