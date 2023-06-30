import uuid
import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, FileOnDiskMixin


class SourceList(Base, FileOnDiskMixin):

    __tablename__ = 'source_lists'

    image_id = sa.Column(
        sa.ForeignKey('images.id'),
        nullable=False,
        index=True,
        doc="ID of the image this source list was generated from. "
    )

    image = orm.relationship(
        'Image',
        doc="The image this source list was generated from. "
    )

    is_sub = sa.Column(
        sa.Boolean,
        nullable=False,
        default=False,
        doc=(
            "Whether this source list is from a subtraction image (detections), "
            "or from a regular image (sources, the default). "
        )
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

    def save(self):
        """
        Save this source list to the database.
        """
        # TODO: Must implement this at some point!
        self.filepath = uuid.uuid4().hex

