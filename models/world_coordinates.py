
import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base


class WorldCoordinates(Base):
    __tablename__ = 'world_coordinates'

    source_list_id = sa.Column(
        sa.ForeignKey('source_lists.id'),
        nullable=False,
        index=True,
        doc="ID of the source list this world coordinate system is associated with. "
    )

    source_list = orm.relationship(
        'SourceList',
        doc="The source list this world coordinate system is associated with. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this world coordinate system. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this world coordinate system. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this world coordinate system. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this world coordinate system. "
        )
    )

