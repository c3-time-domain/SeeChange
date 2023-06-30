
import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base


class ZeroPoint(Base):
    __tablename__ = 'zero_points'

    source_list_id = sa.Column(
        sa.ForeignKey('source_lists.id'),
        nullable=False,
        index=True,
        doc="ID of the source list this zero point is associated with. "
    )

    source_list = orm.relationship(
        'SourceList',
        doc="The source list this zero point is associated with. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this zero point. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this zero point. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this zero point. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this zero point. "
        )
    )
