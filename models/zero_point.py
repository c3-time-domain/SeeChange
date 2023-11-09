
import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, AutoIDMixin


class ZeroPoint(Base, AutoIDMixin):
    __tablename__ = 'zero_points'

    source_list_id = sa.Column(
        sa.ForeignKey('source_lists.id', ondelete='CASCADE', name='zero_points_source_list_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the source list this zero point is associated with. "
    )

    source_list = orm.relationship(
        'SourceList',
        lazy='selectin',
        doc="The source list this zero point is associated with. "
    )

    # TODO : figure this out
    # image = orm.relationship(
    #     'Image',
    #     secondary='source_lists',
    #     primaryjoin='zero_points.c.source_list_id == source_lists.c.id',
    #     secondaryjoin='source_lists.c.image_id == images.c.id',
    #     single_parent=True
    # )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='zero_points_provenance_id_fkey'),
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

    zp = sa.Column(
        sa.Float,
        nullable=False,
        index=False,
        doc="Zeropoint: -2.5*log10(flux_psf) + zp = mag"
    )

    dzp = sa.Column(
        sa.Float,
        nullable=False,
        index=False,
        doc="Uncertainty on zp"
    )
