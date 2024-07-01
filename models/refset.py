import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, SeeChangeBase, AutoIDMixin, SmartSession
from models.provenance import Provenance


# provenance to refset association table:
refset_provenance_association_table = sa.Table(
    'refset_provenance_association',
    Base.metadata,
    sa.Column('provenance_id',
              sa.Text,
              sa.ForeignKey(
                  'provenances.id', ondelete="CASCADE", name='refset_provenances_association_provenance_id_fkey'
              ),
              primary_key=True),
    sa.Column('refset_id',
              sa.Integer,
              sa.ForeignKey('refsets.id', ondelete="CASCADE", name='refsets_provenances_association_refset_id_fkey'),
              primary_key=True),
)


class RefSet(Base, AutoIDMixin):
    __tablename__ = 'refsets'

    name = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        unique=True,
        doc="Name of the reference set. "
    )

    description = sa.Column(
        sa.Text,
        nullable=True,
        doc="Description of the reference set. "
    )

    upstream_hash = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Hash of the upstreams used to make the reference provenance. "
    )

    provenances = orm.relationship(
        Provenance,
        secondary=refset_provenance_association_table,
        backref='refsets',  # add refsets attribute to Provenance
        order_by=Provenance.created_at,
        cascade='all'
    )

    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)


