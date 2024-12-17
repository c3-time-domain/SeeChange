import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, SeeChangeBase, UUIDMixin, SmartSession
from models.provenance import Provenance


class RefSet(Base, UUIDMixin):
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

    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances._id', ondelete='CASCADE', name='refset_provenance_id_fkey' ),
        nullable=False,
        index=True,
        doc="Provenance of the ref for this refset"
    )

    @classmethod
    def get_by_name( cls, name, session=None ):
        with SmartSession( session ) as sess:
            refset = sess.query( RefSet ).filter( RefSet.name==name ).first()
            return refset

    @property
    def provenance( self ):
        if self._provenance is None:
            self._provenance = Provenance.get( self.provenance_id )
        return self._provenance


    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values
        self._provenance = None

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)
        self._provenance = None
