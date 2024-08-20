import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.exc import IntegrityError

from models.base import Base, SeeChangeBase, UUIDMixin, SmartSession
from models.provenance import Provenance


# provenance to refset association table:
refset_provenance_association_table = sa.Table(
    'refset_provenance_association',
    Base.metadata,
    sa.Column('provenance_id',
              sa.Text,
              sa.ForeignKey(
                  'provenances._id', ondelete="CASCADE", name='refset_provenances_association_provenance_id_fkey'
              ),
              primary_key=True),
    sa.Column('refset_id',
              sqlUUID,
              sa.ForeignKey('refsets._id', ondelete="CASCADE", name='refsets_provenances_association_refset_id_fkey'),
              primary_key=True),
)


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

    @classmethod
    def get_by_name( cls, name, session=None ):
        with SmartSession( session ) as sess:
            refset = sess.query( RefSet ).filter( RefSet.name==name ).first()
            return refset

    @property
    def provenances( self ):
        if self._provenances is None:
            self._provenances = self.get_provenances()
        return self._provenances

    @provenances.setter
    def provenances( self, val ):
        raise RuntimeError( "Don't set provenances directly, use append_provenance()" )

    def __init__(self, **kwargs):
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values
        self._provenances = None

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)
        self._provenances = None

    def get_provenances( self, session=None ):
        with SmartSession( session ) as sess:
            provs = ( sess.query( Provenance )
                      .join( refset_provenance_association_table,
                             refset_provenance_association_table.c.provenance_id == Provenance._id )
                      .filter( refset_provenance_association_table.c.refset_id == self.id )
                     ).all()
            self._provenances = provs
            return provs

    def append_provenance( self, prov, session=None ):
        """Add a provenance to this refset.

        Won't do anything if it's already there.

        """
        with SmartSession( session ) as sess:
            try:
                sess.connection().execute(
                    sa.text( 'INSERT INTO refset_provenance_association(provenance_id,refset_id) '
                             'VALUES(:provid,:refsetid)' ),
                    { 'provid': prov.id, 'refsetid': self.id } )
                sess.commit()
            except IntegrityError as ex:
                # It was already there, so we're good
                sess.rollback()

            # Refresh the self-list of provenances to include the added one.
            self._provenances = self.get_provenances( session=sess )
