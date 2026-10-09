import sqlalchemy as sa
from sqlalchemy import orm
from psycopg import sql

from models.base import Base, SeeChangeBase, UUIDMixin, PGDB
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
    def get_by_name( cls, name, pgdb=None, session=None ):
        pgdb = pgdb if pgdb is not None else session
        with PGDB( pgdb, dictcursor=True ) as pgdb:
            rows = pgdb.execute( sql.SQL( "SELECT * FROM refsets WHERE name={name}" ).format( name=name ) )
            if len(rows) == 0:
                return None
            else:
                return RefSet( **(rows[0]) )

    @property
    def provenance( self ):
        if self._provenance is None:
            self._provenance = Provenance.get( self.provenance_id )
        return self._provenance

    @provenance.setter
    def provenance( self, prov ):
        self._provenance = prov
        self.provenance_id = prov.id

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


    def __repr__( self ):
        return f"RefSet(id: {self.id}, name: {self.name}, provenance_id: {self.provenance_id})"
