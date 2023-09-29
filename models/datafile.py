import sqlalchemy as sa
from sqlalchemy import orm

from models.base import Base, SeeChangeBase, AutoIDMixin, FileOnDiskMixin

class DataFile( Base, AutoIDMixin, FileOnDiskMixin ):
    """Miscellaneous data files."""

    __tablename__ = "data_files"

    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances.id', ondelete='CASCADE', name='data_files_provenance_id_fkey' ),
        nullable=False,
        index=True,
        doc="ID of the provenance of this miscellaneous data file"
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this data file. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this file. "
        )
    )

    def __init__( self, *args, **kwargs ):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @orm.reconstructor
    def init_on_load( self ):
        Base.init_on_load( self )
        FileOnDiskMixin.init_on_load( self )
