import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.schema import CheckConstraint

from models.base import Base, SeeChangeBase, UUIDMixin, FileOnDiskMixin


class DataFile( Base, UUIDMixin, FileOnDiskMixin ):
    """Miscellaneous data files."""

    __tablename__ = "data_files"

    @declared_attr
    def __table_args__(cls):
        return (
            CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
                             '(md5sum_extensions IS NULL OR array_position(md5sum_extensions, NULL) IS NOT NULL))',
                             name=f'{cls.__tablename__}_md5sum_check' ),

        )


    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances._id', ondelete='CASCADE', name='data_files_provenance_id_fkey' ),
        nullable=False,
        index=True,
        doc="ID of the provenance of this miscellaneous data file"
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

    def get_downstreams( self, session=None ):
        # DataFile has no downstreams
        return []

    def __repr__(self):
        return (
            f'<DataFile('
            f'id={self.id}, '
            f'filepath={self.filepath}, '
            f'>'
        )

    # ======================================================================
    # The fields below are things that we've deprecated; these definitions
    #   are here to catch cases in the code where they're still used

    @property
    def provenance( self ):
        raise RuntimeError( f"Datafile.provenance is deprecated, don't use it" )

    @provenance.setter
    def provenance( self, val ):
        raise RuntimeError( f"Datafile.provenance is deprecated, don't use it" )

