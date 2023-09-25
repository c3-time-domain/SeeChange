import sqlalchemy as sa

from models.base import Base, AutoIDMixin, FileOnDiskMixin

class DataFile( Base, AutoIDMixin, FileOnDiskMixin ):
    """Miscellaneous data files."""

    __tablename__ = "data_files"

    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances.id', ondelete='CASCADE', name='data_files_provenance_id_fkey' ),
        nullable=False,
        index=True,
        doc="ID of the provenance of this miscellaneous data file"
    )
