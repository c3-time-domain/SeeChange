import sqlalchemy as sa
from sqlalchemy import orm

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

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this data file. "
            "The provenance will contain a record of the code version "
            "and the parameters used to produce this image. "
        )
    )
