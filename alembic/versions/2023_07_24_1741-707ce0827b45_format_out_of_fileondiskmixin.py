"""format_out_of_fileondiskmixin

Revision ID: 707ce0827b45
Revises: 2e42e5319395
Create Date: 2023-07-24 17:41:56.598780

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '707ce0827b45'
down_revision = '2e42e5319395'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('cutouts', 'format')
    op.drop_column('source_lists', 'format')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('source_lists', sa.Column('format', postgresql.ENUM('fits', 'hdf5', name='image_format'), autoincrement=False, nullable=False))
    op.add_column('cutouts', sa.Column('format', postgresql.ENUM('fits', 'hdf5', name='image_format'), autoincrement=False, nullable=False))
    # ### end Alembic commands ###