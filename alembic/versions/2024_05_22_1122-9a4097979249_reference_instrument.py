"""reference instrument

Revision ID: 9a4097979249
Revises: 485334f16c23
Create Date: 2024-05-22 11:22:20.322800

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9a4097979249'
down_revision = '485334f16c23'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('refs', sa.Column('instrument', sa.Text(), nullable=False))
    op.create_index(op.f('ix_refs_instrument'), 'refs', ['instrument'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_refs_instrument'), table_name='refs')
    op.drop_column('refs', 'instrument')
    # ### end Alembic commands ###