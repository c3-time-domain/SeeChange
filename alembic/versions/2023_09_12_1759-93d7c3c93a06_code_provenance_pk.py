"""code_provenance_pk

Revision ID: 93d7c3c93a06
Revises: 04e5cdfa1ad9
Create Date: 2023-09-12 17:59:07.897131

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '93d7c3c93a06'
down_revision = '04e5cdfa1ad9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Alembic autogenerate totally screwed this up, so these were all done manually
    op.drop_column( 'code_hashes', 'id' )
    op.alter_column( 'code_hashes', 'hash', new_column_name='id' )
    op.create_primary_key( 'pk_code_hashes', 'code_hashes', [ 'id' ] )

    op.drop_constraint( 
    op.drop_column( 'code_versions', 'id' )
    op.alter_column( 'code_versions', 'version', new_column_name='id' )
    op.create_primary_key( 'pk_code_versions', 'code_versions', [ 'id' ] )

    op.drop_column( 'provenances', 'id' )
    op.alter_column( 'provenances', 'unique_hash', new_column_name='id' )
    op.create_primary_key( 'pk_provenances', 'provenances', [ 'id' ] )


def downgrade() -> None:
    raise Exception( "Irreversable migration." )
