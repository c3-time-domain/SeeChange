"""oops

Revision ID: de25d73b31db
Revises: 54fd73b6510e
Create Date: 2026-10-13 15:17:37.863720

"""

from alembic import op

from models.base import PGDB

# revision identifiers, used by Alembic.
revision = 'de25d73b31db'
down_revision = '54fd73b6510e'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Alembic totally detected the wrong thing (it just dropped and
    #   added columns rather than renaming), so just do it manually.
    #   (Which means using SQL, because I know it and looking up SA
    #   and/or Alembic notation is painful, and it's absolutely as
    #   byzantine as SQL, if not more.)
    with PGDB( op.get_bind().connection.driver_connection ) as pgdb:
        pgdb.execute( "ALTER TABLE image_subtraction_components "
                      "RENAME COLUMN warped_ref_source_id TO warped_ref_sources_id" )

def downgrade() -> None:
    with PGDB( op.get_bind().connection.driver_connection ) as pgdb:
        pgdb.execute( "ALTER TABLE image_subtraction_components "
                      "RENAME COLUMN warped_ref_sources_id TO warped_ref_source_id" )
