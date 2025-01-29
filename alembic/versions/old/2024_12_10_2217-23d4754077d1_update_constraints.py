"""update_constraints

Revision ID: 23d4754077d1
Revises: a956948a16c4
Create Date: 2024-12-10 22:17:21.825710

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '23d4754077d1'
down_revision = 'a956948a16c4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    # Manually add back in the constraints that alembic failed to detect in the
    #   previous migration.  Use SQL after spending 5 minutes trying to find how
    #   to add an SA constraint to an exsiting table and giving up.  (All the docs
    #   say how to put it in the table when you create it.)
    conn.execute( sa.text( "ALTER TABLE catalog_excerpts "
                           "ADD CONSTRAINT catalog_excerpts_md5sum_check "
                           "CHECK( NOT(md5sum IS NULL AND md5sum_components IS NULL "
                           "           OR array_position(md5sum_components, NULL) IS NOT NULL))" ) )
    conn.execute( sa.text( "ALTER TABLE data_files "
                           "ADD CONSTRAINT data_files_md5sum_check "
                           "CHECK( NOT(md5sum IS NULL AND md5sum_components IS NULL "
                           "           OR array_position(md5sum_components, NULL) IS NOT NULL))" ) )
    conn.execute( sa.text( "ALTER TABLE exposures "
                           "ADD CONSTRAINT exposures_md5sum_check "
                           "CHECK( NOT(md5sum IS NULL AND md5sum_components IS NULL "
                           "           OR array_position(md5sum_components, NULL) IS NOT NULL))" ) )
    conn.execute( sa.text( "ALTER TABLE images "
                           "ADD CONSTRAINT images_md5sum_check "
                           "CHECK( NOT(md5sum IS NULL AND md5sum_components IS NULL "
                           "           OR array_position(md5sum_components, NULL) IS NOT NULL))" ) )
    conn.execute( sa.text( "ALTER TABLE source_lists "
                           "ADD CONSTRAINT source_lists_md5sum_check "
                           "CHECK( NOT(md5sum IS NULL AND md5sum_components IS NULL "
                           "           OR array_position(md5sum_components, NULL) IS NOT NULL))" ) )
    conn.execute( sa.text( "ALTER TABLE backgrounds "
                           "ADD CONSTRAINT backgrounds_md5sum_check "
                           "CHECK( NOT(md5sum IS NULL AND md5sum_components IS NULL "
                           "           OR array_position(md5sum_components, NULL) IS NOT NULL))" ) )
    conn.execute( sa.text( "ALTER TABLE psfs "
                           "ADD CONSTRAINT psfs_md5sum_check "
                           "CHECK( NOT(md5sum IS NULL AND md5sum_components IS NULL "
                           "           OR array_position(md5sum_components, NULL) IS NOT NULL))" ) )
    conn.execute( sa.text( "ALTER TABLE world_coordinates "
                           "ADD CONSTRAINT world_coordinates_md5sum_check "
                           "CHECK( NOT(md5sum IS NULL AND md5sum_components IS NULL "
                           "           OR array_position(md5sum_components, NULL) IS NOT NULL))" ) )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute( sa.text( "ALTER TABLE catalog_excerpts DROP CONSTRAINT catalog_excerpts_md5sum_check" ) )
    conn.execute( sa.text( "ALTER TABLE data_files DROP CONSTRAINT data_files_md5sum_check" ) )
    conn.execute( sa.text( "ALTER TABLE exposures DROP CONSTRAINT exposures_md5sum_check" ) )
    conn.execute( sa.text( "ALTER TABLE images DROP CONSTRAINT images_md5sum_check" ) )
    conn.execute( sa.text( "ALTER TABLE source_lists DROP CONSTRAINT source_lists_md5sum_check" ) )
    conn.execute( sa.text( "ALTER TABLE backgrounds DROP CONSTRAINT backgrounds_md5sum_check" ) )
    conn.execute( sa.text( "ALTER TABLE psfs DROP CONSTRAINT psfs_md5sum_check" ) )
    conn.execute( sa.text( "ALTER TABLE world_coordinates DROP CONSTRAINT world_coordinates_md5sum_check" ) )
    pass
