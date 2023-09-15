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
    # Alembic autogenerate totally screwed this up, so these were all
    # done manually.
    #
    # This migration DOES NOT preserve database information!  To do
    # that, we'd also need to write the commands that connect the ids of
    # the current foreign keys to the versions and hashes that are
    # replacing them.  Since we don't have any existing databases that
    # can't be blow away right now, I'm not worrying about this.
    op.drop_index('ix_code_hashes_hash', table_name='code_hashes')
    op.drop_column( 'code_hashes', 'id' )
    op.alter_column( 'code_hashes', 'hash', new_column_name='id' )
    op.create_primary_key( 'pk_code_hashes', 'code_hashes', [ 'id' ] )

    op.drop_column( 'code_hashes', 'code_version_id' )
    op.drop_column( 'provenances', 'code_version_id' )
    op.drop_index('ix_code_versions_version', table_name='code_versions')
    op.drop_column( 'code_versions', 'id' )
    op.alter_column( 'code_versions', 'version', new_column_name='id' )
    op.create_primary_key( 'pk_code_versions', 'code_versions', [ 'id' ] )
    op.add_column( 'provenances', sa.Column( "code_version_id", sa.String(),  nullable=False ) )
    op.create_foreign_key( 'provenances_code_version_id_fkey', 'provenances', 'code_versions',
                           [ 'code_version_id' ], [ 'id' ], ondelete='CASCADE' )
    op.create_index( 'ix_provenances_code_version_id', 'provenances', ['code_version_id'], unique=False )
    op.add_column( 'code_hashes', sa.Column( "code_version_id", sa.String() ) )
    op.create_foreign_key( 'code_hashes_code_version_id_fkey', 'code_hashes', 'code_versions',
                           [ 'code_version_id' ], [ 'id' ], ondelete='CASCADE' )
    op.create_index( 'ix_code_hashes_code_version_id', 'code_hashes', ['code_version_id'], unique=False )


    op.drop_table( 'provenance_upstreams' )

    op.drop_column( 'cutouts', 'provenance_id' )
    op.drop_column( 'images', 'provenance_id' )
    op.drop_column( 'measurements', 'provenance_id' )
    op.drop_column( 'source_lists', 'provenance_id' )
    op.drop_column( 'world_coordinates', 'provenance_id' )
    op.drop_column( 'zero_points', 'provenance_id' )
    op.drop_column( 'provenances', 'replaced_by' )
    op.drop_index('ix_provenances_unique_hash', table_name='provenances')
    op.drop_column( 'provenances', 'id' )
    op.alter_column( 'provenances', 'unique_hash', new_column_name='id' )
    op.create_primary_key( 'pk_provenances', 'provenances', [ 'id' ] )
    op.add_column( 'provenances', sa.Column( "replaced_by", sa.String(), nullable=True ) )
    op.create_foreign_key( 'provenances_replaced_by_fkey', 'provenances', 'provenances',
                           [ 'replaced_by' ], [ 'id' ], ondelete='SET NULL' )
    op.create_index( 'ix_provenances_replaced_by', 'provenances', ['replaced_by'], unique=False )
    op.add_column( 'zero_points', sa.Column( "provenance_id", sa.String(), nullable=False ) )
    op.create_foreign_key( 'zero_points_provenance_id_fkey', 'zero_points', 'provenances',
                           [ 'provenance_id' ], [ 'id' ], ondelete='CASCADE' )
    op.create_index( 'ix_zero_points_provenance_id', 'zero_points', ['provenance_id'], unique=False )
    op.add_column( 'world_coordinates', sa.Column( "provenance_id", sa.String(), nullable=False ) )
    op.create_foreign_key( 'world_coordinates_provenance_id_fkey', 'world_coordinates', 'provenances',
                           [ 'provenance_id' ], [ 'id' ], ondelete='CASCADE' )
    op.create_index( 'ix_world_coordinates_provenance_id', 'world_coordinates', ['provenance_id'], unique=False )
    op.add_column( 'source_lists', sa.Column( "provenance_id", sa.String(), nullable=False ) )
    op.create_foreign_key( 'source_lists_provenance_id_fkey', 'source_lists', 'provenances',
                           [ 'provenance_id' ], [ 'id' ], ondelete='CASCADE' )
    op.create_index( 'ix_source_lists_provenance_id', 'source_lists', ['provenance_id'], unique=False )
    op.add_column( 'measurements', sa.Column( "provenance_id", sa.String(), nullable=False ) )
    op.create_foreign_key( 'measurements_provenance_id_fkey', 'measurements', 'provenances',
                           [ 'provenance_id' ], [ 'id' ], ondelete='CASCADE' )
    op.create_index( 'ix_measurements_provenance_id', 'measurements', ['provenance_id'], unique=False )
    op.add_column( 'images', sa.Column( "provenance_id", sa.String(), nullable=False ) )
    op.create_foreign_key( 'images_provenance_id_fkey', 'images', 'provenances',
                           [ 'provenance_id' ], [ 'id' ], ondelete='CASCADE' )
    op.create_index( 'ix_images_provenance_id', 'images', ['provenance_id'], unique=False )
    op.add_column( 'cutouts', sa.Column( "provenance_id", sa.String(), nullable=False ) )
    op.create_foreign_key( 'cutouts_provenance_id_fkey', 'cutouts', 'provenances',
                           [ 'provenance_id' ], [ 'id' ], ondelete='CASCADE' )
    op.create_index( 'ix_cutouts_provenance_id', 'cutouts', ['provenance_id'], unique=False )

    op.create_table( 'provenance_upstreams',
                     sa.Column('upstream_id', sa.String(), nullable=False),
                     sa.Column('downstream_id', sa.String(), nullable=False),
                     sa.ForeignKeyConstraint(['downstream_id'], ['provenances.id'], ondelete='CASCADE',
                                             name='provenance_upstreams_downstream_id_fkey'),
                     sa.ForeignKeyConstraint(['upstream_id'], ['provenances.id'], ondelete='CASCADE',
                                             name='provenance_upstreams_upstream_id_fkey'),
                     sa.PrimaryKeyConstraint('upstream_id', 'downstream_id')
                    )


def downgrade() -> None:
    raise Exception( "Irreversable migration." )
