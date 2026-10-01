"""Persist asteroid annotations and orbit snapshot metadata.

Revision ID: a61e0d3cf902
Revises: 8c3f73f0ede0
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'a61e0d3cf902'
down_revision = '8c3f73f0ede0'
branch_labels = None
depends_on = None


def timestamps():
    return [sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column('modified', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)]


def upgrade():
    op.create_table(
        'mpc_orbit_catalog',
        sa.Column('name', sa.Text, primary_key=True),
        sa.Column('version', sa.Text, nullable=False),
        sa.Column('epoch_jd', sa.Double, nullable=False),
        sa.Column('object_count', sa.Integer, nullable=False),
        *timestamps(),
    )
    op.create_index('ix_mpc_orbit_catalog_created_at', 'mpc_orbit_catalog', ['created_at'])
    op.create_table(
        'asteroid_match_sets',
        sa.Column('_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('measurementset_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('measurement_sets._id', ondelete='CASCADE'), nullable=False),
        sa.Column('provenance_id', sa.String,
                  sa.ForeignKey('provenances._id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.Text, nullable=False),
        sa.Column('catalog_version', sa.Text),
        sa.Column('catalog_epoch_jd', sa.Double),
        sa.Column('observation_jd', sa.Double),
        sa.Column('matches', postgresql.JSONB, nullable=False, server_default='{}'),
        *timestamps(),
        sa.UniqueConstraint('measurementset_id', 'provenance_id', name='_asteroid_match_sets_uc'),
        sa.CheckConstraint("status IN ('checked', 'disabled', 'unavailable', 'stale')",
                           name='asteroid_match_status_check'),
    )
    for column in ['created_at', 'measurementset_id', 'provenance_id']:
        op.create_index(f'ix_asteroid_match_sets_{column}', 'asteroid_match_sets', [column])


def downgrade():
    op.drop_table('asteroid_match_sets')
    op.drop_table('mpc_orbit_catalog')
