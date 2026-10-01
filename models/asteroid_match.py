"""Persistent asteroid annotations, independent of measurement quality and alert selection."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from models.base import Base, UUIDMixin, SmartSession


class AsteroidMatchSet(Base, UUIDMixin):
    __tablename__ = 'asteroid_match_sets'
    __table_args__ = (
        sa.UniqueConstraint('measurementset_id', 'provenance_id', name='_asteroid_match_sets_uc'),
        sa.CheckConstraint("status IN ('checked', 'disabled', 'unavailable', 'stale')",
                           name='asteroid_match_status_check'),
    )

    measurementset_id = sa.Column(sa.ForeignKey('measurement_sets._id', ondelete='CASCADE'),
                                  nullable=False, index=True)
    provenance_id = sa.Column(sa.ForeignKey('provenances._id', ondelete='CASCADE'), nullable=False, index=True)
    status = sa.Column(sa.Text, nullable=False)
    catalog_version = sa.Column(sa.Text, nullable=True)
    catalog_epoch_jd = sa.Column(sa.Double, nullable=True)
    observation_jd = sa.Column(sa.Double, nullable=True)
    # Keys are measurement UUIDs, so filtering or reordering measurements cannot shift annotations.
    matches = sa.Column(JSONB, nullable=False, server_default='{}')

    def get_upstreams(self, session=None):
        from models.measurements import MeasurementSet
        with SmartSession(session) as sess:
            return list(sess.scalars(sa.select(MeasurementSet).where(MeasurementSet._id == self.measurementset_id)))

    def get_downstreams(self, session=None):
        return []

    def annotation(self, measurement):
        matches = (self.matches or {}).get(str(measurement.id), [])
        return {
            'mpcDesignation': matches[0]['designation'] if matches else None,
            'mpcMatches': matches,
            'mpcMatchStatus': self.status,
            'mpcCatalogVersion': self.catalog_version,
            'mpcCatalogEpoch': self.catalog_epoch_jd,
        }


class MPCOrbitCatalog(Base):
    """Metadata for the atomically replaced, current heliocentric orbit snapshot."""

    __tablename__ = 'mpc_orbit_catalog'
    name = sa.Column(sa.Text, primary_key=True)
    version = sa.Column(sa.Text, nullable=False)
    epoch_jd = sa.Column(sa.Double, nullable=False)
    object_count = sa.Column(sa.Integer, nullable=False)
