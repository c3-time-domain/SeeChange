import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declared_attr

from models.base import Base

class MPCTable(Base):

    __tablename__ = 'mpc_table'

    # This is the primary key for the table.
    designation = sa.Column(
        sa.Text,
        primary_key=True,
        nullable=False,
        doc=("MPC designation of the object.")
    )

    jd = sa.Column(
        sa.Double,
        nullable=False,
        doc=("Julian Date of the observation.")
    )
    
    ra = sa.Column(
        sa.Double,
        nullable=False,
        doc=("Right Ascension of the object (in degrees).")
    )

    dec = sa.Column(
        sa.Double,
        nullable=False,
        doc=("Declination of the object (in degrees).")
    )

    pm_ra = sa.Column(
        sa.Double,
        nullable=False,
        doc=("Proper motion in Right Ascension (in degrees per day).")
    )

    pm_dec = sa.Column(
        sa.Double,
        nullable=False,
        doc=("Proper motion in Declination (in degrees per day).")
    )

    position_x = sa.Column(
        sa.Double,
        nullable=False,
        doc=("X coordinate of the position (in AU for equatorial coordinates).")
    )

    position_y = sa.Column(
        sa.Double,
        nullable=False,
        doc=("Y coordinate of the position (in AU for equatorial coordinates).")
    )

    position_z = sa.Column(
        sa.Double,
        nullable=False,
        doc=("Z coordinate of the position (in AU for equatorial coordinates).")
    )   

    velocity_x = sa.Column(
        sa.Double,
        nullable=False,
        doc=("X coordinate of the velocity (in AU per day for equatorial coordinates).")
    )

    velocity_y = sa.Column(
        sa.Double,
        nullable=False,
        doc=("Y coordinate of the velocity (in AU per day for equatorial coordinates).")
    )

    velocity_z = sa.Column(
        sa.Double,
        nullable=False,
        doc=("Z coordinate of the velocity (in AU per day for equatorial coordinates).")
    )


    