from models.base import (
    Base,
    AutoIDMixin,
    SpatiallyIndexed,
)

class KnownExposure(Base, AutoIDMixin):
    """A table of exposures we know about that we need to grab and process through the pipeline.

    Most fields are nullable because we can't be sure a priori how much
    information we'll be able to get about known exposures before we
    download them and start processing them -- they may only be a list
    of filenames, for instance.  exposuresurce must be known (because
    that's where they come from), and we're assuming that the instrument
    will be known.  identifier required, and is some sort of unique
    identifier that specifies this exposure; it's interpretation is
    ExposureSource-dependent.  This plus "params" need to be enough
    information to actually pull the exposure from the exposure source.

    """
    
    __tablename__ = "knownexposures"

    instrument = sa.Column( sa.Text, nullable=False, index=True, doc='Instrument this known exposure is from' )
    exposuresource = sa.Column ( sa.Text, nullable=False, index=True, doc='ExposureSource this exposure is from' )
    identifier = sa.Column( sa.Text, nullable=False, doc='Identifies this exposure on the ExposureSource' )
    params = sa.Column( sa.Text, nullable=True, doc='Additional parameters needed to pull this exposure' )
    
    exposure_id = sa.Column( 'exposure_id',
                             sa.BigInteger,
                             sa.ForeignKey( 'exposures.id', name='knownexposure_exposure_id_fkey' ),
                             nullable=True )

    mjd = sa.Column( sa.Double, nullable=True, index=True, doc="MJD of the start of the exposure (MJD=JD-2400000.5)" )
    exp_time = sa.Column( sa.REAL, nullable=True, doc="Exposure time of the exposure" )
    filter = sa.Column( sa.Text, nullable=True, doc="Filter of the exposure" )

    project = sa.Column( sa.Text, nullable=True, doc="Name of the project (or proposal ID)" )
    target = sa.Column( sa.Text, nullable=True, doc="Target of the exposure" )

    cluster_id = sa.Column( sa.Text, nullable=True, doc="ID of the cluster that has been assigned this exposure" )
    claim_time = sa.Column( sa.DateTime, nullable=True, doc="Time when this exposure was assigned to cluster_id" )
    
    # Not using SpatiallyIndexed because we need ra and dec to be nullable
    ra = sa.Column( sa.Double, nullable=True, doc='Right ascension in degrees' )
    dec = sa.Column( sa.Double, nullable=True, doc='Declination in degrees' )
    gallat = sa.Column(sa.Double, nullable=True, index=True, doc="Galactic latitude of the target. ")
    gallon = sa.Column(sa.Double, nullable=True, index=False, doc="Galactic longitude of the target. ")
    ecllat = sa.Column(sa.Double, nullable=True, index=True, doc="Ecliptic latitude of the target. ")
    ecllon = sa.Column(sa.Double, nullable=True, index=False, doc="Ecliptic longitude of the target. ")

    @declared_attr
    def __table_args__(cls):
        tn = cls.__tablename__
        return (
            sa.Index(f"{tn}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
        )
    
    def calculate_coordinates(self):
        """Fill self.gallat, self.gallon, self.ecllat, and self.ecllong based on self.ra and self.dec."""

        if self.ra is None or self.dec is None:
            return

        coords = SkyCoord(self.ra, self.dec, unit="deg", frame="icrs")
        self.gallat = float(coords.galactic.b.deg)
        self.gallon = float(coords.galactic.l.deg)
        self.ecllat = float(coords.barycentrictrueecliptic.lat.deg)
        self.ecllon = float(coords.barycentrictrueecliptic.lon.deg)
                            
