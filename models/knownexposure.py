import sqlalchemy as sa
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as sqlUUID

from models.base import Base, UUIDMixin
from util.radec import radec_to_gal_ecl


class KnownExposure(Base, UUIDMixin):
    """A table of exposures we know about that we need to grab and process through the pipeline.

    Most fields are nullable because we can't be sure a priori how much
    information we'll be able to get about known exposures before we
    download them and start processing them -- they may only be a list
    of filenames, for instance.  exposuresource must be known (because
    that's where they come from), and we're assuming that the instrument
    will be known.  identifier required, and is some sort of unique
    identifier that specifies this exposure; it's interpretation is
    instrument-dependent.  This plus "params" need to be enough
    information to actually pull the exposure from the exposure source.

    """

    __tablename__ = "knownexposures"

    instrument = sa.Column( sa.Text, nullable=False, index=True, doc='Instrument this known exposure is from' )
    identifier = sa.Column( sa.Text, nullable=False, index=True,
                            doc=( 'Identifies this exposure on the ExposureSource; '
                                  'should match exposures.origin_identifier' ) )
    params = sa.Column( JSONB, nullable=True,
                        doc='Additional instrument-specific parameters needed to pull this exposure' )

    hold = sa.Column( 'hold', sa.Boolean, nullable=False, server_default='false',
                      doc="If True, conductor won't release this exposure for processing" )

    exposure_id = sa.Column( 'exposure_id',
                             sqlUUID,
                             sa.ForeignKey( 'exposures._id', name='knownexposure_exposure_id_fkey' ),
                             index=True,
                             nullable=True )

    mjd = sa.Column( sa.Double, nullable=True, index=True,
                     doc="MJD of the start (?) of the exposure (MJD=JD-2400000.5)" )
    exp_time = sa.Column( sa.REAL, nullable=True, doc="Exposure time of the exposure" )
    filter = sa.Column( sa.Text, nullable=True, doc="Filter of the exposure" )

    project = sa.Column( sa.Text, nullable=True, doc="Name of the project (or proposal ID)" )
    target = sa.Column( sa.Text, nullable=True, doc="Target of the exposure" )

    # node_id vs. machine name
    # node_id should match what shows up in the pipelineworkers table.  But, it's not actually the node of
    #   the cluster where it's running (possibly).
    # machine_name is the name of the actual machine where the pipeline most recently ran.
    # Example 1: Perlmutter, there's a single process running on a login node that is polling
    #   the conductor and creating sbatch scripts that will sometime later be submitted.  In this case,
    #   the cluster_id would be "perlmutter" and the "node_id" would be the name of the login node
    #   where the process querying the conductor is running... both here and in pipeline_workers
    #   Then, sometime later, the actualy starts on nid200001; that would be machine_name.
    # Example 2: Running on an interactive node (say nid20001) on Perlmutter, running *two*
    #   pipeline_exposure_launcher.py that both query the conductor.  Both have cluster
    #   "perlmutter".  Because node_id has to be unique in pipeline_workers, the first process has
    #   (say) node_id=nid20001a and the second has node_id=nid20001b.  The machine_name in this case
    #   can match the node_id, or could just be nid20001.
    cluster_id = sa.Column( sa.Text, nullable=True, doc="ID of the cluster that has been assigned this exposure" )
    node_id = sa.Column( sa.Text, nullable=True, doc="ID of the node that has been assigned this exposure" )
    machine_name = sa.Column( sa.Text, nullable=True, doc="Name of the machine where the pipeline is running." )
    claim_time = sa.Column( sa.DateTime, nullable=True, doc="Time when this exposure was assigned to cluster_id" )
    start_time = sa.Column( sa.DateTime, nullable=True, doc="Time when the pipeline actually started" )
    release_time = sa.Column( sa.DateTime, nullable=True, doc="Time when the cluster released this exposure" )
    provenance_tag = sa.Column( sa.Text, nullable=True, doc="Provenance tag of process that claimed the exposure" )
    do_not_subtract = sa.Column( sa.Boolean, nullable=False, server_default='false',
                                 doc="If True, don't ever try to do subtraction or later steps on this exposure." )

    # Not using SpatiallyIndexed because we need ra and dec to be nullable
    ra = sa.Column( sa.Double, nullable=True, doc='Right ascension in degrees' )
    dec = sa.Column( sa.Double, nullable=True, doc='Declination in degrees' )
    gallat = sa.Column(sa.Double, nullable=True, index=True, doc="Galactic latitude of the target. ")
    gallon = sa.Column(sa.Double, nullable=True, index=False, doc="Galactic longitude of the target. ")
    ecllat = sa.Column(sa.Double, nullable=True, index=True, doc="Ecliptic latitude of the target. ")
    ecllon = sa.Column(sa.Double, nullable=True, index=False, doc="Ecliptic longitude of the target. ")

    @declared_attr
    def __table_args__(cls):  # noqa: N805
        tn = cls.__tablename__
        return (
            sa.Index(f"{tn}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
            UniqueConstraint( 'instrument', 'identifier', name='_ke_unique_inst_ident' )
        )

    def calculate_coordinates(self):
        """Fill self.gallat, self.gallon, self.ecllat, and self.ecllong based on self.ra and self.dec."""

        if self.ra is None or self.dec is None:
            return

        self.gallat, self.gallon, self.ecllat, self.ecllon = radec_to_gal_ecl( self.ra, self.dec )


class PipelineWorker(Base, UUIDMixin):
    """A table of currently active pipeline launchers that the conductor knows about."""

    __tablename__ = "pipelineworkers"

    cluster_id = sa.Column( sa.Text, nullable=False, doc="Cluster where the worker is running" )
    node_id = sa.Column( sa.Text, nullable=True, doc="Node where the worker is running" )
    current_knownexp = sa.Column( 'current_knownexp', sqlUUID,
                                  sa.ForeignKey( 'knownexposures._id', name='pipelineworker_knownexp_id' ),
                                  index=True,
                                  nullable=True )
    lastheartbeat = sa.Column( sa.DateTime, nullable=False, doc="Last time this pipeline worker checked in" )


# A singlet with conductor config.
conductor_config = sa.Table(
    'conductor_config',
    Base.metadata,
    sa.Column( 'instrument_name', sa.Text, doc="Name of instrument to poll for" ),
    sa.Column( 'updateargs', JSONB, doc="Arguments to pass to instrument's find_origin_exposures" ),
    sa.Column( 'update_timeout', sa.Integer, doc="Seconds between polling" ),
    sa.Column( 'pause_updates', sa.Boolean, doc="True if we shouldn't automatically update" ),
    sa.Column( 'hold_new_exposures', sa.Boolean, doc="True if we should set the hold flag on new known exposures" ),
    sa.Column( 'configchangetime', sa.DateTime(timezone=True), doc="Last time config was changed" ),
    sa.Column( 'throughstep', sa.Text, doc="Last step to tell pipeline workers to do" ),
    sa.Column( 'pickuppartial', sa.Boolean, doc="(Complicated)" )
)
