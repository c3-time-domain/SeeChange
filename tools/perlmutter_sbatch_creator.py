import sys
import pathlib
import time
import argparse
import signal
import logging

from models.base import SmartSession
from models.knownexposure import KnownExposure
from util.conductor_connector import ConductorConnector
from util.logger import SCLogger


# ======================================================================

class PerlmutterSbatchCreator:
    def __init__( self, podman_image=None, seechange_dir=None, data_dir=None, temp_dir=None, home_dir=None,
                  secrets_dir=None, direc=None, max_jobs=4, sleeptime=30, queue='realtime', account=None, user=None,
                  cpus=None, mem=None, runtime='00:30:00', numprocs=60, worker_log_level='warning' ):
        self.podman_image = podman_image
        self.seechange_dir = seechange_dir
        self.data_dir = data_dir
        self.temp_dir = temp_dir
        self.home_dir = home_dir
        self.secrets_dir = secrets_dir
        self.direc = direc
        self.max_jobs = max_jobs
        self.sleeptime = sleeptime
        self.queue = queue
        self.account = account
        self.user = user
        self.cpus = cpus
        self.mem = mem
        self.runtime = runtime
        self.numprocs = numprocs
        self.worker_log_level = worker_log_level
        self.pipelineworker_id = None
        self.conductor = ConductorConnector()

        self.scriptdir = direc / 'jobs_to_submit'

    def register_worker( self, replace=False ):
        url = f'conductor/registerworker/cluster_id=perlmutter/node_id=sbatch_creator/replace={int(replace)}'
        data = self.conductor.send( url )
        self.pipelineworker_id = data['id']

    def unregister_worker( self ):
        url = f'conductor/unregisterworker/{str(self.pipelineworker_id)}'
        SCLogger.info( f"Sending to {url}" )
        try:
            data = self.conductor.send( url )
            if data['status'] != 'worker deleted':
                SCLogger.error( "Surprising response from conductor unregistering worker: {data}" )
        except Exception:
            SCLogger.exception( f"Exception unregistering worker {self.pipelineworker_id}, continuing" )

    def send_heartbeat( self ):
        url = f'conductor/workerheartbeat/{str(self.pipelineworker_id)}'
        self.conductor.send( url )

    def __call__( self ):
        while True:
            self.send_heartbeat()

            ncurrentscripts = len( list( self.scriptdir.glob("*.sh") ) )
            if ncurrentscripts >= self.max_jobs:
                SCLogger.info( f"Already {ncurrentscripts} scripts waiting to be submitted, "
                               f"sleeping {self.sleeptime} s" )
                time.sleep( self.sleeptime )
                continue

            data = self.conductor.send( 'conductor/requestexposure/cluster_id=perlmutter' )

            if data['status'] == 'not available':
                SCLogger.info( f"No exposures avialable, sleeping {self.sleeptime} s" )
                time.sleep( self.sleeptime )
                continue

            if data['status'] != 'available':
                raise ValueError( f"Unexpected value of data['status']: {data['status']}" )

            with SmartSession() as session:
                knownexp = ( session.query( KnownExposure )
                             .filter( KnownExposure._id==data['knownexposure_id'] ) ).all()
                if len( knownexp ) == 0:
                    raise RuntimeError( f"The conductor gave me KnownExposure id {data['knownexposure_id']}, "
                                        f"but I can't find it in the knownexposures table" )
                if len( knownexp ) > 1:
                    raise RuntimeError( f"More than one KnownExposure with id {data['knownexposure_id']}; "
                                        f"you should never see this error." )
            knownexp = knownexp[0]

            jobscript = self.scriptdir / f"seechange_{knownexp.identifier}.sh"
            SCLogger.info( f"Writing {jobscript}" )
            with open( jobscript, "w" ) as ofp:
                ofp.write( "#!/bin/bash\n" )
                ofp.write( f"#SBATCH --account {self.account}\n" )
                ofp.write( "#SBATCH --constraint cpu\n" )
                ofp.write( "#SBATCH --ntasks 1\n" )
                if self.cpus is not None:
                    ofp.write( f"#SBATCH --cpus-per-task {self.cpus}\n" )
                if self.mem is not None:
                    ofp.write( f"#SBATCH --mem {self.mem}G\n" )
                ofp.write( f"#SBATCH --qos {self.queue}\n" )
                ofp.write( f"#SBATCH --output ../logs/{knownexp.identifier}.log\n" )
                ofp.write( f"#SBATCH --time {self.runtime}\n" )
                ofp.write( "\n" )

                ofp.write( "echo 'slurm job running on ' $SLURM_JOB_NODELIST\n" )
                ofp.write( "echo 'job starting at ' `date`\n\n" )
                ofp.write( "node=`hostname`\n\n" )

                ofp.write( "podman-hpc run \\\n" )
                ofp.write( "  --env OMP_NUM_THREADS=1 \\\n" )
                ofp.write( "  --env OPENBLAS_NUM_THREADS=1 \\\n" )
                ofp.write( "  --env HOME=/seechange-home \\\n" )
                ofp.write( f"  --mount type=bind,source={str(self.home_dir)},target=/seechange-home \\\n" )
                ofp.write( f"  --mount type=bind,source={str(self.seechange_dir)},target=/seechange \\\n" )
                ofp.write( f"  --mount type=bind,source={str(self.secrets_dir)},target=/secrets \\\n" )
                ofp.write( f"  --mount type=bind,source={str(self.data_dir)},target=/data \\\n" )
                ofp.write( f"  --mount type=bind,source={str(self.temp_dir)},target=/temp \\\n" )
                ofp.write( f"  {self.podman_image} \\\n" )
                ofp.write( "  python /seechange/pipeline/exposure_processor.py \\\n" )
                ofp.write( f"     {knownexp.instrument} {knownexp.identifier} \\\n" )
                ofp.write( "     -c perlmutter \\\n" )
                ofp.write( f"     -n {self.numprocs} \\\n" )
                ofp.write( f"     -l {self.worker_log_level} \\\n" )
                ofp.write( "     --assume-claimed\n" )

                ofp.write( "\necho 'job exiting at ' `date`\n" )



# ======================================================================

class ArgFormatter( argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter ):
    def __init__( self, *args, **kwargs ):
        super().__init__( *args, **kwargs )


def main():
    parser = argparse.ArgumentParser( 'perlmutter_sbatch_creator.py', formatter_class=ArgFormatter,
                                      epilog='''Create slurm scripts for running jobs on perlmutter.

Contacts the conductor to find out if there are exposures to process.
Claims exposures, and creates slurm scripts that will process those
exposures.

This needs to run inside a SeeChange podman container.  It doesn't actually
submit the slurm scrips because it doesn't have access to slurm from
inside the container.  Run perlmutter_sbatch_launcher.py to actually
launch the scripts in question.

Note that the --home-dir, --seechange-dir, --data-dir, and --temp-dir parameters
must all be paths on the host, i.e. outside the podman container.
However, --script-and-log-dir a path *inside* the container.

Created scripts will be put in {script_and_log_dir}/jobstorun.

'''
                                      )
    parser.add_argument( '--seechange-dir', default=None, required=True,
                         help="Directory on the host where SeeChange is checked out (defaults to ../)" )
    parser.add_argument( '--data-dir', default=None, required=True,
                         help="SeeChange local file store (probably somewhere under $SCRATCH" )
    parser.add_argument( '--temp-dir', default=None, required=True,
                         help="SeeChange temp file store (probably somewhere under $SCRATCH" )
    parser.add_argument( '--home-dir', default=None, required=True,
                         help=( "Directory to mount to user home directory inside the container."
                                "(This is not in general the user's home directory on the system; "
                                "it's probably a work directory for the project.  The user must be "
                                "able to write to it.)" ) )
    parser.add_argument( '--secrets-dir', default=None, required=True,
                         help="Directory to mount to /secrets inside the container." )
    parser.add_argument( '-m', '--max-jobs', type=int, default=4,
                         help="Maximum number of jobs that should submitted/run at once." )
    parser.add_argument( '-s', '--sleeptime', default=30,
                         help='How many seconds to sleep after getting nothing to do from conductor' )
    parser.add_argument( '-l', '--log-level', default='info', help='Log level for the main process' )
    parser.add_argument( '-u', '--sbatch-user', required=True,
                         help='User under which sbatch jobs will be submitted' )
    parser.add_argument( '-a', '--sbatch-account', required=True,
                         help='NERSC account to submit jobs under' )
    parser.add_argument( '-q', '--queue', required=True,
                         help='Queue to which to submit jobs' )
    parser.add_argument( '-d', '--script-and-log-dir', default='scripts_logs',
                         help="Directory where scripts and logs go *inside* the container." )
    parser.add_argument( '-i', '--podman-image', default='registry.nersc.gov/m2218/raknop/seechange:decat-ddf',
                         help="Podman image to use" )
    parser.add_argument( '-c', '--cpus', default=None, type=int,
                         help='Number of CPUs to ask slurm for; default is not to ask anyting in particular' )
    parser.add_argument( '--mem', default=None, type=int,
                         help='Number of GB of memory to ask slurm for; default is not to ask.' )
    parser.add_argument( '-p', '--chip-processes', type=int, default=60,
                         help=( "Number of chip processes to run; if you specify --cpus, it should be at least "
                                "this plus 1." ) )
    parser.add_argument( '-t', '--runtime', default='00:30:00', help='How long to tell slurm the job will take' )
    parser.add_argument( '-w', '--worker-log-level', default='warning', help='Log level for the worker process' )
    args = parser.parse_args()

    loglookup = { 'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG }
    if ( args.log_level not in loglookup ) or ( args.worker_log_level not in loglookup ):
        raise ValueError( "Valid log levels are: error warning info debug" )
    SCLogger.setLevel( loglookup[ args.log_level.lower() ] )

    sbatcher = PerlmutterSbatchCreator( podman_image=args.podman_image,
                                        seechange_dir=pathlib.Path(args.seechange_dir),
                                        data_dir=pathlib.Path(args.data_dir),
                                        temp_dir=pathlib.Path(args.temp_dir),
                                        home_dir=pathlib.Path(args.home_dir),
                                        secrets_dir=pathlib.Path(args.secrets_dir),
                                        direc=pathlib.Path(args.script_and_log_dir).resolve(),
                                        queue=args.queue,
                                        account=args.sbatch_account,
                                        user=args.sbatch_user,
                                        cpus=args.cpus,
                                        mem=args.mem,
                                        runtime=args.runtime,
                                        numprocs=args.chip_processes,
                                        worker_log_level=args.worker_log_level )
    sbatcher.register_worker()

    def goodbye( signum, frame ):
        SCLogger.warning( "Got INT/TERM signal, unregistering worker and exiting." )
        sys.exit()

    signal.signal( signal.SIGINT, goodbye )
    signal.signal( signal.SIGTERM, goodbye )

    try:
        sbatcher()
    finally:
        sbatcher.unregister_worker()


# ======================================================================
if __name__ == "__main__":
    main()
