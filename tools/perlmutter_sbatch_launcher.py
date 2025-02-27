import sys
import re
import time
import pathlib
import shutil
import logging
import subprocess
import argparse

_logger = logging.getLogger( __file__ )
_logout = logging.StreamHandler( sys.stderr )
_logger.addHandler( _logout )
_formatter = logging.Formatter( '[%(asctime)s - %(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
_logout.setFormatter( _formatter )
_logger.setLevel( logging.INFO )


# ======================================================================

class PerlmutterSbatchLauncher:
    def __init__( self, account='raknop', direc=None, sleeptime=60, maxjobs=4 ):
        self.account = account
        self.incoming_dir = pathlib.Path( direc ) / "jobs_to_submit"
        self.submitted_dir = pathlib.Path( direc ) / "submitted_jobs"
        self.log_dir = pathlib.Path( direc ) / "logs"
        self.submitted_dir.mkdir( parents=True, exist_ok=True )
        self.log_dir.mkdir( parents=True, exist_ok=True )
        self.sleeptime = sleeptime
        self.maxjobs = maxjobs

    def count_jobs( self ):
        # Count number of submitted jobs
        res = subprocess.run( ['sacct', '-X', '-s',  'pd,r',
                               '--format', 'Jobid%24,Account%24,QOS%12,JobName%24,State%12' ],
                              capture_output=True )
        if res.returncode != 0:
            raise RuntimeError( "sacct command returned {res.returncode}" )
        lines = res.stdout.decode('utf-8').split('\n')
        # two header lines
        lines = lines[2:]
        nrunning = 0
        jobnamesearch = re.compile( "seechange" )
        emptysearch = re.compile( r"^\s*$" )
        for line in lines:
            if emptysearch.search( line ):
                continue
            _jobid, account, _qos, jobname, _state = line.split()
            if ( account == self.account ) and jobnamesearch.search( jobname ):
                nrunning += 1

        return nrunning


    def __call__( self ):
        while True:
            tosubmit = list( self.incoming_dir.glob("*.sh"))
            if len( tosubmit ) == 0:
                _logger.info( f"No scripts to submit, sleeping {self.sleeptime} s " )
                time.sleep( self.sleeptime )
                continue

            nrunning = self.count_jobs()
            if nrunning >= self.maxjobs:
                _logger.info( f"Already {nrunning} jobs submitted or running, sleeping {self.sleeptime} s" )
                time.sleep( self.sleeptime )
                continue

            tosubmit.sort( key = lambda f: f.stat().st_mtime )
            src = tosubmit[0]
            dst = self.submitted_dir / src.name
            shutil.move( src, dst )

            _logger.info( f"Submitting {dst.name}" )
            subprocess.run( [ "sbatch", str(dst.name) ], cwd=self.submitted_dir )
            time.sleep( self.sleeptime )


# ======================================================================

def main():
    parser = argparse.ArgumentParser( 'perlmutter_sbatch_launcher.py', 'submit slurm scripts' )
    parser.add_argument( "-a", "--account", required=True,
                         help="SLRUM account jobs are submitted under." )
    parser.add_argument( "-d", "--script-and-log-dir", default=".",
                         help="Directory where scripts and logs go" )
    parser.add_argument( "-s", "--sleeptime", type=int, default=60,
                         help="Number of seconds to sleep after submitting a job, or failing to find a job to submit" )
    parser.add_argument( "-m", "--max-jobs", type=int, default=4,
                         help="Maximum number of jobs submitted to the queue at once." )
    args = parser.parse_args()

    launcher = PerlmutterSbatchLauncher( account=args.account,
                                         direc=pathlib.Path(args.script_and_log_dir).resolve(),
                                         sleeptime=args.sleeptime,
                                         maxjobs=args.max_jobs )
    launcher()


# ======================================================================-
if __name__ == "__main__":
    main()
