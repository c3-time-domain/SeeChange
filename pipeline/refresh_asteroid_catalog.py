"""Refresh the MPC orbit cache independently of image processing."""

import argparse
import os
from pathlib import Path

from astropy.time import Time
from util.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=os.getenv('SEECHANGE_CONFIG', str(Path(__file__).parents[1] /
                                                                            'default_config.yaml')))
    parser.add_argument('--epoch', help='UTC ISO timestamp; defaults to now. Use image time for archival processing.')
    args = parser.parse_args()
    Config.init(args.config, setdefault=True)
    from pipeline.asteroid_checker import download_and_update_db
    epoch = Time(args.epoch, scale='utc') if args.epoch else Time.now()
    download_and_update_db(epoch.tdb.jd)


if __name__ == '__main__':
    main()
