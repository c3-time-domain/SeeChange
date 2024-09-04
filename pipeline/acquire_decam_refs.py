import argparse

import numpy as np

from astropy.io import fits

from models.instrument import get_instrument_instance
from models.provenance import Provenance
from models.decam import DECam
from models.image import Image
from models.exposure import Exposure

class DECamRefFetcher:
    def __init__( self, ra, dec, filter, max_seeing=1.2, min_depth=None, min_mjd=None, max_mjd=None, min_per_chip=9 ):
        self.ra = ra
        self.dec = dec
        self.filter = filter
        self.max_seeing = max_seeing
        self.min_depth = min_depth
        self.min_mjd = min_mjd
        self.max_mjd = max_mjd
        self.min_per_chip = min_per_chip

        self.decam = get_instrument_instance( 'DECam' )
        self.chips = self.decam.get_section_ids()
        self.chippos = { c: self.decam.get_ra_dec_for_section( self.ra, self.dec, c ) for c in self.chips }

        # We're going to demand at least min_per_chip images overlapping four points
        #  on the image, each point being "near" one of the corners.
        # Put those ra/dec points in the self.corners dictionary.
        self.corners = {}
        self.cornercount = {}
        self.chipimgs = {}
        cornerfrac = 0.8
        for chip in self.chips:
            ra = self.chippos[chip][0]
            dec = self.chippos[chip][1]
            cd = np.cos( dec * np.pi / 180. )
            dra = cornerfrac * 2048. * self.decam.pixel_scale / 3600. / cd
            ddec = cornerfrac * 1024. * self.decam.pixel_scale / 3600.
            self.corners[chip] = [ ( ra - dra, dec - ddec ),
                                   ( ra - dra, dec + ddec ),
                                   ( ra + dra, dec - ddec ),
                                   ( ra + dra, dec + ddec ) ]
            self.cornercount[chip] = [ 0, 0, 0, 0 ]
            self.chipimgs[chip] = []

        self.prov = Provenance( process='preprocessing',
                                parameters={ 'preprocessing': 'noirlab_instcal',
                                             'calibset': 'externally_supplied',
                                             'flattype': 'externally_supplied' } )


    def identify_existing_images( self, reset=True ):
        """Find DECam instcal images already in the database we can use.

        These will probably be here from a previous run of this class.
        Looks for images for each chip whose center includes the chip's
        center (based on ra, dec, and the chip's known offset), and that
        overlap the chip by at least 10%.

        Updates self.chipimgs

        Parameters
        ----------
          reset : bool, default True
            reset all counts to 0 before starting.  Not doing this could
            easily lead to redundancies.

        """
        if reset:
            self.cornercount = { c: [ 0, 0, 0, 0 ] for c in self.chips }
            self.chipimgs = { c: [] for c in self.chips }

        for chip in self.chips:
            kwargs = { 'ra': self.chippos[chip][0],
                       'dec': self.chippos[chip][1],
                       'instrument': 'DECam',
                       'filter': self.filter,
                       'provenance_ids': self.prov.id,
                       'order_by': 'quality'
                      }
            if self.min_mjd is not None:
                kwargs['min_mjd'] = self.min_mjd
            if self.max_mjd is not None:
                kwargs['max_mjd'] = self.max_mjd
            if self.max_seeing is not None:
                kwargs['max_seeing'] = self.max_seeing

            imgs = Image.find_images( **kwargs )

            if len( imgs ) > 0:
                for img in imgs:
                    # Totally punting on ra/dec crossing 0°....
                    for corner in range(r):
                        if ( ( img.minra < self.corners[chip][corner][0] ) and
                             ( img.maxra > self.corners[chip][corner][0] ) and
                             ( img.mindec < self.corners[chip][corner][1] ) and
                             ( img.maxdec > self.corners[chip][corner][1] )
                            ):
                            self.cornercount[chip][corner] += 1
                            self.chipimgs[chip].append( img )


    def identify_useful_remote_exposures( self ):
        """Get a DECamOriginExposures objects and a set of indexes of the ones we want.

        Starts with whats in self.cornercount.  Look for exposures
        within 2° of our exposure, then go through all chips and find
        things that overlap.

        """

        # Get all exposures that might be useful

        kwargs = { 'skip_exposures_in_database': True,
                   'ctr_ra': self.ra,
                   'ctr_dec': self.dec,
                   'radius': 2.2,          # DECam has 2.2° field of view, so this gets to edge overlap
                   'proc_type': 'instcal'
                  }
        if self.min_mjd is not None:
            kwargs['minmjd'] = self.min_mjd
        if self.max_mjd is not None:
            kwargs['maxmjd'] = self.max_mjd

        origexps = self.decam.find_origin_exposures( **kwargs )
        usefuldexen = set()

        # For each chip we're looking for:
        for chip in self.chips:
            for expdex in range(len(origexps)):
                filter = origexps.exposure_filter( expdex )
                expra, expdec = origexps.exposure_coords( expdex )

                # Basic cuts
                if self.decam.get_short_filter_name( filter ) != self.decam.get_short_filter_name( self.filter ):
                    continue
                if ( self.max_seeing is not None ) and ( origexps.exposure_seeing(expdex) > self.max_seeing ):
                    continue
                if ( self.min_depth is not None ) and ( origexps.exposure_depth(expdex) < self.min_depth ):
                    continue

                expra, expdec = origexps.exposure_coords( expdex )

                # Go through all chips of the exposure to figure out which chips
                #   overlap which chips of our target.  Add exposures that
                #   help to usefuldexen, and increment self.cornercount
                for expchip in self.chips:
                    ra, dec = self.decam.get_ra_dec_for_section( expra, expdec, expchip )
                    cd = np.cos( dec * np.pi / 180. )
                    dra = 2048. * self.decam.pixel_scale / 3600. / cd
                    ddec = 1024. * self.decam.pixel_scale / 3600.
                    minra = ra - dra
                    maxra = ra + dra
                    mindec = dec - ddec
                    maxdec = dec + ddec

                    for corner in range(4):
                        if ( ( minra < self.corners[chip][corner][0] ) and
                             ( maxra > self.corners[chip][corner][0] ) and
                             ( mindec < self.corners[chip][corner][1] ) and
                             ( maxdec > self.corners[chip][corner][1] )
                            ):
                            self.cornercount[chip][corner] += 1
                            usefuldexen.add( expdex )

                # Are we done?  If so, break out of exposure loop, go on to next chip of target
                if all( [ self.cornercount[chip][c] >= self.min_per_chip for c in [ 0, 1, 2, 3 ] ] ):
                    break

        return origexps, usefuldexen


    def download_and_extract( self, origexps, usefuldexen ):
        for dex in usefuldexen:
            dled = origexps.download_exposures( outdir=Image.temp_path, indexes=[dex], onlyexposures=False,
                                                existing_ok=True, clobber=False, )

            if set( dled[0].keys() ) != { 'exposure', 'wtmap', 'dqmask' }:
                raise RuntimeError( f"Don't have all of exposure, wtmap, dqmask in {dled[0].keys()}" )

            import pdb; pdb.set_trace()
            expsr = Exposure( current_file=str(dled[0]['exposure']),
                              instrument='DECam' )

            wtexp = fits.open( dled[0]['wtmap'] )
            flgsexp = fits.open( dled[0]['dqmask'] )


            # ROB YOU ARE HERE

            import pdb; pdb.set_trace()
            pass




    def __call__( self ):
        self.identify_existing_images( reset=True )
        origexps, usefuldexen = self.identify_useful_remote_exposures()
        self.download_and_extract( origexps, usefuldexen )

        import pdb; pdb.set_trace()
        pass

def main():
    parser = argparse.ArgumentParser( 'acquire_decam_refs',
                                      description="Download NOIRlab-procssed DECam exposures for use as references.",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '-r', '--ra', type=float, required=True, help="Center RA (degrees) of target exposure" )
    parser.add_argument( '-d', '--dec', type=float, required=True, help="Center Dec (degrees) of target exposure" )
    parser.add_argument( '-f', '--filter', required=True, help="Filter (g, r, i, z) to search for" )
    parser.add_argument( '-s', '--max-seeing', type=float, default=1.2, help="Maximum seeing of exposures to pull" )
    parser.add_argument( '-l', '--min-depth', type=float, default=None,
                         help=( "Minimum magnitude limit.  This will be filter-dependent, but you really "
                                "want to set this." ) )
    parser.add_argument( '-n', '--min-mjd', type=float, default=None, help="Minimum mjd; default=no limit" )
    parser.add_argument( '-x', '--max-mjd', type=float, default=None, help="Maximum mjd; default=no limit" )
    parser.add_argument( '-c', '--min-num-per-chip', type=int, default=9,
                         help=( "Make sure to get exposure so that each chip will have at least this many "
                                "images overlapping it." ) )
    args = parser.parse_args()

    fetcher = DECamRefFetcher( args.ra, args.dec, args.filter, max_seeing=args.max_seeing, min_depth=args.min_depth,
                               min_mjd=args.min_mjd, max_mjd=args.max_mjd, min_per_chip=args.min_num_per_chip, )
    fetcher()


# **********************************************************************
if __name__ == "__main__":
    main()
