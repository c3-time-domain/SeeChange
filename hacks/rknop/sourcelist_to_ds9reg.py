import argparse
from astropy.io import fits

parser = argparse.ArgumentParser( "Go from a SExtractor FITS catalog to a ds9 .reg" )
parser.add_argument( "filename", help=".cat file" )
parser.add_argument( "-w", "--world", action="store_true", default=False,
                     help="Use X_WORLD and Y_WORLD (default: X_IMAGE and Y_IMAGE)" )
parser.add_argument( "-c", "--color", default='green', help="Color (default: green)" )
parser.add_argument( "-r", "--radius", default=None,
                     help='Radius of circle (default: 5 pix or 1"); in proper units (pixels or °)!' )
parser.add_argument( "-m", "--maglimit", default=None, nargs=2, type=float, help="Only keep things in this mag range" )
args = parser.parse_args()

if args.radius is None:
    if args.world:
        radius = 1./3600.
    else:
        radius = 5.
else:
    radius = args.radius

if args.world:
    frame = 'icrs'
    xfield = 'X_WORLD'
    yfield = 'Y_WORLD'
else:
    frame = 'image'
    xfield = 'X_IMAGE'
    yfield = 'Y_IMAGE'

if args.maglimit is not None:
    minmag = args.maglimit[0]
    maxmag = args.maglimit[1]
else:
    minmag = None

with fits.open( args.filename, memmap=False ) as hdul:
    for row in hdul[2].data:
        doprint = ( ( minmag is None ) or
                    ( ( minmag is not None ) and
                      ( row['MAG'] >= minmag ) and ( row['MAG'] <= maxmag ) ) )
        if doprint:
            print( f'{frame};circle({row[xfield]},{row[yfield]},{radius}) # color={args.color} width=2' )
