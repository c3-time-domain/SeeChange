import sys
import pathlib
from astropy.io import fits
import re
import numpy

secparse = re.compile( r"^\[(?P<x0>\d+):(?P<x1>\d+),(?P<y0>\d+):(?P<y1>\d+)\]$" )
fnamere = re.compile( r"^(.*)\.fits$" )

direc = pathlib.Path( '/DECam_domeflat' )
for origflat in direc.glob( "*.fits" ):
    match = fnamere.search( origflat.name )
    fbase = match.group(1)
    with fits.open( origflat ) as hdu:
        hdr = hdu[0].header
        data = hdu[0].data

    match = secparse.search( hdr['DATASEC'] )
    x0 = int( match.group( 'x0' ) ) - 1
    x1 = int( match.group( 'x1' ) )
    y0 = int( match.group( 'y0' ) ) - 1
    y1 = int( match.group( 'y1' ) )
    data = data[ y0:y1, x0:x1 ]
    data /= numpy.nanmedian( data )
    fits.writeto( direc / f'{fbase}_trim_med.fits', data, hdr, overwrite=True )
    sys.stderr.write( f'Did {origflat.name}\n' )
