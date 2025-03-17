import math

import healpy
from matplotlib import pyplot
import pyarrow.parquet


tab = pyarrow.parquet.read_table( "gaia_healpix_density.pq" ).to_pandas()
ra, dec = healpy.pixelfunc.pix2ang( 32, tab.healpix32, nest=True, lonlat=True )
dec *= math.pi / 180.
ra *= math.pi / 180.
ra[ ra < -math.pi ] += 2 * math.pi
ra[ ra >= math.pi ] -= 2 * math.pi

fig = pyplot.figure( figsize=(12,6) )
ax = fig.add_subplot( 1, 1, 1, projection='aitoff' )
image = ax.tripcolor( ra, dec, tab['20'], norm='log' )
fig.colorbar( image, ax=ax )
# This has useful informatioh but makes a mess
# ax.tricontour( ra, dec, tab['20'], levels=[ 1e4, 1e5, 1e6 ], color=white )
ax.grid( which='major', color='#999999' )
fig.savefig( 'gaia_healpix_density.png' )
fig.show()
pyplot.show()
