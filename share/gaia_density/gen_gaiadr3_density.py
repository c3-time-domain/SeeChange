import sys
import logging
import pathlib
import pyarrow
import pyarrow.parquet
from astropy.table import Table

_logger = logging.getLogger("main")
_logout = logging.StreamHandler( sys.stderr )
_logger.addHandler( _logout )
_formatter = logging.Formatter( '[%(asctime)s - %(levelname)s] - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S' )
_logout.setFormatter( _formatter )
_logger.setLevel( logging.INFO )


def main():
    basedir = pathlib.Path( "/global/cfs/cdirs/cosmo/data/gaia/dr3/healpix" )
    healpix = []
    gstars = { 16: [],
               17: [],
               18: [],
               19: [],
               20: [],
               21: [],
               22: [] }
    for i in range( 0, 12288 ):
        _logger.info( f"Doing healpix {i} of 12288" )
        tab = Table.read( basedir / f"healpix-{i:05d}.fits" )
        healpix.append( i )
        for mag in gstars.keys():
            gstars[mag].append( ( tab['PHOT_G_MEAN_MAG'] <= mag ).sum() )

    datas = [ healpix ]
    datas.extend( [ gstars[m] for m in range(16, 23) ] )
    names = [ 'healpix32' ]
    names.extend( [ str(m) for m in range(16, 23) ] )
    pqtab = pyarrow.table( datas, names=names )
    pyarrow.parquet.write_table( pqtab, 'gaia_density.pq' )


# ======================================================================
if __name__ == "__main__":
    main()
