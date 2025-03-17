import pathlib

import pyarrow.parquet
import healpy

from util.config import Config
from util.logger import SCLogger
from models.base import CODE_ROOT
from models.exposure import Exposure
from models.image import Image
from pipeline.parameters import Parameters


class ParsConfigChooser( Parameters ):

    def __init__( self, **kwargs ):
        super().__init__()

        self.choice_algorithm = self.add_par(
            'choice_algorithm',
            'star_density',
            ( None, str ),
            ( "The algorithm used to choose the base config file.  None = don't "
              "choose, just stick with the default config that was already read. "
              "star_density = look at RA/Dec of the exposure or image in the "
              "DataStore, and choose either config 'galactic' or 'extragalactic' "
              "based on the star_density_cutoff and star_mag_cutoff parameters" ),
            critical=True
        )

        self.star_mag_cutoff = self.add_par(
            'star_mag_cutoff',
            20,
            int,
            "Magnitude cutoff to look at star density",
            critical=True
        )

        # For the healpix(32,nest=True) healpix, each healpix is about 110' (1.8°) on a side,
        #   so each healpix is roughly 3.4 square degrees.
        # TODO : look more carefully to choose this number
        self.star_density_cutoff = self.add_par(
            'star_density_cutoff',
            1e5,
            float,
            ( "Star densities in stars/healpix at or above this value will lead to the "
              "choice of the 'galactic' config; otherwise, the 'extragalactic' config." ),
            critical=True
        )

        self.config_dir = self.add_par(
            'config_dir',
            None,
            ( None, str ),
            "Parent directory to look for config files and tables.  If None, uses CODE_ROOT",
            critical=False,
        )

        self.gaia_density_catalog = self.add_par(
            'gaia_density_catalog',
            'share/gaia_density/gaia_healpix_density.pq',
            str,
            "Path to gaia density parquet file relative to config_dir",
            critical=False
        )

        self._enforce_no_new_attrs = True

        self.override( kwargs )


class ConfigChooser:
    """Modify the config based on Image or Exposure.

    ConfigChooser modifies the default config (so, it will change
    what any later process gets through a call to exactly
    "Config.get()") based on its configuration and the Image or Exposure
    in the passed DataStore.

    """

    def __init__( self, **kwargs ):
        self.pars = ParsConfigChooser( **kwargs )


    def run( self, *args ):
        """Pass one of three things:

        (1) ra, dec : float, float
              Decimal degrees

        (2) exposure : Exposure
              An exposure, whose ra and dec fields will be read

        (3) image : Image
              An image, whose ra and dec fields will be read
        if self.pars.choice_algorithm is None:
            return

        Returns nothing.

        """

        try:
            if len( args ) == 1:
                if isinstance( args[0], Exposure) or isinstance( args[0], Image ):
                    ra = args[0].ra
                    dec = args[0].dec
                else:
                    raise RuntimeError( "Pass ConfigChooser.run() (ra, dec), Exposure, or Image" )
            elif len( args ) == 2:
                ra = float( args[0] )
                dec = float( args[1] )
            else:
                raise RuntimeError( "Pass ConfigChooser.run() (ra, dec), Exposure, or Image" )

            if self.pars.choice_algorithm == 'star_density':
                self.set_config_based_on_star_density( ra, dec )
            else:
                raise ValueError( f"Unknown ConfigChooser algorithm: {self.pars.choice_algorithm}" )

        except Exception as e:
            SCLogger.exception( f"Exception in ConfigChooser.run: {e}" )
            raise


    def set_config_based_on_star_density( self, ra, dec ):
        """Replaces the default config based on gaia stars / healpix at ra, dec.

        Decides if this is a galactic or extragalactic field by reading
        tablefile and looking up the density of stars per healpix for
        mag≤maglim at the healpix (32, nest=True, lonlat=True) of ra,
        dec.  Uses the current config's value of
        configchoice.configs.galactic or
        configchoice.configs.extragalactic as a config file to read and
        set as the default config going forward.

        Parameters
        ----------
          ra : float
            RA in degress

          dec : float
            DEC in degrees

        """

        cfg = Config.get()
        if cfg.value( 'configchoice.config_dir' ) is not None:
            tablefile = pathlib.Path( cfg.value( 'configchoice.config_dir' ) )
        else:
            tablefile = pathlib.Path( CODE_ROOT )
        tablefile = tablefile / self.pars.gaia_density_catalog
        maglim = self.pars.star_mag_cutoff
        densitycut = self.pars.star_density_cutoff

        densitytab = pyarrow.parquet.read_table( tablefile ).to_pandas()
        if str(maglim) not in densitytab.columns:
            raise ValueError( f"Don't have densities for magnitude limit {maglim}" )

        hp = healpy.ang2pix( 32, ra, dec, nest=True, lonlat=True )
        row = densitytab[ densitytab['healpix32'] == hp ]
        if len(row) == 0:
            raise ValueError( f"Failed to find healpix {hp} in gaia density table" )
        if len(row) > 1:
            raise ValueError( f"Healpix {hp} shows up in gaia density table more than once; this shouldn't happen." )

        dens = row[ str(maglim) ].values[ 0 ]

        cfg = Config.get()

        if dens >= densitycut:
            configfile = cfg.value( 'configchoice.configs.galactic' )
        else:
            configfile = cfg.value( 'configchoice.configs.extragalactic' )
        configfile = cfg._path.parent / configfile
        Config.init( configfile, reread=True, setdefault=True )
