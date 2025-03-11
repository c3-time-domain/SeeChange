from util.config import Config
from models.exposure import Exposure
from pipeline.configchooser import ConfigChooser


def test_config_chooser():
    try:
        origconfig = Config.get()
        assert origconfig.value( 'configchoice.choice_algorithm' ) == 'star_density'
        assert ( origconfig.value( 'configchoice.configs' )
                 == { 'galactic': 'seechange_config_test_galactic.yaml',
                      'extragalactic': 'seechange_config_test_extragalactic.yaml' } )

        # An extragalactic field
        exgalexp = Exposure( ra=x, dec=y )
        chooser = ConfigChooser()
        chooser.run( exgalexp )
        exgalconfig = Config.get()

        assert exgalconfig.value( 'configchoice.choice_algorithm' ) is None
        assert exgalconfig.value( 'configchoice.configs' ) is None
        assert exgalconfig.value( 'extraction.threshold' ) == origconfig.value( 'extraction.threshold' )
        assert exgalconfig.value( 'wcs.max_catalog_mag' ) == origconfig.value( 'wcs.max_catalog_mag' )

        # A galactic field
        galexp = Exposure( ra=x, dec=y )
        chooser = ConfigChooser()
        chooser.run( galexp )
        galconfig = Config.get()

        assert galconfig.value( 'configchoice.choice_algorithm' ) is None
        assert galconfig.value( 'configchoice.configs' ) is None
        assert galconfig.value( 'extraction.threshold' ) != origconfig.value( 'extraction.threshold' )
        assert galconfig.value( 'extraction.threshold' ) == 10.0
        assert galconfig.value( 'wcs.max_catalog_mag' ) != origconfig.value( 'wcs.max_catalog_mag' )
        assert galconfig.value( 'wcs.max_catalog_mag' ) == [15., 16., 17.]

    finally:
        # Poke into the internals of Config to make sure we
        #   reset fully to default
        Config._default = None
        Config._configs = {}
        Config.init()
