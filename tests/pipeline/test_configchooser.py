from util.config import Config
from pipeline.configchooser import ConfigChooser


def test_config_chooser():
    try:
        origconfig = Config.get()
        assert origconfig.value( 'configchoice.choice_algorithm' ) == 'star_density'
        assert ( origconfig.value( 'configchoice.configs' )
                 == { 'galactic': 'seechange_config_test_galactic.yaml',
                      'extragalactic': 'seechange_config_test_extragalactic.yaml' } )

        # Totally abuse internal knowledge that ConfigChooser only looks
        # at ra and dec.  Just set those two fields, not worrying that
        # the Exposure no longer makes sense.  (Note that decam_exposure
        # is a session fixture, so we have to be sure to undo the damage
        # in our finally block below!)

        # An extragalactic field
        chooser = ConfigChooser()
        chooser.run( 15, -15. )
        exgalconfig = Config.get()

        assert exgalconfig.value( 'configchoice.choice_algorithm' ) is None
        assert exgalconfig.value( 'configchoice.configs' ) is None
        assert exgalconfig.value( 'extraction.threshold' ) == origconfig.value( 'extraction.threshold' )
        assert exgalconfig.value( 'astrocal.max_catalog_mag' ) == origconfig.value( 'astrocal.max_catalog_mag' )

        # Reset config before trying the next thing
        Config._default = None
        Config._configs = {}
        Config.init()

        # A galactic field
        chooser = ConfigChooser()
        chooser.run( 270., -30. )
        galconfig = Config.get()

        assert galconfig.value( 'configchoice.choice_algorithm' ) is None
        assert galconfig.value( 'configchoice.configs' ) is None
        assert galconfig.value( 'extraction.threshold' ) != origconfig.value( 'extraction.threshold' )
        assert galconfig.value( 'extraction.threshold' ) == 10.0
        assert galconfig.value( 'astrocal.max_catalog_mag' ) != origconfig.value( 'astrocal.max_catalog_mag' )
        assert galconfig.value( 'astrocal.max_catalog_mag' ) == [15., 16., 17.]

    finally:
        # Poke into the internals of Config to make sure we
        #   reset fully to default
        Config._default = None
        Config._configs = {}
        Config.init()
