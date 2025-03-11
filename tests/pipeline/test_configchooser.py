from util.config import Config
from pipeline.configchooser import ConfigChooser


def test_config_chooser( decam_exposure ):
    origra = decam_exposure.ra
    origdec = decam_exposure.dec
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
        decam_exposure.ra = 15
        decam_exposure.dec = -15.
        chooser = ConfigChooser()
        chooser.run( decam_exposure, 'N1' )
        exgalconfig = Config.get()

        assert exgalconfig.value( 'configchoice.choice_algorithm' ) is None
        assert exgalconfig.value( 'configchoice.configs' ) is None
        assert exgalconfig.value( 'extraction.threshold' ) == origconfig.value( 'extraction.threshold' )
        assert exgalconfig.value( 'wcs.max_catalog_mag' ) == origconfig.value( 'wcs.max_catalog_mag' )

        # Reset config before trying the next thing
        Config._default = None
        Config._configs = {}
        Config.init()

        # A galactic field
        decam_exposure.ra = 270.
        decam_exposure.dec = -30.
        chooser = ConfigChooser()
        chooser.run( decam_exposure, 'N1' )
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
        # Fix the session fixture we may have screwed up
        decam_exposure.ra = origra
        decam_exposure.dec = origdec
