class ExposureSource:
    """A place you can pull exposures from.

    Each exposure source is linked to a fininte number of Instrument
    objects.  (Often, there will be just one.)  These are the
    instruments whose exposures can be pulled from this exposure source.

    Properties:

       name : name of the exposure source (for use in the knownexposures table)

       instrument_names : names of instruments supported by this
          ExposureSource.  Each should work as an argument to
          models.instrument.get_instrument_instance()

    """

    def __init__( self, **kwargs ):
        """Create a new ExposureSource; don't call this direclty, use get_exposuresource_instance.

        Derived classes should call this at the *end* of their own
        __init__ method.

        """

        self.name = getattr( self, 'name', None )
        self.instrument_names = getattr( self, 'instrument_names', [] ) 
         
