import collections.abc

from astropy.coords import SkyCoord

from util.util import get_inheritors

class ExposureSource:
    """A place you can pull exposures from.

    Each exposure source is linked to a fininte number of Instrument
    objects.  (Often, there will be just one.)  These are the
    instruments whose exposures can be pulled from this exposure source.

    Get the one you need by calling get_exposuresource_instance( classname )

    Subclasses must implement:
      find_exposures
      _get_found_exposure_info


    Properties:

       name : name of the exposure source (for use in the knownexposures table)

       instrument_names : names of instruments supported by this
          ExposureSource.  Each should work as an argument to
          models.instrument.get_instrument_instance()

    """

    def __init__( self, **kwargs ):
        """Create a new ExposureSource; don't call this directly, use get_exposuresource_instance.

        Derived classes should call this at the *end* of their own
        __init__ method.

        """

        self.name = getattr( self, 'name', None )
        if self.name is None:
            raise ValueError( f"Class {self.__name__} doesn't define name" )
        self.description = getattr( self, 'description', None )
        self.instrument_names = getattr( self, 'instrument_names', [] )


    def find_exposures( self, *args, **kwargs ):
        """Search the exposure source for exposures.  Returns an opaque object.

        Each ExposureSource subclass may define its own arguments, but here are
        some standard kwargs that subclasses should strongly consider implementing:

        instrument : the instrument to find exposures for
        project : the name of the the project or proposal id
        target : the name of the target
        obsdate_min : the calendar date of the first *night* of observations ( 'yyyy-mm-dd' or datetime.date )
                       (usually an observatory-local date)
        obsdate_max : the calendar date of the last *night* of observations ( 'yyyy-mm-dd' or datetime.date )
        mjd_min : the minimum mjd
        mjd_max : the maximum mjd

        The return value should not be directly interpreted, but passed
        on to methods get_found_exposure_info, download_found_exposure.
        (Even if you look at the return value and it seems to make
        sense, treat it as opaque.  The one thing you can run on it is
        len() to find the number of exposures found.)

        (Subclasses must make sure the returned object properly implements __len__.)

        """
        raise NotImplementedError( f"Class {self.__name__} needs to implement find_exposure" )


    def get_found_exposure_info( self, exps, index=None ):
        """Return a dictionary of information about the found exposure.

        Not all exposure sources will fill all fields; unused fields will be None.

        Parameters
        ----------
          exps : Object
             The return value from find_exposure

          index : int or list of int
             The index into exps

        Returns
        -------
          info: dict or list of dict
             If index is None or a single value, will be a dictionary.  Otherwise,
             will be a list of dictionaries.  Each dictionary has keys:

                identifier : a filename, or some other unique identifier that specifies this exposure
                mjd : the mjd when the exposure was taken
                project : the project or proposal name
                target : the name of the target
                filter : the filter, or some specification of the filter array
                exp_time : exposure time in seconds
                ra : right ascension of the fiducial point in decimal degrees
                dec : declination of the fiducial point in decimal degrees
                gallat :
                gallon :
                ecllat :
                ecllon :

        """

        keys = [ 'identifier', 'mjd', 'project', 'target', 'filter', 'exp_time',
                 'ra', 'dec', 'gallon', 'gallat', 'ecllon', 'ecllat', 'download_params' ]
        if isinstance( index, int ):
            info = [ { k: None for k in keys } ]
            info[0]['index'] = index
        elif isinstance( index, collections.abc.Sequence ):
            if not all( [ isinstance( i, int ) for i in index ] ):
                raise ValueError( "All values of index must be integers" )
            info = [ { k : None for k in keys } for i in range(len(index)) ]
            for i in range(len(index)):
                info[i]['index'] = index[i]
        else:
            raise ValueError( f"Index must be either an integer or a sequence of integers, not {index}" )

        info = self._get_found_exposure_info( info, exps, index )

        def addcoords( infodict ):
            sc = SkyCoord( infodict['ra'], infodict['dec'], unit='deg', frame='icrs' )
            infodict['gallat'] = float( coords.galactic.b.deg )
            infodict['gallon'] = float( coords.galactic.l.deg )
            infodict['ecllat'] = float( coords.barycentrictrueecliptic.lat.deg )
            infodict['ecllon'] = float( coords.barycentrictrueecliptic.lon.deg )

        if instance( info, dict ):
            if ( info['ra'] is not None ) and ( info['dec'] is not None ):
                addcoords( info )
            else:
                for i in range( len(info) ):
                    addcords( info[i] )


    def download_found_exposure( self, exps, index, destfile ):
        """Download a single exposure.

        Parameters
        ----------
          exps : Object
            The return value from find_Exposure

          index: int
            The index into exps for the exposure to be downloaded

          destfile: string or Path
            The path to where the file should be written

        Note that for some exposure sources, it's possible that multipe
        files will be written in which case destfile should be viewed as
        a base.

        Returns
        -------
          written_files: list of Path
             Absolute paths to any files written

        """
        raise NotImplementedError( f"Class {self.__name__} needs to implement download_found_exposure" )





# ======================================================================
# ======================================================================
# ======================================================================

# A dictionary of names to exposure sources, pointing to the relevant class
EXPOSURESOURCE_NAME_TO_CLASS = None

# A dictionary of exposure source object singletons
EXPOSURESOURCE_INSTANCE_CACHE = None

def register_all_exposuresources():

        """Find all subclasses of ExposureSource and add them to global dictionaries.

    """
    global EXPOSURESOURCE_NAME_TO_CLASS

    if EXPOSURESOURCE_NAME_TO_CLASS is None:
        EXPOSURESOURCE_NAME_TO_CLASS = {}

    expsrces = get_inheritors( ExposureSource )
    for expsrc in expsrces:
        EXPOSURESOURCE_NAME_TO_CLASS[ expsrc.__name__ ] = expsrc


def get_exposuresource_instance( expsrc_name ):
    """Get the singleton instance of an ExposureSource given its name."""

    global EXPOSURESOURCE_NAME_TO_CLASS, EXPOSURESOURCE_INSTANCE_CACHE

    if EXPOSURESOURCE_NAME_TO_CLASS is None:
        register_all_exposuresources()

    if EXPOSURESOURCE_INSTANCE_CACHE is None:
        EXPOSURESOURCE_INSTANCE_CACHE = {}

    if expsrc_name not in EXPOSURESOURCE_INSTANCE_CACHE:
        if expsrc_name not in EXPOSURESOURCE_NAME_TO_CLASS.keys():
            raise ValueError( f"Unknown exposure source {expsrc_name}; make sure things are spelled right, "
                              f"and that the module defining the exposure source is imported early enough." )
        EXPOSURESOURCE_INSTANCE_CACHE[ expsrc_name ] = EXPSOURESOURCE_NAME_TO_CLASS[ expsrc_name ]()

    return EXPOSURESOURCE_INSTANCE_CACHE[ expsrc_name ]

