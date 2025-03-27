import os
import pathlib
import types
import copy
import yaml

from util.logger import SCLogger


class NoValue:
    pass


class Config:
    """Interface for yaml config file.

    Read a yaml file that might include other yaml files, and provide an
    interface. The top level of the yaml must be a dict. Only supports
    dicts, lists, and scalars.


    USAGE

    1. Optional: call Config.init(filename)

    2. Instantiate a config object with

           confobj = Config.get()
       or

           confobj = Config.get(filename)

       in the former case, it will get the default file.  (That's the
       file that was passed to Config.init, or specified to the first
       .get call.)  You can't do the first one if you haven't done the
       second one yet, or if you haven't called Config.init yet.  Do NOT
       call __init__ (i.e. don't say confobj = Config() or
       Config(filename)).

    3. Get a config value with

           configval = confobj.value( fieldspec )

       where fieldspec is just the field for a scalar, or .-separated
       fields for lists and dicts.  For lists, it must be a (0-offset)
       integer index.  For example, if the yaml files includes:

         storage:
           images:
             format: fits
             single_file: false
             name_convention: "{inst_name}_{date}_{time}_{section_id}_{filter}_{im_type}_{prov_hash:.6s}"

       then confobj.value("storage.images.format") will return
       "fits". You can also ask configobj.value for higher levels.  For
       example, config.value("storage.images") will return a dictionary:
          { "format": "fits",
            "single_file": False,
            "name_convention": "{inst_name}_{date}_{time}_{section_id}_{filter}_{im_type}_{prov_hash:.6s}"
          }

    4. Change a config value with

           confobj.set_value( fieldspec, value )

       This only changes it for the running session, it does *not*
       affect the YAML files in storage.


    CONFIG FILES

    This class reads yaml files (which can have other included yaml
    file).  The top level structure of a config file must be a
    dictionary.

    When reading a config file, it is processed as follows:

    The "current working config" starts as an empty dictionary ({}).
    When everything is done, it can be a big messy hierarchy.  Each key
    of the top level dictionary can have a value that is a scalar, a
    list, or a dictionary.  The structure is recursive; each element of
    each list can be a scalar, a list, or a dictionary, and the value
    associated with each key in each dictionary can itself be a scalar,
    a list, or a dictionary.

    A config file has three special keys: preloads, augments, and
    overrides.  The value associated with each key is a list of file
    paths relative to the directory where this file is found.

    preloads is a list of files that are read, in order, *before* the
       current config file, populating the current working config.
       Files later in the list override files earlier in the list.

    The current file is parsed next.  It overrides anything in the
       current working config (which will have been built from preload,
       if any).

    augments is a list of files that are read next.  Those files are
       read in order, and augment the current working config.

    overrides is a list off files that are read last.  Those files are
       read in order, and override the current working config.

    Any file that's read can itself have preloads, augments, and
    overrides.  IMPORTANT: there is no circular inclusion detection, so
    if you say file A augments file B and file B augments file A, you
    will get stuck in an infinite recursion when you try to read the
    config.  This will either lead to a python crash because it detects
    too many recursion levels, or to a process crash if you run out of
    memory, or to the end of the local Universe as you create a
    singularity and nucleate a new Big Bang.  Just don't do that.  Keep
    it simple.

    Above, the words "augment" and "override" were used to describe how
    to combine information from two different files.  Exactly what
    happens is complicated; if you *really* want to know, see

      util/config.py::Config._merge_trees()

    Here's an attempt to define it:

    augment
       When one config tree (the "right" tree) "augments" another (the
       "left" tree), generally speaking, stuff in the right tree is
       added to stuff in the left tree, but of course it's more
       complicated than that.  The augment process walks down the trees,
       from top to bottom.  It starts at the very top level, where the
       left dictionary and right dictionary are compared.

          * If the current item being compared have different types
            (scalar vs. list vs. dict), the new value *replaces* the old
            values.  This will never happen at the very top level,
            because both left and right are dictionaries at the top
            level.

          * If the item being compared is a dictionary, then merge the
            two dictionaries.  Keys in the right that don't show up in
            the left key have their key:value added wholesale to
            the left item.  If a key shows up in both the left and
            right trees, then recurse.  (So, the value of right[key]
            augments the value of left[key].)

          * If the item being compared is a list, then then the
            right list extends the left list.  (Literally using
            list.extend().)

          * If the item being compared is a scalar, then the right
            value replaces the left value.

    override
       When one config tree (the "right" tree) "overrides" another (the
       "left" tree), generally speaking, stuff in the right tree
       replaces stuff in the left tree.  The override process walks down
       the trees, from top ot bottom.  It starts at the very top level,
       where the left dictionary and the right dictionary are compared.

          * If the current item being compared have different types
            (scalar vs. list vs. dict), the new value *replaces* the old
            values.  This will never happen at the very top level,
            because both left and right are dictionaries at the top
            level.  This is exactly the same behavior as in augment.

          * If the current item being compared is a dictionary, then
            the dictionaries are merged in exactly the same manner
            as "augment", with the modification that recursing down
            into the dictionary passes along the fact that we're
            overriding rather than augmenting.

          * If the current item being compared is a list, then the
            right list *replaces* the left list.  (This could
            potentially throw away a gigantic hierarchy if lists
            and dicts and scalars from the left wide, which is as
            designed.)

          * If the item being compared is a scalar, then the right value

    This can be very confusing, so keeping your config overrides and
    augments simple is likely what you want.


   WARNINGS

    * Won't work with any old yaml file.  Top level must be a dictionary.
      Don't name your dictionary fields as numbers, as the code will then
      detect it as a list rather than a dict.

    * The yaml parser seems to be parsing "yyyy-mm-dd" strings as a
      datetime.date; beware.

    * python's yaml reading of floats is broken.  It will read 1e10 as a
      string, not a float.  Write 1.0e+10 to make it work right.  There
      are two things here: the first is the + after e (which, I think
      *is* part of the yaml spec, even though we can freely omit that +
      in Python and C).  The second is the decimal point; the YAML spec
      says it's not necessary, but python won't recognize it as a float
      without it.

    """

    # top_level_config is ill-considered.  It works if you are running
    #   from the source tree, but if you're running an installed instance,
    #   then it's not really pointing at the right place.  We should just
    #   return an error if SEECHANGE_CONFIG is not set.
    # top_level_config = str((pathlib.Path(__file__).parent.parent / "default_config.yaml").resolve())
    # _default_default = os.getenv('SEECHANGE_CONFIG', top_level_config)
    _default_default = os.getenv( 'SEECHANGE_CONFIG', None )

    _default = None
    _configs = {}


    @staticmethod
    def init( configfile=None, setdefault=None ):
        """Initialize configuration globally for process.

        Parameters
        ----------
        configfile : str or pathlib.Path, default None
            See documentation of the configfile parameter in Config.get

        setdefault : bool, default None
            See documentation of the setdefault parameter in Config.get

        """
        Config.get( configfile, setdefault=setdefault )


    @staticmethod
    def get( configfile=None, setdefault=None, static=True ):
        """Returns a Config object.

        Parameters
        -----------
        configfile : str or Pathlib.Path, default None
            The config file to read (if it hasn't been read before, or
            if reread is True).  If None, will return the default config
            context for the current session (which is normally the one
            in the file pointed to by environment variable
            SEECHANGE_CONFIG, but see "setdefault" below.  If that env
            var is needed but not set, then an exception will be
            raised).

        setdefault : bool, default None
            Avoid use of this, as it is mucking about with global
            variables and as such can cause confusion.  If True, set the
            Config object read by this method to be the session default
            config.  If False, never set the Config object read by this
            method to be the session default config.  If not specified,
            which is usually what you want, then if configfile is None,
            the configfile in SEECHANGE_CONFIG will be read and set to
            the be the session default Config; if configfile is not
            None, read that config file, but don't make it the session
            default config.

            Normal usage of Config is to make a call early on to either
            Config.init() or Config.get() without parameters.  That will
            read the config file in SEECHANGE_CONFIG and make that the
            default config for the process.  If, for some reason, you
            want to read a different config file and make that the
            default config file for the process, then pass a configfile
            here and make setdefault True.  If, for some truly perverse
            reason, you want to the config in SEECHANGE_CONFIG but not
            set it to the session default, then call
            Config.get(setdefault=False), and question your life
            choices.

        static : bool, default True
            If True (the default), then you get one of the config object
            singletons described below.  In this case, it is not
            permitted to modify the config.  If False, you get back a
            clone of the config singleton, and that clone is not stored
            anywhere other than the return value.  In this case, you may
            modify the config.  Call Config.get(static=False) to get a
            modifiable version of the default config.

        Returns
        -------
            Config object

        Config objects are stored as an array of singletons (as class
        variables of the Config class).  That is, if you pass a config
        file that has been passed before in the current execution
        context, you'll get back exactly the same object each time
        (unless static is True).  If you pass a config file that hasn't
        been passed before, it will read the indicated configuration
        file, cache an object associated with it for future calls, and
        return that object (unless static is True, in which case the the
        object is still cached, but you get a copy of that object).

        If you don't pass a config file, then you will get back the
        default config object.  If there is no default config object
        (because neither Config.get() nor Config.init() have been called
        previously), and if the class is not configured with a "default
        default", then an exception will be raised.

        """
        if configfile is None:
            if Config._default is not None:
                configfile = Config._default
            else:
                if Config._default_default is None:
                    raise RuntimeError( 'No default config defined yet; run Config.init(configfile)' )
                configfile = Config._default_default
                if setdefault is None:
                    setdefault = True

        configfile = str( pathlib.Path(configfile).resolve() )

        if configfile not in Config._configs:
            Config._configs[configfile] = Config( configfile=configfile )

        if setdefault:
            Config._default = configfile

        if static:
            return Config._configs[configfile]
        else:
            return Config( clone=Config._configs[configfile] )


    def __init__( self, configfile=None, clone=None ):
        """Don't call this, call static method Config.get().

        Parameters
        ----------
        configfile : str or Path, or None

        clone : Config object, default None

        If clone is not None, return a deep copy of that.  Otherwise,
        read the config file in configfile and return that.  Only give
        one of configfile or clone.

        """

        self._static = True

        if ( clone is None ) == ( configfile is None ):
            raise ValueError( "Must specify exactly one of configfile or clone" )

        if clone is not None:
            self._data = copy.deepcopy( clone._data )
            self._static = False
            return

        try:
            self._data = {}
            self._path = pathlib.Path( configfile ).resolve()
            curfiledata = yaml.safe_load( open(self._path) )
            if curfiledata is None:
                # Empty file, so self._data can stay as {}
                return
            if not isinstance( curfiledata, dict ):
                raise RuntimeError( f'Config file {configfile} doesn\'t have yaml I like.' )

            imports = { 'preloads': [], 'augments': [], 'overrides': [] }
            for importfile in [ 'preloads', 'augments', 'overrides' ]:
                if importfile in curfiledata:
                    if not isinstance( imports[importfile], list ):
                        raise TypeError( f'{importfile} must be a list' )
                    imports[importfile] = curfiledata[importfile]
                    del curfiledata[importfile]

            for preloadfile in imports['preloads']:
                # Because self._data is currently {}, this
                #   is just the same as setting self._data
                #   to the result of reading preloadfile
                self._override( preloadfile )

            # Override the preloads with the current file
            self._data = Config._merge_trees( self._data, curfiledata )

            for augmentfile in imports['augments']:
                self._augment( augmentfile )

            for overridefile in imports['overrides']:
                self._override( overridefile )

        except Exception as e:
            SCLogger.exception( f'Exception trying to load config from {configfile}' )
            raise e


    def _augment( self, augmentfile ):
        """Read file (or path) augmentfile and augment config data.  Intended for internal use only.

        Parameters
        ----------
        augmentfile: str or Path

        * If the old and new items have different types (scalar vs. list
          vs. dict), the new value replaces the old value.

        * If the value is a scalar, the new value replaces the old value.

        * If the value is a list, extend the existing list with what's
          present in the new file.

        * If the value is a dict, then merge the two dictionaries.  New
          keys are added.  When the key is present in both dictionaries,
          recurse.
        """
        augmentpath = pathlib.Path( augmentfile )
        if not augmentpath.is_absolute():
            augmentpath = ( self._path.parent / augmentfile ).resolve()
        if augmentpath.is_file():
            SCLogger.debug( f'Reading file {augmentfile} as an augment. ' )
            augment = Config( augmentpath )._data
            self._data = Config._merge_trees( self._data, augment, augment=True )
        elif augmentfile is not None:
            SCLogger.debug( f'Augment file {augmentfile} not found. ' )

    def _override( self, overridefile ):
        """Read file (or path) overridefile and override config data.  Intended for internal use only.

        Parameters
        ----------
        augmentfile: str or Path

        * If the old and new items have different types (scalar vs. list
          vs. dict), the new value replaces the old value.

        * If the value is a scalar, the new value replaces the old value.

        * If the value is a list, the old list is tossed out in favor of
          the new list.

        * If the value is a dict, then merge the two dictionaries.  New
          keys are added.  When the key is present in both dictionaries,
          recurse into the conflict rules.
        """
        overridepath = pathlib.Path( overridefile )

        if not overridepath.is_absolute():
            overridepath = ( self._path.parent / overridefile ).resolve()
        if overridepath.is_file():
            SCLogger.debug(f'Reading file {overridepath} as an override. ')
            override = Config( overridepath )._data
            self._data = Config._merge_trees( self._data, override )
        elif overridefile is not None:
            SCLogger.debug( f'Override file {overridefile} not found. ' )

    def value( self, field, default=NoValue(), struct=None ):
        """Get a value from the config structure.

        For trees, separate fields by periods.  If there is
        an array somewhere in the tree, then the array index
        as a number is the field for that branch.

        For example, if the config yaml file is;

        scalar: value

        dict1:
          dict2:
            sub1: 2level1
            sub2: 2level2

        dict3:
          list:
            - list0
            - list1

        then you could get values with:

        configobj.value( "scalar" ) --> returns "value"
        configobj.value( "dict1.dict2.sub2" ) --> returns "2level2"
        configobj.value( "dict3.list.1" ) --> returns "list1"

        You can also specify a branch to get back the rest of the
        subtree; for instance configobj.value( "dict1.dict2" ) would
        return the dictionary { "sub1": "2level1", "sub2": "2level2" }.

        Parameters
        ----------
        field: str
            See below

        struct: dict, default None
            If passed, use this dictionary in place of the object's own
            config dictionary.  Avoid use.

        Returns
        -------
        int, float, str, list, or dict

        If a list or dict, you get a deep copy of the original list or
        dict.  As such, it's safe to modify the return value without
        worrying about changing the internal config.  (If you want to
        change the internal config, use set_value().)

        """

        if struct is None:
            struct = self._data
        fields, isleaf, curfield, ifield = self._fieldsep( field )

        if isinstance( struct, list ):
            if ifield is None:
                raise ValueError( f'Failed to parse {curfield} as an integer index' )
            if ifield >= len(struct):
                raise ValueError( f'{ifield} > {len(struct)}, the length of the list' )
            if isleaf:
                return_value = struct[ifield]
            else:
                try:
                    return_value = self.value( ".".join(fields[1:]), default, struct[ifield] )
                except Exception as e:
                    if isinstance(default, NoValue):
                        raise ValueError( f'Error getting list element {ifield}' ) from e
                    else:
                        return_value = default
        elif isinstance( struct, dict ):
            if curfield not in struct:
                if isinstance(default, NoValue):
                    raise ValueError( f'Field {curfield} doesn\'t exist' )
                else:
                    return_value = default
            elif isleaf:
                return_value = struct[curfield]
            else:
                try:
                    return_value = self.value( ".".join(fields[1:]), default, struct[curfield] )
                except Exception as e:
                    if isinstance(default, NoValue):
                        raise ValueError( f'Error getting field {curfield}' ) from e
                    else:
                        return_value = default
        else:
            if not isleaf:
                raise ValueError( f'Tried to get field {curfield} of scalar!' )
            return_value = struct

        if isinstance(return_value, (dict, list)):
            return_value = copy.deepcopy( return_value )
        return return_value

    def set_value( self, field, value, structpass=None, appendlists=False ):
        """Set a value in the config object.

        If the config object was created with static=True (which is the
        case for all the singleton objects stored in the Config class),
        use of this method raises an exception.

        Parameters
        ----------
        field: str
            See value() for more information

        value: str, int, float, list, or dict

        structpass: some object with a ".struct" field
           Used internally when the Config object is building it's own
           _data field; don't use externally

        appendlists: bool, default False
           If true and if field is a pre-existing list, then value is
           appended to the list.  Otherwise, value replaces the
           pre-existing field if there is one.

        Does not save to disk.  Follows the standard rules docuemnted in
        "augment" and "override"; if appendlists is True, uses
        "augment", else "override".  Will create the whole hierarchy if
        necessary.

        """

        if self._static:
            raise RuntimeError( "Not permitted to modify static Config object." )

        if structpass is None:
            structpass = types.SimpleNamespace()
            structpass.struct = self._data
        elif not hasattr( structpass, 'struct' ):
            raise ValueError( 'structpass must have a field "struct"' )
        fields, isleaf, curfield, ifield = self._fieldsep( field )

        if isleaf:
            if isinstance( structpass.struct, list ):
                if appendlists:
                    if ifield is None:
                        raise TypeError( "Tried to add a non-integer field to a list." )
                    structpass.struct.append( value )
                else:
                    if ifield is None:
                        structpass.struct = { curfield: value }
                    else:
                        structpass.struct = [ value ]
            elif isinstance( structpass.struct, dict ):
                if ifield is not None:
                    raise TypeError( "Tried to add an integer field to a dict." )
                structpass.struct[curfield] = value
            else:
                structpass.struct = { curfield: value }
        else:
            structchuck = types.SimpleNamespace()
            try:
                nextifield = int( fields[1] )
            except ValueError:
                nextifield = None

            if isinstance( structpass.struct, list ):
                structchuck.struct = {} if nextifield is None else []
                self.set_value( ".".join(fields[1:], value, structchuck, appendlists=appendlists ) )
                if appendlists:
                    if ifield is None:
                        raise TypeError( "Tried to add a non-integer field to a list" )
                    structpass.struct.append( structchuck.struct )
                else:
                    if ifield is None:
                        structpass.struct = { curfield: structchuck.struct }
                    else:
                        structpass.struct = [ structchuck.struct ]
            else:
                if ifield is None:
                    if isinstance( structpass.struct, dict ):
                        if curfield in structpass.struct:
                            structchuck.struct = structpass.struct[curfield]
                        else:
                            structchuck.struct = {} if nextifield is None else []
                    else:
                        structpass.struct = {}
                    self.set_value( ".".join(fields[1:]), value, structchuck, appendlists=appendlists )
                    structpass.struct[curfield] = structchuck.struct
                else:
                    if isinstance( structpass.struct, dict ):
                        raise TypeError( "Tried to add an integer field to a dict." )
                    structchuck.struct = {} if nextifield is None else []
                    self.set_value( ".".join(fields[1:]), value, structchuck, appendlists=appendlists )
                    structpass.struct = [ structchuck.struct ]

    @classmethod
    def _fieldsep( cls, field ):
        """Parses a period-separated config specifier string.  Internal use only.

        Parameters
        ----------
        field: str
            A field specifier to parse

        Returns
        -------
        tuple with 4 elements: fields, isleav, curfield, ifield.
          fields : list of the hierarchy (e.g. "val1.val2.val3" returns ["val1","val2","val3"])
          isleaf : True if len(fields) is 1, otherwise false
          curfield : The first element of the field (val1 in the example above)
          ifield : None if curfield is not an integer, otherwise the integer value of curfield

        """
        fields = field.split( "." )
        isleaf = ( len(fields) == 1 )
        curfield = fields[0]
        try:
            ifield = int(curfield)
        except ValueError:
            ifield = None
        return fields, isleaf, curfield, ifield

    @staticmethod
    def _merge_trees( left, right, augment=False ):
        """Internal usage, do not call.

        Parameters
        ----------
        left: dict

        right: dict

        augment: bool

        Merge two config trees, with the right dict overriding the left
        dict using the rules described in the documentation of the
        Config class.

        """

        if isinstance( left, list ):
            if not isinstance( right, list ) or ( not augment ):
                return copy.deepcopy( right )
            else:
                newlist = copy.deepcopy( left )
                newlist.extend( copy.deepcopy( right ) )
                return newlist
        elif isinstance( left, dict ):
            if not isinstance( right, dict ):
                return copy.deepcopy( right )
            newdict = copy.deepcopy( left )
            for key, value in right.items():
                if key in newdict:
                    newdict[key] = Config._merge_trees( newdict[key], right[key], augment=augment )
                else:
                    newdict[key] = copy.deepcopy( right[key] )
            return newdict
        else:
            return copy.deepcopy( right )


if __name__ == "__main__":
    Config.init()
