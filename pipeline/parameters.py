import pathlib
import copy
import random
import warnings

import pyarrow.parquet
import healpy

from util.logger import SCLogger
from util.radec import ra_cross_zero_avgra

# parameters that are propagated from one Parameters object
# to the next when adding embedded objects.
# If the embedded object's Parameters doesn't have any of these
# then that key is just skipped


class Parameters:
    """Keep track of parameters for any of the pipeline classes.

    You can access the parameters as attributes, but also
    as dictionary items (using "in" and "pars[key]").

    Methods
    -------
    - add_par() to add a new parameter (mostly in the __init__).
    - add_alias() to add an alias for a parameter.
    - merge_configs() takes lists of dictionaries and does things with them recursively
    - override() the parameters from a dictionary.
    - augment() takes parameters from a dictionary and updates (instead of overriding dict/set parameters).
    - subconfig_update() [ see it's own docstring ]
    - to_dict() converts the non-private parameters to a dictionary.
    - copy() makes a deep copy of this object.
    - compare() checks if the parameters of two objects are the same.
    - add_defaults_to_dict() adds some attributes that should be shared to all pars.
    - show_pars() shows the parameters and descriptions.
    - vprint() prints the text given if the verbose level exceeds a threshold.

    Adding new parameters
    ---------------------
    To add new parameters directly on the object,
    or in the __init__ of a subclass, use the add_par() method.
    signature: add_par(par_name, default, types, description, critical).
    This allows adding the type, the default, and the description
    of the parameter, which are very useful when printing out
    the details of the configuration. The critical flag is used
    to indicate parameters that have an effect on the output products.
    E.g., a photometric aperture radius is critical, but a verbosity
    level is not. Parameters added this way can be recovered as a dictionary,
    using the to_dict() method (where you can include or exclude hidden and
    non-critical parameters).

    If the values are hard-coded, e.g., in a Parameters subclass,
    use the self._enforce_no_new_attrs = True at the end of the constructor.
    This will raise an error if the user tries to set/get the wrong attribute.

    The types of the parameters are enforced, so you cannot set a parameter
    of the wrong type, unless you specify self._enforce_type_checks = False.

    Parameter matching
    ------------------
    To access a parameter using a key that is similar
    to the parameter name (e.g., as a shorthand) use
    the _ignore_case, _allow_shorthands and _remove_underscores
    parameters. This affects the way comparison between the given
    string and the parameter names are made.
    * _ignore_case: if True, the case of the given string is ignored.
    * _allow_shorthands: if True, the parameter name needs to begin
      with the given string.
    * _remove_underscores: if True, the underscores in the parameter name
      and the input string are removed before comparison.
    If more than one parameter matches, will raise a ValueError.
    Finally, to add a different name as an alias to the original parameter,
    use add_alias(new_name, old_name). The alias name is also subject
    to the same matching rules as the original parameter name,
    but if multiple aliases for the same actual parameter are found,
    only the match to the original is used.
    E.g., if int_par is an alias to int_parameter, and assuming
    _allow_shorthands = True, then both keys will match the same
    parameter (int_parameter) but this will not raise an exception
    as both refer to the same parameter.
    However, if you also define another parameter int_value,
    that is not an alias but a different parameter, then
    calling pars.int_ will match multiple different parameters
    and will raise an exception.

    Sub classes
    -----------
    When sub-classing, make sure to call the super().__init__(),
    and then add all the specific parameters using add_par().
    Then finish by calling self._enforce_no_new_attrs = True.
    If you are sub-sub-classing, then make sure to set
    self._enforce_no_new_attrs = False before adding new
    parameters, and locking it back to True at the end.
    After locking the object, call self.override(**kwargs)
    to set parameters from the initialization call.
    """

    def __init__(self, **kwargs):
        """Set up a Parameters object.

        After setting up, the parameters can be set
        either by hard-coded values or by a YAML file,
        using the load() method,
        or by a dictionary, using the read() method.

        When adding parameters, use the add_par() method,
        that accepts the variable name, the type(s),
        and a docstring describing the parameter.
        This allows for type checking and documentation.

        Subclasses of this class should add their own
        parameters then override the allow_adding_new_attributes()
        method to return False, to prevent the user from
        adding new parameters not defined in the subclass.
        """

        self.__typecheck__ = {}
        self.__defaultpars__ = {}
        self.__docstrings__ = {}
        self.__critical__ = {}
        self.__aliases__ = {}

        self.verbose = self.add_par(
            "verbose", 0, int, "Level of verbosity (0=quiet).", critical=False
        )

        self.inject_warnings = self.add_par(
            "inject_warnings",
            False,
            (bool, float),
            "Inject warnings into the pipeline. If given as float, use as probability to inject. ",
            critical=False,
        )

        self.inject_exceptions = self.add_par(
            "inject_exceptions",
            False,
            (bool, float),
            "Inject exceptions into the pipeline. If given as float, use as probability to inject. ",
            critical=False,
        )

        self.inject_hangups = self.add_par(
            "inject_hangups",
            False,
            (bool, float),
            "Inject hangups into the pipeline. If given as float, use as probability to inject. "
            "Note that if an exception occurs from inject exceptions, the hangup would not be reached. ",
            critical=False,
        )

        self._enforce_type_checks = self.add_par(
            "_enforce_type_checks",
            True,
            bool,
            "Choose if input values should be checked "
            "against the type defined in add_par().",
            critical=False,
        )

        self._enforce_no_new_attrs = self.add_par(
            "_enforce_no_new_attrs",
            False,
            bool,
            "Choose if new attributes should be allowed "
            "to be added to the Parameters object. "
            "Set to True to lock the object from further changes. ",
            critical=False,
        )

        self._allow_shorthands = self.add_par(
            "_allow_shorthands",
            True,
            bool,
            "If true, can refer to a parameter name by a partial string, "
            "as long as the partial string is unique. "
            "This slows down getting and setting parameters. ",
            critical=False,
        )

        self._ignore_case = self.add_par(
            "_ignore_case",
            True,
            bool,
            "If true, the parameter names are case-insensitive. ",
            critical=False,
        )

        self._remove_underscores = self.add_par(
            "_remove_underscores",
            False,
            bool,
            "If true, underscores are ignored in parameter names. ",
            critical=False,
        )

        self._cfg_key = self.add_par(
            "_cfg_key",
            None,
            (None, str),
            "The key to use when loading the parameters from a YAML file. "
            "This is also the key that will be used when writing the parameters "
            "to the output config file. ",
            critical=False,
        )
        self._cfg_sub_key = self.add_par(
            "_cfg_sub_key",
            None,
            (None, str),
            "The sub-key to use when loading the parameters from a YAML file. "
            "E.g., the observatory name under observatories. ",
            critical=False,
        )

        self.override(kwargs)

    def _get_real_par_name(self, key):
        """Get the real parameter name from a partial string.

        Ignores case, and follows the alias dictionary.
        """
        if key in self.__dict__:
            return key

        if key.startswith("__"):
            return key

        if (
            "_allow_shorthands" not in self.__dict__
            or "_ignore_case" not in self.__dict__
            or "_remove_underscores" not in self.__dict__
            or "__aliases__" not in self.__dict__
        ):
            return key

        # get these without passing back through the whole machinery
        allow_shorthands = super().__getattribute__("_allow_shorthands")
        ignore_case = super().__getattribute__("_ignore_case")
        remove_underscores = super().__getattribute__("_remove_underscores")
        aliases_dict = super().__getattribute__("__aliases__")

        if not allow_shorthands and not ignore_case and not remove_underscores:
            return key

        if ignore_case:

            def reducer1(x):
                return x.lower()

        else:

            def reducer1(x):
                return x

        if remove_underscores:

            def reducer(x):
                return reducer1(x.replace("_", ""))

        else:

            def reducer(x):
                return reducer1(x)

        if allow_shorthands:

            def comparator(x, y):
                return x.startswith(y)

        else:

            def comparator(x, y):
                return x == y

        matches = [
            v for k, v in aliases_dict.items() if comparator(reducer(k), reducer(key))
        ]
        matches = list(set(matches))  # remove duplicates

        if len(matches) > 1:
            raise ValueError(
                f"More than one parameter matches the given key: {key}. "
                f"Matches: {matches}. "
            )
        elif len(matches) == 0:
            # this will either raise an AttributeError or not,
            # depending on _enforce_no_new_attrs (in setter):
            return key
        else:
            return matches[0]  # one match is good!

    def __getattr__(self, key):

        real_key = self._get_real_par_name(key)

        # finally get the value:
        return super().__getattribute__(real_key)

    def __setattr__(self, key, value):
        """Set an attribute of this object.

        There are some limitations on what can be set:
        1) if this class has allow_adding_new_attributes=False,
           no new attributes can be added by the user
           (to prevent setting parameters with typoes, etc).
        2) if self._enforce_type_checks=True, then the type of the
           value must match the types allowed by the add_par() method.

        """
        real_key = self._get_real_par_name(key)

        new_attrs_check = (
            hasattr(self, "_enforce_no_new_attrs") and self._enforce_no_new_attrs
        )

        if (
            new_attrs_check
            and real_key not in self.__dict__
            and real_key not in self.propagated_keys()
        ):
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{key}'"
            )

        type_checks = (
            hasattr(self, "_enforce_type_checks") and self._enforce_type_checks
        )
        if (
            type_checks
            and real_key in self.__typecheck__
            and not isinstance(value, self.__typecheck__[real_key])
        ):
            strv = None
            try:
                # Added this in an attempt to diagnose a very mysterious error.
                # (The thing should not have been a float!)
                strv = f"  [Value: {str(value)}]"
            except Exception:
                pass
            raise TypeError(
                f'Parameter "{key}" must be of type {self.__typecheck__[real_key]}, got {type(value)} instead.{strv}'
            )
        super().__setattr__(real_key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def add_par(self, name, default, par_types, docstring, critical=True):
        """Add a parameter to the list of allowed parameters.

        To add a value in one line (in the __init__ method):
        self.new_var = self.add_par('new_var', (bool, NoneType), False, "Description of new_var.")

        Parameters
        ----------
        name: str
            Name of the parameter. Must match the name of the variable
            that the return value is assigned to, e.g., self.new_var.
        default: any
            The default value of the parameter. Must match the given par_types.
        par_types: type or tuple of types
            The type(s) of the parameter. Will be enforced using isinstance(),
            when _enforce_type_checks=True.
        docstring: str
            A description of the parameter. Will be used in the docstring of the class,
            and when using the print() method or _get_par_string() method.
        critical: bool
            If True, the parameter is included in the list of parameters
            that have an important effect on the download/analysis results.

        Returns
        -------

        """
        if name in self.__typecheck__:
            raise ValueError(f"Parameter {name} already exists.")
        if isinstance(par_types, (set, list)):
            par_types = tuple(par_types)
        if not isinstance(par_types, tuple):
            par_types = (par_types,)
        par_types = tuple(type(pt) if pt is None else pt for pt in par_types)
        if float in par_types:
            par_types += (int,)
        self.__typecheck__[name] = par_types
        self.__docstrings__[name] = docstring
        self.__defaultpars__[name] = default
        self.__critical__[name] = critical
        self.__aliases__[name] = name

        self[name] = default  # this should fail if wrong type?
        return default

    def add_alias(self, alias, name):
        """Add an alias for a parameter.

        Whenever the alias is used, either for get or set,
        the call will be re-routed to the original parameter.

        Example: self.add_alias('max_mag', 'maximum_magnitude')
        can be used to quickly refer to the longer name using
        the shorthand "max_mag".

        Parameters
        ----------
        alias: str
            The alias to use.
        name: str
            The original name of the parameter, to redirect to.
        """
        self.__aliases__[alias] = name

    def _recursive_merge( self, left, right, toplevel=True ):
        if toplevel:
            # First, fix the keys for the real keys.
            # This means more deepcopies than strictly necessary, but oh well.
            if ( not isinstance( left, dict ) ) or ( not isinstance( right, dict ) ):
                raise TypeError( f"Top level merges must all be dicts, got {type(left)} and {type(right)}" )
            left = copy.deepcopy( left )
            right = copy.deepcopy( right )
            for which in [ left, right ]:
                for key, val in which.items():
                    newkey = self._get_real_par_name( key )
                    if newkey != key:
                        if key in which:
                            raise ValueError( f"Different keys point to the same parameter: {key}, {newkey}" )
                        which[newkey] = which[key]
                        del which[key]

        # Now we can merge
        if isinstance( left, dict ):
            if not isinstance( right, dict ):
                return copy.deepcopy( right )
            newdict = copy.deepcopy( left )
            for key, value in right.items():
                if key in newdict:
                    newdict[key] = self._recursive_merge( newdict[key], right[key], toplevel=False )
                else:
                    newdict[key] = copy.deepcopy( right[key] )
            return newdict
        else:
            return copy.deepcopy( right )

    def merge_configs( self, before=[], after=[] ):
        """Does a particular kind of merge.

        It's designed for configs that have subconfigs where you might
        want to override just parts of the subconfig without having to
        replace the whole config.  Everyting in before is merged in
        order, then the current value of the parameters in this object
        are merged in, then everything in after is merged in order.

        Each pairwise merge is a "left/right" merge.  left and right
        must both be dictionaries (so before and after have to be lists
        of dicts.)  If left[key] and right[key] are not both
        dictionaries, then left[key] is replaced with right[key].  (So,
        lists are fully replaced, not extended!)  Otherwise, right
        updates left, sorta... because instead of a straight
        left.update(right), it crawls down through both dictionaries so
        that dictionaries of dictionaries will all have things updated.

        Lots of deepcopies are done in the process, so, presuming we
        didn't screw it up, your dictionaries should all be untouched.

        For example:

           left = { 'a': [ 1, 2, 3 ],
                    'b': { 'foo': 'bar', 'baz': qux' },
                    'c': { 'alpha': 13,
                           'beta': 42,
                           'gamma': { 'mercury': 1,
                                      'venus': 2,
                                      'earth': 3
                                    }
                         },
                    'd': 'kittens',
                    'α': 'ω'
                 }

           right = { 'a': [ 4, 5, 6 ],
                     'b': 128,
                     'c': { 'alpha': 137,
                            'gamma': { 'earth': 0.,
                                       'mars': 4
                                     }
                          },
                     'e': 'puppies'
                     'α': { 'β': 'electron',
                            'γ': 'photon'
                          }
                   }

        Then the resultant left/right merge would be:

        { 'a': [ 4, 5, 6 ],
          'b': 128,
          'c': { 'alpha': 137,
                 'beta': 42,
                 'gamma': { 'mercury': 1,
                            'venus': 2,
                            'earth': 0,
                            'mars': 4
                          }
               },
          'd': 'kittens',
          'e': 'puppies',
          'α': { 'β': 'electron',
                 'γ': 'photon'
               }
        }

        """

        if ( not isinstance(before, list) ) or ( not isinstance(after, list) ):
            raise TypeError( "Must pass lists of dicts ot merge_config" )

        everything = before.copy()
        everything.append( self.to_dict() )
        everything.extend( after )

        result = {}
        for item in everything:
            if isinstance( item, Parameters ):
                result = self._recursive_merge( result, item.to_dict() )
            else:
                result = self._recursive_merge( result, item )

        self.override( result )

    def override(self, dictionary, ignore_addons=False):

        """Read parameters from a dictionary.

        If any parameters were already defined,
        they will be overridden by the values in the dictionary.

        Parameters
        ----------
        dictionary: dict
            A dictionary with the parameters.
        ignore_addons: bool
            If True, will ignore any parameters that are not
            already defined in the object. If False, will raise
            an AttributeError if a parameter is not already defined
            (unless the Parameter object's _enforce_no_new_attrs is False,
            in which case you can add new parameters).
        """
        for k, v in dictionary.items():
            try:
                self[k] = v
            except AttributeError as e:
                if not ignore_addons and "has no attribute" in str(e):
                    raise e


    def augment(self, dictionary, ignore_addons=False):
        """Update parameters from a dictionary.

        Any dict or set parameters already defined
        will be updated by the values in the dictionary,
        otherwise values are replaced by the input values.

        TODO : make this recursive.  Cf. util.config.Config._merge_trees

        Parameters
        ----------
        dictionary: dict
            A dictionary with the parameters.
        ignore_addons: bool
            If True, will ignore any parameters that are not
            already defined in the object. If False, will raise
            an AttributeError if a parameter is not already defined
            (unless the Parameter object's _enforce_no_new_attrs is False,
            in which case you can add new parameters).
        """

        for k, v in dictionary.items():
            if self._get_real_par_name(k) in self:  # need to update
                if isinstance(self[k], set) and isinstance(v, (set, list)):
                    self[k].update(v)
                elif isinstance(self[k], dict) and isinstance(v, dict):
                    self[k].update(v)
                # what about list? should we append existing lists?
                else:
                    self[k] = v
            else:  # just add this parameter
                try:
                    self[k] = v
                except AttributeError as e:
                    if not ignore_addons and "has no attribute" in str(e):
                        raise e


    def subconfig_update( self, ds ):
        """Modify the parameters based on the contents of DataStore ds.

        For this to work, there must be a "subconfigs" paramameter, which is a
        dictionary that has (at least) keys "choice_algorithm" and
        "subconfigs".  choice_algorithm is string, and subconfigs is a dict of
        dicts.  The keys of subconfigs are strings that might be returned by
        the function that choice_algorithm specifies.  The values of
        subconfigs are dictionaries with replacement parametrs for any/all of
        the other parameters defined in this Parmeters object.

        Optionally, the subconfigs parameter can also have keys
        "choice_params" and "default_subconfig".  choice_params are passed as
        **kwargs to the function that choice_algorithm specifies.  See below
        for "default_subconifg".

        Optinally, there can also be a "subconfigs_noncritical" parameter,
        with keys "choice_params" (optional) and "subconfigs" (required).
        choice_params here is merged with choiceparams from subconfigs (with
        subconfigs taking priority in the case of the same key showing up in
        both); you would put things like directories and other system-specific
        stuff in the choice_params here, saving choice_params in subconfigs
        for things that would actually change the answer.

        A function, indicated by choice_algorithm, will be called, being
        passed ds and **choice_params.  It will return a string, call it
        chosen_subconfig.

        A replacement dictionary will be built by looking in
        subconfigs['subconfigs'].  It starts with {}.  If
        subconfigs['default_subconfig'] exists, it updates with
        subconfigs['subconfigs'][subconfigs['default_subconfig']].  It then
        updates with subconfigs['subconfigs'][chosen_subconfig].

        Finally, that replacment dictionary is passed to self.override().

        By convention, the parmeter subconfigs should have critical=True and
        the parameter subconfigs_noncritical should have critical=False.  You
        probably also want to make all parameters that will be modified by
        subconfig_update to be critical=False at the top level.  That way, you
        will get the same provenance regardless of which subconfig
        subconfig_udpate chooses.

        """

        if not hasattr( self, 'subconfigs' ):
            raise ValueError( f"subconfig_update requires {self.__class__.__name__} "
                              f"to have parameter subconfigs" )

        if ( not isinstance( self.subconfigs, dict ) ) or ( 'choice_algorithm' not in self.subconfigs ):
            raise ValueError( f"{self.__class__.__name__} parameter subconfigs must have choice_algorithm" )

        if self.subconfigs['choice_algorithm'] is None:
            return

        if 'subconfigs' not in self.subconfigs:
            raise ValueError( f"{self.__class__.__name__} parameter subconfigs must have subconfigs if "
                              f"choice_algorithm is not None." )

        if hasattr( self, 'subconfigs_noncritical' ):
            if not isinstance( self.subconfigs_noncritical, dict ):
                raise TypeError( f"If {self.__class__.__name__} has subconfigs_noncritical, it must be a dict." )
            if not set( self.subconfigs_noncritical ).issubset( { 'choice_params', 'subconfigs' } ):
                raise ValueError( f"{self.__class__.__name__} parameter subconfigs_noncritical can only have keys "
                                  f"choice_params and subconfigs" )

        newconfig = {}
        if 'default_subconfig' in self.subconfigs:
            if self.subconfigs['default_subconfig'] not in self.subconfigs['subconfigs']:
                raise ValueError( f"{self.__class__.__name__} default_subconfig "
                                  f"{self.subconfigs['default_subconfig']} is not in "
                                  f"subconfigs: {self.subconfigs['subconfigs'].keys()}" )
            newconfig.update( self.subconfigs['subconfigs'][ self.subconfigs['default_subconfig'] ] )

        choice_params = {}
        if ( hasattr( self, 'subconfigs_noncritical' ) and ( 'choice_params' in self.subconfigs_noncritical ) ):
            choice_params.update( self.subconfigs_noncritical['choice_params'] )
        if 'choice_params' in self.subconfigs:
            choice_params.update( self.subconfigs['choice_params'] )

        if self.subconfigs['choice_algorithm'] == 'star_density':
            subconfigtag = self.get_tag_for_star_density( ds, **choice_params )
        else:
            raise ValueError( f"Unknown config choice algorithm {self.subconfigs['choice_algorithm']}" )

        if subconfigtag not in self.subconfigs['subconfigs']:
            raise ValueError( f"Tried to set subconfig {subconfigtag}, but it's not present in "
                              f"subconfigs['subconfigs'] for {self.__class__.__name__}" )

        SCLogger.debug( f"Updating config for {self.__class__.__name__} with tag {subconfigtag}" )
        newconfig.update( self.subconfigs['subconfigs'][subconfigtag] )

        if ( ( hasattr( self, 'subconfigs_noncritical' ) ) and
             ( subconfigtag in self.subconfigs_noncritical['subconfigs'] )
            ):
            newconfig.update( self.subconfigs_noncritical['subconfigs'][subconfigtag] )

        self.override( newconfig )


    @classmethod
    def get_tag_for_star_density( cls, ds,
                                  config_dir=None,
                                  gaia_density_catalog="share/gaia_density/gaia_healpix_density.pq",
                                  star_mag_cutoff=20,
                                  star_density_cutoff=1e5
                                 ):
        """The function called by subconfig_update for choice_algorithn 'star_density'.

        Parametrs
        ---------
        ds : DataStore

        config_dir : str or Path
           The root directory where parquet catalogs with star densities are found.  Defaults to
           CODE_ROOT.

        gaia_density_catalog : str, default "share/gaia_density/gaia_healpix_density.pq"

        star_mag_cutoff: int, default 20
          Consider the density of stars at this magnitude and brighter.  This is actually the name
          of a column in the gaia_density_catalog, so only specific magnitudes will work.

        star_density_cutoff: float, default 1e5
          This many stars in a healpix(32,nest=True) healpix is consider "a lot".  More than this,
          the field is considered galactic, less than this, the field is considered extragalactic.

        Returns
        -------
        'galactic' or 'extragalactic'

        """
        # Avoid circular import (which may not be necsesary here, but might be in the future; as a
        #  general principle, parameters should probably not globally import anything other than
        #  standard stuff and util.* stuff.)
        import models.base
        config_dir = models.base.CODE_ROOT if config_dir is None else config_dir
        tablefile = pathlib.Path( config_dir ) / gaia_density_catalog

        if ds.wcs is not None:
            ra0 = ra_cross_zero_avgra( ds.wcs.good_minra, ds.wcs.good_maxra )
            dec0 = ( ds.wcs.good_mindec + ds.wcs.good_maxdec ) / 2.

        elif ds.image is not None:
            ra0 = ds.image.ra
            dec0 = ds.image.dec

        elif ds.exposure is not None:
            ra0 = ds.exposure.ra
            dec0 = ds.exposure.dec

        else:
            raise RuntimeError( "Couldn't find a WorldCoordinates, Image, or Exposure in passed DataStore." )

        densitytab = pyarrow.parquet.read_table( tablefile ).to_pandas()
        if str(star_mag_cutoff) not in densitytab.columns:
            raise ValueError( f"Don't have star densities for magnitude limit {star_mag_cutoff}" )

        hp = healpy.ang2pix( 32, ra0, dec0, nest=True, lonlat=True )
        row = densitytab[ densitytab['healpix32'] == hp ]
        if len(row) == 0:
            raise ValueError( f"Failed to find healpix {hp} in gaia density table" )
        if len(row) > 1:
            raise ValueError( f"Healpix {hp} shows up in gaia density table more than once; this shouldn't happen." )

        dens = row[ str(star_mag_cutoff) ].values[ 0 ]
        return 'galactic' if dens > star_density_cutoff else 'extragalactic'


    def get_critical_pars(self):

        """Get a dictionary of the critical parameters.

        Returns
        -------
        dict
            The dictionary of critical parameters.
        """
        return self.to_dict(critical=True, hidden=True)


    def to_dict(self, critical=False, hidden=False):
        """Convert parameters to a dictionary.

        Only get the parameters that were defined
        using the add_par method.

        Parameters
        ----------
        critical: bool
            If True, only get the critical parameters.
        hidden: bool
            If True, include hidden parameters.
            By default, does not include hidden parameters.

        Returns
        -------
        output: dict
            A dictionary with the parameters.
        """
        output = {}
        for k in self.__defaultpars__.keys():
            if hidden or not k.startswith("_"):
                if not critical or self.__critical__[k]:
                    output[k] = self[k]

        return output

    def copy(self):
        """Create a copy of the parameters."""
        return copy.deepcopy(self)

    @staticmethod
    def propagated_keys():
        """Which parameters are propagated.

        Parameter values that are propagated from one Parameters object
        to the next when adding embedded objects.
        """
        return ['verbose']

    def add_defaults_to_dict(self, inputs):
        """Add some default keywords to the inputs dictionary.

        If these keys already exist in the dictionary,
        they will not be overriden (they're given explicitly by the user).
        This is useful to automatically propagate parameter values that
        need to be shared by sub-objects (e.g., "verbose").

        Note that the "inputs" dictionary is modified in-place.
        """
        keys = self.propagated_keys()
        for k in keys:
            if k in self and k not in inputs:
                inputs[k] = self[k]

    def show_pars(self, owner_pars=None):
        """Print the parameters.

        If given an owner_pars input,
        will not print any of the propagated_keys
        parameters if their values are the same
        in the owner_pars object.
        """
        if owner_pars is not None and not isinstance(owner_pars, Parameters):
            raise ValueError("owner_pars must be a Parameters object.")

        names = []
        desc = []
        defaults = []
        for name in self.__dict__:
            if name.startswith("_"):
                continue
            if owner_pars is not None:
                if name in self.propagated_keys() and self[name] == owner_pars[name]:
                    defaults.append(name)
                    continue

            desc.append(self._get_par_string(name))
            names.append(name)

        if len(defaults) > 0:
            print(f" Propagated pars: {', '.join(defaults)}")
        if len(names) > 0:
            max_length = max(len(n) for n in names)
            for n, d in zip(names, desc):
                print(f" {n:>{max_length}}{d}")

    def vprint(self, text, threshold=1):
        """Print the text to standard output if the verbose level is above a given threshold.

        Parameters
        ----------
        text: str
            The text to print.
        threshold: bool or int
            The minimal level of the verbose parameter needed
            to print out the text. If self.verbose is lower
            than that, nothing will be printed.

        """
        if self.verbose > threshold:
            SCLogger.debug(text)

    def compare(self, other, hidden=False, critical=False, ignore=None, verbose=False):
        """Check that all parameters are the same between two Parameter objects.

        Will only check those parameters that were added using the
        add_par() method.  By default, ignores hidden parameters even if
        they were added using add_par().

        Parameters
        ----------
        other: Parameters object
            The other Parameters object to compare to.
        hidden: bool
            If True, include hidden parameters.
            By default, does not include hidden parameters.
        critical: bool
            If True, only compare the critical parameters.
            By default, ignores critical status.
        ignore: list of str
            A list of parameters to ignore in the comparison.
        verbose: bool
            If True, print the differences between the two
            Parameter objects.

        Returns
        -------
        same: bool
            True if all parameters are the same.

        """
        if ignore is None:
            ignore = []

        same = True
        for k in self.__defaultpars__.keys():
            if k in ignore:
                continue
            if (hidden or not k.startswith("_")) and (not critical or self.__critical__[k]) and self[k] != other[k]:
                same = False
                if not verbose:
                    break
                SCLogger.debug(f'Par "{k}" is different: {self[k]} vs {other[k]}')

        return same

    def _get_par_string(self, name):
        """Get the value, docstring and default of a parameter."""

        desc = default = types = critical = ""
        value = self[name]

        if name in self.__docstrings__:
            desc = self.__docstrings__[name].strip()
            if desc.endswith("."):
                desc = desc[:-1]
        if name in self.__defaultpars__:
            def_value = self.__defaultpars__[name]
            if def_value == value:
                default = "default"
            else:
                default = f"default= {def_value}"
        if name in self.__typecheck__:
            types = self.__typecheck__[name]
            if not isinstance(types, tuple):
                types = (types,)
            types = (t.__name__ for t in types)
            types = f'types= {", ".join(types)}'
        if name in self.__critical__:
            critical = 'critical' if self.__critical__[name] else ''
        extra = ", ".join([s for s in (default, types, critical) if s])
        if extra:
            extra = f" [{extra}]"

        if isinstance(value, str):
            value = f'"{value}"'
        s = f"= {value} % {desc}{extra}"

        return s

    def get_process_name(self):
        """Get the name of the process (pipeline phase) that is relevant for this Parameter object.

        Should be implemented in each subclass.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def do_warning_exception_hangup_injection_here(self):
        """When called, will check if any of the inject_ parameters are set to non-zero value.

        If they are, will raise a warning, exception or hangup, depending on the value.
        If any of the values is a float between 0 and 1, will compare it to a uniform random number,
        and if that random number is lower than the value, will inject the warning, exception or hangup.
        Note that if an exception occurs, it will override the hangup (it would not be reached).

        """
        if self.inject_warnings > 0:
            if random.uniform(0, 1) <= self.inject_warnings:
                warnings.warn(f"Warning injected by pipeline parameters in process '{self.get_process_name()}'.")

        if self.inject_exceptions > 0:
            if random.uniform(0, 1) <= self.inject_exceptions:
                raise RuntimeError(f"Exception injected by pipeline parameters in process '{self.get_process_name()}'.")

        if self.inject_hangups > 0:
            if random.uniform(0, 1) <= self.inject_hangups:
                while True:
                    pass


class ParsDemoSubclass(Parameters):
    def __init__(self, **kwargs):
        super().__init__()  # initialize base Parameters without passing arguments

        self.integer_parameter = self.add_par(
            "integer_parameter", 1, int, "An integer parameter", critical=True
        )
        self.add_alias("int_par", "integer_parameter")  # shorthand

        self.float_parameter = self.add_par(
            "float_parameter", 1.0, float, "A float parameter", critical=True
        )
        self.add_alias("float_par", "float_parameter")  # shorthand

        self.plotting_value = self.add_par(
            "plotting_value", True, bool, "A boolean parameter", critical=False
        )

        self._secret_parameter = self.add_par(
            "_secret_parameter", 1, int, "An internal (hidden) parameter", critical=True
        )

        self.nullable_parameter = self.add_par(
            "nullable_parameter",
            1,
            [int, None],
            "A parameter we can set to None",
            critical=True,
        )

        # lock this object so it can't be accidentally given the wrong name
        self._enforce_no_new_attrs = True

        self.override(kwargs)


if __name__ == "__main__":
    p = Parameters()
