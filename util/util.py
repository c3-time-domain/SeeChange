import collections.abc
import numbers
import os
import io
import re
import pathlib
import time
from datetime import datetime, date
import dateutil.parser
import uuid
import json

import numpy as np

from astropy.time import Time

from util.logger import SCLogger


def asUUID( id ):
    """Pass either a UUID or a string representation of one, get a UUID back."""
    if isinstance( id, uuid.UUID ):
        return id
    if not isinstance( id, str ):
        raise TypeError( f"asUUID requires a UUID or a str, not a {type(id)}" )
    return uuid.UUID( id )


class NumpyAndUUIDJsonEncoder(json.JSONEncoder):
    """Encodes UUID to strings, also encodes numpy stuff to python things, and datetime to a string."""

    def default(self, obj):
        if isinstance( obj, np.integer ):
            return int( obj )
        if isinstance( obj, np.floating ):
            return float( obj )
        if isinstance( obj, np.bool_ ):
            return bool( obj )
        if isinstance( obj, np.ndarray ):
            return obj.tolist()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, datetime ):
            return obj.isoformat()
        if isinstance(obj, date ):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def ensure_file_does_not_exist( filepath, delete=False ):
    """Check if a file exists.  Delete it, or raise an exception, if it does.

    Will always raise a FileExistsError if the file exists but isn't a normal file.

    Parameters
    ----------
    filepath: str or Path
       Path to the file
    delete: bool
       If True, will delete the file if it exists and is a regular
       file.  If False (default), will raise a FileExistsError
    """

    filepath = pathlib.Path( filepath )
    if filepath.exists():
        if not filepath.is_file():
            raise FileExistsError( f"{filepath} exists but is not a regular file" )
        if not delete:
            raise FileExistsError( f"{filepath} exists and delete is False" )
        else:
            filepath.unlink()


def listify( val, require_string=False ):
    """Return a list version of val.

    If val is None, return None.  If val is an iterable (but not a str
    or bytes), return list(val).  Otherwise, return [val].

    Parameters
    ----------
    require_string: bool (default False)
       If true, then val must either be a sequence of strings or a string

    Returns
    -------
    list or None

    """

    if val is None:
        return val

    if isinstance( val, collections.abc.Iterable ):
        if isinstance( val, str ):
            return [ val ]
        elif isinstance( val, bytes ):
            if require_string:
                raise TypeError( "listify wants a string, you gave it bytes.  (use decode?)" )
            return [ val ]
        else:
            if require_string and ( not all( [ isinstance( i, str ) for i in val ] ) ):
                raise TypeError( 'listify: all elements of passed sequence must be strings.' )
            return list( val )
    else:
        if require_string and ( not isinstance( val, str ) ):
            raise TypeError( f'listify wants a string, not a {type(val)}' )
        return [ val ]


def remove_empty_folders(path, remove_root=True):
    """Recursively remove any empty folders in the given path.

    Parameters
    ----------
    path: str or pathlib.Path
        The path to remove empty folders from.
    remove_root: bool
        If True, remove the root folder as well if it is empty.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        for subpath in path.iterdir():
            remove_empty_folders(subpath, remove_root=True)
        if remove_root and not any(path.iterdir()):
            path.rmdir()


def parse_dateobs(dateobs=None, output='datetime'):
    """Parse the dateobs, that can be a float, string, datetime or Time object.

    The output is datetime by default, but can be any of the above types.
    If the dateobs is None, the current time will be returned.
    If int or float, will assume MJD (or JD if bigger than 2400000).

    Parameters
    ----------
    dateobs: float, str, datetime, Time or None
        The dateobs to parse.
    output: str
        Choose one of the output formats:
        'datetime', 'Time', 'float', 'mjd', 'str'.

    Returns
    -------
    datetime, Time, float or str
    """
    if dateobs is None:
        dateobs = Time.now()
    elif isinstance(dateobs, (int, float)):
        if dateobs > 2400000:
            dateobs = Time(dateobs, format='jd')
        else:
            dateobs = Time(dateobs, format='mjd')
    elif isinstance(dateobs, str):
        if dateobs == 'now':
            dateobs = Time.now()
        else:
            # Not using astropy to parse the string here because it can't parse
            #   '2025-03-10T00:00:00-08:00', which is a legal
            #   ISO time!
            dateobs = Time( dateutil.parser.parse( dateobs ) )
    elif isinstance(dateobs, datetime):
        dateobs = Time(dateobs)
    elif isinstance(dateobs, date):
        dateobs = Time( datetime.combine( dateobs, datetime.min.time() ) )
    elif not isinstance(dateobs, Time):
        raise ValueError(f'Cannot parse dateobs of type {type(dateobs)}')

    if output == 'datetime':
        return dateobs.datetime
    elif output == 'Time':
        return dateobs
    elif output in ['float', 'mjd']:
        return dateobs.mjd
    elif output == 'str':
        return dateobs.isot
    else:
        raise ValueError(f'Unknown output type {output}')


def parse_bool(text):
    """Check if a string of text that represents a boolean value is True or False."""
    if text is None:
        return False
    if isinstance(text, bool):
        return text
    elif text.lower() in ['true', 'yes', '1']:
        return True
    elif text.lower() in ['false', 'no', '0']:
        return False
    else:
        raise ValueError(f'Cannot parse boolean value from "{text}"')


def as_UUID( val, canbenone=True ):
    """Convert a string or None to a uuid.UUID

    Parameters
    ----------
       val : uuid.UUID, str, or None
         The UUID to be converted.  Will throw a ValueError if val isn't
         properly formatted.

       canbenone : bool, default True
         If True, when val is None this function returns None.  If
         False, when val is None, when val is None this function returns
         uuid.UUID(''00000000-0000-0000-0000-000000000000').

    Returns
    -------
      uuid.UUID or None

    """

    if val is None:
        if canbenone:
            return None
        else:
            return uuid.UUID( '00000000-0000-0000-0000-000000000000' )
    if isinstance( val, uuid.UUID ):
        return val
    else:
        return uuid.UUID( val )


def as_datetime( string ):
    r"""Convert a string to datetime.date with some error checking, allowing a null op.

    Doesn't do anything to take care of timezone aware vs. timezone
    unaware dates.  It probably should.  Dealing with that is always a
    nightmare.

    Parmeters
    ---------
      string : str or datetime.datetime
         The string to convert.  If a datetime.datetime, the return
         value is just this.  If none or an empty string ("^\\s*$"), will
         return None.  Otherwise, must be a string that
         dateutil.parser.parse can handle.

    Returns
    -------
      datetime.datetime or None

    """

    if string is None:
        return None
    if isinstance( string, datetime ):
        return string
    if not isinstance( string, str ):
        raise TypeError( f'Error, must pass either a datetime or a string to asDateTime, not a {type(string)}' )
    string = string.strip()
    if len(string) == 0:
        return None
    try:
        dateval = dateutil.parser.parse( string )
        return dateval
    except Exception as e:
        if hasattr( e, 'message' ):
            SCLogger.error( f'Exception in asDateTime: {e.message}\n' )
        else:
            SCLogger.error( f'Exception in asDateTime: {e}\n' )
        raise ValueError( f'Error, {string} is not a valid date and time.' )


def env_as_bool(varname):
    """Parse an environmental variable as a boolean."""
    return parse_bool(os.getenv(varname))


def patch_image_overlap_limits( patchwid, x, y, imageshape ):
    """Do the annoying calculation of handling all the edge cases to figure out a patch and image overlap.

    Parameters
    ----------
      patchwid : int
        The size of the patch.  Must be odd.

      x, y : int
        The coordinates on the image that correspond to the center pixel of the patch

      imageshape : 2-element tuple of ints
        The shape of the image (ny, nx)

    Returns
    -------
       Two tuples of four integers: ( (px0, px1, py0, py1 ), ( ix0, ix1, iy0, iy1 ) )

       The first is are the limits on the patch that correspond to
       the limits on the image in the second tuple.  You probably want to do something like:

         image[ iy0:iy1, ix0:iy1 ] += patch[ py0:py1, px0:px1 ]

    """

    if any( [ not isinstance( i, numbers.Integral ) for i in [ patchwid, x, y ] ] ):
        raise TypeError( "patchwid, x, y must all be integers" )
    if ( ( not isinstance( imageshape, collections.abc.Sequence ) )
         or ( len(imageshape) !=2 )
         or ( any( [ not isinstance( i, numbers.Integral ) for i in imageshape ] ) )
        ):
        raise TypeError( "imageshape must be a 2-element tuple (or list) of integers" )

    if patchwid %2 == 0:
        raise ValueError( "Patchwid must be odd" )

    px0 = 0
    px1 = patchwid
    py0 = 0
    py1 = patchwid

    ix0 = x - patchwid // 2
    ix1 = ix0 + patchwid
    if ix0 < 0:
        px0 -= ix0
        ix0 = 0
    if ix1 > imageshape[1]:
        px1 -= ( ix1 - imageshape[1] )
        ix1 = imageshape[1]

    iy0 = y - patchwid // 2
    iy1 = iy0 + patchwid
    if iy0 < 0:
        py0 -= iy0
        iy0 = 0
    if iy1 > imageshape[0]:
        py1 -= ( iy1 - imageshape[0] )
        iy1 = imageshape[0]

    return ( (px0, px1, py0, py1), (ix0, ix1, iy0, iy1) )


def retry_with_sleep( func, sleepmin=0.1, sleept=0.5, sleepfac=2, sleepfuzz=0.1, sleepmax=32,
                      failmessage="to do the thing", exception_on_fail=True, retval_on_fail=None, randseed=None,
                      check_result=None, return_attr=None, good_returns=None, bad_returns=None,
                      badreturn_handler=None, accept_exceptions=Exception ):
    if not ( ( isinstance(accept_exceptions, type) and issubclass(accept_exceptions, Exception) ) or
             ( isinstance(accept_exceptions, tuple) and
               all( isinstance(i, type) and issubclass(i, Exception) for i in accept_exceptions ) )
            ):
        raise TypeError( "accept_exceptions must be a subclass of Exception, or a tuple of same" )

    failedatleastonce = False
    succeeded = False
    done = False
    rng = np.random.default_rng( randseed )
    t0 = time.monotonic()
    tries = 0
    while not done:
        try:
            tries += 1
            result = func()

            if check_result is not None:
                if not check_result( result ):
                    raise ValueError( "Unacceptable result.")

            if ( good_returns is not None ) or ( bad_returns is not None ):
                retval = result if return_attr is None else getattr( result, return_attr )
                try:
                    sretval = str(retval)
                except Exception:
                    sretval = ""
                if good_returns is not None:
                    if retval in good_returns:
                        done = True
                        succeeded = True
                    else:
                        if badreturn_handler is not None:
                            badreturn_handler( retval )
                        raise ValueError( f"Got return {sretval} {'that' if sretval=='' else 'which'} is not good" )
                elif retval in bad_returns:
                    if badreturn_handler is not None:
                        badreturn_handler( retval )
                    raise ValueError( f"Got bad return {sretval}" )

            else:
                done = True
                succeeded = True

            t1 = time.monotonic()

        except accept_exceptions as ex:
            t1 = time.monotonic()
            failedatleastonce = True
            if sleept > sleepmax:
                SCLogger.error( f"Repeated failures {failmessage} after {t1-t0:.2f}s and {tries} tries, giving up.  "
                                f"Last exception: {ex}" )
                done = True
            else:
                actualsleept = max( sleepmin, rng.normal(sleept, sleepfuzz * sleept) )
                SCLogger.warning( f"Failed {failmessage} after {tries} tries, "
                                  f"will sleep {actualsleept:.2f}s (nominally {sleept:.2f}s) and try again.  "
                                  f"Exception: {ex}" )
                time.sleep( actualsleept )
                sleept *= sleepfac

    if succeeded:
        if failedatleastonce:
            SCLogger.info( f"Succeeded {failmessage} after {t1-t0:.2f}s and {tries} tries." )
        return result

    else:
        if exception_on_fail:
            raise RuntimeError( f"Failed {failmessage} after {t1-t0:.2f}s and {tries} tries." )
        else:
            return retval_on_fail


def _reconstruct_commandline_canonical_option( action ):
    optstr = None
    if len( action.option_strings) > 0:
        for o in action.option_strings:
            # Try to keep the first one that starts with --
            optstr = o
            if ( len(o) > 1 ) and ( o[0:2] == '--' ):
                break
        if optstr is None:
            # by constructon
            raise RuntimeError( "This should never happen." )
        if re.search( r"[\s\'\"\\]", optstr ):
            raise ValueError( "Some of the option strings you sent to ArgumentParser.add_argument "
                              "have whitespace and/or quotes and/or backspaces in them!  "
                              "Why would you do that?!?!?" )
    return optstr


def reconstruct_commandline( argparser, args, executable=None,
                             envs=[], error_on_unknown_env=False,
                             showdefaults=False ):
    """Try to reconstruct a command line based on parsed args.

    Parameters
    ----------
      argparser: argparse.ArgumentParser
         The parser object that did the parsing.

      args: ...
         The thing that as returned by argparser.parse_args()

      executable: str, default None
         If not None, put this at the beginning of the returned command string

      envs: list, default []
         A list of key environment variables to output.
         SEECHANGE_CONFIG will always be prepended to this list (if it's
         defined).  Only those environment variables that actually exist
         in os.environ will be printed; set error_on_unknown_env=True if
         you want an exception raised if you ask for something that's
         not set.

      error_on_unknown_env: bool, default False
         If one of the environment variables passed in envs isn't
         actually defined (i.e. isn't in os.environ), normally it will
         be silently dropped from the output.  Set this parameter to
         True to trigger an exception instead.

      showdefaults: bool, default False
         Normally, if args.{option} is at the default value found in
         argparser, then {option} will not be included in the returned
         command string.  Set this to True to include everything.

    Returns
    -------
      str

        Intended for human consumption, not machine parsing.  Just log it.

    """

    def valprint( v ):
        if ( isinstance( v, str ) and '"' in v ):
            if "'" in v:
                raise RuntimeError( "This function is not sophisticated enough to cope with "
                                    "things specified on the command line that have both single "
                                    "and double quotes." )
            return f"'{v}'"
        else:
            return f'"{v}"'

    if ( 'SEECHANGE_CONFIG' not in envs ) and ( 'SEECHANGE_CONFIG' in os.environ ):
        envs.insert( 0, 'SEECHANGE_CONFIG' )

    strio = io.StringIO()
    if error_on_unknown_env and any( e not in os.environ for e in envs ):
        raise ValueError( f"You asked to print some env vars, but the following ones aren't defined: "
                          f"{set(envs) - set(os.environ.keys())}" )
    envs = [ e for e in envs if e in os.environ ]
    if len(envs) > 0:
        nlsp = "\n  "
        strio.write( f"\nKey environment variables:\n  "
                     f"{nlsp.join( f'{v}={valprint(os.getenv(v))}' for v in envs )}\n" )

    if executable is None:
        cmdline = ""
        spaces = "  "
    else:
        cmdline = executable
        spaces = "    "

    # I feel a little queasy using an underscore property of an object....
    for action in argparser._actions:
        if action.dest in args:
            val = getattr( args, action.dest )
            if ( ( val != action.default ) or
                    ( showdefaults and ( action.default is not None ) and
                        ( not isinstance( action.nargs, int ) or ( action.nargs > 0 ) )
                     )
                 ):
                if len(cmdline) > 0:
                    cmdline += f" \\\n{spaces}"
                optstr = _reconstruct_commandline_canonical_option( action )
                if optstr is not None:
                    cmdline += optstr
                    sp = " "
                else:
                    sp = ""

                if action.nargs in ( '*', '+' ):
                    for v in val:
                        cmdline += sp + valprint( v )
                        sp = " "
                elif action.nargs != 0:
                    cmdline += sp + valprint( val )

    if len( cmdline ) > 0:
        strio.write( f"\nCommand line:\n  {cmdline}\n" )

    return strio.getvalue()
