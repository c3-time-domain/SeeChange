import collections.abc
import numbers
import os
import pathlib
from datetime import datetime
import dateutil.parser
import uuid
import json

import numpy as np
import sqlalchemy as sa

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

    If val is already a sequence other than a string, return list(val).
    Otherwise, return [val].  If val is None, return None.

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

    if isinstance( val, collections.abc.Sequence ):
        if isinstance( val, str ):
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
            dateobs = Time(dateobs)
    elif isinstance(dateobs, datetime):
        dateobs = Time(dateobs)
    else:
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


def parse_session(*args, **kwargs):
    """Parse the arguments and keyword arguments to find a SmartSession or SQLAlchemy session.

    If one of the kwargs is called "session" that value will be returned.
    Otherwise, if any of the unnamed arguments is a session, the last one will be returned.
    If neither of those are found, None will be returned.
    Will also return the args and kwargs with any sessions removed.

    Parameters
    ----------
    args: list
        List of unnamed arguments
    kwargs: dict
        Dictionary of named arguments

    Returns
    -------
    args: list
        List of unnamed arguments with any sessions removed.
    kwargs: dict
        Dictionary of named arguments with any sessions removed.
    session: SmartSession or SQLAlchemy session or None
        The session found in the arguments or kwargs.
    """
    session = None
    sessions = [arg for arg in args if isinstance(arg, sa.orm.session.Session)]
    if len(sessions) > 0:
        session = sessions[-1]
    args = [arg for arg in args if not isinstance(arg, sa.orm.session.Session)]

    sesskeys = []
    for key in kwargs.keys():
        if key in ['session']:
            if not isinstance(kwargs[key], sa.orm.session.Session):
                raise ValueError(f'Session must be a sqlalchemy.orm.session.Session, got {type(kwargs[key])}')
            sesskeys.append(key)
    for key in sesskeys:
        session = kwargs.pop(key)

    return args, kwargs, session


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


# from: https://stackoverflow.com/a/5883218
def get_inheritors(klass):
    """Get all classes that inherit from klass. """
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


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
