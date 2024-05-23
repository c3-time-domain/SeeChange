import pathlib
import collections.abc

import sys
import os
import pathlib
import git
from collections import defaultdict
import numpy as np
from datetime import datetime

import sqlalchemy as sa

from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord

from models.base import SmartSession, safe_mkdir

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
                raise TypeError( f'listify: all elements of passed sequence must be strings.' )
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

def get_git_hash():
    """
    Get the commit hash of the current git repo.

    If the environmental variable GITHUB_SHA is set,
    use that as the git commit hash.
    If not, try to find the git commit hash of the current repo.
    If all these methods fail, quietly return None.
    """

    git_hash = os.getenv('GITHUB_SHA')
    if git_hash is None:
        try:
            repo = git.Repo(search_parent_directories=True)
            git_hash = repo.head.object.hexsha
        except Exception:
            git_hash = None

    return git_hash

def get_latest_provenance(process_name, session=None):
    """
    Find the provenance object that fits the process_name
    that is the most recent.
    # TODO: we need to think about what "most recent" means.

    Parameters
    ----------
    process_name: str
        Name of the process that created this provenance object.
        Examples can include: "calibration", "subtraction", "source extraction" or just "level1".
    session: sqlalchemy.orm.session.Session
        Session to use to query the database.
        If not given, a new session will be created,
        and will be closed at the end of the function.

    Returns
    -------
    Provenance
        The most recent provenance object that matches the process_name.
        If not found, returns None.
    """
    # importing the models here to avoid circular imports
    from models.base import SmartSession
    from models.provenance import Provenance

    with SmartSession(session) as session:
        prov = session.scalars(
            sa.select(Provenance).where(
                Provenance.process == process_name
            ).order_by(Provenance.created_at.desc())
        ).first()

    return prov

def parse_dateobs(dateobs=None, output='datetime'):
    """
    Parse the dateobs, that can be a float, string, datetime or Time object.
    The output is datetime by default, but can be any of the above types.
    If the dateobs is None, the current time will be returned.
    If int or float, will assume MJD (or JD if bigger than 2400000).

    Parameters
    ----------
    dateobs: float, str, datetime, Time or None
        The dateobs to parse.
    output: str
        Choose one of the output formats:
        'datetime', 'Time', 'float', 'str'.

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
    elif output == 'float':
        return dateobs.mjd
    elif output == 'str':
        return dateobs.isot
    else:
        raise ValueError(f'Unknown output type {output}')

def parse_session(*args, **kwargs):
    """
    Parse the arguments and keyword arguments to find a SmartSession or SQLAlchemy session.
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

def read_fits_image(filename, ext=0, output='data'):
    """
    Read a standard FITS file's image data and header.
    Assumes this file doesn't have any complicated data structure.
    For more complicated data files, allow each Instrument subclass
    to use its own methods instead of this.

    Parameters
    ----------
    filename: str
        The full path to the file.
    ext: int or str
        The extension number or name to read.
        For files with a single extension,
        the default 0 is fine.
    output: str
        Choose which part of the file to read:
        'data', 'header', 'both'.

    Returns
    -------
    The return value depends on the value of output:
    data: np.ndarray (optional)
        The image data.
    header: astropy.io.fits.Header (optional)
        The header of the image.
    If both are requested, will return a tuple (data, header)
    """
    with fits.open(filename, memmap=False) as hdul:
        if output in ['data', 'both']:
            data = hdul[ext].data
            # astropy will read FITS files as big-endian
            # But, the sep library depends on native byte ordering
            # So, swap if necessary
            if not data.dtype.isnative:
                data = data.astype( data.dtype.name ) 

        if output in ['header', 'both']:
            header = hdul[ext].header

    if output == 'data':
        return data
    elif output == 'header':
        return header
    elif output == 'both':
        return data, header
    else:
        raise ValueError(f'Unknown output type "{output}", use "data", "header" or "both"')

def save_fits_image_file(filename, data, header, extname=None, overwrite=True, single_file=False,
                         just_update_header=False):
    """Save a single dataset (image data, weight, flags, etc) to a FITS file.

    The header should be the raw header, with some possible adjustments,
    including all the information (not just the minimal subset saved into the DB).

    When single_file=False (default) the extname parameter will be appended to the filename,
    and the file will contain only the primary HDU.

    Using the single_file=True option will save each data array into a separate extension
    of the same file, instead of saving each data array into a separate file.
    In this case the extname will be used to name the FITS extension.

    In all cases, if the final filename does not end with .fits,
    that will be appended at the end of it.

    Parameters
    ----------
    filename: str
        The full path to the file.

    data: np.ndarray
        The image data. Can also supply the weight, flags, etc.

    header: dict or astropy.io.fits.Header
        The header of the image.

    extname: str, default None
        The name of the extension to save the data into.
        If writing individual files (default) will just
        append this string to the filename.
        If writing a single file, will use this as the extension name.

    overwrite: bool, default True
        Whether to overwrite the file if it already exists.

    single_file: bool, default False
        Whether to save each data array into a separate extension of the same file.
        if False (default) will save each data array into a separate file.
        If True, will use the extname to name the FITS extension, and save
        each array into the same file.

    just_update_header: bool, default False
       Ignored if single_file is True.  Otherwise, if this is True, and
       the file to be written exists, it will be opened in "update" mode
       and just the header will be rewritten, rather than the entire
       image.  (I'm not 100% sure that astropy will really do this
       right, thereby saving most of the disk I/O, but the existence of
       "update" mode in astropy.io.fits.open() suggests it does, so
       we'll use it.)  This option implies overwrite, so will overwrite
       the file header if overwrite is False.

    Returns
    -------
    The path to the file saved (or written to)

    """

    hdu = fits.ImageHDU( data, name=extname ) if single_file else fits.PrimaryHDU( data )

    if isinstance( header, fits.Header ):
        hdu.header.extend( header )
    else:
        for k, v in header.items():
            hdu.header[k] = v

    if single_file:
        if not filename.endswith('.fits'):
            filename += '.fits'
        safe_mkdir(os.path.dirname(filename))
        with fits.open(filename, memmap=False, mode='append') as hdul:
            if len(hdul) == 0:
                hdul.append( fits.PrimaryHDU() )
            hdul.append(hdu)
        return filename

    else:  # multiple files
        hdul = fits.HDUList([hdu])

        full_name = filename if extname is None else filename + '.' + extname
        if not full_name.endswith('.fits'):
            full_name += '.fits'
        full_name = pathlib.Path( full_name )
        safe_mkdir( str( full_name.parent ) )

        if just_update_header and full_name.is_file():
            with fits.open( full_name, mode="update" ) as filehdu:
                filehdu[0].header = hdul[0].header
        else:
            hdul.writeto(full_name, overwrite=overwrite)

        return str( full_name )


def parse_bool(text):
    """Check if a string of text that represents a boolean value is True or False."""
    if text is None:
        return False
    elif text.lower() in ['true', 'yes', '1']:
        return True
    elif text.lower() in ['false', 'no', '0']:
        return False
    else:
        raise ValueError(f'Cannot parse boolean value from "{text}"')
