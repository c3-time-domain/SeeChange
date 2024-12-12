# DO NOT USE THESE OUTSIDE OF TESTS IN tests/
#
# (The cache has some scariness to it, and we don't want it built into
# the mainstream pipeline.  It's used in test fixtures, and should only
# be used there.)

import shutil
import json
import pathlib

import sqlalchemy as sa

from models.base import FileOnDiskMixin
from util.logger import SCLogger
from util.util import UUIDJsonEncoder, asUUID


def copy_to_cache(FoD, cache_dir, filepath=None, dont_actually_copy_just_return_json_filepath=False ):
    """Save a copy of the object (and, potentially, associated files) into a cache directory.

    If the object is a FileOnDiskMixin, then the file(s) pointed by
    get_fullpath() will be copied to the cache directory with their
    original names, unless filepath is specified, in which case the
    cached files will have a different name than the files in the data
    folder (and the database filepath).  The object's filepath attribute
    will be used to create a JSON file which holds the object's column
    attributes (i.e., only those that are database persistent).

    If caching a non-FileOnDiskMixin object, the filepath argument must
    be given, because it is used to name the JSON file.

    The object must implement a method to_json(path) that serializes
    itself to the file specified by path.

    Parameters
    ----------
    FoD: FileOnDiskMixin or another object that implements to_json()
        The object to cache.

    cache_dir: str or Path
        The path to the cache directory.

    filepath: str or Path (optional)
        Must be given if the FoD is None.
        If it is a FileOnDiskMixin, it will be used to name
        the data files and the JSON file in the cache folder.

    Returns
    -------
    pathlib.Path
        The full path to the output json file.

    """

    cache_dir = pathlib.Path( cache_dir )
    if filepath is not None:
        filepath = pathlib.Path( filepath )
        if filepath.name.endswith( '.json' ):
            jsonpath = cache_dir / filepath
        else:
            jsonpath = cache_dir / filepath.parent / f"{filepath.name}.json"


    if filepath is None:
        if not isinstance(FoD, FileOnDiskMixin):
            raise ValueError("filepath must be given when caching a non-FileOnDiskMixin object")
        else:
            filepath = pathlib.Path( FoD.filepath )
            jsonpath = cache_dir / filepath.parent / f"{filepath.name}.json"

    if dont_actually_copy_just_return_json_filepath:
        return jsonpath

    # Now actually do the saving.
    if isinstance(FoD, FileOnDiskMixin):
        for i, paths in enumerate( zip( FoD.get_relpath(as_list=True), FoD.get_fullpath(as_list=True) ) ):
            relpath, fullpath = paths
            if fullpath is None:
                continue
            fullpath = pathlib.Path( fullpath )
            if fullpath.is_symlink():
                # ...is this actually a problem?  Maybe not.
                raise RuntimeError( f"Trying to copy a simlink to cache: {fullpath}" )
            cachepath = cache_dir / relpath
            SCLogger.debug(f"Copying {fullpath} to {cachepath}")
            cachepath.parent.mkdir( exist_ok=True, parents=True )
            shutil.copy2( fullpath, cachepath )

    jsonpath.parent.mkdir( exist_ok=True, parents=True )
    FoD.to_json( jsonpath )

    return jsonpath


def copy_list_to_cache(obj_list, cache_dir, filepath=None):
    """Copy a correlated list of objects to the cache directory.

    All objects must be of the same type.  The objects implement the
    to_dict() method that serializes themselves to a dictionary (as
    FileOnDiskMixin does).  All of the objects' dictionary data will be
    written to filepath as a JSON list of dictionaries.  (Use case:
    something like Measurements where you expect to generate a whole
    bunch of things together; this lets us write one big (well,
    less-small) file instead of a whole bunch of small files.

    If filepath is None, then all objects in the list must have the same
    "filepath" attribute.

    Parameters
    ----------
    obj_list: list
        A list of objects to save to the cache directory.

    cache_dir: str or path
        The path to the cache directory.

    filepath: str or path (optional)
        Must be given if the objects are not FileOnDiskMixin.
        If it is a FileOnDiskMixin, it will be used to name
        the data files and the JSON file in the cache folder.
        → Currently not optional, as this function doesn't
          actually work with FileOnDiskMixin objects!

    Returns
    -------
    str
        The full path to the output JSON file.

    """

    cache_dir = pathlib.Path( cache_dir )
    filepath = pathlib.Path( filepath )
    if filepath.name.endswith( '.json' ):
        jsonpath = cache_dir / filepath
    else:
        jsonpath = cache_dir / filepath.parent / f"{filepath.name}.json"

    if len(obj_list) > 0:
        if isinstance(obj_list[0], FileOnDiskMixin):
            # See comment in copy_list_from_cache
            raise NotImplementedError( "copy_list_from_cache doesn't work with FileOnDiskMixin objects." )

        types = set( type(obj) for obj in obj_list )
        if len(types) != 1:
            raise ValueError("All objects must be of the same type!")

        # This next bit is commented out since it only really makes
        #   sense when dealing with FileOnDisk objects, which we currently
        #   don't support.  (At some point, we may decide to never
        #   support them for lists (probably a good idea), at which point
        #   we can just remove this block.)
        # filepaths = set( getattr(obj, 'filepath', None) for obj in obj_list )
        # if len(filepaths) != 1:
        #     raise ValueError("All objects must have the same filepath!")

        # save the JSON file and copy associated files
        # filepath = filepath is filepath is not None else filepaths[0]
        # json_filepath = copy_to_cache( obj_list[0], cache_dir, filepath=filepath,
        #                                dont_actually_copy_just_return_json_filepath=True )

    # overwrite the JSON file with the list of dictionaries for all the objects
    with open(jsonpath, 'w') as fp:
        json.dump( [obj.to_dict() for obj in obj_list], fp, indent=2, cls=UUIDJsonEncoder )

    return jsonpath


def realize_column_uuids( obj ):
    """Make sure that all UUID columns that aren't none have type uuid.UUID."""
    for col in sa.inspect( obj ).mapper.columns:
        if ( isinstance( col.type, sa.sql.sqltypes.UUID ) ) and ( getattr( obj, col.key ) is not None ):
            setattr( obj, col.key, asUUID( getattr( obj, col.key ) ) )


def copy_from_cache( cls, cache_dir, filepath, add_to_dict=None, symlink=False ):
    """Copy and reconstruct an object from the cache directory.

    Will need the JSON file that contains all the column attributes of a
    database object.  The object must implement the from_dict class
    method.  Once the object is successfully loaded, and if the object
    is a FileOnDiskMixin, it will be able to figure out where all the
    associated files are saved based on the filepath and components in
    the JSON file.  Those files will be copied into the current data
    directory (i.e., that pointed to by FileOnDiskMixin.local_path).

    Database records are restored exactly as they were saved in the
    cache.  This includes any foreign keys to other tables of the
    database.  This could potentially lead to conflicts if different
    things saved to the cache weren't saved consistently (i.e. with
    cross-references in place), or if the objects are not restored in
    the right order (i.e. something referred to now isn't already
    restored to the database).

    Parameters
    ----------
    cls : Class that derives from FileOnDiskMixin, or that implements from_dict(dict)
        The class of the object that's being copied

    cache_dir: str or path
        The path to the cache directory.

    filepath: str or path
        The name of the JSON file that holds the column attributes.
        Must be underneath cache_dir.  May either be the full absolute
        path (i.e. including cache_dir), or relative to cache_dir.

    add_to_dict: dict (optional)
        Additional parameters to add to the dictionary pulled from the
        cache.  Add things here that aren't saved to the cache but that
        are necessary in order to instantiate the object.  Things here will
        also override anything read from the cache.

    symlink : bool, default False
        Only relevant if copying an object that has stored files
        (i.e. something that is a FileOnDiskMixin).  If True, instead
        of copying, will make a symbolic link for all of those stored
        files rather than fully copying them from the cache.

    Returns
    -------
    output: SeeChangeBase
        The reconstructed object, of the same type as the class.

    """

    filepath = pathlib.Path( filepath )
    cache_dir = pathlib.Path( cache_dir ).resolve()

    # Make filepath relative to cache_dir (making sure it's there if filepath is absolute)
    if filepath.is_relative_to( cache_dir ):
        filepath = filepath.relative_to( cache_dir )
    else:
        if filepath.is_absolute():
            raise ValueError( f"filepath must be relative to cache_dir, but it's not: "
                              f"filepath={filepath}, cache_dir={cache_dir}" )

    # allow the user to give the filepath with or without the .json extension
    if not filepath.name.endswith('.json'):
        filepath = filepath.parent / f'{filepath.name}.json'

    # read the json file, restore the object
    with open( cache_dir / filepath, 'r' ) as fp:
        json_dict = json.load(fp)
    if add_to_dict is not None:
        json_dict.update( add_to_dict )
    output = cls.from_dict(json_dict)
    # Make sure UUIDs are UUIDs; they will have been read as strings from the JSON file
    realize_column_uuids( output )

    # copy any associated files
    if isinstance(output, FileOnDiskMixin):
        for i, paths in enumerate( zip( output.get_relpath(as_list=True), output.get_fullpath(as_list=True) ) ):
            relpath, fullpath = paths
            if fullpath is None:
                continue
            fullpath = pathlib.Path( fullpath )
            cachepath = cache_dir / relpath
            SCLogger.debug(f"Copying {cachepath} to {fullpath}")
            fullpath.parent.mkdir( exist_ok=True, parents=True )
            if symlink:
                if fullpath.exists():
                    fullpath.unlink()
                fullpath.symlink_to( cachepath )
            else:
                shutil.copyfile( cachepath, fullpath )

    return output


def copy_list_from_cache(cls, cache_dir, filepath):
    """Copy and reconstruct a list of objects from the cache directory.

    Will need the JSON file that contains all the column attributes of the file.
    Once those are successfully loaded, and if the object is a FileOnDiskMixin,
    it will be able to figure out where all the associated files are saved
    based on the filepath and extensions in the JSON file.

    Parameters
    ----------
    cls: Class that derives from FileOnDiskMixin, or that implements from_dict(dict)
        The class of the objects that are being copied

    cache_dir: str or Path
        The path to the cache directory.

    filepath: str or Path
        The name of the JSON file that holds the column attributes.
        Must be underneath cache_dir.  May be the full absolute path
        (i.e. including cache_dir), or relative to cache_dir.

    Returns
    -------
    output: list
        The list of reconstructed objects, of type cls.

    """

    if isinstance(cls, FileOnDiskMixin):
        # The use case of this function has evolved since it was
        #   written.  Originally, we had a whole bunch of Cutouts object
        #   that all shared one file.  We needed to reconstruct the
        #   database objects from the list, but then restore just the
        #   one file.  Now, the Cutouts object refers to the whole file,
        #   so we're back to the cleaner case of each database entry
        #   refers to its own files, and doesn't share files with other
        #   database entries.  That means that restoring a list from
        #   cache is less useful for things with files.  (Really, the
        #   whole cache list is just used for Measurements and
        #   DeepScores now.)
        raise NotImplementedError( "copy_list_from_cache doesn't work with FileOnDiskMixin objects." )

    filepath = pathlib.Path( filepath )
    cache_dir = pathlib.Path( cache_dir ).resolve()

    # Make filepath relative to cache_dir (making sure it's there if filepath is absolute)
    if filepath.is_relative_to( cache_dir ):
        filepath = filepath.relative_to( cache_dir )
    else:
        if filepath.is_absolute():
            raise ValueError( f"filepath must be relative to cache_dir, but it's not: "
                              f"filepath={filepath}, cache_dir={cache_dir}" )

    # allow the user to give the filepath with or without the .json extension
    if not filepath.name.endswith('.json'):
        filepath = filepath.parent / f'{filepath.name}.json'

    with open( cache_dir / filepath, 'r') as fp:
        json_list = json.load(fp)

    output = []
    for obj_dict in json_list:
        newobj = cls.from_dict( obj_dict )
        output.append( newobj )

    return output
