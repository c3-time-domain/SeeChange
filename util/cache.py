import os
import shutil
import json

from models.base import FileOnDiskMixin
from util.logger import SCLogger

# ======================================================================
# Functions for copying FileOnDisk objects to/from cache


def copy_to_cache(FoD, cache_dir, filepath=None):
    """Save a copy of the object (and, potentially, associated files) into a cache directory.

    If the object is a FileOnDiskMixin, then the file(s) pointed by get_fullpath()
    will be copied to the cache directory with their original names,
    unless filepath is specified, in which case the cached files will
    have a different name than the files in the data folder (and the database filepath).
    The filepath (with optional first extension) will be used to create a JSON file
    which holds the object's column attributes (i.e., only those that are
    database persistent).

    If caching a non-FileOnDiskMixin object, the filepath argument must
    be given, because it is used to name the JSON file.  The object must
    implement a method to_json(path) that serializes itself to the file
    specified by path.

    Parameters
    ----------
    FoD: FileOnDiskMixin or another object that implements to_json()
        The object to cache.
    cache_dir: str or path
        The path to the cache directory.
    filepath: str or path (optional)
        Must be given if the FoD is None.
        If it is a FileOnDiskMixin, it will be used to name
        the data files and the JSON file in the cache folder.

    Returns
    -------
    str
        The full path to the output json file.

    """
    if filepath is not None and filepath.endswith('.json'):  # remove .json if it exists
        filepath = filepath[:-5]

    json_filepath = filepath
    if not isinstance(FoD, FileOnDiskMixin):
        if filepath is None:
            raise ValueError("filepath must be given when caching a non FileOnDiskMixin object")

    else:  # it is a FileOnDiskMixin
        if filepath is None:  # use the FileOnDiskMixin filepath as default
            filepath = FoD.filepath  # use this filepath for the data files
            json_filepath = FoD.filepath  # use the same filepath for the json file too
        if (
                FoD.filepath_extensions is not None and
                len(FoD.filepath_extensions) > 0 and
                not json_filepath.endswith(FoD.filepath_extensions[0])
        ):
                json_filepath += FoD.filepath_extensions[0]  # only append this extension to the json filename

        for i, source_f in enumerate(FoD.get_fullpath(as_list=True)):
            if source_f is None:
                continue
            target_f = os.path.join(cache_dir, filepath)
            if FoD.filepath_extensions is not None and i < len(FoD.filepath_extensions):
                target_f += FoD.filepath_extensions[i]
            SCLogger.debug(f"Copying {source_f} to {target_f}")
            os.makedirs(os.path.dirname(target_f), exist_ok=True)
            shutil.copy2(source_f, target_f)

    # attach the cache_dir and the .json extension if needed
    json_filepath = os.path.join(cache_dir, json_filepath)
    os.makedirs( os.path.dirname( json_filepath ), exist_ok=True )
    if not json_filepath.endswith('.json'):
        json_filepath += '.json'
    FoD.to_json(json_filepath)

    return json_filepath


def copy_list_to_cache(obj_list, cache_dir, filepath=None):
    """Copy a correlated list of objects to the cache directory.

    All objects must be of the same type.  If they are of type
    FileOnDiskMixin, the files associated with the *first* object in the
    list (only) will be copied to the cache.  (Use case: something like
    Cutouts where a whole bunch of objects all have the same file.)

    The objects implement the to_dict() method that serializes
    themselves to a dictionary (as FileOnDiskMixin does).  All of the
    objects' dictionary data will be written to filepath as a JSON list
    of dictionaries.

    In either case, if the objects have a "filepath" attribute, the
    value of that attribute must be the same for every object in the
    list.

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

    Returns
    -------
    str
        The full path to the output JSON file.

    """
    if len(obj_list) == 0:
        if filepath is None:
            return  # can't do anything without a filepath
        json_filepath = os.path.join(cache_dir, filepath)
        if not json_filepath.endswith('.json'):
            json_filepath += '.json'
    else:
        types = set([type(obj) for obj in obj_list])
        if len(types) != 1:
            raise ValueError("All objects must be of the same type!")

        filepaths = set([getattr(obj, 'filepath', None) for obj in obj_list])
        if len(filepaths) != 1:
            raise ValueError("All objects must have the same filepath!")

        # save the JSON file and copy associated files
        json_filepath = copy_to_cache(obj_list[0], cache_dir, filepath=filepath)

    # overwrite the JSON file with the list of dictionaries
    with open(json_filepath, 'w') as fp:
        json.dump([obj.to_dict() for obj in obj_list], fp, indent=2)

    return json_filepath


def copy_from_cache(cls, cache_dir, filepath):
    """Copy and reconstruct an object from the cache directory.

    Will need the JSON file that contains all the column attributes of the file.
    Once those are successfully loaded, and if the object is a FileOnDiskMixin,
    it will be able to figure out where all the associated files are saved
    based on the filepath and extensions in the JSON file.
    Those files will be copied into the current data directory
    (i.e., that pointed to by FileOnDiskMixin.local_path).
    The reconstructed object should be correctly associated
    with its files but will not necessarily have the correct
    relationships to other objects.

    Parameters
    ----------
    cls : Class that derives from FileOnDiskMixin, or that implements from_dict(dict)
        The class of the object that's being copied
    cache_dir: str or path
        The path to the cache directory.
    filepath: str or path
        The name of the JSON file that holds the column attributes.

    Returns
    -------
    output: SeeChangeBase
        The reconstructed object, of the same type as the class.
    """
    # allow user to give an absolute path, so long as it is in the cache dir
    if filepath.startswith(cache_dir):
        filepath = filepath[len(cache_dir) + 1:]

    # allow the user to give the filepath with or without the .json extension
    if filepath.endswith('.json'):
        filepath = filepath[:-5]

    full_path = os.path.join(cache_dir, filepath)
    with open(full_path + '.json', 'r') as fp:
        json_dict = json.load(fp)

    output = cls.from_dict(json_dict)

    # copy any associated files
    if isinstance(output, FileOnDiskMixin):
        # if fullpath ends in filepath_extensions[0]
        if (
                output.filepath_extensions is not None and
                output.filepath_extensions[0] is not None and
                full_path.endswith(output.filepath_extensions[0])
        ):
            full_path = full_path[:-len(output.filepath_extensions[0])]

        for i, target_f in enumerate(output.get_fullpath(as_list=True)):
            if target_f is None:
                continue
            source_f = os.path.join(cache_dir, full_path)
            if output.filepath_extensions is not None and i < len(output.filepath_extensions):
                source_f += output.filepath_extensions[i]
            SCLogger.debug(f"Copying {source_f} to {target_f}")
            os.makedirs(os.path.dirname(target_f), exist_ok=True)
            shutil.copyfile(source_f, target_f)

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
    cache_dir: str or path
        The path to the cache directory.
    filepath: str or path
        The name of the JSON file that holds the column attributes.

    Returns
    -------
    output: list
        The list of reconstructed objects, of the same type as the class.
    """
    # allow user to give an absolute path, so long as it is in the cache dir
    if filepath.startswith(cache_dir):
        filepath = filepath[len(cache_dir) + 1:]

    # allow the user to give the filepath with or without the .json extension
    if filepath.endswith('.json'):
        filepath = filepath[:-5]

    full_path = os.path.join(cache_dir, filepath)
    with open(full_path + '.json', 'r') as fp:
        json_list = json.load(fp)

    output = []
    for obj_dict in json_list:
        output.append(cls.from_dict(obj_dict))

    if len(output) == 0:
        return []

    if isinstance(output[0], FileOnDiskMixin):
        # if fullpath ends in filepath_extensions[0]
        if (
                output[0].filepath_extensions is not None and
                output[0].filepath_extensions[0] is not None and
                full_path.endswith(output[0].filepath_extensions[0])
        ):
            full_path = full_path[:-len(output[0].filepath_extensions[0])]

        for i, target_f in enumerate(output[0].get_fullpath(as_list=True)):
            if target_f is None:
                continue
            source_f = os.path.join(cache_dir, full_path)
            if output[0].filepath_extensions is not None and i < len(output[0].filepath_extensions):
                source_f += output[0].filepath_extensions[i]
            SCLogger.debug(f"Copying {source_f} to {target_f}")
            os.makedirs(os.path.dirname(target_f), exist_ok=True)
            shutil.copyfile(source_f, target_f)

    return output


# ======================================================================

