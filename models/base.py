import sys
import os
import inspect
import hashlib
import pathlib
import shutil
import logging
from uuid import UUID

from contextlib import contextmanager

from astropy.coordinates import SkyCoord

import sqlalchemy as sa
from sqlalchemy import func, orm
from sqlalchemy.types import Enum

from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy.dialects.postgresql import UUID as sqlUUID

import util.config as config
from util.archive import Archive

utcnow = func.timezone("UTC", func.current_timestamp())

_logger = logging.getLogger("main")
if len(_logger.handlers) == 0:
    _logout = logging.StreamHandler( sys.stdout )
    _logger.addHandler( _logout )
    _formatter = logging.Formatter( f"[%(asctime)s - %(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S" )
    _logout.setFormatter( _formatter )
    _logout.setLevel( logging.INFO )

# this is the root SeeChange folder
CODE_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

_engine = None
_Session = None


def Session():
    """
    Make a session if it doesn't already exist.
    Use this in interactive sessions where you don't
    want to open the session as a context manager.
    If you want to use it in a context manager
    (the "with" statement where it closes at the
    end of the context) use SmartSession() instead.

    Returns
    -------
    sqlalchemy.orm.session.Session
        A session object that doesn't automatically close.
    """
    global _Session, _engine

    if _Session is None:
        cfg = config.Config.get()
        url = (f'{cfg.value("db.engine")}://{cfg.value("db.user")}:{cfg.value("db.password")}'
               f'@{cfg.value("db.host")}:{cfg.value("db.port")}/{cfg.value("db.database")}')
        _engine = sa.create_engine(url, future=True, poolclass=sa.pool.NullPool)

        _Session = sessionmaker(bind=_engine, expire_on_commit=False)

    session = _Session()

    return session


@contextmanager
def SmartSession(input_session=None):
    """
    Return a Session() instance that may or may not
    be inside a context manager.

    If the input is already a session, just return that.
    If the input is None, create a session that would
    close at the end of the life of the calling scope.
    """
    global _Session, _engine

    # open a new session and close it when outer scope is done
    if input_session is None:

        with Session() as session:
            yield session

    # return the input session with the same scope as given
    elif isinstance(input_session, sa.orm.session.Session):
        yield input_session

    # wrong input type
    else:
        raise TypeError(
            "input_session must be a sqlalchemy session or None"
        )


def safe_merge(session, obj):
    """
    Only merge the object if it has a valid ID,
    and if it does not exist on the session.
    Otherwise, return the object itself.

    Parameters
    ----------
    session: sqlalchemy.orm.session.Session
        The session to use for the merge.
    obj: SeeChangeBase
        The object to merge.

    Returns
    -------
    obj: SeeChangeBase
        The merged object, or the unmerged object
        if it is already on the session or if it
        doesn't have an ID.
    """
    if obj is None:
        return None

    if obj.id is None:
        return obj

    if obj in session:
        return obj

    return session.merge(obj)


class SeeChangeBase:
    """Base class for all SeeChange classes."""

    id = sa.Column(
        sa.BigInteger,
        primary_key=True,
        index=True,
        autoincrement=True,
        doc="Unique identifier for this dataset",
    )

    created_at = sa.Column(
        sa.DateTime,
        nullable=False,
        default=utcnow,
        index=True,
        doc="UTC time of insertion of object's row into the database.",
    )

    modified = sa.Column(
        sa.DateTime,
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
        doc="UTC time the object's row was last modified in the database.",
    )

    type_annotation_map = { UUID: sqlUUID }
    
    def __init__(self, **kwargs):
        self.from_db = False  # let users know this object was newly created
        for k, v in kwargs.items():
            setattr(self, k, v)

    @orm.reconstructor
    def init_on_load(self):
        self.from_db = True  # let users know this object was loaded from the database

    def get_attribute_list(self):
        """
        Get a list of all attributes of this object,
        not including internal SQLAlchemy attributes,
        and database level attributes like id, created_at, etc.
        """
        attrs = [
            a for a in self.__dict__.keys()
            if (
                a not in ['_sa_instance_state', 'id', 'created_at', 'modified', 'from_db']
                and not callable(getattr(self, a))
                and not isinstance(getattr(self, a), (
                    orm.collections.InstrumentedList, orm.collections.InstrumentedDict
                ))
            )
        ]

        return attrs

    def recursive_merge(self, session, done_list=None):
        """
        Recursively merge (using safe_merge) all the objects,
        the parent objects (image, ref_image, new_image, etc.)
        and the provenances of all of these, into the given session.

        Parameters
        ----------
        session: sqlalchemy.orm.session.Session
            The session to use for the merge.
        done_list: list (optional)
            A list of objects that have already been merged.

        Returns
        -------
        SeeChangeBase
            The merged object.
        """
        if done_list is None:
            done_list = set()

        if self in done_list:
            return self

        obj = safe_merge(session, self)
        done_list.add(obj)

        # only do the sub-properties if the object was already added to the session
        attributes = ['provenance', 'exposure', 'image', 'ref_image', 'new_image', 'sub_image', 'source_list']

        # recursively call this on the provenance and other parent objects
        for att in attributes:
            try:
                sub_obj = getattr(self, att, None)
                # go over lists:
                if isinstance(sub_obj, list):
                    setattr(obj, att, [o.recursive_merge(session, done_list=done_list) for o in sub_obj])

                if isinstance(sub_obj, SeeChangeBase):
                    setattr(obj, att, sub_obj.recursive_merge(session, done_list=done_list))

            except DetachedInstanceError:
                pass

        return obj


Base = declarative_base(cls=SeeChangeBase)


class FileOnDiskMixin:
    """Mixin for objects that refer to files on disk.

    Files are assumed to live on the local disk (underneath the
    configured path.data_root), and optionally on a archive server
    (configured through the subproperties of "archive" in the yaml
    config file).  The property filepath is the path relative to the
    root in both cases.

    If there is a single file associated with this entry, then filepath
    has the name of that file.  md5sum holds a checksum for the file,
    *if* it has been correctly saved to the archive.  If the file has
    not been saved to the archive, md5sum is null.  In this case,
    filepath_extensions and md5sum_extensions will be null.

    If there are multiple files associated with this entry, then
    filepath is the beginning of the names of all the files.  Each entry
    in filepath_extensions is then appended to filepath to get the
    actual path of the file.  For example, if an image file has the
    image itself, an associated weight, and an associated mask, then
    filepath might be "image" and filepath_extensions might be
    [".fits.fz", ".mask.fits.fz", ".weight.fits.fz"] to indicate that
    the three files image.fits.fz, image.mask.fits.fz, and
    image.weight.fz are all associated with this entry.  When
    filepath_extensions is non-null, md5sum should be null, and
    md5sum_extensions is an array with the same length as
    filepath_extensions.  (For extension files that have not yet been
    saved to the archive, that element of the md5sum_etensions array is
    null.)

    Saving data:

    Any object that implements this mixin must call this class' "save"
    method in order to save the data to disk.  (This may be through
    super() if the subclass has to do custom things.)  The save method
    of this class will save the to the local filestore (undreneath
    path.data_root), and also save it to the archive.  Once a file is
    saved on the archive, the md5sum (or md5sum_extensions) field in the
    database record is updated.  (If the file has not been saved to the
    archive, then the md5sum and md5sum_extensions fields will be null.)

    Loading data:

    When calling get_fullpath(), the object will first check if the file
    exists locally, and then it will import it from archive if missing
    (and if archive is defined).  If you want to avoid downloading, use
    get_fullpath(download=False) or get_fullpath(nofile=True).  (The
    latter case won't even try to find the file on the lcoal disk, it
    will just tell you what the path should be.)  If you want to always
    get a list of filepaths (even if filepath_extensions=None) use
    get_fullpath(as_list=True).  If the file is missing locally, and
    downloading cannot proceed (because no archive is defined, or
    because the download=False flag is used, or because the file is
    missing from server), then the call to get_fullpath() will raise an
    exception (unless you use download=False or nofile=True).

    After all the pulling from the archive is done and the file(s) exist
    locally, the full (absolute) path to the local file is returned.  It
    is then up to the inheriting object (e.g., the Exposure or Image) to
    actually load the file from disk and figure out what to do with the
    data.

    The path to the local file store and the archive object are saved in
    class variables "local_path" and "archive" that are initialized from
    the config system the first time the class is loaded.

    """

    cfg = config.Config.get()
    local_path = cfg.value('path.data_root', None)
    if local_path is None:
        local_path = cfg.value('path.data_temp', None)
    if local_path is None:
        local_path = os.path.join(CODE_ROOT, 'data')
    if not os.path.isdir(local_path):
        os.makedirs(local_path, exist_ok=True)

    archive = cfg.value('archive', None)
    if archive is not None:
        archive = Archive( archive_url=cfg.value('archive.url'),
                           path_base=cfg.value('archive.path_base'),
                           token=cfg.value('archive.token'),
                           verify_cert=cfg.value('archive.verify_cert'),
                           local_read_dir=cfg.value('archive.read_dir'),
                           local_write_dir=cfg.value('archive.write_dir') )
        
    @classmethod
    def safe_mkdir(cls, path):
        if path is None or path == '':
            return  # ignore empty paths, we don't need to make them!
        cfg = config.Config.get()

        allowed_dirs = []
        if cls.local_path is not None:
            allowed_dirs.append(cls.local_path)
        temp_path = cfg.value('path.data_temp', None)
        if temp_path is not None:
            allowed_dirs.append(temp_path)

        allowed_dirs = list(set(allowed_dirs))

        ok = False

        for d in allowed_dirs:
            parent = os.path.realpath(os.path.abspath(d))
            child = os.path.realpath(os.path.abspath(path))

            if os.path.commonpath([parent]) == os.path.commonpath([parent, child]):
                ok = True
                break

        if not ok:
            err_str = "Cannot make a new folder not inside the following folders: "
            err_str += "\n".join(allowed_dirs)
            err_str += f"\n\nAttempted folder: {path}"
            raise ValueError(err_str)

        # if the path is ok, also make the subfolders
        os.makedirs(path, exist_ok=True)

    filepath = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        unique=True,
        doc="Base path (relative to the data root) for a stored file"
    )

    filepath_extensions = sa.Column(
        sa.ARRAY(sa.Text),
        nullable=True,
        doc="If non-null, array of text appended to filepath to get actual saved filenames."
    )

    md5sum = sa.Column(
        sqlUUID(as_uuid=True),
        nullable=True,
        default=None,
        doc = "md5sum of the file, provided by the archive server"
    )

    md5sum_extensions = sa.Column(
        sa.ARRAY(sqlUUID(as_uuid=True)),
        nullable=True,
        default=None,
        doc="md5sum of extension files; must have same number of elements as filepath_extensions"
    )

    def __init__(self, *args, **kwargs):
        """
        Initialize an object that is associated with a file on disk.
        If giving a single unnamed argument, will assume that is the filepath.
        Note that the filepath should not include the global data path,
        but only a path relative to that. # TODO: remove the global path if filepath starts with it?

        Parameters
        ----------
        args: list
            List of arguments, should only contain one string as the filepath.
        kwargs: dict
            Dictionary of keyword arguments.
            These include:
            - filepath: str
                Use instead of the unnamed argument.
            - nofile: bool
                If True, will not require the file to exist on disk.
                That means it will not try to download it from archive, either.
                This should be used only when creating a new object that will
                later be associated with a file on disk (or for tests).
                This property is NOT SAVED TO DB!
                Saving to DB should only be done when a file exists
                This is True by default, except for subclasses that
                override the _do_not_require_file_to_exist() method.
                # TODO: add the check that file exists before committing?
        """
        if len(args) == 1 and isinstance(args[0], str):
            self.filepath = args[0]

        self.filepath = kwargs.pop('filepath', self.filepath)
        self.nofile = kwargs.pop('nofile', self._do_not_require_file_to_exist())

    @orm.reconstructor
    def init_on_load(self):
        self.nofile = self._do_not_require_file_to_exist()

    @staticmethod
    def _do_not_require_file_to_exist():
        """
        The default value for the nofile property of new objects.
        Generally it is ok to make new FileOnDiskMixin derived objects
        without first having a file (the file is created by the app and
        saved to disk before the object is committed).
        Some subclasses (e.g., Exposure) will override this method
        so that the default is that a file MUST exist upon creation.
        In either case the caller to the __init__ method can specify
        the value of nofile explicitly.
        """
        return True

    def __setattr__(self, key, value):
        if key == 'filepath' and isinstance(value, str):
            value = self._validate_filepath(value)

        super().__setattr__(key, value)

    def _validate_filepath(self, filepath):
        """
        Make sure the filepath is legitimate.
        If the filepath starts with the local path
        (i.e., an absolute path is given) then
        the local path is removed from the filepath,
        forcing it to be a relative path.

        Parameters
        ----------
        filepath: str
            The filepath to validate.

        Returns
        -------
        filepath: str
            The validated filepath.
        """
        if filepath.startswith(self.local_path):
            filepath = filepath[len(self.local_path) + 1:]

        return filepath

    def get_fullpath(self, download=True, as_list=False, nofile=None, always_verify_md5=False):
        """
        Get the full path of the file, or list of full paths
        of files if filepath_extensions is not None.
        If the archive is defined, and download=True (default),
        the file will be downloaded from the server if missing.
        If the file is not found on server or locally, will
        raise a FileNotFoundError.
        When setting self.nofile=True, will not check if the file exists,
        or try to download it from server. The assumption is that an
        object with self.nofile=True will be associated with a file later on.

        If the file is found on the local drive, under the local_path,
        (either it was there or after it was downloaded)
        the full path is returned.
        The application is then responsible for loading the content
        of the file.

        When the filepath_extensions is None, will return a single string.
        When the filepath_extensions is an array, will return a list of strings.
        If as_list=False, will always return a list of strings,
        even if filepath_extensions is None.

        Parameters
        ----------
        download: bool
            Whether to download the file from server if missing.
            Must have archive defined. Default is True.
        as_list: bool
            Whether to return a list of filepaths, even if filepath_extensions=None.
            Default is False.
        nofile: bool
            Whether to check if the file exists on local disk.
            Default is None, which means use the value of self.nofile.
        always_verify_md5: bool
            Set True to verify that the file's md5sum matches what's
            in the database (if there is one in the database), and
            raise an exception if it doesn't.  Ignored if nofile=True.

        Returns
        -------
        str or list of str
            Full path to the file(s) on local disk.
        """
        if self.filepath_extensions is None:
            if as_list:
                return [self._get_fullpath_single(download=download, nofile=nofile,
                                                  always_verify_md5=always_verify_md5)]
            else:
                return self._get_fullpath_single(download=download, nofile=nofile,
                                                 always_verify_md5=always_verify_md5)
        else:
            return [
                self._get_fullpath_single(download=download, ext=ext, nofile=nofile,
                                          always_verify_md5=always_verify_md5)
                for ext in self.filepath_extensions
            ]

    def _get_fullpath_single(self, download=True, ext=None, nofile=None, always_verify_md5=False):
        """Get the full path of a single file.
        Will follow the same logic as get_fullpath(),
        of checking and downloading the file from the server
        if it is not on local disk.

        Parameters
        ----------
        download: bool
            Whether to download the file from server if missing.
            Must have archive defined. Default is True.
        ext: str
            Extension to add to the filepath. Default is None
            (indicating that this object is stored in a single file).
        nofile: bool
            Whether to check if the file exists on local disk.
            Default is None, which means use the value of self.nofile.
        always_verify_md5: bool
            Set True to verify that the file's md5sum matches what's
            in the database (if there is one in the database), and
            raise an exception if it doesn't.  Ignored if nofile=True.

        Returns
        -------
        str
            Full path to the file on local disk.

        """
        if nofile is None:
            nofile = self.nofile
        if not nofile and self.local_path is None:
            raise ValueError("Local path not defined!")

        fname = self.filepath
        md5sum = None
        if ext is None:
            md5sum = self.md5sum.hex if self.md5sum is not None else None
        else:
            found = False
            try:
                extdex = self.filepath_extensions.index( ext )
            except ValueError:
                raise ValueError(f"Unknown extension {ext} for {fname}" )
            if (self.md5sum_extensions is None ) or ( extdex >= len(self.md5sum_extensions) ):
                md5sum = None
            else:
                md5sum = self.md5sum_extensions[extdex]
                md5sum = None if md5sum is None else md5sum.hex
            fname += ext

        downloaded = False
        fullname = os.path.join(self.local_path, fname)
        if ( not nofile ) and ( not os.path.exists(fullname) ) and download and ( self.archive is not None ):
            if md5sum is None:
                raise RuntimeError(f"Don't have md5sum in the database for {fname}, can't download")
            self.archive.download( fname, fullname, verifymd5=True, clobbermismatch=False, mkdir=True )
            downloaded = True

        if not nofile:
            if not os.path.exists(fullname):
                raise FileNotFoundError(f"File {fullname} not found!")
            elif always_verify_md5 and not downloaded and md5sum is not None:
                # self.archive.download will have already verified the md5sum
                filemd5 = hashlib.md5()
                with open(fullname, "rb") as ifp:
                    filemd5.update(ifp.read())
                localmd5 = filemd5.hexdigest()
                if localmd5 != md5sum:
                    raise ValueError( f"{fname} has md5sum {localmd5} on disk, which doesn't match the "
                                      f"database value of {md5sum}" )
        
        return fullname

    def save(self, data, extension=None, overwrite=True, exists_ok=True, verify_md5=True, no_archive=False ):
        """Save a file to disk, and to the archive.

        Parametrs
        ---------
        data: bytes, string, or Path
          The data to be saved
        extension: string or None
          The file extension
        overwrite: bool
          True to overwrite existing files (locally and on the archive).
        exists_ok: bool
          Ignored if overwrite is True.  Otherwise: if the file exists
          on disk, and this is False, raise an exception.
        verify_md5: bool
          Used to modify both overwrite and exists_ok
            LOCAL STORAGE
               verify_md5 = True
                  if overwrite = True, check file md5 before actually overwriting
                  if overwrite = False
                      if exists_ok = True, verify existing file
               verify_md5 = False
                  if overwrite = True, always overwrite the file
                  if overwrite = False
                      if exists_ok = True, assume existing file is right
            ARCHIVE
               If self.md5sum (or the appropriate entry in
               md5sum_extensions) is null, then always upload to the
               archive as long as no_archive is False). Otherwise,
               verify_md5 modifies the behavior;
               verify_md5 = True
                 If self.md5sum (or the appropriate entry in
                 md5sum_extensions) matches the md5sum of the passed data,
                 do not upload to the archive.  Otherwise, if overwrite
                 is true, upload to the archive; if overwrite is false,
                 raise an exception.
               verify_md5 = False
                 If overwrite is True, upload to the archive,
                 overwriting what's there.  Otherwise, assume that
                 what's on the archive is right.
        no_archive: bool
          If True, do *not* save to the archive, only to the local filesystem.

        If data is a "bytes" type, then it represents the relevant
        binary data.  It will be written to the right place in the local
        filestore (underneath path.data_root).

        If data is a pathlib.Path or a string, then it is the
        (resolvable) path to a file on disk.  In this case, if the file
        is already in the right place underneath path.data_root, it is
        just left in place (modulo verify_md5).  Otherwise, it is copied
        there.

        Then, in either case, the file is uploaded to the archive (if
        the class property archive is not None, modul verify_md5).  Once
        it's uploaded to the archive, the object's md5sum is set (or
        updated, if overwrite is True and it wasn't null to start with).

        If extension is not None, and it isn't already in the list of
        filepath_extensions, it will be added.

        Performance notes: if you call this with anything other than
        overwrite=False, exists_ok=True, verify_md5=False, you may well
        have redundant I/O.  You may have saved an image before, and
        then (for instance) called
        pipeline.data_store.save_and_commit(), which will at the very
        least read things to verify that md5sums match.

        Of course, not either using overwrite=True or verify_md5=True
        could lead to incorrect files in either the local filestore on
        the server not being detected.

        """

        # (This one is an ugly mess of if statements, done to avoid reading
        # or writing files when not necessary since I/O tends to be much
        # more expensive than processing.)

        # First : figure out if this is an extension or not,
        #   and make sure that's consistent with the object.
        # If it is:
        #   Find the index into the extensions array for
        #   this extension, or append to the array if
        #   it's a new extension that doesn't already exist.
        #   Set the variables curextensions and extmd5s to lists with
        #   extensions and md5sums of extension files,
        #   initially copied from self.filepath_extensions and
        #   self.md5sum_extensions, and modified if necessary
        #   with the saved file.  extensiondex holds the index
        #   into both of these arrays for the current extension.
        # else:
        #   Set curextions, extmd5s, and extensiondex to None.

        # We will either replace these two variables with empty lists,
        #  or make a copy (using list()).  The reason for this: we don't
        #  want to directly modify the lists in self until the saving is
        #  done.  That way, self doesn't get mucked up if this function
        #  exceptions out.
        curextensions = self.filepath_extensions
        extmd5s = self.md5sum_extensions

        extensiondex = None
        if extension is None:
            if curextensions is not None:
                raise RuntimeError( "Tried to save a non-extension file, but this file has extensions" )
            if extmd5s is not None:
                raise RuntimeError( "Data integrity error; filepath_extensions is null, "
                                    "but md5sum_extensions isn't." )
        else:
            if curextensions is None:
                if extmd5s is not None:
                    raise RuntimeError( "Data integrity error; filepath_extensions is null, "
                                        "but md5sum_extensions isn't." )
                curextensions = []
                extmd5s = []
            else:
                if extmd5s is None:
                    raise RuntimeError( "Data integrity error; filepath_extensions is not null, "
                                        "but md5sum_extensions is" )
                curextensions = list(curextensions)
                extmd5s = list(extmd5s)
            if len(curextensions) != len(extmd5s):
                raise RuntimeError( f"Data integrity error; len(md5sum_extensions)={len(extmd5s)}, "
                                    f"but len(filepath_extensions)={len(curextensions)}" )
            try:
                extensiondex = curextensions.index( extension )
            except ValueError:
                curextensions.append( extension )
                extmd5s.append( None )
                extensiondex = len(curextensions) - 1

        # relpath holds the path of the file relative to the data store root
        # origmd5 holds the md5sum (hashlib.hash object) of the original file,
        #   *unless* the original file is already the right file in the local file store
        #   (in which case it's None)
        # localpath holds the absolute path of where the file should be written in the local file store
        relpath = pathlib.Path( self.filepath if extension is None else self.filepath + extension )
        localpath = pathlib.Path( self.local_path ) / relpath
        if isinstance( data, bytes ):
            path = "passed data"
        else:
            if isinstance( data, str ):
                path = pathlib.Path( data )
            elif isinstance( data, pathlib.Path ):
                path = data
            else:
                raise TypeError( f"data must be bytes, str, or Path, not {type(data)}" )
            path = path.absolute()
            data = None

        alreadyinplace = False
        mustwrite = False
        origmd5 = None
        if not localpath.exists():
            mustwrite = True
        else:
            if not localpath.is_file():
                raise RuntimeError( f"{localpath} exists but is not a file!  Can't save." )
            if localpath == path:
                alreadyinplace = True
                _logger.debug( f"FileOnDiskMixin.save: local file store path and original path are the same: {path}" )
            else:
                if ( not overwrite ) and ( not exists_ok ):
                    raise FileExistsError( f"{localpath} already exists, cannot save." )
                if verify_md5:
                    origmd5 = hashlib.md5()
                    if data is None:
                        with open( path, "rb" ) as ifp:
                            data = ifp.read()
                    origmd5.update( data )
                    localmd5 = hashlib.md5()
                    with open( localpath, "rb" ) as ifp:
                        localmd5.update( ifp.read() )
                    if localmd5.hexdigest() != origmd5.hexdigest():
                        if overwrite:
                            _logger.debug( f"Existing {localpath} md5sum mismatch; overwriting." )
                            mustwrite = True
                        else:
                            raise ValueError( f"{localpath} exists, but its md5sum {localmd5.hexdigest()} does not "
                                              f"match md5sum of {path} {origmd5.hexdigest()}" )
                else:
                    if overwrite:
                        _logger.debug( f"Overwriting {localpath}" )
                        mustwrite = True
                    elif exists_ok:
                        _logger.debug( f"{localpath} already exists, not verifying md5 nor overwriting" )
                    else:
                        # raise FileExistsError( f"{localpath} already exists, not saving" )
                        # Logically, should not be able to get here
                        raise RuntimeError( "This should never happen" )
            
        if mustwrite and not alreadyinplace:
            if data is None:
                with open( path, "rb" ) as ifp:
                    data = ifp.read()
            if origmd5 is None:
                origmd5 = hashlib.md5()
                origmd5.update( data )
            with open( localpath, "wb" ) as ofp:
                ofp.write( data )
            # Verify written file
            with open( localpath, "rb" ) as ifp:
                writtenmd5 = hashlib.md5()
                writtenmd5.update( ifp.read() )
                if writtenmd5.hexdigest() != origmd5.hexdigest():
                    raise RuntimeError( f"Error writing {localpath}; written file md5sum mismatches expected!" )

        if no_archive:
            if curextensions is not None:
                self.filepath_extensions = curextensions
                self.md5sum_extensions = extmd5s
            return
        
        # The rest of this deals with the archive

        archivemd5 = self.md5sum if extension is None else extmd5s[extensiondex]

        mustupload = False
        if archivemd5 is None:
            mustupload = True
        else:
            if not verify_md5:
                if overwrite:
                    _logger.debug( f"Uploading {self.filepath} to archive, overwriting existing file" )
                    mustupload = True
                else:
                    _logger.debug( f"Assuming existing {self.filepath} on archive is correct" )
            else:
                if origmd5 is None:
                    origmd5 = hashlib.md5()
                    if data is None:
                        with open( localpath, "rb" ) as ifp:
                            data = ifp.read()
                    origmd5.update( data )
                if origmd5.hexdigest() == archivemd5.hex:
                    _logger.debug( f"Archive md5sum for {self.filepath} matches saved data, not reuploading." )
                else:
                    if overwrite:
                        _logger.debug( f"Archive md5sum for {self.filepath} doesn't match saved data, "
                                       f"overwriting on archive." )
                        mustupload = True
                    else:
                        raise ValueError( f"Archive md5sum for {self.filepath} does not match saved data!" )


        if mustupload:
            remmd5 = self.archive.upload( localpath, relpath.parent, relpath.name, overwrite=overwrite, md5=origmd5 )
            remmd5 = UUID( remmd5 )
            if curextensions is not None:
                extmd5s[extensiondex] = remmd5
                self.md5sum = None
                self.filepath_extensions = curextensions
                self.md5sum_extensions = extmd5s
            else:
                self.md5sum = remmd5


    def remove_data_from_disk(self, remove_folders=True, purge_archive=False, session=None, nocommit=False):

        """Delete the data from disk, if it exists.
        If remove_folders=True, will also remove any folders
        if they are empty after the deletion.

        Parameters
        ----------
        remove_folders: bool
            If True, will remove any folders on the path to the files
            associated to this object, if they are empty.
        purge_archive: bool
            If True, will also remove these files from the archive.
            Make this True when deleteing a file from the database. 
            Make this False when you're just cleaning up local storage.
        session: an sqlalchemy session, or None
            Database session -- IMPORTANT: if you call this method
            when a database session has a transaction in progress,
            pass that session here, otherwise you'll likely get
            database deadlocks.
        nocommit: bool
            Don't commit the md5sum changes to the database. WARNING:
            only use this if the calling function is going to commit the
            self object to the database shortly after calling this
            method!  Otherwise, the database will get out of sync from
            the reality of the archive.  This argument is here so that
            this method may be called inside a database transaction that
            is doing things that would lead to problems (e.g. deleting
            some things that other things that won't be deleted until
            later refer to via foreign keys).

        """
        if self.filepath is None:
            return
        # get the filepath, but don't check if the file exists!
        for f in self.get_fullpath(as_list=True, nofile=True):
            if os.path.exists(f):
                os.remove(f)
                if remove_folders:
                    folder = f
                    for i in range(10):
                        folder = os.path.dirname(folder)
                        if len(os.listdir(folder)) == 0:
                            os.rmdir(folder)
                        else:
                            break
        if purge_archive:
            if self.filepath_extensions is None:
                self.archive.delete( self.filepath, okifmissing=True )
            else:
                for ext in self.filepath_extensions:
                    self.archive.delete( f"{self.filepath}{ext}", okifmissing=True )
            self.md5sum = None
            if self.filepath_extensions is not None:
                self.md5sum_extensions = [ None for i in range(len(self.filepath_extensions)) ]
            # If this image is in the database, make sure the database md5sum fields are updated
            if not nocommit:
                with SmartSession(session) as smsess:
                    safe_merge( smsess, self )
                    smsess.commit()

def safe_mkdir(path):
    FileOnDiskMixin.safe_mkdir(path)


class SpatiallyIndexed:
    """A mixin for tables that have ra and dec fields indexed via q3c."""

    ra = sa.Column(sa.Double, nullable=False, doc='Right ascension in degrees')

    dec = sa.Column(sa.Double, nullable=False, doc='Declination in degrees')

    gallat = sa.Column(sa.Double, index=True, doc="Galactic latitude of the target. ")

    gallon = sa.Column(sa.Double, index=False, doc="Galactic longitude of the target. ")

    ecllat = sa.Column(sa.Double, index=True, doc="Ecliptic latitude of the target. ")

    ecllon = sa.Column(sa.Double, index=False, doc="Ecliptic longitude of the target. ")

    @declared_attr
    def __table_args__(cls):
        tn = cls.__tablename__
        return (
            sa.Index(f"{tn}_q3c_ang2ipix_idx", sa.func.q3c_ang2ipix(cls.ra, cls.dec)),
        )

    def calculate_coordinates(self):
        if self.ra is None or self.dec is None:
            raise ValueError("Object must have RA and Dec set before calculating coordinates! ")

        coords = SkyCoord(self.ra, self.dec, unit="deg", frame="icrs")
        self.gallat = coords.galactic.b.deg
        self.gallon = coords.galactic.l.deg
        self.ecllat = coords.barycentrictrueecliptic.lat.deg
        self.ecllon = coords.barycentrictrueecliptic.lon.deg

if __name__ == "__main__":
    pass
