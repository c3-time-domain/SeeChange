import os
import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.ext.associationproxy import association_proxy

import h5py

from astropy.table import Table

from models.base import (
    SmartSession,
    Base,
    SeeChangeBase,
    AutoIDMixin,
    FileOnDiskMixin,
    SpatiallyIndexed,
    HasBitFlagBadness,
)
from models.enums_and_bitflags import CutoutsFormatConverter, cutouts_badness_inverse
from models.source_list import SourceList


class Cutouts(Base, AutoIDMixin, FileOnDiskMixin, SpatiallyIndexed, HasBitFlagBadness):

    __tablename__ = 'cutouts'

    # a unique constraint on the provenance and the source list, but also on the index in the list
    __table_args__ = (
        UniqueConstraint(
            'index_in_sources', 'sources_id', 'provenance_id', name='_cutouts_index_sources_provenance_uc'
        ),
    )

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=CutoutsFormatConverter.convert('hdf5'),
        doc="Format of the file on disk. Should be fits, hdf5, csv or npy. "
            "Saved as integer but is converted to string when loaded. "
    )

    @hybrid_property
    def format(self):
        return CutoutsFormatConverter.convert(self._format)

    @format.expression
    def format(cls):
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(CutoutsFormatConverter.dict, value=cls._format)

    @format.setter
    def format(self, value):
        self._format = CutoutsFormatConverter.convert(value)

    sources_id = sa.Column(
        sa.ForeignKey('source_lists.id', name='cutouts_source_list_id_fkey', ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of the source list (of detections in the difference image) this cutouts object is associated with. "
    )

    sources = orm.relationship(
        SourceList,
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        lazy='selectin',
        doc="The source list (of detections in the difference image) this cutouts object is associated with. "
    )

    index_in_sources = sa.Column(
        sa.Integer,
        nullable=False,
        doc="Index of this cutout in the source list (of detections in the difference image). "
    )

    sub_image_id = association_proxy('sources', 'image_id')
    sub_image = association_proxy('sources', 'image')

    x = sa.Column(
        sa.Integer,
        nullable=False,
        doc="X pixel coordinate of the center of the cutout. "
    )

    y = sa.Column(
        sa.Integer,
        nullable=False,
        doc="Y pixel coordinate of the center of the cutout. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='cutouts_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this cutout. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this cutout. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this cutout. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this cutout. "
        )
    )

    @property
    def new_image(self):
        """Get the aligned new image using the sub_image. """
        return self.sub_image.new_aligned_image

    @property
    def ref_image(self):
        """Get the aligned reference image using the sub_image. """
        return self.sub_image.ref_aligned_image

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        HasBitFlagBadness.__init__(self)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self.format = 'hdf5'  # the default should match the column-defined default above!

        self._source_row = None

        self._sub_data = None
        self._sub_weight = None
        self._sub_flags = None
        self._sub_psfflux = None
        self._sub_psffluxerr = None

        self._ref_data = None
        self._ref_weight = None
        self._ref_flags = None

        self._new_data = None
        self._new_weight = None
        self._new_flags = None

        self._bitflag = 0

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)

        self._source_row = None

        self._sub_data = None
        self._sub_weight = None
        self._sub_flags = None
        self._sub_psfflux = None
        self._sub_psffluxerr = None

        self._ref_data = None
        self._ref_weight = None
        self._ref_flags = None

        self._new_data = None
        self._new_weight = None
        self._new_flags = None

    def __repr__(self):
        return (
            f"<Cutouts {self.id} "
            f"from SourceList {self.sources_id} "
            f"(number {self.index_in_sources}) "
            f"from Image {self.sub_image_id} "
            f"at x,y= {self.x}, {self.y}>"
        )

    def __setattr__(self, key, value):
        if key in ['x', 'y'] and value is not None:
            value = int(round(value))

        super().__setattr__(key, value)

    @staticmethod
    def get_data_attributes(include_optional=True):
        names = ['source_row']
        for im in ['sub', 'ref', 'new']:
            for att in ['data', 'weight', 'flags']:
                names.append(f'{im}_{att}')

        if include_optional:
            names += ['sub_psfflux', 'sub_psffluxerr']

        return names

    @property
    def has_data(self):
        for att in self.get_data_attributes(include_optional=False):
            if getattr(self, att) is None:
                return False
        return True

    @property
    def sub_nandata(self):
        if self.sub_data is None or self.sub_flags is None:
            return None
        return np.where(self.sub_flags > 0, np.nan, self.sub_data)

    @property
    def ref_nandata(self):
        if self.ref_data is None or self.ref_flags is None:
            return None
        return np.where(self.ref_flags > 0, np.nan, self.ref_data)

    @property
    def new_nandata(self):
        if self.new_data is None or self.new_flags is None:
            return None
        return np.where(self.new_flags > 0, np.nan, self.new_data)

    @staticmethod
    def from_detections(detections, source_index, provenance=None, **kwargs):
        """Create a Cutout object from a row in the SourceList.

        The SourceList must have a valid image attribute, and that image should have exactly two
        upstream_images: the reference and new image. Each Cutout will have three small stamps
        from the new, reference, and subtraction images.

        Parameters
        ----------
        detections: SourceList
            The source list from which to create the cutout.
        source_index: int
            The index of the source in the source list from which to create the cutout.
        provenance: Provenance, optional
            The provenance of the cutout. If not given, will leave as None (to be filled externally).
        kwargs: dict
            Can include any of the following keys, in the format: {im}_{att}, where
            the {im} can be "sub", "ref", or "new", and the {att} can be "data", "weight", or "flags".
            These are optional, to be used to fill the different data attributes of this object.

        Returns
        -------
        cutout: Cutout
            The cutout object.
        """
        cutout = Cutouts()
        cutout.sources = detections
        cutout.index_in_sources = source_index
        cutout.source_row = dict(Table(detections.data)[source_index])
        for key, value in cutout.source_row.items():
            if isinstance(value, np.number):
                cutout.source_row[key] = value.item()  # convert numpy number to python primitive
        cutout.x = detections.x[source_index]
        cutout.y = detections.y[source_index]
        cutout.ra = cutout.source_row['ra']
        cutout.dec = cutout.source_row['dec']
        cutout.calculate_coordinates()
        cutout.provenance = provenance

        # add the data, weight, and flags to the cutout from kwargs
        for im in ['sub', 'ref', 'new']:
            for att in ['data', 'weight', 'flags']:
                setattr(cutout, f'{im}_{att}', kwargs.get(f'{im}_{att}', None))

        # update the bitflag
        cutout._upstream_bitflag = detections.bitflag

        return cutout

    def invent_filepath(self):
        if self.sources is None:
            raise RuntimeError( f"Can't invent a filepath for cutouts without a source list" )
        if self.provenance is None:
            raise RuntimeError( f"Can't invent a filepath for cutouts without a provenance" )

        # base the filename on the image filename, not on the sources filename.
        filename = self.sub_image.filepath
        if filename is None:
            filename = self.sub_image.invent_filepath()

        if filename.endswith(('.fits', '.h5', '.hdf5')):
            filename = os.path.splitext(filename)[0]

        filename += '.cutouts_'
        filename += self.provenance.id[:6]
        if self.format == 'hdf5':
            filename += '.h5'
        elif self.format == ['fits', 'jpg', 'png']:
            filename += f'.{self.format}'
        else:
            raise TypeError( f"Unable to create a filepath for cutouts file of type {self.format}" )

        return filename

    def _save_dataset_to_hdf5(self, file, groupname):
        """Save the dataset from this Cutouts object into an HDF5 group for an open file.

        Parameters
        ----------
        file: h5py.File
            The open HDF5 file to save to.
        groupname: str
            The name of the group to save into. This should be "source_<number>"
        """
        if groupname in file:
            del file[groupname]

        # handle the data arrays
        for att in self.get_data_attributes():
            if att == 'source_row':
                continue

            data = getattr(self, f'_{att}')  # get the private attribute so as not to trigger a load upon hitting None
            if data is not None:
                file.create_dataset(
                    f'{groupname}/{att}',
                    data=data,
                    shape=data.shape,
                    dtype=data.dtype,
                    compression='gzip'
                )

        # handle the source_row dictionary
        target = file[groupname].attrs
        for key in target.keys():  # first clear the existing keys
            del target[key]

        # then add the new ones
        for key, value in self.source_row.items():
            target[key] = value

    def save(self, filename=None, **kwargs):
        """Save a single Cutouts object into a file.

        Parameters
        ----------
        filename: str, optional
            The (relative/full path) filename to save to. If not given, will use the default filename.
        kwargs: dict
            Any additional keyword arguments to pass to the FileOnDiskMixin.save method.
        """
        raise NotImplementedError('Saving only a single cutout into a file is not supported. Use save_list instead.')

        if not self.has_data:
            raise RuntimeError("The Cutouts data is not loaded. Cannot save.")

        if filename is not None:
            self.filepath = filename
        if self.filepath is None:
            self.filepath = self.invent_filepath()

        fullname = self.get_fullpath()
        self.safe_mkdir(os.path.dirname(fullname))

        if self.format == 'hdf5':
            with h5py.File(fullname, 'a') as file:
                self._save_dataset_to_hdf5(file, f'source_{self.index_in_sources}')
        elif self.format == 'fits':
            raise NotImplementedError('Saving cutouts to fits is not yet implemented.')
        elif self.format in ['jpg', 'png']:
            raise NotImplementedError('Saving cutouts to jpg or png is not yet implemented.')
        else:
            raise TypeError(f"Unable to save cutouts file of type {self.format}")

        # make sure to also save using the FileOnDiskMixin method
        super().save(fullname, **kwargs)

    @classmethod
    def save_list(cls, cutouts_list,  filename=None, overwrite=True, **kwargs):
        """Save a list of Cutouts objects into a file.

        Parameters
        ----------
        cutouts_list: list of Cutouts
            The list of Cutouts objects to save.
        filename: str, optional
            The (relative/full path) filename to save to. If not given, will use the default filename.
        overwrite: bool
            If True, will overwrite the file if it already exists.
            If False, will raise an error if the file already exists.
        kwargs: dict
            Any additional keyword arguments to pass to the File
        """
        if not isinstance(cutouts_list, list):
            raise TypeError("The input must be a list of Cutouts objects.")
        if len(cutouts_list) == 0:
            return  # silently do nothing

        for cutout in cutouts_list:
            if not isinstance(cutout, cls):
                raise TypeError("The input must be a list of Cutouts objects.")
            if not cutout.has_data:
                raise RuntimeError("The Cutouts data is not loaded. Cannot save.")

        if filename is None:
            filename = cutouts_list[0].invent_filepath()

        fullname = os.path.join(cutouts_list[0].local_path, filename)
        cutouts_list[0].safe_mkdir(os.path.dirname(fullname))

        if not overwrite and os.path.isfile(fullname):
            raise FileExistsError(f"The file {fullname} already exists and overwrite is False.")

        if cutouts_list[0].format == 'hdf5':
            with h5py.File(fullname, 'a') as file:
                for cutout in cutouts_list:
                    cutout._save_dataset_to_hdf5(file, f'source_{cutout.index_in_sources}')
                    cutout.filepath = filename
        elif cutouts_list[0].format == 'fits':
            raise NotImplementedError('Saving cutouts to fits is not yet implemented.')
        elif cutouts_list[0].format in ['jpg', 'png']:
            raise NotImplementedError('Saving cutouts to jpg or png is not yet implemented.')
        else:
            raise TypeError(f"Unable to save cutouts file of type {cutouts_list[0].format}")

        # make sure to also save using the FileOnDiskMixin method
        FileOnDiskMixin.save(cutouts_list[0], fullname, overwrite=overwrite, **kwargs)

        # after saving one object as a FileOnDiskMixin, all the others should have the same md5sum
        if cutouts_list[0].md5sum is not None:
            for cutout in cutouts_list:
                cutout.md5sum = cutouts_list[0].md5sum

    def _load_dataset_from_hdf5(self, file, groupname):
        """Load the dataset from an HDF5 group into this Cutouts object.

        Parameters
        ----------
        file: h5py.File
            The open HDF5 file to load from.
        groupname: str
            The name of the group to load from. This should be "source_<number>"
        """
        for att in self.get_data_attributes():
            if att == 'source_row':
                self.source_row = dict(file[groupname].attrs)
            elif att in file[groupname]:
                setattr(self, att, np.array(file[f'{groupname}/{att}']))

        self.format = 'hdf5'

    def load(self, filepath=None):
        """Load the data for this cutout from a file.

        Parameters
        ----------
        filepath: str, optional
            The (relative/full path) filename to load from.
            If not given, will use self.get_fullpath() to get the filename.
        """
        if filepath is None:
            filepath = self.get_fullpath()

        if self.format == 'hdf5':
            with h5py.File(filepath, 'r') as file:
                self._load_dataset_from_hdf5(file, f'source_{self.index_in_sources}')
        elif self.format == 'fits':
            raise NotImplementedError('Loading cutouts from fits is not yet implemented.')
        elif self.format in ['jpg', 'png']:
            raise NotImplementedError('Loading cutouts from jpg or png is not yet implemented.')
        else:
            raise TypeError(f"Unable to load cutouts file of type {self.format}")

    @classmethod
    def from_file(cls, filepath, source_number, **kwargs):
        """Create a Cutouts object from a file.

        Will try to guess the format based on the file extension.

        Parameters
        ----------
        filepath: str
            The (relative/full path) filename to load from.
        source_number: int
            The index of the source in the source list from which to create the cutout.
            This relates to the internal storage in file. For HDF5 files, the group
            for this object will be named "source_{source_number}".
        kwargs: dict
            Any additional keyword arguments to pass to the Cutouts constructor.
            E.g., if you happen to know some database values for this object,
            like the ID of related objects or the bitflag, you can pass them here.
        """
        cutout = cls(**kwargs)
        fmt = os.path.splitext(filepath)[1][1:]
        if fmt == 'h5':
            fmt = 'hdf5'

        cutout.format = fmt
        cutout.index_in_sources = source_number
        cutout.load(filepath)

        for att in ['ra', 'dec', 'x', 'y']:
            if att in cutout.source_row:
                setattr(cutout, att, cutout.source_row[att])

        cutout.calculate_coordinates()

        if filepath.startswith(cutout.local_path):
            filepath = filepath[len(cutout.local_path) + 1:]
        cutout.filepath = filepath

        # TODO: should also load the MD5sum automatically?

        return cutout

    @classmethod
    def load_list(cls, filepath, cutout_list=None):
        """Load all Cutouts object that were saved to a file

        Note that these cutouts are not loaded from the database,
        so they will be missing important relationships like provenance and sources.
        If cutout_list is given, it must match the cutouts on the file,
        so that each cutouts object will be loaded the data from file,
        but retain its database relationships.

        Parameters
        ----------
        filepath: str
            The (relative/full path) filename to load from.
            The file format is determined by the extension.
        cutout_list: list of Cutouts, optional
            If given, will load the data from the file into these objects.

        Returns
        -------
        cutouts: Cutouts
            The list of cutouts loaded from the file.
        """
        ext = os.path.splitext(filepath)[1][1:]
        if ext == 'h5':
            format = 'hdf5'
        else:
            format = ext

        if filepath.startswith(Cutouts.local_path):
            rel_filepath = filepath[len(Cutouts.local_path) + 1:]

        cutouts = []

        if format == 'hdf5':
            with h5py.File(filepath, 'r') as file:
                for groupname in file.keys():
                    if groupname.startswith('source_'):
                        number = int(groupname.split('_')[1])
                        if cutout_list is None:
                            cutout = cls()
                            cutout.format = format
                            cutout.index_in_sources = number
                        else:
                            cutout = [c for c in cutout_list if c.index_in_sources == number]
                            if len(cutout) != 1:
                                raise ValueError(f"Could not find a unique cutout with index {number} in the list.")
                            cutout = cutout[0]

                        cutout._load_dataset_from_hdf5(file, groupname)
                        cutout.filepath = rel_filepath
                        for att in ['ra', 'dec', 'x', 'y']:
                            if att in cutout.source_row:
                                setattr(cutout, att, cutout.source_row[att])

                        cutout.calculate_coordinates()

                        cutouts.append(cutout)

        elif format == 'fits':
            raise NotImplementedError('Loading cutouts from fits is not yet implemented.')
        elif format in ['jpg', 'png']:
            raise NotImplementedError('Loading cutouts from jpg or png is not yet implemented.')
        else:
            raise TypeError(f"Unable to load cutouts file of type {format}")

        cutouts.sort(key=lambda x: x.index_in_sources)
        return cutouts

    def remove_data_from_disk(self, remove_folders=True, remove_downstreams=False):
        """Delete the data from local disk, if it exists.
        Will remove the dataset for this specific cutout from the file,
        and remove the file if this is the last cutout in the file.
        If remove_folders=True, will also remove any folders
        if they are empty after the deletion.
        This function will not remove database rows or archive files,
        only cleanup local storage for this object and its downstreams.

        To remove both the files and the database entry, use
        delete_from_disk_and_database() instead.

        Parameters
        ----------
        remove_folders: bool
            If True, will remove any folders on the path to the files
            associated to this object, if they are empty.
        remove_downstreams: bool
            This is not used, but kept here for backward compatibility with the base class.
        """
        raise NotImplementedError(
            'Currently there is no support for removing one Cutout at a time. Use delete_list instead.'
        )

        if self.filepath is not None:
            # get the filepath, but don't check if the file exists!
            for f in self.get_fullpath(as_list=True, nofile=True):
                if os.path.exists(f):
                    need_to_delete = False
                    if self.format == 'hdf5':
                        with h5py.File(f, 'a') as file:
                            del file[f'source_{self.index_in_sources}']
                            if len(file) == 0:
                                need_to_delete = True
                    elif self.format == 'fits':
                        raise NotImplementedError('Removing cutouts from fits is not yet implemented.')
                    elif self.format in ['jpg', 'png']:
                        raise NotImplementedError('Removing cutouts from jpg or png is not yet implemented.')
                    else:
                        raise TypeError(f"Unable to remove cutouts file of type {self.format}")

                    if need_to_delete:
                        os.remove(f)
                        if remove_folders:
                            folder = f
                            for i in range(10):
                                folder = os.path.dirname(folder)
                                if len(os.listdir(folder)) == 0:
                                    os.rmdir(folder)
                                else:
                                    break

    def delete_from_archive(self, remove_downstreams=False):
        """Delete the file from the archive, if it exists.
        Will only
        This will not remove the file from local disk, nor
        from the database.  Use delete_from_disk_and_database()
        to do that.

        Parameters
        ----------
        remove_downstreams: bool
            If True, will also remove any downstream data.
            Will recursively call get_downstreams() and find any objects
            that have remove_data_from_disk() implemented, and call it.
            Default is False.
        """
        raise NotImplementedError(
            'Currently archive does not support removing one Cutout at a time, use delete_list instead.'
        )
        if self.filepath is not None:
            if self.filepath_extensions is None:
                self.archive.delete( self.filepath, okifmissing=True )
            else:
                for ext in self.filepath_extensions:
                    self.archive.delete( f"{self.filepath}{ext}", okifmissing=True )

        # make sure these are set to null just in case we fail
        # to commit later on, we will at least know something is wrong
        self.md5sum = None
        self.md5sum_extensions = None

    def get_upstreams(self, session=None):
        """Get the detections SourceList that was used to make this cutout. """
        with SmartSession(session) as session:
            return session.scalars(sa.select(SourceList).where(SourceList.id == self.sources_id)).all()
        
    def get_downstreams(self, session=None, siblings=False):
        """Get the downstream Measurements that were made from this Cutouts object. """
        from models.measurements import Measurements

        with SmartSession(session) as session:
            return session.scalars(sa.select(Measurements).where(Measurements.cutouts_id == self.id)).all()

    @classmethod
    def merge_list(cls, cutouts_list, session):
        """Merge (or add) the list of Cutouts to the given session. """
        if cutouts_list is None or len(cutouts_list) == 0:
            return cutouts_list

        sources = session.merge(cutouts_list[0].sources)
        for i, cutout in enumerate(cutouts_list):
            cutouts_list[i].sources = sources
            cutouts_list[i] = session.merge(cutouts_list[i])

        return cutouts_list

    @classmethod
    def delete_list(cls, cutouts_list, remove_local=True, archive=True, database=True, session=None, commit=True):
        """
        Remove a list of Cutouts objects from local disk and/or the archive and/or the database.
        This removes the file that includes all the cutouts.
        Can only delete cutouts that share the same filepath.
        WARNING: this will not check that the file contains ONLY the cutouts on the list!
        So, if the list contains a subset of the cutouts on file, the file is still deleted.

        Parameters
        ----------
        cutouts_list: list of Cutouts
            The list of Cutouts objects to remove.
        remove_local: bool
            If True, will remove the file from local disk.
        archive: bool
            If True, will remove the file from the archive.
        database: bool
            If True, will remove the cutouts from the database.
        session: Session, optional
            The database session to use. If not given, will create a new session.
        commit: bool
            If True, will commit the changes to the database.
            If False, will not commit the changes to the database.
            If session is not given, commit must be True.
        """
        if database and session is None and not commit:
            raise ValueError('If session is not given, commit must be True.')

        filepath = set([c.filepath for c in cutouts_list])
        if len(filepath) > 1:
            raise ValueError(
                f'All cutouts must share the same filepath to be deleted together. Got: {filepath}'
            )

        if remove_local:
            fullpath = cutouts_list[0].get_fullpath()
            if fullpath is not None and os.path.isfile(fullpath):
                os.remove(fullpath)

        if archive:
            if cutouts_list[0].filepath is not None:
                cutouts_list[0].archive.delete(cutouts_list[0].filepath, okifmissing=True)

        if database:
            with SmartSession(session) as session:
                for cutout in cutouts_list:
                    cutout.delete_from_database(session=session, commit=False)
                if commit:
                    session.commit()

    def check_equals(self, other):
        """Compare if two cutouts have the same data. """
        if not isinstance(other, Cutouts):
            return super().__eq__(other)  # any other comparisons use the base class

        attributes = self.get_data_attributes()
        attributes += ['ra', 'dec', 'x', 'y', 'filepath', 'format']

        for att in attributes:
            if isinstance(getattr(self, att), np.ndarray):
                if not np.array_equal(getattr(self, att), getattr(other, att)):
                    return False
            else:  # other attributes get compared directly
                if getattr(self, att) != getattr(other, att):
                    return False

        return True

    def _get_inverse_badness(self):
        return cutouts_badness_inverse


# use these two functions to quickly add the "property" accessor methods
def load_attribute(object, att):
    """Load the data for a given attribute of the object."""
    if not hasattr(object, f'_{att}'):
        raise AttributeError(f"The object {object} does not have the attribute {att}.")
    if getattr(object, f'_{att}') is None:
        if object.filepath is None:
            return None  # objects just now created and not saved cannot lazy load data!
        object.load()  # can lazy-load all data

    # after data is filled, should be able to just return it
    return getattr(object, f'_{att}')


def set_attribute(object, att, value):
    """Set the value of the attribute on the object. """
    setattr(object, f'_{att}', value)


# add "@property" functions to all the data attributes
for att in Cutouts.get_data_attributes():
    setattr(
        Cutouts,
        att,
        property(
            fget=lambda self, att=att: load_attribute(self, att),
            fset=lambda self, value, att=att: set_attribute(self, att, value),
        )
    )

