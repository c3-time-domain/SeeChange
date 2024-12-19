import os
import numpy as np

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import UniqueConstraint, CheckConstraint

import h5py

from models.base import (
    SmartSession,
    Base,
    SeeChangeBase,
    UUIDMixin,
    FileOnDiskMixin,
    HasBitFlagBadness,
)
from models.image import Image
from models.enums_and_bitflags import CutoutsFormatConverter
from models.source_list import SourceList


class Co_Dict(dict):
    """Cutouts Dictionary used in Cutouts to store dictionaries which hold data arrays
    for individual cutouts. Acts as a normal dictionary, except when a key is passed
    using bracket notation (such as "co_dict[source_index_7]"), if that key is not present
    in the Co_dict then it will search on disk for the requested data, and if found
    will silently load that data and return it.
    Must be assigned a Cutouts object to its cutouts attribute so that it knows
    how to look for data.
    """
    def __init__(self, *args, **kwargs):
        self.cutouts = None  # this must be assigned before use
        super().__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        if key not in self.keys():
            if self.cutouts.filepath is not None:
                self.cutouts.load_one_co_dict(key) # no change if not found
        return super().__getitem__(key)


class Cutouts(Base, UUIDMixin, FileOnDiskMixin, HasBitFlagBadness):

    __tablename__ = 'cutouts'

    # a unique constraint on the provenance and the source list
    @declared_attr
    def __table_args__(cls):  # noqa: N805
        return (
            CheckConstraint( sqltext='NOT(md5sum IS NULL AND '
                             '(md5sum_components IS NULL OR array_position(md5sum_components, NULL) IS NOT NULL))',
                             name=f'{cls.__tablename__}_md5sum_check' ),
            UniqueConstraint('sources_id', 'provenance_id', name='_cutouts_sources_provenance_uc')
        )

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        server_default=sa.sql.elements.TextClause( str(CutoutsFormatConverter.convert('hdf5')) ),
        doc="Format of the file on disk. Should be fits, hdf5, csv or npy. "
            "Saved as integer but is converted to string when loaded. "
    )

    @hybrid_property
    def format(self):
        return CutoutsFormatConverter.convert(self._format)

    @format.expression
    def format(cls):  # noqa: N805
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(CutoutsFormatConverter.dict, value=cls._format)

    @format.setter
    def format(self, value):
        self._format = CutoutsFormatConverter.convert(value)

    sources_id = sa.Column(
        sa.ForeignKey('source_lists._id', name='cutouts_source_list_id_fkey', ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of the source list (of detections in the difference image) this cutouts object is associated with. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances._id', ondelete="CASCADE", name='cutouts_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this cutout. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this cutout. "
        )
    )

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

        self.co_dict = Co_Dict()
        self.co_dict.cutouts = self

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

        self.co_dict = Co_Dict()
        self.co_dict.cutouts = self

    def __repr__(self):
        return (
            f"<Cutouts {self.id} "
            f"from SourceList {self.sources_id}>"
        )

    @staticmethod
    def get_data_array_attributes(include_optional=True):
        names = []
        for im in ['sub', 'ref', 'new']:
            for att in ['data', 'weight', 'flags']:
                names.append(f'{im}_{att}')

        if include_optional:
            names += ['sub_psfflux', 'sub_psffluxerr']

        return names

    @staticmethod
    def get_data_scalar_attributes( include_optional=True ):
        return [ 'new_x', 'new_y' ]

    def load_all_co_data( self, sources=None ):
        """Intended method for a Cutouts object to ensure that the data for all
        sources is loaded into its co_dict attribute. Will only actually load
        from disk if any subdictionaries (one per source in SourceList) are missing.

        Should be used before, for example, iterating over the dictionary as in
        the creation of Measurements objects. Not necessary for accessing
        individual subdictionaries however, because the Co_Dict class can lazy
        load those as they are requested (eg. co_dict["source_index_0"]).

        Parameters
        ----------
          sources: SourceList
            The detections associated with these cutouts.  Here for
            efficiency, or if the cutouts and sources aren't yet in the
            database.  If not given, will load them from the database.

        """
        if sources is None:
            sources = SourceList.get_by_id( self.sources_id )
        if sources.num_sources is None:
            raise ValueError("The detections of this cutouts has no num_sources attr")
        proper_length = sources.num_sources
        if len(self.co_dict) != proper_length and self.filepath is not None:
            self.load()

    @staticmethod
    def from_detections(detections, provenance=None, **kwargs):
        """Create a Cutout object from a row in the SourceList.

        Each Cutout will have three small stamps from the new,
        reference, and subtraction images.

        Parameters
        ----------
        detections: SourceList
            The source list from which to create the cutout.  It should
            have exactly two upstream_images: the reference and new
            image.

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
        cutout.sources_id = detections.id
        cutout.provenance_id = None if provenance is None else provenance.id

        # update the bitflag
        cutout._upstream_bitflag = detections.bitflag

        return cutout

    def invent_filepath( self, image=None, detections=None ):
        if image is None:
            if detections is None:
                detections = SourceList.get_by_id( self.sources_id )
            if detections is None:
                raise RuntimeError( "Can't invent a filepath for cutouts without a image or detections source list" )
            image = Image.get_by_id( detections.image_id )
        if image is None:
            raise RuntimeError( "Can't invent a filepath for cutouts without an image" )
        if self.provenance_id is None:
            raise RuntimeError( "Can't invent a filepath for cutouts without a provenance" )

        # base the filename on the image filename, not on the sources filename.
        filename = image.filepath
        if filename is None:
            filename = image.invent_filepath()

        if filename.endswith(('.fits', '.h5', '.hdf5')):
            filename = os.path.splitext(filename)[0]

        filename += '.cutouts_'
        filename += self.provenance_id[:6]
        if self.format == 'hdf5':
            filename += '.h5'
        elif self.format == ['fits', 'jpg', 'png']:
            filename += f'.{self.format}'
        else:
            raise TypeError( f"Unable to create a filepath for cutouts file of type {self.format}" )

        return filename

    def _save_dataset_dict_to_hdf5(self, co_subdict, file, groupname):
        """Save the one co_subdict from the co_dict of this Cutouts
        into an HDF5 group for an open file.

        Parameters
        ----------
        co_subdict: dict
            The subdict containing the data for a single cutout
        file: h5py.File
            The open HDF5 file to save to.
        groupname: str
            The name of the group to save into. This should be "source_<number>"
        """
        if groupname in file:
            del file[groupname]

        for key in self.get_data_array_attributes():
            data = co_subdict.get(key)

            if data is not None:
                file.create_dataset(
                    f'{groupname}/{key}',
                    data=data,
                    shape=data.shape,
                    dtype=data.dtype,
                    compression='gzip'
                )

        for key in self.get_data_scalar_attributes():
            data = co_subdict.get(key)
            if data is not None:
                file[groupname].attrs[ key ] = data


    def save(self, filename=None, image=None, sources=None, overwrite=True, **kwargs):
        """Save the data of this Cutouts object into a file.

        Parameters
        ----------
        filename: str, optional
            The (relative/full path) filename to save to. If not given, will use the default filename.

        image: Image
            The sub image that these cutouts are associated with.  (Needed to determine filepath.)

        sources: SourceList
            The SourceList (detections on sub image) that these cutouts are associated with.

        kwargs: dict
            Any additional keyword arguments to pass to the FileOnDiskMixin.save method.
        """
        if len(self.co_dict) == 0:
            return None  # do nothing

        proper_length = sources.num_sources
        if len(self.co_dict) != proper_length:
            raise ValueError(f"Trying to save cutouts dict with {len(self.co_dict)}"
                             f" subdicts, but SourceList has {proper_length} sources")

        for key, value in self.co_dict.items():
            if not isinstance(value, dict):
                raise TypeError("Each entry of co_dict must be a dictionary")

        if filename is None:
            filename = self.invent_filepath( image=image )

        self.filepath = filename

        fullname = os.path.join(self.local_path, filename)
        self.safe_mkdir(os.path.dirname(fullname))

        if not overwrite and os.path.isfile(fullname):
            raise FileExistsError(f"The file {fullname} already exists and overwrite is False.")

        if self.format == 'hdf5':
            with h5py.File(fullname, 'a') as file:
                for key, value in self.co_dict.items():
                    self._save_dataset_dict_to_hdf5(value, file, key)
        elif self.format == 'fits':
            raise NotImplementedError('Saving cutouts to fits is not yet implemented.')
        elif self.format in ['jpg', 'png']:
            raise NotImplementedError('Saving cutouts to jpg or png is not yet implemented.')
        else:
            raise TypeError(f"Unable to save cutouts file of type {self.format}")

        # make sure to also save using the FileOnDiskMixin method
        FileOnDiskMixin.save(self, fullname, overwrite=overwrite, **kwargs)

    def _load_dataset_dict_from_hdf5(self, file, groupname):
        """Load the dataset from an HDF5 group into one co_subdict and return.

        Parameters
        ----------
        file: h5py.File
            The open HDF5 file to load from.
        groupname: str
            The name of the group to load from. This should be "source_index_<number>"
        """

        co_subdict = {}
        found_data = False
        for att in self.get_data_array_attributes(): # remove source index for dict soon [?]
            if att in file[groupname]:
                found_data = True
                co_subdict[att] = np.array(file[f'{groupname}/{att}'])
        for att in self.get_data_scalar_attributes():
            if att in file[groupname].attrs:
                co_subdict[att] = file[groupname].attrs[att]
        if found_data:
            return co_subdict

    def load_one_co_dict(self, groupname, filepath=None):
        """Load data subdict for a single cutout into this Cutouts co_dict. This allows
        a measurement to request only the information relevant to that object, rather
        than populating the entire dictionary when we only need one subdict.
        """

        if filepath is None:
            filepath = self.get_fullpath()

        with h5py.File(filepath, 'r') as file:
            co_subdict = self._load_dataset_dict_from_hdf5(file, groupname)
            if co_subdict is not None:
                self.co_dict[groupname] = co_subdict
        return None

    def load(self, filepath=None):
        """Load the data for this cutout from a file.

        Parameters
        ----------
        filepath: str, optional
            The (relative/full path) filename to load from.
            If not given, will use self.get_fullpath() to get the filename.
        """

        if filepath is None:
            filepath = self.get_fullpath( nofile=False )

        if filepath is None:
            raise ValueError("Could not find filepath to load")

        self.co_dict = Co_Dict()
        self.co_dict.cutouts = self

        if os.path.exists(filepath):
            if self.format == 'hdf5':
                with h5py.File(filepath, 'r') as file:
                    # quirk: the resulting dict is sorted alphabetically... likely harmless
                    for groupname in file:
                        self.co_dict[groupname] = self._load_dataset_dict_from_hdf5(file, groupname)


    def get_upstreams( self, session=None ):
        """Return upstreams of this cutouts object.

        This will be the SourceList that is the detections from which this cutout was made.
        """

        with SmartSession( session ) as session:
            return session.scalars( sa.Select( SourceList ).where( SourceList._id == self.sources_id ) ).all()

    def get_downstreams( self, session=None, siblings=False ):
        """Return downstreams of this cutouts object.

        Only gets immediate downstreams; does not recurse.  (As per the
        docstring in SeeChangeBase.get_downstreams.)

        Returns a list of Measurements objects.

        """

        # Avoid circular imports
        from models.measurements import Measurements

        with SmartSession( session ) as sess:
            measurements = sess.query( Measurements ).filter( Measurements.cutouts_id==self.id )

        return list( measurements )
