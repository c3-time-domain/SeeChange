import os
import pathlib

from functools import partial

import numpy as np
import pandas as pd

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql.functions import coalesce

import astropy.table
from astropy.io import fits

from models.base import Base, AutoIDMixin, FileOnDiskMixin, SeeChangeBase, _logger
from models.image import Image
from models.enums_and_bitflags import (
    SourceListFormatConverter,
    bitflag_to_string,
    string_to_bitflag,
    data_badness_dict,
    source_list_badness_inverse,
)
from util.util import ensure_file_does_not_exist
import util.ldac

class SourceList(Base, AutoIDMixin, FileOnDiskMixin):
    """Encapsulates a source list.

    By default, uses SExtractor.

    Note that internal storage stores image coordinates using the numpy
    convention, i.e. 0-offset.  The load() and save() methods have code
    that converts to the standard sextractor 1-offset when reading and
    writing FITS files, so this should hopefully be handled
    transparently.

    """

    __tablename__ = 'source_lists'

    _format = sa.Column(
        sa.SMALLINT,
        nullable=False,
        default=SourceListFormatConverter.convert('sextrfits'),
        doc="Format of the file on disk. Should be sepnpy or sextrfits. "
            "Saved as integer but is converter to string when loaded. "
    )

    @hybrid_property
    def format(self):
        return SourceListFormatConverter.convert(self._format)

    @format.expression
    def format(cls):
        # ref: https://stackoverflow.com/a/25272425
        return sa.case(SourceListFormatConverter.dict, value=cls._format)

    @format.setter
    def format(self, value):
        self._format = SourceListFormatConverter.convert(value)

    image_id = sa.Column(
        sa.ForeignKey('images.id', name='source_lists_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the image this source list was generated from. "
    )

    image = orm.relationship(
        'Image',
        lazy='selectin',
        doc="The image this source list was generated from. "
    )

    @hybrid_property
    def is_sub(self):
        """Whether this source list is from a subtraction image (detections),
        or from a regular image (sources, the default).
        """
        if self.image is None:
            return None
        else:
            return self.image.is_sub

    @is_sub.expression
    def is_sub(cls):
        return sa.select(Image.is_sub).where(Image.id == cls.image_id).label('is_sub')

    aper_rads= sa.Column(
        sa.ARRAY( sa.REAL ),
        nullable=True,
        default=None,
        index=False,
        doc="Radius of apertures used for aperture photometry in pixels."
    )

    num_sources = sa.Column(
        sa.Integer,
        nullable=False,
        index=True,
        doc="Number of sources in this source list. "
    )

    provenance_id = sa.Column(
        sa.ForeignKey('provenances.id', ondelete="CASCADE", name='source_lists_provenance_id_fkey'),
        nullable=False,
        index=True,
        doc=(
            "ID of the provenance of this source list. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this source list. "
        )
    )

    provenance = orm.relationship(
        'Provenance',
        cascade='save-update, merge, refresh-expire, expunge',
        lazy='selectin',
        doc=(
            "Provenance of this source list. "
            "The provenance will contain a record of the code version"
            "and the parameters used to produce this source list. "
        )
    )

    _bitflag = sa.Column(
        sa.BIGINT,
        nullable=False,
        default=0,
        index=True,
        doc='Bitflag for this source list. Good source lists have a bitflag of 0. '
            'Bad source list are each bad in their own way (i.e., have different bits set). '
            'Will include all the bits from data used to make this source list '
            '(e.g., the exposure it is based on). '
    )

    @hybrid_property
    def bitflag(self):
        return self._bitflag | self.image.bitflag

    @bitflag.inplace.expression
    @classmethod
    def bitflag(cls):
        stmt = sa.select(coalesce(cls._bitflag, 0).op('|')(Image.bitflag))
        stmt = stmt.where(cls.image_id == Image.id)
        stmt = stmt.scalar_subquery()
        return stmt

    @bitflag.setter
    def bitflag(self, value):
        self._bitflag = value

    @property
    def badness(self):
        """
        A comma separated string of keywords describing
        why this data is not good, based on the bitflag.
        This includes all the reasons this data is bad,
        including the parent data models that were used
        to create this data (e.g., the Exposure underlying
        the Image).
        """
        return bitflag_to_string(self.bitflag, data_badness_dict)

    @badness.setter
    def badness(self, value):
        """Set the badness for this image using a comma separated string. """
        self.bitflag = string_to_bitflag(value, source_list_badness_inverse)

    def append_badness(self, value):
        """Add some keywords (in a comma separated string)
        describing what is bad about this image.
        The keywords will be added to the list "badness"
        and the bitflag for this image will be updated accordingly.
        """
        self.bitflag |= string_to_bitflag(value, source_list_badness_inverse)

    description = sa.Column(
        sa.Text,
        nullable=True,
        doc='Free text comment about this source list, e.g., why it is bad. '
    )

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self._data = None
        self._bitflag = 0
        self._info = None

        # manually set all properties (columns or not)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @orm.reconstructor
    def init_on_load(self):
        Base.init_on_load(self)
        FileOnDiskMixin.init_on_load(self)
        self._data = None
        self._info = None

    def __repr__(self):
        output = (
            f'<SourceList(id={self.id}, '
            f'format={self.format}, '
            f'image_id={self.image_id}, '
            f'is_sub={self.is_sub}), '
            f'num_sources= {self.num_sources}>'
        )

        return output

    @property
    def data(self):
        """The data in this source list. A table of sources and their properties."""
        if self._data is None and self.filepath is not None:
            self.load()
        return self._data

    @data.setter
    def data(self, value):
        # TODO: add astropy table?
        if isinstance(value, pd.DataFrame):
            value = value.to_records(index=False)

        if not isinstance(value, np.ndarray) or value.dtype.names is None:
            raise TypeError("data must be a pandas DataFrame or numpy recarray")

        self._data = value
        self.num_sources = len(value)

    @property
    def info(self):
        """Additional info associated with this source list.

        For example, for the sextrfits format, this would be the header
        records from HDU 1 of the output of SExtractor (stored as an
        astropy.io.fits.header.Header)

        """
        if ( self._info is None ) and ( self.filepath is not None ):
            self.load()
        return self._info

    @info.setter
    def info(self, value):
        """Set the info property.  Does no type checking."""
        self._info = value

    @property
    def x( self ):
        """A numpy array with 0-offset based x values of sources"""
        if self.format == 'sextrfits':
            return self.data['X_IMAGE']
        elif self.format == 'sepnpy':
            return self.data['x']
        else:
            raise ValueError( "Unknown format {self.format}" )

    @property
    def y( self ):
        """A numpy array with 0-0ffset based y values of sources"""
        if self.format == 'sextrfits':
            return self.data['Y_IMAGE']
        elif self.format == 'sepnpy':
            return self.data['x']
        else:
            raise ValueError( "Unknown format {self.format}" )

    @property
    def ra( self ):
        """RA of all sources in degrees."""
        if self.format == 'sextrfits':
            return self.data['X_WORLD']
        else:
            raise ValueError( "Can't get RA for source list format {self.format}" )

    @property
    def dec( self ):
        """Dec of all sources in degrees."""
        if self.format == 'sextrfits':
            return self.data['Y_WORLD']
        else:
            raise ValueError( "Can't get Dec for source list format {self.format}" )

    def apfluxadu( self, apnum=0, ap=None ):
        """Return two numpy arrays with aperture flux values and errors

        Parameters
        ----------
          apnum : int, default 0
            The number of the aperture in the list of apertures in
            aper_rads to use.  Ignroed if ap is not None.

          ap: float, default None
            If not None, look for an aperture that's within 0.01 pixels
            of this and return flux in apertures of that radius.  Raises
            an exception if such an aperture doesn't apear in aper_rads

        Returns
        -------
          flux, dflux : numpy arrays
        """

        if self.format != 'sextrfits':
            raise NotImplementedError( f"Not currently implemented for format {self.format}" )

        if ap is None:
            if ( self.aper_rads is None ) or ( apnum < 0 ) or ( apnum >= len(self.aper_rads) ):
                raise ValueError( f"Aperture radius number {apnum} doesn't exist." )
        else:
            w = np.where( np.abs( np.array( self.aper_rads) - ap ) < 0.01 )[0]
            if len(w) == 0:
                raise ValueError( f"Can't find an aperture of radius {ap} pixels; "
                                  f"available apertures = {self.aper_rads}" )
            if len(w) > 1:
                _logger.warning( "Multiple apertures match {ap}; choosing the first one in the list." )
            apnum = w[0]

        if len(self.aper_rads) == 1:
            # In this case, the table has a single value, so will be a 1d array
            return self.data['FLUX_APER'], self.data['FLUXERR_APER']
        else:
            return self.data['FLUX_APER'][:, apnum], self.data['FLUXERR_APER'][:, apnum]


    def load(self, filepath=None):
        """Load this source list from the file.

        Updates self._data and self._info.

        Will update self.aper_rads and self.num_sources if they are null;
        otherwise, will throw an exception if they are inconsistent with
        what is loaded.

        Parameters
        ----------
          filepath: str, Path, or None
             File to read.  Format of the file must match self.format.
             If None, will load the file retunred by self.get_fullpath()

        """

        if filepath is None:
            filepath = self.get_fullpath()

        if self.format == 'sepnpy':
            if self.aper_rads is not None:
                raise ValueError( f"self.aper_rads is not None for a sepnpy format file" )
            self._info = []
            data = np.load( filepath )
            if self.num_sources is None:
                self.num_sources = len( data )
            else:
                if self.num_sources != len( data ):
                    raise ValueError( f"self.num_sources={self.num_sources} but len(self.data)={len(data)}" )
            self._data = data

        elif self.format == 'sextrfits':
            info, tbl = util.ldac.get_table_from_ldac( filepath, frame=1, imghdr_as_header=True )
            tbl = tbl.as_array()
            tbl = self._convert_from_sextractor_to_numpy( tbl )

            if self.num_sources is None:
                self.num_sources = len( tbl )
            else:
                if self.num_sources != len( tbl ):
                    raise ValueError( f"self.num_sources={self.num_sources} but the sextractor file "
                                      f"had {len(tbl)} sources" )

            aps = []
            for apn in range( 1, 5 ):
                kw = f'SEXAPED{apn}'
                if kw in info:
                    if info[kw] == 0.:
                        break
                    aps.append( info[kw] )

            if self.aper_rads is None:
                if ( len( tbl['FLUX_APER'].shape ) > 1 ) and ( tbl['FLUX_APER'].shape[1] > 4 ):
                    raise ValueError( f"Can't load sextractor file, has {tbl['FLUX_APER'].shape[1]} "
                                      f"apertures, but sextractor only saves the radii of the first four." )
                self.aper_rads = aps
            else:
                # SExtractor annoyingly only saves the radii of the first four apertures
                # it used.  So, we're just going to blindly trust that if there are more,
                # they're right.
                if len( self.aper_rads ) != ( tbl['FLUX_APER'].shape[1] if tbl['FLUX_APER'].ndim==2 else 1 ):
                    raise ValueError( f"self.aper_rads doesn't match the number of apertures "
                                      f"found in {filepath}" )
                if ( ( ( len(self.aper_rads) <= 4 ) and ( len( aps ) != len( self.aper_rads ) ) )
                     or
                     ( not ( np.abs( np.array( aps ) - np.array( self.aper_rads[:len(aps)] ) ) < 0.01 ).all() )
                    ):
                    raise ValueError( f"self.aper_rads {self.aper_rads} doesn't match sextractor file "
                                      f"aperture radii {aps}" )

            self._info = info
            self._data = tbl
        else:
            raise NotImplementedError( f"Don't know how to load source lists of format {self.format}" )

    def invent_filepath( self ):
        if self.image is None:
            raise RuntimeError( f"Can't invent a filepath for sources without an image" )

        filename = self.image.filepath
        if filename is None:
            filename = self.image.invent_filepath()

        if filename.endswith(('.fits', '.h5', '.hdf5')):
            filename = os.path.splitext(filename)[0]

        filename += '.sources'
        if self.format == 'sepnpy':
            filename += '.npy'
        elif self.format == 'sextrfits':
            filename += '.fits'
        else:
            raise TypeError( f"Unable to create a filepath for sources file of type {self.format}" )

        return filename

    def save(self, **kwargs):
        """Save the data table to a file on disk.

        Updates self.filepath (if it is None) and self.num_sources
        """

        if self.data is None:
            raise ValueError("Cannot save source list without data")

        if self.filepath is None:
            self.filepath = self.invent_filepath()

        fullname = os.path.join(self.local_path, self.filepath)
        self.safe_mkdir(os.path.dirname(fullname))

        if self.format == "sepnpy":
            np.save(fullname, self.data)
        elif self.format == 'sextrfits':
            data = self._convert_to_sextractor_for_saving( self.data )
            util.ldac.save_table_as_ldac( astropy.table.Table(data), fullname, imghdr=self.info, overwrite=True )
        else:
            raise NotImplementedError( f"Don't know how to save source lists of type {self.format}" )

        self.num_sources = len( self.data )
        super().save(fullname, **kwargs)

    @staticmethod
    def _convert_from_sextractor_to_numpy( arr, copy=False ):
        """Convert from 1-offset to 0-offset coordinates.

        Parameters
        ----------
          arr: numpy array with named records
          copy: if True, copy arr before modfying it

        Returns
        -------
        numpy array with named records

        """
        if copy:
            arr = np.copy( arr, subok=True )
        for col in arr.dtype.names:
            # I hope this is an exhaustive list of sextractor image position columns
            # (We *don't* want to modify the image error, or shape parameter, columns,
            # which is why we can't just search for *_IMAGE.)
            if col in [ 'XMIN_IMAGE', 'XMAX_IMAGE', 'YMIN_IMAGE', 'YMAX_IMAGE',
                        'X_IMAGE', 'Y_IMAGE', 'XPEAK_IMAGE', 'YPEAK_IMAGE',
                        'XWIN_IMAGE', 'YWIN_IMAGE' ]:
                arr[col] -=1
        return arr

    @staticmethod
    def _convert_to_sextractor_for_saving( arr ):
        """Convert array from 0-offset to 1-offset coordinates.

        Parmaeters
        ----------
          arr: numpy array with named records

        Returns
        -------
          numpy array with named records; will be a copy, arr is not touched
        """
        arr = np.copy( arr, subok=True )
        for col in arr.dtype.names:
            if col in [ 'XMIN_IMAGE', 'XMAX_IMAGE', 'YMIN_IMAGE', 'YMAX_IMAGE',
                        'X_IMAGE', 'Y_IMAGE', 'XPEAK_IMAGE', 'YPEAK_IMAGE',
                        'XWIN_IMAGE', 'YWIN_IMAGE' ]:
                arr[col] +=1
        return arr


    def ds9_regfile( self, regfile, color='green', radius=2, width=2, clobber=True ):
        """Write a DS9 region file with circles on the sources.

        See https://ds9.si.edu/doc/ref/region.html for file format

        Parameters
        ----------
        regfile: str or Path
           The output region file
        color: str, default 'green'
           The color to use in the region file (using something standard to DS9)
        radius: float
           The radius of the circles in pixels
        width: float
           The width of the circle line (in whatever unigs DS9 uses)
        clobber: bool, default True
           If the file exists, overwrite it
        """
        ensure_file_does_not_exist( regfile, delete=clobber )

        data = self.data
        with open( regfile, "w" ) as ofp:
            for x, y in zip( self.x, self.y ):
                # +1 to go from C-coordinates to FITS-coordinates
                ofp.write( f"image;circle({x+1},{y+1},{radius}) # color={color} width={width}\n" )

# add "property" attributes to SourceList referencing the image for convenience
for att in [
    'section_id',
    'mjd',
    'filter',
    'filter_short',
    'telescope',
    'instrument',
    'instrument_object',
]:
    setattr(SourceList, att, property(fget=lambda self, att=att: getattr(self.image, att) if self.image is not None else None))
