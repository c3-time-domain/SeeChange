import os

import numpy as np
import pandas as pd

from collections import defaultdict

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY

import astropy.table

from models.base import Base, SmartSession, AutoIDMixin, FileOnDiskMixin, SeeChangeBase, HasBitFlagBadness
from models.image import Image
from models.enums_and_bitflags import (
    SourceListFormatConverter,
    source_list_badness_inverse,
)
from util.util import ensure_file_does_not_exist
from util.logger import SCLogger
import util.ldac


class SourceList(Base, AutoIDMixin, FileOnDiskMixin, HasBitFlagBadness):
    """Encapsulates a source list.

    By default, uses SExtractor.

    Note that internal storage stores image coordinates using the numpy
    convention, i.e. 0-offset.  The load() and save() methods have code
    that converts to the standard sextractor 1-offset when reading and
    writing FITS files, so this should hopefully be handled
    transparently.

    """

    __tablename__ = 'source_lists'

    __table_args__ = (
        UniqueConstraint('image_id', 'provenance_id', name='_source_list_image_provenance_uc'),
    )

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
        sa.ForeignKey('images.id', ondelete='CASCADE', name='source_lists_image_id_fkey'),
        nullable=False,
        index=True,
        doc="ID of the image this source list was generated from. "
    )

    image = orm.relationship(
        Image,
        lazy='selectin',
        cascade='save-update, merge, refresh-expire, expunge',
        passive_deletes=True,
        doc="The image this source list was generated from. "
    )

    is_sub = association_proxy('image', 'is_sub')
    is_coadd = association_proxy('image', 'is_coadd')

    aper_rads = sa.Column(
        ARRAY( sa.REAL, zero_indexes=True ),
        nullable=True,
        default=None,
        index=False,
        doc="Radius of apertures used for aperture photometry in pixels."
    )

    _inf_aper_num = sa.Column(
        sa.SMALLINT,
        nullable=True,
        default=None,
        index=False,
        doc="Which element of aper_rads to use as the 'infinite' aperture; null = last one"
    )

    @property
    def inf_aper_num( self ):
        if self._inf_aper_num is None:
            if self.aper_rads is None:
                return None
            else:
                return len(self.aper_rads) - 1
        else:
            return self._inf_aper_num

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

    def _get_inverse_badness(self):
        """Get a dict with the allowed values of badness that can be assigned to this object"""
        return source_list_badness_inverse

    def __init__(self, *args, **kwargs):
        FileOnDiskMixin.__init__(self, *args, **kwargs)
        SeeChangeBase.__init__(self)  # don't pass kwargs as they could contain non-column key-values

        self._data = None
        self._bitflag = 0
        self._info = None
        self._is_star = None
        self.wcs = None
        self.zp = None
        self.cutouts = None
        self.measurements = None

        # manually set all properties (columns or not)
        self.set_attributes_from_dict(kwargs)

    def __setattr__(self, key, value):
        if key == 'image' and value is not None:
            self._upstream_bitflag = value.bitflag
        # TODO: what happens if setting the image_id instead of the image?

        super().__setattr__(key, value)

    @orm.reconstructor
    def init_on_load(self):
        FileOnDiskMixin.init_on_load(self)
        SeeChangeBase.init_on_load(self)
        self._data = None
        self._info = None
        self._is_star = None

        self.wcs = None
        self.zp = None
        self.cutouts = None
        self.measurements = None

    def merge_all(self, session):
        """Use safe_merge to merge all the downstream products and assign them back to self.

        This includes: wcs, zp, cutouts, measurements.
        Make sure to first assign a merged image to self.image,
        otherwise SQLA will use that relationship to merge a new image,
        which will be different from the one we want to merge into.

        Must provide a session to merge into. Need to commit at the end.

        Returns the merged SourceList with its products on the same session.
        """
        new_sources = self.safe_merge(session=session)
        session.flush()
        for att in ['wcs', 'zp']:
            sub_obj = getattr(self, att, None)
            if sub_obj is not None:
                sub_obj.sources = new_sources  # make sure to first point this relationship back to new_sources
                sub_obj.sources_id = new_sources.id  # make sure to first point this relationship back to new_sources
                if sub_obj not in session:
                    sub_obj = sub_obj.safe_merge(session=session)
                setattr(new_sources, att, sub_obj)

        for att in ['cutouts', 'measurements']:
            sub_obj = getattr(self, att, None)
            if sub_obj is not None:
                new_list = []
                for item in sub_obj:
                    item.sources = new_sources  # make sure to first point this relationship back to new_sources
                    new_list.append(session.merge(item))
                setattr(new_sources, att, new_list)

        return new_sources

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

        if isinstance(value, pd.DataFrame):
            value = value.to_records(index=False)

        if not isinstance(value, (np.ndarray, astropy.table.Table)) or value.dtype.names is None:
            raise TypeError("data must be a pandas.DataFrame, astropy.table.Table or numpy.recarray")

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
        elif self.format == 'filter':
            return self.data['x']
        else:
            raise ValueError( "Unknown format {self.format}" )

    @property
    def y( self ):
        """A numpy array with 0-0ffset based y values of sources"""
        if self.format == 'sextrfits':
            return self.data['Y_IMAGE']
        elif self.format == 'sepnpy':
            return self.data['y']
        elif self.format == 'filter':
            return self.data['y']
        else:
            raise ValueError( "Unknown format {self.format}" )

    @property
    def varx( self ):
        """A numpy array with variances on y position"""
        if self.format == 'sextrfits':
            return self.data['ERRY2_IMAGE']
        elif self.foramt == 'sepnpy':
            # The sep documentation says this is "Second Moment Errors",
            # which may not really be what we want.
            return self.data['erry2']
        elif self.format == 'filter':
            return None
        else:
            raise ValueError( "Unknown format {self.format}" )

    @property
    def vary( self ):
        """A numpy array with variances on x position"""
        if self.format == 'sextrfits':
            return self.data['ERRX2_IMAGE']
        elif self.foramt == 'sepnpy':
            # The sep documentation says this is "Second Moment Errors",
            # which may not really be what we want.
            return self.data['errx2']
        elif self.format == 'filter':
            return None
        else:
            raise ValueError( "Unknown format {self.format}" )

    @property
    def errx( self ):
        """A numpy array with uncertainties on x position"""
        return np.sqrt( self.varx ) if self.varx is not None else None

    @property
    def erry( self ):
        """A numpy array with uncertainties on y position"""
        return np.sqrt( self.vary ) if self.vary is not None else None

    @property
    def good( self ):
        """A numpy array of boolean with length num_sources.

        Each element of returned array corresponds to the corresponding
        element of the arrays returned by the x and y properties, the
        apfluxadu() function, etc.

        True means the object is "good"; False means it's "bad".  Bad
        usually means that there was a saturated pixel, there was as bad
        pixel within some defined area around the center, or there was
        some issue with the extraction (which could be deblending, too
        close to the edge, etc.).

        For sextractor, "bad" is anything that has FLAGS != 0, or that
        has IMAFLAGS_ISO & 0x7fff != 0 (the bitwise AND chosen because
        empirically many objects have bit 0x8000 set; this probably is
        an issue having to do with signed vs. unsigned integers, and
        saving and loading of the FITS files, and should be
        investigated).

        """

        if self.format != 'sextrfits':
            raise NotImplementedError( f"good not currently implemented for format {self.format}" )

        return ( self.data['IMAFLAGS_ISO'] & 0x7fff == 0 ) & ( self.data['FLAGS'] == 0 )

    @property
    def is_star( self ):
        """A numpy array of booleans with length num_sources.

        Each element of returned array corresponds to the corresponding
        element of the arrays returned by the x and y properties, the
        apfluxadu() function, etc.

        True means the object is likely a star, under the assumption
        that the image is clean... but see below.

        Notes for SExtractor:

        SExtrator has two different star/galaxy cateogorizers, CLASS_STAR and SPREAD_MODEL:

          https://sextractor.readthedocs.io/en/latest/Position.html#class-star-def
          https://sextractor.readthedocs.io/en/latest/Model.html#spread-model-def

        SPREAD_MODEL is the more reliable one (based on the
        documentation, and also based on experimentation with one test
        image); CLASS_STAR that test images misses most of the stars.
        However, SPREAD_MODEL takes a lot longer to run (see the
        documentation linked above for a description of what it does).
        Runtime on the test image goes from a few seconds to roughly a
        minute.

        Right now, the code doesn't run SPREAD_MODEL, and the
        classification below is based on CLASS_STAR.  As such, this
        classification should not be considered very reliable.

        """

        if self._is_star is not None:
            return self._is_star

        if self.format != 'sextrfits':
            raise NotImplementedError( f'is_star is only implemented for format sextrfits' )

        # epsilon_2 = 5e-3 ** 2
        # kappa_2 = 4 ** 2
        # thresh = np.sqrt( epsilon_2 + kappa_2 * self.data['SPREADERR_MODEL']**2 )
        # self._is_star = self.data['SPREAD_MODEL'] < thresh

        self._is_star = self.data['CLASS_STAR'] > 0.8

        return self._is_star

    def apfluxadu( self, apnum=0, ap=None ):
        """Return two numpy arrays with aperture flux values and errors

        Parameters
        ----------
          apnum : int, default 0
            The number of the aperture in the list of apertures in
            aper_rads to use.  Ignored if ap is not None.

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
                SCLogger.warning( "Multiple apertures match {ap}; choosing the first one in the list." )
            apnum = w[0]

        if len(self.aper_rads) == 1:
            # In this case, the table has a single value, so will be a 1d array
            return self.data['FLUX_APER'], self.data['FLUXERR_APER']
        else:
            return self.data['FLUX_APER'][:, apnum], self.data['FLUXERR_APER'][:, apnum]

    def psffluxadu( self ):
        """Return two numpy arrays with psf-weighted flux values and errors.

        Returns
        -------
          flux, dflux : numpy arrays

        """

        if self.format != 'sextrfits':
            raise NotImplementedError( f"Not currently implemented for format {self.format}" )
        if 'FLUX_PSF' not in self.data.dtype.names:
            raise ValueError( "Source list doesn't have PSF photometry" )
        return self.data['FLUX_PSF'], self.data['FLUXERR_PSF']

    def calc_aper_cor( self, aper_num=0, inf_aper_num=None, min_stars=20 ):
        """Calculate an aperture correction based on the photometry in this source list.

        The aperture correction apercor is defined so that for a star (or other point source):

           mag = -2.5*log10(fluxadu) + zeropoint + apercor

        where zeropoint is determined through photometric calibration.
        apercor will in general be negative because an aperture will
        have less than the total flux of a star.

        Parameters
        ----------
          aper_num: int, default 0
            The index into self.aper_rads to calculate the aperture correction for.

          inf_aper_num: int
            The index into self.aper_rads to use as the "infinite" aperture.  If None,
            will use self.inf_aper_num

          min_stars: int, default 20
            Must have at least this many stars to measure the aperture correction from.

        Returns
        -------
          apercor: float

        """

        if inf_aper_num is None:
            inf_aper_num = self.inf_aper_num
        if inf_aper_num is None:
            raise RuntimeError( f"Can't determine which aperture to use as the \"infinite\" aperture" )
        if ( inf_aper_num < 0 ) or ( inf_aper_num >= len(self.aper_rads) ):
            raise ValueError( f"inf_aper_num {inf_aper_num} is outside available list of {len(self.aper_rads)}" )

        bigflux, bigfluxerr = self.apfluxadu( apnum=inf_aper_num )
        smallflux, smallfluxerr = self.apfluxadu( apnum=aper_num )
        wgood = self.good & ( bigflux > 5.*bigfluxerr ) & ( smallflux > 5.*smallfluxerr )

        if wgood.sum() < min_stars:
            raise RuntimeError( f'Only {wgood.sum()} stars, less than the minimum of {min_stars} '
                                f'requested for measuring the aperture correction.' )

        bigflux = bigflux[wgood]
        bigfluxerr = bigfluxerr[wgood]
        smallflux = smallflux[wgood]
        smallfluxerr = smallfluxerr[wgood]

        rat = bigflux / smallflux
        ratvar = ( bigfluxerr / smallflux ) **2 + ( smallfluxerr * bigflux / (smallflux)**2 ) **2
        meanrat = ( rat / ratvar ).sum() / ( 1. / ratvar ).sum()

        return -2.5 * np.log10( meanrat )

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

        if self.format in ['sepnpy', 'filter']:
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
                    aps.append( info[kw] / 2. )

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
        if self.provenance is None:
            raise RuntimeError( f"Can't invent a filepath for sources without a provenance" )

        filename = self.image.filepath
        if filename is None:
            filename = self.image.invent_filepath()

        if filename.endswith(('.fits', '.h5', '.hdf5')):
            filename = os.path.splitext(filename)[0]

        filename += '.sources_'
        self.provenance.update_id()
        filename += self.provenance.id[:6]
        if self.format in ['sepnpy', 'filter']:
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

        if self.format in ["sepnpy", "filter"]:
            np.save(fullname, self.data)
        elif self.format == 'sextrfits':
            data = self._convert_to_sextractor_for_saving( self.data )
            util.ldac.save_table_as_ldac( astropy.table.Table(data), fullname, imghdr=self.info, overwrite=True )
        else:
            raise NotImplementedError( f"Don't know how to save source lists of type {self.format}" )

        self.num_sources = len( self.data )
        super().save(fullname, **kwargs)

    def free( self, ):
        """Free loaded source list memory.

        Wipe out the data and info fields, freeing memory.  Depends on
        python garbage collection, so if there are other references to
        those objects, the memory won't actually be freed.

        """
        self._data = None
        self._info = None

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
          arr: numpy array with named records, or astropy.table.Table

        Returns
        -------
          A copy of arr, with pixel positions incremented by 1
        """
        cols = { 'XMIN_IMAGE', 'XMAX_IMAGE', 'YMIN_IMAGE', 'YMAX_IMAGE',
                 'X_IMAGE', 'Y_IMAGE', 'XPEAK_IMAGE', 'YPEAK_IMAGE',
                 'XWIN_IMAGE', 'YWIN_IMAGE' }
        if isinstance( arr, np.ndarray ):
            cols = cols.intersection( set(arr.dtype.names) )
            arr = np.copy( arr, subok=True )
        elif isinstance( arr, astropy.table.Table ):
            cols = cols.intersection( set(arr.columns) )
            arr = astropy.table.Table( arr )

        for col in cols:
            arr[col] +=1

        return arr

    def ds9_regfile( self, regfile, color='green', radius=2, width=2, whichsources='all', clobber=True ):
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

        whichsources: str, one of 'all', 'stars', 'nonstars'
           Which objects to write regions for.  If 'all', all of them.
           If 'stars', only the ones for which self.is_star is True.  If
           'nonstar', only the ones for which self.is_star is False.

        clobber: bool, default True
           If the file exists, overwrite it

        """
        ensure_file_does_not_exist( regfile, delete=clobber )

        if whichsources == 'stars':
            which = self.is_star
        elif whichsources == 'nonstars':
            which = ~self.is_star
        elif whichsources == 'all':
            which = np.full( ( self.num_sources, ), True )
        else:
            raise ValueError( f'whichsources must be one of all, stars, or nonstars, not {whichsources}' )

        with open( regfile, "w" ) as ofp:
            for x, y, use in zip( self.x, self.y, which ):
                if use:
                    # +1 to go from C-coordinates to FITS-coordinates
                    ofp.write( f"image;circle({x+1},{y+1},{radius}) # color={color} width={width}\n" )

    def get_upstreams(self, session=None):
        """Get the image that was used to make this source list. """
        with SmartSession(session) as session:
            return session.scalars(sa.select(Image).where(Image.id == self.image_id)).all()

    def get_downstreams(self, session=None, siblings=False):
        """Get all the data products that are made using this source list.

        If siblings=True then also include the PSFs, WCSes, ZPs and background objects
        that were created at the same time as this SourceList.
        """
        from models.psf import PSF
        from models.world_coordinates import WorldCoordinates
        from models.zero_point import ZeroPoint
        from models.cutouts import Cutouts
        from models.provenance import Provenance

        with SmartSession(session) as session:
            subs = session.scalars(
                sa.select(Image).where(
                    Image.provenance.has(Provenance.upstreams.any(Provenance.id == self.provenance.id)),
                    Image.upstream_images.any(Image.id == self.image_id),
                )
            ).all()
            output = subs

            if self.is_sub:
                cutouts = session.scalars(sa.select(Cutouts).where(Cutouts.sources_id == self.id)).all()
                output += cutouts
            elif siblings:  # for "detections" we don't have siblings
                psfs = session.scalars(
                    sa.select(PSF).where(PSF.image_id == self.image_id, PSF.provenance_id == self.provenance_id)
                ).all()
                if len(psfs) != 1:
                    raise ValueError(f"Expected exactly one PSF for SourceList {self.id}, but found {len(psfs)}")

                # TODO: add background object

                wcs = session.scalars(sa.select(WorldCoordinates).where(WorldCoordinates.sources_id == self.id)).all()
                if len(wcs) != 1:
                    raise ValueError(
                        f"Expected exactly one WorldCoordinates for SourceList {self.id}, but found {len(wcs)}"
                    )
                zps = session.scalars(sa.select(ZeroPoint).where(ZeroPoint.sources_id == self.id)).all()
                if len(zps) != 1:
                    raise ValueError(
                        f"Expected exactly one ZeroPoint for SourceList {self.id}, but found {len(zps)}"
                    )
                output += psfs + wcs + zps

        return output

    def show(self, **kwargs):
        """Show the source positions on top of the image.

        This is a convenience function that uses the Image.show() method.
        The arguments are passed into the Image.show() method.

        """
        import matplotlib.pyplot as plt

        if self.image is None:
            raise ValueError("Can't show source list without an image")
        self.image.show(**kwargs)
        plt.plot(self.x, self.y, 'ro', markersize=5, fillstyle='none')


# TODO: replace these with association proxies?
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
    setattr(
        SourceList,
        att,
        property(fget=lambda self, att=att: getattr(self.image, att) if self.image is not None else None)
    )
