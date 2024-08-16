import math
import copy
import pathlib
import requests
import json
import collections.abc

import numpy as np
import pandas
import astropy.time
from astropy.io import fits

import sqlalchemy as sa

from models.base import SmartSession, FileOnDiskMixin
from models.exposure import Exposure
from models.knownexposure import KnownExposure
from models.instrument import Instrument, InstrumentOrientation, SensorSection, get_instrument_instance
from models.image import Image
from models.datafile import DataFile
from models.provenance import Provenance
from util.config import Config
import util.util
from util.retrydownload import retry_download
from util.logger import SCLogger
from util.radec import radec_to_gal_ecl
from models.enums_and_bitflags import string_to_bitflag, flag_image_bits_inverse


class DECam(Instrument):

    def __init__(self, **kwargs):
        self.name = 'DECam'
        self.telescope = 'CTIO 4.0-m telescope'
        self.aperture = 4.0
        self.focal_ratio = 2.7
        self.square_degree_fov = 3.0
        self.pixel_scale = 0.263
        self.read_time = 20.0
        self.orientation_fixed = True
        self.orientation = InstrumentOrientation.NleftEup
        # read_noise, dark_current, gain, saturation_limit, non_linearity_limit
        # are all approximate values for DECam; it varies by a lot between chips
        self.read_noise = 7.0
        self.dark_current = 0.1
        self.gain = 4.0
        self.saturation_limit = 44000
        self.non_linearity_limit = 44000
        self.allowed_filters = ["g", "r", "i", "z", "Y"]

        # These numbers were measured off of the WCS solution to
        #  c4d_230804_031607_ori.fits as saved by the
        #  decat lensgrinder pipeline.
        # Ra offsets are approximately *linear* degrees -- that is, they
        #  are ΔRA * cos( dec ), where dec is the exposure dec.
        # Chips 31 and 60 are the "bad" chips, and weren't in the
        #  decat database, so their centers used centers of
        #  the nearest aligned chips along each axis.
        #
        # Notice that the "N" chips are to the south and the "S" chips
        # are to the north; this is correct! See:
        # https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/characteristics
        self._chip_radec_off = {
            'S29': { 'ccdnum':  1, 'dra':  -0.30358, 'ddec':   0.90579 },
            'S30': { 'ccdnum':  2, 'dra':   0.00399, 'ddec':   0.90370 },
            'S31': { 'ccdnum':  3, 'dra':   0.31197, 'ddec':   0.90294 },
            'S25': { 'ccdnum':  4, 'dra':  -0.45907, 'ddec':   0.74300 },
            'S26': { 'ccdnum':  5, 'dra':  -0.15079, 'ddec':   0.74107 },
            'S27': { 'ccdnum':  6, 'dra':   0.15768, 'ddec':   0.73958 },
            'S28': { 'ccdnum':  7, 'dra':   0.46585, 'ddec':   0.73865 },
            'S20': { 'ccdnum':  8, 'dra':  -0.61496, 'ddec':   0.58016 },
            'S21': { 'ccdnum':  9, 'dra':  -0.30644, 'ddec':   0.57787 },
            'S22': { 'ccdnum': 10, 'dra':   0.00264, 'ddec':   0.57605 },
            'S23': { 'ccdnum': 11, 'dra':   0.31167, 'ddec':   0.57493 },
            'S24': { 'ccdnum': 12, 'dra':   0.62033, 'ddec':   0.57431 },
            'S14': { 'ccdnum': 13, 'dra':  -0.77134, 'ddec':   0.41738 },
            'S15': { 'ccdnum': 14, 'dra':  -0.46272, 'ddec':   0.41468 },
            'S16': { 'ccdnum': 15, 'dra':  -0.15310, 'ddec':   0.41266 },
            'S17': { 'ccdnum': 16, 'dra':   0.15678, 'ddec':   0.41097 },
            'S18': { 'ccdnum': 17, 'dra':   0.46634, 'ddec':   0.41032 },
            'S19': { 'ccdnum': 18, 'dra':   0.77533, 'ddec':   0.41018 },
            'S8':  { 'ccdnum': 19, 'dra':  -0.77343, 'ddec':   0.25333 },
            'S9':  { 'ccdnum': 20, 'dra':  -0.46437, 'ddec':   0.25010 },
            'S10': { 'ccdnum': 21, 'dra':  -0.15423, 'ddec':   0.24804 },
            'S11': { 'ccdnum': 22, 'dra':   0.15631, 'ddec':   0.24661 },
            'S12': { 'ccdnum': 23, 'dra':   0.46667, 'ddec':   0.24584 },
            'S13': { 'ccdnum': 24, 'dra':   0.77588, 'ddec':   0.24591 },
            'S1':  { 'ccdnum': 25, 'dra':  -0.93041, 'ddec':   0.09069 },
            'S2':  { 'ccdnum': 26, 'dra':  -0.62099, 'ddec':   0.08716 },
            'S3':  { 'ccdnum': 27, 'dra':  -0.31067, 'ddec':   0.08417 },
            'S4':  { 'ccdnum': 28, 'dra':   0.00054, 'ddec':   0.08241 },
            'S5':  { 'ccdnum': 29, 'dra':   0.31130, 'ddec':   0.08122 },
            'S6':  { 'ccdnum': 30, 'dra':   0.62187, 'ddec':   0.08113 },
            'S7':  { 'ccdnum': 31, 'dra':   0.93180, 'ddec':   0.08113 },
            'N1':  { 'ccdnum': 32, 'dra':  -0.93285, 'ddec':  -0.07360 },
            'N2':  { 'ccdnum': 33, 'dra':  -0.62288, 'ddec':  -0.07750 },
            'N3':  { 'ccdnum': 34, 'dra':  -0.31207, 'ddec':  -0.08051 },
            'N4':  { 'ccdnum': 35, 'dra':  -0.00056, 'ddec':  -0.08247 },
            'N5':  { 'ccdnum': 36, 'dra':   0.31077, 'ddec':  -0.08351 },
            'N6':  { 'ccdnum': 37, 'dra':   0.62170, 'ddec':  -0.08335 },
            'N7':  { 'ccdnum': 38, 'dra':   0.93180, 'ddec':  -0.08242 },
            'N8':  { 'ccdnum': 39, 'dra':  -0.77988, 'ddec':  -0.24010 },
            'N9':  { 'ccdnum': 40, 'dra':  -0.46913, 'ddec':  -0.24376 },
            'N10': { 'ccdnum': 41, 'dra':  -0.15732, 'ddec':  -0.24624 },
            'N11': { 'ccdnum': 42, 'dra':   0.15476, 'ddec':  -0.24786 },
            'N12': { 'ccdnum': 43, 'dra':   0.46645, 'ddec':  -0.24819 },
            'N13': { 'ccdnum': 44, 'dra':   0.77723, 'ddec':  -0.24747 },
            'N14': { 'ccdnum': 45, 'dra':  -0.78177, 'ddec':  -0.40426 },
            'N15': { 'ccdnum': 46, 'dra':  -0.47073, 'ddec':  -0.40814 },
            'N16': { 'ccdnum': 47, 'dra':  -0.15836, 'ddec':  -0.41091 },
            'N17': { 'ccdnum': 48, 'dra':   0.15385, 'ddec':  -0.41244 },
            'N18': { 'ccdnum': 49, 'dra':   0.46623, 'ddec':  -0.41260 },
            'N19': { 'ccdnum': 50, 'dra':   0.77755, 'ddec':  -0.41164 },
            'N20': { 'ccdnum': 51, 'dra':  -0.62766, 'ddec':  -0.57063 },
            'N21': { 'ccdnum': 52, 'dra':  -0.31560, 'ddec':  -0.57392 },
            'N22': { 'ccdnum': 53, 'dra':  -0.00280, 'ddec':  -0.57599 },
            'N23': { 'ccdnum': 54, 'dra':   0.30974, 'ddec':  -0.57705 },
            'N24': { 'ccdnum': 55, 'dra':   0.62187, 'ddec':  -0.57650 },
            'N25': { 'ccdnum': 56, 'dra':  -0.47298, 'ddec':  -0.73648 },
            'N26': { 'ccdnum': 57, 'dra':  -0.16038, 'ddec':  -0.73922 },
            'N27': { 'ccdnum': 58, 'dra':   0.15280, 'ddec':  -0.74076 },
            'N28': { 'ccdnum': 59, 'dra':   0.46551, 'ddec':  -0.74086 },
            'N29': { 'ccdnum': 60, 'dra':  -0.31779, 'ddec':  -0.90199 },
            'N30': { 'ccdnum': 61, 'dra':  -0.00280, 'ddec':  -0.90348 },
            'N31': { 'ccdnum': 62, 'dra':   0.30889, 'ddec':  -0.90498 },
        }

        # will apply kwargs to attributes, and register instrument in the INSTRUMENT_INSTANCE_CACHE
        Instrument.__init__(self, **kwargs)

        self.preprocessing_steps_available = [ 'overscan', 'linearity', 'flat', 'illumination', 'fringe' ]
        self.preprocessing_steps_done = []

    @classmethod
    def get_section_ids(cls):

        """
        Get a list of SensorSection identifiers for this instrument.
        We are using the names of the FITS extensions (e.g., N12, S22, etc.).
        See ref: https://noirlab.edu/science/sites/default/files/media/archives/images/DECamOrientation.png
        """
        # CCDs 31 (S7) and 61 (N30) are bad CCDS
        # https://noirlab.edu/science/index.php/programs/ctio/instruments/Dark-Energy-Camera/Status-DECam-CCDs
        n_list = [ f'N{i}' for i in range(1, 32) if i != 30 ]
        s_list = [ f'S{i}' for i in range(1, 32) if i != 7 ]
        return n_list + s_list

    @classmethod
    def check_section_id(cls, section_id):
        """
        Check that the type and value of the section is compatible with the instrument.
        In this case, it must be a string starting with 'N' or 'S' and a number between 1 and 31.
        """
        if not isinstance(section_id, str):
            raise ValueError(f"The section_id must be a string. Got {type(section_id)}. ")

        letter = section_id[0]
        number = int(section_id[1:])

        if letter not in ['N', 'S']:
            raise ValueError(f"The section_id must start with either 'N' or 'S'. Got {letter}. ")

        if not 1 <= number <= 31:
            raise ValueError(f"The section_id number must be in the range [1, 31]. Got {number}. ")

    def get_section_offsets(self, section_id):
        """Find the offset for a specific section.

        For DECam, these offests were determined by using the WCS
        solutions for a given exposure, and then calculated from the
        nominal pixel scale of the instrument.  The exposure center was
        taken as the average of S4 (x=2047, y=2048) and N4 (x=0,
        y=2048).  These pixel offsets do *not* correspond exactly to the
        pixel offsets that you'd get if you laid the chips down flat on
        a table positioned exactly where they are in the camera, and
        measured the centers of each chip.  But, they are the ones you'd
        use to figure out the RA and Dec of the chip centers starting
        with the exposure ra/dec (ASSUMING that it's centered between N4
        and S4) and using the noiminal instrument pixel scale (properly
        including cos(dec)); as of this writing, that nominal instrument
        pixel scale was coded to be 0.263"/pixel, which is the
        three-digit average of what
        https://noirlab.edu/science/programs/ctio/instruments/Dark-Energy-Camera/characteristics
        cites as the center pixel scale (0.2637) and edge pixel scale
        (0.2626).

        Parameters
        ----------
        section_id: int
            The identifier of the section.

        Returns
        -------
        offset_x: int
            The x offset of the section.
        offset_y: int
            The y offset of the section.

        """

        self.check_section_id(section_id)
        if section_id not in self._chip_radec_off:
            raise ValueError( f'Failed to find {section_id} in dictionary of chip offsets' )

        # x increases to the south, y increases to the east
        return ( -self._chip_radec_off[section_id]['ddec'] * 3600. / self.pixel_scale ,
                 self._chip_radec_off[section_id]['dra'] * 3600. / self.pixel_scale )

    def _make_new_section(self, section_id):
        """
        Make a single section for the DECam instrument.
        The section_id must be a valid section identifier (Si or Ni, where i is an int in [1,31])

        Returns
        -------
        section: SensorSection
            A new section for this instrument.
        """
        (dx, dy) = self.get_section_offsets(section_id)
        defective = section_id in { 'N30', 'S7' }
        return SensorSection(section_id, self.name, size_x=2048, size_y=4096,
                             offset_x=dx, offset_y=dy, defective=defective)

    def get_ra_dec_for_section( self, exposure, section_id ):
        if section_id not in self._chip_radec_off:
            raise ValueError( f"Unknown DECam section_id {section_id}" )
        return ( exposure.ra + self._chip_radec_off[section_id]['dra'] / math.cos( exposure.dec * math.pi / 180. ),
                 exposure.dec + self._chip_radec_off[section_id]['ddec'] )

    def get_standard_flags_image( self, section_id ):
        # NOTE : there's a race condition here; multiple
        # processes might try to locally cache the
        # file all at the same time.
        #
        # For now, just not going to worry about it.
        #
        cfg = Config.get()
        ccdnum = f'{self._chip_radec_off[section_id]["ccdnum"]:02d}'
        rempath = pathlib.Path( f'{cfg.value("DECam.calibfiles.bpmbase")}{ccdnum}.fits' )
        filepath = pathlib.Path( FileOnDiskMixin.local_path ) / "DECam_default_calibrators" / "bpm" / rempath.name

        if not filepath.exists():
            url = f'{cfg.value("DECam.calibfiles.urlbase")}{str(rempath)}'
            retry_download( url, filepath )

        with fits.open( filepath, memmap=False ) as hdu:
            rawbpm = hdu[0].data

        # TODO : figure out what the bits mean in this bad pixel mask file!
        #  For now, call anything non-zero as "bad"
        #  (If we ever change this, we have to fix the
        #  flag_image_bits dictionary in enums_and_bitflags,
        #  and everything that has used it...)
        bpm = np.zeros( rawbpm.shape, dtype=np.uint16 )
        bpm[ rawbpm != 0 ] = string_to_bitflag( 'bad pixel', flag_image_bits_inverse )

        return bpm

    def get_gain_at_pixel( self, image, x, y, section_id=None ):
        if image is None:
            return Instrument.get_gain_at_pixel( image, x, y, section_id=section_id )

        # VERIFY THAT THIS IS RIGHT FOR ALL CHIPS
        # Really, we should be looking at the DATASECA, TIRMSECA, etc.
        # header keywords, but those are gone from the NOIRLab-reduced
        # C-pixels x=0-2047 are amp A, 2048-4095 are amp B
        #
        # Empirically, the noirlab-reduced images are *not* gain multiplied,
        # BUT the images are flatfielded, so there's been some effective
        # gain adjustment.
        if ( x <= 2048 ):
            return float( image.header[ 'GAINA' ] )
        else:
            return float( image.header[ 'GAINB' ] )

    def average_gain( self, image, section_id=None ):
        if image is None:
            return Instrument.average_again( self, None, section_id=section_id )
        return ( float( image.header['GAINA'] ) + float( image.header['GAINB'] ) ) / 2.

    def average_saturation_limit( self, image, section_id=None ):
        if image is None:
            return Instrument.average_saturation_limit( self, image, section_id=section_id )
        # Although the method name is "average...", return the lower saturation
        #  limit to be conservative
        return min( float( image.header['SATURATA'] ), float( image.header['SATURATB'] ) )

    @classmethod
    def _get_fits_hdu_index_from_section_id(cls, section_id):
        """
        Return the index of the HDU in the FITS file for the DECam files.
        Since the HDUs have extension names, we can use the section_id directly
        to index into the HDU list.
        """
        cls.check_section_id(section_id)
        return section_id

    @classmethod
    def get_filename_regex(cls):
        return [r'c4d.*\.fits']

    @classmethod
    def get_short_instrument_name(cls):
        """
        Get a short name used for e.g., making filenames.
        """
        return 'c4d'

    @classmethod
    def get_short_filter_name(cls, filter):
        """
        Return the short version of each filter used by DECam.
        In this case we just return the first character of the filter name,
        e.g., shortening "g DECam SDSS c0001 4720.0 1520.0" to "g".
        """
        return filter[0:1]

    @classmethod
    def gaia_dr3_to_instrument_mag( cls, filter, catdata ):
        """Transform Gaia DR3 magnitudes to instrument magnitudes.

        Uses a polynomial transformation from Gaia MAG_G to instrument magnitude.

        The array trns allows a conversion from Gaia MAG_G to
        the magnitude through the desired filter using:

          MAG_filter = Gaia_MAG_G - sum( trns[i] * ( Gaia_MAG _BP - Gaia_MAG_RP ) ** i )

        (with i running from 0 to len(trns)-1).

        Parameters
        ----------
        filter: str
            The (short) filter name of the magnitudes we want.
        catdata: dict or pandas.DataFrame or numpy.recarray or astropy.Table
            A data structure that holds the relevant data,
            that can be indexed on the following keys:
            MAG_G, MAGERR_G, MAG_BP, MAGERR_BP, MAG_RP, MAGERR_RP
            If a single magnitude is required, can pass a dict.
            If an array of magnitudes is required, can be any
            data structure that when indexed on those keys
            returns a 1D numpy array (e.g., a pandas DataFrame,
            or a named structured numpy array, or even a dict
            with ndarray values).

        Returns
        -------
        trans_mag: float or numpy array
            The catalog magnitude(s) transformed to instrument magnitude(s).
        trans_magerr: float or numpy array
            The catalog magnitude error(s) transformed to instrument magnitude error(s).
        """
        if not isinstance(filter, str):
            raise ValueError(f"The filter must be a string. Got {type(filter)}. ")
        if len(filter) > 1:
            filter_short = cls.get_short_filter_name(filter)
        else:
            filter_short = filter

        # Emily Ramey came up with these by fitting polynomials to Gaia
        # magnitudes and DECaLS magnitudes
        transformations = {
            'g': np.array( [  0.07926061, -0.18958323, -0.50588824, 0.11476034 ] ),
            'r': np.array( [ -0.28526417, 0.65444024, -0.25415955, -0.00204337 ] ),
            'i': np.array( [ -0.2491122, 0.51709843, 0.02919352, -0.02097517 ] ),
            'z': np.array( [ -0.38939061, 0.70406435, 0.04190059, -0.01617815 ] )
        }
        if filter_short not in transformations:
            raise ValueError( f"Unknown short DECam filter name {filter}" )

        # instrumental mag is sum(trns[i] * (GaiaBP - GaiaRP) ** i)
        trns = transformations[ filter_short ]
        fitorder = len(trns) - 1

        colors = catdata['MAG_BP'] - catdata['MAG_RP']
        colorerrs = np.sqrt(catdata['MAGERR_BP'] ** 2 + catdata['MAGERR_RP'] ** 2)
        colton = colors[:, np.newaxis] ** np.arange(0, fitorder + 1, 1)
        coltonminus1 = np.zeros(colton.shape)
        coltonminus1[:, 1:] = colors[:, np.newaxis] ** np.arange(0, fitorder, 1)
        coltonerr = np.zeros(colton.shape)
        coltonerr[:, 1:] = np.arange(1, fitorder + 1, 1) * coltonminus1[:, 1:] * colorerrs.value[:, np.newaxis]

        trans_mag = catdata['MAG_G'] - (trns[np.newaxis, :] * colton).sum(axis=1)
        trans_magerr = np.sqrt(catdata['MAGERR_G'] ** 2 + (trns[np.newaxis, :] * coltonerr).sum(axis=1) ** 2)

        return trans_mag, trans_magerr

    def _get_default_calibrator( self, mjd, section, calibtype='dark', filter=None, session=None ):
        # Just going to use the 56876 versions for everything
        # (configured in the yaml files), even though there are earlier
        # versions.  "Good enough."

        # Import CalibratorFile here.  We can't import it at the top of
        # the file because calibrator.py imports image.py, image.py
        # imports exposure.py, and exposure.py imports instrument.py --
        # leading to a circular import
        from models.calibratorfile import CalibratorFile, CalibratorFileDownloadLock

        cfg = Config.get()
        cv = Provenance.get_code_version( session=session )
        prov = Provenance( process='DECam Default Calibrator', code_version_id=cv.id )
        prov.insert_if_needed( session=session )

        reldatadir = pathlib.Path( "DECam_default_calibrators" )
        datadir = pathlib.Path( FileOnDiskMixin.local_path ) / reldatadir

        if calibtype == 'flat':
            rempath = pathlib.Path( f'{cfg.value("DECam.calibfiles.flatbase")}/'
                                    f'{filter}.out.{self._chip_radec_off[section]["ccdnum"]:02d}_trim_med.fits' )

        elif calibtype == 'illumination':
            rempath = pathlib.Path( f'{cfg.value("DECam.calibfiles.illuminationbase")}-'
                                    f'{filter}I_ci_{filter}_{self._chip_radec_off[section]["ccdnum"]:02d}.fits' )
        elif calibtype == 'fringe':
            if filter not in [ 'z', 'Y' ]:
                return None
            rempath = pathlib.Path( f'{cfg.value("DECam.calibfiles.fringebase")}-'
                                    f'{filter}G_ci_{filter}_{self._chip_radec_off[section]["ccdnum"]:02d}.fits' )
        elif calibtype == 'linearity':
            rempath = pathlib.Path( cfg.value( "DECam.calibfiles.linearity" ) )
        else:
            # Other types don't have calibrators for DECam
            return None

        url = f'{cfg.value("DECam.calibfiles.urlbase")}{str(rempath)}'
        filepath = reldatadir / calibtype / rempath.name
        fileabspath = datadir / calibtype / rempath.name

        if calibtype == 'linearity':
            # Linearity requires special handling because it's the same
            # file for all chips.  So, to avoid chaos, we have to get
            # the CalibratorFileDownloadLock for it with section=None.
            # (By the time this this function is called, we should
            # already have the lock for one specific section, but not
            # for the whole instrument.)

            with CalibratorFileDownloadLock.acquire_lock(
                    self.name, None, 'externally_supplied', 'linearity', session=session
            ) as calibfile_lockid:

                with SmartSession( session ) as dbsess:
                    # Gotta check to see if the file was there from
                    # something that didn't go all the way through
                    # before, or if it was downloaded by another process
                    # while we were waiting for the
                    # calibfile_downloadlock
                    datafile = dbsess.scalars(sa.select(DataFile).where(DataFile.filepath == str(filepath))).first()
                    # TODO: what happens if the provenance doesn't match??

                if datafile is None:
                    retry_download( url, fileabspath )
                    datafile = DataFile( filepath=str(filepath), provenance_id=prov.id )
                    datafile.save( str(fileabspath) )
                    datafile.insert( session=session )

                # Linearity file applies for all chips, so load the database accordingly
                # Once again, gotta check to make sure the entry doesn't already exist,
                # because somebody else may have created it while we were waiting for
                # the calibfile_downloadlock.  No race condition here, because nobody
                # else can muck with this table while we have the calibfile_downloadlock.
                with SmartSession( session ) as dbsess:
                    for ssec in self._chip_radec_off.keys():
                        if ( dbsess.query( CalibratorFile )
                             .filter( CalibratorFile.type=='linearity' )
                             .filter( CalibratorFile.calibrator_set=='externally_supplied' )
                             .filter( CalibratorFile.flat_type==None )
                             .filter( CalibratorFile.instrument=='DECam' )
                             .filter( CalibratorFile.sensor_section==ssec )
                             .filter( CalibratorFile.datafile_id==datafile.id ) ).count() == 0:
                            calfile = CalibratorFile( type='linearity',
                                                      calibrator_set="externally_supplied",
                                                      flat_type=None,
                                                      instrument='DECam',
                                                      sensor_section=ssec,
                                                      datafile_id=datafile.id
                                                     )
                            calfile.insert( session=dbsess )

                    # Finally pull out the right entry for the sensor section we were actually asked for
                    calfile = ( dbsess.query( CalibratorFile )
                                .filter( CalibratorFile.type=='linearity' )
                                .filter( CalibratorFile.calibrator_set=='externally_supplied' )
                                .filter( CalibratorFile.flat_type==None )
                                .filter( CalibratorFile.instrument=='DECam' )
                                .filter( CalibratorFile.sensor_section==section )
                                .filter( CalibratorFile.datafile_id==datafile._id )
                               ).first()
                    if calfile is None:
                        raise RuntimeError( f"Failed to get default calibrator file for DECam linearity; "
                                            f"you should never see this error." )
        else:
            # No need to get a new calibfile_downloadlock, we should already have the one for this type and section
            retry_download( url, fileabspath )

            # We know calibtype will be one of fringe, flat, or illumination
            if calibtype == 'fringe':
                dbtype = 'Fringe'
            elif calibtype == 'flat':
                dbtype = 'ComDomeFlat'
            elif calibtype == 'illumination':
                dbtype = 'ComSkyFlat'
            mjd = float( cfg.value( "DECam.calibfiles.mjd" ) )
            image = Image( format='fits', type=dbtype, provenance_id=prov.id, instrument='DECam',
                           telescope='CTIO4m', filter=filter, section_id=section, filepath=str(filepath),
                           mjd=mjd, end_mjd=mjd,
                           info={}, exp_time=0, ra=0., dec=0.,
                           ra_corner_00=0., ra_corner_01=0.,ra_corner_10=0., ra_corner_11=0.,
                           dec_corner_00=0., dec_corner_01=0., dec_corner_10=0., dec_corner_11=0.,
                           minra=0, maxra=0, mindec=0, maxdec=0,
                           target="", project="" )
            # Use FileOnDiskMixin.save instead of Image.save here because we're doing
            # a lower-level operation.  image.save would be if we wanted to read and
            # save FITS data, but here we just want to have it make sure the file
            # is in the right place and check its md5sum.  (FileOnDiskMixin.save, when
            # given a filename, will move that file to where it goes in the local data
            # storage unless it's already in the right place.)
            FileOnDiskMixin.save( image, fileabspath )
            calfile = CalibratorFile( type=calibtype,
                                      calibrator_set='externally_supplied',
                                      flat_type='externally_supplied' if calibtype == 'flat' else None,
                                      instrument='DECam',
                                      sensor_section=section,
                                      image_id=image.id )
            image.insert( session=session )
            calfile.insert( session=session )

        return calfile

    def linearity_correct( self, *args, linearitydata=None ):
        if not isinstance( linearitydata, DataFile ):
            raise TypeError( f'DECam.linearity_correct: linearitydata must be a DataFile' )

        if len(args) == 1:
            if not isinstance( args[0], Image ):
                raise TypeError( 'linearity_correct: pass either an Image as one argument, '
                                 'or header and data as two arguments' )
            data = args[0].data
            header = args[0].header
        elif len(args) == 2:
            # if not isinstance( args[0], <whatever the right header datatype is>:
            #     raise TypeError( "header isn't a <header>" )
            if not isinstance( args[1], np.ndarray ):
                raise TypeError( "data isn't a numpy array" )
            header = args[0]
            data = args[1]
        else:
            raise RuntimeError( 'linearity_correct: pass either an Image as one argument, '
                                'or header and data as two arguments' )

        presecs = self.overscan_and_data_sections( header )
        secs = {}
        for sec in presecs:
            secs[sec['secname']] = sec

        newdata = np.zeros_like( data )
        ccdnum = header[ 'CCDNUM' ]

        with fits.open( linearitydata.get_fullpath(), memmap=False ) as linhdu:
            for amp in ["A", "B"]:
                ampdex = f'ADU_LINEAR_{amp}'
                x0 = secs[amp]['destsec']['x0']
                x1 = secs[amp]['destsec']['x1']
                y0 = secs[amp]['destsec']['y0']
                y1 = secs[amp]['destsec']['y1']

                lindex = np.floor( data[ y0:y1, x0:x1 ] ).astype( int )
                newdata[ y0:y1, x0:x1 ] = ( linhdu[ccdnum].data[lindex][ampdex]
                                            + ( ( data[y0:y1, x0:x1] - linhdu[ccdnum].data[lindex]['ADU'] )
                                                * ( linhdu[ccdnum].data[lindex+1][ampdex]
                                                    - linhdu[ccdnum].data[lindex][ampdex] )
                                                / ( linhdu[ccdnum].data[lindex+1]['ADU']
                                                    - linhdu[ccdnum].data[lindex]['ADU'] )
                                               ) )

        return newdata

    def acquire_origin_exposure( self, identifier, params, outdir=None ):
        """Download exposure from NOIRLab; see Instrument.acquire_origin_exposure

        NOTE : assumes downloading proc_type 'raw' images, so does not
        look for dqmask or weight images, just downlaods the single
        exposure.

        """
        outdir = pathlib.Path( outdir ) if outdir is not None else pathlib.Path( FileOnDiskMixin.temp_path )
        outdir.mkdir( parents=True, exist_ok=True )
        outfile = outdir / identifier
        retry_download( params['url'], outfile, retries=5, sleeptime=5, exists_ok=True,
                        clobber=True, md5sum=params['md5sum'], sizelog='GiB', logger=SCLogger.get() )
        return outfile

    def _commit_exposure( self, origin_identifier, expfile, obs_type='Sci', proc_type='raw', session=None ):
        """Add to the Exposures table in the database an exposure downloaded from NOIRLab.

        Used internally by acquire_and_commit_origin_exposure and
        DECamOriginExposures.download_and_commit_exposures

        Parameters
        ----------
        origin_identifier : str
          The filename part of the archive_filename from the NOIRLab archive

        expfile : str or Path
          The path where the downloaded exposure can be found on disk

        obs_type : str, default 'Sci'
          The obs_type parameter (generally parsed from the exposure header, or pulled from the NOIRLab archive)

        session : Session
          Optional database session

        Returns
        -------
        Exposure

        """
        outdir = pathlib.Path( FileOnDiskMixin.local_path )

        # This import is here rather than at the top of the file
        #  because Exposure imports Instrument, so we've got
        #  a circular import.  Here, instrument will have been
        #  fully initialized before we try to import Exposure,
        #  so we should be OK.
        from models.exposure import Exposure

        obstypemap = { 'object': 'Sci',
                       'dark': 'Dark',
                       'dome flat': 'DomeFlat',
                       'zero': 'Bias'
                      }

        provenance = Provenance(
            process='download',
            parameters={ 'proc_type': proc_type, 'Instrument': 'DECam' },
            code_version_id=Provenance.get_code_version( session=session ).id
        )
        provenance.insert_if_needed( session=session )

        with fits.open( expfile ) as ifp:
            hdr = { k: v for k, v in ifp[0].header.items()
                    if k in ( 'PROCTYPE', 'PRODTYPE', 'FILENAME', 'TELESCOP', 'OBSERVAT', 'INSTRUME'
                              'OBS-LONG', 'OBS-LAT', 'EXPTIME', 'DARKTIME', 'OBSID',
                              'DATE-OBS', 'TIME-OBS', 'MJD-OBS', 'OBJECT', 'PROGRAM',
                              'OBSERVER', 'PROPID', 'FILTER', 'RA', 'DEC', 'HA', 'ZD', 'AIRMASS',
                              'VSUB', 'GSKYPHOT', 'LSKYPHOT' ) }
        exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter',
                                                            'project', 'target' ] )
        ra = util.radec.parse_sexigesimal_degrees( hdr['RA'], hours=True )
        dec = util.radec.parse_sexigesimal_degrees( hdr['DEC'] )

        # NOTE -- there's a possible sort-of race condition here.  (Only
        # sort-of because we'll get an error that we want to get.)  If
        # multiple processes are working on images from the same
        # exposure, then they could all be trying to save the same
        # exposure at the same time, and most of them will become sad.
        # In practice, however, e.g. in
        # pipeline/pipeline_exposure_launcher.py, we'll have a single
        # process dealing with a given exposure, so this shouldn't come
        # up much.
        with SmartSession( session ) as dbsess:
            q = ( dbsess.query( Exposure )
                  .filter( Exposure.instrument == 'DECam' )
                  .filter( Exposure.origin_identifier == origin_identifier )
                 )
            if q.count() > 1:
                raise RuntimeError( f"Database error: got more than one Exposure "
                                    f"with origin_identifier {origin_identifier}" )
            existing = q.first()
            if existing is not None:
                raise FileExistsError( f"Exposure with origin identifier {origin_identifier} "
                                       f"already exists in the database. ({existing.filepath})" )
            if obs_type not in obstypemap:
                SCLogger.warning( f"DECam obs_type {obs_type} not known, assuming Sci" )
                obs_type = 'Sci'
            else:
                obs_type = obstypemap[ obs_type ]
            expobj = Exposure( current_file=expfile, invent_filepath=True,
                               type=obs_type, format='fits', provenance_id=provenance.id, ra=ra, dec=dec,
                               instrument='DECam', origin_identifier=origin_identifier, header=hdr,
                               **exphdrinfo )
            dbpath = outdir / expobj.filepath
            expobj.save( expfile )
            expobj.insert( session=dbsess )

        return expobj


    def acquire_and_commit_origin_exposure( self, identifier, params ):
        """Download exposure from NOIRLab, add it to the database; see Instrument.acquire_and_commit_origin_exposure.

        """
        downloaded = self.acquire_origin_exposure( identifier, params )
        return self._commit_exposure( identifier, downloaded, params['obs_type'], params['proc_type'] )


    def find_origin_exposures( self,
                               skip_exposures_in_database=True,
                               skip_known_exposures=True,
                               minmjd=None,
                               maxmjd=None,
                               filters=None,
                               containing_ra=None,
                               containing_dec=None,
                               minexptime=None,
                               proc_type='raw',
                               projects=None ):
        """Search the NOIRLab data archive for exposures.

        See Instrument.find_origin_exposures for documentation; in addition:

        Parameters
        ----------
        filters: str or list of str
           The short (i.e. single character) filter names ('g', 'r',
           'i', 'z', or 'Y') to search for.  If not given, will
           return images from all filters.
        projects: str or list of str
           The NOIRLab proposal ids to limit the search to.  If not
           given, will not filter based on proposal id.
        proc_type: str
           'raw' or 'instcal' : the processing type to get
           from the NOIRLab data archive.

        TODO -- deal with provenances!  Right now, skip_known_exposures
        will skip exposures of *any* provenance, may or may not be what
        we want.  See Issue #310.

        """

        if ( containing_ra is None ) != ( containing_dec is None ):
            raise RuntimeError( f"Must specify both or neither of (containing_ra, containing_dec)" )
        if ( ( containing_ra is None ) or ( containing_dec is None ) ) and ( minmjd is None ):
            raise RuntimeError( f"Must specify either a containing ra,dec or a minmjd to find DECam exposures." )
        if containing_ra is not None:
            raise NotImplementedError( f"containing_(ra|dec) is not implemented yet for DECam" )

        # Convert mjd to iso format for astroarchive
        starttime, endtime = astropy.time.Time( [ minmjd, maxmjd ], format='mjd').to_value( 'isot' )
        # Make sure there's a Z at the end of these times; astropy
        # doesn't seem to do it, but might in the future
        if starttime[-1] != 'Z': starttime += 'Z'
        if endtime[-1] != 'Z': endtime += 'Z'
        spec = {
            "outfields" : [
                "archive_filename",
                "url",
                "instrument",
                "telescope",
                "proposal",
                "proc_type",
                "prod_type",
                "obs_type",
                "caldat",
                "dateobs_center",
                "ifilter",
                "exposure",
                "ra_center",
                "dec_center",
                "md5sum",
                "OBJECT",
                "MJD-OBS",
                "DATE-OBS",
                "AIRMASS",
            ],
            "search" : [
                [ "instrument", "decam" ],
                [ "proc_type", proc_type ],
                [ "dateobs_center", starttime, endtime ],
            ]
        }

        filters = util.util.listify( filters, require_string=True )
        proposals = util.util.listify( projects, require_string=True )

        if proposals is not None:
            spec["search"].append( [ "proposal" ] + proposals )

        # TODO : implement the ability to log in via a configured username and password
        # For now, will only get exposures that are public
        apiurl = f'https://astroarchive.noirlab.edu/api/adv_search/find/?format=json&limit=0'

        def getoneresponse( json ):
            SCLogger.debug( f"Sending NOIRLab search query to {apiurl} with json={json}" )
            response = requests.post( apiurl, json=json )
            response.raise_for_status()
            if response.status_code == 200:
                files = pandas.DataFrame( response.json()[1:] )
            else:
                SCLogger.error( response.json()['errorMessage'] )
                # SCLogger.error( response.json()['traceback'] )     # Uncomment for API developer use
                raise RuntimeError( response.json()['errorMessage'] )
            return files

        if filters is None:
            files = getoneresponse( spec )
        else:
            files = None
            for filt in filters:
                filtspec = copy.deepcopy( spec )
                filtspec["search"].append( [ "ifilter", filt, "startswith" ] )
                newfiles = getoneresponse( filtspec )
                if not newfiles.empty:
                    files = newfiles if files is None else pandas.concat( [files, newfiles] )

        if files.empty or files is None:
            SCLogger.warning( f"DECam exposure search found no files." )
            return None

        if minexptime is not None:
            files = files[ files.exposure >= minexptime ]
        files.sort_values( by='dateobs_center', inplace=True )
        files['filtercode'] = files.ifilter.str[0]
        files['filter'] = files.ifilter

        if skip_known_exposures:
            identifiers = [ pathlib.Path( f ).name for f in files.archive_filename.values ]
            with SmartSession() as session:
                ke = session.query( KnownExposure ).filter( KnownExposure.identifier.in_( identifiers ) ).all()
            existing = [ i.identifier for i in ke ]
            keep = [ i not in existing for i in identifiers ]
            files = files[keep].reset_index( drop=True )

        if skip_exposures_in_database:
            identifiers = [ pathlib.Path( f ).name for f in files.archive_filename.values ]
            with SmartSession() as session:
                exps = session.query( Exposure ).filter( Exposure.origin_identifier.in_( identifiers ) ).all()
            existing = [ i.origin_identifier for i in exps ]
            keep = [ i not in existing for i in identifiers ]
            files = files[keep].reset_index( drop=True )

        if len(files) == 0:
            SCLogger.info( "DEcam exposure search found no files afters skipping known and databsae exposures" )
            return None

        # If we were downloaded reduced images, we're going to have multiple prod_types
        # for the same image (reason: there are dq mask and weight images in addition
        # to the actual images).  Reflect this in the pandas structure.  Try to set it
        # up so that there's a range()-like index for each dateobs, and then the
        # second index is prod_type.  I really hope the dateobs values all match
        # perfectly.
        dateobsvals = files.dateobs_center.unique()
        dateobsmap = { dateobsvals[i]: i for i in range(len(dateobsvals)) }
        dateobsindexcol = [ dateobsmap[i] for i in files.dateobs_center.values ]
        files['dateobsindex'] = dateobsindexcol

        files.set_index( [ 'dateobsindex', 'prod_type' ], inplace=True )

        return DECamOriginExposures( proc_type, files )


class DECamOriginExposures:
    """An object that encapsulates what was found by DECam.find_origin_exposures()"""

    def __init__( self, proc_type, frame ):
        """Should only be instantiated from DECam.find_origin_exposures()

        Parameters
        ----------
        proc_type: str
           'raw' or 'instcal'
        frame: pandas.DataFrame

        """
        self.proc_type = proc_type
        self._frame = frame
        self.decam = get_instrument_instance( 'DECam' )

    def __len__( self ):
        # The length is the number of values there are in the *first* index
        # as that is the number of different exposures.
        return len( self._frame.index.levels[0] )

    def add_to_known_exposures( self,
                                indexes=None,
                                hold=False,
                                skip_loaded_exposures=True,
                                skip_duplicates=True,
                                session=None):
        """See InstrumentOriginExposures.add_to_known_exposures"""

        if indexes is None:
            indexes = range( len(self._frame) )
        if not isinstance( indexes, collections.abc.Sequence ):
            indexes = [ indexes ]

        with SmartSession( session ) as dbsess:
            identifiers = [ pathlib.Path( self._frame.loc[ dex, 'image' ].archive_filename ).name for dex in indexes ]

            # There is a database race condition here that I've chosen
            # not to worry about.  Reason: in general usage, there will
            # be one conductor process running, and that will be the
            # only process to call this method, so nothing will be
            # racing with it re: the knownexposures table.  While in
            # principle this process is racing with all running
            # instances of the pipeline for the exposures table, in
            # practice we won't be launching a pipeline to work on an
            # exposure until after it was already loaded into the
            # knownexposures table.  So, as long as skip_duplicates is
            # True, there will not (in the usual case) be anything in
            # exposures that both we are searching for right now and
            # that's not already in knownexposures.

            skips = []
            if skip_loaded_exposures:
                skips.extend( list( dbsess.query( Exposure.origin_identifier )
                                    .filter( Exposure.origin_identifier.in_( identifiers ) ) ) )
            if skip_duplicates:
                skips.extend( list( dbsess.query( KnownExposure.identifier )
                                    .filter( KnownExposure.identifier.in_( identifiers ) ) ) )
            skips = set( skips )

            for dex in indexes:
                identifier = pathlib.Path( self._frame.loc[ dex, 'image' ].archive_filename ).name
                if identifier in skips:
                    continue
                expinfo = self._frame.loc[ dex, 'image' ]
                gallat, gallon, ecllat, ecllon = radec_to_gal_ecl( expinfo.ra_center, expinfo.dec_center )
                ke = KnownExposure( instrument='DECam', identifier=identifier,
                                    params={ 'url': expinfo.url,
                                             'md5sum': expinfo.md5sum,
                                             'obs_type': expinfo.obs_type,
                                             'proc_type': expinfo.proc_type },
                                    hold=hold,
                                    exp_time=expinfo.exposure,
                                    filter=expinfo.ifilter,
                                    project=expinfo.proposal,
                                    target=expinfo.OBJECT,
                                    mjd=util.util.parse_dateobs( expinfo.dateobs_center, output='float' ),
                                    ra=expinfo.ra_center,
                                    dec=expinfo.dec_center,
                                    ecllat=ecllat,
                                    ecllon=ecllon,
                                    gallat=gallat,
                                    gallon=gallon
                                   )
                ke.insert( session=dbsess )


    def download_exposures( self, outdir=".", indexes=None, onlyexposures=True,
                            clobber=False, existing_ok=False, session=None ):
        """Download exposures, and maybe weight and data quality frames as well.

        Parameters
        ----------
        Same as InstrumentOriginExposures.download_exposures, plus:

        onlyexposures: bool
          If True, only download the main exposure.  If False, also
          download dqmask and weight exposures; this only makes sense if
          proc_type was 'instcal' in the call to
          DECam.find_origin_exposures that instantiated this object.

        Returns a list of all files downloaded; you will need to parse
        the filenames to figure out if it's exposure (image), weight
        exposure, or dqmask exposure.

        """
        outdir = pathlib.Path( outdir )
        if indexes is None:
            indexes = range( len(self._frame) )
        if not isinstance( indexes, collections.abc.Sequence ):
            indexes = [ indexes ]

        downloaded = []

        for dex in indexes:
            expinfo = self._frame.loc[dex]
            extensions = expinfo.index.values
            fpaths = {}
            for ext in extensions:
                if onlyexposures and ext != 'image':
                    continue
                expinfo = self._frame.loc[ dex, ext ]
                fname = pathlib.Path( expinfo.archive_filename ).name
                fpath = outdir / fname
                retry_download( expinfo.url, fpath, retries=5, sleeptime=5, exists_ok=existing_ok,
                                clobber=clobber, md5sum=expinfo.md5sum, sizelog='GiB', logger=SCLogger.get() )
                fpaths[ 'exposure' if ext=='image' else ext ] = fpath
            downloaded.append( fpaths )

        return downloaded

    def download_and_commit_exposures( self, indexes=None, clobber=False, existing_ok=False,
                                       delete_downloads=True, skip_existing=True, session=None ):
        outdir = pathlib.Path( FileOnDiskMixin.local_path )
        if indexes is None:
            indexes = range( len(self._frame) )
        if not isinstance( indexes, collections.abc.Sequence ):
            indexes = [ indexes ]

        downloaded = self.download_exposures( outdir=outdir, indexes=indexes, clobber=clobber,
                                              existing_ok=existing_ok, session=session )

        exposures = []
        for dex, expfiledict in zip( indexes, downloaded ):
            if set( expfiledict.keys() ) != { 'exposure' }:
                SCLogger.warning( f"Downloaded wtmap and dqmask files in addition to the exposure file "
                                 f"from DECam, but only loading the exposure file into the database." )
                # TODO: load these as file extensions (see
                # FileOnDiskMixin), if we're ever going to actually
                # use the observatory-reduced DECam images It's more
                # work than just what needs to be done here, because
                # we will need to think about that in
                # Image.from_exposure(), and perhaps other places as
                # well.
            expfile = expfiledict[ 'exposure' ]
            origin_identifier = pathlib.Path( self._frame.loc[dex,'image'].archive_filename ).name
            obs_type = self._frame.loc[dex,'image'].obs_type
            proc_type = self._frame.loc[dex,'image'].proc_type
            expobj = self.decam._commit_exposure( origin_identifier, expfile, obs_type, proc_type, session=session )
            exposures.append( expobj )

        return exposures
