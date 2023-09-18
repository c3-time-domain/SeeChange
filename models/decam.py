import io
import math
import time
import copy
import pathlib
import hashlib
import logging
import requests
import collections.abc

import pandas
import astropy.time
from astropy.io import fits

from models.base import _logger, SmartSession, FileOnDiskMixin
from models.instrument import Instrument, InstrumentOrientation, SensorSection
from models.provenance import Provenance
import util
import util.util

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
        # will apply kwargs to attributes, and register instrument in the INSTRUMENT_INSTANCE_CACHE
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

        Instrument.__init__(self, **kwargs)

    @classmethod
    def get_section_ids(cls):

        """
        Get a list of SensorSection identifiers for this instrument.
        We are using the names of the FITS extensions (e.g., N12, S22, etc.).
        See ref: https://noirlab.edu/science/sites/default/files/media/archives/images/DECamOrientation.png
        """
        n_list = [f'N{i}' for i in range(1, 32)]
        s_list = [f'S{i}' for i in range(1, 32)]
        return n_list + s_list

    @classmethod
    def check_section_id(cls, section_id):
        """
        Check that the type and value of the section is compatible with the instrument.
        In this case, it must be an integer in the range [0, 63].
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

    def find_origin_exposures( self, skip_exposures_in_database=True,
                               minmjd=None, maxmjd=None, filters=None,
                               containing_ra=None, containing_dec=None,
                               minexptime=None, proc_type='raw',
                               proposals=None ):
        """Search the NOIRLab data archive for exposures.

        See Instrument.find_origin_exposures for documentation; in addition:

        Parameters
        ----------
        filters: str or list of str
           The short (i.e. single character) filter names ('g', 'r',
           'i', 'z', or 'Y') to search for.  If not given, will
           return images from all filters.
        proposals: str or list of str
           The NOIRLab proposal ids to limit the search to.  If not
           given, will not filter based on proposal id.
        proc_type: str
           'raw' or 'instcal' : the processing type to get 
           from the NOIRLab data archive.

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
                "md5sum",
                "MJD-OBS",
                "DATE-OBS",
                "AIRMASS",
            ],
            "search" : [
                [ "instrument", "decam" ],
                [ "proc_type", proc_type ],
                [ "prod_type", "image" ],
                [ "dateobs_center", starttime, endtime ],
            ]
        }

        filters = util.util.listify( filters, require_string=True )
        proposals = util.util.listify( proposals, require_string=True )

        if proposals is not None:
            spec["search"].append( [ "proposal" ] + proposals )

        # TODO : implement the ability to log in via a configured username and password
        # For now, will only get exposures that are public
        apiurl = f'https://astroarchive.noirlab.edu/api/adv_search/find/?format=json&limit=0'

        def getoneresponse( json ):
            _logger.debug( f"Sending NOIRLab search query to {apiurl} with json={json}" )
            response = requests.post( apiurl, json=json )
            response.raise_for_status()
            if response.status_code == 200:
                files = pandas.DataFrame( response.json()[1:] )
            else:
                _logger.error( response.json()['errorMessage'] )
                # _logger.error( response.json()['traceback'] )     # Uncomment for API developer use
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
            _logger.warning( f"DECam exposure search found no files." )
            return None

        if minexptime is not None:
            files = files[ files.exposure >= minexptime ]
        files.sort_values( by='dateobs_center', inplace=True )
        files['filtercode'] = files.ifilter.str[0]

        if skip_exposures_in_database:
            raise NotImplementedError( "TODO: implement skip_exposures_in_database" )

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

    def __len__( self ):
        return len(self._frame)

    def download_exposures( self, outdir=".", indexes=None, clobber=False, existing_ok=False ):
        outdir = pathlib.Path( outdir )
        if indexes is None:
            indexes = range( len(self._frame) )
        if not isinstance( indexes, collections.abc.Sequence ):
            indexes = [ indexes ]

        downloaded = []

        for dex in indexes:
            expinfo = self._frame.iloc[dex]
            fname = pathlib.Path( expinfo.archive_filename ).name
            fpath = pathlib.Path( outdir / fname )
            if fpath.exists():
                if clobber:
                    if not fpath.is_file():
                        _logger.error( f"download_exposures: {fpath} exists and is not a file, not overwriting." )
                        raise FileExistsError( f"{fpath} exists and is not a file, not overwriting." )
                    else:
                        fpath.unlink()
                elif existing_ok:
                    _logger.info( f"download_exposures: {fpath} exists; trusting it's the right thing" )
                    downloaded.append( fpath )
                    continue
                else:
                    _logger.error( f"download_exposures: {fpath} exists but clobber is False" )
                    raise FileExistsError( f"{fpath} exists but clobber is False" )
            countdown = 5
            success = False
            while not success:
                try:
                    starttime = time.perf_counter()
                    renew = False
                    if _logger.getEffectiveLevel() >= logging.DEBUG:
                        _logger.info( f"download_exposures: Downloading {fname} from {expinfo.url}" )
                    else:
                        _logger.info( f"download_exposures: Downloading {fname}" )
                    response = requests.get( expinfo.url )
                    response.raise_for_status()
                    midtime = time.perf_counter()
                    size = len(response.content) / 1024 / 1024 / 1024
                    _logger.info( f"...downloaded {size:.3f} GiB in {midtime-starttime:.2f} sec" )
                    with open( fpath, "wb" ) as ofp:
                        ofp.write( response.content )
                    endtime = time.perf_counter()
                    _logger.info( f"...written to disk in {endtime-midtime:.2f} sec" )
                    success = True
                except Exception as e:
                    strio = io.StringIO("")
                    traceback.print_exc( file=strio )
                    _logger.warning( f"Exception downloading from {expinfo.url}:\n{strio.getvalue()}" )
                    countdown -= 1
                    if countdown >= 0:
                        _logger.warning( f"download_exposures: Failed to download {fname}, waiting 5s and retrying." )
                        time.sleep( 5 )
                    else:
                        _logger.error( f"download_exposures: Repeated exceptions trying to download {fname}" )
                        raise e

            downloaded.append( fpath )

        return downloaded

    def download_and_commit_exposures( self, indexes=None, clobber=False, existing_ok=False,
                                       delete_downloads=True, skip_existing=True, session=None ):
        outdir = pathlib.Path( FileOnDiskMixin.local_path )
        if indexes is None:
            indexes = range( len(self._frame) )
        if not isinstance( indexes, collections.abc.Sequence ):
            indexes = [ index ]

        exposures = []

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

        with SmartSession(session) as dbsess:
            provenance = Provenance( process='download',
                                     parameters={ 'proc_type': self.proc_type },
                                     code_version=Provenance.get_code_version(session=dbsess) )
            provenance.update_id()
            provenance = provenance.recursive_merge( dbsess )
            dbsess.add( provenance )

            downloaded = self.download_exposures( outdir=outdir, indexes=indexes,
                                                  clobber=clobber, existing_ok=existing_ok )
            for dex, expfile in zip( indexes, downloaded ):
                with fits.open( expfile ) as ifp:
                    hdr = { k: v for k, v in ifp[0].header.items()
                            if k in ( 'PROCTYPE', 'PRODTYPE', 'FILENAME', 'TELESCOP', 'OBSERVAT', 'INSTRUME'
                                      'OBS-LONG', 'OBS-LAT', 'EXPTIME', 'DARKTIME', 'OBSID',
                                      'DATE-OBS', 'TIME-OBS', 'MJD-OBS', 'OBJECT', 'PROGRAM',
                                      'OBSERVER', 'PROPID', 'FILTER', 'RA', 'DEC', 'HA', 'ZD', 'AIRMASS',
                                      'VSUB', 'GSKYPHOT', 'LSKYPHOT' ) }
                exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter',
                                                                    'project', 'target' ] )
                origin_identifier = pathlib.Path( self._frame.iloc[dex].archive_filename ).name

                ra = util.radec.parse_sexigesimal_degrees( hdr['RA'], hours=True )
                dec = util.radec.parse_sexigesimal_degrees( hdr['DEC'] )

                q = ( dbsess.query( Exposure )
                      .filter( Exposure.instrument == 'DECam' )
                      .filter( Exposure.origin_identifier==origin_identifier )
                     )
                existing = q.first()
                # Maybe check that q.count() isn't >1; if it is, throw an exception
                #  about database corruption?
                if existing is not None:
                    if skip_existing:
                        _logger.info( f"download_and_commit_exposures: exposure with origin identifier "
                                      f"{origin_identifier} is already in the database, skipping. "
                                      f"({existing.filepath})" )
                        continue
                    else:
                        raise FileExistsError( f"Exposure with origin identifier {origin_identifier} "
                                               f"already exists in the database. ({existing.filepath})" )
                obstype = self._frame.iloc[dex].obs_type
                if obstype not in obstypemap:
                    _logger.warning( f"DECam obs_type {obstype} not known, assuming Sci" )
                    obstype = 'Sci'
                else:
                    obstype = obstypemap[ obstype ]
                expobj = Exposure( current_file=expfile, invent_filepath=True,
                                   type=obstype, format='fits', provenance=provenance, ra=ra, dec=dec, 
                                   instrument='DECam', origin_identifier=origin_identifier, header=hdr,
                                   **exphdrinfo )
                dbpath = outdir / expobj.filepath
                expobj.save( expfile )
                expobj = expobj.recursive_merge( dbsess )
                dbsess.add( expobj )
                dbsess.commit()
                if delete_downloads and ( dbpath.resolve() != expfile.resolve() ):
                    expfile.unlink()
                exposures.append( expobj )

        return exposures
