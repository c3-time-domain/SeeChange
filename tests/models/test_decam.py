import os
import re
import shutil
import pathlib
import hashlib
import pytest

import numpy as np
import sqlalchemy as sa

from astropy.io import fits

from models.base import SmartSession, FileOnDiskMixin
from models.exposure import Exposure
from models.instrument import get_instrument_instance
from models.datafile import DataFile
from models.calibratorfile import CalibratorFile
from models.image import Image
from models.instrument import Instrument
from models.decam import DECam
import util.radec
from util.logger import SCLogger

from tests.conftest import CODE_ROOT


def test_decam_exposure(decam_filename):
    assert os.path.isfile(decam_filename)

    e = Exposure(filepath=decam_filename)
    e.save()  # make sure to save it to archive, so it has an MD5 sum
    assert e.instrument == 'DECam'
    assert isinstance(e.instrument_object, DECam)
    assert e.telescope == 'CTIO 4.0-m telescope'
    assert e.mjd == 59887.32121458
    assert e.end_mjd == 59887.32232569111
    assert e.ra == 116.32024583333332
    assert e.dec == -26.25
    assert e.exp_time == 96.0
    assert e.filepath == 'c4d_221104_074232_ori.fits.fz'
    assert e.filter == 'g DECam SDSS c0001 4720.0 1520.0'
    assert not e.from_db
    assert e.info == {}
    assert e.id is None
    assert e.target == 'DECaPS-West'
    assert e.project == '2022A-724693'

    # check that we can lazy load the header from file
    assert len(e.header) == 150
    assert e.header['NAXIS'] == 0

    with pytest.raises(ValueError, match=re.escape('The section_id must be a string. ')):
        _ = e.data[0]

    assert isinstance(e.data['N4'], np.ndarray)
    assert e.data['N4'].shape == (4146, 2160)
    assert e.data['N4'].dtype == 'uint16'

    with pytest.raises(ValueError, match=re.escape('The section_id must be a string. ')):
        _ = e.section_headers[0]

    assert len(e.section_headers['N4']) == 100
    assert e.section_headers['N4']['NAXIS'] == 2
    assert e.section_headers['N4']['NAXIS1'] == 2160
    assert e.section_headers['N4']['NAXIS2'] == 4146


def test_image_from_decam_exposure(decam_filename, provenance_base, data_dir):
    with fits.open( decam_filename, memmap=False ) as ifp:
        hdr = ifp[0].header
    exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter', 'project', 'target' ] )
    ra = util.radec.parse_sexigesimal_degrees( hdr['RA'], hours=True )
    dec = util.radec.parse_sexigesimal_degrees( hdr['DEC'] )
    e = Exposure( ra=ra, dec=dec, instrument='DECam', format='fits', **exphdrinfo,
                  filepath=os.path.join(data_dir, pathlib.Path(decam_filename).name ))
    sec_id = 'N4'
    im = Image.from_exposure(e, section_id=sec_id)  # load the first CCD

    assert e.instrument == 'DECam'
    assert e.telescope == 'CTIO 4.0-m telescope'
    assert not im.from_db
    # should not be the same as the exposure!
    # assert im.ra == 116.32024583333332
    # assert im.dec == -26.25
    assert im.ra != e.ra
    assert im.dec != e.dec
    assert im.ra == 116.32126671843677
    assert im.dec == -26.337508447652503
    assert im.mjd == 59887.32121458
    assert im.end_mjd == 59887.32232569111
    assert im.exp_time == 96.0
    assert im.filter == 'g DECam SDSS c0001 4720.0 1520.0'
    assert im.target == 'DECaPS-West'
    assert im.project == '2022A-724693'
    assert im.section_id == sec_id

    assert im.id is None  # not yet on the DB
    assert im.filepath is None  # no file yet!

    assert len(im.header) == 98
    assert im.header['NAXIS'] == 2
    assert im.header['NAXIS1'] == 2160
    assert im.header['NAXIS2'] == 4146
    assert 'BSCALE' not in im.header
    assert 'BZERO' not in im.header

    # check we have the raw data copied into temporary attribute
    assert im.raw_data is not None
    assert isinstance(im.raw_data, np.ndarray)
    assert im.raw_data.shape == (4146, 2160)

    # just for this test we will do preprocessing just by reducing the median
    im.data = np.float32(im.raw_data - np.median(im.raw_data))

    # TODO: check we can save the image using the filename conventions


# Note that these tests are probing the internal state of the opaque
# DECamOriginExposures objects.  If you're looking at this test for
# guidance for how to do things, do *not* write code that mucks about
# with the _frame member of one of those objects; that's internal state
# not intended for external consumption.
def test_decam_search_noirlab( decam_reduced_origin_exposures ):
    origloglevel = SCLogger.get().getEffectiveLevel()
    try:
        # uncomment below to show the things sent to the noirlab API if something goes wrong.
        # SCLogger.setLevel( logging.DEBUG )

        decam = DECam()

        # Make sure it yells at us if we don't give any constraints
        with pytest.raises( RuntimeError ):
            decam.find_origin_exposures()

        # Make sure we can find some reduced exposures without filter/proposals
        originexposures = decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                                       skip_exposures_in_database=False, proc_type='instcal' )
        assert len(originexposures._frame.index.levels[0]) == 9
        assert set(originexposures._frame.index.levels[1]) == { 'image', 'wtmap', 'dqmask' }
        assert set(originexposures._frame.filtercode) == { 'g', 'V', 'i', 'r', 'z' }
        assert set(originexposures._frame.proposal) == { '2023A-716082', '2023A-921384' }

        # Make sure we can find reduced exposures limiting by proposal
        originexposures = decam_reduced_origin_exposures
        assert set(originexposures._frame.filtercode) == { 'i', 'r', 'g', 'z' }
        assert all( [ originexposures._frame.iloc[i].proposal == '2023A-716082'
                      for i in range(len(originexposures._frame)) ] )

        # Make sure we can find reduced exposures limiting by filter
        originexposures = decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                                       filters='r',
                                                       skip_exposures_in_database=False )
        assert len(originexposures._frame.index.levels[0]) == 2
        assert set(originexposures._frame.filtercode) == { 'r' }

        originexposures = decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                                       filters=[ 'r', 'g' ],
                                                       skip_exposures_in_database=False )
        assert len(originexposures._frame.index.levels[0]) == 4
        assert set(originexposures._frame.filtercode) == { 'r', 'g' }
    finally:
        SCLogger.setLevel( origloglevel )


def test_decam_download_origin_exposure( decam_reduced_origin_exposures, cache_dir ):
    assert all( [ row.proc_type == 'instcal' for i, row in decam_reduced_origin_exposures._frame.iterrows() ] )
    try:
        # First try downloading the reduced exposures themselves
        downloaded = decam_reduced_origin_exposures.download_exposures(
            outdir=os.path.join(cache_dir, 'DECam'),
            indexes=[ 1, 3 ],
            onlyexposures=True,
            clobber=False,
            existing_ok=True,
        )
        assert len(downloaded) == 2
        for pathdict, dex in zip( downloaded, [ 1, 3 ] ):
            assert set( pathdict.keys() ) == { 'exposure' }
            md5 = hashlib.md5()
            with open( pathdict['exposure'], "rb") as ifp:
                md5.update( ifp.read() )
            assert md5.hexdigest() == decam_reduced_origin_exposures._frame.loc[ dex, 'image' ].md5sum

        # Now try downloading exposures, weights, and dataquality masks
        downloaded = decam_reduced_origin_exposures.download_exposures(
            outdir=os.path.join(cache_dir, 'DECam'),
            indexes=[ 1, 3 ],
            onlyexposures=False,
            clobber=False,
            existing_ok=True,
        )
        assert len(downloaded) == 2
        for pathdict, dex in zip( downloaded, [ 1, 3 ] ):
            assert set( pathdict.keys() ) == { 'exposure', 'wtmap', 'dqmask' }
            for extname, extpath in pathdict.items():
                if extname == 'exposure':
                    # Translation back to what we know is in the internal dataframe
                    extname = 'image'
                md5 = hashlib.md5()
                with open( extpath, "rb" ) as ifp:
                    md5.update( ifp.read() )
                assert md5.hexdigest() == decam_reduced_origin_exposures._frame.loc[ dex, extname ].md5sum

    finally:
        # Don't clean up for efficiency of rerunning tests.
        pass


def test_decam_download_and_commit_exposure(
        code_version, decam_raw_origin_exposures, cache_dir, data_dir, test_config, archive
):
    eids = []
    try:
        with SmartSession() as session:
            expdexes = [ 1, 2 ]

            # get these downloaded first, to get the filenames to check against the cache
            downloaded = decam_raw_origin_exposures.download_exposures(
                outdir=os.path.join(cache_dir, 'DECam'),
                indexes=expdexes,
                onlyexposures=True,
                clobber=False,
                existing_ok=True,
            )
            for pathdict in downloaded:
                cachedpath = pathdict['exposure']
                assert os.path.isfile( cachedpath )
                shutil.copy2( cachedpath, os.path.join( data_dir, os.path.basename( cachedpath ) ) )

            exposures = decam_raw_origin_exposures.download_and_commit_exposures( indexes=expdexes, clobber=False,
                                                                              existing_ok=True, delete_downloads=False,
                                                                              session=session )
            for i, exposure in zip( expdexes, exposures ):
                eids.append( exposure.id )
                fname = pathlib.Path( decam_raw_origin_exposures._frame.iloc[i].archive_filename ).name

                # cache the files
                os.makedirs( os.path.join( cache_dir, 'DECam' ), exist_ok=True )
                if not os.path.isfile( os.path.join( cache_dir, 'DECam', fname ) ):
                    shutil.copy2( os.path.join( data_dir, fname ),
                                os.path.join( cache_dir, 'DECam', fname ) )

                match = re.search( r'^c4d_(?P<yymmdd>\d{6})_(?P<hhmmss>\d{6})_ori.fits', fname )
                assert match is not None
                # Todo : add the subdirectory to dbfname once that is implemented
                dbfname = ( f'c4d_20{match.group("yymmdd")}_{match.group("hhmmss")}_{exposure.filter[0]}_'
                            f'{exposure.provenance.id[0:6]}.fits' )
                assert exposure.filepath == dbfname
                assert ( pathlib.Path( exposure.get_fullpath( download=False ) ) ==
                         pathlib.Path( FileOnDiskMixin.local_path ) / exposure.filepath )
                assert pathlib.Path( exposure.get_fullpath( download=False ) ).is_file()
                archive_dir = archive.test_folder_path

                assert ( pathlib.Path( archive_dir ) / exposure.filepath ).is_file()
                # Perhaps do m5dsums to verify that the local and archive files are the same?
                # Or, just trust that the archive works because it has its own tests.
                assert exposure.instrument == 'DECam'
                assert exposure.mjd == pytest.approx( decam_raw_origin_exposures._frame.iloc[i]['MJD-OBS'], abs=1e-5)
                assert exposure.filter == decam_raw_origin_exposures._frame.iloc[i].ifilter

        # Make sure they're really in the database
        with SmartSession() as session:
            foundexps = session.query( Exposure ).filter( Exposure.id.in_( eids ) ).all()
            assert len(foundexps) == len(exposures)
            assert set( [ f.id for f in foundexps ] ) == set( [ e.id for e in exposures ] )
            assert set( [ f.filepath for f in foundexps ]) == set( [ e.filepath for e in exposures ] )
    finally:
        # Clean up
        with SmartSession() as session:
            exposures = session.query( Exposure ).filter( Exposure.id.in_( eids ) )
            for exposure in exposures:
                exposure.delete_from_disk_and_database( session=session, commit=False )
            session.commit()
            # remove downloaded files from data_dir (a cached version should remain)
            if 'downloaded' in locals():
                for d in downloaded:
                    path = os.path.join(data_dir, d['exposure'].name)
                    if os.path.isfile(path):
                        os.unlink(path)

            # remove downloaded files from data_dir (a cached version should remain)
            if 'downloaded' in locals():
                for d in downloaded:
                    path = os.path.join(data_dir, d['exposure'].name)
                    if os.path.isfile(path):
                        os.unlink(path)


@pytest.mark.skipif( os.getenv('RUN_SLOW_TESTS') is None, reason="Set RUN_SLOW_TESTS to run this test" )
def test_get_default_calibrators( decam_default_calibrators ):
    sections, filters = decam_default_calibrators
    decam = get_instrument_instance( 'DECam' )

    with SmartSession() as session:
        for sec in sections:
            for filt in filters:
                for ftype in [ 'flat', 'fringe', 'linearity' ]:
                    q = ( session.query( CalibratorFile )
                          .filter( CalibratorFile.instrument=='DECam' )
                          .filter( CalibratorFile.type==ftype )
                          .filter( CalibratorFile.sensor_section==sec )
                          .filter( CalibratorFile.calibrator_set=='externally_supplied' )
                         )
                    if ftype == 'flat':
                        q = q.filter( CalibratorFile.flat_type=='externally_supplied' )
                    if ftype != 'linearity':
                        q = q.join( Image ).filter( Image.filter==filt )

                    if ( ftype == 'fringe' ) and ( filt not in [ 'z', 'Y' ] ):
                        assert q.count() == 0
                    else:
                        assert q.count() == 1
                        cf = q.first()
                        assert cf.validity_start is None
                        assert cf.validity_end is None
                        if ftype == 'linearity':
                            assert cf.image_id is None
                            assert cf.datafile_id is not None
                            p = ( pathlib.Path( FileOnDiskMixin.local_path ) / cf.datafile.filepath )
                            assert p.is_file()
                        else:
                            assert cf.image_id is not None
                            assert cf.datafile_id is None
                            p = ( pathlib.Path( FileOnDiskMixin.local_path ) / cf.image.filepath )
                            assert p.is_file()


def test_linearity( decam_raw_image, decam_default_calibrators ):
    decam = get_instrument_instance( "DECam" )
    im = decam_raw_image
    origdata = im.data
    try:
        with SmartSession() as session:
            info = decam.preprocessing_calibrator_files( 'externally_supplied', 'externally_supplied',
                                                         im.section_id, im.filter_short, im.mjd, session=session )
            lindf = session.get( DataFile, info['linearity_fileid'] )
            im.data = decam.overscan_and_trim( im )
            assert im.data.shape == ( 4096, 2048 )
            newdata = decam.linearity_correct( im, linearitydata=lindf )

            # This is here to uncomment for debugging purposes
            # from astropy.io import fits
            # fits.writeto( 'trimmed.fits', im.data, im.header, overwrite=True )
            # fits.writeto( 'linearitied.fits', newdata, im.header, overwrite=True )

            # Brighter pixels should all have gotten brighter
            # (as nonlinearity will suppress bright pixels )
            w = np.where( im.data > 10000 )
            assert np.all( newdata[w] <= im.data[w] )

            # TODO -- figure out more ways to really test if this was done right
            #  (Could make this a regression test by putting in empirically what
            #  we get, but it'd be nice to actually make sure it really did the
            #  right thing.)
    finally:
        im.data = origdata


def test_preprocessing_calibrator_files( decam_default_calibrators ):
    decam = get_instrument_instance( "DECam" )

    linfile = None
    for filt in [ 'r', 'z' ]:
        info = decam.preprocessing_calibrator_files( 'externally_supplied', 'externally_supplied',
                                                     'N1', filt, 60000. )
        for nocalib in [ 'zero', 'dark', 'illumination' ]:
            # DECam doesn't include these three in its preprocessing steps
            assert f'{nocalib}_isimage' not in info.keys()
            assert f'{nocalib}_fileid' not in info.keys()
            assert f'{nocalib}_set' not in info.keys()
        assert info[ 'flat_isimage' ] == True
        assert 'flat_fileid' in info
        assert info[ 'linearity_isimage' ] == False
        assert 'linearity_fileid' in info
        if filt == 'r':
            assert info[ 'fringe_fileid' ] is None
        else:
            assert info[ 'fringe_isimage' ] == True
            assert 'fringe_fileid' in info
        with SmartSession() as session:
            im = session.get( Image, info[ 'flat_fileid' ] )
            assert im.filter == filt
            # The linearity file is the same for all sensor sections in DECam
            if linfile is None:
                linfile = session.get( DataFile, info[ 'linearity_fileid' ] )
            else:
                linfile = session.merge(linfile)
                assert linfile == session.get( DataFile, info[ 'linearity_fileid' ] )

    # Make sure we can call it a second time and don't get "file exists"
    # database errors (which would happen if _get_default_calibrators
    # gets called the second time around, which it should not.
    for filt in [ 'r', 'z' ]:
        info = decam.preprocessing_calibrator_files( 'externally_supplied', 'externally_supplied',
                                                     'N1', filt, 60000. )


def test_overscan_sections( decam_raw_image, data_dir,  ):
    decam = get_instrument_instance( "DECam" )

    ovsecs = decam.overscan_sections( decam_raw_image.header )
    assert ovsecs == [ { 'secname': 'A',
                         'biassec' : { 'x0': 6, 'x1': 56, 'y0': 0, 'y1': 4096 },
                         'datasec' : { 'x0': 56, 'x1': 1080, 'y0': 0, 'y1': 4096 }
                        },
                       { 'secname': 'B',
                         'biassec': { 'x0': 2104, 'x1': 2154, 'y0': 0, 'y1': 4096 },
                         'datasec': { 'x0': 1080, 'x1': 2104, 'y0': 0, 'y1': 4096 }
                        } ]


def test_overscan_and_data_sections( decam_raw_image, data_dir ):
    decam = get_instrument_instance( "DECam" )

    ovsecs = decam.overscan_and_data_sections( decam_raw_image.header )
    assert ovsecs == [ { 'secname': 'A',
                         'biassec' : { 'x0': 6, 'x1': 56, 'y0': 0, 'y1': 4096 },
                         'datasec' : { 'x0': 56, 'x1': 1080, 'y0': 0, 'y1': 4096 },
                         'destsec' : { 'x0': 0, 'x1': 1024, 'y0': 0, 'y1': 4096 }
                        },
                       { 'secname': 'B',
                         'biassec': { 'x0': 2104, 'x1': 2154, 'y0': 0, 'y1': 4096 },
                         'datasec': { 'x0': 1080, 'x1': 2104, 'y0': 0, 'y1': 4096 },
                         'destsec': { 'x0': 1024, 'x1': 2048, 'y0': 0, 'y1': 4096 }
                        } ]


def test_overscan( decam_raw_image, data_dir ):
    decam = get_instrument_instance( "DECam" )

    # Make sure it fails if it gets bad arguments
    with pytest.raises( TypeError, match='overscan_and_trim: pass either an Image as one argument' ):
        _ = decam.overscan_and_trim( 42 )
    with pytest.raises( RuntimeError, match='overscan_and_trim: pass either an Image as one argument' ):
        _ = decam.overscan_and_trim()
    with pytest.raises( RuntimeError, match='overscan_and_trim: pass either an Image as one argument' ):
        _ = decam.overscan_and_trim( 1, 2, 3 )
    with pytest.raises( TypeError, match="data isn't a numpy array" ):
        _ = decam.overscan_and_trim( decam_raw_image.header, 42 )

    rawdata = decam_raw_image.raw_data
    trimmeddata = decam.overscan_and_trim( decam_raw_image.header, rawdata )

    assert trimmeddata.shape == ( 4096, 2048 )

    # Spot check the image
    # These values are empirical from the actual image (using ds9)
    # (Remember numpy arrays are indexed y, x)
    rawleft = rawdata[ 2296:2297, 227:307 ]
    rawovleft = rawdata[ 2296:2297, 6:56 ]
    trimmedleft = trimmeddata[ 2296:2297, 171:251 ]
    rawright = rawdata[ 2170:2171, 1747:1827 ]
    rawovright = rawdata[ 2170:2171, 2104:2154 ]
    trimmedright = trimmeddata[ 2170:2171, 1691:1771 ]
    assert rawleft.mean() == pytest.approx( 2369.72, abs=0.01 )
    assert np.median( rawovleft ) == pytest.approx( 2176, abs=0.001 )
    assert trimmedleft.mean() == pytest.approx( rawleft.mean() - np.median(rawovleft), abs=0.01 )
    assert rawright.mean() == pytest.approx( 1615.78, abs=0.01 )
    assert np.median( rawovright ) == pytest.approx( 1435, abs=0.001 )
    assert trimmedright.mean() == pytest.approx( rawright.mean() - np.median(rawovright), abs=0.01 )
