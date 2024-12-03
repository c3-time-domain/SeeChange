import os
import re
import shutil
import pathlib
import hashlib
import pytest

import numpy as np

from astropy.io import fits

from models.base import SmartSession, FileOnDiskMixin
from models.exposure import Exposure
from models.knownexposure import KnownExposure
from models.instrument import get_instrument_instance
from models.datafile import DataFile
from models.calibratorfile import CalibratorFile
from models.image import Image
from models.instrument import Instrument
from models.decam import DECam

import util.radec
from util.logger import SCLogger
from util.util import env_as_bool


def test_decam_exposure(decam_exposure):
    e = decam_exposure

    assert e.instrument == 'DECam'
    assert isinstance(e.instrument_object, DECam)
    assert e.telescope == 'CTIO 4.0-m telescope'
    assert e.mjd == 60127.33963431
    assert e.end_mjd == 60127.34062968037
    assert e.ra == 7.874804166666666
    assert e.dec == -43.0096
    assert e.exp_time == 86.0
    assert e.filepath == 'c4d_230702_080904_ori.fits.fz'
    assert e.filter == 'r DECam SDSS c0002 6415.0 1480.0'
    assert not e.from_db
    assert e.info == {}
    assert e.target == 'ELAIS-E1'
    assert e.project == '2023A-716082'

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


def test_image_from_decam_exposure(decam_exposure, provenance_base, data_dir):
    e = decam_exposure
    sec_id = 'N4'
    im = Image.from_exposure(e, section_id=sec_id)  # load the first CCD

    assert e.instrument == 'DECam'
    assert e.telescope == 'CTIO 4.0-m telescope'
    assert not im.from_db
    # should not be the same as the exposure!
    assert im.ra != e.ra
    assert im.dec != e.dec
    assert im.ra == 7.878344279652849
    assert im.dec == -43.0961474371319
    assert im.mjd == 60127.33963431
    assert im.end_mjd == 60127.34062968037
    assert im.exp_time == 86.0
    assert im.filter == 'r DECam SDSS c0002 6415.0 1480.0'
    assert im.target == 'ELAIS-E1'
    assert im.project == '2023A-716082'
    assert im.section_id == sec_id

    assert im._id is None
    assert im.filepath is None

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
@pytest.mark.skipif( env_as_bool('SKIP_NOIRLAB_DOWNLOADS'), reason="SKIP_NOIRLAB_DOWNLOADS is set" )
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
        originexposures = decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667, proc_type='instcal' )
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

        # Make sure that we can search based on RA, Dec center
        originexposures = decam.find_origin_exposures( minmjd=59893.18, maxmjd=60008.01,
                                                       proc_type='instcal', ctr_ra=30., ctr_dec=-7., radius=1. )
        assert len(originexposures._frame.index.levels[0]) == 4
        assert set(originexposures._frame.index.levels[1]) == { 'image', 'wtmap', 'dqmask' }
        assert set(originexposures._frame.filtercode) == { 'r', 'i', 'z' }
        assert set(originexposures._frame.proposal) == { '2020B-0053', '2021B-0909' }
        ras = originexposures._frame.xs( 'image', level='prod_type' ).ra_center
        decs = originexposures._frame.xs( 'image', level='prod_type' ).dec_center
        assert all( ras > 30. - 1. / np.cos( -7 * np.pi/180. ) )
        assert all( ras < 30. + 1. / np.cos( -7 * np.pi/180. ) )
        assert all( decs > -7 - 1. )
        assert all( decs < -7 + 1. )

    finally:
        SCLogger.setLevel( origloglevel )


@pytest.mark.skipif( env_as_bool('SKIP_NOIRLAB_DOWNLOADS'), reason="SKIP_NOIRLAB_DOWNLOADS is set" )
def test_decam_search_noirlab_dedup():
    # Make sure that if we do a search we know has duplicates in it, we don't get
    #  duplicate exposures

    decam = DECam()
    originexposures = decam.find_origin_exposures( minmjd=57664.25347, maxmjd=57664.25556, proc_type='instcal' )
    assert len(originexposures) == 1
    assert ( originexposures.exposure_origin_identifier(0) ==
             '/net/archive/pipe/20161002/ct4m/2012B-0001/c4d_161003_060737_ooi_r_v1.fits.fz' )


@pytest.mark.skipif( env_as_bool('SKIP_NOIRLAB_DOWNLOADS'), reason="SKIP_NOIRLAB_DOWNLOADS is set" )
def test_decam_download_reduced_origin_exposure( decam_reduced_origin_exposures, cache_dir ):

    # See comment in test_decam_download_and_commit_exposure.
    # In the past, we've tested this with a list of two items,
    # which is good, but this is not a fast test, so reduce
    # it to one item.  Leave the list of two commented out here
    # so that we can go back to it trivially (and so we might
    # remember that once we tested that).
    # whichtodownload = [ 1, 3 ]
    whichtodownload = [ 3 ]

    assert all( [ row.proc_type == 'instcal' for i, row in decam_reduced_origin_exposures._frame.iterrows() ] )
    try:
        # First try downloading the reduced exposures themselves
        downloaded = decam_reduced_origin_exposures.download_exposures(
            outdir=os.path.join(cache_dir, 'DECam'),
            indexes=whichtodownload,
            onlyexposures=True,
            clobber=False,
            existing_ok=True,
        )
        assert len(downloaded) == len(whichtodownload)
        for pathdict, dex in zip( downloaded, whichtodownload ):
            assert set( pathdict.keys() ) == { 'exposure' }
            md5 = hashlib.md5()
            with open( pathdict['exposure'], "rb") as ifp:
                md5.update( ifp.read() )
            assert md5.hexdigest() == decam_reduced_origin_exposures._frame.loc[ dex, 'image' ].md5sum

        # Now try downloading exposures, weights, and dataquality masks
        downloaded = decam_reduced_origin_exposures.download_exposures(
            outdir=os.path.join(cache_dir, 'DECam'),
            indexes=whichtodownload,
            onlyexposures=False,
            clobber=False,
            existing_ok=True,
        )
        assert len(downloaded) == len(whichtodownload)
        for pathdict, dex in zip( downloaded, whichtodownload ):
            assert set( pathdict.keys() ) == { 'exposure', 'wtmap', 'dqmask' }
            for extname, extpath in pathdict.items():
                if extname == 'exposure':
                    # Translation back to what we know is in the internal dataframe
                    extname = 'image'
                md5 = hashlib.md5()
                with open( extpath, "rb" ) as ifp:
                    md5.update( ifp.read() )
                assert md5.hexdigest() == decam_reduced_origin_exposures._frame.loc[ dex, extname ].md5sum

    finally:  # cleanup
        for d in downloaded:
            for path in d.values():
                if os.path.isfile( path ):
                    os.unlink( path )

@pytest.mark.skipif( env_as_bool('SKIP_NOIRLAB_DOWNLOADS'), reason="SKIP_NOIRLAB_DOWNLOADS is set" )
def test_decam_download_and_commit_reduced_origin_exposure( decam_reduced_origin_exposures ):
    # See test_decam_download_and_commit_exposure for downloading a raw
    #  exposure.  That one pokes into details a bit more too.
    exps = []
    try:
        exps = decam_reduced_origin_exposures.download_and_commit_exposures( indexes=[0] )
        assert len(exps) == 1
        assert isinstance( exps[0], Exposure )
        assert exps[0].filepath_extensions == [ '.image.fits.fz', '.weight.fits.fz', '.flags.fits.fz' ]
        fpaths = exps[0].get_fullpath()
        assert all( os.path.isfile(p) for p in fpaths )
        # Make sure it's actually in the database
        assert Exposure.get_by_id( exps[0].id ) is not None
    finally:
        for e in exps:
            e.delete_from_disk_and_database()


@pytest.mark.skipif( env_as_bool('SKIP_NOIRLAB_DOWNLOADS'), reason="SKIP_NOIRLAB_DOWNLOADS is set" )
def test_add_to_known_exposures( decam_raw_origin_exposures ):
    # I'm looking inside the decam_raw_origin_exposures structure,
    #  which you're not supposed to do.  This means if the
    #  internal implementation changes, even if the interface
    #  doesn't, I may need to rewrite the test... oh well.
    # (To fix it, we'd need a method that extracts the identifiers
    #  from the opaque origin exposures object.)
    identifiers = [ pathlib.Path( decam_raw_origin_exposures._frame.loc[i,'image'].archive_filename ).name
                    for i in [1,2] ]
    try:
        decam_raw_origin_exposures.add_to_known_exposures( [1, 2] )

        with SmartSession() as session:
            kes = session.query( KnownExposure ).filter( KnownExposure.identifier.in_( identifiers ) ).all()
            assert len(kes) == 2
            assert { k.identifier for k in kes } == { 'c4d_230702_081059_ori.fits.fz', 'c4d_230702_080904_ori.fits.fz' }
            assert all( [ k.instrument == 'DECam' for k in kes ] )
            assert all( [ k.params['url'][0:45] == 'https://astroarchive.noirlab.edu/api/retrieve' for k in kes ] )

    finally:
        with SmartSession() as session:
            kes = session.query( KnownExposure ).filter( KnownExposure.identifier.in_( identifiers ) )
            for ke in kes:
                session.delete( ke )
            session.commit()

    # Make sure all get added when add_to_known_exposures is called with no arguments
    identifiers = [ pathlib.Path( decam_raw_origin_exposures._frame.loc[i,'image'].archive_filename ).name
                    for i in range(len(decam_raw_origin_exposures)) ]
    try:
        decam_raw_origin_exposures.add_to_known_exposures()

        with SmartSession() as session:
            kes = session.query( KnownExposure ).filter( KnownExposure.identifier.in_( identifiers ) )
            assert kes.count() == len( decam_raw_origin_exposures )
            assert all( [ k.instrument == 'DECam' for k in kes ] )
            assert all( [ k.params['url'][0:45] == 'https://astroarchive.noirlab.edu/api/retrieve' for k in kes ] )
    finally:
        with SmartSession() as session:
            kes = session.query( KnownExposure ).filter( KnownExposure.identifier.in_( identifiers ) )
            for ke in kes:
                session.delete( ke )
            session.commit()


@pytest.mark.skipif( env_as_bool('SKIP_NOIRLAB_DOWNLOADS'), reason="SKIP_NOIRLAB_DOWNLOADS is set" )
def test_decam_download_and_commit_exposure(
        code_version, decam_raw_origin_exposures, cache_dir, data_dir, test_config, archive
):
    # This one does a raw exposure;
    # test_decam_download_and_commit_reduced_origin_exposures downloads one
    # processed through the NOIRLab pipeline.
    # (This one also digs in a bit deeper.)

    eids = []
    try:
        with SmartSession() as session:
            # ...yes, it's nice to test that this works with a list, and
            # we did that for a long time, but this is a slow process
            # (how slow depends on how the NOIRLab servers are doing,
            # but each exposure download is typically tens of seconds to
            # a few minutes) and leaves big files on disk (in the
            # cache), so just test it with a single exposure.  Leave
            # this commented out here in case somebody comes back and
            # thinks, hmm, better test this with more than one exposure.
            # expdexes = [ 1, 2 ]
            expdexes = [ 1 ]

            # get these downloaded first, to get the filenames to check against the cache
            downloaded = decam_raw_origin_exposures.download_exposures(
                outdir=os.path.join(cache_dir, 'DECam'),
                indexes=expdexes,
                onlyexposures=True,
                clobber=False,
                existing_ok=True,
            )
            # Make sure we only get exposures
            assert len(downloaded) == len(expdexes)
            assert all( list(d.keys())==['exposure'] for d in downloaded )

            # Copy out of cache
            for pathdict in downloaded:
                cachedpath = pathdict['exposure']
                assert os.path.isfile( cachedpath )
                shutil.copy2( cachedpath, os.path.join( data_dir, os.path.basename( cachedpath ) ) )

            # This could download again, but in this case won't because it will see the
            # files are already in place with the right md5sum
            exposures = decam_raw_origin_exposures.download_and_commit_exposures( indexes=expdexes, clobber=False,
                                                                              existing_ok=True, delete_downloads=False,
                                                                              session=session )
            for i, exposure in zip( expdexes, exposures ):
                eids.append( exposure.id )
                fname = pathlib.Path( decam_raw_origin_exposures._frame.iloc[i].archive_filename ).name

                match = re.search( r'^c4d_(?P<yymmdd>\d{6})_(?P<hhmmss>\d{6})_ori.fits', fname )
                assert match is not None
                # Todo : add the subdirectory to dbfname once that is implemented
                dbfname = ( f'c4d_20{match.group("yymmdd")}_{match.group("hhmmss")}_{exposure.filter[0]}_'
                            f'{exposure.provenance_id[0:6]}.fits' )
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
            foundexps = session.query( Exposure ).filter( Exposure._id.in_( eids ) ).all()
            assert len(foundexps) == len(exposures)
            assert set( [ f.id for f in foundexps ] ) == set( [ e.id for e in exposures ] )
            assert set( [ f.filepath for f in foundexps ]) == set( [ e.filepath for e in exposures ] )
    finally:
        # Clean up
        with SmartSession() as session:
            exposures = session.query( Exposure ).filter( Exposure._id.in_( eids ) )
        for exposure in exposures:
            exposure.delete_from_disk_and_database()
        if 'downloaded' in locals():
            for d in downloaded:
                path = os.path.join(data_dir, d['exposure'].name)
                if os.path.isfile(path):
                    os.unlink(path)
                if os.path.isfile(d['exposure']):
                    os.unlink(d['exposure'])

# This test really isn't *that* slow.  Not compared to so many others nowadays.
# @pytest.mark.skipif( not env_as_bool('RUN_SLOW_TESTS'), reason="Set RUN_SLOW_TESTS to run this test" )
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
                            df = DataFile.get_by_id( cf.datafile_id, session=session )
                            p = ( pathlib.Path( FileOnDiskMixin.local_path ) / df.filepath )
                            assert p.is_file()
                        else:
                            assert cf.image_id is not None
                            assert cf.datafile_id is None
                            i = Image.get_by_id( cf.image_id, session=session )
                            p = ( pathlib.Path( FileOnDiskMixin.local_path ) / i.filepath )
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
                                                     'S3', filt, 60000. )
        for nocalib in [ 'zero', 'dark' ]:
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
                                                     'S3', filt, 60000. )


def test_overscan_sections( decam_raw_image, data_dir,  ):
    decam = get_instrument_instance( "DECam" )

    ovsecs = decam.overscan_sections( decam_raw_image.header )
    assert ovsecs == [ { 'secname' : 'A',
                         'biassec' : { 'x0': 2104, 'x1': 2154, 'y0': 50, 'y1': 4146 },
                         'datasec' : { 'x0': 1080, 'x1': 2104, 'y0': 50, 'y1': 4146 }
                        },
                       { 'secname' : 'B',
                         'biassec' : { 'x0': 6, 'x1': 56, 'y0': 50, 'y1': 4146 },
                         'datasec' : { 'x0': 56, 'x1': 1080, 'y0': 50, 'y1': 4146 }
                        } ]


def test_overscan_and_data_sections( decam_raw_image, data_dir ):
    decam = get_instrument_instance( "DECam" )

    ovsecs = decam.overscan_and_data_sections( decam_raw_image.header )
    assert ovsecs == [ { 'secname': 'A',
                         'biassec' : { 'x0': 2104, 'x1': 2154, 'y0': 50, 'y1': 4146 },
                         'datasec' : { 'x0': 1080, 'x1': 2104, 'y0': 50, 'y1': 4146 },
                         'destsec' : { 'x0': 1024, 'x1': 2048, 'y0': 0, 'y1': 4096 }
                        },
                       { 'secname' : 'B',
                         'biassec' : { 'x0': 6, 'x1': 56, 'y0': 50, 'y1': 4146 },
                         'datasec' : { 'x0': 56, 'x1': 1080, 'y0': 50, 'y1': 4146 },
                         'destsec' : { 'x0': 0, 'x1': 1024, 'y0': 0, 'y1': 4096 }
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
    rawleft = rawdata[ 2296:2297, 100:168 ]
    rawovleft = rawdata[ 2296:2297, 6:56 ]
    trimmedleft = trimmeddata[ 2246:2247, 44:112 ]
    rawright = rawdata[ 2296:2297, 1900:1968 ]
    rawovright = rawdata[ 2296:2297, 2104:2154 ]
    trimmedright = trimmeddata[ 2246:2247, 1844:1912 ]
    assert rawleft.mean() == pytest.approx( 3823.5882, abs=0.01 )
    assert np.median( rawovleft ) == pytest.approx( 1660, abs=0.001 )
    assert trimmedleft.mean() == pytest.approx( rawleft.mean() - np.median(rawovleft), abs=0.1 )
    assert rawright.mean() == pytest.approx( 4530.8971, abs=0.01 )
    assert np.median( rawovright ) == pytest.approx( 2209, abs=0.001 )
    assert trimmedright.mean() == pytest.approx( rawright.mean() - np.median(rawovright), abs=0.1 )
