import re
import logging
import pathlib
import hashlib
import pytest

import sqlalchemy as sa

from models.base import SmartSession, FileOnDiskMixin, _logger
from models.exposure import Exposure
from models.decam import DECam

@pytest.fixture(scope='module')
def decam_origin_exposures():
    decam = DECam()
    yield decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                       proposals='2023A-716082',
                                       skip_exposures_in_database=False,
                                       proc_type='instcal' )

# Note that these tests are probing the internal state of the opaque
# DECamOriginExposures objects.  If you're looking at this test for
# guidance for how to do things, do *not* write code that mucks about
# with the _frame member of one of those objects; that's internal state
# not intended for external consumption.
def test_decam_search_noirlab( decam_origin_exposures ):
    origloglevel = _logger.getEffectiveLevel()
    try:
        # Make sure we'll show the things sent to the noirlab API
        # so that we have a hope of figuring out what went wrong
        # if something does go wrong.
        _logger.setLevel( logging.DEBUG )

        decam = DECam()

        # Make sure it yells at us if we don't give any constraints
        with pytest.raises( RuntimeError ):
            decam.find_origin_exposures()

        # Make sure we can find some raw exposures without filter/proposals
        originexposures = decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                                       skip_exposures_in_database=False )
        assert len(originexposures._frame) == 9
        assert set(originexposures._frame.filtercode) == { 'g', 'V', 'i', 'r', 'z' }
        assert set(originexposures._frame.proposal) == { '2023A-716082', '2023A-921384' }


        # Make sure we can find raw exposures limiting by proposal
        originexposures = decam_origin_exposures
        assert all( [ originexposures._frame.prod_type[i] == 'image' for i in range(len(originexposures)) ] )
        assert set(originexposures._frame.filtercode) == { 'i', 'r', 'g', 'z' }
        assert all( [ originexposures._frame.proposal[i] == '2023A-716082' for i in range(len(originexposures)) ] )

        # Make sure we can find raw exposures limiting by filter
        originexposures = decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                                       filters='r',
                                                       skip_exposures_in_database=False )
        assert len(originexposures._frame) == 2
        assert set(originexposures._frame.filtercode) == { 'r' }

        originexposures = decam.find_origin_exposures( minmjd=60159.15625, maxmjd=60159.16667,
                                                       filters=[ 'r', 'g' ],
                                                       skip_exposures_in_database=False )
        assert len(originexposures._frame) == 4
        assert set(originexposures._frame.filtercode) == { 'r', 'g' }
    finally:
        _logger.setLevel( origloglevel )

def test_decam_download_origin_exposure( decam_origin_exposures ):
    localpath = FileOnDiskMixin.local_path
    assert all( [ row.proc_type=='instcal' for i,row in decam_origin_exposures._frame.iterrows() ] )
    downloaded = decam_origin_exposures.download_exposures( outdir=localpath, indexes=[ 1, 3 ],
                                                            clobber=False, existing_ok=True )
    try:
        assert len(downloaded) == 2
        for path, dex in zip( downloaded, [ 1, 3 ] ):
            md5 = hashlib.md5()
            with open( path, "rb") as ifp:
                md5.update( ifp.read() )
            assert md5.hexdigest() == decam_origin_exposures._frame.iloc[dex].md5sum
    finally:
        # Don't clean up for efficiency of rerunning tests.
        # for path in downloaded:
        #     path.unlink()
        pass

def test_decam_download_and_commit_exposure( code_version, decam_origin_exposures ):
    eids = []
    try:
        with SmartSession() as session:
            # Deliberately choosing different indices here than in test_decam_download_orign_exposure
            # so that we'll make sure the download works when called from download_and_commit_exposures
            expdexes = [ 1, 2 ]
            exposures = decam_origin_exposures.download_and_commit_exposures( indexes=expdexes, clobber=False,
                                                                              existing_ok=True, delete_downloads=False,
                                                                              session=session )
            for i, exposure in zip( expdexes, exposures ):
                eids.append( exposure.id )
                fname = pathlib.Path( decam_origin_exposures._frame.iloc[i].archive_filename ).name
                match = re.search( r'^c4d_(?P<yymmdd>\d{6})_(?P<hhmmss>\d{6})_ooi_._v1\.fits', fname )
                assert match is not None
                # Todo : add the subdirectory to dbfname once that is implemented
                dbfname = ( f'c4d_20{match.group("yymmdd")}_{match.group("hhmmss")}_{exposure.filter[0]}_'
                            f'{exposure.provenance.id[0:6]}.fits' )
                assert exposure.filepath == dbfname
                assert ( pathlib.Path( exposure.get_fullpath( download=False ) ) ==
                         pathlib.Path( FileOnDiskMixin.local_path ) / exposure.filepath )
                assert pathlib.Path( exposure.get_fullpath( download=False ) ).is_file()
                assert ( pathlib.Path( '/archive_storage/base/test' ) / exposure.filepath ).is_file()
                # Perhaps do m5dsums to verify that the local and archive files are the same?
                # Or, just trust that the archive works because it has its own tests.
                assert exposure.instrument == 'DECam'
                assert exposure.mjd == pytest.approx( decam_origin_exposures._frame.iloc[i]['MJD-OBS'], abs=1e-5)
                assert exposure.filter == decam_origin_exposures._frame.iloc[i].ifilter

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
                for base in [ pathlib.Path( FileOnDiskMixin.local_path ),
                              pathlib.Path( '/archive_storage/base/test' ) ]:
                    path = base / exposure.filepath
                    if path.is_file():
                        path.unlink()
            session.execute( sa.delete( Exposure ).where( Exposure.id.in_( eids ) ) )
            session.commit()
            # Not deleteing the original downloaded exposures so that rerunning
            #  tests will be fast (as they'll already be there).
            # Reinitialize the test environment (including cleaning out
            #  non-git-tracked files from data) to force verification of
            #  downloads; this will always happen on github actions.
