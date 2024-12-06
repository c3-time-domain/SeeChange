import os
import pytest
import hashlib
import uuid

import numpy as np

from astropy.wcs import WCS
from astropy.io import fits

from util.exceptions import BadMatchException
from models.base import SmartSession, CODE_ROOT
from models.image import Image
from models.world_coordinates import WorldCoordinates

from util.util import env_as_bool

from tests.conftest import SKIP_WARNING_TESTS


def test_solve_wcs_scamp_failures( ztf_gaia_dr3_excerpt, ztf_datastore_uncommitted, astrometor ):
    catexp = ztf_gaia_dr3_excerpt
    ds = ztf_datastore_uncommitted

    astrometor.pars.method = 'scamp'
    astrometor.pars.max_resid = 0.01

    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        _ = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    # Make sure it fails if we give it too small of a crossid radius.
    # Note that this one is passed directly to _solve_wcs_scamp.
    # _solve_wcs_scamp doesn't read what we pass to AstroCalibrator
    # constructor, because that is an array of crossid_radius values to
    # try, whereas _solve_wcs_scamp needs a single value.  (The
    # iteration happens outside _solve_wcs_scamp.)

    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        _ = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp, crossid_radius=0.01 )

    astrometor.pars.min_frac_matched = 0.8
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        _ = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    astrometor.pars.min_matched_stars = 50
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        _ = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )


def test_solve_wcs_scamp( ztf_gaia_dr3_excerpt, ztf_datastore_uncommitted, astrometor, blocking_plots ):
    catexp = ztf_gaia_dr3_excerpt
    ds = ztf_datastore_uncommitted

    # Make True for visual testing purposes
    if env_as_bool('INTERACTIVE'):
        basename = os.path.join(CODE_ROOT, 'tests/plots')
        catexp.ds9_regfile( os.path.join(basename, 'catexp.reg'), radius=4 )
        ds.sources.ds9_regfile( os.path.join(basename, 'sources.reg'), radius=3 )

    orighdr = ds.image._header.copy()

    astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    # Because this was a ZTF image that had a WCS already, the new WCS
    # should be damn close, but not identical (since there's no way we
    # used exactly the same set of sources and stars, plus this was a
    # cropped ZTF image, not the full image).
    allsame = True
    for i in [ 1, 2 ]:
        for j in range( 17 ):
            diff = np.abs( ( orighdr[f'PV{i}_{j}'] - ds.image._header[f'PV{i}_{j}'] ) / orighdr[f'PV{i}_{j}'] )
            if diff > 1e-6:
                allsame = False
                break
    assert not allsame

    # ...but check that they are close
    wcsold = WCS( orighdr )
    scolds = wcsold.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    wcsnew = WCS( ds.image._header )
    scnews = wcsnew.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    for scold, scnew in zip( scolds, scnews ):
        assert scold.ra.value == pytest.approx( scnew.ra.value, abs=1./3600. )
        assert scold.dec.value == pytest.approx( scnew.dec.value, abs=1./3600. )


def test_run_scamp( decam_datastore_through_bg, astrometor ):
    ds = decam_datastore_through_bg

    # Get the md5sum and WCS from the image before we do things to it
    with open(ds.path_to_original_image, "rb") as ifp:
        md5 = hashlib.md5()
        md5.update(ifp.read())
        origmd5 = uuid.UUID(md5.hexdigest())

    xvals = [0, 0, 2047, 2047]
    yvals = [0, 4095, 0, 4095]
    with fits.open(ds.path_to_original_image) as hdu:
        origwcs = WCS(hdu[ds.section_id].header)

    astrometor.pars.cross_match_catalog = 'gaia_dr3'
    astrometor.pars.solution_method = 'scamp'
    astrometor.pars.max_catalog_mag = [20.]
    astrometor.pars.mag_range_catalog = 4.
    astrometor.pars.min_catalog_stars = 50
    astrometor.pars.max_resid = 0.15
    astrometor.pars.crossid_radii = [2.0]
    astrometor.pars.min_frac_matched = 0.1
    astrometor.pars.min_matched_stars = 10
    astrometor.pars.test_parameter = uuid.uuid4().hex  # make sure it gets a different Provenance

    # The datastore should object when it tries to get the provenance for astrometor
    # params that don't match what we started with
    ds = astrometor.run(ds)
    exc = ds.read_exception()
    assert exc is not None
    assert str(exc) == ( "DataStore getting provenance for extraction whose parameters don't match "
                         "the parameters of the same process in the prov_tree" )

    # Wipe the datastore prov_tree so that we can
    #   run something with paramaters that are
    #   different from what's in there.
    # (This is doing it "wrong", because we're now
    #   going to generate a WCS in the datastore
    #   whose provenance is different from the
    #   provenance of sources.  Doing that for this
    #   test here, but the production pipeline should
    #   never do that.  (Not setting ds.prov_tree to
    #   None would have caught that in this case.))
    ds.prov_tree = None
    ds = astrometor.run(ds)

    assert astrometor.has_recalculated

    # Make sure that the new WCS is different from the original wcs
    # (since we know the one that came in the decam exposure is approximate)
    # BUT, make sure that it's within 40", because the original one, while
    # not great, is *something*
    origscs = origwcs.pixel_to_world( xvals, yvals )
    newscs = ds.wcs.wcs.pixel_to_world( xvals, yvals )
    for origsc, newsc in zip( origscs, newscs ):
        assert not origsc.ra.value == pytest.approx( newsc.ra.value, abs=1./3600. )
        assert not origsc.dec.value == pytest.approx( newsc.dec.value, abs=1./3600. )
        assert origsc.ra.value == pytest.approx( newsc.ra.value, abs=40./3600. )   # cos(dec)...
        assert origsc.dec.value == pytest.approx( newsc.dec.value, abs=40./3600. )

    # NOTE -- because of the cache, the image may well have the "astro_cal_done" flag
    #  set even though we're using the decam_datastore_through_bg fixture, which doesn't
    #  do astro_cal.  So, we can't check that.  But, we know that we've done it,
    #  so we know that we want to update the image header.
    ds.save_and_commit( update_image_header=True, overwrite=True )

    with SmartSession() as session:
        # Make sure the WCS made it into the database
        # (It should be the only one attached to this ds.sources since the fixture only
        # went through backgrounding.)
        q = session.query( WorldCoordinates ).filter( WorldCoordinates.sources_id == ds.sources.id )
        assert q.count() == 1
        dbwcs = q.first()
        dbscs = dbwcs.wcs.pixel_to_world( xvals, yvals )
        for newsc, dbsc in zip( newscs, dbscs ):
            assert dbsc.ra.value == pytest.approx( newsc.ra.value, abs=0.01/3600. )
            assert dbsc.dec.value == pytest.approx( newsc.dec.value, abs=0.01/3600. )

        # Make sure the image got updated properly on the database
        # and on disk
        q = session.query( Image ).filter( Image._id == ds.image.id )
        assert q.count() == 1
        foundim = q.first()
        assert foundim.md5sum_extensions[0] == ds.image.md5sum_extensions[0]
        assert foundim.md5sum_extensions[0] != origmd5
        with open( foundim.get_fullpath()[0], 'rb' ) as ifp:
            md5 = hashlib.md5()
            md5.update( ifp.read() )
            assert uuid.UUID( md5.hexdigest() ) == foundim.md5sum_extensions[0]
        # This is probably redundant given the md5sum test we just did....
        ds.image._header = None
        for kw in foundim.header:
            # SIMPLE can't be an index to a Header.  (This is sort
            # of a weird thing in the astropy Header interface.)
            # BITPIX doesn't match because the ds.image raw header
            # was constructed from the exposure that had been
            # BSCALEd, even though the image we wrote to disk fully
            # a float (BITPIX=-32).
            if kw in [ 'SIMPLE', 'BITPIX' ]:
                continue
            assert foundim.header[kw] == ds.image.header[kw]

        # Make sure the new WCS got written to the FITS file
        with fits.open( foundim.get_fullpath()[0] ) as hdul:
            imhdr = hdul[0].header
        imwcs = WCS( imhdr )
        imscs = imwcs.pixel_to_world( xvals, yvals )
        for newsc, imsc in zip( newscs, imscs ):
            assert newsc.ra.value == pytest.approx( imsc.ra.value, abs=0.05/3600. )
            assert newsc.dec.value == pytest.approx( imsc.dec.value, abs=0.05/3600. )

        # Make sure the archive has the right md5sum
        info = foundim.archive.get_info( f'{foundim.filepath}.image.fits' )
        assert info is not None
        assert uuid.UUID( info['md5sum'] ) == foundim.md5sum_extensions[0]


# TODO : test that it fails when it's supposed to


def test_warnings_and_exceptions(decam_datastore, astrometor):

    # Wipe the datastore's prov_tree so we get the exceptions we're looking for,
    #   not an exception about a provenance parameters mismatch.
    decam_datastore.prov_tree = None

    if not SKIP_WARNING_TESTS:
        astrometor.pars.inject_warnings = 1
        with pytest.warns(UserWarning) as record:
            astrometor.run(decam_datastore)
        assert decam_datastore.exception is None
        assert len(record) > 0
        assert any("Warning injected by pipeline parameters in process 'astrocal'." in str(w.message) for w in record)

    astrometor.pars.inject_warnings = 0
    astrometor.pars.inject_exceptions = 1
    with pytest.raises(Exception) as excinfo:
        ds = astrometor.run(decam_datastore)
        ds.reraise()
    assert "Exception injected by pipeline parameters in process 'astrocal'." in str(excinfo.value)
    ds.read_exception()
