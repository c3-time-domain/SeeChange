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


def test_solve_wcs_scamp_failures( ztf_gaia_dr3_excerpt, ztf_datastore_uncommitted, astrometor ):
    catexp = ztf_gaia_dr3_excerpt
    ds = ztf_datastore_uncommitted

    astrometor.pars.method = 'scamp'
    astrometor.pars.max_resid = 0.01

    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    # Make sure it fails if we give it too small of a crossid radius.
    # Note that this one is passed directly to _solve_wcs_scamp.
    # _solve_wcs_scamp doesn't read what we pass to AstroCalibrator
    # constructor, because that is an array of crossid_radius values to
    # try, whereas _solve_wcs_scamp needs a single value.  (The
    # iteration happens outside _solve_wcs_scamp.)

    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp, crossid_radius=0.01 )

    astrometor.pars.min_frac_matched = 0.8
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )

    astrometor.pars.min_matched_stars = 50
    with pytest.raises( BadMatchException, match="which isn.*t good enough" ):
        wcs = astrometor._solve_wcs_scamp( ds.image, ds.sources, catexp )


def test_solve_wcs_scamp( ztf_gaia_dr3_excerpt, ztf_datastore_uncommitted, astrometor, blocking_plots ):
    catexp = ztf_gaia_dr3_excerpt
    ds = ztf_datastore_uncommitted

    # Make True for visual testing purposes
    if os.getenv('INTERACTIVE', False):
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

    #...but check that they are close
    wcsold = WCS( orighdr )
    scolds = wcsold.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    wcsnew = WCS( ds.image._header )
    scnews = wcsnew.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    for scold, scnew in zip( scolds, scnews ):
        assert scold.ra.value == pytest.approx( scnew.ra.value, abs=1./3600. )
        assert scold.dec.value == pytest.approx( scnew.dec.value, abs=1./3600. )


def test_run_scamp( decam_datastore, astrometor ):
    ds = decam_datastore
    original_filename = ds.cache_base_name + '.image.fits.original'
    with open(original_filename, "rb") as ifp:
        md5 = hashlib.md5()
        md5.update(ifp.read())
        origmd5 = uuid.UUID(md5.hexdigest())

    xvals = [0, 0, 2047, 2047]
    yvals = [0, 4095, 0, 4095]
    with fits.open(original_filename) as hdu:
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

    # These next few lines will need to be done after astrometry is done.  Right now,
    # we don't do saving and committing inside the Astrometor.run method.
    update_image_header = False
    if not ds.image.astro_cal_done:
        ds.image.astro_cal_done = True
        update_image_header = True
    ds.save_and_commit( update_image_header=update_image_header, overwrite=True )

    with SmartSession() as session:
        # Make sure the WCS made it into the databse
        q = ( session.query( WorldCoordinates )
              .filter( WorldCoordinates.sources_id == ds.sources.id )
              .filter( WorldCoordinates.provenance_id == ds.wcs.provenance.id ) )
        assert q.count() == 1
        dbwcs = q.first()
        dbscs = dbwcs.wcs.pixel_to_world( xvals, yvals )
        for newsc, dbsc in zip( newscs, dbscs ):
            assert dbsc.ra.value == pytest.approx( newsc.ra.value, abs=0.01/3600. )
            assert dbsc.dec.value == pytest.approx( newsc.dec.value, abs=0.01/3600. )

        # Make sure the image got updated properly on the database
        # and on disk
        q = session.query( Image ).filter( Image.id == ds.image.id )
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
        imwcs = WCS( hdul[0].header )
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
    astrometor.pars.inject_warnings = 1

    with pytest.warns(UserWarning) as record:
        astrometor.run(decam_datastore)
    assert len(record) > 0
    assert any('Warning injected by pipeline parameters in process "astro_cal".' in str(w.message) for w in record)

    astrometor.pars.inject_warnings = 0
    astrometor.pars.inject_exceptions = 1
    with pytest.raises(Exception) as excinfo:
        ds = astrometor.run(decam_datastore)
        ds.reraise()
    assert 'Exception injected by pipeline parameters in process "astro_cal"' in str(excinfo.value)
