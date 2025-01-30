import pytest

import numpy as np
import sqlalchemy as sa

from astropy.time import Time

from models.base import SmartSession, FourCorners
from models.image import Image

from tests.fixtures.simulated import ImageCleanup


def test_image_coordinates():
    image = Image('coordinates.fits', ra=None, dec=None, nofile=True)
    assert image.ecllat is None
    assert image.ecllon is None
    assert image.gallat is None
    assert image.gallon is None

    image = Image('coordinates.fits', ra=123.4, dec=None, nofile=True)
    assert image.ecllat is None
    assert image.ecllon is None
    assert image.gallat is None
    assert image.gallon is None

    image = Image('coordinates.fits', ra=123.4, dec=56.78, nofile=True)
    assert abs(image.ecllat - 35.846) < 0.01
    assert abs(image.ecllon - 111.838) < 0.01
    assert abs(image.gallat - 33.542) < 0.01
    assert abs(image.gallon - 160.922) < 0.01


def test_image_cone_search( provenance_base ):
    image1 = None
    image2 = None
    image3 = None
    image4 = None
    try:
        rng = np.random.default_rng()
        kwargs = { 'format': 'fits',
                   'exp_time': 60.48,
                   'section_id': 'x',
                   'project': 'x',
                   'target': 'x',
                   'instrument': 'DemoInstrument',
                   'telescope': 'x',
                   'filter': 'r',
                   'ra_corner_00': 0,
                   'ra_corner_01': 0,
                   'ra_corner_10': 0,
                   'ra_corner_11': 0,
                   'minra': 0,
                   'maxra': 0,
                   'dec_corner_00': 0,
                   'dec_corner_01': 0,
                   'dec_corner_10': 0,
                   'dec_corner_11': 0,
                   'mindec': 0,
                   'maxdec': 0,
                  }
        image1 = Image(ra=120., dec=10., provenance_id=provenance_base.id, **kwargs )
        image1.mjd = rng.uniform(0, 1) + 60000
        image1.end_mjd = image1.mjd + 0.007
        _1 = ImageCleanup.save_image( image1 )
        image1.insert()

        image2 = Image(ra=120.0002, dec=9.9998, provenance_id=provenance_base.id, **kwargs )
        image2.mjd = rng.uniform(0, 1) + 60000
        image2.end_mjd = image2.mjd + 0.007
        _2 = ImageCleanup.save_image( image2 )
        image2.insert()

        image3 = Image(ra=120.0005, dec=10., provenance_id=provenance_base.id, **kwargs )
        image3.mjd = rng.uniform(0, 1) + 60000
        image3.end_mjd = image3.mjd + 0.007
        _3 = ImageCleanup.save_image( image3 )
        image3.insert()

        image4 = Image(ra=60., dec=0., provenance_id=provenance_base.id, **kwargs )
        image4.mjd = rng.uniform(0, 1) + 60000
        image4.end_mjd = image4.mjd + 0.007
        _4 = ImageCleanup.save_image( image4 )
        image4.insert()

        with SmartSession() as session:
            sought = session.query( Image ).filter( Image.cone_search(120., 10., rad=1.02) ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id, image2.id }.issubset( soughtids )
            assert len( { image3.id, image4.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.cone_search(120., 10., rad=2.) ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id, image2.id, image3.id }.issubset( soughtids )
            assert len( { image4.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.cone_search(120., 10., 0.017, radunit='arcmin') ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id, image2.id }.issubset( soughtids )
            assert len( { image3.id, image4.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.cone_search(120., 10., 0.0002833, radunit='degrees') ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id, image2.id }.issubset( soughtids )
            assert len( { image3.id, image4.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.cone_search(120., 10., 4.9451e-6, radunit='radians') ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id, image2.id }.issubset( soughtids )
            assert len( { image3.id, image4.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.cone_search(60, -10, 1.) ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert len( { image1.id, image2.id, image3.id, image4.id } & soughtids ) == 0

            with pytest.raises( ValueError, match='.*unknown radius unit' ):
                sought = Image.cone_search( 0., 0., 1., 'undefined_unit' )
    finally:
        for i in [ image1, image2, image3, image4 ]:
            if i is not None:
                i.delete_from_disk_and_database()


# Really, we should also do some speed tests, but that
# is outside the scope of the always-run tests.
def test_four_corners( provenance_base ):
    # TODO : also test prov_id in find_potential_overlapping

    rafar = 30.
    decfar = -20.
    # Include some RAs near 0 to test the seam
    ractrs = [ 120., 120., 0., 0.1, 359.9  ]
    decctrs = [ 40., 80., 20., 20., 20. ]
    dra = 0.2
    ddec = 0.2
    # deltas is in the standard FourCorners sorting order
    deltas = np.array( [ [ -dra/2., -ddec/2. ],
                         [ -dra/2.,  ddec/2. ],
                         [  dra/2., -ddec/2. ],
                         [  dra/2.,  ddec/2. ] ] )


    def makeimage( ra, dec, rot, offscale=1 ):
        if rot == 0:
            off = np.copy( deltas )
        else:
            rotmat = np.array( [ [  np.cos( rot * np.pi/180. ), np.sin( rot * np.pi/180. ) ],
                                 [ -np.sin( rot * np.pi/180. ), np.cos( rot * np.pi/180. ) ] ] )
            # There's probably a clever numpy broadcasting way to do this without a list comprehension
            off = np.array( [ np.matmul( rotmat, d ) for d in deltas ] )

        off *= offscale
        off[:,0] /= np.cos( dec * np.pi / 180. )

        ras = off[:,0] + ra
        decs = off[:,1] + dec
        minra = ras.min()
        maxra = ras.max()
        mindec = decs.min()
        maxdec = decs.max()
        ras[ ras < 0. ] += 360.
        ras[ ras >= 360. ] -= 360.
        minra = minra if minra >= 0 else minra + 360.
        minra = minra if minra < 360. else minra - 360.
        maxra = maxra if maxra >= 0 else maxra + 360.
        maxra = maxra if maxra < 360. else maxra - 360.
        img = Image( ra=ra, dec=dec,
                     ra_corner_00=ras[0], ra_corner_01=ras[1],
                     ra_corner_10=ras[2], ra_corner_11=ras[3],
                     dec_corner_00=decs[0], dec_corner_01=decs[1],
                     dec_corner_10=decs[2], dec_corner_11=decs[3],
                     minra=minra, maxra=maxra, mindec=mindec, maxdec=maxdec,
                     format='fits', exp_time=60.48, section_id='x',
                     project='x', target='x', instrument='DemoInstrument',
                     telescope='x', filter='r', provenance_id=provenance_base.id,
                     nofile=True )
        rng = np.random.default_rng()
        img.mjd = rng.uniform( 0, 1 ) + 60000
        img.end_mjd = img.mjd + 0.007
        return img

    for ra0, dec0 in zip( ractrs, decctrs ):
        image1 = None
        image2 = None
        image3 = None
        imagepoint = None
        imagefar = None
        try:
            # RA numbers are made ugly from cos(dec).
            # image1: centered on ra, dec; square to the sky
            image1 = makeimage( ra0, dec0, 0. )
            _1 = ImageCleanup.save_image( image1 )
            image1.insert()

            # image2: centered on ra, dec, at a 45° angle
            image2 = makeimage( ra0, dec0, 45. )
            _2 = ImageCleanup.save_image( image2 )
            image2.insert()

            # image3: centered offset by (0.025, 0.025) linear degrees from ra, dec, square on sky
            image3 = makeimage( ra0+0.025/np.cos(dec0*np.pi/180.), dec0+0.025, 0. )
            _3 = ImageCleanup.save_image( image3 )
            image3.insert()

            # imagepoint and imagefar are used to test Image.containing and Image.find_containing_siobj,
            # as Image is the only example of a SpatiallyIndexed thing we have so far.
            # imagepoint is in the lower left of image1, so should not be in image2 or image3
            decpoint = dec0 - 0.9 * ddec / 2.
            rapoint = ra0 - 0.9 * dra / 2. / np.cos( decpoint * np.pi / 180. )
            rapoint = rapoint if rapoint >= 0. else rapoint + 360.
            imagepoint = makeimage( rapoint, decpoint, 0., offscale=0.01 )
            _point = ImageCleanup.save_image( imagepoint )
            imagepoint.insert()

            imagefar = makeimage( rafar, decfar, 0. )
            _far = ImageCleanup.save_image( imagefar )
            imagefar.insert()

            with SmartSession() as session:
                # A quick safety check...
                with pytest.raises( TypeError, match=r"\(ra,dec\) must be floats" ):
                    sought = Image.find_containing( "Robert'); DROP TABLE students; --", 23. )
                with pytest.raises( TypeError, match=r"\(ra,dec\) must be floats" ):
                    sought = Image.find_containing( 42., "Robert'); DROP TABLE students; --" )

                sought = Image.find_containing( ra0, dec0, session=session )
                soughtids = set( [ s.id for s in sought ] )
                assert { image1.id, image2.id, image3.id }.issubset( soughtids )
                assert len( { imagepoint.id, imagefar.id } & soughtids ) == 0

                sought = Image.find_containing( ra0, dec0+0.6*ddec, session=session )
                soughtids = set( [ s.id for s in sought ] )
                assert { image2.id, image3.id }.issubset( soughtids )
                assert len( { image1.id, imagepoint.id, imagefar.id } & soughtids ) == 0

                sought = Image.find_containing( ra0, dec0-0.6*ddec, session=session )
                soughtids = set( [ s.id for s in sought ] )
                assert { image2.id }.issubset( soughtids )
                assert len( { image1.id, image3.id, imagepoint.id, imagefar.id } & soughtids ) == 0

                sought = Image.find_containing( imagepoint.ra, imagepoint.dec, session=session )
                soughtids = set( [ s.id for s in sought ] )
                assert { image1.id, imagepoint.id }.issubset( soughtids )
                assert len( { image2.id, image3.id, imagefar.id } & soughtids ) == 0

                sought = Image.find_containing_siobj( imagepoint, session=session )
                soughtids = set( [ s.id for s in sought ] )
                assert { image1.id, imagepoint.id }.issubset( soughtids )
                assert len( { image2.id, image3.id, imagefar.id } & soughtids ) == 0

                # This verifies that find_containing can handle integers in addition to floats
                sought = Image.find_containing( 0, 0, session=session )
                soughtids = set( [ s.id for s in sought ] )
                assert len( { image1.id, image2.id, image3.id, imagepoint.id, imagefar.id } & soughtids ) == 0

                sought = Image.find_containing( imagefar.ra, imagefar.dec, session=session )
                soughtids = set( [ s.id for s in sought ] )
                assert len( { image1.id, image2.id, image3.id, imagepoint.id } & soughtids ) == 0

                sought = Image.find_containing_siobj( imagefar, session=session )
                soughtids = set( [ s.id for s in sought ] )
                assert len( { image1.id, image2.id, image3.id, imagepoint.id } & soughtids ) == 0

                sought = session.query( Image ).filter( Image.within( image1 ) ).all()
                soughtids = set( [ s.id for s in sought ] )
                assert { image1.id, image2.id, image3.id, imagepoint.id }.issubset( soughtids )
                assert len( { imagefar.id } & soughtids ) == 0

                sought = session.query( Image ).filter( Image.within( imagefar ) ).all()
                soughtids = set( [ s.id for s in sought ] )
                assert len( { image1.id, image2.id, image3.id, imagepoint.id } & soughtids ) == 0

                sought = Image.find_potential_overlapping( imagepoint, session=session )
                soughtids = set( [ s.id for s in sought ] )
                assert { image1.id, image2.id, imagepoint.id }.issubset( soughtids )
                assert len( { image3.id, imagefar.id } & soughtids ) == 0

                sought = Image.find_potential_overlapping( imagefar, session=session )
                soughtids = set( [ s.id for s in sought ] )
                assert len( { image1.id, image2.id, image3.id, imagepoint.id } & soughtids ) == 0

        finally:
            for i in [ image1, image2, image3, imagepoint, imagefar ]:
                if ( i is not None ):
                    i.delete_from_disk_and_database()


    # Further overlap test
    with SmartSession() as session:
        image1 = None
        image2 = None
        image3 = None
        image4 = None
        image5 = None
        try:
            image1 = makeimage( 180., 0., 0. )
            _1 = ImageCleanup.save_image( image1 )
            # Make a couple of images offset more than half but less than the full image size
            image2 = makeimage( 180.18, 0.18, 0. )
            _2 = ImageCleanup.save_image( image2 )
            image3 = makeimage( 179.82, -0.18, 0. )
            _3 = ImageCleanup.save_image( image3 )
            # Also make a smaller image to test that that overlap works
            image4 = makeimage( 180., 0., 0., offscale=0.25 )
            _4 = ImageCleanup.save_image( image4 )
            # And make an image at small angle to test that the "no corners
            #  inside" case works
            image5 = makeimage( 180., 0., 10. )
            _5 = ImageCleanup.save_image( image5 )

            session.add( image1 )
            session.add( image2 )
            session.add( image3 )
            session.add( image4 )
            session.add( image5 )
            # These tests don't pass if I don't commit here.  However, the tests above
            # (in the for loop) did pass even though I didn't commit.  I don't understand
            # the difference.  (Mumble mumble typical sqlalchmey mysteriousness mumble mumble.)
            # In practical usage, we're going to be searching stuff that was committed to the
            # database before, so the equivalent of this next commit will have been run.
            session.commit()

            sought = Image.find_potential_overlapping( image1, session=session )
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id, image2.id, image3.id, image4.id, image5.id }.issubset( soughtids )

            sought = Image.find_potential_overlapping( image2, session=session )
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id, image2.id, image5.id }.issubset( soughtids )
            assert len( { image3.id, image4.id } & soughtids ) == 0

            sought = Image.find_potential_overlapping( image4, session=session )
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id, image4.id, image5.id }.issubset( soughtids )
            assert len( { image2.id, image3.id } & soughtids ) == 0

        finally:
            # When a test fails, this doesn't seem to actually
            #  clean up the database.  Is this pytest subverting
            #  the finally block?  Or is it some sqlqlchemy mysteriousness?
            for i in [ image1, image2, image3, image4, image5 ]:
                if ( i is not None ) and sa.inspect( i ).persistent:
                    session.delete( i )
            session.commit()



def im_qual(im, factor=3.0):
    """Helper function to get the "quality" of an image."""
    return im.lim_mag_estimate - factor * im.fwhm_estimate


def test_find_images(ptf_reference_image_datastores, ptf_ref,
                     decam_reference, decam_datastore, decam_default_calibrators):
    # TODO: need to fix some of these values (of lim_mag and quality) once we get actual limiting magnitude measurements
    # (...isn't that done now?  TODO: verify that the limiting magnitude estimates in the tests below come
    # from Dan's code, and if it is, remove these three lines of comments.)

    with SmartSession() as session:
        total_w_calibs = session.query( Image ).count()
        total = session.query( Image ).filter( Image._type.in_([1,2,3,4]) ).count()

    # try finding them all
    all_images_w_calibs = Image.find_images( type=None )
    assert len(all_images_w_calibs) == total_w_calibs

    all_images = Image.find_images()
    assert len(all_images) == total

    results = Image.find_images( order_by='earliest' )
    assert len(results) == total
    assert all( results[i].mjd <= results[i+1].mjd for i in range(len(results)-1) )

    results = Image.find_images( order_by='latest' )
    assert len(results) == total
    assert all( results[i].mjd >= results[i+1].mjd for i in range(len(results)-1) )

    # get only the science images
    found1 = Image.find_images(type=1)
    assert all(im._type == 1 for im in found1)
    assert all(im.type == 'Sci' for im in found1)
    assert len(found1) < total

    # get the coadd and subtraction images
    found2 = Image.find_images(type=[2, 3, 4])
    assert all(im._type in [2, 3, 4] for im in found2)
    assert all(im.type in ['ComSci', 'Diff', 'ComDiff'] for im in found2)
    assert len(found2) < total
    assert len(found1) + len(found2) == total

    # use the names of the types instead of integers, or a mixture of ints and strings
    found3 = Image.find_images(type=['ComSci', 'Diff', 4])
    assert [ f._id for f in found2 ] == [ f._id for f in found3 ]

    # filter by MJD and observation date
    value = 57000.0
    found1 = Image.find_images(min_mjd=value)
    assert all(im.mjd >= value for im in found1)
    assert all(im.instrument == 'DECam' for im in found1)
    assert len(found1) < total

    found2 = Image.find_images(max_mjd=value)
    assert all(im.mjd <= value for im in found2)
    assert all(im.instrument == 'PTF' for im in found2)
    assert len(found2) < total
    assert len(found1) + len(found2) == total

    found3 = Image.find_images(min_mjd=value, max_mjd=value)
    assert len(found3) == 0

    # filter by observation date
    t = Time(57000.0, format='mjd').datetime
    found4 = Image.find_images(min_dateobs=t)
    assert all(im.observation_time >= t for im in found4)
    assert all(im.instrument == 'DECam' for im in found4)
    assert set( f._id for f in found4 ) == set( f._id for f in found1 )
    assert len(found4) < total

    found5 = Image.find_images(max_dateobs=t)
    assert all(im.observation_time <= t for im in found5)
    assert all(im.instrument == 'PTF' for im in found5)
    assert set( f._id for f in found5 ) == set( f._id for f in found2 )
    assert len(found5) < total
    assert len(found4) + len(found5) == total

    # filter by images that contain this point (ELAIS-E1, chip S2)
    ra = 7.025
    dec = -42.923
    found1 = Image.find_containing( ra, dec )   # note: find_containing is a FourCorners method
    found1a = Image.find_images( ra=ra, dec=dec )
    assert set( i.id for i in found1 ) == set( i.id for i in found1a )
    assert all(im.instrument == 'DECam' for im in found1)
    assert all(im.target == 'ELAIS-E1' for im in found1)
    assert len(found1) < total

    # filter by images that contain this point (ELAIS-E1, chip N16)
    ra = 7.659
    dec = -43.420
    found2 = Image.find_images(ra=ra, dec=dec)
    assert all(im.instrument == 'DECam' for im in found2)
    assert all(im.target == 'ELAIS-E1' for im in found2)
    assert len(found2) < total

    # filter by images that contain this point (PTF field number 100014)
    ra = 188.0
    dec = 4.5
    found3 = Image.find_images(ra=ra, dec=dec )
    assert all(im.instrument == 'PTF' for im in found3)
    assert all(im.target == '100014' for im in found3)
    assert len(found3) < total
    assert len(found1) + len(found2) + len(found3) == total

    # find images that overlap
    ptfdses = ptf_reference_image_datastores
    found1 = Image.find_images( image=ptfdses[0].image )
    found1ids = set( f._id for f in found1 )
    assert len(found1) == 6
    assert set( d.image.id for d in ptfdses ).issubset( found1ids )
    assert ptf_ref.image.id in found1ids

    found2 = Image.find_images( minra=ptfdses[0].image.minra,
                                maxra=ptfdses[0].image.maxra,
                                mindec=ptfdses[0].image.mindec,
                                maxdec=ptfdses[0].image.maxdec )
    found2ids = set( f._id for f in found2 )
    assert found1ids == found2ids

    found3 = Image.find_images( image=ptfdses[0].image, overlapfrac=0.98 )
    found3ids = set( f._id for f in found3 )
    assert found3ids.issubset( found1ids )
    assert all( FourCorners.get_overlap_frac( ptfdses[0].image, f ) >= 0.98 for f in found3 )
    assert len(found3ids) == 2

    # filter by the PTF project name
    found1 = Image.find_images(project='PTF_DyC_survey')
    assert all(im.project == 'PTF_DyC_survey' for im in found1)
    assert all(im.instrument == 'PTF' for im in found1)
    assert len(found1) < total

    # filter by the two different project names for DECam:
    found2 = Image.find_images(project=['many', '2021B-0149'])
    assert all(im.project in ['many', '2021B-0149'] for im in found2)
    assert all(im.instrument == 'DECam' for im in found2)
    assert len(found2) < total
    assert len(found1) + len(found2) == total

    # filter by instrument
    found1 = Image.find_images(instrument='PTF')
    assert all(im.instrument == 'PTF' for im in found1)
    assert len(found1) < total

    found2 = Image.find_images(instrument='DECam')
    assert all(im.instrument == 'DECam' for im in found2)
    assert len(found2) < total
    assert len(found1) + len(found2) == total

    found3 = Image.find_images(instrument=['PTF', 'DECam'])
    assert len(found3) == total

    found4 = Image.find_images(instrument=['foobar'])
    assert len(found4) == 0

    # filter by filter
    found6 = Image.find_images(filter='R')
    assert all(im.filter == 'R' for im in found6)
    assert all(im.instrument == 'PTF' for im in found6)
    assert set( f.id for f in found6 ) == set( f.id for f in found1 )

    found7 = Image.find_images(filter='r')
    assert all(im.filter == 'r' for im in found7)
    assert all(im.instrument == 'DECam' for im in found7)
    assert set( f.id for f in found7 ) == set( f.id for f in found2 )

    # filter by seeing FWHM
    value = 3.0
    found1 = Image.find_images(max_seeing=value)
    assert all(im.instrument == 'DECam' for im in found1)
    assert all(im.fwhm_estimate <= value for im in found1)
    assert len(found1) < total

    found2 = Image.find_images(min_seeing=value)
    assert all(im.instrument == 'PTF' for im in found2)
    assert all(im.fwhm_estimate >= value for im in found2)
    assert len(found2) < total
    assert len(found1) + len(found2) == total

    found3 = Image.find_images(min_seeing=value, max_seeing=value)
    assert len(found3) == 0  # we will never have exactly that number

    # filter by limiting magnitude
    value = 21.0
    found1 = Image.find_images(min_lim_mag=value)
    assert all(im.instrument == 'DECam' for im in found1)
    assert all(im.lim_mag_estimate >= value for im in found1)
    assert len(found1) < total

    found2 = Image.find_images(max_lim_mag=value)
    assert all(im.instrument == 'PTF' for im in found2)
    assert all(im.lim_mag_estimate <= value for im in found2)
    assert len(found2) < total
    assert len(found1) + len(found2) == total

    found3 = Image.find_images(min_lim_mag=value, max_lim_mag=value)
    assert len(found3) == 0

    # filter by background
    value = 28.0
    found1 = Image.find_images(min_background=value)
    assert all(im.bkg_rms_estimate >= value for im in found1)
    assert len(found1) < total

    found2 = Image.find_images(max_background=value)
    assert all(im.bkg_rms_estimate <= value for im in found2)
    assert len(found2) < total
    assert len(found1) + len(found2) == total

    found3 = Image.find_images(min_background=value, max_background=value)
    assert len(found3) == 0

    # filter by zero point
    value = 28.0
    found1 = Image.find_images(min_zero_point=value)
    assert all(im.zero_point_estimate >= value for im in found1)
    assert len(found1) < total

    found2 = Image.find_images(max_zero_point=value)
    assert all(im.zero_point_estimate <= value for im in found2)
    assert len(found2) < total
    assert len(found1) + len(found2) == total

    found3 = Image.find_images(min_zero_point=value, max_zero_point=value)
    assert len(found3) == 0

    # filter by exposure time
    value = 60.0 + 1.0
    found1 = Image.find_images(min_exp_time=value)
    assert all(im.exp_time >= value for im in found1)
    assert len(found1) < total

    found2 = Image.find_images(max_exp_time=value)
    assert all(im.exp_time <= value for im in found2)
    assert len(found2) < total

    found3 = Image.find_images(min_exp_time=60.0, max_exp_time=60.0)
    assert len(found3) == len(found2)  # all those under 31s are those with exactly 30s

    # query based on airmass
    value = 1.15
    total_with_airmass = len([im for im in all_images if im.airmass is not None])
    found1 = Image.find_images(max_airmass=value)
    assert all(im.airmass <= value for im in found1)
    assert len(found1) < total_with_airmass

    found2 = Image.find_images(min_airmass=value)
    assert all(im.airmass >= value for im in found2)
    assert len(found2) < total_with_airmass
    assert len(found1) + len(found2) == total_with_airmass

    # order the found by quality (lim_mag - 3 * fwhm)
    # note that we cannot filter by quality, it is not a meaningful number
    # on its own, only as a way to compare images and find which is better.
    # sort all the images by quality and get the best one
    found = Image.find_images(order_by='quality')
    best = found[0]

    # the best overall quality from all images
    assert im_qual(best) == max([im_qual(im) for im in found])

    # get the two best images from the PTF instrument (exp_time chooses the single images only)
    found1 = Image.find_images(max_exp_time=60, order_by='quality')[:2]
    assert len(found1) == 2
    assert all(im_qual(im) > 9.0 for im in found1)

    # change the seeing factor a little:
    factor = 2.8
    found2 = Image.find_images(max_exp_time=60, order_by='quality', seeing_quality_factor=factor)[:2]
    assert [ i.id for i in found2 ] == [ i.id for i in found1 ]

    # quality will be a little bit higher, but the images are the same
    assert [ f._id for f in found2 ] == [ f._id for f in found1 ]
    assert im_qual(found2[0], factor=factor) > im_qual(found1[0])
    assert im_qual(found2[1], factor=factor) > im_qual(found1[1])

    # change the seeing factor dramatically:
    factor = 0.2
    found3 = Image.find_images(max_exp_time=60, order_by='quality', seeing_quality_factor=factor)[:2]
    assert [ i.id for i in found3 ] == [ i.id for i in found1 ]

    # TODO -- assumptions that went into this test aren't right, come up with
    #   a test case where it will actually work
    # quality will be a higher, but also a different image will now have the second-best quality
    # assert [ f._id for f in found3 ] != [ f._id for f in found1 ]
    # assert im_qual(found3[0], factor=factor) > im_qual(found1[0])

    # do a cross filtering of coordinates and background (should only find the PTF coadd)
    ra = 188.0
    dec = 4.5
    background = 5.

    found1 = Image.find_images(ra=ra, dec=dec, max_background=background)
    assert len(found1) == 1
    assert found1[0].instrument == 'PTF'
    assert found1[0].type == 'ComSci'

    # cross the DECam target and section ID with the exposure time that's of the S2 ref image
    target = 'ELAIS-E1'
    section_id = 'S2'
    exp_time = 120.0

    found2 = Image.find_images(target=target, section_id=section_id, min_exp_time=exp_time)
    assert len(found2) == 1
    assert found2[0].instrument == 'DECam'
    assert found2[0].type == 'ComSci'
    assert found2[0].exp_time == 150.0

    # cross filter on MJD and instrument in a way that has no found
    mjd = 55000.0
    instrument = 'PTF'

    found3 = Image.find_images(min_mjd=mjd, instrument=instrument)
    assert len(found3) == 0

    # cross filter MJD and sort by quality to get the coadd PTF image
    mjd = 54926.31913

    found4 = Image.find_images(max_mjd=mjd, order_by='quality')
    assert len(found4) == 2
    assert found4[0].mjd == found4[1].mjd  # same time, as one is a coadd of the other images
    assert found4[0].instrument == 'PTF'
    # TODO : these next two tests don't work right; see Issue #343
    # assert found4[0].type == 'ComSci'  # the first one out is the high quality coadd
    # assert found4[1].type == 'Sci'  # the second one is the regular image

    # check that the DECam difference and new image it is based on have the same limiting magnitude and quality
    # (...this check probably really belongs in a test of subtractions!)
    diff = Image.find_images(instrument='DECam', type=3)
    assert len(diff) == 1
    diff = diff[0]
    new =  Image.find_images(instrument='DECam', type=1, min_mjd=diff.mjd, max_mjd=diff.mjd)
    assert len(new) == 1
    new = new[0]
    assert new.id != diff.id
    assert diff.lim_mag_estimate == new.lim_mag_estimate
    assert diff.fwhm_estimate == new.fwhm_estimate
    assert im_qual(diff) == im_qual(new)
