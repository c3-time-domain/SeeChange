import pytest

import numpy as np
import sqlalchemy as sa

from astropy.time import Time

from models.base import SmartSession
from models.image import Image, image_upstreams_association_table

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
    with SmartSession() as session:
        image1 = None
        image2 = None
        image3 = None
        image4 = None
        try:
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
            image1 = Image(ra=120., dec=10., provenance=provenance_base, **kwargs )
            image1.mjd = np.random.uniform(0, 1) + 60000
            image1.end_mjd = image1.mjd + 0.007
            clean1 = ImageCleanup.save_image( image1 )

            image2 = Image(ra=120.0002, dec=9.9998, provenance=provenance_base, **kwargs )
            image2.mjd = np.random.uniform(0, 1) + 60000
            image2.end_mjd = image2.mjd + 0.007
            clean2 = ImageCleanup.save_image( image2 )

            image3 = Image(ra=120.0005, dec=10., provenance=provenance_base, **kwargs )
            image3.mjd = np.random.uniform(0, 1) + 60000
            image3.end_mjd = image3.mjd + 0.007
            clean3 = ImageCleanup.save_image( image3 )

            image4 = Image(ra=60., dec=0., provenance=provenance_base, **kwargs )
            image4.mjd = np.random.uniform(0, 1) + 60000
            image4.end_mjd = image4.mjd + 0.007
            clean4 = ImageCleanup.save_image( image4 )

            session.add( image1 )
            session.add( image2 )
            session.add( image3 )
            session.add( image4 )

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
                if ( i is not None ) and sa.inspect( i ).persistent:
                    session.delete( i )
            session.commit()


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
                     telescope='x', filter='r', provenance=provenance_base,
                     nofile=True )
        img.mjd = np.random.uniform( 0, 1 ) + 60000
        img.end_mjd = img.mjd + 0.007
        return img

    for ra0, dec0 in zip( ractrs, decctrs ):
        with SmartSession() as session:
            image1 = None
            image2 = None
            image3 = None
            imagepoint = None
            imagefar = None
            try:
                # RA numbers are made ugly from cos(dec).
                # image1: centered on ra, dec; square to the sky
                image1 = makeimage( ra0, dec0, 0. )
                clean1 = ImageCleanup.save_image( image1 )

                # image2: centered on ra, dec, at a 45° angle
                image2 = makeimage( ra0, dec0, 45. )
                clean2 = ImageCleanup.save_image( image2 )

                # image3: centered offset by (0.025, 0.025) linear degrees from ra, dec, square on sky
                image3 = makeimage( ra0+0.025/np.cos(dec0*np.pi/180.), dec0+0.025, 0. )
                clean3 = ImageCleanup.save_image( image3 )

                # imagepoint and imagefar are used to test Image.containing and Image.find_containing_siobj,
                # as Image is the only example of a SpatiallyIndexed thing we have so far.
                # imagepoint is in the lower left of image1, so should not be in image2 or image3
                decpoint = dec0 - 0.9 * ddec / 2.
                rapoint = ra0 - 0.9 * dra / 2. / np.cos( decpoint * np.pi / 180. )
                rapoint = rapoint if rapoint >= 0. else rapoint + 360.
                imagepoint = makeimage( rapoint, decpoint, 0., offscale=0.01 )
                clearpoint = ImageCleanup.save_image( imagepoint )

                imagefar = makeimage( rafar, decfar, 0. )
                clearfar = ImageCleanup.save_image( imagefar )

                session.add( image1 )
                session.add( image2 )
                session.add( image3 )
                session.add( imagepoint )
                session.add( imagefar )

                sought = session.query( Image ).filter( Image.containing( ra0, dec0 ) ).all()
                soughtids = set( [ s.id for s in sought ] )
                assert { image1.id, image2.id, image3.id }.issubset( soughtids )
                assert len( { imagepoint.id, imagefar.id } & soughtids ) == 0

                sought = session.query( Image ).filter( Image.containing( rapoint, decpoint ) ).all()
                soughtids = set( [ s.id for s in sought ] )
                assert { image1.id, imagepoint.id }.issubset( soughtids  )
                assert len( { image2.id, image3.id, imagefar.id } & soughtids ) == 0

                sought = session.query( Image ).filter( Image.containing( ra0, dec0+0.6*ddec ) ).all()
                soughtids = set( [ s.id for s in sought ] )
                assert { image2.id, image3.id }.issubset( soughtids )
                assert len( { image1.id, imagepoint.id, imagefar.id } & soughtids ) == 0

                sought = session.query( Image ).filter( Image.containing( ra0, dec0-0.6*ddec ) ).all()
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

                sought = session.query( Image ).filter( Image.containing( 0, 0 ) ).all()
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
                    if ( i is not None ) and sa.inspect( i ).persistent:
                        session.delete( i )
                session.commit()


    # Further overlap test
    with SmartSession() as session:
        image1 = None
        image2 = None
        image3 = None
        image4 = None
        image5 = None
        try:
            image1 = makeimage( 180., 0., 0. )
            clean1 = ImageCleanup.save_image( image1 )
            # Make a couple of images offset more than half but less than the full image size
            image2 = makeimage( 180.18, 0.18, 0. )
            clean2 = ImageCleanup.save_image( image2 )
            image3 = makeimage( 179.82, -0.18, 0. )
            clean3 = ImageCleanup.save_image( image3 )
            # Also make a smaller image to test that that overlap works
            image4 = makeimage( 180., 0., 0., offscale=0.25 )
            clean4 = ImageCleanup.save_image( image4 )
            # And make an image at small angle to test that the "no corners
            #  inside" case works
            image5 = makeimage( 180., 0., 10. )
            clean5 = ImageCleanup.save_image( image5 )

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


def test_image_query(ptf_ref, decam_reference, decam_datastore, decam_default_calibrators):
    # TODO: need to fix some of these values (of lim_mag and quality) once we get actual limiting magnitude measurements

    with SmartSession() as session:
        stmt = Image.query_images()
        results = session.scalars(stmt).all()
        total = len(results)

        # from pprint import pprint
        # pprint(results)
        #
        # print(f'MJD: {[im.mjd for im in results]}')
        # print(f'date: {[im.observation_time for im in results]}')
        # print(f'RA: {[im.ra for im in results]}')
        # print(f'DEC: {[im.dec for im in results]}')
        # print(f'target: {[im.target for im in results]}')
        # print(f'section_id: {[im.section_id for im in results]}')
        # print(f'project: {[im.project for im in results]}')
        # print(f'Instrument: {[im.instrument for im in results]}')
        # print(f'Filter: {[im.filter for im in results]}')
        # print(f'FWHM: {[im.fwhm_estimate for im in results]}')
        # print(f'LIMMAG: {[im.lim_mag_estimate for im in results]}')
        # print(f'B/G: {[im.bkg_rms_estimate for im in results]}')
        # print(f'ZP: {[im.zero_point_estimate for im in results]}')
        # print(f'EXPTIME: {[im.exp_time for im in results]}')
        # print(f'AIRMASS: {[im.airmass for im in results]}')
        # print(f'QUAL: {[im_qual(im) for im in results]}')

        # get only the science images
        stmt = Image.query_images(type=1)
        results1 = session.scalars(stmt).all()
        assert all(im._type == 1 for im in results1)
        assert all(im.type == 'Sci' for im in results1)
        assert len(results1) < total

        # get the coadd and subtraction images
        stmt = Image.query_images(type=[2, 3, 4])
        results2 = session.scalars(stmt).all()
        assert all(im._type in [2, 3, 4] for im in results2)
        assert all(im.type in ['ComSci', 'Diff', 'ComDiff'] for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        # use the names of the types instead of integers, or a mixture of ints and strings
        stmt = Image.query_images(type=['ComSci', 'Diff', 4])
        results3 = session.scalars(stmt).all()
        assert results2 == results3

        # filter by MJD and observation date
        value = 57000.0
        stmt = Image.query_images(min_mjd=value)
        results1 = session.scalars(stmt).all()
        assert all(im.mjd >= value for im in results1)
        assert all(im.instrument == 'DECam' for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(max_mjd=value)
        results2 = session.scalars(stmt).all()
        assert all(im.mjd <= value for im in results2)
        assert all(im.instrument == 'PTF' for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        stmt = Image.query_images(min_mjd=value, max_mjd=value)
        results3 = session.scalars(stmt).all()
        assert len(results3) == 0

        # filter by observation date
        t = Time(57000.0, format='mjd').datetime
        stmt = Image.query_images(min_dateobs=t)
        results4 = session.scalars(stmt).all()
        assert all(im.observation_time >= t for im in results4)
        assert all(im.instrument == 'DECam' for im in results4)
        assert set(results4) == set(results1)
        assert len(results4) < total

        stmt = Image.query_images(max_dateobs=t)
        results5 = session.scalars(stmt).all()
        assert all(im.observation_time <= t for im in results5)
        assert all(im.instrument == 'PTF' for im in results5)
        assert set(results5) == set(results2)
        assert len(results5) < total
        assert len(results4) + len(results5) == total

        # filter by images that contain this point (ELAIS-E1, chip S3)
        ra = 7.449
        dec = -42.926

        stmt = Image.query_images(ra=ra, dec=dec)
        results1 = session.scalars(stmt).all()
        assert all(im.instrument == 'DECam' for im in results1)
        assert all(im.target == 'ELAIS-E1' for im in results1)
        assert len(results1) < total

        # filter by images that contain this point (ELAIS-E1, chip N16)
        ra = 7.659
        dec = -43.420

        stmt = Image.query_images(ra=ra, dec=dec)
        results2 = session.scalars(stmt).all()
        assert all(im.instrument == 'DECam' for im in results2)
        assert all(im.target == 'ELAIS-E1' for im in results2)
        assert len(results2) < total

        # filter by images that contain this point (PTF field number 100014)
        ra = 188.0
        dec = 4.5
        stmt = Image.query_images(ra=ra, dec=dec)
        results3 = session.scalars(stmt).all()
        assert all(im.instrument == 'PTF' for im in results3)
        assert all(im.target == '100014' for im in results3)
        assert len(results3) < total
        assert len(results1) + len(results2) + len(results3) == total

        # filter by section ID
        stmt = Image.query_images(section_id='S3')
        results1 = session.scalars(stmt).all()
        assert all(im.section_id == 'S3' for im in results1)
        assert all(im.instrument == 'DECam' for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(section_id='N16')
        results2 = session.scalars(stmt).all()
        assert all(im.section_id == 'N16' for im in results2)
        assert all(im.instrument == 'DECam' for im in results2)
        assert len(results2) < total

        stmt = Image.query_images(section_id='11')
        results3 = session.scalars(stmt).all()
        assert all(im.section_id == '11' for im in results3)
        assert all(im.instrument == 'PTF' for im in results3)
        assert len(results3) < total
        assert len(results1) + len(results2) + len(results3) == total

        # filter by the PTF project name
        stmt = Image.query_images(project='PTF_DyC_survey')
        results1 = session.scalars(stmt).all()
        assert all(im.project == 'PTF_DyC_survey' for im in results1)
        assert all(im.instrument == 'PTF' for im in results1)
        assert len(results1) < total

        # filter by the two different project names for DECam:
        stmt = Image.query_images(project=['many', '2023A-716082'])
        results2 = session.scalars(stmt).all()
        assert all(im.project in ['many', '2023A-716082'] for im in results2)
        assert all(im.instrument == 'DECam' for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        # filter by instrument
        stmt = Image.query_images(instrument='PTF')
        results1 = session.scalars(stmt).all()
        assert all(im.instrument == 'PTF' for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(instrument='DECam')
        results2 = session.scalars(stmt).all()
        assert all(im.instrument == 'DECam' for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        stmt = Image.query_images(instrument=['PTF', 'DECam'])
        results3 = session.scalars(stmt).all()
        assert len(results3) == total

        stmt = Image.query_images(instrument=['foobar'])
        results4 = session.scalars(stmt).all()
        assert len(results4) == 0

        # filter by filter
        stmt = Image.query_images(filter='R')
        results6 = session.scalars(stmt).all()
        assert all(im.filter == 'R' for im in results6)
        assert all(im.instrument == 'PTF' for im in results6)
        assert set(results6) == set(results1)

        stmt = Image.query_images(filter='r DECam SDSS c0002 6415.0 1480.0')
        results7 = session.scalars(stmt).all()
        assert all(im.filter == 'r DECam SDSS c0002 6415.0 1480.0' for im in results7)
        assert all(im.instrument == 'DECam' for im in results7)
        assert set(results7) == set(results2)

        # filter by seeing FWHM
        value = 4.0
        stmt = Image.query_images(min_seeing=value)
        results1 = session.scalars(stmt).all()
        assert all(im.instrument == 'DECam' for im in results1)
        assert all(im.fwhm_estimate >= value for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(max_seeing=value)
        results2 = session.scalars(stmt).all()
        assert all(im.instrument == 'PTF' for im in results2)
        assert all(im.fwhm_estimate <= value for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        stmt = Image.query_images(min_seeing=value, max_seeing=value)
        results3 = session.scalars(stmt).all()
        assert len(results3) == 0  # we will never have exactly that number

        # filter by limiting magnitude
        value = 24.0
        stmt = Image.query_images(min_lim_mag=value)
        results1 = session.scalars(stmt).all()
        assert all(im.instrument == 'DECam' for im in results1)
        assert all(im.lim_mag_estimate >= value for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(max_lim_mag=value)
        results2 = session.scalars(stmt).all()
        assert all(im.instrument == 'PTF' for im in results2)
        assert all(im.lim_mag_estimate <= value for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        stmt = Image.query_images(min_lim_mag=value, max_lim_mag=value)
        results3 = session.scalars(stmt).all()
        assert len(results3) == 0

        # filter by background
        value = 25.0
        stmt = Image.query_images(min_background=value)
        results1 = session.scalars(stmt).all()
        assert all(im.bkg_rms_estimate >= value for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(max_background=value)
        results2 = session.scalars(stmt).all()
        assert all(im.bkg_rms_estimate <= value for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        stmt = Image.query_images(min_background=value, max_background=value)
        results3 = session.scalars(stmt).all()
        assert len(results3) == 0

        # filter by zero point
        value = 27.0
        stmt = Image.query_images(min_zero_point=value)
        results1 = session.scalars(stmt).all()
        assert all(im.zero_point_estimate >= value for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(max_zero_point=value)
        results2 = session.scalars(stmt).all()
        assert all(im.zero_point_estimate <= value for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        stmt = Image.query_images(min_zero_point=value, max_zero_point=value)
        results3 = session.scalars(stmt).all()
        assert len(results3) == 0

        # filter by exposure time
        value = 60.0 + 1.0
        stmt = Image.query_images(min_exp_time=value)
        results1 = session.scalars(stmt).all()
        assert all(im.exp_time >= value for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(max_exp_time=value)
        results2 = session.scalars(stmt).all()
        assert all(im.exp_time <= value for im in results2)
        assert len(results2) < total

        stmt = Image.query_images(min_exp_time=60.0, max_exp_time=60.0)
        results3 = session.scalars(stmt).all()
        assert len(results3) == len(results2)  # all those under 31s are those with exactly 30s

        # query based on airmass
        value = 1.15
        total_with_airmass = len([im for im in results if im.airmass is not None])
        stmt = Image.query_images(max_airmass=value)
        results1 = session.scalars(stmt).all()
        assert all(im.airmass <= value for im in results1)
        assert len(results1) < total_with_airmass

        stmt = Image.query_images(min_airmass=value)
        results2 = session.scalars(stmt).all()
        assert all(im.airmass >= value for im in results2)
        assert len(results2) < total_with_airmass
        assert len(results1) + len(results2) == total_with_airmass

        # order the results by quality (lim_mag - 3 * fwhm)
        # note that we cannot filter by quality, it is not a meaningful number
        # on its own, only as a way to compare images and find which is better.
        # sort all the images by quality and get the best one
        stmt = Image.query_images(order_by='quality')
        best = session.scalars(stmt).first()

        # the best overall quality from all images
        assert im_qual(best) == max([im_qual(im) for im in results])

        # get the two best images from the PTF instrument (exp_time chooses the single images only)
        stmt = Image.query_images(max_exp_time=60, order_by='quality')
        results1 = session.scalars(stmt.limit(2)).all()
        assert len(results1) == 2
        assert all(im_qual(im) > 10.0 for im in results1)

        # change the seeing factor a little:
        factor = 2.8
        stmt = Image.query_images(max_exp_time=60, order_by='quality', seeing_quality_factor=factor)
        results2 = session.scalars(stmt.limit(2)).all()

        # quality will be a little bit higher, but the images are the same
        assert results2 == results1
        assert im_qual(results2[0], factor=factor) > im_qual(results1[0])
        assert im_qual(results2[1], factor=factor) > im_qual(results1[1])

        # change the seeing factor dramatically:
        factor = 0.2
        stmt = Image.query_images(max_exp_time=60, order_by='quality', seeing_quality_factor=factor)
        results3 = session.scalars(stmt.limit(2)).all()

        # quality will be a higher, but also a different image will now have the second-best quality
        assert results3 != results1
        assert im_qual(results3[0], factor=factor) > im_qual(results1[0])

        # do a cross filtering of coordinates and background (should only find the PTF coadd)
        ra = 188.0
        dec = 4.5
        background = 5

        stmt = Image.query_images(ra=ra, dec=dec, max_background=background)
        results1 = session.scalars(stmt).all()
        assert len(results1) == 1
        assert results1[0].instrument == 'PTF'
        assert results1[0].type == 'ComSci'

        # cross the DECam target and section ID with the exposure time that's of the S3 ref image
        target = 'ELAIS-E1'
        section_id = 'S3'
        exp_time = 120.0

        stmt = Image.query_images(target=target, section_id=section_id, min_exp_time=exp_time)
        results2 = session.scalars(stmt).all()
        assert len(results2) == 1
        assert results2[0].instrument == 'DECam'
        assert results2[0].type == 'ComSci'
        assert results2[0].exp_time == 150.0

        # cross filter on MJD and instrument in a way that has no results
        mjd = 55000.0
        instrument = 'PTF'

        stmt = Image.query_images(min_mjd=mjd, instrument=instrument)
        results3 = session.scalars(stmt).all()
        assert len(results3) == 0

        # cross filter MJD and sort by quality to get the coadd PTF image
        mjd = 54926.31913

        stmt = Image.query_images(max_mjd=mjd, order_by='quality')
        results4 = session.scalars(stmt).all()
        assert len(results4) == 2
        assert results4[0].mjd == results4[1].mjd  # same time, as one is a coadd of the other images
        assert results4[0].instrument == 'PTF'
        assert results4[0].type == 'ComSci'  # the first one out is the high quality coadd
        assert results4[1].type == 'Sci'  # the second one is the regular image

        # check that the DECam difference and new image it is based on have the same limiting magnitude and quality
        stmt = Image.query_images(instrument='DECam', type=3)
        diff = session.scalars(stmt).first()
        stmt = Image.query_images(instrument='DECam', type=1, min_mjd=diff.mjd, max_mjd=diff.mjd)
        new = session.scalars(stmt).first()
        assert diff.lim_mag_estimate == new.lim_mag_estimate
        assert diff.fwhm_estimate == new.fwhm_estimate
        assert im_qual(diff) == im_qual(new)


def test_image_get_downstream(ptf_ref, ptf_supernova_images, ptf_subtraction1):
    with SmartSession() as session:
        # how many image to image associations are on the DB right now?
        num_associations = session.execute(
            sa.select(sa.func.count()).select_from(image_upstreams_association_table)
        ).scalar()

        assert num_associations > len(ptf_ref.image.upstream_images)

        prov = ptf_ref.image.provenance
        assert prov.process == 'coaddition'
        images = ptf_ref.image.upstream_images
        assert len(images) > 1

        loaded_image = Image.get_image_from_upstreams(images, prov.id)

        assert loaded_image.id == ptf_ref.image.id
        assert loaded_image.id != ptf_subtraction1.id
        assert loaded_image.id != ptf_subtraction1.new_image.id

    new_image = None
    new_image2 = None
    new_image3 = None
    try:
        # make a new image with a new provenance
        new_image = Image.copy_image(ptf_ref.image)
        prov = ptf_ref.provenance
        prov.process = 'copy'
        new_image.provenance = prov
        new_image.upstream_images = ptf_ref.image.upstream_images
        new_image.save()

        with SmartSession() as session:
            new_image = session.merge(new_image)
            session.commit()
            assert new_image.id is not None
            assert new_image.id != ptf_ref.image.id

            loaded_image = Image.get_image_from_upstreams(images, prov.id)
            assert loaded_image.id == new_image.id

        # use the original provenance but take down an image from the upstreams
        prov = ptf_ref.image.provenance
        images = ptf_ref.image.upstream_images[1:]

        new_image2 = Image.copy_image(ptf_ref.image)
        new_image2.provenance = prov
        new_image2.upstream_images = images
        new_image2.save()

        with SmartSession() as session:
            new_image2 = session.merge(new_image2)
            session.commit()
            assert new_image2.id is not None
            assert new_image2.id != ptf_ref.image.id
            assert new_image2.id != new_image.id

            loaded_image = Image.get_image_from_upstreams(images, prov.id)
            assert loaded_image.id != ptf_ref.image.id
            assert loaded_image.id != new_image.id

        # use the original provenance but add images to the upstreams
        prov = ptf_ref.image.provenance
        images = ptf_ref.image.upstream_images + ptf_supernova_images

        new_image3 = Image.copy_image(ptf_ref.image)
        new_image3.provenance = prov
        new_image3.upstream_images = images
        new_image3.save()

        with SmartSession() as session:
            new_image3 = session.merge(new_image3)
            session.commit()
            assert new_image3.id is not None
            assert new_image3.id != ptf_ref.image.id
            assert new_image3.id != new_image.id
            assert new_image3.id != new_image2.id

            loaded_image = Image.get_image_from_upstreams(images, prov.id)
            assert loaded_image.id == new_image3.id

    finally:
        if new_image is not None:
            new_image.delete_from_disk_and_database()
        if new_image2 is not None:
            new_image2.delete_from_disk_and_database()
        if new_image3 is not None:
            new_image3.delete_from_disk_and_database()

