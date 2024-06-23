import pytest

import numpy as np
import sqlalchemy as sa

from astropy.time import Time

from models.base import SmartSession
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
                       'dec_corner_00': 0,
                       'dec_corner_01': 0,
                       'dec_corner_10': 0,
                       'dec_corner_11': 0,
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
                      }
            # RA numbers are made ugly from cos(dec).
            # image1: centered on 120, 40, square to the sky
            image1 = Image( ra=120, dec=40.,
                            ra_corner_00=119.86945927, ra_corner_01=119.86945927,
                            ra_corner_10=120.13054073, ra_corner_11=120.13054073,
                            dec_corner_00=39.9, dec_corner_01=40.1, dec_corner_10=39.9, dec_corner_11=40.1,
                            provenance=provenance_base, nofile=True, **kwargs )
            image1.mjd = np.random.uniform(0, 1) + 60000
            image1.end_mjd = image1.mjd + 0.007
            clean1 = ImageCleanup.save_image( image1 )

            # image2: centered on 120, 40, at a 45° angle
            image2 = Image( ra=120, dec=40.,
                            ra_corner_00=119.81538753, ra_corner_01=120, ra_corner_11=120.18461247, ra_corner_10=120,
                            dec_corner_00=40, dec_corner_01=40.14142136, dec_corner_11=40, dec_corner_10=39.85857864,
                            provenance=provenance_base, nofile=True, **kwargs )
            image2.mjd = np.random.uniform(0, 1) + 60000
            image2.end_mjd = image2.mjd + 0.007
            clean2 = ImageCleanup.save_image( image2 )

            # image3: centered offset by (0.025, 0.025) linear arcsec from 120, 40, square on sky
            image3 = Image( ra=120.03264714, dec=40.025,
                            ra_corner_00=119.90210641, ra_corner_01=119.90210641,
                            ra_corner_10=120.16318787, ra_corner_11=120.16318787,
                            dec_corner_00=39.975, dec_corner_01=40.125, dec_corner_10=39.975, dec_corner_11=40.125,
                            provenance=provenance_base, nofile=True, **kwargs )
            image3.mjd = np.random.uniform(0, 1) + 60000
            image3.end_mjd = image3.mjd + 0.007
            clean3 = ImageCleanup.save_image( image3 )

            # imagepoint and imagefar are used to test Image.containing and Image.find_containing,
            # as Image is the only example of a SpatiallyIndexed thing we have so far.
            # The corners don't matter for these given how they'll be used.
            imagepoint = Image( ra=119.88, dec=39.95,
                                ra_corner_00=-.001, ra_corner_01=0.001, ra_corner_10=-0.001,
                                ra_corner_11=0.001, dec_corner_00=0, dec_corner_01=0, dec_corner_10=0, dec_corner_11=0,
                                provenance=provenance_base, nofile=True, **kwargs )
            imagepoint.mjd = np.random.uniform(0, 1) + 60000
            imagepoint.end_mjd = imagepoint.mjd + 0.007
            clearpoint = ImageCleanup.save_image( imagepoint )

            imagefar = Image( ra=30, dec=-10,
                              ra_corner_00=0, ra_corner_01=0, ra_corner_10=0,
                              ra_corner_11=0, dec_corner_00=0, dec_corner_01=0, dec_corner_10=0, dec_corner_11=0,
                              provenance=provenance_base, nofile=True, **kwargs )
            imagefar.mjd = np.random.uniform(0, 1) + 60000
            imagefar.end_mjd = imagefar.mjd + 0.007
            clearfar = ImageCleanup.save_image( imagefar )

            session.add( image1 )
            session.add( image2 )
            session.add( image3 )
            session.add( imagepoint )
            session.add( imagefar )

            sought = session.query( Image ).filter( Image.containing( 120, 40 ) ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id, image2.id, image3.id }.issubset( soughtids )
            assert len( { imagepoint.id, imagefar.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.containing( 119.88, 39.95 ) ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id }.issubset( soughtids  )
            assert len( { image2.id, image3.id, imagepoint.id, imagefar.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.containing( 120, 40.12 ) ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert { image2.id, image3.id }.issubset( soughtids )
            assert len( { image1.id, imagepoint.id, imagefar.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.containing( 120, 39.88 ) ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert { image2.id }.issubset( soughtids )
            assert len( { image1.id, image3.id, imagepoint.id, imagefar.id } & soughtids ) == 0

            sought = Image.find_containing( imagepoint, session=session )
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id }.issubset( soughtids )
            assert len( { image2.id, image3.id, imagepoint.id, imagefar.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.containing( 0, 0 ) ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert len( { image1.id, image2.id, image3.id, imagepoint.id, imagefar.id } & soughtids ) == 0

            sought = Image.find_containing( imagefar, session=session )
            soughtids = set( [ s.id for s in sought ] )
            assert len( { image1.id, image2.id, image3.id, imagepoint.id, imagefar.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.within( image1 ) ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert { image1.id, image2.id, image3.id, imagepoint.id }.issubset( soughtids )
            assert len( { imagefar.id } & soughtids ) == 0

            sought = session.query( Image ).filter( Image.within( imagefar ) ).all()
            soughtids = set( [ s.id for s in sought ] )
            assert len( { image1.id, image2.id, image3.id, imagepoint.id, imagefar.id } & soughtids ) == 0

        finally:
            session.rollback()


def im_qual(im, factor=3.0):
    """Helper function to get the "quality" of an image."""
    return im.lim_mag_estimate - factor * im.fwhm_estimate


def test_image_query(ptf_ref, decam_reference, decam_datastore, decam_default_calibrators):
    # TODO: need to fix some of these values (of lim_mag and quality) once we get actual limiting magnitude measurements

    with SmartSession() as session:
        stmt = Image.query_images()
        results = session.scalars(stmt).all()
        total = len(results)

        from pprint import pprint
        pprint(results)

        print(f'MJD: {[im.mjd for im in results]}')
        print(f'date: {[im.observation_time for im in results]}')
        print(f'RA: {[im.ra for im in results]}')
        print(f'DEC: {[im.dec for im in results]}')
        print(f'target: {[im.target for im in results]}')
        print(f'section_id: {[im.section_id for im in results]}')
        print(f'project: {[im.project for im in results]}')
        print(f'Instrument: {[im.instrument for im in results]}')
        print(f'Filter: {[im.filter for im in results]}')
        print(f'FWHM: {[im.fwhm_estimate for im in results]}')
        print(f'LIMMAG: {[im.lim_mag_estimate for im in results]}')
        print(f'B/G: {[im.bkg_rms_estimate for im in results]}')
        print(f'ZP: {[im.zero_point_estimate for im in results]}')
        print(f'EXPTIME: {[im.exp_time for im in results]}')
        print(f'QUAL: {[im_qual(im) for im in results]}')

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
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        # filter by MJD and observation date
        value = 55000.0
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
        t = Time(55000.0, format='mjd').datetime
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

        # filter by images that contain this point (DECaPS-West)
        ra = 115.28
        dec = -26.33

        stmt = Image.query_images(ra=ra, dec=dec)
        results1 = session.scalars(stmt).all()
        assert all(im.instrument == 'DECam' for im in results1)
        assert all(im.target == 'DECaPS-West' for im in results1)
        assert len(results1) < total

        # filter by images that contain this point (PTF field number 100014)
        ra = 188.0
        dec = 4.5
        stmt = Image.query_images(ra=ra, dec=dec)
        results2 = session.scalars(stmt).all()
        assert all(im.instrument == 'PTF' for im in results2)
        assert all(im.target == '100014' for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        # filter by section ID
        stmt = Image.query_images(section_id='N1')
        results1 = session.scalars(stmt).all()
        assert all(im.section_id == 'N1' for im in results1)
        assert all(im.instrument == 'DECam' for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(section_id='11')
        results2 = session.scalars(stmt).all()
        assert all(im.section_id == '11' for im in results2)
        assert all(im.instrument == 'PTF' for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        # filter by the PTF project name
        stmt = Image.query_images(project='PTF_DyC_survey')
        results1 = session.scalars(stmt).all()
        assert all(im.project == 'PTF_DyC_survey' for im in results1)
        assert all(im.instrument == 'PTF' for im in results1)
        assert len(results1) < total

        # filter by the two different project names for DECam:
        stmt = Image.query_images(project=['DECaPS', '2022A-724693'])
        results2 = session.scalars(stmt).all()
        assert all(im.project in ['DECaPS', '2022A-724693'] for im in results2)
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

        stmt = Image.query_images(filter='g DECam SDSS c0001 4720.0 1520.0')
        results7 = session.scalars(stmt).all()
        assert all(im.filter == 'g DECam SDSS c0001 4720.0 1520.0' for im in results7)
        assert all(im.instrument == 'DECam' for im in results7)
        assert set(results7) == set(results2)

        # filter by seeing FWHM
        value = 3.5
        stmt = Image.query_images(min_seeing=value)
        results1 = session.scalars(stmt).all()
        assert all(im.fwhm_estimate >= value for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(max_seeing=value)
        results2 = session.scalars(stmt).all()
        assert all(im.fwhm_estimate <= value for im in results2)
        assert len(results2) < total
        assert len(results1) + len(results2) == total

        stmt = Image.query_images(min_seeing=value, max_seeing=value)
        results3 = session.scalars(stmt).all()
        assert len(results3) == 0  # we will never have exactly that number

        # filter by limiting magnitude
        value = 25.0
        stmt = Image.query_images(min_lim_mag=value)
        results1 = session.scalars(stmt).all()
        assert all(im.lim_mag_estimate >= value for im in results1)
        assert len(results1) < total

        stmt = Image.query_images(max_lim_mag=value)
        results2 = session.scalars(stmt).all()
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

        # TODO: this fails because images don't have an "airmass" column
        # stmt = Image.query_images(max_airmass=1.5)

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

        # cross the DECam target and section ID with long exposure time
        target = 'DECaPS-West'
        section_id = 'N1'
        exp_time = 400.0

        stmt = Image.query_images(target=target, section_id=section_id, min_exp_time=exp_time)
        results2 = session.scalars(stmt).all()
        assert len(results2) == 1
        assert results2[0].instrument == 'DECam'
        assert results2[0].type == 'Sci'
        assert results2[0].exp_time == 576.0

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
