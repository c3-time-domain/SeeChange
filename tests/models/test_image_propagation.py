import pytest
import uuid
import sqlalchemy as sa
from models.base import SmartSession
from models.image import Image
from models.exposure import Exposure
from models.provenance import Provenance
from tests.fixtures.simulated import ImageCleanup


def test_image_upstreams_downstreams(sim_image1, sim_reference, provenance_extra, data_dir):

    # make sure the new image matches the reference in all these attributes
    sim_image1.filter = sim_reference.filter
    sim_image1.target = sim_reference.target
    sim_image1.section_id = sim_reference.section_id

    sim_reference_image = Image.get_by_id( sim_reference.image_id )

    diff_image = Image.from_new_and_ref(sim_image1, sim_reference_image)
    diff_image.provenance_id = provenance_extra.id

    # save and delete at the end
    cleanup = ImageCleanup.save_image( diff_image )
    diff_image.insert()

    # Reload the image from the database to make sure all the upstreams and downstreams
    #   are there.
    new = Image.get_by_id( diff_image.id )

    # check the upstreams/downstreams for the new image
    # TODO : make a diff image where there are sources in
    #   the database, because then those ought to be part of the
    #   diff image upstreams.  But, the provenance would have
    #   to have the right upstream provenances for that to work,
    #   so this isn't a simple few-liner.
    upstream_ids = [ u.id for u in new.get_upstreams() ]
    assert sim_image1.id in upstream_ids
    assert sim_reference.image_id in upstream_ids
    downstream_ids = [ d.id for d in new.get_downstreams() ]
    assert len(downstream_ids) == 0

    upstream_ids = [ u.id for u in sim_image1.get_upstreams() ]
    assert [sim_image1.exposure_id] == upstream_ids
    downstream_ids = [ d.id for d in sim_image1.get_downstreams() ]
    assert [new.id] == downstream_ids  # should be the only downstream

    # check the upstreams/downstreams for the reference image
    upstream_images = sim_reference_image.get_upstreams( only_images=True )
    assert len(upstream_images) == 5  # was made of five images
    assert all( [ isinstance(u, Image) for u in upstream_images ] )
    source_images_ids = [ im.id for im in upstream_images ]
    downstream_ids = [d.id for d in sim_reference_image.get_downstreams()]
    assert [new.id] == downstream_ids  # should be the only downstream

    # test for the Image.downstream relationship
    assert len( upstream_images[0].get_downstreams() ) == 1
    assert [ i.id for i in upstream_images[0].get_downstreams() ] == [ sim_reference_image.id ]
    assert len( upstream_images[1].get_downstreams() ) == 1
    assert [ i.id for i in upstream_images[1].get_downstreams() ] == [ sim_reference_image.id ]

    assert len( sim_image1.get_downstreams() ) == 1
    assert [ i.id for i in sim_image1.get_downstreams() ] == [ diff_image.id ]

    assert len( sim_reference_image.get_downstreams() ) == 1
    assert [ i.id for i in sim_reference_image.get_downstreams() ] == [ diff_image.id ]

    assert len( new.get_downstreams() ) == 0

    # add a second "new" image using one of the reference's upstreams instead of the reference
    refupstrim = Image.get_by_id( sim_reference_image.upstream_image_ids[0] )
    new2 = Image.from_new_and_ref( sim_image1, refupstrim )
    new2.provenance_id = provenance_extra.id
    new2.mjd += 1  # make sure this image has a later MJD, so it comes out later on the downstream list!

    # save and delete at the end
    cleanup2 = ImageCleanup.save_image( new2 )
    new2.insert()

    assert len( refupstrim.get_downstreams() ) == 2
    assert set( [ i.id for i in refupstrim.get_downstreams() ] ) == set( [ sim_reference_image.id, new2.id ] )

    refupstrim = Image.get_by_id( sim_reference_image.upstream_image_ids[1] )
    assert len( refupstrim.get_downstreams() )
    assert [ i.id for i in refupstrim.get_downstreams() ] == [ sim_reference_image.id ]

    assert len( sim_image1.get_downstreams() ) == 2
    assert set( [ i.id for i in sim_image1.get_downstreams() ] ) == set( [ diff_image.id, new2.id ] )

    assert len( sim_reference_image.get_downstreams() ) == 1
    assert [ i.id for i in sim_reference_image.get_downstreams() ] == [ diff_image.id ]

    assert len( new2.get_downstreams() ) == 0


def test_image_badness(sim_image1):

    exposure = Exposure.get_by_id( sim_image1.exposure_id )

    # this is not a legit "badness" keyword...
    with pytest.raises(ValueError, match='Keyword "foo" not recognized'):
        sim_image1.set_badness( 'foo' )

    # this is a legit keyword, but for cutouts, not for images
    with pytest.raises(ValueError, match='Keyword "cosmic ray" not recognized'):
        sim_image1.set_badness( 'cosmic ray' )

    # this is a legit keyword, but for images, using no space and no capitalization
    sim_image1.set_badness( 'brightsky' )

    # retrieving this keyword, we do get it capitalized and with a space:
    assert sim_image1.badness == 'bright sky'
    assert sim_image1.bitflag == 2 ** 5  # the bright sky bit is number 5

    # what happens when we add a second keyword?
    sim_image1.set_badness( 'Bright_sky, Banding' )  # try this with capitalization and underscores
    assert sim_image1.bitflag == 2 ** 5 + 2 ** 1  # the bright sky bit is number 5, banding is number 1
    assert sim_image1.badness == 'banding, bright sky'

    # update this in the database, make sure it took
    sim_image1.upsert()
    testim = Image.get_by_id( sim_image1.id )
    assert testim.bitflag == sim_image1.bitflag

    # now add a third keyword, but on the Exposure
    exposure.set_badness( 'saturation' )
    exposure.upsert()

    # a manual way to propagate bitflags downstream
    exposure.update_downstream_badness()  # make sure the downstreams get the new badness

    # Reload the image to make sure it now has the new flag
    sim_image1 = Image.get_by_id( sim_image1.id )
    assert sim_image1.bitflag == 2 ** 5 + 2 ** 3 + 2 ** 1  # saturation bit is 3
    assert sim_image1.badness == 'banding, saturation, bright sky'

    # adding the same keyword on the exposure and the image makes no difference;
    # also make sure that we don't see "saturation" on the image any more
    # once the upstream no longer has it
    exposure.set_badness( 'Banding' )
    exposure.upsert()
    exposure.update_downstream_badness()
    sim_image1 = Image.get_by_id( sim_image1.id )
    assert sim_image1.bitflag == 2 ** 5 + 2 ** 1
    assert sim_image1.badness == 'banding, bright sky'

    # try appending keywords to the image
    sim_image1.append_badness('shaking')
    assert sim_image1.bitflag == 2 ** 5 + 2 ** 2 + 2 ** 1  # shaking bit is 2
    assert sim_image1.badness == 'banding, shaking, bright sky'


def test_multiple_images_badness(
        sim_image1,
        sim_image2,
        sim_image3,
        sim_image5,
        sim_image6,
        provenance_extra
):
    try:
        images = [sim_image1, sim_image2, sim_image3, sim_image5, sim_image6]
        cleanups = []
        filter = 'g'
        target = str(uuid.uuid4())
        project = 'test project'
        for im in images:
            im.filter = filter
            im.target = target
            im.project = project
            im.upsert()

        # the image itself is marked bad because of bright sky
        sim_image2.set_badness( 'BrightSky' )
        assert sim_image2.badness == 'bright sky'
        assert sim_image2.bitflag == 2 ** 5
        sim_image2.upsert()

        # note that this image is not directly bad, but the exposure has banding
        sim_exposure3 = Exposure.get_by_id( sim_image3.exposure_id )
        sim_exposure3.set_badness( 'banding' )
        sim_exposure3.upsert()
        sim_exposure3.update_downstream_badness()

        sim_image3 = Image.get_by_id( sim_image3.id )
        assert sim_image3.badness == 'banding'
        assert sim_image1.own_bitflag == 0  # the exposure is bad!
        assert sim_image3.bitflag == 2 ** 1

        # find the images that are good vs bad
        with SmartSession() as session:
            good_images = session.scalars(sa.select(Image).where(Image.bitflag == 0)).all()
            assert sim_image1.id in [i.id for i in good_images]

            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert sim_image2.id in [i.id for i in bad_images]
            assert sim_image3.id in [i.id for i in bad_images]

        # make an image from the two bad exposures using subtraction

        sim_image4 = Image.from_new_and_ref( sim_image3, sim_image2 )
        improvs = Provenance.get_batch( [ sim_image3.provenance_id, sim_image2.provenance_id ] )
        prov4 = Provenance( process='testsub',
                            upstreams=improvs,
                            parameters={},
                            is_testing=True
                           )
        prov4.insert_if_needed()
        sim_image4.provenance_id = prov4.id
        # cleanups.append( ImageCleanup.save_image(sim_image4) )
        sim_image4.md5sum = uuid.uuid4()   # spoof so we don't need to save data
        sim_image4.filepath = sim_image4.invent_filepath()
        cleanups.append( sim_image4 )
        sim_image4.insert()

        assert sim_image4.id is not None
        assert sim_image4.ref_image_id == sim_image2.id
        assert sim_image4.new_image_id == sim_image3.id

        # check that badness is loaded correctly from both parents
        assert sim_image4.badness == 'banding, bright sky'
        assert sim_image4.own_bitflag == 0  # the image itself is not flagged
        assert sim_image4.bitflag == 2 ** 1 + 2 ** 5

        # check that filtering on this value gives the right bitflag
        with SmartSession() as session:
            bad_images = session.scalars(sa.select(Image).where(Image.bitflag == 2 ** 1 + 2 ** 5)).all()
            assert sim_image4.id in [i.id for i in bad_images]
            assert sim_image3.id not in [i.id for i in bad_images]
            assert sim_image2.id not in [i.id for i in bad_images]

        # check that adding a badness on the image itself is added to the total badness
        sim_image4.set_badness( 'saturation' )
        sim_image4.upsert()
        assert sim_image4.badness == 'banding, saturation, bright sky'
        assert sim_image4.own_bitflag == 2 ** 3  # only this bit is from the image itself

        # make a new subtraction:
        sim_image7 = Image.from_ref_and_new( sim_image6, sim_image5 )
        improvs = Provenance.get_batch( [ sim_image6.provenance_id, sim_image5.provenance_id ] )
        prov7 = Provenance( process='testsub',
                            upstreams=improvs,
                            parameters={},
                            is_testing=True
                           )
        prov7.insert_if_needed()
        sim_image7.provenance_id = prov7.id
        # cleanups.append( ImageCleanup.save_image(sim_image7) )
        sim_image7.md5sum = uuid.uuid4()   # spoof so we don't need to save data
        sim_image7.filepath = sim_image7.invent_filepath()
        cleanups.append( sim_image7 )
        sim_image7.insert()

        # check that the new subtraction is not flagged
        assert sim_image7.badness == ''
        assert sim_image7.own_bitflag == 0
        assert sim_image7.bitflag == 0

        with SmartSession() as session:
            good_images = session.scalars(sa.select(Image).where(Image.bitflag == 0)).all()
            assert sim_image5.id in [i.id for i in good_images]
            assert sim_image5.id in [i.id for i in good_images]
            assert sim_image7.id in [i.id for i in good_images]

            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert sim_image5.id not in [i.id for i in bad_images]
            assert sim_image6.id not in [i.id for i in bad_images]
            assert sim_image7.id not in [i.id for i in bad_images]

        # let's try to coadd an image based on some good and bad images
        # as a reminder, sim_image2 has bright sky (5),
        # sim_image3's exposure has banding (1), while
        # sim_image4 has saturation (3).

        # make a coadded image (without including the subtraction sim_image4):
        sim_image8 = Image.from_images( [sim_image1, sim_image2, sim_image3, sim_image5, sim_image6] )
        sim_image8.is_coadd = True
        improvs = Provenance.get_batch( [ sim_image1.provenance_id, sim_image2.provenance_id,
                                          sim_image3.provenance_id, sim_image5.provenance_id,
                                          sim_image6.provenance_id ] )
        prov8 = Provenance( process='testcoadd',
                            upstreams=improvs,
                            parameters={},
                            is_testing=True
                           )
        prov8.insert_if_needed()
        sim_image8.provenance_id = prov8.id
        # cleanups.append( ImageCleanup.save_image(sim_image8) )
        sim_image8.md5sum = uuid.uuid4()   # spoof so we don't need to save data
        sim_image8.filepath = sim_image8.invent_filepath()
        cleanups.append( sim_image8 )
        images.append( sim_image8 )
        sim_image8.insert()

        assert sim_image8.badness == 'banding, bright sky'
        assert sim_image8.bitflag == 2 ** 1 + 2 ** 5

        # does this work in queries (i.e., using the bitflag hybrid expression)?
        with SmartSession() as session:
            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert sim_image8.id in [i.id for i in bad_images]
            bad_coadd = session.scalars(sa.select(Image).where(Image.bitflag == 2 ** 1 + 2 ** 5)).all()
            assert sim_image8.id in [i.id for i in bad_coadd]

        # get rid of this coadd to make a new one
        sim_image8.delete_from_disk_and_database()
        cleanups.pop()
        images.pop()

        # now let's add the subtraction image to the coadd:
        # make a coadded image (now including the subtraction sim_image4):
        sim_image8 = Image.from_images( [sim_image1, sim_image2, sim_image3, sim_image4, sim_image5, sim_image6] )
        sim_image8.is_coadd = True
        improvs = Provenance.get_batch( [sim_image1.provenance_id, sim_image2.provenance_id,
                                         sim_image3.provenance_id, sim_image4.provenance_id,
                                         sim_image5.provenance_id, sim_image6.provenance_id ] )
        sim_image8.provenance_id = prov8.id
        # cleanups.append( ImageCleanup.save_image(sim_image8) )
        sim_image8.md5sum = uuid.uuid4()   # spoof so we don't need to save data
        sim_image8.filepath = sim_image8.invent_filepath()
        cleanups.append( sim_image8 )
        images.append(sim_image8)
        sim_image8.insert()

        assert sim_image8.badness == 'banding, saturation, bright sky'
        assert sim_image8.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 5  # this should be 42

        # does this work in queries (i.e., using the bitflag hybrid expression)?
        with SmartSession() as session:
            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert sim_image8.id in [i.id for i in bad_images]
            bad_coadd = session.scalars(sa.select(Image).where(Image.bitflag == 42)).all()
            assert sim_image8.id in [i.id for i in bad_coadd]

        # try to add some badness to one of the underlying exposures
        sim_exposure1 = Exposure.get_by_id( sim_image1.exposure_id )
        sim_exposure1.set_badness( 'shaking' )
        sim_exposure1.upsert()
        sim_exposure1.update_downstream_badness()

        sim_image1 = Image.get_by_id( sim_image1.id )
        sim_image8 = Image.get_by_id( sim_image8.id )
        assert 'shaking' in sim_image1.badness
        assert 'shaking' in sim_image8.badness

    finally:
        # I don't know why, but the _del_ method of ImageCleanup was not
        #   getting called for image7 and image8 before the post-yield
        #   parts of the code_version fixture.  So, I've commented out
        #   the ImageCleanups above and am manually cleaning up here.  I
        #   also stopped actually saving data, since we don't use it in
        #   this test, and just spoofed the md5sums.

        # Because some of these images are used as upstreams for other images,
        #   we also have to clear out the association table
        with SmartSession() as sess:
            for cleanup in cleanups:
                sess.execute( sa.text( "DELETE FROM image_upstreams_association "
                                       "WHERE upstream_id=:id" ),
                              { 'id': cleanup.id } )
            sess.commit()
        for cleanup in cleanups:
            cleanup.delete_from_disk_and_database()
