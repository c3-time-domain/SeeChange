import pytest
import uuid
import sqlalchemy as sa
from models.base import SmartSession
from models.image import Image
from tests.fixtures.simulated import ImageCleanup


def test_image_upstreams_downstreams(sim_image1, sim_reference, provenance_extra, data_dir):
    with SmartSession() as session:
        sim_image1 = sim_image1.merge_all(session)
        sim_reference = sim_reference.merge_all(session)

        # make sure the new image matches the reference in all these attributes
        sim_image1.filter = sim_reference.filter
        sim_image1.target = sim_reference.target
        sim_image1.section_id = sim_reference.section_id

        new = Image.from_new_and_ref(sim_image1, sim_reference.image)
        new.provenance = session.merge(provenance_extra)

        # save and delete at the end
        cleanup = ImageCleanup.save_image(new)

        session.add(new)
        session.commit()

    # new make sure a new session can find all the upstreams/downstreams
    with SmartSession() as session:
        sim_image1 = sim_image1.merge_all(session)
        sim_reference = sim_reference.merge_all(session)
        new = new.merge_all(session)

        # check the upstreams/downstreams for the new image
        upstream_ids = [u.id for u in new.get_upstreams(session=session)]
        assert sim_image1.id in upstream_ids
        assert sim_reference.image_id in upstream_ids
        downstream_ids = [d.id for d in new.get_downstreams(session=session)]
        assert len(downstream_ids) == 0

        upstream_ids = [u.id for u in sim_image1.get_upstreams(session=session)]
        assert [sim_image1.exposure_id] == upstream_ids
        downstream_ids = [d.id for d in sim_image1.get_downstreams(session=session)]
        assert [new.id] == downstream_ids  # should be the only downstream

        # check the upstreams/downstreams for the reference image
        upstreams = sim_reference.image.get_upstreams(session=session)
        assert len(upstreams) == 5  # was made of five images
        assert all([isinstance(u, Image) for u in upstreams])
        source_images_ids = [im.id for im in sim_reference.image.upstream_images]
        upstream_ids = [u.id for u in upstreams]
        assert set(upstream_ids) == set(source_images_ids)
        downstream_ids = [d.id for d in sim_reference.image.get_downstreams(session=session)]
        assert [new.id] == downstream_ids  # should be the only downstream

        # test for the Image.downstream relationship
        assert len(upstreams[0].downstream_images) == 1
        assert upstreams[0].downstream_images == [sim_reference.image]
        assert len(upstreams[1].downstream_images) == 1
        assert upstreams[1].downstream_images == [sim_reference.image]

        assert len(sim_image1.downstream_images) == 1
        assert sim_image1.downstream_images == [new]

        assert len(sim_reference.image.downstream_images) == 1
        assert sim_reference.image.downstream_images == [new]

        assert len(new.downstream_images) == 0

        # add a second "new" image using one of the reference's upstreams instead of the reference
        new2 = Image.from_new_and_ref(sim_image1, upstreams[0])
        new2.provenance = session.merge(provenance_extra)
        new2.mjd += 1  # make sure this image has a later MJD, so it comes out later on the downstream list!

        # save and delete at the end
        cleanup2 = ImageCleanup.save_image(new2)

        session.add(new2)
        session.commit()

        session.refresh(upstreams[0])
        assert len(upstreams[0].downstream_images) == 2
        assert set(upstreams[0].downstream_images) == set([sim_reference.image, new2])

        session.refresh(upstreams[1])
        assert len(upstreams[1].downstream_images) == 1
        assert upstreams[1].downstream_images == [sim_reference.image]

        session.refresh(sim_image1)
        assert len(sim_image1.downstream_images) == 2
        assert set(sim_image1.downstream_images) == set([new, new2])

        session.refresh(sim_reference.image)
        assert len(sim_reference.image.downstream_images) == 1
        assert sim_reference.image.downstream_images == [new]

        assert len(new2.downstream_images) == 0


def test_image_badness(sim_image1):

    with SmartSession() as session:
        sim_image1 = session.merge(sim_image1)

        # this is not a legit "badness" keyword...
        with pytest.raises(ValueError, match='Keyword "foo" not recognized'):
            sim_image1.badness = 'foo'

        # this is a legit keyword, but for cutouts, not for images
        with pytest.raises(ValueError, match='Keyword "cosmic ray" not recognized'):
            sim_image1.badness = 'cosmic ray'

        # this is a legit keyword, but for images, using no space and no capitalization
        sim_image1.badness = 'brightsky'

        # retrieving this keyword, we do get it capitalized and with a space:
        assert sim_image1.badness == 'bright sky'
        assert sim_image1.bitflag == 2 ** 5  # the bright sky bit is number 5

        # what happens when we add a second keyword?
        sim_image1.badness = 'Bright_sky, Banding'  # try this with capitalization and underscores
        assert sim_image1.bitflag == 2 ** 5 + 2 ** 1  # the bright sky bit is number 5, banding is number 1
        assert sim_image1.badness == 'banding, bright sky'

        # now add a third keyword, but on the Exposure
        sim_image1.exposure.badness = 'saturation'
        session.add(sim_image1)
        session.commit()

        # a manual way to propagate bitflags downstream
        sim_image1.exposure.update_downstream_badness(session=session)  # make sure the downstreams get the new badness
        session.commit()
        assert sim_image1.bitflag == 2 ** 5 + 2 ** 3 + 2 ** 1  # saturation bit is 3
        assert sim_image1.badness == 'banding, saturation, bright sky'

        # adding the same keyword on the exposure and the image makes no difference
        sim_image1.exposure.badness = 'Banding'
        sim_image1.exposure.update_downstream_badness(session=session)  # make sure the downstreams get the new badness
        session.commit()
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
        with SmartSession() as session:
            sim_image1 = session.merge(sim_image1)
            sim_image2 = session.merge(sim_image2)
            sim_image3 = session.merge(sim_image3)
            sim_image5 = session.merge(sim_image5)
            sim_image6 = session.merge(sim_image6)

            images = [sim_image1, sim_image2, sim_image3, sim_image5, sim_image6]
            cleanups = []
            filter = 'g'
            target = str(uuid.uuid4())
            project = 'test project'
            for im in images:
                im.filter = filter
                im.target = target
                im.project = project
                session.add(im)

            session.commit()

            # the image itself is marked bad because of bright sky
            sim_image2.badness = 'BrightSky'
            assert sim_image2.badness == 'bright sky'
            assert sim_image2.bitflag == 2 ** 5
            session.commit()

            # note that this image is not directly bad, but the exposure has banding
            sim_image3.exposure.badness = 'banding'
            sim_image3.exposure.update_downstream_badness(session=session)
            session.commit()

            assert sim_image3.badness == 'banding'
            assert sim_image1._bitflag == 0  # the exposure is bad!
            assert sim_image3.bitflag == 2 ** 1
            session.commit()

            # find the images that are good vs bad
            good_images = session.scalars(sa.select(Image).where(Image.bitflag == 0)).all()
            assert sim_image1.id in [i.id for i in good_images]

            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert sim_image2.id in [i.id for i in bad_images]
            assert sim_image3.id in [i.id for i in bad_images]

            # make an image from the two bad exposures using subtraction

            sim_image4 = Image.from_new_and_ref(sim_image3, sim_image2)
            sim_image4.provenance = provenance_extra
            sim_image4.provenance.upstreams = sim_image4.get_upstream_provenances()
            cleanups.append(ImageCleanup.save_image(sim_image4))
            sim_image4 = session.merge(sim_image4)
            images.append(sim_image4)
            session.commit()

            assert sim_image4.id is not None
            assert sim_image4.ref_image == sim_image2
            assert sim_image4.new_image == sim_image3

            # check that badness is loaded correctly from both parents
            assert sim_image4.badness == 'banding, bright sky'
            assert sim_image4._bitflag == 0  # the image itself is not flagged
            assert sim_image4.bitflag == 2 ** 1 + 2 ** 5

            # check that filtering on this value gives the right bitflag
            bad_images = session.scalars(sa.select(Image).where(Image.bitflag == 2 ** 1 + 2 ** 5)).all()
            assert sim_image4.id in [i.id for i in bad_images]
            assert sim_image3.id not in [i.id for i in bad_images]
            assert sim_image2.id not in [i.id for i in bad_images]

            # check that adding a badness on the image itself is added to the total badness
            sim_image4.badness = 'saturation'
            session.add(sim_image4)
            session.commit()
            assert sim_image4.badness == 'banding, saturation, bright sky'
            assert sim_image4._bitflag == 2 ** 3  # only this bit is from the image itself

            # make a new subtraction:
            sim_image7 = Image.from_ref_and_new(sim_image6, sim_image5)
            sim_image7.provenance = provenance_extra
            cleanups.append(ImageCleanup.save_image(sim_image7))
            sim_image7 = session.merge(sim_image7)
            images.append(sim_image7)
            session.commit()

            # check that the new subtraction is not flagged
            assert sim_image7.badness == ''
            assert sim_image7._bitflag == 0
            assert sim_image7.bitflag == 0

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
            sim_image8 = Image.from_images([sim_image1, sim_image2, sim_image3, sim_image5, sim_image6])
            sim_image8.provenance = provenance_extra
            cleanups.append(ImageCleanup.save_image(sim_image8))
            images.append(sim_image8)
            sim_image8 = session.merge(sim_image8)
            session.commit()

            assert sim_image8.badness == 'banding, bright sky'
            assert sim_image8.bitflag == 2 ** 1 + 2 ** 5

            # does this work in queries (i.e., using the bitflag hybrid expression)?
            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert sim_image8.id in [i.id for i in bad_images]
            bad_coadd = session.scalars(sa.select(Image).where(Image.bitflag == 2 ** 1 + 2 ** 5)).all()
            assert sim_image8.id in [i.id for i in bad_coadd]

            # get rid of this coadd to make a new one
            sim_image8.delete_from_disk_and_database(session=session)
            cleanups.pop()
            images.pop()

            # now let's add the subtraction image to the coadd:
            # make a coadded image (now including the subtraction sim_image4):
            sim_image8 = Image.from_images([sim_image1, sim_image2, sim_image3, sim_image4, sim_image5, sim_image6])
            sim_image8.provenance = provenance_extra
            cleanups.append(ImageCleanup.save_image(sim_image8))
            sim_image8 = session.merge(sim_image8)
            images.append(sim_image8)
            session.commit()

            assert sim_image8.badness == 'banding, saturation, bright sky'
            assert sim_image8.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 5  # this should be 42

            # does this work in queries (i.e., using the bitflag hybrid expression)?
            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert sim_image8.id in [i.id for i in bad_images]
            bad_coadd = session.scalars(sa.select(Image).where(Image.bitflag == 42)).all()
            assert sim_image8.id in [i.id for i in bad_coadd]

            # try to add some badness to one of the underlying exposures
            sim_image1.exposure.badness = 'shaking'
            session.add(sim_image1)
            sim_image1.exposure.update_downstream_badness(session=session)
            session.commit()

            assert 'shaking' in sim_image1.badness
            assert 'shaking' in sim_image8.badness

    finally:  # cleanup
        with SmartSession() as session:
            session.autoflush = False
            for im in images:
                im = im.merge_all(session)
                exp = im.exposure
                im.delete_from_disk_and_database(session=session, commit=False)

                if exp is not None and sa.inspect(exp).persistent:
                    session.delete(exp)

            session.commit()

