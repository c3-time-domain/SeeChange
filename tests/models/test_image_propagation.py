import pytest
import uuid
import sqlalchemy as sa
from models.base import SmartSession, Psycopg2Connection
from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference
from util.util import asUUID


# TODO: this test could be made faster by replacing the ptf fixtures with sim image fixtures (Issue #367)
# (Originally, there was a test here that did that, but it was replaced by an updated test
# that was moved from test_image_qaurying.)
def test_image_upstreams_downstreams( ptf_ref, ptf_supernova_image_datastores, ptf_subtraction1_datastore ):

    # Upstream of a regular image should be just an exposure
    upstrs = ptf_subtraction1_datastore.image.get_upstreams()
    assert len(upstrs) == 1
    assert isinstance( upstrs[0], Exposure )

    refimg = Image.get_by_id( ptf_ref.image_id )
    prov = Provenance.get( refimg.provenance_id )
    assert prov.process == 'coaddition'

    # Test upstreams of a coadd image
    with Psycopg2Connection() as conn:
        cursor = conn.cursor()
        cursor.execute( "SELECT image_id FROM image_coadd_component WHERE coadd_image_id=%(id)s",
                        { 'id': refimg.id } )
        dbcomps = set( [ asUUID(row[0]) for row in cursor.fetchall() ] )
    # There were 5 images summed together in the ptf_ref
    assert len(dbcomps) == 5
    assert set( asUUID(i) for i in refimg.upstream_image_ids ) == dbcomps
    assert set( asUUID(i) for i in refimg.coadd_component_ids ) == dbcomps
    images = refimg.get_upstreams( only_images=True )
    assert set( asUUID(i.id) for i in images ) == dbcomps

    loaded_image = Image.get_coadd_from_components(images, prov.id)
    assert loaded_image.id == refimg.id
    assert loaded_image.id != ptf_subtraction1_datastore.image.id

    # Test upstreams of a difference image
    subim = ptf_subtraction1_datastore.sub_image
    upstrs = subim.get_upstreams( only_images=True )
    assert len(upstrs) == 1
    assert isinstance( upstrs[0], Image )
    assert upstrs[0].id == ptf_subtraction1_datastore.image.id
    upstrs = subim.get_upstreams( only_images_and_reference=True )
    assert len(upstrs) == 2
    assert isinstance( upstrs[0], Reference )
    assert isinstance( upstrs[1], Image )
    assert [ asUUID(u.id) for u in upstrs ] == [ asUUID(ptf_ref.id),
                                                 asUUID(ptf_subtraction1_datastore.image.id) ]
    assert subim.coadd_component_ids == []
    assert [ asUUID(i) for i in subim.upstream_image_ids ] == [ asUUID(ptf_ref.image_id),
                                                                asUUID(ptf_subtraction1_datastore.image.id) ]
    upstrs = subim.get_upstreams()
    assert set( type(u) for u in upstrs ) == { Reference, Image, SourceList, Background,
                                               PSF, WorldCoordinates, ZeroPoint }

    # Test get_coadd_from_components
    new_image = None
    new_image2 = None
    new_image3 = None
    try:
        # make a new image with a new provenance
        new_image = Image.copy_image(refimg)
        newprov = Provenance.get( ptf_ref.provenance_id )
        newprov.process = 'copy'
        _ = newprov.upstreams    # Force newprov._upstreams to load
        newprov.update_id()
        newprov.insert()
        new_image.provenance_id = newprov.id
        # Not supposed to set _coadd_component_ids directly, so never do
        # it anywhere in code; set the components of a coadded image by
        # building it with Image.from_images().  But, do this here for
        # testing purposes.
        new_image._coadd_component_ids = refimg.coadd_component_ids
        new_image.save()
        new_image.insert()

        loaded_image = Image.get_coadd_from_components(refimg.upstream_image_ids, newprov.id)
        assert loaded_image.id == new_image.id
        assert new_image.id != refimg.id

        # use the original provenance but take down an image from the upstreams
        new_image2 = Image.copy_image( refimg )
        new_image2.provenance_id = prov.id
        # See note above about setting _coadd_component_ids directly (which is naughty)
        new_image2._coadd_component_ids = refimg.coadd_component_ids[1:]
        new_image2.save()
        new_image2.insert()

        images = [ Image.get_by_id( i ) for i in refimg.upstream_image_ids[1:] ]
        loaded_image = Image.get_coadd_from_components(images, prov.id)
        assert loaded_image.id != refimg.id
        assert loaded_image.id != new_image.id
        assert loaded_image.id == new_image2.id

        # use the original provenance but add images to the upstreams
        upstrids = refimg.upstream_image_ids + [ d.image.id for d in ptf_supernova_image_datastores ]

        new_image3 = Image.copy_image(refimg)
        new_image3.provenance_id = prov.id
        # See note above about setting _coadd_component_ids directly (which is naughty)
        new_image3._coadd_component_ids = upstrids
        new_image3.save()
        new_image3.insert()

        images = [ Image.get_by_id( i ) for i in upstrids ]
        loaded_image = Image.get_coadd_from_components(images, prov.id)
        assert loaded_image.id == new_image3.id

    finally:
        if new_image is not None:
            new_image.delete_from_disk_and_database()
        if new_image2 is not None:
            new_image2.delete_from_disk_and_database()
        if new_image3 is not None:
            new_image3.delete_from_disk_and_database()


    # Image downstreams of the original image should only be the sum image
    downstr = ptf_subtraction1_datastore.image.get_downstreams( only_images=True )
    assert len(downstr) == 1
    assert asUUID( downstr[0].id ) == asUUID( subim.id )

    # All downstreams of the original image should be a lot of things
    downstr = ptf_subtraction1_datastore.image.get_downstreams()
    assert len(downstr) == 6
    assert set( type(d) for d in downstr ) == { SourceList, Background, PSF, WorldCoordinates, ZeroPoint, Image }

    # The downstream of the ref image should be the reference in addition to the other data products
    downstr = ptf_ref.image.get_downstreams()
    assert asUUID(ptf_ref.id) in [ asUUID(i.id) for i in downstr ]
    assert set( type(d) for d in downstr ) == { SourceList, Background, PSF, WorldCoordinates, ZeroPoint, Reference }

    # The sub image should have no downstreams because we haven't run anything after subtraction in the fixture
    assert len( subim.get_downstreams() ) == 0


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
        cleanupprovs = []
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

        refprov = Provenance( process='manual_reference',
                              upstreams=[],
                              parameters={},
                              is_testing=True )
        refprov.insert_if_needed()
        cleanupprovs.append( refprov )
        srclst = SourceList( image_id=sim_image2.id, num_sources=0, provenance_id=provenance_extra.id )
        srclst.md5sum = uuid.uuid4()
        srclst.filepath = 'foo'
        srclst.insert()
        ref = Reference( image_id=sim_image2.id, sources_id=srclst.id, provenance_id=refprov.id )
        ref.insert()
        # Make sure the ref badness propagates right
        sim_image2.update_downstream_badness()
        ref = Reference.get_by_id( ref.id )
        assert ref.own_bitflag == 0
        assert ref.bitflag == sim_image2.bitflag
        sim_image4 = Image.from_new_and_ref( sim_image3, ref )
        improvs = Provenance.get_batch( [ refprov.id, sim_image3.provenance_id ] )
        prov4 = Provenance( process='testsub',
                            upstreams=improvs,
                            parameters={},
                            is_testing=True
                           )
        prov4.insert_if_needed()
        sim_image4.provenance_id = prov4.id
        sim_image4.md5sum = uuid.uuid4()   # spoof so we don't need to save data
        sim_image4.filepath = sim_image4.invent_filepath()
        cleanups.append( sim_image4 )
        sim_image4.insert()

        assert sim_image4.id is not None
        assert sim_image4.ref_id == ref.id
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
        srclst = SourceList( image_id=sim_image6.id, num_sources=0, provenance_id=provenance_extra.id )
        srclst.md5sum = uuid.uuid4()
        srclst.filepath = 'bar'
        srclst.insert()
        ref = Reference( image_id=sim_image6.id, sources_id=srclst.id, provenance_id=refprov.id )
        ref.insert()
        sim_image7 = Image.from_ref_and_new( ref, sim_image5 )
        improvs = Provenance.get_batch( [ refprov.id, sim_image6.provenance_id ] )
        prov7 = Provenance( process='testsub',
                            upstreams=improvs,
                            parameters={},
                            is_testing=True
                           )
        prov7.insert_if_needed()
        sim_image7.provenance_id = prov7.id
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
        with SmartSession() as sess:
            # Make sure no coadd association table entries are left behind
            for cleanup in cleanups:
                sess.execute( sa.text( "DELETE FROM image_coadd_component "
                                       "WHERE image_id=:id OR coadd_image_id=:id" ),
                              { 'id': cleanup.id } )
            sess.commit()
        for cleanup in cleanups:
            cleanup.delete_from_disk_and_database()
        with SmartSession() as sess:
            for cleanup in cleanupprovs:
                sess.execute( sa.text( "DELETE FROM provenances WHERE _id=:id" ), { 'id': cleanup.id } )
            sess.commit()
