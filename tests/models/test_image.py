import os
import pytest
import re
import hashlib
import pathlib
import uuid

import numpy as np

from astropy.io import fits

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession, FileOnDiskMixin
from models.image import Image
from models.enums_and_bitflags import image_preprocessing_inverse, string_to_bitflag
from models.psf import PSF
from models.source_list import SourceList
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from tests.conftest import rnd_str
from tests.fixtures.simulated import ImageCleanup


def test_image_no_null_values(provenance_base):

    required = {
        'mjd': 58392.1,
        'end_mjd': 58392.1 + 30 / 86400,
        'exp_time': 30,
        'ra': np.random.uniform(0, 360),
        'dec': np.random.uniform(-90, 90),
        'ra_corner_00': 0,
        'ra_corner_01': 0,
        'ra_corner_10': 0,
        'ra_corner_11': 0,
        'dec_corner_00': 0,
        'dec_corner_01': 0,
        'dec_corner_10': 0,
        'dec_corner_11': 0,
        'instrument': 'DemoInstrument',
        'telescope': 'DemoTelescope',
        'project': 'foo',
        'target': 'bar',
        'provenance_id': provenance_base.id,
    }

    added = {}

    # use non-capturing groups to extract the column name from the error message
    expr = r'(?:null value in column )(".*")(?: of relation "images" violates not-null constraint)'

    try:
        im_id = None  # make sure to delete the image if it is added to DB

        # md5sum is spoofed as we don't have this file saved to archive
        image = Image(f"Demo_test_{rnd_str(5)}.fits", md5sum=uuid.uuid4(), nofile=True, section_id=1)
        with SmartSession() as session:
            for i in range(len(required)):
                image = session.merge(image)
                # set the exposure to the values in "added" or None if not in "added"
                for k in required.keys():
                    setattr(image, k, added.get(k, None))

                # without all the required columns on image, it cannot be added to DB
                with pytest.raises(IntegrityError) as exc:
                    session.add(image)
                    session.commit()
                    im_id = image.id
                session.rollback()

                # a constraint on a column being not-null was violated
                match_obj = re.search(expr, str(exc.value))
                assert match_obj is not None

                # find which column raised the error
                colname = match_obj.group(1).replace('"', '')

                # add missing column name:
                added.update({colname: required[colname]})

        for k in required.keys():
            setattr(image, k, added.get(k, None))
        session.add(image)
        session.commit()
        im_id = image.id
        assert im_id is not None

    finally:
        # cleanup
        with SmartSession() as session:
            found_image = None
            if im_id is not None:
                found_image = session.scalars(sa.select(Image).where(Image.id == im_id)).first()
            if found_image is not None:
                session.delete(found_image)
                session.commit()


def test_image_must_have_md5(sim_image_uncommitted, provenance_base):
    try:
        im = sim_image_uncommitted
        assert im.md5sum is None
        assert im.md5sum_extensions is None

        im.provenance = provenance_base
        _ = ImageCleanup.save_image(im, archive=False)

        im.md5sum = None
        with SmartSession() as session:

            with pytest.raises(IntegrityError, match='violates check constraint'):
                im = session.merge(im)
                session.commit()
            session.rollback()

            # adding md5sums should fix this problem
            _2 = ImageCleanup.save_image(im, archive=True)
            im = session.merge(im)
            session.commit()

    finally:
        with SmartSession() as session:
            im = session.merge(im)
            exp = im.exposure
            im.delete_from_disk_and_database(session)

            if sa.inspect(exp).persistent:
                session.delete(exp)
                session.commit()


def test_image_archive_singlefile(sim_image_uncommitted, provenance_base, archive, test_config):
    im = sim_image_uncommitted
    im.data = np.float32( im.raw_data )
    im.flags = np.random.randint(0, 100, size=im.raw_data.shape, dtype=np.uint16)

    archive_dir = archive.test_folder_path
    single_fileness = test_config.value('storage.images.single_file')

    try:
        with SmartSession() as session:
            # Do single file first
            test_config.set_value('storage.images.single_file', True)
            im.provenance = session.merge(provenance_base)
            im.exposure = session.merge(im.exposure)  # make sure the exposure and provenance/code versions merge
            # Make sure that the archive is *not* written when we tell it not to.
            im.save( no_archive=True )
            assert im.md5sum is None
            archive_path = os.path.join(archive_dir, im.filepath)
            with pytest.raises(FileNotFoundError):
                ifp = open( archive_path, 'rb' )
                ifp.close()
            im.remove_data_from_disk()

            # Save to the archive, make sure it all worked
            im.save()
            localmd5 = hashlib.md5()
            with open( im.get_fullpath( nofile=False ), 'rb' ) as ifp:
                localmd5.update( ifp.read() )
            assert localmd5.hexdigest() == im.md5sum.hex
            archivemd5 = hashlib.md5()
            with open( archive_path, 'rb' ) as ifp:
                archivemd5.update( ifp.read() )
            assert archivemd5.hexdigest() == im.md5sum.hex

            # Make sure that we can download from the archive
            im.remove_data_from_disk()
            with pytest.raises(FileNotFoundError):
                assert isinstance( im.get_fullpath( nofile=True ), str )
                ifp = open( im.get_fullpath( nofile=True ), "rb" )
                ifp.close()
            p = im.get_fullpath( nofile=False )
            localmd5 = hashlib.md5()
            with open( im.get_fullpath( nofile=False ), 'rb' ) as ifp:
                localmd5.update( ifp.read() )
            assert localmd5.hexdigest() == im.md5sum.hex

            # Make sure that the md5sum is properly saved to the database
            im.provenance = session.merge(im.provenance)
            session.add( im )
            session.commit()
            with SmartSession() as differentsession:
                dbimage = differentsession.query(Image).filter(Image.id == im.id)[0]
                assert dbimage.md5sum.hex == im.md5sum.hex

            # Make sure we can purge the archive
            im.delete_from_disk_and_database(session=session, commit=True)
            with pytest.raises(FileNotFoundError):
                ifp = open( archive_path, 'rb' )
                ifp.close()
            assert im.md5sum is None

    finally:
        with SmartSession() as session:
            exp = session.merge(im.exposure)

            if sa.inspect(exp).persistent:
                session.delete(exp)
                session.commit()
        test_config.set_value('storage.images.single_file', single_fileness)


def test_image_archive_multifile(sim_image_uncommitted, provenance_base, archive, test_config):
    im = sim_image_uncommitted
    im.data = np.float32( im.raw_data )
    im.flags = np.random.randint(0, 100, size=im.raw_data.shape, dtype=np.uint16)
    im.weight = None

    archive_dir = archive.test_folder_path
    single_fileness = test_config.value('storage.images.single_file')

    try:
        with SmartSession() as session:
            im.provenance = provenance_base

            # Now do multiple images
            test_config.set_value('storage.images.single_file', False)

            # Make sure that the archive is not written when we tell it not to
            im.save( no_archive=True )
            localmd5s = {}
            assert len(im.get_fullpath(nofile=True)) == 2
            for fullpath in im.get_fullpath(nofile=True):
                localmd5s[fullpath] = hashlib.md5()
                with open(fullpath, "rb") as ifp:
                    localmd5s[fullpath].update(ifp.read())
            assert im.md5sum is None
            assert im.md5sum_extensions == [None, None]
            im.remove_data_from_disk()

            # Save to the archive
            im.save()
            for ext, fullpath, md5sum in zip(im.filepath_extensions,
                                             im.get_fullpath(nofile=True),
                                             im.md5sum_extensions):
                assert localmd5s[fullpath].hexdigest() == md5sum.hex

                with open( fullpath, "rb" ) as ifp:
                    m = hashlib.md5()
                    m.update( ifp.read() )
                    assert m.hexdigest() == localmd5s[fullpath].hexdigest()
                with open( os.path.join(archive_dir, im.filepath) + ext, 'rb' ) as ifp:
                    m = hashlib.md5()
                    m.update( ifp.read() )
                    assert m.hexdigest() == localmd5s[fullpath].hexdigest()

            # Make sure that we can download from the archive
            im.remove_data_from_disk()

            # using nofile=True will make sure the files are not downloaded from archive
            filenames = im.get_fullpath( nofile=True )
            for filename in filenames:
                with pytest.raises(FileNotFoundError):
                    ifp = open( filename, "rb" )
                    ifp.close()

            # this call to get_fullpath will also download the files to local storage
            newpaths = im.get_fullpath( nofile=False )
            assert newpaths == filenames
            for filename in filenames:
                with open( filename, "rb" ) as ifp:
                    m = hashlib.md5()
                    m.update( ifp.read() )
                    assert m.hexdigest() == localmd5s[filename].hexdigest()

            # Make sure that the md5sum is properly saved to the database
            im = session.merge(im)
            session.commit()
            with SmartSession() as differentsession:
                dbimage = differentsession.scalars(sa.select(Image).where(Image.id == im.id)).first()
                assert dbimage.md5sum is None

                filenames = dbimage.get_fullpath( nofile=True )
                for fullpath, md5sum in zip(filenames, dbimage.md5sum_extensions):
                    assert localmd5s[fullpath].hexdigest() == md5sum.hex

    finally:
        with SmartSession() as session:
            im = im.merge_all(session)
            exp = im.exposure
            im.delete_from_disk_and_database(session)

            if sa.inspect(exp).persistent:
                session.delete(exp)
                session.commit()
        test_config.set_value('storage.images.single_file', single_fileness)


def test_image_save_justheader( sim_image1 ):
    try:
        sim_image1.data = np.full( (64, 32), 0.125, dtype=np.float32 )
        sim_image1.flags = np.random.randint(0, 100, size=sim_image1.data.shape, dtype=np.uint16)
        sim_image1.weight = np.full( (64, 32), 4., dtype=np.float32 )

        archive = sim_image1.archive

        icl = ImageCleanup.save_image( sim_image1, archive=True )
        names = sim_image1.get_fullpath( download=False )
        assert names[0].endswith('.image.fits')
        assert names[1].endswith('.flags.fits')
        assert names[2].endswith('.weight.fits')

        # This is tested elsewhere, but for completeness make sure the
        # md5sum of the file on the archive is what's expected
        info = archive.get_info( pathlib.Path( names[0] ).relative_to( FileOnDiskMixin.local_path ) )
        assert uuid.UUID( info['md5sum'] ) == sim_image1.md5sum_extensions[0]

        sim_image1._header['ADDEDKW'] = 'This keyword was added'
        sim_image1.data = np.full( (64, 32), 0.5, dtype=np.float32 )
        sim_image1.weight = np.full( (64, 32), 2., dtype=np.float32 )

        origimmd5sum = sim_image1.md5sum_extensions[0]
        origwtmd5sum = sim_image1.md5sum_extensions[1]
        sim_image1.save( only_image=True, just_update_header=True )

        # Make sure the md5sum is different since the image is different, but that the weight is the same
        assert sim_image1.md5sum_extensions[0] != origimmd5sum
        assert sim_image1.md5sum_extensions[1] == origwtmd5sum

        # Make sure the archive has the new image
        info = archive.get_info( pathlib.Path( names[0] ).relative_to( FileOnDiskMixin.local_path ) )
        assert uuid.UUID( info['md5sum'] ) == sim_image1.md5sum_extensions[0]

        with fits.open( names[0] ) as hdul:
            assert hdul[0].header['ADDEDKW'] == 'This keyword was added'
            assert ( hdul[0].data == np.full( (64, 32), 0.125, dtype=np.float32 ) ).all()

        with fits.open( names[2] ) as hdul:
            assert ( hdul[0].data == np.full( (64, 32), 4., dtype=np.float32 ) ).all()

    finally:
        with SmartSession() as session:
            exp = session.merge(sim_image1.exposure)
            if sa.inspect(exp).persistent:
                session.delete(exp)
                session.commit()


def test_image_save_onlyimage( sim_image1 ):
    sim_image1.data = np.full( (64, 32), 0.125, dtype=np.float32 )
    sim_image1.flags = np.random.randint(0, 100, size=sim_image1.data.shape, dtype=np.uint16)
    sim_image1.weight = np.full( (64, 32), 4., dtype=np.float32 )

    icl = ImageCleanup.save_image( sim_image1, archive=False )
    names = sim_image1.get_fullpath( download=False )
    assert names[0].endswith('.image.fits')
    assert names[1].endswith('.flags.fits')
    assert names[2].endswith('.weight.fits')

    sim_image1._header['ADDEDTOO'] = 'An added keyword'
    sim_image1.data = np.full( (64, 32), 0.0625, dtype=np.float32 )

    with open( names[1], "w" ) as ofp:
        ofp.write( "Hello, world." )

    sim_image1.save( only_image=True, just_update_header=False, no_archive=True )
    with fits.open( names[0] ) as hdul:
        assert hdul[0].header['ADDEDTOO'] == 'An added keyword'
        assert ( hdul[0].data == np.full( (64, 32), 0.0625, dtype=np.float32 ) ).all()

    with open( names[1], "r" ) as ifp:
        assert ifp.read() == "Hello, world."


def test_image_enum_values(sim_image1):
    data_filename = None
    with SmartSession() as session:
        sim_image1 = sim_image1.merge_all(session)

        try:
            with pytest.raises(ValueError, match='ImageTypeConverter must be one of .* not foo'):
                sim_image1.type = 'foo'
                session.add(sim_image1)
                session.commit()
            session.rollback()

            # these should work
            for prepend in ["", "Com"]:
                for t in ["Sci", "Diff", "Bias", "Dark", "DomeFlat"]:
                    sim_image1.type = prepend+t
                    session.add(sim_image1)
                    session.commit()

            # should have an image with ComDomeFlat type
            assert sim_image1._type == 10  # see image_type_dict

            # make sure we can also select on this:
            images = session.scalars(sa.select(Image).where(Image.type == "ComDomeFlat")).all()
            assert sim_image1.id in [i.id for i in images]

            images = session.scalars(sa.select(Image).where(Image.type == "Sci")).all()
            assert sim_image1.id not in [i.id for i in images]

            # check the image format enum works as expected:
            with pytest.raises(ValueError, match='ImageFormatConverter must be one of .* not foo'):
                sim_image1.format = 'foo'
                session.add(sim_image1)
                session.commit()
            session.rollback()

            # these should work
            for f in ['fits', 'hdf5']:
                sim_image1.format = f
                session.add(sim_image1)
                session.commit()

            # should have an image with ComDomeFlat type
            assert sim_image1._format == 2  # see image_type_dict

            # make sure we can also select on this:
            images = session.scalars(sa.select(Image).where(Image.format == "hdf5")).all()
            assert sim_image1.id in [i.id for i in images]

            images = session.scalars(sa.select(Image).where(Image.format == "fits")).all()
            assert sim_image1.id not in [i.id for i in images]

        finally:
            sim_image1.remove_data_from_disk()
            if data_filename is not None and os.path.exists(data_filename):
                os.remove(data_filename)
                folder = os.path.dirname(data_filename)
                if len(os.listdir(folder)) == 0:
                    os.rmdir(folder)


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


def test_image_preproc_bitflag( sim_image1 ):

    with SmartSession() as session:
        im = session.merge(sim_image1)

        assert im.preproc_bitflag == 0
        im.preproc_bitflag |= string_to_bitflag( 'zero', image_preprocessing_inverse )
        assert im.preproc_bitflag == string_to_bitflag( 'zero', image_preprocessing_inverse )
        im.preproc_bitflag |= string_to_bitflag( 'flat', image_preprocessing_inverse )
        assert im.preproc_bitflag == string_to_bitflag( 'zero, flat', image_preprocessing_inverse )
        im.preproc_bitflag |= string_to_bitflag( 'flat, overscan', image_preprocessing_inverse )
        assert im.preproc_bitflag == string_to_bitflag( 'overscan, zero, flat', image_preprocessing_inverse )

        q = ( session.query( Image.filepath )
              .filter( Image.preproc_bitflag.op('&')(string_to_bitflag('zero', image_preprocessing_inverse) )
                       != 0 ) )
        assert (im.filepath,) in q.all()
        q = ( session.query( Image.filepath )
              .filter( Image.preproc_bitflag.op('&')(string_to_bitflag('zero,flat', image_preprocessing_inverse) )
                       !=0 ) )
        assert (im.filepath,) in q.all()
        q = ( session.query( Image.filepath )
              .filter( Image.preproc_bitflag.op('&')(string_to_bitflag('zero, flat', image_preprocessing_inverse ) )
                       == string_to_bitflag( 'flat, zero', image_preprocessing_inverse ) ) )
        assert (im.filepath,) in q.all()
        q = ( session.query( Image.filepath )
              .filter( Image.preproc_bitflag.op('&')(string_to_bitflag('fringe', image_preprocessing_inverse) )
                       !=0 ) )
        assert (im.filepath,) not in q.all()
        q = ( session.query( Image.filepath )
              .filter( Image.preproc_bitflag.op('&')(string_to_bitflag('fringe, overscan',
                                                                       image_preprocessing_inverse) )
                       == string_to_bitflag( 'overscan, fringe', image_preprocessing_inverse ) ) )
        assert q.count() == 0


def test_image_badness(sim_image1):

    with SmartSession() as session:
        sim_image1 = session.merge(sim_image1)

        # this is not a legit "badness" keyword...
        with pytest.raises(ValueError, match='Keyword "foo" not recognized'):
            sim_image1.badness = 'foo'

        # this is a legit keyword, but for cutouts, not for images
        with pytest.raises(ValueError, match='Keyword "Cosmic Ray" not recognized'):
            sim_image1.badness = 'Cosmic Ray'

        # this is a legit keyword, but for images, using no space and no capitalization
        sim_image1.badness = 'brightsky'

        # retrieving this keyword, we do get it capitalized and with a space:
        assert sim_image1.badness == 'Bright Sky'
        assert sim_image1.bitflag == 2 ** 5  # the bright sky bit is number 5

        # what happens when we add a second keyword?
        sim_image1.badness = 'brightsky, banding'
        assert sim_image1.bitflag == 2 ** 5 + 2 ** 1  # the bright sky bit is number 5, banding is number 1
        assert sim_image1.badness == 'Banding, Bright Sky'

        # now add a third keyword, but on the Exposure
        sim_image1.exposure.badness = 'saturation'
        session.add(sim_image1)
        session.commit()

        # a manual way to propagate bitflags downstream
        sim_image1.exposure.update_downstream_badness(session)  # make sure the downstreams get the new badness
        session.commit()
        assert sim_image1.bitflag == 2 ** 5 + 2 ** 3 + 2 ** 1  # saturation bit is 3
        assert sim_image1.badness == 'Banding, Saturation, Bright Sky'

        # adding the same keyword on the exposure and the image makes no difference
        sim_image1.exposure.badness = 'banding'
        sim_image1.exposure.update_downstream_badness(session)  # make sure the downstreams get the new badness
        session.commit()
        assert sim_image1.bitflag == 2 ** 5 + 2 ** 1
        assert sim_image1.badness == 'Banding, Bright Sky'

        # try appending keywords to the image
        sim_image1.append_badness('shaking')
        assert sim_image1.bitflag == 2 ** 5 + 2 ** 2 + 2 ** 1  # shaking bit is 2
        assert sim_image1.badness == 'Banding, Shaking, Bright Sky'


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
            sim_image2.badness = 'brightsky'
            assert sim_image2.badness == 'Bright Sky'
            assert sim_image2.bitflag == 2 ** 5
            session.commit()

            # note that this image is not directly bad, but the exposure has banding
            sim_image3.exposure.badness = 'banding'
            sim_image3.exposure.update_downstream_badness(session)
            session.commit()

            assert sim_image3.badness == 'Banding'
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
            images.append(sim_image4)
            sim_image4 = session.merge(sim_image4)
            session.commit()

            assert sim_image4.id is not None
            assert sim_image4.ref_image == sim_image2
            assert sim_image4.new_image == sim_image3

            # check that badness is loaded correctly from both parents
            assert sim_image4.badness == 'Banding, Bright Sky'
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
            assert sim_image4.badness == 'Banding, Saturation, Bright Sky'
            assert sim_image4._bitflag == 2 ** 3  # only this bit is from the image itself

            # make a new subtraction:
            sim_image7 = Image.from_ref_and_new(sim_image6, sim_image5)
            sim_image7.provenance = provenance_extra
            cleanups.append(ImageCleanup.save_image(sim_image7))
            images.append(sim_image7)
            sim_image7 = session.merge(sim_image7)
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
            # as a reminder, sim_image2 has Bright Sky (5),
            # sim_image3's exposure has banding (1), while
            # sim_image4 has Saturation (3).

            # make a coadded image (without including the subtraction sim_image4):
            sim_image8 = Image.from_images([sim_image1, sim_image2, sim_image3, sim_image5, sim_image6])
            sim_image8.provenance = provenance_extra
            cleanups.append(ImageCleanup.save_image(sim_image8))
            images.append(sim_image8)
            sim_image8 = session.merge(sim_image8)
            session.commit()

            assert sim_image8.badness == 'Banding, Bright Sky'
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

            assert sim_image8.badness == 'Banding, Saturation, Bright Sky'
            assert sim_image8.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 5  # this should be 42

            # does this work in queries (i.e., using the bitflag hybrid expression)?
            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert sim_image8.id in [i.id for i in bad_images]
            bad_coadd = session.scalars(sa.select(Image).where(Image.bitflag == 42)).all()
            assert sim_image8.id in [i.id for i in bad_coadd]

            # try to add some badness to one of the underlying exposures
            sim_image1.exposure.badness = 'Shaking'
            session.add(sim_image1)
            sim_image1.exposure.update_downstream_badness(session)
            session.commit()

            assert 'Shaking' in sim_image1.badness
            assert 'Shaking' in sim_image8.badness

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


def test_image_from_exposure(sim_exposure1, provenance_base):
    sim_exposure1.update_instrument()

    # demo instrument only has one section
    with pytest.raises(ValueError, match='section_id must be 0 for this instrument.'):
        _ = Image.from_exposure(sim_exposure1, section_id=1)

    im = Image.from_exposure(sim_exposure1, section_id=0)
    assert im.section_id == 0
    assert im.mjd == sim_exposure1.mjd
    assert im.end_mjd == sim_exposure1.end_mjd
    assert im.exp_time == sim_exposure1.exp_time
    assert im.end_mjd == im.mjd + im.exp_time / 86400
    assert im.filter == sim_exposure1.filter
    assert im.instrument == sim_exposure1.instrument
    assert im.telescope == sim_exposure1.telescope
    assert im.project == sim_exposure1.project
    assert im.target == sim_exposure1.target
    assert not im.is_coadd
    assert not im.is_sub
    assert im.id is None  # need to commit to get IDs
    assert im.upstream_images == []
    assert im.filepath is None  # need to save file to generate a filename
    assert np.array_equal(im.raw_data, sim_exposure1.data[0])
    assert im.data is None
    assert im.flags is None
    assert im.weight is None
    assert im.nofile  # images are made without a file by default

    # TODO: add check for loading the header after we make a demo header maker
    # TODO: what should the RA/Dec be for an image that cuts out from an exposure?

    im_id = None
    try:
        with SmartSession() as session:
            with pytest.raises(IntegrityError, match='null value in column .* of relation "images"'):
                session.merge(im)
                session.commit()
            session.rollback()

            # must add the provenance!
            im.provenance = provenance_base
            with pytest.raises(IntegrityError, match='null value in column "filepath" of relation "images"'):
                im = session.merge(im)
                session.commit()
            session.rollback()

            _ = ImageCleanup.save_image(im)  # this will add the filepath and md5 sum!

            session.add(im)
            session.commit()

            assert im.id is not None
            assert im.provenance_id is not None
            assert im.provenance_id == provenance_base.id
            assert im.exposure_id is not None
            assert im.exposure_id == sim_exposure1.id

    finally:
        if im_id is not None:
            with SmartSession() as session:
                im.delete_from_disk_and_database(commit=True, session=session)


def test_image_from_exposure_filter_array(sim_exposure_filter_array):
    sim_exposure_filter_array.update_instrument()
    im = Image.from_exposure(sim_exposure_filter_array, section_id=0)
    filt = sim_exposure_filter_array.filter_array[0]
    assert im.filter == filt


def test_image_with_multiple_upstreams(sim_exposure1, sim_exposure2, provenance_base):
    sim_exposure1.update_instrument()
    sim_exposure2.update_instrument()

    # make sure exposures are in chronological order...
    if sim_exposure1.mjd > sim_exposure2.mjd:
        sim_exposure1, sim_exposure2 = sim_exposure2, sim_exposure1

    # get a couple of images from exposure objects
    im1 = Image.from_exposure(sim_exposure1, section_id=0)
    im2 = Image.from_exposure(sim_exposure2, section_id=0)
    im2.filter = im1.filter
    im2.target = im1.target

    im1.provenance = provenance_base
    _1 = ImageCleanup.save_image(im1)

    im2.provenance = provenance_base
    _2 = ImageCleanup.save_image(im2)

    # make a coadd image from the two
    im = Image.from_images([im1, im2])
    im.provenance = provenance_base
    _3 = ImageCleanup.save_image(im)

    try:
        im_id = None
        im1_id = None
        im2_id = None
        with SmartSession() as session:
            im = im.merge_all( session )
            im1 = im1.merge_all( session )
            im2 = im2.merge_all( session )
            sim_exposure1 = session.merge(sim_exposure1)
            sim_exposure2 = session.merge(sim_exposure2)

            session.commit()

            im_id = im.id
            assert im_id is not None
            assert im.exposure_id is None
            assert im.upstream_images == [im1, im2]
            assert np.isclose(im.mid_mjd, (im1.mjd + im2.mjd) / 2)

            # make sure source images are pulled into the database too
            im1_id = im1.id
            assert im1_id is not None
            assert im1.exposure_id is not None
            assert im1.exposure_id == sim_exposure1.id
            assert im1.upstream_images == []

            im2_id = im2.id
            assert im2_id is not None
            assert im2.exposure_id is not None
            assert im2.exposure_id == sim_exposure2.id
            assert im2.upstream_images == []

    finally:  # make sure to clean up all images
        for image in [im, im1, im2]:
            if image is not None:
                with SmartSession() as session:
                    image.delete_from_disk_and_database(session=session, commit=False)
                    session.commit()


def test_image_subtraction(sim_exposure1, sim_exposure2, provenance_base):
    sim_exposure1.update_instrument()
    sim_exposure2.update_instrument()

    # make sure exposures are in chronological order...
    if sim_exposure1.mjd > sim_exposure2.mjd:
        sim_exposure1, sim_exposure2 = sim_exposure2, sim_exposure1

    # get a couple of images from exposure objects
    im1 = Image.from_exposure(sim_exposure1, section_id=0)
    im2 = Image.from_exposure(sim_exposure2, section_id=0)
    im2.filter = im1.filter
    im2.target = im1.target

    im1.provenance = provenance_base
    _1 = ImageCleanup.save_image(im1)
    im2.provenance = provenance_base
    _2 = ImageCleanup.save_image(im2)

    # make a coadd image from the two
    im = Image.from_ref_and_new(im1, im2)
    im.provenance = provenance_base
    _3 = ImageCleanup.save_image(im)

    try:
        im_id = None
        im1_id = None
        im2_id = None
        with SmartSession() as session:
            im = im.merge_all( session )
            im1 = im1.merge_all( session )
            im2 = im2.merge_all( session )
            sim_exposure1 = session.merge(sim_exposure1)
            sim_exposure2 = session.merge(sim_exposure2)

            session.commit()

            im_id = im.id
            assert im_id is not None
            assert im.exposure_id is None
            assert im.ref_image == im1
            assert im.ref_image.id == im1.id
            assert im.new_image == im2
            assert im.new_image.id == im2.id
            assert im.mjd == im2.mjd
            assert im.exp_time == im2.exp_time

            # make sure source images are pulled into the database too
            im1_id = im1.id
            assert im1_id is not None
            assert im1.exposure_id is not None
            assert im1.exposure_id == sim_exposure1.id
            assert im1.upstream_images == []

            im2_id = im2.id
            assert im2_id is not None
            assert im2.exposure_id is not None
            assert im2.exposure_id == sim_exposure2.id
            assert im2.upstream_images == []

    finally:  # make sure to clean up all images
        for id_ in [im_id, im1_id, im2_id]:
            if id_ is not None:
                with SmartSession() as session:
                    im = session.scalars(sa.select(Image).where(Image.id == id_)).first()
                    session.delete(im)
                    session.commit()


def test_image_filename_conventions(sim_image1, test_config):

    # sim_image1 was saved using the naming convention in the config file
    assert re.search(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}\.image\.fits', sim_image1.get_fullpath()[0])
    for f in sim_image1.get_fullpath(as_list=True):
        assert os.path.isfile(f)
        os.remove(f)
    original_filepath = sim_image1.filepath

    # try to set the name convention to None, to load the default hard-coded one
    convention = test_config.value('storage.images.name_convention')
    try:
        test_config.set_value('storage.images.name_convention', None)
        sim_image1.save( no_archive=True )
        assert re.search(r'Demo_\d{8}_\d{6}_\d+_.+_.{6}\.image\.fits', sim_image1.get_fullpath()[0])
        for f in sim_image1.get_fullpath(as_list=True):
            assert os.path.isfile(f)
            os.remove(f)
            folder = os.path.dirname(f)
            if len(os.listdir(folder)) == 0:
                os.rmdir(folder)

        new_convention = '{ra_int:03d}/foo_{date}_{time}_{section_id_int:02d}_{filter}'
        test_config.set_value('storage.images.name_convention', new_convention)
        # invent_filepath will try to use the existing filepath value so we clear it first
        sim_image1.filepath = None
        sim_image1.save( no_archive=True )
        assert re.search(r'\d{3}/foo_\d{8}_\d{6}_\d{2}_.\.image\.fits', sim_image1.get_fullpath()[0])
        for f in sim_image1.get_fullpath(as_list=True):
            assert os.path.isfile(f)
            os.remove(f)
            folder = os.path.dirname(f)
            if len(os.listdir(folder)) == 0:
                os.rmdir(folder)

        new_convention = 'bar_{date}_{time}_{section_id_int:02d}_{ra_int_h:02d}{dec_int:+03d}'
        test_config.set_value('storage.images.name_convention', new_convention)
        sim_image1.filepath = None
        sim_image1.save( no_archive=True )
        assert re.search(r'bar_\d{8}_\d{6}_\d{2}_\d{2}[+-]\d{2}\.image\.fits', sim_image1.get_fullpath()[0])
        for f in sim_image1.get_fullpath(as_list=True):
            assert os.path.isfile(f)
            os.remove(f)
            folder = os.path.dirname(f)
            if len(os.listdir(folder)) == 0:
                os.rmdir(folder)

    finally:  # return to the original convention
        test_config.set_value('storage.images.name_convention', convention)
        sim_image1.filepath = original_filepath  # this will allow the image to delete itself in the teardown


def test_image_multifile(sim_image_uncommitted, provenance_base, test_config):
    im = sim_image_uncommitted
    im.data = np.float32(im.raw_data)
    im.flags = np.random.randint(0, 100, size=im.raw_data.shape, dtype=np.uint32)
    im.weight = None
    im.provenance = provenance_base

    single_fileness = test_config.value('storage.images.single_file')  # store initial value

    try:
        # first use single file
        test_config.set_value('storage.images.single_file', True)
        im.save( no_archive=True )

        assert re.match(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}\.fits', im.filepath)

        files = im.get_fullpath(as_list=True)
        assert len(files) == 1
        assert os.path.isfile(files[0])

        with fits.open(files[0]) as hdul:
            assert len(hdul) == 3  # primary plus one for image data and one for flags
            assert hdul[0].header['NAXIS'] == 0
            assert isinstance(hdul[0], fits.PrimaryHDU)
            assert hdul[1].header['NAXIS'] == 2
            assert isinstance(hdul[1], fits.ImageHDU)
            assert np.array_equal(hdul[1].data, im.data)
            assert hdul[2].header['NAXIS'] == 2
            assert isinstance(hdul[2], fits.ImageHDU)
            assert np.array_equal(hdul[2].data, im.flags)

        for f in im.get_fullpath(as_list=True):
            os.remove(f)

        # now test multiple files
        test_config.set_value('storage.images.single_file', False)
        im.filepath = None
        im.save( no_archive=True )

        assert re.match(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}', im.filepath)
        fullnames = im.get_fullpath(as_list=True)

        assert len(fullnames) == 2

        assert os.path.isfile(fullnames[0])
        assert re.search(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}\.image\.fits', fullnames[0])
        with fits.open(fullnames[0]) as hdul:
            assert len(hdul) == 1  # image data is saved on the primary HDU
            assert hdul[0].header['NAXIS'] == 2
            assert isinstance(hdul[0], fits.PrimaryHDU)
            assert np.array_equal(hdul[0].data, im.data)

        assert os.path.isfile(fullnames[1])
        assert re.search(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}\.flags\.fits', fullnames[1])
        with fits.open(fullnames[1]) as hdul:
            assert len(hdul) == 1
            assert hdul[0].header['NAXIS'] == 2
            assert isinstance(hdul[0], fits.PrimaryHDU)
            assert np.array_equal(hdul[0].data, im.flags)

    finally:
        with SmartSession() as session:
            im = session.merge(im)
            exp = im.exposure
            im.delete_from_disk_and_database(session=session, commit=False)

            if exp is not None and sa.inspect(exp).persistent:
                session.delete(exp)

            session.commit()

        test_config.set_value('storage.images.single_file', single_fileness)


def test_image_products_are_deleted(ptf_datastore, data_dir, archive):
    ds = ptf_datastore  # shorthand

    # check the datastore comes with all these objects
    assert isinstance(ds.image, Image)
    assert isinstance(ds.psf, PSF)
    assert isinstance(ds.sources, SourceList)
    assert isinstance(ds.wcs, WorldCoordinates)
    assert isinstance(ds.zp, ZeroPoint)
    # TODO: add more data types?

    # make sure the image has the same objects
    im = ds.image
    assert im.psf == ds.psf
    assert im.sources == ds.sources
    assert im.wcs == ds.wcs
    assert im.zp == ds.zp

    # make sure the files are there
    local_files = []
    archive_files = []
    for obj in [im, im.psf, im.sources]:  # TODO: add WCS when it becomes a FileOnDiskMixin
        for file in obj.get_fullpath(as_list=True):
            archive_file = file[len(obj.local_path)+1:]  # grap the end of the path only
            archive_file = os.path.join(archive.test_folder_path, archive_file)  # prepend the archive path
            assert os.path.isfile(file)
            assert os.path.isfile(archive_file)
            local_files.append(file)
            archive_files.append(archive_file)

    # delete the image and all its downstreams
    im.delete_from_disk_and_database(remove_folders=True, remove_downstream_data=True)

    # make sure the files are gone
    for file in local_files:
        assert not os.path.isfile(file)

    for file in archive_files:
        assert not os.path.isfile(file)
