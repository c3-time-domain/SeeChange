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

from tests.conftest import ImageCleanup
import util.config as config
import util.radec
from models.base import SmartSession, FileOnDiskMixin
from models.exposure import Exposure
from models.image import Image
from models.instrument import Instrument
from models.references import Reference
from models.enums_and_bitflags import image_preprocessing_inverse, string_to_bitflag

# Have to have this here; otherwise, decam.py never gets loaded, and
# DECam never gets added to the global instrument.INSTRUMENT_INSTANCE_CACHE
#
# There must be a better solution.  Do we just need to stuff all of the instruments
# in the same file?  Or, should we rerun register_all_instruments() at the bottom
# of every instrument's .py file?
# import models.decam


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


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
        'section_id': 1,
    }

    added = {}

    # use non-capturing groups to extract the column name from the error message
    expr = r'(?:null value in column )(".*")(?: of relation "images" violates not-null constraint)'

    try:
        im_id = None  # make sure to delete the image if it is added to DB

        # md5sum is spoofed as we don't have this file saved to archive
        image = Image(f"Demo_test_{rnd_str(5)}.fits", md5sum=uuid.uuid4(), nofile=True)
        with SmartSession() as session:
            for i in range(len(required)):
                image.recursive_merge( session )
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


def test_image_must_have_md5(demo_image, provenance_base):
    assert demo_image.md5sum is None
    assert demo_image.md5sum_extensions is None

    demo_image.provenance = provenance_base
    _ = ImageCleanup.save_image(demo_image, archive=False)

    demo_image.md5sum = None
    with SmartSession() as session:
        demo_image.recursive_merge(session)
        with pytest.raises(IntegrityError, match='violates check constraint'):
            session.add(demo_image)
            session.commit()
        session.rollback()

        # adding md5sums should fix this problem
        _2 = ImageCleanup.save_image(demo_image, archive=True)
        session.add(demo_image)
        session.commit()


def test_image_archive_singlefile(demo_image, provenance_base, archive):
    demo_image.data = np.float32( demo_image.raw_data )
    demo_image.flags = np.random.randint(0, 100, size=demo_image.raw_data.shape, dtype=np.uint16)
    demo_image.provenance = provenance_base

    cfg = config.Config.get()
    archivebase = f"{os.getenv('SEECHANGE_TEST_ARCHIVE_DIR')}/{cfg.value('archive.path_base')}"
    single_fileness = cfg.value( 'storage.images.single_file' )

    try:
        with SmartSession() as session:
            # Do single file first
            cfg.set_value( 'storage.images.single_file', True )
            demo_image.exposure.recursive_merge(session)  # make sure the exposure and provenance/code versions merge
            # Make sure that the archive is *not* written when we tell it not to.
            demo_image.save( no_archive=True )
            assert demo_image.md5sum is None
            with pytest.raises(FileNotFoundError):
                ifp = open( f'{archivebase}{demo_image.filepath}', 'rb' )
                ifp.close()
            demo_image.remove_data_from_disk()

            # Save to the archive, make sure it all worked
            demo_image.save()
            localmd5 = hashlib.md5()
            with open( demo_image.get_fullpath( nofile=False ), 'rb' ) as ifp:
                localmd5.update( ifp.read() )
            assert localmd5.hexdigest() == demo_image.md5sum.hex
            archivemd5 = hashlib.md5()
            with open( f'{archivebase}{demo_image.filepath}', 'rb' ) as ifp:
                archivemd5.update( ifp.read() )
            assert archivemd5.hexdigest() == demo_image.md5sum.hex

            # Make sure that we can download from the archive
            demo_image.remove_data_from_disk()
            with pytest.raises(FileNotFoundError):
                assert isinstance( demo_image.get_fullpath( nofile=True ), str )
                ifp = open( demo_image.get_fullpath( nofile=True ), "rb" )
                ifp.close()
            p = demo_image.get_fullpath( nofile=False )
            localmd5 = hashlib.md5()
            with open( demo_image.get_fullpath( nofile=False ), 'rb' ) as ifp:
                localmd5.update( ifp.read() )
            assert localmd5.hexdigest() == demo_image.md5sum.hex

            # Make sure that the md5sum is properly saved to the database
            demo_image.provenance = session.merge( demo_image.provenance )
            session.add( demo_image )
            session.commit()
            with SmartSession() as differentsession:
                dbimage = differentsession.query(Image).filter(Image.id==demo_image.id)[0]
                assert dbimage.md5sum.hex == demo_image.md5sum.hex

            # Make sure we can purge the archive
            demo_image.delete_from_disk_and_database(session=session, commit=True)
            with pytest.raises(FileNotFoundError):
                ifp = open( f'{archivebase}{demo_image.filepath}', 'rb' )
                ifp.close()
            assert demo_image.md5sum is None

    finally:
        cfg.set_value( 'storage.images.single_file', single_fileness )


def test_image_archive_multifile(exposure, demo_image, provenance_base, archive):
    demo_image.data = np.float32( demo_image.raw_data )
    demo_image.flags = np.random.randint(0, 100, size=demo_image.raw_data.shape, dtype=np.uint16)

    cfg = config.Config.get()
    archivebase = f"{os.getenv('SEECHANGE_TEST_ARCHIVE_DIR')}/{cfg.value('archive.path_base')}"
    single_fileness = cfg.value( 'storage.images.single_file' )

    try:
        with SmartSession() as session:
            # First, work around SQLAlchemy
            demo_image.provenance = session.merge( provenance_base )
            exposure.provenance = session.merge( exposure.provenance )

            # Now do multiple images
            cfg.set_value( 'storage.images.single_file', False )

            # Make sure that the archive is not written when we tell it not to
            demo_image.save( no_archive=True )
            localmd5s = {}
            assert len(demo_image.get_fullpath(nofile=True)) == 2
            for fullpath in demo_image.get_fullpath(nofile=True):
                localmd5s[fullpath] = hashlib.md5()
                with open(fullpath, "rb") as ifp:
                    localmd5s[fullpath].update(ifp.read())
            assert demo_image.md5sum is None
            assert demo_image.md5sum_extensions == [None, None]
            demo_image.remove_data_from_disk()

            # Save to the archive
            demo_image.save()
            for ext, fullpath, md5sum in zip(demo_image.filepath_extensions,
                                             demo_image.get_fullpath(nofile=True),
                                             demo_image.md5sum_extensions):
                assert localmd5s[fullpath].hexdigest() == md5sum.hex

                with open( fullpath, "rb" ) as ifp:
                    m = hashlib.md5()
                    m.update( ifp.read() )
                    assert m.hexdigest() == localmd5s[fullpath].hexdigest()
                with open( f'{archivebase}{demo_image.filepath}{ext}', 'rb' ) as ifp:
                    m = hashlib.md5()
                    m.update( ifp.read() )
                    assert m.hexdigest() == localmd5s[fullpath].hexdigest()

            # Make sure that we can download from the archive
            demo_image.remove_data_from_disk()

            # using nofile=True will make sure the files are not downloaded from archive
            filenames = demo_image.get_fullpath( nofile=True )
            for filename in filenames:
                with pytest.raises(FileNotFoundError):
                    ifp = open( filename, "rb" )
                    ifp.close()

            # this call to get_fullpath will also download the files to local storage
            newpaths = demo_image.get_fullpath( nofile=False )
            assert newpaths == filenames
            for filename in filenames:
                with open( filename, "rb" ) as ifp:
                    m = hashlib.md5()
                    m.update( ifp.read() )
                    assert m.hexdigest() == localmd5s[filename].hexdigest()

            # Make sure that the md5sum is properly saved to the database
            session.add( demo_image )
            session.commit()
            with SmartSession() as differentsession:
                dbimage = differentsession.scalars(sa.select(Image).where(Image.id == demo_image.id)).first()
                assert dbimage.md5sum is None

                filenames = dbimage.get_fullpath( nofile=True )
                for fullpath, md5sum in zip(filenames, dbimage.md5sum_extensions):
                    assert localmd5s[fullpath].hexdigest() == md5sum.hex

    finally:
        cfg.set_value( 'storage.images.single_file', single_fileness )


def test_image_save_justheader( exposure, demo_image, provenance_base ):
    demo_image.provenance = provenance_base
    demo_image.data = np.full( (64, 32), 0.125, dtype=np.float32 )
    demo_image._weight = np.full( (64, 32), 4., dtype=np.float32 )

    archive = demo_image.archive

    icl = ImageCleanup.save_image( demo_image, archive=True )
    names = demo_image.get_fullpath( download=False )
    assert names[0][-11:] == '.image.fits'
    assert names[1][-12:] == '.weight.fits'

    # This is tested elsewhere, but for completeness make sure the
    # md5sum of the file on the archive is what's expected
    info = archive.get_info( pathlib.Path( names[0] ).relative_to( FileOnDiskMixin.local_path ) )
    assert uuid.UUID( info['md5sum'] ) == demo_image.md5sum_extensions[0]

    demo_image._raw_header['ADDEDKW'] = 'This keyword was added'
    demo_image.data = np.full( (64, 32), 0.5, dtype=np.float32 )
    demo_image._weight = np.full( (64, 32), 2., dtype=np.float32 )

    origimmd5sum = demo_image.md5sum_extensions[0]
    origwtmd5sum = demo_image.md5sum_extensions[1]
    demo_image.save( only_image=True, just_update_header=True )

    # Make sure the md5sum is different since the image is different, but that the weight is the same
    assert demo_image.md5sum_extensions[0] != origimmd5sum
    assert demo_image.md5sum_extensions[1] == origwtmd5sum

    # Make sure the archive has the new image
    info = archive.get_info( pathlib.Path( names[0] ).relative_to( FileOnDiskMixin.local_path ) )
    assert uuid.UUID( info['md5sum'] ) == demo_image.md5sum_extensions[0]

    with fits.open( names[0] ) as hdul:
        assert hdul[0].header['ADDEDKW'] == 'This keyword was added'
        assert ( hdul[0].data == np.full( (64, 32), 0.125, dtype=np.float32 ) ).all()

    with fits.open( names[1] ) as hdul:
        assert ( hdul[0].data == np.full( (64, 32), 4., dtype=np.float32 ) ).all()


def test_image_save_onlyimage( exposure, demo_image, provenance_base ):
    demo_image.provenance = provenance_base
    demo_image.data = np.full( (64, 32), 0.125, dtype=np.float32 )
    demo_image._weight = np.full( (64, 32), 4., dtype=np.float32 )

    icl = ImageCleanup.save_image( demo_image, archive=False )
    names = demo_image.get_fullpath( download=False )
    assert names[0][-11:] == '.image.fits'
    assert names[1][-12:] == '.weight.fits'

    demo_image._raw_header['ADDEDTOO'] = 'An added keyword'
    demo_image.data = np.full( (64, 32), 0.0625, dtype=np.float32 )

    with open( names[1], "w" ) as ofp:
        ofp.write( "Hello, world." )

    demo_image.save( only_image=True, just_update_header=False, no_archive=True )
    with fits.open( names[0] ) as hdul:
        assert hdul[0].header['ADDEDTOO'] == 'An added keyword'
        assert ( hdul[0].data == np.full( (64, 32), 0.0625, dtype=np.float32 ) ).all()

    with open( names[1], "r" ) as ifp:
        assert ifp.read() == "Hello, world."


def test_image_enum_values(exposure, demo_image, provenance_base):
    data_filename = None
    with SmartSession() as session:
        demo_image.provenance = session.merge( provenance_base )
        exposure.provenance = session.merge( exposure.provenance )
        with pytest.raises(RuntimeError, match='The image data is not loaded. Cannot save.'):
            demo_image.save( no_archive=True )

        _ = ImageCleanup.save_image(demo_image, archive=True)
        data_filename = demo_image.get_fullpath(as_list=True)[0]
        assert os.path.exists(data_filename)

        try:
            with pytest.raises(ValueError, match='ImageTypeConverter must be one of .* not foo'):
                demo_image.type = 'foo'
                session.add(demo_image)
                session.commit()
            session.rollback()

            # these should work
            for prepend in ["", "Com"]:
                for t in ["Sci", "Diff", "Bias", "Dark", "DomeFlat"]:
                    demo_image.type = prepend+t
                    session.add(demo_image)
                    session.commit()

            # should have an image with ComDomeFlat type
            assert demo_image._type == 10  # see image_type_dict

            # make sure we can also select on this:
            images = session.scalars(sa.select(Image).where(Image.type == "ComDomeFlat")).all()
            assert demo_image.id in [i.id for i in images]

            images = session.scalars(sa.select(Image).where(Image.type == "Sci")).all()
            assert demo_image.id not in [i.id for i in images]

            # check the image format enum works as expected:
            with pytest.raises(ValueError, match='ImageFormatConverter must be one of .* not foo'):
                demo_image.format = 'foo'
                session.add(demo_image)
                session.commit()
            session.rollback()

            # these should work
            for f in ['fits', 'hdf5']:
                demo_image.format = f
                session.add(demo_image)
                session.commit()

            # should have an image with ComDomeFlat type
            assert demo_image._format == 2  # see image_type_dict

            # make sure we can also select on this:
            images = session.scalars(sa.select(Image).where(Image.format == "hdf5")).all()
            assert demo_image.id in [i.id for i in images]

            images = session.scalars(sa.select(Image).where(Image.format == "fits")).all()
            assert demo_image.id not in [i.id for i in images]

        finally:
            demo_image.remove_data_from_disk()
            if data_filename is not None and os.path.exists(data_filename):
                os.remove(data_filename)
                folder = os.path.dirname(data_filename)
                if len(os.listdir(folder)) == 0:
                    os.rmdir(folder)


def test_image_upstreams_downstreams(demo_image, sim_reference, provenance_base, provenance_extra):
    with SmartSession() as session:
        demo_image.provenance = provenance_base
        demo_image = demo_image.recursive_merge(session)

        # make sure the new image matches the reference in all these attributes
        demo_image.filter = sim_reference.filter
        demo_image.target = sim_reference.target
        demo_image.section_id = sim_reference.section_id

        # save and delete at the end
        cleanup1 = ImageCleanup.save_image(demo_image)
        session.add(demo_image)

        sim_reference = session.merge(sim_reference)
        new = Image.from_new_and_ref(demo_image, sim_reference.image)
        new.provenance = provenance_extra
        new = new.recursive_merge(session)

        # save and delete at the end
        cleanup2 = ImageCleanup.save_image(new)

        session.add(new)
        session.commit()

    # new make sure a new session can find all the upstreams/downstreams
    with SmartSession() as session:
        # check the upstreams/downstreams for the new image
        upstream_ids = [u.id for u in new.get_upstreams(session=session)]
        assert demo_image.id in upstream_ids
        assert sim_reference.image_id in upstream_ids
        downstream_ids = [d.id for d in new.get_downstreams(session=session)]
        assert len(downstream_ids) == 0

        upstream_ids = [u.id for u in demo_image.get_upstreams(session=session)]
        assert [demo_image.exposure_id] == upstream_ids
        downstream_ids = [d.id for d in demo_image.get_downstreams(session=session)]
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


def test_image_preproc_bitflag( demo_image, provenance_base ):

    with SmartSession() as session:
        demo_image.provenance = provenance_base
        demo_image.filepath = demo_image.invent_filepath()
        # Spoof an md5sum so we can commit this to the database without saving actual data
        demo_image.md5sum = uuid.uuid4()
        im = demo_image.recursive_merge( session )
        session.add( im )
        # Need to do this for the defaults to get set
        # It will be removed from the database in
        # demo_image teardown
        session.commit()

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


def test_image_badness(demo_image, provenance_base):

    with SmartSession() as session:
        demo_image.provenance = provenance_base
        cleanup = ImageCleanup.save_image(demo_image)
        demo_image = demo_image.recursive_merge(session)
        session.add(demo_image)
        session.commit()

        # this is not a legit "badness" keyword...
        with pytest.raises(ValueError, match='Keyword "foo" not recognized'):
            demo_image.badness = 'foo'

        # this is a legit keyword, but for cutouts, not for images
        with pytest.raises(ValueError, match='Keyword "Cosmic Ray" not recognized'):
            demo_image.badness = 'Cosmic Ray'

        # this is a legit keyword, but for images, using no space and no capitalization
        demo_image.badness = 'brightsky'

        # retrieving this keyword, we do get it capitalized and with a space:
        assert demo_image.badness == 'Bright Sky'
        assert demo_image.bitflag == 2 ** 5  # the bright sky bit is number 5

        # what happens when we add a second keyword?
        demo_image.badness = 'brightsky, banding'
        assert demo_image.bitflag == 2 ** 5 + 2 ** 1  # the bright sky bit is number 5, banding is number 1
        assert demo_image.badness == 'Banding, Bright Sky'

        # now add a third keyword, but on the Exposure
        demo_image.exposure.badness = 'saturation'
        session.add(demo_image)
        session.commit()

        # a manual way to propagate bitflags downstream
        demo_image.exposure.update_downstream_badness(session)  # make sure the downstreams get the new badness
        session.commit()
        assert demo_image.bitflag == 2 ** 5 + 2 ** 3 + 2 ** 1  # saturation bit is 3
        assert demo_image.badness == 'Banding, Saturation, Bright Sky'

        # adding the same keyword on the exposure and the image makes no difference
        demo_image.exposure.badness = 'banding'
        demo_image.exposure.update_downstream_badness(session)  # make sure the downstreams get the new badness
        session.commit()
        assert demo_image.bitflag == 2 ** 5 + 2 ** 1
        assert demo_image.badness == 'Banding, Bright Sky'

        # try appending keywords to the image
        demo_image.append_badness('shaking')
        assert demo_image.bitflag == 2 ** 5 + 2 ** 2 + 2 ** 1  # shaking bit is 2
        assert demo_image.badness == 'Banding, Shaking, Bright Sky'


def test_multiple_images_badness(
        demo_image,
        demo_image2,
        demo_image3,
        demo_image5,
        demo_image6,
        provenance_base,
        provenance_extra
):
    try:
        images = [demo_image, demo_image2, demo_image3, demo_image5, demo_image6]
        cleanups = []
        filter = 'g'
        target = str(uuid.uuid4())
        project = 'test project'
        with SmartSession() as session:
            for im in images:
                im.filter = filter
                im.target = target
                im.project = project
                im.provenance = provenance_base
                im = im.recursive_merge(session)
                cleanups.append(ImageCleanup.save_image(im))
                session.add(im)
            session.commit()

            # the image itself is marked bad because of bright sky
            demo_image2.badness = 'brightsky'
            assert demo_image2.badness == 'Bright Sky'
            assert demo_image2.bitflag == 2 ** 5
            session.commit()

            # note that this image is not directly bad, but the exposure has banding
            demo_image3.exposure.badness = 'banding'
            demo_image3.exposure.update_downstream_badness(session)
            session.commit()

            assert demo_image3.badness == 'Banding'
            assert demo_image._bitflag == 0  # the exposure is bad!
            assert demo_image3.bitflag == 2 ** 1
            session.commit()

            # find the images that are good vs bad
            good_images = session.scalars(sa.select(Image).where(Image.bitflag == 0)).all()
            assert demo_image.id in [i.id for i in good_images]

            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert demo_image2.id in [i.id for i in bad_images]
            assert demo_image3.id in [i.id for i in bad_images]

            # make an image from the two bad exposures using subtraction
            demo_image4 = Image.from_new_and_ref(demo_image3, demo_image2)
            demo_image4.provenance = provenance_extra
            demo_image4.provenance.upstreams = demo_image4.get_upstream_provenances()
            cleanups.append(ImageCleanup.save_image(demo_image4))
            images.append(demo_image4)
            demo_image4 = demo_image4.recursive_merge(session)
            session.add(demo_image4)
            session.commit()

            assert demo_image4.id is not None
            assert demo_image4.ref_image_id == demo_image2.id
            assert demo_image4.new_image_id == demo_image3.id

            # check that badness is loaded correctly from both parents
            assert demo_image4.badness == 'Banding, Bright Sky'
            assert demo_image4._bitflag == 0  # the image itself is not flagged
            assert demo_image4.bitflag == 2 ** 1 + 2 ** 5

            # check that filtering on this value gives the right bitflag
            bad_images = session.scalars(sa.select(Image).where(Image.bitflag == 2 ** 1 + 2 ** 5)).all()
            assert demo_image4.id in [i.id for i in bad_images]
            assert demo_image3.id not in [i.id for i in bad_images]
            assert demo_image2.id not in [i.id for i in bad_images]

            # check that adding a badness on the image itself is added to the total badness
            demo_image4.badness = 'saturation'
            session.add(demo_image4)
            session.commit()
            assert demo_image4.badness == 'Banding, Saturation, Bright Sky'
            assert demo_image4._bitflag == 2 ** 3  # only this bit is from the image itself

            # make a new subtraction:
            demo_image7 = Image.from_ref_and_new(demo_image6, demo_image5)
            demo_image7.provenance = provenance_extra
            cleanups.append(ImageCleanup.save_image(demo_image7))
            images.append(demo_image7)
            demo_image7 = demo_image7.recursive_merge( session )
            session.add(demo_image7)
            session.commit()

            # check that the new subtraction is not flagged
            assert demo_image7.badness == ''
            assert demo_image7._bitflag == 0
            assert demo_image7.bitflag == 0

            good_images = session.scalars(sa.select(Image).where(Image.bitflag == 0)).all()
            assert demo_image5.id in [i.id for i in good_images]
            assert demo_image6.id in [i.id for i in good_images]
            assert demo_image7.id in [i.id for i in good_images]

            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert demo_image5.id not in [i.id for i in bad_images]
            assert demo_image6.id not in [i.id for i in bad_images]
            assert demo_image7.id not in [i.id for i in bad_images]

            # let's try to coadd an image based on some good and bad images
            # as a reminder, demo_image2 has Bright Sky (5),
            # demo_image3's exposure has banding (1), while
            # demo_image4 has Saturation (3).

            # make a coadded image (without including the subtraction demo_image4):
            demo_image8 = Image.from_images([demo_image, demo_image2, demo_image3, demo_image5, demo_image6])
            demo_image8.provenance = provenance_extra
            cleanups.append(ImageCleanup.save_image(demo_image8))
            images.append(demo_image8)
            demo_image8 = demo_image8.recursive_merge( session )
            session.add(demo_image8)
            session.commit()

            assert demo_image8.badness == 'Banding, Bright Sky'
            assert demo_image8.bitflag == 2 ** 1 + 2 ** 5

            # does this work in queries (i.e., using the bitflag hybrid expression)?
            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert demo_image8.id in [i.id for i in bad_images]
            bad_coadd = session.scalars(sa.select(Image).where(Image.bitflag == 2 ** 1 + 2 ** 5)).all()
            assert demo_image8.id in [i.id for i in bad_coadd]

            # get rid of this coadd to make a new one
            demo_image8.delete_from_disk_and_database(session=session)
            cleanups.pop()
            images.pop()

            # now let's add the subtraction image to the coadd:
            # make a coadded image (now including the subtraction demo_image4):
            demo_image8 = Image.from_images([demo_image, demo_image2, demo_image3, demo_image4, demo_image5, demo_image6])
            demo_image8.provenance = provenance_extra
            cleanups.append(ImageCleanup.save_image(demo_image8))
            images.append(demo_image8)
            demo_image8 = demo_image8.recursive_merge(session)
            session.add(demo_image8)
            session.commit()

            session.add(demo_image8)
            session.commit()

            assert demo_image8.badness == 'Banding, Saturation, Bright Sky'
            assert demo_image8.bitflag == 2 ** 1 + 2 ** 3 + 2 ** 5  # this should be 42

            # does this work in queries (i.e., using the bitflag hybrid expression)?
            bad_images = session.scalars(sa.select(Image).where(Image.bitflag != 0)).all()
            assert demo_image8.id in [i.id for i in bad_images]
            bad_coadd = session.scalars(sa.select(Image).where(Image.bitflag == 42)).all()
            assert demo_image8.id in [i.id for i in bad_coadd]

            # try to add some badness to one of the underlying exposures
            demo_image.exposure.badness = 'Shaking'
            session.add(demo_image)
            demo_image.exposure.update_downstream_badness(session)
            session.commit()

            assert 'Shaking' in demo_image.badness
            assert 'Shaking' in demo_image8.badness

    finally:  # cleanup
        with SmartSession() as session:
            session.autoflush = False
            for im in images:
                im = im.recursive_merge(session)
                im.delete_from_disk_and_database(session=session, commit=False)

            session.commit()


def test_image_coordinates():
    image = Image('coordinates.fits', ra=None, dec=None, nofile=True)
    assert image.ecllat is None
    assert image.ecllon is None
    assert image.gallat is None
    assert image.gallon is None

    with pytest.raises(ValueError, match='Object must have RA and Dec set'):
        image.calculate_coordinates()

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


def test_image_from_exposure(exposure, provenance_base):
    exposure.update_instrument()
    exposure.type = 'ComSci'

    # demo instrument only has one section
    with pytest.raises(ValueError, match='section_id must be 0 for this instrument.'):
        _ = Image.from_exposure(exposure, section_id=1)

    im = Image.from_exposure(exposure, section_id=0)
    assert im.section_id == 0
    assert im.mjd == exposure.mjd
    assert im.end_mjd == exposure.end_mjd
    assert im.exp_time == exposure.exp_time
    assert im.end_mjd == im.mjd + im.exp_time / 86400
    assert im.filter == exposure.filter
    assert im.instrument == exposure.instrument
    assert im.telescope == exposure.telescope
    assert im.project == exposure.project
    assert im.target == exposure.target
    assert not im.is_coadd
    assert not im.is_sub
    assert im.id is None  # need to commit to get IDs
    assert im.exposure_id is None  # need to commit to get IDs
    assert im.upstream_images == []
    assert im.filepath is None  # need to save file to generate a filename
    assert np.array_equal(im.raw_data, exposure.data[0])
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
                im = im.recursive_merge( session )
                session.add(im)
                session.commit()
            session.rollback()

            # must add the provenance!
            im.provenance = provenance_base
            im = im.recursive_merge( session )
            with pytest.raises(IntegrityError, match='null value in column "filepath" of relation "images"'):
                session.add(im)
                session.commit()
            session.rollback()

            _ = ImageCleanup.save_image(im)  # this will add the filepath and md5 sum!

            session.add(im)
            session.commit()

            assert im.id is not None
            assert im.provenance_id is not None
            assert im.provenance_id == provenance_base.id
            assert im.exposure_id is not None
            assert im.exposure_id == exposure.id

    finally:
        if im_id is not None:
            with SmartSession() as session:
                im.delete_from_disk_and_database(commit=True, session=session)


def test_image_from_exposure_filter_array(exposure_filter_array):
    exposure_filter_array.update_instrument()
    im = Image.from_exposure(exposure_filter_array, section_id=0)
    filt = exposure_filter_array.filter_array[0]
    assert im.filter == filt


def test_image_with_multiple_upstreams(exposure, exposure2, provenance_base):
    exposure.update_instrument()
    exposure2.update_instrument()

    # make sure exposures are in chronological order...
    if exposure.mjd > exposure2.mjd:
        exposure, exposure2 = exposure2, exposure

    # get a couple of images from exposure objects
    im1 = Image.from_exposure(exposure, section_id=0)
    im2 = Image.from_exposure(exposure2, section_id=0)
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
            im = im.recursive_merge( session )
            im1 = im1.recursive_merge( session )
            im2 = im2.recursive_merge( session )
            exposure = exposure.recursive_merge( session )
            exposure2 = exposure2.recursive_merge( session )
            session.add(im)
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
            assert im1.exposure_id == exposure.id
            assert im1.upstream_images == []

            im2_id = im2.id
            assert im2_id is not None
            assert im2.exposure_id is not None
            assert im2.exposure_id == exposure2.id
            assert im2.upstream_images == []

    finally:  # make sure to clean up all images
        for id_ in [im_id, im1_id, im2_id]:
            if id_ is not None:
                with SmartSession() as session:
                    im = session.scalars(sa.select(Image).where(Image.id == id_)).first()
                    session.delete(im)
                    session.commit()


def test_image_subtraction(exposure, exposure2, provenance_base):
    exposure.update_instrument()
    exposure2.update_instrument()

    # make sure exposures are in chronological order...
    if exposure.mjd > exposure2.mjd:
        exposure, exposure2 = exposure2, exposure

    # get a couple of images from exposure objects
    im1 = Image.from_exposure(exposure, section_id=0)
    im2 = Image.from_exposure(exposure2, section_id=0)
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
            im = im.recursive_merge( session )
            im1 = im1.recursive_merge( session )
            im2 = im2.recursive_merge( session )
            exposure = exposure.recursive_merge( session )
            exposure2 = exposure2.recursive_merge( session )
            session.add(im)
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
            assert im1.exposure_id == exposure.id
            assert im1.upstream_images == []

            im2_id = im2.id
            assert im2_id is not None
            assert im2.exposure_id is not None
            assert im2.exposure_id == exposure2.id
            assert im2.upstream_images == []

    finally:  # make sure to clean up all images
        for id_ in [im_id, im1_id, im2_id]:
            if id_ is not None:
                with SmartSession() as session:
                    im = session.scalars(sa.select(Image).where(Image.id == id_)).first()
                    session.delete(im)
                    session.commit()


def test_image_filename_conventions(demo_image, provenance_base):
    demo_image.data = np.float32(demo_image.raw_data)
    demo_image.provenance = provenance_base

    # use the naming convention in the config file
    demo_image.save( no_archive=True )

    assert re.search(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}\.image\.fits', demo_image.get_fullpath()[0])
    for f in demo_image.get_fullpath(as_list=True):
        assert os.path.isfile(f)
        os.remove(f)

    cfg = config.Config.get()

    # try to set the name convention to None, to load the default hard-coded one
    convention = cfg.value('storage.images.name_convention')
    try:
        cfg.set_value('storage.images.name_convention', None)
        demo_image.save( no_archive=True )
        assert re.search(r'Demo_\d{8}_\d{6}_\d+_.+_.{6}\.image\.fits', demo_image.get_fullpath()[0])
        for f in demo_image.get_fullpath(as_list=True):
            assert os.path.isfile(f)
            os.remove(f)
            folder = os.path.dirname(f)
            if len(os.listdir(folder)) == 0:
                os.rmdir(folder)

        new_convention = '{ra_int:03d}/foo_{date}_{time}_{section_id_int:02d}_{filter}'
        cfg.set_value('storage.images.name_convention', new_convention)
        # This next line was added because I changed image.save() so
        # that it will use self.filepath if it is non-None.  (I needed
        # this change in behavior to be able to control filenames when
        # called from a DataStore save method; the other option would be
        # to add options to those methods and then pass those options
        # on, and that is very cumbersome.)
        demo_image.filepath = None
        demo_image.save( no_archive=True )
        assert re.search(r'\d{3}/foo_\d{8}_\d{6}_\d{2}_.\.image\.fits', demo_image.get_fullpath()[0])
        for f in demo_image.get_fullpath(as_list=True):
            assert os.path.isfile(f)
            os.remove(f)
            folder = os.path.dirname(f)
            if len(os.listdir(folder)) == 0:
                os.rmdir(folder)

        new_convention = 'bar_{date}_{time}_{section_id_int:02d}_{ra_int_h:02d}{dec_int:+03d}'
        cfg.set_value('storage.images.name_convention', new_convention)
        demo_image.filepath = None
        demo_image.save( no_archive=True )
        assert re.search(r'bar_\d{8}_\d{6}_\d{2}_\d{2}[+-]\d{2}\.image\.fits', demo_image.get_fullpath()[0])
        for f in demo_image.get_fullpath(as_list=True):
            assert os.path.isfile(f)
            os.remove(f)
            folder = os.path.dirname(f)
            if len(os.listdir(folder)) == 0:
                os.rmdir(folder)

    finally:  # return to the original convention
        cfg.set_value('storage.images.name_convention', convention)


def test_image_multifile(demo_image, provenance_base):
    demo_image.data = np.float32(demo_image.raw_data)
    demo_image.flags = np.random.randint(0, 100, size=demo_image.raw_data.shape, dtype=np.uint32)
    demo_image.provenance = provenance_base

    cfg = config.Config.get()
    single_fileness = cfg.value('storage.images.single_file')  # store initial value

    try:
        # first use single file
        cfg.set_value('storage.images.single_file', True)
        demo_image.save( no_archive=True )

        assert re.match(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}\.fits', demo_image.filepath)

        files = demo_image.get_fullpath(as_list=True)
        assert len(files) == 1
        assert os.path.isfile(files[0])

        with fits.open(files[0]) as hdul:
            assert len(hdul) == 3  # primary plus one for image data and one for flags
            assert hdul[0].header['NAXIS'] == 0
            assert isinstance(hdul[0], fits.PrimaryHDU)
            assert hdul[1].header['NAXIS'] == 2
            assert isinstance(hdul[1], fits.ImageHDU)
            assert np.array_equal(hdul[1].data, demo_image.data)
            assert hdul[2].header['NAXIS'] == 2
            assert isinstance(hdul[2], fits.ImageHDU)
            assert np.array_equal(hdul[2].data, demo_image.flags)

        for f in demo_image.get_fullpath(as_list=True):
            os.remove(f)

        # now test multiple files
        cfg.set_value('storage.images.single_file', False)
        demo_image.filepath = None
        demo_image.save( no_archive=True )

        assert re.match(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}', demo_image.filepath)
        fullnames = demo_image.get_fullpath(as_list=True)

        assert len(fullnames) == 2

        assert os.path.isfile(fullnames[0])
        assert re.search(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}\.image\.fits', fullnames[0])
        with fits.open(fullnames[0]) as hdul:
            assert len(hdul) == 1  # image data is saved on the primary HDU
            assert hdul[0].header['NAXIS'] == 2
            assert isinstance(hdul[0], fits.PrimaryHDU)
            assert np.array_equal(hdul[0].data, demo_image.data)

        assert os.path.isfile(fullnames[1])
        assert re.search(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}\.flags\.fits', fullnames[1])
        with fits.open(fullnames[1]) as hdul:
            assert len(hdul) == 1
            assert hdul[0].header['NAXIS'] == 2
            assert isinstance(hdul[0], fits.PrimaryHDU)
            assert np.array_equal(hdul[0].data, demo_image.flags)

    finally:
        cfg.set_value('storage.images.single_file', single_fileness)


def test_image_from_decam_exposure(decam_example_file, provenance_base):
    with fits.open( decam_example_file, memmap=False ) as ifp:
        hdr = ifp[0].header
    exphdrinfo = Instrument.extract_header_info( hdr, [ 'mjd', 'exp_time', 'filter', 'project', 'target' ] )
    ra = util.radec.parse_sexigesimal_degrees( hdr['RA'], hours=True )
    dec = util.radec.parse_sexigesimal_degrees( hdr['DEC'] )
    e = Exposure( ra=ra, dec=dec, instrument='DECam', format='fits', **exphdrinfo,
                  filepath=str( pathlib.Path('test_data/DECam_examples') / pathlib.Path(decam_example_file).name ) )
    sec_id = 'N4'
    im = Image.from_exposure(e, section_id=sec_id)  # load the first CCD

    assert e.instrument == 'DECam'
    assert e.telescope == 'CTIO 4.0-m telescope'
    assert not im.from_db
    # should not be the same as the exposure!
    # assert im.ra == 116.32024583333332
    # assert im.dec == -26.25
    assert im.ra != e.ra
    assert im.dec != e.dec
    assert im.ra == 116.32126671843677
    assert im.dec == -26.337508447652503
    assert im.mjd == 59887.32121458
    assert im.end_mjd == 59887.32232569111
    assert im.exp_time == 96.0
    assert im.filter == 'g DECam SDSS c0001 4720.0 1520.0'
    assert im.target == 'DECaPS-West'
    assert im.project == '2022A-724693'
    assert im.section_id == sec_id

    assert im.id is None  # not yet on the DB
    assert im.filepath is None  # no file yet!

    # the header lazy loads alright:
    assert len(im.raw_header) == 98
    assert im.raw_header['NAXIS'] == 2
    assert im.raw_header['NAXIS1'] == 2160
    assert im.raw_header['NAXIS2'] == 4146
    assert 'BSCALE' not in im.raw_header
    assert 'BZERO' not in im.raw_header

    # check we have the raw data copied into temporary attribute
    assert im.raw_data is not None
    assert isinstance(im.raw_data, np.ndarray)
    assert im.raw_data.shape == (4146, 2160)

    # just for this test we will do preprocessing just by reducing the median
    im.data = np.float32(im.raw_data - np.median(im.raw_data))

    # check we can save the image using the filename conventions


