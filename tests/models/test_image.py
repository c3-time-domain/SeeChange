import os
import pytest
import re
import psutil
import gc
import hashlib
import pathlib
import uuid
import time
import warnings

import numpy as np

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from models.base import SmartSession, FileOnDiskMixin
from models.instrument import get_instrument_instance
from models.image import Image, image_upstreams_association_table
from models.enums_and_bitflags import image_preprocessing_inverse, string_to_bitflag, image_badness_inverse
from models.psf import PSF
from models.source_list import SourceList
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from tests.conftest import rnd_str
from tests.fixtures.simulated import ImageCleanup


def test_image_no_null_values(provenance_base):
    rng = np.random.default_rng()
    required = {
        'mjd': 58392.1,
        'end_mjd': 58392.1 + 30 / 86400,
        'exp_time': 30,
        'ra': rng.uniform(0., 360.),
        'dec': rng.uniform(-90., 90.),
        'ra_corner_00': 0.,
        'ra_corner_01': 0.,
        'ra_corner_10': 0.,
        'ra_corner_11': 0.,
        'dec_corner_00': 0.,
        'dec_corner_01': 0.,
        'dec_corner_10': 0.,
        'dec_corner_11': 0.,
        'minra': 0.,
        'maxra': 0.,
        'mindec': 0.,
        'maxdec': 0.,
        'instrument': 'DemoInstrument',
        'telescope': 'DemoTelescope',
        'project': 'foo',
        'target': 'bar',
        'provenance_id': provenance_base.id,
    }

    added = {}

    # use non-capturing groups to extract the column name from the error message
    exc_re = re.compile( r'(?:null value in column )(".*")(?: of relation "images" violates not-null constraint)' )

    try:
        im_id = None  # make sure to delete the image if it is added to DB

        # md5sum is spoofed as we don't have this file saved to archive
        image = Image(f"Demo_test_{rnd_str(5)}.fits", md5sum=uuid.uuid4(), nofile=True, section_id=1)

        for i in range( len(required ) ):
            # set the exposure to the values in "added" or None if not in "added"
            for k in required.keys():
                setattr(image, k, added.get(k, None))

            # without all the required columns on image, it cannot be added to DB
            with pytest.raises( IntegrityError ) as exc:
                image.insert()

            # Figure out which column screamed and yelled about being null
            match_obj = exc_re.search( str(exc.value) )
            assert match_obj is not None

            # Extract the column that raised the error, and add it to things we set
            colname = match_obj.group(1).replace( '"', '' )
            added.update( { colname: required[colname] } )

        # now set all the required keys and make sure that the loading works
        for k in required.keys():
            setattr(image, k, added.get(k, None))
        image.insert()
        im_id = image.id
        assert im_id is not None

    finally:
        # cleanup
        with SmartSession() as session:
            session.execute( sa.delete( Image ).where( Image._id == im_id ) )
            session.commit()


def test_image_insert( sim_image1, sim_image2, sim_image3, sim_image_uncommitted, provenance_base ):
    # FileOnDiskMixin.test_insert() is tested in test_exposure.py::test_insert

    im = sim_image_uncommitted
    im.filepath = im.invent_filepath()
    # Spoof the md5sum as we're not actually going to save this image
    im.md5sum = uuid.uuid4()

    upstreamids = ( [ sim_image1.id, sim_image2.id ]
                    if sim_image1.mjd < sim_image2.mjd
                    else [ sim_image2.id, sim_image1.id ] )

    # Make sure that upstreams get created if the image has them
    im._upstream_ids = [ i for i in upstreamids ]
    im.insert()
    with SmartSession() as sess:
        upstrs = ( sess.query( image_upstreams_association_table )
                   .filter( image_upstreams_association_table.c.downstream_id == im.id ) ).all()
        assert len( upstrs ) == 2
        assert set( [ i.upstream_id for i in upstrs ] ) == set( upstreamids )

    # Make sure that we get the upstream ids from the database if necessary
    im._upstream_ids = None
    assert im.upstream_image_ids == upstreamids
    assert im._upstream_ids == upstreamids

    # clean up
    with SmartSession() as sess:
        sess.execute( sa.delete( Image ).where( Image._id==im.id ) )
        sess.commit()
        # Make sure the delete cascaded to the assocation tabe
        upstrs = ( sess.query( image_upstreams_association_table )
                   .filter( image_upstreams_association_table.c.downstream_id == im.id ) ).all()
        assert len(upstrs) == 0


def test_image_upsert( sim_image1, sim_image2, sim_image_uncommitted ):
    im = sim_image_uncommitted
    im.filepath = im.invent_filepath()
    im.md5sum = uuid.uuid4()
    im.is_coadd = True

    upstreamids = ( [ sim_image1.id, sim_image2.id ]
                    if sim_image1.mjd < sim_image2.mjd
                    else [ sim_image2.id, sim_image1.id ] )
    im._upstream_ids = [ i for i in upstreamids ]
    im.insert()

    expectedupstreams = set( upstreamids )
    expectedupstreams.add( im.exposure_id )

    newim = Image.get_by_id( im.id )
    assert set( [ i.id for i in newim.get_upstreams( only_images=True ) ] ) == expectedupstreams

    # Make sure that if I upsert, it works without complaining, and the upstreams are still in place
    oldfmt = newim._format
    im._format = oldfmt + 1
    im.upsert()
    newim = Image.get_by_id( im.id )
    assert set( [ i.id for i in newim.get_upstreams( only_images=True ) ] ) == expectedupstreams
    assert newim._format == oldfmt + 1

    im._format = None
    im.upsert( load_defaults=True )
    assert im._format == oldfmt + 1


def test_image_must_have_md5(sim_image_uncommitted, provenance_base):
    try:
        im = sim_image_uncommitted
        assert im.md5sum is None
        assert im.md5sum_extensions is None

        im.provenance_id = provenance_base.id
        _ = ImageCleanup.save_image(im, archive=False)

        im.md5sum = None

        with pytest.raises(IntegrityError, match='violates check constraint'):
            im.insert()

        # adding md5sums should fix this problem
        _2 = ImageCleanup.save_image(im, archive=True)
        im.insert()

    finally:
        im.delete_from_disk_and_database()


def test_image_archive_singlefile(sim_image_uncommitted, archive, test_config):
    im = sim_image_uncommitted
    im.data = np.float32( im.raw_data )
    rng = np.random.default_rng()
    im.flags = rng.integers(0, 100, size=im.raw_data.shape, dtype=np.uint16)

    archive_dir = archive.test_folder_path
    single_fileness = test_config.value('storage.images.single_file')

    try:
        # Do single file first
        test_config.set_value('storage.images.single_file', True)

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
        localmd5 = hashlib.md5()
        with open( im.get_fullpath( nofile=False ), 'rb' ) as ifp:
            localmd5.update( ifp.read() )
        assert localmd5.hexdigest() == im.md5sum.hex

        # Make sure that the md5sum is properly saved to the database
        assert im._id is None
        im.insert()
        assert im.id is not None
        with SmartSession() as session:
            dbimage = session.query(Image).filter(Image._id == im.id)[0]
            assert dbimage.md5sum.hex == im.md5sum.hex

        # Make sure we can purge the archive
        im.delete_from_disk_and_database()
        with pytest.raises(FileNotFoundError):
            ifp = open( archive_path, 'rb' )
            ifp.close()
        assert im.md5sum is None

    finally:
        im.delete_from_disk_and_database()
        test_config.set_value('storage.images.single_file', single_fileness)


def test_image_archive_multifile(sim_image_uncommitted, archive, test_config):
    im = sim_image_uncommitted
    im.data = np.float32( im.raw_data )
    rng = np.random.default_rng()
    im.flags = rng.integers(0, 100, size=im.raw_data.shape, dtype=np.uint16)
    im.weight = None

    archive_dir = archive.test_folder_path
    single_fileness = test_config.value('storage.images.single_file')

    try:
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
        assert im._id is None
        im.insert()
        assert im.id is not None
        with SmartSession() as session:
            dbimage = session.scalars(sa.select(Image).where(Image._id == im.id)).first()
        assert dbimage.md5sum is None
        filenames = dbimage.get_fullpath( nofile=True )
        for fullpath, md5sum in zip(filenames, dbimage.md5sum_extensions):
            assert localmd5s[fullpath].hexdigest() == md5sum.hex

    finally:
        im.delete_from_disk_and_database()
        test_config.set_value('storage.images.single_file', single_fileness)


def test_image_save_justheader( sim_image1 ):
    try:
        sim_image1.data = np.full( (64, 32), 0.125, dtype=np.float32 )
        rng = np.random.default_rng()
        sim_image1.flags = rng.integers(0, 100, size=sim_image1.data.shape, dtype=np.uint16)
        sim_image1.weight = np.full( (64, 32), 4., dtype=np.float32 )

        archive = sim_image1.archive

        _ = ImageCleanup.save_image( sim_image1, archive=True )
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
        # The fixtures should do all the necessary deleting
        pass


def test_image_save_onlyimage( sim_image1 ):
    sim_image1.data = np.full( (64, 32), 0.125, dtype=np.float32 )
    rng = np.random.default_rng()
    sim_image1.flags = rng.integers(0, 100, size=sim_image1.data.shape, dtype=np.uint16)
    sim_image1.weight = np.full( (64, 32), 4., dtype=np.float32 )

    _ = ImageCleanup.save_image( sim_image1, archive=False )
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


def test_image_enum_values( sim_image_uncommitted ):
    im = sim_image_uncommitted
    im.filepath = im.invent_filepath()
    # Spoof the md5sum since we aren't actually saving anything
    im.md5sum = uuid.uuid4()

    try:
        with pytest.raises(ValueError, match='ImageTypeConverter must be one of .* not foo'):
            im.type = 'foo'
            im.insert()

        # these should work
        for prepend in ["", "Com"]:
            for t in ["Sci", "Diff", "Bias", "Dark", "DomeFlat"]:
                im.type = prepend+t
                im.insert()
                # Now remove it so the next insert can work
                if not ( ( t == 'DomeFlat' ) and ( prepend == "Com" ) ):
                    im._delete_from_database()

        # should have an image with ComDomeFlat type
        assert im._type == 10  # see image_type_dict

        # make sure we can also select on this:
        with SmartSession() as session:
            images = session.scalars(sa.select(Image).where(Image.type == "ComDomeFlat")).all()
            assert im.id in [i.id for i in images]

            images = session.scalars(sa.select(Image).where(Image.type == "Sci")).all()
            assert im.id not in [i.id for i in images]


        im._delete_from_database()

        # check the image format enum works as expected:
        with pytest.raises(ValueError, match='ImageFormatConverter must be one of .* not foo'):
            im.format = 'foo'
            im.insert()

        # these should work
        for f in ['fits', 'hdf5']:
            im.format = f
            im.insert()
            if f != 'hdf5':
                im._delete_from_database()

        # should have an image with ComDomeFlat type
        assert im._format == 2  # see image_type_dict

        # make sure we can also select on this:
        with SmartSession() as session:
            images = session.scalars(sa.select(Image).where(Image.format == "hdf5")).all()
            assert im.id in [i.id for i in images]

            images = session.scalars(sa.select(Image).where(Image.format == "fits")).all()
            assert im.id not in [i.id for i in images]

    finally:
        # The fixtures should do all necessary deletion
        pass


def test_image_preproc_bitflag( sim_image1 ):
    # Reload the image from the database so the default values that get
    # set when the image is saved to the database are filled.
    with SmartSession() as session:
        im = session.query( Image ).filter( Image._id==sim_image1.id ).first()

    assert im.preproc_bitflag == 0
    im.preproc_bitflag |= string_to_bitflag( 'zero', image_preprocessing_inverse )
    assert im.preproc_bitflag == string_to_bitflag( 'zero', image_preprocessing_inverse )
    im.preproc_bitflag |= string_to_bitflag( 'flat', image_preprocessing_inverse )
    assert im.preproc_bitflag == string_to_bitflag( 'zero, flat', image_preprocessing_inverse )
    im.preproc_bitflag |= string_to_bitflag( 'flat, overscan', image_preprocessing_inverse )
    assert im.preproc_bitflag == string_to_bitflag( 'overscan, zero, flat', image_preprocessing_inverse )

    im2 = None
    try:
        # Save a new image with the preproc bitflag that we've set
        im2 = im.copy()
        im2.id = uuid.uuid4()
        im2.filepath = "delete_this_file.fits"       # Shouldn't actually get saved
        im2.insert()

        with SmartSession() as session:
            images = session.scalars(sa.select(Image).where(
                Image.preproc_bitflag.op('&')(string_to_bitflag('zero', image_preprocessing_inverse)) != 0
            )).all()
            assert im2.id in [i.id for i in images]

            images = session.scalars(sa.select(Image).where(
                Image.preproc_bitflag.op('&')(string_to_bitflag('zero,flat', image_preprocessing_inverse)) !=0
            )).all()
            assert im2.id in [i.id for i in images]

            images = session.scalars(sa.select(Image).where(
                Image.preproc_bitflag.op('&')(
                    string_to_bitflag('zero, flat', image_preprocessing_inverse)
                ) == string_to_bitflag('flat, zero', image_preprocessing_inverse)
            )).all()
            assert im2.id in [i.id for i in images]

            images = session.scalars(sa.select(Image).where(
                Image.preproc_bitflag.op('&')(string_to_bitflag('fringe', image_preprocessing_inverse) ) !=0
            )).all()
            assert im2.id not in [i.id for i in images]

            images = session.scalars(sa.select(Image.filepath).where(
                Image.id == im.id,  # only find the original image, if any
                Image.preproc_bitflag.op('&')(
                    string_to_bitflag('fringe, overscan', image_preprocessing_inverse)
                ) == string_to_bitflag( 'overscan, fringe', image_preprocessing_inverse )
            )).all()
            assert len(images) == 0

    finally:
        if im2 is not None:
            im2._delete_from_database()


def test_image_from_exposure( provenance_base, sim_exposure1 ):
    sim_exposure1.update_instrument()

    # demo instrument only has one section
    with pytest.raises(ValueError, match='section_id must be 0 for this instrument.'):
        _ = Image.from_exposure(sim_exposure1, section_id=1)

    im = Image.from_exposure(sim_exposure1, section_id=0)
    assert im._id is None
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
    assert im._id is None  # need to commit to get IDs
    assert im._upstream_ids is None
    assert im.filepath is None  # need to save file to generate a filename
    assert np.array_equal(im.raw_data, sim_exposure1.data[0])
    assert im.data is None
    assert im.flags is None
    assert im.weight is None
    assert im.nofile  # images are made without a file by default

    # TODO: add check for loading the header after we make a demo header maker
    # TODO: what should the RA/Dec be for an image that cuts out from an exposure?
    #  (It should be the RA/Dec at the center of the image; ideally, Image.from_exposure
    #  will do that right.  We should test that, if we don't already somewhere.  It is
    #  almost certainly instrument-specific code to do this.)

    try:
        with pytest.raises(IntegrityError, match='null value in column .* of relation "images"'):
            im.insert()

        # must add the provenance!
        im.provenance_id = provenance_base.id
        with pytest.raises(IntegrityError, match='null value in column "filepath" of relation "images"'):
            im.insert()

        im.data = im.raw_data
        im.save()   # This will add the filepath and md5sum

        im.insert()

        assert im.id is not None
        assert im.provenance_id is not None
        assert im.provenance_id == provenance_base.id
        assert im.exposure_id is not None
        assert im.exposure_id == sim_exposure1.id

    finally:
        # All necessary cleanup *should* be done in fixtures
        # (The sim_exposure1 fixture will delete images derived from it)
        pass


def test_image_from_exposure_filter_array(sim_exposure_filter_array):
    sim_exposure_filter_array.update_instrument()
    im = Image.from_exposure(sim_exposure_filter_array, section_id=0)
    filt = sim_exposure_filter_array.filter_array[0]
    assert im.filter == filt


def test_image_from_reduced_exposure( decam_reduced_origin_exposure_loaded_in_db ):
    # We get a whole bunch of annoying AstropyUserWarning about an invalid
    #  header record.  It's hard to work up a care.
    with warnings.catch_warnings():
        warnings.simplefilter( 'ignore', AstropyWarning )
        decam = get_instrument_instance( 'DECam' )
        exp = decam_reduced_origin_exposure_loaded_in_db

        img = Image.from_exposure( exp, section_id='N16' )

    assert img.format == 'fits'
    assert img.exposure_id == exp.id
    assert img.ref_image_id is None
    with pytest.raises( RuntimeError, match="new_image_id is not defined for images that aren't subtractions" ):
        assert img.new_image_id is None
    assert img.upstream_image_ids == []
    assert not img.is_sub
    assert not img.is_coadd
    assert img.type == 'Sci'
    assert img.provenance_id is None    # from_exposure doesn't set provenance
    assert img.mjd == exp.mjd
    assert img.exp_time == exp.exp_time
    assert img.instrument == exp.instrument
    assert img.telescope == exp.telescope
    assert img.filter == exp.filter
    assert img.section_id == 'N16'
    assert img.project == exp.project
    assert img.preproc_bitflag == 127           # Same as what was set for exp in the fixture
    assert not img.astro_cal_done
    assert not img.sky_sub_done
    assert img.airmass == exp.airmass
    assert img.fwhm_estimate is None
    assert img.zero_point_estimate is None
    assert img.lim_mag_estimate is None
    assert img.bkg_mean_estimate is None
    assert img.bkg_rms_estimate is None
    ra, dec = decam.get_ra_dec_for_section_of_exposure( exp, 'N16' )
    assert img.ra == pytest.approx( ra , 10./3600. / np.cos( exp.dec * np.pi / 180. ) )
    assert img.dec == pytest.approx( dec, 10./3600. )
    assert img._data.shape == ( 4094, 2046 )
    assert img._weight.shape == img._data.shape
    assert img._flags.shape == img._data.shape
    assert img._data.mean() == pytest.approx( 373.123, abs=0.001 )
    assert img._data.std() == pytest.approx( 1558.650, abs=0.001 )
    assert img._flags.dtype == np.int16
    assert img._flags.max() == 1
    assert img._flags.min() == 0
    assert img._flags.sum() == 266089


def test_image_with_multiple_upstreams(sim_exposure1, sim_exposure2, provenance_base):

    im = None
    try:
        sim_exposure1.update_instrument()
        sim_exposure2.update_instrument()

        # make sure exposures are in chronological order...
        if sim_exposure1.mjd > sim_exposure2.mjd:
            sim_exposure1, sim_exposure2 = sim_exposure2, sim_exposure1

        # get a couple of images from exposure objects
        im1 = Image.from_exposure(sim_exposure1, section_id=0)
        im1.provenance_id = provenance_base.id
        im1.filepath = im1.invent_filepath()
        im1.md5sum = uuid.uuid4()                # Spoof so we can save to database without writing a file
        im1.data = im1.raw_data
        im1.weight = np.ones_like(im1.raw_data)
        im1.flags = np.zeros_like(im1.raw_data)
        im2 = Image.from_exposure(sim_exposure2, section_id=0)
        im2.provenance_id = provenance_base.id
        im2.filepath = im2.invent_filepath()
        im2.md5sum = uuid.uuid4()                # Spoof so we can save to database without writing a file
        im2.data = im2.raw_data
        im2.weight = np.ones_like(im2.raw_data)
        im2.flags = np.zeros_like(im2.raw_data)
        im2.filter = im1.filter
        im2.target = im1.target

        # Since these images were created fresh, not loaded from the
        #  database, they don't have ids yet
        assert im1._id is None
        assert im2._id is None

        # make a "coadd" image from the two
        im = Image.from_images([im1, im2])
        assert im.is_coadd
        im.provenance_id = provenance_base.id
        # Spoof the md5sum so we can save to the database
        im.md5sum = uuid.uuid4()

        # im1, im2 provenances should have been filled in
        # when we ran from_images
        assert im1.id is not None
        assert im2.id is not None
        assert im1.upstream_image_ids == []
        assert im2.upstream_image_ids == []

        assert im._id is None
        assert im.exposure_id is None
        assert im.upstream_image_ids == [im1.id, im2.id]
        assert np.isclose(im.mid_mjd, (im1.mjd + im2.mjd) / 2)

        # Make sure we can save all of this to the database

        with pytest.raises( IntegrityError, match='null value in column "filepath" of relation "images" violates' ):
            im.insert()
        im.filepath = im.invent_filepath()

        # It should object if we haven't saved the upstreams first
        with pytest.raises( IntegrityError,
                            match='insert or update on table "images" violates foreign key constraint' ):
            im.insert()

        # So try to do it right
        im1.insert()
        im2.insert()
        im.insert()

        assert im.id is not None

        upstrimgs = im.get_upstreams( only_images=True )
        assert [ i.id for i in upstrimgs ] == [ im1.id, im2.id ]

        with SmartSession() as session:
            newim = session.query( Image ).filter( Image._id==im.id ).first()
        assert newim.upstream_image_ids == [ im1.id, im2.id ]
        assert newim.exposure_id is None
        assert np.isclose( im.mid_mjd, ( im1.mjd + im2.mjd ) / 2. )

    finally:
        # im1, im2 will be cleaned up by the exposure fixtures
        if im is not None:
            im.delete_from_disk_and_database()


def test_image_subtraction(sim_exposure1, sim_exposure2, provenance_base):
    im1 = None
    im2 = None
    im = None
    try:
        sim_exposure1.update_instrument()
        sim_exposure2.update_instrument()

        # make sure exposures are in chronological order...
        if sim_exposure1.mjd > sim_exposure2.mjd:
            sim_exposure1, sim_exposure2 = sim_exposure2, sim_exposure1

        # get a couple of images from exposure objects
        im1 = Image.from_exposure(sim_exposure1, section_id=0)
        im1.weight = np.ones_like(im1.raw_data)
        im1.flags = np.zeros_like(im1.raw_data)
        im2 = Image.from_exposure(sim_exposure2, section_id=0)
        im2.weight = np.ones_like(im2.raw_data)
        im2.flags = np.zeros_like(im2.raw_data)
        im2.filter = im1.filter
        im2.target = im1.target

        im1.provenance_id = provenance_base.id
        _1 = ImageCleanup.save_image(im1)
        im1.insert()

        im2.provenance_id = provenance_base.id
        _2 = ImageCleanup.save_image(im2)
        im2.insert()

        # make a coadd image from the two
        im = Image.from_ref_and_new(im1, im2)

        assert im._id is None
        assert im.exposure_id is None
        assert im.ref_image_id == im1.id
        assert im.new_image_id == im2.id
        assert im.mjd == im2.mjd
        assert im.exp_time == im2.exp_time
        assert im.upstream_image_ids == [ im1.id, im2.id ]

        im.provenance_id = provenance_base.id
        _3 = ImageCleanup.save_image(im)
        im.insert()

        # Reload from database, make sure all is well
        im = Image.get_by_id( im.id )
        assert im.id is not None
        assert im.exposure_id is None
        assert im.ref_image_id == im1.id
        assert im.new_image_id == im2.id
        assert im.mjd == im2.mjd
        assert im.exp_time == im2.exp_time
        assert im.upstream_image_ids == [ im1.id, im2.id ]

    finally:  # make sure to clean up all images
        # Make sure images are delete from the database.  Have to do it in this order;
        #   If im1 or im2 is deleted first, we'll get an error about the image still
        #   existing in image_upstreams_association.  That's because the association
        #   cascades when deleting a downstream, but *not* when deleting an upstream.
        for i in [ im, im1, im2 ]:
            if i is not None:
                with SmartSession() as session:
                    session.execute( sa.text( "DELETE FROM images WHERE _id=:id" ), {'id': i.id } )
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
    rng = np.random.default_rng()
    im.flags = rng.integers(0, 100, size=im.raw_data.shape, dtype=np.uint32)
    im.weight = None
    im.provenance_id = provenance_base.id

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
        test_config.set_value('storage.images.single_file', single_fileness)


# Note: ptf_datastore is a pretty heavyweight fixture, since it has to
#   build the ptf reference.  Perhaps this isn't a big deal, because
#   it'll get cached so later tests that run it will not be so slow.
#   (Still not instant, because there is all the disk writing and
#   archive uploading.)  But, we might want to think about using a
#   lighter weight fixture for this particular test.
def test_image_products_are_deleted(ptf_datastore, data_dir, archive):
    ds = ptf_datastore  # shorthand

    # check the datastore comes with all these objects
    assert isinstance(ds.image, Image)
    assert isinstance(ds.psf, PSF)
    assert isinstance(ds.sources, SourceList)
    assert isinstance(ds.wcs, WorldCoordinates)
    assert isinstance(ds.zp, ZeroPoint)
    # TODO: add more data types?

    # make sure the files are there
    local_files = []
    archive_files = []
    for obj in [ds.image, ds.psf, ds.sources, ds.wcs]:
        for file in obj.get_fullpath(as_list=True):
            archive_file = file[len(obj.local_path)+1:]  # grap the end of the path only
            archive_file = os.path.join(archive.test_folder_path, archive_file)  # prepend the archive path
            assert os.path.isfile(file)
            assert os.path.isfile(archive_file)
            local_files.append(file)
            archive_files.append(archive_file)

    # delete the image and all its downstreams
    ds.image.delete_from_disk_and_database(remove_folders=True, remove_downstreams=True)

    # make sure the files are gone (including cascading down to things dependent on the image)
    for file in local_files:
        assert not os.path.isfile(file)

    for file in archive_files:
        assert not os.path.isfile(file)


@pytest.mark.skip(reason="This test regularly fails, even when flaky is used. See Issue #263")
def test_free( decam_exposure, decam_raw_image, ptf_ref ):
    proc = psutil.Process()
    origmem = proc.memory_info()

    sleeptime = 0.5 # in seconds

    # Make sure that only_free behaves as expected
    decam_raw_image._weight = 'placeholder'
    decam_raw_image.free( only_free={'weight'} )
    time.sleep(sleeptime)
    assert decam_raw_image._weight is None
    assert decam_raw_image._data is not None
    assert decam_raw_image.raw_data is not None

    with pytest.raises( RuntimeError, match="Unknown image property to free" ):
        decam_raw_image.free( only_free={'this_is_not_a_property_that_actually_exists'} )

    # Make sure that things are None and that we get the memory back
    # when we free

    decam_raw_image.free( )
    time.sleep(sleeptime)
    assert decam_raw_image._data is None
    # The image is ~4k by 2k, data is 32-bit
    # so expect to free ~( 4000*2000 ) *4 >~ 30MiB of data
    gc.collect()
    freemem = proc.memory_info()
    assert origmem.rss - freemem.rss > 30 * 1024 * 1024

    # Make sure that if we clear the exposure's caches, the raw_data is now gone too
    # decam_exposure has session scope, but that file is on disk so it can reload
    # as necessary):

    assert decam_raw_image.raw_data is None
    decam_exposure.data.clear_cache()
    decam_exposure.section_headers.clear_cache()
    time.sleep(sleeptime)
    gc.collect()
    freemem = proc.memory_info()
    assert origmem.rss - freemem.rss > 45 * 1024 * 1024

    # Note that decam_raw_image.data will now raise an exception,
    #  because the weight and flags files aren't yet written to disk for
    #  this fixture.  (It was constructed in the fixture by manually
    #  setting data to a float32 copy of raw_data.)  decam_raw_image
    #  has test scope, not session scope, so that should be OK.


    # This next test is only meaningful if the ptf_ref fixture had to
    # rebuild the coadded ref.  If it loaded it from cache, then the
    # data for ptf_ref.image.aligned_images will not have been loaded.
    # So, make sure that happened before even bothering with the test.
    # (Our github actions tests always start with a clean environment,
    # so it will get run there.  If you want to run it locally, you
    # have to make sure your test cache is cleaned out.)

    # NOTE: at some point in the future, we may have coadd do
    # the freeing of the aligned images, at which point this
    # test will fail.  This note is here for you to find if
    # you're the one who made coadd do the freeing....

    if ptf_ref.image.aligned_images[0]._data is not None:
        origmem = proc.memory_info()
        # Make sure that the ref image data has been loaded
        _ = ptf_ref.image.data
        # Free the image and all the refs.  Expected savings: 6 4k × 2k
        # 32-bit images =~ 6 * (4000*2000) * 4 >~ 180MiB.
        ptf_ref.image.free( free_aligned=True )
        time.sleep(sleeptime)
        gc.collect()
        freemem = proc.memory_info()
        assert origmem.rss - freemem.rss > 180 * 1024 * 1024

    # the free_derived_products parameter is tested in test_source_list.py
    # and test_psf.py


# There are other tests of badness elsewhere that do upstreams and downstreams
# See: test_sources.py::test_source_list_bitflag ; test_pipeline.py::test_bitflag_propagation
def test_badness_basic( sim_image_uncommitted, provenance_base ):
    im = sim_image_uncommitted
    im.provenance_id = provenance_base.id
    im.filepath = im.invent_filepath()
    im.md5sum = uuid.uuid4()             # Spoof md5sum since we aren't really saving data

    # Make sure we can set it
    assert im.badness == ''
    im.set_badness( 'banding,shaking' )
    assert im._bitflag == ( 2**image_badness_inverse['banding'] | 2**image_badness_inverse['shaking'] )

    # Make sure it's not saved to the database even if we ask to commit and it has an id
    im.set_badness( None )
    assert im._bitflag == ( 2**image_badness_inverse['banding'] | 2**image_badness_inverse['shaking'] )

    with SmartSession() as session:
        assert session.query( Image ).filter( Image._id==im.id ).first() is None

    # Save it to the database
    im.insert()

    # Make sure it's there with the expected bitflag
    with SmartSession() as session:
        dbim = session.query( Image ).filter( Image._id==im.id ).first()
        assert dbim._bitflag == im._bitflag

    # Make a change to the bitflag and make sure it doesn't get committed if we don't want it to
    im.set_badness( '', commit=False )
    with SmartSession() as session:
        dbim = session.query( Image ).filter( Image._id==im.id ).first()
        assert dbim._bitflag == ( 2**image_badness_inverse['banding'] | 2**image_badness_inverse['shaking'] )

    # Make sure it gets saved if we do set_badness with None
    im.set_badness( None )
    with SmartSession() as session:
        dbim = session.query( Image ).filter( Image._id==im.id ).first()
        assert dbim._bitflag == 0

    # Make sure it gets saved if we set_badness without commit=False

    im.set_badness( 'saturation' )
    assert im._bitflag == 2**image_badness_inverse['saturation']
    with SmartSession() as session:
        dbim = session.query( Image ).filter( Image._id==im.id ).first()
        assert dbim._bitflag == im._bitflag

    # Make sure we can append without committing

    im.append_badness( 'shaking', commit=False )
    assert im._bitflag == 2**image_badness_inverse['saturation'] | 2**image_badness_inverse['shaking']
    with SmartSession() as session:
        dbim = session.query( Image ).filter( Image._id==im.id ).first()
        assert dbim._bitflag == 2**image_badness_inverse['saturation']

    # Make sure we can append with committing

    im.append_badness( 'banding' )
    assert im._bitflag == ( 2**image_badness_inverse['saturation'] |
                            2**image_badness_inverse['shaking'] |
                            2**image_badness_inverse['banding'] )
    with SmartSession() as session:
        dbim = session.query( Image ).filter( Image._id==im.id ).first()
        assert dbim._bitflag == im._bitflag

    # No need to clean up, the exposure from which sim_image_uncommitted was generated
    #  will clean up all its downstreams.
