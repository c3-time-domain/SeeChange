import os
import pytest
import re

import numpy as np

from astropy.io import fits

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, DataError

import util.config as config
from models.base import SmartSession
from models.exposure import Exposure
from models.image import Image


def rnd_str(n):
    return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


def test_image_no_null_values(provenance_base):

    required = {
        'mjd': 58392.1,
        'end_mjd': 58392.1 + 30 / 86400,
        'exp_time': 30,
        'filter': 'r',
        'ra': np.random.uniform(0, 360),
        'dec': np.random.uniform(-90, 90),
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
        image = Image(f"Demo_test_{rnd_str(5)}.fits", nofile=True)
        with SmartSession() as session:
            for i in range(len(required)):
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
            exposure = None
            if im_id is not None:
                exposure = session.scalars(sa.select(Image).where(Image.id == im_id)).first()
            if exposure is not None:
                session.delete(exposure)
                session.commit()


def test_image_enum_values(demo_image, provenance_base):
    data_filename = None
    with SmartSession() as session:
        demo_image.provenance = provenance_base
        with pytest.raises(RuntimeError, match='The image data is not loaded. Cannot save.'):
            demo_image.save()

        demo_image.data = np.float32(demo_image.raw_data)
        demo_image.save()
        data_filename = demo_image.get_fullpath(as_list=True)[0]
        assert os.path.exists(data_filename)

        try:
            with pytest.raises(DataError, match='invalid input value for enum image_type: "foo"'):
                demo_image.type = 'foo'
                session.add(demo_image)
                session.commit()
            session.rollback()

            for t in ["science", "reference", "difference", "bias", "dark", "flat"]:
                demo_image.type = t
                session.add(demo_image)
                session.commit()

        finally:
            if data_filename is not None and os.path.exists(data_filename):
                os.remove(data_filename)
                folder = os.path.dirname(data_filename)
                if len(os.listdir(folder)) == 0:
                    os.rmdir(folder)


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


def test_image_from_exposure(exposure, provenance_base):
    exposure.update_instrument()
    exposure.type = 'reference'

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
    assert im.source_images == []
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
            with pytest.raises(IntegrityError, match='null value in column "provenance_id" of relation "images"'):
                session.add(im)
                session.commit()
            session.rollback()

            # must add the provenance!
            im.provenance = provenance_base
            with pytest.raises(IntegrityError, match='null value in column "filepath" of relation "images"'):
                session.add(im)
                session.commit()
            session.rollback()

            # must add the filepath!
            im.filepath = 'foo_exposure.fits'

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
                im = session.scalars(sa.select(Image).where(Image.id == im_id)).first()
                session.delete(im)
                session.commit()


def test_image_from_exposure_filter_array(exposure_filter_array):
    exposure_filter_array.update_instrument()
    im = Image.from_exposure(exposure_filter_array, section_id=0)
    filt = exposure_filter_array.filter_array[0]
    assert im.filter == filt


def test_image_with_multiple_source_images(exposure, exposure2, provenance_base):
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
    im1.filepath = 'foo1.fits'
    im2.provenance = provenance_base
    im2.filepath = 'foo2.fits'

    # make a coadd image from the two
    im = Image.from_images([im1, im2])
    im.filepath = 'foo.fits'
    im.provenance = provenance_base

    try:
        im_id = None
        im1_id = None
        im2_id = None
        with SmartSession() as session:
            session.add(im)
            session.commit()

            im_id = im.id
            assert im_id is not None
            assert im.exposure_id is None
            assert im.is_coadd
            assert im.source_images == [im1, im2]
            assert np.isclose(im.mid_mjd, (im1.mjd + im2.mjd) / 2)

            # make sure source images are pulled into the database too
            im1_id = im1.id
            assert im1_id is not None
            assert im1.exposure_id is not None
            assert im1.exposure_id == exposure.id
            assert not im1.is_coadd
            assert im1.source_images == []

            im2_id = im2.id
            assert im2_id is not None
            assert im2.exposure_id is not None
            assert im2.exposure_id == exposure2.id
            assert not im2.is_coadd
            assert im2.source_images == []

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
    im1.filepath = 'foo1.fits'
    im2.provenance = provenance_base
    im2.filepath = 'foo2.fits'

    # make a coadd image from the two
    im = Image.from_ref_and_new(im1, im2)
    im.filepath = 'foo.fits'
    im.provenance = provenance_base

    try:
        im_id = None
        im1_id = None
        im2_id = None
        with SmartSession() as session:
            session.add(im)
            session.commit()

            im_id = im.id
            assert im_id is not None
            assert im.exposure_id is None
            assert im.is_sub
            assert im.ref_image == im1
            assert im.ref_image_id == im1.id
            assert im.new_image == im2
            assert im.new_image_id == im2.id
            assert im.mjd == im2.mjd
            assert im.exp_time == im2.exp_time

            # make sure source images are pulled into the database too
            im1_id = im1.id
            assert im1_id is not None
            assert im1.exposure_id is not None
            assert im1.exposure_id == exposure.id
            assert not im1.is_coadd
            assert im1.source_images == []

            im2_id = im2.id
            assert im2_id is not None
            assert im2.exposure_id is not None
            assert im2.exposure_id == exposure2.id
            assert not im2.is_coadd
            assert im2.source_images == []

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
    demo_image.save()

    assert re.search(r'\d{3}/Demo_\d{8}_\d{6}_\d+_.+_.{6}\.image\.fits', demo_image.get_fullpath()[0])
    for f in demo_image.get_fullpath(as_list=True):
        assert os.path.isfile(f)
        os.remove(f)

    cfg = config.Config.get()

    # try to set the name convention to None, to load the default hard-coded one
    convention = cfg.value('storage.images.name_convention')
    try:
        cfg.set_value('storage.images.name_convention', None)
        demo_image.save()
        assert re.search(r'Demo_\d{8}_\d{6}_\d+_.+_.{6}\.image\.fits', demo_image.get_fullpath()[0])
        for f in demo_image.get_fullpath(as_list=True):
            assert os.path.isfile(f)
            os.remove(f)
            folder = os.path.dirname(f)
            if len(os.listdir(folder)) == 0:
                os.rmdir(folder)

        new_convention = '{ra_int:03d}/foo_{date}_{time}_{section_id:02d}_{filter}'
        cfg.set_value('storage.images.name_convention', new_convention)
        demo_image.save()
        assert re.search(r'\d{3}/foo_\d{8}_\d{6}_\d{2}_.\.image\.fits', demo_image.get_fullpath()[0])
        for f in demo_image.get_fullpath(as_list=True):
            assert os.path.isfile(f)
            os.remove(f)
            folder = os.path.dirname(f)
            if len(os.listdir(folder)) == 0:
                os.rmdir(folder)

        new_convention = 'bar_{date}_{time}_{section_id:02d}_{ra_int_h:02d}{dec_int:+03d}'
        cfg.set_value('storage.images.name_convention', new_convention)
        demo_image.save()
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
        demo_image.save()

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
        demo_image.save()

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
    e = Exposure(decam_example_file)
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
    assert im.ra == 116.23984530727733
    assert im.dec == -26.410038282561345
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
    assert len(im.raw_header) == 100
    assert im.raw_header['NAXIS'] == 2
    assert im.raw_header['NAXIS1'] == 2160
    assert im.raw_header['NAXIS2'] == 4146

    # check we have the raw data copied into temporary attribute
    assert im.raw_data is not None
    assert isinstance(im.raw_data, np.ndarray)
    assert im.raw_data.shape == (4146, 2160)

    # just for this test we will do preprocessing just by reducing the median
    im.data = np.float32(im.raw_data - np.median(im.raw_data))

    # check we can save the image using the filename conventions
