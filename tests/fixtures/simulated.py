import pytest
import os
import warnings
import uuid

import numpy as np

import sqlalchemy as sa

from astropy.io import fits
from astropy.time import Time

from models.base import SmartSession, _logger
from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.reference import Reference
from models.source_list import SourceList

from tests.conftest import rnd_str


def make_sim_exposure():
    e = Exposure(
        filepath=f"Demo_test_{rnd_str(5)}.fits",
        section_id=0,
        exp_time=np.random.randint(1, 4) * 10,  # 10 to 40 seconds
        mjd=np.random.uniform(58000, 58500),
        filter=np.random.choice(list('grizY')),
        ra=np.random.uniform(0, 360),
        dec=np.random.uniform(-90, 90),
        project='foo',
        target=rnd_str(6),
        nofile=True,
        md5sum=uuid.uuid4(),  # this should be done when we clean up the exposure factory a little more
    )
    return e


def add_file_to_exposure(exposure):
    fullname = exposure.get_fullpath()
    open(fullname, 'a').close()

    yield exposure  # don't use this, but let it sit there until going out of scope of the test

    if fullname is not None and os.path.isfile(fullname):
        os.remove(fullname)


def commit_exposure(exposure, session=None):
    with SmartSession(session) as session:
        exposure = session.merge(exposure)
        exposure.nofile = True  # avoid calls to the archive to find this file
        session.commit()

    return exposure


# idea taken from: https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
def generate_exposure_fixture():
    @pytest.fixture
    def new_exposure():
        e = make_sim_exposure()
        add_file_to_exposure(e)
        e = commit_exposure(e)

        yield e

        with SmartSession() as session:
            e = session.merge(e)
            if sa.inspect(e).persistent:
                session.delete(e)
                session.commit()

    return new_exposure


# this will inject 10 exposures named sim_exposure1, sim_exposure2, etc.
for i in range(1, 10):
    globals()[f'sim_exposure{i}'] = generate_exposure_fixture()


@pytest.fixture
def sim_exposure_filter_array():
    e = make_sim_exposure()
    e.filter = None
    e.filter_array = ['r', 'g', 'r', 'i']
    add_file_to_exposure(e)
    e = commit_exposure(e)

    yield e

    if 'e' in locals():
        with SmartSession() as session:
            e = session.merge(e)
            if sa.inspect( e ).persistent:
                session.delete(e)
                session.commit()


# tools for making Image fixtures
class ImageCleanup:
    """
    Helper function that allows you to take an Image object
    with fake data (for testing) and save it to disk,
    while also making sure that the data is removed from disk
    when the object goes out of scope.

    Usage:
    >> im_clean = ImageCleanup.save_image(image)
    at end of test the im_clean goes out of scope and removes the file
    """

    @classmethod
    def save_image(cls, image, archive=True):
        """
        Save the image to disk, and return an ImageCleanup object.

        Parameters
        ----------
        image: models.image.Image
            The image to save (that is used to call remove_data_from_disk)
        archive:
            Whether to save to the archive or not. Default is True.
            Controls the save(no_archive) flag and whether the file
            will be cleaned up from database and archive at the end.

        Returns
        -------
        ImageCleanup:
            An object that will remove the image from disk when it goes out of scope.
            This should be put into a variable that goes out of scope at the end of the test.
        """
        if image.data is None:
            if image.raw_data is None:
                image.raw_data = np.random.uniform(0, 100, size=(100, 100))
            image.data = np.float32(image.raw_data)

        if image.instrument is None:
            image.instrument = 'DemoInstrument'

        if image._header is None:
            image._header = fits.Header()

        image.save(no_archive=not archive)

        return cls(image, archive=archive)  # don't use this, but let it sit there until going out of scope of the test

    def __init__(self, image, archive=True):
        self.image = image
        self.archive = archive

    def __del__(self):
        # _logger.debug('removing file at end of test!')
        try:
            if self.archive:
                self.image.delete_from_disk_and_database()
            else:
                self.image.remove_data_from_disk()
        except Exception as e:
            warnings.warn(str(e))


# idea taken from: https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
def generate_image_fixture(commit=True):

    @pytest.fixture
    def new_image(provenance_preprocessing):
        exp = make_sim_exposure()
        add_file_to_exposure(exp)
        if commit:
            exp = commit_exposure(exp)
        exp.update_instrument()

        im = Image.from_exposure(exp, section_id=0)
        im.data = np.float32(im.raw_data)  # this replaces the bias/flat preprocessing
        im.flags = np.random.randint(0, 100, size=im.raw_data.shape, dtype=np.uint32)
        im.weight = np.full(im.raw_data.shape, 4., dtype=np.float32)

        if commit:
            with SmartSession() as session:
                im.provenance = provenance_preprocessing
                im.save()
                merged_image = session.merge(im)
                merged_image.raw_data = im.raw_data
                merged_image.data = im.data
                merged_image.flags = im.flags
                merged_image.weight = im.weight
                merged_image.header = im.header
                im = merged_image
                session.commit()

        yield im

        with SmartSession() as session:
            im = session.merge(im)
            exp = im.exposure
            im.delete_from_disk_and_database(session=session, commit=True)
            if sa.inspect( im ).persistent:
                session.delete(im)
                session.commit()

            if im in session:
                session.expunge(im)

            if exp is not None and sa.inspect( exp ).persistent:
                session.delete(exp)
                session.commit()

    return new_image


# this will inject 10 images named sim_image1, sim_image2, etc.
for i in range(1, 10):
    globals()[f'sim_image{i}'] = generate_image_fixture()


# use this Image if you want the test to do the saving
sim_image_uncommitted = generate_image_fixture(commit=False)


@pytest.fixture
def sim_reference(provenance_preprocessing, provenance_extra):
    filter = np.random.choice(list('grizY'))
    target = rnd_str(6)
    ra = np.random.uniform(0, 360)
    dec = np.random.uniform(-90, 90)
    images = []
    with SmartSession() as session:
        provenance_extra = session.merge(provenance_extra)

        for i in range(5):
            exp = make_sim_exposure()
            add_file_to_exposure(exp)
            exp = commit_exposure(exp, session)
            exp.filter = filter
            exp.target = target
            exp.project = "coadd_test"
            exp.ra = ra
            exp.dec = dec

            exp.update_instrument()
            im = Image.from_exposure(exp, section_id=0)
            im.data = im.raw_data - np.median(im.raw_data)
            im.provenance = provenance_preprocessing
            im.ra = ra
            im.dec = dec
            im.save()
            im.provenance = session.merge(im.provenance)
            session.add(im)
            images.append(im)

        ref_image = Image.from_images(images)
        ref_image.is_coadd = True
        ref_image.data = np.mean(np.array([im.data for im in images]), axis=0)

        provenance_extra.process = 'coaddition'
        ref_image.provenance = provenance_extra
        ref_image.save()
        session.add(ref_image)

        ref = Reference()
        ref.image = ref_image
        ref.provenance = Provenance(
            code_version=provenance_extra.code_version,
            process='reference',
            parameters={'test_parameter': 'test_value'},
            upstreams=[provenance_extra],
            is_testing=True,
        )
        ref.validity_start = Time(50000, format='mjd', scale='utc').isot
        ref.validity_end = Time(58500, format='mjd', scale='utc').isot
        ref.section_id = 0
        ref.filter = filter
        ref.target = target
        ref.project = "coadd_test"

        session.add(ref)
        session.commit()

    yield ref

    if 'ref' in locals():
        with SmartSession() as session:
            ref = ref.merge_all(session)
            for im in ref.image.upstream_images:
                im.exposure.delete_from_disk_and_database(session=session, commit=False)
                im.delete_from_disk_and_database(session=session, commit=False)
            ref.image.delete_from_disk_and_database(session=session, commit=False)
            if sa.inspect(ref).persistent:
                session.delete(ref.provenance)  # should also delete the reference
            session.commit()


@pytest.fixture
def sim_sources(sim_image1):
    num = 100
    x = np.random.uniform(0, sim_image1.raw_data.shape[1], num)
    y = np.random.uniform(0, sim_image1.raw_data.shape[0], num)
    flux = np.random.uniform(0, 1000, num)
    flux_err = np.random.uniform(0, 100, num)
    rhalf = np.abs(np.random.normal(0, 3, num))

    data = np.array(
        [x, y, flux, flux_err, rhalf],
        dtype=([('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('flux_err', 'f4'), ('rhalf', 'f4')])
    )
    s = SourceList(image=sim_image1, data=data, format='sepnpy')

    prov = Provenance(
        code_version=sim_image1.provenance.code_version,
        process='extraction',
        parameters={'test_parameter': 'test_value'},
        upstreams=[sim_image1.provenance],
        is_testing=True,
    )

    with SmartSession() as session:
        s.provenance = prov
        s.save()
        s = session.merge(s)
        session.commit()

    yield s

    with SmartSession() as session:
        s = s.merge_all(session)
        s.delete_from_disk_and_database(session=session, commit=True)
