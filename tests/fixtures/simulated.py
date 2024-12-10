import pytest
import os
import uuid

import numpy as np

import sqlalchemy as sa

from astropy.io import fits
from astropy.wcs import WCS

from models.base import SmartSession
from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference
from models.instrument import DemoInstrument

from pipeline.data_store import DataStore

from improc.tools import make_gaussian

from tests.conftest import rnd_str


def make_sim_exposure():
    rng = np.random.default_rng()
    e = Exposure(
        filepath=f"Demo_test_{rnd_str(5)}.fits",
        section_id=0,
        exp_time=rng.integers(1, 4) * 10,  # 10 to 40 seconds
        mjd=rng.uniform(58000, 58500),
        filter=rng.choice(list('grizY')),
        ra=rng.uniform(0, 360),
        dec=rng.uniform(-90, 90),
        project='foo',
        target=rnd_str(6),
        nofile=True,
        format='fits',
        md5sum=uuid.uuid4(),  # this should be done when we clean up the exposure factory a little more
    )
    return e


def add_file_to_exposure(exposure):
    """Creates an empty file at the exposure's filepath if one doesn't exist already."""

    fullname = exposure.get_fullpath()
    open(fullname, 'a').close()

    yield exposure  # don't use this, but let it sit there until going out of scope of the test

    if fullname is not None and os.path.isfile(fullname):
        os.remove(fullname)


def commit_exposure(exposure):
    exposure.insert()
    exposure.nofile = True  # avoid calls to the archive to find this file
    return exposure


# idea taken from: https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
def generate_exposure_fixture():
    @pytest.fixture
    def new_exposure():
        e = make_sim_exposure()
        add_file_to_exposure(e)
        e = commit_exposure(e)

        yield e

        e.delete_from_disk_and_database()

        with SmartSession() as session:
            # The provenance will have been automatically created
            session.execute( sa.delete( Provenance ).where( Provenance._id==e.provenance_id ) )
            session.commit()

    return new_exposure


# this will inject 9 exposures named sim_exposure1, sim_exposure2, etc.
for i in range(1, 10):
    globals()[f'sim_exposure{i}'] = generate_exposure_fixture()


@pytest.fixture
def unloaded_exposure():
    e = make_sim_exposure()
    return e


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

            session.execute( sa.delete( Provenance ).where( Provenance._id==e.provenance_id ) )
            session.commit()


# tools for making Image fixtures
class ImageCleanup:
    """Helper function that allows you to take an Image object with fake data and save it to disk.

    Also makes sure that the data is removed from disk when the object
    goes out of scope.

    Usage:
    >> im_clean = ImageCleanup.save_image(image)
    at end of test the im_clean goes out of scope and removes the file

    """

    @classmethod
    def save_image(cls, image, archive=True):
        """Save the image to disk, and return an ImageCleanup object.

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
                rng = np.random.default_rng()
                image.raw_data = rng.uniform(0, 100, size=(100, 100))
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
        try:
            # Just in case this image was used in a test and became an upstream, we
            #   need to clean out those entries.  (They won't automatically clean out
            #   because ondelete is RESTRICT for upstream_id in image_upstreams_associaton.)
            # We're trusting that whoever made the downstream will clean themselves up.
            with SmartSession() as sess:
                sess.execute( sa.text( "DELETE FROM image_upstreams_association "
                                       "WHERE upstream_id=:id" ),
                              { "id": self.image.id } )
                sess.commit()
            self.image.delete_from_disk_and_database()
        finally:
            pass


# idea taken from: https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
def generate_image_fixture(commit=True):

    @pytest.fixture
    def new_image(provenance_preprocessing):
        im = None
        exp = None
        exp = make_sim_exposure()
        add_file_to_exposure(exp)
        # Have to commit the exposure even if commit=False
        #  because otherwise tests that use this fixture
        #  would get an error about unknown exposure id
        #  when trying to commit the image.
        exp = commit_exposure(exp)
        exp.update_instrument()

        rng = np.random.default_rng()
        im = Image.from_exposure(exp, section_id=0)
        im.provenance_id = provenance_preprocessing.id
        im.data = np.float32(im.raw_data)  # this replaces the bias/flat preprocessing
        im.flags = rng.integers(0, 100, size=im.raw_data.shape, dtype=np.uint32)
        im.weight = np.full(im.raw_data.shape, 1.0, dtype=np.float32)
        im.format = 'fits'

        if commit:
            im.save()
            im.insert()

        yield im

        # Just in case this image got added as an upstream to anything,
        #   need to clean out the association table.  (See comment in
        #   ImageCleanup.__del__.)
        with SmartSession() as sess:
            sess.execute( sa.text( "DELETE FROM image_upstreams_association "
                                   "WHERE upstream_id=:id" ),
                          { "id": im.id } )
            sess.commit()

        # Clean up the exposure that got created; this will recusrively delete im as well
        if exp is not None:
            exp.delete_from_disk_and_database()

        # Cleanup provenances?  We seem to be OK with those lingering in the database at the end of tests.

    return new_image


# this will inject 9 images named sim_image1, sim_image2, etc.
for i in range(1, 10):
    globals()[f'sim_image{i}'] = generate_image_fixture()


# use this Image if you want the test to do the saving
sim_image_uncommitted = generate_image_fixture(commit=False)


@pytest.fixture
def sim_reference(provenance_preprocessing, provenance_extra):
    rng = np.random.default_rng()
    filter = rng.choice(list('grizY'))
    target = rnd_str(6)
    ra = rng.uniform(0, 360)
    dec = rng.uniform(-90, 90)
    images = []
    exposures = []

    for i in range(5):
        exp = make_sim_exposure()
        add_file_to_exposure(exp)
        exp = commit_exposure( exp )
        exp.filter = filter
        exp.target = target
        exp.project = "coadd_test"
        exp.ra = ra
        exp.dec = dec
        exposures.append( exp )

        exp.update_instrument()
        im = Image.from_exposure(exp, section_id=0)
        im.data = im.raw_data - np.median(im.raw_data)
        im.flags = rng.integers(0, 100, size=im.raw_data.shape, dtype=np.uint32)
        im.weight = np.full(im.raw_data.shape, 1.0, dtype=np.float32)
        im.provenance_id = provenance_preprocessing.id
        im.ra = ra
        im.dec = dec
        im.save()
        im.insert()
        images.append(im)

    ref_image = Image.from_images(images)
    ref_image.is_coadd = True
    ref_image.data = np.mean(np.array([im.data for im in images]), axis=0)
    ref_image.flags = np.max(np.array([im.flags for im in images]), axis=0)
    ref_image.weight = np.mean(np.array([im.weight for im in images]), axis=0)

    coaddprov = Provenance( process='coaddition',
                            code_version_id=provenance_extra.code_version_id,
                            parameters={},
                            upstreams=[provenance_extra],
                            is_testing=True )
    coaddprov.insert_if_needed()
    ref_image.provenance_id = coaddprov.id
    ref_image.save()
    ref_image.insert()

    ref = Reference()
    ref.image_id = ref_image.id
    refprov = Provenance(
        code_version_id=provenance_extra.code_version_id,
        process='referencing',
        parameters={'test_parameter': 'test_value'},
        upstreams=[provenance_extra],
        is_testing=True,
    )
    refprov.insert_if_needed()
    ref.provenance_id = refprov.id
    ref.instrument = 'Simulated'
    ref.section_id = 0
    ref.filter = filter
    ref.target = target
    ref.project = "coadd_test"
    ref.insert()

    yield ref

    if 'ref_image' in locals():
        ref_image.delete_from_disk_and_database()   # Should also delete the Reference

    # Deleting exposure should cascade to images
    for exp in exposures:
        exp.delete_from_disk_and_database()

    with SmartSession() as session:
        session.execute( sa.delete( Provenance ).where( Provenance._id.in_([coaddprov.id, refprov.id]) ) )
        session.commit()


@pytest.fixture
def sim_sources(sim_image1):
    num = 100
    rng = np.random.default_rng()
    x = rng.uniform(0, sim_image1.raw_data.shape[1], num)
    y = rng.uniform(0, sim_image1.raw_data.shape[0], num)
    flux = rng.uniform(0, 1000, num)
    flux_err = rng.uniform(0, 100, num)
    rhalf = np.abs(rng.normal(0, 3, num))

    data = np.array(
        [x, y, flux, flux_err, rhalf],
        dtype=([('x', 'f4'), ('y', 'f4'), ('flux', 'f4'), ('flux_err', 'f4'), ('rhalf', 'f4')])
    )
    s = SourceList(image_id=sim_image1.id, data=data, format='sepnpy')

    iprov = Provenance.get( sim_image1.provenance_id )
    prov = Provenance(
        code_version_id=iprov.code_version_id,
        process='extraction',
        parameters={'test_parameter': 'test_value'},
        upstreams=[ iprov ],
        is_testing=True,
    )
    prov.insert()
    s.provenance_id=prov.id

    s.save()
    s.insert()

    yield s
    # No need to delete, it will be deleted
    #   as a downstream of the exposure parent
    #   of sim_image1


@pytest.fixture
def sim_image_list_datastores(
        provenance_preprocessing,
        provenance_extraction,
        provenance_extra,
        fake_sources_data,
        ztf_filepaths_image_sources_psf
):
    rng = np.random.default_rng()
    ra = rng.uniform(30, 330)
    dec = rng.uniform(-30, 30)
    num = 5
    width = 1.0
    # use the ZTF files to generate a legitimate PSF (that has get_clip())
    # TODO: remove this ZTF dependence when doing issue #242
    _, _, _, _, psf, psfxml = ztf_filepaths_image_sources_psf

    # make images with all associated data products
    dses = []

    for i in range(num):
        ds = DataStore()
        exp = make_sim_exposure()
        ds.exposure = exp
        ds.exposure_id = exp.id
        add_file_to_exposure(exp)
        exp.update_instrument()

        im = Image.from_exposure(exp, section_id=0)
        im.data = np.float32(im.raw_data)  # this replaces the bias/flat preprocessing
        im.flags = rng.uniform(0, 1.01, size=im.raw_data.shape)  # 1% bad pixels
        im.flags = np.floor(im.flags).astype(np.uint16)
        im.weight = np.full(im.raw_data.shape, 4., dtype=np.float32)
        # TODO: remove ZTF depenedence and make a simpler PSF model (issue #242)

        # save the images to disk and database
        im.provenance_id = provenance_preprocessing.id

        # add some additional products we may need down the line
        ds.sources = SourceList(format='filter', data=fake_sources_data)
        # must randomize the sources data to get different MD5sum
        ds.sources.data['x'] += rng.normal(0, .1, len(fake_sources_data))
        ds.sources.data['y'] += rng.normal(0, .1, len(fake_sources_data))

        for j in range(len(ds.sources.data)):
            dx = ds.sources.data['x'][j] - ds.raw_data.shape[1] / 2
            dy = ds.sources.data['y'][j] - ds.raw_data.shape[0] / 2
            gaussian = make_gaussian(imsize=im.raw_data.shape, offset_x=dx, offset_y=dy, norm=1, sigma_x=width)
            gaussian *= rng.normal(ds.sources.data['flux'][j], ds.sources.data['flux_err'][j])
            im.data += gaussian

        im.save()
        ds.image = im

        ds.sources.provenance = provenance_extraction.id
        im.sources.image_id = im.id
        im.sources.save()
        ds.psf = PSF(filepath=str(psf.relative_to(im.local_path)), format='psfex')
        im.psf.load(download=False, psfpath=psf, psfxmlpath=psfxml)
        # must randomize to get different MD5sum
        ds.psf.data += rng.normal(0, 0.001, im.psf.data.shape)
        ds.psf.info = im.psf.info.replace('Emmanuel Bertin', uuid.uuid4().hex)

        ds.psf.fwhm_pixels = width * 2.3  # this is a fake value, but we need it to be there
        ds.psf.provenance_id = provenance_extraction.id
        ds.psf.sources_id = ds.sources.id
        im.psf.save()
        ds.zp = ZeroPoint()
        ds.zp.zp = rng.uniform(25, 30)
        ds.zp.dzp = rng.uniform(0.01, 0.1)
        ds.zp.aper_cor_radii = [1.0, 2.0, 3.0, 5.0]
        ds.zp.aper_cors = rng.normal(0, 0.1, len(im.zp.aper_cor_radii))
        ds.zp.provenance_id = provenance_extra.id
        ds.zp.sources_id = provenance_extra.id
        ds.wcs = WorldCoordinates()
        ds.wcs.wcs = WCS()
        # hack the pixel scale to reasonable values (0.3" per pixel)
        ds.wcs.wcs.wcs.pc = np.array([[0.0001, 0.0], [0.0, 0.0001]])
        ds.wcs.wcs.wcs.crval = np.array([ra, dec])
        ds.wcs.provenance_id = im.wcs.provenance.id
        ds.wcs.sources_id = ds.sources.id
        ds.wcs.save()

        ds.image.insert()
        ds.sources.insert()
        ds.psf.insert()
        ds.zp.insert()
        ds.wcs.insert()

    yield dses

    for ds in dses:
        ds.delete_everything()


@pytest.fixture
def provenance_subtraction(code_version, subtractor):
    with SmartSession() as session:
        prov = Provenance(
            code_version_id=code_version.id,
            process='subtraction',
            parameters=subtractor.pars.get_critical_pars(),
            upstreams=[],
            is_testing=True,
        )
        session.add(prov)
        session.commit()

    yield prov

    with SmartSession() as session:
        prov = session.merge(prov)
        if sa.inspect(prov).persistent:
            session.delete(prov)
            session.commit()


@pytest.fixture
def provenance_detection(code_version, detector):
    with SmartSession() as session:
        prov = Provenance(
            code_version_id=code_version.id,
            process='detection',
            parameters=detector.pars.get_critical_pars(),
            upstreams=[],
            is_testing=True,
        )
        session.add(prov)
        session.commit()

    yield prov

    with SmartSession() as session:
        prov = session.merge(prov)
        if sa.inspect(prov).persistent:
            session.delete(prov)
            session.commit()


@pytest.fixture
def provenance_cutting(code_version, cutter):
    with SmartSession() as session:
        prov = Provenance(
            code_version_id=code_version.id,
            process='cutting',
            parameters=cutter.pars.get_critical_pars(),
            upstreams=[],
            is_testing=True,
        )
        session.add(prov)
        session.commit()

    yield prov

    with SmartSession() as session:
        prov = session.merge(prov)
        if sa.inspect(prov).persistent:
            session.delete(prov)
            session.commit()


@pytest.fixture
def provenance_measuring(code_version, measurer):
    with SmartSession() as session:
        prov = Provenance(
            code_version_id=code_version.id,
            process='measuring',
            parameters=measurer.pars.get_critical_pars(),
            upstreams=[],
            is_testing=True,
        )
        session.add(prov)
        session.commit()

    yield prov

    with SmartSession() as session:
        prov = session.merge(prov)
        if sa.inspect(prov).persistent:
            session.delete(prov)
            session.commit()


@pytest.fixture
def fake_sources_data():
    num_x = 2
    num_y = 2
    size_x = DemoInstrument.fake_image_size_x
    size_y = DemoInstrument.fake_image_size_y

    x_list = np.linspace(size_x * 0.2, size_x * 0.8, num_x)
    y_list = np.linspace(size_y * 0.2, size_y * 0.8, num_y)

    rng = np.random.default_rng()
    xx = np.array([rng.normal(x, 1) for x in x_list for _ in y_list]).flatten()
    yy = np.array([rng.normal(y, 1) for _ in x_list for y in y_list]).flatten()
    ra = rng.uniform(0, 360)
    ra = [x / 3600 + ra for x in xx]  # assume pixel scale is 1"/pixel
    dec = rng.uniform(-10, 10)  # make it close to the equator to avoid having to consider cos(dec)
    dec = [y / 3600 + dec for y in yy]  # assume pixel scale is 1"/pixel
    flux = rng.uniform(1000, 2000, num_x * num_y)
    flux_err = rng.uniform(100, 200, num_x * num_y)
    dtype = [('x', 'f4'), ('y', 'f4'), ('ra', 'f4'), ('dec', 'f4'), ('flux', 'f4'), ('flux_err', 'f4')]
    data = np.empty(len(xx), dtype=dtype)
    data['x'] = xx
    data['y'] = yy
    data['ra'] = ra
    data['dec'] = dec
    data['flux'] = flux
    data['flux_err'] = flux_err

    yield data


# You will have trouble if you try to use this fixture
#   at the same time as sim_image_list_datastores,
#   because this one just adds things to the former's
#   elements.
@pytest.fixture
def sim_sub_image_list_datastores(
        sim_image_list_datastores,
        sim_reference,
        fake_sources_data,
        cutter,
        provenance_subtraction,
        provenance_detection,
        provenance_measuring,
):
    sub_dses = []
    for ds in sim_image_list_datastores:
        ds.reference = sim_reference
        ds.image.filter = ds.ref_image.filter
        ds.image.target = ds.ref_image.target
        ds.image.upsert()
        ds.sub_image = Image.from_ref_and_new( ds.ref_image, ds.image)
        assert ds.sub.is_sub == True
        # we are not actually doing any subtraction here, just copying the data
        # TODO: if we ever make the simulations more realistic we may want to actually do subtraction here
        ds.sub_image.data = ds.image.data.copy()
        ds.sub_image.flags = ds.image.flags.copy()
        ds.sub_image.weight = ds.image.weight.copy()
        ds.sub_image.insert()

        ds.detections = SourceList(format='filter', num_sources=len(fake_sources_data))
        ds.detections.provenance_id = provenance_detection.id
        ds.detections.image_id = ds.sub_image.id
        # must randomize the sources data to get different MD5sum
        rng = np.random.default_rng()
        fake_sources_data['x'] += rng.normal(0, 1, len(fake_sources_data))
        fake_sources_data['y'] += rng.normal(0, 1, len(fake_sources_data))
        ds.detections.data = fake_sources_data
        ds.detections.save()
        ds.detections.insert()

        # hack the images as though they are aligned
        # sim_reference.image.info['alignment_parameters'] = sub.provenance.parameters['alignment']
        # sim_reference.image.info['original_image_filepath'] = sim_reference.image.filepath
        # sim_reference.image.info['original_image_id'] = sim_reference.image.id
        # im.info['alignment_parameters'] = sub.provenance.parameters['alignment']
        # im.info['original_image_filepath'] = im.filepath
        # im.info['original_image_id'] = im.id

        # sub.aligned_images = [sim_reference.image, im]

        ds = cutter.run( ds )
        ds.cutouts.save()
        ds.cutouts.insert()

        sub_dses.append( ds )

    # The sim_image_list_datastores cleanup will clean our new mess up
    return sub_dses


# This fixture is broken until we do Issue #346
# @pytest.fixture
# def sim_lightcurves(sim_sub_image_list_datastores, measurer):
#     # a nested list of measurements, each one for a different part of the images,
#     # for each image contains a list of measurements for the same source
#     measurer.pars.thresholds['bad pixels'] = 100  # avoid losing measurements to random bad pixels
#     measurer.pars.deletion_thresholds['bad pixels'] = 100
#     measurer.pars.thresholds['offsets'] = 10  # avoid losing measurements to random offsets
#     measurer.pars.deletion_thresholds['offsets'] = 10
#     measurer.pars.association_radius = 5.0  # make it harder for random offsets to dis-associate the measurements
#     lightcurves = []

#     for ds in sim_sub_image_list_datastores:
#         ds = measurer.run( ds )
#         ds.save_and_commit()

#         # grab all the measurements associated with each Object
#         for m in ds.measurements:
#             m = session.merge(m)
#             lightcurves.append(m.object.measurements)  # <--- need to update with obejct measurement list

#     # sim_sub_image_list_datastores cleanup will clean up our mess too
#     return lightcurves
