import pytest
import os
import uuid
import copy

import numpy as np

import sqlalchemy as sa

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from models.base import SmartSession, Psycopg2Connection
from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference
from models.refset import RefSet
from models.instrument import get_instrument_instance

from pipeline.data_store import DataStore
from pipeline.top_level import Pipeline

from improc.simulator import Simulator

from util.util import patch_image_overlap_limits

from tests.conftest import rnd_str


def make_sim_exposure( filter=None, seed=None ):
    rng = np.random.default_rng( seed=seed )
    e = Exposure(
        filepath=f"Demo_test_{rnd_str(5)}.fits",
        section_id=0,
        exp_time=rng.integers(1, 4) * 10,  # 10 to 40 seconds
        mjd=rng.uniform(58000, 58500),
        filter=rng.choice(list('grizY')) if filter is None else filter,
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
def generate_exposure_fixture( seed=None ):
    @pytest.fixture
    def new_exposure():
        e = make_sim_exposure( seed=seed )
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
_rng = np.random.default_rng( seed=1611012017 )
for i in range(1, 10):
    globals()[f'sim_exposure{i}'] = generate_exposure_fixture( seed=_rng.integers( 0, 2**31 ) )


@pytest.fixture
def unloaded_exposure():
    e = make_sim_exposure( seed=1951785053 )
    return e


@pytest.fixture
def sim_exposure_filter_array():
    e = make_sim_exposure( seed=1072089705 )
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
    >> im_clean = ImageCleanup.save_image(image,seed=<integer>)
    at end of test the im_clean goes out of scope and removes the file

    """

    @classmethod
    def save_image(cls, image, archive=True, seed=None):
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
                rng = np.random.default_rng( seed=seed )
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
            self.image.delete_from_disk_and_database()
        finally:
            pass


# idea taken from: https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
def generate_image_fixture(commit=True, filter=None, seed=None ):

    @pytest.fixture
    def new_image(provenance_preprocessing):
        rng = np.random.default_rng( seed=seed )
        im = None
        exp = None
        exp = make_sim_exposure( filter=filter, seed=rng.integers(0, 2**31) )
        add_file_to_exposure(exp)
        # Have to commit the exposure even if commit=False
        #  because otherwise tests that use this fixture
        #  would get an error about unknown exposure id
        #  when trying to commit the image.
        exp = commit_exposure(exp)
        exp.update_instrument()

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

        # Clean up the exposure that got created; this will recusrively delete im as well
        if exp is not None:
            exp.delete_from_disk_and_database()

        # Cleanup provenances?  We seem to be OK with those lingering in the database at the end of tests.

    return new_image


# this will inject 9 images named sim_image1, sim_image2, etc.
for i in range(1, 10):
    globals()[f'sim_image{i}'] = generate_image_fixture( seed=1011888316 )

for i in range(1, 10):
    globals()[f'sim_image_r{i}'] = generate_image_fixture( filter='r', seed=869327863 )

# use this Image if you want the test to do the saving
sim_image_uncommitted = generate_image_fixture(commit=False, seed=1093069384)


# This fixture is currently broken
@pytest.fixture
def sim_reference(provenance_preprocessing, provenance_extraction, provenance_extra, fake_sources_data):
    rng = np.random.default_rng( seed=332378872 )
    filter = rng.choice(list('grizY'))
    target = rnd_str(6)
    ra = rng.uniform(0, 360)
    dec = rng.uniform(-90, 90)
    images = []
    zps = []
    exposures = []

    for i in range(5):
        exp = make_sim_exposure( seed=rng.integers(0, 2**31) )
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
        sl = SourceList( format='filter', num_sources=len(fake_sources_data) )
        sl.provenance_id = provenance_extraction.id
        sl.image_id = im.id
        # must randomize the sources data to get different MD5sum
        fake_sources_data['x'] += rng.normal(0, 1, len(fake_sources_data))
        fake_sources_data['y'] += rng.normal(0, 1, len(fake_sources_data))
        sl.data = fake_sources_data
        sl.save()
        sl.insert()
        wcs = WorldCoordinates()
        wcs.wcs = WCS()
        # hack the pixel scale to reasonable values (0.3" per pixel)
        wcs.wcs.wcs.pc = np.array([[0.0001, 0.0], [0.0, 0.0001]])
        wcs.wcs.wcs.crval = np.array([ra, dec])
        wcs.provenance_id = provenance_extra.id
        wcs.sources_id = sl.id
        wcs.save( image=im )
        wcs.insert()
        zp = ZeroPoint()
        zp.zp = rng.uniform( 25, 30 )
        zp.dzp = rng.uniform( 0.01, 0.1 )
        zp.aper_cor_radii = [1.0, 2.0, 3.0, 5.0]
        zp.aper_cors = rng.normal(0, 0.1, len(zp.aper_cor_radii))
        zp.provenance_id = provenance_extra.id
        zp.wcs_id = wcs.id
        zp.insert()
        zps.append( zp )


    ref_image = Image.from_image_zps(zps)
    ref_image.is_coadd = True
    ref_image.data = np.mean(np.array([im.data for im in images]), axis=0)
    ref_image.flags = np.max(np.array([im.flags for im in images]), axis=0)
    ref_image.weight = np.mean(np.array([im.weight for im in images]), axis=0)

    coaddprov = Provenance( process='coaddition',
                            code_version_id=provenance_extra.code_version_id,
                            parameters={},
                            upstreams=[provenance_extra,provenance_extraction],
                            is_testing=True )
    coaddprov.insert_if_needed()
    ref_image.provenance_id = coaddprov.id
    ref_image.save()
    ref_image.insert()

    # This is a garbage throwaway sources that doesn't have a file associated,
    #   just so the refs table has something to chew on.  If we ever need
    #   these sources for real, then we need to make them, which may be
    #   distressingly slow.

    sc = SourceList( format='sextrfits', image_id=ref_image.id, best_aper_num=-1, num_sources=0,
                     provenance_id=provenance_extraction.id, md5sum=uuid.uuid4(), filepath="foo" )
    # Likewise, garbage throwaway wcs and zp
    wcs = WorldCoordinates( wcs=WCS(), provenance_id=provenance_extra.id, md5sum=uuid.uuid4(), filepath="foo",
                            sources_id=sc.id )
    zp = ZeroPoint( wcs_id=wcs.id, zp=25., dzp=0.1, provenacne_id=provenance_extra.id )

    sc.insert()
    wcs.insert()
    zp.insert()


    ref = Reference()
    ref.zp_id = zp.id
    refprov = Provenance(
        code_version_id=provenance_extra.code_version_id,
        process='referencing',
        parameters={'test_parameter': 'test_value'},
        upstreams=[provenance_extra],
        is_testing=True,
    )
    refprov.insert_if_needed()
    ref.provenance_id = refprov.id
    ref.insert()

    yield ref

    if 'ref_image' in locals():
        ref_image.delete_from_disk_and_database()   # Should also delete the Reference

    # Deleting exposure should cascade to images
    for exp in exposures:
        exp.delete_from_disk_and_database()

    with SmartSession() as session:
        session.execute( sa.delete( SourceList ).where( SourceList._id==sc.id ) )
        session.execute( sa.delete( Provenance ).where( Provenance._id.in_([coaddprov.id, refprov.id]) ) )
        session.commit()


@pytest.fixture
def sim_sources(sim_image1):
    num = 100
    rng = np.random.default_rng( seed=61532483 )
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


# ======================================================================
# The following fixtures create simulated images using Guy's image
# simulator.  It makes a reference image, and then six new images which
# are just the reference image plus added noise, plus added sources.
# They're a way-too-idealized case: the images are all already aligned,
# and the images have the same PSF.  So, they're not good tests of
# everything, but they're suitable for tests of data structures,
# database searching, and as very easy tests of things like source
# finding.

@pytest.fixture( scope='session' )
def sim_lightcurve_image_parameters():
    instr = get_instrument_instance( 'DemoInstrument' )
    ra = 123.45678
    dec = -3.14159
    wid = 256
    minra = ra - wid/2. * instr.pixel_scale / 3600. / np.cos( dec * np.pi / 180. )
    maxra = ra + wid/2. * instr.pixel_scale / 3600. / np.cos( dec * np.pi / 180. )
    mindec = dec - wid/2. * instr.pixel_scale / 3600.
    maxdec = dec + wid/2. * instr.pixel_scale / 3600.

    # Put this parameter in just so that this provenance won't be the same as anybody else's
    #  "test_image" provenance.  (Don't want to have things like unexpected references
    #  showing up, etc.)
    improv = Provenance( process='test_image', parameters={ 'fixture': 'sim_lightcurve_image_parameters' } )
    improv.insert_if_needed()

    imageinfo = { 'prov': improv,
                  'size': wid,
                  'refmjd': 60000.,
                  'refexptime': 600.,
                  'refmjdend': 60000. + 600. / 3600. / 24.,
                  'refskye-': 200.,
                 }
    imageargs = { 'ra': ra,
                  'dec': dec,
                  'minra': minra,
                  'maxra': maxra,
                  'mindec': mindec,
                  'maxdec': maxdec,
                  'ra_corner_00': minra,
                  'ra_corner_01': minra,
                  'ra_corner_10': maxra,
                  'ra_corner_11': maxra,
                  'dec_corner_00': mindec,
                  'dec_corner_01': maxdec,
                  'dec_corner_10': mindec,
                  'dec_corner_11': maxdec,
                  'instrument': 'DemoInstrument',
                  'telescope': 'DemoTelescope',
                  'format': 'fits',
                  'type': 'Sci',
                  'filter': 'r',
                  'section_id': '0',
                  'project': 'tests',
                  'target': 'gratuitous',
                  'airmass': 1.0,
                  'preprocessing_done': True,
                  'preproc_bitflag': 0x7f,
                  'provenance_id': improv.id,
                 }

    yield imageinfo, imageargs

    with Psycopg2Connection() as con:
        cursor = con.cursor()
        cursor.execute( "DELETE FROM provenances WHERE _id=%(id)s", { 'id': improv.id } )
        con.commit()


@pytest.fixture( scope="session" )
def sim_lightcurve_persistent_sources():
    # These positions were chosen visually to be near galaxies on the
    #   generated image.  If the simulator code is edited at all, it's
    #   possible that the image generated with the same random seed
    #   will look nothing like the image that was used to choose these
    #   positions.  (But, the tests should still work....)
    sources = [
        { 'ra': 123.46525473,   # At x=56.2, y=76.9
          'dec': -3.14735274,
          'mjdmaxoff': 29.5,
          'sigmadays': 5.,
          'maxflux': 10000.
         },
        { 'ra': 123.45663172,   # At x=128.8, y=229.3
          'dec': -3.12999611,
          'mjdmaxoff': 42.2,
          'sigmadays': 10.,
          'maxflux': 2000.,
         },
        { 'ra': 123.45339241,
          'dec': -3.14136222,   # At x=157.2, y=129.5
          'mjdmaxoff': 49.7,
          'sigmadays': 18.,
          'maxflux': 3000.,
          }
    ]

    return sources


@pytest.fixture( scope="session" )
def sim_lightcurve_wcs_headers( sim_lightcurve_image_parameters ):
    imageinfo, imageargs = sim_lightcurve_image_parameters
    instr = get_instrument_instance( 'DemoInstrument' )
    return { 'CTYPE1': 'RA---TAN',
             'CTYPE2': 'DEC--TAN',
             'CRPIX1': imageinfo['size'] / 2. + 0.5,    # 1-offset center of image
             'CRPIX2': imageinfo['size'] / 2. + 0.5,
             'CDELT1': -instr.pixel_scale / 3600.,   #  Should there be a cos(dec) here and on CD1_1?
             'CDELT2': instr.pixel_scale / 3600.,
             'CRVAL1': imageargs['ra'],
             'CRVAL2': imageargs['dec'],
             'CD1_1': -instr.pixel_scale / 3600.,
             'CD1_2': 0.,
             'CD2_1': 0.,
             'CD2_2': instr.pixel_scale / 3600.
            }


@pytest.fixture( scope="session" )
def sim_lightcurve_pipeline_parameters():
    return { 'pipeline': { 'provenance_tag': 'sim_lightcurve' },
             'subtraction': { 'refset': 'sim_lightcurve_reference' },
             'extraction': { 'backgrounding': { 'format': 'map',
                                                'method': 'sep',
                                                'poly_order': 1,
                                                'box_size': 64,
                                                'filt_size': 3,
                                               },
                            },
            }


# This is a datastore of unsaved data products.  Because the image is
# 256×256, it does't use up too terribly much memory, so hopefully it
# won't be a big deal to have this sitting around for the whole session.
#
# As a side effect, this will create several provenances.
#
# The simulator isn't that slow, but it and psfex do take a little bit of time.
@pytest.fixture( scope="session" )
def sim_lightcurve_reference_image_unsaved( sim_lightcurve_image_parameters, sim_lightcurve_pipeline_parameters,
                                            sim_lightcurve_wcs_headers ):
    imageinfo, imageargs = sim_lightcurve_image_parameters
    parms = sim_lightcurve_pipeline_parameters.copy()
    parms['provenance_tag'] = 'sim_lightcurve_reference'
    pip = Pipeline( **parms )

    instr = get_instrument_instance( 'DemoInstrument' )
    s = Simulator( image_size_x=imageinfo['size'],
                   gain_mean=instr.gain,
                   star_number=60,
                   galaxy_number=100,
                   background_mean=imageinfo['refskye-'],
                   background_std=np.sqrt( imageinfo['refskye-'] ),
                   star_min_flux=400.,
                   star_flux_power_law=1.5,
                   galaxy_min_flux=400.,
                   galaxy_flux_power_law=1.8,
                   bias_mean=0.,
                   bias_std=0.,
                   dark_current=0.,
                   random_seed=42 )
    s.make_image()
    image = np.array( s.image, dtype=np.float32 )


    img = Image( mjd=imageinfo['refmjd'], exp_time=imageinfo['refexptime'], end_mjd=imageinfo['refmjdend'],
                 **imageargs )
    img.calculate_coordinates()
    # It seems that scamp fails if it doesn't have at least an approximate wcs already
    #  in the header (?).  This makes me very sad.  May be time to write custom
    #  WCS solving software.  (Or resurrect what we used in the SCP 30 years ago?)
    #  The hard part is figuring out all the WCS nonlinear transformations....
    #  It would take a long time.  We want scamp to work.
    img._header = fits.Header( sim_lightcurve_wcs_headers )
    img.data = image
    img.weight = 1. / ( instr.gain * image )
    img.flags = np.zeros_like( image, dtype=np.int16 )
    img.filepath = img.invent_filepath()

    ds = DataStore( img )
    # Running the pipeline extraction manually, because we can't really
    # run WCS or ZP as these images don't match anything in catalogs; we
    # have to hack the WCS and the ZP.
    pip.make_provenance_tree( ds, ok_no_ref_prov=True )
    ds = pip.extractor.run( ds )

    # Make a fake WCS
    ds.wcs = WorldCoordinates()
    ds.wcs.wcs = WCS( sim_lightcurve_wcs_headers )
    ds.wcs.sources_id = ds.sources.id
    # This is a cheat, as we didn't really use the params in the provenance, but, whatevs
    ds.wcs.provenance_id = ds.prov_tree['astrocal'].id

    # Likewise, make a fake zeropoint, cheating again on provenance
    # (Re: number of stars, there just aren't that many not-deblended
    # high-sn stars in the image :( ).
    apercors = ds.sources.calc_aper_cors( min_stars=10 )
    ds.zp = ZeroPoint( wcs_id=ds.wcs.id, zp=27.50, dzp=0.02,
                       aper_cor_radii=ds.sources.aper_rads, aper_cors=apercors,
                       provenance_id=ds.prov_tree['photocal'].id )

    yield ds

    # This shouldn't be necessary; ideally anything that called this and
    # saved stuff cleaned up after itself.  But, just to be sure....

    ds.delete_everything()
    with Psycopg2Connection() as conn:
        cursor = conn.cursor()
        cursor.execute( "DELETE FROM provenance_tags WHERE tag='sim_lightcurve_reference'" )
        conn.commit()


# Function used by the next two fixtures
def _do_sim_lightcurve_reference( ds ):
    ds.save_and_commit()
    refprov = Provenance( process='referencing', code_version_id=Provenance.get_code_version('referencing').id,
                          parameters={ 'description': 'sim_lightcurve fixture' } )
    refprov.insert_if_needed()
    ref = Reference( zp_id=ds.zp.id, provenance_id=refprov.id )
    ref.insert()
    refset = RefSet( name='sim_lightcurve_reference', provenance_id=refprov.id )
    refset.insert()

    return ref, ds


# This saves the reference image and dataproducts, and creates a Reference object in the database
@pytest.fixture
def sim_lightcurve_reference( sim_lightcurve_reference_image_unsaved ):
    ref = None
    try:
        ref, ds = _do_sim_lightcurve_reference( sim_lightcurve_reference_image_unsaved )
        yield ref, ds
    finally:
        if ref is not None:
            with Psycopg2Connection() as conn:
                cursor = conn.cursor()
                cursor.execute( "DELETE FROM refsets WHERE name='sim_lightcurve_reference'" )
                cursor.execute( "DELETE FROM refs WHERE _id=%(id)s", { 'id': ref.id } )
                conn.commit()

        sim_lightcurve_reference_image_unsaved.delete_everything( do_not_clear=True )


# Same as previous fixture, but with module scope for efficiency
@pytest.fixture( scope='module' )
def sim_lightcurve_reference_module(  sim_lightcurve_reference_image_unsaved ):
    ref = None
    try:
        ref, ds = _do_sim_lightcurve_reference( sim_lightcurve_reference_image_unsaved )
        yield ref, ds
    finally:
        if ref is not None:
            with Psycopg2Connection() as conn:
                cursor = conn.cursor()
                cursor.execute( "DELETE FROM refsets WHERE name='sim_lightcurve_reference'" )
                cursor.execute( "DELETE FROM refs WHERE _id=%(id)s", { 'id': ref.id } )
                conn.commit()

        sim_lightcurve_reference_image_unsaved.delete_everything( do_not_clear=True )


# Usually don't use this fixture directly, use the sim_lightcurve_new_ds_factory fixture
@pytest.fixture( scope='session' )
def sim_lightcurve_image_datastore_maker_factory( sim_lightcurve_image_parameters, sim_lightcurve_pipeline_parameters,
                                                  sim_lightcurve_reference_image_unsaved ):
    refds = sim_lightcurve_reference_image_unsaved
    dsentocleanup = []

    def maker( image ):
        ds = DataStore( image )
        pipparams = copy.deepcopy( sim_lightcurve_pipeline_parameters )
        # The psf is the same as the reference psf, so pass that to avoid running psfex again.
        # (We also then need to pass input_psf to extractor.run.)
        pipparams['extraction'].update( { 'measure_psf': False } )
        pip = Pipeline( **pipparams )
        ds.prov_tree = pip.make_provenance_tree( ds, no_provtag=True, ok_no_ref_prov=True )

        ds = pip.extractor.run( ds, input_psf=refds.psf )
        ds.sources.save( image=ds.image )
        ds.sources.insert()
        # Fix the psf sources_id now that we have a sources
        ds.psf.sources_id = ds.sources.id
        ds.psf.save( image=ds.image, sources=ds.sources )
        ds.psf.insert()
        ds.bg.save( image=ds.image, sources=ds.sources )
        ds.bg.insert()

        # The WCS is the same as the reference image wcs
        # Make a fake WCS, because these sources are simulated so don't match the sky
        ds.wcs = WorldCoordinates()
        ds.wcs.wcs = refds.wcs.wcs
        ds.wcs.sources_id = ds.sources.id
        ds.wcs.provenance_id = ds.prov_tree['astrocal'].id
        ds.wcs.save( image=ds.image )
        ds.wcs.insert()

        # Likewise, make a fake zeropoint, cheating again on provenance
        # (Sad about the number of stars for aperture correction; probably
        # means the star/galaxy discriminator failed?  Look into this-- Issue #513.)
        apercors = ds.sources.calc_aper_cors( min_stars=4 )
        ds.zp = ZeroPoint( wcs_id=ds.wcs.id, zp=27.50, dzp=0.02,
                           aper_cor_radii=ds.sources.aper_rads, aper_cors=apercors,
                           provenance_id=ds.prov_tree['photocal'].id )
        ds.zp.insert()

        dsentocleanup.append( ds )
        return ds

    yield maker

    # Ideally, things that use this fixture will clean up their own
    # datastores, so none of this cleanup here should be necessary.
    # However, do it just to be anal.
    for ds in dsentocleanup:
        ds.delete_everything()


# This function is used by the next two fixtures
def _do_sim_lightcurve_new_ds_factory( imageinfo, imageargs, refds, sources, wcshdrs, maker, dsentodel ):
    def add_source_to_data( data, x0, y0, flux, seesig, patchwid ):
        ix0 = int( np.floor( x0 ) )
        iy0 = int( np.floor( y0 ) )
        patchx, patchy = np.meshgrid( np.arange(patchwid, dtype=np.float32),
                                      np.arange(patchwid, dtype=np.float32) )
        patchx += x0 - float( patchwid // 2 )
        patchy += y0 - float( patchwid // 2 )
        patch = ( ( flux / ( 2 * np.pi * seesig**2 ) ) *
                  np.exp( -( (ix0 - patchx)**2 + (iy0 - patchy)**2 ) / ( 2. * seesig**2 ) ) )

        # TODO : scatter by poisson noise?
        # Also update image weight?  (Will have to pass it in that case.)

        ( px0, px1, py0, py1 ), ( lx, hx, ly, hy ) = patch_image_overlap_limits( patchwid, ix0, iy0, data.shape )
        data[ ly:hy, lx:hx ] += patch[ py0:py1, px0:px1 ]


    def make_new_ds( mjdoff, extranoise=25., extrarandsourcefluxes=[], random_seed=64738 ):
        instr = get_instrument_instance( 'DemoInstrument' )
        mjd = imageinfo['refmjd'] + mjdoff
        rng = np.random.default_rng( seed=random_seed )
        data = refds.image.data + instr.gain * rng.normal( 0., extranoise, size=refds.image.data.shape )
        weight = 1. / ( ( 1. / refds.image.weight ) + ( instr.gain  * extranoise ) ** 2 )
        flags = refds.image.flags.copy()

        # Add sources.  Not bothering to poisson scatter them.  Oh well.
        seesig = refds.psf.fwhm_pixels / 2.35482
        patchwid = int( np.ceil( 6. * refds.psf.fwhm_pixels ) )
        patchwid += 1 if patchwid % 2 == 0 else 0

        for source in sources:
            flux = source['maxflux'] * np.exp( -( mjdoff - source['mjdmaxoff'] )**2 / ( 2 * source['sigmadays']**2 ) )
            tmp = refds.wcs.wcs.world_to_pixel( SkyCoord( source['ra'], source['dec'], unit='deg' ) )
            # I HATE that astropy/numpy now returns array(1.5) instead of just a regular float 1.5
            x = float( tmp[0] )
            y = float( tmp[1] )
            add_source_to_data( data, x, y, flux, seesig, patchwid )

        # Add random sources
        for flux in extrarandsourcefluxes:
            x = rng.uniform( 0., data.shape[1] )
            y = rng.uniform( 0., data.shape[0] )
            add_source_to_data( data, x, y, flux, seesig, patchwid )

        image = Image( mjd=mjd, exp_time=200., end_mjd=mjd+200./3600./24., **imageargs )
        image.data = data
        image.weight = weight
        image.flags = flags
        # Add some jitter to the WCS... I think scamp can still cope.
        # (Doing it here to test, but then just left it in.)
        wcsparm = wcshdrs.copy()
        wcsparm['CRVAL1'] += rng.normal( 0., 20./3600. )
        wcsparm['CRVAL2'] += rng.normal( 0., 20./3600. )
        image._header = fits.Header( wcsparm )
        image.save()
        image.insert()

        ds = maker( image )
        dsentodel.append( ds )
        return ds

    return make_new_ds


# This is the fixture to run to actually get new image datastores...
# But also probably don't run this directly, use the sim_lightcurve_news fixture!
@pytest.fixture
def sim_lightcurve_new_ds_factory( sim_lightcurve_image_parameters,
                                   sim_lightcurve_wcs_headers,
                                   sim_lightcurve_reference,
                                   sim_lightcurve_image_datastore_maker_factory,
                                   sim_lightcurve_persistent_sources
                                  ):
    imageinfo, imageargs = sim_lightcurve_image_parameters
    _, refds = sim_lightcurve_reference
    sources = sim_lightcurve_persistent_sources
    wcshdrs = sim_lightcurve_wcs_headers
    maker = sim_lightcurve_image_datastore_maker_factory
    dsentodel = []

    yield _do_sim_lightcurve_new_ds_factory( imageinfo, imageargs, refds, sources, wcshdrs, maker, dsentodel )

    for ds in dsentodel:
        ds.delete_everything()
    with Psycopg2Connection() as conn:
        cursor = conn.cursor()
        cursor.execute( "DELETE FROM provenance_tags WHERE tag='sim_lightcurve'" )
        conn.commit()


# Same as previous fixture, but module scope for efficiency.
# Don't use this, use sim_lightcurve_news_module.
@pytest.fixture( scope="module" )
def sim_lightcurve_new_ds_factory_module( sim_lightcurve_image_parameters,
                                          sim_lightcurve_wcs_headers,
                                          sim_lightcurve_reference_module,
                                          sim_lightcurve_image_datastore_maker_factory,
                                          sim_lightcurve_persistent_sources
                                         ):
    imageinfo, imageargs = sim_lightcurve_image_parameters
    _, refds = sim_lightcurve_reference_module
    sources = sim_lightcurve_persistent_sources
    wcshdrs = sim_lightcurve_wcs_headers
    maker = sim_lightcurve_image_datastore_maker_factory
    dsentodel = []

    yield _do_sim_lightcurve_new_ds_factory( imageinfo, imageargs, refds, sources, wcshdrs, maker, dsentodel )

    for ds in dsentodel:
        ds.delete_everything()
    with Psycopg2Connection() as conn:
        cursor = conn.cursor()
        cursor.execute( "DELETE FROM provenance_tags WHERE tag='sim_lightcurve'" )
        conn.commit()


# This fixture still takes a minute or two to run.  Too much time is
#   watsed determining the psf on warped images... as the images won't
#   be warped, really, so we could have just used the input psf.  But,
#   oh well.  Code for the general case, watch it be inefficient in
#   a specific case.
@pytest.fixture
def sim_lightcurve_news( sim_lightcurve_new_ds_factory ):
    rng = np.random.default_rng( seed=221084103 )

    dses = []
    mjdoffs = np.array( [ 30., 32., 37., 40., 45., 55. ] )
    for mjdoff in mjdoffs:
        nextrafluxes = rng.integers( 1, 4 )
        extrafluxes = rng.uniform( 2000., 20000., size=nextrafluxes )
        dses.append( sim_lightcurve_new_ds_factory( mjdoff, random_seed=rng.integers( 0, 2**31 ),
                                                    extrarandsourcefluxes=extrafluxes ) )

    # sim_lightcurve_new_ds_factory handles cleanup
    return dses


# Same as previous fixture, but module scope
@pytest.fixture( scope='module' )
def sim_lightcurve_news_module( sim_lightcurve_new_ds_factory_module ):
    rng = np.random.default_rng( seed=221084103 )

    dses = []
    mjdoffs = np.array( [ 30., 32., 37., 40., 45., 55. ] )
    for mjdoff in mjdoffs:
        nextrafluxes = rng.integers( 1, 4 )
        extrafluxes = rng.uniform( 2000., 20000., size=nextrafluxes )
        dses.append( sim_lightcurve_new_ds_factory_module( mjdoff, random_seed=rng.integers( 0, 2**31 ),
                                                           extrarandsourcefluxes=extrafluxes ) )

    # sim_lightcurve_new_ds_factory handles cleanup
    return dses


@pytest.fixture
def sim_lightcurve_complete_dses( sim_lightcurve_reference, sim_lightcurve_news,
                                  sim_lightcurve_pipeline_parameters ):
    ref, refds = sim_lightcurve_reference
    newdsen = []
    pips = []
    for ds in sim_lightcurve_news:
        pip = Pipeline( **sim_lightcurve_pipeline_parameters )
        ds = pip.run( ds )
        newdsen.append( ds )
        pips.append( pip )

    return ref, refds, newdsen, pips


# Same as previous fixture, but module scope
@pytest.fixture( scope="module" )
def sim_lightcurve_complete_dses_module( sim_lightcurve_reference_module, sim_lightcurve_news_module,
                                         sim_lightcurve_pipeline_parameters ):
    ref, refds = sim_lightcurve_reference_module
    newdsen = []
    pips = []
    for ds in sim_lightcurve_news_module:
        pip = Pipeline( **sim_lightcurve_pipeline_parameters )
        ds = pip.run( ds )
        newdsen.append( ds )
        pips.append( pip )

    return ref, refds, newdsen, pips


@pytest.fixture
def sim_lightcurve_one_new( sim_lightcurve_new_ds_factory ):
    rng = np.random.default_rng( seed=1708950305 )
    nextrafluxes = rng.integers( 1, 4 )
    extrafluxes = rng.uniform( 2000., 20000., size=nextrafluxes )
    ds = sim_lightcurve_new_ds_factory( 30., random_seed=rng.integers( 0, 2**31 ),
                                        extrarandsourcefluxes=extrafluxes )
    return ds


@pytest.fixture( scope="module" )
def sim_lightcurve_one_new_module( sim_lightcurve_new_ds_factory_module ):
    rng = np.random.default_rng( seed=1708950305 )
    nextrafluxes = rng.integers( 1, 4 )
    extrafluxes = rng.uniform( 2000., 20000., size=nextrafluxes )
    ds = sim_lightcurve_new_ds_factory_module( 30., random_seed=rng.integers( 0, 2**31 ),
                                               extrarandsourcefluxes=extrafluxes )
    return ds


@pytest.fixture
def sim_lightcurve_one_complete_ds( sim_lightcurve_reference, sim_lightcurve_one_new,
                                    sim_lightcurve_pipeline_parameters ):
    ref, refds = sim_lightcurve_reference
    ds = sim_lightcurve_one_new
    pip = Pipeline( **sim_lightcurve_pipeline_parameters )
    ds = pip.run( ds )
    return ref, refds, ds, pip


@pytest.fixture( scope="module" )
def sim_lightcurve_one_complete_ds_module( sim_lightcurve_reference, sim_lightcurve_one_new_module,
                                           sim_lightcurve_pipeline_parameters ):
    ref, refds = sim_lightcurve_reference
    ds = sim_lightcurve_one_new
    pip = Pipeline( **sim_lightcurve_pipeline_parameters )
    ds = pip.run( ds )
    return ref, refds, ds, pip
