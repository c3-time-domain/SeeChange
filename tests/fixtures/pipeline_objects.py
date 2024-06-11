import os
import warnings
import shutil
import pytest

import numpy as np

import sqlalchemy as sa

import sep

from models.base import SmartSession, FileOnDiskMixin
from models.provenance import Provenance
from models.enums_and_bitflags import BitFlagConverter
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements

from pipeline.data_store import DataStore
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from pipeline.coaddition import Coadder
from pipeline.subtraction import Subtractor
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer
from pipeline.top_level import Pipeline

from util.logger import SCLogger
from util.cache import copy_to_cache, copy_list_to_cache, copy_from_cache, copy_list_from_cache

from improc.bitmask_tools import make_saturated_flag

@pytest.fixture(scope='session')
def preprocessor_factory(test_config):

    def make_preprocessor():
        prep = Preprocessor(**test_config.value('preprocessing'))
        prep.pars._enforce_no_new_attrs = False
        prep.pars.test_parameter = prep.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        prep.pars._enforce_no_new_attrs = True

        return prep

    return make_preprocessor


@pytest.fixture
def preprocessor(preprocessor_factory):
    return preprocessor_factory()


@pytest.fixture(scope='session')
def extractor_factory(test_config):

    def make_extractor():
        extr = Detector(**test_config.value('extraction'))
        extr.pars._enforce_no_new_attrs = False
        extr.pars.test_parameter = extr.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        extr.pars._enforce_no_new_attrs = True

        return extr

    return make_extractor


@pytest.fixture
def extractor(extractor_factory):
    return extractor_factory()


@pytest.fixture(scope='session')
def astrometor_factory(test_config):

    def make_astrometor():
        astrom = AstroCalibrator(**test_config.value('astro_cal'))
        astrom.pars._enforce_no_new_attrs = False
        astrom.pars.test_parameter = astrom.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        astrom.pars._enforce_no_new_attrs = True

        return astrom

    return make_astrometor


@pytest.fixture
def astrometor(astrometor_factory):
    return astrometor_factory()


@pytest.fixture(scope='session')
def photometor_factory(test_config):

    def make_photometor():
        photom = PhotCalibrator(**test_config.value('photo_cal'))
        photom.pars._enforce_no_new_attrs = False
        photom.pars.test_parameter = photom.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        photom.pars._enforce_no_new_attrs = True

        return photom

    return make_photometor


@pytest.fixture
def photometor(photometor_factory):
    return photometor_factory()


@pytest.fixture(scope='session')
def coadder_factory(test_config):

    def make_coadder():

        coadd = Coadder(**test_config.value('coaddition.coaddition'))
        coadd.pars._enforce_no_new_attrs = False
        coadd.pars.test_parameter = coadd.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        coadd.pars._enforce_no_new_attrs = True

        return coadd

    return make_coadder


@pytest.fixture
def coadder(coadder_factory):
    return coadder_factory()


@pytest.fixture(scope='session')
def subtractor_factory(test_config):

    def make_subtractor():
        sub = Subtractor(**test_config.value('subtraction'))
        sub.pars._enforce_no_new_attrs = False
        sub.pars.test_parameter = sub.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        sub.pars._enforce_no_new_attrs = True

        return sub

    return make_subtractor


@pytest.fixture
def subtractor(subtractor_factory):
    return subtractor_factory()


@pytest.fixture(scope='session')
def detector_factory(test_config):

    def make_detector():
        det = Detector(**test_config.value('detection'))
        det.pars._enforce_no_new_attrs = False
        det.pars.test_parameter = det.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        det.pars._enforce_no_new_attrs = True

        return det

    return make_detector


@pytest.fixture
def detector(detector_factory):
    return detector_factory()


@pytest.fixture(scope='session')
def cutter_factory(test_config):

    def make_cutter():
        cut = Cutter(**test_config.value('cutting'))
        cut.pars._enforce_no_new_attrs = False
        cut.pars.test_parameter = cut.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        cut.pars._enforce_no_new_attrs = True

        return cut

    return make_cutter


@pytest.fixture
def cutter(cutter_factory):
    return cutter_factory()


@pytest.fixture(scope='session')
def measurer_factory(test_config):

    def make_measurer():
        meas = Measurer(**test_config.value('measuring'))
        meas.pars._enforce_no_new_attrs = False
        meas.pars.test_parameter = meas.pars.add_par(
            'test_parameter', 'test_value', str, 'parameter to define unique tests', critical=True
        )
        meas.pars._enforce_no_new_attrs = True

        return meas

    return make_measurer


@pytest.fixture
def measurer(measurer_factory):
    return measurer_factory()


@pytest.fixture(scope='session')
def pipeline_factory(
        preprocessor_factory,
        extractor_factory,
        astrometor_factory,
        photometor_factory,
        subtractor_factory,
        detector_factory,
        cutter_factory,
        measurer_factory,
        test_config,
):
    def make_pipeline():
        p = Pipeline(**test_config.value('pipeline'))
        p.preprocessor = preprocessor_factory()
        p.extractor = extractor_factory()
        p.astro_cal = astrometor_factory()
        p.photo_cal = photometor_factory()
        p.subtractor = subtractor_factory()
        p.detector = detector_factory()
        p.cutter = cutter_factory()
        p.measurer = measurer_factory()
        return p

    return make_pipeline


@pytest.fixture
def pipeline_for_tests(pipeline_factory):
    return pipeline_factory()


@pytest.fixture(scope='session')
def datastore_factory(data_dir, pipeline_factory):
    """Provide a function that returns a datastore with all the products based on the given exposure and section ID.

    To use this data store in a test where new data is to be generated,
    simply change the pipeline object's "test_parameter" value to a unique
    new value, so the provenance will not match and the data will be regenerated.

    If "save_original_image" is True, then a copy of the image before
    going through source extraction, WCS, etc. will be saved along side
    the image, with ".image.fits.original" appended to the filename;
    this path will be in ds.path_to_original_image.  In this case, the
    thing that calls this factory must delete that file when done.

    EXAMPLE
    -------
    extractor.pars.test_parameter = uuid.uuid().hex
    extractor.run(datastore)
    assert extractor.has_recalculated is True

    """
    def make_datastore(
            *args,
            cache_dir=None,
            cache_base_name=None,
            session=None,
            overrides={},
            augments={},
            bad_pixel_map=None,
            save_original_image=False
    ):
        code_version = args[0].provenance.code_version
        ds = DataStore(*args)  # make a new datastore

        if ( cache_dir is not None ) and ( cache_base_name is not None ) and ( not os.getenv( "LIMIT_CACHE_USE" ) ):
            ds.cache_base_name = os.path.join(cache_dir, cache_base_name)  # save this for testing purposes

        p = pipeline_factory()

        # allow calling scope to override/augment parameters for any of the processing steps
        p.override_parameters(**overrides)
        p.augment_parameters(**augments)

        with SmartSession(session) as session:
            code_version = session.merge(code_version)
            if ds.image is not None:  # if starting from an externally provided Image, must merge it first
                ds.image = ds.image.merge_all(session)

            ############ preprocessing to create image ############
            if (   ( not os.getenv( "LIMIT_CACHE_USAGE" ) ) and
                   ( ds.image is None ) and ( cache_dir is not None ) and ( cache_base_name is not None )
                ):
                # check if preprocessed image is in cache
                cache_name = cache_base_name + '.image.fits.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    SCLogger.debug('loading image from cache. ')
                    ds.image = copy_from_cache(Image, cache_dir, cache_name)
                    # assign the correct exposure to the object loaded from cache
                    if ds.exposure_id is not None:
                        ds.image.exposure_id = ds.exposure_id
                    if ds.exposure is not None:
                        ds.image.exposure = ds.exposure

                    # Copy the original image from the cache if requested
                    if save_original_image:
                        ds.path_to_original_image = ds.image.get_fullpath()[0] + '.image.fits.original'
                        cache_path = os.path.join(cache_dir, ds.image.filepath + '.image.fits.original')
                        shutil.copy2( cache_path, ds.path_to_original_image )

                    # add the preprocessing steps from instrument (TODO: remove this as part of Issue #142)
                    preprocessing_steps = ds.image.instrument_object.preprocessing_steps
                    prep_pars = p.preprocessor.pars.get_critical_pars()
                    prep_pars['preprocessing_steps'] = preprocessing_steps

                    upstreams = [ds.exposure.provenance] if ds.exposure is not None else []  # images without exposure
                    prov = Provenance(
                        code_version=code_version,
                        process='preprocessing',
                        upstreams=upstreams,
                        parameters=prep_pars,
                        is_testing=True,
                    )
                    prov = session.merge(prov)

                    # if Image already exists on the database, use that instead of this one
                    existing = session.scalars(sa.select(Image).where(Image.filepath == ds.image.filepath)).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.image).mapper.columns.keys():
                            value = getattr(ds.image, key)
                            if (
                                    key not in ['id', 'image_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.image = existing  # replace with the existing row
                    ds.image.provenance = prov

                    # make sure this is saved to the archive as well
                    ds.image.save(verify_md5=False)

            if ds.image is None:  # make the preprocessed image
                SCLogger.debug('making preprocessed image. ')
                ds = p.preprocessor.run(ds)
                ds.image.provenance.is_testing = True
                if bad_pixel_map is not None:
                    ds.image.flags |= bad_pixel_map
                    if ds.image.weight is not None:
                        ds.image.weight[ds.image.flags.astype(bool)] = 0.0

                # flag saturated pixels, too (TODO: is there a better way to get the saturation limit? )
                mask = make_saturated_flag(ds.image.data, ds.image.instrument_object.saturation_limit, iterations=2)
                ds.image.flags |= (mask * 2 ** BitFlagConverter.convert('saturated')).astype(np.uint16)

                ds.image.save()
                if not os.getenv( "LIMIT_CACHE_USAGE" ):
                    output_path = copy_to_cache(ds.image, cache_dir)

                    if cache_dir is not None and cache_base_name is not None and output_path != cache_path:
                        warnings.warn(f'cache path {cache_path} does not match output path {output_path}')
                    elif cache_dir is not None and cache_base_name is None:
                        ds.cache_base_name = output_path
                        SCLogger.debug(f'Saving image to cache at: {output_path}')

                # In test_astro_cal, there's a routine that needs the original
                # image before being processed through the rest of what this
                # factory function does, so save it if requested
                if save_original_image:
                    ds.path_to_original_image = ds.image.get_fullpath()[0] + '.image.fits.original'
                    shutil.copy2( ds.image.get_fullpath()[0], ds.path_to_original_image )
                    if not os.getenv( "LIMIT_CACHE_USAGE" ):
                        shutil.copy2( ds.image.get_fullpath()[0],
                                      os.path.join(cache_dir, ds.image.filepath + '.image.fits.original') )

            # check if background was calculated
            if ds.image.bkg_mean_estimate is None or ds.image.bkg_rms_estimate is None:
                # Estimate the background rms with sep
                boxsize = ds.image.instrument_object.background_box_size
                filtsize = ds.image.instrument_object.background_filt_size

                # Dysfunctionality alert: sep requires a *float* image for the mask
                # IEEE 32-bit floats have 23 bits in the mantissa, so they should
                # be able to precisely represent a 16-bit integer mask image
                # In any event, sep.Background uses >0 as "bad"
                fmask = np.array(ds.image.flags, dtype=np.float32)
                backgrounder = sep.Background(ds.image.data, mask=fmask,
                                              bw=boxsize, bh=boxsize, fw=filtsize, fh=filtsize)

                ds.image.bkg_mean_estimate = backgrounder.globalback
                ds.image.bkg_rms_estimate = backgrounder.globalrms

            ############# extraction to create sources / PSF #############
            if (   ( not os.getenv( "LIMIT_CACHE_USAGE" ) ) and
                   ( cache_dir is not None ) and ( cache_base_name is not None )
                ):
                # try to get the SourceList from cache
                prov = Provenance(
                    code_version=code_version,
                    process='extraction',
                    upstreams=[ds.image.provenance],
                    parameters=p.extractor.pars.get_critical_pars(),
                    is_testing=True,
                )
                prov = session.merge(prov)
                cache_name = f'{cache_base_name}.sources_{prov.id[:6]}.fits.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    SCLogger.debug('loading source list from cache. ')
                    ds.sources = copy_from_cache(SourceList, cache_dir, cache_name)

                    # if SourceList already exists on the database, use that instead of this one
                    existing = session.scalars(
                        sa.select(SourceList).where(SourceList.filepath == ds.sources.filepath)
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.sources).mapper.columns.keys():
                            value = getattr(ds.sources, key)
                            if (
                                key not in ['id', 'image_id', 'created_at', 'modified'] and
                                value is not None
                            ):
                                setattr(existing, key, value)
                        ds.sources = existing  # replace with the existing row

                    ds.sources.provenance = prov
                    ds.sources.image = ds.image

                    # make sure this is saved to the archive as well
                    ds.sources.save(verify_md5=False)

                # try to get the PSF from cache
                cache_name = f'{cache_base_name}.psf_{prov.id[:6]}.fits.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    SCLogger.debug('loading PSF from cache. ')
                    ds.psf = copy_from_cache(PSF, cache_dir, cache_name)

                    # if PSF already exists on the database, use that instead of this one
                    existing = session.scalars(
                        sa.select(PSF).where(PSF.filepath == ds.psf.filepath)
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.psf).mapper.columns.keys():
                            value = getattr(ds.psf, key)
                            if (
                                    key not in ['id', 'image_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.psf = existing  # replace with the existing row

                    ds.psf.provenance = prov
                    ds.psf.image = ds.image

                    # make sure this is saved to the archive as well
                    ds.psf.save(verify_md5=False, overwrite=True)

            if ds.sources is None or ds.psf is None:  # make the source list from the regular image
                SCLogger.debug('extracting sources. ')
                ds = p.extractor.run(ds)
                ds.sources.save()
                ds.psf.save(overwrite=True)
                if not os.getenv( "LIMIT_CACHE_USAGE" ):
                    copy_to_cache(ds.sources, cache_dir)
                    output_path = copy_to_cache(ds.psf, cache_dir)
                    if cache_dir is not None and cache_base_name is not None and output_path != cache_path:
                        warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

            ############## astro_cal to create wcs ################
            if cache_dir is not None and cache_base_name is not None:
                prov = Provenance(
                        code_version=code_version,
                        process='astro_cal',
                        upstreams=[ds.sources.provenance],
                        parameters=p.astro_cal.pars.get_critical_pars(),
                        is_testing=True,
                    )
                cache_name = f'{cache_base_name}.wcs_{prov.id[:6]}.txt.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    SCLogger.debug('loading WCS from cache. ')
                    ds.wcs = copy_from_cache(WorldCoordinates, cache_dir, cache_name)
                    prov = session.merge(prov)

                    # check if WCS already exists on the database
                    existing = session.scalars(
                        sa.select(WorldCoordinates).where(
                            WorldCoordinates.sources_id == ds.sources.id,
                            WorldCoordinates.provenance_id == prov.id
                        )
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.wcs).mapper.columns.keys():
                            value = getattr(ds.wcs, key)
                            if (
                                    key not in ['id', 'sources_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.wcs = existing  # replace with the existing row

                    ds.wcs.provenance = prov
                    ds.wcs.sources = ds.sources
                    # make sure this is saved to the archive as well
                    ds.wcs.save(verify_md5=False, overwrite=True)

            if ds.wcs is None:  # make the WCS
                SCLogger.debug('Running astrometric calibration')
                ds = p.astro_cal.run(ds)
                ds.wcs.save()
                if ( ( cache_dir is not None ) and ( cache_base_name is not None ) and
                     ( not os.getenv( "LIMIT_CACHE_USAGE" ) ) ):
                    output_path = copy_to_cache(ds.wcs, cache_dir)
                    if output_path != cache_path:
                        warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

            ########### photo_cal to create zero point ############
            if (   ( not os.getenv( "LIMIT_CACHE_USAGE" ) ) and
                   ( cache_dir is not None ) and ( cache_base_name is not None )
                ):
                cache_name = cache_base_name + '.zp.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    SCLogger.debug('loading zero point from cache. ')
                    ds.zp = copy_from_cache(ZeroPoint, cache_dir, cache_name)
                    prov = Provenance(
                        code_version=code_version,
                        process='photo_cal',
                        upstreams=[ds.sources.provenance, ds.wcs.provenance],
                        parameters=p.photo_cal.pars.get_critical_pars(),
                        is_testing=True,
                    )
                    prov = session.merge(prov)

                    # check if ZP already exists on the database
                    existing = session.scalars(
                        sa.select(ZeroPoint).where(
                            ZeroPoint.sources_id == ds.sources.id,
                            ZeroPoint.provenance_id == prov.id
                        )
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.zp).mapper.columns.keys():
                            value = getattr(ds.zp, key)
                            if (
                                    key not in ['id', 'sources_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.zp = existing  # replace with the existing row

                    ds.zp.provenance = prov
                    ds.zp.sources = ds.sources

            if ds.zp is None:  # make the zero point
                SCLogger.debug('Running photometric calibration')
                ds = p.photo_cal.run(ds)
                if ( ( cache_dir is not None ) and ( cache_base_name is not None ) and
                     ( not os.getenv( "LIMIT_CACHE_USAGE" ) ) ):
                    output_path = copy_to_cache(ds.zp, cache_dir, cache_name)
                    if output_path != cache_path:
                        warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

            ds.save_and_commit(session=session)

            try:  # if no reference is found, simply return the datastore without the rest of the products
                ref = ds.get_reference()  # first make sure this actually manages to find the reference image
            except ValueError as e:
                if 'No reference image found' in str(e):
                    return ds
                raise e  # if any other error comes up, raise it

            # try to find the subtraction image in the cache
            if cache_dir is not None:
                prov = Provenance(
                    code_version=code_version,
                    process='subtraction',
                    upstreams=[
                        ds.image.provenance,
                        ds.sources.provenance,
                        ds.wcs.provenance,
                        ds.zp.provenance,
                        ref.image.provenance,
                        ref.sources.provenance,
                        ref.wcs.provenance,
                        ref.zp.provenance,
                    ],
                    parameters=p.subtractor.pars.get_critical_pars(),
                    is_testing=True,
                )
                sub_im = Image.from_new_and_ref(ds.image, ref.image)
                sub_im.provenance = prov
                cache_sub_name = sub_im.invent_filepath()
                cache_name = cache_sub_name + '.image.fits.json'
                if os.path.isfile(os.path.join(cache_dir, cache_name)):
                    SCLogger.debug('loading subtraction image from cache. ')
                    ds.sub_image = copy_from_cache(Image, cache_dir, cache_name)

                    ds.sub_image.provenance = prov
                    ds.sub_image.upstream_images.append(ref.image)
                    ds.sub_image.ref_image_id = ref.image_id
                    ds.sub_image.new_image = ds.image
                    ds.sub_image.save(verify_md5=False)  # make sure it is also saved to archive

                    # try to load the aligned images from cache
                    prov_aligned_ref = Provenance(
                        code_version=code_version,
                        parameters={
                            'method': 'swarp',
                            'to_index': 'new',
                            'max_arcsec_residual': 0.2,
                            'crossid_radius': 2.0,
                            'max_sources_to_use': 2000,
                            'min_frac_matched': 0.1,
                            'min_matched': 10,
                        },
                        upstreams=[
                            ds.image.provenance,
                            ds.sources.provenance,  # this also includes the PSF's provenance
                            ds.wcs.provenance,
                            ds.ref_image.provenance,
                            ds.ref_image.sources.provenance,
                            ds.ref_image.wcs.provenance,
                            ds.ref_image.zp.provenance,
                        ],
                        process='alignment',
                        is_testing=True,
                    )
                    # TODO: can we find a less "hacky" way to do this?
                    f = ref.image.invent_filepath()
                    f = f.replace('ComSci', 'Warped')  # not sure if this or 'Sci' will be in the filename
                    f = f.replace('Sci', 'Warped')     # in any case, replace it with 'Warped'
                    f = f[:-6] + prov_aligned_ref.id[:6]  # replace the provenance ID
                    filename_aligned_ref = f

                    prov_aligned_new = Provenance(
                        code_version=code_version,
                        parameters=prov_aligned_ref.parameters,
                        upstreams=[
                            ds.image.provenance,
                            ds.sources.provenance,  # this also includes the PSF's provenance
                            ds.wcs.provenance,
                            ds.zp.provenance,
                        ],
                        process='alignment',
                        is_testing=True,
                    )
                    f = ds.sub_image.new_image.invent_filepath()
                    f = f.replace('ComSci', 'Warped')
                    f = f.replace('Sci', 'Warped')
                    f = f[:-6] + prov_aligned_new.id[:6]
                    filename_aligned_new = f

                    cache_name_ref = filename_aligned_ref + '.fits.json'
                    cache_name_new = filename_aligned_new + '.fits.json'
                    if (
                            os.path.isfile(os.path.join(cache_dir, cache_name_ref)) and
                            os.path.isfile(os.path.join(cache_dir, cache_name_new))
                    ):
                        SCLogger.debug('loading aligned reference image from cache. ')
                        image_aligned_ref = copy_from_cache(Image, cache_dir, cache_name)
                        image_aligned_ref.provenance = prov_aligned_ref
                        image_aligned_ref.info['original_image_id'] = ds.ref_image_id
                        image_aligned_ref.info['original_image_filepath'] = ds.ref_image.filepath
                        image_aligned_ref.save(verify_md5=False, no_archive=True)
                        # TODO: should we also load the aligned image's sources, PSF, and ZP?

                        SCLogger.debug('loading aligned new image from cache. ')
                        image_aligned_new = copy_from_cache(Image, cache_dir, cache_name)
                        image_aligned_new.provenance = prov_aligned_new
                        image_aligned_new.info['original_image_id'] = ds.image_id
                        image_aligned_new.info['original_image_filepath'] = ds.image.filepath
                        image_aligned_new.save(verify_md5=False, no_archive=True)
                        # TODO: should we also load the aligned image's sources, PSF, and ZP?

                        if image_aligned_ref.mjd < image_aligned_new.mjd:
                            ds.sub_image._aligned_images = [image_aligned_ref, image_aligned_new]
                        else:
                            ds.sub_image._aligned_images = [image_aligned_new, image_aligned_ref]

            if ds.sub_image is None:  # no hit in the cache
                ds = p.subtractor.run(ds)
                ds.sub_image.save(verify_md5=False)  # make sure it is also saved to archive
                if not os.getenv( "LIMIT_CACHE_USAGE" ):
                    copy_to_cache(ds.sub_image, cache_dir)

            # make sure that the aligned images get into the cache, too
            if (
                    ( not os.getenv( "LIMIT_CACHE_USAGE" ) ) and
                    'cache_name_ref' in locals() and
                    os.path.isfile(os.path.join(cache_dir, cache_name_ref)) and
                    'cache_name_new' in locals() and
                    os.path.isfile(os.path.join(cache_dir, cache_name_new))
            ):
                for im in ds.sub_image.aligned_images:
                    copy_to_cache(im, cache_dir)

            ############ detecting to create a source list ############
            prov = Provenance(
                code_version=code_version,
                process='detection',
                upstreams=[ds.sub_image.provenance],
                parameters=p.detector.pars.get_critical_pars(),
                is_testing=True,
            )
            cache_name = os.path.join(cache_dir, cache_sub_name + f'.sources_{prov.id[:6]}.npy.json')
            if ( not os.getenv( "LIMIT_CACHE_USAGE" ) ) and ( os.path.isfile(cache_name) ):
                SCLogger.debug('loading detections from cache. ')
                ds.detections = copy_from_cache(SourceList, cache_dir, cache_name)
                ds.detections.provenance = prov
                ds.detections.image = ds.sub_image
                ds.sub_image.sources = ds.detections
                ds.detections.save(verify_md5=False)
            else:  # cannot find detections on cache
                ds = p.detector.run(ds)
                ds.detections.save(verify_md5=False)
                if not os.getenv( "LIMIT_CACHE_USAGE" ):
                    copy_to_cache(ds.detections, cache_dir, cache_name)

            ############ cutting to create cutouts ############
            prov = Provenance(
                code_version=code_version,
                process='cutting',
                upstreams=[ds.detections.provenance],
                parameters=p.cutter.pars.get_critical_pars(),
                is_testing=True,
            )
            cache_name = os.path.join(cache_dir, cache_sub_name + f'.cutouts_{prov.id[:6]}.h5')
            if ( not os.getenv( "LIMIT_CACHE_USAGE" ) ) and ( os.path.isfile(cache_name) ):
                SCLogger.debug('loading cutouts from cache. ')
                ds.cutouts = copy_list_from_cache(Cutouts, cache_dir, cache_name)
                ds.cutouts = Cutouts.load_list(os.path.join(ds.cutouts[0].local_path, ds.cutouts[0].filepath))
                [setattr(c, 'provenance', prov) for c in ds.cutouts]
                [setattr(c, 'sources', ds.detections) for c in ds.cutouts]
                Cutouts.save_list(ds.cutouts)  # make sure to save to archive as well
            else:  # cannot find cutouts on cache
                ds = p.cutter.run(ds)
                Cutouts.save_list(ds.cutouts)
                if not os.getenv( "LIMIT_CACHE_USAGE" ):
                    copy_list_to_cache(ds.cutouts, cache_dir)

            ############ measuring to create measurements ############
            prov = Provenance(
                code_version=code_version,
                process='measuring',
                upstreams=[ds.cutouts[0].provenance],
                parameters=p.measurer.pars.get_critical_pars(),
                is_testing=True,
            )

            cache_name = os.path.join(cache_dir, cache_sub_name + f'.measurements_{prov.id[:6]}.json')

            if os.path.isfile(cache_name):  # note that the cache contains ALL the measurements, not only the good ones
                SCLogger.debug('loading measurements from cache. ')
                ds.all_measurements = copy_list_from_cache(Measurements, cache_dir, cache_name)
                [setattr(m, 'provenance', prov) for m in ds.all_measurements]
                [setattr(m, 'cutouts', c) for m, c in zip(ds.all_measurements, ds.cutouts)]

                ds.measurements = []
                for m in ds.all_measurements:
                    threshold_comparison = p.measurer.compare_measurement_to_thresholds(m)
                    if threshold_comparison != "delete":  # all disqualifiers are below threshold
                        m.is_bad = threshold_comparison == "bad"
                        ds.measurements.append(m)

                [m.associate_object(session) for m in ds.measurements]  # create or find an object for each measurement
                # no need to save list because Measurements is not a FileOnDiskMixin!
            else:  # cannot find measurements on cache
                ds = p.measurer.run(ds)
                copy_list_to_cache(ds.all_measurements, cache_dir, cache_name)  # must provide filepath!

            ds.save_and_commit(session=session)

            return ds

    return make_datastore
