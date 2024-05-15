import os
import warnings
import shutil
import pytest

import numpy as np

import sqlalchemy as sa

import sep

from models.base import SmartSession
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

from util.logger import SCLogger

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
        det.pars._enforce_no_new_attrs = False

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
        cut.pars._enforce_no_new_attrs = False

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
        meas.pars._enforce_no_new_attrs = False

        return meas

    return make_measurer


@pytest.fixture
def measurer(measurer_factory):
    return measurer_factory()


@pytest.fixture(scope='session')
def datastore_factory(
        data_dir,
        preprocessor_factory,
        extractor_factory,
        astrometor_factory,
        photometor_factory,
        subtractor_factory,
        detector_factory,
        cutter_factory,
        measurer_factory,

):
    """Provide a function that returns a datastore with all the products based on the given exposure and section ID.

    To use this data store in a test where new data is to be generated,
    simply change the pipeline object's "test_parameter" value to a unique
    new value, so the provenance will not match and the data will be regenerated.

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
    ):
        code_version = args[0].provenance.code_version
        ds = DataStore(*args)  # make a new datastore

        if cache_dir is not None and cache_base_name is not None:
            ds.cache_base_name = os.path.join(cache_dir, cache_base_name)  # save this for testing purposes

        # allow calling scope to override/augment parameters for any of the processing steps
        preprocessor = preprocessor_factory()
        preprocessor.pars.override(overrides.get('preprocessing', {}))
        preprocessor.pars.augment(augments.get('preprocessing', {}))

        extractor = extractor_factory()
        extractor.pars.override(overrides.get('extraction', {}))
        extractor.pars.augment(augments.get('extraction', {}))

        astrometor = astrometor_factory()
        astrometor.pars.override(overrides.get('astro_cal', {}))
        astrometor.pars.augment(augments.get('astro_cal', {}))

        photometor = photometor_factory()
        photometor.pars.override(overrides.get('photo_cal', {}))
        photometor.pars.augment(augments.get('photo_cal', {}))

        subtractor = subtractor_factory()
        subtractor.pars.override(overrides.get('subtraction', {}))
        subtractor.pars.augment(augments.get('subtraction', {}))

        detector = detector_factory()
        detector.pars.override(overrides.get('detection', {}))
        detector.pars.augment(augments.get('detection', {}))

        cutter = cutter_factory()
        cutter.pars.override(overrides.get('cutting', {}))
        cutter.pars.augment(augments.get('cutting', {}))

        measurer = measurer_factory()
        measurer.pars.override(overrides.get('measurement', {}))
        measurer.pars.augment(augments.get('measurement', {}))

        with SmartSession(session) as session:
            code_version = session.merge(code_version)
            if ds.image is not None:  # if starting from an externally provided Image, must merge it first
                ds.image = ds.image.merge_all(session)

            ############ preprocessing to create image ############
            if ds.image is None and cache_dir is not None and cache_base_name is not None:
                # check if preprocessed image is in cache
                cache_name = cache_base_name + '.image.fits.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    SCLogger.debug('loading image from cache. ')
                    ds.image = Image.copy_from_cache(cache_dir, cache_name)
                    # assign the correct exposure to the object loaded from cache
                    if ds.exposure_id is not None:
                        ds.image.exposure_id = ds.exposure_id
                    if ds.exposure is not None:
                        ds.image.exposure = ds.exposure

                    # add the preprocessing steps from instrument (TODO: remove this as part of Issue #142)
                    preprocessing_steps = ds.image.instrument_object.preprocessing_steps
                    prep_pars = preprocessor.pars.get_critical_pars()
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
                ds = preprocessor.run(ds)
                ds.image.provenance.is_testing = True
                if bad_pixel_map is not None:
                    ds.image.flags |= bad_pixel_map
                    if ds.image.weight is not None:
                        ds.image.weight[ds.image.flags.astype(bool)] = 0.0

                # flag saturated pixels, too (TODO: is there a better way to get the saturation limit? )
                mask = make_saturated_flag(ds.image.data, ds.image.instrument_object.saturation_limit, iterations=2)
                ds.image.flags |= (mask * 2 ** BitFlagConverter.convert('saturated')).astype(np.uint16)

                ds.image.save()
                output_path = ds.image.copy_to_cache(cache_dir)
                # also save the original image to the cache as a separate file
                shutil.copy2(
                    ds.image.get_fullpath()[0],
                    os.path.join(cache_dir, ds.image.filepath + '.image.fits.original')
                )

                if cache_dir is not None and cache_base_name is not None and output_path != cache_path:
                    warnings.warn(f'cache path {cache_path} does not match output path {output_path}')
                elif cache_dir is not None and cache_base_name is None:
                    ds.cache_base_name = output_path
                    SCLogger.debug(f'Saving image to cache at: {output_path}')

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
            if cache_dir is not None and cache_base_name is not None:
                # try to get the SourceList from cache
                prov = Provenance(
                    code_version=code_version,
                    process='extraction',
                    upstreams=[ds.image.provenance],
                    parameters=extractor.pars.get_critical_pars(),
                    is_testing=True,
                )
                prov = session.merge(prov)
                cache_name = f'{cache_base_name}.sources_{prov.id[:6]}.fits.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    SCLogger.debug('loading source list from cache. ')
                    ds.sources = SourceList.copy_from_cache(cache_dir, cache_name)

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
                    ds.psf = PSF.copy_from_cache(cache_dir, cache_name)

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
                ds = extractor.run(ds)
                ds.sources.save()
                ds.sources.copy_to_cache(cache_dir)
                ds.psf.save(overwrite=True)
                output_path = ds.psf.copy_to_cache(cache_dir)
                if cache_dir is not None and cache_base_name is not None and output_path != cache_path:
                    warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

            ############## astro_cal to create wcs ################
            if cache_dir is not None and cache_base_name is not None:
                cache_name = cache_base_name + '.wcs.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    SCLogger.debug('loading WCS from cache. ')
                    ds.wcs = WorldCoordinates.copy_from_cache(cache_dir, cache_name)
                    prov = Provenance(
                        code_version=code_version,
                        process='astro_cal',
                        upstreams=[ds.sources.provenance],
                        parameters=astrometor.pars.get_critical_pars(),
                        is_testing=True,
                    )
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

            if ds.wcs is None:  # make the WCS
                SCLogger.debug('Running astrometric calibration')
                ds = astrometor.run(ds)
                if cache_dir is not None and cache_base_name is not None:
                    # must provide a name because this one isn't a FileOnDiskMixin
                    output_path = ds.wcs.copy_to_cache(cache_dir, cache_name)
                    if output_path != cache_path:
                        warnings.warn(f'cache path {cache_path} does not match output path {output_path}')

            ########### photo_cal to create zero point ############
            if cache_dir is not None and cache_base_name is not None:
                cache_name = cache_base_name + '.zp.json'
                cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(cache_path):
                    SCLogger.debug('loading zero point from cache. ')
                    ds.zp = ZeroPoint.copy_from_cache(cache_dir, cache_name)
                    prov = Provenance(
                        code_version=code_version,
                        process='photo_cal',
                        upstreams=[ds.sources.provenance],
                        parameters=photometor.pars.get_critical_pars(),
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
                ds = photometor.run(ds)
                if cache_dir is not None and cache_base_name is not None:
                    output_path = ds.zp.copy_to_cache(cache_dir, cache_name)
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
                    parameters=subtractor.pars.get_critical_pars(),
                    is_testing=True,
                )
                sub_im = Image.from_new_and_ref(ds.image, ref.image)
                sub_im.provenance = prov
                cache_sub_name = sub_im.invent_filepath()
                cache_name = cache_sub_name + '.image.fits.json'
                if os.path.isfile(os.path.join(cache_dir, cache_name)):
                    SCLogger.debug('loading subtraction image from cache. ')
                    ds.sub_image = Image.copy_from_cache(cache_dir, cache_name)
                    ds.sub_image.provenance = prov
                    ds.sub_image.upstream_images.append(ref.image)
                    ds.sub_image.ref_image_id = ref.image_id
                    ds.sub_image.new_image = ds.image
                    ds.sub_image.save(verify_md5=False)  # make sure it is also saved to archive
            if ds.sub_image is None:  # no hit in the cache
                ds = subtractor.run(ds)

            prov = Provenance(
                code_version=code_version,
                process='detection',
                upstreams=[ds.sub_image.provenance],
                parameters=detector.pars.get_critical_pars(),
                is_testing=True,
            )
            cache_name = os.path.join(cache_dir, cache_sub_name + f'.sources_{prov.id[:6]}.npy.json')
            if os.path.isfile(cache_name):
                SCLogger.debug('loading detections from cache. ')
                ds.detections = SourceList.copy_from_cache(cache_dir, cache_name)
                ds.detections.provenance = prov
                ds.detections.image = ds.sub_image
                ds.sub_image.sources = ds.detections
                ds.detections.save(verify_md5=False)
            else:  # cannot find detections on cache
                ds = detector.run(ds)
                ds.detections.save(verify_md5=False)
                ds.detections.copy_to_cache(cache_dir, cache_name)

            prov = Provenance(
                code_version=code_version,
                process='cutting',
                upstreams=[ds.detections.provenance],
                parameters=cutter.pars.get_critical_pars(),
                is_testing=True,
            )
            cache_name = os.path.join(cache_dir, cache_sub_name + f'.cutouts_{prov.id[:6]}.h5')
            if os.path.isfile(cache_name):
                SCLogger.debug('loading cutouts from cache. ')
                ds.cutouts = Cutouts.copy_list_from_cache(cache_dir, cache_name)
                ds.cutouts = Cutouts.load_list(os.path.join(ds.cutouts[0].local_path, ds.cutouts[0].filepath))
                [setattr(c, 'provenance', prov) for c in ds.cutouts]
                [setattr(c, 'sources', ds.detections) for c in ds.cutouts]
                Cutouts.save_list(ds.cutouts)  # make sure to save to archive as well
            else:  # cannot find cutouts on cache
                ds = cutter.run(ds)
                Cutouts.save_list(ds.cutouts)
                Cutouts.copy_list_to_cache(ds.cutouts, cache_dir)

            prov = Provenance(
                code_version=code_version,
                process='measuring',
                upstreams=[ds.cutouts[0].provenance],
                parameters=measurer.pars.get_critical_pars(),
                is_testing=True,
            )

            cache_name = os.path.join(cache_dir, cache_sub_name + f'.measurements_{prov.id[:6]}.json')

            if os.path.isfile(cache_name):  # note that the cache contains ALL the measurements, not only the good ones
                SCLogger.debug('loading measurements from cache. ')
                ds.all_measurements = Measurements.copy_list_from_cache(cache_dir, cache_name)
                [setattr(m, 'provenance', prov) for m in ds.all_measurements]
                [setattr(m, 'cutouts', c) for m, c in zip(ds.all_measurements, ds.cutouts)]
                ds.measurements = [m for m in ds.all_measurements if m.passes()]
                [m.associate_object(session) for m in ds.measurements]  # create or find an object for each measurement
                # no need to save list because Measurements is not a FileOnDiskMixin!
            else:  # cannot find measurements on cache
                ds = measurer.run(ds)
                Measurements.copy_list_to_cache(ds.all_measurements, cache_dir, cache_name)  # must provide filepath!

            ds.save_and_commit(session=session)

            return ds

    return make_datastore
