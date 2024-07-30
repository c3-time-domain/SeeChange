import os
import warnings
import shutil
import pytest

import numpy as np

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import Provenance
from models.enums_and_bitflags import BitFlagConverter
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.refset import RefSet
from pipeline.data_store import DataStore

from util.logger import SCLogger
from util.cache import copy_to_cache, copy_list_to_cache, copy_from_cache, copy_list_from_cache
from util.util import env_as_bool

from improc.bitmask_tools import make_saturated_flag


@pytest.fixture(scope='session')
def datastore_factory(data_dir, pipeline_factory, request):
    """Provide a function that returns a datastore with all the products based on the given exposure and section ID.

    To use this data store in a test where new data is to be generated,
    simply change the pipeline object's "test_parameter" value to a unique
    new value, so the provenance will not match and the data will be regenerated.

    If "save_original_image" is True, then a copy of the image before
    going through source extraction, WCS, etc. will be saved alongside
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
            save_original_image=False,
            skip_sub=False,
            provtag='datastore_factory'
    ):
        code_version = args[0].provenance.code_version
        SCLogger.debug( f"make_datastore called with args {args}, overrides={overrides}, augments={augments}" )
        ds = DataStore(*args)  # make a new datastore
        use_cache = cache_dir is not None and cache_base_name is not None and not env_as_bool( "LIMIT_CACHE_USAGE" )

        if cache_base_name is not None:
            cache_name = cache_base_name + '.image.fits.json'
            image_cache_path = os.path.join(cache_dir, cache_name)
        else:
            image_cache_path = None

        if use_cache:
            ds.cache_base_name = os.path.join(cache_dir, cache_base_name)  # save this for testing purposes

        p = pipeline_factory( provtag )

        # allow calling scope to override/augment parameters for any of the processing steps
        p.override_parameters(**overrides)
        p.augment_parameters(**augments)

        with SmartSession(session) as session:
            code_version = session.merge(code_version)

            if ds.image is not None:  # if starting from an externally provided Image, must merge it first
                SCLogger.debug( f"make_datastore was provided an external image; merging it" )
                ds.image = ds.image.merge_all(session)

            ############ load the reference set ############

            inst_name = ds.image.instrument.lower() if ds.image else ds.exposure.instrument.lower()
            refset_name = f'test_refset_{inst_name}'
            if inst_name == 'ptf':  # request the ptf_refset fixture dynamically:
                request.getfixturevalue('ptf_refset')
            if inst_name == 'decam':  # request the decam_refset fixture dynamically:
                request.getfixturevalue('decam_refset')

            refset = session.scalars(sa.select(RefSet).where(RefSet.name == refset_name)).first()

            if refset is None:
                raise ValueError(f'make_datastore found no reference with name {refset_name}')

            ref_prov = refset.provenances[0]

            ############ preprocessing to create image ############
            if ds.image is None and use_cache:  # check if preprocessed image is in cache
                if os.path.isfile(image_cache_path):
                    SCLogger.debug('make_datastore loading image from cache. ')
                    ds.image = copy_from_cache(Image, cache_dir, cache_name)
                    # assign the correct exposure to the object loaded from cache
                    if ds.exposure_id is not None:
                        ds.image.exposure_id = ds.exposure_id
                    if ds.exposure is not None:
                        ds.image.exposure = ds.exposure
                        ds.image.exposure_id = ds.exposure.id

                    # Copy the original image from the cache if requested
                    if save_original_image:
                        ds.path_to_original_image = ds.image.get_fullpath()[0] + '.image.fits.original'
                        image_cache_path_original = os.path.join(cache_dir, ds.image.filepath + '.image.fits.original')
                        shutil.copy2( image_cache_path_original, ds.path_to_original_image )

                    upstreams = [ds.exposure.provenance] if ds.exposure is not None else []  # images without exposure
                    prov = Provenance(
                        code_version=code_version,
                        process='preprocessing',
                        upstreams=upstreams,
                        parameters=p.preprocessor.pars.get_critical_pars(),
                        is_testing=True,
                    )
                    prov = session.merge(prov)
                    session.commit()

                    # if Image already exists on the database, use that instead of this one
                    existing = session.scalars(sa.select(Image).where(Image.filepath == ds.image.filepath)).first()
                    if existing is not None:
                        SCLogger.debug( f"make_datastore updating existing image {existing.id} "
                                        f"({existing.filepath}) with image loaded from cache" )
                        # overwrite the existing row data using the JSON cache file
                        for key in sa.inspect(ds.image).mapper.columns.keys():
                            value = getattr(ds.image, key)
                            if (
                                    key not in ['id', 'image_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.image = existing  # replace with the existing row
                    else:
                        SCLogger.debug( f"make_datastore did not find image with filepath "
                                        f"{ds.image.filepath} in database" )

                    ds.image.provenance = prov

                    # make sure this is saved to the archive as well
                    ds.image.save(verify_md5=False)

            if ds.image is None:  # make the preprocessed image
                SCLogger.debug('make_datastore making preprocessed image. ')
                ds = p.preprocessor.run(ds, session)
                ds.image.provenance.is_testing = True
                if bad_pixel_map is not None:
                    ds.image.flags |= bad_pixel_map
                    if ds.image.weight is not None:
                        ds.image.weight[ds.image.flags.astype(bool)] = 0.0

                # flag saturated pixels, too (TODO: is there a better way to get the saturation limit? )
                mask = make_saturated_flag(ds.image.data, ds.image.instrument_object.saturation_limit, iterations=2)
                ds.image.flags |= (mask * 2 ** BitFlagConverter.convert('saturated')).astype(np.uint16)

                ds.image.save()
                # even if cache_base_name is None, we still need to make the manifest file, so we will get it next time!
                if not env_as_bool( "LIMIT_CACHE_USAGE" ) and os.path.isdir(cache_dir):
                    output_path = copy_to_cache(ds.image, cache_dir)

                    if image_cache_path is not None and output_path != image_cache_path:
                        warnings.warn(f'cache path {image_cache_path} does not match output path {output_path}')
                    else:
                        cache_base_name = output_path[:-16]  # remove the '.image.fits.json' part
                        ds.cache_base_name = output_path
                        SCLogger.debug(f'Saving image to cache at: {output_path}')
                        use_cache = True  # the two other conditions are true to even get to this part...

                # In test_astro_cal, there's a routine that needs the original
                # image before being processed through the rest of what this
                # factory function does, so save it if requested
                if save_original_image:
                    ds.path_to_original_image = ds.image.get_fullpath()[0] + '.image.fits.original'
                    shutil.copy2( ds.image.get_fullpath()[0], ds.path_to_original_image )
                    if use_cache:
                        shutil.copy2(
                            ds.image.get_fullpath()[0],
                            os.path.join(cache_dir, ds.image.filepath + '.image.fits.original')
                        )

            ############# extraction to create sources / PSF / BG / WCS / ZP #############
            if use_cache:  # try to get the SourceList, PSF, BG, WCS and ZP from cache
                prov = Provenance(
                    code_version=code_version,
                    process='extraction',
                    upstreams=[ds.image.provenance],
                    parameters=p.extractor.pars.get_critical_pars(),  # the siblings will be loaded automatically
                    is_testing=True,
                )
                prov = session.merge(prov)
                session.commit()

                # try to get the source list from cache
                cache_name = f'{cache_base_name}.sources_{prov.id[:6]}.fits.json'
                sources_cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(sources_cache_path):
                    SCLogger.debug('make_datastore loading source list from cache. ')
                    ds.sources = copy_from_cache(SourceList, cache_dir, cache_name)
                    # if SourceList already exists on the database, use that instead of this one
                    existing = session.scalars(
                        sa.select(SourceList).where(SourceList.filepath == ds.sources.filepath)
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        SCLogger.debug( f"make_datastore updating existing source list {existing.id} "
                                        f"({existing.filepath}) with source list loaded from cache" )
                        for key in sa.inspect(ds.sources).mapper.columns.keys():
                            value = getattr(ds.sources, key)
                            if (
                                key not in ['id', 'image_id', 'created_at', 'modified'] and
                                value is not None
                            ):
                                setattr(existing, key, value)
                        ds.sources = existing  # replace with the existing row
                    else:
                        SCLogger.debug( f"make_datastore did not find source list with filepath "
                                        f"{ds.sources.filepath} in the database" )

                    ds.sources.provenance = prov
                    ds.sources.image = ds.image

                    # make sure this is saved to the archive as well
                    ds.sources.save(verify_md5=False)

                # try to get the PSF from cache
                cache_name = f'{cache_base_name}.psf_{prov.id[:6]}.fits.json'
                psf_cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(psf_cache_path):
                    SCLogger.debug('make_datastore loading PSF from cache. ')
                    ds.psf = copy_from_cache(PSF, cache_dir, cache_name)
                    # if PSF already exists on the database, use that instead of this one
                    existing = session.scalars(
                        sa.select(PSF).where(PSF.filepath == ds.psf.filepath)
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        SCLogger.debug( f"make_datastore updating existing psf {existing.id} "
                                        f"({existing.filepath}) with psf loaded from cache" )
                        for key in sa.inspect(ds.psf).mapper.columns.keys():
                            value = getattr(ds.psf, key)
                            if (
                                    key not in ['id', 'image_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.psf = existing  # replace with the existing row
                    else:
                        SCLogger.debug( f"make_datastore did not find psf with filepath "
                                        f"{ds.psf.filepath} in the database" )

                    ds.psf.provenance = prov
                    ds.psf.image = ds.image

                    # make sure this is saved to the archive as well
                    ds.psf.save(verify_md5=False, overwrite=True)

                # try to get the background from cache
                cache_name = f'{cache_base_name}.bg_{prov.id[:6]}.h5.json'
                bg_cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(bg_cache_path):
                    SCLogger.debug('make_datastore loading background from cache. ')
                    ds.bg = copy_from_cache(Background, cache_dir, cache_name)
                    # if BG already exists on the database, use that instead of this one
                    existing = session.scalars(
                        sa.select(Background).where(Background.filepath == ds.bg.filepath)
                    ).first()
                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        SCLogger.debug( f"make_datastore updating existing background {existing.id} "
                                        f"({existing.filepath}) with source list loaded from cache" )
                        for key in sa.inspect(ds.bg).mapper.columns.keys():
                            value = getattr(ds.bg, key)
                            if (
                                    key not in ['id', 'image_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.bg = existing
                    else:
                        SCLogger.debug( f"make_datastore did not find background with filepath "
                                        f"{ds.bg.filepath} in the database" )

                    ds.bg.provenance = prov
                    ds.bg.image = ds.image

                    # make sure this is saved to the archive as well
                    ds.bg.save(verify_md5=False, overwrite=True)

                # try to get the WCS from cache
                cache_name = f'{cache_base_name}.wcs_{prov.id[:6]}.txt.json'
                wcs_cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(wcs_cache_path):
                    SCLogger.debug('make_datastore loading WCS from cache. ')
                    ds.wcs = copy_from_cache(WorldCoordinates, cache_dir, cache_name)
                    prov = session.merge(prov)

                    # check if WCS already exists on the database
                    if ds.sources is not None:
                        existing = session.scalars(
                            sa.select(WorldCoordinates).where(
                                WorldCoordinates.sources_id == ds.sources.id,
                                WorldCoordinates.provenance_id == prov.id
                            )
                        ).first()
                    else:
                        existing = None

                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        SCLogger.debug( f"make_datastore updating existing wcs {existing.id} "
                                        f"with wcs loaded from cache" )
                        for key in sa.inspect(ds.wcs).mapper.columns.keys():
                            value = getattr(ds.wcs, key)
                            if (
                                    key not in ['id', 'sources_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.wcs = existing  # replace with the existing row
                    else:
                        SCLogger.debug( f"make_datastore did not find existing wcs in database" )

                    ds.wcs.provenance = prov
                    ds.wcs.sources = ds.sources
                    # make sure this is saved to the archive as well
                    ds.wcs.save(verify_md5=False, overwrite=True)

                # try to get the ZP from cache
                cache_name = cache_base_name + '.zp.json'
                zp_cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(zp_cache_path):
                    SCLogger.debug('make_datastore loading zero point from cache. ')
                    ds.zp = copy_from_cache(ZeroPoint, cache_dir, cache_name)

                    # check if ZP already exists on the database
                    if ds.sources is not None:
                        existing = session.scalars(
                            sa.select(ZeroPoint).where(
                                ZeroPoint.sources_id == ds.sources.id,
                                ZeroPoint.provenance_id == prov.id
                            )
                        ).first()
                    else:
                        existing = None

                    if existing is not None:
                        # overwrite the existing row data using the JSON cache file
                        SCLogger.debug( f"make_datastore updating existing zp {existing.id} "
                                        f"with zp loaded from cache" )
                        for key in sa.inspect(ds.zp).mapper.columns.keys():
                            value = getattr(ds.zp, key)
                            if (
                                    key not in ['id', 'sources_id', 'created_at', 'modified'] and
                                    value is not None
                            ):
                                setattr(existing, key, value)
                        ds.zp = existing  # replace with the existing row
                    else:
                        SCLogger.debug( "make_datastore did not find existing zp in database" )

                    ds.zp.provenance = prov
                    ds.zp.sources = ds.sources

            # if any data product is missing, must redo the extraction step
            if ds.sources is None or ds.psf is None or ds.bg is None or ds.wcs is None or ds.zp is None:
                SCLogger.debug('make_datastore extracting sources. ')
                ds = p.extractor.run(ds, session)

                ds.sources.save(overwrite=True)
                if use_cache:
                    output_path = copy_to_cache(ds.sources, cache_dir)
                    if output_path != sources_cache_path:
                        warnings.warn(f'cache path {sources_cache_path} does not match output path {output_path}')

                ds.psf.save(overwrite=True)
                if use_cache:
                    output_path = copy_to_cache(ds.psf, cache_dir)
                    if output_path != psf_cache_path:
                        warnings.warn(f'cache path {psf_cache_path} does not match output path {output_path}')

                SCLogger.debug('Running background estimation')
                ds = p.backgrounder.run(ds, session)

                ds.bg.save(overwrite=True)
                if use_cache:
                    output_path = copy_to_cache(ds.bg, cache_dir)
                    if output_path != bg_cache_path:
                        warnings.warn(f'cache path {bg_cache_path} does not match output path {output_path}')

                SCLogger.debug('Running astrometric calibration')
                ds = p.astrometor.run(ds, session)
                ds.wcs.save(overwrite=True)
                if use_cache:
                    output_path = copy_to_cache(ds.wcs, cache_dir)
                    if output_path != wcs_cache_path:
                        warnings.warn(f'cache path {wcs_cache_path} does not match output path {output_path}')

                SCLogger.debug('Running photometric calibration')
                ds = p.photometor.run(ds, session)
                if use_cache:
                    cache_name = cache_base_name + '.zp.json'
                    output_path = copy_to_cache(ds.zp, cache_dir, cache_name)
                    if output_path != zp_cache_path:
                        warnings.warn(f'cache path {zp_cache_path} does not match output path {output_path}')

            SCLogger.debug( "make_datastore running ds.save_and_commit on image (before subtraction)" )
            ds.save_and_commit(session=session)

            # make a new copy of the image to cache, including the estimates for lim_mag, fwhm, etc.
            if not env_as_bool("LIMIT_CACHE_USAGE"):
                output_path = copy_to_cache(ds.image, cache_dir)

            # If we were told not to try to do a subtraction, then we're done
            if skip_sub:
                SCLogger.debug( "make_datastore : skip_sub is True, returning" )
                return ds

            # must provide the reference provenance explicitly since we didn't build a prov_tree
            ref = ds.get_reference(ref_prov, session=session)
            if ref is None:
                SCLogger.debug( "make_datastore : could not find a reference, returning" )
                return ds  # if no reference is found, simply return the datastore without the rest of the products

            if use_cache:  # try to find the subtraction image in the cache
                SCLogger.debug( "make_datstore looking for subtraction image in cache..." )
                prov = Provenance(
                    code_version=code_version,
                    process='subtraction',
                    upstreams=[
                        ds.image.provenance,
                        ds.sources.provenance,
                        ref.image.provenance,
                        ref.sources.provenance,
                    ],
                    parameters=p.subtractor.pars.get_critical_pars(),
                    is_testing=True,
                )
                prov = session.merge(prov)
                session.commit()

                sub_im = Image.from_new_and_ref(ds.image, ref.image)
                sub_im.provenance = prov
                cache_sub_name = sub_im.invent_filepath()
                cache_name = cache_sub_name + '.image.fits.json'
                sub_cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(sub_cache_path):
                    SCLogger.debug('make_datastore loading subtraction image from cache: {sub_cache_path}" ')
                    ds.sub_image = copy_from_cache(Image, cache_dir, cache_name)

                    ds.sub_image.provenance = prov
                    ds.sub_image.upstream_images.append(ref.image)
                    ds.sub_image.ref_image_id = ref.image_id
                    ds.sub_image.ref_image = ref.image
                    ds.sub_image.new_image = ds.image
                    ds.sub_image.save(verify_md5=False)  # make sure it is also saved to archive

                    # try to load the aligned images from cache
                    prov_aligned_ref = Provenance(
                        code_version=code_version,
                        parameters=prov.parameters['alignment'],
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
                        parameters=prov.parameters['alignment'],
                        upstreams=[
                            ds.image.provenance,
                            ds.sources.provenance,  # this also includes provs for PSF, BG, WCS, ZP
                        ],
                        process='alignment',
                        is_testing=True,
                    )
                    f = ds.sub_image.new_image.invent_filepath()
                    f = f.replace('ComSci', 'Warped')
                    f = f.replace('Sci', 'Warped')
                    f = f[:-6] + prov_aligned_new.id[:6]
                    filename_aligned_new = f

                    cache_name_ref = filename_aligned_ref + '.image.fits.json'
                    cache_name_new = filename_aligned_new + '.image.fits.json'
                    if (
                            os.path.isfile(os.path.join(cache_dir, cache_name_ref)) and
                            os.path.isfile(os.path.join(cache_dir, cache_name_new))
                    ):
                        SCLogger.debug('loading aligned reference image from cache. ')
                        image_aligned_ref = copy_from_cache(Image, cache_dir, cache_name)
                        image_aligned_ref.provenance = prov_aligned_ref
                        image_aligned_ref.info['original_image_id'] = ds.ref_image.id
                        image_aligned_ref.info['original_image_filepath'] = ds.ref_image.filepath
                        image_aligned_ref.info['alignment_parameters'] = prov.parameters['alignment']
                        image_aligned_ref.save(verify_md5=False, no_archive=True)
                        # TODO: should we also load the aligned image's sources, PSF, and ZP?

                        SCLogger.debug('loading aligned new image from cache. ')
                        image_aligned_new = copy_from_cache(Image, cache_dir, cache_name)
                        image_aligned_new.provenance = prov_aligned_new
                        image_aligned_new.info['original_image_id'] = ds.image_id
                        image_aligned_new.info['original_image_filepath'] = ds.image.filepath
                        image_aligned_new.info['alignment_parameters'] = prov.parameters['alignment']
                        image_aligned_new.save(verify_md5=False, no_archive=True)
                        # TODO: should we also load the aligned image's sources, PSF, and ZP?

                        if image_aligned_ref.mjd < image_aligned_new.mjd:
                            ds.sub_image._aligned_images = [image_aligned_ref, image_aligned_new]
                        else:
                            ds.sub_image._aligned_images = [image_aligned_new, image_aligned_ref]
                else:
                    SCLogger.debug( "make_datastore didn't find subtraction image in cache" )

            if ds.sub_image is None:  # no hit in the cache
                SCLogger.debug( "make_datastore running subtractor to create subtraction image" )
                ds = p.subtractor.run(ds, session)
                ds.sub_image.save(verify_md5=False)  # make sure it is also saved to archive
                if use_cache:
                    output_path = copy_to_cache(ds.sub_image, cache_dir)
                    if output_path != sub_cache_path:
                        warnings.warn(f'cache path {sub_cache_path} does not match output path {output_path}')

            if use_cache:  # save the aligned images to cache
                SCLogger.debug( "make_datastore saving aligned images to cache" )
                for im in ds.sub_image.aligned_images:
                    im.save(no_archive=True)
                    copy_to_cache(im, cache_dir)

            ############ detecting to create a source list ############
            prov = Provenance(
                code_version=code_version,
                process='detection',
                upstreams=[ds.sub_image.provenance],
                parameters=p.detector.pars.get_critical_pars(),
                is_testing=True,
            )
            prov = session.merge(prov)
            session.commit()

            cache_name = os.path.join(cache_dir, cache_sub_name + f'.sources_{prov.id[:6]}.npy.json')
            if use_cache and os.path.isfile(cache_name):
                SCLogger.debug( "make_datastore loading detections from cache." )
                ds.detections = copy_from_cache(SourceList, cache_dir, cache_name)
                ds.detections.provenance = prov
                ds.detections.image = ds.sub_image
                ds.sub_image.sources = ds.detections
                ds.detections.save(verify_md5=False)
            else:  # cannot find detections on cache
                SCLogger.debug( "make_datastore running detector to find detections" )
                ds = p.detector.run(ds, session)
                ds.detections.save(verify_md5=False)
                if use_cache:
                    copy_to_cache(ds.detections, cache_dir, cache_name)

            ############ cutting to create cutouts ############
            prov = Provenance(
                code_version=code_version,
                process='cutting',
                upstreams=[ds.detections.provenance],
                parameters=p.cutter.pars.get_critical_pars(),
                is_testing=True,
            )
            prov = session.merge(prov)
            session.commit()

            cache_name = os.path.join(cache_dir, cache_sub_name + f'.cutouts_{prov.id[:6]}.h5')
            if use_cache and ( os.path.isfile(cache_name) ):
                SCLogger.debug( 'make_datastore loading cutouts from cache.' )
                ds.cutouts = copy_from_cache(Cutouts, cache_dir, cache_name)
                ds.cutouts.provenance = prov
                ds.cutouts.sources = ds.detections
                ds.cutouts.load_all_co_data()  # sources must be set first
                ds.cutouts.save()  # make sure to save to archive as well
            else:  # cannot find cutouts on cache
                SCLogger.debug( "make_datastore running cutter to create cutouts" )
                ds = p.cutter.run(ds, session)
                ds.cutouts.save()
                if use_cache:
                    copy_to_cache(ds.cutouts, cache_dir)

            ############ measuring to create measurements ############
            prov = Provenance(
                code_version=code_version,
                process='measuring',
                upstreams=[ds.cutouts.provenance],
                parameters=p.measurer.pars.get_critical_pars(),
                is_testing=True,
            )
            prov = session.merge(prov)
            session.commit()

            cache_name = os.path.join(cache_dir, cache_sub_name + f'.measurements_{prov.id[:6]}.json')

            if use_cache and ( os.path.isfile(cache_name) ):
                # note that the cache contains ALL the measurements, not only the good ones
                SCLogger.debug( 'make_datastore loading measurements from cache.' )
                ds.all_measurements = copy_list_from_cache(Measurements, cache_dir, cache_name)
                [setattr(m, 'provenance', prov) for m in ds.all_measurements]
                [setattr(m, 'cutouts', ds.cutouts) for m in ds.all_measurements]

                ds.measurements = []
                for m in ds.all_measurements:
                    threshold_comparison = p.measurer.compare_measurement_to_thresholds(m)
                    if threshold_comparison != "delete":  # all disqualifiers are below threshold
                        m.is_bad = threshold_comparison == "bad"
                        ds.measurements.append(m)

                [m.associate_object(session) for m in ds.measurements]  # create or find an object for each measurement
                # no need to save list because Measurements is not a FileOnDiskMixin!
            else:  # cannot find measurements on cache
                SCLogger.debug( "make_datastore running measurer to create measurements" )
                ds = p.measurer.run(ds, session)
                if use_cache:
                    copy_list_to_cache(ds.all_measurements, cache_dir, cache_name)  # must provide filepath!

            SCLogger.debug( "make_datastore running ds.save_and_commit after subtraction/etc" )
            ds.save_and_commit(session=session)

            return ds

    return make_datastore
