import os
import warnings
import shutil
import pytest

import numpy as np

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import Provenance, CodeVersion
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

    (...this whole thing is a sort of more verbose implementation of
    pipeline/top_level.py...)

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
        SCLogger.debug( f"make_datastore called with args {args}, overrides={overrides}, augments={augments}" )

        if session is not None:
            raise RuntimeError( "Don't pass a session to make_datastore.  You're just asking for deadlocks." )
        
        code_version = None
        with SmartSession( session ) as sess:
            code_version = ( sess.query( CodeVersion )
                             .join( Provenance, Provenance.code_version_id == CodeVersion._id )
                             .filter( Provenance._id == args[0].provenance_id )
                            ).first()
        if code_version is None:
            raise RuntimeError( "make_datastore failed to get code_version" )

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

        ############ preprocessing to create image ############

        upstream_provs = []
        if ds.exposure is not None:
            upstream_provs.append( Provenance.get( ds.exposure.provenance_id ) )
        preprocessing_prov = Provenance(
            code_version=code_version,
            process='preprocessing',
            upstreams=upstream_provs,
            parameters=p.preprocessor.pars.get_critical_pars(),
            is_testing=True,
        )
        preprocessing_prov.insert_if_needed()
        ds.prov_tree = { 'preprocessing':  preprocessing_prov }

        if ds.image is None and use_cache:  # check if preprocessed image is in cache
            if os.path.isfile(image_cache_path):
                SCLogger.debug('make_datastore loading image from cache. ')
                img = copy_from_cache(Image, cache_dir, cache_name)
                # assign the correct exposure to the object loaded from cache
                if ds.exposure_id is not None:
                    img.exposure_id = ds.exposure_id
                if ds.exposure is not None:
                    img.exposure_id = ds.exposure.id
                ds.image = img

                # Copy the original image from the cache if requested
                if save_original_image:
                    ds.path_to_original_image = ds.image.get_fullpath()[0] + '.image.fits.original'
                    image_cache_path_original = os.path.join(cache_dir, ds.image.filepath + '.image.fits.original')
                    shutil.copy2( image_cache_path_original, ds.path_to_original_image )

                ds.image.provenance_id = preprocessing_prov.id

                # make sure this is saved to the archive as well
                ds.image.save(verify_md5=False)

        if ds.image is None:  # make the preprocessed image
            SCLogger.debug('make_datastore making preprocessed image. ')
            ds = p.preprocessor.run(ds)
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
        extraction_prov = Provenance(
            code_version=code_version,
            process='extraction',
            upstreams=[ preprocessing_prov ],
            parameters=p.extractor.pars.get_critical_pars(),  # the siblings will be loaded automatically
            is_testing=True,
        )
        extraction_prov.insert_if_needed()
        ds.prov_tree['extraction'] = extraction_prov

        if use_cache:  # try to get the SourceList, PSF, BG, WCS and ZP from cache

            # try to get the source list from cache
            cache_name = f'{cache_base_name}.sources_{extraction_prov.id[:6]}.fits.json'
            sources_cache_path = os.path.join(cache_dir, cache_name)
            if os.path.isfile(sources_cache_path):
                SCLogger.debug('make_datastore loading source list from cache. ')
                ds.sources = copy_from_cache(SourceList, cache_dir, cache_name)
                ds.sources.provenance_id = extraction_prov.id
                ds.sources.image_id = ds.image.id
                # make sure this is saved to the archive as well
                ds.sources.save(verify_md5=False)

            # try to get the PSF from cache
            cache_name = f'{cache_base_name}.psf_{extraction_prov.id[:6]}.fits.json'
            psf_cache_path = os.path.join(cache_dir, cache_name)
            if os.path.isfile(psf_cache_path):
                SCLogger.debug('make_datastore loading PSF from cache. ')
                ds.psf = copy_from_cache(PSF, cache_dir, cache_name)
                ds.psf.sources_id = ds.sources.id
                # make sure this is saved to the archive as well
                ds.psf.save( image=ds.image, sources=ds.sources, verify_md5=False, overwrite=True )

            # try to get the background from cache
            cache_name = f'{cache_base_name}.bg_{extraction_prov.id[:6]}.h5.json'
            bg_cache_path = os.path.join(cache_dir, cache_name)
            if os.path.isfile(bg_cache_path):
                SCLogger.debug('make_datastore loading background from cache. ')
                ds.bg = copy_from_cache( Background, cache_dir, cache_name,
                                         add_to_dict={ 'image_shape': ds.image.data.shape } )
                ds.bg.sources_id = ds.sources.id
                # make sure this is saved to the archive as well
                ds.bg.save( image=ds.image, sources=ds.sources, verify_md5=False, overwrite=True )

            # try to get the WCS from cache
            cache_name = f'{cache_base_name}.wcs_{extraction_prov.id[:6]}.txt.json'
            wcs_cache_path = os.path.join(cache_dir, cache_name)
            if os.path.isfile(wcs_cache_path):
                SCLogger.debug('make_datastore loading WCS from cache. ')
                ds.wcs = copy_from_cache(WorldCoordinates, cache_dir, cache_name)
                ds.wcs.sources_id = ds.sources.id
                # make sure this is saved to the archive as well
                ds.wcs.save( image=ds.image, sources=ds.sources, verify_md5=False, overwrite=True )

            # try to get the ZP from cache
            cache_name = cache_base_name + '.zp.json'
            zp_cache_path = os.path.join(cache_dir, cache_name)
            if os.path.isfile(zp_cache_path):
                SCLogger.debug('make_datastore loading zero point from cache. ')
                ds.zp = copy_from_cache(ZeroPoint, cache_dir, cache_name)
                ds.zp.sources_ids = ds.sources.id

        # if any data product is missing, must redo the extraction step
        if ds.sources is None or ds.psf is None or ds.bg is None or ds.wcs is None or ds.zp is None:
            # Clear out the existing database records
            for attr in [ 'zp', 'wcs', 'psf', 'bg', 'sources' ]:
                if getattr( ds, attr ) is not None:
                    getattr( ds, attr ).delete_from_disk_and_database()
                setattr( ds, attr, None )

            SCLogger.debug('make_datastore extracting sources. ')
            ds = p.extractor.run(ds, session)

            ds.sources.save( image=ds.image, overwrite=True )
            if use_cache:
                output_path = copy_to_cache(ds.sources, cache_dir)
                if output_path != sources_cache_path:
                    warnings.warn(f'cache path {sources_cache_path} does not match output path {output_path}')

            ds.psf.save( image=ds.image, sources=ds.sources, overwrite=True )
            if use_cache:
                output_path = copy_to_cache(ds.psf, cache_dir)
                if output_path != psf_cache_path:
                    warnings.warn(f'cache path {psf_cache_path} does not match output path {output_path}')

            SCLogger.debug('Running background estimation')
            ds = p.backgrounder.run(ds, session)

            ds.bg.save( image=ds.image, sources=ds.sources, overwrite=True )
            if use_cache:
                output_path = copy_to_cache(ds.bg, cache_dir)
                if output_path != bg_cache_path:
                    warnings.warn(f'cache path {bg_cache_path} does not match output path {output_path}')

            SCLogger.debug('Running astrometric calibration')
            ds = p.astrometor.run(ds, session)
            ds.wcs.save( image=ds.image, sources=ds.sources, overwrite=True )
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

        ########### Done with image and image data products; save and commit #############
                    
        SCLogger.debug( "make_datastore running ds.save_and_commit on image (before subtraction)" )
        ds.save_and_commit( session=session )

        # make a new copy of the image to cache, including the estimates for lim_mag, fwhm, etc.
        if not env_as_bool("LIMIT_CACHE_USAGE"):
            output_path = copy_to_cache(ds.image, cache_dir)

        ############ Now do subtraction / detection / measurement / etc. ##############
            
        # If we were told not to try to do a subtraction, then we're done
        if skip_sub:
            SCLogger.debug( "make_datastore : skip_sub is True, returning" )
            return ds

        ############ get the reference #############

        inst_name = ds.image.instrument.lower() if ds.image else ds.exposure.instrument.lower()
        refset_name = f'test_refset_{inst_name}'
        if inst_name == 'ptf':  # request the ptf_refset fixture dynamically:
            request.getfixturevalue('ptf_refset')
        if inst_name == 'decam':  # request the decam_refset fixture dynamically:
            request.getfixturevalue('decam_refset')

        with SmartSession() as sess:
            refset = sess.scalars(sa.select(RefSet).where(RefSet.name == refset_name)).first()

        if refset is None:
            raise ValueError(f'make_datastore found no reference with name {refset_name}')

        if len( refset.provenances ) == 0:
            SCLogger.debug( f"No reference provenances defined for refset {refset.name}, returning." )
            return ds

        # must provide the reference provenance explicitly since we didn't build a prov_tree
        ref = ds.get_reference(refset.provenances[0], session=session)
        if ref is None:
            SCLogger.debug( "make_datastore : could not find a reference, returning" )
            return ds  # if no reference is found, simply return the datastore without the rest of the products
        refimg = Image.get_by_id( ref.image_id )
        
        improvs = Provenance.get_batch( [ ds.image.provenance_id, ds.sources.provenance_id ] )
        refprovs = Provenance.get_batch( [ ds.ref_image.provenance_id, ds.ref_sources.provenance_id ] )
        bothprovs = improvs + refprovs

        sub_prov = Provenance(
            code_version=code_version,
            process='subtraction',
            upstreams=bothprovs,
            parameters=p.subtractor.pars.get_critical_pars(),
            is_testing=True,
        )
        sub_prov.insert_if_needed()
        ds.prov_tree['subtraction'] = sub_prov

        ########### find or run the subtraction ##########

        if use_cache:  # try to find the subtraction image in the cache
            SCLogger.debug( "make_datstore looking for subtraction image in cache..." )

            sub_im = Image.from_new_and_ref(ds.image, refimg)
            sub_im.provenance_id = sub_prov.id
            cache_sub_name = sub_im.invent_filepath()
            cache_name = cache_sub_name + '.image.fits.json'
            sub_cache_path = os.path.join(cache_dir, cache_name)
            zogy_score_cache_path = sub_cache_path.replace( ".image.fits.json", ".zogy_score.npy" )
            zogy_alpha_cache_path = sub_cache_path.replace( ".image.fits.json", ".zogy_alpha.npy" )

            prov_aligned_ref = Provenance(
                code_version=code_version,
                parameters=sub_prov.parameters['alignment'],
                upstreams=bothprovs,
                process='alignment',
                is_testing=True,
            )
            f = refimg.invent_filepath()
            f = f.replace('ComSci', 'Warped')  # not sure if this or 'Sci' will be in the filename
            f = f.replace('Sci', 'Warped')     # in any case, replace it with 'Warped'
            f = f[:-6] + prov_aligned_ref.id[:6]  # replace the provenance ID
            filename_aligned_ref = f
            cache_name_aligned_ref = filename_aligned_ref + '.image.fits.json'
            aligned_ref_cache_path = os.path.join( cache_dir, cache_name_aligned_ref )

            # Commenting this out -- we know that we're aligning to new,
            #   do don't waste cache on aligned_new
            # prov_aligned_new = Provenance(
            #     code_version=code_version,
            #     parameters=sub_prov.parameters['alignment'],
            #     upstreams=bothprovs,
            #     process='alignment',
            #     is_testing=True,
            # )
            # f = ds.image.invent_filepath()
            # f = f.replace('ComSci', 'Warped')
            # f = f.replace('Sci', 'Warped')
            # f = f[:-6] + prov_aligned_new.id[:6]
            # filename_aligned_new = f

            if ( ( os.path.isfile(sub_cache_path) ) and
                 ( os.path.isfile(zogy_score_cache_path) ) and
                 ( os.path.isfile(zogy_alpha_cache_path) ) and
                 ( os.path.isfile(aligned_ref_cache_path) ) ):
                SCLogger.debug('make_datastore loading subtraction image from cache: {sub_cache_path}" ')
                tmpsubim =  copy_from_cache(Image, cache_dir, cache_name)
                tmpsubim.provenance_id = sub_prov.id
                tmpsubim._upstream_ids = sub_im._upstream_ids
                tmpsubim.ref_image_id = ref.image_id
                tmpsubim.save(verify_md5=False)  # make sure it is also saved to archive
                ds.sub_image = tmpsubim
                ds.zogy_score = np.load( zogy_score_cache_path )
                ds.zogy_alpha = np.load( zogy_alpha_cache_path )

                ds.aligned_new_image = ds.image

                SCLogger.debug('loading aligned reference image from cache. ')
                image_aligned_ref = copy_from_cache( Image, cache_dir, cache_name_aligned_ref )
                image_aligned_ref.provenance_id = prov_aligned_ref.id
                image_aligned_ref.info['original_image_id'] = ds.ref_image.id
                image_aligned_ref.info['original_image_filepath'] = ds.ref_image.filepath
                image_aligned_ref.info['alignment_parameters'] = sub_prov.parameters['alignment']
                # TODO FIGURE OUT WHAT'S GOING ON HERE
                # Not sure why the md5sum_extensions was [], but it was
                image_aligned_ref.md5sum_extensions = [ None, None, None ]
                image_aligned_ref.save(verify_md5=False, no_archive=True)
                # TODO: should we also load the aligned image's sources, PSF, and ZP?
                ds.aligned_ref_image = image_aligned_ref

            else:
                SCLogger.debug( "make_datastore didn't find subtraction image in cache" )

        if ds.sub_image is None:  # no hit in the cache
            SCLogger.debug( "make_datastore running subtractor to create subtraction image" )
            ds = p.subtractor.run( ds, session )
            ds.sub_image.save(verify_md5=False)  # make sure it is also saved to archive
            if use_cache:
                output_path = copy_to_cache(ds.sub_image, cache_dir)
                if output_path != sub_cache_path:
                    raise ValueError( f'cache path {sub_cache_path} does not match output path {output_path}' )
                    # warnings.warn(f'cache path {sub_cache_path} does not match output path {output_path}')
                np.save( zogy_score_cache_path, ds.zogy_score, allow_pickle=False )
                np.save( zogy_alpha_cache_path, ds.zogy_alpha, allow_pickle=False )

                SCLogger.debug( "make_datastore saving aligned ref image to cache" )
                ds.aligned_ref_image.save( no_archive=True )
                copy_to_cache( ds.aligned_ref_image, cache_dir )

        ############ detecting to create a source list ############
        detection_prov =  Provenance(
            code_version=code_version,
            process='detection',
            upstreams=[ sub_prov ],
            parameters=p.detector.pars.get_critical_pars(),
            is_testing=True,
        )
        detection_prov.insert_if_needed()

        cache_name = os.path.join(cache_dir, cache_sub_name + f'.sources_{detection_prov.id[:6]}.npy.json')
        if use_cache and os.path.isfile(cache_name):
            SCLogger.debug( "make_datastore loading detections from cache." )
            ds.detections = copy_from_cache(SourceList, cache_dir, cache_name)
            ds.detections.provenance_id = detection_prov.id
            ds.detections.image_id = ds.sub_image.id
            ds.detections.save(verify_md5=False)
        else:  # cannot find detections on cache
            SCLogger.debug( "make_datastore running detector to find detections" )
            ds = p.detector.run(ds, session)
            ds.detections.save( image=ds.sub_image, verify_md5=False )
            if use_cache:
                copy_to_cache( ds.detections, cache_dir, cache_name )

        ############ cutting to create cutouts ############
        cutting_prov = Provenance(
            code_version=code_version,
            process='cutting',
            upstreams=[ detection_prov ],
            parameters=p.cutter.pars.get_critical_pars(),
            is_testing=True,
        )
        cutting_prov.insert_if_needed()
        ds.prov_tree['cutting'] = cutting_prov

        cache_name = os.path.join(cache_dir, cache_sub_name + f'.cutouts_{cutting_prov.id[:6]}.h5')
        if use_cache and ( os.path.isfile(cache_name) ):
            SCLogger.debug( 'make_datastore loading cutouts from cache.' )
            ds.cutouts = copy_from_cache(Cutouts, cache_dir, cache_name)
            ds.cutouts.provenance_id = cutting_prov.id
            ds.cutouts.sources_id = ds.detections.id
            ds.cutouts.load_all_co_data( sources=ds.detections )
            ds.cutouts.save( image=ds.sub_image, sources=ds.detections )  # make sure to save to archive as well
        else:  # cannot find cutouts on cache
            SCLogger.debug( "make_datastore running cutter to create cutouts" )
            ds = p.cutter.run(ds, session)
            ds.cutouts.save( image=ds.sub_image, sources=ds.detections )
            if use_cache:
                copy_to_cache(ds.cutouts, cache_dir)

        ############ measuring to create measurements ############
        measuring_prov = Provenance(
            code_version=code_version,
            process='measuring',
            upstreams=[ cutting_prov ],
            parameters=p.measurer.pars.get_critical_pars(),
            is_testing=True,
        )
        measuring_prov.insert_if_needed()
        ds.prov_tree['measuring'] = measuring_prov

        all_measurements_cache_name = os.path.join( cache_dir,
                                                    cache_sub_name + f'.all_measurements_{measuring_prov.id[:6]}.json')
        measurements_cache_name = os.path.join(cache_dir, cache_sub_name + f'.measurements_{measuring_prov.id[:6]}.json')

        if use_cache and ( os.path.isfile(measurements_cache_name) ) and ( os.path.isfile(all_measurements_cache_name) ):
            SCLogger.debug( 'make_datastore loading measurements from cache.' )
            ds.measurements = copy_list_from_cache(Measurements, cache_dir, measurements_cache_name)
            [ setattr(m, 'provenance_id', measuring_prov.id) for m in ds.measurements ]
            [ setattr(m, 'cutouts_id', ds.cutouts.id) for m in ds.measurements ]

            # Note that the actual measurement objects in the two lists
            # won't be the same objects (they will be equivalent
            # objects), whereas when they are created in the first place
            # I think they're the same objects.  As long as we're
            # treating measurements as read-only, except for a bit of
            # memory usage this shouldn't matter.
            ds.all_measurements = copy_list_from_cache(Measurements, cache_dir, all_measurements_cache_name)
            [ setattr(m, 'provenance_id', measuring_prov.id) for m in ds.all_measurements ]
            [ setattr(m, 'cutouts_id', ds.cutouts.id) for m in ds.all_measurements ]

            # Because the Object association wasn't run, we have to do that manually
            with SmartSession( session ) as sess:
                for m in ds.measurements:
                    m.associate_object( p.measurer.pars.association_radius,
                                        is_testing=measuring_prov.is_testing,
                                        session=sess )
                                        
        else:  # cannot find measurements on cache
            SCLogger.debug( "make_datastore running measurer to create measurements" )
            ds = p.measurer.run(ds, session)
            if use_cache:
                copy_list_to_cache(ds.all_measurements, cache_dir, all_measurements_cache_name)
                copy_list_to_cache(ds.measurements, cache_dir, measurements_cache_name)

            

        # Make sure there are no residual exceptions caught in the datastore
        assert ds.exception is None
                
        SCLogger.debug( "make_datastore running ds.save_and_commit after subtraction/etc" )
        ds.save_and_commit(session=session)

        return ds

    return make_datastore
