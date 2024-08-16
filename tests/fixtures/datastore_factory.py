import os
import warnings
import shutil
import pytest

import numpy as np

import sqlalchemy as sa

from models.base import SmartSession
from models.provenance import Provenance, CodeVersion
from models.enums_and_bitflags import BitFlagConverter
from models.exposure import Exposure
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
            exporim,
            section_id=None,
            cache_dir=None,
            cache_base_name=None,
            overrides=None,
            augments=None,
            bad_pixel_map=None,
            save_original_image=False,
            skip_sub=False,
            through_step=None,
            provtag='datastore_factory'
    ):
        """Create a DataStore for testing purposes.

        The datastore you get back will at least have the .image field
        loaded; whether or not further fields are loaded depend on the
        setting of through_step and whether or not there's a reference
        available.  If there is a reference available (regardless of the
        setting of through_step), the .reference field will also be
        loaded.  prov_tree will be loaded with preprocessing and
        extraction, and if there's a reference available, with
        everything else as well.

        The datastore will also come with a custom _pipeline attribute.
        This is not standard for DataStore, but is used in a lot of the
        tests (to get the various pipeline processing objects that are
        consistent with the provenances loaded into the DataStore's
        prov_tree).

        Parameters
        ----------
          exporim: Exposure or Image

          section_id: str or None
            Ignored if exporim is an Image

          cache_dir: str, default None

          cache_base_name: str, defautl None

          overrides: dict, default None
            If passed, overrides parameters sent to pipeline_factory

          augments: dict, default None
            If passed, augments parameters sent to pipeline_factory

          bad_pixel_mnap:

          save_original_image: bool, default False
            If True, will write a file '....image.fits.original' next to
            '....image.fits' for the main image of the DataSTore (used
            in some tests).

          skip_sub: bool, default False
            Equvialent through_step='zp'; ignored if through_step is not None

          through_step: str, default None
            If passed, will only run processing through this step.  One
            of preprocessing, extraction, bg, wcs, zp, subtraction,
            detection, cutting, measuring.  (Can't do extraction without
            psf, as those are done in a single function call.)

          provtag: str, default 'datastore_factory'

        """

        SCLogger.debug( f"make_datastore called with a {type(exporim).__name__}, "
                        f"overrides={overrides}, augments={augments}" )

        overrides = {} if overrides is None else overrides
        augments = {} if augments is None else augments

        stepstodo = [ 'preprocessing', 'extraction', 'bg', 'wcs', 'zp',
                      'subtraction', 'detection', 'cutting', 'measuring' ]
        if through_step is None:
            if skip_sub:
                through_step = 'zp'
            else:
                through_step = 'measuring'
        dex = stepstodo.index( through_step )
        stepstodo = stepstodo[:dex+1]

        # Make the datastore
        if isinstance( exporim, Exposure ):
            ds = DataStore( exporim, section_id )
        elif isinstance( exporim, Image ):
            ds = DataStore( exporim )
        else:
            raise RuntimeError( "Error, datastory_factory must start from either an exposure or an image." )

        # Set up the cache if appropriate
        use_cache = cache_dir is not None and cache_base_name is not None and not env_as_bool( "LIMIT_CACHE_USAGE" )
        if cache_base_name is not None:
            cache_name = cache_base_name + '.image.fits.json'
            image_cache_path = os.path.join(cache_dir, cache_name)
        else:
            image_cache_path = None
        if use_cache:
            ds.cache_base_name = os.path.join(cache_dir, cache_base_name)  # save this for testing purposes

        # This fixture uses a standard refset.  Update the pipline parameters accordingly.
        refset_name = None
        if 'subtraction' in stepstodo:
            inst_name = ds.image.instrument.lower() if ds.image else ds.exposure.instrument.lower()
            refset_name = f'test_refset_{inst_name}'
            if inst_name == 'ptf':  # request the ptf_refset fixture dynamically:
                request.getfixturevalue('ptf_refset')
            if inst_name == 'decam':  # request the decam_refset fixture dynamically:
                request.getfixturevalue('decam_refset')

            if 'subtraction' not in overrides:
                overrides['subtraction'] = {}
            overrides['subtraction']['refset'] = refset_name

        # Create the pipeline and build the provenance tree

        p = pipeline_factory( provtag )
        ds._pipeline = p

        # allow calling scope to override/augment parameters for any of the processing steps
        p.override_parameters(**overrides)
        p.augment_parameters(**augments)

        ds.prov_tree = p.make_provenance_tree( ds.exposure if ds.exposure is not None else ds.image,
                                               ok_no_ref_provs=True )

        # Remove all steps past subtraction if there's no referencing provenance
        if ( 'subtraction' in stepstodo ) and ( 'referencing' not in ds.prov_tree ):
            SCLogger.debug( "datastore_factory: No reference set, or no reference image provenances, found; "
                            "removing all steps from subtraction on from steps to  perform." )
            subdex = stepstodo.index( 'subtraction' )
            stepstodo = stepstodo[:subdex]


        ############ preprocessing to create image ############

        if 'preprocessing' in stepstodo:

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

                    ds.image.provenance_id = ds.prov_tree['preprocessing'].id

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
                    # Don't copy the image to the cache -- the image database record
                    #  is going to get further modified in subsequent setps.  We don't
                    #  want an incomplete cache if those steps aren't done.
                    # Image copying to cache happens after the zp step.
                    output_path = copy_to_cache(ds.image, cache_dir, dont_actually_copy_just_return_json_filepath=True)
                    if image_cache_path is not None and output_path != image_cache_path:
                        warnings.warn(f'cache path {image_cache_path} does not match output path {output_path}')
                    else:
                        cache_base_name = output_path[:-16]  # remove the '.image.fits.json' part
                        ds.cache_base_name = output_path
                        SCLogger.debug(f'Saving image to cache at: {output_path}')
                        # use_cache = True  # the two other conditions are true to even get to this part...

                # In test_astro_cal, there's a routine that needs the original
                # image before being processed through the rest of what this
                # factory function does, so save it if requested
                if save_original_image:
                    ds.path_to_original_image = ds.image.get_fullpath()[0] + '.image.fits.original'
                    shutil.copy2( ds.image.get_fullpath()[0], ds.path_to_original_image )
                    if use_cache:
                        shutil.copy2( ds.image.get_fullpath()[0],
                                      os.path.join(cache_dir, ds.image.filepath + '.image.fits.original') )


        ############# extraction to create sources / PSF  #############

        filename_barf = ds.prov_tree['extraction'].id[:6]

        if 'extraction' in stepstodo:

            found_sources_in_cache = False
            if use_cache:
                # try to get the source list from cache
                cache_name = f'{cache_base_name}.sources_{filename_barf}.fits.json'
                sources_cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(sources_cache_path):
                    SCLogger.debug('make_datastore loading source list from cache. ')
                    ds.sources = copy_from_cache(SourceList, cache_dir, cache_name)
                    ds.sources.provenance_id = ds.prov_tree['extraction'].id
                    ds.sources.image_id = ds.image.id
                    # make sure this is saved to the archive as well
                    ds.sources.save(verify_md5=False)
                    found_sources_in_cache = True

                # try to get the PSF from cache
                cache_name = f'{cache_base_name}.psf_{filename_barf}.fits.json'
                psf_cache_path = os.path.join(cache_dir, cache_name)
                if os.path.isfile(psf_cache_path):
                    SCLogger.debug('make_datastore loading PSF from cache. ')
                    ds.psf = copy_from_cache(PSF, cache_dir, cache_name)
                    ds.psf.sources_id = ds.sources.id
                    # make sure this is saved to the archive as well
                    ds.psf.save( image=ds.image, sources=ds.sources, verify_md5=False, overwrite=True )
                else:
                    found_sources_in_cache = False

            # if sources or psf is missing, have to redo the extraction step
            if ds.sources is None or ds.psf is None:
                # Clear out the existing database records
                for attr in [ 'zp', 'wcs', 'psf', 'bg', 'sources' ]:
                    if getattr( ds, attr ) is not None:
                        getattr( ds, attr ).delete_from_disk_and_database()
                    setattr( ds, attr, None )

                SCLogger.debug('make_datastore extracting sources. ')
                ds = p.extractor.run(ds)

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

        ########## Background ##########

        if 'bg' in stepstodo:
            cache_name = f'{cache_base_name}.bg_{filename_barf}.h5.json'
            bg_cache_path = os.path.join(cache_dir, cache_name)
            if use_cache and found_sources_in_cache:
                # try to get the background from cache
                if os.path.isfile(bg_cache_path):
                    SCLogger.debug('make_datastore loading background from cache. ')
                    ds.bg = copy_from_cache( Background, cache_dir, cache_name,
                                             add_to_dict={ 'image_shape': ds.image.data.shape } )
                    ds.bg.sources_id = ds.sources.id
                    # make sure this is saved to the archive as well
                    ds.bg.save( image=ds.image, sources=ds.sources, verify_md5=False, overwrite=True )


            if ds.bg is None:
                SCLogger.debug('Running background estimation')
                ds = p.backgrounder.run(ds)

                ds.bg.save( image=ds.image, sources=ds.sources, overwrite=True )
                if use_cache:
                    output_path = copy_to_cache(ds.bg, cache_dir)
                    if output_path != bg_cache_path:
                        warnings.warn(f'cache path {bg_cache_path} does not match output path {output_path}')

        ########## Astrometric calibration ##########

        if 'wcs' in stepstodo:
            cache_name = f'{cache_base_name}.wcs_{filename_barf}.txt.json'
            wcs_cache_path = os.path.join(cache_dir, cache_name)
            if use_cache and found_sources_in_cache:
                # try to get the WCS from cache
                if os.path.isfile(wcs_cache_path):
                    SCLogger.debug('make_datastore loading WCS from cache. ')
                    ds.wcs = copy_from_cache(WorldCoordinates, cache_dir, cache_name)
                    ds.wcs.sources_id = ds.sources.id
                    # make sure this is saved to the archive as well
                    ds.wcs.save( image=ds.image, sources=ds.sources, verify_md5=False, overwrite=True )

            if ds.wcs is None:
                SCLogger.debug('Running astrometric calibration')
                ds = p.astrometor.run(ds)
                ds.wcs.save( image=ds.image, sources=ds.sources, overwrite=True )
                if use_cache:
                    output_path = copy_to_cache(ds.wcs, cache_dir)
                    if output_path != wcs_cache_path:
                        warnings.warn(f'cache path {wcs_cache_path} does not match output path {output_path}')

        ########## Photometric calibration ##########

        if 'zp' in stepstodo:
            cache_name = cache_base_name + '.zp.json'
            zp_cache_path = os.path.join(cache_dir, cache_name)
            if use_cache and found_sources_in_cache:
                # try to get the ZP from cache
                if os.path.isfile(zp_cache_path):
                    SCLogger.debug('make_datastore loading zero point from cache. ')
                    ds.zp = copy_from_cache(ZeroPoint, cache_dir, cache_name)
                    ds.zp.sources_ids = ds.sources.id

            if ds.zp is None:
                SCLogger.debug('Running photometric calibration')
                ds = p.photometor.run(ds)
                if use_cache:
                    cache_name = cache_base_name + '.zp.json'
                    output_path = copy_to_cache(ds.zp, cache_dir, cache_name)
                    if output_path != zp_cache_path:
                        warnings.warn(f'cache path {zp_cache_path} does not match output path {output_path}')

        ########### Done with image and image data products; save and commit #############

        SCLogger.debug( "make_datastore running ds.save_and_commit on image (before subtraction)" )
        ds.save_and_commit()

        # *Now* copy the image to cache, including the estimates for lim_mag, fwhm, etc.
        if not env_as_bool("LIMIT_CACHE_USAGE"):
            output_path = copy_to_cache(ds.image, cache_dir)

        # Make sure there are no residual errors in the datastore
        assert ds.exception is None

        ############ Now do subtraction / detection / measurement / etc. ##############

        ########## subtraction ##########

        if 'subtraction' in stepstodo:

            ########## look for a reference ##########

            with SmartSession() as sess:
                refset = sess.scalars(sa.select(RefSet).where(RefSet.name == refset_name)).first()

                if refset is None:
                    SCLogger.debug( f"No refset found with name {refset_name}, returning." )
                    return ds

                if len( refset.provenances ) == 0:
                    SCLogger.debug( f"No reference provenances defined for refset {refset.name}, returning." )
                    return ds

                ref = ds.get_reference()
                if ( ref is None ) and ( 'subtraction' in stepstodo ):
                    SCLogger.debug( "make_datastore : could not find a reference, returning." )
                    return ds


            ########### find or run the subtraction ##########

            if use_cache:  # try to find the subtraction image in the cache
                SCLogger.debug( "make_datstore looking for subtraction image in cache..." )

                sub_im = Image.from_new_and_ref( ds.image, ds.ref_image )
                sub_im.provenance_id = ds.prov_tree['subtraction'].id
                cache_sub_name = sub_im.invent_filepath()
                cache_name = cache_sub_name + '.image.fits.json'
                sub_cache_path = os.path.join(cache_dir, cache_name)
                zogy_score_cache_path = sub_cache_path.replace( ".image.fits.json", ".zogy_score.npy" )
                zogy_alpha_cache_path = sub_cache_path.replace( ".image.fits.json", ".zogy_alpha.npy" )

                alignupstrprovs = Provenance.get_batch( [ ds.image.provenance_id,
                                                          ds.sources.provenance_id,
                                                          ds.ref_image.provenance_id,
                                                          ds.ref_sources.provenance_id ] )

                prov_aligned_ref = Provenance(
                    code_version_id=ds.prov_tree['subtraction'].code_version_id,
                    parameters=ds.prov_tree['subtraction'].parameters['alignment'],
                    upstreams=alignupstrprovs,
                    process='alignment',
                    is_testing=True,
                )
                f = ds.ref_image.invent_filepath()
                f = f.replace('ComSci', 'Warped')  # not sure if this or 'Sci' will be in the filename
                f = f.replace('Sci', 'Warped')     # in any case, replace it with 'Warped'
                f = f[:-6] + prov_aligned_ref.id[:6]  # replace the provenance ID
                filename_aligned_ref = f
                cache_name_aligned_ref = filename_aligned_ref + '.image.fits.json'
                aligned_ref_cache_path = os.path.join( cache_dir, cache_name_aligned_ref )

                # Commenting this out -- we know that we're aligning to new,
                #   do don't waste cache on aligned_new
                # prov_aligned_new = Provenance(
                #     code_version_id=code_version.id,
                #     parameters=ds.prov_tree['subtraction'].parameters['alignment'],
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
                    tmpsubim.provenance_id = ds.prov_tree['subtraction'].id
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
                    image_aligned_ref.info['alignment_parameters'] = ds.prov_tree['subtraction'].parameters['alignment']
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
                ds = p.subtractor.run( ds )
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

        if 'detection' in stepstodo:
            cache_name = os.path.join(cache_dir, cache_sub_name +
                                      f'.sources_{ds.prov_tree["detection"].id[:6]}.npy.json')
            if use_cache and os.path.isfile(cache_name):
                SCLogger.debug( "make_datastore loading detections from cache." )
                ds.detections = copy_from_cache(SourceList, cache_dir, cache_name)
                ds.detections.provenance_id = ds.prov_tree['detection'].id
                ds.detections.image_id = ds.sub_image.id
                ds.detections.save(verify_md5=False)
            else:  # cannot find detections on cache
                SCLogger.debug( "make_datastore running detector to find detections" )
                ds = p.detector.run(ds)
                ds.detections.save( image=ds.sub_image, verify_md5=False )
                if use_cache:
                    copy_to_cache( ds.detections, cache_dir, cache_name )

        ############ cutting to create cutouts ############

        if 'cutting' in stepstodo:
            cache_name = os.path.join(cache_dir, cache_sub_name +
                                      f'.cutouts_{ds.prov_tree["cutting"].id[:6]}.h5')
            if use_cache and ( os.path.isfile(cache_name) ):
                SCLogger.debug( 'make_datastore loading cutouts from cache.' )
                ds.cutouts = copy_from_cache(Cutouts, cache_dir, cache_name)
                ds.cutouts.provenance_id = ds.prov_tree['cutting'].id
                ds.cutouts.sources_id = ds.detections.id
                ds.cutouts.load_all_co_data( sources=ds.detections )
                ds.cutouts.save( image=ds.sub_image, sources=ds.detections )  # make sure to save to archive as well
            else:  # cannot find cutouts on cache
                SCLogger.debug( "make_datastore running cutter to create cutouts" )
                ds = p.cutter.run(ds)
                ds.cutouts.save( image=ds.sub_image, sources=ds.detections )
                if use_cache:
                    copy_to_cache(ds.cutouts, cache_dir)

        ############ measuring to create measurements ############

        if 'measuring' in stepstodo:
            all_measurements_cache_name = os.path.join( cache_dir,
                                                        cache_sub_name +
                                                        f'.all_measurements_{ds.prov_tree["measuring"].id[:6]}.json')
            measurements_cache_name = os.path.join(cache_dir, cache_sub_name +
                                                   f'.measurements_{ds.prov_tree["measuring"].id[:6]}.json')

            if ( use_cache and
                 os.path.isfile(measurements_cache_name) and
                 os.path.isfile(all_measurements_cache_name)
                ):
                SCLogger.debug( 'make_datastore loading measurements from cache.' )
                ds.measurements = copy_list_from_cache(Measurements, cache_dir, measurements_cache_name)
                [ setattr(m, 'provenance_id', ds.prov_tree['measuring'].id) for m in ds.measurements ]
                [ setattr(m, 'cutouts_id', ds.cutouts.id) for m in ds.measurements ]

                # Note that the actual measurement objects in the two lists
                # won't be the same objects (they will be equivalent
                # objects), whereas when they are created in the first place
                # I think they're the same objects.  As long as we're
                # treating measurements as read-only, except for a bit of
                # memory usage this shouldn't matter.
                ds.all_measurements = copy_list_from_cache(Measurements, cache_dir, all_measurements_cache_name)
                [ setattr(m, 'provenance_id', ds.prov_tree['measuring'].id) for m in ds.all_measurements ]
                [ setattr(m, 'cutouts_id', ds.cutouts.id) for m in ds.all_measurements ]

                # Because the Object association wasn't run, we have to do that manually
                with SmartSession() as sess:
                    for m in ds.measurements:
                        m.associate_object( p.measurer.pars.association_radius,
                                            is_testing=ds.prov_tree['measuring'].is_testing,
                                            session=sess )

            else:  # cannot find measurements on cache
                SCLogger.debug( "make_datastore running measurer to create measurements" )
                ds = p.measurer.run(ds)
                if use_cache:
                    copy_list_to_cache(ds.all_measurements, cache_dir, all_measurements_cache_name)
                    copy_list_to_cache(ds.measurements, cache_dir, measurements_cache_name)



        # Make sure there are no residual exceptions caught in the datastore
        assert ds.exception is None

        SCLogger.debug( "make_datastore running ds.save_and_commit after subtraction/etc" )
        ds.save_and_commit()

        return ds

    return make_datastore
