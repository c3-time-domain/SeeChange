import io
import pathlib
import warnings
import shutil
import pytest
import datetime

import numpy as np

import astropy.time
import sqlalchemy as sa

from models.base import SmartSession, Psycopg2Connection
from models.provenance import Provenance
from models.enums_and_bitflags import BitFlagConverter #, string_to_bitflag, flag_image_bits_inverse
from models.report import Report
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.cutouts import Cutouts
from models.measurements import Measurements, MeasurementSet
from models.deepscore import DeepScore, DeepScoreSet
from models.refset import RefSet
from models.object import Object
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

    The returned DataStore will have a property _pipeline that holds the
    pipeline used to create the data products.

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
            If (at least) it's an Exposure, it must already be loaded into the database.

          section_id: str or None
            Ignored if exporim is an Image

          cache_dir: str or Path, default None
            The top-level directory of the cache.

          cache_base_name: str, default None
            The filepath of the Image that will be created; this is used
            to search the cache for the desired object.  This should
            *not* include the ".image.fits" suffix, or the ".json"
            suffix.  It should be relative to cache_dir, not an
            absolute path.

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

        SCLogger.debug( f"make_datastore called with a {type(exporim).__name__};\n"
                        f"      overrides={overrides}\n"
                        f"      augments={augments}\n"
                        f"      cache_dir={cache_dir}\n"
                        f"      cache_base_name={cache_base_name}\n"
                        f"      bad_pixel_map is a {type(bad_pixel_map)}\n"
                        f"      save_original_image={save_original_image}\n"
                        f"      skip_sub={skip_sub}\n"
                        f"      through_step={through_step}\n"
                        f"      provtag={provtag}" )

        overrides = {} if overrides is None else overrides
        augments = {} if augments is None else augments

        if env_as_bool('SEECHANGE_TRACEMALLOC'):
            import tracemalloc
            tracemalloc.start()

        cache_dir = pathlib.Path( cache_dir ) if cache_dir is not None else None

        stepstodo = [ 'preprocessing', 'extraction', 'bg', 'wcs', 'zp',
                      'subtraction', 'detection', 'cutting', 'measuring', 'scoring' ]
        if through_step is None:
            if skip_sub:
                through_step = 'zp'
            else:
                through_step = 'scoring'
        dex = stepstodo.index( through_step )
        stepstodo = stepstodo[:dex+1]

        # Make the datastore
        if isinstance( exporim, Exposure ):
            ds = DataStore( exporim, section_id )
        elif isinstance( exporim, Image ):
            ds = DataStore( exporim )
        else:
            raise RuntimeError( "Error, datastory_factory must start from either an exposure or an image." )

        ds.update_runtimes = True
        ds.update_memory_usages = env_as_bool( 'SEECHANGE_TRACEMALLOC' )

        # Set up the cache if appropriate
        # cache_base_path is the path of the image cache file, minus .json, relative to cache_dir
        # ds.cache_base_path is the absolute equivalent
        use_cache = cache_dir is not None and cache_base_name is not None and not env_as_bool( "LIMIT_CACHE_USAGE" )
        image_was_loaded_from_cache = False
        if cache_base_name is not None:
            cache_base_path = pathlib.Path( cache_base_name )
        else:
            cache_base_path = None
        if use_cache:
            ds.cache_base_path = cache_dir / cache_base_path

        # This fixture uses a standard refset.  Update the pipline parameters accordingly.
        refset_name = None
        if 'subtraction' in stepstodo:
            inst_name = ds.image.instrument.lower() if ds.image else ds.exposure.instrument.lower()
            refset_name = f'test_refset_{inst_name}'
            if 'subtraction' not in overrides:
                overrides['subtraction'] = {}
            overrides['subtraction']['refset'] = refset_name

        # Create the pipeline and build the provenance tree
        # We're not just going to call pipeline.run() because of all the
        #   cache reading/writing below.  Instead, we call the
        #   individual steps directly.  That makes this fixture way
        #   bigger than it would be without the cache, but if a fixture
        #   is reused, we can save a bunch of time by caching the
        #   results.  (The fixture is still kind of slow because even
        #   restoring the cache takes time — ~tens of seconds for a full
        #   subtraction/measurement datastore.)
        # Note that we access the .id field of lots of things before copying
        #   them to the cache.  The reason for that is that we want the
        #   uuid to be generated before it's saved to the cache.  Some later
        #   steps may behave differently based on that id.  For instance, the
        #   invent_filepath method of image will based the "utag" part of the
        #   filepath on the uuids of the zeropoints that went into a coadd,
        #   or of the reference and zeropoint that went into a subtraction.

        p = pipeline_factory( provtag )
        ds._pipeline = p

        # allow calling scope to override/augment parameters for any of the processing steps
        p.override_parameters(**overrides)
        p.augment_parameters(**augments)

        p.setup_datastore( ds, ok_no_ref_prov=True )

        if isinstance( exporim, Exposure ) and ( not env_as_bool("LIMIT_CACHE_USAGE") ) and cache_dir.is_dir():
            # If we didn't know the cache base path before, we should
            #   figure it out from the exposure and section id.
            # (OMG all this cache stuff is tangled.)
            im = Image.from_exposure( exporim, section_id )
            im.provenance_id = ds.prov_tree['preprocessing'].id
            new_cache_base_path = pathlib.Path( im.invent_filepath() )
            use_cache = True
            if cache_base_path is None:
                cache_base_path = new_cache_base_path
                report_cache_path = cache_dir / cache_base_path.parent / f'{cache_base_path.name}.report.json'
                ds.cache_base_path = cache_dir / cache_base_path
            elif new_cache_base_path != cache_base_path:
                warnings.warn( f'Determined cache_base_path {new_cache_base_path} '
                               f'from exposure/section_id, but it does not match the '
                               f'pre-existing cache_base_path {cache_base_path}' )

        # Try to read the report from the cache.  If it's there, then _hopefully_ everything
        # else is there.  (Report cache is written at the end, but it's possible that there
        # will be provenance mismatches.  If that happens, then the report read from the cache
        # will be wrong.  Clear your cache to be sure.)
        #
        # (We can only make reports if given an exposure, so skip all report stuff if we
        # were given an image.)
        if p._generate_report:
            report_was_loaded_from_cache = False
            report_cache_path = None
            if cache_base_path is not None:
                report_cache_path = cache_dir / cache_base_path.parent / f'{cache_base_path.name}.report.json'
                SCLogger.debug( f'make_datastore searching cache for report {report_cache_path}' )
            if use_cache and ( report_cache_path is not None ) and report_cache_path.is_file():
                SCLogger.debug( 'make_datastore loading report from cache' )
                cached_report = copy_from_cache( Report, cache_dir, report_cache_path, symlink=True )
                # The cached exposure id won't be right
                cached_report.exposure_id = exporim.id
                # TODO -- I want this next line to be ds.report.insert().  And, indeed,
                #   when I run all the tests on my local machine, it works.  However,
                #   when running the tests on github actions, in two tests this was
                #   raising an error, saying that the report id already existed.
                #   This is of course very hard to track down, since if you can't
                #   find it on your local machine, doing any debugging is basically
                #   impossible.  This is almost certainly some cache handling thing,
                #   and the cache is only used in the tests, so to get on with life
                #   I replaced this insert with upsert.  See Issue #378 ; if we ever
                #   care enough to track this down, and have the time to do so,
                #   we should probably do that.  (Or, if we happen to find the solution
                #   while doing something else, make this upsert into an insert and
                #   close the issue.)
                # cached_report.insert()
                cached_report.upsert()
                # We may have made a report earlier when calling Pipeline.setup_datastore,
                #   so we need to remove that one from the database now that we've replaced it.
                if ds.report is not None:
                    with Psycopg2Connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute( "DELETE FROM reports WHERE _id=%(id)s",
                                        { 'id': ds.report.id } )
                        conn.commit()
                ds.report = cached_report
                report_was_loaded_from_cache = True
            else:
                if ds.report is not None:
                    ds.report.start_time = datetime.datetime.now( tz=datetime.UTC )
                else:
                    raise RuntimeError( "ds.report is None and I'm surprised" )

        # Remove all steps past subtraction if there's no referencing provenance
        if ( 'subtraction' in stepstodo ) and ( 'referencing' not in ds.prov_tree ):
            SCLogger.debug( "datastore_factory: No reference set, or no reference image provenances, found; "
                            "removing all steps from subtraction on from steps to  perform." )
            subdex = stepstodo.index( 'subtraction' )
            stepstodo = stepstodo[:subdex]


        ############ preprocessing to create image ############

        if 'preprocessing' in stepstodo:

            image_cache_path = None
            if ds.image is None and use_cache:  # check if preprocessed image is in cache
                image_cache_path = cache_base_path.parent / f'{cache_base_path.name}.json'
                SCLogger.debug( f'make_datastore searching cache for {image_cache_path}' )
                if ( cache_dir / image_cache_path).is_file():
                    SCLogger.debug('make_datastore loading image from cache')
                    img = copy_from_cache(Image, cache_dir, image_cache_path, symlink=True)
                    # assign the correct exposure to the object loaded from cache
                    if ds.exposure_id is not None:
                        img.exposure_id = ds.exposure_id
                    if ds.exposure is not None:
                        img.exposure_id = ds.exposure.id
                    ds.image = img

                    # Copy the original image from the cache if requested
                    if save_original_image:
                        pth = pathlib.Path( ds.image.get_fullpath()[0] )
                        ds.path_to_original_image = pth.parent / f'{pth.name}.image.fits.original'
                        pth = cache_dir / ds.image.filepath
                        image_cache_path_original = pth.parent / f'{pth.name}.image.fits.original'
                        shutil.copy2( image_cache_path_original, ds.path_to_original_image )

                    ds.image.provenance_id = ds.prov_tree['preprocessing'].id

                    # make sure this is saved to the archive as well
                    # this is inefficient, because it's going to overwrite the
                    # (presumably identical) image files we just copied from the cache!
                    ds.image.save(verify_md5=False)
                    image_was_loaded_from_cache = True

            if ds.image is None:  # make the preprocessed image
                SCLogger.debug('make_datastore making preprocessed image')
                ds = p.preprocessor.run(ds)
                ds.update_report( 'preprocessing' )
                if bad_pixel_map is not None:
                    # AUGH.  Don't want to be doing processing here
                    #   that's not normally done in the pipeline.  The
                    #   DECam instrument has the concept of a standard
                    #   instrument bad pixel flag; could we add that to
                    #   PTF, and remove the explicitly bad pixel map
                    #   setting from the ptf fixtures?  Will take some
                    #   work.  (See comment on ptf_bad_pixel_map fixture
                    #   in tests/fixtures.ptf.py) → Issue #386
                    SCLogger.warning( "datastore_factory adding to bad_pixel_map; this should "
                                      "be refactored so that it's part of preprocessing, not "
                                      "something bespoke in datastore_factory" )
                    ds.image.flags |= bad_pixel_map
                    if ds.image.weight is not None:
                        ds.image.weight[ds.image.flags.astype(bool)] = 0.0

                    # I don't know that we really want to be dilating
                    #   these masked pixels, but it's been done for a
                    #   bunch of ptf fixtures, and changing it now would
                    #   change test results, leading to pain of updating
                    #   it all.  Leave this here for now, but think
                    #   about changing it (ideally as part of the AUGH
                    #   change above, which will require some changes to
                    #   preprocessing and probably to the PTF
                    #   instrument), and then probably a bunch of test
                    #   result numbers are going to change too.  Fix
                    #   this when we fix Issue #386.  flag saturated
                    #   pixels, too (TODO: is there a better way to get
                    #   the saturation limit? )
                    mask = make_saturated_flag(ds.image.data, ds.image.instrument_object.saturation_limit,
                                               iterations=2, no_really_i_know_i_want_to_run_this=True)
                    ds.image.flags |= (mask * 2 ** BitFlagConverter.convert('saturated')).astype(np.uint16)
                    # This is what I'd rather be doing, and is what's done in preprocessing already
                    #   (so it doesn't have to be done here).
                    # wsat = image.data >= ds.image.instrument_object.average_saturation_limit( ds.image )
                    # ds.image.flags[wsat] |= string_to_bitflag( "saturated", flag_image_bits_inverse )

                ds.image.save()

                if not env_as_bool( "LIMIT_CACHE_USAGE" ) and cache_dir.is_dir():
                    # Don't copy the image to the cache -- the image database record
                    #  is going to get further modified in subsequent setps.  We don't
                    #  want an incomplete cache if those steps aren't done.
                    # Image copying to cache happens after the zp step.
                    # However, verify that the thing we will copy to the cache matches
                    #   what was expected.
                    _ = ds.image.id
                    output_path = copy_to_cache( ds.image, cache_dir,
                                                 dont_actually_copy_just_return_json_filepath=True )
                    if ( ( image_cache_path is not None ) and
                         ( output_path.resolve() != ( cache_dir / image_cache_path ).resolve() )
                        ):
                        warnings.warn(f'cache path {image_cache_path} does not match output path {output_path}')
                    else:
                        # Update some of the cache paths.  These may
                        #   have been None before, because they weren't
                        #   passed, and we didn't have enough
                        #   information to figure them out.  Now that
                        #   we do, unNoneify these variables.
                        cache_base_path = output_path.parent / output_path.name[:-5]    # Strip the .json
                        report_cache_path = cache_base_path.parent / f'{cache_base_path.name}.report.json'
                        ds.cache_base_path = cache_base_path
                        cache_base_path = cache_base_path.relative_to( cache_dir )
                        SCLogger.debug( f'Later, will save image to cache at: {output_path}' )
                        # If LIMIT_CACHE_USAGE is not set and cache_dir
                        #   exists (requirements to be in this block of
                        #   code), then we want to be using the cache.
                        #   It's possible that we set use_cache to false
                        #   before because cache_base_name wasn't set
                        #   yet, but now that we've set it, we should
                        #   flip use_cache back to True.  (All of this
                        #   cache stuff is very spaghettiesque.  Perhaps
                        #   inevitable given that the cache is a hack
                        #   put in to make tests run faster, and it's
                        #   a kludge.)
                        use_cache = True

                # In test_astro_cal, there's a routine that needs the original
                # image before being processed through the rest of what this
                # factory function does, so save it if requested
                if save_original_image:
                    ds.path_to_original_image = str(ds.image.get_fullpath()[0]) + '.image.fits.original'
                    shutil.copy2( ds.image.get_fullpath()[0], ds.path_to_original_image )
                    if use_cache:
                        ( cache_dir / ds.image.filepath ).parent.mkdir( exist_ok=True, parents=True )
                        pth = pathlib.Path( ds.image.filepath )
                        shutil.copy2( ds.image.get_fullpath()[0],
                                      cache_dir / pth.parent / f'{pth.name}.image.fits.original' )

        ############# extraction to create sources / PSF  #############

        if 'extraction' in stepstodo:

            found_sources_in_cache = False
            if use_cache:
                # try to get the source list from cache
                filename_barf = ds.prov_tree['extraction'].id[:6]
                sources_cache_path = ( cache_dir / cache_base_path.parent /
                                       f'{cache_base_path.name}.sources_{filename_barf}.fits.json' )
                SCLogger.debug( f'make_datastore searching cache for source list {sources_cache_path}' )
                if sources_cache_path.is_file():
                    SCLogger.debug('make_datastore loading source list from cache')
                    ds.sources = copy_from_cache(SourceList, cache_dir, sources_cache_path, symlink=True)
                    ds.sources.provenance_id = ds.prov_tree['extraction'].id
                    ds.sources.image_id = ds.image.id
                    # make sure this is saved to the archive as well
                    ds.sources.save(verify_md5=False)
                    found_sources_in_cache = True

                # try to get the PSF from cache
                psf_cache_path = ( cache_dir / cache_base_path.parent /
                                   f'{cache_base_path.name}.psf_{filename_barf}.json' )
                SCLogger.debug( f'make_datastore searching cache for psf {psf_cache_path}' )
                if psf_cache_path.is_file():
                    SCLogger.debug('make_datastore loading PSF from cache')
                    ds.psf = copy_from_cache(PSF, cache_dir, psf_cache_path, symlink=True)
                    ds.psf.sources_id = ds.sources.id
                    # make sure this is saved to the archive as well
                    ds.psf.load()
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
                ds.psf.save( image=ds.image, sources=ds.sources, overwrite=True )
                ds.update_report( 'extraction' )

                if use_cache:
                    _ = ds.sources.id
                    output_path = copy_to_cache(ds.sources, cache_dir)
                    if output_path.resolve() != sources_cache_path.resolve():
                        warnings.warn(f'cache path {sources_cache_path} does not match output path {output_path}')

                    _ = ds.psf.id
                    output_path = copy_to_cache(ds.psf, cache_dir)
                    if output_path.resolve() != psf_cache_path.resolve():
                        warnings.warn(f'cache path {psf_cache_path} does not match output path {output_path}')

        ########## Background ##########

        if 'bg' in stepstodo:
            filename_barf = ds.prov_tree['backgrounding'].id[:6]
            bg_cache_path = ( cache_dir / cache_base_path.parent /
                              f'{cache_base_path.name}.bg_{filename_barf}.h5.json' )
            if use_cache and found_sources_in_cache:
                # try to get the background from cache
                SCLogger.debug( f'make_datastore searching cache for background {bg_cache_path}' )
                if bg_cache_path.is_file():
                    SCLogger.debug('make_datastore loading background from cache. ')
                    ds.bg = copy_from_cache( Background, cache_dir, bg_cache_path,
                                             add_to_dict={ 'image_shape': ds.image.data.shape },
                                             symlink=True )
                    ds.bg.sources_id = ds.sources.id
                    # make sure this is saved to the archive as well
                    ds.bg.save( image=ds.image, verify_md5=False, overwrite=True )


            if ds.bg is None:
                SCLogger.debug('make_datastore running background estimation')
                ds = p.backgrounder.run(ds)
                ds.bg.save( image=ds.image, overwrite=True )
                ds.update_report( 'backgrounding' )
                if use_cache:
                    _ = ds.bg.id
                    output_path = copy_to_cache(ds.bg, cache_dir)
                    if output_path.resolve() != bg_cache_path.resolve():
                        warnings.warn(f'cache path {bg_cache_path} does not match output path {output_path}')

        ########## Astrometric calibration ##########

        if 'wcs' in stepstodo:
            filename_barf = ds.prov_tree['wcs'].id[:6]
            wcs_cache_path = ( cache_dir / cache_base_path.parent /
                               f'{cache_base_path.name}.wcs_{filename_barf}.txt.json' )
            if use_cache and found_sources_in_cache:
                # try to get the WCS from cache
                SCLogger.debug( f'make_datastore searching cache for wcs {wcs_cache_path}' )
                if wcs_cache_path.is_file():
                    SCLogger.debug('make_datastore loading WCS from cache. ')
                    ds.wcs = copy_from_cache(WorldCoordinates, cache_dir, wcs_cache_path, symlink=True)
                    ds.wcs.sources_id = ds.sources.id
                    # make sure this is saved to the archive as well
                    ds.wcs.save( image=ds.image, verify_md5=False, overwrite=True )

            if ds.wcs is None:
                SCLogger.debug('make_datastore running astrometric calibration')
                ds = p.astrometor.run(ds)
                ds.wcs.save( image=ds.image, overwrite=True )
                ds.update_report( 'astrocal' )
                if use_cache:
                    _ = ds.wcs.id
                    output_path = copy_to_cache(ds.wcs, cache_dir)
                    if output_path.resolve() != wcs_cache_path.resolve():
                        warnings.warn(f'cache path {wcs_cache_path} does not match output path {output_path}')

        ########## Photometric calibration ##########

        if 'zp' in stepstodo:
            zp_cache_path = ( cache_dir / cache_base_path.parent /
                              f'{cache_base_path.name}.zp.json' )
            if use_cache and found_sources_in_cache:
                # try to get the ZP from cache
                SCLogger.debug( f'make_datastore searching cache for zero point {zp_cache_path}' )
                if zp_cache_path.is_file():
                    SCLogger.debug('make_datastore loading zero point from cache. ')
                    ds.zp = copy_from_cache(ZeroPoint, cache_dir, zp_cache_path, symlink=True)
                    ds.zp.sources_ids = ds.sources.id

            if ds.zp is None:
                SCLogger.debug('make_datastore running photometric calibration')
                ds = p.photometor.run(ds)
                ds.update_report( 'photocal' )
                if use_cache:
                    _ = ds.zp.id
                    output_path = copy_to_cache(ds.zp, cache_dir, zp_cache_path)
                    if output_path.resolve() != zp_cache_path.resolve():
                        warnings.warn(f'cache path {zp_cache_path} does not match output path {output_path}')

        ########### Done with image and image data products; save and commit #############

        SCLogger.debug( "make_datastore running ds.save_and_commit on image (before subtraction)" )
        ds.save_and_commit()

        # *Now* copy the image to cache, including the estimates for lim_mag, fwhm, etc.
        if ( not env_as_bool("LIMIT_CACHE_USAGE") ) and ( not image_was_loaded_from_cache ):
            _ = ds.image.id
            output_path = copy_to_cache(ds.image, cache_dir)

        ############ Now do subtraction / detection / measurement / etc. ##############

        ########## subtraction ##########

        if 'subtraction' in stepstodo:

            ########## look for a reference ##########

            with SmartSession() as sess:
                refset = sess.scalars(sa.select(RefSet).where(RefSet.name == refset_name)).first()

                if refset is None:
                    SCLogger.debug( f"No refset found with name {refset_name}, returning." )
                    return ds

                ref = ds.get_reference()
                if ( ref is None ) and ( 'subtraction' in stepstodo ):
                    SCLogger.debug( "make_datastore : could not find a reference, returning." )
                    return ds


            ########### find or run the subtraction ##########

            if use_cache:  # try to find the subtraction image in the cache
                SCLogger.debug( "make_datstore looking for subtraction image in cache..." )

                sub_im = Image.from_new_and_ref( ds.zp, ds.reference )
                sub_im.provenance_id = ds.prov_tree['subtraction'].id
                cache_sub_name = pathlib.Path( sub_im.invent_filepath() )

                sub_cache_path = cache_dir / cache_sub_name.parent / f'{cache_sub_name.name}.json'
                zogy_score_cache_path = cache_dir / cache_sub_name.parent / f'{cache_sub_name.name}.zogy_score.npy'
                zogy_alpha_cache_path = cache_dir / cache_sub_name.parent / f'{cache_sub_name.name}.zogy_alpha.npy'
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
                f = f[:-9]                         # strip the u-tag
                f = f[:-6] + prov_aligned_ref.id[:6]  # replace the provenance ID
                filename_aligned_ref = pathlib.Path( f )
                filename_aligned_ref_sources = pathlib.Path( f'{f}_sources.fits' )
                filename_aligned_ref_psf = pathlib.Path( f'{f}.psf' )
                filename_aligned_ref_bg = pathlib.Path( f'{f}_bg' )

                aligned_ref_cache_path = ( cache_dir / filename_aligned_ref.parent /
                                           f'{filename_aligned_ref.name}.json' )
                aligned_ref_sources_cache_path = ( cache_dir / filename_aligned_ref_sources.parent /
                                                   f'{filename_aligned_ref_sources.name}.json' )
                aligned_ref_psf_cache_path = ( cache_dir / filename_aligned_ref_psf.parent /
                                               f'{filename_aligned_ref_psf.name}.json' )
                aligned_ref_bg_cache_path = ( cache_dir / filename_aligned_ref_bg.parent /
                                              f'{filename_aligned_ref_bg.name}.h5.json' )
                aligned_ref_zp_cache_path = ( cache_dir / filename_aligned_ref.parent /
                                              f'{filename_aligned_ref.name}.zp.json' )

                SCLogger.debug( f'make_datastore searching for subtraction cache including {sub_cache_path}' )
                files_needed = [ sub_cache_path, aligned_ref_cache_path, aligned_ref_sources_cache_path,
                                 aligned_ref_psf_cache_path, aligned_ref_bg_cache_path, aligned_ref_zp_cache_path ]
                if p.subtractor.pars.method == 'zogy':
                    files_needed.extend( [ zogy_score_cache_path, zogy_alpha_cache_path ] )

                if all( f.is_file() for f in files_needed ):
                    SCLogger.debug('make_datastore loading subtraction image from cache: {sub_cache_path}" ')
                    tmpsubim =  copy_from_cache(Image, cache_dir, sub_cache_path, symlink=True)
                    tmpsubim.provenance_id = ds.prov_tree['subtraction'].id
                    tmpsubim._ref_id = ds.reference.id
                    tmpsubim._new_zp_id = ds.zp.id
                    tmpsubim.save(verify_md5=False)  # make sure it is also saved to archive
                    ds.sub_image = tmpsubim
                    if p.subtractor.pars.method == 'zogy':
                        ds.zogy_score = np.load( zogy_score_cache_path )
                        ds.zogy_alpha = np.load( zogy_alpha_cache_path )

                    SCLogger.debug('loading aligned reference image from cache. ')
                    image_aligned_ref = copy_from_cache( Image, cache_dir, aligned_ref_cache_path, symlink=True )
                    image_aligned_ref.provenance_id = prov_aligned_ref.id
                    image_aligned_ref.info['original_image_id'] = ds.ref_image.id
                    image_aligned_ref.info['original_image_filepath'] = ds.ref_image.filepath
                    image_aligned_ref.info['alignment_parameters'] = ds.prov_tree['subtraction'].parameters['alignment']
                    # TODO FIGURE OUT WHAT'S GOING ON HERE
                    # Not sure why the md5sum_components was [], but it was
                    image_aligned_ref.md5sum_components = [ None, None, None ]
                    image_aligned_ref.save(verify_md5=False, no_archive=True)
                    # TODO: should we also load the aligned images' sources and PSF?
                    #  (We've added bg and zp because specific tests need them.)
                    ds.aligned_ref_image = image_aligned_ref

                    ds.aligned_ref_sources = copy_from_cache( SourceList, cache_dir, aligned_ref_sources_cache_path,
                                                              symlink=True )
                    ds.aligned_ref_psf = copy_from_cache( PSF, cache_dir, aligned_ref_psf_cache_path,
                                                          symlink=True )
                    ds.aligned_ref_bg = copy_from_cache( Background, cache_dir, aligned_ref_bg_cache_path,
                                                         symlink=True )
                    ds.aligned_ref_zp = copy_from_cache( ZeroPoint, cache_dir, aligned_ref_zp_cache_path, symlink=True )
                    ds.aligned_new_image = ds.image
                    ds.aligned_new_sources = ds.sources
                    ds.aligned_new_bg = ds.bg
                    ds.aligned_new_zp = ds.zp

                else:
                    strio = io.StringIO()
                    strio.write( "make_datastore didn't find subtraction image in cache\n" )
                    for f in files_needed:
                        strio.write( f"   ... {f} : {'found' if f.is_file() else 'NOT FOUND'}\n" )
                    SCLogger.debug( strio.getvalue() )

            if ds.sub_image is None:  # no hit in the cache
                SCLogger.debug( "make_datastore running subtractor to create subtraction image" )
                ds = p.subtractor.run( ds )
                ds.sub_image.save(verify_md5=False)  # make sure it is also saved to archive
                ds.update_report( 'subtraction' )
                if use_cache:
                    _ = ds.sub_image.id
                    output_path = copy_to_cache(ds.sub_image, cache_dir)
                    if output_path.resolve() != sub_cache_path.resolve():
                        raise ValueError( f'cache path {sub_cache_path} does not match output path {output_path}' )
                        # warnings.warn(f'cache path {sub_cache_path} does not match output path {output_path}')
                    if p.subtractor.pars.method == 'zogy':
                        np.save( zogy_score_cache_path, ds.zogy_score, allow_pickle=False )
                        np.save( zogy_alpha_cache_path, ds.zogy_alpha, allow_pickle=False )

                    # Normally the aligned ref (and associated products) doesn't get saved
                    #  to disk.  But, we need it in the cache, since it's used in the
                    #  pipeline.
                    # (This might actually require some thought.  Right now, if the
                    #  pipeline has run through subtraction, you *can't* pick it up at
                    #  cutting becasue cutting needs the aligned refs!  So perhaps
                    #  we should be saving it.)
                    SCLogger.debug( "make_datastore saving aligned ref image to cache" )
                    ds.aligned_ref_image.save( no_archive=True )
                    _ = ds.aligned_ref_image.id
                    outpath = copy_to_cache( ds.aligned_ref_image, cache_dir )
                    if outpath.resolve() != aligned_ref_cache_path.resolve():
                        warnings.warn( f"Aligned ref cache path {outpath} "
                                       f"doesn't match expected {aligned_ref_cache_path}" )
                    ds.aligned_ref_sources.filepath = str( filename_aligned_ref_sources )
                    ds.aligned_ref_sources.save( no_archive=True )
                    _ = ds.aligned_ref_sources.id
                    outpath = copy_to_cache( ds.aligned_ref_sources, cache_dir,
                                             filepath=aligned_ref_sources_cache_path )
                    if outpath.resolve() != aligned_ref_sources_cache_path.resolve():
                        warnings.warn( f"Aligned ref sources cache path {outpath} "
                                       f"doesn't match expected {aligned_ref_sources_cache_path}" )
                    ds.aligned_ref_psf.save( no_archive=True, filename=str( filename_aligned_ref_psf ) )
                    _ = ds.aligned_ref_psf.id
                    outpath = copy_to_cache( ds.aligned_ref_psf, cache_dir,
                                             filepath=aligned_ref_psf_cache_path )
                    if outpath.resolve() != aligned_ref_psf_cache_path.resolve():
                        warnings.warn( f"Aligned ref psf cache path {outpath} "
                                       f"doesn't match expected {aligned_ref_psf_cache_path}" )
                    _ = ds.aligned_ref_zp.id
                    outpath = copy_to_cache( ds.aligned_ref_zp, cache_dir, filepath=aligned_ref_zp_cache_path )
                    if outpath.resolve() != aligned_ref_zp_cache_path.resolve():
                        warnings.warn( f"Aligned ref zp cache path {outpath} "
                                       f"doesn't match expected {aligned_ref_zp_cache_path}" )
                    ds.aligned_ref_bg.save( no_archive=True, filename=f'{ds.aligned_ref_image.filepath}_bg.h5' )
                    _ = ds.aligned_ref_bg.id
                    outpath = copy_to_cache( ds.aligned_ref_bg, cache_dir )
                    if outpath.resolve() != aligned_ref_bg_cache_path.resolve():
                        warnings.warn( f"Aligned ref bg cache path {outpath} "
                                       f"doesn't match expected {aligned_ref_bg_cache_path}" )

        ############ detecting to create a source list ############

        if 'detection' in stepstodo:
            detection_cache_path = ( cache_dir / cache_sub_name.parent /
                                     f'{cache_sub_name.name}.sources_{ds.prov_tree["detection"].id[:6]}.npy.json' )
            if use_cache:
                SCLogger.debug( f'make_datastore searching cache for detections {detection_cache_path}' )
                if detection_cache_path.is_file():
                    SCLogger.debug( "make_datastore loading detections from cache." )
                    ds.detections = copy_from_cache(SourceList, cache_dir, detection_cache_path, symlink=True)
                    ds.detections.provenance_id = ds.prov_tree['detection'].id
                    ds.detections.image_id = ds.sub_image.id
                    ds.detections.save(verify_md5=False)

            if ds.detections is None:
                SCLogger.debug( "make_datastore running detector to find detections" )
                ds = p.detector.run(ds)
                ds.detections.save( image=ds.sub_image, verify_md5=False )
                ds.update_report( 'detection' )
                if use_cache:
                    _ = ds.detections.id
                    outpath = copy_to_cache( ds.detections, cache_dir, detection_cache_path )
                    if outpath.resolve() != detection_cache_path.resolve():
                        warnings.warn( f"Detection cache path {outpath} "
                                       f"doesn't match expected {detection_cache_path}" )

        ############ cutting to create cutouts ############

        if 'cutting' in stepstodo:
            cutouts_cache_path = ( cache_dir / cache_sub_name.parent /
                                   f'{cache_sub_name.name}.cutouts_{ds.prov_tree["cutting"].id[:6]}.h5.json' )
            if use_cache:
                SCLogger.debug( f'make_datastore searching cache for cutouts {cutouts_cache_path}' )
                if cutouts_cache_path.is_file():
                    SCLogger.debug( 'make_datastore loading cutouts from cache.' )
                    ds.cutouts = copy_from_cache(Cutouts, cache_dir, cutouts_cache_path, symlink=True)
                    ds.cutouts.provenance_id = ds.prov_tree['cutting'].id
                    ds.cutouts.sources_id = ds.detections.id
                    ds.cutouts.load_all_co_data( sources=ds.detections )
                    ds.cutouts.save( image=ds.sub_image, sources=ds.detections )  # make sure to save to archive as well

            if ds.cutouts is None:
                SCLogger.debug( "make_datastore running cutter to create cutouts" )
                ds = p.cutter.run(ds)
                ds.cutouts.save( image=ds.sub_image, sources=ds.detections )
                ds.update_report( 'cutting' )
                if use_cache:
                    _ = ds.cutouts.id
                    outpath = copy_to_cache(ds.cutouts, cache_dir)
                    if outpath.resolve() != cutouts_cache_path.resolve():
                        warnings.warn( f"Cutouts cache path {outpath} "
                                       f"doesn't match expected {cutouts_cache_path}" )

        ############ measuring to create measurements ############

        if 'measuring' in stepstodo:
            measurement_set_cache_path = ( cache_dir / cache_sub_name.parent /
                                           ( f'{cache_sub_name.name}.measurement_set_'
                                             f'{ds.prov_tree["measuring"].id[:6]}.json' ) )
            all_measurements_cache_path = ( cache_dir / cache_sub_name.parent /
                                            ( f'{cache_sub_name.name}.all_measurements_'
                                              f'{ds.prov_tree["measuring"].id[:6]}.json' ) )
            measurements_cache_path = ( cache_dir / cache_sub_name.parent /
                                        f'{cache_sub_name.name}.measurements_{ds.prov_tree["measuring"].id[:6]}.json' )

            SCLogger.debug( 'make_datastore searching cache for measurements and associated' )
            if ( use_cache and
                 measurements_cache_path.is_file() and
                 all_measurements_cache_path.is_file() and
                 measurement_set_cache_path.is_file()
                ):
                SCLogger.debug( 'make_datastore loading measurements from cache.' )
                ds.measurement_set = copy_from_cache( MeasurementSet, cache_dir, measurement_set_cache_path )
                ds.measurement_set.cutouts_id = ds.cutouts.id
                ds.measurement_set.provenance_id = ds.prov_tree['measuring'].id
                ds.measurement_set.measurements = copy_list_from_cache(Measurements, cache_dir, measurements_cache_path)
                [ setattr(m, 'measurementset_id', ds.measurement_set.id) for m in ds.measurement_set.measurements ]

                # Note that the actual measurement objects in the two lists
                # won't be the same objects (they will be equivalent
                # objects), whereas when they are created in the first place
                # I think they're the same objects.  As long as we're
                # treating measurements as read-only, except for a bit of
                # memory usage this shouldn't matter.
                ds.all_measurements = copy_list_from_cache(Measurements, cache_dir, all_measurements_cache_path)
                [ setattr(m, 'provenance_id', ds.prov_tree['measuring'].id) for m in ds.all_measurements ]

                # Because the Object association wasn't run, we have to do that manually
                # (Warning: this means object names next not be the same from one run of tests
                # to the next, if things are loaded from the cache in a different order from
                # that in which they were created.  Currently, no tests using cached datastores
                # look at actual object names.)
                year = int( np.floor( astropy.time.Time( ds.image.mjd, format='mjd' ).jyear ) )
                Object.associate_measurements( ds.measurements, p.measurer.pars.association_radius, year=year,
                                               is_testing=ds.prov_tree['measuring'].is_testing )

            if ds.measurements is None:
                SCLogger.debug( "make_datastore running measurer to create measurements" )
                ds = p.measurer.run(ds)
                ds.update_report( 'measuring' )
                # assign each measurements an ID to be saved in cache - needed for scores cache
                [m.id for m in ds.measurements]
                if use_cache:
                    copy_to_cache( ds.measurement_set, cache_dir, measurement_set_cache_path )
                    copy_list_to_cache(ds.all_measurements, cache_dir, all_measurements_cache_path)
                    copy_list_to_cache(ds.measurements, cache_dir, measurements_cache_path)

        if 'scoring' in stepstodo:
            deepscore_set_cache_path = ( cache_dir / cache_sub_name.parent /
                                         f'{cache_sub_name.name}.deepscore_set_{ds.prov_tree["scoring"].id[:6]}.json' )
            deepscores_cache_path = ( cache_dir / cache_sub_name.parent /
                                      f'{cache_sub_name.name}.deepscores_{ds.prov_tree["scoring"].id[:6]}.json' )

            SCLogger.debug( f'make_datastore searching cache for deepscores {deepscores_cache_path} and '
                            f'deepscore set {deepscore_set_cache_path}' )
            if use_cache and deepscores_cache_path.is_file() and deepscore_set_cache_path.is_file():
                deepscore_set = copy_from_cache( DeepScoreSet, cache_dir, deepscore_set_cache_path )
                deepscore_set.measurementset_id = ds.measurement_set.id
                deepscore_set.provenance_id = ds.prov_tree['scoring'].id
                scores = copy_list_from_cache(DeepScore, cache_dir, deepscores_cache_path)
                deepscore_set.deepscores = scores
                ds.deepscore_set = deepscore_set
            else:
                SCLogger.debug( "make_datastore running scorer to create scores" )
                ds = p.scorer.run(ds)
                ds.update_report( 'scoring' )
                # assign each score an ID to be saved in cache
                [sc.id for sc in ds.deepscores]
                if use_cache:
                    copy_to_cache( ds.deepscore_set, cache_dir, deepscore_set_cache_path )
                    copy_list_to_cache( ds.deepscores, cache_dir, deepscores_cache_path )

        # If necessary, save the report to the cache
        if ( ( p._generate_report) and
             isinstance( exporim, Exposure ) and
             use_cache and
             ( not report_was_loaded_from_cache )
            ):
            ds.finalize_report()
            if ds.report is not None:
                _ = ds.report.id
                output_path = copy_to_cache( ds.report, cache_dir, report_cache_path )
                if output_path.resolve() != report_cache_path.resolve():
                    warnings.warn( f'report cache path {report_cache_path} does not match output path {output_path}' )
            else:
                SCLogger.warning( "Report not available!" )

        SCLogger.debug( "make_datastore running final ds.save_and_commit" )
        ds.save_and_commit()

        return ds

    return make_datastore
