import datetime
import time

import numpy as np
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from pipeline.parameters import Parameters
from pipeline.coaddition import CoaddPipeline
from pipeline.top_level import Pipeline

from models.base import SmartSession
from models.provenance import Provenance
from models.reference import Reference
from models.exposure import Exposure
from models.image import Image
from models.refset import RefSet

from util.config import Config
from util.logger import SCLogger
from util.util import parse_session, listify
from util.radec import parse_sexigesimal_degrees


class ParsRefMaker(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = self.add_par(
            'name',
            'default',
            str,
            'Name of the reference set. ',
            critical=False,  # the name of the refset is not in the Reference provenance!
            # this means multiple refsets can refer to the same Reference provenance
        )

        self.description = self.add_par(
            'description',
            '',
            str,
            'Description of the reference set. ',
            critical=False,
        )

        self.allow_append = self.add_par(
            'allow_append',
            True,
            bool,
            'If True, will append new provenances to an existing reference set with the same name. '
            'If False, will raise an error if a reference set with the same name '
            'and a different provenance already exists',
            critical=False,  # can decide to turn this option on or off as an administrative decision
        )

        self.start_time = self.add_par(
            'start_time',
            None,
            (None, str, float, datetime.datetime),
            'Only use images taken after this time (inclusive). '
            'Time format can be MJD float, ISOT string, or datetime object. '
            'If None, will not limit the start time. ',
            critical=True,
        )

        self.end_time = self.add_par(
            'end_time',
            None,
            (None, str, float, datetime.datetime),
            'Only use images taken before this time (inclusive). '
            'Time format can be MJD float, ISOT string, or datetime object. '
            'If None, will not limit the end time. ',
            critical=True,
        )

        self.instruments = self.add_par(
            'instruments',
            None,
            (None, list),
            'Only use images from these instruments. If None, will use all instruments. '
            'If given as a list, will use any of the instruments in the list. '
            'In both these cases, cross-instrument references will be made. '
            'To make sure single-instrument references are made, make a different refset '
            'with a single item on this list, one for each instrument. '
            'This does not have a default value, but you MUST supply a list with at least one instrument '
            'in order to get a reference provenance and create a reference set. ',
            critical=True,
        )

        self.filters = self.add_par(
            'filters',
            None,
            (None, list),
            'Only use images with these filters. If None, will not limit the filters. '
            'If given as a list, will use any of the filters in the list. '
            'For multiple instruments, can match any filter to any instrument. ',
            critical=True,
        )

        self.projects = self.add_par(
            'projects',
            None,
            (None, list),
            'Only use images from these projects. If None, will not limit the projects. '
            'If given as a list, will use any of the projects in the list. ',
            critical=True,
        )

        self.__image_query_pars__ = ['airmass', 'background', 'seeing', 'lim_mag', 'exp_time']

        for name in self.__image_query_pars__:
            for min_max in ['min', 'max']:
                self.add_limit_parameter(name, min_max)

        self.__docstrings__['min_lim_mag'] = ('Only use images with lim_mag larger (fainter) than this. '
                                             'If None, will not limit the minimal lim_mag. ')
        self.__docstrings__['max_lim_mag'] = ('Only use images with lim_mag smaller (brighter) than this. '
                                                'If None, will not limit the maximal lim_mag. ')

        self.min_number = self.add_par(
            'min_number',
            1,
            int,
            'Construct a reference only if there are at least this many images that pass all other criteria. ',
            critical=True,
        )

        self.max_number = self.add_par(
            'max_number',
            None,
            (None, int),
            'If there are more than this many images, pick the ones with the highest "quality". ',
            critical=True,
        )

        self.seeing_quality_factor = self.add_par(
            'seeing_quality_factor',
            3.0,
            float,
            'linear combination coefficient for adding limiting magnitude and seeing FWHM '
            'when calculating the "image quality" used to rank images. ',
            critical=True,
        )

        self.save_new_refs = self.add_par(
            'save_new_refs',
            True,
            bool,
            'If True, will save the coadd image and commit it and the newly created reference to the database. '
            'If False, will only return it. ',
            critical=False,
        )

        self._enforce_no_new_attrs = True  # lock against new parameters

        self.override(kwargs)

    def add_limit_parameter(self, name, min_max='min'):
        """Add a parameter in a systematic way. """
        if min_max not in ['min', 'max']:
            raise ValueError('min_max must be either "min" or "max"')
        compare = 'larger' if min_max == 'min' else 'smaller'
        setattr(
            self,
            f'{min_max}_{name}',
            self.add_par(
                f'{min_max}_{name}',
                None,
                (None, float),
                f'Only use images with {name} {compare} than this value. '
                f'If None, will not limit the {min_max}imal {name}.',
                critical=True,
            )
        )

    def get_process_name(self):
        return 'referencing'


class RefMaker:
    def __init__(self, **kwargs):
        """
        Initialize a reference maker object.

        The possible keywords that can be given are: maker, pipeline, coaddition. Each should be a dictionary.

        The object will load the config file and use the following hierarchy to set the parameters:
        - first loads up the regular pipeline parameters, namely those for preprocessing and extraction.
        - override those with the parameters given by the "referencing" dictionary in the config file.
        - override those with kwargs['pipeline'] that can have "preprocessing" or "extraction" keys.
        - parameters for the coaddition step, and the extraction done on the coadd image are taken from "coaddition"
        - those are overriden by the "referencing.coaddition" dictionary in the config file
        - those are overriden by the kwargs['coaddition'] dictionary, if it exists.
        - the parameters to the reference maker its (e.g., how to choose images) are given from the
          config['referencing.maker'] dictionary and are overriden by the kwargs['maker'] dictionary.

        The maker contains a pipeline object, that doesn't do any work, but is instantiated so it can build up the
        provenances of the images and their products, that go into the coaddition.
        Those images need to already exist in the database before calling run().
        Pass kwargs into the pipeline object using kwargs['pipeline'].
        TODO: what about multiple instruments that go into the coaddition? we'd need multiple pipeline objects
         in order to have difference parameter sets for preprocessing/extraction for each instrument.
        The maker also contains a coadd_pipeline object, that has two roles: one is to build the provenances of the
        coadd image and the products of that image (extraction on the coadd) and the second is to actually
        do the work of coadding the chosen images.
        Pass kwargs into this object using kwargs['coaddition'].
        The choice of which images are loaded into the reference coadd is determined by the parameters object of the
        maker itself (and the provenances of the images and their products).
        To set these parameters, use the "referencing.maker" dictionary in the config, or pass them in kwargs['maker'].
        """
        # first break off some pieces of the kwargs dict
        maker_overrides = kwargs.pop('maker', {})  # to set the parameters of the reference maker itself
        pipe_overrides = kwargs.pop('pipeline', {})  # to allow overriding the regular image pipeline
        coadd_overrides = kwargs.pop('coaddition', {})  # to allow overriding the coaddition pipeline

        if len(kwargs) > 0:
            raise ValueError(f'Unknown parameters given to RefMaker: {kwargs.keys()}')

        # now read the config file
        config = Config.get()

        # initialize an object to get the provenances of the regular images and their products
        pipe_dict = config.value('referencing.pipeline', {})  # this is the reference pipeline override
        pipe_dict.update(pipe_overrides)
        self.pipeline = Pipeline(**pipe_dict)  # internally loads regular pipeline config, overrides with pipe_dict

        coadd_dict = config.value('referencing.coaddition', {})  # allow overrides from config's referencing.coaddition
        coadd_dict.update(coadd_overrides)  # allow overrides from kwargs['coaddition']
        self.coadd_pipeline = CoaddPipeline(**coadd_dict)  # coaddition parameters, overrides with coadd_dict

        maker_dict = config.value('referencing.maker')
        maker_dict.update(maker_overrides)  # user can provide override arguments in kwargs
        self.pars = ParsRefMaker(**maker_dict)  # initialize without the pipeline/coaddition parameters

        # first, make sure we can assemble the provenances up to extraction:
        self.im_provs = None  # the provenances used to make images going into the reference (these are coadds!)
        self.ex_provs = None  # the provenances used to make other products like SourceLists, that go into the reference
        self.coadd_im_prov = None  # the provenance used to make the coadd image
        self.coadd_ex_prov = None  # the provenance used to make the products of the coadd image
        self.ref_upstream_hash = None  # a hash identifying all upstreams of the reference provenance
        self.ref_prov = None  # the provenance of the reference itself
        self.refset = None  # the RefSet object that was found / created

        # these attributes tell us the place in the sky where we want to look for objects (given to run())
        # optionally it also specifies which filter we want the reference to be in
        self.ra = None  # in degrees
        self.dec = None  # in degrees
        self.target = None  # the name of the target / field ID / Object ID
        self.section_id = None  # a string with the section ID
        self.filter = None  # a string with the (short) name of the filter

    def setup_provenances(self, session=None):
        """Make the provenances for the images and all their products, including the coadd image.

        These are used both to establish the provenance of the reference itself,
        and to look for images and associated products (like SourceLists) when
        building the reference.
        """
        if self.pars.instruments is None or len(self.pars.instruments) == 0:
            raise ValueError('No instruments given to RefMaker!')

        self.im_provs = []
        self.ex_provs = []

        for inst in self.pars.instruments:
            load_exposure = Exposure.make_provenance(inst)
            pars = self.pipeline.preprocessor.pars.get_critical_pars()
            preprocessing = Provenance(
                process='preprocessing',
                code_version=load_exposure.code_version,  # TODO: allow loading versions for each process
                parameters=pars,
                upstreams=[load_exposure],
                is_testing='test_parameter' in pars,
            )
            pars = self.pipeline.extractor.pars.get_critical_pars()  # includes parameters of siblings
            extraction = Provenance(
                process='extraction',
                code_version=preprocessing.code_version,  # TODO: allow loading versions for each process
                parameters=pars,
                upstreams=[preprocessing],
                is_testing='test_parameter' in pars,
            )

            # the exposure provenance is not included in the reference provenance's upstreams
            self.im_provs.append(preprocessing)
            self.ex_provs.append(extraction)

        upstreams = self.im_provs + self.ex_provs  # all the provenances that go into the coadd
        # TODO: we are using preprocess.code_version but should really load the right code_version for this process.
        coadd_provs = self.coadd_pipeline.make_provenance_tree(upstreams, preprocessing.code_version, session=session)
        self.coadd_im_prov = coadd_provs['coaddition']
        self.coadd_ex_prov = coadd_provs['extraction']

        pars = self.pars.get_critical_pars()
        self.ref_prov = Provenance(
            process=self.pars.get_process_name(),
            code_version=self.coadd_im_prov.code_version,  # TODO: allow loading versions for each process
            parameters=pars,
            upstreams=[self.coadd_im_prov, self.coadd_ex_prov],
            is_testing='test_parameter' in pars,
        )

        # this hash uniquely identifies all the preprocessing and extraction hashes in this provenance's upstreams
        self.ref_upstream_hash = self.ref_prov.get_combined_upstream_hash()
        # NOTE: we could have just used the coadd_ex_prov hash, because that uniquely identifies the coadd_im_prov
        #  (it is an upstream) and through that the preprocessing and extraction provenances of the regular images.
        #  but I am not sure how the provenance tree will look like in the future, I am leaving this additional hash
        #  here to be safe. The important part is that this hash must be singular for each RefSet, so that the
        #  downstreams of the subtractions will have a well-defined provenance, one for each RefSet.

    def parse_arguments(self, *args, **kwargs):
        """Figure out if the input parameters are given as coordinates or as target + section ID pairs.

        Possible combinations:
        - float + float + string: interpreted as RA/Dec in degrees
        - str + str: try to interpret as sexagesimal (RA as hours, Dec as degrees)
                     if it fails, will interpret as target + section ID
        # TODO: can we identify a reference with only a target/field ID without a section ID? Issue #320
        In addition to the first two arguments, can also supply a filter name as a string
        and can provide a session object as an argument (in any position) to be used and kept open
        for the entire run. If not given a session, will open a new one and close it when done using it internally.

        Alternatively, can provide named arguments with the same combinations for either
        (ra, dec) or (target, section_id) and filter.

        Returns
        -------
        session: sqlalchemy.orm.session.Session object or None
            The session object, if it was passed in as a positional argument.
            If not given, the ref maker will just open and close sessions internally
            when needed.
        """
        self.ra = None
        self.dec = None
        self.target = None
        self.section_id = None
        self.filter = None

        args, kwargs, session = parse_session(*args, **kwargs)  # first pick out any sessions

        if len(args) == 3:
            if not isinstance(args[2], str):
                raise ValueError('Third argument must be a string, the filter name!')
            self.filter = args[2]
            args = args[:2]  # remove the last one

        if len(args) == 2:
            if isinstance(args[0], (float, int, np.number)) and isinstance(args[1], (float, int, np.number)):
                self.ra = float(args[0])
                self.dec = float(args[1])
            if isinstance(args[0], str) and isinstance(args[1], str):
                try:
                    self.ra = parse_sexigesimal_degrees(args[0], hours=True)
                    self.dec = parse_sexigesimal_degrees(args[1], hours=False)
                except ValueError:
                    self.target, self.section_id = args[0], args[1]
        elif len(args) == 0:  # parse kwargs instead!
            if 'ra' in kwargs and 'dec' in kwargs:
                self.ra = kwargs.pop('ra')
                if isinstance(self.ra, str):
                    self.ra = parse_sexigesimal_degrees(self.ra, hours=True)

                self.dec = kwargs.pop('dec')
                if isinstance(self.dec, str):
                    self.dec = parse_sexigesimal_degrees(self.dec, hours=False)

            elif 'target' in kwargs and 'section_id' in kwargs:
                self.target = kwargs.pop('target')
                self.section_id = kwargs.pop('section_id')
            else:
                raise ValueError('Cannot find ra/dec or target/section_id in any of the inputs! ')

            if 'filter' in kwargs:
                self.filter = kwargs.pop('filter')

        else:
            raise ValueError('Invalid number of arguments given to RefMaker.parse_arguments()')

        if self.filter is None:
            raise ValueError('No filter given to RefMaker.parse_arguments()!')

        return session

    def make_refset(self, session=None):
        """Create or load an existing RefSet with the required name.

        Will also make all the required provenances (using the config) and
        possibly append the reference provenance to the list of provenances
        on the RefSet.
        """
        with SmartSession(session) as dbsession:
            self.setup_provenances(session=dbsession)

            # first merge the reference provenance
            self.ref_prov = self.ref_prov.merge_concurrent(session=dbsession, commit=True)

            # now load or create a RefSet
            for i in range(5):  # a concurrent merge sort of loop
                self.refset = dbsession.scalars(sa.select(RefSet).where(RefSet.name == self.pars.name)).first()

                if self.refset is not None:
                    break
                else:  # not found any RefSet with this name
                    try:
                        self.refset = RefSet(
                            name=self.pars.name,
                            description=self.pars.description,
                            upstream_hash=self.ref_upstream_hash,
                        )
                        dbsession.add(self.refset)
                        dbsession.commit()
                    except IntegrityError as e:
                        # there was a violation on unique constraint on the "name" column:
                        if 'duplicate key value violates unique constraint "ix_refsets_name' in str(e):
                            session.rollback()
                            time.sleep(0.1 * 2 ** i)  # exponential sleep
                        else:
                            raise e
            else:  # if we didn't break out of the loop, there must have been some integrity error
                raise e

            if self.refset is None:
                raise RuntimeError(f'Failed to find or create a RefSet with the name "{self.pars.name}"!')

            if self.refset.upstream_hash != self.ref_upstream_hash:
                raise RuntimeError(
                    f'Found a RefSet with the name "{self.pars.name}", but it has a different upstream_hash!'
                )

            # If the provenance is not already on the RefSet, add it (or raise, if allow_append=False)
            if self.ref_prov.id not in [p.id for p in self.refset.provenances]:
                if self.pars.allow_append:
                    prov_list = self.refset.provenances
                    prov_list.append(self.ref_prov)
                    self.refset.provenances = prov_list  # not sure if appending directly will trigger an update to DB
                    dbsession.commit()
                else:
                    raise RuntimeError(
                        f'Found a RefSet with the name "{self.pars.name}", but it has a different provenance! '
                        f'Use "allow_append" parameter to add new provenances to this RefSet. '
                    )

    def run(self, *args, **kwargs):
        """Check if a reference exists for the given coordinates/field ID, and filter, and make it if it is missing.

        Will check if a RefSet exists with the same provenance and name, and if it doesn't, will create a new
        RefSet with these properties, to keep track of the reference provenances.

        Arguments specifying where in the sky to look for / create the reference are parsed by parse_arguments().
        Same is true for the filter choice.
        The remaining policy regarding which images to pick, and what provenance to use to find references,
        is defined by the parameters object of self and of self.pipeline.

        If one of the inputs is a session, will use that in the entire process.
        Otherwise, will open internal sessions and close them whenever they are not needed.

        Will return a Reference, or None in case it doesn't exist and cannot be created
        (e.g., because there are not enough images that pass the criteria).
        """
        session = self.parse_arguments(*args, **kwargs)

        with SmartSession(session) as dbsession:
            self.make_refset(session=dbsession)

            # look for the reference at the given location in the sky (via ra/dec or target/section_id)
            ref = Reference.get_references(
                ra=self.ra,
                dec=self.dec,
                target=self.target,
                section_id=self.section_id,
                filter=self.filter,
                provenance_ids=self.ref_prov.id,
                session=dbsession,
            )

            if ref:  # found a reference, can skip the next part of the code!
                if len(ref) == 0:
                    return None
                elif len(ref) == 1:
                    return ref[0]
                else:
                    raise RuntimeError(
                        f'Found multiple references with the same provenance {self.ref_prov.id} and location!'
                    )
            ############### no reference found, need to build one! ################

            # first get all the images that could be used to build the reference
            images = []  # can get images from different instruments
            for inst in self.pars.instrument:
                prov = [p for p in self.im_provs if p.upstreams[0].parameters['instrument'] == inst]
                if len(prov) == 0:
                    raise RuntimeError(f'Cannot find a provenance for instrument "{inst}" in im_provs!')
                if len(prov) > 1:
                    raise RuntimeError(f'Found multiple provenances for instrument "{inst}" in im_provs!')
                prov = prov[0]

                query_pars = dict(
                    instrument=inst,
                    ra=self.ra,  # can be None!
                    dec=self.dec,  # can be None!
                    target=self.target,  # can be None!
                    section_id=self.section_id,  # can be None!
                    filter=self.pars.filters,  # can be None!
                    project=self.pars.project,  # can be None!
                    min_dateobs=self.pars.start_time,
                    max_dateobs=self.pars.end_time,
                    seeing_quality_factor=self.pars.seeing_quality_factor,
                    provenance_ids=prov.id,
                )

                for key in self.pars.__image_query_pars__:
                    for min_max in ['min', 'max']:
                        query_pars[f'{min_max}_{key}'] = getattr(self.pars, f'{min_max}_{key}')  # can be None!

                # get the actual images that match the query

                images += dbsession.scalars(Image.query_images(**query_pars).limit(self.pars.max_number)).all()

            if len(images) < self.pars.min_number:
                SCLogger.info(f'Found {len(images)} images, need at least {self.pars.min_number} to make a reference!')
                return None

            # note that if there are multiple instruments, each query may load the max number of images,
            # that's why we must also limit the number of images after all queries have returned.
            if len(images) > self.pars.max_number:
                coeff = abs(self.pars.seeing_quality_factor)  # abs is used to make sure the coefficient is negative
                for im in images:
                    im.quality = im.lim_mag_estimate - coeff * im.fwhm_estimate

                # sort the images by the quality
                images = sorted(images, key=lambda x: x.quality, reverse=True)
                images = images[:self.pars.max_number]

            # make the reference (note that we are out of the session block, to release it while we coadd)
            images = sorted(images, key=lambda x: x.mjd)  # sort the images in chronological order for coaddition

            # load the extraction products of these images using the ex_provs
            for im in images:
                im.load_products(self.ex_provs, session=dbsession)
                prods = {p: getattr(im, p) for p in ['sources', 'psf', 'bg', 'wcs', 'zp']}
                if any([p is None for p in prods.values()]):
                    raise RuntimeError(
                        f'Image {im} is missing products {prods} for coaddition! '
                        f'Make sure to produce products using the provenances in ex_provs: '
                        f'{self.ex_provs}'
                    )

        # release the session when making the coadd image

        coadd_image = self.coadd_pipeline.run(images)

        ref = Reference(image=coadd_image, provenance=self.ref_prov)

        if self.pars.save_new_refs:
            with SmartSession(session) as dbsession:
                ref.image.save(overwrite=True)
                ref.image.sources.save(overwrite=True)
                ref.image.psf.save(overwrite=True)
                ref.image.bg.save(overwrite=True)
                ref.image.wcs.save(overwrite=True)
                # zp is not a FileOnDiskMixin!

                ref = ref.merge_all(dbsession)
                dbsession.commit()

        return ref
