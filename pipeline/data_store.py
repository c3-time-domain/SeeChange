import sqlalchemy as sa

from pipeline.utils import get_latest_provenance, parse_session

from models.base import SmartSession, FileOnDiskMixin, safe_merge
from models.provenance import CodeVersion, Provenance
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.references import ReferenceEntry
from models.cutouts import Cutouts
from models.measurements import Measurements


UPSTREAM_NAMES = {
    'preprocessing': [],
    'extraction': ['preprocessing'],
    'astro_cal': ['extraction'],
    'photo_cal': ['extraction', 'astro_cal'],
    'subtraction': ['preprocessing', 'extraction', 'astro_cal', 'photo_cal'],
    'detection': ['subtraction'],
    'cutting': ['detection'],
    'measurement': ['detection', 'photo_cal'],
}

UPSTREAM_OBJECTS = {
    'preprocessing': 'image',
    'coaddition': 'image',
    'extraction': 'sources',
    'astro_cal': 'wcs',
    'photo_cal': 'zp',
    'subtraction': 'sub_image',
    'detection': 'detections',
    'cutting': 'cutouts',
    'measurement': 'measurements',
}


class DataStore:
    """
    Create this object to parse user inputs and identify which data products need
    to be fetched from the database, and keep a cached version of the products for
    use downstream in the pipeline.
    """
    @staticmethod
    def from_args(*args, **kwargs):
        """
        Create a DataStore object from the given arguments.
        See the parse_args method for details on the different input parameters.

        Returns
        -------
        ds: DataStore
            The DataStore object.
        session: sqlalchemy.orm.session.Session or SmartSession or None
        """
        if len(args) == 0:
            raise ValueError('No arguments given to DataStore constructor!')
        if len(args) == 1 and isinstance(args[0], DataStore):
            return args[0], None
        if (
                len(args) == 2 and isinstance(args[0], DataStore) and
                (isinstance(args[1], sa.orm.session.Session) or args[1] is None)
        ):
            return args[0], args[1]
        else:
            ds = DataStore()
            session = ds.parse_args(*args, **kwargs)
            return ds, session

    def __init__(self, *args, **kwargs):
        """
        See the parse_args method for details on how to initialize this object.
        """
        # these are data products that can be cached in the store
        self.exposure = None  # single image, entire focal plane
        self.image = None  # single image from one sensor section
        self.sources = None  # extracted sources (a SourceList object, basically a catalog)
        self.wcs = None  # astrometric solution
        self.zp = None  # photometric calibration
        self.ref_image = None  # to be used to make subtractions
        self.sub_image = None  # subtracted image
        self.detections = None  # a SourceList object for sources detected in the subtraction image
        self.cutouts = None  # cutouts around sources
        self.measurements = None  # photometry and other measurements for each source

        self.upstream_provs = None  # provenances to override the upstreams if no upstream objects exist

        # these are identifiers used to find the data products in the database
        self.exposure_id = None  # use this and section_id to find the raw image
        self.section_id = None  # use this and exposure_id to find the raw image
        self.image_id = None  # use this to specify an image already in the database

        self.parse_args(*args, **kwargs)

    def parse_args(self, *args, **kwargs):
        """
        Parse the arguments to the DataStore constructor.
        Can initialize based on exposure and section ids,
        or give a specific image id or coadd id.

        Parameters
        ----------
        args: list
            A list of arguments to parse.
            Possible argument combinations are:
            - exposure_id, section_id: give two integers or integer and string
            - image_id: give a single integer

        kwargs: dict
            A dictionary of keyword arguments to parse.
            Using named arguments allows the user to
            explicitly assign the values to the correct
            attributes. These are parsed after the args
            list and can override it!

        Returns
        -------
        output_session: sqlalchemy.orm.session.Session or SmartSession
            If the user provided a session, return it to the scope
            that called "parse_args" so it can be used locally by
            the function that received the session as one of the arguments.
            If no session is given, will return None.
        """
        if len(args) == 1 and isinstance(args[0], DataStore):
            # if the only argument is a DataStore, copy it
            self.__dict__ = args[0].__dict__.copy()
            return

        args, kwargs, output_session = parse_session(*args, **kwargs)

        # remove any provenances from the args list
        for arg in args:
            if isinstance(arg, Provenance):
                self.upstream_provs.append(arg)
        args = [arg for arg in args if not isinstance(arg, Provenance)]

        # parse the args list
        arg_types = [type(arg) for arg in args]
        if arg_types == []:  # no arguments, quietly skip
            pass
        elif arg_types == [int, int] or arg_types == [int, str]:  # exposure_id, section_id
            self.exposure_id, self.section_id = args
        elif arg_types == [int]:
            self.image_id = args[0]
        # TODO: add more options here
        #  example: get a string filename to parse a specific file on disk
        else:
            raise ValueError(
                'Invalid arguments to DataStore constructor, '
                f'got {arg_types}. '
                f'Expected [int, int] or [int]'
            )

        # parse the kwargs dict
        for key, val in kwargs.items():
            # override these attributes explicitly
            if key in ['exposure_id', 'section_id', 'image_id']:
                if not isinstance(val, int):
                    raise ValueError(f'{key} must be an integer, got {type(val)}')
                setattr(self, key, val)

            # check for provenances
            if key in ['prov', 'provenances', 'upstream_provs', 'upstream_provenances']:
                new_provs = val
                if not isinstance(new_provs, list):
                    new_provs = [new_provs]

                for prov in new_provs:
                    if not isinstance(prov, Provenance):
                        raise ValueError(f'Provenance must be a Provenance object, got {type(prov)}')
                    self.upstream_provs.append(prov)

        return output_session

    def __setattr__(self, key, value):
        """
        Check some of the inputs before saving them.
        """
        if value is not None:
            if key in ['exposure_id', 'image_id'] and not isinstance(value, int):
                raise ValueError(f'{key} must be an integer, got {type(value)}')

            if key in ['section_id'] and not isinstance(value, (int, str)):
                raise ValueError(f'{key} must be an integer or a string, got {type(value)}')

            if key == 'image' and not isinstance(value, Image):
                raise ValueError(f'image must be an Image object, got {type(value)}')

            if key == 'sources' and not isinstance(value, SourceList):
                raise ValueError(f'sources must be a SourceList object, got {type(value)}')

            if key == 'wcs' and not isinstance(value, WorldCoordinates):
                raise ValueError(f'WCS must be a WorldCoordinates object, got {type(value)}')

            if key == 'zp' and not isinstance(value, ZeroPoint):
                raise ValueError(f'ZP must be a ZeroPoint object, got {type(value)}')

            if key == 'ref_image' and not isinstance(value, Image):
                raise ValueError(f'ref_image must be an Image object, got {type(value)}')

            if key == 'sub_image' and not isinstance(value, Image):
                raise ValueError(f'sub_image must be a Image object, got {type(value)}')

            if key == 'detections' and not isinstance(value, SourceList):
                raise ValueError(f'detections must be a SourceList object, got {type(value)}')

            if key == 'cutouts' and not isinstance(value, list):
                raise ValueError(f'cutouts must be a list of Cutout objects, got {type(value)}')

            if key == 'cutouts' and not all([isinstance(c, Cutouts) for c in value]):
                raise ValueError(f'cutouts must be a list of Cutouts objects, got list with {[type(c) for c in value]}')

            if key == 'measurements' and not isinstance(value, list):
                raise ValueError(f'measurements must be a list of Measurements objects, got {type(value)}')

            if key == 'measurements' and not all([isinstance(m, Measurements) for m in value]):
                raise ValueError(
                    f'measurements must be a list of Measurement objects, got list with {[type(m) for m in value]}'
                )

            if key == 'upstream_provs' and not isinstance(value, list):
                raise ValueError(f'upstream_provs must be a list of Provenance objects, got {type(value)}')

            if key == 'upstream_provs' and not all([isinstance(p, Provenance) for p in value]):
                raise ValueError(
                    f'upstream_provs must be a list of Provenance objects, got list with {[type(p) for p in value]}'
                )

            if key == 'session' and not isinstance(value, (sa.orm.session.Session, SmartSession)):
                raise ValueError(f'Session must be a SQLAlchemy session or SmartSession, got {type(value)}')

        super().__setattr__(key, value)

    def get_inputs(self):
        """Get a string with the relevant inputs. """

        if self.image_id is not None:
            return f'image_id={self.image_id}'
        elif self.exposure_id is not None and self.section_id is not None:
            return f'exposure_id={self.exposure_id}, section_id={self.section_id}'
        else:
            raise ValueError('Could not get inputs for DataStore.')

    def get_provenance(self, process, pars_dict, upstream_provs=None, session=None):
        """
        Get the provenance for a given process.
        Will try to find a provenance that matches the current code version
        and the parameter dictionary, and if it doesn't find it,
        it will create a new Provenance object.

        This function should be called externally by applications
        using the DataStore, to get the provenance for a given process,
        or to make it if it doesn't exist.

        Parameters
        ----------
        process: str
            The name of the process, e.g., "preprocess", "calibration", "subtraction".
            Use a Parameter object's get_process_name().
        pars_dict: dict
            A dictionary of parameters used for the process.
            These include the critical parameters for this process.
            Use a Parameter object's get_critical_pars().
        upstream_provs: list of Provenance objects
            A list of provenances to use as upstreams for the current
            provenance that is requested. Any upstreams that are not
            given will be filled using objects that already exist
            in the data store, or by getting the most up-to-date
            provenance from the database.
            The upstream provenances can be given directly as
            a function parameter, or using the DataStore constructor.
            If given as a parameter, it will override the DataStore's
            self.upstream_provs attribute for that call.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        prov: Provenance
            The provenance for the given process.
        """
        if upstream_provs is None:
            upstream_provs = self.upstream_provs

        with SmartSession(session) as session:
            code_version = Provenance.get_code_version(session=session)
            if code_version is None:
                # this "null" version should never be used in production
                code_version = CodeVersion(version='v0.0.0')
                code_version.update()  # try to add current git hash to version object

            # check if we can find the upstream provenances
            upstreams = []
            for name in UPSTREAM_NAMES[process]:
                # first try to load an upstream that was given explicitly:
                obj = getattr(self, UPSTREAM_OBJECTS[name], None)
                if upstream_provs is not None and name in [p.process for p in upstream_provs]:
                    prov = [p for p in upstream_provs if p.process == name][0]

                # second, try to get a provenance from objects saved to the store:
                elif obj is not None and hasattr(obj, 'provenance') and obj.provenance is not None:
                    prov = obj.provenance

                # last, try to get the latest provenance from the database:
                else:
                    prov = get_latest_provenance(name, session=session)

                # can't find any provenance upstream, therefore
                # there can't be any provenance for this process
                if prov is None:
                    return None

                upstreams.append(prov)

            if len(upstreams) != len(UPSTREAM_NAMES[process]):
                raise ValueError(f'Could not find all upstream provenances for process {process}.')

            # we have a code version object and upstreams, we can make a provenance
            prov = Provenance(
                process=process,
                code_version=code_version,
                parameters=pars_dict,
                upstreams=upstreams,
            )
            prov.update_hash()  # need a new object to calculate the hash, then check if it exists on the DB:
            existing_p = session.scalars(
                sa.select(Provenance).where(
                    Provenance.unique_hash == prov.unique_hash
                )
            ).first()

            if existing_p is not None:
                prov = existing_p

        return prov

    def _get_provanance_for_an_upstream(self, process, session=None):
        """
        Get the provenance for a given process, without knowing
        the parameters or code version.
        This simply looks for a matching provenance in the upstream_provs
        attribute, and if it is not there, it will call the latest provenance
        (for that process) from the database.
        This is used to get the provenance of upstream objects,
        only when those objects are not found in the store.
        Example: when looking for the upstream provenance of a
        photo_cal process, the upstream process is preprocess,
        so this function will look for the preprocess provenance.
        If the ZP object is from the DB then there must be provenance
        objects for the Image that was used to create it.
        If the ZP was just created, the Image should also be
        in memory even if the provenance is not on DB yet,
        in which case this function should not be called.

        This will raise if no provenance can be found.
        """
        # see if it is in the upstream_provs
        if self.upstream_provs is not None:
            prov_list = [p for p in self.upstream_provs if p.process == process]
            provenance = prov_list[0] if len(prov_list) > 0 else None
        else:
            provenance = None

        # try getting the latest from the database
        if provenance is None:  # check latest provenance
            provenance = get_latest_provenance(process, session=session)

        return provenance

    def get_raw_exposure(self, session=None):
        """
        Get the raw exposure from the database.
        """
        if self.exposure is None:
            if self.exposure_id is None:
                raise ValueError('Cannot get raw exposure without an exposure_id!')

            with SmartSession(session) as session:
                self.exposure = session.scalars(sa.select(Exposure).where(Exposure.id == self.exposure_id)).first()

        return self.exposure

    def get_image(self, provenance=None, session=None):
        """
        Get the pre-processed (or coadded) image, either from
        memory or from the database.
        If the store is initialized with an image_id,
        that image is returned, no matter the
        provenances or the local parameters.
        This is the only way to ask for a coadd image.
        If an image with such an id is not found,
        in memory or in the database, will raise
        an ValueError.
        If exposure_id and section_id are given, will
        load an image that is consistent with
        that exposure and section ids, and also with
        the code version and critical parameters
        (using a matching of provenances).
        In this case we will only load a regular
        image, not a coadded image.
        If no matching image is found, will return None.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the image.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "preprocessing" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        image: Image object
            The image object, or None if no matching image is found.

        """
        process_name = 'preprocessing'
        # we were explicitly asked for a specific image id:
        if self.image_id is not None:
            if isinstance(self.image, Image) and self.image.id == self.image_id:
                pass  # return self.image at the end of function...
            else:  # not found in local memory, get from DB
                with SmartSession(session) as session:
                    self.image = session.scalars(sa.select(Image).where(Image.id == self.image_id)).first()

            # we asked for a specific image, it should exist!
            if self.image is None:
                raise ValueError(f'Cannot find image with id {self.image_id}!')

        # this option is for when we are not sure which image id we need
        elif self.exposure_id is not None and self.section_id is not None:
            # check if self.image is the correct image:
            if (
                isinstance(self.image, Image) and self.image.exposure_id == self.exposure_id
                    and self.image.section_id == str(self.section_id)
            ):
                # make sure the image has the correct provenance
                if self.image is not None:
                    if self.image.provenance is None:
                        raise ValueError('Image has no provenance!')
                    if self.upstream_provs is not None:
                        provenances = [p for p in self.upstream_provs if p.process == process_name]
                    else:
                        provenances = []

                    if len(provenances) > 1:
                        raise ValueError(f'More than one "{process_name}" provenance found!')
                    if len(provenances) == 1:
                        # a mismatch of provenance and cached image:
                        if self.image.provenance.unique_hash != provenances[0].unique_hash:
                            self.image = None  # this must be an old image, get a new one

            if self.image is None:  # load from DB
                # this happens when the image is required as an upstream for another process (but isn't in memory)
                if provenance is None:  # check if in upstream_provs/database
                    provenance = self._get_provanance_for_an_upstream(process_name, session=session)

                if provenance is not None:  # if we can't find a provenance, then we don't need to load from DB
                    with SmartSession(session) as session:
                        self.image = session.scalars(
                            sa.select(Image).where(
                                Image.exposure_id == self.exposure_id,
                                Image.section_id == str(self.section_id),
                                Image.provenance.has(unique_hash=provenance.unique_hash)
                            )
                        ).first()

        else:
            raise ValueError('Cannot get processed image without exposure_id and section_id or image_id!')

        return self.image  # could return none if no image was found

    def get_sources(self, provenance=None, session=None):
        """
        Get a SourceList from the original image,
        either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the source list.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "extraction" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        sl: SourceList object
            The list of sources for this image (the catalog),
            or None if no matching source list is found.

        """
        process_name = 'extraction'
        # if sources exists in memory, check the provenance is ok
        if self.sources is not None:
            # make sure the sources object has the correct provenance
            if self.sources.provenance is None:
                raise ValueError('SourceList has no provenance!')

            if self.upstream_provs is not None:
                provenances = [p for p in self.upstream_provs if p.process == process_name]
            else:
                provenances = []
            if len(provenances) > 1:
                raise ValueError(f'More than one {process_name} provenance found!')
            if len(provenances) == 1:
                # a mismatch of given provenance and self.sources' provenance:
                if self.sources.provenance.unique_hash != provenances[0].unique_hash:
                    self.sources = None  # this must be an old sources object, get a new one

        # not in memory, look for it on the DB
        if self.sources is None:
            # this happens when the source list is required as an upstream for another process (but isn't in memory)
            if provenance is None:  # check if in upstream_provs/database
                provenance = self._get_provanance_for_an_upstream(process_name, session=session)

            if provenance is not None:  # if we can't find a provenance, then we don't need to load from DB
                with SmartSession(session) as session:
                    image = self.get_image(session=session)
                    self.sources = session.scalars(
                        sa.select(SourceList).where(
                            SourceList.image_id == image.id,
                            SourceList.is_sub.is_(False),
                            SourceList.provenance.has(unique_hash=provenance.unique_hash),
                        )
                    ).first()

        return self.sources

    def get_wcs(self, provenance=None, session=None):
        """
        Get an astrometric solution (in the form of a WorldCoordinates),
        either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the wcs.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "astro_cal" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        wcs: WorldCoordinates object
            The WCS object, or None if no matching WCS is found.

        """
        process_name = 'astro_cal'
        # make sure the wcs has the correct provenance
        if self.wcs is not None:
            if self.wcs.provenance is None:
                raise ValueError('WorldCoordinates has no provenance!')
            if self.upstream_provs is not None:
                provenances = [p for p in self.upstream_provs if p.process == process_name]
            else:
                provenances = []
            if len(provenances) > 1:
                raise ValueError(f'More than one "{process_name}" provenance found!')
            if len(provenances) == 1:
                # a mismatch of provenance and cached wcs:
                if self.wcs.provenance.unique_hash != provenances[0].unique_hash:
                    self.wcs = None  # this must be an old wcs object, get a new one

        # not in memory, look for it on the DB
        if self.wcs is None:
            with SmartSession(session) as session:
                # this happens when the wcs is required as an upstream for another process (but isn't in memory)
                if provenance is None:  # check if in upstream_provs/database
                    provenance = self._get_provanance_for_an_upstream(process_name, session=session)

                if provenance is not None:  # if None, it means we can't find it on the DB
                    sources = self.get_sources(session=session)
                    self.wcs = session.scalars(
                        sa.select(WorldCoordinates).where(
                            WorldCoordinates.source_list_id == sources.id,
                            WorldCoordinates.provenance.has(unique_hash=provenance.unique_hash),
                        )
                    ).first()

        return self.wcs

    def get_zp(self, provenance=None, session=None):
        """
        Get a photometric calibration (in the form of a ZeroPoint object),
        either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the wcs.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "photo_cal" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        wcs: ZeroPoint object
            The photometric calibration object, or None if no matching ZP is found.
        """
        process_name = 'photo_cal'
        # make sure the zp has the correct provenance
        if self.zp is not None:
            if self.zp.provenance is None:
                raise ValueError('ZeroPoint has no provenance!')

            if self.upstream_provs is not None:
                provenances = [p for p in self.upstream_provs if p.process == process_name]
            else:
                provenances = []
            if len(provenances) > 1:
                raise ValueError(f'More than one "{process_name}" provenance found!')
            if len(provenances) == 1:
                # a mismatch of provenance and cached zp:
                if self.zp.provenance.unique_hash != provenances[0].unique_hash:
                    self.zp = None  # this must be an old zp, get a new one

        # not in memory, look for it on the DB
        if self.zp is None:
            with SmartSession(session) as session:
                sources = self.get_sources(session=session)
                # TODO: do we also need the astrometric solution (to query for the ZP)?
                # this happens when the wcs is required as an upstream for another process (but isn't in memory)
                if provenance is None:  # check if in upstream_provs/database
                    provenance = self._get_provanance_for_an_upstream(process_name, session=session)

                if provenance is not None:  # if None, it means we can't find it on the DB
                    self.zp = session.scalars(
                        sa.select(ZeroPoint).where(
                            ZeroPoint.source_list_id == sources.id,
                            ZeroPoint.provenance.has(unique_hash=provenance.unique_hash),
                        )
                    ).first()

        return self.zp

    def get_reference_image(self, provenance=None, session=None):
        """
        Get the reference image for this image.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the coaddition.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "coaddition" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        ref: Image object
            The reference image for this image, or None if no reference is found.

        """
        if self.ref_image is None:

            with SmartSession(session) as session:
                image = self.get_image(session=session)

                ref_entry = session.scalars(
                    sa.select(ReferenceEntry).where(
                        sa.or_(
                            ReferenceEntry.validity_start.is_(None),
                            ReferenceEntry.validity_start <= image.observation_time
                        ),
                        sa.or_(
                            ReferenceEntry.validity_end.is_(None),
                            ReferenceEntry.validity_end >= image.observation_time
                        ),
                        ReferenceEntry.filter == image.filter,
                        ReferenceEntry.target == image.target,
                        ReferenceEntry.is_bad.is_(False),
                    )
                ).first()

                if ref_entry is None:
                    raise ValueError(f'No reference image found for image {image.id}')

                self.ref_image = ref_entry.image

        return self.ref_image

    def get_subtraction(self, provenance=None, session=None):
        """
        Get a subtraction Image, either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the subtraction.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "subtraction" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        sub: Image
            The subtraction Image,
            or None if no matching subtraction image is found.

        """
        process_name = 'subtraction'
        # make sure the subtraction has the correct provenance
        if self.sub_image is not None:
            if self.sub_image.provenance is None:
                raise ValueError('Subtraction image has no provenance!')
            if self.upstream_provs is not None:
                provenances = [p for p in self.upstream_provs if p.process == process_name]
            else:
                provenances = []
            if len(provenances) > 1:
                raise ValueError(f'More than one "{process_name}" provenance found!')
            if len(provenances) > 0:
                # a mismatch of provenance and cached subtraction image:
                if self.sub_image.provenance.unique_hash != provenances[0].unique_hash:
                    self.sub_image = None  # this must be an old subtraction image, need to get a new one

        # not in memory, look for it on the DB
        if self.sub_image is None:
            with SmartSession(session) as session:
                image = self.get_image(session=session)
                ref = self.get_reference_image(session=session)

                # this happens when the subtraction is required as an upstream for another process (but isn't in memory)
                if provenance is None:  # check if in upstream_provs/database
                    provenance = self._get_provanance_for_an_upstream(process_name, session=session)

                if provenance is not None:  # if None, it means we can't find it on the DB
                    self.sub_image = session.scalars(
                        sa.select(Image).where(
                            Image.ref_image_id == ref.id,
                            Image.new_image_id == image.id,
                            Image.provenance.has(unique_hash=provenance.unique_hash),
                        )
                    ).first()

        return self.sub_image

    def get_detections(self, provenance=None, session=None):
        """
        Get a SourceList for sources from the subtraction image,
        either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the source list.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "detection" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        sl: SourceList object
            The list of sources for this subtraction image (the catalog),
            or None if no matching source list is found.

        """
        process_name = 'detection'
        # not in memory, look for it on the DB
        if self.detections is not None:
            # make sure the wcs has the correct provenance
            if self.detections.provenance is None:
                raise ValueError('SourceList has no provenance!')

            if self.upstream_provs is not None:
                provenances = [p for p in self.upstream_provs if p.process == process_name]
            else:
                provenances = []
            if len(provenances) > 1:
                raise ValueError(f'More than one "{process_name}" provenance found!')
            if len(provenances) == 1:
                # a mismatch of provenance and cached detections:
                if self.detections.provenance.unique_hash != provenances[0].unique_hash:
                    self.detections = None  # this must be an old detections object, need to get a new one

        if self.detections is None:
            with SmartSession(session) as session:
                sub_image = self.get_subtraction(session=session)

                # this happens when the wcs is required as an upstream for another process (but isn't in memory)
                if provenance is None:  # check if in upstream_provs/database
                    provenance = self._get_provanance_for_an_upstream(process_name, session=session)

                if provenance is not None:  # if None, it means we can't find it on the DB
                    self.detections = session.scalars(
                        sa.select(SourceList).where(
                            SourceList.image_id == sub_image.id,
                            SourceList.is_sub.is_(True),
                            SourceList.provenance.has(unique_hash=provenance.unique_hash),
                        )
                    ).first()

        return self.detections

    def get_cutouts(self, provenance=None, session=None):
        """
        Get a list of Cutouts, either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the measurements.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "cutting" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        measurements: list of Measurement objects
            The list of measurements, or None if no matching measurements are found.

        """
        process_name = 'cutting'
        # make sure the cutouts have the correct provenance
        if self.cutouts is not None:
            if any([c.provenance is None for c in self.cutouts]):
                raise ValueError('One of the Cutouts has no provenance!')

            if self.upstream_provs is not None:
                provenances = [p for p in self.upstream_provs if p.process == process_name]
            else:
                provenances = []
            if len(provenances) > 1:
                raise ValueError(f'More than one "{process_name}" provenance found!')
            if len(provenances) == 1:
                # a mismatch of provenance and cached cutouts:
                if any([c.provenance.unique_hash != provenances[0].unique_hash for c in self.cutouts]):
                    self.cutouts = None  # this must be an old cutouts list, need to get a new one

        # not in memory, look for it on the DB
        if self.cutouts is None:
            with SmartSession(session) as session:
                sub_image = self.get_subtraction(session=session)

                # this happens when the cutouts are required as an upstream for another process (but aren't in memory)
                if provenance is None:
                    provenance = self._get_provanance_for_an_upstream(process_name, session=session)

                if provenance is not None:  # if None, it means we can't find it on the DB
                    self.cutouts = session.scalars(
                        sa.select(Cutouts).where(
                            Cutouts.sub_image_id == sub_image.id,
                            Cutouts.provenance.has(unique_hash=provenance.unique_hash),
                        )
                    ).all()

        return self.cutouts

    def get_measurements(self, provenance=None, session=None):
        """
        Get a list of Measurements, either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the measurements.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "measurement" process.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.

        Returns
        -------
        measurements: list of Measurement objects
            The list of measurements, or None if no matching measurements are found.

        """
        process_name = 'measurement'
        # make sure the measurements have the correct provenance
        if self.measurements is not None:
            if any([m.provenance is None for m in self.measurements]):
                raise ValueError('One of the Measurements has no provenance!')

            if self.upstream_provs is not None:
                provenances = [p for p in self.upstream_provs if p.process == process_name]
            else:
                provenances = []
            if len(provenances) > 1:
                raise ValueError(f'More than one "{process_name}" provenance found!')
            if len(provenances) == 1:
                # a mismatch of provenance and cached image:
                if any([m.provenance.unique_hash != provenances[0].unique_hash for m in self.measurements]):
                    self.measurements = None

        # not in memory, look for it on the DB
        if self.measurements is None:
            with SmartSession(session) as session:
                cutouts = self.get_cutouts(session=session)
                cutout_ids = [c.id for c in cutouts]

                # this happens when the measurements are required as an upstream (but aren't in memory)
                if provenance is None:
                    provenance = self._get_provanance_for_an_upstream(process_name, session=session)

                if provenance is not None:  # if None, it means we can't find it on the DB
                    self.measurements = session.scalars(
                        sa.select(Measurements).where(
                            Measurements.cutouts_id.in_(cutout_ids),
                            Measurements.provenance.has(unique_hash=provenance.unique_hash),
                        )
                    ).all()

        return self.measurements

    def get_all_data_products(self, output='dict'):
        """
        Get all the data products associated with this Exposure.
        By default, this returns a dict with named entries.
        If using output='list', will return a flattened list of all
        objects, including lists (e.g., Cutouts will be concatenated,
        no nested). Any None values will be removed.

        Parameters
        ----------
        output: str, optional
            The output format. Can be 'dict' or 'list'.
            Default is 'dict'.

        Returns
        -------
        data_products: dict or list
            A dict with named entries, or a flattened list of all
            objects, including lists (e.g., Cutouts will be concatenated,
            no nested). Any None values will be removed.
        """
        attributes = ['exposure', 'image', 'sources', 'wcs', 'zp', 'sub_image', 'detections', 'cutouts', 'measurements']
        result = {att: getattr(self, att) for att in attributes}
        if output == 'dict':
            return result
        if output == 'list':
            list_result = []
            for k, v in result.items():
                if isinstance(v, list):
                    list_result.extend(v)
                else:
                    list_result.append(v)

            return [v for v in list_result if v is not None]

        else:
            raise ValueError(f'Unknown output format: {output}')

    def save_and_commit(self, session=None):
        """
        Go over all the data products and add them to the session.
        If any of the data products are associated with a file on disk,
        that would be saved as well.

        Parameters
        ----------
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.
            Note that this method calls session.commit()
        """
        with SmartSession(session) as session:
            autoflush_state = session.autoflush
            try:
                # session.autoflush = False
                for obj in self.get_all_data_products(output='list'):
                    # print(f'saving {obj} with provenance: {getattr(obj, "provenance", None)}')

                    if isinstance(obj, FileOnDiskMixin):
                        obj.save()

                    obj = obj.recursive_merge(session)
                    session.add(obj)

                session.commit()
            finally:
                session.autoflush = autoflush_state

    def delete_everything(self, session=None):
        """
        Delete everything associated with this sub-image.
        All data products in the data store are removed from the DB,
        and all files on disk are deleted.

        Parameters
        ----------
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will open a new session and close it at
            the end of the function.
            Note that this method calls session.commit()
        """
        with SmartSession(session) as session:
            autoflush_state = session.autoflush
            try:
                session.autoflush = False
                for obj in self.get_all_data_products(output='list'):
                    # if hasattr(obj, 'provenance'):
                    #     print(f'Deleting {obj} with provenance= {obj.provenance}')
                    obj = safe_merge(session, obj)
                    if isinstance(obj, FileOnDiskMixin):
                        obj.remove_data_from_disk()
                    if obj in session:
                        session.delete(obj)

                session.commit()
            finally:
                session.autoflush = autoflush_state

