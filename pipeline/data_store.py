import warnings
import datetime
import sqlalchemy as sa

from util.util import parse_session, listify
from util.logger import SCLogger

from models.base import SmartSession, FileOnDiskMixin, FourCorners
from models.provenance import CodeVersion, Provenance
from models.exposure import Exposure
from models.image import Image, image_upstreams_association_table
from models.source_list import SourceList
from models.psf import PSF
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference
from models.cutouts import Cutouts
from models.measurements import Measurements

# for each process step, list the steps that go into its upstream
UPSTREAM_STEPS = {
    'exposure': [],  # no upstreams
    'preprocessing': ['exposure'],
    'extraction': ['preprocessing'],
    'subtraction': ['referencing', 'preprocessing', 'extraction'],
    'detection': ['subtraction'],
    'cutting': ['detection'],
    'measuring': ['cutting'],
}

# The products that are made at each processing step.
# Usually it is only one, but sometimes there are multiple products for one step (e.g., extraction)
PROCESS_PRODUCTS = {
    'exposure': 'exposure',
    'preprocessing': 'image',
    'coaddition': 'image',
    'extraction': ['sources', 'psf', 'bg', 'wcs', 'zp'],
    'referencing': 'reference',
    'subtraction': 'sub_image',
    'detection': 'detections',
    'cutting': 'cutouts',
    'measuring': 'measurements',
}


class DataStore:
    """
    Create this object to parse user inputs and identify which data products need
    to be fetched from the database, and keep a cached version of the products for
    use downstream in the pipeline.
    """
    # the products_to_save are also getting cleared along with products_to_clear
    products_to_save = [
        'exposure',
        'image',
        'sources',
        'psf',
        'bg',
        'wcs',
        'zp',
        'sub_image',
        'detections',
        'cutouts',
        'measurements'
    ]

    # these get cleared but not saved
    products_to_clear = [
        'ref_image',
        'sub_image',
        'reference',
        'exposure_id',
        'section_id',
        'image_id',
        'session',
    ]

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

    def parse_args(self, *args, **kwargs):
        """
        Parse the arguments to the DataStore constructor.
        Can initialize based on exposure and section ids,
        or give a specific image id or coadd id.

        If given an Image that is already loaded with related products
        (SourceList, PSF, etc.) then these will also be added to the
        datastore's attributes, to be checked against Provenance in the
        usual way when the relevant getter is called (e.g., get_sources).

        Parameters
        ----------
        args: list
            A list of arguments to parse.
            Possible argument combinations are:
            - DataStore: makes a copy of the other DataStore's __dict__
            - exposure_id, section_id: give two integers or integer and string
            - Exposure, section_id: an Exposure object, and an integer or string
            - Image: an Image object.
            - image_id: give a single integer

        kwargs: dict
            A dictionary of keyword arguments to parse.
            Using named arguments allows the user to
            explicitly assign the values to the correct
            attributes. These are parsed after the args
            list and can override it!

        Additional things that can get automatically parsed,
        either by keyword or by the content of one of the args:
            - provenances / prov_tree: a dictionary of provenances for each process.
            - session: a sqlalchemy session object to use.
            -

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

        self.session = output_session

        # look for a user-given provenance tree
        provs = [
            arg for arg in args
            if isinstance(arg, dict) and all([isinstance(value, Provenance) for value in arg.values()])
        ]
        if len(provs) > 0:
            self.prov_tree = provs[0]
            # also remove the provenances from the args list
            args = [
                arg for arg in args
                if not isinstance(arg, dict) or not all([isinstance(value, Provenance) for value in arg.values()])
            ]
        found_keys = []
        for key, value in kwargs.items():
            if key in ['prov', 'provs', 'provenances', 'prov_tree', 'provs_tree', 'provenance_tree']:
                if not isinstance(value, dict) or not all([isinstance(v, Provenance) for v in value.values()]):
                    raise ValueError('Provenance tree must be a dictionary of Provenance objects.')
                self.prov_tree = value
                found_keys.append(key)

        for key in found_keys:
            del kwargs[key]

        # parse the args list
        arg_types = [type(arg) for arg in args]
        if arg_types == []:  # no arguments, quietly skip
            pass
        elif arg_types == [int, int] or arg_types == [int, str]:  # exposure_id, section_id
            self.exposure_id, self.section_id = args
        elif arg_types == [Exposure, int] or arg_types == [Exposure, str]:
            self.exposure, self.section_id = args
            self.exposure_id = self.exposure.id
        elif arg_types == [int]:
            self.image_id = args[0]
        elif arg_types == [Image]:
            self.image = args[0]
        # TODO: add more options here
        #  example: get a string filename to parse a specific file on disk
        else:
            raise ValueError(
                'Invalid arguments to DataStore constructor, '
                f'got {arg_types}. '
                f'Expected [int, int] or [int], or [<image>] or [<exposure>, <section id>]. '
            )

        # parse the kwargs dict
        for key, val in kwargs.items():
            # override these attributes explicitly
            if key in ['exposure_id', 'section_id', 'image_id']:
                if not isinstance(val, int):
                    raise ValueError(f'{key} must be an integer, got {type(val)}')
                setattr(self, key, val)

            if key == 'exposure':
                if not isinstance(val, Exposure):
                    raise ValueError(f'exposure must be an Exposure object, got {type(val)}')
                self.exposure = val

            if key == 'image':
                if not isinstance(val, Image):
                    raise ValueError(f'image must be an Image object, got {type(val)}')
                self.image = val

        if self.image is not None:
            for att in ['sources', 'psf', 'bg', 'wcs', 'zp', 'detections', 'cutouts', 'measurements']:
                if getattr(self.image, att, None) is not None:
                    setattr(self, att, getattr(self.image, att))

        return output_session

    @staticmethod
    def catch_failure_to_parse(exception, *args):
        """Call this when the from_args() function fails.
        It is gaurenteed to return a DataStore object,
        and will set the error attribute to the exception message.
        """
        datastores = [a for a in args if isinstance(a, DataStore)]
        if len(datastores) > 0:
            ds = datastores[0]
        else:
            ds = DataStore()  # return an empty datastore, as we cannot create it and cannot find one in args

        ds.exception = exception

        return ds

    def catch_exception(self, exception):
        """Store the exception into the datastore for later use. """
        self.exception = exception
        # This is a trivial function now, but we may want to do more complicated stuff down the road

    def read_exception(self):
        """Return the stored exception and clear it from the datastore. """
        output = self.exception
        self.exception = None
        return output

    def reraise(self):
        """If an exception is logged to the datastore, raise it. Otherwise pass. """
        if self.exception is not None:
            e = self.read_exception()
            raise e

    def __init__(self, *args, **kwargs):
        """
        See the parse_args method for details on how to initialize this object.

        Please make sure to add any new attributes to the products_to_save list.
        """
        # these are data products that can be cached in the store
        self._exposure = None  # single image, entire focal plane
        self._section = None  # SensorSection

        self.prov_tree = None  # provenance dictionary keyed on the process name

        # these all need to be added to the products_to_save list
        self.image = None  # single image from one sensor section
        self.sources = None  # extracted sources (a SourceList object, basically a catalog)
        self.psf = None  # psf determined from the extracted sources
        self.bg = None  # background from the extraction phase
        self.wcs = None  # astrometric solution
        self.zp = None  # photometric calibration
        self.reference = None  # the Reference object needed to make subtractions
        self.sub_image = None  # subtracted image
        self.detections = None  # a SourceList object for sources detected in the subtraction image
        self.cutouts = None  # cutouts around sources
        self.measurements = None  # photometry and other measurements for each source
        self.objects = None  # a list of Object associations of Measurements

        # these need to be added to the products_to_clear list
        self.ref_image = None  # to be used to make subtractions
        self.sub_image = None  # subtracted image
        self.reference = None  # the Reference object needed to make subtractions
        self.exposure_id = None  # use this and section_id to find the raw image
        self.section_id = None  # corresponds to SensorSection.identifier (*not* .id)
        self.image_id = None  # use this to specify an image already in the database

        self.warnings_list = None  # will be replaced by a list of warning objects in top_level.Pipeline.run()
        self.exception = None  # the exception object (so we can re-raise it if needed)
        self.runtimes = {}  # for each process step, the total runtime in seconds
        self.memory_usages = {}  # for each process step, the peak memory usage in MB
        self.products_committed = ''  # a comma separated list of object names (e.g., "image, sources") saved to DB
        self.report = None  # keep a reference to the report object for this run

        # The database session parsed in parse_args; it could still be None even after parse_args
        self.session = None
        self.parse_args(*args, **kwargs)

    @property
    def exposure( self ):
        if self._exposure is None:
            if self.exposure_id is not None:
                self._exposure = self.get_raw_exposure( session=self.session )
        return self._exposure

    @exposure.setter
    def exposure( self, value ):
        self._exposure = value
        self.exposure_id = value.id if value is not None else None

    @property
    def section( self ):
        if self._section is None:
            if self.section_id is not None:
                if self.exposure is not None:
                    self.exposure.instrument_object.fetch_sections()
                    self._section = self.exposure.instrument_object.get_section( self.section_id )
        return self._section

    @property
    def ref_image( self ):
        if self.reference is not None:
            return self.reference.image
        return None

    @ref_image.setter
    def ref_image( self, value ):
        if self.reference is None:
            self.reference = Reference()
        self.reference.image = value

    def __getattribute__(self, key):
        # if this datastore has a pending error, will raise it as soon as any other data is used
        if (
                key not in ['exception', 'read_exception', 'update_report', 'reraise', 'report'] and
                not key.startswith('__') and hasattr(self, 'exception') and self.exception is not None
        ):
            SCLogger.warning('DataStore has a pending exception. Call read_exception() to get it, or reraise() to raise it.')
            SCLogger.warning(f'Exception was triggered by trying to access attribute {key}.')
            raise self.exception

        value = super().__getattribute__(key)
        if key == 'image' and value is not None:
            self.append_image_products(value)

        return value

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

            if key == 'psf' and not isinstance(value, PSF):
                raise ValueError(f'psf must be a PSF object, got {type(value)}')

            if key == 'bg' and not isinstance(value, Background):
                raise ValueError(f'bg must be a Background object, got {type(value)}')

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

            if key == 'cutouts' and not isinstance(value, Cutouts):
                raise ValueError(f'cutouts must be a Cutouts object, got {type(value)}')

            if key == 'measurements' and not isinstance(value, list):
                raise ValueError(f'measurements must be a list of Measurements objects, got {type(value)}')

            if key == 'measurements' and not all([isinstance(m, Measurements) for m in value]):
                raise ValueError(
                    f'measurements must be a list of Measurement objects, got list with {[type(m) for m in value]}'
                )

            if (
                key == 'prov_tree' and not isinstance(value, dict) and
                not all([isinstance(v, Provenance) for v in value.values()])
            ):
                raise ValueError(f'prov_tree must be a list of Provenance objects, got {value}')

            if key == 'session' and not isinstance(value, sa.orm.session.Session):
                raise ValueError(f'Session must be a SQLAlchemy session or SmartSession, got {type(value)}')

        super().__setattr__(key, value)

    def update_report(self, process_step, session=None):
        """Update the report object with the latest results from a processing step that just finished. """
        self.report = self.report.scan_datastore(self, process_step=process_step, session=session)

    def finalize_report(self, session=None):
        """Mark the report as successful and set the finish time."""
        self.report.success = True
        self.report.finish_time = datetime.datetime.utcnow()
        with SmartSession(session) as session:
            new_report = session.merge(self.report)
            session.commit()
        self.report = new_report

    def get_inputs(self):
        """Get a string with the relevant inputs. """

        if self.image_id is not None:
            return f'image_id={self.image_id}'
        if self.image is not None:
            return f'image={self.image}'
        elif self.exposure_id is not None and self.section_id is not None:
            return f'exposure_id={self.exposure_id}, section_id={self.section_id}'
        elif self.exposure is not None and self.section_id is not None:
            return f'exposure={self.exposure}, section_id={self.section_id}'
        else:
            raise ValueError('Could not get inputs for DataStore.')

    def get_provenance(self, process, pars_dict, session=None):
        """Get the provenance for a given process.
        Will try to find a provenance that matches the current code version
        and the parameter dictionary, and if it doesn't find it,
        it will create a new Provenance object.

        This function should be called externally by applications
        using the DataStore, to get the provenance for a given process,
        or to make it if it doesn't exist.

        Getting upstreams:
        Will use the prov_tree attribute of the datastore (if it exists)
        and if not, will try to get the upstream provenances from objects
        it has in memory already.
        If it doesn't find an upstream in either places it would use the
        most recently created provenance as an upstream, but this should
        rarely happen.

        Note that the output provenance can be different for the given process,
        if there are new parameters that differ from those used to make this provenance.
        For example: a prov_tree contains a preprocessing provenance "A",
        and an extraction provenance "B". This function is called for
        the "extraction" step, but with some new parameters (different than in "B").
        The "A" provenance will be used as the upstream, but the output provenance
        will not be "B" because of the new parameters.
        This will not change the prov_tree or affect later calls to this function
        for downstream provenances.

        Parameters
        ----------
        process: str
            The name of the process, e.g., "preprocess", "extraction", "subtraction".
        pars_dict: dict
            A dictionary of parameters used for the process.
            These include the critical parameters for this process.
            Use a Parameter object's get_critical_pars().
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        prov: Provenance
            The provenance for the given process.

        """
        with SmartSession(session, self.session) as session:
            code_version = Provenance.get_code_version(session=session)
            if code_version is None:
                # this "null" version should never be used in production
                code_version = CodeVersion(version='v0.0.0')
                code_version.update()  # try to add current git hash to version object

            # check if we can find the upstream provenances
            upstreams = []
            for name in UPSTREAM_STEPS[process]:
                prov = None
                # first try to load an upstream that was given explicitly:
                if self.prov_tree is not None and name in self.prov_tree:
                    prov = self.prov_tree[name]

                if prov is None:  # if that fails, see if the correct object exists in memory
                    obj_names = PROCESS_PRODUCTS[name]
                    if isinstance(obj_names, str):
                        obj_names = [obj_names]
                    obj = getattr(self, obj_names[0], None)  # only need one object to get the provenance
                    if isinstance(obj, list):
                        obj = obj[0]  # for cutouts or measurements just use the first one

                    if obj is not None and hasattr(obj, 'provenance') and obj.provenance is not None:
                        prov = obj.provenance

                if prov is not None:  # if we don't find one of the upstreams, it will raise an exception
                    upstreams.append(prov)

            if len(upstreams) != len(UPSTREAM_STEPS[process]):
                raise ValueError(f'Could not find all upstream provenances for process {process}.')

            for u in upstreams:  # check if "referencing" is in the list, if so, replace it with its upstreams
                if u.process == 'referencing':
                    upstreams.remove(u)
                    for up in u.upstreams:
                        upstreams.append(up)

            # we have a code version object and upstreams, we can make a provenance
            prov = Provenance(
                process=process,
                code_version=code_version,
                parameters=pars_dict,
                upstreams=upstreams,
                is_testing="test_parameter" in pars_dict,  # this is a flag for testing purposes
            )
            prov = prov.merge_concurrent(session=session, commit=True)

        return prov

    def _get_provenance_for_an_upstream(self, process, session=None):
        """Get the provenance for a given process, without parameters or code version.
        This is used to get the provenance of upstream objects.
        Looks for a matching provenance in the prov_tree attribute.

        Example:
        When making a SourceList in the extraction phase, we will want to know the provenance
        of the Image object (from the preprocessing phase).
        To get it, we'll call this function with process="preprocessing".
        If prov_tree is not None, it will provide the provenance for the preprocessing phase.

        Will raise if no provenance can be found.
        """
        # see if it is in the prov_tree
        if self.prov_tree is not None:
            if process in self.prov_tree:
                return self.prov_tree[process]
            else:
                raise ValueError(f'No provenance found for process "{process}" in prov_tree!')

        return None  # if not found in prov_tree, just return None

    def get_raw_exposure(self, session=None):
        """
        Get the raw exposure from the database.
        """
        if self._exposure is None:
            if self.exposure_id is None:
                raise ValueError('Cannot get raw exposure without an exposure_id!')

            with SmartSession(session, self.session) as session:
                self._exposure = session.scalars(sa.select(Exposure).where(Exposure.id == self.exposure_id)).first()

        return self._exposure

    def get_image(self, provenance=None, session=None):
        """
        Get the pre-processed (or coadded) image, either from
        memory or from the database.
        If the store is initialized with an image_id,
        that image is returned, no matter the
        provenances or the local parameters.
        This is the only way to ask for a coadd image.
        If an image with such an id is not found,
        in memory or in the database, will raise a ValueError.
        If exposure_id and section_id are given, will
        load an image that is consistent with
        that exposure and section ids, and also with
        the code version and critical parameters
        (using a matching of provenances).
        In this case we will only load a regular image, not a coadd.
        If no matching image is found, will return None.

        Note that this also updates self.image with the found image (or None).

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the image.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the prov_tree and if that is None,
            will use the latest provenance for the "preprocessing" process.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        image: Image object
            The image object, or None if no matching image is found.

        """
        session = self.session if session is None else session

        if (
                (self.exposure is None or self.section is None) and
                (self.exposure_id is None or self.section_id is None) and
                self.image is None and self.image_id is None
        ):
            raise ValueError('Cannot get image without one of (exposure_id, section_id), '
                             '(exposure, section), image, or image_id!')

        if self.image_id is not None:  # we were explicitly asked for a specific image id:
            if isinstance(self.image, Image) and self.image.id == self.image_id:
                pass  # return self.image at the end of function...
            else:  # not found in local memory, get from DB
                with SmartSession(session) as session:
                    self.image = session.scalars(sa.select(Image).where(Image.id == self.image_id)).first()

            # we asked for a specific image, it should exist!
            if self.image is None:
                raise ValueError(f'Cannot find image with id {self.image_id}!')

        else:  # try to get the image based on exposure_id and section_id
            process = 'preprocessing'
            if self.image is not None and self.image.provenance is not None:
                process = self.image.provenance.process  # this will be "coaddition" sometimes!
            if provenance is None:  # try to get the provenance from the prov_tree
                provenance = self._get_provenance_for_an_upstream(process, session=session)

            if self.image is not None:
                # If an image already exists and image_id is none, we may be
                # working with a datastore that hasn't been committed to the
                # database; do a quick check for mismatches.
                # (If all the ids are None, it'll match even if the actual
                # objects are wrong, but, oh well.)
                if (
                    self.exposure_id is not None and self.section_id is not None and
                    (self.exposure_id != self.image.exposure_id or self.section_id != self.image.section_id)
                ):
                    self.image = None
                if self.exposure is not None and self.image.exposure_id != self.exposure.id:
                    self.image = None
                if ( self.section is not None and self.image is not None and
                        str(self.image.section_id) != self.section.identifier ):
                    self.image = None
                if self.image is not None and provenance is not None and self.image.provenance.id != provenance.id:
                    self.image = None

                # If we get here, self.image is presumed to be good

            if self.image is None:  # load from DB
                # this happens when the image is required as an upstream for another process (but isn't in memory)
                if provenance is not None:
                    with SmartSession(session) as session:
                        self.image = session.scalars(
                            sa.select(Image).where(
                                Image.exposure_id == self.exposure_id,
                                Image.section_id == str(self.section_id),
                                Image.provenance_id == provenance.id,
                            )
                        ).first()

        return self.image  # can return none if no image was found

    def append_image_products(self, image):
        """Append the image products to the image and sources objects.
        This is a convenience function to be used by the
        pipeline applications, to make sure the image
        object has all the data products it needs.
        """
        for att in ['sources', 'psf', 'bg', 'wcs', 'zp', 'detections', 'cutouts', 'measurements']:
            if getattr(self, att, None) is not None:
                setattr(image, att, getattr(self, att))
        if image.sources is not None:
            for att in ['wcs', 'zp']:
                if getattr(self, att, None) is not None:
                    setattr(image.sources, att, getattr(self, att))

    def get_sources(self, provenance=None, session=None):
        """Get the source list, either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use to get the source list.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, uses the appropriate provenance
            from the prov_tree dictionary.
            If prov_tree is None, will use the latest provenance
            for the "extraction" process.
            Usually the provenance is not given when sources are loaded
            in order to be used as an upstream of the current process.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        sl: SourceList object
            The list of sources for this image (the catalog),
            or None if no matching source list is found.

        """
        process_name = 'extraction'
        if provenance is None:  # try to get the provenance from the prov_tree
            provenance = self._get_provenance_for_an_upstream(process_name, session)

        # if sources exists in memory, check the provenance is ok
        if self.sources is not None:
            # make sure the sources object has the correct provenance
            if self.sources.provenance is None:
                raise ValueError('SourceList has no provenance!')
            if provenance is not None and provenance.id != self.sources.provenance.id:
                self.sources = None

        # TODO: do we need to test the SourceList Provenance has upstreams consistent with self.image.provenance?

        # not in memory, look for it on the DB
        if self.sources is None:
            with SmartSession(session, self.session) as session:
                image = self.get_image(session=session)
                if image is not None and provenance is not None:
                    self.sources = session.scalars(
                        sa.select(SourceList).where(
                            SourceList.image_id == image.id,
                            SourceList.is_sub.is_(False),
                            SourceList.provenance_id == provenance.id,
                        )
                    ).first()

        return self.sources

    def get_psf(self, provenance=None, session=None):
        """Get a PSF, either from memory or from the database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the PSF.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, uses the appropriate provenance
            from the prov_tree dictionary.
            Usually the provenance is not given when the psf is loaded
            in order to be used as an upstream of the current process.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        psf: PSF object
            The point spread function object for this image,
            or None if no matching PSF is found.

        """
        process_name = 'extraction'
        if provenance is None:  # try to get the provenance from the prov_tree
            provenance = self._get_provenance_for_an_upstream(process_name, session)

        # if psf exists in memory, check the provenance is ok
        if self.psf is not None:
            # make sure the psf object has the correct provenance
            if self.psf.provenance is None:
                raise ValueError('PSF has no provenance!')
            if provenance is not None and provenance.id != self.psf.provenance.id:
                self.psf = None

        # TODO: do we need to test the PSF Provenance has upstreams consistent with self.image.provenance?

        # not in memory, look for it on the DB
        if self.psf is None:
            with SmartSession(session, self.session) as session:
                image = self.get_image(session=session)
                if image is not None:
                    self.psf = session.scalars(
                        sa.select(PSF).where(PSF.image_id == image.id, PSF.provenance_id == provenance.id)
                    ).first()

        return self.psf

    def get_background(self, provenance=None, session=None):
        """Get a Background object, either from memory or from the database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the background.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, uses the appropriate provenance
            from the prov_tree dictionary.
            Usually the provenance is not given when the background is loaded
            in order to be used as an upstream of the current process.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        bg: Background object
            The background object for this image,
            or None if no matching background is found.

        """
        process_name = 'extraction'
        if provenance is None:  # try to get the provenance from the prov_tree
            provenance = self._get_provenance_for_an_upstream(process_name, session)

        # if background exists in memory, check the provenance is ok
        if self.bg is not None:
            # make sure the background object has the correct provenance
            if self.bg.provenance is None:
                raise ValueError('Background has no provenance!')
            if provenance is not None and provenance.id != self.bg.provenance.id:
                self.bg = None

        # TODO: do we need to test the b/g Provenance has upstreams consistent with self.image.provenance?

        # not in memory, look for it on the DB
        if self.bg is None:
            with SmartSession(session, self.session) as session:
                image = self.get_image(session=session)
                if image is not None:
                    self.bg = session.scalars(
                        sa.select(Background).where(
                            Background.image_id == image.id,
                            Background.provenance_id == provenance.id,
                        )
                    ).first()

        return self.bg

    def get_wcs(self, provenance=None, session=None):
        """Get an astrometric solution in the form of a WorldCoordinates object, from memory or from the database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the WCS.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, uses the appropriate provenance
            from the prov_tree dictionary.
            If prov_tree is None, will use the latest provenance
            for the "extraction" process.
            Usually the provenance is not given when the wcs is loaded
            in order to be used as an upstream of the current process.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        wcs: WorldCoordinates object
            The world coordinates object for this image,
            or None if no matching WCS is found.

        """
        process_name = 'extraction'
        if provenance is None:  # try to get the provenance from the prov_tree
            provenance = self._get_provenance_for_an_upstream(process_name, session)

        # if psf exists in memory, check the provenance is ok
        if self.wcs is not None:
            # make sure the psf object has the correct provenance
            if self.wcs.provenance is None:
                raise ValueError('WorldCoordinates has no provenance!')
            if provenance is not None and provenance.id != self.wcs.provenance.id:
                self.wcs = None

        # TODO: do we need to test the WCS Provenance has upstreams consistent with self.sources.provenance?

        # not in memory, look for it on the DB
        if self.wcs is None:
            with SmartSession(session, self.session) as session:
                sources = self.get_sources(session=session)
                if sources is not None and sources.id is not None:
                    self.wcs = session.scalars(
                        sa.select(WorldCoordinates).where(
                            WorldCoordinates.sources_id == sources.id, WorldCoordinates.provenance_id == provenance.id
                        )
                    ).first()

        return self.wcs

    def get_zp(self, provenance=None, session=None):
        """Get a photometric solution in the form of a ZeroPoint object, from memory or from the database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the ZP.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, uses the appropriate provenance
            from the prov_tree dictionary.
            If prov_tree is None, will use the latest provenance
            for the "extraction" process.
            Usually the provenance is not given when the zp is loaded
            in order to be used as an upstream of the current process.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        zp: ZeroPoint object
            The zero point object for this image,
            or None if no matching ZP is found.

        """
        process_name = 'extraction'
        if provenance is None:  # try to get the provenance from the prov_tree
            provenance = self._get_provenance_for_an_upstream(process_name, session)

        # if psf exists in memory, check the provenance is ok
        if self.zp is not None:
            # make sure the psf object has the correct provenance
            if self.zp.provenance is None:
                raise ValueError('ZeroPoint has no provenance!')
            if provenance is not None and provenance.id != self.zp.provenance.id:
                self.zp = None

        # TODO: do we need to test the ZP Provenance has upstreams consistent with self.sources.provenance?

        # not in memory, look for it on the DB
        if self.zp is None:
            with SmartSession(session, self.session) as session:
                sources = self.get_sources(session=session)
                if sources is not None and sources.id is not None:
                    self.zp = session.scalars(
                        sa.select(ZeroPoint).where(
                            ZeroPoint.sources_id == sources.id, ZeroPoint.provenance_id == provenance.id
                        )
                    ).first()

        return self.zp

    def get_reference(self, provenances=None, min_overlap=0.85, match_filter=True,
                      ignore_target_and_section=False, skip_bad=True, session=None ):
        """Get the reference for this image.

        Parameters
        ----------
        provenances: list of provenance objects
            A list of provenances to use to identify a reference.
            Will check for existing references for each one of these provenances,
            and will apply any additional criteria to each resulting reference, in turn,
            until the first one qualifies and is the one returned
            (i.e, it is possible to take the reference matching the first provenance
            and never load the others).
            If not given, will try to get the provenances from the prov_tree attribute.
            If those are not given, or if no qualifying reference is found, will return None.
        min_overlap: float, default 0.85
            Area of overlap region must be at least this fraction of the
            area of the search image for the reference to be good.
            (Warning: calculation implicitly assumes that images are
            aligned N/S and E/W.)  Make this <= 0 to not consider
            overlap fraction when finding a reference.
        match_filter: bool, default True
            If True, only find a reference whose filter matches the
            DataStore's images' filter.
        ignore_target_and_section: bool, default False
            If False, will try to match based on the datastore image's target and
            section_id parameters (if they are not None) and only use RA/dec to match
            if they are missing. If True, will only use RA/dec to match.
        skip_bad: bool, default True
            If True, will skip references that are marked as bad.
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        ref: Image object
            The reference image for this image, or None if no reference is found.

        If min_overlap is given, it will return the reference that has the
        highest overlap fraction.  (If, by unlikely chance, more than one have
        identical overlap fractions, an undeterministically chosen
        reference will be returned.  Ideally, by construction, you will
        never have this situation in your database; you will only have a
        single valid reference image for a given instrument/filter/date
        that has an appreciable overlap with any possible image from
        that instrument.  The software does not enforce this, however.)
        """
        image = self.get_image(session=session)
        if image is None:
            return None  # cannot find a reference without a new image to match

        if provenances is None:  # try to get it from the prov_tree
            provenances = self._get_provenance_for_an_upstream('referencing')

        provenances = listify(provenances)

        if provenances is None:
            self.reference = None  # cannot get a reference without any associated provenances

        # first, some checks to see if existing reference is ok
        if self.reference is not None and provenances is not None:  # check for a mismatch of reference to provenances
            if self.reference.provenance_id not in [p.id for p in provenances]:
                self.reference = None

        if self.reference is not None and min_overlap is not None and min_overlap > 0:
            ovfrac = FourCorners.get_overlap_frac(image, self.reference.image)
            if ovfrac < min_overlap:
                self.reference = None

        if self.reference is not None and skip_bad:
            if self.reference.is_bad:
                self.reference = None

        if self.reference is not None and match_filter:
            if self.reference.filter != image.filter:
                self.reference = None

        if (
                self.reference is not None and not ignore_target_and_section and
                image.target is not None and image.section_id is not None
        ):
            if self.reference.target != image.target or self.reference.section_id != image.section_id:
                self.reference = None

        # if we have survived this long without losing the reference, can return it here:
        if self.reference is not None:
            return self.reference

        # No reference was found (or it didn't match other parameters) must find a new one
        with SmartSession(session, self.session) as session:
            if ignore_target_and_section or image.target is None or image.section_id is None:
                arguments = dict(ra=image.ra, dec=image.dec)
            else:
                arguments = dict(target=image.target, section_id=image.section_id)

            if match_filter:
                arguments['filter'] = image.filter
            else:
                arguments['filter'] = None

            arguments['skip_bad'] = skip_bad
            arguments['provenance_ids'] = provenances
            references = Reference.get_references(**arguments, session=session)

            self.reference = None
            for ref in references:
                if min_overlap is not None and min_overlap > 0:
                    ovfrac = FourCorners.get_overlap_frac(image, ref.image)
                    # print(
                    #     f'ref.id= {ref.id}, ra_left= {ref.image.ra_corner_00:.2f}, '
                    #     f'ra_right= {ref.image.ra_corner_11:.2f}, ovfrac= {ovfrac}'
                    # )
                    if ovfrac >= min_overlap:
                        self.reference = ref
                        break

        return self.reference

    def get_subtraction(self, provenance=None, session=None):
        """Get a subtraction Image, either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the subtraction.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "subtraction" process.
            Usually the provenance is not given when the subtraction is loaded
            in order to be used as an upstream of the current process.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        sub: Image
            The subtraction Image,
            or None if no matching subtraction image is found.

        """
        process_name = 'subtraction'
        # make sure the subtraction has the correct provenance
        if provenance is None:  # try to get the provenance from the prov_tree
            provenance = self._get_provenance_for_an_upstream(process_name, session)

        # if subtraction exists in memory, check the provenance is ok
        if self.sub_image is not None:
            # make sure the sub_image object has the correct provenance
            if self.sub_image.provenance is None:
                raise ValueError('Subtraction Image has no provenance!')
            if provenance is not None and provenance.id != self.sub_image.provenance.id:
                self.sub_image = None

        # TODO: do we need to test the subtraction Provenance has upstreams consistent with upstream provenances?

        # not in memory, look for it on the DB
        if self.sub_image is None:
            with SmartSession(session, self.session) as session:
                image = self.get_image(session=session)
                ref = self.get_reference(session=session)

                aliased_table = sa.orm.aliased(image_upstreams_association_table)
                self.sub_image = session.scalars(
                    sa.select(Image).join(
                        image_upstreams_association_table,
                        sa.and_(
                            image_upstreams_association_table.c.upstream_id == ref.image_id,
                            image_upstreams_association_table.c.downstream_id == Image.id,
                        )
                    ).join(
                        aliased_table,
                        sa.and_(
                            aliased_table.c.upstream_id == image.id,
                            aliased_table.c.downstream_id == Image.id,
                        )
                    ).where(Image.provenance_id == provenance.id)
                ).first()

        if self.sub_image is not None:
            self.sub_image.load_upstream_products()
            self.sub_image.coordinates_to_alignment_target()

        return self.sub_image

    def get_detections(self, provenance=None, session=None):
        """Get a SourceList for sources from the subtraction image, from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the source list.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "detection" process.
            Usually the provenance is not given when the subtraction is loaded
            in order to be used as an upstream of the current process.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        sl: SourceList object
            The list of sources for this subtraction image (the catalog),
            or None if no matching source list is found.

        """
        process_name = 'detection'
        if provenance is None:  # try to get the provenance from the prov_tree
            provenance = self._get_provenance_for_an_upstream(process_name, session)

        # not in memory, look for it on the DB
        if self.detections is not None:
            # make sure the detections have the correct provenance
            if self.detections.provenance is None:
                raise ValueError('SourceList has no provenance!')
            if provenance is not None and provenance.id != self.detections.provenance.id:
                self.detections = None

        if self.detections is None:
            with SmartSession(session, self.session) as session:
                sub_image = self.get_subtraction(session=session)

                self.detections = session.scalars(
                    sa.select(SourceList).where(
                        SourceList.image_id == sub_image.id,
                        SourceList.is_sub.is_(True),
                        SourceList.provenance_id == provenance.id,
                    )
                ).first()

        return self.detections

    def get_cutouts(self, provenance=None, session=None):
        """Get a list of Cutouts, either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the cutouts.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "cutting" process.
            Usually the provenance is not given when the subtraction is loaded
            in order to be used as an upstream of the current process.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        cutouts: list of Cutouts objects
            The list of cutouts, that will be empty if no matching cutouts are found.

        """
        process_name = 'cutting'
        if provenance is None:  # try to get the provenance from the prov_tree
            provenance = self._get_provenance_for_an_upstream(process_name, session)

        if self.cutouts is not None:
            self.cutouts.load_all_co_data()
            if len(self.cutouts.co_dict) == 0:
                self.cutouts = None  # TODO: what about images that actually don't have any detections?

            # make sure the cutouts have the correct provenance
            if self.cutouts is not None:
                if self.cutouts.provenance is None:
                    raise ValueError('Cutouts have no provenance!')
                if provenance is not None and provenance.id != self.cutouts.provenance.id:
                    self.cutouts = None

        # not in memory, look for it on the DB
        if self.cutouts is None:
            with SmartSession(session, self.session) as session:
                sub_image = self.get_subtraction(session=session)

                if sub_image is None:
                    return None

                if sub_image.sources is None:
                    sub_image.sources = self.get_detections(session=session)

                if sub_image.sources is None:
                    return None

                self.cutouts = session.scalars(
                    sa.select(Cutouts).where(
                        Cutouts.sources_id == sub_image.sources.id,
                        Cutouts.provenance_id == provenance.id,
                    )
                ).first()

        return self.cutouts

    def get_measurements(self, provenance=None, session=None):
        """Get a list of Measurements, either from memory or from database.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the measurements.
            This provenance should be consistent with
            the current code version and critical parameters.
            If none is given, will use the latest provenance
            for the "measurement" process.
            Usually the provenance is not given when the subtraction is loaded
            in order to be used as an upstream of the current process.
        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.

        Returns
        -------
        measurements: list of Measurement objects
            The list of measurements, that will be empty if no matching measurements are found.

        """
        process_name = 'measurement'
        if provenance is None:  # try to get the provenance from the prov_tree
            provenance = self._get_provenance_for_an_upstream(process_name, session)

        # make sure the measurements have the correct provenance
        if self.measurements is not None:
            if any([m.provenance is None for m in self.measurements]):
                raise ValueError('One of the Measurements has no provenance!')
            if provenance is not None and any([m.provenance.id != provenance.id for m in self.measurements]):
                self.measurements = None

        # not in memory, look for it on the DB
        if self.measurements is None:
            with SmartSession(session, self.session) as session:
                cutouts = self.get_cutouts(session=session)

                self.measurements = session.scalars(
                    sa.select(Measurements).where(
                        Measurements.cutouts_id == cutouts.id,
                        Measurements.provenance_id == provenance.id,
                    )
                ).all()

        return self.measurements

    def get_all_data_products(self, output='dict', omit_exposure=False):
        """Get all the data products associated with this Exposure.

        By default, this returns a dict with named entries.
        If using output='list', will return a list of all
        objects, including sub-lists. None values are skipped.

        This list does not include the reference image.

        Parameters
        ----------
        output: str, optional
            The output format. Can be 'dict' or 'list'.
            Default is 'dict'.
        omit_exposure: bool, default False
            If True, does not include the exposure in the list of data products

        Returns
        -------
        data_products: dict or list
            A dict with named entries, or a flattened list of all
            objects, including lists (e.g., Cutouts will be concatenated,
            no nested). Any None values will be removed.
        """
        attributes = [] if omit_exposure else [ '_exposure' ]
        attributes.extend( [ 'image', 'wcs', 'sources', 'psf', 'bg', 'zp', 'sub_image',
                             'detections', 'cutouts', 'measurements' ] )
        result = {att: getattr(self, att) for att in attributes}
        if output == 'dict':
            return result
        if output == 'list':
            return [result[att] for att in attributes if result[att] is not None]
        else:
            raise ValueError(f'Unknown output format: {output}')

    def save_and_commit(self, exists_ok=False, overwrite=True, no_archive=False,
                        update_image_header=False, force_save_everything=True, session=None):
        """Go over all the data products and add them to the session.

        If any of the data products are associated with a file on disk,
        that would be saved as well.

        In general, it will *not* save data products that have a
        non-null md5sum (or md5sum_extensions) line in the database.
        Reason: once that line is written, it means that that data
        product is "done" and will not change again.  As such, this
        routine assumes that it's all happily saved at least to the
        archive, so nothing needs to be written.

        There is one exception: the "image" (as opposed to weight or
        flags) extension of an Image.  If "update_image_header" is true,
        then the DataStore will save and overwrite just the image
        extension (not the weight or flags extensions) both to disk and
        to the archive, and will update the database md5sum line
        accordingly.  The *only* change that should have been made to
        the image file is in the header; the WCS and zeropoint keywords
        will have been updated.  The pipeline that uses the DataStore
        will be set to only do this once for each step (once the
        astro_cal_done and TODO_PHOTOMETRIC fields change from False to
        True), as the image headers get "first-look" values, not
        necessarily the latest and greatest if we tune either process.

        Parameters
        ----------
        exists_ok: bool, default False
            Ignored if overwrite is True.  Otherwise, this indicates
            what to do if the file exists on disk.  If exists_ok is
            True, then the file is assumed to be right on disk (and on
            the archive), and is not checked.  This is most efficient;
            if the file has already been saved, I/O won't be wasted
            saving it again and pushing it to the archive again.  If
            exists_ok is False, raise an exception if the file exists
            (and overwrite is False)

        overwrite: bool, default True
            If True, will overwrite any existing files on disk.

        no_archive: bool, default False
            If True, will not push files up to the archive, will only
            save on local disk.

        update_image_header: bool, default False
            See above.  If this is true, then the if there is an Image
            object in the data store, its "image" extension will be
            overwritten both on the local store and on the archive, and
            appropriate entry in the md5sum_extensions array of the
            Image object (and in row in the database) will be updated.
            THIS OPTION SHOULD BE USED WITH CARE.  It's an exception to
            the basic design of the pipeline, and adds redundant I/O
            (since the data hasn't changed, but at the very least the
            entire image will be sent back to the archive).  This should
            only be used when there are changes to the image header that
            need to be saved (e.g. to save a "first look" WCS or
            zeropoint).

        force_save_everything: bool, default False
            Write all files even if the md5sum exists in the database.
            Usually you don't want to use this, but it may be useful for
            testing purposes.

        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.
            Note that this method calls session.commit()

        """
        # save to disk whatever is FileOnDiskMixin
        for att in self.products_to_save:
            obj = getattr(self, att, None)
            if obj is None:
                continue

            SCLogger.debug( f'save_and_commit considering a {obj.__class__.__name__} with filepath '
                            f'{obj.filepath if isinstance(obj,FileOnDiskMixin) else "<none>"}' )

            if isinstance(obj, FileOnDiskMixin):
                mustsave = True
                # TODO : if some extensions have a None md5sum and others don't,
                #  right now we'll re-save everything.  Improve this to only
                #  save the necessary extensions.  (In practice, this should
                #  hardly ever come up.)
                if ( ( not force_save_everything )
                     and
                     ( ( obj.md5sum is not None )
                       or ( ( obj.md5sum_extensions is not None )
                            and
                            ( all( [ i is not None for i in obj.md5sum_extensions ] ) )
                           )
                      ) ):
                    mustsave = False

                # Special case handling for update_image_header for existing images.
                # (Not needed if the image doesn't already exist, hence the not mustsave.)
                if isinstance( obj, Image ) and ( not mustsave ) and update_image_header:
                    SCLogger.debug( 'Just updating image header.' )
                    try:
                        obj.save( only_image=True, just_update_header=True )
                    except Exception as ex:
                        SCLogger.error( f"Failed to update image header: {ex}" )
                        raise ex

                elif mustsave:
                    try:
                        obj.save( overwrite=overwrite, exists_ok=exists_ok, no_archive=no_archive )
                    except Exception as ex:
                        SCLogger.error( f"Failed to save a {obj.__class__.__name__}: {ex}" )
                        raise ex

                else:
                    SCLogger.debug( f'Not saving the {obj.__class__.__name__} because it already has '
                                   f'a md5sum in the database' )

        # carefully merge all the objects including the products
        with SmartSession(session, self.session) as session:
            if self.image is not None:
                self.image = self.image.merge_all(session)
                for att in ['sources', 'psf', 'bg', 'wcs', 'zp']:
                    setattr(self, att, None)  # avoid automatically appending to the image self's non-merged products
                for att in ['exposure', 'sources', 'psf', 'bg', 'wcs', 'zp']:
                    if getattr(self.image, att, None) is not None:
                        setattr(self, att, getattr(self.image, att))

            # This may well have updated some ids, as objects got added to the database
            if self.exposure_id is None and self._exposure is not None:
                self.exposure_id = self._exposure.id
            if self.image_id is None and self.image is not None:
                self.image_id = self.image.id

            self.sources = self.image.sources
            self.psf = self.image.psf
            self.bg = self.image.bg
            self.wcs = self.image.wcs
            self.zp = self.image.zp

            session.commit()
            self.products_committed = 'image, sources, psf, wcs, zp, bg'

            if self.sub_image is not None:
                if self.reference is not None:
                    self.reference = self.reference.merge_all(session)
                self.sub_image.new_image = self.image  # update with the now-merged image
                self.sub_image = self.sub_image.merge_all(session)  # merges the upstream_images and downstream products
                self.sub_image.ref_image.id = self.sub_image.ref_image_id
                self.detections = self.sub_image.sources

                session.commit()
                self.products_committed += ', sub_image'

            if self.detections is not None:
                more_products = 'detections'
                if self.cutouts is not None:
                    self.cutouts.sources = self.detections
                    self.cutouts = session.merge(self.cutouts)
                    more_products += ', cutouts'

                if self.measurements is not None:
                    for i, m in enumerate(self.measurements):
                        # use the new, merged cutouts
                        self.measurements[i].cutouts = self.cutouts
                        self.measurements[i].associate_object(session)
                        self.measurements[i] = session.merge(self.measurements[i])
                        self.measurements[i].object.measurements.append(self.measurements[i])
                    more_products += ', measurements'

                session.commit()
                self.products_committed += ', ' + more_products

    def delete_everything(self, session=None, commit=True):
        """Delete everything associated with this sub-image.

        All data products in the data store are removed from the DB,
        and all files on disk and in the archive are deleted.

        NOTE: does *not* delete the exposure.  (There may well be other
        data stores out there with different images from the same
        exposure.)
        This does not delete the reference either.

        Parameters
        ----------
        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it at the end of the function.
            Note that this method calls session.commit()
        commit: bool, default True
            If True, will commit the transaction.  If False, will not
            commit the transaction, so the caller can do more work
            before committing.
            If session is None, commit must also be True.
        """
        if session is None and not commit:
            raise ValueError('If session is None, commit must be True')

        with SmartSession( session, self.session ) as session, warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                message=r'.*DELETE statement on table .* expected to delete \d* row\(s\).*',
            )
            autoflush_state = session.autoflush
            try:
                # no flush to prevent some foreign keys from being voided before all objects are deleted
                session.autoflush = False
                obj_list = self.get_all_data_products(output='list', omit_exposure=True)
                for i, obj in enumerate(obj_list):  # first make sure all are merged
                    if isinstance(obj, list):
                        for j, o in enumerate(obj):
                            if o.id is not None:
                                for att in ['image', 'sources']:
                                    try:
                                        setattr(o, att, None)  # clear any back references before merging
                                    except AttributeError:
                                        pass  # ignore when the object doesn't have attribute, or it has no setter
                                obj_list[i][j] = session.merge(o)
                        continue
                    if sa.inspect(obj).transient:  # don't merge new objects, as that just "adds" them to DB!
                        obj_list[i] = session.merge(obj)

                for obj in obj_list:  # now do the deleting without flushing
                    # call the special delete method for list-arranged objects (e.g., cutouts, measurements)
                    if isinstance(obj, list):
                        if len(obj) > 0:
                            if hasattr(obj[0], 'delete_list'):
                                obj[0].delete_list(obj, session=session, commit=False)
                        continue
                    if isinstance(obj, FileOnDiskMixin):
                        obj.delete_from_disk_and_database(session=session, commit=False, archive=True)
                    if obj in session and sa.inspect(obj).pending:
                        session.expunge(obj)
                    if obj in session and sa.inspect(obj).persistent:
                        session.delete(obj)

                    if (
                            not sa.inspect(obj).detached and
                            hasattr(obj, 'provenance') and
                            obj.provenance is not None
                            and obj.provenance in session
                    ):
                        session.expunge(obj.provenance)

                session.flush()  # flush to finalize deletion of objects before we delete the Image

                # verify that the objects are in fact deleted by deleting the image at the root of the datastore
                if self.image is not None and self.image.id is not None:
                    session.execute(sa.delete(Image).where(Image.id == self.image.id))
                    # also make sure aligned images are deleted from disk and archive

                if self.sub_image is not None and self.sub_image._aligned_images is not None:
                    for im in self.sub_image._aligned_images:  # do not autoload, which happens if using aligned_images
                        im.remove_data_from_disk()

                # verify that no objects were accidentally added to the session's "new" set
                for obj in obj_list:
                    if isinstance(obj, list):
                        continue  # skip cutouts and measurements, as they could be slow to check

                    for new_obj in session.new:
                        if type(obj) is type(new_obj) and obj.id is not None and obj.id == new_obj.id:
                            session.expunge(new_obj)  # remove this object

                session.commit()

            finally:
                session.flush()
                session.autoflush = autoflush_state

        self.products_committed = ''  # TODO: maybe not critical, but what happens if we fail to delete some of them?

    def clear_products(self):
        """ Make sure all data products are None so that they aren't used again. """
        for att in self.products_to_save:
            setattr(self, att, None)

        for att in self.products_to_clear:
            setattr(self, att, None)
