import io
import datetime
import sqlalchemy as sa
import uuid
# import traceback

from util.util import parse_session, listify, asUUID
from util.logger import SCLogger

from models.base import SmartSession, FileOnDiskMixin, FourCorners
from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference
from models.cutouts import Cutouts
from models.measurements import Measurements
from models.deepscore import DeepScore

# for each process step, list the steps that go into its upstream
# backgrounding, astrocal, photocal don't appear in this list because
# they share a provenance with 'extraction', so where you see 'extraction',
# treat it as the union of (extraction, bbackgrounding, astrocal, photocal).
UPSTREAM_STEPS = {
    'exposure': [],  # no upstreams
    'preprocessing': ['exposure'],
    'extraction': ['preprocessing'],
    'referencing': [],               # This is a special case; it *does* have upstreams, but outside the main pipeline
    'subtraction': ['referencing', 'preprocessing', 'extraction'],
    'detection': ['subtraction'],
    'cutting': ['detection'],
    'measuring': ['cutting'],
    'scoring': ['measuring'],
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
    'scoring': 'scores',
}


class DataStore:
    """An object that stores all of the data products from a run through a pipeline.

    Can be created in a few ways.  Standard is to initialize it either
    with an Exposure and a (string) section_id, or with an Image.  You
    can also initilize it by passing another DataStore, in which case
    the it will copy all the attributes (shallow copy) of the passed
    DataStore.  (It copies the __dict__ atribute.)

    Most pipeline tasks take a DataStore as an argument, and return
    another DataStore with updated products.  (Usually it's the same
    DataStore object, modified, that is returned.)

    To work best, you want the DataStore's provenance tree to be loaded
    with provenances consistent with the parmeters that you will be
    using in the various pipeline tasks.  The easiest way to do this is
    to have a fully initilized Pipeline object (see
    pipeline/top_level.py) and run

      ds.prov_tree = pipeline.make_provenance_tree()

    You can get the provenances from a DataStore with get_provenance;
    that will try to load a default if there isn't one already in the
    tree.  You can also use that function to update the provenances
    stored in the provenance tree.  You can manually update the
    provenances stored in the provenance tree with set_prov_tree.

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
        'measurements',
        'scores'
    ]

    # these get cleared but not saved
    products_to_clear = [
        'reference',
        '_ref_image',
        '_ref_sources'
        '_ref_bg',
        '_ref_psf',
        '_ref_wcs',
        '_ref_zp',
        'aligned_ref_image',
        'aligned_ref_sources'
        'aligned_ref_bg',
        'aligned_ref_psf',
        'aligned_ref_zp',
        'aligned_new_image',
        'aligned_new_sources'
        'aligned_new_bg',
        'aligned_new_psf',
        'aligned_new_zp'
        'aligned_wcs',
        '_sub_image',
        'reference',
        'exposure_id',
        'section_id',
        'image_id',
        'session',
        # Things specific to the zogy subtraction method
        'zogy_score',
        'zogy_alpha',
        'zogy_alpha_err',
        'zogy_psf'
    ]

    # These are the various data products that the DataStore can hold
    # These getters and setters make sure that the relationship IDs
    # between them are all set, and that if something is set to None,
    # everything downstream is also set to None.

    @property
    def exposure_id( self ):
        return self._exposure_id

    @exposure_id.setter
    def exposure_id( self, val ):
        if val is None:
            self._exposure_id = None
        else:
            if isinstance( val, uuid.UUID ):
                self._exposure_id = val
            else:
                # This will raise an exception if it's not a well-formed UUID string
                self._exposure_id = asUUID( val )
            if self._exposure is not None:
                self._exposure.id = self._exposure_id

    @property
    def image_id( self ):
        return self._image_id

    @image_id.setter
    def image_id( self, val ):
        if val is None:
            self._image_id = None
        else:
            if isinstance( val, uuid.UUID ):
                self._image_id = val
            else:
                # This will raise an execption if it's not a well-formed UUID string
                self._image_id = asUUID( val )
            if self._image is not None:
                self._image.id = self.image_id

    @property
    def exposure( self ):
        if self._exposure is None:
            if self.exposure_id is not None:
                self._exposure = self.get_raw_exposure( session=self.session )
        return self._exposure

    @exposure.setter
    def exposure( self, value ):
        self._exposure = value
        if self._exposure is not None:
            self._exposure.id
            self.exposure_id = self._exposure.id

    @property
    def section( self ):
        if self._section is None:
            if self.section_id is not None:
                if self.exposure is not None:
                    self.exposure.instrument_object.fetch_sections()
                    self._section = self.exposure.instrument_object.get_section( self.section_id )
        return self._section

    @section.setter
    def section( self, val ):
        raise NotImplementedError( "Don't set DataStore section, set section_id" )

    @property
    def image( self ):
        return self._image

    @image.setter
    def image( self, val ):
        if val is None:
            self._image = None
            self.sources = None
        else:
            if not isinstance( val, Image ):
                raise TypeError( f"DataStore.image must be an Image, not a {type(val)}" )
            if ( self._sources is not None ) and ( self._sources.image_id != val.id ):
                raise ValueError( "Can't set a DataStore image inconsistent with sources" )
            if self._exposure is not None:
                if ( val.exposure_id is not None ) and ( val.exposure_id != self._exposure.id ):
                    raise ValueError( "Setting an image whose exposure_id doesn't match DataStore's exposure's id" )
                val.exposure_id = self._exposure.id
            elif self.exposure_id is not None:
                if ( val.exposure_id is not None ) and ( val.exposure_id != self.exposure_id ):
                    raise ValueError( "Setting an image whose exposure_id doesn't match Datastore's exposure_id" )
                val.exposure_id = self.exposure_id

            if ( self.image_id is not None ) and ( val.id != self.image_id ):
                raise ValueError( "Setting an image whose id doesn't match DataStore's image_id" )

            self.image_id = val.id
            self._image = val


    @property
    def sources( self ):
        return self._sources

    @sources.setter
    def sources( self, val ):
        if val is None:
            self._sources = None
            self._bg = None
            self._psf = None
            self._wcs = None
            self._zp = None
            self.sub_image = None
        else:
            if self._image is None:
                raise RuntimeError( "Can't set DataStore sources until it has an image." )
            if not isinstance( val, SourceList ):
                raise TypeError( f"DatatStore.sources must be a SourceList, not a {type(val)}" )
            if ( ( ( self._bg is not None ) and ( self._bg.sources_id != val.id ) ) or
                 ( ( self._psf is not None ) and ( self._psf.sources_id != val.id ) ) or
                 ( ( self._wcs is not None ) and ( self._wcs.sources_id != val.id ) ) or
                 ( ( self._zp is not None ) and ( self._zp.sources_id != val.id ) ) ):
                raise ValueError( "Can't set a DataStore sources inconsistent with other data products" )
            self._sources = val
            self._sources.image_id = self._image.id

    @property
    def bg( self ):
        return self._bg

    @bg.setter
    def bg( self, val ):
        if val is None:
            self._bg = None
            self.sub_image = None
        else:
            if self._sources is None:
                raise RuntimeError( "Can't set DataStore bg until it has a sources." )
            if not isinstance( val, Background ):
                raise TypeError( f"DataStore.bg must be a Background, not a {type(val)}" )
            self._bg = val
            self._bg.sources_id = self._sources.id

    @property
    def psf( self ):
        return self._psf

    @psf.setter
    def psf( self, val ):
        if val is None:
            self._psf = None
            self.sub_image = None
        else:
            if self._sources is None:
                raise RuntimeError( "Can't set DataStore psf until it has a sources." )
            if not isinstance( val, PSF ):
                raise TypeError( f"DataStore.psf must be a PSF, not a {type(val)}" )
            self._psf = val
            self._psf.sources_id = self._sources.id

    @property
    def wcs( self ):
        return self._wcs

    @wcs.setter
    def wcs( self, val ):
        if val is None:
            self._wcs = None
            self.sub_image = None
        else:
            if self._sources is None:
                raise RuntimeError( "Can't set DataStore wcs until it has a sources." )
            if not isinstance( val, WorldCoordinates ):
                raise TypeError( f"DataStore.wcs must be a WorldCoordinates, not a {type(val)}" )
            self._wcs = val
            self._wcs.sources_id = self._sources.id

    @property
    def zp( self ):
        return self._zp

    @zp.setter
    def zp( self, val ):
        if val is None:
            self._zp = None
            self.sub_image = None
        else:
            if self._sources is None:
                raise RuntimeError( "Can't set DataStore zp until it has a sources." )
            if not isinstance( val, ZeroPoint ):
                raise TypeError( f"DataStore.zp must be a ZeroPoint, not a {type(val)}" )
            self._zp = val
            self._zp.sources_id = self._sources.id

    @property
    def ref_image( self ):
        if self._ref_image is None:
            if self.reference is not None:
                self._ref_image = Image.get_by_id( self.reference.image_id )
        return self._ref_image

    @ref_image.setter
    def ref_image( self, val ):
        raise RuntimeError( "Don't directly set ref_image, call get_reference" )

    @property
    def ref_sources( self ):
        if self._ref_sources is None:
            if self.reference is not None:
                ( self._ref_sources, self._ref_bg, self._ref_psf,
                  self._ref_wcs, self._ref_zp ) = self.reference.get_ref_data_products()
        return self._ref_sources

    @ref_sources.setter
    def ref_sources( self, val ):
        raise RuntimeError( "Don't directly set ref_sources, call get_reference" )

    @property
    def ref_bg( self ):
        if self._ref_bg is None:
            if self.reference is not None:
                ( self._ref_sources, self._ref_bg, self._ref_psf,
                  self._ref_wcs, self._ref_zp ) = self.reference.get_ref_data_products()
        return self._ref_bg

    @ref_bg.setter
    def ref_bg( self, val ):
        raise RuntimeError( "Don't directly set ref_bg, call get_reference" )

    @property
    def ref_psf( self ):
        if self._ref_psf is None:
            if self.reference is not None:
                ( self._ref_sources, self._ref_bg, self._ref_psf,
                  self._ref_wcs, self._ref_zp ) = self.reference.get_ref_data_products()
        return self._ref_psf

    @ref_psf.setter
    def ref_psf( self, val ):
        raise RuntimeError( "Don't directly set ref_psf, call get_reference" )

    @property
    def ref_wcs( self ):
        if self._ref_wcs is None:
            if self.reference is not None:
                ( self._ref_sources, self._ref_bg, self._ref_psf,
                  self._ref_wcs, self._ref_zp ) = self.reference.get_ref_data_products()
        return self._ref_wcs

    @ref_wcs.setter
    def ref_wcs( self, val ):
        raise RuntimeError( "Don't directly set ref_wcs, call get_reference" )

    @property
    def ref_zp( self ):
        if self._ref_zp is None:
            if self.reference is not None:
                ( self._ref_sources, self._ref_bg, self._ref_psf,
                  self._ref_wcs, self._ref_zp ) = self.reference.get_ref_data_products()
        return self._ref_zp

    @ref_zp.setter
    def ref_zp( self, val ):
        raise RuntimeError( "Don't directly set ref_zp, call get_reference" )

    @property
    def sub_image( self ):
        return self._sub_image

    @sub_image.setter
    def sub_image( self, val ):
        if val is None:
            self._sub_image = None
            self.detections = None
        else:
            if ( self._zp is None ) or ( self.ref_image is None ):
                raise RuntimeError( "Can't set DataStore sub_image until it has a zp and a ref_image" )
            if not isinstance( val, Image ):
                raise TypeError( f"DataStore.sub_image must be an Image, not a {type(val)}" )
            if not val.is_sub:
                raise ValueError( "DataStore.sub_image must have is_sub set" )
            if ( ( self._detections is not None ) and ( self._detections.image_id != val.id ) ):
                raise ValueError( "Can't set a sub_image inconsistent with detections" )
            if val.ref_id != self.reference.id:
                raise ValueError( "Can't set a sub_image inconsistent with reference" )
            if val.new_image_id != self.image.id:
                raise ValueError( "Can't set a sub image inconsistent with image" )
            # TODO : check provenance upstream of sub_image to make sure it's consistent
            #   with ds.sources?
            self._sub_image = val

    @property
    def detections( self ):
        return self._detections

    @detections.setter
    def detections( self, val ):
        if val is None:
            self._detections = None
            self.cutouts = None
        else:
            if self.sub_image is None:
                raise RuntimeError( "Can't set DataStore detections until it has a sub_image" )
            if not isinstance( val, SourceList ):
                raise TypeError( f"DataStore.detections must be a SourceList, not a {type(val)}" )
            if ( ( self._cutouts is not None ) and ( self._cutouts.sources_id != val.id ) ):
                raise ValueError( "Can't set a cutouts inconsistent with detections" )
            self._detections = val
            self._detections.image_id = self._sub_image.id

    @property
    def cutouts( self ):
        return self._cutouts

    @cutouts.setter
    def cutouts( self, val ):
        if val is None:
            self._cutouts = None
            self.measurements = None
        else:
            if self._detections is None:
                raise RuntimeError( "Can't set DataStore cutouts until it has a detections" )
            if not isinstance( val, Cutouts ):
                raise TypeError( f"DataStore.cutouts must be a Cutouts, not a {type(val)}" )
            if ( ( self._measurements is not None ) and
                 ( any( [ m.cutouts_id != val.id for m in self.measurements ] ) )
                ):
                raise ValueError( "Can't set a cutouts inconsistent with measurements" )
            self._cutouts = val
            self._cutouts.detections_id = self._detections.id

    @property
    def measurements( self ):
        return self._measurements

    @measurements.setter
    def measurements( self, val ):
        if val is None:
            self._measurements = None
            self.scores = None
        else:
            if self._cutouts is None:
                raise RuntimeError( "Can't set DataStore measurements until it has a cutouts" )
            if not isinstance( val, list ):
                raise TypeError( f"Datastore.measurements must be a list of Measurements, not a {type(val)}" )
            wrongtypes = set( [ type(m) for m in val if not isinstance( m, Measurements ) ] )
            if len(wrongtypes) > 0:
                raise TypeError( f"Datastore.measurements must be a list of Measurements, but the passed list "
                                 f"included {wrongtypes}" )
            self._measurements = val
            for m in self._measurements:
                m.cutouts_id = self._cutouts.id

    @property
    def scores( self ):
        return self._scores

    @scores.setter
    def scores( self, val ):
        if val is None:
            self._scores = None
        else:
            if ( self._measurements is None ):
                raise RuntimeError( " Can't set DataStore scores until it has measurements" )
            if not isinstance( val, list ):
                raise TypeError( f"Datastore.scores must be a list of scores, not a {type(val)}" )
            wrongtypes = set( [ type(s) for s in val if not isinstance( s, DeepScore ) ] )
            if len(wrongtypes) > 0:
                raise TypeError( f"Datastore.scores must be a list of DeepScores, but the passed list "
                                 f"included {wrongtypes}" )

            # ensure that there is a score for each measurement, otherwise reject
            if ( len( val ) != len(self._measurements) ):
                raise ValueError( "Score and measurements list not the same length" )

            # ensure that the scores relate to measurements in the datascore
            if ( set([str(score.measurements_id) for score in val])
                    .issubset(set([str(m.id) for m in self._measurements])) ):
                self._scores = val
            else:
                raise RuntimeError( "Attempted to set scores corresponding to wrong measurements")

    @staticmethod
    def from_args(*args, **kwargs):
        """Create a DataStore object from the given arguments.

        See the parse_args method for details on the different input parameters.

        Returns
        -------
        ds: DataStore
            The DataStore object.

        session: sqlalchemy.orm.session.Session or SmartSession or None
            Never use this.

        """
        if len(args) == 0:
            raise ValueError('No arguments given to DataStore constructor!')
        if len(args) == 1 and isinstance(args[0], DataStore):
            return args[0], None
        if (
                len(args) == 2 and isinstance(args[0], DataStore) and
                (isinstance(args[1], sa.orm.session.Session) or args[1] is None)
        ):
            if isinstance( args[1], sa.orm.session.Session ):
                SCLogger.error( "You passed a session to a DataStore constructor.  This is usually a bad idea." )
                raise RuntimeError( "Don't pass a session to the DataStore constructor." )
            return args[0], args[1]
        else:
            ds = DataStore()
            session = ds.parse_args(*args, **kwargs)
            if session is not None:
                SCLogger.error( "You passed a session to a DataStore constructor.  This is usually a bad idea." )
                raise RuntimeError( "Don't pass a session to the DataStore constructor." )
            return ds, session


    def parse_args(self, *args, **kwargs):
        """Parse the arguments to the DataStore constructor.

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
            - session: a sqlalchemy session object to use.  (Usually you do not want to give this!)

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
        if output_session is not None:
            raise RuntimeError( "You passed a session to DataStore.  Don't." )

        self.session = output_session

        # look for a user-given provenance tree
        provs = [ arg for arg in args
                  if isinstance(arg, dict) and all([isinstance(value, Provenance) for value in arg.values()])
                 ]
        if len(provs) > 0:
            self.prov_tree = provs[0]
            # also remove the provenances from the args list
            args = [ arg for arg in args
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
        self.inputs_str = "(unknown)"
        arg_types = [type(arg) for arg in args]
        if arg_types == []:   # no arguments, quietly skip
            self.inputs_str = "(no inputs)"
        elif ( ( arg_types == [ uuid.UUID, int ] ) or
               ( arg_types == [ uuid.UUID, str ] ) or
               ( arg_types == [ str, int ] ) or
               ( arg_types == [ str, str ] ) ):  #exposure_id, section_id
            self.exposure_id, self.section_id = args
            self.inputs_str = f"exposure_id={self.exposure_id}, section_id={self.section_id}"
        elif arg_types == [ Exposure, int ] or arg_types == [ Exposure, str ]:
            self.exposure, self.section_id = args
            self.inputs_str = f"exposure={self.exposure}, section_id={self.section_id}"
        elif ( arg_types == [ uuid.UUID ] ) or ( arg_types == [ str ] ):     # image_id
            self.image_id = args[0]
            self.inputs_str = f"image_id={self.image_id}"
        elif arg_types == [ Image ]:
            self.image = args[0]
            self.inputs_str = f"image={self.image}"
        # TODO: add more options here?
        #  example: get a string filename to parse a specific file on disk
        else:
            raise ValueError( f'Invalid arguments to DataStore constructor, got {arg_types}. '
                              f'Expected [<image_id>], or [<image>], or [<exposure id>, <section id>], '
                              f'or [<exposure>, <section id>]. ' )

        # parse the kwargs dict
        for key, val in kwargs.items():
            # The various setters will do type checking
            setattr( self, key, val )

        if output_session is not None:
            raise RuntimeError( "DataStore parse_args found a session.  "
                                "Don't pass sessions to DataStore constructors." )
        return output_session


    def __init__(self, *args, **kwargs):
        """Make a DataStore.

        See the parse_args method for details on how to initialize this object.

        Please make sure to add any new attributes to the products_to_save list.
        """
        # these are data products that can be cached in the store
        self._exposure = None  # single image, entire focal plane
        self._section = None  # SensorSection

        self.prov_tree = None  # provenance dictionary keyed on the process name

        # these all need to be added to the products_to_save list
        self._image = None  # single image from one sensor section
        self._sources = None  # extracted sources (a SourceList object, basically a catalog)
        self._psf = None  # psf determined from the extracted sources
        self._bg = None  # background from the extraction phase
        self._wcs = None  # astrometric solution
        self._zp = None  # photometric calibration
        self._reference = None  # the Reference object needed to make subtractions
        self._sub_image = None  # subtracted image
        self._detections = None  # a SourceList object for sources detected in the subtraction image
        self._cutouts = None  # cutouts around sources
        self._measurements = None  # photometry and other measurements for each source
        self._objects = None  # a list of Object associations of Measurements
        self._scores = None  # a list of r/b and ML/DL scores for Measurements

        # these need to be added to the products_to_clear list
        self.reference = None
        self._ref_image = None
        self._ref_sources = None
        self._ref_bg = None
        self._ref_psf = None
        self._ref_wcs = None
        self._ref_zp = None
        self.aligned_ref_image = None
        self.aligned_ref_sources = None
        self.aligned_ref_bg = None
        self.aligned_ref_psf = None
        self.aligned_ref_zp = None
        self.aligned_new_image = None
        self.aligned_new_sources = None
        self.aligned_new_bg = None
        self.aligned_new_psf = None
        self.aligned_new_zp = None
        self.aligned_wcs = None
        self._sub_image = None  # subtracted image
        self._reference = None  # the Reference object needed to make subtractions
        self._exposure_id = None  # use this and section_id to find the raw image
        self.section_id = None  # corresponds to SensorSection.identifier (*not* .id)
        self._image_id = None  # use this to specify an image already in the database

        self.warnings_list = None  # will be replaced by a list of warning objects in top_level.Pipeline.run()
        self.exceptions = []   # Stored list of exceptions
        self.runtimes = {}  # for each process step, the total runtime in seconds
        self.memory_usages = {}  # for each process step, the peak memory usage in MB
        self.products_committed = ''  # a comma separated list of object names (e.g., "image, sources") saved to DB
        self.report = None  # keep a reference to the report object for this run

        # The database session parsed in parse_args; it could still be None even after parse_args
        self.session = None
        self.parse_args(*args, **kwargs)


    def __setattr__(self, key, value):
        """Check some of the inputs before saving them.

        TODO : since we're only checking a couple of them, it might make sense to
        write specific handlers just for those instead of having every single attribute
        access of a DataStore have to make this function call.

        """

        if value is not None:
            if key in ['section_id'] and not isinstance(value, (int, str)):
                raise TypeError(f'{key} must be an integer or a string, got {type(value)}')

            # This is a tortured condition
            elif ( ( key == 'prov_tree' ) and
                 ( ( not isinstance(value, dict) ) or
                   ( not all( [ isinstance(v, Provenance) or
                                ( ( k == 'referencing' ) and
                                  ( isinstance( v, list ) and
                                    ( [ all( isinstance(i, Provenance) for i in v ) ] )
                                   )
                                 )
                                for k, v in value.items() ] )
                    )
                  )
                ):
                raise TypeError(f'prov_tree must be a dict of Provenance objects, got {value}')

            elif key == 'session' and not isinstance(value, sa.orm.session.Session):
                raise ValueError(f'Session must be a SQLAlchemy session or SmartSession, got {type(value)}')

        super().__setattr__(key, value)

    def update_report(self, process_step, session=None):
        """Update the report object with the latest results from a processing step that just finished. """
        if self.report is not None:
            self.report.scan_datastore( self, process_step=process_step )

    def finalize_report( self ):
        """Mark the report as successful and set the finish time."""
        if self.report is not None:
            self.report.scan_datastore( self, process_step='finalize' )
            self.report.success = True
            self.report.finish_time = datetime.datetime.now( datetime.UTC )
            self.report.upsert()


    def set_prov_tree( self, provdict, wipe_tree=False ):
        """Update the DataStore's provenance tree.

        Assumes that the passed provdict is self-consistent (i.e. the
        upstreams of downstream processes are the actual upstream
        processes in provdict).  Don't pass a provdict that doesn't fit
        this.  (NOTE: UPSTREAM_STEPS is a little deceptive when it comes
        to referencing and subtraction.  While 'referencing' is listed
        as an upstream to subtraction, in reality the subtraction
        provenance upstreams are supposed to be the upstreams of the
        referencing provenance (plus the preprocessing and extraction
        provenances), not the referencing provenance itself.)

        Will set any provenances downstream of provenances in provdict
        to None (to keep the prov_tree self-consistent).  (Of course, if
        there are multiple provenances in provdict, and one is
        downstream of another, the first one will not not be None after
        this function runs, it will be what was passed in provdict.

        Parameters
        ----------
           provdict: dictionary of process: Provenance
              Each key of the dictionary must be one of the keys in
              UPSTREAM_STEPS ('exposure', 'preprocessing','extractin',
              'referencing', 'subtraction', 'detection', 'cutting').

           wipe_tree: bool, default False
              If True, will wipe out the provenance tree before setting
              the provenances for the processes in provdict.
              Otherwise, will only wipe out provenances downstream from
              the provenances in provdict.

        """

        if wipe_tree:
            self.prov_tree = None

        givenkeys = list( provdict.keys() )
        # Sort according to UPSTREAM_STEPS
        givenkeys.sort( key=lambda x: list( UPSTREAM_STEPS.keys() ).index( x ) )

        for process in givenkeys:
            if self.prov_tree is None:
                self.prov_tree = { process: provdict[ process ] }
            else:
                self.prov_tree[ process ] = provdict[ process ]
                # Have to wipe out all downstream provenances, because
                # they will change!  (There will be a bunch of redundant
                # work here if multiple provenances are passed in
                # provdict, but, whatever, it's quick enough, and this
                # will never be called in an inner loop.)
                mustwipe = set( [ k for k,v in UPSTREAM_STEPS.items() if process in v ] )
                while len( mustwipe ) > 0:
                    for towipe in mustwipe:
                        if towipe in self.prov_tree:
                            del self.prov_tree[ towipe ]
                    mustwipe = set( [ k for k,v in UPSTREAM_STEPS.items() if towipe in v ] )


    def get_provenance(self, process, pars_dict, session=None,
                       pars_not_match_prov_tree_pars=False,
                       replace_tree=False ):
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
            A dictionary of parameters used for the process.  These
            include the critical parameters for this process.  Use a
            Parameter object's get_critical_pars() if you are setting
            this; otherwise, the provenance will be wrong, as it may
            include things in parameters that aren't supposed to be
            there.

            WARNING : be careful creating an extraction provenance.
            The pars_dict there is more complicated because of
            siblings.

        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, new sessions
            will be opened and closed as necessary.

        pars_not_match_prov_tree_pars: bool, default False
            If you're consciously asking for a provenance with
            parameters that you know won't match the provenance in the
            DataStore's provenance tree, set this to True.  Otherwise,
            an exception will be raised if you ask for a provenance
            that's inconsistent with one in the prov_tree.

        replace_tree: bool, default False
            Replace whatever's in the provenance tree with the newly
            generated provenance.  This requires upstream provenances to
            exist-- either in the prov tree, or in upstream objects
            already saved to the data store.  It (effectively) implies
            pars_not_match_prov_tree_pars.

        Returns
        -------
        prov: Provenance
            The provenance for the given process.

        """

        # First, check the provenance tree:
        prov_found_in_tree = None
        if ( not replace_tree ) and ( self.prov_tree is not None ) and ( process in self.prov_tree ):
            if self.prov_tree[ process ].parameters != pars_dict:
                if not pars_not_match_prov_tree_pars:
                    raise ValueError( f"DataStore getting provenance for {process} whose parameters "
                                      f"don't match the parameters of the same process in the prov_tree" )
            else:
                prov_found_in_tree = self.prov_tree[ process ]

        if prov_found_in_tree is not None:
            return prov_found_in_tree

        # If that fails, see if we can make one

        session = self.session if session is None else session

        code_version = Provenance.get_code_version(session=session)
        if code_version is None:
            raise RuntimeError( "No code_version in the database, can't make a Provenance" )

        # check if we can find the upstream provenances
        upstreams = []
        for name in UPSTREAM_STEPS[process]:
            prov = None
            if ( self.prov_tree is not None ) and ( name in self.prov_tree ):
                # first try to load an upstream that was given explicitly:
                prov = self.prov_tree[name]
            else:
                # if that fails, see if the correct object exists in memory
                obj_names = PROCESS_PRODUCTS[name]
                if isinstance(obj_names, str):
                    obj_names = [obj_names]
                obj = getattr(self, obj_names[0], None)  # only need one object to get the provenance
                if isinstance(obj, list):
                    obj = obj[0]  # for cutouts or measurements just use the first one
                if ( obj is not None ) and ( obj.provenance_id is not None ):
                    prov = Provenance.get( obj.provenance_id, session=session )

            if prov is not None:  # if we don't find one of the upstreams, it will raise an exception
                upstreams.append(prov)

        if len(upstreams) != len(UPSTREAM_STEPS[process]):
            raise ValueError(f'Could not find all upstream provenances for process {process}.')

        # check if "referencing" is in the list, if so, replace it with its upstreams
        # (Reason: referencing is upstream of subtractions, but subtraction upstreams
        # are *not* the Reference entry, but rather the images that went into the subtractions.)
        for u in upstreams:
            if u.process == 'referencing':
                upstreams.remove(u)
                for up in u.upstreams:
                    upstreams.append(up)

        # we have a code version object and upstreams, we can make a provenance
        prov = Provenance(
            process=process,
            code_version_id=code_version.id,
            parameters=pars_dict,
            upstreams=upstreams,
            is_testing="test_parameter" in pars_dict,  # this is a flag for testing purposes
        )
        prov.insert_if_needed( session=session )

        if replace_tree:
            self.set_prov_tree( { process: prov }, wipe_tree=False )

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
        raise RuntimeError( "Deprecated; just look in prov_tree" )

        # # see if it is in the prov_tree
        # if self.prov_tree is not None:
        #     if process in self.prov_tree:
        #         return self.prov_tree[process]
        #     else:
        #         raise ValueError(f'No provenance found for process "{process}" in prov_tree!')

        # return None  # if not found in prov_tree, just return None

    def get_raw_exposure(self, session=None):
        """Get the raw exposure from the database."""
        if self._exposure is None:
            if self.exposure_id is None:
                raise ValueError('Cannot get raw exposure without an exposure_id!')

            with SmartSession(session, self.session) as session:
                self.exposure = session.scalars(sa.select(Exposure).where(Exposure._id == self.exposure_id)).first()

        return self._exposure

    def get_image( self, provenance=None, reload=False, session=None ):
        """Get the pre-processed (or coadded) image, either from memory or from the database.

        If the store is initialized with an image or an image_id, that
        image is returned, no matter the provenances or the local
        parameters.  This is the only way to ask for a coadd image.  If
        an image with such an id is not found, in memory or in the
        database, will raise a ValueError.

        If exposure_id and section_id are given, will load an image that
        is consistent with that exposure and section ids, futher qualified by:
          * with provenance matching the passed provenance, if provided, else:
          * with provenance matching the 'preprocessing' provenance in self.prov.tree,
            or an exception if there is no suchy thing in prov_tree.
        Will return None if there is no match.

        Note that this also updates self.image with the found image (or None).

        Parameters
        ----------
        provenance: Provenance object, or None
            The provenance to use for the image.  This provenance should
            be consistent with the current code version and critical
            parameters.  If None, will get the 'preprocessing' provenance
            from self.prov_tree.

        reload: bool, default False
            If True, ignore the image saved in the Datastore and reload
            the image from the databse using either the image_id (if one
            is available) or the exposure, section_id, and
            'preprocessing' provenance.

        session: sqlalchemy.orm.session.Session
            An optional session to use for the database query.
            If not given, will use the session stored inside the
            DataStore object; if there is none, will open a new session
            and close it when done with it.

        Returns
        -------
        image: Image object
            The image object, or None if no matching image is found.

        """
        session = self.session if session is None else session

        # See if we have the image

        if reload:
            self.image = None

        if self.image is not None:
            return self.image

        if self.image_id is not None:
            self.image = Image.get_by_id( self.image_id, session=session )
            if self.image is None:
                raise RuntimeError( f"Failed to load image {self.image_id}" )
            return self.image

        if ( self.exposure is None ) or ( self.section is None ):
            raise ValueError( "Cannot get image without either (exposure, section) or (image) or (image_id )" )

        # We don't have the image yet, try to get it based on exposure and section

        if provenance is None:
            if 'preprocessing' not in self.prov_tree:
                raise RuntimeError( "Can't get an image without a provenance; there is no preprocessing "
                                    "provenance in the DataStore's provenance tree." )
            provenance = self.prov_tree[ 'preprocessing' ]

        with SmartSession( session ) as sess:
            self.image = ( sess.query( Image )
                           .filter( Image.exposure_id == self.exposure_id )
                           .filter( Image.section_id == str(self.section_id) )
                           .filter( Image.provenance_id == provenance._id )
                          ).first()

        # Will return None if no image was found in the search
        return self.image

    def _get_data_product( self,
                           att,
                           cls,
                           upstream_att,
                           cls_upstream_id_att,
                           process,
                           is_list=False,
                           upstream_is_list=False,
                           match_prov=True,
                           provenance=None,
                           reload=False,
                           session=None ):
        """Get a data product (e.g. sources, detections, etc.).

        First sees if the data product is already in the DataStore.  If
        so, returns it, without worrying about provenance.

        If it's not there, gets the upstream data product first.
        Searches the database for an object whose upstream matches, and
        whose provenance matches.  Provenance is set from the provenance
        tree for the appropriate process if it is not passed explicitly.

        Returns an object or None (if is_list is False), or a
        (potentially empty) list of objects if is_list is True.

        Also updates the self.{att} property.

        Parameters
        ----------
          att: str
            The attribute of the DataStore we're trying to get (sources, psf, wcs, bg, cutouts, etc.)

          cls: class
            The class associated with att (Sources, PSF, WorldCoordinates, etc.)

          upstream_att: str
            The name of the attribute of the DataStore that represents the upstream product.

          cls_upstream_id_att: THING The actual attribute from the
            class that holds the id of the upstream. E.g., if
            att="sources" and cls=SourceList, then
            upstream_att="image_id" and att=SourceList.image_id

          process: str
            The name of the process that produces this data product ('extraction', 'detection', ';measuring', etc.)

          is_list: bool, default False
            True if a list is expected (which currently is only for measurements).

          upstream_is_list: bool, default False
            True if the attribute represented by upstream_att is a list (eg measurements)

          match_prov: bool, default True
            True if the provenance must match.  (For some things,
            i.e. the SourceList siblings, it's a 1:1 relationship, so
            there's no need to match provenance.)

          provenance: Provenance or None
            The provenance of the data product.  If this isn't passed,
            will look in the provenance tree for a provenance of the
            indicated process.  If there's nothing there, and the data
            product isn't already in the DataStore, it's an error.

          reload: bool, default False
            Igonore an existing data product if one is already in the
            DataStore, and always reload it from the database using the
            parent products and the provenance.

          session: SQLAlchemy session or None
            If not passed, may make and close a sesion.

        """
        # First, see if we already have one
        if hasattr( self, att ):
            if reload:
                setattr( self, att, None )
            else:
                obj = getattr( self, att )
                if obj is not None:
                    return obj
        else:
            raise RuntimeError( f"DataStore has no {att} attribute." )

        # If not, find it in the database

        if match_prov and ( provenance is None ):
            if ( self.prov_tree is None ) or ( process not in self.prov_tree ):
                raise RuntimeError( f"DataStore: can't get {att}, no provenance, and provenance not in prov_tree" )
            provenance = self.prov_tree[ process ]

        upstreamobj = getattr( self, upstream_att )
        if upstreamobj is None:
            getattr( self, f'get_{upstream_att}' )( session=session )
            upstreamobj = getattr( self, upstream_att )
        if upstreamobj is None:
            # It's not obvious to me if we should return None, or if we should
            #  be raising an exception here.  Some places in the code assume
            #  it will just be None, so that's what it is.
            # raise RuntimeError( f"Datastore can't get a {att}, it isn't able to get the parent {upstream_att}" )
            setattr( self, att, None )
            return None

        if not upstream_is_list:
            with SmartSession( session ) as sess:
                obj = sess.query( cls ).filter( cls_upstream_id_att == upstreamobj._id )
                if ( match_prov ):
                    obj = obj.filter( cls.provenance_id == provenance._id )
                obj = obj.all()

        else: # should only be scoring atm
            upstream_ids = [obj.id for obj in upstreamobj]
            with SmartSession( session ) as sess:
                obj = sess.query( cls ).filter( cls_upstream_id_att.in_( upstream_ids ) )
                if ( match_prov ):
                    obj = obj.filter( cls.provenance_id == provenance._id )
                obj = obj.all()

        if is_list:
            setattr( self, att, None if len(obj)==0 else list(obj) )
        else:
            if len( obj ) > 1:
                raise RuntimeError( f"DataStore found multiple matching {cls.__name__} and shouldn't have" )
            elif len( obj ) == 0:
                setattr( self, att, None )
            else:
                setattr( self, att, obj[0] )

        return getattr( self, att )



    def get_sources(self, provenance=None, reload=False, session=None):
        """Get the source list, either from memory or from database.

        If there is already a sources will return that one, or raise an
        error if its provenance doesn't match what's expected.
        (Expected provenance is defined by the provenance parameter if
        its passed, otherwise the 'extraction' provenance in
        self.prov_tree, otherwise anything with the image provenance in
        its upstreams.)

        Otherwise, will try to get the image (with get_image), and will
        Try to use the image_id and the provenance to find one in the
        database.  Returns None if none is found.

        Updates self.sources and self.sources_id as it loads new things.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use to get the source list.  This
            provenance should be consistent with the current code
            version and critical parameters.  If none is given, uses the
            appropriate provenance from the prov_tree dictionary.  If
            prov_tree is None, then that's an error.

        reload: bool, default False
            If True, ignore any .sources already present and reload the
            sources from the databse using the image and the
            'extraction' provenance.

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

        return self._get_data_product( "sources", SourceList, "image", SourceList.image_id, "extraction",
                                       provenance=provenance, reload=reload, session=session )

    def get_psf(self, session=None, reload=False, provenance=None):
        """Get a PSF, either from memory or from the database."""
        return self._get_data_product( 'psf', PSF, 'sources', PSF.sources_id, 'extraction',
                                       match_prov=False, provenance=provenance, reload=reload, session=session )

    def get_background(self, session=None, reload=False, provenance=None):
        """Get a Background object, either from memory or from the database."""
        return self._get_data_product( 'bg', Background, 'sources', Background.sources_id, 'extraction',
                                       match_prov=False, provenance=provenance, reload=reload, session=session )

    def get_wcs(self, session=None, reload=False, provenance=None):
        """Get an astrometric solution in the form of a WorldCoordinates object, from memory or from the database."""
        return self._get_data_product( 'wcs', WorldCoordinates, 'sources', WorldCoordinates.sources_id, 'extraction',
                                       match_prov=False, provenance=provenance, reload=reload, session=session )

    def get_zp(self, session=None, reload=False, provenance=None):
        """Get a zeropoint as a ZeroPoint object, from memory or from the database."""
        return self._get_data_product( 'zp', ZeroPoint, 'sources', ZeroPoint.sources_id, 'extraction',
                                       match_prov=False, provenance=provenance, reload=reload, session=session )


    def get_reference(self,
                      search_by='image',
                      provenances=None,
                      match_instrument=True,
                      match_filter=True,
                      min_overlap=0.85,
                      skip_bad=True,
                      reload=False,
                      multiple_ok=False,
                      randomly_pick_if_multiple=False,
                      session=None ):
        """Get the reference for this image.

        Also sets the self.reference property.

        Parameters
        ----------
        search_by: str, default 'image'
            One of 'image', 'ra/dec', or 'target/section'.  If 'image',
            will pass the DataStore's image to
            Reference.get_references(), which will find references that
            overlap the area of the image by at least min_overlap.  If
            'ra/dec', will pass the central ra/dec of the image to
            Reference.get_references(), and then post-filter them by
            overlapfrac (if that is not None).  If 'target/section',
            will pass target and section_id of the image to
            Reference.get_references().  You almost always want to use
            the default of 'image', unles you're working with a survey
            that has very well-defined targets and the image headers are
            always completely reliable; in that case, use
            'target/section'.  'ra/dec' might be useful if you're doing
            forced photometry and the image is a targeted image with the
            target right at the center of the image (which is probably a
            fairly contrived situation, though you may have created
            subset images constructed that way).

        provenances: list of Provenance objects, or None
            A list of provenances to use to identify a reference.  Any
            found references must have one of these provenances.  If not
            given, will try to get the provenances from the prov_tree
            attribute.  If it can't find them there and provenance isn't
            given, raise an exception.

        match_filter: bool, default True
            If True, only find a reference whose filter matches the
            DataStore's image's filter.

        match_instrument: bool, default True
            If True, only find a refernce whose instrument matches the
            Datastore's images' instrument.

        min_overlap: float or None, default 0.85
            Area of overlap region must be at least this fraction of the
            area of the search image for the reference to be good.  Make
            this None to not consider overlap fraction when finding a
            reference.  (Sort of; it will still return the one with the
            higehst overlap, it's just it will return that one even if
            the overlap is tiny.)

        skip_bad: bool, default True
            If True, will skip references that are marked as bad.

        reload: bool, default False
            If True, set the self.reference property (as well as derived
            things like ref_image, ref_sources, etc.) to None and try to
            re-acquire the reference from the databse.  (Normally, if
            there already is a self.reference and it matches all the
            other criteria, it will just be returned.)

        multiple_ok: bool, default False
            Ignored for 'ra/dec' and 'target/section' search, or if
            min_overlap is None or <=0.  For 'image' search, normally,
            if more the one matching reference is found, it will return
            an error.  If this is True, then it will pick the reference
            with the highest overlap (depending on
            randomly_pick_if_multiple).

        randomly_pick_if_multiple: bool, default False
            Normally, if there multiple references with exactly the same
            maximum overlap fraction with the DataStore's image (which
            should be _very_ rare), an exception will be raised.  If
            randomly_pick_if_multiple is True, the code will not raise
            an exception, and will just return whichever one the
            database and code happend to sort first (which is
            non-deterministic).

        session: sqlalchemy.orm.session.Session or SmartSession
            An optional session to use for the database query.  If not
            given, then functions called by this function will open and
            close sessions as necessary.

        Returns
        -------
        ref: Image object
            The reference image for this image, or None if no reference is found.

        Behavior when more than one reference is found:

        * For search_by='image':
            * If multiple_ok=True or min_overlap is None or <=0, return
              the reference with the highest overlap fraction with the
              DataStore's image.

            * If multiple_ok=False and min_overlap is positive, raise an
              exception.

        * Otherwise:
            * Return the refrence with the highest overlap fraction with
              the DataStore's image.

        * Special case for both of the above: if there are multiple
          images and, by unlikely chance, there are more than one that
          have exactly the same highest overlap fraction, then raise an
          exception if randomly_pick_if_multiple is False, otherwise
          pick whichever one the database and code happened to sort
          first.

        """

        if reload:
            self.reference = None
            self._ref_image = None
            self._ref_sources = None
            self._ref_bg = None
            self._ref_psf = None
            self._ref_wcs = None
            self._ref_zp = None
            self.sub_image = None

        image = self.get_image(session=session)
        if image is None:
            return None  # cannot find a reference without a new image to match

        if provenances is None:  # try to get it from the prov_tree
            if ( self.prov_tree is not None ) and ( 'referencing' in self.prov_tree ):
                provenances = self.prov_tree[ 'referencing' ]

        provenances = listify(provenances)

        if ( provenances is None ) or ( len(provenances) == 0 ):
            raise RuntimeError( "DataStore can't get a reference, no provenances to search" )

        provenance_ids = [ p.id for p in provenances ]

        # first, some checks to see if existing reference is ok
        if self.reference is not None:
            if self.reference.provenance_id not in provenance_ids:
                self.reference = None

            elif skip_bad and ( self.reference.bitflag != 0 ):
                self.reference = None

            elif match_filter and self.reference.image.filter != image.filter:
                self.reference = None

            elif match_instrument and self.reference.image.instrument != image.instrument:
                self.reference = None

            elif ( ( search_by in [ 'target/section', 'target/section_id' ] ) and
                   ( ( self.reference.imagetarget != image.target ) or
                     ( self.reference.imagesection_id != image.section_id ) )
                  ):
                self.reference = None

            elif ( min_overlap is not None ) and ( min_overlap > 0 ):
                refimg = Image.get_by_id( self.reference.image_id )
                ovfrac = FourCorners.get_overlap_frac(image, refimg)
                if ovfrac < min_overlap:
                    self.reference = None

            # if we have survived this long without losing the reference, can return it here:
            if self.reference is not None:
                return self.reference

        # No reference was found (or it didn't match other parameters) must find a new one
        # First, clear out all data products that are downstream of reference.
        # (Setting sub_image will cascade to detections, cutouts, measurements.)

        self._ref_image = None
        self._ref_sources = None
        self._ref_bg = None
        self._ref_psf = None
        self._ref_wcs = None
        self._ref_zp = None
        self.sub_image = None

        arguments = {}
        if search_by == 'image':
            arguments['image'] = image
            arguments['overlapfrac'] = min_overlap
        elif search_by == 'ra/dec':
            arguments['ra'] = image.ra
            arguments['dec'] = image.dec
        elif search_by in [ 'target/section', 'target/section_id' ]:
            arguments['target'] = image.target
            arguments['section_id'] = image.section_id

        if match_filter:
            arguments['filter'] = image.filter

        if match_instrument:
            arguments['instrument'] = image.instrument

        if skip_bad:
            arguments['skip_bad'] = True

        arguments['provenance_ids'] = provenance_ids

        # SCLogger.debug( f"DataStore calling Reference.get_references with arguments={arguments}" )

        refs, imgs = Reference.get_references( **arguments, session=session )
        if len(refs) == 0:
            # SCLogger.debug( f"DataStore: Reference.get_references returned nothing." )
            self.reference = None
            return None

        elif len(refs) == 1:
            # One reference found.  Return it if it's OK.
            self.reference = refs[0]

            # For image search, Reference.get_references() will
            #  already have filtered by min_overlap if relevant.
            if search_by != 'image':
                if ( ( min_overlap is not None ) and
                     ( min_overlap > 0 ) and
                     ( FourCorners.get_overlap_frac( image, imgs[0] ) < min_overlap )
                    ):
                    self.reference = None

            return self.reference

        else:
            # Multiple references found; deal with it.

            # Sort references by overlap fraction descending
            ovfrac = [ FourCorners.get_overlap_frac( image, i ) for i in imgs ]
            sortdex = list( range( len(refs) ) )
            sortdex.sort( key=lambda x: -ovfrac[x] )

            if search_by == 'image':
                # For image search, raise an exception if multiple_ok is
                #   False, as Reference.get_references() will already
                #   have thrown out things with ovfrac < min_overlap.
                #   If multiple_ok is True, or if we didn't give a
                #   min_overlap, then return the one with the highest
                #   overlap, except in the
                #   randomly_pick_if_multiple=False edge case.
                if ( not multiple_ok ) and ( min_overlap is not None ) and ( min_overlap > 0 ):
                    self.reference = None
                    raise RuntimeError( f"More than one reference overlapped the image by at least {min_overlap}" )
                if ( not randomly_pick_if_multiple ) and ( ovfrac[sortdex[0]] == ovfrac[sortdex[1]] ):
                    self.reference = None
                    raise RuntimeError( f"More than one reference had exactly the same overlap of "
                                        f"{ovfrac[sortdex[0]]}" )
                self.reference = refs[ sortdex[0] ]
                return self.reference

            else:
                # For ra/dec or target/section search,
                # References.get_reference() will not have filtered by
                # min_overlap, so do that here.
                if ( min_overlap is not None ) and ( min_overlap > 0 ):
                    sortdex = [ s for s in sortdex if ovfrac[s] >= min_overlap ]
                    if len(sortdex) == 0:
                        self.reference = None
                        return self.reference
                    # Edge case
                    if ( ( len(sortdex) > 1 ) and
                         ( not randomly_pick_if_multiple ) and
                         ( ovfrac[sortdex[0]] == ovfrac[sortdex[1]] )
                        ):
                        self.reference = None
                        raise RuntimeError( f"More than one reference had exactly the same overlap of "
                                            f"{ovfrac[sortdex[0]]}" )
                    # Return the one with highest overlap
                    self.reference = refs[ sortdex[0] ]
                    return self.reference
                else:
                    # We can just return the one with highest overlap, even if it's tiny, because we
                    #   didn't ask to filter on min_overlap, except in the edge case
                    if ( ( len(sortdex) > 1 ) and
                         ( not randomly_pick_if_multiple ) and
                         ( ovfrac[sortdex[0]] == ovfrac[sortdex[1]] )
                        ):
                        self.reference = None
                        raise RuntimeError( f"More than one reference had exactly the same overlap of "
                                            f"{ovfrac[sortdex[0]]}" )
                    self.reference = refs[ sortdex[0] ]
                    return self.reference

        raise RuntimeError( "The code should never get to this line." )


    def get_subtraction(self, provenance=None, reload=False, session=None):
        """Get a subtraction Image, either from memory or from database.

        If sub_image is not None, return that.  Otherwise, if
        self.reference is None, raise an exception.  Otherwise, use
        self.get_image() to get the image, and find the subtraction
        image that has the self.image as its new, self.ref_image as its
        ref, and the right provenance.

        Updates sub_image.

        Parameters
        ----------
        provenance: Provenance object
            The provenance to use for the subtraction.  This provenance
            should be consistent with the current code version and
            critical parameters.  If None, then gets the "subtraction"
            provenance from the provenance tree, raising an exception if
            one isn't found.

        reload: bool, default False
            Set .sub_image to None, and always try to reload from the database.

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

        # This one has a more complicated query, so
        #  we can't just use _get_data_product

        if reload:
            self.sub_image = None

        # First see if we already have one
        if self.sub_image is not None:
            return self.sub_image

        # If not, look for it in the database

        if provenance is None:
            if 'subtraction' not in self.prov_tree:
                raise RuntimeError( "Can't get a subtraction without a provenance; there's no subtraction "
                                    "provenance in the DataStore's provenance tree." )
            provenance = self.prov_tree[ 'subtraction' ]

        if self.reference is None:
            # We could do the call here, but there are so many configurable parameters to
            #   get_reference() that it's safer to make the user do it
            raise RuntimeError( "Can't get a subtraction without a reference; call get_reference" )

        # Really, sources and its siblings ought to be loaded too, but we don't strictly need it
        #   for the search.

        with SmartSession( session ) as sess:
            if self.image_id is None:
                self.get_image( session=sess )
            if self.image_id is None:
                raise RuntimeError( "Can't get sub_image, don't have an image_id" )

            imgs = ( sess.query( Image )
                     .filter( Image.provenance_id==provenance.id )
                     .filter( Image.new_image_id==self.image.id )
                     .filter( Image.ref_id==self.reference.id )
                     .filter( Image.is_sub ) ).all()
            if len(imgs) > 1:
                raise RuntimeError( "Found more than one matching sub_image in the database!  This shouldn't happen!" )
            if len(imgs) == 0:
                self.sub_image = None
            else:
                self.sub_image = imgs[0]

        return self.sub_image

    def get_detections(self, provenance=None, reload=False, session=None):
        """Get a SourceList for sources from the subtraction image, from memory or from database."""
        return self._get_data_product( "detections", SourceList, "sub_image", SourceList.image_id, "detection",
                                       provenance=provenance, reload=reload, session=session )

    def get_cutouts(self, provenance=None, reload=False, session=None):
        """Get a list of Cutouts, either from memory or from database."""
        return self._get_data_product( "cutouts", Cutouts, "detections", Cutouts.sources_id, "cutting",
                                       provenance=provenance, reload=reload, session=session )


    def get_measurements(self, provenance=None, reload=False, session=None):
        """Get a list of Measurements, either from memory or from database."""
        return self._get_data_product( "measurements", Measurements, "cutouts", Measurements.cutouts_id, "measuring",
                                       is_list=True, provenance=provenance, reload=reload, session=session )

    def get_scores(self, provenance=None, reload=False, session=None):
        """Get a list of DeepScores, either from memory or from database"""
        scores =  self._get_data_product( "scores", DeepScore, "measurements", DeepScore.measurements_id,
                                      "scoring", is_list=True, upstream_is_list=True,
                                      provenance=provenance, reload=reload, session=session)

        # sort the scores so the list order matches measurements
        if scores is not None and len(scores) > 0:

            if len(scores) != len(self.measurements):
                raise RuntimeError(f"get_scores found {len(scores)} scores for {len(self.measurements)} measurements")

            m_ids = [str(m.id) for m in self.measurements]
            scores.sort( key=lambda x: m_ids.index( str(x.measurements_id) ) )

        return scores


    def get_all_data_products(self, output='dict', omit_exposure=False):
        """Get all the data products associated with this Exposure.

        Does *not* try to load missing ones from the databse.  Just
        returns what the DataStore already knows about.  (Use
        load_all_data_products to load missing ones from the database.)

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
        attributes.extend( [ 'image', 'sources', 'psf', 'bg', 'wcs', 'zp', 'sub_image',
                             'detections', 'cutouts', 'measurements', 'scores' ] )
        result = {att: getattr(self, att) for att in attributes}
        if output == 'dict':
            return result
        if output == 'list':
            return [result[att] for att in attributes if result[att] is not None]
        else:
            raise ValueError(f'Unknown output format: {output}')

    def load_all_data_products( self, reload=False, omit_exposure=False ):
        """Load all of the data products that exist on the database into DataStore attributes.

        Will return existing ones, or try to load them from the database
        using the provenance in self.prov_tree and the parent objects.
        If reload is True, will set the attribute to None and always try
        to reload from the database.

        If omit_exposure is True, will not touch the self.exposure
        attribute.  Otherwise, will try to load it based on
        self.exposure_id.

        """

        if not omit_exposure:
            if reload:
                self.exposure = None
            if self.exposure is None:
                if self.exposure_id is not None:
                    if self.section_id is None:
                        raise RuntimeError( "DataStore has exposure_id but not section_id, I am surprised." )
                    self.exposure = Exposure.get_by_id( self.exposure_id )

        self.get_image( reload=reload )
        self.get_sources( reload=reload )
        self.get_psf( reload=reload )
        self.get_background( reload=reload )
        self.get_wcs( reload=reload )
        self.get_zp( reload=reload )
        self.get_reference( reload=reload )
        self.get_subtraction( reload=reload )
        self.get_detections( reload=reload )
        self.get_cutouts( reload=reload )
        self.get_measurements( reload=reload )


    def save_and_commit(self,
                        exists_ok=False,
                        overwrite=True,
                        no_archive=False,
                        update_image_header=False,
                        update_image_record=True,
                        force_save_everything=False,
                        session=None):
        """Go over all the data products, saving them to disk if necessary, saving them to the database as necessary.

        In general, it will *not* save data products that have a
        non-null md5sum (or md5sum_components) line in the database.
        Reason: once that line is written, it means that that data
        product is "done" and will not change again.  As such, this
        routine assumes that it's all happily saved at least to the
        archive, so nothing needs to be written.

        There is one exception: the "image" (as opposed to weight or
        flags) component of an Image.  If "update_image_header" is true,
        then the DataStore will save and overwrite just the image
        component (not the weight or flags components) both to disk and
        to the archive, and will update the database md5sum line
        accordingly.  The *only* change that should have been made to
        the image file is in the header; the WCS and zeropoint keywords
        will have been updated.  The pipeline that uses the DataStore
        will be set to only do this once for each step (once the
        astro_cal_done and TODO_PHOTOMETRIC fields change from False to
        True), as the image headers get "first-look" values, not
        necessarily the latest and greatest if we tune either process.

        It will run an upsert on the database record for all data
        products.  This means that if the object is not in the databse,
        it will get added.  (In this case, the object is then reloaded
        back from the database, so that the database-default fields will
        be filled.)  If it already is in the database, its fields will
        be updated with what's in the objects in the DataStore.  Most of
        the time, this should be a null operation, as if we're not
        inserting, we have all the fields that were already loaded.
        However, it does matter for self.image, as some fields (such as
        background level, fwhm, zp) get set during processes that happen
        after the image's record in the database is first created.

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
            (and overwrite is False).  This parameter is ignored for
            data products that already have a md5sum unless
            force_save_everything is True.

        overwrite: bool, default True
            If True, will overwrite any existing files on disk.

        no_archive: bool, default False
            If True, will not push files up to the archive, will only
            save on local disk.

        update_image_header: bool, default False
            See above.  If this is true, then the if there is an Image
            object in the data store, its "image" component will be
            overwritten both on the local store and on the archive, and
            appropriate entry in the md5sum_components array of the
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

            strio = io.StringIO()
            strio.write( f"save_and_commit of {att} considering a {obj.__class__.__name__}" )
            if isinstance( obj, FileOnDiskMixin ):
                strio.write( f" with filepath {obj.filepath}" )
            elif isinstance( obj, list ):
                strio.write( f" including types {set([type(i) for i in obj])}" )
            SCLogger.debug( strio.getvalue() )

            if isinstance(obj, FileOnDiskMixin):
                mustsave = True
                # TODO : if some components have a None md5sum and others don't,
                #  right now we'll re-save everything.  Improve this to only
                #  save the necessary components.  (In practice, this should
                #  hardly ever come up.)
                if ( ( not force_save_everything )
                     and
                     ( ( obj.md5sum is not None )
                       or ( ( obj.md5sum_components is not None )
                            and
                            ( all( [ i is not None for i in obj.md5sum_components ] ) )
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
                        SCLogger.debug( f"save_and_commit saving a {obj.__class__.__name__}" )
                        SCLogger.debug( f"self.image={self.image}" )
                        SCLogger.debug( f"self.sources={self.sources}" )
                        basicargs = { 'overwrite': overwrite, 'exists_ok': exists_ok, 'no_archive': no_archive }
                        # Various things need other things to invent their filepath
                        if att == "sources":
                            obj.save( image=self.image, **basicargs )
                        elif att in [ "psf", "bg", "wcs" ]:
                            obj.save( image=self.image, sources=self.sources, **basicargs )
                        elif att == "detections":
                            obj.save( image=self.sub_image, **basicargs )
                        elif att == "cutouts":
                            obj.save( image=self.sub_image, sources=self.detections, **basicargs )
                        else:
                            obj.save( overwrite=overwrite, exists_ok=exists_ok, no_archive=no_archive )
                    except Exception as ex:
                        SCLogger.error( f"Failed to save a {obj.__class__.__name__}: {ex}" )
                        raise ex

                else:
                    SCLogger.debug( f'Not saving the {obj.__class__.__name__} because it already has '
                                    f'a md5sum in the database' )

        # Save all the data products.  Cascade our way down so that we can
        #   set upstream ids as necessary.  (Many of these will already have been
        #   set/saved before.)

        commits = []

        # Exposure
        # THINK.  Should we actually upsert this?
        # Almost certainly it hasn't changed, and
        # it was probably already in the database
        # anyway.
        if self.exposure is not None:
            SCLogger.debug( "save_and_commit upserting exposure" )
            self.exposure.upsert( load_defaults=True )
            # commits.append( 'exposure' )
            # exposure isn't in the commit bitflag

        # Image
        if self.image is not None:
            if self.exposure is not None:
                self.image.exposure_id = self.exposure.id
            SCLogger.debug( "save_and_commit upserting image" )
            self.image.upsert( load_defaults=True )
            commits.append( 'image' )

        # SourceList
        if self.sources is not None:
            if self.image is not None:
                self.sources.image_id = self.image.id
            SCLogger.debug( "save_and_commit upserting sources" )
            self.sources.upsert( load_defaults=True )
            commits.append( 'sources' )

        # SourceList siblings
        for att in [ 'psf', 'bg', 'wcs', 'zp' ]:
            if getattr( self, att ) is not None:
                if self.sources is not None:
                    setattr( getattr( self, att ), 'sources_id', self.sources.id )
                SCLogger.debug( f"save_and_commit upserting {att}" )
                getattr( self, att ).upsert( load_defaults=True )
                commits.append( att )

        # subtraction Image
        if self.sub_image is not None:
            self.sub_image.upsert( load_defaults=True )
            SCLogger.debug( "save_and_commit upserting sub_image" )
            commits.append( 'sub_image' )

        # detections
        if self.detections is not None:
            if self.sub_image is not None:
                self.detections.sources_id = self.sub_image.id
            SCLogger.debug( "save_and_commit detections" )
            self.detections.upsert( load_defaults=True )
            commits.append( 'detections' )

        # cutouts
        if self.cutouts is not None:
            if self.detections is not None:
                self.cutouts.detections_id = self.detections.id
            SCLogger.debug( "save_and_commit upserting cutouts" )
            self.cutouts.upsert( load_defaults=True )
            commits.append( 'cutouts' )

        # track which score goes with which measurement
        if ( ( ( self.measurements is not None ) and ( len(self.measurements) > 0 ) ) and
             ( ( self.scores is not None ) and ( len(self.scores) > 0 ) ) ):

            # make sure there is one score per measurement
            if ( len( self.scores ) != len(self.measurements) ):
                raise ValueError("Score and measurements list not the same length")

            sm_index_list = []
            m_ids = [str(m.id) for m in self.measurements]
            for i, s in enumerate(self.scores):
                if str(s.measurements_id) not in m_ids:
                    raise ValueError("score points to nonexistant measurement")
                sm_index_list.append(m_ids.index(str(s.measurements_id)))

        # measurements
        if ( self.measurements is not None ) and ( len(self.measurements) > 0 ):
            if self.cutouts is not None:
                for m in self.measurements:
                    m.cutouts_id = self.cutouts.id
            Measurements.upsert_list( self.measurements, load_defaults=True )
            SCLogger.debug( "save_and_commit measurements" )
            commits.append( 'measurements' )

        # scores
        if ( self.scores is not None ) and ( len(self.scores) > 0 ):
            if ( self.measurements is not None ) and ( len(self.measurements) > 0 ):
                for i, s in enumerate(self.scores):
                    s.measurements_id = m_ids[sm_index_list[i]]
            DeepScore.upsert_list( self.scores, load_defaults=True )
            SCLogger.debug( "save_and_commit scores" )
            commits.append( 'scores' )

        self.products_committed = ",".join( commits )


    def delete_everything(self):
        """Delete (almost) everything associated with this DataStore.

        All data products in the data store are removed from the DB,
        and all files on disk and in the archive are deleted.

        Does *not* delete the exposure.  (There may well be other
        data stores out there with different images from the same
        exposure.)

        For similar reasons, does not delete the reference either.

        Clears out all data product fields in the datastore.

        """

        # Special case handling for report, since it was never in
        #   products_to_save.  We don't want it there, because it's
        #   handled differently from the actual data products.  (Most
        #   notably: although there are exceptions (image and WCS), the
        #   default idea for our data products is that once a database
        #   entry is written, it stays the same.  The reports database
        #   entry is very much a "update with status" thing, though.)
        if self.report is not None:
            self.report.delete_from_disk_and_database()

        # Not just deleting the image and allowing it to recurse through its
        #   downstreams because it's possible that the data products weren't
        #   all added to the databse, so the downstreams wouldn't be found.
        # Go in reverse order so that things that reference other things will
        #   be deleted before the things they reference.

        del_list = [ getattr( self, i ) for i in self.products_to_save if i != 'exposure' ]
        del_list.reverse()
        for obj in del_list:
            if obj is not None:
                if isinstance( obj, list ):
                    for o in obj:
                        o.delete_from_disk_and_database()
                else:
                    obj.delete_from_disk_and_database()

        self.clear_products()


    def clear_products(self):
        """ Make sure all data products are None so that they aren't used again. """
        for att in self.products_to_save:
            setattr(self, att, None)

        for att in self.products_to_clear:
            setattr(self, att, None)

        self.report = None


    def free( self, not_zogy_specific_products=False ):
        """Set lazy-loaded data product fields to None in an attempt to save memory.

        If data products have not been saved to the file store and
        database yet, then things will probably break, because they will not
        be lazy-loadable.

        """

        if self.exposure is not None:
            for field in [ '_data', '_section_headers', '_weight', '_weight_section_headers',
                           '_flags', '_flags_section_headers' ]:
                if getattr( self.exposure, field ) is not None:
                    getattr( self.exposure, field ).clear_cache()
            if self.exposure._header is not None:
                self.exposure._header = None

        for prop in [ self._image, self._ref_image, self.aligned_ref_image, self.aligned_new_image,
                      self._sub_image,
                      self._bg, self._ref_bg, self.aligned_ref_bg, self.aligned_new_bg,
                      self._sources, self._ref_sources, self.aligned_ref_sources, self.aligned_new_sources,
                      self._psf, self._ref_psf, self.aligned_ref_psf, self.aligned_new_psf,
                      self._wcs, self._ref_wcs ]:
            if prop is not None:
                prop.free()

        if not not_zogy_specific_products:
            for prop in [ 'zogy_score', 'zogy_alpha', 'zogy_alpha_err', 'zogy_psf' ]:
                setattr( self, prop, None )
