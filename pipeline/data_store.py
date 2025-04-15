import io
import datetime
import sqlalchemy as sa
import uuid
# import traceback

from util.util import listify, asUUID, env_as_bool
from util.logger import SCLogger

from models.base import SmartSession, FileOnDiskMixin, FourCorners
from models.provenance import CodeVersion, Provenance, ProvenanceTag
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference, image_subtraction_components
from models.cutouts import Cutouts
from models.measurements import Measurements, MeasurementSet
from models.deepscore import DeepScore, DeepScoreSet
from models.refset import RefSet
from models.fakeset import FakeSet, FakeAnalysis

# The products that are made at each processing step.
# Usually it is only one, but sometimes there are multiple products for one step (e.g., extraction)
_PROCESS_PRODUCTS = {
    'exposure': 'exposure',
    'preprocessing': 'image',
    'coaddition': 'image',
    'extraction': ['sources', 'psf', 'bg'],
    'astrocal': 'wcs',
    'photocal': 'zp',
    'referencing': 'reference',
    'subtraction': 'sub_image',
    'detection': 'detections',
    'cutting': 'cutouts',
    'measuring': 'measurements',
    'scoring': 'scores',
}


class ProvenanceTree(dict):
    """Used internally by DataStore and Pipeline.

    This keeps track of all the provenances for all the steps that could
    be run for data products in this DataStore.  It's normally set by
    DataStore.make_prov_tree()

    This isn't internally enforced, but a self-consistent provenance
    tree has an entry in self.upstream_steps for each entry in its
    provenance dictionary, and all of the values in the upstream steps
    are also in the keys of the provenance dictionary.

    """

    def __init__( self, provs={}, upstream_steps={} ):
        """Create a ProvenanceTree.

        Once created, the provences can be accessed from the
        ProvenanceTree object just like a dictionary.  (In fact, it *is*
        a dictionary.)  The upstream_steps property has the dictionary
        of upstream steps.

        Parameters
        ----------
          provs : dict
            A dictionary of processname : provenance

          upstream_steps : dict
            A dictionary of processname : list of upstream process
            names.  This dictionary must be ordered, so that all of the
            upstreams of a process are earlier in the upstream_steps
            dictionary.

        """
        super().__init__( provs )
        self.upstream_steps = upstream_steps


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
    pipeline/top_level.py) and run pipeline.make_provenance_tree().  You
    can also use the make_prov_tree() method of DataStore, but in that
    case you must also build up a proper set of dictionaries of critical
    parameters yourself.

    You can get the provenances from a DataStore with get_provenance.

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
        'measurement_set',
        'deepscore_set',
        'fakes',
        'fakeanal'
    ]

    # these get cleared but not saved
    products_to_clear = [
        'reference',
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
        'all_measurements',
        'fakeanal',
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
                self._exposure = self.get_raw_exposure()
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
        return self.reference.image if self.reference is not None else None

    @ref_image.setter
    def ref_image( self, val ):
        raise RuntimeError( "Don't directly set ref_image, call get_reference" )

    @property
    def ref_sources( self ):
        return self.reference.sources if self.reference is not None else None

    @ref_sources.setter
    def ref_sources( self, val ):
        raise RuntimeError( "Don't directly set ref_sources, call get_reference" )

    @property
    def ref_bg( self ):
        return self.reference.bg if self.reference is not None else None

    @ref_bg.setter
    def ref_bg( self, val ):
        raise RuntimeError( "Don't directly set ref_bg, call get_reference" )

    @property
    def ref_psf( self ):
        return self.reference.psf if self.reference is not None else None

    @ref_psf.setter
    def ref_psf( self, val ):
        raise RuntimeError( "Don't directly set ref_psf, call get_reference" )

    @property
    def ref_wcs( self ):
        return self.reference.wcs if self.reference is not None else None

    @ref_wcs.setter
    def ref_wcs( self, val ):
        raise RuntimeError( "Don't directly set ref_wcs, call get_reference" )

    @property
    def ref_zp( self ):
        return self.reference.zp if self.reference is not None else None

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
            if val.new_zp_id != self.zp.id:
                raise ValueError( "Can't set a sub image inconsistent with image zeropoint" )
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
            self.measurement_set = None
        else:
            if self._detections is None:
                raise RuntimeError( "Can't set DataStore cutouts until it has a detections" )
            if not isinstance( val, Cutouts ):
                raise TypeError( f"DataStore.cutouts must be a Cutouts, not a {type(val)}" )
            if ( self._measurement_set is not None ) and ( self._measurement_set.cutouts_id != val.id ):
                raise ValueError( "Can't set a cutouts inconsistent with measurement set." )
            self._cutouts = val
            self._cutouts.detections_id = self._detections.id

    @property
    def measurement_set( self ):
        return self._measurement_set

    @measurement_set.setter
    def measurement_set( self, val ):
        if val is None:
            self._measurement_set = None
            self.deepscore_set = None
        else:
            if self._cutouts is None:
                raise RuntimeError( "Can't set DataStore measurement_set until it has a cutouts" )
            if not isinstance( val, MeasurementSet ):
                raise TypeError( f"Datastore.measurement_set must be a MeasurementSet, not a {type(val)}" )
            if ( self._deepscore_set is not None ) and ( self._deepscore_set.measurementset_id != val.id ):
                raise ValueError( "Can't set a measurement_set inconsistent with deepscore_set" )
            self._measurement_set = val
            self._measurement_set.cutouts_id = self._cutouts.id

    @property
    def measurements( self ):
        if self.measurement_set is None:
            return None
        return self.measurement_set.measurements


    @property
    def deepscore_set( self ):
        return self._deepscore_set

    @deepscore_set.setter
    def deepscore_set( self, val ):
        if val is None:
            self._deepscore_set = None
        else:
            if ( self._measurement_set is None ):
                raise RuntimeError( "Can't set DataStore deepscore_set until it has a measurement_set" )
            if not isinstance( val, DeepScoreSet ):
                raise TypeError( f"Datastore.deepscore_set must be a DeepScoreSet, not a {type(val)}" )

            # ensure that there is a score for each measurement, otherwise reject
            if ( len( val.deepscores ) != len( self.measurement_set.measurements ) ):
                raise ValueError( "Score and measurements list not the same length" )
            if not all( d.index_in_sources == m.index_in_sources
                        for d, m in zip( val.deepscores, self.measurements ) ):
                raise ValueError( "Score and measurements index_in_sources must match" )

            self._deepscore_set = val
            self._deepscore_set.measurementset_id = self._measurement_set.id

    @property
    def deepscores( self ):
        if self._deepscore_set is None:
            return None
        return self.deepscore_set.deepscores


    @property
    def fakes( self ):
        return self._fakes

    @fakes.setter
    def fakes( self, val ):
        if val is None:
            self._fakes = None
            self._fakeanal = None
        else:
            if self._zp is None:
                raise RuntimeError( "Can't set DataStore fakes until it has a zp." )
            if not isinstance( val, FakeSet ):
                raise TypeError( f"DataStore.fakes must be a FakeSet, not a {type(val)}" )
            if self._zp.id != val.zp_id:
                raise ValueError( "Can't set a fakes inconsistent with zp" )
            self._fakes = val

    @property
    def fakeanal( self ):
        return self._fakeanal

    @fakeanal.setter
    def fakeanal( self, val ):
        if val is None:
            self._fakeanal = None
        else:
            if self._fakes is None:
                raise RuntimeError( "Can't set DataStore fakeanal until it has fakes." )
            if not isinstance( val, FakeAnalysis ):
                raise TypeError( f"DataStore.fakeanal must be a FakeAnalysis, not a {type(val)}" )
            if self._fakes.id != val.fakeset_id:
                raise ValueError( "Can't set a fakeanal inconsistent with fakes" )
            if val.orig_deepscore_set_id is None:
                # No way to verify this, because the datastore doesn't have the original deescore set
                raise ValueError( "fakeanal must have orig_deepscore_set_id set" )
            self._fakeanal = val


    @staticmethod
    def from_args(*args, **kwargs):
        """Create a DataStore object from the given arguments.

        See the parse_args method for details on the different input parameters.

        Returns
        -------
        ds: DataStore
            The DataStore object.

        """
        if len(args) == 0:
            raise ValueError('No arguments given to DataStore constructor!')
        if len(args) == 1 and isinstance(args[0], DataStore):
            return args[0]
        else:
            ds = DataStore( *args, **kwargs )
            return ds


    def parse_args(self, *args, prov_tree=None ):
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

        prov_tree : ProvenanceTree or None
           Initialize the DataStore's provenance tree (stored in the
           prov_tree property) with this.

        """
        if len(args) == 1 and isinstance(args[0], DataStore):
            # if the only argument is a DataStore, copy it
            self.__dict__ = args[0].__dict__.copy()
            return

        if prov_tree is not None:
            if not isinstance( prov_tree, ProvenanceTree ):
                raise TypeError( f"provtree must be a ProvenanceTree, not a {type(prov_tree)}" )
            self.prov_tree = prov_tree

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


    def __init__(self, *args, **kwargs):
        """Make a DataStore.

        See the parse_args method for details on how to initialize this object.

        Please make sure to add any new attributes to the products_to_save list.
        """
        # these are data products that can be cached in the store
        self._exposure = None  # single image, entire focal plane
        self._section = None  # SensorSection

        self.prov_tree = None  # ProvenanceTree object
        self._provtag = None
        self._code_version = None

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
        self._measurement_set = None  # photometry and other measurements for each source
        self._deepscore_set = None  # a list of r/b and ML/DL scores for Measurements
        self._fakes = None
        self._fakeanal = None

        # these need to be added to the products_to_clear list
        self.reference = None
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

        # These are flags that tell processes running the data store some things to do nor not do
        self.update_runtimes = True
        self.update_memory_usages = env_as_bool( 'SEECHANGE_TRACEMALLOC' )

        self.parse_args(*args, **kwargs)


    def update_report(self, process_step):
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


    def make_prov_tree( self, steps, pars, provtag=None, ok_no_ref_prov=False, upstream_steps=None,
                        starting_point=None ):
        """Create the DataStore's provenance tree.

        Also creates provenances and saves them to the database if
        they're not there already.

        Will base the provenance tree off of starting_point if that's
        given, otherwise off of the provenance of self.exposure if
        that's defined, otherwise off of the provenance of self.image.

        As a side effect, if 'subtraction' is in the steps, it tries to
        identify a reference for the image based on
        pars['subtraction']['refset'].  If a reference is not found,
        referencing and everything downstream from it will not have
        provenances identified or generated.  (This is necessary because
        the provenance of the subtraction and later products depends on
        the provenance of the reference.  References are not built as
        part of the main pipeline, but external to the main pipeline.)

        Parameters
        ----------
          steps : list of str The steps that we want to generate
             provenances for.  Must be in order (i.e. anything later in
             the list has all of its upstreams earlier in the list).
             This list should *not* include "referencing"; that will be
             added automatically if "subtraction" is in the list of
             steps.

          pars : a dictionary of step -> dict
             The dictionary for a given step must be what you'd get from
             a call to get_critical_pars on an object that performs that
             step.  (The get_critical_pars_dicts method of a Pipeline
             object returns what's needed here.)

          provtag : str or None
             If not None, add all created provenances to this provenance tag

          ok_no_ref_prov: bool, default False
             If True, and if 'subtraction' is in steps, and a reference isn't found,
             usually that's an exception.  if ok_no_ref_prov is True, then instead
             just stop generating provenances at the step before subtraction.

          upstream_steps: dict or None
             You usually don't want to specify this.  This a dict of
             str: list.  Each key is the name of a process, the value is
             a list of process names that are upstream to this process.
             It must be ordered so that all of the upstream processes
             are keys earlier in the dict.  There is a default built in
             that is usually what you want to use.

          starting_point: Provenance or None
             The provenance that the tree starts from; the first
             step in steps will base put this into its upstreams.

        """

        # Make a copy of steps so we can modify it
        steps = steps.copy()

        code_version = None
        is_testing = None

        if not isinstance( pars, dict ):
            raise TypeError( "pars must be a dictionary" )
        if ( ( not all( isinstance( v, dict ) for v in pars.values() ) ) or
             ( not all( isinstance( k, str ) for k in pars.keys() ) ) ):
            raise TypeError( "pars must be a dictionary of str:dict" )
        for step in steps:
            if step not in pars:
                raise ValueError( f"Step {step} not in pars" )

        if 'referencing' in steps:
            raise ValueError( "Steps must not include referencing" )

        provs = ProvenanceTree()

        if upstream_steps is not None:
            if ( ( not isinstance( upstream_steps, dict ) ) or
                 ( not all( isinstance( k, str ) for k in upstream_steps.keys() ) ) or
                 ( not all( isinstance( v, list ) for v in upstream_steps.values() ) ) or
                 ( not all( all( isinstance( vv, str ) for vv in v ) for v in upstream_steps.values() ) ) ):
                raise TypeError( "upstream_steps must be a dict of str: list of str" )
            k0 = next( iter( upstream_steps.keys() ) )
            if upstream_steps[k0] != []:
                ValueError( f"The first step in upstream_steps cannot have prerequisites! "
                            f"Got first step {k0} had prereqs {upstream_steps[k0]}" )
            provs.upstream_steps = upstream_steps.copy()
            if 'starting_point' not in provs.upstream_steps:
                keyorder = ['starting_point'] + list( provs.upstream_steps.keys() )
                provs.upstream_steps['starting_point'] = []
                provs.upstream_steps = { k: provs.upstream_steps[k] for k in keyorder }
                provs.upstream_steps[k0] = [ 'starting_point' ]
        else:
            provs.upstream_steps = {
                'starting_point': [],
                'preprocessing': ['starting_point'],
                'extraction': ['preprocessing'],
                'astrocal': ['extraction'],
                'photocal': ['astrocal'],
                'referencing': [],   # This is a special case; it *does* have upstreams, but outside the main pipeline
                'subtraction': ['referencing', 'photocal'],
                'detection': ['subtraction'],
                'cutting': ['detection'],
                'measuring': ['cutting'],
                'scoring': ['measuring'],
                'alerting': []
            }
            # Put code here to modify upstream_steps based on things in pars
            # (This will happen with fake injection in a future PR.; the
            # subtraction's upstreams will change to have a fake injector,
            # and we'll add a fake injector key.)

        # Get started with the passed Exposure (usual case) or Image
        if starting_point is not None:
            if not isinstance( starting_point, Provenance ):
                raise TypeError( f"starting_point must be a Provenance, not a {type(starting_point)}" )
            provs['starting_point'] = starting_point
        elif self.exposure is not None:
            if not isinstance( self.exposure, Exposure ):
                raise TypeError( f"DataStore's exposure field is a {type(self.exposure)}, not Exposure!" )
            provs['starting_point'] = Provenance.get( self.exposure.provenance_id )
        elif self.image is not None:
            if not isinstance( self.image, Image ):
                raise TypeError( f"DataStore's image field is a {type(self.image)}, not Image!" )
            provs['starting_point'] = Provenance.get( self.image.provenance_id )
        else:
            raise RuntimeError( "make_prov_tree requires either a starting_point, or the "
                                "DataStore must have either an exposure or an image" )
        code_version = CodeVersion.get_by_id( provs['starting_point'].code_version_id )
        is_testing  = provs['starting_point'].is_testing

        # Get the reference
        ref_prov = None
        if 'subtraction' in steps:
            refset_name = pars['subtraction']['refset']
            if refset_name is None:
                raise ValueError( "'subtraction' is in steps but refset_name is None; this is inconsistent" )
            refset = RefSet.get_by_name( refset_name )
            if refset is None:
                if ok_no_ref_prov:
                    SCLogger.warning( "No ref provenance found, "
                                      "not generating provenances for subtraction or later steps" )
                    subdex = steps.index( 'subtraction' )
                    steps = steps[:subdex]
                else:
                    raise ValueError(f'No reference set with name {refset_name} found in the database!')
            else:
                ref_prov = Provenance.get( refset.provenance_id )
                if ref_prov is None:
                    raise RuntimeError( f"Ref provenance {refset.provenance_id} not found; database corrupted." )

        if ref_prov is not None:
            provs['referencing'] = ref_prov

        for step in steps:
            # figure out which provenances go into the upstreams for this step
            up_steps = provs.upstream_steps[ step ]
            if isinstance( up_steps, str ):
                up_steps = [ up_steps ]
            upstream_provs = [ provs[u] for u in up_steps ]
            provs[step] = Provenance( code_version_id=code_version.id,
                                      process=step,
                                      parameters=pars[step],
                                      upstreams=upstream_provs,
                                      is_testing=is_testing )
            provs[step].insert_if_needed()

            # Set the provenance tag if requested.
            # (Chances are it's already set, but somebody will be first.)
            self._provtag = provtag
            if self._provtag is not None:
                ProvenanceTag.addtag( self._provtag, provs.values(), add_missing_processes_to_provtag=True )

        self._code_version = code_version
        self.prov_tree = provs


    def edit_prov_tree( self, step, params_dict=None, prov=None, new_step=False, provtag=None, donotinsert=False ):
        """Update the DataStore's provenance tree.

        Parameters
        ----------
           step: ProvenanceTree or str
              If this is a ProvenanceTree, completely replace the
              DataStore's provenance tree with this value.  (It's stored
              by reference, so don't make changes to what you pass in
              provs after passing it here if you don't want those
              changes to show up in the DataStore's provenance tree!)

              If this is a string, then use params_dict to generate a
              new provenance for this step, and for all steps downstream
              of this step.  If a provenance tag was previously passed
              to make_prov_tree, tag all newly created provenances with
              that provenance tag.  (Note, however, that this will
              probably raise an exception, because the provenance tag
              will have a pre-existing provenance for that step!  The
              anticipated use for edit_prov_tree is really in tests
              where we haven't set a provenance tag.)

           params_dict: dict
              A parameters dictionary for step, such as is produced by
              Parameters.get_critical_pars().  Ignored if prov is not
              None.

           prov: Provenance or None
              The provenance for this step.  WARNING: this code does
              not verify that the upstreams of this provenance are
              correct!  It's up to you to make sure you pass in a valid
              provenance.  (Downstream provenances will be still
              recreated using this as the upstream.)

           new_step: bool, default False
              If True, then this may be new step that doesn't currently
              exist in the prov tree.  However, that step must exist in
              the provenance tree's upstream_steps, and all of the
              upstreams of this provenance (as defined by the prov
              tree's upstream_steps) must already be in the prov tree.

           provtag: str or None
              This may only be passed if step is a ProvenanceTree.  In
              that case, assume that all provenances in ProvenanceTree
              are already tagged with this provenance tag.  (It's up to
              the user to ensure that's the case.)  Save this in the
              DataStore, so that future provenances created with
              edit_prov_tree will be tagged with this provenance tag.

          donotinsert: bool, default False
              If True, don't insert any newly created Provenances into
              the database.  (By default, they will be inserted.)

        """

        if isinstance( step, ProvenanceTree ):
            if params_dict is not None:
                raise ValueError( "params_dict must be None when passing a ProvenanceTree to edit_prov_tree" )
            self.prov_tree = step
            self._provtag = provtag
            self._code_version = CodeVersion.get_by_id( ( next(iter(step.values())) ).code_version_id )
            return

        if self.prov_tree is None:
            raise TypeError( "Can't edit provenance tree, DataStore doesn't have one yet." )

        if provtag is not None:
            raise ValueError( "Can't pass a provtag unless step is a ProvenanceTree" )

        if ( params_dict is None ) and ( prov is None ):
            raise ValueError( f"Can't edit provenance for step {step}, no params_dict nor prov passed." )

        if step not in self.prov_tree:
            if not new_step:
                raise RuntimeError( f"Can't modify provenance for step {step}, it's not in the current prov tree." )
            if step not in self.prov_tree.upstream_steps:
                raise RuntimeError( f"Can't add provenance for step {step}, it's not a known step." )
            if not all( s in self.prov_tree for s in self.prov_tree.upstream_steps[step] ):
                raise RuntimeError( f"Can't add provenance for step {step}, it's upstreams aren't "
                                    f"already in the current prov tree." )
            self.prov_tree[ step ] = prov

        mustmodify = { step }
        for s, ups in self.prov_tree.upstream_steps.items():
            if any( i in mustmodify for i in ups ):
                mustmodify.add( s )

        for curstep in self.prov_tree.keys():
            if curstep in mustmodify:
                if curstep == step:
                    if prov is not None:
                        self.prov_tree[curstep] = prov
                        continue
                    else:
                        params = params_dict
                else:
                    params = self.prov_tree[ curstep ].parameters

                upstream_provs = [ self.prov_tree[u] for u in self.prov_tree.upstream_steps[curstep] ]
                self.prov_tree[curstep] = Provenance( code_version_id=self._code_version.id,
                                                      process=curstep,
                                                      parameters=params,
                                                      upstreams=upstream_provs )
        if ( len(mustmodify) > 0 ) and ( not donotinsert ):
            with SmartSession() as sess:
                for curstep in self.prov_tree.keys():
                    if curstep in mustmodify:
                        self.prov_tree[curstep].insert_if_needed( session=sess )

        if self._provtag is not None:
            ProvenanceTag.addtag( self._provtag, self.prov_tree.values(), add_missing_processes_to_provtag=True )


    def get_provenance(self, process, pars_dict=None ):
        """Get the provenance for a given process.

        Will return the provenance from the DataStore's internal provenance tree.

        For historic reasons, if pars_dict is passed, will verify that
        the provenance is consistent with pars_tree, and raise an
        exception if it's not.  (Previously, we could ask for
        provenances with any pars_dict from DataStore, and there is code
        that uses that interface.)

        Parameters
        ----------
        process: str
            The name of the process, e.g., "preprocess", "extraction", "subtraction".

        pars_dict: dict
            A dictionary of critical parameters used for the process.

        Returns
        -------
        prov: Provenance
            The provenance for the given process.

        """

        # Need a prov_tree so we know what upstream steps there are to each process
        if self.prov_tree is None:
            raise RuntimeError( "get_provenance requires the DataStore to have a provenance tree" )

        if process not in self.prov_tree:
            raise ValueError( f"No provenance for {process} in provenance tree" )

        if ( pars_dict is not None ) and ( self.prov_tree[process].parameters != pars_dict ):
            raise ValueError( f"Passed pars_dict does not match parameters for internal provenance of {process}" )

        return self.prov_tree[process]


    def get_raw_exposure(self, session=None):
        """Get the raw exposure from the database."""
        if self._exposure is None:
            if self.exposure_id is None:
                raise ValueError('Cannot get raw exposure without an exposure_id!')

            with SmartSession(session) as session:
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
            An optional session to use for the database query.  If not
            given, will open a new session and close it when done with
            it.

        Returns
        -------
        image: Image object
            The image object, or None if no matching image is found.

        """
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
        elif ( self.prov_tree is not None ) and ( provenance.id != self.prov_tree['preprocessing'].id ):
            raise ValueError( "Passed image provenance doesn't match what's in the DataStore's provenance tree." )

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

          cls_upstream_id_att:
            The actual attribute from the class that holds the id of the
            upstream. E.g., if att="sources" and cls=SourceList, then
            upstream_att="image_id" and att=SourceList.image_id

          process: str
            The name of the process that produces this data product ('extraction', 'detection', ';measuring', etc.)

          is_list: bool, default False
            True if a list is expected (which currently is only for measurements).

          upstream_is_list: bool, default False
            True if the attribute represented by upstream_att is a list (eg measurements)

          match_prov: bool, default True
            True if the provenance must match.  (For psf, there is no
            provenance, it must just match sources_id.  There may be
            others.)

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
            An optional session to use for the database query.  If not
            given, will open a new session and close it at the end of
            the function.

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

    def get_background(self, session=None, reload=False):
        """Get a Background object, either from memory or from the database."""
        return self._get_data_product( 'bg', Background, 'sources', Background.sources_id, 'extraction',
                                       match_prov=False, reload=reload, session=session )

    def get_wcs(self, session=None, reload=False, provenance=None):
        """Get an astrometric solution in the form of a WorldCoordinates object, from memory or from the database."""
        return self._get_data_product( 'wcs', WorldCoordinates, 'sources', WorldCoordinates.sources_id, 'astrocal',
                                       match_prov=True, provenance=provenance, reload=reload, session=session )

    def get_zp(self, session=None, reload=False, provenance=None):
        """Get a zeropoint as a ZeroPoint object, from memory or from the database."""
        return self._get_data_product( 'zp', ZeroPoint, 'wcs', ZeroPoint.wcs_id, 'photocal',
                                       match_prov=True, provenance=provenance, reload=reload, session=session )


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
                ovfrac = FourCorners.get_overlap_frac(image, self.reference.image)
                if ovfrac < min_overlap:
                    self.reference = None

            # if we have survived this long without losing the reference, can return it here:
            if self.reference is not None:
                return self.reference

        # No reference was found (or it didn't match other parameters) must find a new one
        # First, clear out all data products that are downstream of reference.
        # (Setting sub_image will cascade to detections, cutouts, measurements.)

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
                    strio = io.StringIO()
                    strio.write( f"More than one reference overlapped the image by at least {min_overlap}:\n" )
                    for oopsi in sortdex:
                        strio.write( f"  {ovfrac[oopsi]:.2f} : ref {refs[oopsi]._id}  img {imgs[oopsi].filepath}\n" )
                    raise RuntimeError( strio.getvalue() )

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


    def get_sub_image(self, provenance=None, reload=False, session=None):
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
            An optional session to use for the database query.  If not
            given, will open a new session and close it at the end of
            the function.

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

        if self.zp is None:
            raise RuntimeError( "Can't get subtraction without a zp; try calling get_zp" )

        if self.reference is None:
            # We could do the call here, but there are so many configurable parameters to
            #   get_reference() that it's safer to make the user do it
            raise RuntimeError( "Can't get a subtraction without a reference; try calling get_reference" )

        with SmartSession( session ) as sess:
            if self.image_id is None:
                self.get_image( session=sess )
            if self.image_id is None:
                raise RuntimeError( "Can't get sub_image, don't have an image_id" )

            imgs = ( sess.query( Image )
                     .join( image_subtraction_components, Image._id==image_subtraction_components.c.image_id )
                     .filter( Image.provenance_id==provenance.id )
                     .filter( image_subtraction_components.c.new_zp_id==self.zp.id )
                     .filter( image_subtraction_components.c.ref_id==self.reference.id )
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

    def get_measurement_set( self, provenance=None, reload=False, session=None ):
        """Get the MeasurementsSet, either form memory, or from database."""
        return self._get_data_product( "measurement_set", MeasurementSet, "cutouts", MeasurementSet.cutouts_id,
                                       "measuring", is_list=False, provenance=provenance, reload=reload,
                                       session=session  )

    def get_deepscore_set( self, provenance=None, reload=False, session=None ):
        """Get the DeepScore set, either from memory or from the database."""
        return self._get_data_product( "deepscore_set", DeepScoreSet, "measurement_set",
                                       DeepScoreSet.measurementset_id, "scoring", is_list=False,
                                       provenance=provenance, reload=reload, session=session )

    def get_deepscores(self, provenance=None, reload=False, session=None):
        """Get a list of DeepScores, either from memory or from database.

        By construction, will be sorted the same as self.measurements (by index_in_sources).

        """
        return self.get_deepscore_set( self, provenance=provenance, reload=reload, session=session ).deepscores

    def get_fakes( self, provenance=None, reload=False, session=None ):
        """Get a FakeSet"""

        return self._get_data_product( "fakes", FakeSet, "zp", FakeSet.zp_id, "fakeinjection",
                                       match_prov=True, provenance=provenance, reload=reload, session=session )


    def get_fakeanal( self, orig_deepscore_set_id, reload=False, session=None ):
        if ( self._fakeanal is not None ) and ( not reload ):
            return self._fakeanal

        with SmartSession( session ) as session:
            fakeset = self.get_fakes( session=session )
            fakeanal = session.scalars( sa.select( FakeAnalysis )
                                        .where( FakeAnalysis.fakeset_id == fakeset.id )
                                        .where( FakeAnalysis.orig_deepscore_set_id == orig_deepscore_set_id )
                                       ).all()
            if len( fakeanal ) > 1:
                raise RuntimeError( "This should never happen." )
            elif len( fakeanal ) == 0:
                self._fakeanal = None
            else:
                self._fakeanal = fakeanal

        return self._fakeanal


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
        self.get_sub_image( reload=reload )
        self.get_detections( reload=reload )
        self.get_cutouts( reload=reload )
        self.get_measurement_set( reload=reload )
        _ = self.measurement_set.measurements   # Force the measurements to load
        self.get_deespcore_set( reload=reload )
        _ = self.deepscore_set.deepscores       # Force the deepscores to load



    def save_and_commit(self,
                        exists_ok=False,
                        overwrite=True,
                        no_archive=False,
                        update_image_header=False,
                        update_image_record=True,
                        force_save_everything=False ):
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
                raise RuntimeError( "There shouldn't be list objects any more; fix the code." )
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
                        if att in [ "psf", "bg" ]:
                            obj.save( image=self.image, sources=self.sources, **basicargs )
                        elif att in [ "sources", "wcs" ]:
                            obj.save( image=self.image, **basicargs )
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

        # psf
        if self.psf is not None:
            if self.sources is not None:
                self.psf.sources_id = self.sources.id
            SCLogger.debug( "save_and_commit upserting psf" )
            self.psf.upsert( load_defaults=True )
            commits.append( 'psf' )

        # bg
        if self.bg is not None:
            if self.sources is not None:
                self.bg.sources_id = self.sources.id
            SCLogger.debug( "save_and_commit upsertting bg" )
            self.bg.upsert( load_defaults=True )
            commits.append( 'bg' )

        # wcs
        if self.wcs is not None:
            if self.sources is not None:
                self.wcs.sources_id = self.sources.id
            SCLogger.debug( "save_and_commit upserting wcs" )
            self.wcs.upsert( load_defaults=True )
            commits.append( 'wcs' )

        # zp
        if self.zp is not None:
            if self.wcs is not None:
                self.zp.wcs_id = self.wcs.id
            if self.bg is not None:
                self.zp.background_id = self.bg.id
            SCLogger.debug( "save_and_commit upsertting zp" )
            self.zp.upsert( load_defaults=True )
            commits.append( 'zp' )

        # subtraction Image
        if self.sub_image is not None:
            self.sub_image.upsert( load_defaults=True )
            SCLogger.debug( "save_and_commit upserting sub_image" )
            commits.append( 'sub_image' )

        # detections
        if self.detections is not None:
            if self.sub_image is not None:
                self.detections.image_id = self.sub_image.id
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

        # measurements
        if self.measurement_set is not None:
            if self.cutouts is not None:
                self.measurement_set.cutouts_id = self.cutouts.id
            SCLogger.debug( "save_and_commit measurements" )
            self.measurement_set.upsert( load_defaults=True )
            if len( self.measurement_set.measurements ) > 0:
                for m in self.measurement_set.measurements:
                    m.measurementset_id = self.measurement_set.id
                Measurements.upsert_list( self.measurement_set.measurements, load_defaults=True )
            commits.append( 'measurement_set' )

        # scores
        if self.deepscore_set is not None:
            if self.measurement_set is not None:
                self.deepscore_set.measurementset_id = self.measurement_set.id
            SCLogger.debug( "save_and_commit scores" )
            self.deepscore_set.upsert( load_defaults=True )
            if len( self.deepscore_set.deepscores ) > 0:
                for d in self.deepscore_set.deepscores:
                    d.deepscoreset_id = self.deepscore_set.id
                DeepScore.upsert_list( self.deepscore_set.deepscores, load_defaults=True )
            commits.append( 'deepscore_set' )

        self.products_committed = ",".join( commits )

        # fakes
        if self.fakes is not None:
            if self.zp is not None:
                self.fakes.zp_id = self.zp.id
            SCLogger.debug( "save_and_commit fakes" )
            self.fakes.upsert( load_defaults=True )
            commits.append( "fakes" )

        # fake analysis
        if self.fakeanal is not None:
            if self.fakes is not None:
                self.fakeanal.fakeset_id = self.fakes.id
            # NO!  Not setting orig_deepscore_set_id.  The deepscore set
            #   in the DataStore is almost certainly *not* the original
            #   deepscore set, but the one from the with-fakes
            #   subtraction!  If somebody hasn't properly set
            #   orig_deepscore_set_id, then we'll just get a database
            #   error when we try to insert, which is fine.
            #   pipeline/top_level.py does the right thing.
            # if self.deepscore_set is not None:
            #     self.fakeanal.orig_deepscore_set_id = ...uhoh
            SCLogger.debug( "save_and_commit fakeanal" )
            self.fakeanal.upsert( load_defaults=True )
            commits.append( "fakeanal" )


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

        # TODO : free() for fakes and fakeanal
        for prop in [ self._image, self.aligned_ref_image, self.aligned_new_image,
                      self.reference, self._sub_image,
                      self._bg, self.aligned_ref_bg, self.aligned_new_bg,
                      self._sources, self.aligned_ref_sources, self.aligned_new_sources,
                      self._psf, self.aligned_ref_psf, self.aligned_new_psf,
                      self._wcs ]:
            if prop is not None:
                prop.free()

        if self.measurement_set is not None:
            self.measurement_set._measurements = None
        if self.deepscore_set is not None:
            self.deepscore_set.deepscores = None

        if not not_zogy_specific_products:
            for prop in [ 'zogy_score', 'zogy_alpha', 'zogy_alpha_err', 'zogy_psf' ]:
                setattr( self, prop, None )
