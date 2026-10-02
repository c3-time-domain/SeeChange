## Overview

SeeChange is designed to be used as a pipeline and archiving tool for imaging sky surveys, primarily for the La Silla Schmidt Southern Survey (LS4).

SeeChange consists of a main pipeline that takes raw images and produces a few data products:
 - Calibrated images (after preprocessing such as bias, flat, etc).

 - Source catalogs (after source detection and photometry).

 - Point Spread Function (PSF) for the image.

 - Calibrations of the sources to external catalogs: astrometric calibration in the form of a World Coordinate System (WCS) and a photometric calibration in the form of a magnitude zero point (ZP).

 - Difference images.

 - Source catalogs from the difference images
   (dubbed "detections" as we are mostly interested in transients).

 - Cutouts around the sources detected in the difference images, along with the corresponding image cutouts from the reference and the newly acquired images.

 - Measurements on those cutouts, including the photometric flux, the shapes, and some metrics that indicate if the source is astronomical or an artefact (using analytical cuts).

Additional pipelines for making bias frames, flat frames, and to produce deep coadded references are described separately.  The coadd pipeline is described in :doc:`references`.

Data is saved into a database, on the local filesystem, and on a data archive (usually this is a remote server).  Saving and loading data from these locations is described in :doc:`data_storage`.

To read more about developing and contributing to the project, see :doc:`contribution` and :doc:`testing`.  To set up an instance of the pipeline to run on a local machine or on a remote server, see :doc:`setup`.

### Project architecture

The project is organized around two main subdirectories, `models` and `pipeline`.  `models` contains the objects that represent the data products, most of which are mapped (using SQL Alchemy) to a PostgreSQL database. `pipeline` contains the code that runs the analysis and processing required to produce the data products.

Additional folders include:

 - `alembic`: for migrating the database.

 - `conductor`: The conductor is a web application that lives on a server somewhere and keeps track of what exposures are available for acquisition and processing.  It exists so that multiple instances of the pipeline can run on different clusters and all work on the same survey.

 - `data`: for storing local data, either from the tests or from the actual pipeline (not to be confused with the long term storing of data on the "archive").  (You probably don't really want to store data here, as your code is likely to be checked out on a filesystem that's different from the most efficient large-file storage system on your machine.  The actual data storage location is configured in a YAML file.)

 - `devshell`: docker definitions for running the pipeline in a local dockerized environment.

 - `docs`: documentation.

 - `extern`: external packages that are used by SeeChange, including the `nersc-upload-connector` package that is used to connect the archive.

 - `improc: image processing code that is used by the pipeline, generally manipulating images in ways that are not specific to a single point in the pipeline (e.g., image alignment or inpainting).  (There is some scope seepage between `improc` and `pipeline`.)

 - `tests`: tests for the pipeline (more on that below).

 - `utils`: generic utility functions that are used by the pipeline.

 - `webap`: A web application for browsing what exposures, images, and detections are in the database.

The actual source code for the pipeline is found in `pipeline`, `models`, `improc`, and `utils`.  Notable files in the `pipeline` folder include `data_store.py` (described below) and the `top_level.py` file that defines the `Pipeline` object, which is the main entry point for running the pipeline.  In `models` we have the `base.py` file, which contains tools for database communications, along with some useful mixin classes, and the `instrument.py` file, which contains the `Instrument` base class used to define various instruments from different surveys, plus files for each type of data product (images, source lists, etc.).


### Pipeline segments (processes)

The pipeline is broken into several "processes" that take one data product and produce the next.
The idea is that the pipeline can start and stop at any of these junctions and can still be started up
from that point with the existing data products.
Here is a list of the processes and their data products (including the object classes that track them):

 - preprocessing: dark, bias, flat, fringe corrections, etc. For large, segmented focal planes,
   will also segment the input raw data into "sections" that usually correspond to individual CCDs.
   This process takes an `Exposure` object and produces `Image` objects, one for each section/CCD.

 - extraction: find the sources in the pre-processed image, measure their PSF, cross-match them
   for astrometric and photometric calibration.
   This process takes an `Image` object and produces a `SourceList` and a 'PSF'.

 - astrocal: create a solution from pixel position to RA/Dec on the sky by matching objects in the image to the Gaia DR3 catalog.  This process takes a `SourceList` and produces a `WorldCoordinates.`

 - photocal: Use `SourceList` and the Gaia DR3 catalog to find a zeropoint so that `m=-2.5*log10(flux)`, where `flux` is the full-psf flux of a star in ADU in the image.  As this is a discovery pipeline, not a lightcurve-building pipeline, we don't do really careful photmetric calibration, and you shouldn't expect this to be better than a couple of percent.  Produces a `ZeroPoint` object.

 - subtraction: taking a reference image of the same part of the sky (usually a deep coadd) and subtracting it from the "new" image (the one being processed by the pipeline).  Different algorithms can be used to match the PSFs of the new and reference image (we currently implement ZOGY and Alard/Lupton, and hope to add SFFT later).  Uses an `Image`,`SourceList`, `WorldCoordinates`, and `ZeroPoint` object for the science (also called search or new) image, and a second `Image`, `SourceList`, and (maybe) `ZeroPoint` object for a reference (also called ref or template) image identified by a `Reference` object.  It produdes new `Image`, `WorldCoordinates`, and `ZeroPoint` objects for the difference image.  (By default, the `WorldCoordinates` and `ZeroPoint` are just inherited directly from the science image, as the reference image is warped to the science image and the difference image is scaled to have the same zeropoint as the science image.)

 - detection: finding the sources in the difference image.  This process uses the difference `Image` object and produces a `SourceList` object.  This new source list is different from the previous one, as it contains information only on variable or transient sources, and does not map the constant sources in the field.

 - cutting: this part of the pipeline identifies the area of the image around each source that was detected in the previous process step, and the corresponding area in the reference and the new image, and saves the list of those stamps as a `Cutouts` object.  There are three stamps for each detection: new, ref, and sub.

 - measuring: this part of the pipeline measures the fluxes and shapes of the sources in the cutouts. It uses a set of analytical cuts to distinguish between astronomical sources and artefacts.  This process uses `Cutouts` object to produce a list of `Measurements` objects, one for each source.

 - scoring: this part of the pipeline assigns to each measurement a deep learning/machine learning score based on a given algorithm and parameters. In addition to a column for the score, the resulting deepscores contain a JSONB column which can contain additional relevant information.  This process uses the list of `Measurements` objects to product a list of `Deepscore` objects, one for each measurement.

The final stage of the pipeline adds the new measurements that pass configurable thresholds to the database and attempts to link them to existing `Object`s.  If no object exists (in that location on the sky), a new one is created.

Alerts are then optinally and sent to a kafka server for any new measurements, and the information on the associated `Object` is added to the alert to provide historical context.

More details on each step in the pipeline and their parameters can be found in `docs/pipeline.md`.


### The `DataStore` object

Every time we run the pipeline, objects need to be generated and pulled from the database.
The `DataStore` object is generated by the first process that is called in the pipeline,
and is then passed from one process to another.

The datastore contains all the data products for the "new" image being processed,
and also data products for the reference image (that are produced earlier by the reference pipeline).
The datastore can also be used to query for any data product on the database,
getting the "latest" version (see below for more on versioning).

*WARNING: I think this next section won't work as is.  Needs updating to deal with loading a provenance tree into the DataStore.*

To quickly find the relevant data, initialize the datastore using the exposure ID and the section ID,
or using a single integer as the image ID.

```python
from pipeline.data_store import DataStore
ds = DataStore( exposure_id=expid, section_id=1 )
# or, using the image ID:
ds = DataStore( image_id=imgid )
```

where `expid` and `imgid` would be either strings or `uuid.UUID` objects holding the id of the exposure or image you want.  (These are found in the database in the `_id` column, and can be accessed in a model object via the `id` property.)  `Image` and `Exposure` ids are internal database identifiers, while the section ID is defined by the instrument used, and usually refers to the CCD number or name (it can be an integer or a string).  E.g., the DECam sections are named `N1`, `N2`, ... `S1`, `S2`, etc.

Once a datastore is initialized, it can be used to query for any data product:

```python
# get the image object:
image = ds.get_image()
# get the source list:
source_list = ds.get_sources()
# get the difference image:
diff_image = ds.get_subtraction()
# and so on...
```

There could be multiple versions of the same data product, produced with different parameters or code versions.  A user may choose to pass a `provenance` input to the `get` methods, to specify which version of the data product is requested.  If no provenance is specified, the provenance is loaded either from the datastore's general `prov_tree` dictionary, or if it doesn't exist, will just load the most recently created provenance for that pipeline step.

```python
from models.provenance import Provenance
prov = Provenance(
   process='extraction',
   code_version=code_version,
   parameters=parameters,
   upstreams=upstream_provs
)
# or, using the datastore's tool to get the "right" provenance:
prov = ds.get_provenance(process='extraction', pars_dict=parameters)

# then you can get a specific data product, with the parameters and code version:
sources = ds.get_sources(provenance=prov)
```

See :ref:`overview-provenance` below for more information about versioning using the provenance model.


### Configuring the pipeline

Each part of the pipeline (each process) is conducted using a dedicated object.  In practice, you will not instantiate and run these objects individually, but would rather instantiate a single `Pipeline` object (defined in `pipeline/top_level.py`).

 - preprocessing: using a `Preprocessor` object defined in `pipeline/preprocessing.py`.

 - extraction: using a `Detector` object defined in `pipeline/detection.py` to produce the `SourceList` and `PSF`
   objects.

 - backgrounding: using a `Backgrounder` object defined in `pipeline/backgrounding.py` to produce a `Background` (i.e. sky level) object.

 - astrocal: using the `AstroCalibrator` object defined in `pipeline/astro_cal.py`, to produce the `WorldCoordinates` object.

 - photocal: Using the `PhotCalibrator` object defined in `pipeline/photo_cal.py`, to produce the `ZeroPoint` object.

 - subtraction: using the `Subtractor` object defined in `pipeline/subtraction.py`, producing an `Image` object (and some others)

 - detection: again using the `Detector` object, with different parameters, also producing a `SourceList` object.

 - cutting: using the `Cutter` object defined in `pipeline/cutting.py`, producing a list of `Cutouts` objects.

 - measuring: using the `Measurer` object defined in `pipeline/measuring.py`, producing a list of `Measurements` objects.

 - scoring: using the `Scorer` object defined in `pipeline/scoring.py`, producing a list of `Deepscore` objects.

All these objects are initialized as attributes of a top level `Pipeline` object, which is defined in `pipeline/top_level.py`.  Each of these objects can be configured using a dictionary of parameters.

There are a few ways to configure any object in the pipeline.  By default, the pipeline will use the config system defined in `util/config.py` to read a configuration YAML file.  An example file with documentation in comments (and the default location of this file) can be found as `default_config.yaml` in the directory of the project.  The `SEECHANGE_CONFIG` environment variable allows you to change the location of the config file that will be read when you run the code.

Parameters from the config file can be overidden at runtime by passing dictionary arguments to the `Pipeline` constructor.  Keys in this dictionary can include `pipeline`, `preprocessing`, etc.  Each of those keys should map to another dictionary, with parameter choices for that process.  More information will (eventually) be in :doc:`configuration`.  For example:


```python
from pipeline.top_level import Pipeline
p = Pipeline(
   pipeline={pipeline_par1': pl_val1, 'pipeline_par2': pl_val2},
   preprocessing={'preprocessing_par1': pp_val1, 'preprocessing_par2': pp_val2},
   extraction={'sources': {'sources_par1: sr_val1, 'sources_par2': sr_val2},
               'bg': {'bg_par1': bg_val1, 'bg_par2': bg_val2},
               'wcs': {'wcs_par1': wcs_val1, 'wcs_par2': wcs_val2},
               'zp': {'zp_par1': zp_val1, 'zp_par2': zp_val2}
              },
   ...
)
```

(Note that although extraction, backgrounding, astrocal, and photocal are separate steps, they are configured all as substeps of the extraction step (with "sources" holding the configuration for actual extraction).  This is because all four steps share the same provenance, which was done to simplify some issues of data management in the pipeline.  One consequence of this is that if you want to (say) change a parameter in how the zeropoint is calculated, you have to redo all of these steps, you can't just redo the (probably fast) zeropoint step.)

If only a single object needs to be initialized, pass the parameters directly to the object's constructor.  Note that in this case the parmeters from the config file will *not* be used; only when you instantiate a top-level pipeline are the values in the config file automatically used to configure your process object.

```python
from pipeline.preprocessing import Preprocessor
pp = Preprocessor(
   preprocessing_par1=pp_value1,
   preprocessing_par2=pp_value2
)
```

If you do want to use the configuration file to configure an individual processing step's object, you can use the config system to pass that:

```python
from util.config import Config
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
cfg = Config.get()
pp = Preprocessor( **cfg.value('preprocessing') )
ex = Detector( **cfg.value('extraction.sources') )
ac = AstroCalibrator( **cfg.value('extraction.wcs') )
```

(As described above, parameters for extraction, backgrounding, astrocal, and photocal are all stored as sub-dictionaries of the `extraction` dictionary in the config file.  That why we used the `extraction.sources` and `extraction.wcs` values of the Config object in the example above.)

Finally, after all objects are initialized with their parameters, a user (e.g., in an interactive session) can modify any of the parameters using the `pars` attribute of the object.

```python
pp.pars['preprocessing_par1'] = new_pp_value1
# or
pp.pars.preprocessing_par1 = new_pp_value1
```

To get a list of all the parameters that can be modified, their descriptions and default values, use

```python
pp.pars.show_pars()
```

The definition of the base `Parameters` object is at `pipeline/parameters.py`, but each process class has a dedicated `Parameters` subclass where the parameters are defined in the `__init__` method.

Additional information on using config files and the `Config` class
can be found (eventually) at :doc:`configuration`.

.. _overview-provenance:
### Versioning using the `Provenance` model

Each of the output data products is stamped with a `Provenance` object.  This object tracks the code version, the parameters chosen for that processing step, and the provenances of the data products used to produce this product (the "upstreams").  The `Provenance` object is defined in `models/provenance.py`.

Users interacting with the database outside of the main pipeline are likely want to use provenance tags; see :doc:`versioning`.

The `Provenance` object is initialized with the following inputs:

 - `process`: the name of the process that produced this data product ('preprocessing', 'subtraction', etc.).

 - `code_version`: the version object for the code that was used to produce this data product.

 - `parameters`: a dictionary of parameters that were used to produce this data product.

 - `upstreams`: a list of `Provenance` objects that were used to produce this data product.

The code version is a `CodeVersion` object, defined also in `models/provenance.py`.  The ID of the `CodeVersion` object is any string, *but this will change soon as we move to semantic versioning for individual steps in the pipeline*. It is recommended that new code versions are added to the database for major changes in the code, e.g., when regression tests indicate that data products have different values with the new code.

The parameters dictionary should include only "critical" parameters, as defined by the `__init__` method of the specific process object, and should not include auxiliary parameters like verbosity or number of processors.  Only parameters that affect the product values are included.

The upstreams are other `Provenance` objects defined for the data products that are an input to the current processing step.  The flowchart of the different process steps is defined in `pipeline/datastore.py::UPSTREAM_STEPS`.  E.g., the upstreams for the `subtraction` object are `['preprocessing', 'extraction', 'referencing']`.  `referencing` is a special case; its upstream is replaced by the provenances of the reference's `preprocessing` and `extraction` steps.

When a `Provenance` object has all the required inputs, it will produce a hash identifier that is unique to that combination of inputs.  So, when given all those inputs, a user (or a datastore) can create a `Provenance` object and then query the database for the data product that has that provenance.  This process is implemented in the `DataStore` object, using the different `get` methods.

Additional details on data versioning can be found at :doc:`versioning`.

### Database schema

The database structure is defined using SQLAlchemy's object-relational model (ORM).  Each table is mapped to a python-side object.  The definitions of the models can be found in the various files in the `models` subdiretory.

The following classes define models and database tables, each associated with a data product.  Usually, the file in which a given model's definition can be found is obvious, but if not, a quick search for `class <modelname>` in the `models` subdirectory should suffice.

 - `Exposure`: a single exposure of all CCDs on the focal plane, linked to (usually) raw data on disk.

 - `Image`: a simple image, that has been preprocessed to remove bias, dark, and flat fields, etc.  An `Image` can be linked to one `Exposure`, or it can be linked to a list of other `Image` objects (if it is a coadded image) or it can be linked to a reference and new `Image` objects (if it is a difference image).

 - `SourceList`: a catalog of light sources found in an image. It can be extracted from a regular image or from a subtraction. The 'SourceList' is linked to a single `Image` and will have the coordinates of all the sources detected.

 - `PSF`: a model of the point spread function (PSF) of an image.  This is linked to a SourceList object, and holds the PSF model for the image that that source list is linked to.

 - `WorldCoordinates`: a set of transformations used to convert between image pixel coordinates and sky coordinates.  This is linked to a single `SourceList` and will contain the WCS information for that image that the source list is linked to.

 - `ZeroPoint`: a photometric solution that converts image flux to magnitudes.
   This is linked to a single `SourceList` and will contain the zeropoint information for the image that the source list is linked to.

 - `Object`: a table that contains information about a single object found on a difference image (real or bogus).  Practically speaking, an object is defined (for the most part) as a position on the sky (modulo some flags like `is_fake` and `is_test`).  (When the pipeline finds something on a difference image, it will link the measurements of what it finds to an `Object` within a small radius (1") of the discovery's position, creating a new object if an appropriate one does not exist.)

 - `Cutouts`: contain a list of small pixel stamps around a point in the sky in a new image, reference image, and subtraction image. Each `Cutouts` object is linked back to a single subtraction based `SourceList`, and will contain the three cutouts for each source in that source list.

 - `Measurements`: contains measurements made on something detected in a difference image.  Measurements are made on the three stamps (new, ref, sub) of one of the list of such stamps in a linked `Cutouts`.  Values include flux+errors, magnitude+errors, centroid positions, spot width, analytical cuts, etc.

 - `DeepScore` : contains a score in the range 0-1 (where higher means more likely to be real) assigned based on machine learning/deep learning algorithms.  Additionally contains a JSONB field which can contain additional information.

 - `Provenance`: A table containing the code version and critical parameters that are unique to this version of the data.  Each data product above must link back to a provenance row, so we can recreate the conditions that produced this data.

 - `Reference`: An object that identifies a linked `Image` and associated `SourceList` as being a potential reference for the relevant filter and location on the sky.

 - `CalibratorFile`: An object that tracks data needed to apply calibration (preprocessing) for a specific instrument.  The calibration could include an `Image` data file, or a generic non-image `DataFile` object.

 - `DataFile`: An object that tracks non-image data on disk for use in, e.g., calibration/preprocessing.

 - `CatalogExcerpt`: An object that tracks a small subset of a large catalog, e.g., a small region of Gaia DR3 that is relevant to a specific image.  The excerpts are used as cached parts of the catalog, that can be reused for multiple images with the same pointing.


#### Additional classes

The `Instrument` class is defined in `models/instrument.py`.  Although it is not mapped to a database table, it contains important tools for processing images from different instruments.  For each instrument we define a subclass of `Instrument` (e.g., `DECam`) which defines the properties of that instrument and methods for loading the data, reading the headers, and other instrument-specific tasks.

More on instruments can be found at :doc:`instruments`.

Mixins are defined in `models/base.py`.  These are used as additional base classes for some of the models used in the pipeline, to give them certain attributes and methods that are shared across many classes.

These include:

 - `UUIDMixin`: Most of our database tables use a uuid as a primary key, and they do this by deriving from this class.  The choice of this over an auto-incrementing integer makes it easier to run the pipeline in multiple places at once, building up cross-references between objects without having to contact the database as each individual reference is constructed.  The actual database column is `_id`, but usually you should refer to the `id` (without underscore) property of an object.  (Accessing that property will automatically generate an id for a new object if it does not already have one.)


 - `FileOnDiskMixin`: adds a `filepath` attribute to a model (among other things), which make it possible to save/load the file to the archive, to find it on local storage, and to delete it.

 - `SpatiallyIndexed`: adds a right ascension and declination attributes to a model, and also adds a q3c spatial index to the database table.  This is used for many of the objects that are associated with a point on the celestial sphere, e.g., an `Image` or an `Object`.

 - `FourCorners`: adds the RA/dec for the four corners of an object, describing the bounding box of the object on the sky.  This is particularly useful for images but also for catalog excerpts, that span a small region of the sky.

 - `HasBitFlagBadness`: adds a `_bitflag` and `_upstream_bitflag` columns to the model.  These allow flagging of bad data products, either because they are bad themselves, or because one of their upstreams is bad. It also adds some methods and attributes to access the badness like `badness` and `append_badness()`.  If you change the bitflag of such an object, and it was already used to produce downstream products, make sure to use `update_downstream_badness()` to recursively update the badness of all downstream products.

Enums and bitflag are stored on the database as integers (short integers for Enums and long integers for bitflags).  These are mapped when loading/saving each database object using a set of dictionaries defined in `models/enums_and_bitflags.py`.

More information on data storage and retrieval can be found at :doc:`data_storage`.


### Parallelization

The pipeline is built around an assumption that large surveys have a natural way of segmenting their imaging data (e.g., by CCDs).  We also assume the processing of each section is independent of the others.  Thus, it is easy to parallelize the pipeline to work with each section separately.  We can just run a completely independent process to analyze each CCD of each exposure.  (While this is *almost* embarassingly parallel, there are some resource contention and race conditions we deal with in the pipline to handle cases of, e.g., several processes all trying to get the Gaia catalog for the same region of the sky at once, or several processes all trying to load a shared calibraiton file into the database at the same time.)

The executable that runs the main pipeline is `pipeline/pipeline_exposure_launcher.py`.  It contacts the conductor (a server that keeps track of which exposures are available for processing and which ones have been "claimed" by systems and clusters) and waits to be given an exposure to process.  Once it gets that exposure, it will use python's `mutliprocessing` to launch a configurable number of processes to analyze all of the CCDs of that exposure in parallel.  (Ideally, this will be as many processes as there are CCDs.  However, it works just fine with fewer, as some processes will serially run more than one CCD until all of them are done.)  Once it's done, it will contact the conductor again and ask for a new exposure to do.  (TODO: we need to configure it to be able to exit after finishing an exposure once it's processed a certain number of exposures, or after a certain runtime.  This will allow us to run it on a batch queue system where we can only allocate a job on a node for a limited amount of time.  Alternatively, we could use a different architecture where something like `pipeline_exposure_launcher` runs on a single (e.g. login) node, and launches batch jobs to process exposures it hears about.)

Currently, the pipeline does not (intentionally) use multiprocessing while working on a single chip.  Because we will have a large number of images and chips to process, it's easier to parallelize by just dividing up the work by chip rather than trying to make a single chip run as fast as possible.
