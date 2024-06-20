## Overview

SeeChange is designed to be used as a pipeline and archiving tool 
for imaging sky surveys, primarily for the La Silla Schmidt Southern Survey (LS4). 

SeeChange consists of a main pipeline that takes raw images and produces a few data products:
 - Calibrated images (after preprocessing such as bias, flat, etc). 
 - Source catalogs (after source detection and photometry).
 - Point Spread Function (PSF) for the image. 
 - Calibrations of the sources to external catalogs:
   astrometric calibration in the form of a World Coordinate System (WCS) 
   and a photometric calibration in the form of a magnitude zero point (ZP). 
 - Difference images. 
 - Source catalogs from the difference images 
   (dubbed "detections" as we are mostly interested in transients).
 - Cutouts around the sources detected in the difference images, along with the corresponding 
   image cutouts from the reference and the newly acquired images. 
 - Measurements on those cutouts, including the photometric flux, the shapes, and some 
   metrics that indicate if the source is astronomical or an artefact (using analytical cuts). 

Additional pipelines for making bias frames, flat frames, and to produce deep coadded references
are described separately. 
The coadd pipeline is described in `docs/coadd_pipeline.md`.

Data is saved into a database, on the local filesystem, 
and on a data archive (usually this is a remote server). 
Saving and loading data from these locations is described in
`docs/data_storage.md`. 

To read more about developing and contributing to the project, see `docs/contribution.md`. 
To set up an instance of the pipeline to run on a local machine or on a remote server,
see `docs/setup.md`.

### Project architecture

The project is organized around two main folders, `models` and `pipeline`.
`models` contains the objects that represent the data products,
most of which are mapped (using SQL Alchemy) to a database (we use postgresql).
`pipeline` contains the code that runs the analysis and processing required to 
produce the data products. 

Additional folders include:
 - `alembic`: for migrating the database. 
 - `data`: for storing local data, either from the tests or from the actual pipeline 
   (not to be confused with the long term storing of data on the "archive"). 
 - `devshell`: docker definitions for running the pipeline in a local dockerized environment.
 - `docs`: documentation.
 - `extern`: external packages that are used by SeeChange, including the `nersc-upload-connector` 
   package that is used to connect the archive.
 - `improc: image processing code that is used by the pipeline, generally manipulating images
   in ways that are not specific to a single point in the pipeline (e.g., image alignment or inpainting).
 - `tests`: tests for the pipeline (more on that below). 
 - `utils`: generic utility functions that are used by the pipeline. 

The source code is found in `pipeline`, `models`, `improc` and `utils`.
Notable files in the `pipeline` folder include `data_store.py` (described below)
and the `top_level.py` file that defines the `Pipeline` object, 
which is the main entry point for running the pipeline.
In `models` we have the `base.py` file, which contains tools for 
database communications, along with some useful mixin classes, 
and the `instrument.py` file, which contains the `Instrument` 
base class used to define various instruments from different surveys. 


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
   This process takes an `Image` object and produces a `SourceList`, a 'PSF', a 'WorldCoordinates', 
   and a 'ZeroPoint' object. 
   The astrometric and photometric steps were integrated into "extraction" to simplify the pipeline.
   The WorldCoordinates object is a WCS solution that maps image pixel coordinates to sky coordinates.
   The ZeroPoint object is a photometric solution that maps instrumental fluxes to magnitudes.
 - subtraction: taking a reference image of the same part of the sky (usually a deep coadd)
   and subtracting it from the "new" image (the one being processed by the pipeline). 
   Different algorithms can be used to match the PSFs of the new and reference image
   (we currently implement ZOGY, but HOTPANTS and SFFT will be added later). 
 - This process uses the `Image` object, along with all the other data products 
   produced so far in the pipeline, and another `Image` object for the reference 
   (this image comes with its own set of data products) and produces a subtraction `Image` object. 
 - detection: finding the sources in the difference image. 
   This process uses the difference `Image` object and produces a `SourceList` object.
   This new source list is different from the previous one, as it contains information only
   on variable or transient sources, and does not map the constant sources in the field. 
 - cutting: this part of the pipeline identifies the area of the image around each source 
   that was detected in the previous process step, and the corresponding area in the reference 
   and the new image, and saves those stamps as a `Cutouts` object. 
   Additional pixel data could optionally be scraped from other surveys (like PanSTARRS or DECaLS).
   Each source that was detected in the difference image gets a separate `Cutouts` object. 
 - measuring: this part of the pipeline measures the fluxes and shapes of the sources 
   in the cutouts. It uses a set of analytical cuts to 
   distinguish between astronomical sources and artefacts. 
   This process uses the list of `Cutouts` objects 
   to produce a list of `Measurements` objects, one for each source. 

The final stage of the pipeline adds the new measurements to the database 
and attempts to link them to existing `Object`s. 
If no object exists (in that location on the sky), a new one is created.

Alerts are then generated (as `Alert` objects) for any new measurements, 
and the information on the associated `Object` is added to the alert 
to provide historical context.

More details on each step in the pipeline and their parameters
can be found in `docs/pipeline.md`.

### The `DataStore` object

Every time we run the pipeline, objects need to be generated and pulled from the database.
The `DataStore` object is generated by the first process that is called in the pipeline, 
and is then passed from one process to another. 

The datastore contains all the data products for the "new" image being processed, 
and also data products for the reference image (that are produced earlier by the reference pipeline).
The datastore can also be used to query for any data product on the database, 
getting the "latest" version (see below for more on versioning). 
To quickly find the relevant data, initialize the datastore using the exposure ID and the section ID, 
or using a single integer as the image ID. 

```python
from pipeline.data_store import DataStore
ds = DataStore(exposure_id=12345, section_id=1)
# or, using the image ID:
ds = DataStore(image_id=123456)
```

Note that the `Image` and `Exposure` IDs are internal database identifiers, 
while the section ID is defined by the instrument used, and usually refers
to the CCD number or name (it can be an integer or a string). 
E.g., the DECam sections are named `N1`, `N2`, ... `S1`, `S2`, etc.

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

There could be multiple versions of the same data product, 
produced with different parameters or code versions.
A user may choose to pass a `provenance` input to the `get` methods,
to specify which version of the data product is requested. 
If no provenance is specified, the provenance is loaded either
from the datastore's general `prov_tree` dictionary, or if it doesn't exist, 
will just load the most recently created provenance for that pipeline step. 

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

See below for more information about versioning using the provenance model. 


### Configuring the pipeline

Each part of the pipeline (each process) is conducted using a dedicated object. 
 - preprocessing: using the `Preprocessor` object defined in `pipeline/preprocessing.py`.
 - extraction: using the `Detector` object defined in `pipeline/detection.py` to produce the `SourceList` and `PSF` 
   objects. A sub dictionary keyed by "sources" is used to define the parameters for these objects. 
   The astrometric and photometric calibration are also done in this step.
   The astrometric calibration using the `AstroCalibrator` object defined in `pipeline/astro_cal.py`, 
   with a sub dictionary keyed by "wcs", produces the `WorldCoordinates` object. 
   The photometric calibration is done using the `PhotoCalibrator` object defined in
   `pipeline/photo_cal.py`, with a sub dictionary keyed by "zp", produces the `ZeroPoint` object.
 - subtraction: using the `Subtractor` object defined in `pipeline/subtraction.py`, producing an `Image` object.
 - detection: again using the `Detector` object, with different parameters, also producing a `SourceList` object.
 - cutting: using the `Cutter` object defined in `pipeline/cutting.py`, producing a list of `Cutouts` objects.
 - measuring: using the `Measurer` object defined in `pipeline/measuring.py`, producing a list of `Measurements` objects.

All these objects are initialized as attributes of a top level `Pipeline` object,
which is defined in `pipeline/top_level.py`. 
Each of these objects can be configured using a dictionary of parameters.

There are three ways to configure any object in the pipeline. 
The first is using a `Config` object, which is defined in `util/config.py`.
This object reads one or more YAML files and stores the parameters in a dictionary hierarchy.
More on how to initialize this object can be found in the `configuration.md` document. 
Keys in this dictionary can include `pipeline`, `preprocessing`, etc. 
Each of those keys should map to another dictionary, with parameter choices for that process.

After the config files are read in, the `Pipeline` object can also be initialized using
a hierarchical dictionary: 

```python
from pipeline.top_level import Pipeline
p = Pipeline(
   pipeline={'pipeline_par1': pl_value1, 'pipeline_par2': pl_value2}, 
   preprocessing={'preprocessing_par1': pp_value1, 'preprocessing_par2': pp_value2},
   extraction={'extraction_par1': ex_value1, 'extraction_par2': ex_value2},
   ...
)
```

If only a single object needs to be initialized, 
pass the parameters directly to the object's constructor:

```python
from pipeline.preprocessing import Preprocessor
pp = Preprocessor(
   preprocessing_par1=pp_value1, 
   preprocessing_par2=pp_value2
)
```

Finally, after all objects are initialized with their parameters, 
a user (e.g., in an interactive session) can modify any of the parameters
using the `pars` attribute of the object. 

```python
pp.pars['preprocessing_par1'] = new_pp_value1
# or
pp.pars.preprocessing_par1 = new_pp_value1
```

To get a list of all the parameters that can be modified, 
their descriptions and default values, use 

```python
pp.pars.show_pars()
```

The definition of the base `Parameters` object is at `pipeline/parameters.py`, 
but each process class has a dedicated `Parameters` subclass where the parameters
are defined in the `__init__` method. 

Additional information on using config files and the `Config` class 
can be found at `docs/configuration.md`.

### Versioning using the `Provenance` model

Each of the output data products is stamped with a `Provenance` object. 
This object tracks the code version, the parameters chosen for that processing step, 
and the provenances of the data products used to produce this product (the "upstreams"). 
The `Provenance` object is defined in `models/provenance.py`.

The `Provenance` object is initialized with the following inputs:
 - `process`: the name of the process that produced this data product ('preprocessing', 'subtraction', etc.).
 - `code_version`: the version object for the code that was used to produce this data product.
 - `parameters`: a dictionary of parameters that were used to produce this data product.
 - `upstreams`: a list of `Provenance` objects that were used to produce this data product.

The code version is a `CodeVersion` object, defined also in `models/provenance.py`.
The ID of the `CodeVersion` object is any string, but we recommend a string that is 
monotonically increasing with newer versions, such that it is easy to find the latest version.
E.g., use the date of the code version release, or semantic versioning (e.g., `v1.0.0`).
It is recommended that new code versions are added to the database for major changes in the code, 
e.g., when regression tests indicate that data products have different values with the new code. 

The parameters dictionary should include only "critical" parameters, 
as defined by the `__init__` method of the specific process object, 
and should not include auxiliary parameters like verbosity or number of processors. 
Only parameters that affect the product values are included. 

The upstreams are other `Provenance` objects defined for the data products that 
are an input to the current processing step. 
The flowchart of the different process steps is defined in `pipeline.datastore.UPSTREAM_STEPS`. 
E.g., the upstreams for the `subtraction` object are `['preprocessing', 'extraction', 'reference']`.
Note that the `reference` upstream is replaced by the provenances 
of the reference's `preprocessing` and `extraction` steps.

When a `Provenance` object has all the required inputs, it will produce a hash identifier
that is unique to that combination of inputs.
So, when given all those inputs, a user (or a datastore) can create a `Provenance` object 
and then query the database for the data product that has that provenance.
This process is implemented in the `DataStore` object, using the different `get` methods.

Additional details on data versioning can be found at `docs/versioning.md`.

### Database schema

The database interactions are managed using SQLAlchemy's object-relational model (ORM). 
Each table is mapped to a python-side object, that can be added or deleted using a session object.

It is useful to get familiar with the naming convention for different data products: 

 - `Exposure`: a single exposure of all CCDs on the focal plane, linked to raw data on disk. 
 - `Image`: a simple image, that has been preprocessed to remove bias, dark, and flat fields, etc. 
   An `Image` can be linked to one `Exposure`, or it can be linked to a list of other `Image` objects
   (if it is a coadded image) or it can be linked to a reference and new `Image` objects (if it is a difference image).
 - `SourceList`: a catalog of light sources found in an image. It can be extracted from a regular image or from a 
   subtraction. The 'SourceList' is linked to a single `Image` and will have the coordinates of all the sources detected.
 - `PSF`: a model of the point spread function (PSF) of an image. 
   This is linked to a single `Image` and will contain the PSF model for that image.
 - `WorldCoordinates`: a set of transformations used to convert between image pixel coordinates and sky coordinates. 
   This is linked to a single `SourceList` (and from it to an `Image`) and will contain the WCS information for that image.
 - `ZeroPoint`: a photometric solution that converts image flux to magnitudes. 
   This is linked to a single `SourceList` (and from it to an  `Image`) and will contain the zeropoint information for that image.
 - `Object`: a table that contains information about a single astronomical object (real or bogus), 
   such as its RA, Dec, and magnitude. Each `Object` is linked to a list of `Measurements` objects.  
 - `Cutouts`: contain the small pixel stamps around a point in the sky in a new image, reference image, and 
   subtraction image. Could contain additional, external imaging data from other surveys.
   Each `Cutouts` object is linked back to a subtraction based `SourceList`. 
 - `Measurements`: contains measurements made on the information in the `Cutouts`.  
   These include flux+errors, magnitude+errors, centroid positions, spot width, analytical cuts, etc. 
 - `Provenance`: A table containing the code version and critical parameters that are unique to this version of the data. 
   Each data product above must link back to a provenance row, so we can recreate the conditions that produced this data. 
 - `Reference`: An object that links a reference `Image` with a specific field/target, a section ID, 
   and a time validity range, that allows users to quickly identify which reference goes with a new image. 
 - `CalibratorFile`: An object that tracks data needed to apply calibration (preprocessing) for a specific instrument.
   The calibration could include an `Image` data file, or a generic non-image `DataFile` object. 
 - `DataFile`: An object that tracks non-image data on disk for use in, e.g., calibration/preprocessing. 
 - `CatalogExcerpt`: An object that tracks a small subset of a large catalog, 
   e.g., a small region of Gaia DR3 that is relevant to a specific image.
   The excerpts are used as cached parts of the catalog, that can be reused for multiple images with the same pointing. 
 

#### Additional classes

The `Instrument` class is defined in `models/instrument.py`.
Although it is not mapped to a database table, it contains important tools
for processing images from different instruments.
For each instrument we define a subclass of `Instrument` (e.g., `DECam`) 
which defines the properties of that instrument and methods for loading the data, 
reading the headers, and other instrument-specific tasks.

More on instruments can be found at `docs/instruments.md`.

Mixins are defined in `models/base.py`. 
These are used as additional base classes for some of the models 
used in the pipeline, to give them certain attributes and methods
that are shared across many classes. 

These include:
 - `AutoIDMixin`: adds a unique integer ID that auto increments with each new object.
   This is the default behavior for most of the mapped classes, 
   such that each row has an integer primary key using an internal ID.
   Some objects will use a different ID, which corresponds to some meaningful string
   or hash value (e.g., the provenance hash is also its ID). 
 - `FileOnDisk`: adds a `filepath` attribute to a model
   (among other things), which make it possible to save/load the file
   to the archive, to find it on local storage, and to delete it.
 - `SpatiallyIndexed`: adds a right ascension and declination attributes to a model,
   and also adds a spatial index to the database table. 
   This is used for many of the objects that are associated with a point on the celestial sphere,
   e.g., an `Image` or an `Object`.
 - `FourCorners`: adds the RA/dec for the four corners of an object, 
   describing the bounding box of the object on the sky.
   This is particularly useful for images but also for catalog excerpts, 
   that span a small region of the sky.
 - `HasBitFlagBadness`: adds a `_bitflag` and `_upstream_bitflag` columns to the model.
   These allow flagging of bad data products, either because they are bad themselves, or 
   because one of their upstreams is bad. It also adds some methods and attributes to access
   the badness like `badness` and `append_badness()`. 
   If you change the bitflag of such an object, and it was already used to produce downstream products, 
   make sure to use `update_downstream_badness()` to recursively update the badness of all downstream products.

Enums and bitflag are stored on the database as integers
(short integers for Enums and long integers for bitflags).
These are mapped when loading/saving each database object
using a set of dictionaries defined in `models/enums_and_bitflags.py`. 


More information on data storage and retrieval can be found at `docs/data_storage.md`.


### Parallelization

The pipeline is built around an assumption that 
large surveys have a natural way of segmenting their imaging data
(e.g., by CCDs). 
We also assume the processing of each section is independent of the others. 
Thus, it is easy to parallelize the pipeline to work with each section separately.

Ideally, a single CPU running a single python process will be used for each section. 
If the same section is used on the same local machine, some caching can occur
(e.g., loading the instrument data for that section) and the processing will be slightly faster.
If the same region of the sky is processed on the same machine, 
some caching of cross-match catalogs also helps speed things up. 

When running on a cluster/supercomputer, there is usually an abundance of CPU cores, 
so running multiple sections at once, or even multiple exposures (each with many sections), 
is not a problem, and simplifies the processing. 

Additional parallelization can be achieved by using multi-threaded code
on specific bottlenecks in the pipeline, but this is not yet implemented.
The reason is that image sections are typically not very large, 
and we generally manage to keep the processing times lower than the exposure 
time of typical astronomical instruments. 
Thus, with more CPU cores than CCD sections, we can process the data in real time, 
without needing to call additional threads for each process. 