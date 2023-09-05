# SeeChange
A time-domain data reduction pipeline (e.g., for handling images->lightcurves) for surveys like DECam and LS4


## Database schema

The database interactions are managed using SQLAlchemy's object-relational model (ORM). 
Each table is mapped to a python-side object, that can be added or deleted using a session object.

It is useful to get familiar with the naming convention for different pieces of data: 

- `Exposure`: a single exposure of all CCDs on the focal plane, linked to raw data on disk. 
- `Image`: a simple image, that has been processed to remove bias, dark, and flat fields.
  An `Image` can be linked to one `Exposure`, or it can be linked to a list of other 'Image' objects
  (if it is a coadded image) and it can be linked to a reference and new image (if it is a difference image).
- `SourceList`: a catalog of light sources found in an image. It can be extracted from a regular image or from a subtraction. 
  The 'SourceList' is linked to a single `Image` and will have the coordinates of all the sources detected.
- `WorldCoordinates`: a set of transformations used to convert between image pixel coordinates and sky coordinates. 
  This is linked to a single `Image` and will contain the WCS information for that image.
- `ZeroPoint`: a photometric solution that converts image flux to magnitudes. 
  This is linked to a single `Image` and will contain the zeropoint information for that image.
- `Object`: a table that contains information about a single astronomical object (real or bogus), 
  such as its RA, Dec, and magnitude. Each `Object` is linked to a list of `Sighting`s. 
- `Cutouts`: contain the small pixel stamps around a point in the sky in one image, linked to a single `Sighting`.
  Will contain links to disk locations of three cutouts: the reference image, the new image, and the difference image.
- `Measurement`: contains measurements made on the information in the `Cutouts`.  
  These include flux+errors, magnitude+errors, centroid positions, spot width, machine learning scores, etc. 
- `Provenance`: A table containing the code version and critical parameters that are unique to this version of the data. 
  Each data product above must link back to a provenance row, so we can recreate the conditions that produced this data. 

## Development notes

### Installing Docker

At the moment, some of the things below will not work if you install Docker Desktop.  
It has to do with permissions and bind-mounting system volumes; 
because of how Docker Desktop works, the files inside the container all end up owned as root, 
not as you, even if they are owned by you on your own filesystem.  Hopefully there's a way to fix this, 
but in the mean time, install Docker Engine instead of Docker Desktop; instructions are here:

- Installing Docker Engine : https://docs.docker.com/engine/install/
- Setting up rootless mode (so you don't have to sudo everything) : https://docs.docker.com/engine/security/rootless/

### Tests

To run the tests on your local system in an environment that approximates how they'll be run on github, 
cd into `tests` and run the following command (which requires the "docker compose CLI plugin" installed to work):
```
   export GITHUB_REPOSITORY_OWNER=<yourname>
   docker compose build
   USERID=<uid> GROUIP=<gid> docker compose run runtests
```
where you replace `<uid>` and `<gid>` with your own userid and groupid; if you don't do this, the tests will run, 
but various pycache files will get created in your checkout owned by root, which is annoying. 
At the end, `echo $?`; if 0, that's a pass, if 1 (or anything else not 0), that's a fail.  
(The output you see to the screen should tell you the same information.)  
This will take a long time the first time you do it, as it has to build the docker images, 
but after that, it should be fast (unless the Dockerfile has changed for either image).  
The variable GITHUB_RESPOSITORY_OWNER must be set to *something*; it only matters if you try to push or pull the images.  
Try setting it to your github username, though if you really want to push and pull you're going to have to look up 
making tokens on github.  (The docker-compose.yaml file is written to run on github, which is why it includes this variable.)

After the test is complete, run
```
    docker compose down
```
(otherwise, the postgres container will still be running).

### Development shell -- local database

To create a database on your local machine and get a development shell in which to run code, cd into the `devshell` directory and run
```
   USERID=<UID> GROUPID=<GID> docker compose up -d
```
replacing `<UID>` with your uid, and `<GID>` with your GID.  
You can avoid typing this all the time by creating a file `.env` in the `devshell` directory with contents
```
  USERID=<UID>
  GROUPID=<GID>
```
again replacing `<UID>` and `<GID>` with the right things (you can find your uid and gid with `id -u` and `id -g`). 

Once things are started, there will be two containers running, 
one for your database, one to which you can attach with a shell.  Do
```
   docker ps
```
to see what the created containers are named; the shell container is usually `devshell-seechange-1`.  
You can then get a shell inside that environment with
```
   docker exec -it devshell-seechange-1 /bin/bash
```
and you're in.  (Put in the right name for the container if it's not devshell-seechange-1.)

This docker image bind-mounts your seechange checkout 
(the parent directory of the `devshell` directory where you're working) at `/seechange`.  
That means if you work in that directory, it's the same as working in the checkout.  
If you edit something outside the container, 
the differences will be immediately available inside the container (since it's the same physical filesystem).  
This means there's no need to rebuild the container every time you change any bit of code.

When you're done, exit the container, and run
```
  USERID=<UID> GROUPID=<GID> docker compose down
```
to stop and delete the container images.  (If you created the `.env` file mentioned above, 
you don't need the USERID and GROUPID definitions, and can just type `docker compose down`.)

The `docker-compose.yaml` file in this directory defines a volume where postgres stores its data.  
This means that every time you restart the environment, 
the database will still be as it was before.  
This isn't what you want for running tests, but it's often what you want for development.  
You can see what volumes are defined with
```
  docker volume list
```
In that list, you should see something that has a name `devshell_seechange-postgres-dbdata`.  
If you want to wipe this volume out and start with a fresh database, you can run
```
  docker volume rm devshell_seechange-postgres-dbdata
```

### Development shell -- using an external existing database

TBD

### Database migrations

Database migrations are handled with alembic.

If you've just created a database and want to initialize it with all the tables, run
```
  alembic upgrade head
```

After editing any schema, you have to create new database migrations to apply them.  Do this by running something like:
```
  alembic revision --autogenerate -m "<put a short comment here>"
```
The comment will go in the filename, so it should really be short.  
Look out for any warnings, and review the created migration file before applying it (with `alembic upgrade head`).

## Usage:

Here we give some instructions for using and extending the code. 

### Exposures and data files

An Exposure object is a reference to the raw data taken by an instrument/telescope
at one point in time and space. It can include multiple images if the instrument
has multiple sections, e.g., for a mosaic camera, or a multi-channel instrument, 
e.g., using a dichroic. 

The Exposure must have some basic properties such as the time of observation
(which is represented as the modified Julian date, or MJD when the image was taken), 
the coordinates the telescope was pointing at (RA and Dec),
and some housekeeping information like the project/propsal under which the images were taken, 
the name of the object/target/field that was observed, 
the exposure time, and the filter used (or filter array). 

The Exposure object will contain the filename and path to the underyling raw data files. 
Ideally these data files are not modified at all from when they were generated at the telescope. 
Additional processing steps, such as bias/flat should be done on the next processing level
(i.e., when generating Image objects) so the details of the processing can be tracked.
The Exposure's `filename` column will contain information on the name and path of the data
files relative to some data root folder. The root folder can be changed using the config
(e.g., when moving a mount point) and can contain multiple root folders for storing
data on a server and locally. The `filename` is always relative to that, so the database
entry remains valid even if the data is moved.
For multiple-section instruments, there would be multiple data arrays saved to disk. 
These can be saved in one file (e.g., using a multi-extension FITS file) or in multiple files.
In that case, the `filename_extensions` will contain an array of strings, 
each containing the last part of the filename that completes the partial filename in `filename`.
These can be normal file extensions:

```
filename = 'Camera_Project_2021-01-01T00:00:00.000.fits'
filename_extensions = ['.image', '.mask', '.weight'] 
```

or they can be arbitrary parts of the filename, or even entire filenames, 
leaving the `filename` to contain just the path:

```
filename = 'path/to/files/Camera_Project_'
filename_extensions = ['image.fits', 'mask.fits', 'weight.fits'] 
```

To get the full path to the file, use the `get_fullpath()` method. 
This will return a string with the local path to the file. 
Note that this method will try to download the file from the server
if it doesn't exist locally. 
The path to the server root folder is defined in the config, under `path.server_data`.
The local root folder is defined under `path.data_root`. 
If the file is not found locally or on the server, this command raises a `FileNotFoundError`.
To prevent downloading, use the `get_fullpath(download=False)` method.
If the Exposure has `filename_extensions`, the `get_fullpath()` method will return
a list of strings, even if there is only one extension. 
To consistently get a list of strings, use `get_fullpath(as_list=True)`.

The most common use case for Exposure objects is to get a file with data,
make a new Exposure object that links to it, and then commit that object
to track the file. There are many tools built into the Exposure class
to read the header data to automatically fill in all the information needed
to populate the Exposure object. The Exposure can then be used directly
or loaded at a different time from the database. 

Once linked to a file, the Exposure's `data` attributes can be accessed, 
which will lazy download the file from server and then lazy load the image 
data into numpy arrays.
For example: 

```python
import matplotlib.pyplot as plt 
from models.exposure import Exposure
exp = Exposure(filename='Camera_Project_2021-01-01T00:00:00.000.fits')
plt.show(exp.data[0])
```

In some cases, the data is in memory and needs to be saved to disk. 
An exposure can then be generated without a file using `Exposure(nofile=True)`. 
The user will then need to fill in the column data (such as MJD) and use 
some method like `Exposure.save()` to generate the file. 

The `header` column of the Exposure table is a JSONB column (a dictionary)
that contains a small subset of the original header data. 
The choice of keywords that get copied into the header are the combination of the keys 
in `exposure.EXPOSURE_HEADER_KEYS` and `Instrument.get_auxiliary_exposure_header_keys()`. 
The latter is used to let each Instrument subclass to define additional keywords that
are useful to have in the header. 
Note that this header contains only global exposure information, 
not information about the individual images (e.g., from different CCD chips).
That information is saved in the Image object header. 
In addition, many of the classical header keywords such as exposure time and MJD
are stored as column-attributes of the Exposure, where they are indexed and searchable. 
These are not saved again in the header. 
The header keywords are saved in lower-case, to match the names of the column attributes, 
and to emphasize that these are uniform across all instruments, 
whereas uppercase keywords are usually seen in the raw header files (e.g., FITS files).

### Instruments

An Exposure also keeps track of the names of the instrument and telescope
that were used to make the exposure. 
Each instrument name corresponds to a subclass of Instrument. 
Since each instrument would have different implementations on how to load
data and read headers, the code for each instrument is kept in a separate class. 
Each instrument has a `read_header()` and `load_section_image()` methods 
to interact with files made by that instrument. 

Instruments also keep a record of the properties of the device, 
such as the gain, read noise, and so on. 
Each Exposure will have an `instrument` attribute with the name of the instrument, 
and an `instrument_object` attribute which lazy loads an instance of the Instrument class.
This instrument object is shared in memory across all Exposures that use the same instrument, 
using the `instrument.get_instrument_instance()` method, 
which caches instruments by name in the `instrument.INSTRUMENT_INSTANCE_CACHE` dictionary.

Note that currently the telescope properties, including the optical system,
are saved in the instrument class. This is simply to save having to define 
additional objects. Since the same instrument can sometimes be used on different telescopes,
the system admin can choose to make multiple instruments on different telescopes 
by subclassing that instrument class (see more details below).

#### SensorSections

Some instruments contain multiple sections, e.g., CCD chips or channels. 
To allow instruments to have different properties for each section, 
the SensorSection class is used.
Each Instrument object has one or more SensorSection objects,
each of which can override some or all of the properties of the instrument. 
For example, the DemoInstrument has `gain=2.0`, 
but it can have a SensorSection with `gain=2.1` which keeps a more accurate record of the gain. 
It should be noted that in cases where the value is critical for processing the exposure
(such as the case for image gain), the value should be read from the file header. 
The Instrument class, in that case, is only used as a general reference. 
In cases where some data is missing from the header, the Instrument data can
be used as a fallback (first using the SensorSection data, and only then the global Instrument data). 

If multiple sections exist on an Instrument, they could have different properties
across the sections. When instantiating an Instrument object, 
the user must call `fetch_sections()` to populate a dictionary of sections. 
Then each property can be queried using `get_property(section_id, prop)` 
to get the value of `prop` for the section with `section_id`.
If that section has `None` for that property, 
the global value from the Instrument object is returned instead. 

Sensor sections can also be used to track changes in the instrument over time. 
For example, a bad CCD can be replaced, so that at some point in time the 
read noise or gain of the section can change. 
To accommodate these changes, SensorSections can be saved to the database, 
optionally with a `validity_start` and `validity_end` dates. 
The full signature would then be `fetch_sections(session, dateobs)`, 
which will query the database for sections that are valid during 
the time of the observation. The `dateobs` can be a `datetime` or 
`astropy.time.Time` object, the MJD, or a string in the 
format `YYYY-MM-DDTHH:MM:SS.SSS`.
If no sections are found on the database, they are generated using
the Instrument subclass `_make_new_section()` method. 
This defaults to the subclass hard coded values, which is usually
what is needed for most instruments where there are no dramatic changes
in the properties of the sections.

To add sections to the database, edit the properties of the 
relevant sections and then call `commit_sections(session, validity_start, validity_end)`. 
The start/end dates would apply to all sections that do not already have validity values. 
The user can thus apply a uniform validity range or manually add validity dates to each 
section individually. 
Once committed, these new sections are saved in the `sensor_sections` table and
will be loaded using `fetch_sections()`, if the validity dates match the observation date.
Note that calling `fetch_sections()` without a date will default to current time. 
When working with a specific Exposure object, 
calling `exp.update_instrument(session)` will call `fetch_sections()` 
with the Exposure object's MJD as the observation date. 
Exposures loaded from the database will automatically have their instrument
updated when loaded.

#### Adding a new instrument

To add a new instrument, create a subclass of the Instrument class. 
Some of the methods in the Instrument should be left alone (e.g., `fetch_sections()`), 
some must be overriden, and some are optionally overriden or expanded. 

The methods that must be overriden for the new Instrument to function properly are:
 - `__init__`: must define the properties of the instrument and telescope. 
   At the end of the method, call the `super().__init__()` method to initialize the Instrument
   and add the new instrument to the list of registered instruments. 
 - `get_section_ids`: this gives a list of the sensor section IDs. 
   Since each instrument can have a different number of sections, 
   and a different naming convention, this function is fairly general. 
   Simple examples can be `return [0]` for a single section instrument, 
   or `return range(10)` for a 10-CCD instrument with integer section IDs.
   A more general case could be `return ['A', 'B', 'C']`, which highlights 
   the fact the section IDs can be strings, not only integers. 
 - `check_section_id`: verify the input section ID is of the correct type and in range. 
 - `_make_new_section`: make a new section with hard coded properties. 
   If any of the properties are identical across all sections, leave them as `None`. 
   If the properties are different but known in advance, this method will be used
   to fill them up for each section ID, using a lookup table or data file. 
 - `get_section_offsets`: the geometric layout of the instrument's sections.  
   if each section does not define an `offset_x` and `offset_y`, these values 
   need to be globally defined for the instrument. Since even a global offset table
   needs to have a different value for each section, this method returns the `(offset_x, offset_y)`
   for the given section_id. Instruments that have a single section can return `(0, 0)`.
 - `get_section_filter_array_index`: the same as `get_section_offsets` only will return
   the global value of `filter_array_index` for the given section_id.
   This is only relevant for instruments with a filter array (e.g., LS4) where different
   sections of the instruments are located under different parts of the filter array.
   E.g., if the array is `['R', 'V', 'R', 'I']`, then some sections under the `V` filter
   would have `filter_array_index=1`, and so on. 
   Instruments without a filter array do not need to use this method. 
 - `load_section_image`: the actual code to load the image data for a section of the instrument. 
   The default Instrument class will raise a `NotImplementedError` exception.
   (TODO: need to add a default FITS reader).
 - `read_header`: the actual code to read the header data from file. This reads only the global header, 
   not the individual header info for each section.
   (TODO: add a generic FITS reader). This function returns a dictionary of header keywords and values, 
   but does not attempt to parse or normalize the keywords. 
 - `get_auxiliary_exposure_header_keys`: a list of additional keywords that should be added to the Exposure
   header column. These are lower-case strings that contain important information which is specific to the instrument. 
 - `get_filename_regex`: return a list of regular expression patterns to search in the Exposure's filename. 
   These expressions help quickly match the correct instrument based on the format of the filename.
   This process occurs in the `guess_instrument()` method of the `instrument` module. 
 - `_get_header_keyword_translations`: return a dictionary that translates the uniform column names and header keys
   (all lower case) with the raw header keywords (usually upper case). 
   The Instrument base class defines a generic dictionary but subclasses can augment or replace any of these translations.
   Note that each raw header keyword is first passed through the `normalize_keyword()` function before comparing it
   to the various "translations". This includes making it uppercase and removing spaces and underscores. 
 - `_get_header_values_converters`: a dictionary of keywords (lowercase) and lambda functions that convert the 
   raw header data into the correct units. For example if the specific instrument tracks exposure time in milliseconds, 
   then `{'exp_time': lambda x: x/1000}` will convert the raw header value into seconds.
   The Instrument base class returns an empty dictionary for this method, but additional entries can be added 
   by the subclasses if needed. 

Some examples for subclassing the Instrument base class are given in the `instrument.py` file in the  `models` folder, 
and in the `test_instrument.py` file in the `tests/models` folder. 

#### Same instrument, different telescope (or configuration)

To be added... 



### Image data and headers

A really important requirement from this pipeline is to make the data quickly accessible. 
So where is the data stored and how to get it quickly? 

Each Exposure object is associated with a single FITS file (or sometimes multiple files, for different sections). 
To get the imaging data for an Exposure, simply call the `data` property. This dictionary-like object will
provide a numpy array for each section of the instrument:

```python
exp = Exposure('path/to/file.fits')
print(type(exp.data))  # this is a SectionData object, defined in models/exposure.py
print(type(exp.data[0]))  # numpy array for section zero

for section_id in exp.instrument_object.get_section_ids():
    print(exp.data[section_id].shape)  # print the shape of each section
```

The `data` property is a SectionData object, which acts like a dictionary 
that lazy loads the data array from the FITS file when needed
(in most cases these will be FITS files, but other formats can be added just as well). 
For single-section instruments, `data[0]` will usually be good enough. 
When there are several sections, use `exp.instrument_object.get_section_ids()` to get the list of section IDs. 
Note that these could be integers or strings, but the SectionData can use either type. 

Header information is also loaded from the FITS file, but this information can be kept in three different places. 
The first is the `header` property of the Exposure object. This is a dictionary-like object that contains
a small subset of the full FITS header. Generally only the properties we intend to query on will be saved here. 
Since some of this information is given as independent columns (like `exp_time`), the `header` column does not 
necessarily keep much information beyond that. Note that this header is filled using the global header, 
not the header of individual sections.
The keywords in this header are all lower-case, and are translated to standardized names using the
`_get_header_keyword_translations()` method of the Instrument class. This makes it easy to tell them apart
from the raw header information (in upper case) which also uses instrument-specific keywords. 
The value of the header cards are also converted to standard units using the `_get_header_values_converters()`

In addition to the `header` column which is saved to the database, the Exposure also has a `raw_header` and
a `section_headers` properties. The `raw_header` is a dictionary-like object that contains the full FITS header
of the file. This is not saved to the database, but is lazy loaded from the file when needed. 
The raw headers use all upper case keywords. They are saved with the file on disk, and are not kept in the database. 
The `section_headers` property is a SectionHeaders object (also defined in `models/exposure.py`) 
which acts like a dictionary that lazy loads the FITS header from file for a specific section when needed. 
Note that the "global" header could be the same as the header of the first section. 
This usually happens if the raw data includes a separate FITS file for each section. 
Each one would have a different raw header, and the "exposure global header" would arbitrarily be the header of the 
first file. If the instrument only has one section, this is trivially true as well. 
In cases where multiple section data is saved in one FITS file, there would usually be a primary HDU that contains
the global exposure header information, and additional extension HDUs with their own image data and headers. 
In this case the `section_headers` are all different from the `raw_header`. 

After running basic pre-processing, we split each Exposure object into one or more Image objects. 
These are already section-specific, so we have less properties to track when looking for the data or headers. 
The Image object's `data` property contains the pixel values (usually after some pre-processing). 
In addition to the pixel values, we also keep some more data arrays relevant to the image. 
These include the `flags` array, which is an integer bit-flag array marking things like bad pixels,
the `weight` array, giving the inverse variance of each pixel (noise model), 
and additional, optional arrays like the `score` array which is a "match-filtered" image, 
normalized to units of signal-to-noise. 
If the point spread function (PSF) of the image is calculated, it can be stored in the `psf` property.
These arrays are all numpy arrays, and are saved to disk using the format defined in the config file. 

The Image object's `raw_header` property contains the section-specific header, copied directly from 
the Exposure's `section_headers` property. Some header keywords may be added or modified in the pre-processing step. 
This header is saved to the file, and not the database. 
The Image object's `header` property contains a subset of the section-specific header. 
Again this header uses standardized names, in lower case, that are searchable on the database. 
This header is produced during the pre-processing step and contains only the important (searchable) keywords. 

The Image data is saved to disk using the format defined in the config file (using `storage.images.format`). 
Unlike the Exposure object, which is linked to files that were created by an instrument we are not in control of, 
the files associated with an Image object are created by the pipeline. 
We can also choose to save all the different arrays (data, weight, flags, etc.) in different files, 
or in the same file (using multiple extensions). This is defined in the config file using `storage.images.single_file`. 
In either case, the additional arrays are saved with their own headers, which are all identical to the Image object's
`raw_header` dictionary. 


