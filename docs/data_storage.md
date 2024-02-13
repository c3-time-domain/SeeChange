## Data storage

Data is stored in three main places: 
 - The database
 - The local filesystem
 - The data archive (e.g., NERSC)

### Database

We use `postgres` as the database backend,
which allows us to use the `q3c` extension for fast spatial queries, 
as well as types like JSONB for storing dictionaries.
This needs to be set up as a service on the local machine,
or as a separate container in a docker-compose environment.
See the `docs/setup.md` file for more details.

Database communications is done using SQLAlchemy, 
by opening and closing sessions that talk to the database, 
and by defining classes that map to database tables.

The database communications is defined in `models/base.py`, 
where we define a `Base` class for all database mapped objects. 
Each class that inherits from it can have columns and relationships
mapped to object attributes.
The `base` module also defines session opening functions
for database communications. 
See the documentation for SQLAlchemy for more details.

#### Sessions and smart sessions

The `base` module also defines two types of session
starting functions, `Session()` and `SmartSession()`. 
To open a regular session, use the `Session()` function, 
and make sure to close it when you are done. 
This is useful mainly for interactive use, 
and highly discouraged for using in the code base. 
The session is a `sqlalchemy.orm.session.Session` object. 

```python
from models.base import Session
session = Session()
# do stuff with the session
obj = MappedObject()  # can be any of the models
session.add(obj)
session.commit()

# make sure to close at the end
session.close()
```

The `Session()` function can also be used in a context manager,
which will automatically close the session at the end of the block.

```python
from models.base import Session
with Session() as session:
    # do stuff with the session
    obj = MappedObject()  # can be any of the models
    session.add(obj)
    session.commit()
```

However, we encourage developers to use the `SmartSession()` function 
for such uses. The main difference is that a smart session accepts inputs
that can be either another session object or None. 
If the input is None, it will open a new session,
which will close at the end of the context. 
If the input given to it is a session object, 
it will not close it at the end of the context, 
rather it will depend on the external context that 
made the session to close it when done. 

This is particularly useful in nested functions:
```python
from models.base import SmartSession

def my_function(session=None):
    with SmartSession(session) as session:
        # do stuff with the session
        obj = MappedObject()  # can be any of the models
        session.add(obj)
        session.commit()

# call the function with a self-contained session
my_function()  # session opens and closes inside the function scope
        
# call the function with an externally created session
with SmartSession() as session:
    my_function(session)
    # session will be closed here, not in the function 
```

#### Defining mapped classes

The `models` folder contains a module for each database table,
which defines a mapped class for that table.
Each mapped class inherits from the `Base` class defined in `models/base.py`,
and has a `__tablename__` attribute that defines the table name.

The class definition of each mapped class includes columns and relationships
that map to the database table columns and relationships.
For example, a SourceList will have a `num_sources` column attribute 
that corresponds to a column in the database table.
It also has a relationship attribute to the `images` table, 
such that it has an `Image` type attribute loaded from the database.

The `__init__()` method of each mapped class can define other attributes
that are not mapped to the database. 
Also, the `init_on_load()` function should be defined 
to initialize any such attributes when an object is loaded, 
rather than created from scratch.
Use the `orm.reconstructor` decorator to define this function.

```python   
from sqlalchemy.orm import reconstructor

class MyObject:
    def __init__(self):
        self.my_attribute = None
        
    @reconstructor
    def init_on_load(self):
        self.my_attribute = None
```

This makes sure those properties are defined and initialized
even when an object is not created with its `__init__()` function.

### Files on disk

Some of the data products are too large to keep fully in the database. 
For example, the image data itself is stored in FITS files on disk. 
Each such resource must also be mapped to a database object, 
so it can be queried and loaded when needed. 

In `models/base.py`, we define a `FileOnDisk` mixin class,
that allows a database row to also store a file (or multiple files)
on disk in a local directory. The mixin class will also have the ability
to save and load the data from a remote archive. 

The `FileOnDisk` class defines a `filepath` attribute,
which is relative to some base directory, defined by `FileOnDisk.local_path`. 
Changing the `local_path` allows the same database entries to correspond
to different filesystems, but with similar relative paths.
The relative `filepath` can include subfolders. 

If the `FileOnDisk` subclass needs to store multiple files, 
it should have a `filepath` that includes the first part of the filename,
and `filepath_extensions` that is an array of strings with the ends
of the filenames, which are different for each file.

```python
from models.image import Image
image = Image()  # is a FileOnDisk subclass
image.filepath = 'images/2020-01-01/some_image'
image.filepath_extensions = ['image.fits', 'weight.fits', 'flags.fits']
```

To save the file to disk, use the `save()` method of the object.
Most of the time, the subclass will implement a `save()` method, 
which will call the `save()` method of the `FileOnDisk` class
after it has actually done the saving of the data
(which is specific to each subclass and the data it contains).
The `FileOnDisk` class will make sure the file is in the right place, 
will check the MD5 checksum, and will push the file to the archive
(see below). 
The object will generally not be saved to the database until after
it has a legal `filepath` and MD5 checksum. 
This makes sure database objects are mapping to actual files on disk. 
It should be noted that those files could later be removed without
the database knowing about it. 

To remove the file from local storage only, use `remove_data_from_disk()`. 
To remove the file from local disk, archive and database, use
`delete_from_disk_and_database()`. 

Note that loading of the data is not done in the `FileOnDisk` mixin, 
as that is specific to each subclass. 
In general, we add a private `_data` attribute to the subclass,
which is `None` when the object is initialized or loaded from the database. 
Then a public `data` property will cause a lazy load of the data from disk
when it is first accessed, putting it into `_data`. 
To get the full path of the file stored on disk (e.g., for loading it)
use the `FileOnDisk.get_fullpath()` method, which attaches the `local_path`
to the `filepath` attribute, and includes extensions if they exist. 


#### Exposures and data files

An `Exposure` object is a reference to the raw data taken by an instrument/telescope
at one point in time and space. It can include multiple sub-images if the instrument
has multiple sections, e.g., for a mosaic camera, or a multi-channel instrument, 
e.g., using a dichroic. 

The `Exposure` must have some basic properties such as the time of observation
(which is represented as the modified Julian date, or MJD, when the image was taken), 
the coordinates the telescope was pointing at (RA and Dec),
and some housekeeping information like the project/proposal under which the images were taken, 
the name of the object/target/field that was observed, 
the exposure time, and the filter used (or filter array). 

The `Exposure` object will contain the filename and path to the underlying raw data files. 
Ideally, these data files are not modified at all from when they were generated at the telescope. 
Additional processing steps, such as bias/flat should be done on the next processing level
(i.e., when generating `Image` objects) so the details of the processing can be tracked.

The most common use case for `Exposure` objects is to get a file with data,
make a new `Exposure` object that links to it, and then commit that object
to track the file. There are many tools built into the `Exposure` class
to read the header data to automatically fill in all the information needed
to populate the `Exposure` object. The `Exposure` can then be used directly
or loaded at a different time from the database. 

Once linked to a file, the `Exposure` object's `data` attribute can be accessed, 
which will lazy download the file from server and then lazy load the image 
data into numpy arrays.
For example: 

```python
from models.exposure import Exposure
exp = Exposure(filename='Camera_Project_2021-01-01T00:00:00.000.fits')

print(type(exp.data))  # this is a SectionData object, defined in models/exposure.py
print(type(exp.data[0]))  # numpy array for section zero

for section_id in exp.instrument_object.get_section_ids():
    print(exp.data[section_id].shape)  # print the shape of each section
```

In some cases, the data is in memory and needs to be saved to disk. 
An `Exposure` can then be generated without a file using `Exposure(nofile=True)`. 
The user will then need to fill in the column data (such as MJD) and use 
some method like `Exposure.save()` to generate the file. 


#### Exposure and image headers

Each `Exposure` object is associated with a single FITS file (or sometimes multiple files, for different sections). 
Header information is loaded from the FITS file, but this information can be kept in three different places. 
The first is the `info` property of the `Exposure` object. This is a dictionary that contains
a small subset of the full FITS header. Generally only the properties we intend to query on will be saved here. 
The choice of keywords that get copied into the `info` dictionary are the combination of the keys 
in `exposure.EXPOSURE_HEADER_KEYS` and `Instrument.get_auxiliary_exposure_header_keys()`. 
The latter is used to let each `Instrument` subclass to define additional keywords that
are useful to have in the info dictionary. 
The `info` dictionary is saved to the database as a JSONB column.
Since some of this information is saved as independent columns (like `exp_time`), the `info` column does not 
necessarily keep much information (information is not duplicated). 
Note that this dictionary is filled using the global `Exposure` header, not the header of individual sections.
The keywords in this dictionary are all lower-case, and are translated to standardized names using the
`_get_header_keyword_translations()` method of the `Instrument` class. This makes it easy to tell them apart
from the header information (in upper case) which also uses instrument-specific keywords. 
The value of the `info` values are also converted to standard units using the `_get_header_values_converters()`

In addition to the `info` column which is saved to the database, the `Exposure` also has a `header` and
a `section_headers` attributes. The `header` is a dictionary-like object (using `astropy.io.fits.Header`) 
that contains the full FITS header of the file.  
This is not saved to the database, but is lazy loaded from the file when needed. 
The headers use all upper case keywords.  
The `section_headers` property is a `SectionHeaders` object (also defined in `models/exposure.py`) 
which acts like a dictionary that lazy loads the FITS header from file for a specific section when needed.
In cases where multiple section data is saved in one FITS file, there would usually be a primary HDU that contains
the global exposure header information, and additional extension HDUs with their own image data and headers. 
In this case the `section_headers` are all different from the Exposure's `header`. 
If the raw data includes a separate FITS file for each section, 
then each file would have a different header, 
and the "exposure global header" would arbitrarily be the header of the first file. 
If the instrument only has one section, this is trivially true as well. 

After running basic pre-processing, we split each `Exposure` object into one or more `Image` objects. 
These are already section-specific, so we have less properties to track when looking for the data or headers. 
The `Image` object's `data` property contains the pixel values (usually after some pre-processing). 
In addition to the pixel values, we also keep some more data arrays relevant to the image. 
These include the `flags` array, which is an integer bit-flag array marking things like bad pixels,
the `weight` array, giving the inverse variance of each pixel (noise model), 
and additional, optional arrays like the `score` array which is a "match-filtered" image, 
normalized to units of signal-to-noise. 
These arrays are all numpy arrays, and are saved to disk using the format defined in the config file. 

If the point spread function (PSF) of the image is calculated, it can be stored in the `psf` attribute. 
If sources are extracted into a `SourceList` object this can be saved into a `sources` attribute. 
The same is true for astrometric solution saved into a `WorldCoordinates` object in `wcs`
and a photometric solution saved into a `ZeroPoint` object in `zp`.

The `Image` object's `header` property contains the section-specific header, copied directly from 
the `Exposure` object's `section_headers` property. 
Some header keywords may be added or modified in the pre-processing step. 
This header is saved to the file, and not the database. 
The `Image` object's `info` attribute contains a subset of the section-specific header. 
Again, this dictionary uses standardized names, in lower case, that are searchable on the database. 
This `info` dictionary is produced during the pre-processing step and contains only the important (searchable) keywords, 
but not those that are important enough to get their own columns in the `images` table
(e.g., the MJD of the image). 

The `Image` data is saved to disk using the format defined in the config file (using `storage.images.format`). 
Unlike the `Exposure` object, which is linked to files that were created by an instrument we are not in control of, 
the files associated with an `Image` object are created by the pipeline. 
We can also choose to save all the different arrays (data, weight, flags, etc.) in different files, 
or in the same file (using multiple extensions). This is defined in the config file using `storage.images.single_file`. 
In either case, the additional arrays are saved with their own headers, which are all identical to the `Image` object's
`header` dictionary. 


### Data archive

The data archive could be a local filesystem, for example on an external hard drive 
with lots of storage space, or it can be a remote server. 
The archive parameters must be defined in the config YAML file,
under the `archive` key.

If using a local folder, or even a networked folder that is mounted locally, 
the only thing to define is a path to the base folder 
(underneath it, the `filepath` should locate the file in the archive). 

When using a remote server, a few more parameters are needed, 
and some code must be in place to allow the transfer of files.
In the case of LS4, we use the NERSC archive, 
where communications are done using the `nersc-upload-connector` package, 
which is a submodule that resides in the `extern` folder. 
The code can be found at <https://github.com/c3-time-domain/nersc-upload-connector>. 

In general, whenever using the `save()` command, specify `no_archive=True` 
to avoid writing to the archive (only to local storage). 
If not specified, the archive will be used.

In the other direction, `FileOnDisk` will look for the underlying file
on the local filesystem first, and only if it is not found,
it will try to load it from the archive.
This happens when requesting the filename using `get_fullpath()`. 
To disable this automatic download, use `download=False`. 
Note that `FileOnDisk` lazy downloads the file from the archive, 
but does not, by itself, load it into memory. 
This is the responsibility of the subclass. 
`FileOnDisk` simply supplies the full path to the file
(while silently making sure it is downloaded from the archive), 
and the subclass can use that path to load the data.

### Relationships 

Some of the data models (mapped objects)
are related to other objects across tables
(and sometimes in the same table). 
This means that when an object with a relationship is 
added to the session (i.e., saved to database) it will
automatically save the related objects as well. 
It means that the related objects are often loaded along with the object. 
It also means that sometimes deleting an object will trigger a deletion 
of related objects. 

In general, we put relationships to objects that must exist 
for the current object to exist. 
For example, a `SourceList` has an `image` attribute
that is a relationship. The `SourceList` has been created
from exactly one `Image`, and cannot exist without it.
If the `Image` is deleted, the `SourceList` should be deleted as well.
When loading the `SourceList`, the image is also loaded at the same time. 
We do not lazy-load the related classes in most instances, 
as this causes an error if the related object is accessed outside a session scope. 
Note that loading the `Image` object just pulls down the table row from the database, 
including the file path to the image data, but does not also load the data. 
That would be lazy loaded when accessing the `data` attribute of the `Image` object. 

On the other hand, the `Image` has a `source_list` attribute wich is not
defined as a relationship. This is because the `Image` can exist without
a `SourceList` (e.g., if it has not been processed yet).
Furthermore, it is possible to have multiple `SourceList` objects
that were created from the same `Image` object. 
Each one of these `SourceList` objects will have a different provenance
(see below), and will be stored in a different table row.
It is not trivial to know which `SourceList` object is the one
that should be associated with the `Image`, so this attribute is 
filled either manually by the code or, more likely, by the `DataStore` 
object that manages the models when the pipeline is working.
This becomes useful when passing an `Image` object along
with the related objects like a source list, a zero point, 
a WCS, and a PSF. 


### DataStore and Provenances

For a short discussion of the `Provenance` model, 
see the `docs/overview.md` file (under "Versioning using the `Provenance` model")
For a more in-depth discussion of versioning, see `docs/versioning.md`. 

TBA

