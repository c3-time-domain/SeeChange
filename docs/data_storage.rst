************
Data storage
************

Data is stored in three main places: 

* The database
* The local filesystem
* The data archive

.. _provenances:
Provenances
===========

**If you don't understand provenances, you will probably do the wrong thing when you query the database.**

The concept of Provenances is fundamental to the SeeChange data model.  SeeChange is designed with the understanding that we are likely to want to run the same data through the pipeline more than once.  (We'll redo things with different parameters, or we may want to redo a year's worth of supernova searching once we have better references.)  To avoid the chaos of overwriting data when something is redone, SeeChange uses a "provenance" system to keep track of different versions of data products that come from the same original data.  Each data product produced by the pipeline is associated with a ``Provenance`` object.  A ``Provenance`` is defined (in ``models/provenance.py``) by the version of the code used to produce the data product, the name of the process (one of a set of standardized names used by the code), the parameters that configure the details of how the process runs, and, crucially, immediately upstream provenances.

Thus, when searching the database, it's not enough to search for (say) measurements of an object at a given RA an Dec in a given time range.  You *must* also specify the provenance of the measurements you wish to find.  If you don't, you may well get reudndant data, and you will probably get a mess of heterogeneous data.

For example, ``Image`` objects usually have a provenance whose process is ``preprocessing``.  The parameters include which steps were applied (bias subtraction, flatfielding, nonlinearity correction, etc.) and any configuration parameters for those steps (e.g. what set of flatfields were used).  From the image, we might run something like SExtractor to produce a ``SourceList`` object.  The ``SourceList`` provenance process is "extraction", the parameters include which code was used to generate the source list (SExtractor or something else), and the configuration settings for that code.  An extraction provenance has a preprocessing provenance as an upstream.  Even if you run exactly the same code, with exactly the same parameters, on an image, but you changed something about the way the image was preprocessed, you don't expect to get exactly the same source list out the other end!  This is why the upstream preprocessing provenance is included as part of the definition of a downstream provenance.  If you change the parameters for preprocessing, that will change the provenances for *every* step of the pipeline (as they are all downstream of preprocessing, so the provenance change will cascaed through each step).

(Technically, the parameters for a Provenance are defined by the pipeline object that performs the process.  See :doc:`development` for more information.)

Provenance Tags
---------------

In practice, provenances are treated as (mostly) opaque objects, and are a field of objects of classes associated with data products.  Every Provenance has a simple string id which can be used to refer to it, and to use when searching for objects.  These strings are hashes, so are unpleasant for humans to use.  The database also allows for the definition of "provenance tags", which are human readable strings that point to a provenacne for each process.  You might, for instance, use the "DR1" provenance tag to get all of the provenacne ids for all processes, and then use those provenance ids to get all the data products associated with "data release 1".  It's up to the managers of the pipeline to maintain which provenance tags exist and to keep track of what they mean.

  
Database
========

We use ``postgres`` as the database backend, which allows us to use the ``q3c`` extension for fast spatial queries, as well as types like JSONB for storing dictionaries.  A SeeChange instance must have a database server associated with it, and any running SeeChange code needs to be able to connect to the PostgreSQL port (5432) of that server.  If you're doing this for local development, see :doc:`development` for a self-contained docker environment that creates a throw-away database.  Alternatively, you can run PostgreSQL on your local system, or you can set up a server somewehre else you have access.

Database models are defined using SQLAlchemy, and database migrations are tracked with Alembic.  Database communication is done using a combination of SQLALchmey and direct SQL using ``psycopg``.  The database communications is defined in ``models/base.py``, where we define a ``Base`` class for all database mapped objects.  Each class that inherits from ``Base`` can have columns mapped to object attributes.

Unless you're writing one-off code that you expect to run by itself (i.e. not as part of a pipline that is mass-processing data), do not hold database connections open for a long time.  Sometimes you do want to have a connection open for several operations, either because they're linked, or because they're happening fast enough that the inefficiency of opening a new connection between the operations isn't worth it.  However, if code that runs in a pipeline designed to run many times concurrently hold open database connections for extended periods of time, we run the risk of exhausting database connection resources.  See :ref:`sqla-sessions` and :ref:`psycopg` below.

Database communcation should usually be done inside ``with SmartSession(...) as session:`` or ``with PsycopgConnection(...) as conn:`` blocks, where these two functions are defined in ``models/base.py``.  See :ref:`sqla-sessions` and :ref:`psycopg` below.

Many of our basic database functions take either a SQLAlchemy ``Session`` object, or a psycopg ``connection`` object, as an optional argument.  If you are inside a ``with`` block that holds a database connection, pass that connection to the function, and the function will use the same session or connection that you're already using.  (Otherwise, the function will open a new connection.)  For example, see ``test.py::SeeChangeBase.insert()``.  (``SeeChangeBase`` is a class that most (all?) of our database model objects inherit from, so ``insert`` (and others) are defined for all models.)  Often, however, you will not want to open your own session, but just let these functions open sessions themselves.  This is usually the case when you are doing any non-trivial computation between database calls, as the overhead of establishing a new database connection will become small compared to the computation you're doing.

.. _sqla-sessions:
SQLAlchemy sessions
-------------------

While the SQLAlchemy syntactic sugar can, in simple cases, simplify the mapping of database rows to object properties, we aren't using the full infrastructure of SQLAlchemy.  The reason is that SQLAlchemy implicitly assumes you're going to keep a database connection open as long as an object that might want to communicate with the database exists.  This is fine for web applications whose total runtime is less than a second.  It's a total disaster for a pipeline that might run for a couple of minutes, and for which there may more than a hundred copies of that pipeline all running at once.  (And, no, connection pools does not solve the issue, but in fact makes it worse because each parallel process ends up keeping its own pool, so connections don't even get closed when you close them!)  Previously, use used SQLAlchemy relations, but those caused *constant* development nightmares and debugging tangles when we didn't hold open database connections; eventually, we refactored the code to get rid of those, and it was the cause of much relief.  However, we find that there are things you want to do in PostgreSQL that the SQLAlchemy intermediary makes very difficult, and for all but the simplest queries, the SQLAlchmy syntax is just as byzantine as SQL (and frustratingly different for somebody who already knows SQL).  What's more, whereas you might have some idea what you're doing with SQL, you can't really be sure what an SQLAlchmey construction is actually going to do with the database.  Sometimes, it's very frustrating to figure out how to get SQAlchemy to *actually close* conections on the backend in a reproducible manner, meaning we were stuck with lingering locks (which only appeared intermittently, to make debugging as hard as possible) that would have been trivial to deal with if we hadn't foolishly wedded ourselves to SQLAlchemy.  As such, at some point we may refactor the code to remove SQLAlchemy entirely (to more great relief)— see `Issue #516 <https://github.com/c3-time-domain/SeeChange/issues/516>`_.  For the most part, except for simplest things, when we write new code we use direct connections with `psycopg` (see :ref:`psycopg` below).

If you want to communicate with the databse using SQLAlchemy constructions, this section describes how to get a connection.

The file ``models/base.py`` defines two SQLAlchemy session starting functions, ``Session()`` and ``SmartSession()``.  If you're using SQLAlchemy, we strongly urge developers to use ``SmartSession()``; see below.  To open a regular session, use the ``Session()`` function, and make sure to close it when you are done.  This is useful mainly for interactive use, and highly discouraged for using in the code base.  The session is a ``sqlalchemy.orm.session.Session`` object::

  from models.base import Session
  session = Session()
  # do stuff with the session
  obj = MappedObject()  # can be any of the models
  session.add(obj)
  session.commit()

  # make sure to close at the end
  session.close()

The ``Session()`` function can also be used in a context manager, which will automatically close the session at the end of the block::

  from models.base import Session
  with Session() as session:
      # do stuff with the session
      obj = MappedObject()  # can be any of the models
      session.add(obj)
      session.commit()

However, we encourage developers to use the ``SmartSession()`` function for such uses. The main difference is that a ``SmartSession`` accepts inputs that can be either another session object or None.  If the input is None, it will open a new session, which will close at the end of the context.  If the input given to it is a session object, it will not close it at the end of the context, rather it will depend on the external context that made the session to close it when done.

This is particularly useful in nested functions::

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

**Make sure not to hold open connections for a long time.**  In normal operation of the pipeline, many processes are running in parallel.  If they all hold connections open at once, they may exhaust database resources.  There is some overhead in creating a databse connection, so if you're going to do several fast operations in a row, do them within one connection.  But, if you're going to do substantial computations between two calls to the database, make sure to close the database connection in between.  This usually means exiting the ``with`` block that started the connection.  Additionally, be careful about passing a session nested functions that do substantial computation, as that will hold the database connection open throughout the running of the nested function.

.. _psycopg:
Psycopg connections
-------------------

While it's possible to do direct SQL with SQLAlchemy sessions, either with the session itself, or by (painfully) extracting the underlying database connection, if what you really need to do is direct SQL, you're probably better off with a simple psycopg connection.  If you are going to do anything that locks tables, you must do it this way.  Empirically, SQLAlchemy does not seem to actually close the underlying databse connection when you call the ``close`` method of a session, but rather markes it for closing sometime later during garbage collection.  The result is that if you lock a table using SQL passed to an SQLAlchemy session, most of the time it will work, but rarely the table will not unlock when you close the session (or exist the relevantg ``with`` block).  If you really need control over your database, use an actual database connection and save yourself the complication of another opininated layer between you and the database.

``models/base.py`` provides a function ``PsycopgConnection()`` that must be used as a context manager (i.e. in a ``with`` block).  It takes one optional argument, and returns a standard ``psycopg.connection`` object.  If that argument is ``None``, it opens a new connection to the databse, and then calls ``rollback`` and closes that connection when the context exits.  (This means that if you want any changes you made to presist, you must make sure to call ``commit`` on the connection.)  If that argument is an existing psycopg connection, it just immediately returns that, and does not automatically rollback or close it.  (In that case, whatever created the connection in the first place is responsible for doing that.)  This latter use is analgous to passing an existing SQLAlchemy session to ``SmartSession``, as described above.

As a trivial example::

  with PsycopgConnection() as conn:
      cursor = conn.cursor()
      cursor.execute( "SELECT _id FROM images WHERE mjd>60000. AND Mjd<60001." )
      foundids = [ row[0] for row in cursor.fetchall() ]

For a less trivial example, see ``models/provenance.py::ProvenanceTag.addtag()`` (which locks a table).

Defining mapped classes
-----------------------

(Users working with the code to analyze existing data can probably skip this section; this is necessary for developers who are adding to the pipeline code itself.)

The ``models`` folder contains a module for each database table, which defines a mapped class for that table.  Each mapped class inherits from the ``Base`` class defined in ``models/base.py``, and has a ``__tablename__`` attribute that defines the table name.

The class definition of each mapped class includes columns and relationships that map to the database table columns and relationships.  For example, a SourceList will have a ``num_sources`` column attribute that corresponds to a column in the database table.

**Do not use SQLAlchemy relationships**.  Previously, the code defined things like a ``image`` field of the SourceList table that use SQLAlchemy lazy loading to load the ``Image`` record associated with the source list when you accessed it.  This led to a constant stream of painful debugging of detached instances and other mysterious and hard-to-debug errors that were the result of SQLAlchemy's design assuming that you were holding a connection open throughout your entire computation (which, while reasonable for a webap that completes its computations in less than a second, is not reasonable for a data analysis pipeline that will run for two minutes).  Instead, code relationships with standard SQL foreign keys.  For example, SourceList has the ``image_id`` field, which is a foreign key pointing at the ``_id`` column of the ``images`` table (the table behind the Images model).  If you want to get the actual image data, just run::

  image = Image.get_by_id( sources.image_id )

(This method is defined in ``models/base.py::UUIDMixin.get_by_id``.)

The ``__init__()`` method of each mapped class can define other attributes that are not mapped to the database.  Also, the ``init_on_load()`` function should be defined to initialize any such attributes when an object is loaded, rather than created from scratch.  Use the ``orm.reconstructor`` decorator to define this function::

  from sqlalchemy.orm import reconstructor

  class MyObject:
      def __init__(self):
          self.my_attribute = None
        
      @reconstructor
      def init_on_load(self):
          self.my_attribute = None

This makes sure those properties are defined and initialized even when an object is not created with its ``__init__()`` function.

If you change a mapped class, or define a new mapped class, make sure also to update the database migrations.  See "Database migrations" in TODO FIGURE OUT WHERE THESE ARE, PROBABLY DEVELOPMENT.

Files on disk
=============

Some of the data products are far too large to store in PostgreSQL tables.  For example, the image data itself is stored in FITS files on disk.  Each such resource must also be mapped to a database object, so it can be queried and loaded when needed.  These files have an "official" location in the archive (see :ref:`archive` below), as well as a (potentially transitory) location in a local file store.

In ``models/base.py``, we define a ``FileOnDisk`` mixin class, that allows a database row to also store a file (or multiple files) on disk in a local directory. The mixin class also has the ability to save and load the data from the archive.  The ``FileOnDisk`` class defines a ``filepath`` attribute, which is relative to some base directory, defined by ``FileOnDisk.local_path``.  Changing the ``local_path`` allows the same database entries to correspond to different filesystems, but with identical relative paths.  The relative ``filepath`` can include subdirectories.  If the ``FileOnDisk`` subclass needs to store multiple files, it should have a ``filepath`` that includes the first part of the filename, and ``filepath_extensions`` that is an array of strings with the ends of the filenames, which are different for each file.  ``Image`` is an example of a class that does this::

  from models.image import Image
  image = Image()      # is a FileOnDisk subclass
  image.filepath = 'images/2020-01-01/some_image'
  image.filepath_extensions = ['image.fits', 'weight.fits', 'flags.fits']

To save the file to disk, use the ``save()`` method of the object.  Most of the time, the subclass will implement a ``save()`` method, which will call the ``save()`` method of the ``FileOnDisk`` class after it has actually done the saving of the data (which is specific to each subclass and the data it contains).  The ``FileOnDisk`` class will make sure the file is in the right place, will check the MD5 checksum, and will push the file to the archive (see below).  The object will generally not be saved to the database until after it has a legal ``filepath`` and MD5 checksum.  This makes sure database objects are mapping to actual files on disk.  It should be noted that those files could later be removed without the database knowing about it.

Usually, to load data from disk, you just access the relevant properties of the object.  For instance, if you have an ``Image`` object that corresponds to a pre-existing image file, just access the ``.data`` property of the ``Image`` object; if there isn't already data in memory, just accessing the property triggers it to be laoded from disk.  (It will also make a copy from the archive to the local file store if necessary.)

To remove the file from local storage only, use ``remove_data_from_disk()``.  To remove the file from local disk, archive and database, use ``delete_from_disk_and_database()``.

(From an implementation point of view, note that loading of the data is usually not done entirely in the ``FileOnDisk`` mixin, as that is specific to each subclass.  In general, we add a private ``_data`` attribute to the subclass, which is ``None`` when the object is initialized or loaded from the database.  Then a public ``data`` property will cause a lazy load of the data from disk when it is first accessed, putting it into ``_data``.  To get the full path of the file stored on disk (e.g., for loading it) use the ``FileOnDisk.get_fullpath()`` method, which attaches the ``local_path`` to the ``filepath`` attribute, and includes extensions if they exist.)


Exposures and data files
------------------------

An ``Exposure`` object encapsulates the (usually) raw data taken by an instrument/telescope at one point in time and space. It can include multiple sub-images if the instrument has multiple sections, e.g., for a mosaic camera, or a multi-channel instrument, e.g., using a dichroic.

The ``Exposure`` must have some basic properties such as the time of observation (which is represented as the modified Julian date, or MJD, when the image was taken), the coordinates the telescope was pointing at (RA and Dec), and some housekeeping information like the project/proposal under which the images were taken, the name of the object/target/field that was observed, the exposure time, and the filter used (or filter array).

The ``Exposure`` object will contain the path(s) to the underlying raw data files.  Ideally, these data files are not modified at all from when they were generated at the telescope.  Additional processing steps, such as bias/flat should be done on the next processing level (i.e., when generating ``Image`` objects) so the details of the processing can be tracked.  However, the code does support downloading pre-reduced exposures, e.g. exposures processed through the standard NOIRLab pipeline and available on the NOIRLab archive.

The most common use case for ``Exposure`` objects is to (somehow) acquire a file with data, make a new ``Exposure`` object that links to it, and then commit that object to track the file. There are many tools built into the ``Exposure`` class to read the header data to automatically fill in all the information needed to populate the ``Exposure`` object. The ``Exposure`` can then be used directly or loaded at a different time from the database.

Once linked to a file, the ``Exposure`` object's ``data`` attribute can be accessed, which will lazy download the file from server and then lazy load the image data into numpy arrays.  For example::

  from models.exposure import Exposure
  exp = Exposure(filename='Camera_Project_2021-01-01T00:00:00.000.fits')

  print(type(exp.data))  # this is a SectionData object, defined in models/exposure.py
  print(type(exp.data[0]))  # numpy array for section zero

  for section_id in exp.instrument_object.get_section_ids():
      print(exp.data[section_id].shape)  # print the shape of each section

In some cases, the data is in memory and needs to be saved to disk.  An ``Exposure`` can then be generated without a file using ``Exposure(nofile=True)``.  The user will then need to fill in the column data (such as MJD) and use some method like ``Exposure.save()`` to generate the file.

Images
------

An ``Image`` encapsulates a single two-dimensional array of data, typically from one chip (though we call it "sensor section" in the code) of an exposure.  For instances, for DECam, an ``Exposure`` has all of the data (usually raw) taken for all the chips for one pointing of the telescope.  That will be broken into 60 or 62 (modulo bad chips) different ``Image`` objects, each of which encapsulates the data for each chip.

Although we violate this principle in a few places, for the most part SeeChange is designed so that data products tracked in the database are WORM ("Write Once, Read Many").  That means that once an ``Image`` object is created, neither the data, nor the header in the data file, nor the columns of the database table row for that image, ever change again in the future.  For this reason, we perform basic preprocessing (flatfielding, etc.) before loading an ``Image`` object into the database. If you want to get access to raw image data, you have to do so through the ``Exposure`` object.  (Another side effect of this is that things like the best WCS and the best zeropoint will **not** generally be in the image headers!  Rather, you will need to get the WCS and zeropoint products separately from the database.  This is covered in :doc:`usage`.)


Exposure and image headers
--------------------------

Each ``Exposure`` object is associated with a single FITS file (or sometimes multiple files, for different sections, though at the moment the code assumes that all of an exposure is packed into a single file).  Header information is loaded from the FITS file, but this information can be accessed in a few different places.  First, and what should be used most of the time (because they should have the same meanings for exposures from any source, whereas detailed header keywords are always observatory-specific), some of the properties of the ``Exposure`` object probably originally came from the exposure file's header (e.g. ``mjd``, ``filter``, ``exp-time``).  ``Exposure`` object have a ``info`` property which is a dictionary that contains a small subset of the full FITS header. Generally only the properties we intend to query on will be saved here.  The choice of keywords that get copied into the ``info`` dictionary are the combination of the keys in ``exposure.EXPOSURE_HEADER_KEYS`` and ``Instrument.get_auxiliary_exposure_header_keys()``.  The latter is used to let each ``Instrument`` subclass to define additional keywords that are useful to have in the info dictionary.  The ``info`` dictionary is saved to the database as a JSONB column.  Since some of this information is saved as independent columns (like ``exp_time``), the ``info`` column does not necessarily keep much information (information is not duplicated).  Note that this dictionary is filled using the global ``Exposure`` header, not the header of individual sections.  The keywords in this dictionary are all lower-case, and are translated to standardized names using the ``_get_header_keyword_translations()`` method of the ``Instrument`` class. This makes it easy to tell them apart from the header information (in upper case) which also uses instrument-specific keywords.  The value of the ``info`` values are also converted to standard units using the ``_get_header_values_converters()``

In addition to the ``info`` column which is saved to the database, the ``Exposure`` also has a ``header`` and a ``section_headers`` attributes. The ``header`` is a dictionary-like object (using ``astropy.io.fits.Header``) that contains the full FITS header of the file.  This is not saved to the database, but is lazy loaded from the exposure FITS file when needed.  The headers use all upper case keywords, as per the FITS standard.  The ``section_headers`` property is a ``SectionHeaders`` object (also defined in ``models/exposure.py``) which acts like a dictionary that lazy loads the FITS header from file for a specific section when needed.  In cases where multiple section data is saved in one FITS file, there would usually be a primary HDU that contains the global exposure header information, and additional extension HDUs with their own image data and headers.  In this case the ``section_headers`` are all different from the Exposure's ``header``.  If the raw data includes a separate FITS file for each section, then each file would have a different header, and the "exposure global header" would arbitrarily be the header of the first file.  If the instrument only has one section, this is trivially true as well.

After running basic pre-processing, we split each ``Exposure`` object into one or more ``Image`` objects.  These are already section-specific, so we have fewer properties to track when looking for the data or headers.  The ``Image`` object's ``data`` property contains the pixel values (usually after some pre-processing).  In addition to the pixel values, we also keep some more data arrays relevant to the image.  These include the ``flags`` array, which is an integer bit-flag array marking things like bad pixels, the ``weight`` array, giving the inverse variance of each pixel (noise model), and additional, optional arrays like the ``score`` array which is a "match-filtered" image, normalized to units of signal-to-noise.  These arrays are all numpy arrays, and are saved to disk using the format defined in the config file.

The ``Image`` object's ``header`` property contains the section-specific header, copied directly from the ``Exposure`` object's ``section_headers`` property.  Some header keywords may be added or modified in the pre-processing step.  This header is saved to the file, and not the database.  The ``Image`` object's ``info`` attribute contains a subset of the section-specific header.  Again, this dictionary uses standardized names, in lower case, that are searchable on the database.  This ``info`` dictionary is produced during the pre-processing step and contains only the important (searchable) keywords, but not those that are important enough to get their own columns in the ``images`` table (e.g., the MJD of the image).


Exposure and Image Data on disk
-------------------------------

The ``Image`` data is saved to disk using the format defined in the config file (using the config options under ``storage.images``).  Currently, the code base only supports FITS files.  Unlike the ``Exposure`` object, which is linked to files that were created by an instrument we are not in control of, the files associated with an ``Image`` object are created by the pipeline.  We can also choose to save all the different arrays (data, weight, flags, etc.) in different files, or in the same file (using multiple extensions).  (Currently, only saving to different files is well-supported and well-tested.)  This is defined in the config file using ``storage.images.single_file``.  In either case, the additional arrays are saved with their own headers, which are all identical to the ``Image`` object's ``header`` dictionary.  (...Maybe.  To be safe, totgally ignore weight and flags headers, and only use image headers.)

Many subsequenct data products (including source lists, world coordinate systems, difference images, detections on difference images, etc.) are also stored on disk; some subsequent data projects (e.g. zeropoints) only have database records.

.. _archive:
Data archive
============

Local storage may be limited.  It may, for instance, be on a "scratch" filesystem where old files are purged, or it may be on a small enough file system that occasionally you have to clean it out manually.  What's more, you may want to run the pipeline in multiple different places that don't share the same filesystem.  For this reason, the pipeline defines an "archive" where all of the files referenced the database are supposed to exist.  The idea is that the archive may be less performant than the local filesystem.  The local filesystem can be thought of as a cache for the archive.  When an object is saved and comitted to the database, it is also pushed up to the archive; it's not considered fully saved until it exists on the archive.  (Unless you've defined your entire archive as ``null`` in the config YAML file, in which case you don't have an archive.)  When the code (in particular, the ``FileOnDisk`` mixin in ``models/base.py``) loads a file, it first looks for it on the local filesystem; if it doesn't find it, it pulls it down from the archive, and then loads it.  Optionally, if the file is found on the local filesystem, the code verifies that the md5sum of the local file matches what's in the database; if there's a mismatch, it's either an error, or the file is pulled from the archive and overwritten, based on how the archive functions are called.

Multiple different installations of SeeChange can work on the same collection of data as long as they're configured in a consistent manner, using consistent versions of the code, and point to the same database and archive.  They do not need to share a local data store.

The class that defines interaction with the archive is in ``Util/archive.py`` (which is actually a symbolic link to the same file in a submodule located in the ``extern`` subdirectory.)  Users of the pipeline will usually not need to directly run the archive, as it's all run transparently in the data saving and loading functions in the pipeline.  (However, it might be useful if you want to make sure you have a local copy of a file for, e.g., viewing with an image viewer.)  If you do want to use it, rather than instantiating an ``Archive`` object directly, call ``models/base.py::get_archive_object()``.  That will get you an archive object based on the configuration parameters on the configuration YAML file.

The data archive could be a local filesystem, for example on an external hard drive with lots of storage space, or it can be a remote server.  The archive parameters must be defined in the config YAML file, under the ``archive`` key; see the comments there for more documentation on keywords that define the archive.

In general, whenever using the ``save()`` command, specify ``no_archive=True`` to avoid writing to the archive (only to local storage).  This is useful for testing, or doing one-off calculations where you want to write data files, but are not saving records to the database.  If not specified, the archive will be used.  Whenever adding something to the databse, you must use the archive.
