
## Instruments

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

### SensorSections

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

### Adding a new instrument

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

### Same instrument, different telescope (or configuration)

To be added... 