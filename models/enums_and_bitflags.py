"""
Here we put all the dictionaries and conversion functions for getting/setting enums and bitflags.
"""

from util.classproperty import classproperty

class EnumConverter:
    """Base class for creating an (effective) enum that is saved to the database as an int.

    This avoids the pain of dealing with Postgres enums and migrations.

    To use this:

    1. Create a subclass of EnumConverter (called <class> here).

    2. Define the _dict property of that subclass to have the mapping from integer to string.

    3. If not all of the strings in the _dict are allowed formats for
       this class, define a property _allowed_values with a list of
       values (strings) that are allowed.  (See, for example,
       ImageFormatConverter.)  If they are all allowed formats, you can
       instead just define _allowed_values as None.  (See, for example,
       ImageTypeConverter.)

    4. Make sure that every class has its own initialized values of
       _dict_filtered and _dict_inverse, both initialized to None.
       (This is necessary because we're using these two class variables
       as mutable variables, so we have to make sure that inheritance
       doesn't confuse the different classes with each other.)

    5. In the database model that uses the enum, create fields and properties like:

       _format = sa.Column( sa.SMALLINT, nullable=False, default=<class>.convert('<default_value>' )

       @hybrid_property
       def format(self):
           return <class>.convert( self._format )

       @format.expression
       def format(cls):
           return sa.case( <class>.dict, value=cls._format )

       @format.setter
       def format( self, value ):
           self._format = <class>.convert( value )

    6. Anywhere in code where you want to convert between the string and
       the corresponding integer key (in either direction), just call
       <class>.convert( value )

    """

    _dict = {}
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None

    @classmethod
    def c( cls, keyword ):
        """Convert the key to something more compatible. """
        return keyword.lower().replace(' ', '')

    @classproperty
    def dict( cls ):
        if cls._dict_filtered is None:
            if cls._allowed_values is None:
                cls._dict_filtered = cls._dict
            else:
                cls._dict_filtered = { k: v for k, v in cls._dict.items() if v in cls._allowed_values }
        return cls._dict_filtered

    @classproperty
    def dict_inverse( cls ):
        if cls._dict_inverse is None:
            cls._dict_inverse = { cls.c(v): k for k, v in cls._dict.items() }
        return cls._dict_inverse

    @classmethod
    def convert( cls, value ):
        """Convert between a string and corresponding integer key.

        If given a string, will return the integer key.  If given an
        integer key, will return the corresponding string.  String
        identification is case-insensitive and ignores spaces.

        If given None, will return None.

        """
        if isinstance(value, str):
            if cls.c(value) not in cls.dict_inverse:
                raise ValueError(f'{cls.__name__} must be one of {cls.dict_inverse.keys()}, not {value}')
            return cls.dict_inverse[cls.c(value)]
        elif isinstance(value, (int, float)):
            if value not in cls.dict:
                raise ValueError(f'{cls.__name__} integer key must be one of {cls.dict.keys()}, not {value}')
            return cls.dict[value]
        elif value is None:
            return None
        else:
            raise ValueError(f'{cls.__name__} must be integer/float key or string value, not {type(value)}')


class FormatConverter( EnumConverter ):
    # This is the master format dictionary, that contains all file types for
    # all data models. Each model will get a subset of this dictionary.
    _dict = {
        1: 'fits',
        2: 'hdf5',
        3: 'csv',
        4: 'json',
        5: 'yaml',
        6: 'xml',
        7: 'pickle',
        8: 'parquet',
        9: 'npy',
        10: 'npz',
        11: 'avro',
        12: 'netcdf',
        13: 'jpg',
        14: 'png',
        15: 'pdf',
        16: 'fitsldac',
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None

class ImageFormatConverter( FormatConverter ):
    _allowed_values = ['fits', 'hdf5']
    _dict_filtered = None
    _dict_inverse = None

class CutoutsFormatConverter( FormatConverter ):
    _dict = ImageFormatConverter._dict
    _allowed_values = ['fits', 'hdf5', 'jpg', 'png']
    _dict_filtered = None
    _dict_inverse = None

class SourceListFormatConverter( FormatConverter ):
    _allowed_values = ['npy', 'csv', 'hdf5', 'parquet', 'fits']
    _dict_filtered = None
    _dict_inverse = None

class ImageTypeConverter( EnumConverter ):
    _dict = {
        1: 'Sci',
        2: 'ComSci',
        3: 'Diff',
        4: 'ComDiff',
        5: 'Bias',
        6: 'ComBias',
        7: 'Dark',
        8: 'ComDark',
        9: 'DomeFlat',
        10: 'ComDomeFlat',
        11: 'SkyFlat',
        12: 'ComSkyFlat',
        13: 'TwiFlat',
        14: 'ComTwiFlat',
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None


def bitflag_to_string(value, dictionary):

    """
    Takes a 64 bit integer bit-flag and converts it to a comma separated string,
    using the given dictionary.
    If any of the bits are not recognized, will raise a ValueError.

    To use this function, you must first define a dictionary with the bit-flag values as keys,
    and the corresponding strings as values. This should include all the possible bits that could
    be encountered. For example, the badness bitflag will include the general data badness dictionary,
    as that bitflag is usually a combination of the badness of all the data models.

    If given None, will return None.

    Parameters
    ----------
    value: int or None.
        64 bit integer bit-flag.
    dictionary: dict
        Dictionary with the bit-flag values as keys, and the corresponding strings as values.

    Returns
    -------
    output: str or None
        Comma separated string with all the different ways the data is bad.
        If given None, will return None.
        If given zero, will return an empty string.
    """
    if value is None:
        return None
    elif value == 0:
        return ''
    elif isinstance(value, (int, float)):
        output = []
        for i in range(64):
            if value >> i & 1:
                if i not in dictionary:
                    raise ValueError(f'Bitflag {i} not recognized in dictionary')
                output.append(dictionary[i])
        return ', '.join(output)
    else:
        raise ValueError(f'Bitflag must be integer/float, not {type(value)}')


def string_to_bitflag(value, dictionary):
    """
    Takes a comma separated string, and converts it to a 64 bit integer bit-flag,
    using the given dictionary (the inverse dictionary).
    If any of the keywords are not recognized, will raise a ValueError.

    To use this function, you must first define a dictionary with the keywords as keys,
    and the corresponding bit-flag values as values. This should include all the possible keywords that could
    be appended to this specific data model. For example, the badness bitflag for cutouts will not include
    the keywords for images, as those will be set on the image model, not on the cutouts model.

    If given an empty string, will return zero.
    If given None, will return None.

    Parameters
    ----------
    value: str or None
        Comma separated string with all the different ways the data is bad.
    dictionary: dict
        Dictionary with the keywords as keys, and the corresponding bit-flag values as values.

    Returns
    -------
    output: int or None
        64 bit integer bit-flag.
        If given None, will return None.
        If given zero, will return an empty string.
    """

    if isinstance(value, str):
        if value == '':
            return 0
        output = 0
        for keyword in value.split(','):
            original_keyword = keyword
            keyword = EnumConverter.c(keyword)
            if keyword not in dictionary:
                raise ValueError(f'Keyword "{original_keyword}" not recognized in dictionary')
            output += 2 ** dictionary[keyword]
        return output


# these are the ways an Image or Exposure are allowed to be bad
image_badness_dict = {
    1: 'Banding',
    2: 'Shaking',
    3: 'Saturation',
    4: 'Bad Subtraction',
    5: 'Bright Sky',
}
image_badness_inverse = {EnumConverter.c(v): k for k, v in image_badness_dict.items()}

# these are the ways a Cutouts object is allowed to be bad
cutouts_badness_dict = {
    21: 'Cosmic Ray',
    22: 'Ghost',
    23: 'Satellite',
    24: 'Offset',
    25: 'Bad Pixel',
    26: 'Bleed Trail',
}
cutouts_badness_inverse = {EnumConverter.c(v): k for k, v in cutouts_badness_dict.items()}

# these are the ways a SourceList object is allowed to be bad
source_list_badness_dict = {
    41: 'X-Match Failed',
    42: 'Big Residuals',
    43: 'Few Sources',
    44: 'Many Sources',
}
source_list_badness_inverse = {EnumConverter.c(v): k for k, v in source_list_badness_dict.items()}

# join the badness:
data_badness_dict = {0: 'Good'}
data_badness_dict.update(image_badness_dict)
data_badness_dict.update(cutouts_badness_dict)
data_badness_dict.update(source_list_badness_dict)
data_badness_inverse = {EnumConverter.c(v): k for k, v in data_badness_dict.items()}
if 0 in data_badness_inverse:
    raise ValueError('Cannot have a badness bitflag of zero. This is reserved for good data.')
