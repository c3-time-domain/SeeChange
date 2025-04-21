"""Here we put all the dictionaries and conversion functions for getting/setting enums and bitflags."""

from util.classproperty import classproperty


class EnumConverter:
    """Base class for creating an (effective) enum that is saved to the database as an int.

    This avoids the pain of dealing with Postgres enums and migrations.

    To use this:

    1. Create a subclass of EnumConverter (called <class> here).

    2. Define the _dict property of that subclass to have the mapping from integer to string.

    3. If not all the strings in the _dict are allowed formats for
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
        """Convert the key to something more compatible.

        ...compatible with _what_?  Why no underscores??
        """
        return keyword.lower().replace(' ', '').replace('_', '')

    @classproperty
    def dict( cls ):  # noqa: N805
        if cls._dict_filtered is None:
            if cls._allowed_values is None:
                cls._dict_filtered = cls._dict
            else:
                cls._dict_filtered = { k: v for k, v in cls._dict.items() if v in cls._allowed_values }
        return cls._dict_filtered

    @classproperty
    def dict_inverse( cls ):  # noqa: N805
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

    @classmethod
    def to_int(cls, value):
        if isinstance(value, int):
            return value
        else:
            return cls.convert(value)

    @classmethod
    def to_string(cls, value):
        if isinstance(value, str):
            return value
        else:
            return cls.convert(value)


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
        17: 'fitsfz',
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None


class ImageFormatConverter( FormatConverter ):
    _allowed_values = ['fits', 'fitsfz', 'hdf5']
    _dict_filtered = None
    _dict_inverse = None


class CutoutsFormatConverter( FormatConverter ):
    _dict = ImageFormatConverter._dict
    _allowed_values = ['fits', 'hdf5', 'jpg', 'png']
    _dict_filtered = None
    _dict_inverse = None


class SourceListFormatConverter( EnumConverter ):
    _dict = {
        1: 'sepnpy',
        2: 'sextrfits',
        3: 'filter',  # when manually constructing a source table from a matched-filter image (e.g., on the subtraction)
    }
    _allowed_values = None
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
        15: 'Fringe',
        16: 'Warped',
        17: 'ComWarped',
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None


class CatalogExcerptFormatConverter( FormatConverter ):
    _allowed_values = [ 'fitsldac' ]
    _dict_filtered = None
    _dict_inverse = None


class CatalogExcerptOriginConverter( EnumConverter ):
    _dict = {
        0: 'unknown',
        1: 'gaia_dr3'
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None


class CalibratorTypeConverter( EnumConverter ):
    _dict = {
        0: 'unknown',
        1: 'zero',
        2: 'dark',
        3: 'flat',
        4: 'fringe',
        5: 'illumination',
        6: 'linearity',
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None


class CalibratorSetConverter( EnumConverter ):
    _dict = {
        0: 'unknown',
        1: 'externally_supplied',
        2: 'general',             # A calib file you built yourself and use for a long time
        3: 'nightly'              # A calib built each night (or for a very limited mjd range)
    }


class FlatTypeConverter( EnumConverter ):
    _dict = {
        0: 'unknown',
        1: 'externally_supplied',
        2: 'sky',
        3: 'twilight',
        4: 'dome'
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None


class PSFFormatConverter( EnumConverter ):
    _dict = {
        0: 'unknown',
        1: 'psfex',
        2: 'delta',
        3: 'gaussian',
        4: 'image'
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None


class BackgroundFormatConverter( EnumConverter ):
    _dict = {
        0: 'scalar',
        1: 'map',
        2: 'polynomial',
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None


class BackgroundMethodConverter( EnumConverter ):
    _dict = {
        0: 'zero',
        1: 'sep',
        2: 'sextr',
        3: 'iter_sextr',
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None


class DeepscoreAlgorithmConverter( EnumConverter ):
    # These algorithms are implemented in models/deepscore.py
    _dict = {
        0: 'random',  # for testing only
        1: 'allperfect', # for testing only
        2: 'RBbot-quiet-shadow-131-cut0.55',
    }
    _allowed_values = None
    _dict_filtered = None
    _dict_inverse = None


def bitflag_to_string(value, dictionary):
    """Convert 64-bit bitflag into a comma separated string.

    Takes a 64-bit integer bit-flag and converts it to a comma separated
    string, using the given dictionary.  If any of the bits are not
    recognized, will raise a ValueError.

    To use this function, you must first define a dictionary with the bit-flag values as keys,
    and the corresponding strings as values. This should include all the possible bits that could
    be encountered. For example, the badness bitflag will include the general data badness dictionary,
    as that bitflag is usually a combination of the badness of all the data models.

    If given None, will return None.

    NOTE : flags images are stored as 16-bit unsigned integers.  To be
    safe, only use the first 15 bits, and then there won't be issues if
    there are signed/unsigned conversions.

    Parameters
    ----------
    value: int or None.
        64-bit integer bit-flag.

    dictionary: dict
        Dictionary with the bit-flag values as keys, and the
        corresponding strings as values.

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
    """Takes a comma separated string, and converts it to a 64-bit integer bit-flag.

    Uses the given dictionary (the inverse dictionary).  If any of the
    keywords are not recognized, will raise a ValueError.

    To use this function, you must first define a dictionary with the
    keywords as keys, and the corresponding bit-flag values as
    values. This should include all the possible keywords that could be
    appended to this specific data model. For example, the badness
    bitflag for cutouts will not include the keywords for images, as
    those will be set on the image model, not on the cutouts model.

    If given an empty string, will return zero.  If given None, will
    return None.

    NOTE : flags images are stored as 16-bit unsigned integers.  To be
    safe, only use the first 15 bits, and then there won't be issues if
    there are signed/unsigned conversions.

    Parameters
    ----------
    value: str or None
        Comma separated string with all the different ways the data is
        bad.

    dictionary: dict
        Dictionary with the keywords as keys, and the corresponding
        bit-flag values as values.

    Returns
    -------
    output: int or None
        64-bit integer bit-flag.
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
                raise ValueError(f'Keyword "{original_keyword.strip()}" not recognized in dictionary')
            output += 2 ** dictionary[keyword]
        return output


# bitflag for image preprocessing steps that have been done
image_preprocessing_dict = {
    0: 'overscan',
    1: 'zero',
    2: 'dark',
    3: 'linearity',
    4: 'flat',
    5: 'fringe',
    6: 'illumination'
}
image_preprocessing_inverse = {EnumConverter.c(v):k for k, v in image_preprocessing_dict.items()}


# these are the ways an Image or Exposure are allowed to be bad
image_badness_dict = {
    1: 'banding',
    2: 'shaking',
    3: 'saturation',
    4: 'bad subtraction',
    5: 'bright sky',
}
image_badness_inverse = {EnumConverter.c(v): k for k, v in image_badness_dict.items()}


# these are all the ways a PSF object is allowed to be bad
psf_badness_dict = {
    11: 'psf fit failed',
}
psf_badness_inverse = {EnumConverter.c(v): k for k, v in psf_badness_dict.items()}


# these are the ways a SourceList object is allowed to be bad
source_list_badness_dict = {
    16: 'few sources',
    17: 'many sources',
}
source_list_badness_inverse = {EnumConverter.c(v): k for k, v in source_list_badness_dict.items()}


# these are the ways a WorldCoordinates/ZeroPoint object is allowed to be bad
# mostly due to bad matches to the catalog
catalog_match_badness_dict = {
    21: 'no catalog',
    22: 'x-match failed',
    23: 'big residuals',
}
catalog_match_badness_inverse = {EnumConverter.c(v): k for k, v in catalog_match_badness_dict.items()}


# these are the ways a Background object is allowed to be bad
# TODO: need to consider what kinds of bad backgrounds we really might have
# TODO: make sure we are not repeating the same keywords in other badness dictionaries
bg_badness_dict = {
    31: 'too dense',
    32: 'bad fit',
}
bg_badness_inverse = {EnumConverter.c(v): k for k, v in bg_badness_dict.items()}


# These are the ways a Reference object may be bad
reference_badness_dict = {
    35: 'ref is bad',
    36: 'ref is superceded'
}
reference_badness_inverse = {EnumConverter.c(v): k for k, v in reference_badness_dict.items()}


# these are the ways a Measurements object is allowed to be bad
measurements_badness_dict = {
    41: 'cosmic ray',
    42: 'ghost',
    43: 'satellite',
    44: 'offset',
    45: 'bad pixel',
    46: 'bleed trail',
}
measurements_badness_inverse = {EnumConverter.c(v): k for k, v in measurements_badness_dict.items()}


# join the badness:
data_badness_dict = {}
data_badness_dict.update(image_badness_dict)
data_badness_dict.update(measurements_badness_dict)
data_badness_dict.update(source_list_badness_dict)
data_badness_dict.update(psf_badness_dict)
data_badness_dict.update(bg_badness_dict)
data_badness_dict.update(catalog_match_badness_dict)
data_badness_dict.update(bg_badness_dict)
data_badness_dict.update(reference_badness_dict)
data_badness_inverse = {EnumConverter.c(v): k for k, v in data_badness_dict.items()}
if 0 in data_badness_inverse:
    raise ValueError('Cannot have a badness bitflag of zero. This is reserved for good data.')


class BadnessConverter( EnumConverter ):
    _dict = data_badness_dict
    _allowed_values = data_badness_dict
    _dict_filtered = None
    _dict_inverse = None


# bitflag for image preprocessing steps that have been done
image_preprocessing_dict = {
    0: 'overscan',
    1: 'zero',
    2: 'dark',
    3: 'linearity',
    4: 'flat',
    5: 'fringe',
    6: 'illumination'
}
image_preprocessing_inverse = {EnumConverter.c(v):k for k, v in image_preprocessing_dict.items()}


# bitflag used in flag images
# Stored as 16-bit integers, only use bits 0 through 14
flag_image_bits = {
    0: 'bad pixel',        # Bad pixel flagged by the instrument
    1: 'zero weight',
    2: 'saturated',
    3: 'out of bounds',     # caused by alignment (swarp etc)
}
flag_image_bits_inverse = { EnumConverter.c(v):k for k, v in flag_image_bits.items() }


class BitFlagConverter( EnumConverter ):
    _dict = flag_image_bits
    _allowed_values = flag_image_bits
    _dict_filtered = None
    _dict_inverse = None


# the list of possible processing steps from a section of an exposure up to measurements, r/b scores, and report
# Used by reports
process_steps_dict = {
    1: 'preprocessing',   # creates an Image from a section of the Exposure
    2: 'extraction',      # creates a SourceList, PSF, and Background from an Image
    4: 'astrocal',        # creates a WorldCoordinates from the SourceList and GAIA catalogs
    5: 'photocal',        # creates s ZeroPoint from the SourceList and GAIA catalogs
    6: 'subtraction',     # creates a subtraction Image
    7: 'detection',       # creates a SourceList from a subtraction Image
    8: 'cutting',         # creates Cutouts from a subtraction Image
    9: 'measuring',       # creates Measurements from Cutouts
    10: 'scoring',        # creates DeepScore from Measurements
    11: 'fakeanalysis',   # inject fakes, resubtract, tabluate what's found
    12: 'alerting',       # send alerts
    30: 'finalize'
}
process_steps_inverse = {EnumConverter.c(v): k for k, v in process_steps_dict.items()}


# the list of objects that could be loaded to a datastore after running the pipeline
pipeline_products_dict = {
    1: 'image',
    2: 'sources',
    3: 'psf',
    4: 'bg',
    5: 'wcs',
    6: 'zp',
    7: 'sub_image',
    8: 'detections',
    9: 'cutouts',
    10: 'measurement_set',
    11: 'deepscore_set',
    25: 'fakes',
    26: 'fakeanal'
}

pipeline_products_inverse = {EnumConverter.c(v): k for k, v in pipeline_products_dict.items()}
