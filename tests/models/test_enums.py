import pytest

from models.enums_and_bitflags import (
    FormatConverter,
    ImageFormatConverter,
    CutoutsFormatConverter,
    SourceListFormatConverter,
    ImageTypeConverter,
    data_badness_dict,
)


def test_enums_zero_values():
    assert 0 not in FormatConverter.dict
    assert 0 not in ImageTypeConverter.dict
    assert data_badness_dict[0] == 'good'


def test_converter_dict():
    # Probably should test them all, but test just these
    #  three and trust that if it works, then the inheritance
    #  is working for all of them.

    assert ImageTypeConverter.dict == {
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
    assert FormatConverter.dict == {
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
    assert ImageFormatConverter.dict == { 1: 'fits', 2: 'hdf5' }


def test_converter_convert():
    for cls in ( ImageFormatConverter, CutoutsFormatConverter, SourceListFormatConverter, ImageTypeConverter ):
        for key, val in cls.dict.items():
            assert cls.convert( key ) == val
            assert cls.convert( val ) == key
            assert cls.convert( val.lower().replace( ' ', '' ) ) == key

    # Check a few bad ones

    with pytest.raises( ValueError, match='.*must be one of' ):
        val = ImageFormatConverter.convert( 'non existent format' )

    with pytest.raises( ValueError, match='.*integer key must be one of' ):
        val = ImageFormatConverter.convert( -1 )

    with pytest.raises( ValueError, match='.*must be integer/float key' ):
        val = ImageFormatConverter.convert( [] )
