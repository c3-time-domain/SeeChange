from models.enums_and_bitflags import (
    file_format_dict,
    image_type_dict,
    data_badness_dict,
)


def test_enums_zero_values():
    assert 0 not in file_format_dict
    assert 0 not in image_type_dict
    assert data_badness_dict[0] == 'Good'