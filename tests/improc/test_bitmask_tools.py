import numpy as np

from improc.bitmask_tools import dilate_bitflag


def test_bitmask_dilation():
    array = np.zeros((10, 10), dtype=np.uint16)
    array[3, 3] = 1
    array[5, 5] = 2
    array[7, 7] = 3
    array[9, 9] = 4

    dilated = dilate_bitflag(array, iterations=1)
    assert dilated.dtype == array.dtype
    assert np.all(dilated[0:1, :] == 0)
    assert np.all(dilated[:, 0:1] == 0)

    assert dilated[3, 3] == 1
    assert dilated[2, 3] == 1
    assert dilated[3, 2] == 1
    assert dilated[4, 3] == 1
    assert dilated[3, 4] == 1

    assert dilated[5, 5] == 2
    assert dilated[4, 5] == 2
    assert dilated[5, 4] == 2
    assert dilated[6, 5] == 2
    assert dilated[5, 6] == 2

    assert dilated[7, 7] == 3
    assert dilated[6, 7] == 3
    assert dilated[7, 6] == 3
    assert dilated[8, 7] == 3
    assert dilated[7, 8] == 3

    assert dilated[9, 9] == 4
    assert dilated[8, 9] == 4
    assert dilated[9, 8] == 4

    struct = np.ones((3, 3), dtype=bool)
    dilated = dilate_bitflag(array, structure=struct, iterations=1)
    assert dilated.dtype == array.dtype
    assert np.all(dilated[0:1, :] == 0)
    assert np.all(dilated[:, 0:1] == 0)

    assert dilated[3, 3] == 1
    assert dilated[2, 3] == 1
    assert dilated[3, 2] == 1
    assert dilated[4, 3] == 1
    assert dilated[3, 4] == 1

    # corners are also included with this structure
    assert dilated[2, 2] == 1
    assert dilated[2, 4] == 1
    assert dilated[4, 2] == 1
    assert dilated[5, 2] == 0  # out of the structure element's reach
    assert dilated[4, 4] == 3  # overlapping corner with 2

    assert dilated[5, 5] == 2
    assert dilated[4, 5] == 2
    assert dilated[5, 4] == 2
    assert dilated[6, 5] == 2
    assert dilated[5, 6] == 2
    assert dilated[6, 6] == 3  # overlaps 3, but 2 is bit-wise included in 3

    assert dilated[7, 7] == 3
    assert dilated[6, 7] == 3
    assert dilated[7, 6] == 3
    assert dilated[8, 7] == 3
    assert dilated[7, 8] == 3
    assert dilated[8, 8] == 7  # overlaps 4

    assert dilated[9, 9] == 4
    assert dilated[8, 9] == 4
    assert dilated[9, 8] == 4

    dilated = dilate_bitflag(array.astype('uint16'), iterations=1)
    assert dilated.dtype == array.dtype
