// Namespace, which is the only thing exported

var seechange = {};

// utils

seechange.nullorfixed = function( val, num ) { return val == null ? null : val.toFixed(num); };
seechange.nullorexpo = function( val, num ) { return val == null ? null : val.toExponential(num); };

// Keep this synced with models/enums_and_bitflags.py::ImageTypeConverter
seechange.imagetypes = {
    0: 'Unknown',
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
    18: 'DiffWarped',
    19: 'ComDiffWarped',
    20: 'ExternComSci',
    21: 'ExternComDiff',
    22: 'ExternComBias',
    23: 'ExternComDark',
    24: 'ExternComDomeFlat',
    25: 'ExternComSkyFlat',
    26: 'ExternComTwiFlat',
    27: 'ExternComWarped',
    28: 'ExternComDiffWarped'
};

export { seechange };
