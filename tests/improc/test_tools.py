from astropy.io import fits
from improc.tools import strip_wcs_keywords


def test_strip_wcs_keywords():
    hdr = fits.Header()

    keeps = {
        'NAXIS': 2,
        'NAXIS1': 1024,
        'NAXIS2': 2048,
        'AKW': 0,
        'A_1': 1,
        'PV1_1A': 3,
        'APb1_2': 4,
        'CDELT0': 5
    }
    loses = {
        'CRVAL1': 180.,
        'CRVAL2': -2.,
        'CRPIX1': 512.,
        'CRPIX2': 1024.,
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CTYPE1': 'something',
        'CTYPE2': 'something',
        'CD1_1': 1.,
        'CD1_2': 0.,
        'CD2_1': 0.,
        'CD2_2': 1.,
        'PC1_1': 1.,
        'PC1_2': 0.,
        'PC2_1': 0.,
        'PC2_2': 1.,
        'A_ORDER': 3,
        'AP_ORDER': 3,
        'B_ORDER': 3,
        'BP_ORDER': 3,
    }
    for i in range(0,2):
        for j in range(0,7):
            loses[f'PV{i}_{j}'] = 1.
            loses[f'AP_{i}_{j}'] = 1.
            loses[f'A_{i}_{j}'] = 1.
            loses[f'BP_{i}_{j}'] = 1.
            loses[f'B_{i}_{j}'] = 1.


    for kw, val in keeps.items():
        hdr[kw] = val
    for kw, val in loses.items():
        hdr[kw] = val

    strip_wcs_keywords(hdr)

    hdrkeys = [ k for k in hdr.keys() ]  # Oh, astropy
    for kw in loses.keys():
        assert kw not in hdrkeys

    for kw, val in keeps.items():
        assert hdr[kw] == val
