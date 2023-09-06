# Utilities for dealing with ra and dec

import re

import astropy.coordinates

_radecparse = re.compile( '^ *(?P<sign>[\-\+])? *(?P<d>[0-9]{1,2}): *(?P<m>[0-9]{1,2}):'
                          ' *(?P<s>[0-9]{1,2}(\.[0-9]*)?) *$' )

def parse_sexigesimal_degrees( strval ):
    """Parse [+-]dd:mm::ss to decimal degrees

    Parameters
    ----------
    strval: string
       Sexigesimal value in the form [-+]dd:mm:ss (+ may be omitted)

    Returns
    -------
    float, the value in degrees

    """

    match = _radecparse.search( strval )
    if match is None:
        raise RuntimeError( f"Error parsing {strval} for [+-]dd:mm::ss" )
    val = float(match.group('d')) + float(match.group('m'))/60. + float(match.group('s'))/3600.
    val *= -1 if match.group('sign') == '-' else 1
    return val

def radec_to_gal_and_eclip( ra, dec ):
    """Convert ra/dec to galactic and ecliptic coordinates

    Parameters
    ----------
    ra: float
      RA in decimal degrees
    dec: float
      dec in decimal degrees

    Returns
    -------
    4-elements tuple: (l, b, ecl_lon, ecl_lat) in degrees

    """
    sc = astropy.coordinates.SkyCoord( ra, dec, unit='deg' )
    gal_l = sc.galactic.l.to( astropy.units.deg ).value
    gal_b = sc.galactic.b.to( astropy.units.deg ).value
    eclipsc = sc.transform_to( astropy.coordinates.BarycentricTrueEcliptic )
    ecl_lon = eclipsc.lon.to( astropy.units.deg ).value
    ecl_lat = eclipsc.lat.to( astropy.units.deg ).value

    return ( gal_l, gal_b, ecl_lon, ecl_lat )
