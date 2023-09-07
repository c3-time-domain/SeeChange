# Utilities for dealing with ra and dec

import re

import astropy.coordinates

_radecparse = re.compile( '^ *(?P<sign>[\-\+])? *(?P<d>[0-9]{1,2}): *(?P<m>[0-9]{1,2}):'
                          ' *(?P<s>[0-9]{1,2}(\.[0-9]*)?) *$' )

def parse_sexigesimal_degrees( strval, hours=False, **kwargs ):
    """Parse [+-]dd:mm::ss to decimal degrees in the range [0, 360) or (-180, 180]

    Parameters
    ----------
    strval: string
       Sexigesimal value in the form [-+]dd:mm:ss (+ may be omitted)
    hours: bool
       If True, strval is hh:mm:ss ; parse that to degrees.  Implies positive.
    positive: bool
       If True, deg is in the range [0, 360).  If False, deg is in the
       range (180, 180].  Defaults to False, unless hours is true, in
       which case positive defaults to True.

    Returns
    -------
    float, the value in degrees

    """

    keys = list( kwargs.keys() )
    if ( keys != [ 'positive' ] ) and ( keys != [] ):
        raise RuntimeError( f'parse_sexigesimal_degrees: unknown keyword arguments '
                            f'{[ k for k in keys if k != "positive"]}' )
    positive = kwargs['positive'] if 'positive' in keys else hours

    match = _radecparse.search( strval )
    if match is None:
        raise RuntimeError( f"Error parsing {strval} for [+-]dd:mm::ss" )
    val = float(match.group('d')) + float(match.group('m'))/60. + float(match.group('s'))/3600.
    val *= -1 if match.group('sign') == '-' else 1
    val *= 15. if hours else 1.
    if positive:
        while val < 0: val += 360.
        while val >= 360: val -= 360.
    else:
        while val > 180: val -= 360.
        while val <= -180: val += 360.
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
