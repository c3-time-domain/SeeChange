# Utilities for dealing with ra and dec

import re

from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
import astropy.units as u

_radecparse = re.compile( r'^ *(?P<sign>[\-\+])? *(?P<d>[0-9]{1,2}): *(?P<m>[0-9]{1,2}):'
                          r' *(?P<s>[0-9]{1,2}(\.[0-9]*)?) *$' )


def parse_sexigesimal_degrees( strval, hours=False, positive=None ):
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
    if positive is None:
        positive = hours

    match = _radecparse.search( strval )
    if match is None:
        raise ValueError( f"Error parsing {strval} for [+-]dd:mm::ss" )
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
    sc = SkyCoord( ra, dec, unit='deg' )
    gal_l = sc.galactic.l.to( u.deg ).value
    gal_b = sc.galactic.b.to( u.deg ).value
    eclipsc = sc.transform_to( BarycentricTrueEcliptic )
    ecl_lon = eclipsc.lon.to( u.deg ).value
    ecl_lat = eclipsc.lat.to( u.deg ).value

    return ( gal_l, gal_b, ecl_lon, ecl_lat )


def parse_ra_deg_to_hms(ra):
    """
    Convert an RA in degrees to a string in sexagesimal format (in hh:mm:ss).
    """
    if ra < 0 or ra > 360:
        raise ValueError("RA out of range.")
    ra /= 15.0  # convert to hours
    return f"{int(ra):02d}:{int((ra % 1) * 60):02d}:{((ra % 1) * 60) % 1 * 60:05.2f}"


def parse_dec_deg_to_dms(dec):
    """
    Convert a Dec in degrees to a string in sexagesimal format (in dd:mm:ss).
    """
    if dec < -90 or dec > 90:
        raise ValueError("Dec out of range.")
    return (
        f"{int(dec):+03d}:{int((dec % 1) * 60):02d}:{((dec % 1) * 60) % 1 * 60:04.1f}"
    )


def parse_ra_hms_to_deg(ra):
    """
    Convert the input right ascension from sexagesimal string (hh:mm:ss format) into a float of decimal degrees.

    """
    if not isinstance(ra, str):
        raise ValueError(f"RA ({ra}) is not a string.")
    c = SkyCoord(ra=ra, dec=0, unit=(u.hourangle, u.degree))
    ra = c.ra.value  # output in degrees

    if not 0.0 < ra < 360.0:
        raise ValueError(f"Value of RA ({ra}) is outside range (0 -> 360).")

    return ra


def parse_dec_dms_to_deg(dec):
    """
    Convert the input declination from sexagesimal string (dd:mm:ss format) into a float of decimal degrees.
    """
    if not isinstance(dec, str):
        raise ValueError(f"Dec ({dec}) is not a string.")

    c = SkyCoord(ra=0, dec=dec, unit=(u.degree, u.degree))
    dec = c.dec.value  # output in degrees

    if not -90.0 < dec < 90.0:
        raise ValueError(f"Value of dec ({dec}) is outside range (-90 -> +90).")

    return dec

def radec_to_gal_ecl( ra, dec ):
    """Convert ra and dec to galactic and ecliptic coordinates.

    Parameters
    ----------
      ra : float
        RA in decimal degrees.

      dec : float
        Dec in decimal degreese

    Returns
    -------
    gallat, gallon, ecllat, ecllon

    """
    coords = SkyCoord(ra, dec, unit="deg", frame="icrs")
    gallat = float(coords.galactic.b.deg)
    gallon = float(coords.galactic.l.deg)
    ecllat = float(coords.barycentrictrueecliptic.lat.deg)
    ecllon = float(coords.barycentrictrueecliptic.lon.deg)

    return gallat, gallon, ecllat, ecllon
