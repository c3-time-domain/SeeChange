# Copyright 2015 Fred Moolekamp
# BSD 3-clause license
# Modified by RKNOP in 2021, 2023 to add imghdr

"""Functions to convert FITS files or astropy Tables to FITS_LDAC files and vice versa.

The FITS LDAC format is a specific convention for FITS files for storing
catalogs:
https://marvinweb.astro.uni-bonn.de/data_products/THELIWWW/LDAC/LDAC_concepts.html

It's the format used by Emmanuel Bertin's Astromatic utilities (SExtractor et al).

This code may not fully implement the specification; it does what's
necessary to read and write SExtractor catalogs.

"""

import io
import numpy as np
from astropy.io import fits
from astropy.io.fits.hdu.table import BinTableHDU
from astropy.io.fits.header import Header


def convert_hdu_to_ldac(hdu, imghdr=None, imghdr_as_header=False):
    """Convert an hdu table to a fits_ldac table (format used by astromatic suite)

    Parameters
    ----------
    hdu: `astropy.io.fits.BinTableHDU` or `astropy.io.fits.TableHDU`
        HDUList to convert to fits_ldac HDUList

    imghdr: `astropy.io.fits.Header`, `astropy.fits.hdu.BinTableHDU`, str, or None
        What gets encoded into the LDAC_IMHEAD HDU.  Must either be a
        header, a Binary Table HDU read from an existing LDAC file, or a
        header already properly written to a string (80×nrecs
        characters, fully space padded, no separators).  If this is
        None, will use the header from hdu (which is almost certainly
        not the right thing, but you probably don't care as you're
        probably going to ignore the first return value).

    imghdr_as_header: bool, default False
        Controls the format of the first return; see Returns

    Returns
    -------
    hdr: `astropy.io.fits.Header` OR `astropy.io.fits.BinTableHDU`
        The information from the LDAC_IMHEAD binary table passed (somehow) in imghdr.
        If hdu_as_header is True, this will be an astropy.io.fits.Header.
        If hdu_as_header is False, this will be a properly formatted LDAC_IMHEAD BinTableHDU.
    tbl: `astropy.io.fits.BinTableHDU`
        Data table (LDAC_OBJECTS)

    NOTE!  Because of how it constructs tbl1, calling writeto on an
    HDULIST constructed from these tables will fail if you don't also
    pass output_verify='ignore'

    """

    # Do the easy part first: the table that's a table
    tbl2 = fits.BinTableHDU( hdu.data, header=hdu.header )
    tbl2.header["EXTNAME"] = "LDAC_OBJECTS"

    # Now deal with the LDAC mess that encodes a header as a binary table...
    if not ( ( imghdr is None ) or isinstance( imghdr, ( str, Header, BinTableHDU ) ) ):
        raise TypeError( f'imghdr must be a str, Header, or BinTableHDU, or None, not a {type(imghdr)}' )

    hdr = None
    if imghdr_as_header:
        if imghdr is None:
            hdr = hdu.header
        if isinstance( imghdr, Header ):
            hdr = imghdr
        elif isinstance( imghdr, str ):
            hdr = Header.fromstring( imghdr )
        else:
            # I suspect the FITS standard rquires ASCII, but I know
            # that at least Emmanuel Bertin uses é in some headers.
            # Since latin-1 is also a single-byte encoding, using
            # it instead of ascii shouldn't hurt anything, and might
            # increase compatibility.
            hdr = Header.fromstring( imghdr.data.tobytes().decode( 'latin-1' ) )

    else:
        if imghdr is None:
            imghdr = hdu.header
        if isinstance( imghdr, str ):
            imghdrstr = imghdr
        elif isinstance( imghdr, BinTableHDU ):
            imghdrstr = imghdr.data.tobytes().decode( 'latin-1' )
        else:
            imghdrstr = imghdr.tostring( padding=False )
        imghdrstrarr = np.array( [ imghdrstr[i:i+80] for i in range(0, len(imghdrstr), 80) ] )

        # This doesn't work; see column below
        # col = fits.Column(name="Field Header Card", format=f'{len(imghdrstr)}A',
        #                   dim=f'(80, {len(imghdrstrarr)})', array=np.array([imghdrstrarr]))
        # cols = fits.ColDefs([col])
        # tbl1 = fits.BinTableHDU.from_columns(cols)
        # tbl1.header["EXTNAME"] = "LDAC_IMHEAD"

        # I failed trying to do it the way the documentation seemed to
        # say I was supposed to.  The problem was that the "Column"
        # definition above was stripping blank spaces from the end of
        # each line in the hdrstrarr.  They got replaced with 0 bytes in
        # the table data, which was *NOT* what I wanted.  I haven't
        # figured out how to tell Column *not* to strip training spaces
        # from its strings.  (Among other things, it made the FITS
        # reader in scamp not work.  One might blame scamp for using a
        # FITS header routine to read binary table data, but I need this
        # to work.)  Looking at the astropy code, it looks like
        # fits.Column will convert it to a numpy chararay, which strips
        # whitespace, irritatingly.  I don't think there's a way to work
        # around this.  So, just do it manually; this also requires
        # adding output_verify='ignore' to the .writeto call in
        # save_table_as_ldac, because astropy FITS gets all pissy when I
        # did the thing that it documented that I could do.
        hdrfields = [ ( 'XTENSION', 'BINTABLE', True ),
                      ( 'BITPIX', 8, False ),
                      ( 'NAXIS', 2, False ),
                      ( 'NAXIS1', len(imghdrstr), False ),
                      ( 'NAXIS2', 1, False ),
                      ( 'PCOUNT', 0, False ),
                      ( 'GCOUNT', 1, False ),
                      ( 'TFIELDS', 1, False ),
                      ( 'TTYPE1', 'Field Header Card', True ),
                      ( 'TFORM1', f'{len(imghdrstr)}A  ', True ),
                      ( 'TDIM1', f'(80, {len(imghdrstrarr)})', True ),
                      ( 'EXTNAME', 'LDAC_IMHEAD', True ),
        ]
        hdr = ""
        for field in hdrfields:
            hdrline = f'{field[0]:<8s}= '
            if field[2]:
                hdrline += f'\'{field[1]}\''
            else:
                hdrline += f'{field[1]:>20n}'
            hdrline += " "*80
            hdrline = hdrline[0:80]
            hdr += hdrline
        hdr += f'{"END":<80s}'
        hdr += " " * ( 2880 - ( len(hdr) % 2880 ) )
        imghdrstr += " " * ( 2880 - ( len(imghdrstr) % 2880 ) )
        hdr = fits.BinTableHDU.fromstring( hdr.encode('latin-1') + imghdrstr.encode('latin-1') )

    return ( hdr, tbl2 )


def convert_table_to_ldac(tbl, imghdr=None):
    """Convert an astropy table to a fits_ldac

    Parameters
    ----------
    tbl: `astropy.table.Table`
        Data table to convert to ldac format
    imghdr: `astropy.io.fits.Header`, `astropy.fits.hdu.BinTableHDU`, str, or None
        Optional: image header data to store in the LDAC_IMHEAD HDU.
        See `convert_hdu_to_ldac` for details.
    Returns
    -------
    hdulist: `astropy.io.fits.HDUList`
        FITS_LDAC 3-HDU hdulist that can be read by astromatic software

    """
    # ...it seems like it ought to be easier to construct a BinTableHDU
    # from a Table, but the only way I've figured out how to do it is to
    # write with Table.write and then read back the thing that was
    # written.

    f = io.BytesIO()
    tbl.write(f, format="fits")
    f.seek(0)
    with fits.open(f, mode="update", memmap=False) as hdulist:
        tblhdu = BinTableHDU( data=hdulist[1].data, header=hdulist[1].header )
    tbl1, tbl2 = convert_hdu_to_ldac(tblhdu, imghdr=imghdr, imghdr_as_header=False)
    new_hdulist = fits.HDUList( [fits.PrimaryHDU(), tbl1, tbl2] )
    return new_hdulist


def save_table_as_ldac(tbl, filename, imghdr=None, **kwargs):
    """Save a table as a fits LDAC file

    Parameters
    ----------
    tbl: `astropy.table.Table`
        Table to save
    filename: str
        Filename to save table
    imghdr: `astropy.io.fits.Header`
        Optional: image header data to store in the LDAC
    kwargs:
        Keyword arguments to pass to hdulist.writeto
    """
    hdulist = convert_table_to_ldac(tbl, imghdr=imghdr)
    hdulist.writeto(filename, output_verify='ignore', **kwargs)


def get_table_from_ldac( filename, frame=1, imghdr_as_header=False ):
    """Load an astropy table from a fits_ldac by frame.

    Since the ldac format has column info for odd tables, giving it twce
    as many tables as a regular fits BinTableHDU, the HDU of the primary
    table info is actually twice the frame.  (I.e. frame 1 has
    LDAC_IMHEAD in HDU 1 (0-offset) and LDAC_OBJECTS in HDU 2 (with HDUs
    being 0 offset here, i.e. the first HDU in the fits file is HDU 0.)

    Parameters
    ----------
    filename: str
        Name of the file to open
    frame: int
        Number of the frame in a regular fits file
    imghdr_as_header: bool, default False
        If False, will return the LDAC_IMHEAD object as a binary table.
        If True, will return it as a Header.  (If frame=0, then there is
        no LDAC_IMHEAD.)

    Returns
    -------
    hdr, tbl
       hdr: Header or BinTableHDU; the information from the LDAC_IMHEAD HDU (or None if frame=0)
       tbl: astropy.table.Table; the information from the LDAC_OBJECTS HDU

    """
    from astropy.table import Table

    if frame > 0:
        frame = frame * 2

    with fits.open( filename ) as hdul:
        hdr, tbl = convert_hdu_to_ldac( hdul[frame], None if frame==0 else hdul[frame-1],
                                        imghdr_as_header=imghdr_as_header )
        tbl = Table( tbl.data )

    return hdr, tbl
