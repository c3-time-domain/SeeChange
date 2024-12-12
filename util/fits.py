import io
import pathlib
import subprocess

from astropy.io import fits
from util.logger import SCLogger


def read_fits_image(filename, ext=None, output='data'):
    """Read a standard FITS file's image data and header.

    Assumes this file doesn't have any complicated data structure.
    For more complicated data files, allow each Instrument subclass
    to use its own methods instead of this.  (...maybe?)

    Parameters
    ----------
    filename: str or Path
        The full path to the file.

    ext: int or str or None
        The extension number or name to read.  If None (default), will
        find the first image extension (defined here as something with
        NAXIS=2 and NAXIS1 and NAXIS2 both positive).

    For files with a single
        extension, the default 0 is fine.  For .fz FITS files, if this
        is an integer, will actually read one plus this number (i.e. the
        extension of the FITS file if it were funpacked first).

    output: str
        Choose which part of the file to read:
        'data', 'header', 'both'.

    Returns
    -------
    The return value depends on the value of output:
    data: np.ndarray (optional)
        The image data.
    header: astropy.io.fits.Header (optional)
        The header of the image.
    If both are requested, will return a tuple (data, header)

    """

    with fits.open(filename, memmap=False) as hdul:
        if ext is None:
            ext = 0
            while ext < len(hdul):
                hdr = hdul[ext].header
                if ( hdr['NAXIS'] == 2) and ( hdr['NAXIS1'] > 0 ) and ( hdr['NAXIS2'] > 0 ):
                    break
                ext += 1
            if ext >= len(hdul):
                raise RuntimeError( f"Failed to find an image HDU in {filename}" )

        if output in ['data', 'both']:
            data = hdul[ext].data
            # astropy will read FITS files as big-endian
            # But, the sep library depends on native byte ordering
            # So, swap if necessary
            if not data.dtype.isnative:
                data = data.astype( data.dtype.name )

        if output in ['header', 'both']:
            header = hdul[ext].header

    if output == 'data':
        return data
    elif output == 'header':
        return header
    elif output == 'both':
        return data, header
    else:
        raise ValueError(f'Unknown output type "{output}", use "data", "header" or "both"')


def save_fits_image_file( filename,
                          data,
                          header,
                          extname=None,
                          overwrite=True,
                          single_file=False,
                          fpack=False,
                          lossless=False,
                          fpackq=None,
                          just_update_header=False ):
    """Save a single dataset (image data, weight, flags, etc) to a FITS file.

    The header should be the raw header, with some possible adjustments,
    including all the information (not just the minimal subset saved
    into the DB).

    Using the single_file=True option means that the pipeline is saving
    multiple data arrays for one data product (e.g. image data, weight,
    and flags for an Image object) to different extensions of a single
    FITS file, instead of saving each data array into a separate file.
    In this case, extname will be used to name the FITS extension, and
    the header and data will be appended to an existing FITS file if
    there is one.

    In all cases, if the final filename does not end with .fits, that
    will be appended at the end of it.

    Parameters
    ----------
    filename: str
        The full path to the file to write.  The actual written file
        will end in .<extname>.fits or .<extname>.fits.fz.  If the
        passed filename ends in '.fits' or '.fits.fz', that will be
        stripped off, and one will be re-appended based on whether
        fpack is set.

    data: np.ndarray
        The image data. Can also supply the weight, flags, etc.

    header: dict or astropy.io.fits.Header
        The header of the image.

    extname: str, default None
        The name of the extension to save the data into.  If writing
        individual files (default) will just append this string to the
        filename after a . and before '.fits' or '.fits.fz'.  If writing
        a single file, will use this as the extension name.

    overwrite: bool, default True
        Whether to overwrite the file if it already exists.  Ignored if
        single_file=True.

    single_file: bool, default False
        Whether to save each data array into a separate extension of the
        same file.  if False (default) will save each data array into a
        separate file.  If True, will use the extname to name the FITS
        extension, and save each array into the same file.

    fpack: bool, default False
        If true, will run fpack on the image after writing it.

    lossless: bool, default False
       Ignored if fpack is false.  If fpack is true, and lossless is
       true, then will use lossless compression (fpack parameters "-q 0
       -g2").

    fpackq: int, default None
       Ignored if fpack is false or lossless is True.  The q-factor to
       pass to fpack.  If None, will use fpack's default (which is 4).
       See
       https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/docs/fpackguide.pdf
       for more information.

    just_update_header: bool, default False
       Only works if single_file is False.  If this is True, and
       the file to be written exists, it will be opened in "update" mode
       and just the header will be rewritten, rather than the entire
       image.  (I'm not 100% sure that astropy will really do this
       right, thereby saving most of the disk I/O, but the existence of
       "update" mode in astropy.io.fits.open() suggests it does, so
       we'll use it.)  This option implies overwrite, so will overwrite
       the file header if overwrite is False.

    Returns
    -------
    The full absolute path to the file saved (or written to).

    """

    # avoid circular imports
    from models.base import safe_mkdir

    filepath = pathlib.Path( filename ).resolve()
    direc = filepath.parent
    filename = filepath.name
    filebase = ( filename[:-5] if filename.endswith(".fits")
                 else filename[:-8] if filename.endswith(".fits.fz")
                 else filename )

    # Figure out output filename
    if single_file:
        if fpack:
            raise NotImplementedError( "fpacking of multi-HDU files not currently supported" )
        finalfilepath = filepath.parent / f'{filebase}.fits'
    else:
        filename = filebase if extname is None else filebase + '.' + extname
        filename += '.fits'
        finalfilename = filename + '.fz' if fpack else filename
        intermedfilepath = direc / filename
        finalfilepath = direc / finalfilename

    # Build the header if necessary
    if not isinstance( header, fits.Header ):
        tmp = fits.Header()
        if header is not None:
            for k, v in header.items():
                tmp.header[k] = v
        header = tmp

    # If we're just updating the header, do that and be done
    if just_update_header:
        if single_file:
            raise NotImplementedError( "just_update_header doesn't work with single_file" )
        if not finalfilepath.is_file():
            raise FileNotFoundError( f"just_update_header failure: missing file {finalfilepath}" )
        with fits.open( finalfilepath, mode="update" ) as filehdu:
            ext = 1 if fpack else 0

            # This doesn't work.  I think it works for non-fpacked fits files,
            #   but for fpacked fits files the new header doesn't actually get
            #   written.  So, instead of doing this, we have to arduously
            #   manipulate the header in place.
            # filehdu[ext].header = header

            del filehdu[ext].header[0:len(filehdu[ext].header)]
            # ....aaaaand, a hack, because there needs to be a SIMPLE header,
            # but for some reason sometimes that wasn't there.
            filehdu[ext].header['SIMPLE'] = True
            # Some FITS keywords are perverse -- COMMENT and HISTORY (at least),
            #   as they aren't unique.  astropy's handling of them is a bit
            #   perverse too, at least for trying to do what I'm doing here.
            #   I'm a little nervous using an underscored class name from
            #   astropy, but you gotta do what you gotta do.  (I haven't
            #   found a way to do this in the astroyp documentation.)
            dedup = set()
            for kw in header:
                if kw in dedup:
                    continue
                dedup.add( kw )
                if isinstance( header[kw], fits.header._HeaderCommentaryCards ):
                    for val in header[kw]:
                        filehdu[ext].header[kw] = val
                else:
                    filehdu[ext].header[kw] = ( header[kw], header.comments[kw] )

        return str( finalfilepath )

    # Make sure the directory exists and is in a legal place
    safe_mkdir( direc )

    # Create the image we're gonna write
    hdu = fits.ImageHDU( data, header, name=extname ) if single_file else fits.PrimaryHDU( data, header )

    # Write
    if single_file:
        # TODO: what happens if there already is an extension in an existinf file with name extname?
        with fits.open( finalfilepath, memmap=False, mode='append' ) as hdul:
            if len(hdul) == 0:
                hdul.append( fits.PrimaryHDU() )
            hdul.append( hdu )
        return str( finalfilepath )

    else:
        # Single-HDU FITS file in the case where an object uses a different file for each extension
        hdul = fits.HDUList( [hdu] )

        if finalfilepath.exists():
            if not overwrite:
                raise FileExistsError( f"save_fits_image_file not overwriting {finalfilepath}" )
            else:
                finalfilepath.unlink()
        if intermedfilepath.exists():
            if not overwrite:
                raise FileExistsError( f"save_fits_image_file not overwriting {intermedfilepath}" )
            else:
                intermedfilepath.unlink()

        hdul.writeto( intermedfilepath )
        # If fpack is false, intermedfilepath and finalfilepath are the same

        if fpack:
            com = [ 'fpack' ]
            if lossless:
                com += [ '-q', '0', '-g2' ]
            elif fpackq is not None:
                com += [ '-q', 'fpackq' ]
            com += [ str(intermedfilepath.resolve()) ]
            try:
                res = subprocess.run( com, capture_output=True, timeout=60 )
            except subprocess.TimeoutExpired as ex:
                SCLogger.error( f"fpack subprocess timed out after {ex.timeout} seconds" )
                strio = io.StringIO()
                strio.write( f"fpack subprocess timed out after {ex.timeout} seconds\n" )
                strio.write( f"    Command: {' '.join(com)}\n" )
                strio.write( f"    stdout:\n{ex.stdout}\n-----------\nstderr:\n{ex.stderr}\n" )
                SCLogger.debug( strio.getvalue() )
                raise RuntimeError( "FITS writing failed: timed out on fpack" )
            except Exception as ex:
                SCLogger.error( f"fpack subprocess raised exception: {ex}" )
                raise
            finally:
                intermedfilepath.unlink()
            if res.returncode != 0:
                SCLogger.error( f"fpack subprocess failed: return code {res.returncode}" )
                strio = io.StringIO()
                strio.write( f"fpack subprocess failed: return code {res.returncode}" )
                strio.write( f"    Command: {' '.join(com)}\n" )
                strio.write( f"    stdout:\n{res.stdout}\n-----------\nstderr:\n{res.stderr}\n" )
                SCLogger.debug( strio.getvalue() )
                intermedfilepath.unlink()
                raise RuntimeError( "FITS writing failed: fpack failed" )

        return str( finalfilepath )
