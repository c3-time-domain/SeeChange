import io
import hashlib
import logging
import pathlib
import requests
import time
import traceback

def retry_download( url, fpath, md5sum=None, retries=5, sleeptime=5, exists_ok=True, clobber=False,
                    logger=logging.getLogger("main"), sizelog='MiB' ):
    """Download a file from a url, retrying upon failure.

    Parameters
    ----------
    url : str
      The URL to download from
    fpath : str or Path
      Where to write the file
    md5sum : None or  str
      If not None, compare the md5sum of the downloaded file to this
      (hexdigest)
    retries : int
      Number of times to retry before failing with an exception
    sleeptime : int or float
      Seconds to sleep after a failed download before retrying (default 5)
    exists_ok : bool
      If False, always raise an exception if the file already exists.
      If True (default), and fpath is not a regular file (e.g. a
      directory), raise an exception.  Otherwise, trust that the file
      there is correct if md5sum is None, or verify the md5sum if it's
      not.
    clobber : bool
      Only matters if exists_ok is True and md5sum is not None.  If the
      md5sum of a pre-existing file at fpath doesn't matched the passed
      one, delete the file and redownload.
    logger : logging.logger
    sizelog : str
      One of 'GiB', 'MiB', or 'kib', defaulting to 'MiB' (which will
      also be usede if the string doesn't match any of the three).
      Units to report download size and rate in to logger.info().

    """

    fpath = pathlib.Path( fpath )
    fname = fpath.name
    if fpath.exists():
        if not fpath.is_file():
            logger.error( f"retry_download: {fpath} exists and is not a file, not overwriting." )
            raise FileExistsError( f"{fpath} exists and is not a file, not overwriting." )
        if not exists_ok:
            logger.error( f"retry_download: {fpath} already exists and exists_ok is false." )
            raise FileExistsError( f"{fpath} already exists and exists_ok is false." )
        else:
            if md5sum is None:
                logger.info( f"retry_download: {fpath} exists, trusting it's the right thing." )
                return
            md5 = hashlib.md5()
            with open( fpath, 'rb' ) as ifp:
                md5.update( ifp.read() )
            if md5.hexdigest() == md5sum:
                logger.info( f"retry_download: {fname} already exists with the right md5sum, not redownloading" )
                return
            mmerr = f"md5sum {md5.hexdigest()} doesn't match expected {md5sum}"
            if not clobber:
                logger.error( f"retry_download: {fpath} exists but {mmerr} and clobber is False" )
                raise FileExistsError( f"{fpath} exists but {mmerr} and clobber is False" )
            logger.info( f"retry_download: existing {fpath} {mmerr}, clobbering and redownloading"  )
            fpath.unlink()

    # If we get this far, then there is no file at fpath, either because
    # it was never there, or because it was clobbered.

    if logger.getEffectiveLevel() >= logging.DEBUG:
        logger.info( f"download_exposures: Downloading {url} to {fpath}" )
    else:
        logger.info( f"download_exposures: Downloading {fname}" )

    countdown = retries
    success = False
    while not success:
        if countdown < retries:
            logger.info( f"...retrying download of {fname}" )
        countdown -= 1
        try:
            starttime = time.perf_counter()
            renew = False
            response = requests.get( url )
            response.raise_for_status()
            midtime = time.perf_counter()
            if sizelog == 'GiB':
                size = len(response.content) / 1024 / 1024 / 1024
            elif sizelog == 'kiB':
                size = len(response.content) / 1024
            else:
                size = len(response.content) / 1024 / 1024
                sizelog = 'MiB'
            dt = float( midtime-starttime )
            logger.info( f"...downloaded {size:.3f} {sizelog} in {midtime-starttime:.2f} sec "
                         f"({size/dt:.3f} {sizelog}/sec)" )
            fpath.parent.mkdir( exist_ok=True, parents=True )
            with open( fpath, "wb" ) as ofp:
                ofp.write( response.content )
            endtime = time.perf_counter()
            logger.info( f"...written to disk in {endtime-midtime:.2f} sec" )
            success = True
            if md5sum is not None:
                md5 = hashlib.md5()
                with open( fpath, 'rb' ) as ifp:
                    md5.update( ifp.read() )
                if md5.hexdigest() != md5sum:
                    success = False
                    logger.warning( f"Downloaded {fname} md5sum {md5.hexdigest()} doesn't match "
                                    f"expected {md5sum}, retrying" )
                    time.sleep( sleeptime )
        except Exception as e:
            strio = io.StringIO("")
            traceback.print_exc( file=strio )
            logger.warning( f"Exception downloading from {url}:\n{strio.getvalue()}" )
            if countdown > 0:
                logger.warning( f"retry_download: Failed to download {fname}, "
                                 f"waiting {sleeptime} sec and retrying." )
                time.sleep( sleeptime )
            else:
                err = f'{retries} exceptions trying to download {fname}, failing.'
                logger.error( f"retry_download: {err}" )
                raise RuntimeError( err )
