import sys
import multiprocessing
import logging

class SCLogger:
    _instance = None
    _ordinal = 0

    @classmethod
    def instance( cls, midformat=None, datefmt='%Y-%m-%d %H:%M:%S', level=logging.WARNING ):
        if cls._instance is None:
            cls._instance = cls( midformat=midformat, datefmt=datefmt, level=level )
        return cls._instance

    @classmethod
    def get( cls ):
        return cls.instance()._logger

    @classmethod
    def replace( cls, midformat=None, datefmt=None, level=None ):
        if cls._instance is not None:
            midformat = cls._instance.midformat if midformat is None else midformat
            datefmt = cls._instance.datefmt if datefmt is None else datefmt
            level = cls._instance._logger.level if level is None else level
        cls._instance = cls( midformat=midformat, datefmt=datefmt, level=level )
        return cls._instance

    @classmethod
    def set_level( cls, level=logging.WARNING ):
        cls.instance()._logger.setLevel( level )

    def __init__( self, midformat=None, datefmt='%Y-%m-%d %H:%M:%S', level=logging.WARNING ):
        lock = multiprocessing.Lock()
        with lock:
            SCLogger._ordinal += 1
            self._logger = logging.getLogger( f"SeeChange_{SCLogger._ordinal}" )

        self.midformat = midformat
        self.datefmt = datefmt

        logout = logging.StreamHandler( sys.stderr )
        fmtstr = f"[%(asctime)s - "
        if midformat is not None:
            fmtstr += f"{midformat} - "
        fmtstr += "%(levelname)s] - %(message)s"
        formatter = logging.Formatter( fmtstr, datefmt=datefmt )
        logout.setFormatter( formatter )
        self._logger.addHandler( logout )
        self._logger.setLevel( level )



