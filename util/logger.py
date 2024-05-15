import sys
import multiprocessing
import logging

class SCLogger:
    """Holds the logging instance that we use throught SeeChange.

    Normal use: get the logger object with SCLogger.get(), which is a
    stander logging logger object.  Or, just call SCLogger.debug,
    SCLogger.info, SCLogger.warning, or SCLogger.error, which will
    instantiate the singleton as necessary.

    The single instance is held inside a SCLogger singleton.  In
    principle, you could also make other SCLogger objects, but that's
    not the standard way to use it.

    """

    _instance = None
    _ordinal = 0

    @classmethod
    def instance( cls, midformat=None, datefmt='%Y-%m-%d %H:%M:%S', level=logging.WARNING ):
        """Return the singleton instance of SCLogger."""
        if cls._instance is None:
            cls._instance = cls( midformat=midformat, datefmt=datefmt, level=level )
        return cls._instance

    @classmethod
    def get( cls ):
        """Return the logging.Logger object."""
        return cls.instance()._logger

    @classmethod
    def replace( cls, midformat=None, datefmt=None, level=None ):
        """Replace the logging.Logger object with a new one.

        Subsequent calls to SCLogger.get(), .info(), etc. will now
        return the new one.  Will inherit the midformat, datefmt, and
        level from the current logger if they aren't specified here.

        See __init__ for parameters.

        Returns the logging.Logger object you'd get from get().

        """
        if cls._instance is not None:
            midformat = cls._instance.midformat if midformat is None else midformat
            datefmt = cls._instance.datefmt if datefmt is None else datefmt
            level = cls._instance._logger.level if level is None else level
        else:
            datefmt = '%Y-%m-%d %H:%M:%S' if datefmt is None else datefmt
            level = logging.WARNING if level is None else level
        cls._instance = cls( midformat=midformat, datefmt=datefmt, level=level )
        return cls._instance

    @classmethod
    def set_level( cls, level=logging.WARNING ):
        """Set the log level of the logging.Logger object."""
        cls.instance()._logger.setLevel( level )

    @classmethod
    def setLevel( cls, level=logging.WARNING ):
        """Set the log level of the logging.Logger object."""
        cls.instance()._logger.setLevel( level )

    @classmethod
    def debug( cls, *args, **kwargs ):
        cls.get().debug( *args, **kwargs )

    @classmethod
    def info( cls, *args, **kwargs ):
        cls.get().info( *args, **kwargs )

    @classmethod
    def warning( cls, *args, **kwargs ):
        cls.get().info( *args, **kwargs )

    @classmethod
    def error( cls, *args, **kwargs ):
        cls.get().error( *args, **kwargs )

    @classmethod
    def critical( cls, *args, **kwargs ):
        cls.get().critical( *args, **kwargs )

    @classmethod
    def exception( cls, *args, **kwargs ):
        cls.get().exception( *args, **kwargs )

    def __init__( self, midformat=None, datefmt='%Y-%m-%d %H:%M:%S', level=logging.WARNING ):
        """Initialize a SCLogger object, and the logging.Logger object it holds.

        Parameters
        ----------
        midformat : string, default None
            The standard formatter emits log messages like "[yyyy-mm-dd
            HH:MM:SS - INFO] Message".  If given, this adds something between the
            date and the log level ("[yyyy-mm-dd HH:MM:SS - {midformat}
            - INFO]...").  Useful, for instance, in multiprocessing to
            keep track of which process the message came from.

        datefmt : string, default '%Y-%m-%d %H:%M:%S'
            The date format to use, using standard logging.Formatter
            datefmt syntax.

        level : logging level constnat, default logging.WARNING
            This can be changedl lated with set_level().

        """
        # (This lock is probably not necessary, since memory isn't by default shared in multiprocessing.)
        # This __init__ will be called rarely (typically once at the beginning of a process), so
        # the brief lock is not a big deal.
        # lock = multiprocessing.Lock()
        # with lock:
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
