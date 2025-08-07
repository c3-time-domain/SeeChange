import sys
import re
import multiprocessing
import logging

_default_log_level = logging.INFO
# _default_log_level = logging.DEBUG

# NOTE : tests/test_logger.py assumes this default date format is '%Y-%m-%d %H:%M:%S'
_default_datefmt = '%Y-%m-%d %H:%M:%S'

# Normally you don't want to show milliseconds, because it's additional gratuitous information
#  that makes log output lines longer.  But, if you're debugging timing stuff, you might want
#  temporarily to set this to True (either by editing it here, or by passing the argument
#  to SCLogger.replace().)
# NOTE: tests/test_logger.py assumes that _show_millisec is False
_show_millisec = False


class SCLogger:
    """Holds a unified logging instance that can be used throughout SeeChange.

    Normal use: just call one of

       SCLogger.exception( message )
       SCLogger.critical( message )
       SCLogger.error( message )
       SCLogger.warning( message )
       SCLogger.info( message )
       SCLogger.debug( message )

    and the message will be printed, with a header, to standard error.

    If you're using the logger in a case where you have many processes
    going via python's mutiprocessing module, at the beginning of a
    subprocess you might want to call

       SCLogger.multiprocessing_replace()

    That will add something to the header of each logging line that
    includes a process number, making it easier to track down which log
    messages came from the same process.  That process number is parsed
    from the process' name, if it has one and there's a number in it;
    otherwise, it's the PID of the process.  (Often with something like
    a multiprocessing pool, processes are named something like
    ForkPoolWorker-1, where the 1 increments.)

    If you want access to the underlying logging.Logger object, call
    SCLogger.get().  The underlying object is instantiated the first
    time it's used.  If you want to manually instantiate it, call
    instance(), with arguments defined in __init__.

    The single instance is held inside a SCLogger singleton.  (Sort
    of. The actual singleton changes sometimes.)  In principle, you
    could also make other SCLogger objects, but that's not the standard
    way to use it.  (In principle, this could be a memory leak, as
    python logging will remember all of the previously existing loggers.
    As long as SCLogger.replace() is not called frequently, this should
    not be a big deal.)

    """

    _instance = None
    _ordinal = 0

    @classmethod
    def instance( cls, *args, **kwargs ):
        """Return the singleton instance of SCLogger.

        The first time this is called, it will create the object.
        Normally, you never need to call this class method explicitly,
        as all the various other class methods call it.  If you want any
        of the defaults to be different from the defaults set in
        __init__ below, you will need to call this class method *before*
        the first time you use SCLogger for anything else.  (However,
        you can also call SCLogger.replace() to change the options
        used.)

        """
        if cls._instance is None:
            cls._instance = cls( *args, **kwargs )
        return cls._instance

    @classmethod
    def get( cls ):
        """Return the logging.Logger object."""
        return cls.instance()._logger

    @classmethod
    def replace( cls, midformat=None, datefmt=None, show_millisec=None, level=None, handler=None ):
        """Replace the logging.Logger object with a new one.

        Subsequent calls to SCLogger.get(), .info(), etc. will now
        return the new one.  Will inherit the various arguments from the
        current logger if they aren't specified here.

        See __init__ for parameters.

        Returns the logging.Logger object you'd get from get().

        """
        if cls._instance is not None:
            kwargs = {
                'midformat': cls._instance._midformat if midformat is None else midformat,
                'datefmt': cls._instance._datefmt if datefmt is None else datefmt,
                'show_millisec': cls._instance._show_millisec if show_millisec is None else show_millisec,
                'level': cls._instance._logger.level if level is None else level,
                'handler': cls._instance._handler if handler is None else handler
            }
        else:
            kwargs = {
                'midformat': midformat,
                'datefmt': _default_datefmt if datefmt is None else datefmt,
                'show_millisec': _show_millisec if show_millisec is None else show_millisec,
                'level': _default_log_level if level is None else level,
                'handler': handler
            }
        cls._instance = cls( **kwargs )
        return cls._instance

    @classmethod
    def multiprocessing_replace( cls, *args, **kwargs ):
        """Shorthand for replace with midformat parsed from the current multiprocessing process."""

        if ( ( cls._instance is not None ) and ( not cls._instance._using_default_handler ) and
             ( "handler" not in kwargs ) ):
            raise RuntimeError( "If you use multiprocessing_replace and you aren't using the "
                                "default handler, you need to pass a handler created in the subprocess." )

        me = multiprocessing.current_process()
        # Usually processes are named things like ForkPoolWorker-{number}, or something
        match = re.search( '([0-9]+)', me.name )
        if match is not None:
            num = f'{int(match.group(1)):3d}'
        else:
            num = str(me.pid)
        cls.replace( *args, midformat=num, **kwargs )

    @classmethod
    def set_level( cls, level=_default_log_level ):
        """Set the log level of the logging.Logger object."""
        cls.instance()._logger.setLevel( level )

    @classmethod
    def setLevel( cls, level=_default_log_level ):
        """Set the log level of the logging.Logger object."""
        cls.instance()._logger.setLevel( level )

    @classmethod
    def getEffectiveLevel( cls ):
        return cls.instance()._logger.getEffectiveLevel()

    @classmethod
    def debug( cls, *args, **kwargs ):
        cls.get().debug( *args, **kwargs )

    @classmethod
    def info( cls, *args, **kwargs ):
        cls.get().info( *args, **kwargs )

    @classmethod
    def warning( cls, *args, **kwargs ):
        cls.get().warning( *args, **kwargs )

    @classmethod
    def error( cls, *args, **kwargs ):
        cls.get().error( *args, **kwargs )

    @classmethod
    def critical( cls, *args, **kwargs ):
        cls.get().critical( *args, **kwargs )

    @classmethod
    def exception( cls, *args, **kwargs ):
        cls.get().exception( *args, **kwargs )

    def __init__( self, midformat=None, datefmt=_default_datefmt,
                  show_millisec=_show_millisec, level=_default_log_level,
                  handler=None ):
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

        show_millisec: bool, default False
            Add millseconds after a . following the date formatted by datefmt.

        level : logging level constant, default logging.WARNING
            This can be changed later with set_level().

        handler : a logging Handler or None
            Normally, SCLogger will send all log messages to sys.stderr.
            If you want it to go somewhere else, create an approprite
            logging.Handler subclass and pass it here.

        """
        self._midformat = midformat
        self._datefmt = datefmt
        self._show_millisec = show_millisec
        self._using_default_handler = handler is None
        self._handler = logging.StreamHandler( sys.stderr ) if self._using_default_handler else handler

        SCLogger._ordinal += 1
        self._logger = logging.getLogger( f"SeeChange_{SCLogger._ordinal}" )

        fmtstr = "[%(asctime)s"
        if self._show_millisec:
            fmtstr += ".%(msecs)03d"
        fmtstr += " - "
        if self._midformat is not None:
            fmtstr += f"{self._midformat} - "
        fmtstr += "%(levelname)s] - %(message)s"
        formatter = logging.Formatter( fmtstr, datefmt=self._datefmt )
        self._handler.setFormatter( formatter )
        self._logger.addHandler( self._handler )
        self._logger.setLevel( level )
