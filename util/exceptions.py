"""Standard exceptions for the pipeline."""

import io


class CatalogNotFoundError( RuntimeError ):
    """A CatalogExcerpt was not found."""

    def __init__( self, *args, **kwargs ):
        super().__init__( self, *args, **kwargs )


class SubprocessFailure( RuntimeError ):
    """A subprocess didn't return success."""

    def __init__( self, response, *args, **kwargs ):
        strio = io.StringIO()
        strio.write( f"Subprocess {response.args[0] if isinstance(response.args,list) else response.args} "
                     f"returned {response.returncode}\n" )
        strio.write( f"arguments: {response.args}\n" )
        strio.write( f"stdout:\n{response.stdout.decode('utf-8')}\n" )
        strio.write( f"stderr:\n{response.stderr.decode('utf-8')}\n" )
        self.message = strio.getvalue()
        super().__init__( self, self.message, *args, **kwargs )

    def __str__( self ):
        return self.message


class BadMatchException( RuntimeError ):
    """A process matching two catalogs/source lists found too few matches, or the residuals were too large."""

    def __init__( self, *args, **kwargs ):
        super().__init__( self, *args, **kwargs )
