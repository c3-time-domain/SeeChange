import flask

# NOTE: for get_instrument_instrance to work, must manually import all
#  known instrument classes we might want to use here.
# If models.instrument gets imported somewhere else before this file
#  is imported, then even this won't work.  There must be a better way....
import models.decam  # noqa: F401
from models.instrument import get_instrument_instance

sys.path.insert( 0, pathlib.Path(__name__).resolve().parent )
from baseview import BaseView, BadUpdaterReturnError


# ======================================================================

class Ltcv( BaseView ):
    def do_the_things( self, argstr=None ):
        args = self.argstr_to_args( argstr, { 'objid': "", 'provtag': "default" } )
        return flask.render_template( "ltcv.html", objid=args['objid'], provtag=args['provtag'] )
        


# ======================================================================

bp = flask.Blueprint( 'ltcv', __name__, url_prefix='/ltcv' )

urls = {
    '/': Ltcv,
    '/<path:argstr>': Ltcv
}

usedurls = {}
for url, cls in urls.items():
    if url not in usedurls.keys():
        usedurls[ url ] = 0
        name = url
    else:
        usedurls[ url ] += 1
        name = f"url.{usedurls[url]}"

    bp.add_url_rule( url, view_func=cls.as_view(name), methods=["GET", "POST"], strict_slashes=False )
