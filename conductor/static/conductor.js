import { rkAuth } from "./rkauth.js"
import { rkWebUtil } from "./rkwebutil.js"

// Namespace, which is the only thing exported

var scconductor = {};

// **********************************************************************
// **********************************************************************
// **********************************************************************
// The global context

scconductor.Context = class
{
    constructor()
    {
        this.parentdiv = document.getElementById( "pagebody" );
        this.authdiv = document.getElementById( "authdiv" );
        this.maindiv = rkWebUtil.elemaker( "div", this.parentdiv );
        this.connector = new rkWebUtil.Connector( "/" );
    };

    init()
    {
        this.auth = new rkAuth( this.authdiv, "",
                                function() { self.render_page(); },
                                function() { window.location.reload(); } );
        this.auth.checkAuth();
    };
    
    render_page()
    {
        rkWebUtil.wipeDiv( this.maindiv );
        let div = rkWebUtil.elemaker( "div", this.maindiv );
        if ( ! this.auth.authenticated ) {
            let p = rkWebUtil.elemaker( "p", "Not authenticated" );
        } else {
            let p = rkWebUtil.elemaker( "p", this.frontpagediv, { "text": "Hello, world." } );
        }
    }
}

// **********************************************************************
// **********************************************************************
// **********************************************************************
// Make this into a module

export { scconductor };
