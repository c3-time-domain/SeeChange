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
        let self = this;

        this.auth = new rkAuth( this.authdiv, "",
                                () => { self.render_page(); },
                                () => { window.location.reload(); } );
        this.auth.checkAuth();
    };

    render_page()
    {
        let self = this;

        let p, span;
        
        rkWebUtil.wipeDiv( this.authdiv );
        p = rkWebUtil.elemaker( "p", this.authdiv,
                                { "text": "Logged in as " + this.auth.username
                                  + " (" + this.auth.userdisplayname + ") — ",
                                  "classes": [ "italic" ] } );
        span = rkWebUtil.elemaker( "span", p,
                                   { "classes": [ "link" ],
                                     "text": "Log Out",
                                     "click": () => { self.auth.logout( () => { window.location.reload(); } ) }
                                   } );
        
        rkWebUtil.wipeDiv( this.maindiv );
        this.frontpagediv = rkWebUtil.elemaker( "div", this.maindiv );
        p = rkWebUtil.elemaker( "p", this.frontpagediv, { "text": "Hello, world." } );
    }
}

// **********************************************************************
// **********************************************************************
// **********************************************************************
// Make this into a module

export { scconductor };
