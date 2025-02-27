import { seechange } from "./seechange_ns.js"
import { rkAuth } from "./rkauth.js"
import { rkWebUtil } from "./rkwebutil.js";
import "./provenancetags.js";
import "./exposuresearch.js";
import "./exposurelist.js";
import "./exposure.js";
import "./conductor.js";

// Everything is going in the seechange namespace
//  (which is imported from seechange_ns.js so that
//   we can avoid circular imports).
// Because of how we do this, it's important to
//  import all the files from here, and only
//  to import rkwebutil and seechange_ns in the
//  various other .js files.  (seechange_start.js
//  is a special case.)


// **********************************************************************
// The organization of the webap on the screen, and in classes below,
// is hierarchical.  At the top is Context.  (Above and outside
// the hierarchy is user auth.)  Contextholds a tabbed div with several
// broad "applications", and those applications have tabbed divs inside them.
//
//
// Each class initializes itself with its constructor, and then actually
// renders itself with render_page().
//
//   Context
//     Conductor (admins only)
//     Exposure Search
//        Exposure List
//        Exposure
//     Provenance Tags
//

// **********************************************************************
// **********************************************************************
// **********************************************************************
// The global context

seechange.Context = class
{
    constructor()
    {
        this.parentdiv = document.getElementById( "pagebody" );
        this.authdiv = document.getElementById( "authdiv" );
        this.maindiv = rkWebUtil.elemaker( "div", this.parentdiv, { 'id': 'parentdiv' } );
        this.frontpagediv = null;

        // TODO : make this configurable?  Or at least remember how to
        //   detect in javascript what the URL you're running from is.  (In
        //   case the webap is not running as the root ap of the webserver.)
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
        var self = this;
        let p, span, button;

        if ( this.frontpagediv == null ) {

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

            this.frontpagediv = rkWebUtil.elemaker( "div", this.maindiv, { 'id': 'frontpagediv' } );

            this.frontpagetabs = new rkWebUtil.Tabbed( this.frontpagediv );
            this.frontpagetabs.div.setAttribute( 'id', 'frontpagetabs' );

            if ( this.auth.usergroups.includes( 'admin' ) || this.auth.usergroups.includes( 'root' ) ) {
                this.conductor = new seechange.Conductor( this );
                this.conductor.render();
                this.frontpagetabs.addTab( "conductor", "Conductor", this.conductor.div );
            }

            this.exposuresearch = new seechange.ExposureSearch( this );
            this.exposuresearch.render();
            this.frontpagetabs.addTab( "exposuresearch", "Exposure Search", this.exposuresearch.div );


            this.provenancetags = new seechange.ProvenanceTags( this );
            this.provenancetags.render();
            this.frontpagetabs.addTab( "provtags", "Provenance Tags", this.provenancetags.div );


            let div = rkWebUtil.elemaker( "div", null );
            rkWebUtil.elemaker( "p", div, { "text": "42" } );
            this.frontpagetabs.addTab( "gratuitous", "Answer", div );

            this.frontpagetabs.selectTab( "exposuresearch" );

            // This next element is used in our tests as a way of quickly figuring
            //   out that this function has finished.
            rkWebUtil.elemaker( "input", this.frontpagediv, { "id": "seechange_context_render_page_complete",
                                                              "attributes": { "type": "hidden" } } );
        }
        else {
            alert( "Reinstalling frontpagediv" );
            rkWebUtil.wipeDiv( this.maindiv );
            this.maindiv.appendChild( this.frontpagediv );
        }
    };


}


// **********************************************************************
// Make this into a module

export { }

