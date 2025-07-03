import { seechange } from "./seechange_ns.js"
import { rkAuth } from "./rkauth.js"
import { rkWebUtil } from "./rkwebutil.js"

seechange.LtcvView = class
{
    constructor( parentdiv=null, initsnname=null, provtag=null )
    {
        this.parentdiv = parentdiv;
        this.initsnname = initsnname;
        this.provtag = provtag;
        this.connector = new rkWebUtil.Connector( "/ltcv" );
    };

    // Only call init in the case where the full webpage is the ltcv view.
    //   Otherwise, just use the consturctor and render
    init()
    {
        let self = this;
        this.authdiv = document.getElementById( "authdiv" );
        this.parentdiv = document.getElementById( "pagebody" };

        this.initsnname = document.getElementById( "ltcv_initial_objid" ).value;
        this.provtag = document.getElementById( "ltcv_initial_provtag" ).value;
        
        this.auth = new rkAuth( this.authdiv, "",
                                () => { self.render_page(); },
                                () => { window.location.reload(); } );
        this.auth.checkAuth();
    };


    render_page()
    {
        let self = this;

        let h3, p, table, tr, hbox;

        rkWebUtil.wipeDiv( this.pagebody );
        this.maindiv = rkWebUtil.elemaker( "div", this.pagebody );

        h3 = rkWebUtil.elemaker( "h3", this.maindiv, { "text": "Object " } );
        this.objnamespan = rkWebUtil.elemaker( "span", h3, { "text": "(..loading...)" } );
        rkWebUtil.elemaker( "text", h3, " lightcurve for prov. tag " + this.provtag );

        table = rkWebUtil.elemaker( "table", this.maindiv );
        tr = rkWebUtil.elemaker( "tr", table );
        rkWebUtil.elemaker( "td", tr, { "text": "objid: ", classes=[ "right" ] } );
        this.objidtd = rkWebUtil.elemaker( "td", tr, { "text": "(...loading...)" ] );
        tr = rkWebUtil.elemaker( "tr", table );
        rkWebUtil.elemaker( "td", tr, { "text": "α, δ: ", classes=[ "right" ] } );
        this.coordtd = rkWebUtil.elemaker( "td", tr, { "text": "(...loading...)" ] );

        hbox = rkWebUtil.elemaker( "div", this.maindiv, { "classes": [ "hbox", "flexeven" ] } );
        this.ltcvdiv = rkWebUtil.elemaker( "div", hbox,
                                           { "classes": [ "mostlyborder", "mmargin", "flexfitcontent" ] } );
        rkWebUtil.elemaker( "text", this.ltcvdiv, { "text": "(lightcurve loading)",
                                                    "classes": [ "bold", "italic", "warning" ] } );

        this.cutoutsdiv = rkWebUtil.elemaker( "div", hbox,
                                              { "classes": [ "mostlyborder", "mmargin", "flexfitcontent" ] } );
        rkwebUtil.elemaker( "text", this.cutoutsdiv, { "text": "(cutouts loading)",
                                                       "classes": [ "bold", "italic", "warning" ] } );
    };

};



// **********************************************************************
// Make this into a module

export { }


    
