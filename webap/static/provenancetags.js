import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"

// **********************************************************************

seechange.ProvenanceTags = class
{
    constructor( context )
    {
        this.context = context;
        this.div = rkWebUtil.elemaker( "div", null, { 'id': 'provenancetagsdiv' } );
    }


    render()
    {
        let self = this;
        let p, hbox, vbox;

        rkWebUtil.wipeDiv( this.div );
        p = rkWebUtil.elemaker( "p", this.div, );
        rkWebUtil.button( p, "Show", () => { self.show_provtag(); } );
        p.appendChild( document.createTextNode( " Provenance Tag: " ) );
        this.provtags = rkWebUtil.elemaker( "select", p, { 'id': 'provtaglistwid'  } );
        this.context.connector.sendHttpRequest( "provtags", {}, (data) => { self.populate_provtag_list(data) } );

        hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox" ] } )
        this.provtagdiv = rkWebUtil.elemaker( "div", hbox, { "id": "provenancetaginfodiv",
                                                             "classes": [ "mmargin" ] } );
        this.provinfodiv = rkWebUtil.elemaker( "div", hbox, { "id": "provenancetagprovinfodiv",
                                                              "classes": [ "mmargin"] } );
    };


    populate_provtag_list( data )
    {
        rkWebUtil.wipeDiv( self.provtags );
        for ( let provtag of data.provenance_tags ) {
            rkWebUtil.elemaker( "option", this.provtags, { "text": provtag, "attributes": { "value": provtag } } );
        }
    };


    show_provtag()
    {
        let self = this;
        let provtag = this.provtags.value;
        rkWebUtil.wipeDiv( this.provtagdiv );
        rkWebUtil.elemaker( "p", this.provtagdiv, { "text": "Loading provenance tag " + provtag + "...",
                                                    "classes": [ "warning", "bold", "italic" ] } );
        this.context.connector.sendHttpRequest( "provtaginfo/" + provtag, {},
                                                (data) => { self.actually_show_provtag(data) } );
    };


    actually_show_provtag( data )
    {
        let self = this;
        let table, tr, td, span, ttspan, a;

        rkWebUtil.wipeDiv( this.provtagdiv );
        rkWebUtil.elemaker( "h3", this.provtagdiv, { "text": "Provenance Tag " + data.tag } );
        table = rkWebUtil.elemaker( "table", this.provtagdiv, { "id": "provenancetagprovenancetable" } );
        tr = rkWebUtil.elemaker( "tr", table );
        for ( let col of [ "process", "code ver.", "id" ] )
            rkWebUtil.elemaker( "th", tr, { "text": col } );

        for ( let i in data['_id'] ) {
            tr = rkWebUtil.elemaker( "tr", table );
            td = rkWebUtil.elemaker( "td", tr, { "text": data['process'][i] } );
            td = rkWebUtil.elemaker( "td", tr, { "text": data['code_version_id'][i] } );
            td = rkWebUtil.elemaker( "td", tr, { "classes": [ "monospace" ] } );
            a = rkWebUtil.elemaker( "a", td, { "text": data['_id'][i],
                                               "classes": [ "link" ],
                                               "click": () => { self.show_prov( data['_id'][i] ) } } );
        }
    };


    show_prov( provid )
    {
        let self = this;
        rkWebUtil.wipeDiv( this.provinfodiv );
        rkWebUtil.elemaker( "p", this.provinfodiv, { "text": "Loading provenance " + provid + "...",
                                                     "classes": [ "warning", "bold", "italic" ] } );
        this.context.connector.sendHttpRequest( "/provenanceinfo/" + provid, {},
                                                (data) => { self.actually_show_prov(data) } );
    };


    actually_show_prov( data )
    {
        let self = this;
        let p, table, tr,td

        rkWebUtil.wipeDiv( this.provinfodiv );
        rkWebUtil.elemaker( "h3", this.provinfodiv, { "text": "Provenance " + data._id } );
        rkWebUtil.elemaker( "p", this.provinfodiv, { "text": "Process: " + data.process } );
        rkWebUtil.elemaker( "p", this.provinfodiv, { "text": "Code Version: " + data.code_version_id } );
        if ( data.is_bad )
            rkWebUtil.elemaker( "p", this.provinfodiv, { "text": "Is bad: " + data.bad_comment } );
        if ( data.is_testing )
            rkWebUtil.elemaker( "p", this.provinfodiv, { "text": "Is a testing provenance" } );
        if ( data.is_outdated ) {
            p = rkWebUtil.elemaker( "p", this.provinfodiv, { "text": "Outdated; replaced by " } );
            rkWebUtil.elemaker( "a", p, { "text": data.replaced_by,
                                          "classes": [ "link", "monospace" ],
                                          "click": () => { self.show_prov( data.replaced_by ) } } );
        }

        p = rkWebUtil.elemaker( "p", this.provinfodiv, { "text": "Parameters: " } );
        rkWebUtil.elemaker( "pre", p, { "text": JSON.stringify( data.parameters, null, 2 ) } );

        if ( data.upstreams._id.length > 0 ) {
            p = rkWebUtil.elemaker( "p", this.provinfodiv, { "text": "Upstreams:" } );
            table = rkWebUtil.elemaker( "table", this.provinfodiv );
            for ( let i in data.upstreams._id ) {
                tr = rkWebUtil.elemaker( "tr", table );
                rkWebUtil.elemaker( "td", tr, { "text": data.upstreams.process[i] } );
                td = rkWebUtil.elemaker( "td", tr, { "classes": [ "monospace" ] } );
                rkWebUtil.elemaker( "a", td, { "text": data.upstreams._id[i],
                                               "classes": [ "link" ],
                                               "click": () => { self.show_prov( data.upstreams._id[i] ) } } );
            }
        }
    };

}

// **********************************************************************
// Make this into a module

export { }
