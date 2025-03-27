import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"

// **********************************************************************

seechange.ExposureSearch = class
{
    constructor( context )
    {
        this.context = context;
        this.div = rkWebUtil.elemaker( "div", null, { 'id': 'exposuresearchdiv' } );
    }


    render()
    {
        let self = this;
        let p, button, hbox, vbox;

        rkWebUtil.wipeDiv( this.div );

        hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox" ] } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "hmargin" ] } );

        p = rkWebUtil.elemaker( "p", vbox );
        button = rkWebUtil.button( p, "Show Exposures", function() { self.parse_widgets_and_show_exposures(); } );
        p.appendChild( document.createTextNode( " from " ) );
        this.startdatewid = rkWebUtil.elemaker( "input", p,
                                                { "id": "show_exposures_from_wid",
                                                  "attributes": { "type": "text",
                                                                  "size": 20 } } );
        this.startdatewid.addEventListener( "blur", function(e) {
            rkWebUtil.validateWidgetDateUTC( self.startdatewid );
        } );
        p.appendChild( document.createTextNode( " to " ) );
        this.enddatewid = rkWebUtil.elemaker( "input", p,
                                              { "id": "show_exposures_to_wid",
                                                "attributes": { "type": "text",
                                                                "size": 20 } } );
        this.enddatewid.addEventListener( "blur", function(e) {
            rkWebUtil.validateWidgetDateUTC( self.enddatewid );
        } );
        rkWebUtil.elemaker( "br", p );
        p.appendChild( document.createTextNode( "    (YYYY-MM-DD [HH:MM] — leave blank for no limit)" ) );

        p = rkWebUtil.elemaker( "p", vbox, { "id": "exposureprovtagsearchdiv",
                                             "text": "Search provenance tag: " } );
        this.provtag_wid = rkWebUtil.elemaker( "select", p, { 'id': 'provtag_wid' } );
        this.context.connector.sendHttpRequest( "provtags", {}, (data) => { self.populate_provtag_wid(data) } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "hmargin" ] } );
        vbox.appendChild( document.createTextNode( "Search projects:" ) );
        rkWebUtil.elemaker( "br", vbox );

        this.project_wid = rkWebUtil.elemaker( "select", vbox, { 'id': 'project_wid',
                                                                 "attributes": { "multiple": 1 } } );
        rkWebUtil.elemaker( "option", this.project_wid,
                            { "text": "<all>",
                              "attributes": { "value": "<all>",
                                              "selected": 1 } } );
        this.context.connector.sendHttpRequest( "projects", {}, (data) => { self.populate_project_wid(data) } );

        rkWebUtil.elemaker( "hr", this.div );
        this.subdiv = rkWebUtil.elemaker( "div", this.div, { "id": "exposuresearchsubdiv" } );
    }


    populate_provtag_wid( data )
    {
        for ( let provtag of data.provenance_tags ) {
            rkWebUtil.elemaker( "option", this.provtag_wid, { "text": provtag, "attributes": { "value": provtag } } );
        }
    };

    populate_project_wid( data )
    {
        for ( let project of data.projects ) {
            rkWebUtil.elemaker( "option", this.project_wid, { "text": project, "attributes": { "value": project } } );
        }
    }


    parse_widgets_and_show_exposures()
    {
        var startdate, enddate, provtag, projects;
        try {
            startdate = this.startdatewid.value.trim();
            if ( startdate.length > 0 )
                startdate = rkWebUtil.parseStandardDateString( startdate ).toISOString();
            else startdate = null;
            enddate = this.enddatewid.value.trim();
            if ( enddate.length > 0 )
                enddate = rkWebUtil.parseStandardDateString( enddate ).toISOString();
            else enddate = null;
        }
        catch (ex) {
            window.alert( "Error parsing at least one of the two dates:\n" + this.startdatewid.value +
                          "\n" + this.enddatewid.value );
            console.log( "Exception parsing dates: " + ex.toString() );
            return;
        }
        provtag = this.provtag_wid.value;
        if ( provtag == '<all>' ) provtag = null;
        projects = Array.from(this.project_wid.selectedOptions).map( ({ value }) => value );
        if ( projects.includes( '<all>' ) ) projects = null;

        this.show_exposures( startdate, enddate, provtag, projects );
    }


    show_exposures( startdate, enddate, provtag, projects )
    {
        var self = this;

        rkWebUtil.wipeDiv( this.subdiv );
        rkWebUtil.elemaker( "p", this.subdiv, { "text": "Loading exposures...",
                                                "classes": [ "warning", "bold", "italic" ] } );

        this.context.connector.sendHttpRequest( "exposures",
                                                { "startdate": startdate,
                                                  "enddate": enddate,
                                                  "provenancetag": provtag,
                                                  "projects": projects },
                                                function( data ) {
                                                    self.actually_show_exposures( data ); } );
    }


    actually_show_exposures( data )
    {
        if ( ! data.hasOwnProperty( "status" ) ) {
            console.log( "return has no status: " + data.toString() );
            window.alert( "Unexpected response from server when looking for exposures." );
            return
        }
        let exps = new seechange.ExposureList( this.context, this, this.subdiv,
                                               data["exposures"],
                                               data["startdate"],
                                               data["enddate"],
                                               data["provenance_tag"],
                                               data["projects"] );
        exps.render_page();
    };

}

// **********************************************************************
// Make this into a module

export { }
