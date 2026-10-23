import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"
import "./exposurelist.js"
import "./imagelist.js"

// **********************************************************************

seechange.ExposureSearch = class
{
    constructor( context )
    {
        this.context = context;
        this.div = rkWebUtil.elemaker( "div", null, { 'id': 'exposuresearchdiv',
                                                      'classes': [ "grow1", "vbox" ] } );
    }


    render()
    {
        let self = this;
        let p, button, hbox, vbox;

        rkWebUtil.wipeDiv( this.div );

        hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox" ] } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "hmargin" ] } );

        p = rkWebUtil.elemaker( "p", vbox );
        button = rkWebUtil.button( p, "Show", function() { self.parse_widgets_and_show_exposures(); } );
        p.appendChild( document.createTextNode( " " ) );
        this.imageorexposure = rkWebUtil.elemaker( "select", p );
        rkWebUtil.elemaker( "option", this.imageorexposure, { "text": "Exposures",
                                                              "attributes": { "value": "exposures", "selected": 1 } } );
        rkWebUtil.elemaker( "option", this.imageorexposure, { "text": "Images",
                                                              "attributes": { "value": "images" } } );
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
        this.subdiv = rkWebUtil.elemaker( "div", this.div, { "id": "exposuresearchsubdiv",
                                                             "classes": [ "vbox", "grow1" ] } );

        this.emptiabledivs = {};
        this.tabbed = new rkWebUtil.Tabbed( this.subdiv );
        // Supposedly javascript objects (treated as dictionaries) are
        //   ordered with integer keys first in numerical order,
        //   then string keys in insertion order, then symbol
        //   keys in insertion order.  So, I'm counting
        //   on the order below being the order we pull.
        // Dictionary of tabname : title
        //   tabname will be referred to elsewhere in the code
        //   so is painful to change.  Title only shows up here
        //   so should be painless to change.
        let tabstoadd = { "exposurelist": "Exposure List",
                          "exposuredetail": "Exposure Detail",
                          "reports": "Reports",
                          "imagelist": "Image List",
                          "imagedetail": "Image Details",
                          "sources": "Detections",
                          "fakes": "Fakes"
                        };
        for ( let tab in tabstoadd ) {
            let focused = ( tab=="exposurelist" ? true : false );
            this.emptiabledivs[tab] = new rkWebUtil.EmptiableDiv( "div", null,
                                                                  { "id": "exposuresearch_tab_" + tab,
                                                                    "classes": [ "vbox", "grow1" ] } );
            this.tabbed.addTab( tab, tabstoadd[tab], this.emptiabledivs[tab].div, focused );
        }
    }


    populate_project_wid( data )
    {
        for ( let project of data.projects ) {
            rkWebUtil.elemaker( "option", this.project_wid, { "text": project, "attributes": { "value": project } } );
        }
    }

    parse_widgets_and_show_exposures()
    {
        var which, startdate, enddate, provtag, projects;

        which = this.imageorexposure.value;
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
        provtag = this.context.provtag_wid.value;
        if ( provtag == '<all>' ) provtag = null;
        projects = Array.from(this.project_wid.selectedOptions).map( ({ value }) => value );
        if ( projects.includes( '<all>' ) ) projects = null;

        this.show_exposures_or_images( which, startdate, enddate, provtag, projects );
    }


    show_exposures_or_images( which, startdate, enddate, provtag, projects )
    {
        var self = this;
        var whichtab, endpoint, callback, div;

        if ( which == "images" ) {
            whichtab = "imagelist";
            endpoint = "images/";
            callback = (data) => { self.actually_show_images( data ); };
            this.tabbed.showTabs();
            // THINK -- reports should be shown for image?
            this.tabbed.hideTabs( [ "exposurelist", "exposuredetail", "reports" ] );
        } else {
            whichtab = "exposurelist";
            endpoint = "exposures/";
            callback = (data) => { self.actually_show_exposures( data ); };
            this.tabbed.showTabs();
            this.tabbed.hideTabs( "imagelist" );
        }
        for ( let tab in this.emptiabledivs ) {
            if ( tab == whichtab ) {
                this.emptiabledivs[tab].reset( { "text": "Loading " + which + "...",
                                                 "classes": [ "warning", "bold", "italic" ] } );
            } else {
                this.emptiabledivs[tab].reset();
            }
        }
        this.tabbed.selectTab( whichtab );
        
        this.context.connector.sendHttpRequest( endpoint + provtag,
                                                { "startdate": startdate,
                                                  "enddate": enddate,
                                                  "projects": projects },
                                                callback );
    }


    actually_show_exposures( data )
    {
        if ( ! data.hasOwnProperty( "status" ) ) {
            console.log( "return has no status: " + data.toString() );
            window.alert( "Unexpected response from server when looking for exposures." );
            return
        }
        this.exposurelist = new seechange.ExposureList( this.context, this, this.tabbed,
                                                        this.tabbed.divs["exposurelist"],
                                                        data["exposures"], data["startdate"], data["enddate"],
                                                        data["provenance_tag"], data["projects"] );
        this.exposurelist.render_page();
    };


    actually_show_images( data )
    {
        this.imagelist = new seechange.ImageList( this.context, this, this.tabbed,
                                                  this.emptiabledivs["imagelist"].div,
                                                  this.emptiabledivs["imagedetail"].div,
                                                  this.emptiabledivs["sources"].div,
                                                  data.images, data.startdate, data.enddate,
                                                  data.provenance_tag, data.projects );
        this.imagelist.render_page();
    }
}



// **********************************************************************
// Make this into a module

export { }
