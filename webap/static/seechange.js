import { rkAuth } from "./rkauth.js"
import { rkWebUtil } from "./rkwebutil.js";

// Namespace, which is the only thing exported

var seechange = {};

seechange.nullorfixed = function( val, num ) { return val == null ? null : val.toFixed(num); }

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
// **********************************************************************
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
// **********************************************************************
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
            rkWebUtil.validateWidgetDate( self.startdatewid );
        } );
        p.appendChild( document.createTextNode( " to " ) );
        this.enddatewid = rkWebUtil.elemaker( "input", p,
                                              { "id": "show_exposures_to_wid",
                                                "attributes": { "type": "text",
                                                                "size": 20 } } );
        this.enddatewid.addEventListener( "blur", function(e) {
            rkWebUtil.validateWidgetDate( self.enddatewid );
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
// **********************************************************************
// **********************************************************************

seechange.ExposureList = class
{
    constructor( context, parent, parentdiv, exposures, fromtime, totime, provtag, projects )
    {
        this.context = context;
        this.parent = parent;
        this.parentdiv = parentdiv;
        this.exposures = exposures;
        this.fromtime = fromtime;
        this.totime = totime;
        this.provtag = provtag;
        this.projects = projects;
        this.masterdiv = null;
        this.listdiv = null;
        this.exposurediv = null;
        this.exposure_displays = {};
    };


    render_page()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.parentdiv );

        if ( this.masterdiv != null ) {
            this.parentdiv.appendChild( this.masterdiv );
            return
        }

        this.masterdiv = rkWebUtil.elemaker( "div", this.parentdiv, { 'id': 'exposurelistmasterdiv' } );

        this.tabbed = new rkWebUtil.Tabbed( this.masterdiv );
        this.listdiv = rkWebUtil.elemaker( "div", null, { 'id': 'exposurelistlistdiv' } );
        this.tabbed.addTab( "exposurelist", "Exposure List", this.listdiv, true );
        this.exposurediv = rkWebUtil.elemaker( "div", null, { 'id': 'exposurelistexposurediv' } );
        this.tabbed.addTab( "exposuredetail", "Exposure Details", this.exposurediv, false );
        rkWebUtil.elemaker( "p", this.exposurediv,
                            { "text": 'No exposure listed; click on an exposure in the "Exposure List" tab.' } );

        var table, th, tr, td;

        // let p = rkWebUtil.elemaker( "p", this.listdiv );
        // rkWebUtil.elemaker( "span", p, { "text": "[Back to exposure search]",
        //                                  "classes": [ "link" ],
        //                                  "click": () => { self.context.render_page() } } );
        // p.appendChild( document.createTextNode( "  —  " ) );
        // rkWebUtil.elemaker( "span", p, { "text": "[Refresh]",
        //                                  "classes": [ "link" ],
        //                                  "click": () => { rkWebUtil.wipeDiv( self.div );
        //                                                   self.context.show_exposures(); } } );

        let h2 = rkWebUtil.elemaker( "h2", this.listdiv, { "text": "Exposures" } );
        if ( ( this.fromtime == null ) && ( this.totime == null ) ) {
            h2.appendChild( document.createTextNode( " from all time" ) );
        } else if ( this.fromtime == null ) {
            h2.appendChild( document.createTextNode( " up to " + this.totime ) );
        } else if ( this.totime == null ) {
            h2.appendChild( document.createTextNode( " from " + this.fromtime + " on" ) );
        } else {
            h2.appendChild( document.createTextNode( " from " + this.fromtime + " to " + this.totime ) );
        }

        if ( this.provtag == null ) {
            h2.appendChild( document.createTextNode( " including all provenances" ) );
        } else {
            h2.appendChild( document.createTextNode( " with provenance tag " + this.provtag ) );
        }

        rkWebUtil.elemaker( "p", this.listdiv,
                            { "text": '"Detections" are everything found on subtratcions; ' +
                              '"Sources" are things that passed preliminary cuts.' } )

        table = rkWebUtil.elemaker( "table", this.listdiv, { "classes": [ "exposurelist" ],
                                                             "attributes": { "id": "exposure_list_table" } } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Exposure" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "MJD" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "target" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "filter" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "t_exp (s)" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "subs" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "detections" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "sources" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "n_successim" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "n_errors" } );

        this.tablerows = [];
        let exps = this.exposures;   // For typing convenience...
        // Remember, in javascript, "i in x" is like python "i in range(len(x))" or "i in x.keys()"
        let fade = 1;
        let countdown = 3;
        for ( let i in exps["name"] ) {
            let row = rkWebUtil.elemaker( "tr", table, { "classes": [ fade ? "bgfade" : "bgwhite" ] } );
            this.tablerows.push( row );
            td = rkWebUtil.elemaker( "td", row );
            rkWebUtil.elemaker( "a", td, { "text": exps["name"][i],
                                           "classes": [ "link" ],
                                           "click": function() {
                                               self.show_exposure( exps["id"][i],
                                                                   exps["name"][i],
                                                                   exps["mjd"][i],
                                                                   exps["filter"][i],
                                                                   exps["target"][i],
                                                                   exps["project"][i],
                                                                   exps["exp_time"][i] );
                                           }
                                         } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["mjd"][i].toFixed(2) } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["target"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["filter"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["exp_time"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["n_subs"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["n_sources"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["n_measurements"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["n_successim"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["n_errors"][i] } );
            countdown -= 1;
            if ( countdown == 0 ) {
                countdown = 3;
                fade = 1 - fade;
            }
        }
    };


    show_exposure( id, name, mjd, filter, target, project, exp_time )
    {
        let self = this;

        this.tabbed.selectTab( "exposuredetail" );

        if ( this.exposure_displays.hasOwnProperty( id ) ) {
            this.exposure_displays[id].render_page();
        }
        else {
            rkWebUtil.wipeDiv( this.exposurediv );
            rkWebUtil.elemaker( "p", this.exposurediv, { "text": "Loading...",
                                                         "classes": [ "warning", "bold", "italic" ] } );
            this.context.connector.sendHttpRequest( "exposure_images/" + id + "/" + this.provtag,
                                                    null,
                                                    (data) => {
                                                        self.actually_show_exposure( id, name, mjd, filter,
                                                                                     target, project,
                                                                                     exp_time, data );
                                                    } );
        }
    };


    actually_show_exposure( id, name, mjd, filter, target, project, exp_time, data )
    {
        let exp = new seechange.Exposure( this, this.context, this.exposurediv,
                                          id, name, mjd, filter, target, project, exp_time, data );
        this.exposure_displays[id] = exp;
        exp.render_page();
    };
}



// **********************************************************************
// **********************************************************************
// **********************************************************************

seechange.Exposure = class
{
    constructor( exposurelist, context, parentdiv, id, name, mjd, filter, target, project, exp_time, data )
    {
        this.exposurelist = exposurelist;
        this.context = context;
        this.parentdiv = parentdiv;
        this.id = id;
        this.name = name;
        this.mjd = mjd;
        this.filter = filter;
        this.target = target;
        this.project = project;
        this.exp_time = exp_time;
        this.data = data;
        this.div = null;
        this.tabs = null;
        this.imagesdiv = null;
        this.cutoutsdiv = null;
        this.cutoutsallimages_checkbox = null;
        this.cutoutsimage_checkboxes = {};
        this.cutouts = {};
        this.cutouts_pngs = {};
    };


    // Copy and adapt these next two from models/enums_and_bitflags.py
    static process_steps = {
        1: 'preprocessing',
        2: 'extraction',
        3: 'backgrounding',
        4: 'astrocal',
        5: 'photocal',
        6: 'subtraction',
        7: 'detection',
        8: 'cutting',
        9: 'measuring',
        10: 'scoring',
        11: 'alerting',
        30: 'finalize'
    };

    static pipeline_products = {
        1: 'image',
        2: 'sources',
        3: 'psf',
        5: 'wcs',
        6: 'zp',
        7: 'sub_image',
        8: 'detections',
        9: 'cutouts',
        10: 'measurements',
        11: 'scores',
    };


    render_page()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.parentdiv );

        if ( this.div != null ) {
            this.parentdiv.appendChild( this.div );
            return;
        }

        this.div = rkWebUtil.elemaker( "div", this.parentdiv );

        var h2, h3, ul, li, table, tr, td, th, hbox, p, span, tiptext, ttspan;

        // rkWebUtil.elemaker( "p", this.div, { "text": "[Back to exposure list]",
        //                                      "classes": [ "link" ],
        //                                      "click": () => { self.exposurelist.render_page(); } } );

        h2 = rkWebUtil.elemaker( "h2", this.div, { "text": "Exposure " + this.name } );
        ul = rkWebUtil.elemaker( "ul", this.div );
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>provenance tag:</b> " + this.data.provenancetag;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>project:</b> " + this.project;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>target:</b> " + this.target;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>mjd:</b> " + this.mjd
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>filter:</b> " + this.filter;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>t_exp (s):</b> " + this.exp_time;

        this.tabs = new rkWebUtil.Tabbed( this.div );


        this.imagesdiv = rkWebUtil.elemaker( "div", null, { 'id': 'exposureimagesdiv' } );

        let totncutouts = 0;
        let totnsources = 0;
        for ( let i in this.data['id'] ) {
            totncutouts += this.data['numsources'][i];
            totnsources += this.data['nummeasurements'][i];
        }

        let numsubs = 0;
        for ( let sid of this.data.subid ) if ( sid != null ) numsubs += 1;
        p = rkWebUtil.elemaker( "p", this.imagesdiv,
                                { "text": ( "Exposure has " + this.data.id.length + " images and " + numsubs +
                                            " completed subtractions" ) } )
        p = rkWebUtil.elemaker( "p", this.imagesdiv,
                                { "text": ( totnsources.toString() + " out of " +
                                            totncutouts.toString() + " detections pass preliminary cuts " +
                                            "(i.e. are \"sources\")." ) } );

        p = rkWebUtil.elemaker( "p", this.imagesdiv );

        this.cutoutsallimages_checkbox =
            rkWebUtil.elemaker( "input", p, { "attributes":
                                              { "type": "radio",
                                                "id": "cutouts_all_images",
                                                "name": "whichimages_cutouts_checkbox",
                                                "checked": "checked" } } );
        rkWebUtil.elemaker( "span", p, { "text": " Show sources for all images" } );
        p.appendChild( document.createTextNode( "      " ) );

        this.cutoutssansmeasurements_checkbox =
            rkWebUtil.elemaker( "input", p, { "attributes":
                                              { "type": "checkbox",
                                                "id": "cutouts_sans_measurements",
                                                "name": "cutouts_sans_measurements_checkbox" } } );
        rkWebUtil.elemaker( "label", p, { "text": ( "Show detections that failed the preliminary cuts " +
                                                    "(i.e. aren't sources)" ),
                                          "attributes": { "for": "cutouts_sans_measurements_checkbox" } } );


        table = rkWebUtil.elemaker( "table", this.imagesdiv, { "classes": [ "exposurelist" ] } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr );
        th = rkWebUtil.elemaker( "th", tr, { "text": "name" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "section" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "α" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "δ" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "b" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "fwhm" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "zp" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "mag_lim" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "detections" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "sources" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "compl. step" } );
        th = rkWebUtil.elemaker( "th", tr, {} ); // products exist
        th = rkWebUtil.elemaker( "th", tr, {} ); // error
        th = rkWebUtil.elemaker( "th", tr, {} ); // warnings

        let fade = 1;
        let countdown = 4;
        for ( let i in this.data['id'] ) {
            countdown -= 1;
            if ( countdown <= 0 ) {
                countdown = 3;
                fade = 1 - fade;
            }
            tr = rkWebUtil.elemaker( "tr", table, { "classes": [ fade ? "bgfade" : "bgwhite" ] } );
            td = rkWebUtil.elemaker( "td", tr );
            this.cutoutsimage_checkboxes[ this.data['id'][i] ] =
                rkWebUtil.elemaker( "input", td, { "attributes":
                                                   { "type": "radio",
                                                     "id": this.data['id'][i],
                                                     "name": "whichimages_cutouts_checkbox" } } )
            td = rkWebUtil.elemaker( "td", tr, { "text": this.data['name'][i] } );
            td = rkWebUtil.elemaker( "td", tr, { "text": this.data['section_id'][i] } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data["ra"][i], 4 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data["dec"][i], 4 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data["gallat"][i], 1 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data["fwhm_estimate"][i], 2 ) } );
            td = rkWebUtil.elemaker( "td", tr,
                                     { "text": seechange.nullorfixed( this.data["zero_point_estimate"][i], 2 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text":
                                                 seechange.nullorfixed( this.data["lim_mag_estimate"][i], 1 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": this.data["numsources"][i] } );
            td = rkWebUtil.elemaker( "td", tr, { "text": this.data["nummeasurements"][i] } );

            td = rkWebUtil.elemaker( "td", tr );
            tiptext = "";
            let laststep = "(none)";
            for ( let j of Object.keys( seechange.Exposure.process_steps ) ) {
                if ( this.data["progress_steps_bitflag"][i] & ( 2**j ) ) {
                    tiptext += seechange.Exposure.process_steps[j] + " done<br>";
                    laststep = seechange.Exposure.process_steps[j];
                } else {
                    tiptext += "(" + seechange.Exposure.process_steps[j] + " not done)<br>";
                }
            }
            span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ],
                                                     "text": laststep } );
            ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
            ttspan.innerHTML = tiptext;

            td = rkWebUtil.elemaker( "td", tr );
            tiptext = "Products created:";
            for ( let j of Object.keys( seechange.Exposure.pipeline_products ) ) {
                if ( this.data["products_exist_bitflag"][i] & ( 2**j ) )
                    tiptext += "<br>" + seechange.Exposure.pipeline_products[j];
            }
            span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ],
                                                     "text": "data products" } );
            ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
            ttspan.innerHTML = tiptext;

            // Really I should be doing some HTML sanitization here on error message and, below, warnings....

            td = rkWebUtil.elemaker( "td", tr );
            if ( this.data["error_step"][i] != null ) {
                span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ],
                                                         "text": "error" } );
                tiptext = ( this.data["error_type"][i] + " error in step " +
                            seechange.Exposure.process_steps[this.data["error_step"][i]] +
                            " (" + this.data["error_message"][i].replaceAll( "\n", "<br>") + ")" );
                ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
                ttspan.innerHTML = tiptext;
            }

            td = rkWebUtil.elemaker( "td", tr );
            if ( ( this.data["warnings"][i] != null ) && ( this.data["warnings"][i].length > 0 ) ) {
                span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ],
                                                         "text": "warnings" } );
                ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
                ttspan.innerHTML = this.data["warnings"][i].replaceAll( "\n", "<br>" );
            }
        }


        this.cutoutsdiv = rkWebUtil.elemaker( "div", null, { 'id': 'exposurecutoutsdiv' } );

        // TODO : buttons for next, prev, etc.

        this.tabs.addTab( "Images", "Images", this.imagesdiv, true );
        this.tabs.addTab( "Cutouts", "Sources", this.cutoutsdiv, false, ()=>{ self.update_cutouts() } );
    };


    update_cutouts()
    {
        var self = this;

        rkWebUtil.wipeDiv( this.cutoutsdiv );

        let withnomeas = this.cutoutssansmeasurements_checkbox.checked ? 1 : 0;

        if ( this.cutoutsallimages_checkbox.checked ) {
            rkWebUtil.elemaker( "p", this.cutoutsdiv,
                                { "text": "Sources for all successfully completed chips" } );
            let div = rkWebUtil.elemaker( "div", this.cutoutsdiv );
            rkWebUtil.elemaker( "p", div,
                                { "text": "...updating cutouts...",
                                  "classes": [ "bold", "italic", "warning" ] } )

            // TODO : offset and limit

            let prop = "cutouts_for_all_images_for_exposure_" + withnomeas;
            if ( this.cutouts_pngs.hasOwnProperty( prop ) ) {
                this.show_cutouts_for_image( div, prop, this.cutouts_pngs[ prop ] );
            }
            else {
                this.context.connector.sendHttpRequest(
                    "png_cutouts_for_sub_image/" + this.id + "/" + this.data.provenancetag + "/0/" + withnomeas,
                    {},
                    (data) => { self.show_cutouts_for_image( div, prop, data ); }
                );
            }
        }
        else {
            for ( let i in this.data['id'] ) {
                if ( this.cutoutsimage_checkboxes[this.data['id'][i]].checked ) {
                    rkWebUtil.elemaker( "p", this.cutoutsdiv,
                                        { "text": "Sources for chip " + this.data['section_id'][i]
                                          + " (image " + this.data['name'][i] + ")" } );

                    let div = rkWebUtil.elemaker( "div", this.cutoutsdiv, { 'id': 'exposureactualcutoutsdiv' } );
                    rkWebUtil.elemaker( "p", div,
                                        { "text": "...updating cutouts...",
                                          "classes": [ "bold", "italic", "warning" ] } )

                    // TODO : offset and limit

                    let prop = this.data['id'][i].toString() + "_" + withnomeas;

                    if ( this.cutouts_pngs.hasOwnProperty( prop ) ) {
                        this.show_cutouts_for_image( div, prop, this.cutouts_pngs[ prop ] );
                    }
                    else {
                        this.context.connector.sendHttpRequest(
                            "png_cutouts_for_sub_image/" + this.data['subid'][i] + "/" + this.data.provenancetag +
                                "/1/" + withnomeas,
                            {},
                            (data) => { self.show_cutouts_for_image( div, prop, data ); }
                        );
                    }

                    return;
                }
            }
        }
    };


    show_cutouts_for_image( div, dex, indata )
    {
        var table, tr, th, td, img;
        var oversample = 5;

        if ( ! this.cutouts_pngs.hasOwnProperty( dex ) )
            this.cutouts_pngs[dex] = indata;

        var data = this.cutouts_pngs[dex];

        rkWebUtil.wipeDiv( div );

        table = rkWebUtil.elemaker( "table", div, { 'id': 'exposurecutoutstable' } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr );
        th = rkWebUtil.elemaker( "th", tr, { "text": "new" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "ref" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "sub" } );

        // Sorting is now done server-side... TODO, think about this
        // // TODO : sort by r/b, make sort configurable
        // let dexen = [...Array(data.cutouts.sub_id.length).keys()];
        // dexen.sort( (a, b) => {
        //     if ( ( data.cutouts['flux'][a] == null ) && ( data.cutouts['flux'][b] == null ) ) return 0;
        //     else if ( data.cutouts['flux'][a] == null ) return 1;
        //     else if ( data.cutouts['flux'][b] == null ) return -1;
        //     else if ( data.cutouts['flux'][a] > data.cutouts['flux'][b] ) return -1;
        //     else if ( data.cutouts['flux'][a] < data.cutouts['flux'][b] ) return 1;
        //     else return 0;
        // } );

        // for ( let i of dexen ) {
        for ( let i in data.cutouts.sub_id ) {
            tr = rkWebUtil.elemaker( "tr", table );
            td = rkWebUtil.elemaker( "td", tr );
            if ( data.cutouts.objname[i] != null ) {
                let text = "Object: " + data.cutouts.objname[i];
                if ( data.cutouts.is_fake[i] ) text += " [FAKE]";
                if ( data.cutouts.is_test[i] ) text += " [TEST]";
                td.appendChild( document.createTextNode( text ) );
            }
            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts['new_png'][i],
                                          "width": oversample * data.cutouts['w'][i],
                                          "height": oversample * data.cutouts['h'][i],
                                          "alt": "new" } } );
            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts['ref_png'][i],
                                          "width": oversample * data.cutouts['w'][i],
                                          "height": oversample * data.cutouts['h'][i],
                                          "alt": "ref" } } );
            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts['sub_png'][i],
                                          "width": oversample * data.cutouts['w'][i],
                                          "height": oversample * data.cutouts['h'][i],
                                          "alt": "sub" } } );

            td = rkWebUtil.elemaker( "td", tr );
            let subdiv = rkWebUtil.elemaker( "div", td );
            // TODO: use "warning" color for low r/b
            if ( ( data.cutouts['flux'][i] == null ) ||
                 ( data.cutouts['rb'][i] == null ) ||
                 ( data.cutouts['rb'][i] < data.cutouts['rbcut'][i] )
               )
                td.classList.add( 'bad' )
            else td.classList.add( 'good' );
            let textblob = ( "<b>chip:</b> " + data.cutouts.section_id[i] + "<br>" +
                             // "<b>cutout (α, δ):</b> (" + data.cutouts['ra'][i].toFixed(5) + " , "
                             // + data.cutouts['dec'][i].toFixed(5) + ")<br>" +
                             "<b>(α, δ):</b> (" + seechange.nullorfixed( data.cutouts['measra'][i], 5 ) + " , "
                             + seechange.nullorfixed( data.cutouts['measdec'][i],5 ) + ")<br>" +
                             // TODO : put x, y back if the server ever starts returning it again! -- Issue #340
                             // "<b>(x, y):</b> (" + data.cutouts['x'][i].toFixed(2) + " , "
                             // + data.cutouts['y'][i].toFixed(2) + ")<br>" +
                             "<b>Flux:</b> " + seechange.nullorfixed( data.cutouts['flux'][i], 0 )
                             + " ± " + seechange.nullorfixed( data.cutouts['dflux'][i], 0 )
                           );
            if ( ( data.cutouts['aperrad'][i] == null ) || ( data.cutouts['aperrad'][i] <= 0 ) )
                textblob += "  (psf)";
            else
                textblob +=  + "  (aper r=" + seechange.nullorfixed( data.cutouts['aperrad'][i], 2) + " px)";
            textblob += ("<br>" + "<b>Mag:</b> " + seechange.nullorfixed( data.cutouts['mag'][i], 2 )
                         + " ± " + seechange.nullorfixed( data.cutouts['dmag'][i], 2 )
                        );
            textblob += "<br><b>R/B:</b> " + seechange.nullorfixed( data.cutouts['rb'][i], 3 );
            subdiv.innerHTML = textblob;
        }
    };
}


// **********************************************************************
// **********************************************************************
// **********************************************************************
// Make this into a module

export { seechange }

