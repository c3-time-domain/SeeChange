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

    // **********************************************************************

    render_page()
    {
        let self = this;

        let p, span, hbox;

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

        hbox = rkWebUtil.elemaker( "div", this.frontpagediv, { "classes": [ "hbox" ] } );

        this.configdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "conductorconfig" ] } );
        this.workersdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "conductorworkers" ] } );

        this.contentdiv = rkWebUtil.elemaker( "div", this.frontpagediv );

        rkWebUtil.elemaker( "hr", this.contentdiv );

        this.forceconductorpoll_p = rkWebUtil.elemaker( "p", this.contentdiv );
        rkWebUtil.button( this.forceconductorpoll_p, "Force Conductor Poll", () => { self.force_conductor_poll(); } );

        p = rkWebUtil.elemaker( "p", this.contentdiv );
        rkWebUtil.button( p, "Refresh", () => { self.update_known_exposures(); } );
        p.appendChild( document.createTextNode( " known exposures taken from " ) );
        this.knownexp_mintwid = rkWebUtil.elemaker( "input", p, { "attributes": { "size": 20 } } );
        p.appendChild( document.createTextNode( " to " ) );
        this.knownexp_maxtwid = rkWebUtil.elemaker( "input", p, { "attributes": { "size": 20 } } );
        p.appendChild( document.createTextNode( " UTC (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)" ) );

        this.knownexpdiv = rkWebUtil.elemaker( "div", this.contentdiv );

        this.show_config_status();
        this.update_workers_div();
    }

    // **********************************************************************

    show_config_status( edit=false )
    {
        var self = this;

        let p;

        rkWebUtil.wipeDiv( this.configdiv )
        rkWebUtil.elemaker( "p", this.configdiv,
                            { "text": "Loading status...",
                              "classes": [ "warning", "bold", "italic" ] } )

        if ( edit )
            this.connector.sendHttpRequest( "/status", {}, (data) => { self.edit_config_status(data) } );
        else
            this.connector.sendHttpRequest( "/status", {}, (data) => { self.actually_show_config_status(data) } );
    }

    // **********************************************************************

    actually_show_config_status( data )
    {
        let self = this;

        let table, tr, th, td, p;

        rkWebUtil.wipeDiv( this.configdiv );
        rkWebUtil.elemaker( "h3", this.configdiv,
                            { "text": "Conductor polling config" } );

        p = rkWebUtil.elemaker( "p", this.configdiv );
        rkWebUtil.button( p, "Refresh", () => { self.show_config_status() } );
        p.appendChild( document.createTextNode( "  " ) );
        rkWebUtil.button( p, "Modify", () => { self.show_config_status( true ) } );

        if ( data.pause )
            rkWebUtil.elemaker( "p", this.configdiv, { "text": "Automatic updating is paused." } )
        if ( data.hold )
            rkWebUtil.elemaker( "p", this.configdiv, { "text": "Newly added known exposures are being held." } )

        let instrument = ( data.instrument == null ) ? "" : data.instrument;
        let minmjd = "(None)";
        let maxmjd = "(None)";
        let minexptime = "(None)";
        let projects = "(Any)";
        if ( data.updateargs != null ) {
            minmjd = data.updateargs.hasOwnProperty( "minmjd" ) ? data.updateargs.minmjd : minmjd;
            maxmjd = data.updateargs.hasOwnProperty( "maxmjd" ) ? data.updateargs.maxmjd : maxmjd;
            minexptime = data.updateargs.hasOwnProperty( "minexptime" ) ? data.updateargs.minexptime : minexptime;
            projects = data.updateargs.hasOwnProperty( "projects" ) ? data.updateargs.projects.join(",") : projects;
        }

        table = rkWebUtil.elemaker( "table", this.configdiv );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Instrument" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": data.instrument } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Min MJD" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": minmjd } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Max MJD" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": maxmjd } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Max Exp. Time" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": minexptime } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Projects" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": projects } );
    }

    // **********************************************************************

    edit_config_status( data )
    {
        let self = this;

        let table, tr, th, td, p;

        rkWebUtil.wipeDiv( this.configdiv );
        rkWebUtil.elemaker( "h3", this.configdiv,
                            { "text": "Conductor polling config" } );

        p = rkWebUtil.elemaker( "p", this.configdiv );
        rkWebUtil.button( p, "Save Changes", () => { self.update_conductor_config(); } );
        p.appendChild( document.createTextNode( "  " ) );
        rkWebUtil.button( p, "Cancel", () => { self.show_config_status() } );

        p = rkWebUtil.elemaker( "p", this.configdiv );
        this.status_pause_wid = rkWebUtil.elemaker( "input", p, { "attributes": { "type": "checkbox",
                                                                                  "id": "status_pause_checkbox" } } );
        if ( data.pause ) this.status_pause_wid.setAttribute( "checked", "checked" );
        rkWebUtil.elemaker( "label", p, { "text": "Pause automatic updating",
                                          "attributes": { "for": "status_pause_checkbox" } } );

        p = rkWebUtil.elemaker( "p", this.configdiv );
        this.status_hold_wid = rkWebUtil.elemaker( "input", p, { "attributes": { "type": "checkbox",
                                                                                 "id": "status_hold_checkbox" } } );
        if ( data.hold ) this.status_hold_wid.setAttribute( "checked", "checked" );
        rkWebUtil.elemaker( "label", p, { "text": "Hold newly added exposures",
                                          "attributes": { "for": "status_hold_checkbox" } } );


        let minmjd = "";
        let maxmjd = "";
        let minexptime = "";
        let projects = "";
        if ( data.updateargs != null ) {
            minmjd = data.updateargs.hasOwnProperty( "minmjd" ) ? data.updateargs.minmjd : minmjd;
            maxmjd = data.updateargs.hasOwnProperty( "maxmjd" ) ? data.updateargs.maxmjd : maxmjd;
            minexptime = data.updateargs.hasOwnProperty( "minexptime" ) ? data.updateargs.minexptime : minexptime;
            projects = data.updateargs.hasOwnProperty( "projects" ) ? data.updateargs.projects.join(",") : projects;
        }
        let instrument = ( data.instrument == null ) ? "" : data.instrument;

        table = rkWebUtil.elemaker( "table", this.configdiv );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Instrument" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.status_instrument_wid = rkWebUtil.elemaker( "input", td,
                                                         { "attributes": { "value": instrument,
                                                                           "size": 20 } } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Start time" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.status_minmjd_wid = rkWebUtil.elemaker( "input", td,
                                                     { "attributes": { "value": minmjd,
                                                                       "size": 20 } } );
        td = rkWebUtil.elemaker( "td", tr, { "text": " (MJD or YYYY-MM-DD HH:MM:SS)" } )
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "End time" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.status_maxmjd_wid = rkWebUtil.elemaker( "input", td,
                                                     { "attributes": { "value": maxmjd,
                                                                       "size": 20 } } );
        td = rkWebUtil.elemaker( "td", tr, { "text": " (MJD or YYYY-MM-DD HH:MM:SS)" } )
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Max Exp. Time" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.status_minexptime_wid = rkWebUtil.elemaker( "input", td,
                                                         { "attributes": { "value": minexptime,
                                                                           "size": 20 } } );
        td = rkWebUtil.elemaker( "td", tr, { "text": " seconds" } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Projects" } );
        td = rkWebUtil.elemaker( "td", tr );
        this.status_projects_wid = rkWebUtil.elemaker( "input", td,
                                                       { "attributes": { "value": projects,
                                                                         "size": 20 } } );
        td = rkWebUtil.elemaker( "td", tr, { "text": " (comma-separated)" } );
    }


    // **********************************************************************

    update_conductor_config()
    {
        let self = this;

        let instrument = this.status_instrument_wid.value.trim();
        instrument = ( instrument.length == 0 ) ? null : instrument;

        // Parsing is often verbose
        let minmjd = this.status_minmjd_wid.value.trim();
        if ( minmjd.length == 0 )
            minmjd = null;
        else if ( minmjd.search( /^ *([0-9]*\.)?[0-9]+ *$/ ) >= 0 )
            minmjd = parseFloat( minmjd );
        else {
            try {
                minmjd = rkWebUtil.mjdOfDate( rkWebUtil.parseDateAsUTC( minmjd ) );
            } catch (e) {
                window.alert( e );
                return;
            }
        }

        let maxmjd = this.status_maxmjd_wid.value.trim();
        if ( maxmjd.length == 0 )
            maxmjd = null;
        else if ( maxmjd.search( /^ *([0-9]*\.)?[0-9]+ *$/ ) >= 0 )
            maxmjd = parseFloat( maxmjd );
        else {
            try {
                maxmjd = rkWebUtil.mjdOfDate( rkWebUtil.parseDateAsUTC( maxmjd ) );
            } catch (e) {
                window.alert( e );
                return;
            }
        }

        let minexptime = this.status_minexptime_wid.value.trim();
        minexptime = ( minexptime.length == 0 ) ? null : parseFloat( minexptime );

        let projects = this.status_projects_wid.value.trim();
        if ( projects.length == 0 )
            projects = null;
        else {
            let tmp = projects.split( "," );
            projects = [];
            for ( let project of tmp ) projects.push( project.trim() );
        }

        let params = {};
        if ( minmjd != null ) params['minmjd'] = minmjd;
        if ( maxmjd != null ) params['maxmjd'] = maxmjd;
        if ( minexptime != null ) params['minexptime'] = minexptime;
        if ( projects != null ) params['projects'] = projects;
        if ( Object.keys(params).length == 0 ) params = null;

        this.connector.sendHttpRequest( "/updateparameters", { 'instrument': instrument,
                                                               'pause': this.status_pause_wid.checked ? 1 : 0,
                                                               'hold': this.status_hold_wid.checked ? 1 : 0,
                                                               'updateargs': params },
                                        () => self.show_config_status() );
    }

    // **********************************************************************

    force_conductor_poll()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.forceconductorpoll_p );
        rkWebUtil.elemaker( "span", this.forceconductorpoll_p,
                            { "text": "...forcing conductor poll...",
                              "classes": [ "warning", "bold", "italic" ] } );
        this.connector.sendHttpRequest( "/forceupdate", {}, () => self.did_force_conductor_poll() );
    }

    // **********************************************************************

    did_force_conductor_poll()
    {
        let self = this;
        rkWebUtil.wipeDiv( this.forceconductorpoll_p );
        rkWebUtil.button( this.forceconductorpoll_p, "Force Conductor Poll", () => { self.force_conductor_poll(); } );
        this.update_known_exposures();
    }


    // **********************************************************************

    update_workers_div()
    {
        let self = this;
        rkWebUtil.wipeDiv( this.workersdiv );
        rkWebUtil.elemaker( "h3", this.workersdiv, { "text": "Known Pipeline Workers" } );
        this.connector.sendHttpRequest( "/getworkers", {}, (data) => { self.show_workers(data); } );
    }

    // **********************************************************************

    show_workers( data )
    {
        let self = this;
        let table, tr, th, td, p;

        p = rkWebUtil.elemaker( "p", this.workersdiv );
        rkWebUtil.button( p, "Refresh", () => { self.update_workers_div(); } );

        table = rkWebUtil.elemaker( "table", this.workersdiv, { "classes": [ "borderedcells" ] } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "id" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "cluster_id" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "node_id" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "nexps" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "last heartbeat" } );

        let grey = 0;
        let coln = 3;
        for ( let worker of data['workers'] ) {
            if ( coln == 0 ) {
                grey = 1 - grey;
                coln = 3;
            }
            coln -= 1;
            tr = rkWebUtil.elemaker( "tr", table );
            if ( grey ) tr.classList.add( "greybg" );
            td = rkWebUtil.elemaker( "td", tr, { "text": worker.id } );
            td = rkWebUtil.elemaker( "td", tr, { "text": worker.cluster_id } );
            td = rkWebUtil.elemaker( "td", tr, { "text": worker.node_id } );
            td = rkWebUtil.elemaker( "td", tr, { "text": worker.nexps } );
            td = rkWebUtil.elemaker( "td", tr,
                                     { "text": rkWebUtil.dateUTCFormat(
                                         rkWebUtil.parseDateAsUTC( worker.lastheartbeat ) ) } );
        }
    }

    // **********************************************************************

    update_known_exposures()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.knownexpdiv );
        let p = rkWebUtil.elemaker( "p", this.knownexpdiv,
                                    { "text": "Loading known exposures...",
                                      "classes": [ "warning", "bold", "italic" ] } );
        let url = "/getknownexposures";
        if ( this.knownexp_mintwid.value.trim().length > 0 ) {
            let minmjd = rkWebUtil.mjdOfDate( rkWebUtil.parseDateAsUTC( this.knownexp_mintwid.value ) );
            url += "/minmjd=" + minmjd.toString();
        }
        if ( this.knownexp_maxtwid.value.trim().length > 0 ) {
            let maxmjd = rkWebUtil.mjdOfDate( rkWebUtil.parseDateAsUTC( this.knownexp_maxtwid.value ) );
            url += "/maxmjd=" + maxmjd.toString();
        }
        this.connector.sendHttpRequest( url, {}, (data) => { self.show_known_exposures(data); } );
    }

    // **********************************************************************

    show_known_exposures( data )
    {
        let self = this;

        let table, tr, td, th, p, button;

        this.known_exposures = [];
        this.known_exposure_checkboxes = {};
        this.known_exposure_rows = {};
        this.known_exposure_hold_tds = {};
        // this.known_exposure_checkbox_manual_state = {};

        rkWebUtil.wipeDiv( this.knownexpdiv );

        p = rkWebUtil.elemaker( "p", this.knownexpdiv );
        this.select_all_checkbox = rkWebUtil.elemaker( "input", p,
                                                       { "attributes": {
                                                           "type": "checkbox",
                                                           "id": "knownexp-select-all-checkbox" } } );
        rkWebUtil.elemaker( "label", p, { "text": "Select all",
                                          "attributes": { "for": "knownexp-select-all-checkbox" } } );
        this.select_all_checkbox.addEventListener(
            "change",
            () => {
                for ( let ke of self.known_exposures ) {
                    self.known_exposure_checkboxes[ ke.id ].checked = self.select_all_checkbox.checked;
                }
            } );
        p.appendChild( document.createTextNode( "      Apply to selected: " ) );
        button = rkWebUtil.button( p, "Delete", () => { window.alert( "Not implemented." ) } );
        button.classList.add( "hmargin" );
        button = rkWebUtil.button( p, "Hold", () => { self.hold_release_exposures( true ); } );
        button.classList.add( "hmargin" );
        button = rkWebUtil.button( p, "Release", () => { self.hold_release_exposures( false ); } );
        button.classList.add( "hmargin" );
        button = rkWebUtil.button( p, "Clear Cluster Claim", () => { window.alert( "Not implemented." ) } );
        button.classList.add( "hmargin" );

        table = rkWebUtil.elemaker( "table", this.knownexpdiv, { "classes": [ "borderedcells" ] } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr );
        th = rkWebUtil.elemaker( "th", tr, { "text": "held?" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "instrument" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "identifier" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "mjd" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "target" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "ra" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "dec" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "b" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "filter" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "exp_time" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "project" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "cluster" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "claim_time" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "exposure" } );

        let grey = 0;
        let coln = 3;
        for ( let ke of data.knownexposures ) {
            if ( coln == 0 ) {
                grey = 1 - grey;
                coln = 3;
            }
            coln -= 1;

            this.known_exposures.push( ke );

            tr = rkWebUtil.elemaker( "tr", table );
            if ( grey ) tr.classList.add( "greybg" );
            if ( ke.hold ) tr.classList.add( "heldexposure" );
            this.known_exposure_rows[ ke.id ] = tr;

            td = rkWebUtil.elemaker( "td", tr );
            this.known_exposure_checkboxes[ ke.id ] =
                rkWebUtil.elemaker( "input", td, { "attributes": { "type": "checkbox" } } );
            // this.known_exposure_checkbox_manual_state[ ke.id ] = 0;
            // this.known_exposure_checkboxes[ ke.id ].addEventListener(
            //     "click", () => {
            //         self.known_exposure_checkbox_manual_state[ ke.id ] =
            //             ( self.known_exposure_checkboxes[ ke.id ].checked ? 1 : 0 );
            //         console.log( "Setting " + ke.id + " to " + self.known_exposures_checkboxes[ ke.id ].checked );
            //     } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.hold ? "***" : "" } );
            this.known_exposure_hold_tds[ ke.id ] = td;
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.instrument } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.identifier } );
            td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.mjd ).toFixed( 5 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.target } );
            td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.ra ).toFixed( 5 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.dec ).toFixed( 5 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.gallat ).toFixed( 3 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.filter } );
            td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.exp_time ).toFixed( 1 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.project } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.cluster_id } );
            td = rkWebUtil.elemaker( "td", tr,
                                     { "text": ( ke.claim_time == null ) ?
                                       "" : rkWebUtil.dateUTCFormat(rkWebUtil.parseDateAsUTC(ke.claim_time)) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.exposure_id } );
        }
    }

    // **********************************************************************

    hold_release_exposures( hold )
    {
        let self = this;

        let tohold = [];
        for ( let ke of this.known_exposures ) {
            if ( this.known_exposure_checkboxes[ ke.id ].checked )
                tohold.push( ke.id );
        }

        if ( tohold.length > 0 ) {
            let url = hold ? "/holdexposures" : "/releaseexposures"
            this.connector.sendHttpRequest( url, { 'knownexposure_ids': tohold },
                                            (data) => { self.process_hold_release_exposures(data, hold); } );
        }
    }

    // **********************************************************************

    process_hold_release_exposures( data, hold )
    {
        for ( let keid of data[ hold ? "held" : "released" ] ) {
            if ( this.known_exposure_rows.hasOwnProperty( keid ) ) {
                if ( hold ) {
                    this.known_exposure_rows[ keid ].classList.add( "heldexposure" );
                    this.known_exposure_hold_tds[ keid ].innerHTML = "***";
                } else {
                    this.known_exposure_rows[ keid ].classList.remove( "heldexposure" );
                    this.known_exposure_hold_tds[ keid ].innerHTML = "";
                }
            }
        }
        if ( data['missing'].length != 0 )
            console.log( "WARNING : tried to hold/release the following unknown knownexposures: " + data['missing'] );
    }



}

// **********************************************************************
// **********************************************************************
// **********************************************************************
// Make this into a module

export { scconductor };
