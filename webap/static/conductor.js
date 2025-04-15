import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js";


// **********************************************************************

seechange.Conductor = class
{

    constructor( context )
    {
        this.context = context;
        this.div = rkWebUtil.elemaker( "div", null, { 'id': 'conductordiv' } );
        this.connector = this.context.connector;
        this.hide_exposure_details_checkbox = null;
    };

    // **********************************************************************

    render()
    {
        let self = this;

        let p, h3, span, hbox, vbox, subhbox;

        rkWebUtil.wipeDiv( this.div );
        this.frontpagediv = rkWebUtil.elemaker( "div", this.div );

        this.pipelineworkers = new seechange.PipelineWorkers( this.context, this );

        hbox = rkWebUtil.elemaker( "div", this.frontpagediv, { "classes": [ "hbox" ] } );
        this.pollingdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "conductorconfig" ] } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "conductorconfig" ] } );
        h3 = rkWebUtil.elemaker( "h3", vbox, { "text": "Pipeline Config  " } );
        rkWebUtil.button( h3, "Refresh", () => { self.show_config_status() } );
        p = rkWebUtil.elemaker( "p", vbox, { "text": "Run through step " } )
        this.throughstep_select = rkWebUtil.elemaker( "select", p );
        for ( let step of seechange.Conductor.ALL_STEPS ) {
            rkWebUtil.elemaker( "option", this.throughstep_select,
                                { "text": step,
                                  "attributes": { "value": step,
                                                  "id": "throughstep_select",
                                                  "name": "throughstep_select",
                                                  "selected": ( step=='alerting' ) ? 1 : 0 } } );
        }

        // UNCOMMENT ALL THIS WHEN IT'S ACTUALLY IMPLEMENTED -- see Issue #446
        // The conductor doesn't currently consider this when choosing exposures to assign.
        // p = rkWebUtil.elemaker( "p", vbox );
        // this.pickup_partial_checkbox = rkWebUtil.elemaker( "input", p,
        //                                                    { "attributes": { "type": "checkbox",
        //                                                                      "id": "pickup_partial_checkbox",
        //                                                                      "name": "pickup_partial_checkbox" } } );
        // p.appendChild( document.createTextNode( " run partially completed exposures?" ) );

        hbox.appendChild( this.pipelineworkers.div );

        this.contentdiv = rkWebUtil.elemaker( "div", this.frontpagediv );

        rkWebUtil.elemaker( "hr", this.contentdiv );

        p = rkWebUtil.elemaker( "p", this.contentdiv );
        rkWebUtil.button( p, "Refresh", () => { self.update_known_exposures(); } );
        p.appendChild( document.createTextNode( " known exposures taken from " ) );
        this.knownexp_mintwid = rkWebUtil.elemaker( "input", p, { "attributes": { "size": 20 } } );
        p.appendChild( document.createTextNode( " to " ) );
        this.knownexp_maxtwid = rkWebUtil.elemaker( "input", p, { "attributes": { "size": 20 } } );
        p.appendChild( document.createTextNode( " UTC (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)" ) );

        this.knownexpdiv = rkWebUtil.elemaker( "div", this.contentdiv );

        this.show_config_status();
        this.pipelineworkers.render();
    }

    // **********************************************************************

    show_config_status( edit=false )
    {
        var self = this;

        let p;

        rkWebUtil.wipeDiv( this.pollingdiv )
        rkWebUtil.elemaker( "p", this.pollingdiv,
                            { "text": "Loading status...",
                              "classes": [ "warning", "bold", "italic" ] } )

        if ( edit )
            this.connector.sendHttpRequest( "conductor/status", {}, (data) => { self.edit_config_status(data) } );
        else
            this.connector.sendHttpRequest( "conductor/status", {},
                                            (data) => { self.actually_show_config_status(data) } );
    }

    // **********************************************************************

    actually_show_config_status( data )
    {
        let self = this;

        let table, tr, th, td, p;

        rkWebUtil.wipeDiv( this.pollingdiv );
        rkWebUtil.elemaker( "h3", this.pollingdiv,
                            { "text": "Conductor polling config" } );

        p = rkWebUtil.elemaker( "p", this.pollingdiv );
        rkWebUtil.button( p, "Refresh", () => { self.show_config_status() } );
        p.appendChild( document.createTextNode( "  " ) );
        rkWebUtil.button( p, "Modify", () => { self.show_config_status( true ) } );

        if ( data.pause )
            rkWebUtil.elemaker( "p", this.pollingdiv, { "text": "Automatic updating is paused." } )
        if ( data.hold )
            rkWebUtil.elemaker( "p", this.pollingdiv, { "text": "Newly added known exposures are being held." } )

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

        table = rkWebUtil.elemaker( "table", this.pollingdiv );
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
        th = rkWebUtil.elemaker( "th", tr, { "text": "Min Exp. Time" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": minexptime } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "Projects" } );
        td = rkWebUtil.elemaker( "td", tr, { "text": projects } );

        this.forceconductorpoll_p = rkWebUtil.elemaker( "p", this.pollingdiv );
        rkWebUtil.button( this.forceconductorpoll_p, "Force Conductor Poll", () => { self.force_conductor_poll(); } );

        this.throughstep_select.value = data.throughstep;
        this.partial_pickup_checkbox = ( data.pickuppartial ? 1 : 0 );
    }

    // **********************************************************************

    edit_config_status( data )
    {
        let self = this;

        let table, tr, th, td, p;

        rkWebUtil.wipeDiv( this.pollingdiv );
        rkWebUtil.elemaker( "h3", this.pollingdiv,
                            { "text": "Conductor polling config" } );

        p = rkWebUtil.elemaker( "p", this.pollingdiv );
        rkWebUtil.button( p, "Save Changes", () => { self.update_conductor_config(); } );
        p.appendChild( document.createTextNode( "  " ) );
        rkWebUtil.button( p, "Cancel", () => { self.show_config_status() } );

        p = rkWebUtil.elemaker( "p", this.pollingdiv );
        this.status_pause_wid = rkWebUtil.elemaker( "input", p, { "attributes": { "type": "checkbox",
                                                                                  "id": "status_pause_checkbox" } } );
        if ( data.pause ) this.status_pause_wid.setAttribute( "checked", "checked" );
        rkWebUtil.elemaker( "label", p, { "text": "Pause automatic updating",
                                          "attributes": { "for": "status_pause_checkbox" } } );

        p = rkWebUtil.elemaker( "p", this.pollingdiv );
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

        table = rkWebUtil.elemaker( "table", this.pollingdiv );
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

        this.connector.sendHttpRequest( "conductor/updateparameters",
                                        { 'instrument': instrument,
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
        this.connector.sendHttpRequest( "conductor/forceupdate", {}, () => self.did_force_conductor_poll() );
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

    update_known_exposures()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.knownexpdiv );
        let p = rkWebUtil.elemaker( "p", this.knownexpdiv,
                                    { "text": "Loading known exposures...",
                                      "classes": [ "warning", "bold", "italic" ] } );
        let url = "conductor/getknownexposures";
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

        let table, tr, td, th, p, button, span, ttspan, hide_exposure_details;

        this.known_exposures = [];
        this.known_exposure_checkboxes = {};
        this.known_exposure_rows = {};
        this.known_exposure_hold_tds = {};
        // this.known_exposure_checkbox_manual_state = {};

        rkWebUtil.wipeDiv( this.knownexpdiv );

        p = rkWebUtil.elemaker( "p", this.knownexpdiv );

        if ( this.hide_exposure_details_checkbox == null ) {
            this.hide_exposure_details_checkbox =
                rkWebUtil.elemaker( "input", null,
                                    { "change": () => { self.show_known_exposures( data ) },
                                      "attributes": { "type": "checkbox",
                                                      "id": "knownexp-hide-exposure-details-checkbox" } } );
        }
        p.appendChild( this.hide_exposure_details_checkbox );
        p.appendChild( document.createTextNode( "Hide exposure detail columns    " ) );
        hide_exposure_details = this.hide_exposure_details_checkbox.checked;

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
        button = rkWebUtil.button( p, "Delete", () => { self.delete_known_exposures() } );
        button.classList.add( "hmargin" );
        button = rkWebUtil.button( p, "Hold", () => { self.hold_release_exposures( true ); } );
        button.classList.add( "hmargin" );
        button = rkWebUtil.button( p, "Release", () => { self.hold_release_exposures( false ); } );
        button.classList.add( "hmargin" );
        button = rkWebUtil.button( p, "Clear Cluster Claim", () => { self.clear_cluster_claim() } );
        button.classList.add( "hmargin" );

        table = rkWebUtil.elemaker( "table", this.knownexpdiv, { "classes": [ "borderedcells" ] } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr );
        th = rkWebUtil.elemaker( "th", tr, { "text": "held?" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "instrument" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "identifier" } );
        if ( ! hide_exposure_details ) {
            th = rkWebUtil.elemaker( "th", tr, { "text": "mjd" } );
            th = rkWebUtil.elemaker( "th", tr, { "text": "target" } );
            th = rkWebUtil.elemaker( "th", tr, { "text": "ra" } );
            th = rkWebUtil.elemaker( "th", tr, { "text": "dec" } );
            th = rkWebUtil.elemaker( "th", tr, { "text": "b" } );
            th = rkWebUtil.elemaker( "th", tr, { "text": "filter" } );
            th = rkWebUtil.elemaker( "th", tr, { "text": "exp_time" } );
            th = rkWebUtil.elemaker( "th", tr, { "text": "project" } );
        }
        th = rkWebUtil.elemaker( "th", tr, { "text": "cluster" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "claim_time" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "relase_time" } );
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
            //         console.log( "Setting " + ke.id + " to " + self.known_exposure_checkboxes[ ke.id ].checked );
            //     } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.hold ? "***" : "" } );
            this.known_exposure_hold_tds[ ke.id ] = td;
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.instrument } );
            td = rkWebUtil.elemaker( "td", tr, { "text": ke.identifier } );
            if ( ! hide_exposure_details ) {
                td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.mjd ).toFixed( 5 ) } );
                td = rkWebUtil.elemaker( "td", tr, { "text": ke.target } );
                td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.ra ).toFixed( 5 ) } );
                td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.dec ).toFixed( 5 ) } );
                td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.gallat ).toFixed( 1 ) } );
                td = rkWebUtil.elemaker( "td", tr, { "text": ke.filter } );
                td = rkWebUtil.elemaker( "td", tr, { "text": parseFloat( ke.exp_time ).toFixed( 1 ) } );
                td = rkWebUtil.elemaker( "td", tr, { "text": ke.project } );
            }
            td = rkWebUtil.elemaker( "td", tr );
            span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ], "text": ke.cluster_id } );
            ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } )
            ttspan.innerHTML = "node: " + ke.node_id + "<br>machine: " + ke.machine_id;
            td = rkWebUtil.elemaker( "td", tr,
                                     { "text": ( ke.claim_time == null ) ?
                                       "" : rkWebUtil.dateUTCFormat(rkWebUtil.parseDateAsUTC(ke.claim_time)) } );
            td = rkWebUtil.elemaker( "td", tr,
                                     { "text": ( ke.release_time == null ) ?
                                       "" : rkWebUtil.dateUTCFormat(rkWebUtil.parseDateAsUTC(ke.release_time)) } );
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
            let url = hold ? "conductor/holdexposures" : "conductor/releaseexposures"
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

    // **********************************************************************

    delete_known_exposures()
    {
        let self = this;

        let todel = [];
        for ( let ke of this.known_exposures ) {
            if ( this.known_exposure_checkboxes[ ke.id ].checked )
                todel.push( ke.id );
        }

        if ( todel.length > 0 ) {
            if ( window.confirm( "Delete " + todel.length.toString() + " known exposures? " +
                                 "(This cannot be undone.)" ) )
                this.connector.sendHttpRequest( "conductor/deleteknownexposures", { 'knownexposure_ids': todel },
                                                (data) => { self.process_delete_known_exposures(data, todel) } );
        }
    }

    // **********************************************************************

    process_delete_known_exposures( data, todel )
    {
        for ( let keid of todel ) {
            // Ugh, n²
            let dex = 0;
            while ( dex < this.known_exposures.length ) {
                if ( this.known_exposures[dex].id == keid )
                    this.known_exposures.splice( dex, 1 );
                else
                    dex += 1;
            }
            this.known_exposure_rows[ keid ].parentNode.removeChild( this.known_exposure_rows[ keid ] );
            delete this.known_exposure_rows[ keid ];
            delete this.known_exposure_checkboxes[ keid ];
            delete this.known_exposure_hold_tds[ keid ];
        }
    }

    // **********************************************************************

    clear_cluster_claim()
    {
        let self = this;

        let toclear = [];
        for ( let ke of this.known_exposures) {
            if ( this.known_exposure_checkboxes[ ke.id ].checked )
                toclear.push( ke.id );
        }

        if ( toclear.length > 0 ) {
            if ( window.confirm( "Clear cluster claim on " + toclear.length.toString() + " known exposures?" ) ) {
                rkWebUtil.wipeDiv( this.knownexposidv );
                rkWebUtil.elemaker( "p", this.knownexpdiv,
                                    { "text": "Loading known exposures...",
                                      "classes": [ "warning", "bold", "italic" ] } );
                this.connector.sendHttpRequest( "/conductor/clearclusterclaim",
                                                { 'knownexposure_ids': toclear },
                                                (data) => { self.update_known_exposures() } );
            }
        }
    }
}



// **********************************************************************

seechange.PipelineWorkers = class
{
    constructor( context, conductor )
    {
        this.context = context;
        this.conductor = conductor;
        this.connector = this.context.connector;
        this.div = rkWebUtil.elemaker( "div", null, { 'id': 'conductorworkers-div',
                                                      'classes': [ 'conductorworkers' ] } )
    };

    // **********************************************************************

    render()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.div );
        let hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox" ] } );
        this.workersdiv = rkWebUtil.elemaker( "div", hbox, {} );
        this.update_workers();
    };

    // **********************************************************************

    update_workers()
    {
        let self = this;
        this.connector.sendHttpRequest( "conductor/getworkers", {}, (data) => { self.show_workers(data); } );
    }

    // **********************************************************************

    show_workers( data )
    {
        let self = this;
        let table, tr, th, td, p, h3;

        rkWebUtil.wipeDiv( this.workersdiv );

        h3 = rkWebUtil.elemaker( "h3", this.workersdiv, { "text": "Known Pipeline Workers  " } );
        rkWebUtil.button( h3, "Refresh", () => { self.update_workers(); } );

        table = rkWebUtil.elemaker( "table", this.workersdiv, { "classes": [ "borderedcells" ] } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "cluster_id" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "node_id" } );
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
            td = rkWebUtil.elemaker( "td", tr, { "text": worker.cluster_id } );
            td = rkWebUtil.elemaker( "td", tr, { "text": worker.node_id } );
            td = rkWebUtil.elemaker( "td", tr,
                                     { "text": rkWebUtil.dateUTCFormat(
                                         rkWebUtil.parseDateAsUTC( worker.lastheartbeat ) ) } );
        }
    }

}

// **********************************************************************
// Keep this synced with top_level.py::Pipeline::ALL_STEPS

seechange.Conductor.ALL_STEPS = [ 'preprocessing', 'extraction', 'astrocal', 'photocal', 'subtraction',
                                  'detection', 'cutting', 'measuring', 'scoring', 'alerting' ];


// **********************************************************************
// **********************************************************************
// **********************************************************************
// Make this into a module

export { };
