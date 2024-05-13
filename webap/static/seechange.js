import { rkWebUtil } from "./rkwebutil.js";

// Namespace, which is the only thing exported

var seechange = {};

seechange.nullorfixed = function( val, num ) { return val == null ? null : val.toFixed(num); }

// **********************************************************************
// **********************************************************************
// **********************************************************************
// The global context

seechange.Context = function()
{
    this.parentdiv = document.getElementById( "pagebody" );
    let h1 = rkWebUtil.elemaker( "h1", this.parentdiv, { "text": "SeeChange Webap" } );
    this.maindiv = rkWebUtil.elemaker( "div", this.parentdiv );
    this.frontpagediv = null;

    // TODO : make this configurable?  Or at least remember how to
    //   detect in javascript what the URL you're running from is.  (In
    //   case the webap is not running as the root ap of the webserver.)
    this.connector = new rkWebUtil.Connector( "/" );
}

seechange.Context.prototype.render_page = function()
{
    var self = this;

    if ( this.frontpagediv == null ) {

        // TODO : users, login

        this.frontpagediv = rkWebUtil.elemaker( "div", this.maindiv );
        let p = rkWebUtil.elemaker( "p", this.frontpagediv );
        let button = rkWebUtil.button( p, "Show Exposures", function() { self.show_exposures(); } );
        p.appendChild( document.createTextNode( " from " ) );
        this.startdatewid = rkWebUtil.elemaker( "input", p,
                                                { "attributes": { "type": "text",
                                                                  "size": 20 } } );
        this.startdatewid.addEventListener( "blur", function(e) {
            rkWebUtil.validateWidgetDate( self.startdatewid );
        } );
        p.appendChild( document.createTextNode( " to " ) );
        this.enddatewid = rkWebUtil.elemaker( "input", p,
                                              { "attributes": { "type": "text",
                                                                "size": 20 } } );
        this.enddatewid.addEventListener( "blur", function(e) {
            rkWebUtil.validateWidgetDate( self.enddatewid );
        } );
        p.appendChild( document.createTextNode( " (YYYY-MM-DD [HH:MM] — leave blank for no limit)" ) );
    }
    else {
        rkWebUtil.wipeDiv( this.maindiv );
        this.maindiv.appendchild( this.frontpagediv );
    }
}

seechange.Context.prototype.show_exposures = function()
{
    var self = this;
    var startdate, enddate;
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

    this.connector.sendHttpRequest( "exposures", { "startdate": startdate, "enddate": enddate },
                                    function( data ) { self.actually_show_exposures( data ); } );
}

seechange.Context.prototype.actually_show_exposures = function( data )
{
    if ( ! data.hasOwnProperty( "status" ) ) {
        console.log( "return has no status: " + data.toString() );
        window.alert( "Unexpected response from server when looking for exposures." );
        return
    }
    let exps = new seechange.ExposureList( this, this.maindiv, data["exposures"], data["startdate"], data["enddate"] );
    exps.render_page();
}

// **********************************************************************
// **********************************************************************
// **********************************************************************

seechange.ExposureList = function( context, parentdiv, exposures, fromtime, totime )
{
    this.context = context;
    this.parentdiv = parentdiv;
    this.exposures = exposures;
    this.fromtime = fromtime;
    this.totime = totime;
    this.div = null;
}

seechange.ExposureList.prototype.render_page = function()
{
    let self = this;

    rkWebUtil.wipeDiv( this.parentdiv );

    if ( this.div != null ) {
        this.parentdiv.appendChild( this.div );
        return
    }

    this.div = rkWebUtil.elemaker( "div", this.parentdiv );

    var table, th, tr, td;

    let h2 = rkWebUtil.elemaker( "h2", this.div, { "text": "Exposures" } );
    if ( ( this.fromtime == null ) && ( this.totime == null ) ) {
        h2.appendChild( document.createTextNode( " from all time" ) );
    } else if ( this.fromtime == null ) {
        h2.appendChild( document.createTextNode( " up to " + this.totime ) );
    } else if ( this.totime == null ) {
        h2.appendChild( document.createTextNode( " from " + this.fromtime + " on" ) );
    } else {
        h2.appendChild( document.createTextNode( " from " + this.fromtime + " to " + this.totime ) );
    }

    table = rkWebUtil.elemaker( "table", this.div, { "classes": [ "exposurelist" ] } );
    tr = rkWebUtil.elemaker( "tr", table );
    th = rkWebUtil.elemaker( "th", tr, { "text": "Exposure" } );
    th = rkWebUtil.elemaker( "th", tr, { "text": "MJD" } );
    th = rkWebUtil.elemaker( "th", tr, { "text": "target" } );
    th = rkWebUtil.elemaker( "th", tr, { "text": "filter" } );
    th = rkWebUtil.elemaker( "th", tr, { "text": "t_exp (s)" } );
    th = rkWebUtil.elemaker( "th", tr, { "text": "n_images" } );
    th = rkWebUtil.elemaker( "th", tr, { "text": "n_cutouts" } );
    th = rkWebUtil.elemaker( "th", tr, { "text": "n_sources" } );

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
                                                               exps["exp_time"][i] );
                                       }
                                     } );
        td = rkWebUtil.elemaker( "td", row, { "text": exps["mjd"][i].toFixed(2) } );
        td = rkWebUtil.elemaker( "td", row, { "text": exps["target"][i] } );
        td = rkWebUtil.elemaker( "td", row, { "text": exps["filter"][i] } );
        td = rkWebUtil.elemaker( "td", row, { "text": exps["exp_time"][i] } );
        td = rkWebUtil.elemaker( "td", row, { "text": exps["n_images"][i] } );
        td = rkWebUtil.elemaker( "td", row, { "text": exps["n_cutouts"][i] } );
        td = rkWebUtil.elemaker( "td", row, { "text": exps["n_sources"][i] } );
        countdown -= 1;
        if ( countdown == 0 ) {
            countdown = 3;
            fade = 1 - fade;
        }
    }
}

seechange.ExposureList.prototype.show_exposure = function( id, name, mjd, filter, target, exp_time )
{
    let self = this;
    this.context.connector.sendHttpRequest( "exposure_images/" + id, null,
                                            (data) => {
                                                self.actually_show_exposure( id, name, mjd, filter,
                                                                             target, exp_time, data );
                                            } );
}

seechange.ExposureList.prototype.actually_show_exposure = function( id, name, mjd, filter, target, exp_time, data )
{
    let exp = new seechange.Exposure( this.context, this.parentdiv, id, name, mjd, filter, target, exp_time, data );
    exp.render_page();
}


// **********************************************************************
// **********************************************************************
// **********************************************************************

seechange.Exposure = function( context, parentdiv, id, name, mjd, filter, target, exp_time, data )
{
    this.context = context;
    this.parentdiv = parentdiv;
    this.id = id;
    this.name = name;
    this.mjd = mjd;
    this.filter = filter;
    this.target = target;
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
}

seechange.Exposure.prototype.render_page = function()
{
    let self = this;

    rkWebUtil.wipeDiv( this.parentdiv );

    if ( this.div != null ) {
        this.parentdiv.appendChild( this.div );
        return;
    }

    this.div = rkWebUtil.elemaker( "div", this.parentdiv );

    var h2, h3, ul, li, table, tr, td, th, hbox, p;

    h2 = rkWebUtil.elemaker( "h2", this.div, { "text": "Exposure " + this.name } );
    ul = rkWebUtil.elemaker( "ul", this.div );
    li = rkWebUtil.elemaker( "li", ul );
    li.innerHTML = "<b>target:</b> " + this.target;
    li = rkWebUtil.elemaker( "li", ul );
    li.innerHTML = "<b>mjd:</b> " + this.mjd
    li = rkWebUtil.elemaker( "li", ul );
    li.innerHTML = "<b>filter:</b> " + this.filter;
    li = rkWebUtil.elemaker( "li", ul );
    li.innerHTML = "<b>t_exp (s):</b> " + this.exp_time;

    this.tabs = new rkWebUtil.Tabbed( this.parentdiv );


    this.imagesdiv = rkWebUtil.elemaker( "div", null );

    let totncutouts = 0;
    let totnsources = 0;
    for ( let i in this.data['id'] ) {
        totncutouts += this.data['numcutouts'][i];
        totnsources += this.data['nummeasurements'][i];
    }

    p = rkWebUtil.elemaker( "p", this.imagesdiv,
                            { "text": "Exposure has " + this.data.id.length + " completed subtractions." } )
    p = rkWebUtil.elemaker( "p", this.imagesdiv,
                            { "text": ( totnsources.toString() + " out of " +
                                        totncutouts.toString() + " sources pass preliminary cuts." ) } );

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
    rkWebUtil.elemaker( "label", p, { "text": "Show cutouts that failed the preliminary cuts",
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
    th = rkWebUtil.elemaker( "th", tr, { "text": "n_cutouts" } );
    th = rkWebUtil.elemaker( "th", tr, { "text": "n_sources" } );

    let fade = 1;
    let countdown = 3;
    for ( let i in this.data['id'] ) {
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
        td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data["lim_mag_estimate"][i], 1 ) } );
        td = rkWebUtil.elemaker( "td", tr, { "text": this.data["numcutouts"][i] } );
        td = rkWebUtil.elemaker( "td", tr, { "text": this.data["nummeasurements"][i] } );
    }


    this.cutoutsdiv = rkWebUtil.elemaker( "div", null );

    // TODO : buttons for next, prev, etc.

    this.tabs.addTab( "Images", "Images", this.imagesdiv, true );
    this.tabs.addTab( "Cutouts", "Sources", this.cutoutsdiv, false, ()=>{ self.update_cutouts() } );
}


seechange.Exposure.prototype.update_cutouts = function()
{
    var self = this;

    rkWebUtil.wipeDiv( this.cutoutsdiv );

    let withnomeas = this.cutoutssansmeasurements_checkbox.checked ? 1 : 0;

    if ( this.cutoutsallimages_checkbox.checked ) {
        rkWebUtil.elemaker( "p", this.cutoutsdiv,
                            { "text": "Sources for all succesfully completed chips" } );
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
                "png_cutouts_for_sub_image/" + this.id + "/0/" + withnomeas,
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

                let div = rkWebUtil.elemaker( "div", this.cutoutsdiv );
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
                        "png_cutouts_for_sub_image/" + this.data['subid'][i] + "/1/" + withnomeas,
                        {},
                        (data) => { self.show_cutouts_for_image( div, prop, data ); }
                    );
                }

                return;
            }
        }
    }
}


seechange.Exposure.prototype.show_cutouts_for_image = function( div, dex, indata )
{
    var table, tr, th, td, img;
    var oversample = 5;

    if ( ! this.cutouts_pngs.hasOwnProperty( dex ) )
        this.cutouts_pngs[dex] = indata;

    var data = this.cutouts_pngs[dex];

    rkWebUtil.wipeDiv( div );

    table = rkWebUtil.elemaker( "table", div );
    tr = rkWebUtil.elemaker( "tr", table );
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
        if ( data.cutouts['flux'][i] == null ) td.classList.add( 'bad' );
        else td.classList.add( 'good' );
        subdiv.innerHTML = ( "<b>chip:</b> " + data.cutouts.section_id[i] + "<br>" +
                             // "<b>cutout (α, δ):</b> (" + data.cutouts['ra'][i].toFixed(5) + " , "
                             // + data.cutouts['dec'][i].toFixed(5) + ")<br>" +
                             "<b>(α, δ):</b> (" + seechange.nullorfixed( data.cutouts['measra'][i], 5 ) + " , "
                             + seechange.nullorfixed( data.cutouts['measdec'][i],5 ) + ")<br>" +
                             "<b>(x, y):</b> (" + data.cutouts['x'][i].toFixed(2) + " , "
                             + data.cutouts['y'][i].toFixed(2) + ")<br>" +
                             "<b>Flux:</b> " + seechange.nullorfixed( data.cutouts['flux'][i], 0 )
                             + " ± " + seechange.nullorfixed( data.cutouts['dflux'][i], 0 )
                             + "  (aper r=" + seechange.nullorfixed( data.cutouts['aperrad'][i], 2) + " px)"
                             + "<br>" + "<b>Mag:</b> " + seechange.nullorfixed( data.cutouts['mag'][i], 2 )
                             + " ± " + seechange.nullorfixed( data.cutouts['dmag'][i], 2 )
                           );
    }
}

// **********************************************************************
// **********************************************************************
// **********************************************************************
// Make this into a module

export { seechange }

