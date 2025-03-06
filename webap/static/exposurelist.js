import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"

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
            h2.appendChild( document.createTextNode( " up to MJD " + this.totime.toFixed(2) ) );
        } else if ( this.totime == null ) {
            h2.appendChild( document.createTextNode( " from MJD " + this.fromtime.toFixed(2) + " and later" ) );
        } else {
            h2.appendChild( document.createTextNode( " from MJD " + this.fromtime.toFixed(2) + " to "
                                                     + this.totime.toFixed(2) ) );
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
                                                                   exps["airmass"][i],
                                                                   exps["filter"][i],
                                                                   exps["seeingavg"][i],
                                                                   exps["limmagavg"][i],
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


    show_exposure( id, name, mjd, airmass, filter, seeingavg, limmagavg, target, project, exp_time )
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
                                                        self.actually_show_exposure( id, name, mjd, airmass,
                                                                                     filter, seeingavg, limmagavg,
                                                                                     target, project,
                                                                                     exp_time, data );
                                                    } );
        }
    };


    actually_show_exposure( id, name, mjd, airmass, filter, seeingavg, limmagavg, target, project, exp_time, data )
    {
        let exp = new seechange.Exposure( this.context, this.exposurediv,
                                          id, name, mjd, airmass, filter, seeingavg, limmagavg,
                                          target, project, exp_time, data );
        this.exposure_displays[id] = exp;
        exp.render_page();
    };
}

// **********************************************************************
// Make this into a module

export { }
