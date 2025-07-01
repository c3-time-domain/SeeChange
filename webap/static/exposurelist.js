import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"

// **********************************************************************

seechange.ExposureList = class
{
    constructor( context, parent, parentdiv )
    {
        this.context = context;
        this.parent = parent;
        this.parentdiv = parentdiv;
        this.masterdiv = null;
        this.listdiv = null;
        this.exposures = {};
        this.exposurediv = null;
        this.exposure_displays = {};
    };


    // **********************************************************************

    new_exposure_list( exposures, fromtime, totime, projects ) {
        this.exposures = exposures;
        this.fromtime = fromtime;
        this.totime = totime;
        this.projects = projects;
        this.render_page();
    }


    // **********************************************************************

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

        h2.appendChild( document.createTextNode( " with provenance tag " + this.cointext.provtag_wid.value ) );

        rkWebUtil.elemaker( "p", this.listdiv,
                            { "text": '"Detections" are everything found on subtratcions; ' +
                              '"Sources" are things that passed preliminary cuts.' } )

        let rowrenderer = (exps, i) => {
            let row = rkWebUtil.elemaker( "tr", null );
            let td = rkWebUtil.elemaker( "td", row );
            rkWebUtil.elemaker( "a", td, { "text": exps["name"][i],
                                           "classes": [ "link" ],
                                           "click": function() { self.show_exposure( exps["id"][i] ) }
                                         }
                              );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["project"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["mjd"][i].toFixed(2) } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["target"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["filter"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["exp_time"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["n_subs"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["n_sources"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["n_measurements"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["n_successim"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["n_errors"][i] } );
            return row
        };

        let fields = [ "Exposure", "project", "MJD", "target", "filter", "t_exp (s)",
                       "subs", "detections", "sources", "n_successim", "n_errors" ];
        let fieldmap = { 'Exposure': "name",
                         'project': "project",
                         'MJD': "mjd",
                         'target': "target",
                         'filter': "filter",
                         't_exp (s)': "exp_time",
                         'subs': "n_subs",
                         'detections': "n_sources",
                         'sources': "n_measurements",
                         'n_successim': "n_successim",
                         'n_errors': "n_errors",
                       };
        let tab = new rkWebUtil.SortableTable( this.exposures, rowrenderer, fields,
                                               { 'fieldmap': fieldmap,
                                                 'dictoflists': true,
                                                 'initsort': [ '+MJD' ],
                                                 'tableclasses': [ 'exposure_list_table' ],
                                                 'colorclasses': [ 'bgfade', 'bgwhite' ],
                                                 'colorlength': 3 } );
        tab.table.id = 'exposure_list_table';
        this.listdiv.appendChild( tab.table );

    };


    show_exposure( id, provtag=null )
    {
        let self = this;
        if ( provtag == null ) provtag = this.context.provtag_wid.value;

        this.tabbed.selectTab( "exposuredetail" );

        if ( this.exposure_displays.hasOwnProperty( id ) ) {
            this.exposure_displays[id].render_page();
        }
        else {
            rkWebUtil.wipeDiv( this.exposurediv );
            rkWebUtil.elemaker( "p", this.exposurediv, { "text": "Loading...",
                                                         "classes": [ "warning", "bold", "italic" ] } );
            this.context.connector.sendHttpRequest( "exposure_images/" + id + "/" + provtag,
                                                    null,
                                                    (data) => { self.actually_show_exposure( data ); }
                                                  );
        }
    };


    actually_show_exposure( data )
    {
        let exp = new seechange.Exposure( this.context, this.exposurediv, data );
        this.exposure_displays[data.exposure._id] = exp;
        exp.render_page();
    };
}

// **********************************************************************
// Make this into a module

export { }
