import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"

// **********************************************************************

seechange.ExposureList = class
{
    constructor( context, parent, parenttabs, parentdiv, exposures, fromtime, totime, provtag, projects )
    {
        this.context = context;
        this.parent = parent;
        this.parenttabs = parenttabs;
        this.parentdiv = parentdiv;
        this.exposures = exposures;
        this.masterdiv = null;
        this.fromtime = fromtime;
        this.totime = totime;
        this.provtag = provtag;
        this.projects = projects;
        this.exposure_displays = {};
        this.render_page();
    }

    render_page()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.parentdiv );
        let div = rkWebUtil.elemaker( "div", this.parentdiv, { 'id': 'epxosurelistdiv',
                                                               "classes": [ "scrolly" ] } );
        let h2 = rkWebUtil.elemaker( "h2", div, { "text": "Exposures" } );
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

        h2.appendChild( document.createTextNode( " with provenance tag " + this.provtag ) );

        rkWebUtil.elemaker( "p", div,
                            { "text": '"Detections" are everything found on subtratctions; ' +
                              '"Sources" are things that passed preliminary cuts.' } )

        let rowrenderer = (exps, fields, i) => {
            let row = rkWebUtil.elemaker( "tr", null );
            let td = rkWebUtil.elemaker( "td", row );
            rkWebUtil.elemaker( "a", td, { "text": exps["name"][i],
                                           "classes": [ "link" ],
                                           "click": function() { self.show_exposure( exps["id"][i] ) }
                                         }
                              );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["project"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["mjd"][i].toFixed(2) } );
            td = rkWebUtil.elemaker( "td", row, { "text": exps["imgtype"][i] } );
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

        let fields = [ "name", "project", "mjd", "imgtype", "target", "filter", "exp_time",
                       "n_subs", "n_sources", "n_measurements", "n_successim", "n_errors" ];
        let fieldmap = { "name": 'Exposure',
                         "project": 'project',
                         "mjd": 'MJD',
                         "imgtype": 'type',
                         "target": 'target',
                         "filter": 'filter',
                         "exp_time": 't_exp (s)',
                         "n_subs": 'subs',
                         "n_sources": 'detections',
                         "n_measurements": 'sources',
                         "n_successim": 'n_successim',
                         "n_errors": 'n_errors',
                       };
        let tab = new rkWebUtil.SortableTable( this.exposures, fields, rowrenderer,
                                               { 'hdrs': fieldmap,
                                                 'dictoflists': true,
                                                 'initsort': [ '+MJD' ],
                                                 'tableclasses': [ 'exposure_list_table' ],
                                                 'colorclasses': [ 'bgfade', 'bgwhite' ],
                                                 'colorlength': 3 } );
        tab.table.id = 'exposure_list_table';
        div.appendChild( tab.table );
    };


    show_exposure( id, provtag=null )
    {
        let self = this;
        if ( provtag == null ) provtag = this.provtag;

        this.parenttabs.selectTab( "exposuredetail" );
        let target = this.parent.emptiabledivs["exposuredetail"];

        if ( this.exposure_displays.hasOwnProperty( id ) ) {
            taget.reset( { 'body': this.exposure_displays[id].div } );
        }
        else {
            target.reset( { "text": "Loading...", "classes": [ "warning", "bold", "italic" ] } );
            this.context.connector.sendHttpRequest( "exposure_images/" + id + "/" + provtag,
                                                    null,
                                                    (data) => { self.actually_show_exposure( data ); }
                                                  );
        }
    };


    actually_show_exposure( data )
    {
        let exp = new seechange.Exposure( this.context, this.parenttabs, data,
                                          { 'imagedetailsdiv': this.parent.emptiabledivs["imagedetail"].div,
                                            'reportsdiv': this.parent.emptiabledivs["reports"].div,
                                            'detectionsdiv': this.parent.emptiabledivs["sources"].div,
                                            'fakesdiv': this.parent.emptiabledivs["fakes"].div
                                          } );
        this.exposure_displays[data.exposure._id] = exp;
        this.parent.emptiabledivs["exposuredetail"].reset( { "body": exp.div } );
        exp.render_page();
    };
}

// **********************************************************************
// Make this into a module

export { }
