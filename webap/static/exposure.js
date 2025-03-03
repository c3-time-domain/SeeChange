import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"

// **********************************************************************

seechange.Exposure = class
{
    // data is what is filled by the exposure_images/ api endpoint
    //   (ExpousreImages in seechange_webap.py.)  It is a dictionary
    //   with contents:
    //      status: 'ok',
    //      provenncetag: provenance tag (str)
    //      name: exposure name (str)
    //      id: array of image uuids
    //      ra: array of image ras (array of float)
    //      dec: array of image decs (array of floats)
    //      gallat: array of galactic latitudes (array of floats)
    //      section_id: array of image section ids (array of str)
    //      fwhm_estimate: array of image fwhm estimates (array of float)
    //      zero_point_estimate: array of image zeropoints (array of float)
    //      lim_mag_estimate: array of image limiting magnitudes (array of float)
    //      bkg_mean_estimate: array of image sky levels (array of float)
    //      bkt_rms_estimate: array of image 1σ sky noise levels (array of float)
    //      numsources: array of number of sources on each difference image (array of int)
    //      nummeasurements: array of number of sources that passed initial cuts on each diff im (array of int)
    //      subid: uuid of the difference image
    //      error_step: step where the pipeline errored out (str or null)
    //      error_type: class of python exception raised where hte pipeline errored out (str or null)
    //      error_message: error message(s) given with exception(s) (str or null)
    //      warnings: warnings issued during pipeline (str or null)
    //      start_time: when pipeline on this image begam
    //      end_time: when pipeline on this image finished
    //      process_memory: empty dictionary, or dictionary of process: MB of peak memory usage
    //                      (only filled if SEECHANGE_TRACEMALLOC env var was set when pipeline was run)
    //      process_runtime: dictionary of process: sections runtime for pipeline segments
    //      process_setps_bitflag: bitflag of which pipeline steps completed
    //      products_exist_bitflag: bitflag of which data products were saved to database/archive

    constructor( context, parentdiv, id, name, mjd, airmass, filter, seeingavg, limmagavg,
                 target, project, exp_time, data )
    {
        this.context = context;
        this.parentdiv = parentdiv;
        this.id = id;
        this.name = name;
        this.mjd = mjd;
        this.airmass = airmass;
        this.filter = filter;
        this.seeingavg = seeingavg;
        this.limmagavg = limmagavg;
        this.target = target;
        this.project = project;
        this.exp_time = exp_time;
        this.data = data;
        this.div = null;
        this.tabs = null;
        this.imagesdiv = null;
        this.cutoutsdiv = null;

        this.cutoutsimage_checkboxes = {};
        this.cutoutssansmeasurements_checkbox = null;
        this.cutoutssansmeasurements_label = null;
        this.cutoutsimage_dropdown = null;

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
        11: 'fakeanalysis',
        12: 'alerting',
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
        25: 'fakes',
        26: 'fakeanalysis'
    };


    // ****************************************

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
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>airmass:</b> " + this.airmass;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>avg. seeing (¨):</b> " + seechange.nullorfixed( this.seeingavg, 2 );
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>avg. 5σ lim mag:</b> " + seechange.nullorfixed( this.limmagavg, 2 );

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

    // ****************************************

    update_cutouts()
    {
        var self = this;
        let p;

        rkWebUtil.wipeDiv( this.cutoutsdiv );

        p = rkWebUtil.elemaker( "p", this.cutoutsdiv, { "text": "Sources for " } )
        if ( this.cutoutsimage_dropdown == null ) {
            this.cutoutsimage_dropdown = rkWebUtil.elemaker( "select", p, { "change": () => self.select_cutouts() } );
            rkWebUtil.elemaker( "option", this.cutoutsimage_dropdown, { "text": "<Choose Image For Cutouts>",
                                                                        "attributes": {
                                                                            "value": "_select_image",
                                                                            "selected": 1 } } );
            rkWebUtil.elemaker( "option", this.cutoutsimage_dropdown, { "text": "All Successful Images",
                                                                        "attributes": { "value": "_all_images" } } );
            for ( let i in this.data['id'] ) {
                rkWebUtil.elemaker( "option", this.cutoutsimage_dropdown, { "text": this.data["section_id"][i],
                                                                            "attributes": {
                                                                                "value": this.data["subid"][i] } }  );
            }
        } else {
            p.appendChild( this.cutoutsimage_dropdown );
        }
        p.appendChild( document.createTextNode( "    " ) );

        let withnomeas = 0;
        if ( this.cutoutssansmeasurements_checkbox == null ) {
            this.cutoutssansmeasurements_checkbox =
                rkWebUtil.elemaker( "input", p, { "change": () => self.select_cutouts(),
                                                  "attributes":
                                                  { "type": "checkbox",
                                                    "id": "cutouts_sans_measurements",
                                                    "name": "cutouts_sans_measurements_checkbox" } } );
            this.cutoutssansmeasurements_label =
                rkWebUtil.elemaker( "label", p, { "text": ( "Show detections that failed the preliminary cuts " +
                                                            "(i.e. aren't sources) " +
                                                            "(Ignored for \"All Successful Images\")" ),
                                                  "attributes": { "for": "cutouts_sans_measurements_checkbox" } } );
        } else {
            p.appendChild( this.cutoutssansmeasurements_checkbox );
            p.appendChild( this.cutoutssansmeasurements_label );
            withnomeas = this.cutoutssansmeasurements_checkbox.checked ? 1 : 0;
        }

        this.cutouts_content_div = rkWebUtil.elemaker( "div", this.cutoutsdiv );

        // Issue a change event on the chip dropdown to make sure
        //   the images are rendered if necessary.
        this.cutoutsimage_dropdown.dispatchEvent( new Event('change') );
    }

    // ****************************************
    // TODO : implement limit and offset in this and the next method

    select_cutouts()
    {
        let self = this;

        rkWebUtil.wipeDiv( this.cutouts_content_div );

        let dex = this.cutoutsimage_dropdown.value.toString();
        if ( dex == "_select_image" )
            return;

        let url = "png_cutouts_for_sub_image/";
        if ( dex == "_all_images" ) {
            url += this.id + "/" + this.data.provenancetag + "/0/0";
            dex += "/0/0";
        } else {
            let sansmeas = ( this.cutoutssansmeasurements_checkbox.checked  ? 1 : 0 ).toString();
            url += dex + "/" + this.data.provenancetag + "/1/" + sansmeas;
            dex += "/1/" + sansmeas;
        }

        if ( this.cutouts_pngs.hasOwnProperty( dex ) ) {
            this.show_cutouts_for_image( this.cutouts_content_div, dex, this.cutouts_pngs[dex] );
        } else {
            this.context.connector.sendHttpRequest( url, {},
                                                    (data) => { self.show_cutouts_for_image( this.cutouts_content_div,
                                                                                             dex, data ) } );
        }
    };

    // ****************************************

    show_cutouts_for_image( div, dex, indata )
    {
        var table, tr, th, td, img, span, ttspan;
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

            // *** The info td, which is long
            td = rkWebUtil.elemaker( "td", tr );
            // row; chip
            rkWebUtil.elemaker( "b", td, { "text": "chip: " } );
            td.appendChild( document.createTextNode( data.cutouts.section_id[i] ) );
            rkWebUtil.elemaker( "br", td );
            // row: source index, with good/bad
            rkWebUtil.elemaker( "b", td, { "text": "source index: " } )
            td.appendChild( document.createTextNode( data.cutouts.source_index[i] + "  " ) );
            span = rkWebUtil.elemaker( "b", td, { "text": ( data.cutouts.is_bad[i] ?
                                                            "fails cuts" : "passes cuts" ) } );
            span.classList.add( "tooltipcolorlesssource" );
            ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
            ttspan.innerHTML = ( "<p>major_width: " + seechange.nullorfixed( data.cutouts.major_width[i], 2 ) + "<br>" +
                                 "minor_width: " + seechange.nullorfixed( data.cutouts.minor_width[i], 2 ) + "<br>" +
                                 "nbadpix: " + data.cutouts.nbadpix[i] + "<br>" +
                                 "negfrac: " + data.cutouts.negfrac[i] + "<br>" +
                                 "negfluxfrac: " + data.cutouts.negfluxfrac[i] + "<br>" +
                                 "Gauss fit pos: " + seechange.nullorfixed( data.cutouts['gfit_x'][i], 2 )
                                 + " , " + seechange.nullorfixed( data.cutouts['gfit_y'][i], 2 ) +
                                 "</p>" )
            span.classList.add( data.cutouts.is_bad[i] ? "bad" : "good" );
            rkWebUtil.elemaker( "br", td );
            // row: ra/dec
            rkWebUtil.elemaker( "b", td, { "text": "(α, δ): " } );
            td.appendChild( document.createTextNode( "(" +
                                                     seechange.nullorfixed( data.cutouts['measra'][i], 5 )
                                                     + " , " +
                                                     seechange.nullorfixed( data.cutouts['measdec'][i], 5 )
                                                     + ")" ) );
            rkWebUtil.elemaker( "br", td );
            // row: x, y from cutouts; this is where it was originally detected.
            rkWebUtil.elemaker( "b", td, { "text": "det. (x, y): " } );
            td.appendChild( document.createTextNode( "(" +
                                                     seechange.nullorfixed( data.cutouts['cutout_x'][i], 2 )
                                                     + " , " +
                                                     seechange.nullorfixed( data.cutouts['cutout_y'][i], 2 )
                                                     + ")" ) );
            rkWebUtil.elemaker( "br", td );
            // row: x, y from measurement table.
            rkWebUtil.elemaker( "b", td, { "text": "meas. (x, y): " } );
            td.appendChild( document.createTextNode( "(" +
                                                     seechange.nullorfixed( data.cutouts['x'][i], 2 )
                                                     + " , " +
                                                     seechange.nullorfixed( data.cutouts['y'][i], 2 )
                                                     + ")" ) );
            rkWebUtil.elemaker( "br", td );
            // row: flux
            rkWebUtil.elemaker( "b", td, { "text": "Flux: " } );
            td.appendChild( document.createTextNode( seechange.nullorfixed( data.cutouts["flux"][i], 0 )
                                                     + " ± " +
                                                     seechange.nullorfixed( data.cutouts["dflux"][i], 0 )
                                                     + "  " +
                                                     + ( ( ( data.cutouts['aperrad'][i] == null ) ||
                                                           ( data.cutouts['aperrad'][i] <= 0 ) ) ?
                                                         '(psf)' :
                                                         ( "(aper r=" +
                                                           seechange.nullorfixed( data.cutouts['aperrad'][i], 2 )
                                                           + " px)" ) ) ) );
            rkWebUtil.elemaker( "br", td );
            // row: mag
            rkWebUtil.elemaker( "b", td, { "text": "Mag: " } );
            td.appendChild( document.createTextNode( seechange.nullorfixed( data.cutouts['mag'][i], 2 )
                                                     + " ± " +
                                                     seechange.nullorfixed( data.cutouts['dmag'][i], 2 ) ) );
            rkWebUtil.elemaker( "br", td );
            // row; R/B
            span = rkWebUtil.elemaker( "span", td );
            if ( ( data.cutouts['rb'][i] == null ) || ( data.cutouts['rb'][i] < data.cutouts['rbcut'][i] ) )
                span.classList.add( 'bad' );
            else
                span.classList.add( 'good' );
            rkWebUtil.elemaker( "b", span, { "text": "R/B: " } );
            span.appendChild( document.createTextNode( seechange.nullorfixed( data.cutouts['rb'][i], 3 ) ) );

        }
    };
}


// **********************************************************************
// Make this into a module

export { }
