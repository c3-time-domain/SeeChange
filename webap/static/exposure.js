import { rkWebUtil } from "./rkwebutil.js";
import { SVGPlot } from "./svgplot.js";
import { ImView } from "./imview.js";
import { seechange } from "./seechange_ns.js";
import "./image.js";

// **********************************************************************

seechange.Exposure = class
{
    // data is what is filled by the exposure_images/ api endpoint
    //   (ExposureImages in seechange_webap.py.)  It is a dictionary
    //   with contents:
    //      status: 'ok',
    //      exposure: dict with information about the exposure
    //      provenncetag: provenance tag (str)
    //      name: exposure name (str) -- this is the filepath in the database
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

    constructor( context, parenttabs, data, params )
    {
        // Things in params
        let default_params = { 'imagedetailsdiv': null,
                               'reportsdiv': null,
                               'detectionsdiv': null,
                               'fakesdiv': null
                             }
        Object.assign( this, default_params );
        Object.assign( this, params );

        this.context = context;
        this.parenttabs = parenttabs;
        this.id = data.exposure._id;

        this.data = data;
        let mat = this.data.exposure.filepath.match( /\/?([^\/]+)$/ );
        this.exposurename = (mat === null) ? "????" : mat[1];

        this.cutoutssansmeasurements_checkbox = null;
        this.cutoutsimage_dropdown = null;

        this.sectionfordetails_dropdown = null;
        this.sectionforfakes_dropdown = null;

        this.imgrows = {};
        this.fakeanalysis_data = {};
        this.image_objects = {};
        this.image_promises = {};
        this.image_view_div = null;
        this.cutoutsdiv = null;
        this.reports = null;
        this.reports_subdiv = null;

        this.div = rkWebUtil.elemaker( "div", null );
    };


    // Copy and adapt these next two from models/enums_and_bitflags.py
    static process_steps = {
        1: 'preprocessing',
        2: 'extraction',
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

        var div, h2, h3, ul, li, but, table, tr, td, th, hbox, p, span, tiptext, ttspan;

        h2 = rkWebUtil.elemaker( "h2", this.div, { "text": "Exposure " + this.data.exposure.filepath } );

        ul = rkWebUtil.elemaker( "ul", null );
        ul.style.display="none";
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>provenance tag:</b> " + this.data.provenancetag;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>filepath:</b> " + this.data.exposure.filepath;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>instrument:</b> " + this.data.exposure.instrument;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>project:</b> " + this.data.exposure.project;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>target:</b> " + this.data.exposure.target;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>(ra, dec):</b> " + ( parseFloat(this.data.exposure.ra).toFixed(4).toString() + ", " +
                                                parseFloat(this.data.exposure.dec).toFixed(4).toString() );
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>b:</b> " + this.data.exposure.gallat;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>mjd:</b> " + this.data.exposure.mjd
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>filter:</b> " + this.data.exposure.filter;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>filter_array:</b> " + this.data.exposure.filter_array;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>t_exp (s):</b> " + this.data.exposure.exp_time;
        li = rkWebUtil.elemaker( "li", ul );
        li.innerHTML = "<b>airmass:</b> " + this.data.exposure.airmass;
        but = rkWebUtil.hideOrShowButton( null, ul, "Exp Info", false );
        p = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox", "alignflexstart", "marginbotex" ] } );
        p.appendChild( but );
        p.appendChild( ul );

        let totncutouts = 0;
        let totngoodmeas = 0;
        let totnmeas = 0;
        let nsubs = 0;
        for ( let imgrow of this.data.images ) {
            this.imgrows[ imgrow.section_id ] = imgrow;
            totncutouts += imgrow['ncutout'];
            totngoodmeas += imgrow['ngoodmeas'];
            totnmeas += imgrow['nmeas'];
            if ( imgrow.has_sub ) nsubs += 1;
        }

        p = rkWebUtil.elemaker( "p", this.div,
                                { "text": ( "Exposure has " + this.data.images.length + " images and " +
                                            nsubs + " completed subtractions" ) } )
        p = rkWebUtil.elemaker( "p", this.div,
                                { "text": ( totnmeas.toString() + " (" + totngoodmeas.toString() + " good) out of " +
                                            totncutouts.toString() + " detections pass preliminary cuts " +
                                            "(i.e. are \"sources\")." ) } );

        table = rkWebUtil.elemaker( "table", this.div, { "classes": [ "exposurelist" ] } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr, { "text": "image name" } );
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
        for ( let dex in this.data.images ) {
            let imgrow = this.data.images[dex]
            countdown -= 1;
            if ( countdown <= 0 ) {
                countdown = 3;
                fade = 1 - fade;
            }
            tr = rkWebUtil.elemaker( "tr", table, { "classes": [ fade ? "bgfade" : "bgwhite" ] } );
            td = rkWebUtil.elemaker( "td", tr, { "text": imgrow.filename,
                                                 "classes": [ "link" ],
                                                 "click": () => { self.update_image_details(imgrow.section_id); }
                                                 }
                                               );
            td = rkWebUtil.elemaker( "td", tr, { "text": imgrow.section_id } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( imgrow.ra, 4 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( imgrow.dec, 4 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( imgrow.gallat, 1 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( imgrow.fwhm_estimate, 2 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( imgrow.zero_point_estimate, 2 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( imgrow.lim_mag_estimate, 1 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": imgrow.ncutout,
                                                 "classes": [ "link" ],
                                                 "click": () => { self.update_cutouts(imgrow.subid, true); }
                                               } );
            td = rkWebUtil.elemaker( "td", tr, { "text": imgrow.nmeas,
                                                 "classes": [ "link" ],
                                                 "click": () => { self.update_cutouts(imgrow.subid, false); }
                                               } );

            td = rkWebUtil.elemaker( "td", tr );
            tiptext = "";
            let laststep = "(none)";
            for ( let j of Object.keys( seechange.Exposure.process_steps ) ) {
                if ( imgrow.report.progress_steps_bitflag & ( 2**j ) ) {
                    tiptext += seechange.Exposure.process_steps[j] + " done<br>";
                    if ( seechange.Exposure.process_steps[j] != 'finalize' )
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
                if ( imgrow.report.products_exist_bitflag & ( 2**j ) )
                    tiptext += "<br>" + seechange.Exposure.pipeline_products[j];
            }
            span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ],
                                                     "text": "data products" } );
            ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
            ttspan.innerHTML = tiptext;

            // // Really I should be doing some HTML sanitization here on error message and, below, warnings....

            td = rkWebUtil.elemaker( "td", tr );
            if ( imgrow.report.error_step != null ) {
                span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ],
                                                         "text": "error" } );
                tiptext = ( imgrow.report.error_type + " error in step " +
                            seechange.Exposure.process_steps[imgrow.report.error_step] +
                            " (" + imgrow.report.error_message.replaceAll( "\n", "<br>") + ")" );
                ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
                ttspan.innerHTML = tiptext;
            }

            td = rkWebUtil.elemaker( "td", tr );
            if ( ( imgrow.report.warnings != null ) && ( imgrow.report.warnings > 0 ) ) {
                span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ],
                                                         "text": "warnings" } );
                ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
                ttspan.innerHTML = imgrow.report.warnings.replaceAll( "\n", "<br>" );
            }
        }

        // Selection dropdown in the image details div
        div = this.imagedetailsdiv;
        rkWebUtil.wipeDiv( div );
        rkWebUtil.elemaker( "h3", div, { "text": "Images for exposure " + this.exposurename } );
        hbox = rkWebUtil.elemaker( "div", div, { "classes": [ "hbox" ] } );
        rkWebUtil.elemaker( "text", hbox, "Show image for section: " );
        this.sectionfordetails_dropdown = rkWebUtil.elemaker( "select", hbox,
                                                              { "change": () => { self.select_image_details() } } );
        rkWebUtil.elemaker( "option", this.sectionfordetails_dropdown, { "text": "<Choose Section For Details>",
                                                                         "attributes": {
                                                                             "value": "_select_section",
                                                                             "selected": 1 } } );
        for ( let imgrow of this.data.images ) {
            rkWebUtil.elemaker( "option", this.sectionfordetails_dropdown, { "text": imgrow.section_id,
                                                                             "attributes": {
                                                                                 "value": imgrow.section_id
                                                                             } } );
        }
        this.image_view_div = rkWebUtil.elemaker( "div", div );


        // Selection dropdown in the fakes div
        div = this.fakesdiv;
        this.sectionforfakes_dropdown = rkWebUtil.elemaker( "select", div,
                                                            { "change": () => self.select_section_fakes() } );
        rkWebUtil.elemaker( "option", this.sectionforfakes_dropdown, { "text": "<Choose Section For Fakes>",
                                                                       "attributes": {
                                                                           "value": "_select_section",
                                                                           "selected": 1 } } );
        for ( let imgrow of this.data.images) {
            rkWebUtil.elemaker( "option", this.sectionforfakes_dropdown, { "text": imgrow.section_id,
                                                                           "attributes": {
                                                                               "value": imgrow.section_id
                                                                           } } );
        }
        this.fakesdiv = rkWebUtil.elemaker( "div", div );

        // Selection dropdowns in the detections div
        p = rkWebUtil.elemaker( "p", this.detectionsdiv );
        this.cutoutsimage_dropdown = rkWebUtil.elemaker( "select", p,
                                                         { "change": () => self.select_cutouts() } );
        rkWebUtil.elemaker( "option", this.cutoutsimage_dropdown, { "text": "<Choose Image For Cutouts>",
                                                                    "attributes": {
                                                                        "value": "_select_image",
                                                                        "selected": 1 } } );
        // Currently don't have code to support showing cutouts from all images at once...
        // rkWebUtil.elemaker( "option", this.cutoutsimage_dropdown, { "text": "All Successful Images",
        //                                                             "attributes": { "value": "_all_images" } } );
        for ( let imgrow of this.data.images ) {
            rkWebUtil.elemaker( "option", this.cutoutsimage_dropdown, { "text": imgrow.section_id,
                                                                        "attributes": {
                                                                            "value": imgrow.section_id } }  );
        }
        this.cutoutssansmeasurements_checkbox =
            rkWebUtil.elemaker( "input", p, { "change": () => { self.select_cutouts() },
                                              "attributes":
                                              { "type": "checkbox",
                                                "id": "cutouts_sans_measurements",
                                                "name": "cutouts_sans_measurements_checkbox" } } );
        rkWebUtil.elemaker( "label", p, { "text": ( "Show detections that failed the preliminary cuts " +
                                                    "(i.e. aren't sources)" ),
                                                    // + " (Ignored for \"All Successful Images\")" ),
                                          "attributes": { "for": "cutouts_sans_measurements_checkbox" } } );
        this.cutoutsdiv = rkWebUtil.elemaker( "div", this.detectionsdiv );

        // // TODO : buttons for next, prev, etc.

    };

    // ****************************************

    show_reports() {
        let self = this;
        let p, button;

        rkWebUtil.wipeDiv( this.reportsdiv );
        p = rkWebUtil.elemaker( "p", this.reportsdiv );
        rkWebUtil.button( p, "Refresh", () => { self.update_reports() } );

        if ( this.reports_subdiv != null ) {
            this.reportsdiv.appendChild( this.reports_subdiv );
        }
        else {
            this.reports_subdiv = rkWebUtil.elemaker( "div", this.reportsdiv );
            this.update_reports();
        }
    }

    // ****************************************

    update_reports() {
        let self = this;

        rkWebUtil.wipeDiv( this.reports_subdiv );
        rkWebUtil.elemaker( "p", this.reports_subdiv, { "text": "Loading reports...",
                                                        "classes": [ "bold", "italic", "warning" ] } );
        this.context.connector.sendHttpRequest( "exposure_reports/" + this.id + "/" + this.data.provenancetag,
                                                {}, (data) => { self.render_reports(data) } );
    }

    // ****************************************

    render_reports( data ) {
        let self = this;
        let h3, p, a, comma, text, table, tr, th, td, span, ttspan;

        this.reports = data.reports;

        rkWebUtil.wipeDiv( this.reports_subdiv );

        // Sort the sections
        let secs = Object.getOwnPropertyNames( this.reports );
        secs.sort()

        let innerhtml = "Jump to section: ";
        comma = false;
        for ( let secid of secs ) {
            if ( comma ) innerhtml += ", ";
            comma = true;
            innerhtml += '<a href="#exposure-report-section-' + secid + '">' + secid + '</a>';
        }
        p = rkWebUtil.elemaker( "p", this.reports_subdiv );
        p.innerHTML = innerhtml;

        let fields = { 'image_id': "Image ID",
                       'start_time': "Start Time",
                       'finish_time': "Finish Time",
                       'success': "Successful?",
                       'cluster_id': "Cluster ID",
                       'node_id': "Node ID",
                       'progress_steps_bitflag': "Steps Completed",
                       'products_exist_bitflag': "Existing Data Products",
                       'products_committed_bitflag': "Committed Data Products",
                       'process_provid': "Provenances",
                       'process_memory': "Memory Usage",
                       'process_runtime': "Runtimes",
                       'warnings': "Warnings",
                       'error': "Error",
                     }
        let steporder = { 'preprocessing': 0,
                          'extraction': 1,
                          'backgrounding': 2,
                          'astrocal': 3,
                          'photocal': 4,
                          'save_intermediate': 5,
                          'subtraction': 6,
                          'detection': 7,
                          'cutting': 8,
                          'measuring': 9,
                          'scoring': 10,
                          'save_final': 11,
                          'fakeanalysis': 12 };

        for ( let secid of secs ) {
            h3 = rkWebUtil.elemaker( "h3", this.reports_subdiv, { "text": "Section " + secid,
                                                                  "attributes": {
                                                                      "id": "exposure-report-section-" + secid,
                                                                      "name": "exposure-report-section-" + secid
                                                                  } } );
            table = rkWebUtil.elemaker( "table", this.reports_subdiv );

            for ( let field in fields ) {

                if ( ( field == "warnings" ) && ( ( this.reports[secid][field] == null ) ||
                                                  ( this.reports[secid][field].length == 0 ) ) )
                    continue;

                if ( ( field == "error" ) && ( this.reports[secid]['error_step'] == null  ) )
                    continue;

                tr = rkWebUtil.elemaker( "tr", table );
                th = rkWebUtil.elemaker( "th", tr, { "text": fields[field] } );
                if ( field == "error" ) th.classList.add( "bad" );
                if ( field == "warnings" ) th.classList.add( "warning" );

                if ( field == "progress_steps_bitflag" ) {
                    comma = false;
                    text = "";
                    for ( let i in seechange.Exposure.process_steps ) {
                        if ( this.reports[secid][field] & ( 2**i ) ) {
                            if ( comma ) text += ", ";
                            comma = true;
                            text += seechange.Exposure.process_steps[i]
                        }
                    }
                    td = rkWebUtil.elemaker( "td", tr, { "text": text } );
                }
                else if ( ( field == "products_exist_bitflag" ) || ( field == "products_committed_bitflag" ) ) {
                    comma = false;
                    text = "";
                    for ( let i in seechange.Exposure.pipeline_products ) {
                        if ( this.reports[secid][field] & ( 2**i ) ) {
                            if ( comma ) text += ", ";
                            comma = true;
                            text += seechange.Exposure.pipeline_products[i];
                        }
                    }
                    td = rkWebUtil.elemaker( "td", tr, { "text": text } );
                }
                else if ( field == "process_provid" ) {
                }
                else if ( ( field == "process_memory" ) || ( field == "process_runtime" ) ) {
                    // We want the processes to show up in a certain order,
                    //  so sort them using steporder (above) to define that order.
                    let procs = Object.getOwnPropertyNames( this.reports[secid][field] );
                    procs.sort( (a, b) => {
                        if ( steporder.hasOwnProperty(a) && steporder.hasOwnProperty(b) ) {
                            if ( steporder[a] < steporder[b] )
                                return -1
                            else if ( steporder[b] < steporder[a] )
                                return 1
                            else
                                return 0;
                        }
                        else if ( steporder.hasOwnProperty(a) ) {
                            return -1;
                        }
                        else if ( steporder.hasOwnProperty(b) ) {
                            return 1;
                        }
                        else {
                            return 0;
                        }
                    } );
                    td = rkWebUtil.elemaker( "td", tr );
                    let subtab = rkWebUtil.elemaker( "table", td, { "classes": [ "borderless" ] } );
                    for ( let proc of procs ) {
                        let subtr = rkWebUtil.elemaker( "tr", subtab );
                        rkWebUtil.elemaker( "th", subtr, { "text": proc,
                                                           "attributes": {
                                                               "style": "text-align: right; padding-right: 1em"
                                                           }
                                                         } );
                        if ( field == "process_memory" ) {
                            rkWebUtil.elemaker( "td", subtr, {
                                "text": seechange.nullorfixed( this.reports[secid][field][proc], 1 ) + " MiB"
                            } );
                        }
                        else {
                            rkWebUtil.elemaker( "td", subtr, {
                                "text": seechange.nullorfixed( this.reports[secid][field][proc], 2 ) + " s"
                            } );
                        }
                    }
                }
                else if ( field == "warnings" ) {
                    td = rkWebUtil.elemaker( "td", tr );
                    span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ],
                                                             "text": "(hover to see)" } );
                    ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
                    ttspan.innerHTML = this.reports[secid][field].replaceAll( "\n", "<br>" );
                }
                else if ( field == "error" ) {
                    td = rkWebUtil.elemaker( "td", tr );
                    span = rkWebUtil.elemaker( "span", td, { "classes": [ "tooltipsource" ],
                                                             "text": ( this.reports[secid]['error_type']
                                                                       + " in step "
                                                                       + this.reports[secid]['error_step'] )
                                                           } );
                    ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
                    ttspan.innerHTML = this.reports[secid]['error_message'].replaceAll( "\n", "<br>" );
                }
                else {
                    td = rkWebUtil.elemaker( "td", tr, { "text": this.reports[secid][field] } )
                }
            }
        }
    }


    // ****************************************

    update_image_details( secid=null, nosel=false ) {
        if ( secid != null ) {
            // I don't think all of this is necessary.  Javascript doesn't
            //    issue events when you change the value of a widget
            //    programmatically, only when it's done by the UI>
            //     let oldevent = this.sectionfordetails_dropdown.onchange;
            //     this.sectionfordetails_dropdown.onchange = null;
            //     this.sectionfordetails_dropdown.value = secid;
            //     this.sectionfordetails_dropdown.onchange = oldevent;
            // }
            this.sectionfordetails_dropdown.value = secid;
        }

        if ( ! nosel )
            this.select_image_details();
    }

    // ****************************************

    get_image_promise( sec )
    {
        if ( ! this.image_promises.hasOwnProperty( sec ) ) {
            let url = 'image_info/' + ( ( this.imgrows[sec].subid !== null ) ?
                                        this.imgrows[sec].subid : this.imgrows[sec]._id );
            this.image_promises[sec] = this.context.connector.httpPromise( url );
        }
        return this.image_promises[sec];
    }

    select_image_details()
    {
        let self = this;
        let p;

        rkWebUtil.wipeDiv( this.image_view_div );

        let sec = this.sectionfordetails_dropdown.value.toString();
        if ( sec == "_select_section" )
            return;

        if ( this.image_objects.hasOwnProperty( sec ) ) {
            this.image_objects[ sec ].show_imagediv();
            rkWebUtil.wipeDiv( this.cutoutsdiv );
            // TODO : sans/avec
            let div = this.image_objects[ sec ].show_detectionsdiv( true, this.cutoutsdiv );
            return;
        }

        this.parenttabs.selectTab( "imagedetail" );
        rkWebUtil.elemaker( "p", this.image_view_div, { "text": "Loading image info...",
                                                        "classes": [ "warning", "bold", "italic" ] } );
        let promise = this.get_image_promise( sec );
        // OK, I really don't understand javascript callbacks.  Could I
        //   just stick self.make_image_details here instead of the
        //   arrow function?  I don't think so, but I'm not sure why, I
        //   only know that when I contemplate it I start to feel
        //   anxiety.
        promise.then( (info) => { self.make_image_details(info);
                                  self.image_objects[sec].show_imagediv();
                                } );
    }

    make_image_details( info )
    {
        let self = this;
        let sec = info.section_id;

        if ( ! this.image_objects.hasOwnProperty( sec ) ) {
            this.image_objects[ sec ] = new seechange.Image( this.context, info,
                                                             { 'provtag': this.data.provenancetag,
                                                               'show_detections': true
                                                             } );
            this.image_objects[ sec ].listen( 'image',
                                              (signal, stuff) => {
                                                  self.parenttabs.selectTab( "imagedetail" );
                                                  rkWebUtil.wipeDiv( self.image_view_div );
                                                  self.image_view_div.appendChild( stuff.imagediv );
                                              },
                                              'exposure_image_listener' );
        }
    }


    // ****************************************

    update_cutouts( dex, sansmeasurements ) {
        if ( dex != null ) {
            // See comment in update_image_details
            // let oldevent = this.cutoutsimage_dropdown.onchange;
            // this.cutoutsimage_dropdown.onchange = null;
            // this.cutoutsimage_dropdown.value = dex;
            // this.cutoutsimage_dropdown.onchange = oldevent;
            this.cutoutsimage_dropdown.value = dex;
        }

        if ( sansmeasurements != null ) {
            // See comment in update_image_details
            // let oldevent = this.cutoutssansmeasurements_checkbox.onchange;
            // this.cutoutssansmeasurements_checkbox.onchange = null;
            // this.cutoutssansmeasurements_checkbox.checked = sansmeasurements
            // this.cutoutssansmeasurements_checkbox.onchange = oldevent;
            this.cutoutssansmeasurements_checkbox.checked = sansmeasurements
        }

        this.select_cutouts();
    }

    // ****************************************

    select_cutouts()
    {
        let self = this;
        let p;

        rkWebUtil.wipeDiv( this.cutoutsdiv );
        this.parenttabs.selectTab( "sources" );


        let sec = this.cutoutsimage_dropdown.value.toString();
        let sans = this.cutoutssansmeasurements_checkbox.checked;
        if ( sec == "_select_image" )
            return;

        rkWebUtil.elemaker( "p", this.cutoutsdiv, { "text": "Loading cutouts...",
                                                    "classes": [ "warning", "bold", "italic" ] });

        let detections_callback = (signal, stuff) => { self.parenttabs.selectTab( "sources" );
                                                       rkWebUtil.wipeDiv( self.cutoutsdiv );
                                                       self.cutoutsdiv.appendChild( stuff.detectionsdiv );
                                                     };

        if ( this.image_objects.hasOwnProperty( sec ) ) {
            this.image_objects[ sec ].listen( 'detections', detections_callback,
                                              'exposure_detections_listener', true );
            this.image_objects[ sec ].show_detectionsdiv( sans );
            return
        }

        let promise = this.get_image_promise(sec );
        promise.then( (info) => { self.make_image_detail(info);
                                  this.image_objects[sec].listen( 'detections', detections_callback,
                                                                  'exposure_detections_listener' );
                                  self.image_objects[sec].show_detectionsdiv( sans ); } );
    }

    // // ****************************************

    // clicked_on_image( imgrow, x, y ) {
    //     let self=this;
    //     this.load_cutouts( imgrow.subid, (which, data) => { self.process_clicked_on_image(x, y, data); } );
    // }

    // process_clicked_on_image( x, y, data ) {
    //     let table, tr, td, ul, li, span, ttspan;
    //     let oversample = 5;

    //     for ( let dex in data.cutouts.x ) {
    //         if ( ( x <= data.cutouts.x[dex] + this.box_width / 2. ) &&
    //              ( x >= data.cutouts.x[dex] - this.box_width / 2. ) &&
    //              ( y <= data.cutouts.y[dex] + this.box_width / 2. ) &&
    //              ( y >= data.cutouts.y[dex] - this.box_width / 2. ) )
    //         {
    //             rkWebUtil.wipeDiv( this.cutouts_div );

    //             let table = rkWebUtil.elemaker( "table", this.cutouts_div );
    //             let tr = rkWebUtil.elemaker( "tr", table );
    //             rkWebUtil.elemaker( "th", tr, { "text": "new" } );
    //             rkWebUtil.elemaker( "th", tr, { "text": "ref" } );
    //             rkWebUtil.elemaker( "th", tr, { "text": "sub" } );
    //             tr = rkWebUtil.elemaker( "tr", table );
    //             for ( let which of [ 'new', 'ref', 'sub' ] ) {
    //                 td = rkWebUtil.elemaker( "td", tr );
    //                 rkWebUtil.elemaker( "img", td,
    //                                     { "attributes": {
    //                                         "src": ( "data:image/png;base64," +
    //                                                  data.cutouts[which + '_png'][dex] ),
    //                                         "width": oversample * data.cutouts['w'][dex],
    //                                         "height": oversample * data.cutouts['h'][dex],
    //                                         "alt": which } } );
    //             }

    //             // A bunch of the rest of this code is very similar to show_cutouts_for_image;
    //             //   should efficientize

    //             ul = rkWebUtil.elemaker( "ul", this.cutouts_div );
    //             // row: source index, with good/bad
    //             li = rkWebUtil.elemaker( "li", ul );
    //             rkWebUtil.elemaker( "b", li, "source index:" );
    //             rkWebUtil.elemaker( "text", li, { "text": data.cutouts.source_index[dex] + "  "  } );
    //             span = rkWebUtil.elemaker( "b", li, { "text": ( data.cutouts.is_bad[dex] ?
    //                                                             "fails cuts" : "passes cuts" ) } );
    //             span.classList.add( "tooltipcolorlesssource" );
    //             ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
    //             ttspan.innerHTML = ( "<p>major_width: " + seechange.nullorfixed( data.cutouts.major_width[dex], 2 )
    //                                  + "<br>" +
    //                                  "minor_width: " + seechange.nullorfixed( data.cutouts.minor_width[dex], 2 )
    //                                  + "<br>" +
    //                                  "nbadpix: " + data.cutouts.nbadpix[dex]
    //                                  + "<br>" +
    //                                  "negfrac: " + data.cutouts.negfrac[dex]
    //                                  + "<br>" +
    //                                  "negfluxfrac: " + data.cutouts.negfluxfrac[dex]
    //                                  + "<br>" +
    //                                  "Gauss fit pos: " + seechange.nullorfixed( data.cutouts['gfit_x'][dex], 2 )
    //                                  + " , " + seechange.nullorfixed( data.cutouts['gfit_y'][dex], 2 ) +
    //                                  "</p>" )
    //             span.classList.add( data.cutouts.is_bad[dex] ? "bad" : "good" );
    //             // row: ra/dec
    //             li = rkWebUtil.elemaker( "li", ul );
    //             rkWebUtil.elemaker( "b", li, { "text": "(α, δ): " } );
    //             rkWebUtil.elemaker( "text", li, { "text": "(" +
    //                                               seechange.nullorfixed( data.cutouts['measra'][dex], 5 )
    //                                               + " , " +
    //                                               seechange.nullorfixed( data.cutouts['measdec'][dex], 5 )
    //                                               + ")" } );
    //             // row: x, y from cutouts; this is where it was originally detected.
    //             li = rkWebUtil.elemaker( "li", ul );
    //             rkWebUtil.elemaker( "b", li, { "text": "det. (x, y): " } );
    //             rkWebUtil.elemaker( "text", li, { "text": "(" +
    //                                               seechange.nullorfixed( data.cutouts['cutout_x'][dex], 2 )
    //                                               + " , " +
    //                                               seechange.nullorfixed( data.cutouts['cutout_y'][dex], 2 )
    //                                               + ")" } );
    //             // row: x, y from measurement table.
    //             li = rkWebUtil.elemaker( "li", ul );
    //             rkWebUtil.elemaker( "b", li, { "text": "meas. (x, y): " } );
    //             rkWebUtil.elemaker( "text", li, { "text": "(" +
    //                                               seechange.nullorfixed( data.cutouts['x'][dex], 2 )
    //                                               + " , " +
    //                                               seechange.nullorfixed( data.cutouts['y'][dex], 2 )
    //                                               + ")" } );
    //             // row: flux
    //             li = rkWebUtil.elemaker( "li", ul );
    //             rkWebUtil.elemaker( "b", li, { "text": "Flux: " } );
    //             rkWebUtil.elemaker( "text", li, { "text": seechange.nullorfixed( data.cutouts["flux"][dex], 0 )
    //                                               + " ± " +
    //                                               seechange.nullorfixed( data.cutouts["dflux"][dex], 0 )
    //                                               + "  " +
    //                                               + ( ( ( data.cutouts['aperrad'][dex] == null ) ||
    //                                                     ( data.cutouts['aperrad'][dex] <= 0 ) ) ?
    //                                                   '(psf)' :
    //                                                   ( "(aper r=" +
    //                                                     seechange.nullorfixed( data.cutouts['aperrad'][dex], 2 )
    //                                                     + " px)" ) ) } );
    //             // row: mag
    //             li = rkWebUtil.elemaker( "li", ul );
    //             rkWebUtil.elemaker( "b", li, { "text": "Mag: " } );
    //             rkWebUtil.elemaker( "text", li, { "text": seechange.nullorfixed( data.cutouts['mag'][dex], 2 )
    //                                               + " ± " +
    //                                               seechange.nullorfixed( data.cutouts['dmag'][dex], 2 ) } );
    //             // row; R/B
    //             li = rkWebUtil.elemaker( "li", ul );
    //             span = rkWebUtil.elemaker( "span", li );
    //             if ( ( data.cutouts['rb'][dex] == null ) || ( data.cutouts['rb'][dex] < data.cutouts['rbcut'][dex] ) )
    //                 span.classList.add( 'bad' );
    //             else
    //                 span.classList.add( 'good' );
    //             rkWebUtil.elemaker( "b", span, { "text": "R/B: " } );
    //             rkWebUtil.elemaker( "text", span, { "text": seechange.nullorfixed( data.cutouts['rb'][dex], 3 ) } );

    //             // Found one, don't bother looking at the rest
    //             break;
    //         }
    //     }
    // }

    // ****************************************

    update_image_fakes( secid ) {
        if ( secid != null ) {
            let oldevent = this.sectionforfakes_dropdown.onchange;
            this.sectionforfakes_dropdown.onchange = null;
            this.sectionforfakes_dropdown.value = secid;
            this.sectionforfakes_dropdown.onchange = oldevent;
        }

        this.select_image_fakes();
    }

    // ****************************************

    select_image_fakes()
    {
        let self = this;
        let p;

        rkWebUtil.wipeDiv( this.fakesdiv );

        p = rkWebUtil.elemaker( "p", this.fakesdiv, { "text": "Fakes for " } );
        p.appendChild( this.sectionforfakes_dropdown );

        this.fakes_content_div = rkWebUtil.elemaker( "div", this.fakesdiv );
        let sec = this.sectionforfakes_dropdown.value.toString();
        if ( sec == "_select_section" )
            return;

        if ( this.fakeanalysis_data.hasOwnProperty( sec ) ) {
            this.show_fakes_for_section( this.fakes_content_div, sec, this.fakeanalysis_data[sec] );
        }
        else {
            let url = "fakeanalysisdata/" + this.id + "/" + this.data.provenancetag + "/" + sec;
            this.context.connector.sendHttpRequest( url, {},
                                                    (data) => { self.show_fakes_for_section(
                                                        this.fakes_content_div,
                                                        sec, data ) } );
        }
    }

    // ****************************************

    show_fakes_for_section( div, sec, indata )
    {
        let p, table, tr, th, td;

        rkWebUtil.wipeDiv( div );

        if ( ! this.fakeanalysis_data.hasOwnProperty( sec ) ) {
            if ( indata.hasOwnProperty( "sections" ) ) {
                if ( indata.sections.hasOwnProperty( sec ) ) {
                    this.fakeanalysis_data[ sec ] = indata.sections[sec];
                }
            }
        }

        if ( ! this.fakeanalysis_data.hasOwnProperty( sec ) ) {
            rkWebUtil.elemaker( "p", div, { "text": "No fake analysis for current section id " + sec,
                                            "classes": [ "bad" ] } )
            return;
        }

        var data = this.fakeanalysis_data[ sec ];

        var dex = this.data.section_id.indexOf( sec );
        if ( dex < 0 ) {
            p = rkWebUtil.elemaker( "p", div, { "text": "Error: unknown section id " + sec,
                                                "classes": [ "bad" ] } );
            return;
        }

        table = rkWebUtil.elemaker( "table", div );

        tr = rkWebUtil.elemaker( "tr", table )
        rkWebUtil.elemaker( "th", tr, { "text": "Image ID" } );
        rkWebUtil.elemaker( "td", tr, { "text": this.data.id[dex] } );

        tr = rkWebUtil.elemaker( "tr", table );
        rkWebUtil.elemaker( "th", tr, { "text": "RA" } );
        rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data.ra[dex], 5 ) } );

        tr = rkWebUtil.elemaker( "tr", table );
        rkWebUtil.elemaker( "th", tr, { "text": "Dec" } );
        rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data.dec[dex], 5 ) } );

        tr = rkWebUtil.elemaker( "tr", table );
        rkWebUtil.elemaker( "th", tr, { "text": "ZP" } );
        rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data.zero_point_estimate[dex], 2 ) } );

        tr = rkWebUtil.elemaker( "tr", table );
        rkWebUtil.elemaker( "th", tr, { "text": "Lim. Mag." } );
        rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data.lim_mag_estimate[dex], 2 ) } );

        tr = rkWebUtil.elemaker( "tr", table );
        rkWebUtil.elemaker( "th", tr, { "text": "Detections" } );
        rkWebUtil.elemaker( "td", tr, { "text": this.data.numsources[dex] } );

        tr = rkWebUtil.elemaker( "tr", table );
        rkWebUtil.elemaker( "th", tr, { "text": "Sources" } );
        rkWebUtil.elemaker( "td", tr, { "text": this.data.nummeasurements[dex] } );

        // fakeanalysis_data is an array because it's possible that
        //   multiple fake analyses were run on the same subtraction
        //   (with all the same provenances, except for the provenance
        //   of the fakes, which is troublesome and perhaps we should
        //   think about that -- see Issue #444).  Combine them all
        //   together here.

        let fake_x = [];
        let fake_y = [];
        let fake_mag = [];
        let is_detected = [];
        let is_kept = [];
        let is_bad = [];
        let mag_psf = [];
        let mag_psf_err = [];
        let score = [];

        for ( let row of data ) {
            fake_x = fake_x.concat( row['fake_x'] );
            fake_y = fake_y.concat( row['fake_y'] );
            fake_mag = fake_mag.concat( row['fake_mag'] );
            is_detected = is_detected.concat( row['is_detected'] );
            is_kept = is_kept.concat( row['is_kept'] );
            is_bad = is_bad.concat( row['is_bad'] );
            mag_psf = mag_psf.concat( row['mag_psf'] );
            mag_psf_err = mag_psf_err.concat( row['mag_psf_err'] );
            score = score.concat( row['score'] )
        }

        // Histogram in steps of 0.25 mag
        let dmag = 0.25;
        let minmag = Math.floor( Math.min( ...fake_mag ) );
        let maxmag = Math.ceil( Math.max( ...fake_mag ) + dmag );
        let mags = [];
        let all_magplotval = [];
        let total = [];
        let detected = [];
        let kept = [];
        for ( let mag = minmag ; mag < maxmag ; mag += dmag ) {
            mags.push( mag );
            // Where we want the histogram point to actually show up.
            //  (Todo: implement step plots in svgplot.)
            all_magplotval.push( mag + dmag / 2. );
            total.push( 0 );
            detected.push( 0 );
            kept.push( 0 );
        }
        for ( let i in fake_mag ) {
            let histdex = Math.floor( ( fake_mag[i] - minmag ) / dmag );
            total[ histdex ] += 1;
            if ( is_detected[i] ) detected[ histdex ] += 1;
            if ( is_kept[i] ) kept[ histdex ] += 1;
        }
        let magplotval = [];
        let fracdetected = [];
        let dfracdetected = [];
        let frackept = [];
        let dfrackept = [];
        for ( let i in all_magplotval ) {
            if ( total[i] > 0 ) {
                magplotval.push( all_magplotval[i] );
                fracdetected.push( detected[i] / total[i] );
                dfracdetected.push( Math.sqrt(detected[i]) / total[i] );
                frackept.push( kept[i] / total[i] );
                dfrackept.push( Math.sqrt(kept[i]) / total[i] );
            }
        }

        // Also make det_ variables for the magnitudes of the
        //   detected fakes.  (TODO: different datasets,
        //   and thus colors, for kept vs. non-kept.  Also R/B?)
        let det_fakemag = [];
        let det_detmag = [];
        let det_detmagerr = [];
        let kept_fakemag = [];
        let kept_detmag = [];
        let kept_detmagerr = [];
        for ( let i in fake_mag ) {
            if ( is_detected[i] ) {
                if ( is_kept[i] ) {
                    kept_fakemag.push( fake_mag[i] );
                    kept_detmag.push( mag_psf[i] - fake_mag[i] );
                    kept_detmagerr.push( mag_psf_err[i] );
                }
                else {
                    det_fakemag.push( fake_mag[i] );
                    det_detmag.push( mag_psf[i] - fake_mag[i] );
                    det_detmagerr.push( mag_psf_err[i] );
                }
            }
        }

        let hbox = rkWebUtil.elemaker( "div", div, { "classes": [ "hbox" ] } )

        // Plot histogram of detected fakes
        let dethist = new SVGPlot.Plot( { 'divid': 'svgplot-fake-hist-div-' + sec,
                                          'svgid': 'svgplot-fake-hist-svg-' + sec,
                                          'title': 'Detection of ' + mags.length.toString() + ' fakes',
                                          'xtitle': 'Fake mag',
                                          'ytitle': 'Frac. detected (red), kept (green)',
                                        } );
        dethist.topdiv.classList.add( "xmargin", "padhalfex", "mostlyborder" );
        hbox.appendChild( dethist.topdiv );
        let detected_dataset = new SVGPlot.Dataset( { 'name': 'fake hist detected',
                                                      'x': magplotval,
                                                      'y': fracdetected,
                                                      'dy': dfracdetected,
                                                      'color': '#cc0000',
                                                      'linewid': 0 } );
        dethist.addDataset( detected_dataset );
        let kept_dataset = new SVGPlot.Dataset( { 'name': 'fake hist kept',
                                                  'x': magplotval,
                                                  'y': frackept,
                                                  'dy': dfrackept,
                                                  'color': '#008800',
                                                  'linewid': 0 } );
        dethist.addDataset( kept_dataset );
        dethist.redraw();
        let limmag_dataset = new SVGPlot.Dataset( { 'name': 'lim mag',
                                                    'x': [ this.data.lim_mag_estimate[dex],
                                                           this.data.lim_mag_estimate[dex] ],
                                                    'y': [ 0, dethist.ymax ],
                                                    'marker': null,
                                                    'linewid': 2,
                                                    'color': '#0000cc' } );
        dethist.addDataset( limmag_dataset );
        dethist.ymin = 0;

        // Plot detected magnitude vs. original magnitude
        let magvmag = new SVGPlot.Plot( { 'divid': 'svgplot-fake-magvmag-div-' + sec,
                                          'svgid': 'svgplot-fake-magvmag-svg-' + sec,
                                          'title': 'Detected mag. vs. fake mag.',
                                          'xtitle': 'Fake mag',
                                          'ytitle': 'Detected mag - Fake mag'
                                        } );
        magvmag.topdiv.classList.add( "xmargin", "padhalfex", "mostlyborder" );
        hbox.appendChild( magvmag.topdiv );
        let magvmag_det_dataset = new SVGPlot.Dataset( { 'name': 'magvmag detected',
                                                         'x': det_fakemag,
                                                         'y': det_detmag,
                                                         'dy': det_detmagerr,
                                                         'color': '#cc0000',
                                                         'linewid': 0 } );
        magvmag.addDataset( magvmag_det_dataset );
        let magvmag_kept_dataset = new SVGPlot.Dataset( { 'name': 'magvmag kept',
                                                          'x': kept_fakemag,
                                                          'y': kept_detmag,
                                                          'dy': kept_detmagerr,
                                                          'color': '#008800',
                                                          'marker': 'filledsquare',
                                                          'linewid': 0 } );
        magvmag.addDataset( magvmag_kept_dataset );
        magvmag.redraw();
        let hline = new SVGPlot.Dataset( { 'name': 'zero',
                                           'x': [ magvmag.xmin, magvmag.xmax ],
                                           'y': [ 0, 0 ],
                                           'color': '#0000cc',
                                           'marker': null,
                                           'linewid': 2 } )
        magvmag.addDataset( hline );
        let vline = new SVGPlot.Dataset( { 'name': 'lim mag',
                                           'x': [ this.data.lim_mag_estimate[dex],
                                                  this.data.lim_mag_estimate[dex] ],
                                           'y': [ magvmag.ymin, magvmag.ymax ],
                                           'color': '#0000cc',
                                           'marker': null,
                                           'linewid': 2 } );
        magvmag.addDataset( vline );
        magvmag.redraw();
    }


}


// **********************************************************************
// Make this into a module

export { }
