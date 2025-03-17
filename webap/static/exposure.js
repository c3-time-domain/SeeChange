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

        this.cutoutssansmeasurements_checkbox = null;
        this.cutoutssansmeasurements_label = null;
        this.cutoutsimage_dropdown = null;

        this.cutouts = {};
        this.cutouts_pngs = {};
        this.reports = null;
        this.reports_subdiv = null;
    };


    // Copy and adapt these next two from models/enums_and_bitflags.py
    static process_steps = {
        1: 'preprocessing',
        2: 'extraction',
        4: 'wcs',
        5: 'zp',
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
            td = rkWebUtil.elemaker( "td", tr, { "text": this.data['name'][i],
                                                 "classes": [ "link" ],
                                                 "click": function() { self.show_image_details( this.id[i] ) }
                                               } );
            td = rkWebUtil.elemaker( "td", tr, { "text": this.data['section_id'][i] } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data["ra"][i], 4 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data["dec"][i], 4 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data["gallat"][i], 1 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": seechange.nullorfixed( this.data["fwhm_estimate"][i], 2 ) } );
            td = rkWebUtil.elemaker( "td", tr,
                                     { "text": seechange.nullorfixed( this.data["zero_point_estimate"][i], 2 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text":
                                                 seechange.nullorfixed( this.data["lim_mag_estimate"][i], 1 ) } );
            td = rkWebUtil.elemaker( "td", tr, { "text": this.data["numsources"][i],
                                                 "classes": [ "link" ],
                                                 "click": function() { self.update_cutouts( i, true );
                                                                       self.tabs.selectTab( "Cutouts" ); }
                                               } );
            td = rkWebUtil.elemaker( "td", tr, { "text": this.data["nummeasurements"][i],
                                                 "classes": [ "link" ],
                                                 "click": function() { self.update_cutouts( i, false );
                                                                       self.tabs.selectTab( "Cutouts" ); }
                                               } );

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

        let temp_remove_this = rkWebUtil.elemaker( "div", null, { "text": "TODO" } );
        this.tabs.addTab( "Image Details", "Image Details", temp_remove_this, false );

        this.reports_div = rkWebUtil.elemaker( "div", null, { 'id': 'exposurereportsdiv' } );
        this.tabs.addTab( "Reports", "Reports", this.reports_div, false, ()=>{ self.show_reports() } );

        this.tabs.addTab( "Cutouts", "Sources", this.cutoutsdiv, false, ()=>{ self.select_cutouts() } );
        this.create_cutouts_widgets();
    };

    // ****************************************

    show_image_details( imageid ) {
        window.alert( "show image details not impmlemented yet" );
    };

    // ****************************************

    show_reports() {
        let self = this;
        let p, button;

        rkWebUtil.wipeDiv( this.reports_div );
        p = rkWebUtil.elemaker( "p", this.reports_div );
        rkWebUtil.button( p, "Refresh", () => { self.update_reports() } );

        if ( this.reports_subdiv != null ) {
            this.reports_div.appendChild( this.reports_subdiv );
        }
        else {
            this.reports_subdiv = rkWebUtil.elemaker( "div", this.reports_div );
            this.update_reports();
        }
    }

    // ****************************************

    update_reports() {
        let self = this;

        rkWebUtil.wipeDiv( this.resports_subdiv );
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

    create_cutouts_widgets() {
        let self = this;

        if ( this.cutoutsimage_dropdown == null ) {
            this.cutoutsimage_dropdown = rkWebUtil.elemaker( "select", null,
                                                             { "change": () => self.select_cutouts() } );
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
        }
        if ( this.cutoutssansmeasurements_checkbox == null ) {
            this.cutoutssansmeasurements_checkbox =
                rkWebUtil.elemaker( "input", null, { "change": () => { self.select_cutouts() },
                                                     "attributes":
                                                     { "type": "checkbox",
                                                       "id": "cutouts_sans_measurements",
                                                       "name": "cutouts_sans_measurements_checkbox" } } );
            this.cutoutssansmeasurements_label =
                rkWebUtil.elemaker( "label", null, { "text": ( "Show detections that failed the preliminary cuts " +
                                                               "(i.e. aren't sources) " +
                                                               "(Ignored for \"All Successful Images\")" ),
                                                     "attributes": { "for": "cutouts_sans_measurements_checkbox" } } );
        }
    };

    // ****************************************

    update_cutouts( dex, sansmeasurements ) {
        if ( dex != null ) {
            let oldevent = this.cutoutsimage_dropdown.onchange;
            this.cutoutsimage_dropdown.onchange = null;
            this.cutoutsimage_dropdown.value = this.data["subid"][dex];
            this.cutoutsimage_dropdown.onchange = oldevent;
        }

        if ( sansmeasurements != null ) {
            let oldevent = this.cutoutssansmeasurements_checkbox.onchange;
            this.cutoutssansmeasurements_checkbox.onchange = null;
            this.cutoutssansmeasurements_checkbox.checked = sansmeasurements
            this.cutoutssansmeasurements_checkbox.onchange = oldevent;
        }

        this.select_cutouts;
    }

    // ****************************************

    select_cutouts()
    {
        let self = this;
        let p;

        rkWebUtil.wipeDiv( this.cutoutsdiv );

        p = rkWebUtil.elemaker( "p", this.cutoutsdiv, { "text": "Sources for " } )
        p.appendChild( this.cutoutsimage_dropdown );
        p.appendChild( document.createTextNode( "    " ) );

        p.appendChild( this.cutoutssansmeasurements_checkbox );
        p.appendChild( this.cutoutssansmeasurements_label );

        this.cutouts_content_div = rkWebUtil.elemaker( "div", this.cutoutsdiv );

        let dex = this.cutoutsimage_dropdown.value.toString();
        if ( dex == "_select_image" )
            return;

        rkWebUtil.elemaker( "p", this.cutouts_content_div, { "text": "Loading cutouts...",
                                                             "classes": [ "bold", "italic", "warning" ] } );

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
    // TODO : implement limit and offset
    //   (will require modifing select_cutouts too)

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
