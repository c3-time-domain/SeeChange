import { rkWebUtil } from "./rkwebutil.js";
import { SVGPlot } from "./svgplot.js";
import { seechange } from "./seechange_ns.js";

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

        this.sectionfordetails_dropdown = null;

        this.cutouts = {};
        this.cutouts_pngs = {};
        this.fakeanalysis_data = {};
        this.reports = null;
        this.reports_subdiv = null;
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
                                                 "click": function() {
                                                     self.update_image_details( self.data.section_id[i] );
                                                     self.tabs.selectTab( "Image Details" );
                                                 }
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


        // TODO : buttons for next, prev, etc.

        this.tabs.addTab( "Images", "Images", this.imagesdiv, true );

        this.image_details_div = rkWebUtil.elemaker( "div", null, { 'id': 'exposureimagedetaildiv' } );
        this.tabs.addTab( "Image Details", "Image Details", this.image_details_div, false,
                          () => { self.select_image_details() } );
        this.create_image_details_widgets();

        this.reports_div = rkWebUtil.elemaker( "div", null, { 'id': 'exposurereportsdiv' } );
        this.tabs.addTab( "Reports", "Reports", this.reports_div, false, ()=>{ self.show_reports() } );

        this.cutoutsdiv = rkWebUtil.elemaker( "div", null, { 'id': 'exposurecutoutsdiv' } );
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

    create_image_details_widgets() {
        let self = this;

        if ( this.sectionfordetails_dropdown == null ) {
            this.sectionfordetails_dropdown = rkWebUtil.elemaker( "select", null,
                                                                  { "change": () => self.select_section_details() } );
            rkWebUtil.elemaker( "option", this.sectionfordetails_dropdown, { "text": "<Choose Section For Details>",
                                                                             "attributes": {
                                                                                 "value": "_select_section",
                                                                                 "selected": 1 } } );
            for ( let i in this.data['id'] ) {
                rkWebUtil.elemaker( "option", this.sectionfordetails_dropdown, { "text": this.data["section_id"][i],
                                                                                 "attributes": {
                                                                                     "value": this.data["section_id"][i]
                                                                                 } } );
            }
        }
    }


    // ****************************************

    update_image_details( secid ) {
        if ( secid != null ) {
            let oldevent = this.sectionfordetails_dropdown.onchange;
            this.sectionfordetails_dropdown.onchange = null;
            this.sectionfordetails_dropdown.value = secid;
            this.sectionfordetails_dropdown.onchange = oldevent;
        }

        this.select_image_details();
    }

    // ****************************************

    select_image_details()
    {
        let self = this;
        let p;

        rkWebUtil.wipeDiv( this.image_details_div );

        p = rkWebUtil.elemaker( "p", this.image_details_div, { "text": "Image details for " } );
        p.appendChild( this.sectionfordetails_dropdown );

        this.image_details_content_div = rkWebUtil.elemaker( "div", this.image_details_div );
        let sec = this.sectionfordetails_dropdown.value.toString();
        if ( sec == "_select_section" )
            return;

        if ( this.fakeanalysis_data.hasOwnProperty( sec ) ) {
            this.show_image_details_for_section( this.image_details_content_div, sec, this.fakeanlaysis_data[sec] );
        }
        else {
            let url = "fakeanalysisdata/" + this.id + "/" + this.data.provenancetag + "/" + sec;
            this.context.connector.sendHttpRequest( url, {},
                                                    (data) => { self.show_image_details_for_section(
                                                        this.image_details_content_div,
                                                        sec, data ) } );
        }
    }

    // ****************************************

    show_image_details_for_section( div, sec, indata )
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

        this.select_cutouts();
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
