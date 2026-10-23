import { rkWebUtil } from "./rkwebutil.js";
import { ImView } from "./imview.js";
import { seechange } from "./seechange_ns.js";

// **********************************************************************

seechange.Image = class extends( rkWebUtil.listener )
{
    static _n_images = 0;

    constructor( context, info, params={} )
    {
        super();
        let default_params = { 'nomeas_cutouts': false,
                               'provtag': null,
                               'oversample': 5,
                               'show_detections': false,
                               'detections_box_size': 10,
                               'detections_box_color': '#008000'
                             };
        rkWebUtil.paramParser( default_params, params, this );

        this._addsignal( 'image' );
        this._addsignal( 'detections' );

        let self = this;
        let p, fncomps;

        this.context = context;
        this.info = info;
        this.issub = this.info.is_sub;
        fncomps = this.info.filepath.match( /\/?([^\/]+)$/ );
        this.filename = (fncomps === null) ? "????" : fncomps[1];

        this._imagediv = null;
        this.filename_h2 = null;
        this.imview_div = null;
        this.info_div = null;
        this.thumbnails_div = null;
        this.infopromises = {};
        this.image_promises = {};

        this.detections_wrapper = rkWebUtil.elemaker( 'div', null );
        this.detections_shown = 'neither';
        this.detectionsdiv_sans = null;
        this.detectionsdiv_avec = null;
        this.cutouts_promise = null;
    }

    get imagediv()
    {
        return this.get_imagediv();
    }

    get_imagediv() {
        if ( this._imagediv == null ) this.make_image_div();
        return this._imagediv;
    }

    show_imagediv()
    {
        let div = this.get_imagediv();
        this.emit_signal( 'image', { 'imagediv': div } );
        return div;
    }

    get detectionsdiv()
    {
        if ( this.detections_shown == 'avec' ) {
            return get_detectionsdiv( false );
        }
        else {
            return get_detectionsdiv( true );
        }
    }

    get_detectionsdiv( sans_measurements=true )
    {
        if ( ! this.issub ) return null;

        let self = this;

        if ( sans_measurements ) {
            let showsans = (data) => {
                if ( self.detections_shown != 'sans' ) {
                    rkWebUtil.wipeDiv( self.detections_wrapper );
                    self.detections_warpper.appendChild( self.detectionsdiv_sans );
                    self.detections_shown = 'sans';
                }
            };
            if ( this.detectionsdiv_sans == null ) this.make_detections_div( true, false, showsans );
            else showsans( null );
        }
        else {
            let showavec = (data) => {
                if ( self.detections_shown != 'avec' ) {
                    rkWebUtil.wipeDiv( self.detections_wrapper );
                    self.detections_wrapper.appendChild( self.detectionsdiv_avec );
                    self.detections_shown = 'avec';
                }
            };
            if ( this.detectionsdiv_avec == null ) this.make_detections_div( false, false, showavec );
            else showavec( null );
        }
        return this.detections_wrapper;
    }

    show_detectionsdiv( sans_measurements=true, parent=null )
    {
        let div = this.get_detectionsdiv( sans_measurements );
        if ( parent != null ) parent.appendChild( this.detections_wrapper );
        this.emit_signal( 'detections', { 'detectionsdiv': this.detections_wrapper } );
        return this.detections_wrapper;
    }



    // **********************************************************************
    // Image div

    make_image_div( remake=false )
    {
        let self = this;

        if ( ( this._imagediv !== null ) && ( ! remake ) ) return;

        this._imagediv = rkWebUtil.elemaker( "div", null, { "id": "_image_div_" + Image.n_images,
                                                            "classes": [ "vbox", "grow1" ] } );
        Image.n_images += 1;

        this.filename_h2 = rkWebUtil.elemaker( "h2", this._imagediv,
                                               { "text": "Image: " + this.filename + "( " + this.info._id + " )" } );

        let hbox = rkWebUtil.elemaker( "div", this._imagediv, { "classes": [ "hbox", "width100p", "grow1" ] } );

        this.imview_div = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "marginrightem",
                                                                          "grow1", "selfstretch" ] } );
        this.info_div = rkWebUtil.elemaker( "div", hbox, { "classes": [ "imageinfodiv" ] } );
        this.thumbnails_div = rkWebUtil.elemaker( "div", hbox );

        this.shown_imview = null;
        this.shown_info = null;
        this.zoomx0 = null;
        this.zoomy0 = null;
        this.zoomscale = null;
        if ( this.issub ) {
            this.imview = new ImView( { 'parent_div': this.imview_div,
                                        'clickcallback': (group, image, x, y) => {
                                            self.clicked_on_image( image, x, y ) },
                                        'show_image_selector': true,
                                        'show_copy_zoom_button': true,
                                        'image_change_callback': (image, group) => { self.load_info_view(image.name) },
                                  } );
            let ids = { 'sub': this.info._id, 'new': this.info.new_id, 'ref': this.info.warped_ref_id };
            for ( let name in ids ) {
                this.imview.add_image( { "name": name,
                                         "data_getter": this.context._image_data_getter( "image_data/" + ids[name] )
                                       } );
            }
            this.imview.render_image( false );

            this.infoview = this.show_info_view( this.info, 'sub' );
            this.infopromises = { 'new': null, 'ref': null };
            this.thumbnails_promise = null;
            if ( ( this.provtag !== null ) && this.show_detections )
                this.load_cutouts( (data) => { self.show_detections_on_image(data); } );
        }
        else {
            this.imview = new ImView( { 'parent_div': this.imview_div } );
            this.imview.add_image( { "name": "image",
                                     "data_getter": this.context._image_data_getter( "image_data/" + this.info._id )
                                   } );
            this.infoview = this.show_info_view( this.info, 'image' );
        }
    }

    load_info_view( which )
    {
        let self = this;

        if ( ( which == 'sub' ) || ( which == 'image' ) ) {
            this.show_info_view( this.info, which );
        }
        else {
            if ( this.infopromises[which] === null ) {
                let url = "image_info/";
                if ( which == 'new' ) {
                    url += this.info['new_id'];
                }
                else if ( which == 'ref' ) {
                    url += this.info['warped_ref_id'];
                }
                else {
                    window.alert( "Invalid thing to load_info_view: " + which );
                    return;
                }
                this.infopromises[which] = this.context.connector.httpPromise( url );
            }
            rkWebUtil.wipeDiv( this.info_div );
            rkWebUtil.elemaker( "p", this.info_div, { "text": "Loading image info...",
                                                      "classes": [ "warning", "bold", "italic" ] } );
            this.infopromises[which].then( (info) => { self.show_info_view( info, which ); } );
        }
    }

    show_info_view( info, which='image' )
    {
        rkWebUtil.wipeDiv( this.info_div );
        if ( which == 'new' ) {
            rkWebUtil.elemaker( "h3", this.info_div, { "text": "New Image" } );
        }
        else if ( which == 'ref' ) {
            rkWebUtil.elemaker( "h3", this.info_div, { "text": "Warped Ref Image" } );
        }
        else if ( which == 'sub' ) {
            rkWebUtil.elemaker( "h3", this.info_div, { "text": "Sub Image" } );
        }
        else if ( which != 'image' ) {
            rkWebUtil.elemaker( "h3", this.info_div, { "text": "Something has gone wrong." } );
        }

        let propvalues = { 'mjd': seechange.nullorfixed( info['mjd'], 5),
                           'ra': seechange.nullorfixed( info['ra'], 5 ),
                           'dec': seechange.nullorfixed( info['dec'], 5 ),
                           'filter': info['filter'],
                           'exptime': seechange.nullorfixed( info['exp_time'], 1 ),
                           'type': seechange.imagetypes[ info['_type'] ],
                           'instrument': info['instrument'],
                           'airmass': seechange.nullorfixed( info['airmass'], 2 ),
                           'seeing (pix)': seechange.nullorfixed( info['fwhm_estimate'], 2 ),
                           'zp': seechange.nullorfixed( info['zero_point_estimate'], 2 ),
                           'm_lim': seechange.nullorfixed( info['lim_mag_estimate'], 1 ),
                           'bkg': seechange.nullorexpo( info['bkg_mean_estimate'], 4 ),
                           'σ_bkg': seechange.nullorexpo( info['bkg_rms_esitmate'], 4 ),
                           'provid': info['provenance_id'],
                           'filename': this.filename,
                           'id': info['_id'],
                         }
        let table = rkWebUtil.elemaker( "table", this.info_div );
        for ( let prop in propvalues ) {
            let tr = rkWebUtil.elemaker( "tr", table );
            let td = rkWebUtil.elemaker( "td", tr, { "text": prop, "classes": [ "bold", "alignright" ] } );
            if ( prop == 'filename' ) {
                td = rkWebUtil.elemaker( "td", tr, { "text": propvalues[prop], "classes": [ "alignleft",
                                                                                            "tooltipsource" ] } );
                rkWebUtil.elemaker( "span", td, { "text": this.info.filepath, "classes": [ "tooltiptext" ] } );
            } else {
                td = rkWebUtil.elemaker( "td", tr, { "text": propvalues[prop], "classes": [ "alignleft" ] } );
            }
        }
    }

    load_cutouts( callback=null )
    {
        // Should only ever be called if the image is a sub image
        let self = this;
        if ( this.cutouts_promise === null ) {
            let url = ( "png_cutouts_for_sub_image/" + this.info._id + "/" + this.provtag +
                        "/1/" + (this.nomeas_cutouts ? "1" : "0") );
            this.cutouts_promise = this.context.connector.httpPromise( url );
        }
        this.cutouts_promise.then( callback );
    }

    show_detections_on_image( data )
    {
        if ( ! this.imview.regionsets.has( this.filename + "-detections" ) ) {
            this.imview.current_group.add_regionset( { 'name': this.filename + "-detections",
                                                       'stroke': this.detections_box_color,
                                                       'size': this.detections_box_size,
                                                       'x': data.cutouts.x,
                                                       'y': data.cutouts.y } );
            this.imview.render_image();
        }
    }

    clicked_on_image( which, x, y )
    {
        let self=this;
        if ( this.issub && this.show_detections ) {
            this.load_cutouts( (data) => { self.process_clicked_on_image(x, y, data); } );
        }
    }

    process_clicked_on_image( x, y, data )
    {
        let table, tr, td, ul, li, span, ttspan;

        for ( let dex in data.cutouts.x ) {
            if ( ( x <= data.cutouts.x[dex] + this.detections_box_size / 2. ) &&
                 ( x >= data.cutouts.x[dex] - this.detections_box_size / 2. ) &&
                 ( y <= data.cutouts.y[dex] + this.detections_box_size / 2. ) &&
                 ( y >= data.cutouts.y[dex] - this.detections_box_size / 2. ) )
            {
                rkWebUtil.wipeDiv( this.thumbnails_div );

                let table = rkWebUtil.elemaker( "table", this.thumbnails_div );
                let tr = rkWebUtil.elemaker( "tr", table );
                rkWebUtil.elemaker( "th", tr, { "text": "new" } );
                rkWebUtil.elemaker( "th", tr, { "text": "ref" } );
                rkWebUtil.elemaker( "th", tr, { "text": "sub" } );
                tr = rkWebUtil.elemaker( "tr", table );
                for ( let which of [ 'new', 'ref', 'sub' ] ) {
                    td = rkWebUtil.elemaker( "td", tr );
                    rkWebUtil.elemaker( "img", td,
                                        { "attributes": {
                                            "src": ( "data:image/png;base64," +
                                                     data.cutouts[which + '_png'][dex] ),
                                            "width": this.oversample * data.cutouts['w'][dex],
                                            "height": this.oversample * data.cutouts['h'][dex],
                                            "alt": which } } );
                }

                ul = rkWebUtil.elemaker( "ul", this.thumbnails_div );
                // row: source index, with good/bad
                li = rkWebUtil.elemaker( "li", ul );
                rkWebUtil.elemaker( "b", li, "source index: " );
                rkWebUtil.elemaker( "text", li, { "text": data.cutouts.source_index[dex] + "  "  } );
                span = rkWebUtil.elemaker( "b", li, { "text": ( data.cutouts.is_bad[dex] ?
                                                                "fails cuts" : "passes cuts" ) } );
                span.classList.add( "tooltipcolorlesssource" );
                ttspan = rkWebUtil.elemaker( "span", span, { "classes": [ "tooltiptext" ] } );
                ttspan.innerHTML = ( "<p>major_width: " + seechange.nullorfixed( data.cutouts.major_width[dex], 2 )
                                     + "<br>" +
                                     "minor_width: " + seechange.nullorfixed( data.cutouts.minor_width[dex], 2 )
                                     + "<br>" +
                                     "nbadpix: " + data.cutouts.nbadpix[dex]
                                     + "<br>" +
                                     "negfrac: " + data.cutouts.negfrac[dex]
                                     + "<br>" +
                                     "negfluxfrac: " + data.cutouts.negfluxfrac[dex]
                                     + "<br>" +
                                     "Gauss fit pos: " + seechange.nullorfixed( data.cutouts['gfit_x'][dex], 2 )
                                     + " , " + seechange.nullorfixed( data.cutouts['gfit_y'][dex], 2 ) +
                                     "</p>" )
                span.classList.add( data.cutouts.is_bad[dex] ? "bad" : "good" );
                // row: ra/dec
                li = rkWebUtil.elemaker( "li", ul );
                rkWebUtil.elemaker( "b", li, { "text": "(α, δ): " } );
                rkWebUtil.elemaker( "text", li, { "text": "(" +
                                                  seechange.nullorfixed( data.cutouts['measra'][dex], 5 )
                                                  + " , " +
                                                  seechange.nullorfixed( data.cutouts['measdec'][dex], 5 )
                                                  + ")" } );
                // row: x, y from cutouts; this is where it was originally detected.
                li = rkWebUtil.elemaker( "li", ul );
                rkWebUtil.elemaker( "b", li, { "text": "det. (x, y): " } );
                rkWebUtil.elemaker( "text", li, { "text": "(" +
                                                  seechange.nullorfixed( data.cutouts['cutout_x'][dex], 2 )
                                                  + " , " +
                                                  seechange.nullorfixed( data.cutouts['cutout_y'][dex], 2 )
                                                  + ")" } );
                // row: x, y from measurement table.
                li = rkWebUtil.elemaker( "li", ul );
                rkWebUtil.elemaker( "b", li, { "text": "meas. (x, y): " } );
                rkWebUtil.elemaker( "text", li, { "text": "(" +
                                                  seechange.nullorfixed( data.cutouts['x'][dex], 2 )
                                                  + " , " +
                                                  seechange.nullorfixed( data.cutouts['y'][dex], 2 )
                                                  + ")" } );
                // row: flux
                li = rkWebUtil.elemaker( "li", ul );
                rkWebUtil.elemaker( "b", li, { "text": "Flux: " } );
                rkWebUtil.elemaker( "text", li, { "text": seechange.nullorfixed( data.cutouts["flux"][dex], 0 )
                                                  + " ± " +
                                                  seechange.nullorfixed( data.cutouts["dflux"][dex], 0 )
                                                  + "  " +
                                                  + ( ( ( data.cutouts['aperrad'][dex] == null ) ||
                                                        ( data.cutouts['aperrad'][dex] <= 0 ) ) ?
                                                      '(psf)' :
                                                      ( "(aper r=" +
                                                        seechange.nullorfixed( data.cutouts['aperrad'][dex], 2 )
                                                        + " px)" ) ) } );
                // row: mag
                li = rkWebUtil.elemaker( "li", ul );
                rkWebUtil.elemaker( "b", li, { "text": "Mag: " } );
                rkWebUtil.elemaker( "text", li, { "text": seechange.nullorfixed( data.cutouts['mag'][dex], 2 )
                                                  + " ± " +
                                                  seechange.nullorfixed( data.cutouts['dmag'][dex], 2 ) } );
                // row; R/B
                li = rkWebUtil.elemaker( "li", ul );
                span = rkWebUtil.elemaker( "span", li );
                if ( ( data.cutouts['rb'][dex] == null ) || ( data.cutouts['rb'][dex] < data.cutouts['rbcut'][dex] ) )
                    span.classList.add( 'bad' );
                else
                    span.classList.add( 'good' );
                rkWebUtil.elemaker( "b", span, { "text": "R/B: " } );
                rkWebUtil.elemaker( "text", span, { "text": seechange.nullorfixed( data.cutouts['rb'][dex], 3 ) } );

                // Found one, don't bother looking at the rest
                break;
            }
        }
    }

    // **********************************************************************
    // Detections div

    make_detections_div( sans=true, remake=false, callback=null )
    {
        if ( ! this.issub ) {
            throw Error( "Can't make detections div when an image isn't a subtraction" );
        }

        rkWebUtil.wipeDiv( this.detections_wrapper );
        rkWebUtil.elemaker( 'h3', this.detections_wrapper, { "text": "Loading detections...",
                                                             "classes": [ "warning", "bold", "italic" ] } );

        let self = this;

        if ( sans ) {
            if ( this.detections_sans_promise == null ) {
                remake = true;
                let url = "png_cutouts_for_sub_image/" + this.info._id + "/" + this.provtag + "/1/0";
                this.detections_sans_promise = this.context.connector.httpPromise( url );
            }
            if ( remake )
                this.detections_sans_promise.then( (data) => { self.actually_make_detections_div( true, data ); } );
            if ( callback != null )
                this.detections_sans_promise.then( (data) => { callback(data) } );
        } else {
            if ( this.detections_avec_promise == null ) {
                remake = true;
                let url = "png_cutouts_for_sub_image/" + this.info._id + "/" + this.provtag + "/1/1";
                this.detections_avec_promise = this.context.connector.httpPromise( url );
            }
            if ( remake )
                this.detections_avec_promise.then( (data) => { self.actually_make_detections_div( false, data ); } );
            if ( callback != null )
                this.detections_avec_promise.then( (data) => { callback(data) } );
        }
    }

    actually_make_detections_div( sans_measurements, data )
    {
        let div, table, tr, th, td, img, span, ttspan;

        rkWebUtil.wipeDiv( this.detections_wrapper );

        if ( sans_measurements ) {
            this.detectionsdiv_sans = rkWebUtil.elemaker( "div", this.detections_wrapper );
            this.detections_shown = 'sans';
            div = this.detectionsdiv_sans;
        } else {
            this.detectionsdiv_avec = rkWebUtil.elemaker( "div", this.detections_wrapper );
            this.detections_shown = 'avec';
            div = this.detectionsdiv_avec;
        }

        table = rkWebUtil.elemaker( "table", div, { "id": "detectionstable" } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr );
        th = rkWebUtil.elemaker( "th", tr, { "text": "new" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "ref" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "sub" } );

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
                                          "width": this.oversample * data.cutouts['w'][i],
                                          "height": this.oversample * data.cutouts['h'][i],
                                          "alt": "new" } } );
            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts['ref_png'][i],
                                          "width": this.oversample * data.cutouts['w'][i],
                                          "height": this.oversample * data.cutouts['h'][i],
                                          "alt": "ref" } } );
            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts['sub_png'][i],
                                          "width": this.oversample * data.cutouts['w'][i],
                                          "height": this.oversample * data.cutouts['h'][i],
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
    }

}

// **********************************************************************
// Make this into a module

export { }
