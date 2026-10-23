import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"
import { SVGPlot } from "./svgplot.js"
import { ImView } from "./imview.js"

// **********************************************************************

seechange.DiaForcedPhotSearch = class
{
    constructor( context )
    {
        this.context = context;
        this.div = rkWebUtil.elemaker( "div", null, { "classes": [ "vbox", "grow1" ],
                                                      'id': 'diaforcedphotsearchdiv' } );
        this.ltcv_divs = {};
    }

    render()
    {
        let self = this;
        let hbox, box, vbox, p;

        hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox", "padex", "xmargin", "mostlyborder",
                                                                   "nogrow" ] } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "padex", "xmargin", "mostlyborder" ] } );
        rkWebUtil.elemaker( "p", vbox, { "text": "Show lightcurve for object:" } );
        p = rkWebUtil.elemaker( "p", vbox )
        this.ltcvid = rkWebUtil.elemaker( "input", p, { "id": "diaforcedphot_ltcvid",
                                                        "attributes": { "type": "text", "size": 36} } );
        p = rkWebUtil.elemaker( "p", vbox )
        rkWebUtil.button( p, "Show Object", () => { self.show_lightcurve_for_widget_object(); } );

        vbox = rkWebUtil.elemaker( "div", hbox, { "classes": [ "vbox", "padex", "xmargin", "mostlyborder" ] } );
        rkWebUtil.elemaker( "p", vbox, { "text": "(Searching not implemented yet; for now:)" } );
        rkWebUtil.button( vbox, "Show Them All!!!!!1!11!!1one11!!", (e) => { self.list_all_objs() } );

        this.ltcv_parent_div = rkWebUtil.elemaker( "div", this.div, { "classes": [ "vbox", "grow1" ] } );
    }

    list_all_objs() {
        let self = this;
        let provtag = this.context.provtag_wid.value;
        let promise = this.context.connector.httpPromise( "objswithdiaforcedphot/" + provtag );
        promise.then( (data) => { self.list_objects(data, provtag) } );
    }

    list_objects( data, provtag ) {
        let self = this;

        rkWebUtil.wipeDiv( this.ltcv_parent_div );
        rkWebUtil.elemaker( "h3", this.ltcv_parent_div, ( "All objects in provtag " + provtag
                                                          + " with dia_forced_photometry" ) );
        let tab = rkWebUtil.elemaker( "table", this.ltcv_parent_div, { "classes": [ "borderedcells", "selfstart" ] } );
        let tr = rkWebUtil.elemaker( "tr", tab );
        rkWebUtil.elemaker( "th", tr, "Name" );
        rkWebUtil.elemaker( "th", tr, "RA" );
        rkWebUtil.elemaker( "th", tr, "Dec" );

        for ( let i in data.name ) {
            tr = rkWebUtil.elemaker( "tr", tab );
            rkWebUtil.elemaker( "td", tr, { "text": data.name[i],
                                            "click": (e) =>{ self.show_lightcurve_for_object(data._id[i]) },
                                            "classes": [ "link" ] } );
            rkWebUtil.elemaker( "td", tr, data.ra[i].toFixed(4) );
            rkWebUtil.elemaker( "td", tr, data.dec[i].toFixed(4) );
        }
    }


    show_lightcurve_for_widget_object()
    {
        let objid = this.ltcvid.value.trim();
        this.show_lightcurve_for_object( objid );
    }

    show_lightcurve_for_object( objid )
    {
        // TODO checkif we already have this in ltcv_divs
        let self = this;
        let provtag = this.context.provtag_wid.value;
        rkWebUtil.wipeDiv( this.ltcv_parent_div );
        rkWebUtil.elemaker( "h3", this.ltcv_parent_div, { "text": "Loading...",
                                                          "classes": [ "warning", "italic" ] } );
        this.context.connector.sendHttpRequest( "diaforcedphotforobj/" + objid + "/" + provtag, {},
                                                (data) => { self.actually_show_lightcurve( data ); } );
    }


    actually_show_lightcurve( data )
    {
        let ltcv = new seechange.DiaForcedPhot( this.context, data );
        this.ltcv_divs[ data.provtag + "/" + data.objid ] = ltcv.div;
        rkWebUtil.wipeDiv( this.ltcv_parent_div );
        this.ltcv_parent_div.appendChild( ltcv.div );
        ltcv.render();
    }

}

// **********************************************************************

seechange.DiaForcedPhot = class
{
    static _n_diaforcedphots = 0;

    constructor( context, data, params )
    {
        this.context = context;
        this.data = data;
        // For convenience
        this.datadex = {};
        for ( let i = 0 ; i < data.diaforcedphotid.length ; ++i ) {
            this.datadex[ data.diaforcedphotid[i] ] = i;
        }

        let defaultparams = {
            "clipsize": 25,
            "oversamp": 5,
            "zscale_new_contrast": 0.1,
            "zscale_sub_contrast": 0.1
        }
        Object.assign( this, defaultparams );
        Object.assign( this, params );

        this.div = rkWebUtil.elemaker( "div", null, { "classes": [ "vbox", "grow1" ] } );
        this.clipcache = {};
        this.clippromise = {};
        this.clips_clipsize = null;
        this.clips_photid = null;
        this.imviewcache = {};
        this.imagepromise = {};
        this.ordinal = seechange.DiaForcedPhot._n_diaforcedphots;
        seechange.DiaForcedPhot._n_diaforcedphots += 1;
    }

    render()
    {
        let self = this;

        rkWebUtil.elemaker( "h2", this.div, { "text": "DIA Forced Photometry for " + this.data.objname
                                              + " ( " + this.data.objid + " )" } );

        let hbox = rkWebUtil.elemaker( "div", this.div, { "classes": [ "hbox", "grow1" ] } );
        this.ltcvdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "mostlyborder", "mmargin", "padex",
                                                                       "flexfitcontent", "scrolly",
                                                                       "vbox", "grow1" ] } );
        this.cutoutsdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "mostlyborder", "padex", "scrolly",
                                                                          "flexfitcontent", "vbox", "grow0" ] } );
        this.imagediv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "mostlyborder", "mmargin", "padex", "vbox",
                                                                        "flexfitcontent", "selfstretch", "grow1" ] } );

        // ****************************************
        // ltcvdiv

        this.svgplot = new SVGPlot.Plot( { "name": "ltcv-" + this.data.objid,
                                           "divid": "ltcv-svgplot-" + this.data.objid,
                                           "showerrbar": true,
                                           "manuallimits": true,
                                           "xtitle": "MJD",
                                           "ytitle": "Flux (nJy)"
                                         } );
        this.svgplot.addClickListener( (info) => { self.click_on_svgplot(info) } );
        this.ltcvdiv.appendChild( this.svgplot.topdiv );

        this.apervspsf = rkWebUtil.elemaker( "select", this.svgplot.buttonbox,
                                             { "change": (e) => { self.update_datasets(); } } );
        rkWebUtil.elemaker( "option", this.apervspsf, { "text": "PSF photometry",
                                                        "attributes": { "value": "psf", "selected": 1 } } );
        rkWebUtil.elemaker( "option", this.apervspsf, { "text": "Aperture Photometry",
                                                        "attributes": { "value": "aperture" } } );

        let filterorder = [ 'u', 'b', 'v', 'g', 'r', 'i', 'z', 'y' ]
        let filtercolors = { 'u': "#8800cc",
                             'b': "#0000cc",
                             'v': "#88cc00",
                             'g': "#00cc00",
                             'r': "#cc0000",
                             'i': "#cc8800",
                             'z': "#884400",
                             'y': "#442200" }
        let filters = new Set();
        for ( let filt of this.data.filter ) {
            filters.add( filt );
            if ( ! filterorder.includes( filt ) ) {
                filterorder.push( filt );
                filtercolors[filt] = "#000000";  // TODO, be better
            }
        }

        this.filters = [];
        this.filtercheckboxes = {};
        this.psfphotids = {};
        this.aperphotids = {};
        this.psfphotdexen = {};
        this.aperphotdexen = {};
        this.psfsets = {};
        this.apersets = {};
        this.phottype = "psf";
        for ( let filt of filterorder ) {
            if ( filters.has( filt ) ) {
                this.filters.push( filt )
                this.psfphotids[filt] = [];
                this.aperphotids[filt] = [];
                this.psfphotdexen[filt] = [];
                this.aperphotdexen[filt] = [];
                let psf_x = [];
                let psf_y = [];
                let psf_dy = [];
                let aper_x = [];
                let aper_y = [];
                let aper_dy = [];
                let psf_pointdata = [];
                let aper_pointdata = [];
                for ( let i in this.data.mjd ) {
                    if ( this.data.filter[i] == filt ) {
                        if ( ! this.data.psfflags[i] ) {
                            psf_x.push( this.data.mjd[i] );
                            psf_y.push( this.data.flux_psf[i] );
                            psf_dy.push( this.data.flux_psf_err[i] );
                            this.psfphotids[filt].push( this.data.diaforcedphotid[i] );
                            this.psfphotdexen[filt].push( i );
                            psf_pointdata.push( { 'id': this.data.diaforcedphotid[i], 'dex': i } );
                        }
                        if ( ! this.data.aperflags_0[i] ) {
                            aper_x.push( this.data.mjd[i] );
                            aper_y.push( this.data.flux_aper_0[i] );
                            aper_dy.push( this.data.flux_aper_0_err[i] );
                            this.aperphotids[filt].push( this.data.diaforcedphotid[i] );
                            this.aperphotdexen[filt].push( i );
                            aper_pointdata.push( { 'id': this.data.diaforcedphotid[i], 'dex': i } );
                        }
                    }
                }
                this.psfsets[filt] = new SVGPlot.Dataset( { 'x': psf_x, 'y': psf_y, 'dy': psf_dy,
                                                            'pointdata': psf_pointdata, linewid: 0,
                                                            'color': filtercolors[filt] } );
                this.apersets[filt] = new SVGPlot.Dataset( { 'x': aper_x, 'y': aper_y, 'dy': aper_dy,
                                                             'pointdata': aper_pointdata, linewid: 0,
                                                             'color': filtercolors[filt] } );
                this.svgplot.addDataset( this.psfsets[filt] );

                rkWebUtil.elemaker( "span", this.svgplot.buttonbox, { "text": "   " } );
                let checkboxid = "diaforcedphot-" + this.ordinal + "-checkbox-" + filt;
                this.filtercheckboxes[filt] = rkWebUtil.checkbox( this.svgplot.buttonbox,
                                                                  { "attributes": { "checked": 1 },
                                                                    "change": (e) => { self.update_datasets() },
                                                                    "id": checkboxid,
                                                                    "text": filt,
                                                                    "classes": [ "bold" ],
                                                                    "style": { "color": filtercolors[filt] }
                                                                  } );
            }
        }

        this.svgplot.redraw();

        // ****************************************
        // cutouts div

        this.clipboxes = {};
        this.clipcanvases = {};
        let dispsize = Math.round( this.clipsize * this.oversamp );
        let tab = rkWebUtil.elemaker( "table", this.cutoutsdiv );
        for ( let which of [ "new", "ref", "sub" ] ) {
            let tr = rkWebUtil.elemaker( "tr", tab, { "classes": [ "vcenter" ] } );
            rkWebUtil.elemaker( "th", tr, { "text": which } );
            let td = rkWebUtil.elemaker( "td", tr );
            this.clipboxes[which] = rkWebUtil.elemaker( "div", td, { "classes": [ "mostlyborder", "padex" ] } );
            this.clipboxes[which].style.poition = "relative";
            this.clipboxes[which].style.width = dispsize.toString() + "px";
            this.clipboxes[which].style.height = dispsize.toString() + "px";
            this.clipcanvases[which] = rkWebUtil.elemaker( "canvas", this.clipboxes[which],
                                                           { "attributes": { "width": dispsize,
                                                                             "height": dispsize
                                                                           },
                                                             "classes": [ "clipcanvas" ],
                                                             "click": (e) => { self.click_on_thumbnail(which) }
                                                           } );
            this.clipcanvases[which].style.width = dispsize.toString() + "px";
            this.clipcanvases[which].style.height = dispsize.toString() + "px";
        }

        tab = rkWebUtil.elemaker( "table", this.cutoutsdiv, { "classes": [ "borderless", "margintopem",
                                                                           "font75" ] } );
        this.cliptabcells = {};
        let tr;
        for ( let prop of [ "mjd", "filter", "flux_psf", "flux_aper_0", "flux_aper_n", "psfflags", "x", "y" ] ) {
            tr = rkWebUtil.elemaker( "tr", tab );
            rkWebUtil.elemaker( "th", tr, { "classes": [ "right" ], "text": prop + ": " } );
            this.cliptabcells[prop] = rkWebUtil.elemaker( "td", tr );
        }
        tr = rkWebUtil.elemaker( "tr", tab );
        // Hurm... looks like font size % is relative to what it would have been
        //   if there were no font-size already specified... so doing font75 here
        //   will make it .75 * .75 = 56.25%
        this.cliptabcells['id'] = rkWebUtil.elemaker( "td", tr, { "attributes": { "colspan": 2 },
                                                                  "classes": [ "font75" ] } );

        // ****************************************
        // image div

        this.imview = new ImView( { 'parent_div': this.imagediv,
                                    'show_image_name': false,
                                    'show_image_selector': true,
                                    'show_group_selector': true,
                                    'group_header': "Point:",
                                    'display_name_in_group_selector': true,
                                    'sort_group_selector': true,
                                    'display_name_in_image_selector': false,
                                    'show_copy_zoom_button': true,
                                    'group_change_callback': (group) => { self.imview_group_change( group.name ) }
                                  } );
    }

    update_datasets()
    {
        this.photttype = this.apervspsf.value;
        let sets = ( this.photttype == "psf" ) ? this.psfsets : this.apersets;
        let filts = [];
        for ( let filt of Object.keys(this.filtercheckboxes) ) {
            if ( this.filtercheckboxes[filt].checked ) filts.push( filt );
        }

        this.svgplot.clear( false );
        for ( let filt of filts ) {
            this.svgplot.addDataset( sets[filt] );
        }
        this.svgplot.redraw();
    }

    update_imview( photid, which=null )
    {
        let self = this;

        if ( which === null ) {
            let curim = this.imview.current_image;
            which = ( curim === null ) ? 'sub' : curim.name;
        }

        if ( ( this.imview.curent_group !== null ) && ( this.imview.current_group.name == photid ) &&
             ( this.imview.current_iamge !== null ) && ( this.imview.current_image.name == which ) )
            // Nothing to do
            return;

        let dex = this.datadex[ photid ];
        let displayname = "MJD " + this.data.mjd[dex].toFixed(4) + " - " + this.data.filter[dex];
        let group = this.imview.add_group( { 'name': photid, 'displayname': displayname }, false, true );

        if ( ! this.imview.has_image( which, group ) ) {
            let imgid = null;
            if      ( which == "new" ) imgid = this.data.newid[ dex ];
            else if ( which == "ref" ) imgid = this.data.refid[ dex ];
            else                       imgid = this.data.subid[ dex ];
            if ( imgid == null ) {
                // TODO IMVIEW ERROR MESSAGE
                window.alert( "Requested image doesn't exist." )
                return;
            }
            else {
                let getter = this.context._image_data_getter( "image_data/" + imgid );
                this.imview.add_image( { 'name': which, 'data_getter': getter }, true, group );
                this.imview.current_image.add_regionset( { "name": photid + "-" + which,
                                                           "stroke": "#c00000",
                                                           "shape": "box",
                                                           "size": 5,
                                                           "x": [ this.data.x[dex] ],
                                                           "y": [ this.data.y[dex] ]
                                                         } );
                this.imview.render_image();
            }
        }
        else {
            this.imview.set_current_image( group, which, true );
            this.imview.render_image();
        }
    }

    click_on_svgplot( info )
    {
        this.select_thumbnails( info.pointdata.id );
        this.update_imview( info.pointdata.id );
    }

    imview_group_change( photid )
    {
        // This callback will get called with the default group when the imview is first created;
        //   try to detect that.
        if ( ! this.datadex.hasOwnProperty( photid ) ) return;

        this.select_thumbnails( photid );

        let i = this.datadex[ photid ];
        let filt = this.data.filter[ i ];
        let setdex = this.filters.indexOf( filt );
        let pointdex = -1;
	if ( this.phottype == "psf" )
            pointdex = this.psfphotids[ filt ].indexOf( photid );
        else
            pointdex = this.aperphotids[ filt ].indexOf( photid );
        pointdex = ( pointdex >= 0 ) ? pointdex : null;
        this.svgplot.highlight( setdex, pointdex );
    }

    select_thumbnails( photid )
    {
        // TODO : update highlight on svgplot !!!

        let self = this;
        let i = this.datadex[ photid ];

        let clipsdone = false;
        if ( this.clipcache.hasOwnProperty( this.clipsize ) ) {
            if ( this.clipcache[ this.clipsize ].hasOwnProperty( photid ) ) {
                this.show_thumbnails( this.clipcache[ this.clipsize ][ photid ] );
                clipsdone = true;
            }
        }
        if ( ! clipsdone ) {
            let promise = null;
            if ( this.clippromise.hasOwnProperty( this.clipsize ) ) {
                if ( this.clippromise[ this.clipsize ].hasOwnProperty( photid ) ) {
                    promise = this.clippromise[ this.clipsize ][ photid ];
                }
            }
            if ( promise === null ) {
                promise = this.context.connector.httpPromise( "diaforcedphotclips/" + photid + "/" + this.clipsize,
                                                              null, false );
                if ( ! this.clippromise.hasOwnProperty( this.clipsize ) ) {
                    this.clippromise[ this.clipsize ] = {};
                }
                this.clippromise[ this.clipsize ][ photid ] = promise;
            }
            promise.then( (data) => self.receive_thumbnails(data, photid, self.clipsize) );
        }

        let tmp = { "digits": 4, "fixedprecision":2, "expprecision": 3, "nullok": true };

        this.cliptabcells["mjd"].innerHTML = this.data.mjd[i].toFixed( 5 );
        this.cliptabcells["filter"].innerHTML = this.data.filter[i];
        this.cliptabcells["flux_psf"].innerHTML = ( rkWebUtil.floatToString( this.data.flux_psf[i], tmp )
                                                    + " ± " +
                                                    rkWebUtil.floatToString( this.data.flux_psf_err[i], tmp )
                                                  );
        this.cliptabcells["flux_aper_0"].innerHTML = ( rkWebUtil.floatToString( this.data.flux_aper_0[i], tmp )
                                                       + " ± " +
                                                       rkWebUtil.floatToString( this.data.flux_aper_0_err[i], tmp )
                                                     );
        this.cliptabcells["flux_aper_n"].innerHTML = ( rkWebUtil.floatToString( this.data.flux_aper_n[i], tmp )
                                                       + " ± " +
                                                       rkWebUtil.floatToString( this.data.flux_aper_n_err[i], tmp )
                                                     );
        this.cliptabcells["psfflags"].innerHTML = "0x" + this.data.psfflags[i].toString(16);
        this.cliptabcells["x"].innerHTML = this.data.x[i].toFixed( 2 );
        this.cliptabcells["y"].innerHTML = this.data.y[i].toFixed( 2 );
        this.cliptabcells["id"].innerHTML = photid;
    }

    receive_thumbnails( data, photid, clipsize )
    {
        if ( ! ImView.is_little_endian() ) {
            window.alert( "This webap doesn't work on a platform that's not little-endian." );
            return;
        }
        let dv = new DataView( data );
        if ( clipsize != dv.getUint16( 0, true ) ) {
            window.alert( "Expected clipsize " + clipsize + " doesn't match what we got back: " +
                          dv.getUint16( 0, true ) );
            return;
        }
        let clipsq = clipsize * clipsize;
        let newdata = new Float32Array( dv.buffer, 4, clipsq );
        let refdata = new Float32Array( dv.buffer, 4 + 4 * clipsq, clipsq );
        let subdata = new Float32Array( dv.buffer, 4 + 8 * clipsq, clipsq );
        let tmp = ImView.zsc_limits( newdata, this.zscale_new_contrast );
        let newmin = tmp[0];
        let newmax = tmp[1];
        tmp = ImView.zsc_limits( subdata, this.zscale_sub_contrast );
        let submin = tmp[0];
        let submax = tmp[1];

        if ( ! this.clipcache.hasOwnProperty( clipsize ) ) {
            this.clipcache[ clipsize ] = {};
        }

        this.clipcache[ clipsize ][ photid ] = {
            'photid': photid,
            'clipsize': clipsize,
            'new': newdata,
            'ref': refdata,
            'sub': subdata,
            'newmin': newmin,
            'newmax': newmax,
            'submin': submin,
            'submax': submax
        };

        this.show_thumbnails( this.clipcache[ clipsize ][ photid ] );
    }

    show_thumbnails( clips )
    {
        this.clips_photid = clips['photid'];
        this.clips_clipsize = clips['clipsize'];
        let newdata = clips['new'];
        let refdata = clips['ref'];
        let subdata = clips['sub'];
        let clipsize = clips['clipsize'];

        // THOUGHT REQUIRED.  If clipsize has changed since the canvases were
        //   made, this will be broken.
        let dispsize = Math.round( clipsize * this.oversamp );
        let newctx = this.clipcanvases['new'].getContext( "2d", { 'alpha': false } );
        let refctx = this.clipcanvases['ref'].getContext( "2d", { 'alpha': false } );
        let subctx = this.clipcanvases['sub'].getContext( "2d", { 'alpha': false } );
        // I *think* this initialized all the data to 0 (based on Uint8ClampedArray docs)
        let newdisp = newctx.createImageData( dispsize, dispsize );
        let refdisp = refctx.createImageData( dispsize, dispsize );
        let subdisp = subctx.createImageData( dispsize, dispsize );
        // Remember that astronomical images think the lower-left is (0, 0),
        //    whereas browsers think the upper-left is (0, 0)
        for ( let y = 0 ; y < clipsize ; ++y ) {
            let starty = Math.round( (clipsize - y - 1) * this.oversamp );
            let nexty = Math.round( (clipsize - y) * this.oversamp );
            if ( nexty <= starty ) continue;
            if ( nexty > dispsize ) nexty = dispsize;
            for ( let x = 0 ; x < clipsize ; ++x ) {
                let startx = Math.round( x * this.oversamp );
                let nextx = Math.round( (x+1) * this.oversamp );
                if ( nextx <= startx ) continue;
                if ( nextx > dispsize ) nextx = dispsize;

                let dex = y * clipsize + x;
                let newbval = ImView.float_to_byte( newdata[dex], clips.newmin, clips.newmax );
                let refbval = ImView.float_to_byte( refdata[dex], clips.newmin, clips.newmax );
                let subbval = ImView.float_to_byte( subdata[dex], clips.submin, clips.submax );

                for ( let dispy = starty ; dispy < nexty ; ++dispy ) {
                    for ( let dispx = startx ; dispx < nextx ; ++dispx ) {
                        let off = 4 * ( ( dispy * dispsize ) + dispx );
                        newdisp.data[ off     ] = newbval;
                        newdisp.data[ off + 1 ] = newbval;
                        newdisp.data[ off + 2 ] = newbval;
                        newdisp.data[ off + 3 ] = 255;  // alpha
                        refdisp.data[ off     ] = refbval;
                        refdisp.data[ off + 1 ] = refbval;
                        refdisp.data[ off + 2 ] = refbval;
                        refdisp.data[ off + 3 ] = 255;
                        subdisp.data[ off     ] = subbval;
                        subdisp.data[ off + 1 ] = subbval;
                        subdisp.data[ off + 2 ] = subbval;
                        subdisp.data[ off + 3 ] = 255;
                    }
                }
            }
        }
        newctx.putImageData( newdisp, 0, 0 );
        refctx.putImageData( refdisp, 0, 0 );
        subctx.putImageData( subdisp, 0, 0 );
    }

    click_on_thumbnail( which )
    {
        if ( ( this.clips_clipsize === null ) || ( this.clips_photid === null ) ) return;
        this.update_imview( this.clips_photid, which );
    }
}
