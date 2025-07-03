import { seechange } from "./seechange_ns.js"
import { rkAuth } from "./rkauth.js"
import { rkWebUtil } from "./rkwebutil.js"
import { SVGPlot } from "./svgplot.js"

seechange.LtcvView = class
{
    constructor( parentdiv=null, initsnname=null, provtag=null )
    {
        this.parentdiv = parentdiv;
        this.initsnname = initsnname;
        this.provtag = provtag;
        this.zp = 31.4;
        this.zpunits = 'nJy';
        this.connector = new rkWebUtil.Connector( "/ltcv" );
    };

    // Only call init in the case where the full webpage is the ltcv view.
    //   Otherwise, just use the consturctor and render
    init()
    {
        let self = this;
        this.authdiv = document.getElementById( "authdiv" );
        this.parentdiv = document.getElementById( "pagebody" );

        this.initsnname = document.getElementById( "ltcv_initial_objid" ).value;
        this.provtag = document.getElementById( "ltcv_initial_provtag" ).value;
        this.zp = parseFloat( document.getElementById( "ltcv_initial_zp" ).value );
        this.zpunits = document.getElementById( "ltcv_initial_zpunits" ).value;
        
        this.auth = new rkAuth( this.authdiv, "",
                                () => { self.render_page(); },
                                () => { window.location.reload(); } );
        this.auth.checkAuth();
    };


    render_page()
    {
        let self = this;

        let h2, p, span, table, tr, hbox;

        rkWebUtil.wipeDiv( this.authdiv );
        p = rkWebUtil.elemaker( "p", this.authdiv,
                                { "text": "Logged in as " + this.auth.username
                                  + " (" + this.auth.userdisplayname + ") — ",
                                  "classes": [ "italic" ] } );
        span = rkWebUtil.elemaker( "span", p,
                                   { "classes": [ "link" ],
                                     "text": "Log Out",
                                     "click": () => { self.auth.logout( () => { window.location.reload(); } ) }
                                   } );
        
        rkWebUtil.wipeDiv( this.parentdiv );

        h2 = rkWebUtil.elemaker( "h2", this.parentdiv, { "text": "Detection lightcurve for " } );
        this.objnamespan = rkWebUtil.elemaker( "span", h2, { "text": "(..loading...)" } );
        rkWebUtil.elemaker( "text", h2, { "text": " (using prov. tag " + this.provtag + ")" } );

        table = rkWebUtil.elemaker( "table", this.parentdiv, { 'classes': [ "borderless" ] } );
        tr = rkWebUtil.elemaker( "tr", table );
        rkWebUtil.elemaker( "th", tr, { "text": "objid: ", 'classes': [ "right", "mmarginright" ] } );
        this.objidtd = rkWebUtil.elemaker( "td", tr, { "text": "(...loading...)" } );
        tr = rkWebUtil.elemaker( "tr", table );
        rkWebUtil.elemaker( "th", tr, { "text": "α, δ: ", 'classes': [ "right", "mmarginright" ] } );
        this.coordtd = rkWebUtil.elemaker( "td", tr, { "text": "(...loading...)" } );
    
        hbox = rkWebUtil.elemaker( "div", this.parentdiv, { "classes": [ "hbox", "flexeven" ] } );
        this.ltcvdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "mostlyborder", "mmargin", "padex",
                                                                       "flexfitcontent", "minwid40p",
                                                                       "scrolly" ] } );
        rkWebUtil.elemaker( "span", this.ltcvdiv, { "text": "(lightcurve loading)",
                                                    "classes": [ "bold", "italic", "warning" ] } );

        this.cutoutsdiv = rkWebUtil.elemaker( "div", hbox, { "classes": [ "mostlyborder", "mmargin", "padex",
                                                                          "flexfitcontent", "minwid40p",
                                                                          "scrolly" ] } );
        rkWebUtil.elemaker( "span", this.cutoutsdiv, { "text": "(cutouts loading)",
                                                       "classes": [ "bold", "italic", "warning" ] } );

        this.connector.sendHttpRequest( "/objectinfo/" + this.initsnname, {},
                                        (data) => { self.update_obj_info(data); } );
    };


    update_obj_info( data )
    {
        let self = this;
        this.objid = data.id;
        this.objnamespan.innerHTML = data.name;
        this.objidtd.innerHTML = data.id;
        this.coordtd.innerHTML = data.ra.toFixed(5).toString() + " , " + data.dec.toFixed(5).toString();
        // Only send the request for the cutouts and lightcurve here, because in render_page() we
        //   might not have known the uuid (the thing it know was either a uuid or a name), but
        //   we need the uuid for those API endpoints.
        this.connector.sendHttpRequest( ( "/objectltcv/" + this.objid + "/" + this.provtag + "/zp=" +
                                          this.zp.toString() + "/zpunits=" + this.zpunits ),
                                        {}, (data) => { self.plot_ltcv(data); } );
        this.connector.sendHttpRequest( "/objectcutouts/" + this.objid + "/" + this.provtag, {},
                                        (data) => { self.show_cutouts(data); } );
    }


    plot_ltcv( data )
    {
        let tmp
        let xdatas = {};
        let ydatas = {};
        let dydatas = {};
        let ymins = {};
        let ymaxes = {};
        let xmin = 1e32;
        let xmax = -1e32;
        this.titles = [];
        this.datasets = [];
        this.plots = [];

        let colortab = { 'g': '#00cc00',
                         'r': '#cc0000',
                         'i': '#aa4400',
                         'z': '#888800',
                         'Y': '#222200' };

        rkWebUtil.wipeDiv( this.ltcvdiv );
        
        for ( let i in data.instruments ) {
            let dex = data.instruments[i] + ':' + data.filters[i];
            xdatas[ dex ] = [];
            ydatas[ dex ] = [];
            dydatas[ dex ] = []
            ymins[ dex ] = 1e32;
            ymaxes[ dex ] = -1e32;
            for ( let j in data.measid ) {
                if ( ( data.instrument[j] == data.instruments[i] ) &&
                     ( data.filter[j] == data.filters[i] ) ) {
                    xdatas[ dex ].push( data.mjd[j] );
                    ydatas[ dex ].push( data.flux_psf[j] );
                    dydatas[ dex ].push( data.flux_psf_err[j] );
                    if ( data.mjd[j] < xmin ) xmin = data.mjd[j];
                    if ( data.mjd[j] > xmax ) xmax = data.mjd[j];
                    if ( data.flux_psf[j] < ymins[dex] ) ymins[dex] = data.flux_psf[j];
                    if ( data.flux_psf[j] > ymaxes[dex] ) ymaxes[dex] = data.flux_psf[j];
                }
            }
            if ( ymins[dex] > 0. ) ymins[dex] = 0.;
            tmp = 0.05 * ( ymaxes[dex] - ymins[dex] );
            ymins[dex] -= tmp;
            ymaxes[dex] += tmp;
        }
        tmp = 0.05 * (xmax - xmin);
        xmin -= tmp;
        xmax += tmp;

        for ( let i in data.instruments ) {
            let dex = data.instruments[i] + ':' + data.filters[i];
            let color;
            if ( colortab.hasOwnProperty( data.filters[i] ) )
                color = colortab[ data.filters[i] ];
            else
                color = '#000000';

            let dataset = new SVGPlot.Dataset( { 'name': dex,
                                                 'x': xdatas[ dex ],
                                                 'y': ydatas[ dex ],
                                                 'dy': dydatas[ dex ],
                                                 'color': color,
                                                 'linewid': 0,
                                                 'markersize': 16
                                               } );
            let plot = new SVGPlot.Plot( { 'divid': 'ltcv-' + dex,
                                           'svgid': 'svg-' + dex,
                                           'title': dex,
                                           'xtitle': 'mjd',
                                           'ytitle': 'flux (' + data.flux_unit + ')',
                                           'defaultlimits': [ xmin, xmax, ymins[dex], ymaxes[dex] ],
                                           'zoommode': 'default'
                                         } );
            plot.addDataset( dataset );

            this.ltcvdiv.appendChild( plot.topdiv );
            this.titles.push( dex );
            this.datasets.push( dataset );
            this.plots.push( plot );
        }
    }


    show_cutouts( data )
    {
        let table, tr, th, td, img;
        let oversample = 5;
        
        rkWebUtil.wipeDiv( this.cutoutsdiv );

        table = rkWebUtil.elemaker( "table", this.cutoutsdiv, { 'id': 'exposurecutoutstable' } );
        tr = rkWebUtil.elemaker( "tr", table );
        th = rkWebUtil.elemaker( "th", tr );
        th = rkWebUtil.elemaker( "th", tr, { "text": "new" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "ref" } );
        th = rkWebUtil.elemaker( "th", tr, { "text": "sub" } );

        for ( let i in data.cutouts.mjd ) {
            tr = rkWebUtil.elemaker( "tr", table );
            td = rkWebUtil.elemaker( "td", tr, { "text": "MJD: " + data.cutouts.mjd[i].toString() } );
            rkWebUtil.elemaker( "br", td );
            rkWebUtil.elemaker( "text", td, { "text": "Filter: " + data.cutouts.filter[i] } );

            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts.new_png[i],
                                          "width": oversample * data.cutouts.w[i],
                                          "height": oversample * data.cutouts.h[i],
                                          "alt": "new" } } );
            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts.ref_png[i],
                                          "width": oversample * data.cutouts.w[i],
                                          "height": oversample * data.cutouts.h[i],
                                          "alt": "ref" } } );
            td = rkWebUtil.elemaker( "td", tr );
            img = rkWebUtil.elemaker( "img", td,
                                      { "attributes":
                                        { "src": "data:image/png;base64," + data.cutouts.sub_png[i],
                                          "width": oversample * data.cutouts.w[i],
                                          "height": oversample * data.cutouts.h[i],
                                          "alt": "sub" } } );
        }
        
    };
    
};



// **********************************************************************
// Make this into a module

export { }


    
