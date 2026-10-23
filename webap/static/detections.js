import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js";

// **********************************************************************

seechange.Detections = class
{
    constructor( context, data, inparams={} )
    {
        let self = this;
        this.params = {}
        Object.assign( this.params, inparams );

        this.context = context;
        this.data = data;

        this.div = rkWebUtil.elemaker( "div", null );

        table = rkWebUtil.elemaker( "table", this.div, { 'id': 'exposurecutoutstable' } );
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
    }
}
