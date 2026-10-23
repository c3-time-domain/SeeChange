import { rkWebUtil } from "./rkwebutil.js";
import { seechange } from "./seechange_ns.js"

// **********************************************************************

seechange.ImageList = class
{
    constructor( context, parent, parenttabs, parentdiv, imagediv, cutoutsdiv,
                 images, fromtime, totime, provtag, projects )
    {
        this.context = context;
        this.parent = parent
        this.parenttabs = parenttabs;
        this.parentdiv = parentdiv;
        this.imagediv = imagediv;
        this.cutoutsdiv = cutoutsdiv;
        this.images = images;
        this.fromtime = fromtime;
        this.totime = totime;
        this.provtag = provtag;
        this.projects = projects;
        this.image_promises = {};
        this.image_objects = {};
    }

    render_page()
    {
        let self = this;
        let h2 = this;

        rkWebUtil.wipeDiv( this.parentdiv );
        this.maindiv = rkWebUtil.elemaker( "div", this.parentdiv, { "classes": [ "scrolly" ] } );
        
        h2 = rkWebUtil.elemaker( "h2", this.maindiv, { "text": "Images" } );
        if ( ( this.fromtime !== null ) && ( this.totime != null ) ) {
            rkWebUtil.text( h2, " from MJD " + this.fromtime.toFixed(2) + " to " + this.totime.toFixed(2) );
        }
        else if ( this.fromtime !== null ) {
            rkWebUtil.text( h2, " from MJD " + this.fromtime.toFixed(2) + " and later" );
        }
        else if ( this.totime !== null ) {
            rkWebUtil.text( h2, " up to MJD " + this.totime.toFixed(2) );
        }
        else {
            rkWebUtil.text( h2, " from all time" ); 
        }
        rkWebUtil.text( h2, " with provenance tag " + this.provtag );

        let rowrenderer = (images, fields, i) => {
            let row = rkWebUtil.elemaker( "tr", null );
            let td = rkWebUtil.elemaker( "td", row );
            rkWebUtil.elemaker( "a", td, { "text": images['filename'][i],
                                           "classes": [ "link" ],
                                           "click": () => { self.show_image( images["_id"][i] ) }
                                         }
                              );
            td = rkWebUtil.elemaker( "td", row, { "text": images["mjd"][i].toFixed(4) } );
            td = rkWebUtil.elemaker( "td", row, { "text": seechange.imagetypes[images["_type"][i]] } );
            td = rkWebUtil.elemaker( "td", row, { "text": images["filter"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": images["ra"][i].toFixed(4) } );
            td = rkWebUtil.elemaker( "td", row, { "text": images["dec"][i].toFixed(4) } );
            td = rkWebUtil.elemaker( "td", row, { "text": images["exp_time"][i].toFixed(1) } );
            td = rkWebUtil.elemaker( "td", row, { "text": images["section_id"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": images["project"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": images["target"][i] } );
            td = rkWebUtil.elemaker( "td", row, { "text": seechange.nullorfixed( images["airmass"][i], 2 ) } );
            td = rkWebUtil.elemaker( "td", row, { "text": seechange.nullorfixed( images["fwhm_estimate"][i], 2 ) } );
            td = rkWebUtil.elemaker( "td", row, { "text": seechange.nullorfixed( images["lim_mag_estimate"][i], 2 ) } );
            td = rkWebUtil.elemaker( "td", row, { "text": seechange.nullorfixed( images["zero_point_estimate"][i],
                                                                                 2 ) } );
            td = rkWebUtil.elemaker( "td", row, { "text": rkWebUtil.floatToString( images["bkg_mean_estimate"][i],
                                                                                   { 'digits': 2,
                                                                                     'nullok': true }
                                                                                 ) } );
            td = rkWebUtil.elemaker( "td", row, { "text": rkWebUtil.floatToString( images["bkg_rms_estimate"][i],
                                                                                   { 'digits': 2,
                                                                                     'nullok': true }
                                                                                 ) } );
            return row;
        };

        let fields = [ "filename", "mjd", "_type", "filter", "ra", "dec", "exp_time", "section_id", "project", "target",
                       "airmass", "fwhm_estimate", "lim_mag_estimate", "zero_point_estimate",
                       "bkg_mean_estimate", "bkg_rms_estimate" ]
        let fieldmap = { "_type": "type",
                         "exp_time": "texp",
                         "section_id": "sec",
                         "fwhm_estimate": "seeing",
                         "lim_mag_estimate": "m_lim",
                         "zero_point_estimate": "zp",
                         "bkg_mean_estimate": "bkg",
                         "bkg_rms_estimate": "σ_bkg"
                       }
        let tab = new rkWebUtil.SortableTable( this.images, fields, rowrenderer,
                                               { 'hdrs': fieldmap,
                                                 'dictoflists': true,
                                                 'initsort': [ '+mjd', '+section_id', '+_type' ],
                                                 'tableclasses': [ 'exposure_list_table' ],
                                                 'colorclasses': [ 'bgfade', 'bgwhite' ],
                                                 'colorlength': 3 } );
        tab.table.id = 'image_list_table';
        this.shown_image_list_sortabletable = tab;
        this.maindiv.appendChild( tab.table );
    }

    // THOUGHT REQUIRED.
    //   Right now we cache everything.  If the user clicks on too many
    //     images, they will use up their own memory.
    //   Right now, that's their problem.  But maybe we should expire
    //     things from the this.image_objects cache.
    show_image( imgid )
    {
        let self = this;
        
        rkWebUtil.wipeDiv( this.imagediv );
        if ( this.image_objects.hasOwnProperty( imgid ) ) {
            this.image_objects[ imgid ].show_imagediv();
            return;
        }

        this.parenttabs.selectTab( "imagedetail" );
        rkWebUtil.elemaker( "p", this.imagediv, { "text": "Loading image info...",
                                                  "classes": [ "warning", "bold", "italic" ] } );
        if ( ! this.image_promises.hasOwnProperty( imgid ) ) {
            this.image_promises[imgid] = this.context.connector.httpPromise( 'image_info/' + imgid );
        }
        this.image_promises[ imgid ].then( (info) => { self.make_image_details(info);
                                                       self.image_objects[imgid].show_imagediv(); } );
    }

    make_image_details( info )
    {
        let self = this
        let imgid = info._id;

        if ( ! this.image_objects.hasOwnProperty( imgid ) ) {
            this.image_objects[ imgid ] = new seechange.Image( this.context, info,
                                                               { 'provtag': this.provtag,
                                                                 'show_detections': info.is_sub
                                                               } );
            this.image_objects[ imgid ].listen( 'image',
                                                (signal, stuff) => {
                                                    self.parenttabs.selectTab( "imagedetail" );
                                                    rkWebUtil.wipeDiv( self.imagediv );
                                                    self.imagediv.appendChild( stuff.imagediv );
                                                },
                                                'image_list_image_listener' );
            if ( info.is_sub ) {
                // TODO, deal with avec/sans
                let div = this.image_objects[imgid].show_detectionsdiv();
                rkWebUtil.wipeDiv( this.cutoutsdiv );
                this.cutoutsdiv.appendChild( div );
            }
        }
    }

}

// **********************************************************************
// Make this into a module

export { }
