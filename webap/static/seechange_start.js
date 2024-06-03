import { seechange } from "./seechange.js"

// **********************************************************************
// **********************************************************************
// **********************************************************************
// Here is the thing that will make the code run when the document has loaded
// It only make sense when included in a HTML document after seechange.js.

seechange.started = false

// console.log("About to window.setInterval...");
seechange.init_interval = window.setInterval(
    function()
    {
        var requestdata, renderer;

        if (document.readyState == "complete")
        {
            // console.log( "document.readyState is complete" );
            if ( !seechange.started )
            {
                seechange.started = true;
                window.clearInterval( seechange.init_interval );
                renderer = new seechange.Context();
                renderer.render_page();
            }
        }
    },
    100
);

export { }
