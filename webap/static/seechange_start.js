import { seechange } from "./seechange_ns.js"
import "./seechange.js"

// **********************************************************************
// **********************************************************************
// **********************************************************************
// Here is the thing that will make the code run when the document has loaded

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
                renderer.init();
            }
        }
    },
    100
);

export { }
