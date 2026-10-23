import { seechange } from "./seechange_ns.js"
import "./seechange.js"

// **********************************************************************
// Make sure the user sees errors.  *Maybe* take this out once
//   we think we're out of development.  If that ever happens.

seechange._erralert = function(event) {
    if ( event.message.substring( 0, 19 ) == "ResizeObserver loop" ) {
        console.log( event.message );
    }
    else {
        alert( "Error in javascript: " + event.message
               + "\nFile: " + event.filename
               + "\nLine: " + event.lineno );
    }
}

seechange._promisealert = function(event) {
    alert( "Unhandled promise rejection: " + event.reason );
    // console.log( "Unhandled promise rejection: " + event.reason );
}

window.addEventListener( 'error', seechange._erralert );
window.addEventListener( 'unhandledrejection', seechange._promisealert );

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

export { seechange }
