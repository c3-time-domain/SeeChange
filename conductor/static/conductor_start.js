import { scconductor } from "./conductor.js";

scconductor.started = false;
scconductor.init_interval = window.setInterval(
    function()
    {
        var requestdata, renderer;
        if ( document.readyState == "complete" ) {
            if ( !scconductor.started ) {
                scconductor.started = true;
                window.clearInterval( scconductor.init_interval );
                renderer = new scconductor.Context();
                renderer.init();
            }
        }
    },
    100 );
