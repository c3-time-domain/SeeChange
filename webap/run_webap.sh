#!/bin/bash

port=8080
if [ $# \> 0 ]; then
    port=$1
fi

bogus=0
if [ $# \> 1 ]; then
    bogus=1
fi


echo "Going to listen on port ${port}"

python updater.py &

if [ $bogus -ne 0 ]; then
    echo "WARNING : running with bogus self-signed certificate (OK for tests, not for anything public)"
    gunicorn -w 4 -b 0.0.0.0:${port} --timeout 0 --certfile /webap_code/bogus_cert.pem --keyfile /webap_code/bogus_key.pem seechange_webap:app
else
    gunicorn -w 4 -b 0.0.0.0:${port} --timeout 0 seechange_webap:app
fi
