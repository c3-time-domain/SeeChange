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
    gunicorn -w 4 -b 0.0.0.0:${port} --timeout 0 --certfile /webservice_code/conductor_bogus_cert.pem --keyfile /webservice_code/conductor_bogus_key.pem webservice:app
else
    gunicorn -w 4 -b 0.0.0.0:${port} --timeout 0 webservice:app
fi
