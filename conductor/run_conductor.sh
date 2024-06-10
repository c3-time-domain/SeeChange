#!/bin/bash

port=8082
if [ $# \> 0 ]; then
    port=$1
fi

echo "Going to listen on port ${port}"

python updater.py &
gunicorn -w 4 -b 0.0.0.0:${port} --timeout 0 webservice:app
