#!/bin/bash

if [ ! -f $POSTGRES_DATA_DIR/PG_VERSION ]; then
    echo "Running initdb in $POSTGRES_DATA_DIR"
    echo $POSTGRES_PASSWORD > $HOME/pwfile
    /usr/lib/postgresql/15/bin/initdb -U postgres --pwfile=$HOME/pwfile $POSTGRES_DATA_DIR
    rm $HOME/pwfile
    /usr/lib/postgresql/15/bin/pg_ctl -D $POSTGRES_DATA_DIR start
    psql --command "CREATE DATABASE seechange OWNER postgres"
    psql --command "CREATE EXTENSION q3c" seechange
    /usr/lib/postgresql/15/bin/pg_ctl -D $POSTGRES_DATA_DIR stop
fi
exec /usr/lib/postgresql/15/bin/postgres -c config_file=/etc/postgresql/15/main/postgresql.conf
