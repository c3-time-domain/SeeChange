import sqlalchemy as sa

from models.image import Image
from models.base import SmartSession

# Verify that locks created with sess.connection().execute() get released with sess.rollback()

with SmartSession() as sess:
    sess.connection().execute( sa.text( f'LOCK TABLE images' ) )
    import pdb; pdb.set_trace()
    # Run the following query on the database, there should be a lock on table images:
    #    SELECT d.datname, c.relname, l.transactionid, l.mode, l.granted
    #    FROM pg_locks l
    #    INNER JOIN pg_database d ON l.database=d.oid
    #    INNER JOIN pg_class c ON l.relation=c.oid
    #    WHERE c.relname NOT LIKE 'pg_%';
    sess.rollback()
    import pdb; pdb.set_trace()
    # Run the query again, make sure there are no locks
    pass

