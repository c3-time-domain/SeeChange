import sys
import os
import pytest
import psycopg2
import psycopg2.extras

@pytest.fixture(scope='module')
def db_con_and_cursor():
    host = 'seechange_postgres'
    port = 5432
    user = 'postgres'
    password = 'fragile'
    dbname = 'seechange'

    con = psycopg2.connect( dbname=dbname, host=host, port=port, user=user, password=password )
    cursor = con.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
    yield con, cursor
    con.rollback()
    con.close()

@pytest.fixture(scope='module')
def testtable( db_con_and_cursor ):
    con, cursor = db_con_and_cursor
    cursor.execute( "CREATE TABLE testing ( string text, value int primary key )" )
    con.commit()
    yield True
    cursor.execute( "DROP TABLE testing" )
    con.commit()

@pytest.fixture(scope='module')
def insertrows( db_con_and_cursor, testtable ):
    con, cursor = db_con_and_cursor
    cursor.execute( "INSERT INTO testing(string,value) VALUES ('one',1)" )
    cursor.execute( "INSERT INTO testing(string,value) VALUES ('two',2)" )
    con.commit()
    yield True
    cursor.execute( "DELETE FROM testing WHERE value IN (1,2)" )
    con.commit()
                        
def test_database( db_con_and_cursor, insertrows ):
    con, cursor = db_con_and_cursor
    cursor.execute( "SELECT * FROM testing ORDER BY value" )
    rows = cursor.fetchall()
    assert len(rows) == 2
    assert rows[0]['string'] == 'one' and rows[0]['value'] == 1
    assert rows[1]['string'] == 'two' and rows[1]['value'] == 2
