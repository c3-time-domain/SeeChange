import sys
import os
import pytest
import psycopg2
import psycopg2.extras

@pytest.fixture(scope='module')
def db():
    host = 'seechange-postgres'
    port = 5432
    user = 'postgres'
    password = 'fragile'
    dbname = 'seechange'

    con = psycopg2.connect( dbname=dbname, host=host, port=port, user=user, password=password )
    yield con
    con.close()

@pytest.fixture
def cursor( db ):
    cursor = db.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
    yield cursor
    cursor.rollback()
    cursor.close()
    
@pytest.fixture(scope='module')
def testtable( cursor ):
    cursor.execute( "CREATE TABLE testing ( string text, value int primary key )" )
    cursor.commit()
    yield True
    cursor.execute( "DROP TABLE testing" )
    cursor.commit()

@pytest.fixture(scope='module')
def insertrows( cursor, testtable ):
    cursor.execute( "INSERT INTO testing(string,value) VALUES ('one',1)" )
    cursor.execute( "INSERT INTO testing(string,value) VALUES ('two',2)" )
    cursor.commit()
    yield True
    cursor.execute( "DELETE FROM testing WHERE value IN (1,2)" )
    cursor.commit()
                        
def test_database( cursor, insertrows ):
    cursor.execute( "SELECT * FROM testing ORDER BY value" )
    rows = cursor.fetchall()
    assert len(rows) == 2
    assert rows[0]['text'] == 'one' and rows[0]['value'] == 1
    assert rows[1]['text'] == 'two' and rows[1]['value'] == 2
