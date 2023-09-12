### How to install and use SeeChange on a local machine (not dockerized)

As always, checkout the code from github: <https://github.com/c3-time-domain/SeeChange>.
We recommend using a virtual environment to install the dependencies. For example:

```bash
python3 -m venv venv
```

Then activate the virtual environment and install the dependencies:

```bash
cd SeeChange
source venv/bin/activate
pip install -r requirements.txt
```

This covers the basic python dependencies. 
Now we need to install postgres and set it up 
to run on the default port (5432) with a database called `seechange`.

On a mac, you can do this with homebrew:

```bash
brew install postgresql
brew services start postgresql
/usr/local/opt/postgres/bin/createdb seechange
```

Usually you will want to add the default user:
    
```bash
/usr/local/opt/postgres/bin/createuser -s postgres
```

On linux/debian use
 
```bash
sudo apt install postgresql
```

Make sure the default port is 5432 in /etc/postgresql/14/main/postgresql.conf
(assuming the version of postgres is 14).
To restart the service do 

```bash
sudo service postgresql restart
```


To login to postgres (as the user "postgres"): 
```bash
sudo -u postgres psql
```

From here you can create or drop the database:

```sql
CREATE DATABASE seechange;
DROP DATABASE seechange WITH(force);
```

To use the database, login as above but then change into the database:

```sql
\c seechange
```


#### Installing Q3C extension for postgres

Get the code from <https://github.com/segasai/q3c>. 
Installing following the instructions: 

```bash
make
make install
```

Login to psql and do:

```sql
\c seechange
CREATE EXTENSION q3c;
```

#### Getting the database schema up-to-date

The database schema is managed by alembic.
If the database is in a fresh state (just created), do:

```bash
alembic upgrade head
```

If the database has already been used (e.g., on a differnt branch), you may need to do:

```bash
alembic downgrade base
alembic upgrade head
```

To generate a new migration script, do:

```bash
alembic revision --autogenerate -m "message"
```

#### Installing submodules

To install submodules used by SeeChange 
(e.g., the `nersc-upload-connector` package that is used to connect the archive)
do the following:

```bash
cd extern/nersc-upload-connector
git submodule init
```

If those packages require updates, you can do that from the root SeeChange directory
using:

```bash
git submodule update
```

Note that you can just do 

```bash
git submodule update --init
```

from the root directory, which will also initialize any 
submodules that have not been initialized yet.

#### Setting up environment variables

Some environmental variables are used by SeeChange.
 - `GITHUB_REPOSITORY_OWNER` is the name of your github user (used only for dockerized tests). 
 - `SEECHANGE_TEST_ARCHIVE_DIR` is used to setup a local directory for test data archive. 
Usually this will point to a folder outside the SeeChange directory, 
where data can be downloaded and stored.
 - `SEECHANGE_CONFIG` can be used to specify the location of the main config file,
but if that is not defined, SeeChange will just use the default config at the top level 
of the SeeChange directory, or the one in the `tests` directory (when running local tests). 

#### Adding local config files for tests

To allow tests to find the archive and the local database, 
a custom config file needs to be loaded. 
The default file, in `SeeChange/tests/seechange_config_test.yaml`,
will automatically look for (and load) two local config files named
`local_overrides.yaml` and `local_augments.yaml`. 
The first will override any keys in the default config, 
and the second one will update the existing parameter dictionaries and lists. 

One way to set things up is to put the following into 
`SeeChange/tests/local_augments.yaml`:

```yaml
archive:
  local_read_dir: /path/to/local/archive
  archive_url: null

db:
  engine: postgresql
  user: postgres
  password: fragile
  host: localhost
  port: 5432
  database: seechange
```

Replace `/path/to/local/archive` with the path to the local archive directory, 
which should also be defined as the environmental variable `SEECHANGE_TEST_ARCHIVE_DIR`.

The same files (`local_overrides.yaml` and `local_augments.yaml`) can be used
on the main SeeChange directory, where they have the same effect, 
just for running a real instance of the SeeChange pipeline locally. 

#### Running the tests

At this point the tests should be working from the IDE or from the command line:

```bash
pytest --ignore=extern
```

The extern folder includes submodules that do not support local testing at this point. 


#### Testing data downloads

The `data` folder contains persistent local data that is used for testing. 
For example, a folder called `DECam_examples` is created with some files in it, 
when running some of the tests. This relatively slow download needs to happen only once. 
If you want to re-test that the download works, simply delete that folder. 

Additional files are created in that folder during tests, 
but they should all be cleaned up at the end of the testing run. 

