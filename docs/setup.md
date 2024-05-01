## Setting up a SeeChange instance

### Installing using Docker

At the moment, some of the things below will not work if you install Docker Desktop.  
It has to do with permissions and bind-mounting system volumes; 
because of how Docker Desktop works, the files inside the container all end up owned as root, 
not as you, even if they are owned by you on your own filesystem.  Hopefully there's a way to fix this, 
but in the meantime, install Docker Engine instead of Docker Desktop; instructions are here:

- Installing Docker Engine : https://docs.docker.com/engine/install/
- Setting up rootless mode (so you don't have to sudo everything) : https://docs.docker.com/engine/security/rootless/

#### Development shell -- local database

The `devshell` directory has a docker compose file that can create a development environment for you.  To set it up, you need to set three environment variables.  You can either manually set these with each and every `docker compose` command, you can set them ahead of time with `export` commands, or, recommended, you can create a file `.env` in the `devshell` directory with contents:
```
  COMPOSE_PROJECT_NAME=<yourname>
  USERID=<UID>
  GROUPID=<GID>
```

`<yourname>` can be any string you want.  If you are also using `docker compose` in the tests subdirectory, you will be happier if you use a different string here than you use there.  `<UID>` and `<GID>` are your userid and groupid respectively; you can find these on Linux by running the command `id`; use the numbers after `uid=` and `gid=`. (Do not include the name in parentheses, just the number.)

Once you've set these environment variables— either in a `.env` file, with three `export` commands, or by prepending them to every `docker compose` command you see below, you can start up a development shell in which to run code by running, while in the `devshell` subdirectory:
```
  docker compose up -d seechange
```

That will start several services.  You can see what's there by running
```
   docker compose ps
```

The services started include an archive server, a postgres database server, and a shell host.  The database server should have all of the schema necessary for SeeChange already created.  To connect to the shell host in order to run within this environment, run
```
   docker compose exec -it seechange /bin/bash
```

Do whatever you want inside that shell; most likely, this will involve running `python` together with either some SeeChange test, or some SeeChange executable. This docker image bind-mounts your seechange checkout (the parent directory of the `devshell` directory where you're working) at `/seechange`.  That means if you work in that directory, it's the same as working in the checkout.  If you edit something outside the container, the differences will be immediately available inside the container (since it's the same physical filesystem).  This means there's no need to rebuild the container every time you change any bit of code.

When you're done running things, you can just `exit` out of the seechange shell.  Making sure you're back in a shell on the host machine, and in the `devshell` subdirectory, bring down all of the services you started with:
```
   docker compose down
```

By default, the volumes with archived files and the database files will still be there, so next time you run `docker compose up -d seechange`, the database contents and archived images will all still be there.  If you want to create a completely fresh environment, instead run
```
   docker compose down -v
```

If all is well, the `-v` will delete the volumnes that stored the database and archive files.

You can see what volumes docker knows about with
```
  docker volume list
```

Note that this will almost certainly show you more than you care about; it will show all volumes that you or anybody else have on the system for any context.

There is one other bit of cleanup.  Any images created while you work in the devshell docker image will be written under the `devshell/temp_data` directory.  When you exit and come back into the docker compose environment, all those files will still be there.  If you want to clean up, in addition to adding `-v` to `docker compose down`, you will also want to `rm -rf temp_data`.

#### Development shell -- using an external existing database

TBD


#### Running tests

To run the tests on your local system in an environment that approximates how they'll be run on github, 
cd into `tests` and run the following command (which requires the "docker compose CLI plugin" installed to work):
```
   export GITHUB_REPOSITORY_OWNER=<yourname>
   export USERID=<uid>
   export GROUPID=<gid>
   docker compose build
   COMPOSE_PROJECT_NAME=<yourname> docker compose run runtests
```
where you replace `<uid>` and `<gid>` with your own userid and groupid; if you don't do this, the tests will run, but various pycache files will get created in your checkout owned by root, which is annoying.  `<yourname>` can be any string you want.  If you are working on a single-user machine, you can omit the `COMPOSE_PROJECT_NAME` variable; the purpose if it is to avoid colliding with other users on the same machine.

At the end, `echo $?`; if 0, that's a pass, if 1 (or anything else not 0), that's a fail.  
(The output you see to the screen should tell you the same information.)  
This will take a long time the first time you do it, as it has to build the docker images, 
but after that, it should be fast (unless the Dockerfile has changed for either image).  
The variable GITHUB_RESPOSITORY_OWNER must be set to *something*; it only matters if you try to push or pull the images.  
Try setting it to your github username, though if you really want to push and pull you're going to have to look up 
making tokens on github.  (The docker-compose.yaml file is written to run on github, which is why it includes this variable.)

After the test is complete, run
```
    COMPOSE_PROJECT_NAME=<yourname> docker compose down -v
```
(otherwise, the postgres container will still be running).


### Database migrations

Database migrations are handled with alembic.

If you've just created a database and want to initialize it with all the tables, run
```
  alembic upgrade head
```

After editing any schema, you have to create new database migrations to apply them.  Do this by running something like:
```
  alembic revision --autogenerate -m "<put a short comment here>"
```
The comment will go in the filename, so it should really be short.  
Look out for any warnings, and review the created migration file before applying it (with `alembic upgrade head`).

Note that in the devshell and test docker environments above, database migrations are automatically run when you create the environment with `docker compose up -d`, so there is no need for an initial `alembic upgrade head`.   However, if you then create additional migrations, and you haven't since run `docker compose down -v` (the `-v` being the thing that deletes the database), then you will need to run `alembic upgrade head` to apply those migrations to the running database inside your docker environment.

### Installing SeeChange on a local machine (not dockerized)

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

Install some of the standalone executables needed for 
analyzing astronomical images:

```bash
sudo apt install source-extractor psfex scamp swarp
sudo ln -sf /usr/bin/python3 /usr/bin/python
sudo ln -sf /usr/bin/SWarp /usr/bin/swarp
```

The last two lines will create links to the executables
with more commonly used spellings. 

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


To log in to postgres (as the user "postgres"): 
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

If the database has already been used (e.g., on a different branch), you may need to do:

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

#### Setting up environmental variables

Some environmental variables are used by SeeChange.
 - `GITHUB_REPOSITORY_OWNER` is the name of your github user (used only for dockerized tests). 
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
  local_write_dir: /path/to/local/archive
  archive_url: null

db:
  engine: postgresql
  user: postgres
  password: fragile
  host: localhost
  port: 5432
  database: seechange
```

Replace `/path/to/local/archive` with the path to the local archive directory.

The same files (`local_overrides.yaml` and `local_augments.yaml`) can be used
on the main SeeChange directory, where they have the same effect, 
just for running a real instance of the SeeChange pipeline locally. 

#### Running the tests

At this point the tests should be working from the IDE or from the command line:

```bash
pytest --ignore=extern
```

The extern folder includes submodules that do not support local testing at this point. 

You can also add a `.pytest.ini` file with the following: 

```
[pytest]
testpaths =
    tests
```

Which will limit pytest to automatically only run the tests in that folder, 
and ignore other test folders, e.g., those in the `extern` folder.
