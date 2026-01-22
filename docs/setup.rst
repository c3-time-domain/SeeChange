.. toctree::

*******************************
Setting up a SeeChange instance
*******************************

While it is probably possible to set up an environment on any machine to run SeeChange, all of our development and testing, and the production running for LS4, is done in a Dockerized environment.  As such, that is all that is documented here.

Most of these instructions assume you're setting up a SeeChange instance to hook into an existing installation.  If you want to start a local isolated environment for development and testing, see :doc:`development`.  If, however, you're setting up an all new installation of SeeChange that's going to run on a survey that somebody hasn't already set up for you, see :ref:`full-server-setup` below.

Checking out the code
=====================

You can pull the code with::

   git clone https://github.com/c3-time-domain/SeeChange.git
   git submodule update --init

(The second command is needed to pull embedded git archives within SeeChange.)  If you're developing the code, you may want to pull it differently; see (TODO reference).

.. _docker-image:
Acquiring or Building the Docker Image
======================================

A "current" docker image for SeeChange can be pulled (with ``docker pull``) from the following locations (where YYYMMDD are used for different releases; sometimes there may be an "a", "b", etc. after YYYYMMDD if we uploaded multiple versions in one day).  It should usually be safe to use the "latest" version, but you might want to pull a specific dated version for reproducibility.  The ``_nocode`` docker images do *not* actually include the SeeChange code, just the necessary environment.  To use this image, you will need to bind-mount an installation of SeeChange (see below).  This is what you want to use for development (and, right now, SeeChange is under heavy development).

 - ``docker.io/rknop/seechange:YYYYMMDD``
 - ``docker.io/rknop/seechange:latest``
 - ``docker.io/rknop/seechange_nocode:YYYYMMDD``
 - ``docker.io/rknop/seechange_nocode:latest``
 - ``registry.nersc.gov/m4616/seechange:YYYYMMDD``
 - ``registry.nersc.gov/m4616/seechange:latest``
 - ``registry.nersc.gov/m4616/seechange_nocode:YYYYMMDD``
 - ``registry.nersc.gov/m4616/seechange_nocode:latest``

**NOTE**: Currently, these images are only for the ``x86_64`` architecture.  If you are on an ARM machine (which is the case for all recent Macs), you may be able to use these images, but they could be very inefficient.

Pulling the docker image on NERSC
---------------------------------

On NERSC, use ``podman-hpc`` instead of ``docker``.  You pull the image with ``podman-hpc pull <imagename>``.  (Do **not** do ``podman-hpc image pull ...``, as that will seem to work, but not pull the images in such a way that they will work on any node other than the one you pulled it on.  If you do ``podman-hpc pull <imagename>``, the image will be available on all nodes.)

Building the Docker Image
-------------------------

If you need to build the docker image yourself, you should be able to accomplish this by running the following in the top level of the SeeChange checkout::

   docker build --target included_code -t seechange:<tag> -f docker/application/Dockerfile .

Replace ``<tag>`` with whatever you want (this is just part of the name of the docker image you are building); if you omit ``:<tag>``, then the "latest" image will be build.  This command will build a docker image that has the version of the code in the checkout included in the image.  If you want to build an image that doesn't include the code, and plan to bind-mount the code yourself, then replace ``included_code`` with ``bindmount_code``.  If you want to be able to run all of the tests, instead use ``--target test_included`` or ``--target test_bindmount`` in place of ``--target included_code``.  (The reason to have a separate image for tests is that they require several additional things that bloats the docker image.  The non-test versions of the image are slightly smaller, though still distressingly large.)

.. _dirs:
Setting up necessary directories
================================

For an installation of SeeChange that is hooking into a pre-existing database, you need to identify the following directories:

- A working directory; below we'll call this ``workdir``.  This is where you will store configuration files, and where you may store other things.
- A "local file store" for data; below we'll call this ``data_root``.  This should be on a fast disk, and should have a lot of space.  E.g., in NERSC, this should be somewhere under your scratch space.
- A temporary directory; below will call this ``data_temp``.  This is where temporary files are written (and, currently, not cleaned up often enough).  If no code is running and you find files here, it's safe to delete them.  To be safe, this directory should have at least a few hundred GB of space availble.

Optionally, if you aren't using the docker image with the code included, you may also need to set up an install directory; see :ref:`installing-code` below.

If you're setting up a whole new installation of SeeChange, there are additional directories and servers you will need; see (TODO reference).


.. _installing-code:
Installing the code
===================

**Note**: This step is unnecessary if you're using a docker image with the SeeChange code included (see :ref:`docker-image`).

Get the latest "production" version of the code by cloning the ``main`` branch of this archive: https://github.com/c3-time-domain/SeeChange/

This can (and should?) be done outside of the docker image.  Pick a location for the code to be installed, and run, at the top level of a SeeChange checkout::

   ./configure --with-installdir=<installdir>
   make install


If that fails, try running::

   autoreconf
   ./configure --with-installdir=<installdir>
   make install

You will then need to add ``<installdir>`` to your ``PYTHONPATH``.  (Instructions for doing this in the Docker environment are included below.)

Instead of installing the code, it is *probably* possible to just run the code in place, and just add the top level of your SeeChange checkout to your ``PYTHONPATH``.  (This is how our tests usually work.)

.. _config:
Creating a configuration file
=============================

SeeChange depends on a configuration file to tell it where the database is, where necessary external servers are, where it can read and write images and other data products, and, last but not least, the parameters to use when running the various steps of the pipeline.  You can find a default configuration file in ``default_config.yaml`` in the top level of the git checkout.

If you're just going to be running tests or doing local development, you mostly don't need to worry about this, but can use the configuration file that is automatically set up by the tests or the dvelopment environment.

To make your own configuration file, we actually do not recommend editing ``default_config.yaml``.  Rather, just copy it exactly as it is to (or, better, make a symbolic in) your work directory (which may well be the top level of your SeeChange checkout, in which case no coyping is needed).  If you look near the top of ``default_config.yaml``, you'll see that it refers to two files ``local_overrides.yaml`` and ``local_augments.yaml``.  These are the ones we recommend you edit.  Easiest is just to leave ``local_augments.yaml`` alone, and only edit ``local_overrides.yaml``.  In ``local_overrides.yaml``, you can override any config values from the default file with your own versions.  At runtime, your own versions will be used.

There are a few things you definitely need to set.  If you're hooking into an existing SeeChange instance, then you need to point to the right web application server, archive server, and database.  This means setting all the right values underneath ``db``, ``archive``, ``conductor``, and ``webap``.  You may also need to make sure you have any standard configuration values that this existing SeeChange instance uses.  Hopefully, the person who maintains this existing SeeChange instance will have a configuration file to give you with these values; below, we'll call that ``<survey_config>.yaml``.  Stick this file in your working directory.  Set yourself up to read this survey config by putting the following at the top of your ``local_overrides.yaml``::

  preloads:
    - <survey_config>.yaml

replacing ``<survey_config>`` with whatever is right for the file you are given.

Next, you need to add ``path`` section to your ``local_overrides.yaml`` file::

  path:
    data_root: /data_root
    data_temp: /data_temp

The directories in this code snippit are exactly what you'll use if you follow the instructions for running the docker container below.  You can, of course, use different directories if you know what you're doing.

Ideally, again assuming you were given a configuration file for the survey you are hooking into, this is everything you need to do.  However, you may wish to make other changes.  You might want to be trying actual different things in the pipeline, in which case hopefully you know what you're doing.  You may also want to enable or disable things relative to what's the survey config (e.g. the sending of alerts).

Running the Docker Container
============================

Assuming you're using the included-code version, cd into your ``workdir`` and run the docker container with::

   docker run -it \
      --mount type=bind,source=<workdir>,target=/workdir \
      --mount type=bind,source=<data_root>,target=/data_root \
      --mount type=bind,source=<data_temp>,target=/data_temp \
      --env SEECHANGE_CONFIG=/workdir/default_config.yaml \
      <image> /bin/bash

Replacing ``<workdir>``, ``<data_root>``, and ``<data_temp>`` with the directories you identified above in :ref:`dirs`.
Replace ``<image>`` with the name of your docker image (which you built or pulled above in :ref:`docker-image`).  This command assumes that you're using an included-code image; see "Bind-mounting the code" below otherwise.

This docker command will give you a shell inside the container.  (You probably want to ``cd /workdir`` once you're inside the container.)  If you're on NERSC, you should be able to just replace ``docker`` with ``podman-hpc`` above.  Notice that this sets the environment variable ``SEECHANGE_CONFIG`` to point at the config file we recommended in :ref:`config` above.  If you are using a configuration file somewhere else, replace ``/workdir/default_config.yaml`` with the *in-container* location of your actual configuration file.

You can verify that things are working by running a python interactive session and trying to import something from SeeChange::

  root@3b6df0c581ff:/seechange# python
  Python 3.13.5 (main, Jun 25 2025, 18:55:22) [GCC 14.2.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> from util.config import Config
  >>> cfg = Config.get()
  >>> print( cfg.value( 'path.data_root' ) )
  /data_root

If all is well, you won't get any error messages doing this.

Bind-mounting the code
----------------------

If you're developing, and you're going to be editing the code, you probably don't want to have to rebuild the docker image every time you change the code.  In this case, you should either bind mount the place where you [installed the code](#installing-code), or the top level of your SeeChange checkout.  Do this by two more arguments to the ``docker`` command::

   docker run -it \
      --mount type=bind,source=<workdir>,target=/workdir \
      --mount type=bind,source=<data_root>,target=/data_root \
      --mount type=bind,source=<data_temp>,target=/data_temp \
      --mount type=bind,source=<install_dir>,target=/seechange \
      --env SEECHANGE_CONFIG=/workdir/default_config.yaml \
      --env PYTHONPATH=/seechange \
      <image> /bin/bash

where `<install_dir>` is where the seechange code is located-- either the place you installed it, or the top level of the SeeChange checkout.  (For development, the latter is probably more convenient.)  In this case, ``<image>`` should be one of the ``_nocode`` images from :ref:`docker-image`.


.. _full-server-setup:
Full Server Setup
=================

In addition to the running SeeChange code, there are a few servers that need to be set up and run.  This includes at the very least PostgreSQL database server.  You probably also want to set up a web application server, and an archive server.

TODO DOCUMENT THIS








----

OLD



### Installing using Docker

At the moment, some of the things below will not work if you install Docker Desktop.  It has to do with permissions and bind-mounting system volumes; because of how Docker Desktop works, the files inside the container all end up owned as root, not as you, even if they are owned by you on your own filesystem.  Hopefully there's a way to fix this, but in the meantime, install Docker Engine instead of Docker Desktop; instructions are here:

- Installing Docker Engine : https://docs.docker.com/engine/install/

- Setting up rootless mode (so you don't have to sudo everything) : https://docs.docker.com/engine/security/rootless/ (There is some indication that this is difficult to get working; you may be happier just installing docker engine and adding yourself to the `docker` group in `/etc/group`.)

In order to actually _use_ the installation, you need to have a configuration file that points to the database, file store, and webap server relevant for what you're doing.  (TODO: document for LS4.)



.. _dev_shell_local_database:
#### Development shell — local transient database

*Warning: we don't always keep the devshell docker compose file up to date.  The docker compose file in tests will always be up to date.  You can often just use that as a dev environment.  Basically everything in this section will work as is if you start in the tests directory rahter than the devshell directory.*

The `devshell` directory has a docker compose file that can create a development environment for you.  To set it up, you need to set three environment variables.  You can either manually set these with each and every `docker compose` command, you can set them ahead of time with `export` commands, or, recommended, you can create a file `.env` in the `devshell` directory with contents:
```
  IMGTAG=[yourname]_dev
  COMPOSE_PROJECT_NAME=[yourname]
  USERID=[UID]
  GROUPID=[GID]
  CONDUCTOR_PORT=[port]
  WEBAP_PORT=[port]
  MAILHOG_PORT=[port]
```

`[yourname]` can be any string you want.  If you are also using `docker compose` in the tests subdirectory, you will be happier if you use a different string here than you use there.  `[UID]` and `[GID]` are your userid and groupid respectively; you can find these on Linux by running the command `id`; use the numbers after `uid=` and `gid=`. (Do not include the name in parentheses, just the number.)  The three [port] lines are optional.  CONDUCTOR_PORT defaults to 8082, WEBAP_PORT to 8081, and MAILHOG_PORT to 8025.  If multiple people are running docker on the same machine, you will probably need to configure these; otherwise, the defaults are probably fine.  (If, when running `docker compose up` below, you get errors about ports in use, that means you probably need to set these numbers.)  Once you start a container, services inside the container will be available on those ports of `localhost` on the host machine.  That is, if you've set `CONDUCTOR_PORT=8082` (or just left it at the default), a web browser on the host machine pointed at `https://localhost:8082/` will show the conductor's web interface.  (Because it uses a self-signed SSL certificate inside the dev environment, your browser will give you a security warning that you need to agree to override in order to actually load the page.)

Once you've set these environment variables— either in a `.env` file, with three `export` commands, or by prepending them to every `docker compose` command you see below, you can start up a development shell in which to run code by running, while in the `devshell` subdirectory:

```
  docker compose build
  docker compose up -d seechange
```

The `build` command doesn't need to be run every time, but should be run every time you update from the archive, or make any changes to the dockerfiles, requirements file, or docker compose files.  (In pratice: run this every so often.  It will be pretty fast (less than 1 minute) if no rebuilds are actually needed.)

The `docker compose up...` command will start several services.  You can see what's there by running
```
   docker compose ps
```

The services started include an archive server, a postgres database server, a webap, a conductor, a test mail server, and a shell host.  The database server should have all of the schema necessary for SeeChange already created.  To connect to the shell host in order to run within this environment, run
```
   docker compose exec -it seechange /bin/bash
```

Do whatever you want inside that shell; most likely, this will involve running `python` together with either some SeeChange test, or some SeeChange executable. This docker image bind-mounts your seechange checkout (the parent directory of the `devshell` directory where you're working) at `/seechange`.  That means if you work in that directory, it's the same as working in the checkout.  If you edit something outside the container, the differences will be immediately available inside the container (since it's the same physical filesystem).  This means there's no need to rebuild the container every time you change any bit of code.

Assuming you're running this on your local machine (i.e. you are running your web browser on the same machine as where you did `docker compose up -d seechange`), there are a couple of web servers available to you.  The SeeChange webap will be running at `localhost:8081` (with the value you specified in the env var `WEBAP_PORT` in place of 8081, if applicable), and the conductor's web interface will be running at `localhost:8082` (or the value you specified in `CONDUCTOR_PORT` in place of 8082).

When you're done running things, you can just `exit` out of the seechange shell.  Making sure you're back in a shell on the host machine, and in the `devshell` subdirectory, bring down all of the services you started with:
```
   docker compose down
```

By default, the volumes with archived files and the database files will still be there, so next time you run `docker compose up -d seechange`, the database contents and archived images will all still be there.  If you want to create a completely fresh environment, instead run
```
   docker compose down -v
```

If all is well, the `-v` will delete the volumes that stored the database and archive files.

You can see what volumes docker knows about with
```
  docker volume list
```

Note that this will almost certainly show you more than you care about; it will show all volumes that you or anybody else have on the system for any context.

There is one other bit of cleanup.  Any images created while you work in the devshell docker image will be written under the `devshell/temp_data` directory.  When you exit and come back into the docker compose environment, all those files will still be there.  If you want to clean up, in addition to adding `-v` to `docker compose down`, you will also want to `rm -rf temp_data`.

#### Development shell — using an external existing database

TBD


#### Running tests

You can run tests in an environment that approximates how they'll be run via CI in github.  Go into the `tests` directory and create a file `.env` with contents:
```
  IMGTAG=[yourname]_test
  COMPOSE_PROJECT_NAME=[yourname]
  USERID=[UID]
  GROUPID=[GID]
  CONDUCTOR_PORT=[port]
  WEBAP_PORT=[port]
  MAILHOG_PORT=[port]
```

(See :ref:`dev_shell_local_database` for a description of what all these environment variables mean.)

Make sure your docker images are up to date with
```
   docker compose build
```
then run
```
   docker compose run runtests
```

At the end, `echo $?`; if 0, that's a pass, if 1 (or anything else not 0), that's a fail.  (The output you see to the screen should tell you the same information.)  This will take a long time the first time you do it, as it has to build the docker images, but after that, it should be fast (unless the Dockerfile has changed for either image).

After the test is complete, run
```
    docker compose down -v
```
(otherwise, the postgres container will still be running).

As with :ref:`dev_shell_local_database`, you can also get a shell in the test environment with
```
   docker compose up -d shell
   docker compose exec -it shell /bin/bash
```
in which you can manually run all the tests, run individual tests, etc.


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

Note that in the devshell and test docker environments above, database migrations are automatically run when you create the environment with `docker compose up -d ...`, so there is no need for an initial `alembic upgrade head`.   However, if you then create additional migrations, and you haven't since run `docker compose down -v` (the `-v` being the thing that deletes the database), then you will need to run `alembic upgrade head` to apply those migrations to the running database inside your docker environment.

### Installing SeeChange on a local machine (not dockerized)

**WARNING:** This section is no longer complete.  What's described here will get you partway there.  However, the tests now require a number of external servers to be running, and setting all of them up is not documented here.  The best way to get everything set up as necessary to run all of the code, and all of the tests, is to use a dockerized environment as described above.

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

This covers (most of) the basic python dependencies. 

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
