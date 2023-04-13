# SeeChange
A time-domain data reduction pipeline (e.g., for handling images->lightcurves) for surveys like DECam and LS4


## Database schema

The database interactions are managed using SQLAlchemy's object-relational model (ORM). 
Each table is mapped to a python-side object, that can be added or deleted using a session object.

It is useful to get familiar with the naming convention for different pieces of data: 

- `Exposure`: a single exposure of a single CCD, linked to raw data on disk. 
- `Image`: a simple image, that has been processed to remove bias, dark, and flat fields.
  Each `Image` is linked to exactly one `Exposure`. 
- `CoaddImage`: a table similar to `Image` (using `ImageMixin`), but linked to a list of `Image`s 
  that were used to produce the coadd.  
- `SubtractionImage`: another table similar to `Image` (using `ImageMixin`), but linked to a new `Image` and 
  a `CoaddImage` that acts as a reference for the subtraction. 
- `Sighting`: a small region on a single image, where something interesting has been detected. 
  Each `Sighting` is linked to exactly one `Image`. The `Sighting` also contains a list of `Cutout` objects,
  a list of `Photometry` objects, and a many-to-one association to an `Object` table.  
- `Object`: a table that contains information about a single astronomical object (real or bogus), 
  such as its RA, Dec, and magnitude. Each `Object` is linked to a list of `Sighting`s. 
- `Cutouts`: contain the small pixel stamps around a point in the sky in one image, linked to a single `Sighting`.
  Will contain links to disk locations of three cutouts: the reference image, the new image, and the difference image.
- `Photometry`: contains measurements made on the information in the `Cutouts` for a single `Sighting`. 
  These include flux+errors, magnitude+errors, centroid positions, spot width, etc. 
- `Provenance`: A table containing the code version and critical parameters that are unique to this version of the data. 
  Each data product above must link back to a provenance row, so we can recreate the conditions that produced this data. 

## Development notes

### Installing Docker

At the moment, some of the things below will not work if you install Docker Desktop.  It has to do with permissions and bind-mounting system volumes; because of how Docker Desktop works, the files inside the container all end up owned as root, not as you, even if they are owned by you on your own filesystem.  Hopefully there's a way to fix this, but in the mean time, install Docker Engine instead of Docker Desktop; instructions are here:

- Installing Docker Engine : https://docs.docker.com/engine/install/
- Setting up rootless mode (so you don't have to sudo everything) : https://docs.docker.com/engine/security/rootless/

### Tests

To run the tests on your local system in an environment that approximates how they'll be run on github, cd into `tests` and run the following command (which requires the "docker compose CLI plugin" installed to work):
```
   export GITHUB_RESPOSITORY_OWNER=<yourname>
   docker compose build
   USERID=<uid> GROUIP=<gid> docker compose run runtests
```
where you replace `<uid>` and `<gid>` with your own userid and groupid; if you don't do this, the tests will run, but various pycache files will get created in your checkout owned by root, which is annoying.  At the end, `echo $?`; if 0, that's a pass, if 1 (or anything else not 0), that's a fail.  (The output you see to the screen should tell you the same information.)  This will take a long time the first time you do it, as it has to build the docker images, but after that, it should be fast (unless the Dockerfile has changed for either image).  The variable GITHUB_RESPOSITORY_OWNER must be set to *something*; it only matters if you try to push or pull the images.  Try setting it to your github username, though if you really want to push and pull you're going to have to look up making tokens on github.  (The docker-compose.yaml file is written to run on github, which is why it includes this variable.)

After the test is complete, run
```
    docker compose down
```
(otherwise, the postgres container will still be running).

### Development shell -- local database

To create a database on your local machine and get a development shell in which to run code, cd into the `devshell` directory and run
```
   USERID=<UID> GROUPID=<GID> docker compose up -d
```
replacing `<UID>` with your uid, and `<GID>` with your GID.  You can avoid typing this all the time by creating a file `.env` in the `devshell` directory with contents
```
  USERID=<UID>
  GROUPID=<GID>
```
again replacing `<UID>` and `<GID>` with the right things.

Once things are started, there will be two containers running, one for your database, one to which you can attach with a shell.  Do
```
   docker ps
```
to see what the created containers are named; in my experience, the shell container is always `devshell-seechange-1`.  You can then get a shell inside that environment with
```
   docker exec -it devshell-seechange-1 /bin/bash
```
and you're in.  (Put in the right name for the container if it's not devshell-seechange-1.)

This docker image bind-mounts your seechange checkout (the parent directory of the `devshell` directory where you're working) at `/seechange`.  That means if you work in that directory, it's the same as working in the checkout.  If you edit something outside of the container, the differences will be immediately available inside the container (since it's the same physical filesystem).  This means there's no need to rebuild the container every time you change any bit of code.

When you're done, exit the container, and run
```
  USERID=<UID> GROUPID=<GID> docker compose down
```
to stop and delete the container images.  (If you created the `.env` file mentioned above, you don't need the USERID and GROUPID definitions, and can just type `docker compose down`.)

The `docker-compose.yaml` file in this directory defines a volume where postgres stores its data.  This means that every time you restart the enviornment, the database will still be as it was before.  This isn't what you want for running tests, but it's often what you want for development.  You can see what volumes are defined with
```
  docker volume list
```
In that list, you should see something that has name `devshell_seechange-postgres-dbdata`.  If you want to wipe this volume out and start with a fresh database, you can run
```
  docker volume rm devshell_seechange-postgres-dbdata
```

### Development shell -- using an external existing database

TBD

### Database migrations

Database migrations are handled with alembic.

If you've just created a database and want to initialize it with all the tables, run
```
  SEECHANGE_CONFIG=<configfile> alembic upgrade head
```

After editing any schema, you have to create new database migrations to apply them.  Do this by running something like:
```
  SEECHANGE_CONFIG=<configfile> alembic revision --autogenerate -m "<put a short comment here>"
```
The comment will go in the filename, so it should really be short.  Look out for any warnings, and review the created migration file before applying it (with `alembic upgrade head`).

*NOTE:* Alembic doesn't properly handle q3c indices when autogenerating migrations. If you add any columns or tables that need to use a q3c index, you will need either to edit the migration by hand, or create a new migration (by running `alembic revision` without `--autogenerate`) and edit it by hand.
