## Setting up the environment for the demo -- Dirac

### Database and Archive

See under Brahms

### Running a shell

```
  apptainer exec --cleanenv \
    --bind /clusterfs/dirac1/raknop/SeeChange:/seechange \
    --bind /global/scratch/users/raknop/seechange:/data \
    --bind /global/home/users/raknop/secrets:/secrets \
    --env "SEECHANGE_CONFIG=/seechange/hacks/rknop/rknop-dev-dirac.yaml" \
    /global/scratch/users/raknop/seechange-mpich.sif /bin/bash
```

## Seeting up the environment for the demo -- Brahms

### Set up the archive : see below under Perlmutter, it's the same archive

### Set up the database machine

Use `decatdb.lbl.gov`.  Database `seechange_rknop_dev`, user `ls4_rknop_dev`.

### Running a shell

```
docker run --user 1000:1000 -it \
   --mount type=bind,source=/home/raknop/SeeChange,target=/seechange \
   --mount type=bind,source=/data/raknop/seechange,target=/data \
   --mount type=bind,source=/home/raknop/secrets,target=/secrets \
   --env "SEECHANGE_CONFIG=/seechange/hacks/rknop/rknop-dev-brahms.yaml" \
   registry.nersc.gov/m4616/seechange:mpich \
   /bin/bash
```   
   

## Setting up the NERSC environment for the demo -- Perlmtuter

When `.yaml` files for spin are referenced, they are in the directory SeeChange/spin/rknop-dev

There are assumptions in the `.yaml` files and the notes below that this is Rob doing this.

### Set up the database machine

* Generate a postgres password and put it after `pgpass:` in `postgres-secrets.yaml`; apply the secrets yaml.  (Remember to remove this password from the file before committing anything to a git repository!)
* Build the docker image and push it to `registry.nersc.gov/m4616/raknop/seechange-postgres`
* Create the postgres volume in Spin using `postgres-pvc.yaml`
* Create the postgres deployment Spin using `postgres.yaml`
* Verify you can connect to it with
```
psql -h postgres-loadbalancer.ls4-rknop-dev.production.svc.spin.nersc.org -U postgres
```
using the password you generated above.

### Set up the archive

* Generate a token for the archive and put that after `base/` in `connector-secrets.yaml`; apply the secrets yaml.  (Remember to remove this token from the file before committing anything to a git repository!)
* Build the docker image with --build-arg="UID=95089" --build-arg="GID=103988"
* Push the docker image to `registry.nersc.gov/m4616/raknop/nersc-upload-connector:raknop`
* Create the archive directory in /global/cfs/cdirs/m4616/archive-rknop-dev
* Make sure the subdirectory `base` under that directory exists.
* Comment out the `host: ls4-rknop-dev-archive.lbl.gov` and `tls:` blocks in `archive.yaml`
* Create the archive with `archive.yaml`
* Secure DNS name `ls4-rknop-dev-archive.lbl.gov` as a CNAME to `archive.ls4-rknop-dev.production.svc.spin.nersc.org`
* Get a SSL cert for `ls4-rknop-dev-archive.lbl.gov`
* Put the b64 encoded stuff in `archive-cert.yaml` and apply it
* Uncomment the stuff in `archive.yaml` and apply it

### Create the secrets config

* Edit `/global/homes/r/raknop/secrets/ls4-rknop-dev.yaml` with contents
```
db:
  password: ...

archive:
  token: ...
```
replacing the `...` with the database password and archive tokens generated above.

### Create data directories

* Make sure `/pscratch/sd/r/raknop/ls4-rknop-dev/data` exists, and that the subdirectories `seechange` and `temp` exist under it.
* TODO : acls

### Get the podman image

* Build the application docker image and push it to `registry.nersc.gov/m4616/raknop/seechange`
* Get it on nersc with `podman-hpc pull registry.nersc.gov/m4616/raknop/seechange`
* Verify it's there with `podman-hpc images` ; make sure in particular that there's a readonly image.

### Running a shell

```
podman-hpc run -it \
  --mount type=bind,source=/global/homes/r/raknop/SeeChange,target=/seechange \
  --mount type=bind,source=/pscratch/sd/r/raknop/ls4-rknop-dev/data,target=/data \
  --mount type=bind,source=/global/homes/r/raknop/secrets,target=/secrets \
  --env "SEECHANGE_CONFIG=/seechange/hacks/rknop/rknop-dev.yaml" \
  registry.nersc.gov/m4616/raknop/seechange \
  /bin/bash
```

Optionally, for importing references, also add
```
  --mount type=bind,source=/global/cfs/cdirs/m937/www/decat/decat/templatecache,target=/refs
```

### Running a shell with shifter

__Don't do this__, it doesn't seem to help.  Use podman.

Pull the image with
```
  shifterimg --user rknop pull docker:rknop/seechange:latest
```
(May need to `shifterimg login` first.)

Run with:
```
   shifter --volume="/global/homes/r/raknop/SeeChange:/seechange;/pscratch/sd/r/raknop/ls4-rknop-dev/data:/data;/global/homes/r/raknop/secrets:/secrets;/global/cfs/cdirs/m937/www/decat/decat/templatecache:/refs" -e SEECHANGE_CONFIG=/seechange/hacks/rknop/rknop-dev.yaml -e TERM=xterm --image=rknop/seechange /bin/bash
```

### Initializing the database

Inside the container, cd to `/seechange` and run
```
alembic upgrade head
```

### Import the references

cd into `/seechange/hacks/rknop`

For each file in `/refs/COSMOS-1-?-templ` (that's 180 files), run
```
   python import_decam_reference.py <img> <wgt> <msk> --hdu 1 -t COSMOS-1 -s <sec>
```
where:
* `<img>` = `/refs/COSMOS-1-<filt>-templ/COSMOS-1-<filt>-templ.<chip>.fits.fz`
* `<wgt>` = `/refs/COSMOS-1-<filt>-templ/COSMOS-1-<filt>-templ.<chip>.weight.fits.fz`
* `<msk>` = `/refs/COSMOS-1-<filt>-templ/COSMOS-1-<filt>-templ.<chip>.bpm.fits.fz`
* `<filt>` is one of `g`, `r`, or `i`
* `<chip>` is a two digit number between 01 and 62
* `<sec>` is the sensor section that goes with the chip; see https://noirlab.edu/science/images/decamorientation-0 (chip numbers are in green, sensor sections are in black)
