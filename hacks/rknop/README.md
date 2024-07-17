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
    /global/scratch/users/raknop/seechange-rknop-dev.sif /bin/bash
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
   --env "SEECHANGE_CONFIG=/seechange/hacks/rknop/rknop-dev.yaml" \
   registry.nersc.gov/m4616/seechange:rknop-dev \
   /bin/bash
```   
   

## Setting up the NERSC environment for the demo -- Perlmutter

When `.yaml` files for spin are referenced, they are in the directory SeeChange/spin/rknop-dev

There are assumptions in the `.yaml` files and the notes below that this is Rob doing this.

### Set up the conductor

* Edit `conductor/local_overrides.yaml` and give it contents:
```
conductor:
  conductor_url: https://ls4-conductor-rknop-dev.lbl.gov/
  email_from: 'Seechange conductor ls4 rknop dev <raknop@lbl.gov>'
  email_subject: 'Seechange conductor (ls4 rknop dev) password reset'
  email_system_name: 'Seechange conductor (ls4 rknop dev)'
  smtp_server: smtp.lbl.gov
  smtp_port: 25
  smtp_use_ssl: false
  smtp_username: null
  smtp_password: null

db:
  host: ls4db.lbl.gov
  port: 5432
  database: seechange_rknop_dev
  user: seechange_rknop_dev
  password_file: /secrets/postgres_passwd
```
* Build the docker image with the command below, and push it
```
   docker docker build --target conductor -t registry.nersc.gov/m4616/seechange:conductor-rknop-dev -f docker/application/Dockerfile .
```
* Edit `conductor-secrets.yaml` to put the right postgres password in. (Remember to take it out before committing anything to a git repository!)
* Create the conductor with `conductor-pvc.yaml`, `conductor-secrets.yaml`, and `conductor.yaml`
* Secure DNS name `ls4-conductor-rknop-dev.lbl.gov` as a CNAME to `onductor.ls4-rknop-dev.production.svc.spin.nersc.org`
* Get a SSL cert for `ls4-conductor-rknop-dev.lbl.gov`
* Put the b64 encoded stuff in `conductor-cert.yaml` and apply it
* Uncomment the stuff in `conductor.yaml` and apply it
* Create user rknop manually in the conductor database (until such a time as the conductor has an actual interface for this).
* (Once the ls4-conductor-rknop-dev.lbl.gov address is available.)  Use the web interface to ls4-conductor-rknop-dev.lbl.gov to set rknop's password on the conductor.  (Consider saving the database barf for the public and private keys to avoid having to do this upon database recreation.)

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

conductor:
  password: ...
```
replacing the `...` with the database password, archive token, and conductor password generated above.

### Create data directories

* Make sure `/pscratch/sd/r/raknop/ls4-rknop-dev/data` exists, and that the subdirectories `seechange` and `temp` exist under it.
* TODO : acls

### Get the podman image

* Build the application docker image and push it to `registry.nersc.gov/m4616/seechange:rknop-dev`
* Get it on nersc with `podman-hpc pull registry.nersc.gov/m4616/seechange:rknop-dev`
* Verify it's there with `podman-hpc images` ; make sure in particular that there's a readonly image.

### Running a shell

```
podman-hpc run -it \
  --mount type=bind,source=/global/homes/r/raknop/SeeChange,target=/seechange \
  --mount type=bind,source=/pscratch/sd/r/raknop/ls4-rknop-dev/data,target=/data \
  --mount type=bind,source=/global/homes/r/raknop/secrets,target=/secrets \
  --env "SEECHANGE_CONFIG=/seechange/hacks/rknop/rknop-dev.yaml" \
  registry.nersc.gov/m4616/seechange:rknop-dev \
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

There is a script `import_cosmos1.py` that runs all of this in parallel.  It's poorly named, because it can work on any of the reference fields I have defined for DECAT (COSMOS 1-3 and ELAIS E1-2).