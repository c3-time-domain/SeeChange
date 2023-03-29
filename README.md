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

### Tests

To run the tests on your local system in an environment that approximates how they'll be run on github, cd into `tests` and run
   `docker compose build`
   `docker compose run runtests`
(You must have the "docker compose CLI plugin" installed for this to work.)  At the end, `echo $?`; if 0, that's a pass, if 1 (or anything else not 0), that's a fail.  (The output you see to the screen should tell you the same information.)  This will take a long time the first time you do it, as it has to build the docker images, but after that, it should be fast (unless the Dockerfile has changed for either image).

After the test is complete, run
    `docker compose down`
(otherwise, the postgres container will still be running).
