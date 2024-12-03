## Testing

Tests are an integral part of the development process. 
We run mostly unit tests that test specific parts of the code, 
but a few integration tests are also included (end-to-end tests). 
We plan to add some regression tests that verify the results 
of the pipeline are consistent with previous code versions. 

### Running tests

If you are in an environment that has all of the SeeChange prerequisites, you can run the tests by simply running the following command from the root directory of the project:

```bash
pytest
```

The tests have a lot of infrastructure necessary to run, however.  If you really know what you're doing, you may be able to set up the full environment.  However, most users will find it easier to use the dockerized environment designed to run with our tests.  See "Setting up a SeeChange instance" for more information.

### Testing tips

#### Database deadlocks

(This will only work when testing on your local machine; you won't be able to use this procedure if you see a deadlock on github actions.)  If your tests seem to just freeze up, it's possible you've hit a database deadlock, where two processes are waiting for the same table lock.  To see if this is the case, use `psql` to connect to your database server; if you're using either the devshell or the test docker environments, from a machine inside that environment run

```psql -h postgres -U postgres seechange````

and enter the database password (`fragile`).  Then run:

```  SELECT pid,usename,pg_blocking_pids(pid) as blocked_by,query as blocked_query
     FROM pg_stat_activity WHERE cardinality(pg_blocking_pids(pid))>0;
```

If you get any results, it means there's a database lock.  To be able to go on with your life, look at the number in the `blocked_by` column and run

```SELECT pg_terminate_backend(<number>)```

That will allow things to continue, though of course tests will fail.

The next task is figuring out where the database deadlock came from and fixing it....

#### Files left over in database / archive / disk at end of tests

The tests are supposed to clean up after themselves, so at the end of a test run there should be nothing left in the database or on the archive.  (There are some exceptions of things allowed to linger.)  If things are found at the end of the tests, this will raise errors.  Unfortunately, these errors can hide the real errors you had in your test (which may also be the reasons things were left behind!)  When debugging, you often want to turn off the check that things are left over at the end, so you can see the real errors you're getting.  Edit `tests/fixtures/conftest.py` and set the variable `verify_archive_database_empty` to `False`.  (Remember to set it back to `True` before pushing your final commit for a PR, to re-enable the leftover file tests!)

#### Test caching and data folders

Some of our tests require large datasets (mostly images).  We include a few example images in the repo itself, but most of the required data is lazy downloaded from the appropriate servers (e.g., from Noirlab).

To avoid downloading the same data over and over again, we cache the data in the `data/cache` folder.  To make sure the downloading process works as expected, users can choose to delete this folder.  Sometimes, also, tests may fail because things have changed, but there are older versions left behind in the cache; in this case, clearing out the cache directory will also solve the problem.  (One may also need to delete the `tests/temp_data` folder, if tests were interrupted.  Ideally, the tests don't depend on anything specific in there, but there may be things left behind.)  In the tests, the path to this folder is given by the `cache_dir` fixture.

Note that the persistent data, that comes with the repo, is anything else in the `data` folder, which is pointed to by the `persistent_dir` fixture.

Finally, the working directory for local storage, which is referenced by the `FileOnDiskMixin.local_path` class variable, is defined in the test config YAML file, and can be accessed using the `data_dir` fixture.  This folder is systematically wiped when the tests are completed.

### Running tests on github actions

In the https://github.com/c3-time-domain/SeeChange repository, we have set up some github actions to automatically run tests upon pushes to main (which should never happen) and upon pull requests.  These tests run from the files in `tests/docker-compose.yaml`.  For them to run, some docker images must exist in the github repository at `ghcr.io/c3-time-domain`.  As of this writing, the images needed are:
* `ghcr.io/c3-time-domain/upload-connector:[tag]`
* `ghcr.io/c3-time-domain/postgres:[tag]`
* `ghcr.io/c3-time-domain/seechange:[tag]`
* `ghcr.io/c3-time-domain/conductor:[tag]`
* `ghcr.io/c3-time-domain/seechange-webap:[tag]`

where `[tag]` is the current tag expected by that compose file.You can figure out what this should be by finding a line in `tests/docker-compose.yaml` like:
```
    image: ghcr.io/${GITHUB_REPOSITORY_OWNER:-c3-time-domain}/seechange:${IMGTAG:-tests20240628}
```
The thing in between `${IMTAG:-` and `}` is the tag— in this example, `tests20240628`.

(If you look at `docker-compose.yaml`, you will also see that it depends on the `mailhog`, image, but this is a standard image that can be pulled from `docker.io`, so you don't need to worry about it.)

Normally, all of these image should already be present in the `ghcr.io` archive, and you don't need to do anything.  However, if they aren't there, you need to build and push them.

To build and push the docker images, in the `tests` subdirectory run:
```
   IMGTAG=[tag] docker compose build
```
where [tag] is exactly what you see in the `docker-compose.yaml` file, followed by:
```
   docker push ghcr.io/c3-time-domain/upload-connector:[tag]
   docker push ghcr.io/c3-time-domain/postgres:[tag]
   docker push ghcr.io/c3-time-domain/seechange:[tag]
   docker push ghcr.io/c3-time-domain/conductor:[tag]
   docker push ghcr.io/c3-time-domain/seechange-webap:[tag]
   docker push ghcr.io/c3-time-domain/kafka:[tag]
```

For this push to work, you must have the requisite permissions on the `c3-time-domain` organizaton at github.

### When making changes to dockerfiles or pip requirements

This set of docker images depend on the following files:
* `docker/application/*`
* `docker/postgres/*`
* `webap/*`
* `requirements.txt`

If you change any of those files, you will need to build and push new docker images.  Before doing that, edit `tests/docker-compose.yaml` and bump the date part of the tag for _every_ image (search and replace is your friend), so that your changed images will only get used for your branch while you're still finalizing your pull request, and so that the updated images will get used by everybody else once your branch has been merged to main.
