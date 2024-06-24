## Testing

Tests are an integral part of the development process. 
We run mostly unit tests that test specific parts of the code, 
but a few integration tests are also included (end-to-end tests). 
We plan to add some regression tests that verify the results 
of the pipeline are consistent with previous code versions. 

### Running tests

To run the tests, simply run the following command from the root directory of the project:

```bash
pytest
```

To run the tests in a dockerized environment, see the setup.md file, under "Running tests". 

### Test caching and data folders

Some of our tests require large datasets (mostly images). 
We include a few example images in the repo itself, 
but most of the required data is lazy downloaded from 
the appropriate servers (e.g., from Noirlab). 

To avoid downloading the same data over and over again, we cache the
data in the `data/cache` folder.  To make sure the downloading process
works as expected, users can choose to delete this folder. (One may also
need to delete the `tests/temp_data` folder, if tests were interrupted.
Ideally, the tests don't depend on anything specific in there, but there
may be things left behind.)  In the tests, the path to this folder is
given by the `cache_dir` fixture.

Note that the persistent data, that comes with the 
repo, is anything else in the `data` folder, 
which is pointed to by the `persistent_dir` fixture. 

Finally, the working directory for local storage, 
which is referenced by the `FileOnDiskMixin.local_path` 
class variable, is defined in the test config YAML file, 
and can be accessed using the `data_dir` fixture. 
This folder is systematically wiped when the tests
are completed. 



