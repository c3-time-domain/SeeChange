To build the nersc-upload-connector image used by seechange-archive.yaml, need to do

  `docker build -t <image> --build-arg "UID=<uid>" --build-arg "gid=<GID>" .`

where <image> is the image name, and <uid> and <gid> are the UID and GID
the container needs to run under.  For instance, for an installation on
nersc spin run by raknop for ls4, I'd use:

* `<image>` = `registry.nersc.gov/m4616/nersc-upload-connector:raknop`
* `uid` = `95089`
* `gid` = `103988`
