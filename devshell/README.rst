The purpose of this directory is to create a self-contained dockerized with enough services running that you can run full-cycle tests.  It is used by the tests in the ../tests directory.  **Most** of the images are defined in subdirectories under ``../docker`` (while ``mailhog`` just uses the standard ``mailhog`` image).

You bring the environment up with:

.. code-block:: bash

   docker compose up -d shell

If all is well, it will tell you that a whole bunch of containers are ``Created``.  If you get any errors, then you have more work to do.  (TODO: document this.)

You can then connect to a shell in the running container (technically, one of the running containers; see below) with:

.. code-block:: bash

   docker compose exec -it shell /usr/bin/bash

You can tell that you're in the container because your prompt will change to something like ``root@ec98e49b45c2:/seechange#``, where the hex barf after ``root@`` will be different every time.  You can get out of the container and back to your host system with ``exit``, just like exiting any other shell.

When you're done, you can remove all the running containers with:

.. code-block:: bash

   docker compose down -v

That will wipe out the database and the archive.  However, any data files that you downloaded will (unless you played with the configuratoin) still exist in ``tests/test_tempdata`` and ``tests/test_filestore`` (underneath the top level of your SeeChange checkout).


Containers Created
------------------

When it's running, the hosts are available within the docker environment:

* shell : a bash shell with various directories mounted (see below)
* webap : a web server running the SeeChange webap
* postgres : a postgres database server
* archive : an archive server
* mailhog : a simple SMTP server (used for testing password reset)
* kafka-server : apache kafka server for sending alerts to


Filesystems available in shell
-------------------------------

Inside the ``shell`` container, the following paths are defined:

* ``/seechange`` : The root of your SeeChange git checkout
* ``/archive-storage`` : The directory that the archive server reads and write sfrom

Poking at the database
----------------------

If you want to look directly at the PostgreSQL database, you can connect to host ``postgres`` with user ``postgres`` and password ``fragile``.  The relevant database is ``seechange``.  With psql, this would be:

.. code-block:: bash

   PGPASSWORD=fragile psql -h postgres -U postgres seechange

Alternate things you can bring up
----------------------------------

TODO
