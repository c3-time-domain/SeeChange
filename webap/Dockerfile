# To build:
#
# Look at the file seechange_webap_config.py ; make a copy of this
#   to a secrets location, and put in the appropriate database
#   information.  This secrets location must then be bind-mounted
#   to where the webap runs.
#
# docker build -t seechange-webap .
#
# ...you might want to use a name other than "seechange-webap", to
# reflect the specifics of the database you built for.  If you
# omit the variables, it will build with the database suitable
# for the tests environment (in ../tests/docker_compose.yaml).
#
# Run a container built from this image with:
#
#   docker run -d \
#     --mount type=bind,source=<secretsdir>,target=/secrets \
#     --name seechange-webap -p 8080:8080 seechange-webap
#
# ...putting whatever you want after "--name", replacing the
# "seechange-webap" at the end of the line with what you gave the -t
# argument of docker build, and replacing <secretsdir> with the place
# wher eyou wrote the seechange_webap_config.py file.
#
# Thereafer, you can find the webap at localhost:8080
#
# To bind-mount the default install directory for testing purposes, after -d:
#    --mount type=bind,source=$PWD/install,target=/code
#
# You can of course bind mount different directories to which you
# installed the code, if you know what you're doing.
#
# This dockerfile is also used in tests, with appropriate options for the test database and
# opening up port 8080 on the machine you're running the tests from.

FROM rknop/devuan-daedalus-rknop

MAINTAINER Rob Knop <raknop@lbl.gov>

SHELL [ "/bin/bash", "-c" ]

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get -y upgrade \
    && DEBIAN_FRONTEND="noninteractive" TZ="US/Pacific" apt-get -y install -y \
         python3 \
         python3-pip \
         postgresql-client \
         python3-psycopg2 \
         pkg-config \
         libhdf5-dev \
         build-essential \
         netcat-openbsd \
    && apt-get -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN rm /usr/lib/python3.11/EXTERNALLY-MANAGED

RUN pip install gunicorn flask pyyaml numpy pandas matplotlib astropy h5py pillow \
    && rm -rf /.cache/pip

# Todo : multi-stage docker file so that the stuff in /build
#  isn't in any of the saved image layers

# I'm using knowledge of what's in the Makefile
# to construct this...

RUN mkdir /code
RUN mkdir -p /build/seechange_webap/rkwebutil
COPY Makefile seechange_webap.py /build/seechange_webap/
COPY static/ /build/seechange_webap/static/
COPY templates/ /build/seechange_webap/templates/
COPY rkwebutil/rkwebutil.js /build/seechange_webap/rkwebutil/
WORKDIR /build/seechange_webap
RUN make INSTALLDIR=/code install
RUN rm -rf /build/seechange_webap

WORKDIR /code

# TODO : think about -w 4 below
CMD [ "gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "--timeout", "0", "seechange_webap:app" ]
