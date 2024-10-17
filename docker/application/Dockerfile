# This dockerfile is complicated.  It has lots of targets, many of which
# are build targets.  They are there so that the ~0.5GB of code needed
# for building (including python pip) doesn't have to be included in the
# final image.
#
# The docker-compose.yaml files in tests and devshell each do a few
# different builds from this Dockerfile.
#
# There's also two (well, four) different versions of the image.  There
# are "bindmount_code" and "included_code" images; "bindmount" images
# expect there to be a bind-mounted directory with the actual SeeChange
# code.  "included" images run a make in order to install the SeeChange
# code inside the image.  For a production environment, you'd probably
# want to use the "included_code" version, but for development and tests
# you want to use "bindmount" so you don't have to rebuild the docker
# image every time you edit a line of code.
#
# The targets with selenium included (the "test_*" environments) add
# ~0.7GB to the image because web browsers are gigantic unwieldy beasts.
# We need that for our tests, but it is not needed for a production
# environment, and (may) not be needed for a dev environment (depending
# on what you're working on).
#
# The targets in this Dockerfile
#
# base
#   build
#     included_code_build
#       conductor_build
#     build_selenium
#   included_code
#     conductor
#   bindmount_code
#     conductor_bindmount
#   base_selenium
#     test_included    ** Image used in tests to run the tests
#     test_bindmount
#
# As of 2024-06-10, the sizes of images from the various stages are as follows:
#   base                1.18GB
#   build               2.82GB
#   included_code_build 2.82GB
#   conductor_build     2.82GB
#   build_selenium      2.89GB
#   included_code       2.06GB
#   conductor           2.06GB
#   bindmount_code      2.06GB
#   conductor_bindmount 2.06GB
#   base_selenium       1.80GB
#   test_bindmount      2.73GB
#   test_included       2.73GB
#
# The "smallest usable image" is bindmount_code, as it includes all the
# pip requirements, but does not include firefox or selenium.
#
# (Note that installing chromium instead of firefox into base_selenium
# changes the image size by only 20MB.)

# ======================================================================
# The base target defines the base system that all other images
#   will be built on top of.

FROM rknop/devuan-daedalus-rknop AS base

LABEL maintainer="Rob Knop <raknop@lbl.gov>"

SHELL ["/bin/bash", "-c"]

# Note: the purge of some pythons below is because there's a
#  version conflict with stuff installed later as a dependency by pip3.
#  (Gotta love duelling package managers.)

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y upgrade && \
    DEBIAN_FRONTEND="nonintearctive" TZ="UTC" apt-get -y install -y \
      lsb-release \
      curl \
      emacs-nox \
      fitsverify \
      flex \
      git \
      imagemagick \
      libcfitsio-bin \
      libfftw3-bin \
      liblapack3 \
      locales \
      missfits \
      netcat-openbsd \
      postgresql-client \
      procps \
      psfex \
      python3 \
      scamp \
      source-extractor \
      swarp \
      tzdata \
      unzip \
      vim-nox \
      wcslib-dev \
      wcslib-tools \
      wcstools \
      wget  \
    && apt-get -y purge python3-cffi-backend python3-requests python3-dateutil && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# We want to be able to just run python

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Make the sanitized Debian install of sextractor compatible with standard sextractor
RUN ln -sf /usr/bin/source-extractor /usr/bin/sex

# Also fix Debian's ideosyncratic capitalization of swarp
RUN ln -sf /usr/bin/SWarp /usr/bin/swarp

# Create a home directory so that packages can create
# astromatic files if needed.
RUN mkdir /home/seechange
RUN mkdir /home/seechange/.astropy
ENV HOME=/home/seechange
RUN chmod -R 777 /home/seechange

RUN cat /etc/locale.gen | perl -pe 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' > /etc/locale.gen.new \
    && mv /etc/locale.gen.new /etc/locale.gen
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

ENV PYTHONSTARTUP=""
ENV PYTHONNOUSERSITE="1"
ENV PYTHONUSERBASE="/tmp"
ENV PYTHONPATH=""

# Some final setups for sanity
ENV LESS=-XLRi

# ======================================================================
# Use a multistage Docker file so that we can install the packages
#   needed for compilation and the like while building things,
#   but don't include those packages in the final Docker image.
#   This saves about ~0.5GB from the final Docker image.
#   (Alas, the pip requirements eat up ~1GB, and there's not a lot we
#   can do about that because a big fraction of that is stuff like
#   scipy, astropy.)

FROM base AS build

RUN apt-get update && \
    DEBIAN_FRONTEND="nonintearctive" TZ="UTC" apt-get -y install -y \
      autoconf \
      automake \
      build-essential \
      cmake \
      gdb \
      gfortran \
      git \
      libcairo2-dev \
      libcfitsio-dev \
      libcurl4-openssl-dev \
      libfftw3-dev \
      libgsl-dev \
      liblapacke-dev \
      libopenblas-openmp-dev \
      libpq-dev \
      libssl-dev \
      libtool \
      m4 \
      pkg-config \
      python3-ipykernel \
      python3-pip \
      python3-venv \
    && apt-get -y purge python3-cffi-backend python3-requests python3-dateutil && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#      libatlas-base-dev \

RUN mkdir /usr/src/seechange
WORKDIR /usr/src/seechange

# Copy all patch files to current working directory
RUN mkdir ./rules
ADD docker/application/patches/patch_* ./rules/

## NOT INCLUDING MPI FOR NOW SINCE ITS HUGE AND WE DON'T CURRENTLY USE IT
## TODO : if we ever uncomment this, need to edit the other stages
##   below to properly copy all the mpich stuff
#
# Need to install mpich here rather than via package manager to ensure
# ABI compatibility.
# ARG mpich_version=4.0.2
# ARG mpich_prefix=mpich-$mpich_version
# RUN curl -L https://www.mpich.org/static/downloads/$mpich_version/$mpich_prefix.tar.gz -O \
#     && tar xf $mpich_prefix.tar.gz \
#     && cd $mpich_prefix \
#     && ./configure FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch \
#     && make -j 16 \
#     && make install \
#     && make clean \
#     && cd .. \
#     && rm -rf $mpich_prefix $mpich_prefix.tar.gz

# Hotpants Alard/Lupton image subtraction
RUN git clone https://github.com/acbecker/hotpants.git \
&& cd hotpants \
    && patch < /usr/src/seechange/rules/patch_hotpants \
    && make -j8 CFITSIOINCDIR=/usr/include CFITSIOLIBDIR=/usr/lib \
    && cp hotpants /usr/bin \
    && cd .. \
    && rm -rf hotpants

RUN mkdir /venv
RUN python3 -mvenv /venv

COPY requirements.txt /seechange/requirements.txt

RUN source /venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install  -r /seechange/requirements.txt

# Pytorch
RUN source /venv/bin/activate \
    && uv pip install torch==2.3.0+cpu --find-links https://download.pytorch.org/whl/torch_stable.html

# RBbot
# Remove the requirements.txt file because it will install
# a GPU version of pytorch, and we've already installed
# a CPU version above.
COPY extern/RBbot_inference /usr/src/seechange/RBbot_inference
RUN cd RBbot_inference \
    && rm -f requirements.txt \
    && touch requirements.txt \
    && source /venv/bin/activate \
    && pip install . \
    && cd .. \
    && rm -rf RBbot_inference


# ======================================================================
# A small addendum to build that adds selenium-firefox to /venv
# (I hope it works to pip install selenium-firefox when firefox
# itself is not installed, since firefox isn't in build.)

FROM build AS build_selenium

RUN source /venv/bin/activate \
    && pip install selenium-firefox==2.0.8

# ======================================================================
# This target is for running the "make install" to install
#   the SeeChange code inside the image.  It needs to build
#   off of the build target so all the autotools stuff is
#   there, but we don't want the autotools stuff
#   in the final image.
#
# Right now, this reruns every time you touch a file in this
#  directory.  Could be made faster by moving the data
#  and cache directories out of this tree, though that might
#  require thinking about directory config for tests.  (The
#  goal of this would be to make the ADD . have less to import.)
# (Acutally, it seems to rerun every time...?  Thought required.)

FROM build AS included_code_build

RUN mkdir -p /usr/src/seechange/seechange

# Need to make sure that everything necessary for the SeeChange make
#  install is loaded here.  Don't just ADD . because every time any file
#  anywhere in the tree got touched it would trigger a rebuild of this
#  step; also, that is fairly slow because there may be a lot of
#  gratuitous cached test data in the tree that we really don't need
#  docker to be thinking about.  (This step will still get redone a lot
#  anyway because we will regularly be editing things in the improc,
#  models, pipeline, and util subdirectories.  Fortunately, it should be
#  a pretty fast step.)  (The one is the longest and most wasteful, and
#  is just there so that the creation of util/githash.py works in
#  util/Makefile.am.)
ADD .git /usr/src/seechange/seechange/.git
ADD configure.ac /usr/src/seechange/seechange/configure.ac
ADD Makefile.am /usr/src/seechange/seechange/Makefile.am
ADD requirements.txt /usr/src/seechange/seechange/requirements.txt
ADD improc/ /usr/src/seechange/seechange/improc/
ADD models/ /usr/src/seechange/seechange/models/
ADD pipeline/ /usr/src/seechange/seechange/pipeline/
ADD util/ /usr/src/seechange/seechange/util/
ADD share/ /usr/src/seechange/seechange/share/
# archive.py needs special treatment because it's a symbolic link in the repo
ADD util/archive.py /usr/src/seechange/seechange/util/archive.py
# special treatment for share/RBbot_inference/models because of sybmolic links
ADD share/RBbot_models/* /usr/src/seechange/seechange/share/RBbot_models/

WORKDIR /usr/src/seechange/seechange
RUN autoreconf --install
RUN ./configure --with-installdir=/seechange/lib
RUN make install

# ======================================================================
# This is the target used to do a make install for the conductor's
#   specific webap code.

FROM included_code_build AS conductor_build

RUN mkdir -p /usr/src/seechange/conductor
ADD conductor/ /usr/src/seechange/conductor/
# Need special handling of symlinks, because the
#  ADD above copied the links as is, but they point
#  to something that won't exist at the destination.
ADD conductor/rkauth_flask.py /usr/src/seechange/conductor/rkauth_flask.py
ADD conductor/static/rkwebutil.js /usr/src/seechange/conductor/static/rkwebutil.js
ADD conductor/static/rkauth.js /usr/src/seechange/conductor/static/rkauth.js
ADD conductor/static/seechange.css /usr/src/seechange/conductor/static/seechange.css
ADD conductor/static/resetpasswd_start.js /usr/src/seechange/conductor/static/resetpasswd_start.js
WORKDIR /usr/src/seechange/conductor

RUN make INSTALLDIR=/webservice_code install

# ======================================================================
# This target is for an image that will not include any
#  of the SeeChange code inside the image, but will need
#  to bind-mount it from the host.

FROM base AS bindmount_code

COPY --from=build /usr/bin/hotpants /usr/bin/hotpants
COPY --from=build /venv/ /venv/
ENV PATH=/venv/bin:$PATH

ENV PYTHONPATH="/seechange"
WORKDIR /seechange

# A gratuitous command so that the container will persist
CMD ["tail", "-f", "/etc/issue"]

# =====================================================================
# This is the target for an image that includes the SeeChange code.

FROM base AS included_code

COPY --from=build /usr/bin/hotpants /usr/bin/hotpants
COPY --from=build /venv/ /venv/
ENV PATH=/venv/bin:$PATH

COPY --from=included_code_build /seechange/lib /seechange/lib
COPY default_config.yaml /seechange/default_config.yaml
COPY docker/application/local_overrides.yaml /seechange/local_overrides.yaml
ENV SEECHANGE_CONFIG=/seechange/default_config.yaml
ENV PYTHONPATH="/seechange/lib"

RUN mkdir /seechange/data
RUN mkdir /seechange/temp

WORKDIR /seechange

# A gratuitous command so that the container will persist
CMD ["tail", "-f", "/etc/issue"]


# ======================================================================
# This is the target for the conductor as used in production.

FROM included_code AS conductor

COPY --from=conductor_build /webservice_code/ /webservice_code/
WORKDIR /webservice_code/
RUN chmod 755 run_conductor.sh

RUN mkdir /sessions
RUN chmod 777 /sessions

ADD conductor/seechange_conductor.yaml /seechange/seechange_conductor.yaml
ADD conductor/local_overrides.yaml /seechange/local_overrides.yaml
ADD conductor/local_augments.yaml /seechange/local_augments.yaml

# Copy in the self-signed certificate we need to use in tests.
# This should *not* be used in production, even though it's
# sitting there in the image.
COPY docker/application/conductor_bogus_key.pem /webservice_code/
COPY docker/application/conductor_bogus_cert.pem /webservice_code/
RUN chmod a+r /webservice_code/conductor*pem

ENV SEECHANGE_CONFIG=/seechange/seechange_conductor.yaml
ENV PYTHONPATH="/seechange/lib"

CMD [ "./run_conductor.sh", "8080" ]

# ======================================================================
# This is the target for the conductor as used in tests It expects to
# bind-mount its code at /webservice_code The only diference from the
# conductor target is that it doesn't COPY the conductor code from the
# conductor_build target.

FROM bindmount_code AS conductor_bindmount

RUN mkdir /webservice_code
WORKDIR /webservice_code

RUN mkdir /sessions
RUN chmod 777 /sessions

ADD conductor/seechange_conductor.yaml /seechange/seechange_conductor.yaml
ADD conductor/local_overrides.yaml /seechange/local_overrides.yaml
ADD conductor/local_augments.yaml /seechange/local_augments.yaml
ENV SEECHANGE_CONFIG=/seechange/seechange_conductor.yaml
ENV PYTHONPATH="/seechange/lib"

CMD [ "./run_conductor.sh", "8080" ]


# ======================================================================
# This is the base image that includes selenium and a web browser

FROM base AS base_selenium

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get -y upgrade \
    && DEBIAN_FRONTEND="noninteractive" apt-get -y install firefox-esr \
    && apt-get -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ======================================================================
# This is the image that includes a web browser and includes the code
# built into the docker image.  This is what's used in our automatically
# run tests.

FROM base_selenium AS test_included

COPY --from=build_selenium /usr/bin/hotpants /usr/bin/hotpants
COPY --from=build_selenium /venv/ /venv/
ENV PATH=/venv/bin:$PATH

COPY --from=included_code_build /seechange/lib /seechange/lib
COPY default_config.yaml /seechange/default_config.yaml
COPY docker/application/local_overrides.yaml /seechange/local_overrides.yaml
ENV SEECHANGE_CONFIG=/seechange/default_config.yaml
ENV PYTHONPATH="/seechange/lib"

RUN mkdir /seechange/data
RUN mkdir /seechange/temp

ENV PYTHONPATH="/seechange"

# A gratuitous command so that the container will persist
CMD ["tail", "-f", "/etc/issue"]


# ======================================================================
# This is the image that includes a web browser but expects the code
# to be bind-mounted at /seechange.  This is for dev purposes.

FROM base_selenium AS test_bindmount

COPY --from=build_selenium /usr/bin/hotpants /usr/bin/hotpants
COPY --from=build_selenium /venv/ /venv/
ENV PATH=/venv/bin:$PATH

ENV PYTHONPATH="/seechange"
WORKDIR /seechange

# A gratuitous command so that the container will persist
CMD ["tail", "-f", "/etc/issue"]
