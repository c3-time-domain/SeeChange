FROM rknop/devuan-daedalus-rknop

MAINTAINER Rob Knop <raknop@lbl.gov>

SHELL ["/bin/bash", "-c"]

# Note 1: I install libatlas-base-dev, which is the serial version of
#   ATLAS in debian.  This defeats the whole purpose of ATLAS, which is
#   to properly use OMP.  However, ATLAS is very anal about being
#   compiled on the machine where it's going to run, so it can detect
#   and set up timing parameters.  That, unfortunately, then defeats one
#   of the purposes of this Dockerfile, which is to be able to run on a
#   bunch of different machines.  So, I'm punting on really doing ATLAS
#   right here.
# Note 2: the purge of some pythons below is because there's a
#  version confict with stuff installed later as a dependency by pip3.
#  (Gotta love duelling package managers.)

RUN DEBIAN_FRONTEND="nonintearctive" TZ="US/Pacific" \
    apt-get update && \
    apt-get -y upgrade && \
    DEBIAN_FRONTEND="nonintearctive" TZ="US/Pacific" apt-get -y install -y \
      autoconf \
      automake \
      build-essential \
      cmake \
      lsb-release \
      curl \
      emacs-nox \
      fitsverify \
      flex \
      gdb \
      gfortran \
      git \
      imagemagick \
      libatlas-base-dev \
      libcairo2-dev \
      libcfitsio-dev \
      libcfitsio-bin \
      libcurl4-openssl-dev \
      libfftw3-bin \
      libfftw3-dev \
      libgsl-dev \
      liblapack3 \
      liblapacke-dev \
      libopenblas-openmp-dev \
      libpq-dev \
      libssl-dev \
      libtool \
      locales \
      m4 \
      missfits \
      pkg-config \
      postgresql-client \
      procps \
      psfex \
      python3 \
      python3-ipykernel \
      python3-pip \
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
ENV HOME /home/seechange
RUN chmod -R 777 /home/seechange

RUN mkdir /usr/src/seechange
WORKDIR /usr/src/seechange

# Copy all patch files to current working directory
RUN mkdir ./rules
ADD patches/patch_* ./rules/

RUN cat /etc/locale.gen | perl -pe 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' > /etc/locale.gen.new \
    && mv /etc/locale.gen.new /etc/locale.gen
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV PYTHONSTARTUP ""
ENV PYTHONNOUSERSITE "1"
ENV PYTHONUSERBASE "/tmp"
ENV PYTHONPATH ""

# I can't believe I have to do this
RUN echo "import sys; sys.path.append('/usr/lib/python3.9/site-packages')" > /usr/lib/python3/dist-packages/site39.pth

ENV PYTHONPATH "/seechange"

## NOT INCLUDING MPI FOR NOW SINCE ITS HUGE AND WE DON'T CURRENTLY USE IT
#
# # Need to install mpich here rather than via package manager to ensure
# # ABI compatibility.
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

# So, the Debian python now has a thing that tells pip that it really
# shouldn't be mucking about with the directories that apt manages.
# This sounds like a really good idea!  Duelling package managers are a
# nightmare!  But... it breaks the dockerfile.  So, work around it.
RUN rm /usr/lib/python3.11/EXTERNALLY-MANAGED

RUN pip install --upgrade pip setuptools wheel

# Listing all these versions ensures that this is going to fall behind
# and out of date and eventually break.  Versions of packages will no
# longer be available, or will not be compatible with updates to the
# base OS.
#
# Not listing versions ensures that eventually a package will evolve
# to being incompatible with my code.  (This happened to me when
# the default sqlalchemy jumped from 1.4.x to 2.0.x.)
#
# There is no good answer.
#
# NOTE : this current package list is a superset of what Rob had in
# lensgrinder and what Guy put in "requirements.txt" under the top
# level.  I'm guessing the latter came out of skyportal?  We should trim
# this down to what we actually need.  We should also remove things
# that are here only as dependencies of other things (as opposed to things
# we explicitly use) and let pip resolve the dependencies.
#
RUN pip install \
      alembic==1.9.3 \
      astro-datalab==2.20.1 \
      astropy==5.2.1 \
      astroquery==0.4.6 \
      fitsio==0.9.12 \
      flaky>=3.7.0 \
      gitdb==4.0.10 \
      GitPython==3.1.31 \
      healpy==1.16.5 \
      numpy==1.24.2 \
      pandas==1.5.3 \
      photutils==1.6.0 \
      psycopg2==2.9.5 \
      pytest==7.2.2 \
      pytest-timeout==2.1.0 \
      python-dateutil==2.8.2 \
      pytz==2022.7.1 \
      pyyaml==6.0 \
      requests==2.28.2 \
      scipy==1.10.1 \
      sep==1.2.1 \
      simplejson==3.18.3 \
      sqlalchemy==2.0.7 \
      sqlalchemy-utils==0.40.0 \
      urllib3==1.26.15 \
      wget==3.2 \
   && rm -rf /home/seechange/.cache/pip

# Other pip packages I removed from above that were
# previously there in the list, but that as of now
# the codebase doesn't use.  Certainly some of these
# will need to get added back in as we import more
# code from lensgrinder.
#      attrs==22.2.0 \
#      avro==1.11.1 \
#      blosc2==2.0.0 \
#      bs4==0.0.1 \
#      certifi==2022.12.7 \
#      charset-normalizer==3.1.0 \
#      contourpy==1.0.7 \
#      cycler==0.11.0 \
#      Cython==0.29.33 \
#      fastavro==1.7.1 \
#      fonttools==4.39.2 \
#      galsim==2.4.7 \
#      greenlet==2.0.2 \
#      idna==3.4 \
#      image_registration==0.2.6 \
#      importlib-resources==5.12.0 \
#      iniconfig==2.0.0 \
#      kafka-python==2.0.2 \
#      kiwisolver==1.4.4 \
#      matplotlib==3.7.1 \
#      mpi4py==3.1.4 \
#      msgpack==1.0.5 \
#      numexpr==2.8.4 \
#      packaging==23.0 \
#      Pillow==9.4.0 \
#      pluggy==1.0.0 \
#      py-cpuinfo==9.0.0 \
#      pyerfa==2.0.0.2 \
#      pyparsing==2.4.7 \
#      smmap==5.0.0 \
#      sncosmo==2.9.0 \
#      tables==3.7.0 \
#      tensorflow==2.11.0 \
#      tomli==2.0.1 \
#      typing-extensions==4.5.0 \
#      zipp==3.15.0 \

# Some final setups for sanity
ENV LESS -XLRi

# A gratuitous command so that the container will persist
CMD ["/bin/bash"]
