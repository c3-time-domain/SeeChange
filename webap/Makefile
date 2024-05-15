INSTALLDIR = install

#PG_HOST = placeholder
#PG_PORT = placeholder
#PG_USER = placeholder
#PG_PASS = placeholder
#PG_NAME = placeholder
#ARCHIVE_DIR = placeholder

toinstall = seechange_webap.py \
	static/seechange.css static/seechange.js static/seechange_start.js static/rkwebutil.js static/favicon.ico \
	templates/base.html templates/seechange_webap.html
#	seechange_webap_config.py

.PHONY: install # seechange_webap_config.py

install: $(patsubst %, $(INSTALLDIR)/%, $(toinstall))

$(INSTALLDIR)/%: %
	install -Dp $< $@

#seechange_webap_config.py:
#	echo import pathlib > seechange_webap_config.py
#	echo PG_HOST = \"$(PG_HOST)\" >> seechange_webap_config.py
#	echo PG_PORT = $(PG_PORT) >> seechange_webap_config.py
#	echo PG_USER = \"$(PG_USER)\" >> seechange_webap_config.py
#	echo PG_PASS = \"$(PG_PASS)\" >> seechange_webap_config.py
#	echo PG_NAME = \"$(PG_NAME)\" >> seechange_webap_config.py
#	echo ARCHIVE_DIR = pathlib.Path\( \"$(ARCHIVE_DIR)\" \) >> seechange_webap_config.py

static/rkwebutil.js: rkwebutil/rkwebutil.js
	ln -s ../rkwebutil/rkwebutil.js static/rkwebutil.js
