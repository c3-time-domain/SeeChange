INSTALLDIR = install

toinstall = seechange_webap.py \
	static/seechange.css static/seechange.js static/seechange_start.js static/rkwebutil.js static/favicon.ico \
	templates/base.html templates/seechange_webap.html
#	seechange_webap_config.py

.PHONY: install # seechange_webap_config.py

install: $(patsubst %, $(INSTALLDIR)/%, $(toinstall))

$(INSTALLDIR)/%: %
	install -Dp $< $@

static/rkwebutil.js: rkwebutil/rkwebutil.js
	ln -s ../rkwebutil/rkwebutil.js static/rkwebutil.js
