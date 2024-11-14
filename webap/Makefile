INSTALLDIR = install

toinstall = seechange_webap.py rkauth_flask.py run_webap.sh \
	static/seechange.css static/seechange.js static/seechange_start.js \
	static/rkwebutil.js static/rkauth.js static/resetpasswd_start.js \
	static/favicon.ico \
	templates/base.html templates/seechange_webap.html

.PHONY: install

install: $(patsubst %, $(INSTALLDIR)/%, $(toinstall))

$(INSTALLDIR)/%: %
	install -Dp $< $@

rkauth_flask.py: rkwebutil/rkauth_flask.py
	ln -s rkwebutil/rkauth_flask.py rkauth_flask.py

static/rkwebutil.js: rkwebutil/rkwebutil.js
	ln -s ../rkwebutil/rkwebutil.js static/rkwebutil.js

static/rkauth.js: rkwebutil/rkauth.js
	ln -s ../rkwebutil/rkauth.js static/rkauth.js

static/resetpasswd_start.js: rkwebutil/resetpasswd_start.js
	ln -s ../rkwebutil/resetpasswd_start.js static/resetpasswd_start.js
