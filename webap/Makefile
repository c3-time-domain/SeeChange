INSTALLDIR = install

toinstall = seechange_webap.py baseview.py conductor.py updater.py rkauth_flask.py run_webap.sh \
	static/seechange.css static/seechange_ns.js static/seechange.js static/seechange_start.js \
	static/provenancetags.js static/exposuresearch.js static/exposurelist.js static/exposure.js \
	static/conductor.js \
	static/rkwebutil.js static/rkauth.js static/resetpasswd_start.js \
	static/favicon.ico \
	templates/base.html templates/seechange_webap.html

.PHONY: install

install: $(patsubst %, $(INSTALLDIR)/%, $(toinstall))

$(INSTALLDIR)/%: %
	install -Dp $< $@

rkauth_flask.py: ../extern/rkwebutil/rkauth_flask.py
	ln -s ../extern/rkwebutil/rkauth_flask.py rkauth_flask.py

static/rkwebutil.js: ../extern/rkwebutil/rkwebutil.js
	ln -s ../../extern/rkwebutil/rkwebutil.js static/rkwebutil.js

static/rkauth.js: ../extern/rkwebutil/rkauth.js
	ln -s ../../extern/rkwebutil/rkauth.js static/rkauth.js

static/resetpasswd_start.js: ../extern/rkwebutil/resetpasswd_start.js
	ln -s ../../extern/rkwebutil/resetpasswd_start.js static/resetpasswd_start.js
