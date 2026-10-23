INSTALLDIR = install

toinstall = seechange_webap.py baseview.py conductor.py ltcv.py updater.py rkauth_flask.py run_webap.sh \
	static/seechange.css static/seechange_ns.js static/seechange.js \
	static/seechange_ltcv_start.js static/seechange_start.js \
	static/provenancetags.js static/exposuresearch.js static/exposurelist.js static/exposure.js \
        static/imagelist.js static/image.js static/ltcv.js static/diaforcedphot.js \
        static/conductor.js \
	static/rkwebutil.js static/svgplot.js static/imview.js static/rkauth.js static/resetpasswd_start.js \
	static/favicon.ico \
	templates/base.html templates/ltcv.html templates/seechange_webap.html

.PHONY: install

install: $(patsubst %, $(INSTALLDIR)/%, $(toinstall))

$(INSTALLDIR)/%: %
	install -Dp $< $@

rkauth_flask.py: ../extern/rkwebutil/rkwebutil/rkauth_flask.py
	ln -s ../extern/rkwebutil/rkwebutil/rkauth_flask.py rkauth_flask.py

static/rkwebutil.js: ../extern/rkwebutil/rkwebutil/static/rkwebutil.js
	ln -s ../../extern/rkwebutil/rkwebutil/static/rkwebutil.js static/rkwebutil.js

static/svgplot.js: ../extern/rkwebutil/rkwebutil/static/svgplot.js
	ln -s ../../extern/rkwebutil/rkwebutil/static/svgplot.js static/svgplot.js

static/imview.js: ../extern/rkwebutil/rkwebutil/static/imview.js
	ln -s ../../extern/rkwebutil/rkwebutil/static/imview.js static/imview.js

static/rkauth.js: ../extern/rkwebutil/rkwebutil/static/rkauth.js
	ln -s ../../extern/rkwebutil/rkwebutil/static/rkauth.js static/rkauth.js

static/resetpasswd_start.js: ../extern/rkwebutil/rkwebutil/static/resetpasswd_start.js
	ln -s ../../extern/rkwebutil/rkwebutil/static/resetpasswd_start.js static/resetpasswd_start.js
