INSTALLDIR = test_install

toinstall = webservice.py rkauth_flask.py updater.py run_conductor.sh \
	static/conductor.js static/conductor_start.js static/rkwebutil.js \
	static/resetpasswd_start.js static/rkauth.js static/seechange.css \
	templates/base.html templates/conductor_root.html

default :
	@echo Do "make install INSTALLDIR=<dir>"

install : $(patsubst %, $(INSTALLDIR)/%, $(toinstall))

$(INSTALLDIR)/% : %
	install -Dcp $< $@
