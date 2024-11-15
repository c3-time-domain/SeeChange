utildir = @installdir@/util
util_SCRIPTS = __init__.py archive.py cache.py classproperty.py conductor_connector.py config.py \
	exceptions.py githash.py ldac.py logger.py radec.py retrydownload.py rkauth_client.py runner.py util.py

githash.py: FORCE
	echo '__git_hash = "'`git rev-parse HEAD`'"' > githash.py

FORCE:

