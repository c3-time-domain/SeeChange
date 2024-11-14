import pathlib
PG_HOST = 'postgres'
PG_PORT = 5432
PG_USER = 'postgres'
PG_PASS = 'fragile'
PG_NAME = 'seechange'
ARCHIVE_DIR = pathlib.Path( '/archive-storage/base/test' )
FLASK_SECRET_KEY = 'xcy8cmyk87dgznom6locs159igd66mjbbcuehnfcjtk9khccj4z2np4tk52qxwj4'
WEBAP_CONFIG = { 'email_from': 'Seechange <nobody@nowhere.org>',
                 'email_subject': 'Seechange password reset',
                 'email_system_name': 'Seechange',
                 'smtp_server': 'mailhog',
                 'smtp_port': 1025,
                 'smtp_use_ssl': False,
                 'smtp_username': None,
                 'smtp_password': None }
