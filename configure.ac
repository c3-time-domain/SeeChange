AC_INIT([SeeChange], [0.0.1], [raknop@lbl.gov])
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_FILES([
  Makefile
  improc/Makefile
  models/Makefile
  pipeline/Makefile
  util/Makefile
  ])

AC_ARG_WITH(installdir,
            [AS_HELP_STRING([--with-installdir=DIR], [Where to install [/usr/local/lib/SeeChange]])],
            [installdir=$withval],
            [installdir=/usr/local/lib/SeeChange])

AC_SUBST(installdir)

AC_OUTPUT
