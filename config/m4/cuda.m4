#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

#
# Check for CUDA support
#
cuda_happy="no"

AC_ARG_WITH([cuda],
           [AS_HELP_STRING([--with-cuda=(DIR)], [Enable the use of CUDA (default is no).])],
           [], [with_cuda=no])

AS_IF([test "x$with_cuda" != "xno"],

      [AS_IF([test "x$with_cuda" == "x" || test "x$with_cuda" == "xguess" || test "x$with_cuda" == "xyes"],
             [
              AC_MSG_NOTICE([CUDA path was not specified. Guessing ...])
              with_cuda=/usr/local/cuda
              ],
              [:])
        AC_CHECK_HEADERS([$with_cuda/include/cuda.h $with_cuda/include/cuda_runtime.h],
                       [AC_CHECK_DECLS([cuPointerGetAttribute],
                           [cuda_happy="yes"],
                           [AC_MSG_WARN([CUDA runtime not detected. Disable.])
                            cuda_happy="no"],
                            [#include <$with_cuda/include/cuda.h>])
                           AS_IF([test "x$cuda_happy" == "xyes"],
                            [AC_DEFINE([HAVE_CUDA], 1, [Enable CUDA support])
                            AC_SUBST(CUDA_CPPFLAGS, "-I$with_cuda/include/ ")
                            AC_SUBST(CUDA_CFLAGS, "-I$with_cuda/include/ ")
                            AC_SUBST(CUDA_LDFLAGS, "-lcudart -lcuda -L$with_cuda/lib64")
                            CFLAGS="$CFLAGS $CUDA_CFLAGS"
                            CPPFLAGS="$CPPFLAGS $CUDA_CPPFLAGS"
                            LDFLAGS="$LDFLAGS $CUDA_LDFLAGS"],
                        [])],
                       [AC_MSG_WARN([CUDA not found])
                        AC_DEFINE([HAVE_CUDA], [0], [Disable the use of CUDA])])],
      [AC_MSG_WARN([CUDA was explicitly disabled])
      AC_DEFINE([HAVE_CUDA], [0], [Disable the use of CUDA])]
)


AM_CONDITIONAL([HAVE_CUDA], [test "x$cuda_happy" != xno])

AC_ARG_WITH([gdrcopy],
           [AS_HELP_STRING([--with-gdrcopy=(DIR)], [Enable the use of GDR_COPY (default is no).])],
           [], [with_gdrcopy=no])

AS_IF([test "x$with_gdrcopy" != "xno"],

      [AS_IF([test "x$with_gdrcopy" == "x" || test "x$with_gdrcopy" == "xguess" || test "x$with_gdrcopy" == "xyes"],
             [
              AC_MSG_NOTICE([GDR_COPY path was not specified. Guessing ...])
              with_gdrcopy=/usr/local/gdrcopy
              ],
              [:])
        AC_CHECK_HEADERS([$with_gdrcopy/include/gdrapi.h],
                       [AC_CHECK_DECLS([gdr_pin_buffer],
                           [gdrcopy_happy="yes"],
                           [AC_MSG_WARN([GDR_COPY runtime not detected. Disable.])
                            gdrcopy_happy="no"],
                            [#include <$with_gdrcopy/include/gdrapi.h>])
                           AS_IF([test "x$gdrcopy_happy" == "xyes"],
                            [AC_DEFINE([HAVE_GDR_COPY], 1, [Enable GDR_COPY support])
                            AC_SUBST(GDR_COPY_CPPFLAGS, "-I$with_gdrcopy/include/ ")
                            AC_SUBST(GDR_COPY_CFLAGS, "-I$with_gdrcopy/include/ ")
                            AC_SUBST(GDR_COPY_LDFLAGS, "-lgdrapi -L$with_gdrcopy/lib64")
                            CFLAGS="$CFLAGS $GDR_COPY_CFLAGS"
                            CPPFLAGS="$CPPFLAGS $GDR_COPY_CPPFLAGS"
                            LDFLAGS="$LDFLAGS $GDR_COPY_LDFLAGS"],
                        [])],
                       [AC_MSG_WARN([GDR_COPY not found])
                        AC_DEFINE([HAVE_GDR_COPY], [0], [Disable the use of GDR_COPY])])],
      [AC_MSG_WARN([GDR_COPY was explicitly disabled])
      AC_DEFINE([HAVE_GDR_COPY], [0], [Disable the use of GDR_COPY])]
)


AM_CONDITIONAL([HAVE_GDR_COPY], [test "x$gdrcopy_happy" != xno])

AC_DEFINE([HAVE_CUDA_GDR], [1], [Eanble GPU Direct RDMA])]
AM_CONDITIONAL([HAVE_CUDA_GDR], [1])
