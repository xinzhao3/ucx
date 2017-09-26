#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

#
# Check for CUDA support
#
cuda_happy="no"
gdrcopy_happy="no"

AC_ARG_WITH([cuda],
           [AS_HELP_STRING([--with-cuda=(DIR)], [Enable the use of CUDA (default is no).])],
           [], [with_cuda=no])

AS_IF([test "x$with_cuda" != "xno"],
        [AS_IF([test ! -z "$with_cuda" -a "x$with_cuda" != "xyes"],
            [
            ucx_check_cuda_dir="$with_cuda"
            ucx_check_cuda_libdir="$with_cuda/lib64 "
            ])
        AS_IF([test ! -z "$with_cuda_libdir" -a "x$with_cuda_libdir" != "xyes"],
            [ucx_check_cuda_libdir="$with_nccl_libdir"])

        AC_CHECK_HEADERS([cuda.h cuda_runtime.h],
                       [AC_CHECK_DECLS([cuPointerGetAttribute],
                           [cuda_happy="yes"],
                           [AC_MSG_WARN([CUDA runtime not detected. Disable.])
                            cuda_happy="no"],
                            [#include <cuda.h>])
                           AS_IF([test "x$cuda_happy" == "xyes"],
                            [AC_DEFINE([HAVE_CUDA], 1, [Enable CUDA support])
                            AC_SUBST(CUDA_CPPFLAGS, "-I$ucx_check_cuda_dir/include ")
                            AC_SUBST(CUDA_CFLAGS, "-I$ucx_check_cuda_dir/include ")
                            AC_SUBST(CUDA_LDFLAGS, "-lcudart -lcuda -L$ucs_check_cuda_libdir/ ")
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

      [AS_IF([test "x$cuda_happy" == "xno"],
             [AC_MSG_ERROR([--with-cuda not specified ...])],[:])
        AS_IF([test ! -z "$with_gdrcopy" -a "x$with_gdrcopy" != "xyes"],
            [
            ucx_check_gdrcopy_dir="$with_gdrcopy"
            ucx_check_gdrcopy_libdir="$with_gdrcopy/lib64 "
            ])
        AS_IF([test ! -z "$with_gdrcopy_libdir" -a "x$with_gdrcopy_libdir" != "xyes"],
            [ucx_check_gdrcopy_libdir="$with_nccl_libdir"])

        AC_CHECK_HEADERS([gdrapi.h],
                       [AC_CHECK_DECLS([gdr_pin_buffer],
                           [gdrcopy_happy="yes"],
                           [AC_MSG_WARN([GDR_COPY runtime not detected. Disable.])
                            gdrcopy_happy="no"],
                            [#include <gdrapi.h>])
                           AS_IF([test "x$gdrcopy_happy" == "xyes"],
                            [AC_DEFINE([HAVE_GDR_COPY], 1, [Enable GDR_COPY support])
                            AC_SUBST(GDR_COPY_CPPFLAGS, "-I$ucx_check_gdrcopy_dir/include/ ")
                            AC_SUBST(GDR_COPY_CFLAGS, "-I$ucx_check_gdrcopy_dir/include/ ")
                            AC_SUBST(GDR_COPY_LDFLAGS, "-lgdrapi -L$ucx_check_gdrcopy_dir/lib64")
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
