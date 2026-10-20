dnl Copyright (c) 2026 Roo J Solutions Ltd.
dnl Author: Leon <leon@roojs.com>
dnl Licensed under the PHP License, version 3.01. See LICENSE.

ext_shared=yes

dnl Debian packages llama.pc under a private multiarch dir. amd64 only.
if test -d /usr/lib/x86_64-linux-gnu/llama/pkgconfig; then
  PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/llama/pkgconfig:${PKG_CONFIG_PATH}"
  export PKG_CONFIG_PATH
fi

PKG_CHECK_MODULES([LLAMA], [llama], [], [
  AC_MSG_ERROR([llama.pc not found — install libllama-dev; set PKG_CONFIG_PATH to …/llama/pkgconfig])
])
PHP_EVAL_INCLINE([$LLAMA_CFLAGS])
PHP_EVAL_LIBLINE([$LLAMA_LIBS], [LLAMA_SHARED_LIBADD])

PHP_NEW_EXTENSION(llamacpp, php_llamacpp.c, yes)
PHP_SUBST(LLAMA_SHARED_LIBADD)
