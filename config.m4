PHP_ARG_ENABLE(
  llama, dnl extension name
  whether to enable llama embed support, dnl text shown during ./configure
  [  --enable-llama           Enable LlamaEmbed (libllama) support] dnl help text shown in ./configure --help
)

if test "$PHP_LLAMA" != "no"; then

  dnl Debian packages llama.pc under a private multiarch dir
  if test -d /usr/lib/x86_64-linux-gnu/llama/pkgconfig; then
    PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/llama/pkgconfig:${PKG_CONFIG_PATH}"
    export PKG_CONFIG_PATH
  elif test -d /usr/lib/aarch64-linux-gnu/llama/pkgconfig; then
    PKG_CONFIG_PATH="/usr/lib/aarch64-linux-gnu/llama/pkgconfig:${PKG_CONFIG_PATH}"
    export PKG_CONFIG_PATH
  fi

  PKG_CHECK_MODULES([LLAMA], [llama], [], [
    AC_MSG_ERROR([llama.pc not found — install libllama-dev; set PKG_CONFIG_PATH to …/llama/pkgconfig])
  ])
  PHP_EVAL_INCLINE([$LLAMA_CFLAGS])
  PHP_EVAL_LIBLINE([$LLAMA_LIBS], [LLAMA_SHARED_LIBADD])

  dnl compile php_llama.c into a PHP extension named llama
  PHP_NEW_EXTENSION(llama, php_llama.c, $ext_shared)
  dnl exports a shell variable from ./configure into the generated Makefile so the build system knows which library flags to use when compiling
  PHP_SUBST(LLAMA_SHARED_LIBADD)
fi
