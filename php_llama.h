#ifndef PHP_LLAMA_H
#define PHP_LLAMA_H

extern zend_module_entry llama_module_entry;
#define phpext_llama_ptr &llama_module_entry

#define PHP_LLAMA_VERSION "0.1.0"

#ifdef PHP_WIN32
# define PHP_LLAMA_API __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
# define PHP_LLAMA_API __attribute__ ((visibility("default")))
#else
# define PHP_LLAMA_API
#endif

#ifdef ZTS
#include "TSRM.h"
#endif

#include <llama.h>

typedef struct {
	struct llama_model *model;
	struct llama_context *ctx;
	int32_t dimension;
	zend_object std;
} php_llama_embed_object;

extern zend_class_entry *php_llama_embed_ce;

PHP_MINIT_FUNCTION(llama);
PHP_MSHUTDOWN_FUNCTION(llama);
PHP_RINIT_FUNCTION(llama);
PHP_RSHUTDOWN_FUNCTION(llama);
PHP_MINFO_FUNCTION(llama);

#endif
