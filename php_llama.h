/*
   +----------------------------------------------------------------------+
   | Copyright (c) 2026 Roo J Solutions Ltd.                              |
   +----------------------------------------------------------------------+
   | This source file is subject to version 3.01 of the PHP license,      |
   | that is bundled with this package in the file LICENSE, and is        |
   | available through the world-wide-web at the following url:           |
   | https://www.php.net/license/3_01.txt                                 |
   | If you did not receive a copy of the PHP license and are unable to   |
   | obtain it through the world-wide-web, please send a note to          |
   | license@php.net so we can mail you a copy immediately.               |
   +----------------------------------------------------------------------+
   | Author: Leon <leon@roojs.com>                                        |
   +----------------------------------------------------------------------+
*/

#ifndef PHP_LLAMA_H
#define PHP_LLAMA_H

extern zend_module_entry llama_module_entry;
#define phpext_llama_ptr &llama_module_entry

#define PHP_LLAMA_VERSION "0.1.0"

#include <llama.h>

typedef struct {
	struct llama_model *model;
	struct llama_context *ctx;
	int32_t dimension;
	int debug;
	zend_object std;
} php_llama_embed_object;

extern zend_class_entry *php_llama_embed_ce;

PHP_MINIT_FUNCTION(llama);
PHP_MSHUTDOWN_FUNCTION(llama);
PHP_MINFO_FUNCTION(llama);

#endif
