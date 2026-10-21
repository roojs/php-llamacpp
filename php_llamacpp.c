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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "zend_exceptions.h"
#include "php_llamacpp.h"
#include "ext/standard/info.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

zend_class_entry *php_llama_embed_ce;

ZEND_BEGIN_MODULE_GLOBALS(llamacpp)
	zend_long n_gpu_layers;
ZEND_END_MODULE_GLOBALS(llamacpp)

ZEND_DECLARE_MODULE_GLOBALS(llamacpp)

#define LLAMACPP_G(v) ZEND_MODULE_GLOBALS_ACCESSOR(llamacpp, v)

/* 0 = CPU only. -1 = every layer on the GPU. N > 0 = that many layers. */
PHP_INI_BEGIN()
	STD_PHP_INI_ENTRY("llamacpp.n_gpu_layers", "0", PHP_INI_ALL, OnUpdateLong,
		n_gpu_layers, zend_llamacpp_globals, llamacpp_globals)
PHP_INI_END()

static int32_t php_llama_n_gpu_layers(void)
{
	zend_long n = LLAMACPP_G(n_gpu_layers);

	if (n > INT32_MAX) {
		return INT32_MAX;
	}
	if (n < INT32_MIN) {
		return INT32_MIN;
	}
	return (int32_t)n;
}

static zend_object_handlers php_llama_embed_object_handlers;
static int php_llama_backend_ready = 0;

/* Max sequences packed into one llama_encode. */
#define PHP_LLAMA_EMBED_SEQ_MAX 24

static void php_llama_embed_release(php_llama_embed_object *intern)
{
	if (intern->ctx) {
		llama_free(intern->ctx);
		intern->ctx = NULL;
	}
	if (intern->model) {
		llama_model_free(intern->model);
		intern->model = NULL;
	}
	intern->dimension = 0;
}

static void php_llama_embed_free_obj(zend_object *object)
{
	php_llama_embed_object *intern = (php_llama_embed_object *)((char *)(object)
		- XtOffsetOf(php_llama_embed_object, std));
	php_llama_embed_release(intern);
	zend_object_std_dtor(object);
}

static zend_object *php_llama_embed_create_object(zend_class_entry *class_type)
{
	php_llama_embed_object *intern = zend_object_alloc(sizeof(php_llama_embed_object), class_type);

	zend_object_std_init(&intern->std, class_type);
	object_properties_init(&intern->std, class_type);
	intern->model = NULL;
	intern->ctx = NULL;
	intern->dimension = 0;
	intern->debug = 0;
	intern->std.handlers = &php_llama_embed_object_handlers;

	return &intern->std;
}

static php_llama_embed_object *php_llama_embed_from_zend(zend_object *obj)
{
	php_llama_embed_object *intern = (php_llama_embed_object *)((char *)(obj)
		- XtOffsetOf(php_llama_embed_object, std));

	if (!intern->model || !intern->ctx) {
		zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed is not loaded");
		return NULL;
	}

	return intern;
}

/* Drop libllama/ggml stderr INFO spam; failures still throw PHP exceptions. */
static void php_llama_log_quiet(enum ggml_log_level level, const char *text, void *user_data)
{
	(void)level;
	(void)text;
	(void)user_data;
}

/**
 * Pack up to PHP_LLAMA_EMBED_SEQ_MAX sequences into one encode when token sum fits n_ubatch.
 */
static int php_llama_embed_texts(php_llama_embed_object *intern, zval *texts_zv, zval *return_arr)
{
	const struct llama_vocab *vocab;
	HashTable *ht;
	uint32_t n;
	uint32_t i;
	uint32_t j;
	llama_token **toks = NULL;
	int32_t *n_toks = NULL;
	int32_t dim;
	int32_t n_ubatch;
	int32_t n_seq_max;
	zval *entry;

	ht = Z_ARRVAL_P(texts_zv);
	n = zend_hash_num_elements(ht);
	array_init_size(return_arr, n);
	if (n == 0) {
		return SUCCESS;
	}

	vocab = llama_model_get_vocab(intern->model);
	dim = intern->dimension;
	n_ubatch = (int32_t)llama_n_ubatch(intern->ctx);
	n_seq_max = (int32_t)llama_n_seq_max(intern->ctx);
	if (n_seq_max < 1) {
		n_seq_max = 1;
	}
	if (n_seq_max > PHP_LLAMA_EMBED_SEQ_MAX) {
		n_seq_max = PHP_LLAMA_EMBED_SEQ_MAX;
	}

	toks = ecalloc(n, sizeof(llama_token *));
	n_toks = ecalloc(n, sizeof(int32_t));

	i = 0;
	ZEND_HASH_FOREACH_VAL(ht, entry) {
		zend_string *s;
		int32_t n_tokens_max;
		int32_t n_tokens;

		if (Z_TYPE_P(entry) != IS_STRING) {
			for (j = 0; j < i; j++) {
				efree(toks[j]);
			}
			efree(toks);
			efree(n_toks);
			zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed::embedBatch expects string[]");
			return FAILURE;
		}
		s = Z_STR_P(entry);
		n_tokens_max = (int32_t)ZSTR_LEN(s) + 8;
		if (n_tokens_max < 64) {
			n_tokens_max = 64;
		}
		toks[i] = emalloc(sizeof(llama_token) * (size_t)n_tokens_max);
		n_tokens = llama_tokenize(vocab, ZSTR_VAL(s), (int32_t)ZSTR_LEN(s),
			toks[i], n_tokens_max, true, true);
		if (n_tokens < 0) {
			n_tokens_max = -n_tokens + 8;
			toks[i] = erealloc(toks[i], sizeof(llama_token) * (size_t)n_tokens_max);
			n_tokens = llama_tokenize(vocab, ZSTR_VAL(s), (int32_t)ZSTR_LEN(s),
				toks[i], n_tokens_max, true, true);
		}
		if (n_tokens <= 0) {
			for (j = 0; j <= i; j++) {
				efree(toks[j]);
			}
			efree(toks);
			efree(n_toks);
			zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed: tokenize failed");
			return FAILURE;
		}
		if (n_tokens > n_ubatch) {
			for (j = 0; j <= i; j++) {
				efree(toks[j]);
			}
			efree(toks);
			efree(n_toks);
			zend_throw_exception_ex(zend_ce_exception, 0,
				"LlamaEmbed: text too long (%d tokens, max %d)", n_tokens, n_ubatch);
			return FAILURE;
		}
		n_toks[i] = n_tokens;
		i++;
	} ZEND_HASH_FOREACH_END();

	i = 0;
	while (i < n) {
		int32_t pack_n = 0;
		int32_t pack_tokens = 0;
		int32_t pack_start = (int32_t)i;
		struct llama_batch batch;
		int32_t pos;
		int32_t s;
		int32_t t;
		int rc;

		while (i < n && pack_n < n_seq_max && pack_tokens + n_toks[i] <= n_ubatch) {
			pack_tokens += n_toks[i];
			pack_n++;
			i++;
		}
		if (pack_n < 1) {
			for (j = 0; j < n; j++) {
				efree(toks[j]);
			}
			efree(toks);
			efree(n_toks);
			zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed: cannot pack batch");
			return FAILURE;
		}

		if (intern->debug) {
			fprintf(stderr, "LlamaEmbed pack: n=%d tokens=%d\n", pack_n, pack_tokens);
			fflush(stderr);
		}

		llama_memory_clear(llama_get_memory(intern->ctx), true);
		batch = llama_batch_init(pack_tokens, 0, 1);
		batch.n_tokens = pack_tokens;
		pos = 0;
		for (s = 0; s < pack_n; s++) {
			int32_t nt = n_toks[pack_start + s];
			for (t = 0; t < nt; t++) {
				batch.token[pos] = toks[pack_start + s][t];
				batch.pos[pos] = t;
				batch.n_seq_id[pos] = 1;
				batch.seq_id[pos][0] = (llama_seq_id)s;
				batch.logits[pos] = 0;
				pos++;
			}
			batch.logits[pos - 1] = 1;
		}

		rc = llama_encode(intern->ctx, batch);
		if (rc != 0) {
			rc = llama_decode(intern->ctx, batch);
		}
		if (rc != 0) {
			llama_batch_free(batch);
			for (j = 0; j < n; j++) {
				efree(toks[j]);
			}
			efree(toks);
			efree(n_toks);
			zend_throw_exception_ex(zend_ce_exception, 0,
				"LlamaEmbed: encode/decode failed (rc=%d)", rc);
			return FAILURE;
		}

		llama_synchronize(intern->ctx);
		for (s = 0; s < pack_n; s++) {
			float *embd;
			float *copy;
			zval vec;
			int32_t k;
			double sum = 0.0;

			embd = llama_get_embeddings_seq(intern->ctx, (llama_seq_id)s);
			if (!embd && pack_n == 1) {
				embd = llama_get_embeddings_ith(intern->ctx, -1);
			}
			if (!embd) {
				llama_batch_free(batch);
				for (j = 0; j < n; j++) {
					efree(toks[j]);
				}
				efree(toks);
				efree(n_toks);
				zend_throw_exception_ex(zend_ce_exception, 0,
					"LlamaEmbed: missing embeddings for seq %d", s);
				return FAILURE;
			}
			copy = emalloc(sizeof(float) * (size_t)dim);
			memcpy(copy, embd, sizeof(float) * (size_t)dim);
			/* L2-normalize so FAISS cosine ≈ inner product */
			for (k = 0; k < dim; k++) {
				sum += (double)copy[k] * (double)copy[k];
			}
			if (sum > 0.0) {
				sum = sqrt(sum);
				for (k = 0; k < dim; k++) {
					copy[k] = (float)((double)copy[k] / sum);
				}
			}
			array_init_size(&vec, dim);
			for (k = 0; k < dim; k++) {
				add_next_index_double(&vec, (double)copy[k]);
			}
			efree(copy);
			add_next_index_zval(return_arr, &vec);
		}
		llama_batch_free(batch);
	}

	for (j = 0; j < n; j++) {
		efree(toks[j]);
	}
	efree(toks);
	efree(n_toks);
	return SUCCESS;
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_llama_load, 0, 1, LlamaEmbed, 0)
	ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
	ZEND_ARG_ARRAY_INFO(0, options, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_embed_batch, 0, 1, IS_ARRAY, 0)
	ZEND_ARG_ARRAY_INFO(0, texts, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_dimension, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(LlamaEmbed, load)
{
	zend_string *path;
	zval *options = NULL;
	php_llama_embed_object *intern;
	struct llama_model_params mparams;
	struct llama_context_params cparams;
	ggml_backend_dev_t cpu_only[2] = { NULL, NULL };
	int32_t n_gpu_layers;

	ZEND_PARSE_PARAMETERS_START(1, 2)
		Z_PARAM_STR(path)
		Z_PARAM_OPTIONAL
		Z_PARAM_ARRAY_OR_NULL(options)
	ZEND_PARSE_PARAMETERS_END();

	if (!php_llama_backend_ready) {
		llama_backend_init();
		php_llama_backend_ready = 1;
	}

	n_gpu_layers = php_llama_n_gpu_layers();
	mparams = llama_model_default_params();
	mparams.n_gpu_layers = n_gpu_layers;
	/* 0 keeps weights and the context off the GPU. Vulkan was OOMing at startup. */
	if (n_gpu_layers == 0) {
		ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

		if (cpu) {
			cpu_only[0] = cpu;
			cpu_only[1] = NULL;
			mparams.devices = cpu_only;
		}
	}

	object_init_ex(return_value, php_llama_embed_ce);
	intern = (php_llama_embed_object *)((char *)(Z_OBJ_P(return_value))
		- XtOffsetOf(php_llama_embed_object, std));
	intern->debug = 0;

	if (options && Z_TYPE_P(options) == IS_ARRAY) {
		zval *z = zend_hash_str_find(Z_ARRVAL_P(options), "debug", sizeof("debug") - 1);
		if (z) {
			intern->debug = zend_is_true(z) ? 1 : 0;
		}
	}

	intern->model = llama_model_load_from_file(ZSTR_VAL(path), mparams);
	if (!intern->model) {
		zend_throw_exception_ex(zend_ce_exception, 0,
			"LlamaEmbed: failed to load model %s", ZSTR_VAL(path));
		RETURN_THROWS();
	}

	cparams = llama_context_default_params();
	cparams.embeddings = true;
	/* GGUF/hparams only; no PHP override — missing metadata resolves to NONE */
	cparams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
	cparams.n_batch = 4096;
	cparams.n_ubatch = 4096;
	cparams.n_ctx = 4096;
	cparams.n_seq_max = PHP_LLAMA_EMBED_SEQ_MAX;
	if (n_gpu_layers == 0) {
		cparams.offload_kqv = false;
		cparams.op_offload = false;
	}

	intern->ctx = llama_init_from_model(intern->model, cparams);
	if (!intern->ctx) {
		llama_model_free(intern->model);
		intern->model = NULL;
		zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed: failed to create context");
		RETURN_THROWS();
	}

	if (llama_pooling_type(intern->ctx) == LLAMA_POOLING_TYPE_NONE) {
		llama_free(intern->ctx);
		intern->ctx = NULL;
		llama_model_free(intern->model);
		intern->model = NULL;
		zend_throw_exception_ex(zend_ce_exception, 0,
			"LlamaEmbed: model has no pooling_type in GGUF metadata");
		RETURN_THROWS();
	}

	if (intern->debug) {
		fprintf(stderr,
			"LlamaEmbed load: n_gpu_layers=%d pooling=%d requested n_batch=%d n_ubatch=%d n_ctx=%d n_seq_max=%d; "
			"actual n_batch=%u n_ubatch=%u n_ctx=%u n_seq_max=%u\n",
			(int)n_gpu_layers,
			(int)llama_pooling_type(intern->ctx),
			(int)cparams.n_batch, (int)cparams.n_ubatch, (int)cparams.n_ctx, (int)cparams.n_seq_max,
			llama_n_batch(intern->ctx), llama_n_ubatch(intern->ctx),
			llama_n_ctx(intern->ctx), llama_n_seq_max(intern->ctx));
		fflush(stderr);
	}

	llama_set_embeddings(intern->ctx, true);

	intern->dimension = llama_model_n_embd_out(intern->model);
	if (intern->dimension <= 0) {
		intern->dimension = llama_model_n_embd(intern->model);
	}
}

PHP_METHOD(LlamaEmbed, embedBatch)
{
	zval *texts;
	php_llama_embed_object *intern;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_ARRAY(texts)
	ZEND_PARSE_PARAMETERS_END();

	intern = php_llama_embed_from_zend(Z_OBJ_P(ZEND_THIS));
	if (!intern) {
		RETURN_THROWS();
	}

	if (php_llama_embed_texts(intern, texts, return_value) != SUCCESS) {
		RETURN_THROWS();
	}
}

PHP_METHOD(LlamaEmbed, dimension)
{
	php_llama_embed_object *intern;

	ZEND_PARSE_PARAMETERS_NONE();

	intern = php_llama_embed_from_zend(Z_OBJ_P(ZEND_THIS));
	if (!intern) {
		RETURN_THROWS();
	}
	RETURN_LONG(intern->dimension);
}

static const zend_function_entry php_llama_embed_methods[] = {
	PHP_ME(LlamaEmbed, load, arginfo_llama_load, ZEND_ACC_PUBLIC|ZEND_ACC_STATIC)
	PHP_ME(LlamaEmbed, embedBatch, arginfo_llama_embed_batch, ZEND_ACC_PUBLIC)
	PHP_ME(LlamaEmbed, dimension, arginfo_llama_dimension, ZEND_ACC_PUBLIC)
	PHP_FE_END
};

static const zend_function_entry llamacpp_functions[] = {
	PHP_FE_END
};

static PHP_GINIT_FUNCTION(llamacpp)
{
#if defined(COMPILE_DL_LLAMACPP) && defined(ZTS)
	ZEND_TSRMLS_CACHE_UPDATE();
#endif
	llamacpp_globals->n_gpu_layers = 0;
}

PHP_MINIT_FUNCTION(llamacpp)
{
	zend_class_entry ce;

	REGISTER_INI_ENTRIES();
	llama_log_set(php_llama_log_quiet, NULL);

	INIT_CLASS_ENTRY(ce, "LlamaEmbed", php_llama_embed_methods);
	php_llama_embed_ce = zend_register_internal_class_ex(&ce, NULL);
	php_llama_embed_ce->create_object = php_llama_embed_create_object;

	memcpy(&php_llama_embed_object_handlers, &std_object_handlers, sizeof(zend_object_handlers));
	php_llama_embed_object_handlers.offset = XtOffsetOf(php_llama_embed_object, std);
	php_llama_embed_object_handlers.free_obj = php_llama_embed_free_obj;
	php_llama_embed_object_handlers.clone_obj = NULL;

	return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(llamacpp)
{
	if (php_llama_backend_ready) {
		llama_backend_free();
		php_llama_backend_ready = 0;
	}
	UNREGISTER_INI_ENTRIES();
	return SUCCESS;
}

PHP_MINFO_FUNCTION(llamacpp)
{
	php_info_print_table_start();
	php_info_print_table_header(2, "llamacpp support", "enabled");
	php_info_print_table_row(2, "Version", PHP_LLAMACPP_VERSION);
	php_info_print_table_end();

	DISPLAY_INI_ENTRIES();
}

zend_module_entry llamacpp_module_entry = {
	STANDARD_MODULE_HEADER,
	"llamacpp",
	llamacpp_functions,
	PHP_MINIT(llamacpp),
	PHP_MSHUTDOWN(llamacpp),
	NULL,
	NULL,
	PHP_MINFO(llamacpp),
	PHP_LLAMACPP_VERSION,
	PHP_MODULE_GLOBALS(llamacpp),
	PHP_GINIT(llamacpp),
	NULL,
	NULL,
	STANDARD_MODULE_PROPERTIES_EX
};

#ifdef COMPILE_DL_LLAMACPP
ZEND_GET_MODULE(llamacpp)
#endif
