#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "zend_exceptions.h"
#include "php_llama.h"
#include "ext/standard/info.h"

#include <math.h>
#include <string.h>

zend_class_entry *php_llama_embed_ce;

static zend_object_handlers php_llama_embed_object_handlers;
static int php_llama_backend_ready = 0;

/* Max sequences packed into one llama_encode (align RAG array_chunk). */
#define PHP_LLAMA_EMBED_SEQ_MAX 10


#define Z_LLAMA_EMBED_P(zv) php_llama_embed_fetch_object(Z_OBJ_P(zv))

static inline php_llama_embed_object *php_llama_embed_fetch_object(zend_object *obj)
{
	return (php_llama_embed_object *)((char *)(obj) - XtOffsetOf(php_llama_embed_object, std));
}

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
	php_llama_embed_object *intern = php_llama_embed_fetch_object(object);
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
	intern->std.handlers = &php_llama_embed_object_handlers;

	return &intern->std;
}

static php_llama_embed_object *php_llama_embed_from_zend(zend_object *obj)
{
	php_llama_embed_object *intern = php_llama_embed_fetch_object(obj);

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

static void php_llama_ensure_backend(void)
{
	if (php_llama_backend_ready) {
		return;
	}
	llama_log_set(php_llama_log_quiet, NULL);
	llama_backend_init();
	php_llama_backend_ready = 1;
}

static enum llama_pooling_type php_llama_parse_pooling(zval *opts)
{
	zval *z;

	if (!opts || Z_TYPE_P(opts) != IS_ARRAY) {
		return LLAMA_POOLING_TYPE_CLS;
	}
	z = zend_hash_str_find(Z_ARRVAL_P(opts), "pooling", sizeof("pooling") - 1);
	if (!z) {
		return LLAMA_POOLING_TYPE_CLS;
	}
	if (Z_TYPE_P(z) == IS_STRING) {
		if (zend_string_equals_literal_ci(Z_STR_P(z), "mean")) {
			return LLAMA_POOLING_TYPE_MEAN;
		}
		if (zend_string_equals_literal_ci(Z_STR_P(z), "last")) {
			return LLAMA_POOLING_TYPE_LAST;
		}
		if (zend_string_equals_literal_ci(Z_STR_P(z), "none")) {
			return LLAMA_POOLING_TYPE_NONE;
		}
	}
	return LLAMA_POOLING_TYPE_CLS;
}

static int32_t php_llama_parse_n_gpu_layers(zval *opts)
{
	zval *z;

	if (!opts || Z_TYPE_P(opts) != IS_ARRAY) {
		return 99;
	}
	z = zend_hash_str_find(Z_ARRVAL_P(opts), "n_gpu_layers", sizeof("n_gpu_layers") - 1);
	if (!z) {
		return 99;
	}
	return (int32_t)zval_get_long(z);
}

static void php_llama_l2_normalize(float *v, int32_t n)
{
	double sum = 0.0;
	int32_t i;

	for (i = 0; i < n; i++) {
		sum += (double)v[i] * (double)v[i];
	}
	if (sum <= 0.0) {
		return;
	}
	sum = sqrt(sum);
	for (i = 0; i < n; i++) {
		v[i] = (float)((double)v[i] / sum);
	}
}

static int php_llama_embed_text(php_llama_embed_object *intern, const char *text, size_t text_len, zval *return_vec)
{
	const struct llama_vocab *vocab;
	llama_token *tokens = NULL;
	int32_t n_tokens;
	int32_t n_tokens_max;
	struct llama_batch batch;
	float *embd;
	int32_t dim;
	int32_t i;
	int rc;

	vocab = llama_model_get_vocab(intern->model);
	dim = intern->dimension;
	n_tokens_max = (int32_t)text_len + 8;
	if (n_tokens_max < 64) {
		n_tokens_max = 64;
	}

	tokens = emalloc(sizeof(llama_token) * (size_t)n_tokens_max);
	n_tokens = llama_tokenize(vocab, text, (int32_t)text_len, tokens, n_tokens_max, true, true);
	if (n_tokens < 0) {
		n_tokens_max = -n_tokens + 8;
		tokens = erealloc(tokens, sizeof(llama_token) * (size_t)n_tokens_max);
		n_tokens = llama_tokenize(vocab, text, (int32_t)text_len, tokens, n_tokens_max, true, true);
	}
	if (n_tokens <= 0) {
		efree(tokens);
		zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed: tokenize failed");
		return FAILURE;
	}

	{
		int32_t n_ubatch = (int32_t)llama_n_ubatch(intern->ctx);
		if (n_tokens > n_ubatch) {
			efree(tokens);
			zend_throw_exception_ex(zend_ce_exception, 0,
				"LlamaEmbed: text too long (%d tokens, max %d)", n_tokens, n_ubatch);
			return FAILURE;
		}
	}

	llama_memory_clear(llama_get_memory(intern->ctx), true);

	batch = llama_batch_init(n_tokens, 0, 1);
	batch.n_tokens = n_tokens;
	for (i = 0; i < n_tokens; i++) {
		batch.token[i] = tokens[i];
		batch.pos[i] = i;
		batch.n_seq_id[i] = 1;
		batch.seq_id[i][0] = 0;
		batch.logits[i] = 0;
	}
	/* request embeddings for all tokens when pooling; last token is enough for CLS/LAST */
	batch.logits[n_tokens - 1] = 1;

	rc = llama_encode(intern->ctx, batch);
	if (rc != 0) {
		rc = llama_decode(intern->ctx, batch);
	}
	efree(tokens);
	if (rc != 0) {
		llama_batch_free(batch);
		zend_throw_exception_ex(zend_ce_exception, 0,
			"LlamaEmbed: encode/decode failed (rc=%d)", rc);
		return FAILURE;
	}

	llama_synchronize(intern->ctx);
	embd = llama_get_embeddings_seq(intern->ctx, 0);
	if (!embd) {
		embd = llama_get_embeddings_ith(intern->ctx, -1);
	}
	if (!embd) {
		llama_batch_free(batch);
		zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed: missing embeddings");
		return FAILURE;
	}

	{
		float *copy = emalloc(sizeof(float) * (size_t)dim);
		memcpy(copy, embd, sizeof(float) * (size_t)dim);
		php_llama_l2_normalize(copy, dim);

		array_init_size(return_vec, dim);
		for (i = 0; i < dim; i++) {
			add_next_index_double(return_vec, (double)copy[i]);
		}
		efree(copy);
	}

	llama_batch_free(batch);
	return SUCCESS;
}
{
	float *copy;
	int32_t i;

	copy = emalloc(sizeof(float) * (size_t)dim);
	memcpy(copy, embd, sizeof(float) * (size_t)dim);
	php_llama_l2_normalize(copy, dim);

	array_init_size(return_vec, dim);
	for (i = 0; i < dim; i++) {
		add_next_index_double(return_vec, (double)copy[i]);
	}
	efree(copy);
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
			zval_ptr_dtor(return_arr);
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
			zval_ptr_dtor(return_arr);
			zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed: tokenize failed");
			return FAILURE;
		}
		if (n_tokens > n_ubatch) {
			for (j = 0; j <= i; j++) {
				efree(toks[j]);
			}
			efree(toks);
			efree(n_toks);
			zval_ptr_dtor(return_arr);
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
			zval_ptr_dtor(return_arr);
			zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed: cannot pack batch");
			return FAILURE;
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
			zval_ptr_dtor(return_arr);
			zend_throw_exception_ex(zend_ce_exception, 0,
				"LlamaEmbed: encode/decode failed (rc=%d)", rc);
			return FAILURE;
		}

		llama_synchronize(intern->ctx);
		for (s = 0; s < pack_n; s++) {
			float *embd;
			zval vec;

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
				zval_ptr_dtor(return_arr);
				zend_throw_exception_ex(zend_ce_exception, 0,
					"LlamaEmbed: missing embeddings for seq %d", s);
				return FAILURE;
			}
			ZVAL_UNDEF(&vec);
			php_llama_embd_to_zval(embd, dim, &vec);
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

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_embed, 0, 1, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO(0, text, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_embed_batch, 0, 1, IS_ARRAY, 0)
	ZEND_ARG_ARRAY_INFO(0, texts, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_dimension, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_llama_close, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(LlamaEmbed, load)
{
	zend_string *path;
	zval *options = NULL;
	php_llama_embed_object *intern;
	struct llama_model_params mparams;
	struct llama_context_params cparams;
	int32_t n_gpu_layers;
	enum llama_pooling_type pooling;

	ZEND_PARSE_PARAMETERS_START(1, 2)
		Z_PARAM_STR(path)
		Z_PARAM_OPTIONAL
		Z_PARAM_ARRAY_OR_NULL(options)
	ZEND_PARSE_PARAMETERS_END();

	php_llama_ensure_backend();

	n_gpu_layers = php_llama_parse_n_gpu_layers(options);
	pooling = php_llama_parse_pooling(options);

	mparams = llama_model_default_params();
	mparams.n_gpu_layers = n_gpu_layers;

	object_init_ex(return_value, php_llama_embed_ce);
	intern = Z_LLAMA_EMBED_P(return_value);

	intern->model = llama_model_load_from_file(ZSTR_VAL(path), mparams);
	if (!intern->model) {
		zend_throw_exception_ex(zend_ce_exception, 0,
			"LlamaEmbed: failed to load model %s", ZSTR_VAL(path));
		RETURN_THROWS();
	}

	cparams = llama_context_default_params();
	cparams.embeddings = true;
	cparams.pooling_type = pooling;
	cparams.n_batch = 2048;
	cparams.n_ubatch = 2048;
	cparams.n_ctx = 2048;
	cparams.n_seq_max = PHP_LLAMA_EMBED_SEQ_MAX;


	intern->ctx = llama_init_from_model(intern->model, cparams);
	if (!intern->ctx) {
		llama_model_free(intern->model);
		intern->model = NULL;
		zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed: failed to create context");
		RETURN_THROWS();
	}

	llama_set_embeddings(intern->ctx, true);

	intern->dimension = llama_model_n_embd_out(intern->model);
	if (intern->dimension <= 0) {
		intern->dimension = llama_model_n_embd(intern->model);
	}
}

PHP_METHOD(LlamaEmbed, embed)
{
	zend_string *text;
	php_llama_embed_object *intern;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_STR(text)
	ZEND_PARSE_PARAMETERS_END();

	intern = php_llama_embed_from_zend(Z_OBJ_P(ZEND_THIS));
	if (!intern) {
		RETURN_THROWS();
	}

	if (php_llama_embed_text(intern, ZSTR_VAL(text), ZSTR_LEN(text), return_value) != SUCCESS) {
		RETURN_THROWS();
	}
}

PHP_METHOD(LlamaEmbed, embedBatch)
{
	zval *texts;
	zval *entry;
	php_llama_embed_object *intern;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_ARRAY(texts)
	ZEND_PARSE_PARAMETERS_END();

	intern = php_llama_embed_from_zend(Z_OBJ_P(ZEND_THIS));
	if (!intern) {
		RETURN_THROWS();
	}

	array_init(return_value);
	ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(texts), entry) {
		zval vec;
		zend_string *s;

		if (Z_TYPE_P(entry) != IS_STRING) {
			zval_ptr_dtor(return_value);
			zend_throw_exception_ex(zend_ce_exception, 0, "LlamaEmbed::embedBatch expects string[]");
			RETURN_THROWS();
		}
		s = Z_STR_P(entry);
		ZVAL_UNDEF(&vec);
		if (php_llama_embed_text(intern, ZSTR_VAL(s), ZSTR_LEN(s), &vec) != SUCCESS) {
			zval_ptr_dtor(return_value);
			RETURN_THROWS();
		}
		add_next_index_zval(return_value, &vec);
	} ZEND_HASH_FOREACH_END();
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

PHP_METHOD(LlamaEmbed, close)
{
	php_llama_embed_object *intern;

	ZEND_PARSE_PARAMETERS_NONE();

	intern = php_llama_embed_fetch_object(Z_OBJ_P(ZEND_THIS));
	php_llama_embed_release(intern);
}

static const zend_function_entry php_llama_embed_methods[] = {
	PHP_ME(LlamaEmbed, load, arginfo_llama_load, ZEND_ACC_PUBLIC|ZEND_ACC_STATIC)
	PHP_ME(LlamaEmbed, embed, arginfo_llama_embed, ZEND_ACC_PUBLIC)
	PHP_ME(LlamaEmbed, embedBatch, arginfo_llama_embed_batch, ZEND_ACC_PUBLIC)
	PHP_ME(LlamaEmbed, dimension, arginfo_llama_dimension, ZEND_ACC_PUBLIC)
	PHP_ME(LlamaEmbed, close, arginfo_llama_close, ZEND_ACC_PUBLIC)
	PHP_FE_END
};

static const zend_function_entry llama_functions[] = {
	PHP_FE_END
};

PHP_MINIT_FUNCTION(llama)
{
	zend_class_entry ce;

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

PHP_MSHUTDOWN_FUNCTION(llama)
{
	if (php_llama_backend_ready) {
		llama_backend_free();
		php_llama_backend_ready = 0;
	}
	return SUCCESS;
}

PHP_RINIT_FUNCTION(llama)
{
	return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(llama)
{
	return SUCCESS;
}

PHP_MINFO_FUNCTION(llama)
{
	php_info_print_table_start();
	php_info_print_table_header(2, "llama support", "enabled");
	php_info_print_table_row(2, "Version", PHP_LLAMA_VERSION);
	php_info_print_table_end();
}

zend_module_entry llama_module_entry = {
	STANDARD_MODULE_HEADER,
	"llama",
	llama_functions,
	PHP_MINIT(llama),
	PHP_MSHUTDOWN(llama),
	PHP_RINIT(llama),
	PHP_RSHUTDOWN(llama),
	PHP_MINFO(llama),
	PHP_LLAMA_VERSION,
	STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_LLAMA
ZEND_GET_MODULE(llama)
#endif
