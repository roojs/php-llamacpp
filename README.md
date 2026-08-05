# php-llama — PHP Zend extension for local GGUF embeddings

Exposes `LlamaEmbed` (load / embed / embedBatch / dimension / close) via Debian `libllama`.

Develop under `/home/press/php-llama` for now. Later copy into `gitlive/php-llama` when `git.roojs.com/php-llama` exists (same pattern as `php-fiass`).

## Prerequisites

- PHP 8.4 dev headers (`php8.4-dev`)
- `libllama0` / `libllama-dev` (and ggml backends) — see `OLLMchat/scripts/fetch-libllama.sh`
- `ldconfig` already listing `/usr/lib/*/llama` (fetch script registers it)
- `pkg-config` finds `llama.pc` under `/usr/lib/x86_64-linux-gnu/llama/pkgconfig`

Build on **each host** against that host’s PHP — do not copy `llama.so` between machines.

## Model files (not in this tree)

Put GGUFs under storedir (shared, not under `Translate/`):

```text
{Pman.storedir}/Llama/models/bge-m3/bge-m3-q4_k_m.gguf
```

Example: `/home/press/Llama/models/bge-m3/bge-m3-q4_k_m.gguf`

## Build

```bash
cd /home/press/php-llama
phpize
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/llama/pkgconfig
./configure --enable-llama
make -j$(nproc)
```

Output: `modules/llama.so`

Optional install:

```bash
sudo make install
```

Or load without installing:

```bash
php -d extension=$PWD/modules/llama.so test_llama.php \
  /home/press/Llama/models/bge-m3/bge-m3-q4_k_m.gguf
```

## API

```php
$e = LlamaEmbed::load('/home/press/Llama/models/bge-m3/bge-m3-q4_k_m.gguf', [
    'n_gpu_layers' => 99,
    'pooling' => 'cls',
]);
$dim = $e->dimension();
$vec = $e->embed($text);          // float[], L2-normalized
$vecs = $e->embedBatch([$a, $b]);
$e->close();
```

## Notes

- No RAG / MediaOutreach wiring in this phase — extension + smoke only.
- Do not keep the model in MINIT globals; `close()` / object dtor frees model+context.
- RAG wiring and corpus rebuild come in a later Translate RAG version.
