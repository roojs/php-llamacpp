# php-llamacpp — PHP Zend extension for local GGUF embeddings

Exposes `LlamaEmbed` (load / embed / embedBatch / dimension / close) via Debian `libllama`.

Sibling repo at `gitlive/php-llamacpp` (used by Translate RAG local embed).

## Prerequisites

- PHP 8.4 dev headers (`php8.4-dev` or matching API version)
- `build-essential`, `pkg-config`, `curl`
- `libllama0` / `libllama-dev` (and ggml backends: vulkan, blas)

On hosts where apt’s llama packages are too old or missing, install from Debian pools:

```bash
cd php-llamacpp
./scripts/fetch-libllama.sh
```

That script downloads/installs `libggml*` + `libllama*` debs, skips HIP on ROCm, and registers `/usr/lib/*/llama` with `ldconfig`. Helper: `scripts/ci/debian-pool-deb.sh`.

Optional (CLI smoke / `llama-server` only — not required to build this extension):

```bash
# match libllama version, e.g. from the same Debian llama.cpp pool
# sudo dpkg -i llama.cpp-tools_….deb
```

Build on **each host** against that host’s PHP — do not copy `llama.so` between machines.

## Model files (not in this tree)

Put GGUFs under storedir (shared, not under `Translate/`):

```text
{Pman.storedir}/Llama/models/bge-m3/bge-m3-q4_k_m.gguf
```

Example: `/home/press/Llama/models/bge-m3/bge-m3-q4_k_m.gguf`

## Build

```bash
cd php-llamacpp
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

Restart php-fpm / Apache after enabling for those SAPIs.

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

- Links against system `libllama` (`<llama.h>`); does not vendor llama.cpp sources.
- Do not keep the model in MINIT globals; `close()` / object dtor frees model+context.
- RAG / MediaOutreach wiring is separate (Translate RAG plans).
