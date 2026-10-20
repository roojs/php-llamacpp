# php-llamacpp — PHP Zend extension for local GGUF embeddings

Exposes `LlamaEmbed` (load / embedBatch / dimension / close) via Debian `libllama`.

Sibling repo at `gitlive/php-llamacpp` (used by Translate RAG local embed).

## Prerequisites

- PHP 8.4 dev headers (`php8.4-dev` or matching API version)
- `build-essential`, `pkg-config`, `curl`
- `libllama0` / `libllama-dev` (and ggml backends: vulkan, blas)

**amd64 only.** Deploy hosts are Debian 13 (`trixie`). Ubuntu 25.04+ (`plucky`, `questing`, `resolute`) is for local testing on the same architecture.

Those suites cannot install Debian sid’s newest `libllama` (it needs glibc 2.43). Use the libc-matching builds from the [roojs APT repository](https://roojs.github.io/repos/):

```bash
cd php-llamacpp
./scripts/fetch-libllama.sh
```

That script adds the roojs source for this suite, installs `libggml*` + `libllama*` plus the vulkan and blas backends, skips HIP on ROCm, and registers `/usr/lib/x86_64-linux-gnu/llama` with `ldconfig`.

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
./configure
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
$e = LlamaEmbed::load('/home/press/Llama/models/bge-m3/bge-m3-q4_k_m.gguf');
$dim = $e->dimension();
$vecs = $e->embedBatch([$a, $b]); // float[][], each L2-normalized
$e->close();
```

## Notes

- GPU offload is always all layers (`n_gpu_layers = -1`); no PHP option.
- Pooling comes from GGUF metadata only (no `pooling` option). Missing metadata → exception at load.

- Single-string embed: `$e->embedBatch([$text])[0]` (or `Pman_Translate_RAG_Llama::embed()`).
- Links against system `libllama` (`<llama.h>`); does not vendor llama.cpp sources.
- Do not keep the model in MINIT globals; `close()` / object dtor frees model+context.
- RAG / MediaOutreach wiring is separate (Translate RAG plans).
