# PHP extension for llama.cpp (embedding only currently)

A PHP extension that links against the llama.cpp library (`libllama`). It currently only implements embeddings: `LlamaEmbed` (load / embedBatch / dimension) for local GGUF models. Further llama.cpp features would not take much to add; embeddings is the only part implemented so far.

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

Build on **each host** against that host’s PHP — do not copy `llamacpp.so` between machines.

## Model files (not in this tree)

GGUF weights are not shipped with this extension. Pass any readable GGUF path to `LlamaEmbed::load()`.

Example: `/path/to/bge-m3-q4_k_m.gguf`

## Build

```bash
cd php-llamacpp
phpize
./configure
make -j$(nproc)
```

Output: `modules/llamacpp.so`

Optional install:

```bash
sudo make install
```

Or load without installing:

```bash
php -d extension=$PWD/modules/llamacpp.so test_llamacpp.php \
  /path/to/bge-m3-q4_k_m.gguf
```

Restart php-fpm / Apache after enabling for those SAPIs.

## Debian package

Each suite build ships the module for that suite's PHP (`php8.4-llamacpp` on Debian 13 and Ubuntu 25.04/25.10, `php8.5-llamacpp` on Ubuntu 26.04) plus an empty `php-llamacpp` package that depends on it. Install the alias. When the distro moves to a newer default PHP, that dependency follows, and the upgrade installs the matching module. `dh-php` enables `llamacpp` for every SAPI of that PHP — `cli`, `fpm`, `apache2`, and the rest. A SAPI installed later picks the module up on its own.

```bash
sudo apt-get install php-llamacpp
php -m | grep llamacpp
```

`scripts/release.sh` tags `vX.Y.Z` from `CHANGELOG.md` and pushes it. GitHub Actions builds the suite `.deb` files and attaches them all to that one release. The [roojs APT repository](https://roojs.github.io/repos/) can pull that release in later. Do not run the release script from an agent.

## API

```php
$e = LlamaEmbed::load('/path/to/bge-m3-q4_k_m.gguf');
$dim = $e->dimension();
$vecs = $e->embedBatch([$a, $b]); // float[][], each L2-normalized
```

## Notes

- GPU offload is always all layers (`n_gpu_layers = -1`); no PHP option.
- Pooling comes from GGUF metadata only (no `pooling` option). Missing metadata → exception at load.

- Single-string embed: `$e->embedBatch([$text])[0]`.
- Links against system `libllama` (`<llama.h>`); does not vendor llama.cpp sources.
- Do not keep the model in MINIT globals; the object destructor frees model+context. `unset($e)` releases it immediately.

## Artificial Intelligence Usage

This project was developed with the assistance of artificial intelligence.

- API design was done by the author
- Other than the API design, very little of this extension was hand crafted
- AI was used to create and build the implementation

## License

Copyright (c) 2026 Roo J Solutions Ltd. Author: Leon <leon@roojs.com>.

php-llamacpp is distributed under the [PHP License, version 3.01](LICENSE), the same license used by PHP extensions.

Debian packages are maintained by Alan Knowles <alan@roojs.com>.

This product includes PHP software, freely available from <http://www.php.net/software/>.
