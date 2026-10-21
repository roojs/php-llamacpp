# Changelog

All notable changes to php-llamacpp are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for git tags (`v0.1.0`).

`debian/changelog` is generated from this file at release time
(`scripts/release/changelog.sh`).

## [Unreleased]

## [0.1.2] - 2026-10-21

### Changed

- `LlamaEmbed::load()` stays on the CPU unless `llamacpp.n_gpu_layers` is set. `0` (default) uses no GPU memory; `-1` offloads every layer.

## [0.1.1] - 2026-10-21

### Fixed

- Link `llamacpp.so` against `libllama`. The 0.1.0 module did not depend on it, so PHP failed to load the extension (`undefined symbol: llama_model_n_embd_out`).

## [0.1.0] - 2026-10-20

### Added

- Initial release of the LlamaEmbed PHP extension
- Debian packages `php8.4-llamacpp` and `php8.5-llamacpp`, each enabled for every SAPI of that PHP
- Empty `php-llamacpp` package depending on the default PHP's module
