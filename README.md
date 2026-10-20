# PHP extension for llama.cpp (embedding only currently)

A PHP extension that links against the llama.cpp library (`libllama`). It currently only implements embeddings: `LlamaEmbed` (load / embedBatch / dimension) for local GGUF models. Further llama.cpp features would not take much to add; embeddings is the only part implemented so far.

## Install

Debian packages are published from the [roojs package repositories](https://roojs.github.io/repos/). That page has the APT setup. Install the meta package; it depends on the module for the default PHP on that suite (`php8.4-llamacpp` or `php8.5-llamacpp`).

```bash
sudo apt-get install php-llamacpp
```

## Model files (not in this tree)

GGUF weights are not shipped with this extension. Pass any readable GGUF path to `LlamaEmbed::load()`.

Example: `/path/to/bge-m3-q4_k_m.gguf`

## Build

See [BUILD.md](BUILD.md).

## API

```php
$e = LlamaEmbed::load('/path/to/bge-m3-q4_k_m.gguf');
$dim = $e->dimension();
$vecs = $e->embedBatch([$a, $b]); // float[][], each L2-normalized
```

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
