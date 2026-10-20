# Building php-llamacpp

## Dependencies

- PHP dev headers for the version you are building (`php8.4-dev`, `php8.5-dev`, or `php-all-dev`)
- `build-essential`, `pkgconf`
- `libllama-dev` (and the ggml backends you want at runtime: vulkan, blas)

`libllama` is not in every distro archive. `./scripts/fetch-libllama.sh` adds the [roojs APT repository](https://roojs.github.io/repos/) and installs `libllama` and `libggml` for the current suite.

Do not copy `llamacpp.so` onto a machine with a different PHP. The `.so` is built against that PHP's module API.

## Extension

```bash
cd php-llamacpp
phpize
./configure
make -j$(nproc)
```

Output: `modules/llamacpp.so`

```bash
sudo make install
```

Or load it without installing:

```bash
php -d extension=$PWD/modules/llamacpp.so test_llamacpp.php \
  /path/to/bge-m3-q4_k_m.gguf
```

## Debian packages

`debian/control.in` is the template. The build writes `debian/control` with one module package per PHP version reported by `phpquery` (`php8.4-llamacpp`, `php8.5-llamacpp`, …) and an empty `php-llamacpp` package that depends on the default PHP's module.

```bash
sudo apt-get install -y build-essential dpkg-dev debhelper dh-php php-all-dev pkgconf
./scripts/release/set-debian-php-package.sh
dpkg-buildpackage -us -uc -b
```

`dh-php` installs `llamacpp.so` and `llamacpp.ini`, and the maintainer scripts run `php_invoke enmod <version> ALL llamacpp`.

GitHub Actions runs that build for Debian 13 and Ubuntu 25.04, 25.10, and 26.04, and attaches the `.deb` files to the version's GitHub Release. Published builds so far are amd64 because that is the architecture those jobs use.
