#!/bin/bash
set -euo pipefail
# Install libllama/libggml from the roojs APT repo:
#   https://roojs.github.io/repos/
# That archive republishes the newest Debian pool build that fits each
# suite's libc. Grabbing sid-latest directly fails on Ubuntu 25.04
# (plucky, libc 2.41): those debs need libc6 >= 2.43 and apt removes them.
#
# Installs core packages plus vulkan and blas backends. Never install
# libggml0-backend-hip on ROCm — it breaks the stack. Debian has no NVIDIA
# CUDA backend package.

run() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

suite="${ROOJS_APT_SUITE:-}"
if [ -z "$suite" ]; then
  suite="$(lsb_release -cs 2>/dev/null || true)"
fi
if [ -z "$suite" ] && [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  suite="${VERSION_CODENAME:-}"
fi
if [ -z "$suite" ]; then
  echo "Set ROOJS_APT_SUITE (trixie, plucky, questing, or resolute) or install lsb-release." >&2
  exit 1
fi

case "$suite" in
  trixie|plucky|questing|resolute) ;;
  *)
    echo "Suite '${suite}' is not published at https://roojs.github.io/repos/" >&2
    echo "Use Debian 13 (trixie) or Ubuntu 25.04+ (plucky/questing/resolute)." >&2
    exit 1
    ;;
esac

run apt-get update
run apt-get install -y --no-install-recommends ca-certificates curl gnupg
run install -d -m 0755 /etc/apt/keyrings
run rm -f /etc/apt/keyrings/roojs.gpg
curl -fsSL https://roojs.github.io/repos/key.gpg \
  | run gpg --batch --yes --dearmor -o /etc/apt/keyrings/roojs.gpg
arch="$(dpkg --print-architecture 2>/dev/null || true)"
if [ -z "$arch" ]; then
  case "$(uname -m)" in
    x86_64) arch="amd64" ;;
    aarch64) arch="arm64" ;;
    *) arch="$(uname -m)" ;;
  esac
fi
curl -fsSL https://roojs.github.io/repos/sources \
  | sed "s/@suite@/${suite}/" \
  | sed "/^Architectures:/d" \
  | run tee /etc/apt/sources.list.d/roojs.sources >/dev/null
printf 'Architectures: %s\n' "${arch}" \
  | run tee -a /etc/apt/sources.list.d/roojs.sources >/dev/null
# Ubuntu/Debian stay preferred when a name exists in both archives.
# Packages that only we ship (libllama, libggml) still install when named.
run tee /etc/apt/preferences.d/roojs >/dev/null <<'EOF'
Package: *
Pin: origin roojs.github.io
Pin-Priority: 100
EOF
run chmod 0644 /etc/apt/keyrings/roojs.gpg \
  /etc/apt/sources.list.d/roojs.sources \
  /etc/apt/preferences.d/roojs
run apt-get update

has_rocm() {
  [ -e /dev/kfd ] || command -v rocminfo >/dev/null 2>&1
}

if has_rocm; then
  echo "ROCm detected; skipping libggml0-backend-hip (breaks ROCm)."
  if dpkg -l libggml0-backend-hip 2>/dev/null | grep -q '^ii'; then
    echo "Removing installed libggml0-backend-hip."
    run apt-get remove -y libggml0-backend-hip
  fi
else
  echo "No ROCm detected; skipping HIP backend."
fi

echo "Installing libllama/libggml for suite ${suite} from https://roojs.github.io/repos/"
run apt-get install -y \
  libggml0 \
  libggml-dev \
  libllama0 \
  libllama-dev \
  libggml0-backend-vulkan \
  libggml0-backend-blas

MULTIARCH="$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || true)"
if [ -z "$MULTIARCH" ]; then
  case "$arch" in
    amd64) MULTIARCH="x86_64-linux-gnu" ;;
    arm64) MULTIARCH="aarch64-linux-gnu" ;;
    *) MULTIARCH="${arch}-linux-gnu" ;;
  esac
fi

LLAMA_LIBDIR="/usr/lib/${MULTIARCH}/llama"
LD_CONF="/etc/ld.so.conf.d/llama.conf"
if [ ! -f "$LD_CONF" ] || ! grep -qF "$LLAMA_LIBDIR" "$LD_CONF" 2>/dev/null; then
  echo "Registering ${LLAMA_LIBDIR} with ldconfig (libllama installs outside default search path)."
  echo "$LLAMA_LIBDIR" | run tee "$LD_CONF" >/dev/null
  run ldconfig
fi
