#!/usr/bin/env bash
# Write debian/control from debian/control.in, one binary package per
# PHP version reported by phpquery: php8.4-llamacpp, php8.5-llamacpp, …
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

template="debian/control.in"
output="debian/control"

if [[ ! -f "${template}" ]]; then
  echo "missing ${template}" >&2
  exit 1
fi

mapfile -t versions < <(phpquery -V)
if ((${#versions[@]} == 0)); then
  echo "phpquery -V returned no PHP versions; install php-all-dev" >&2
  exit 1
fi

source_stanza="$(awk 'BEGIN { keep = 1 } /^Package:/ { exit } keep { print }' "${template}")"
package_stanza="$(awk 'BEGIN { keep = 0 } /^Package:/ { keep = 1 } keep { print }' "${template}")"

if [[ "${package_stanza}" != *'php@PHP_VERSION@-llamacpp'* ]]; then
  echo "${template} is missing Package: php@PHP_VERSION@-llamacpp" >&2
  exit 1
fi

{
  printf '%s\n\n' "${source_stanza}"
  for ver in "${versions[@]}"; do
    printf '%s\n' "${package_stanza}" | sed "s/@PHP_VERSION@/${ver}/g"
    echo
  done
} > "${output}"

echo "Wrote ${output}:"
grep '^Package:' "${output}"
