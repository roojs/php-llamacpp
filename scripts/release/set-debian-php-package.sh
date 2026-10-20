#!/usr/bin/env bash
# Write debian/control from debian/control.in.
# One real binary package per PHP version (php8.4-llamacpp, …) and an
# unversioned php-llamacpp that depends on the default PHP's package.
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

default_ver="$(php-config --version | sed -n 's/^\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
if [[ -z "${default_ver}" ]]; then
  echo "could not read the default PHP version from php-config --version" >&2
  exit 1
fi

found=0
for ver in "${versions[@]}"; do
  if [[ "${ver}" == "${default_ver}" ]]; then
    found=1
    break
  fi
done
if [[ "${found}" -ne 1 ]]; then
  echo "default PHP ${default_ver} is not in phpquery -V: ${versions[*]}" >&2
  exit 1
fi

source_stanza="$(awk 'BEGIN { keep = 1 } /^Package:/ { exit } keep { print }' "${template}")"
alias_stanza="$(awk '
  /^Package: php-llamacpp$/ { keep = 1 }
  /^Package: php@PHP_VERSION@-llamacpp$/ { keep = 0 }
  keep { print }
' "${template}")"
versioned_stanza="$(awk '
  /^Package: php@PHP_VERSION@-llamacpp$/ { keep = 1 }
  keep { print }
' "${template}")"

if [[ "${alias_stanza}" != *'php@PHP_DEFAULT@-llamacpp'* ]]; then
  echo "${template} is missing the php-llamacpp alias dependency" >&2
  exit 1
fi
if [[ "${versioned_stanza}" != *'php@PHP_VERSION@-llamacpp'* ]]; then
  echo "${template} is missing Package: php@PHP_VERSION@-llamacpp" >&2
  exit 1
fi

{
  printf '%s\n\n' "${source_stanza}"
  printf '%s\n' "${alias_stanza}" | sed "s/@PHP_DEFAULT@/${default_ver}/g"
  echo
  for ver in "${versions[@]}"; do
    printf '%s\n' "${versioned_stanza}" | sed "s/@PHP_VERSION@/${ver}/g"
    echo
  done
} > "${output}"

echo "Wrote ${output}:"
grep '^Package:\|^Depends:' "${output}"
