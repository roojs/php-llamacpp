#!/usr/bin/env bash
# Copyright (c) 2026 Roo J Solutions Ltd.
# Author: Leon <leon@roojs.com>
# Licensed under the PHP License, version 3.01. See LICENSE.
# Write GitHub release notes from CHANGELOG.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <tag> [output-file]" >&2
  exit 1
fi

TAG="$1"
OUTPUT="${2:-$ROOT/release-notes.md}"
"$ROOT/scripts/release/changelog.sh" release-notes "$TAG" -o "$OUTPUT"

tmp="$(mktemp)"
{
  echo "This release includes \`php8.4-llamacpp\` (Debian 13 trixie, Ubuntu 25.04 plucky, Ubuntu 25.10 questing) and \`php8.5-llamacpp\` (Ubuntu 26.04 resolute), plus an empty \`php-llamacpp\` package that depends on the default PHP's module. \`phpenmod\` enables \`llamacpp\` for every SAPI."
  echo
  echo "The [roojs package repositories](https://roojs.github.io/repos/) can pull this release in later."
  echo
  echo "---"
  echo
  cat "$OUTPUT"
} > "$tmp"
mv "$tmp" "$OUTPUT"
