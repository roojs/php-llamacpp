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
  echo "This release includes the \`php-llama\` packages built for Debian 13 (trixie) and Ubuntu 25.04, 25.10, and 26.04 (plucky, questing, resolute). Each \`.deb\` is built on that suite against its PHP, and \`phpenmod\` enables \`llama\` for every SAPI."
  echo
  echo "The [roojs package repositories](https://roojs.github.io/repos/) can pull this release in later."
  echo
  echo "---"
  echo
  cat "$OUTPUT"
} > "$tmp"
mv "$tmp" "$OUTPUT"
