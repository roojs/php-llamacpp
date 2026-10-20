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

repo="${GITHUB_REPOSITORY:-roojs/php-llamacpp}"
tmp="$(mktemp)"
{
  echo "Debian packages for this version are published per suite:"
  echo
  echo "- [\`${TAG}-trixie\`](https://github.com/${repo}/releases/tag/${TAG}-trixie) (Debian 13)"
  echo "- [\`${TAG}-plucky\`](https://github.com/${repo}/releases/tag/${TAG}-plucky) (Ubuntu 25.04)"
  echo "- [\`${TAG}-questing\`](https://github.com/${repo}/releases/tag/${TAG}-questing) (Ubuntu 25.10)"
  echo "- [\`${TAG}-resolute\`](https://github.com/${repo}/releases/tag/${TAG}-resolute) (Ubuntu 26.04)"
  echo
  echo "Each \`.deb\` is built on that suite against its PHP, and \`phpenmod\` enables \`llama\` for every SAPI. The [roojs package repositories](https://roojs.github.io/repos/) can pull these releases in later."
  echo
  echo "---"
  echo
  cat "$OUTPUT"
} > "$tmp"
mv "$tmp" "$OUTPUT"
