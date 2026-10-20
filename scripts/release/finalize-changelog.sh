#!/usr/bin/env bash
# Copyright (c) 2026 Roo J Solutions Ltd.
# Author: Leon <leon@roojs.com>
# Licensed under the PHP License, version 3.01. See LICENSE.
# Promote [Unreleased] to a versioned section after a successful GitHub release.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <tag>   e.g. v0.1.0" >&2
  exit 1
fi

exec "$ROOT/scripts/release/changelog.sh" finalize "$1"
