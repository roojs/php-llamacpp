#!/usr/bin/env bash
# Copyright (c) 2026 Roo J Solutions Ltd.
# Author: Leon <leon@roojs.com>
# Licensed under the PHP License, version 3.01. See LICENSE.
# Generate debian/changelog from CHANGELOG.md (single source of truth).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$ROOT/scripts/release/changelog.sh" sync "$@"
