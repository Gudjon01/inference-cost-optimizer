#!/usr/bin/env bash
# Runs the Go toolchain for the gateway inside a container, so the host needs
# Docker but not Go.   scripts/go.sh test ./...
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

# Git Bash rewrites /paths in arguments unless told not to.
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$root:/src" \
  -v ico-go-mod:/go/pkg/mod \
  -v ico-go-build:/root/.cache/go-build \
  -w /src/gateway \
  -e CGO_ENABLED="${CGO_ENABLED:-0}" \
  golang:1.27 go "$@"
