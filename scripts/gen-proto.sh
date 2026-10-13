#!/usr/bin/env bash
# Regenerates the Go and Python code for proto/. The output is committed, so
# this only needs to run after a .proto file changes.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
proto=ico/telemetry/v1/telemetry.proto

# Go: protoc and its plugins live only in this throwaway container.
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$root:/src" \
  -v ico-go-mod:/go/pkg/mod \
  -w /src \
  golang:1.27 sh -ec '
    apt-get update -qq && apt-get install -y -qq protobuf-compiler >/dev/null
    go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.12
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.6.2
    protoc -I proto \
      --go_out=gateway/internal/telemetrypb --go_opt=paths=source_relative \
      --go-grpc_out=gateway/internal/telemetrypb --go-grpc_opt=paths=source_relative \
      '"$proto"'
    # paths=source_relative mirrors the proto tree; the Go package is flat.
    mv gateway/internal/telemetrypb/ico/telemetry/v1/*.go gateway/internal/telemetrypb/
    rm -r gateway/internal/telemetrypb/ico
  '

# Python: grpcio-tools ships its own protoc.
(cd "$root/py" && uv run python -m grpc_tools.protoc -I ../proto \
  --python_out=src --pyi_out=src --grpc_python_out=src "$proto")
