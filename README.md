# inference-cost-optimizer

A self-hosted cost optimizer for teams that serve their own LLMs. A gateway
records every request, a collector maps engine and GPU data into ClickHouse,
and a replay lab finds the cheaper setup on the team's own traffic.

`ico` is a codename until the product has a name.

## Status

| Build step | State |
| --- | --- |
| 1. Compose, ClickHouse schema, vLLM and GPU exporter services | Done |
| 2. Gateway: pass-through, streaming, records, gRPC export | Done |
| 3. Collector: gateway ingest, vLLM and GPU scrapers, OTLP receiver, trace-id join | Done |
| 4. Analyzer views and `ico report` | Next |
| 5. Replay lab | Not started |

Checked against real hardware so far: vLLM 0.10.1.1 on a GeForce RTX 2060 with
nvidia_gpu_exporter 1.15.1. Still to check on a rented datacenter GPU:

- dcgm-exporter. Its adapter is tested against a hand-written page
  (`py/tests/fixtures/dcgm-exporter.handwritten.metrics.txt`), not a real one.
- A current vLLM. Its metric names were read from the v0.31.0 source, not a
  running server.
- Prefill and decode time in engine spans. vLLM 0.10.1.1 leaves them out, so
  `prefill_ms` and `decode_ms` read 0 there.

## Layout

```text
gateway/   Go module: OpenAI-compatible pass-through, plus a fake engine for tests
py/        Python package `ico`: the collector today; analyzer, replay lab and report later
proto/     The gRPC contract between gateway and collector
schema/    ClickHouse migrations, applied by the collector at startup
deploy/    docker-compose.yml, example config, dcgm-exporter counters
scripts/   Toolchain wrappers
```

## Run it

Needs Docker. From `deploy/`:

```bash
cp .env.example .env            # then set the two secrets in it
cp config.example.yaml config.yaml
docker compose --profile vllm --profile dcgm up -d --build
```

On a GeForce card use `--profile geforce` instead of `--profile dcgm`, and
switch the exporter in `config.yaml` as its comment says.

Point clients at `http://<host>:8080/v1` instead of the engine. Tag requests
with `X-Tenant-Id` and `X-Feature` to get cost per customer and feature later.

An engine you run yourself needs two things to be traced: start it with
`--otlp-traces-endpoint grpc://<collector>:4317`, and set `OTEL_SERVICE_NAME`
to its replica name and `OTEL_EXPORTER_OTLP_TRACES_INSECURE=true`.

Without a GPU, two fake engines stand in for vLLM:

```bash
ICO_CONFIG_FILE=./config.mock.yaml docker compose --profile mock up -d --build
```

Look at what was recorded:

```bash
docker compose exec clickhouse sh -c 'clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" -d ico -q "SELECT ts, replica, tenant, prompt_tokens, completion_tokens, ttft_ms, e2e_ms FROM requests ORDER BY ts DESC LIMIT 10"'
```

## What the gateway does

- Proxies `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` and
  `/v1/models` to the configured replicas, round robin, skipping drained ones.
  A replica that refuses the connection is skipped for that request.
- Relays streams event by event. It turns on `stream_options.include_usage` so
  token counts arrive, and withholds the extra usage chunk from clients that
  did not ask for it.
- Sends a W3C `traceparent` and an `X-Request-Id` upstream so engine spans and
  logs join the gateway's record.
- Emits one record per request: ids, replica, model, tenant, feature, token
  counts, time to first token, end-to-end latency, inter-chunk gap p50 and p95,
  status, and prefix hashes.
- Never stores prompt or completion text. Request text is hashed in memory in
  256-character blocks (salted per install) and dropped; the hashes only show
  which requests share a prefix.
- Drops telemetry, never requests: export runs off the request path, and a
  slow or absent collector costs records, not latency.

Admin API on port 8081, bearer token from `ICO_ADMIN_TOKEN`:

```bash
curl -H "Authorization: Bearer $ICO_ADMIN_TOKEN" localhost:8081/admin/replicas
curl -H "Authorization: Bearer $ICO_ADMIN_TOKEN" -X POST localhost:8081/admin/replicas/r1/drain
curl -H "Authorization: Bearer $ICO_ADMIN_TOKEN" -X POST localhost:8081/admin/replicas/r1/undrain
curl -H "Authorization: Bearer $ICO_ADMIN_TOKEN" localhost:8081/admin/stats
```

A drained replica gets no new requests; it is idle once `inflight` reads 0.

## What the collector stores

| Table | Source | Holds |
| --- | --- | --- |
| `requests` | Gateway, over gRPC | One row per request |
| `engine_spans` | Engine OTLP traces, same gRPC port | Queue, prefill, decode and end-to-end time per request |
| `request_spans` (view) | The two above | Each request with its engine span, joined on trace id |
| `engine_metrics` | Each replica's `/metrics`, every 5 s | Running and waiting requests, KV cache use, token counters, latency histograms |
| `gpu_metrics` | dcgm-exporter or nvidia_gpu_exporter, every 5 s | Busy time, memory, power, and profiling counters where the card has them |

Engine and GPU metrics are stored under the collector's own names and units,
so a report reads the same across vLLM versions and exporters. Every GPU row
carries its telemetry level: L3 profiling counters, L2 basic counters, L1
memory and power only.

## Develop

The gateway builds and tests inside a container, so Go does not have to be
installed:

```bash
scripts/go.sh test ./...
CGO_ENABLED=1 scripts/go.sh test -race ./...
scripts/go.sh test -run '^$' -bench . ./internal/proxy
```

Two tests hold the gateway to its budget: `TestOverheadBudget` (under 2 ms
added at p50) and `TestNoRequestFailsWhileCollectorIsDown`.

The Python package uses [uv](https://docs.astral.sh/uv/):

```bash
cd py
uv sync
uv run pytest
```

Three of its tests need a ClickHouse server and skip without one. With the
Compose stack up, set `ICO_TEST_CLICKHOUSE_HOST=localhost`,
`ICO_TEST_CLICKHOUSE_USER=ico` and `ICO_TEST_CLICKHOUSE_PASSWORD` to run them.

After editing `proto/`, regenerate both sides with `scripts/gen-proto.sh`.
Migrations in `schema/` are applied once and recorded by file name: add a new
numbered file, never edit one that has shipped.
