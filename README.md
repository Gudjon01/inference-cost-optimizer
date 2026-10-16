# ico

**ico**, the inference cost optimizer, finds out what your self-hosted LLMs
cost per million tokens, how much of that pays for idle GPUs, and which engine
config serves the same traffic for less, tested on a replay of your own
traffic, on your own machines.

Documentation: [www.inference-cost-optimizer.com](https://www.inference-cost-optimizer.com).

**From a real run.** An RTX 2060 serves Qwen3-0.6B on vLLM, capped at
`--max-num-seqs 8`. `ico replay` took five minutes of traffic the gateway had
recorded, rebuilt it as synthetic prompts of the same shape, and tested the
current config and three others in 19 minutes:

| Config | Requests/s within the limits | Per million tokens |
| --- | ---: | ---: |
| `--max-num-seqs 8`, the starting config | 2.56 | $0.0218 |
| `--max-num-seqs 16` | 10.69 | $0.0052 |
| `--max-num-seqs 256`, vLLM's default | 12.18 | $0.0046 |
| `--max-num-seqs 64` | 15.12 | $0.0037 |

$0.0218 to $0.0037 per million tokens: 83% less, at the same latency limits
(p95 time to first token under 1 s, p95 gap between tokens under 50 ms). The
starting cap was chosen for this demo; against vLLM's own default, 64 is still
about 20% cheaper on this traffic. Costs assume $0.10 per GPU-hour, and the
ratios do not depend on the price.

## What it does

- **Gateway** (Go): an OpenAI-compatible pass-through in front of your engine
  replicas. It writes one record per request, with token counts, latency,
  tenant, feature and salted prefix hashes, and never the prompt or the
  completion. It adds about 0.2 ms at p50. When its collector is down it keeps
  serving and drops telemetry, never requests.
- **Collector** (Python): puts the gateway's records, the engine's metrics and
  traces, and the GPU exporters' counters into ClickHouse.
- **Report** (`ico report`): cost per million tokens, the share of GPU cost
  that paid for idle time, cost per model and per tenant, and the replay run
  on that traffic, as one static HTML file.
- **Replay lab** (`ico replay`): restarts the engine under each candidate
  config on a drained replica, raises your recorded traffic shape until a
  latency limit breaks, and names the cheapest config that holds. It puts the
  original back whatever happens.

All of it runs inside your network, and no text leaves it: the replay's
prompts are synthetic, rebuilt from token counts and prefix hashes.

## Status

Works today with vLLM on NVIDIA GPUs, deployed with Docker Compose or as plain
processes. It has run on real hardware on two cards:

| | Engine | GPU telemetry | What ran |
| --- | --- | --- | --- |
| H100 SXM, RunPod, 2026-10-14 and 16 | vLLM 0.31.0, Qwen3-8B | dcgm-exporter 4.5.2-4.8.1, with profiling counters (level L3) | 16 minutes of four simulated tenants, the report, and two 20-minute replays of that traffic |
| GeForce RTX 2060 | vLLM 0.10.1.1, Qwen3-0.6B | nvidia_gpu_exporter 1.15.1 (level L2) | Recorded traffic, the report, and the replays above |

On the H100, everything the 2060 could not show worked: DCGM's profiling
counters, without the `SYS_ADMIN` capability; vLLM 0.31's metric names; and
prefill and decode time in its engine spans, for every request, so the tenant
split followed engine time. The collector, ClickHouse and the report ran on
another machine, across an SSH tunnel. The dcgm-exporter adapter's test
fixture (`py/tests/fixtures/dcgm-exporter.handwritten.metrics.txt`) is still
the hand-written one.

Still to check on real hardware: successive halving. On the H100, a full test
of the configs fit the budget, so the replay never needed it.

The first H100 replay found a bug in the replay itself. It rebuilt each
conversation's own history the same on every pass through its sample, so from
the second pass on nearly every prompt was a prefix-cache hit: 84% by vLLM's
count, against 38-50% on the live traffic, and the current config measured 43
requests/s. With the fix (#5), the same recorded traffic replayed two days
later kept the cache at 36-39% and gave:

| Config, on the H100 | Requests/s within the limits | Per million tokens |
| --- | ---: | ---: |
| vLLM's defaults, the current config | 22.5 | $0.0237 |
| `--max-num-batched-tokens 16384` | 22.5 | $0.0237 |
| `--max-num-batched-tokens 2048` | 22.5 | $0.0237 |
| `--kv-cache-dtype fp8`, quality unverified | 22.5 | $0.0237 |
| `--no-enable-prefix-caching` | 15.9 | $0.0335 |

The live traffic had been comfortable at about 22 requests/s and queued near
28, so the replay now agrees with it. On this traffic vLLM's defaults were
already the cheapest config that holds, and turning off the prefix cache would
make each token 41% dearer. Same limits as above, at $3.99 per GPU-hour.

Not built yet: SGLang and AMD adapters, Kubernetes and Helm, live-mirror
replay, and a check of answer quality. Until that check exists, configs that
change precision, such as FP8, are measured but cannot win.

## Layout

```text
gateway/   Go module: OpenAI-compatible pass-through, plus a fake engine for tests
py/        Python package `ico`: collector, analyzer, report and replay lab
proto/     The gRPC contract between gateway and collector
schema/    ClickHouse migrations, applied by the collector at startup
deploy/    docker-compose.yml, example config, dcgm-exporter counters
scripts/   Toolchain wrappers
```

## Run it

This runs everything on one machine. To install on rented GPU machines, with
ClickHouse and the collector on a separate small host, follow the
[install guide](https://www.inference-cost-optimizer.com/install/overview).

Needs Docker and git. Take the latest release:

```bash
git clone --branch v0.1.1 https://github.com/Gudjon01/inference-cost-optimizer.git
cd inference-cost-optimizer/deploy
```

Then, from `deploy/`:

```bash
cp .env.example .env            # then set the two secrets in it
cp config.example.yaml config.yaml
docker compose --profile vllm --profile dcgm up -d
```

Compose pulls ico's two images, `ghcr.io/gudjon01/ico-gateway` and
`ghcr.io/gudjon01/ico-collector`, for amd64 or arm64. Add `--build` to build
them from the checkout instead.

On a GeForce card use `--profile geforce` instead of `--profile dcgm`, and
switch the exporter in `config.yaml` as its comment says.

Point clients at `http://<host>:8080/v1` instead of the engine. Tag requests
with `X-Tenant-Id` and `X-Feature` to get cost per customer and feature later.

An engine you run yourself needs two things to be traced: start it with
`--otlp-traces-endpoint grpc://<collector>:4317`, and set `OTEL_SERVICE_NAME`
to its replica name and `OTEL_EXPORTER_OTLP_TRACES_INSECURE=true`.

A machine without Docker, such as a RunPod pod, runs the gateway as one static
binary from the [release](https://github.com/Gudjon01/inference-cost-optimizer/releases/latest):

```bash
curl -fLO https://github.com/Gudjon01/inference-cost-optimizer/releases/download/v0.1.1/ico-gateway-linux-amd64
curl -fLO https://github.com/Gudjon01/inference-cost-optimizer/releases/download/v0.1.1/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
install -m 755 ico-gateway-linux-amd64 gateway
ICO_ADMIN_TOKEN=... ./gateway -config gateway.yaml
```

Without a GPU, two fake engines stand in for vLLM:

```bash
ICO_CONFIG_FILE=./config.mock.yaml docker compose --profile mock up -d
```

Look at what was recorded:

```bash
docker compose exec clickhouse sh -c 'clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" -d ico -q "SELECT ts, replica, tenant, prompt_tokens, completion_tokens, ttft_ms, e2e_ms FROM requests ORDER BY ts DESC LIMIT 10"'
```

## Get the report

```bash
docker compose exec collector ico report --window 24h --out /tmp/report.html
docker compose cp collector:/tmp/report.html report.html
```

`report.html` is one static file with no scripts and nothing loaded from
elsewhere. It shows, for the window:

- cost per million tokens, and what it would be without idle time;
- serving and idle cost per model, and idle GPU-hours;
- where the GPUs report profiling counters, what was paid for compute cores
  that had nothing to run while requests were being served;
- the split of each replica's cost across tenants;
- the replay run on that traffic: the current config against the cheapest one
  that held the latency limits;
- the telemetry level of each host, and the inputs every number came from.

How the numbers are made:

- **Cost** is GPUs x price x hours, from the first to the last sign of a
  replica in the window. Idle hours are billed like any other. Prices come
  from `gpu_price_per_hour` in the config.
- **Idle** is time with no request in flight: paid for, nothing to serve.
  The rest is **serving** time.
- **Cores unused while serving** is serving time minus the time DCGM's
  profiling counters show the compute cores active, on hosts that report them
  and run one replica. Requests were in flight, but part of the cores had
  nothing to run: small batches, waits between engine steps. On the H100 run,
  6% of the cost was idle and another 9% paid for unused cores.
- **Tenant split** follows engine time where spans report prefill and decode
  time, so prompt and completion tokens weigh what they took. Otherwise it
  follows token count.

The views behind it take the window as parameters and can be queried
directly, for example
`SELECT * FROM usage_by_tenant(since = '2026-10-14 00:00:00', until = '2026-10-15 00:00:00')`.
They are defined in `schema/002_analyzer.sql`.

## Find a cheaper config

`ico replay` tests engine configs against the traffic the gateway recorded and
names the cheapest one that holds the latency limits. Run it on the GPU host
of the replica, from `deploy/`, with the CLI installed (`pip install ./py`):

```bash
ico replay --config config.yaml --replica r1 --configs sweep.yaml --window 1h --slo ttft_p95=500ms,itl_p95=50ms --budget 45m
```

It needs `ICO_ADMIN_TOKEN` in the environment, and it has to reach ClickHouse
and the engine from where it runs: set `ICO_CLICKHOUSE_HOST` and pass
`--replica-url` when their addresses in the config file only resolve inside
Compose.

What a run does, in order:

1. Takes a run of 300 consecutive recorded requests that looks like the whole
   window in prompt and completion lengths, shared prefixes and burstiness.
2. Rebuilds them as prompts of token ids: exactly the recorded lengths, and
   the same leading tokens wherever the recorded prefix hashes matched. No
   recorded text exists to replay, and none is needed.
3. Drops configs that cannot fit in GPU memory on paper, before any restart.
4. Drains the replica at the gateway.
5. Restarts the engine on each config, the current one first, and raises the
   request rate until a latency limit breaks. The highest rate that held gives
   the cost of a million tokens on that config. Every config, the current one
   included, is measured on a freshly started engine, so they are compared
   under the same conditions.
6. Restores the original flags and undrains the replica.

A config is recommended only if it measures at least 10% cheaper than the
current one (`--min-saving`). One measurement of a config varies from run to
run: the same config measured twice on the RTX 2060 held rates 8.5% apart.
Configs that measure cheaper by less are listed as within that variation, not
as savings.

Once the current config is measured it knows how long a load and a test take,
prints its plan, and waits for a yes (`--yes` skips it). When a full test of
every config fits the budget, every config gets one. When it does not but
successive halving does, every config gets a short run, the better half a
longer one, and the final two a full test; only a full test can win. Short
runs use shorter steps and fewer requests per step, which makes halving pay
off when there are many configs and loading one is quick next to testing it.

Pick latency limits with room above what the traffic sees today. The command
prints the time to first token recorded in the window: when that is close to
the limit, whether a rate holds comes down to a few requests in each step, and
the rate found swings from run to run.

The replica is put back whatever happens in between: a config that crashes, a
failed health check, Ctrl-C. A watchdog process does the same if the command
itself is killed. A replica whose original config will not start again stays
drained, so it gets no traffic. One replay runs per replica at a time, and
draining the only replica needs `--single-replica`.

How the engine is restarted is set per replica under `launcher` in the config
file:

- `docker` runs each config in a copy of the engine's container: same image,
  GPUs, volumes, environment, ports and networks, other engine flags. It reads
  the flags in production from the running container. The original container
  is only stopped, never changed, so putting it back is starting it again.
  It needs access to the Docker socket.
- `command` runs your own start and stop commands, with `{flags}` filled in.
- `manual` prints the flags and waits for you to restart the engine.

`sweep.example.yaml` shows the sweep file. Configs that change precision, such as FP8, are measured but
cannot win until answer quality is checked.

To try it without a GPU, start the `mock` profile, send some traffic through
the gateway, then from `deploy/`:

```bash
ico replay --config config.mock.yaml --replica mock-1 --configs sweep.mock.yaml --replica-url http://localhost:8001 --step-seconds 3 --min-step-requests 15
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

Some of its tests need a ClickHouse server and skip without one. With the
Compose stack up, set `ICO_TEST_CLICKHOUSE_HOST=localhost`,
`ICO_TEST_CLICKHOUSE_USER=ico` and `ICO_TEST_CLICKHOUSE_PASSWORD` to run them.

After editing `proto/`, regenerate both sides with `scripts/gen-proto.sh`.
Migrations in `schema/` are applied once and recorded by file name: add a new
numbered file, never edit one that has shipped.

## Questions and help

[Open an issue](https://github.com/Gudjon01/inference-cost-optimizer/issues/new/choose):
a bug, a number that looks wrong, a question, or a request for help
installing it. With the install-help template we set ico up with you, on your
machines. Never paste secrets or your users' prompts; ico never needs them.

## License

Apache 2.0. See `LICENSE`.
