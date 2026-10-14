-- Tables are created in the database named in the config file (clickhouse.database).
-- Raw rows expire after 30 days; partitions are daily so expiry drops whole parts.

-- One row per request the gateway proxied.
CREATE TABLE IF NOT EXISTS requests
(
    ts                DateTime64(6, 'UTC'),
    request_id        String,
    trace_id          String,
    replica           LowCardinality(String),
    model             LowCardinality(String),
    tenant            String,
    feature           LowCardinality(String),
    endpoint          LowCardinality(String),
    prompt_tokens     UInt32,
    completion_tokens UInt32,
    cached_tokens     UInt32,
    ttft_ms           Float32,
    e2e_ms            Float32,
    gap_p50_ms        Float32,
    gap_p95_ms        Float32,
    status            UInt16,
    stream            Bool,
    max_tokens        UInt32,
    prompt_chars      UInt32,
    prefix_hashes     Array(UInt64) CODEC(ZSTD(3))
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (replica, ts)
TTL toDateTime(ts) + INTERVAL 30 DAY
SETTINGS ttl_only_drop_parts = 1;

-- One row per engine span (vLLM OTLP traces). trace_id joins requests.
CREATE TABLE IF NOT EXISTS engine_spans
(
    ts                DateTime64(6, 'UTC'),
    trace_id          String,
    span_id           String,
    replica           LowCardinality(String),
    queue_ms          Float32,
    prefill_ms        Float32,
    decode_ms         Float32,
    e2e_ms            Float32,
    ttft_ms           Float32,
    prompt_tokens     UInt32,
    completion_tokens UInt32
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (trace_id, ts)
TTL toDateTime(ts) + INTERVAL 30 DAY
SETTINGS ttl_only_drop_parts = 1;

-- Every request with its engine span, joined on the trace id the gateway
-- started. has_span is false when the engine sent no span for the request.
CREATE VIEW IF NOT EXISTS request_spans AS
SELECT
    r.*,
    s.queue_ms,
    s.prefill_ms,
    s.decode_ms,
    s.e2e_ms AS engine_e2e_ms,
    s.ttft_ms AS engine_ttft_ms,
    s.span_id != '' AS has_span
FROM requests AS r
LEFT ANY JOIN engine_spans AS s ON r.trace_id = s.trace_id;

-- One row per scraped engine metric sample, under names that do not depend on
-- the engine or its version (see py/src/ico/collector/adapters).
CREATE TABLE IF NOT EXISTS engine_metrics
(
    ts      DateTime64(3, 'UTC') CODEC(Delta, ZSTD),
    replica LowCardinality(String),
    metric  LowCardinality(String),
    labels  Map(LowCardinality(String), String),
    value   Float64 CODEC(Gorilla, ZSTD)
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (replica, metric, ts)
TTL toDateTime(ts) + INTERVAL 30 DAY
SETTINGS ttl_only_drop_parts = 1;

-- One row per scraped GPU metric sample, in the same names and units whichever
-- exporter it came from. level is the telemetry level of the source:
-- 3 profiling counters, 2 basic counters, 1 memory and power only.
CREATE TABLE IF NOT EXISTS gpu_metrics
(
    ts        DateTime64(3, 'UTC') CODEC(Delta, ZSTD),
    host      LowCardinality(String),
    gpu_uuid  LowCardinality(String),
    gpu_model LowCardinality(String),
    metric    LowCardinality(String),
    value     Float64 CODEC(Gorilla, ZSTD),
    level     UInt8
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (host, gpu_uuid, metric, ts)
TTL toDateTime(ts) + INTERVAL 30 DAY
SETTINGS ttl_only_drop_parts = 1;

-- One row per replay run, rewritten as the run progresses: read it with FINAL.
-- Kept without a TTL: runs are the product's output, not raw telemetry.
CREATE TABLE IF NOT EXISTS replay_runs
(
    run_id       UUID,
    trigger      LowCardinality(String),
    mode         LowCardinality(String),
    replica      LowCardinality(String),
    window_start DateTime64(3, 'UTC'),
    window_end   DateTime64(3, 'UTC'),
    slo          String,
    budget_s     UInt32,
    started_at   DateTime64(3, 'UTC'),
    finished_at  Nullable(DateTime64(3, 'UTC')),
    status       LowCardinality(String)
)
ENGINE = ReplacingMergeTree
ORDER BY run_id;

-- One row per config tested in a run.
CREATE TABLE IF NOT EXISTS replay_results
(
    run_id        UUID,
    config_id     String,
    stage         UInt8,
    config        String,
    rate_rps      Float32,
    tokens_s      Float32,
    ttft_p95_ms   Float32,
    itl_p95_ms    Float32,
    error_rate    Float32,
    cost_per_mtok Float64,
    held_slo      Bool
)
ENGINE = MergeTree
ORDER BY (run_id, stage, config_id);
