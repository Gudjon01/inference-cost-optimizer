-- Analyzer views. Each takes the report window as parameters and is queried as
--   SELECT * FROM usage_by_tenant(since = '2026-10-14 00:00:00', until = '2026-10-15 00:00:00')
-- The window is half-open: since <= ts < until. The views only count and sum;
-- prices live in the config file, so the dollar math is done by the analyzer.

-- Requests and tokens per replica, model and tenant. Requests the gateway
-- answered itself (no replica reachable) are left out: no GPU served them.
CREATE VIEW IF NOT EXISTS usage_by_tenant AS
SELECT
    replica,
    model,
    tenant,
    count()                AS requests,
    countIf(status >= 400) AS errors,
    sum(prompt_tokens)     AS prompt_tokens,
    sum(completion_tokens) AS completion_tokens,
    sum(cached_tokens)     AS cached_tokens
FROM requests
WHERE ts >= {since:DateTime64(6, 'UTC')} AND ts < {until:DateTime64(6, 'UTC')} AND replica != ''
GROUP BY replica, model, tenant;

-- Seconds each replica had at least one request in flight, and the span of
-- time its requests cover. Overlapping requests count once. A request that
-- began up to an hour before the window still counts for the part inside it.
CREATE VIEW IF NOT EXISTS replica_busy AS
WITH
    toUnixTimestamp64Micro({since:DateTime64(6, 'UTC')}) AS window_start,
    toUnixTimestamp64Micro({until:DateTime64(6, 'UTC')}) AS window_end,
    toUnixTimestamp64Micro(ts)                           AS started,
    started + toInt64(e2e_ms * 1000)                     AS ended
SELECT
    replica,
    intervalLengthSum(greatest(started, window_start), least(ended, window_end)) / 1e6 AS busy_seconds,
    fromUnixTimestamp64Micro(min(greatest(started, window_start)), 'UTC')             AS first_seen,
    fromUnixTimestamp64Micro(max(least(ended, window_end)), 'UTC')                    AS last_seen
FROM requests
WHERE ts >= {since:DateTime64(6, 'UTC')} - INTERVAL 1 HOUR AND ts < {until:DateTime64(6, 'UTC')}
  AND ended > window_start AND replica != ''
GROUP BY replica;

-- Engine time per replica, by phase, with the tokens of the same requests.
-- Only requests whose span reports phase times count, so time and tokens
-- always describe the same set. requests and with_span say how much of the
-- traffic the engine traced at all.
CREATE VIEW IF NOT EXISTS replica_phases AS
WITH (prefill_ms > 0 OR decode_ms > 0) AS has_phases
SELECT
    replica,
    count()                              AS requests,
    countIf(has_span)                    AS with_span,
    sumIf(prefill_ms, has_phases)        AS phase_prefill_ms,
    sumIf(decode_ms, has_phases)         AS phase_decode_ms,
    sumIf(prompt_tokens, has_phases)     AS phase_prompt_tokens,
    sumIf(completion_tokens, has_phases) AS phase_completion_tokens
FROM request_spans
WHERE ts >= {since:DateTime64(6, 'UTC')} AND ts < {until:DateTime64(6, 'UTC')} AND replica != ''
GROUP BY replica;

-- When each replica's engine was heard from, and the model it reports serving.
CREATE VIEW IF NOT EXISTS replica_engine AS
SELECT
    replica,
    min(ts)                                                  AS first_seen,
    max(ts)                                                  AS last_seen,
    anyIf(labels['model_name'], labels['model_name'] != '')  AS served_model
FROM engine_metrics
WHERE ts >= {since:DateTime64(3, 'UTC')} AND ts < {until:DateTime64(3, 'UTC')}
GROUP BY replica;

-- GPU telemetry per host: how many cards, at which telemetry level, and the
-- window average of each counter. An average is NaN when the host never
-- reported that counter.
CREATE VIEW IF NOT EXISTS host_gpus AS
SELECT
    host,
    uniqExact(gpu_uuid)                               AS gpus,
    arraySort(groupUniqArray(gpu_model))              AS gpu_models,
    min(level)                                        AS level,
    min(ts)                                           AS first_seen,
    max(ts)                                           AS last_seen,
    avgIf(value, metric = 'sm_active_ratio')          AS sm_active,
    avgIf(value, metric = 'tensor_active_ratio')      AS tensor_active,
    avgIf(value, metric = 'dram_active_ratio')        AS dram_active,
    avgIf(value, metric = 'util_ratio')               AS util,
    maxIf(value, metric = 'memory_used_bytes')        AS memory_used_peak_bytes,
    avgIf(value, metric = 'power_watts')              AS power_watts
FROM gpu_metrics
WHERE ts >= {since:DateTime64(3, 'UTC')} AND ts < {until:DateTime64(3, 'UTC')}
GROUP BY host;
