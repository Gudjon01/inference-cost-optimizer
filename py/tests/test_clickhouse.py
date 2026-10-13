"""Runs against a real ClickHouse. Skipped unless ICO_TEST_CLICKHOUSE_HOST is set:

    ICO_TEST_CLICKHOUSE_HOST=localhost ICO_TEST_CLICKHOUSE_USER=ico \
    ICO_TEST_CLICKHOUSE_PASSWORD=... uv run pytest tests/test_clickhouse.py
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ico.collector import store
from ico.collector.ingest import REQUESTS_COLUMNS, to_row
from ico.config import ClickHouse
from tests.test_ingest import sample_record

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"

pytestmark = [
    pytest.mark.clickhouse,
    pytest.mark.skipif("ICO_TEST_CLICKHOUSE_HOST" not in os.environ, reason="no ClickHouse to test against"),
]


@pytest.fixture
def cfg():
    """A throwaway database, dropped afterwards."""
    cfg = ClickHouse(
        host=os.environ["ICO_TEST_CLICKHOUSE_HOST"],
        port=int(os.environ.get("ICO_TEST_CLICKHOUSE_PORT", "8123")),
        username=os.environ.get("ICO_TEST_CLICKHOUSE_USER", "default"),
        password=os.environ.get("ICO_TEST_CLICKHOUSE_PASSWORD", ""),
        database=f"ico_test_{uuid.uuid4().hex[:12]}",
    )
    yield cfg
    store.connect(cfg, database="default").command(f"DROP DATABASE IF EXISTS `{cfg.database}`")


def test_migrations_apply_once(cfg):
    assert store.migrate(cfg, SCHEMA_DIR) == ["001_init.sql"]
    assert store.migrate(cfg, SCHEMA_DIR) == [], "a second run must find nothing to do"
    tables = {r[0] for r in store.connect(cfg).query("SHOW TABLES").result_rows}
    assert {"requests", "engine_spans", "engine_metrics", "gpu_metrics", "replay_runs", "replay_results"} <= tables


def test_a_record_round_trips_through_the_requests_table(cfg):
    store.migrate(cfg, SCHEMA_DIR)
    client = store.connect(cfg)
    writer = store.Writer(client, max_buffered_rows=100, flush_interval_s=1)
    writer.add("requests", REQUESTS_COLUMNS, [to_row(sample_record())])
    asyncio.run(writer.flush())
    assert writer.written == 1

    row = client.query(
        "SELECT ts, tenant, prompt_tokens, cached_tokens, ttft_ms, status, stream, prefix_hashes FROM requests"
    ).result_rows[0]
    ts, tenant, prompt_tokens, cached_tokens, ttft_ms, status, stream, prefix_hashes = row
    assert ts.replace(tzinfo=timezone.utc) == datetime(2026, 10, 14, 0, 0, 0, 123456, tzinfo=timezone.utc)
    assert (tenant, prompt_tokens, cached_tokens, ttft_ms, status, stream) == ("acme", 120, 64, 81.5, 200, True)
    assert list(prefix_hashes) == [1, 2**64 - 1]


def test_scraped_and_traced_rows_fit_their_tables_and_the_join_works(cfg):
    from ico.collector.adapters.base import now
    from ico.collector.adapters.gpu import GPU_METRICS_COLUMNS
    from ico.collector.adapters.vllm import ENGINE_METRICS_COLUMNS
    from ico.collector.otlp import ENGINE_SPANS_COLUMNS, span_rows
    from tests.test_otlp import TRACE_ID, vllm_export

    store.migrate(cfg, SCHEMA_DIR)
    client = store.connect(cfg)
    writer = store.Writer(client, max_buffered_rows=100, flush_interval_s=1)
    writer.add("engine_metrics", ENGINE_METRICS_COLUMNS, [(now(), "r1", "ttft_seconds_bucket", {"le": "+Inf", "model_name": "m"}, 6.0)])
    writer.add("gpu_metrics", GPU_METRICS_COLUMNS, [(now(), "gpu-1", "GPU-1", "NVIDIA L4", "sm_active_ratio", 0.41, 3)])
    writer.add("engine_spans", ENGINE_SPANS_COLUMNS, span_rows(vllm_export()))
    # Two gateway records: one the engine traced, one it did not.
    writer.add("requests", REQUESTS_COLUMNS, [
        to_row(sample_record(request_id="traced", trace_id=TRACE_ID)),
        to_row(sample_record(request_id="untraced", trace_id="f" * 32)),
    ])
    asyncio.run(writer.flush())
    assert writer.written == 5

    assert client.query("SELECT labels['le'], value FROM engine_metrics").result_rows == [("+Inf", 6.0)]
    assert client.query("SELECT gpu_model, metric, value, level FROM gpu_metrics").result_rows == [("NVIDIA L4", "sm_active_ratio", 0.41, 3)]
    joined = client.query(
        "SELECT request_id, has_span, queue_ms, prefill_ms, decode_ms, engine_e2e_ms FROM request_spans ORDER BY request_id"
    ).result_rows
    assert joined == [("traced", True, 2.0, 50.0, 160.0, 220.5), ("untraced", False, 0.0, 0.0, 0.0, 0.0)]
