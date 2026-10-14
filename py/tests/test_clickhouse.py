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
    assert store.migrate(cfg, SCHEMA_DIR) == sorted(p.name for p in SCHEMA_DIR.glob("*.sql"))
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


def seed_an_hour(cfg):
    """One hour on replica r1, 12:00 to 13:00: three requests, engine and GPU samples at both ends."""
    from datetime import timedelta

    from ico.collector.adapters.gpu import GPU_METRICS_COLUMNS
    from ico.collector.adapters.vllm import ENGINE_METRICS_COLUMNS
    from ico.collector.otlp import ENGINE_SPANS_COLUMNS

    noon = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
    nanos = int(noon.timestamp()) * 1_000_000_000
    second = 1_000_000_000

    def request(request_id, tenant, offset_s, e2e_ms, prompt, completion, **more):
        return to_row(sample_record(
            request_id=request_id, trace_id=request_id.ljust(32, "0"), tenant=tenant, model="m", replica="r1",
            arrival_unix_nanos=nanos + offset_s * second, e2e_ms=e2e_ms, prompt_tokens=prompt, completion_tokens=completion, **more,
        ))

    store.migrate(cfg, SCHEMA_DIR)
    client = store.connect(cfg)
    writer = store.Writer(client, max_buffered_rows=1000, flush_interval_s=1)
    writer.add("requests", REQUESTS_COLUMNS, [
        # Two overlapping ten-minute requests cover 12:00 to 12:15; a third covers 12:30 to 12:45.
        request("a", "acme", 0, 600_000, 1_000_000, 500_000),
        request("b", "globex", 300, 600_000, 400_000, 100_000),
        request("c", "acme", 1800, 900_000, 0, 0, status=500),
        # Answered by the gateway itself: no replica, so no GPU to charge.
        to_row(sample_record(request_id="d", replica="", arrival_unix_nanos=nanos + 60 * second, status=503)),
        # Before the window.
        request("e", "acme", -7200, 1000, 999, 999),
    ])
    writer.add("engine_spans", ENGINE_SPANS_COLUMNS, [
        (noon, "a".ljust(32, "0"), "s1", "r1", 1.0, 2_000.0, 5_000.0, 7_001.0, 100.0, 1_000_000, 500_000),
    ])
    writer.add("engine_metrics", ENGINE_METRICS_COLUMNS, [
        (noon, "r1", "requests_running", {"model_name": "m"}, 1.0),
        (noon + timedelta(hours=1) - timedelta(seconds=1), "r1", "requests_running", {"model_name": "m"}, 0.0),
    ])
    writer.add("gpu_metrics", GPU_METRICS_COLUMNS, [
        (noon, "gpu-1", "GPU-1", "NVIDIA L4", "util_ratio", 0.5, 2),
        (noon, "gpu-1", "GPU-1", "NVIDIA L4", "memory_used_bytes", 8 * 2**30, 2),
        (noon + timedelta(minutes=30), "gpu-1", "GPU-1", "NVIDIA L4", "util_ratio", 0.1, 2),
        (noon + timedelta(minutes=30), "gpu-1", "GPU-1", "NVIDIA L4", "memory_used_bytes", 20 * 2**30, 2),
    ])
    asyncio.run(writer.flush())
    return client, noon


def test_views_measure_a_window(cfg):
    from datetime import timedelta

    from ico.analyzer import measure

    client, noon = seed_an_hour(cfg)
    m = measure.fetch(client, measure.Window(noon, noon + timedelta(hours=1)))

    assert sorted((u.tenant, u.requests, u.errors, u.prompt_tokens, u.completion_tokens) for u in m.usage) == [
        ("acme", 2, 1, 1_000_000, 500_000),
        ("globex", 1, 0, 400_000, 100_000),
    ]
    # 15 minutes for the overlapping pair, counted once, plus 15 for the third.
    assert m.traffic["r1"].busy_seconds == 30 * 60
    assert m.traffic["r1"].seen == measure.Seen(noon, noon + timedelta(minutes=45))
    # One of the three requests has a span, and it reports phase times.
    assert m.phases["r1"] == measure.Phases(3, 1, 2_000.0, 5_000.0, 1_000_000, 500_000)
    assert m.engines["r1"].served_model == "m"
    assert m.engines["r1"].seen == measure.Seen(noon, noon + timedelta(hours=1) - timedelta(seconds=1))
    gpus = m.hosts["gpu-1"]
    assert (gpus.gpus, gpus.gpu_models, gpus.level) == (1, ("NVIDIA L4",), 2)
    assert gpus.util == pytest.approx(0.3) and gpus.memory_used_peak_bytes == 20 * 2**30
    assert gpus.sm_active is None and gpus.power_watts is None, "counters the host never sent read as missing, not zero"


def test_report_command_writes_the_page_from_stored_telemetry(cfg, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from ico.cli import main

    seed_an_hour(cfg)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "replicas:\n  - {name: r1, url: 'http://vllm:8000', host: gpu-1, gpu_model: L4, gpu_count: 2}\n"
        "gpu_price_per_hour: {L4: 0.50}\n"
        f"clickhouse: {{host: clickhouse-inside-compose, username: {cfg.username}, database: {cfg.database}}}\n",
        encoding="utf-8",
    )
    # The file names ClickHouse as the collector sees it; the environment says where it is from here.
    monkeypatch.setenv("ICO_CLICKHOUSE_HOST", cfg.host)
    monkeypatch.setenv("ICO_CLICKHOUSE_PORT", str(cfg.port))
    monkeypatch.setenv("ICO_CLICKHOUSE_PASSWORD", cfg.password)
    out = tmp_path / "report.html"

    result = CliRunner().invoke(
        main, ["report", "--config", str(config_file), "--window", "60m", "--until", "2026-10-14T13:00", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output

    from tests.test_report import text_of

    page = text_of(out.read_text(encoding="utf-8"))
    # 2 GPUs x $0.50 x the hour less one second the engine was seen for = $1.00; 30 of 60 minutes busy.
    assert "GPUs cost $1.00 in this window, and $0.50 of it (50%)" in page
    assert "$0.50 per million tokens" in page  # $1.00 over 2M tokens
    assert "Prefill and decode time" in page and "33%" in page  # split by spans; 1 of 3 requests traced
    assert "r1 gpu-1 2 × L4 $0.50 1.00 $1.00 50% Time with a request in flight" in page
    assert "L2 , basic counters" in page and "20.0 GiB" in page
