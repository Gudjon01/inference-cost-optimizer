import asyncio
from pathlib import Path

import httpx
import pytest

from ico.collector import scraper
from ico.collector.adapters.base import Adapter, Batch, Capabilities
from ico.collector.adapters.gpu import GPU_METRICS_COLUMNS, GpuExporterAdapter
from ico.collector.adapters.vllm import ENGINE_METRICS_COLUMNS, VllmAdapter

FIXTURES = Path(__file__).parent / "fixtures"


def page(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def serving(text: str, status: int = 200) -> httpx.AsyncClient:
    """An HTTP client whose every request is answered with ``text``."""
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status, text=text)))


def discover_and_scrape(adapter: Adapter) -> tuple[Capabilities, Batch]:
    async def go():
        await adapter.discover()
        return adapter.capabilities(), await adapter.scrape()

    return asyncio.run(go())


def by_metric(batch: Batch) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in batch.rows:
        r = dict(zip(batch.columns, row, strict=True))
        out.setdefault(r["metric"], []).append(r)
    return out


# vLLM -----------------------------------------------------------------------

# The page of a current vLLM (names from the v0.31.0 source), cut down.
VLLM_NEW_NAMES = """\
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="m"} 0.25
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total{model_name="m"} 900.0
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{model_name="m"} 300.0
# TYPE vllm:inter_token_latency_seconds histogram
vllm:inter_token_latency_seconds_bucket{le="0.01",model_name="m"} 5.0
vllm:inter_token_latency_seconds_bucket{le="+Inf",model_name="m"} 7.0
vllm:inter_token_latency_seconds_sum{model_name="m"} 0.08
vllm:inter_token_latency_seconds_count{model_name="m"} 7.0
# TYPE vllm:estimated_flops_per_gpu counter
vllm:estimated_flops_per_gpu_total{model_name="m"} 1.5e+15
# TYPE vllm:some_metric_we_do_not_use gauge
vllm:some_metric_we_do_not_use{model_name="m"} 1.0
"""


def test_vllm_page_from_a_real_0_10_engine():
    caps, batch = discover_and_scrape(
        VllmAdapter("r1", "http://engine:8000/", serving(page("vllm-0.10.1.1.metrics.txt")))
    )
    assert batch.table == "engine_metrics" and batch.columns == ENGINE_METRICS_COLUMNS
    # This version says gpu_cache_usage_perc and time_per_output_token_seconds;
    # both land under the version-independent names.
    assert {"requests_running", "requests_waiting", "kv_cache_usage", "prompt_tokens_total",
            "generation_tokens_total", "preemptions_total", "ttft_seconds", "inter_token_latency_seconds",
            "queue_time_seconds", "prefill_time_seconds", "decode_time_seconds"} <= caps.metrics
    assert "prefix_cache_hits_total" not in caps.metrics, "0.10.1.1 does not publish prefix cache counters"
    assert caps.level is None

    m = by_metric(batch)
    assert [r["value"] for r in m["prompt_tokens_total"]] == [1429.0]
    assert m["prompt_tokens_total"][0]["replica"] == "r1"
    assert m["prompt_tokens_total"][0]["labels"] == {"model_name": "Qwen/Qwen3-0.6B"}
    assert [r["value"] for r in m["ttft_seconds_count"]] == [6.0]
    buckets = m["ttft_seconds_bucket"]
    assert len(buckets) > 5 and all("le" in r["labels"] for r in buckets)
    assert buckets[-1]["labels"]["le"] == "+Inf" and buckets[-1]["value"] == 6.0
    assert not any(name.startswith("vllm:") for name in m), "engine names must not reach the table"


def test_vllm_current_names_map_to_the_same_canonical_names():
    caps, batch = discover_and_scrape(VllmAdapter("r1", "http://engine:8000", serving(VLLM_NEW_NAMES)))
    assert caps.metrics == {"kv_cache_usage", "prefix_cache_queries_total", "prefix_cache_hits_total",
                            "inter_token_latency_seconds", "estimated_flops_total"}
    m = by_metric(batch)
    assert m["kv_cache_usage"][0]["value"] == 0.25
    assert m["prefix_cache_hits_total"][0]["value"] == 300.0
    assert m["estimated_flops_total"][0]["value"] == 1.5e15
    assert set(m) == {"kv_cache_usage", "prefix_cache_queries_total", "prefix_cache_hits_total",
                      "inter_token_latency_seconds_bucket", "inter_token_latency_seconds_sum",
                      "inter_token_latency_seconds_count", "estimated_flops_total"}


def test_vllm_discovery_waits_for_the_engine_to_register_its_metrics():
    # While the model loads, /metrics answers with only the Python process metrics.
    still_loading = "# TYPE python_gc_objects_collected counter\npython_gc_objects_collected_total 9.0\n"
    with pytest.raises(RuntimeError, match="no vLLM metrics yet"):
        asyncio.run(VllmAdapter("r1", "http://engine:8000", serving(still_loading)).discover())


def test_vllm_capabilities_grow_when_the_engine_publishes_more():
    # vLLM 0.10 shows its histograms only after the first request.
    pages = iter(["# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 0.0\n", VLLM_NEW_NAMES])
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=next(pages))))
    adapter = VllmAdapter("r1", "http://engine:8000", http)

    async def go():
        await adapter.discover()
        before = adapter.capabilities().metrics
        await adapter.scrape()
        return before, adapter.capabilities().metrics

    before, after = asyncio.run(go())
    assert before == {"requests_running"}
    assert "inter_token_latency_seconds" in after and "kv_cache_usage" in after


def test_metrics_url_is_built_from_the_replica_url():
    assert VllmAdapter("r1", "http://engine:8000/", serving("")).url == "http://engine:8000/metrics"


# GPU exporters --------------------------------------------------------------


def test_nvidia_smi_exporter_on_a_real_geforce_card_is_level_2():
    caps, batch = discover_and_scrape(GpuExporterAdapter(
        "nvidia_smi", "desktop", "http://exporter:9835/metrics", serving(page("nvidia_gpu_exporter-1.15.1.metrics.txt"))
    ))
    assert caps.level == 2
    assert caps.metrics == {"util_ratio", "memory_used_bytes", "memory_total_bytes", "power_watts"}
    assert batch.table == "gpu_metrics" and batch.columns == GPU_METRICS_COLUMNS

    m = by_metric(batch)
    used = m["memory_used_bytes"][0]
    assert used["value"] == 3650093056.0
    assert used["host"] == "desktop" and used["level"] == 2
    assert used["gpu_uuid"] == "00000000-1111-2222-3333-444444444444"
    assert used["gpu_model"] == "NVIDIA GeForce RTX 2060", "the card's name comes from the exporter's info metric"
    assert m["power_watts"][0]["value"] == 21.99
    assert m["memory_total_bytes"][0]["value"] == 6 * 2**30, "the roofline check needs the size of the card"
    assert 0.0 <= m["util_ratio"][0]["value"] <= 1.0


def test_dcgm_exporter_with_profiling_counters_is_level_3():
    caps, batch = discover_and_scrape(GpuExporterAdapter(
        "dcgm", "gpu-1", "http://dcgm:9400/metrics", serving(page("dcgm-exporter.handwritten.metrics.txt"))
    ))
    assert caps.level == 3
    m = by_metric(batch)
    assert m["sm_active_ratio"][0]["value"] == 0.41
    assert m["dram_active_ratio"][0]["value"] == 0.63
    # Units are normalised: percent to a ratio, MiB to bytes.
    assert m["util_ratio"][0]["value"] == pytest.approx(0.97)
    assert m["memory_used_bytes"][0]["value"] == 20480 * 1024 * 1024
    assert m["memory_total_bytes"][0]["value"] == 23034 * 1024 * 1024
    assert m["power_watts"][0]["value"] == 61.5
    assert {r["gpu_model"] for rows in m.values() for r in rows} == {"NVIDIA L4"}
    assert {r["level"] for rows in m.values() for r in rows} == {3}


def test_dcgm_without_profiling_counters_drops_to_level_2():
    basic = "\n".join(line for line in page("dcgm-exporter.handwritten.metrics.txt").splitlines() if "_PROF_" not in line)
    caps, _ = discover_and_scrape(GpuExporterAdapter("dcgm", "gpu-1", "http://dcgm:9400/metrics", serving(basic)))
    assert caps.level == 2
    memory_only = "\n".join(line for line in basic.splitlines() if "GPU_UTIL" not in line)
    caps, _ = discover_and_scrape(GpuExporterAdapter("dcgm", "gpu-1", "http://dcgm:9400/metrics", serving(memory_only)))
    assert caps.level == 1


def test_discovery_rejects_a_page_that_is_not_the_exporter():
    adapter = GpuExporterAdapter("dcgm", "gpu-1", "http://wrong:9400/metrics", serving(VLLM_NEW_NAMES))
    with pytest.raises(RuntimeError, match="none of the metrics"):
        asyncio.run(adapter.discover())
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(GpuExporterAdapter("dcgm", "gpu-1", "http://down:9400/metrics", serving("", status=503)).discover())


# Scraper --------------------------------------------------------------------


class Flaky(Adapter):
    """Fails its second scrape, then works again."""

    name = "flaky"

    def __init__(self) -> None:
        self.discoveries = 0
        self.scrapes = 0

    async def discover(self) -> None:
        self.discoveries += 1

    async def scrape(self) -> Batch:
        self.scrapes += 1
        if self.scrapes == 2:
            raise ConnectionError("source restarted")
        return Batch("engine_metrics", ("n",), [(self.scrapes,)])

    def capabilities(self) -> Capabilities:
        return Capabilities(metrics=frozenset({"n"}))


class Collected:
    def __init__(self) -> None:
        self.rows: list = []

    def add(self, table, columns, rows):
        self.rows.extend(rows)


def test_scraper_rediscovers_after_a_failure_and_keeps_going():
    adapter, sink = Flaky(), Collected()

    async def go():
        task = asyncio.create_task(scraper.run(adapter, sink, interval_s=0.01))
        while adapter.scrapes < 4:
            await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(asyncio.wait_for(go(), timeout=5))
    assert sink.rows[:3] == [(1,), (3,), (4,)], "the failed scrape adds nothing; the next ones do"
    assert adapter.discoveries == 2, "one discovery at start, one after the failure"
