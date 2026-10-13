"""vLLM's /metrics page, mapped to names that do not change with the vLLM version.

vLLM has renamed metrics between releases, so the analyzer never sees its
names. Each canonical name lists every name vLLM has used for it; the lists
were read from the running page of 0.10.1.1 and the source of 0.31.0.
"""

from __future__ import annotations

import httpx

from ico.collector.adapters.base import Batch, Capabilities, PrometheusAdapter, now

ENGINE_METRICS_COLUMNS = ("ts", "replica", "metric", "labels", "value")

# Gauges and counters: one sample each. Counters are stored as the engine
# reports them (cumulative); rates are the analyzer's job.
_SCALARS = {
    "requests_running": ("vllm:num_requests_running",),
    "requests_waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "prompt_tokens_total": ("vllm:prompt_tokens_total",),
    "generation_tokens_total": ("vllm:generation_tokens_total",),
    "prefix_cache_queries_total": ("vllm:prefix_cache_queries_total",),
    "prefix_cache_hits_total": ("vllm:prefix_cache_hits_total",),
    "preemptions_total": ("vllm:num_preemptions_total",),
    # vLLM's own estimate of work done, the numerators of MFU and MBU.
    "estimated_flops_total": ("vllm:estimated_flops_per_gpu_total",),
    "estimated_read_bytes_total": ("vllm:estimated_read_bytes_per_gpu_total",),
    "estimated_write_bytes_total": ("vllm:estimated_write_bytes_per_gpu_total",),
}

# Histograms: stored as <name>_bucket (with its "le" label), _sum and _count.
_HISTOGRAMS = {
    "ttft_seconds": ("vllm:time_to_first_token_seconds",),
    "inter_token_latency_seconds": ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"),
    "queue_time_seconds": ("vllm:request_queue_time_seconds",),
    "prefill_time_seconds": ("vllm:request_prefill_time_seconds",),
    "decode_time_seconds": ("vllm:request_decode_time_seconds",),
}

# Engine sample name -> (canonical sample name, canonical metric it belongs to).
_BY_SAMPLE: dict[str, tuple[str, str]] = {}
for _canonical, _names in _SCALARS.items():
    for _name in _names:
        _BY_SAMPLE[_name] = (_canonical, _canonical)
for _canonical, _names in _HISTOGRAMS.items():
    for _name in _names:
        for _suffix in ("_bucket", "_sum", "_count"):
            _BY_SAMPLE[_name + _suffix] = (_canonical + _suffix, _canonical)


class VllmAdapter(PrometheusAdapter):
    def __init__(self, replica: str, base_url: str, http: httpx.AsyncClient) -> None:
        super().__init__(f"vllm:{replica}", base_url.rstrip("/") + "/metrics", http)
        self._replica = replica

    async def discover(self) -> None:
        await self.scrape()
        if not self._capabilities.metrics:
            # The page answers before the engine has loaded its model and
            # registered its metrics; failing here makes the scraper ask again.
            raise RuntimeError(f"{self.url} has no vLLM metrics yet")

    async def scrape(self) -> Batch:
        ts = now()
        rows, found = [], set()
        for s in await self._samples():
            mapped = _BY_SAMPLE.get(s.name)
            if mapped is None:
                continue
            found.add(mapped[1])
            rows.append((ts, self._replica, mapped[0], dict(s.labels), s.value))
        # Kept current on every scrape: some vLLM versions publish their
        # histograms only after the first request.
        self._capabilities = Capabilities(metrics=frozenset(found))
        return Batch("engine_metrics", ENGINE_METRICS_COLUMNS, rows)
