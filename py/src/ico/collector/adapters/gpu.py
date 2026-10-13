"""GPU telemetry from the vendors' own Prometheus exporters.

Two exporters cover NVIDIA: dcgm-exporter for datacenter cards, which adds
profiling counters, and nvidia_gpu_exporter for GeForce cards, which only has
what nvidia-smi prints. Both are mapped to the same metric names and units,
so a report reads the same whichever one fed it.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ico.collector.adapters.base import Batch, Capabilities, PrometheusAdapter, now

GPU_METRICS_COLUMNS = ("ts", "host", "gpu_uuid", "gpu_model", "metric", "value", "level")

# Canonical metrics, grouped by the telemetry level their presence proves.
PROFILING = frozenset({"sm_active_ratio", "sm_occupancy_ratio", "tensor_active_ratio", "dram_active_ratio"})
BASIC = frozenset({"util_ratio"})


def telemetry_level(metrics: frozenset[str]) -> int:
    """3 with profiling counters, 2 with busy time, 1 with memory and power only."""
    if metrics & PROFILING:
        return 3
    if metrics & BASIC:
        return 2
    return 1


@dataclass(frozen=True)
class _Mapping:
    """How one exporter names things.

    ``metrics`` maps an exporter sample name to (canonical name, factor to the
    canonical unit).
    """

    metrics: dict[str, tuple[str, float]]
    uuid_label: str
    # A sample's own label naming the card, or None when names come from an info metric.
    model_label: str | None = None
    info_metric: str | None = None
    info_model_label: str | None = None


_MIB = 1024.0 * 1024.0

_DCGM = _Mapping(
    metrics={
        "DCGM_FI_PROF_SM_ACTIVE": ("sm_active_ratio", 1.0),
        "DCGM_FI_PROF_SM_OCCUPANCY": ("sm_occupancy_ratio", 1.0),
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": ("tensor_active_ratio", 1.0),
        "DCGM_FI_PROF_DRAM_ACTIVE": ("dram_active_ratio", 1.0),
        "DCGM_FI_DEV_FB_USED": ("memory_used_bytes", _MIB),
        "DCGM_FI_DEV_POWER_USAGE": ("power_watts", 1.0),
        "DCGM_FI_DEV_GPU_UTIL": ("util_ratio", 0.01),
    },
    uuid_label="UUID",
    model_label="modelName",
)

_NVIDIA_SMI = _Mapping(
    metrics={
        "nvidia_smi_utilization_gpu_ratio": ("util_ratio", 1.0),
        "nvidia_smi_memory_used_bytes": ("memory_used_bytes", 1.0),
        "nvidia_smi_power_draw_watts": ("power_watts", 1.0),
    },
    uuid_label="uuid",
    info_metric="nvidia_smi_gpu_info",
    info_model_label="name",
)

KINDS = {"dcgm": _DCGM, "nvidia_smi": _NVIDIA_SMI}


class GpuExporterAdapter(PrometheusAdapter):
    def __init__(self, kind: str, host: str, url: str, http: httpx.AsyncClient) -> None:
        super().__init__(f"{kind}:{host}", url, http)
        self._mapping = KINDS[kind]
        self._host = host

    async def discover(self) -> None:
        await self.scrape()
        if not self._capabilities.metrics:
            raise RuntimeError(f"{self.url} has none of the metrics a {self.name.split(':')[0]} exporter publishes")

    async def scrape(self) -> Batch:
        m = self._mapping
        ts = now()
        readings, found, models = [], set(), {}
        for s in await self._samples():
            uuid = s.labels.get(m.uuid_label, "")
            if s.name in m.metrics:
                canonical, factor = m.metrics[s.name]
                found.add(canonical)
                readings.append((uuid, canonical, s.value * factor))
                if m.model_label:
                    models[uuid] = s.labels.get(m.model_label, "")
            elif s.name == m.info_metric:
                models[uuid] = s.labels.get(m.info_model_label or "", "")

        # The level is that of this very page, so a card that loses its
        # profiling counters is never reported at the level it used to have.
        level = telemetry_level(frozenset(found))
        self._capabilities = Capabilities(metrics=frozenset(found), level=level if found else None)
        rows = [(ts, self._host, uuid, models.get(uuid, ""), metric, value, level) for uuid, metric, value in readings]
        return Batch("gpu_metrics", GPU_METRICS_COLUMNS, rows)
