"""The shared YAML config file, as the Python side reads it.

The gateway reads the same file; keys only it uses (the ``gateway`` section)
are ignored here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


GPU_EXPORTER_KINDS = ("dcgm", "nvidia_smi")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Replica:
    name: str
    url: str
    host: str = ""
    gpu_model: str = ""
    gpu_count: int = 1


@dataclass(frozen=True)
class Model:
    """What MFU and MBU need to know about a served model."""

    params: int
    weight_bytes_per_param: float
    kv_bytes_per_token: int


@dataclass(frozen=True)
class Slo:
    ttft_p95_ms: float = 500.0
    itl_p95_ms: float = 50.0


@dataclass(frozen=True)
class ClickHouse:
    host: str = "localhost"
    port: int = 8123
    username: str = "default"
    password: str = ""
    database: str = "ico"
    secure: bool = False


@dataclass(frozen=True)
class GpuExporter:
    """A vendor GPU exporter to scrape."""

    # "dcgm" for NVIDIA's dcgm-exporter, "nvidia_smi" for nvidia_gpu_exporter.
    kind: str
    url: str
    # The machine the exporter reports on; matches a replica's host.
    host: str


@dataclass(frozen=True)
class Collector:
    grpc_listen: str = "0.0.0.0:4317"
    scrape_interval_s: float = 5.0
    gpu_exporters: tuple[GpuExporter, ...] = ()
    flush_interval_s: float = 1.0
    # Rows held in memory while ClickHouse is unreachable. Past this the
    # oldest rows are dropped.
    max_buffered_rows: int = 500_000


@dataclass(frozen=True)
class Config:
    replicas: tuple[Replica, ...]
    gpu_price_per_hour: dict[str, float] = field(default_factory=dict)
    models: dict[str, Model] = field(default_factory=dict)
    slo: Slo = Slo()
    clickhouse: ClickHouse = ClickHouse()
    collector: Collector = Collector()


def load(path: str | os.PathLike[str]) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the config must be a YAML mapping")
    return parse(raw)


def parse(raw: dict[str, Any]) -> Config:
    replicas = tuple(
        _build(Replica, r, f"replicas[{i}]") for i, r in enumerate(raw.get("replicas") or [])
    )
    if not replicas:
        raise ConfigError("at least one replica is required")
    names = [r.name for r in replicas]
    if len(set(names)) != len(names):
        raise ConfigError("replica names must be unique")

    clickhouse = _build(ClickHouse, raw.get("clickhouse") or {}, "clickhouse")
    # The password stays out of the file when the environment provides it.
    if password := os.environ.get("ICO_CLICKHOUSE_PASSWORD"):
        clickhouse = replace(clickhouse, password=password)

    return Config(
        replicas=replicas,
        gpu_price_per_hour={str(k): float(v) for k, v in (raw.get("gpu_price_per_hour") or {}).items()},
        models={
            str(name): _build(Model, m, f"models.{name}") for name, m in (raw.get("models") or {}).items()
        },
        slo=_build(Slo, raw.get("slo") or {}, "slo"),
        clickhouse=clickhouse,
        collector=_collector(raw.get("collector") or {}),
    )


def _collector(raw: Any) -> Collector:
    if not isinstance(raw, dict):
        raise ConfigError("collector: expected a mapping")
    exporters = tuple(
        _build(GpuExporter, e, f"collector.gpu_exporters[{i}]")
        for i, e in enumerate(raw.get("gpu_exporters") or [])
    )
    for e in exporters:
        if e.kind not in GPU_EXPORTER_KINDS:
            raise ConfigError(
                f"collector.gpu_exporters: unknown kind {e.kind!r}, expected one of {', '.join(GPU_EXPORTER_KINDS)}"
            )
    return replace(_build(Collector, {**raw, "gpu_exporters": ()}, "collector"), gpu_exporters=exporters)


def _build(cls: type, raw: Any, where: str):
    """Build a config dataclass from a mapping, naming the section on error."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping")
    known = cls.__dataclass_fields__
    # The gateway owns some per-replica keys (models); skip what this side
    # does not model instead of rejecting a file the gateway accepts.
    try:
        return cls(**{k: v for k, v in raw.items() if k in known})
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from None
