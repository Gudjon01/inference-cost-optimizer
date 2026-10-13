"""What the analyzer reads from ClickHouse for one report window.

Each function below reads one of the views in schema/002_analyzer.sql. The
views count and sum; nothing here knows a price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from ico.collector.store import Client


@dataclass(frozen=True)
class Window:
    """A half-open stretch of time, since <= t < until, in UTC."""

    since: datetime
    until: datetime

    def __post_init__(self) -> None:
        if self.since.tzinfo is None or self.until.tzinfo is None:
            raise ValueError("a window needs timezone-aware bounds")
        if self.since >= self.until:
            raise ValueError("a window must end after it starts")

    @property
    def seconds(self) -> float:
        return (self.until - self.since).total_seconds()


@dataclass(frozen=True)
class Seen:
    """The first and last moment a source was heard from inside the window."""

    first: datetime
    last: datetime


@dataclass(frozen=True)
class Usage:
    replica: str
    model: str
    tenant: str
    requests: int
    errors: int
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Traffic:
    """busy_seconds is time with at least one request in flight; overlaps count once."""

    busy_seconds: float
    seen: Seen


@dataclass(frozen=True)
class Phases:
    """Engine time by phase, over the requests whose span reported it."""

    requests: int
    with_span: int
    prefill_ms: float
    decode_ms: float
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Engine:
    seen: Seen
    served_model: str


@dataclass(frozen=True)
class HostGpus:
    """A host's GPU telemetry. A counter the host never reported is None."""

    host: str
    gpus: int
    gpu_models: tuple[str, ...]
    level: int
    seen: Seen
    sm_active: float | None
    tensor_active: float | None
    dram_active: float | None
    util: float | None
    memory_used_peak_bytes: float | None
    power_watts: float | None


@dataclass(frozen=True)
class Measurements:
    usage: tuple[Usage, ...]
    traffic: Mapping[str, Traffic]
    phases: Mapping[str, Phases]
    engines: Mapping[str, Engine]
    hosts: Mapping[str, HostGpus]


def fetch(client: Client, window: Window) -> Measurements:
    usage = tuple(
        Usage(r["replica"], r["model"], r["tenant"], r["requests"], r["errors"], r["prompt_tokens"], r["completion_tokens"])
        for r in _view(client, "usage_by_tenant", 6, window)
    )
    traffic = {r["replica"]: Traffic(r["busy_seconds"], _seen(r)) for r in _view(client, "replica_busy", 6, window)}
    phases = {
        r["replica"]: Phases(
            r["requests"],
            r["with_span"],
            r["phase_prefill_ms"],
            r["phase_decode_ms"],
            r["phase_prompt_tokens"],
            r["phase_completion_tokens"],
        )
        for r in _view(client, "replica_phases", 6, window)
    }
    engines = {r["replica"]: Engine(_seen(r), r["served_model"]) for r in _view(client, "replica_engine", 3, window)}
    hosts = {
        r["host"]: HostGpus(
            host=r["host"],
            gpus=r["gpus"],
            gpu_models=tuple(m for m in r["gpu_models"] if m),
            level=r["level"],
            seen=_seen(r),
            sm_active=_reported(r["sm_active"]),
            tensor_active=_reported(r["tensor_active"]),
            dram_active=_reported(r["dram_active"]),
            util=_reported(r["util"]),
            # maxIf over no rows is 0, and no card with an engine on it uses 0 bytes.
            memory_used_peak_bytes=r["memory_used_peak_bytes"] or None,
            power_watts=_reported(r["power_watts"]),
        )
        for r in _view(client, "host_gpus", 3, window)
    }
    return Measurements(usage, traffic, phases, engines, hosts)


def _view(client: Client, name: str, precision: int, window: Window) -> list[dict[str, Any]]:
    kind = f"DateTime64({precision}, 'UTC')"
    result = client.query(
        f"SELECT * FROM {name}(since = {{since:{kind}}}, until = {{until:{kind}}})",
        parameters={"since": _stamp(window.since), "until": _stamp(window.until)},
    )
    return [dict(zip(result.column_names, row)) for row in result.result_rows]


def _stamp(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def _seen(row: Mapping[str, Any]) -> Seen:
    # The server answers in UTC without saying so.
    return Seen(row["first_seen"].replace(tzinfo=timezone.utc), row["last_seen"].replace(tzinfo=timezone.utc))


def _reported(average: float) -> float | None:
    """An average over no samples comes back as NaN."""
    return None if math.isnan(average) else average
