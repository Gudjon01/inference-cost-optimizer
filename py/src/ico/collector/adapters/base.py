"""What every pulled source implements, and the Prometheus plumbing most share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import httpx
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from ico.collector.store import Row


@dataclass(frozen=True)
class Capabilities:
    """What a source was found to report.

    ``metrics`` holds the canonical names it can fill. ``level`` is the
    telemetry level of a GPU source and None for anything else:
    3 profiling counters, 2 basic counters, 1 memory and power only.
    """

    metrics: frozenset[str] = frozenset()
    level: int | None = None


@dataclass(frozen=True)
class Batch:
    table: str
    columns: Sequence[str]
    rows: list[Row]


class Adapter(ABC):
    """One source of telemetry.

    The scraper calls ``discover`` until it succeeds, then ``scrape`` on every
    tick. A failed scrape sends the adapter back to discovery, so a source that
    restarts with a different version or different hardware is picked up.
    """

    name: str

    @abstractmethod
    async def discover(self) -> None:
        """Probe the source. Raise if it is not there or not what was configured."""

    @abstractmethod
    async def scrape(self) -> Batch:
        """Read the source once and return rows for one table."""

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """What the source reported the last time it was read."""


class PrometheusAdapter(Adapter):
    """An adapter whose source is a Prometheus text page."""

    def __init__(self, name: str, url: str, http: httpx.AsyncClient) -> None:
        self.name = name
        self.url = url
        self._http = http
        self._capabilities = Capabilities()

    def capabilities(self) -> Capabilities:
        return self._capabilities

    async def _samples(self) -> list[Sample]:
        response = await self._http.get(self.url)
        response.raise_for_status()
        return parse_samples(response.text)


def parse_samples(text: str) -> list[Sample]:
    return [s for family in text_string_to_metric_families(text) for s in family.samples]


def now() -> datetime:
    return datetime.now(timezone.utc)
