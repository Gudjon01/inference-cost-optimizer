"""Runs one adapter on a fixed interval, forever."""

from __future__ import annotations

import asyncio
import logging
import time

from ico.collector.adapters.base import Adapter
from ico.collector.store import Writer

log = logging.getLogger(__name__)


async def run(adapter: Adapter, writer: Writer, interval_s: float) -> None:
    """Discover, then scrape every ``interval_s`` seconds.

    A source that is down or changes under us is not an error to stop on: the
    failure is logged once, and discovery is retried on every tick until the
    source answers again.
    """
    discovered = failing = False
    reported = None
    while True:
        started = time.monotonic()
        try:
            if not discovered:
                await adapter.discover()
                discovered = True
            batch = await adapter.scrape()
            writer.add(batch.table, batch.columns, batch.rows)
            if failing:
                log.info("%s: scraping again", adapter.name)
                failing = False
            # Said once, and again whenever the source starts reporting more or less.
            if (caps := adapter.capabilities()) != reported:
                level = f" at telemetry level L{caps.level}" if caps.level else ""
                log.info("%s: reports %s%s", adapter.name, ", ".join(sorted(caps.metrics)), level)
                reported = caps
        except Exception as e:  # noqa: BLE001 - network, HTTP and parse errors all mean "try again"
            if not failing:
                log.warning("%s: scrape failed, retrying every %gs: %s", adapter.name, interval_s, e or type(e).__name__)
                failing = True
            discovered = False
        await asyncio.sleep(max(0.0, interval_s - (time.monotonic() - started)))
