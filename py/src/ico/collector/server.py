"""The collector process: one gRPC server feeding one ClickHouse writer."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path

import grpc
import httpx
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc

from ico.collector import scraper, store
from ico.collector.adapters.base import Adapter
from ico.collector.adapters.gpu import GpuExporterAdapter
from ico.collector.adapters.vllm import VllmAdapter
from ico.collector.ingest import TelemetryService
from ico.collector.otlp import TraceService
from ico.config import Config
from ico.telemetry.v1 import telemetry_pb2_grpc

log = logging.getLogger(__name__)


def migrate_when_ready(cfg: Config, schema_dir: Path, *, timeout_s: float = 120.0) -> None:
    """Apply migrations, waiting for ClickHouse to accept connections first."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            store.migrate(cfg.clickhouse, schema_dir)
            return
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001 - connection errors come in several types
            if time.monotonic() > deadline:
                raise RuntimeError(f"ClickHouse was not reachable within {timeout_s:.0f}s: {e}") from e
            log.info("waiting for ClickHouse at %s:%s (%s)", cfg.clickhouse.host, cfg.clickhouse.port, type(e).__name__)
            time.sleep(2)


def adapters(cfg: Config, http: httpx.AsyncClient) -> list[Adapter]:
    """The pulled sources named by the config: every replica's engine, then the GPU exporters."""
    engines: list[Adapter] = [VllmAdapter(r.name, r.url, http) for r in cfg.replicas]
    gpus: list[Adapter] = [GpuExporterAdapter(e.kind, e.host, e.url, http) for e in cfg.collector.gpu_exporters]
    return engines + gpus


async def serve(cfg: Config) -> None:
    writer = store.Writer(
        store.connect(cfg.clickhouse),
        max_buffered_rows=cfg.collector.max_buffered_rows,
        flush_interval_s=cfg.collector.flush_interval_s,
    )
    # One gRPC port takes both pushed sources: the gateway's records and the
    # engines' OTLP traces.
    server = grpc.aio.server()
    telemetry_pb2_grpc.add_TelemetryServicer_to_server(TelemetryService(writer), server)
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(TraceService(writer), server)
    server.add_insecure_port(cfg.collector.grpc_listen)
    await server.start()
    log.info("collector listening on %s", cfg.collector.grpc_listen)

    http = httpx.AsyncClient(timeout=min(3.0, cfg.collector.scrape_interval_s))
    scraping = [
        asyncio.create_task(scraper.run(adapter, writer, cfg.collector.scrape_interval_s))
        for adapter in adapters(cfg, http)
    ]

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows event loops have no signal handlers
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))

    writing = asyncio.create_task(writer.run())
    await stop.wait()

    log.info("shutting down")
    for task in scraping:
        task.cancel()
    await asyncio.gather(*scraping, return_exceptions=True)
    await http.aclose()
    await server.stop(grace=5)  # let in-flight Export calls land in the buffer
    writing.cancel()  # run() flushes once more on its way out
    await asyncio.gather(writing, return_exceptions=True)
    log.info("stopped; %d rows written, %d dropped", writer.written, writer.dropped)
