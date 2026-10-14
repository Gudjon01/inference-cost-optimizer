"""The gateway adapter: request records arriving over gRPC become rows of ``requests``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import grpc

from ico.collector.store import Row, Writer
from ico.telemetry.v1 import telemetry_pb2, telemetry_pb2_grpc

REQUESTS_COLUMNS = (
    "ts",
    "request_id",
    "trace_id",
    "replica",
    "model",
    "tenant",
    "feature",
    "endpoint",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "ttft_ms",
    "e2e_ms",
    "gap_p50_ms",
    "gap_p95_ms",
    "status",
    "stream",
    "max_tokens",
    "prompt_chars",
    "prefix_hashes",
)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def to_row(r: telemetry_pb2.RequestRecord) -> Row:
    """One record as a row, in REQUESTS_COLUMNS order."""
    return (
        # Integer microseconds: a float timestamp would lose them at this magnitude.
        _EPOCH + timedelta(microseconds=r.arrival_unix_nanos // 1000),
        r.request_id,
        r.trace_id,
        r.replica,
        r.model,
        r.tenant,
        r.feature,
        r.endpoint,
        r.prompt_tokens,
        r.completion_tokens,
        r.cached_tokens,
        r.ttft_ms,
        r.e2e_ms,
        r.gap_p50_ms,
        r.gap_p95_ms,
        r.status,
        r.stream,
        r.max_tokens,
        r.prompt_chars,
        list(r.prefix_hashes),
    )


class TelemetryService(telemetry_pb2_grpc.TelemetryServicer):
    def __init__(self, writer: Writer) -> None:
        self._writer = writer

    async def Export(  # noqa: N802 - the name is fixed by the proto
        self, request: telemetry_pb2.ExportRequest, context: grpc.aio.ServicerContext
    ) -> telemetry_pb2.ExportResponse:
        # Accepted means buffered. The writer owns delivery to ClickHouse from here.
        self._writer.add("requests", REQUESTS_COLUMNS, [to_row(r) for r in request.records])
        return telemetry_pb2.ExportResponse(accepted=len(request.records))
