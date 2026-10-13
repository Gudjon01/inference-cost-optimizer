import asyncio
from datetime import datetime, timezone

import grpc

from ico.collector.ingest import REQUESTS_COLUMNS, TelemetryService, to_row
from ico.collector.store import Writer
from ico.telemetry.v1 import telemetry_pb2, telemetry_pb2_grpc


def sample_record(**overrides) -> telemetry_pb2.RequestRecord:
    fields = dict(
        request_id="req-1",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        arrival_unix_nanos=1_791_936_000_123_456_789,
        replica="r1",
        model="m",
        tenant="acme",
        feature="search",
        endpoint="chat",
        prompt_tokens=120,
        completion_tokens=40,
        cached_tokens=64,
        ttft_ms=81.5,
        e2e_ms=900.25,
        gap_p50_ms=15.0,
        gap_p95_ms=22.0,
        status=200,
        stream=True,
        max_tokens=256,
        prompt_chars=480,
        prefix_hashes=[1, 2**64 - 1],
    )
    fields.update(overrides)
    return telemetry_pb2.RequestRecord(**fields)


def test_row_matches_the_column_list():
    row = dict(zip(REQUESTS_COLUMNS, to_row(sample_record()), strict=True))
    assert row["request_id"] == "req-1"
    assert row["tenant"] == "acme" and row["feature"] == "search" and row["endpoint"] == "chat"
    assert (row["prompt_tokens"], row["completion_tokens"], row["cached_tokens"]) == (120, 40, 64)
    assert row["status"] == 200 and row["stream"] is True and row["max_tokens"] == 256
    assert row["ttft_ms"] == 81.5 and row["e2e_ms"] == 900.25
    # The full unsigned 64-bit range must survive: hashes use the top bit.
    assert row["prefix_hashes"] == [1, 2**64 - 1]


def test_arrival_time_keeps_microseconds():
    row = dict(zip(REQUESTS_COLUMNS, to_row(sample_record()), strict=True))
    assert row["ts"] == datetime(2026, 10, 14, 0, 0, 0, 123456, tzinfo=timezone.utc)


class Buffer(Writer):
    """A writer that only collects, so no ClickHouse is needed."""

    def __init__(self) -> None:
        self.added: list[tuple[str, tuple, list]] = []

    def add(self, table, columns, rows):
        self.added.append((table, tuple(columns), list(rows)))


def test_export_over_grpc_buffers_every_record():
    async def scenario():
        buffer = Buffer()
        server = grpc.aio.server()
        telemetry_pb2_grpc.add_TelemetryServicer_to_server(TelemetryService(buffer), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = telemetry_pb2_grpc.TelemetryStub(channel)
                response = await stub.Export(
                    telemetry_pb2.ExportRequest(
                        gateway_id="gw",
                        records=[sample_record(request_id="a"), sample_record(request_id="b")],
                    )
                )
        finally:
            await server.stop(grace=None)
        return buffer, response

    buffer, response = asyncio.run(scenario())
    assert response.accepted == 2
    [(table, columns, rows)] = buffer.added
    assert table == "requests" and columns == REQUESTS_COLUMNS
    assert [r[1] for r in rows] == ["a", "b"]
