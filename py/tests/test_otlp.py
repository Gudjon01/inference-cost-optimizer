import asyncio
from datetime import datetime, timezone

import grpc
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2, trace_service_pb2_grpc
from opentelemetry.proto.common.v1 import common_pb2
from opentelemetry.proto.resource.v1 import resource_pb2
from opentelemetry.proto.trace.v1 import trace_pb2

from ico.collector.otlp import ENGINE_SPANS_COLUMNS, TraceService, span_rows

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"


def attr(key: str, value) -> common_pb2.KeyValue:
    if isinstance(value, str):
        return common_pb2.KeyValue(key=key, value=common_pb2.AnyValue(string_value=value))
    if isinstance(value, int):
        return common_pb2.KeyValue(key=key, value=common_pb2.AnyValue(int_value=value))
    return common_pb2.KeyValue(key=key, value=common_pb2.AnyValue(double_value=value))


def vllm_export(service: str = "r1") -> trace_service_pb2.ExportTraceServiceRequest:
    """What vLLM sends for one request: its llm_request span, plus an unrelated span."""
    engine_span = trace_pb2.Span(
        trace_id=bytes.fromhex(TRACE_ID),
        span_id=bytes.fromhex(SPAN_ID),
        name="llm_request",
        start_time_unix_nano=1_791_936_000_123_456_789,
        attributes=[
            attr("gen_ai.response.model", "Qwen/Qwen3-0.6B"),
            attr("gen_ai.request.id", "chatcmpl-abc"),
            attr("gen_ai.usage.prompt_tokens", 688),
            attr("gen_ai.usage.completion_tokens", 24),
            attr("gen_ai.latency.time_in_queue", 0.002),
            attr("gen_ai.latency.time_to_first_token", 0.061),
            attr("gen_ai.latency.e2e", 0.2205),
            attr("gen_ai.latency.time_in_model_prefill", 0.05),
            attr("gen_ai.latency.time_in_model_decode", 0.16),
        ],
    )
    http_span = trace_pb2.Span(
        trace_id=bytes.fromhex(TRACE_ID), span_id=bytes.fromhex("1111111111111111"), name="POST /v1/chat/completions"
    )
    return trace_service_pb2.ExportTraceServiceRequest(
        resource_spans=[
            trace_pb2.ResourceSpans(
                resource=resource_pb2.Resource(attributes=[attr("service.name", service)]),
                scope_spans=[trace_pb2.ScopeSpans(spans=[engine_span, http_span])],
            )
        ]
    )


def test_engine_span_becomes_one_row_in_milliseconds():
    rows = span_rows(vllm_export())
    assert len(rows) == 1, "only the span with engine latencies is kept"
    row = dict(zip(ENGINE_SPANS_COLUMNS, rows[0], strict=True))

    assert row["trace_id"] == TRACE_ID, "the id must print exactly as the gateway records it"
    assert row["span_id"] == SPAN_ID
    assert row["replica"] == "r1"
    assert row["ts"] == datetime(2026, 10, 14, 0, 0, 0, 123456, tzinfo=timezone.utc)
    assert (row["queue_ms"], row["prefill_ms"], row["decode_ms"]) == (2.0, 50.0, 160.0)
    assert (row["e2e_ms"], row["ttft_ms"]) == (220.5, 61.0)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (688, 24)


def test_missing_attributes_read_as_zero():
    export = vllm_export()
    span = export.resource_spans[0].scope_spans[0].spans[0]
    kept = [a for a in span.attributes if a.key in ("gen_ai.latency.e2e", "gen_ai.usage.prompt_tokens")]
    del span.attributes[:]
    span.attributes.extend(kept)
    row = dict(zip(ENGINE_SPANS_COLUMNS, span_rows(export)[0], strict=True))
    assert row["e2e_ms"] == 220.5 and row["prompt_tokens"] == 688
    assert (row["queue_ms"], row["prefill_ms"], row["decode_ms"], row["ttft_ms"], row["completion_tokens"]) == (0, 0, 0, 0, 0)


class Collected:
    def __init__(self) -> None:
        self.added: list = []

    def add(self, table, columns, rows):
        self.added.append((table, tuple(columns), list(rows)))


def test_export_over_grpc_is_accepted_like_any_otlp_collector():
    async def scenario():
        sink = Collected()
        server = grpc.aio.server()
        trace_service_pb2_grpc.add_TraceServiceServicer_to_server(TraceService(sink), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = trace_service_pb2_grpc.TraceServiceStub(channel)
                await stub.Export(vllm_export("r2"))
                # An export with nothing for us is still a success, and adds nothing.
                await stub.Export(trace_service_pb2.ExportTraceServiceRequest())
        finally:
            await server.stop(grace=None)
        return sink

    sink = asyncio.run(scenario())
    [(table, columns, rows)] = sink.added
    assert table == "engine_spans" and columns == ENGINE_SPANS_COLUMNS
    assert rows[0][1] == TRACE_ID and rows[0][3] == "r2"
