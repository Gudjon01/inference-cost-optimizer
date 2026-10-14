"""The OTLP trace receiver: the engine's request spans become rows of ``engine_spans``.

vLLM exports one span per request when started with ``--otlp-traces-endpoint``.
It continues the trace the gateway started, so a span's trace id is the trace
id of the gateway's record for the same request.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import grpc
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2, trace_service_pb2_grpc
from opentelemetry.proto.common.v1 import common_pb2

from ico.collector.store import Row, Writer

ENGINE_SPANS_COLUMNS = (
    "ts",
    "trace_id",
    "span_id",
    "replica",
    "queue_ms",
    "prefill_ms",
    "decode_ms",
    "e2e_ms",
    "ttft_ms",
    "prompt_tokens",
    "completion_tokens",
)

# vLLM span attributes (vllm/tracing, SpanAttributes), all in seconds.
_LATENCIES = (
    "gen_ai.latency.time_in_queue",
    "gen_ai.latency.time_in_model_prefill",
    "gen_ai.latency.time_in_model_decode",
    "gen_ai.latency.e2e",
    "gen_ai.latency.time_to_first_token",
)
_TOKENS = ("gen_ai.usage.prompt_tokens", "gen_ai.usage.completion_tokens")

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _number(value: common_pb2.AnyValue) -> float:
    kind = value.WhichOneof("value")
    return float(getattr(value, kind)) if kind in ("double_value", "int_value") else 0.0


def span_rows(request: trace_service_pb2.ExportTraceServiceRequest) -> list[Row]:
    """Rows for the engine request spans in an export, in ENGINE_SPANS_COLUMNS order.

    Spans without engine latency attributes (HTTP server spans and the like)
    are left out.
    """
    rows = []
    for resource_spans in request.resource_spans:
        # Replicas are told apart by service name: start each one with
        # OTEL_SERVICE_NAME set to its replica name.
        replica = next(
            (a.value.string_value for a in resource_spans.resource.attributes if a.key == "service.name"), ""
        )
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                attrs = {a.key: a.value for a in span.attributes}
                if not any(key in attrs for key in _LATENCIES):
                    continue
                rows.append(
                    (
                        _EPOCH + timedelta(microseconds=span.start_time_unix_nano // 1000),
                        span.trace_id.hex(),
                        span.span_id.hex(),
                        replica,
                        *(_number(attrs[k]) * 1000.0 if k in attrs else 0.0 for k in _LATENCIES),
                        *(int(_number(attrs[k])) if k in attrs else 0 for k in _TOKENS),
                    )
                )
    return rows


class TraceService(trace_service_pb2_grpc.TraceServiceServicer):
    def __init__(self, writer: Writer) -> None:
        self._writer = writer

    async def Export(  # noqa: N802 - the name is fixed by the proto
        self, request: trace_service_pb2.ExportTraceServiceRequest, context: grpc.aio.ServicerContext
    ) -> trace_service_pb2.ExportTraceServiceResponse:
        rows = span_rows(request)
        if rows:
            self._writer.add("engine_spans", ENGINE_SPANS_COLUMNS, rows)
        return trace_service_pb2.ExportTraceServiceResponse()
