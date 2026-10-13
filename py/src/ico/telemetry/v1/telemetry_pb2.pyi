from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExportRequest(_message.Message):
    __slots__ = ("gateway_id", "records")
    GATEWAY_ID_FIELD_NUMBER: _ClassVar[int]
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    gateway_id: str
    records: _containers.RepeatedCompositeFieldContainer[RequestRecord]
    def __init__(self, gateway_id: _Optional[str] = ..., records: _Optional[_Iterable[_Union[RequestRecord, _Mapping]]] = ...) -> None: ...

class ExportResponse(_message.Message):
    __slots__ = ("accepted",)
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    accepted: int
    def __init__(self, accepted: _Optional[int] = ...) -> None: ...

class RequestRecord(_message.Message):
    __slots__ = ("request_id", "trace_id", "arrival_unix_nanos", "replica", "model", "tenant", "feature", "endpoint", "prompt_tokens", "completion_tokens", "cached_tokens", "ttft_ms", "e2e_ms", "gap_p50_ms", "gap_p95_ms", "status", "stream", "max_tokens", "prompt_chars", "prefix_hashes")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    ARRIVAL_UNIX_NANOS_FIELD_NUMBER: _ClassVar[int]
    REPLICA_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TENANT_FIELD_NUMBER: _ClassVar[int]
    FEATURE_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TTFT_MS_FIELD_NUMBER: _ClassVar[int]
    E2E_MS_FIELD_NUMBER: _ClassVar[int]
    GAP_P50_MS_FIELD_NUMBER: _ClassVar[int]
    GAP_P95_MS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_CHARS_FIELD_NUMBER: _ClassVar[int]
    PREFIX_HASHES_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    trace_id: str
    arrival_unix_nanos: int
    replica: str
    model: str
    tenant: str
    feature: str
    endpoint: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    ttft_ms: float
    e2e_ms: float
    gap_p50_ms: float
    gap_p95_ms: float
    status: int
    stream: bool
    max_tokens: int
    prompt_chars: int
    prefix_hashes: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, request_id: _Optional[str] = ..., trace_id: _Optional[str] = ..., arrival_unix_nanos: _Optional[int] = ..., replica: _Optional[str] = ..., model: _Optional[str] = ..., tenant: _Optional[str] = ..., feature: _Optional[str] = ..., endpoint: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., ttft_ms: _Optional[float] = ..., e2e_ms: _Optional[float] = ..., gap_p50_ms: _Optional[float] = ..., gap_p95_ms: _Optional[float] = ..., status: _Optional[int] = ..., stream: _Optional[bool] = ..., max_tokens: _Optional[int] = ..., prompt_chars: _Optional[int] = ..., prefix_hashes: _Optional[_Iterable[int]] = ...) -> None: ...
