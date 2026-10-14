"""Rules out configs that cannot work, before any restart is spent on them.

For configs on the GPU the replica already has, the limit that can be worked
out on paper is memory: the weights, and room for at least one full context
of KV cache, must fit in the share of GPU memory the engine is allowed.
"""

from __future__ import annotations

from ico.collector.store import Client
from ico.config import Model
from ico.replay.sweep import Flags

GIB = 2**30

# vLLM's default for --gpu-memory-utilization.
DEFAULT_MEMORY_FRACTION = 0.9

# Bytes per weight under the quantization schemes a sweep may name.
QUANTIZED_BYTES = {"fp8": 1.0, "int8": 1.0, "int4": 0.5, "awq": 0.5, "gptq": 0.5}


def rule_out(flags: Flags, model: Model | None, gpu_memory_bytes: float | None, gpu_count: int) -> str | None:
    """Why this config cannot start, or None if nothing on paper says so.

    Without the model's numbers or the GPU's memory size there is nothing to
    check, and the config goes on to be tested.
    """
    if model is None or not gpu_memory_bytes:
        return None

    allowed = gpu_memory_bytes * gpu_count * float(flags.get("gpu-memory-utilization", DEFAULT_MEMORY_FRACTION))
    bytes_per_weight = QUANTIZED_BYTES.get(str(flags.get("quantization", "")).lower(), model.weight_bytes_per_param)
    weights = model.params * bytes_per_weight
    if weights > allowed:
        return f"the weights need {weights / GIB:.1f} GiB and this memory fraction allows {allowed / GIB:.1f} GiB"

    context = flags.get("max-model-len")
    if context:
        one_context = model.kv_bytes_per_token * int(context)
        if weights + one_context > allowed:
            return (
                f"the weights ({weights / GIB:.1f} GiB) and one full context of KV cache ({one_context / GIB:.1f} GiB) "
                f"need more than the {allowed / GIB:.1f} GiB this memory fraction allows"
            )
    return None


def gpu_memory(client: Client, host: str) -> float | None:
    """Memory in bytes of the smallest GPU on a host, from the last week of GPU telemetry. None if none was reported."""
    result = client.query(
        "SELECT min(total) FROM ("
        "SELECT gpu_uuid, argMax(value, ts) AS total FROM gpu_metrics "
        "WHERE host = {host:String} AND metric = 'memory_total_bytes' AND ts > now() - INTERVAL 7 DAY "
        "GROUP BY gpu_uuid)",
        parameters={"host": host},
    )
    total = result.result_rows[0][0] if result.result_rows else 0
    return float(total) or None
