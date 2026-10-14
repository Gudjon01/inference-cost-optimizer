"""The recorded traffic of a window, and the sample of it a replay uses.

A window can hold millions of requests; a replay uses a run of a few hundred
consecutive ones. Consecutive, because arrival gaps and shared prefixes only
mean something between neighbours. The run is chosen to look like the whole
window in the ways that drive cost: prompt and completion lengths, how much
prompt text repeats an earlier prompt, and how bursty the arrivals are.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from ico.analyzer.measure import Window, stamp
from ico.collector.store import Client


@dataclass(frozen=True)
class Recorded:
    """One request as the gateway saw it. No text: lengths, timing and prefix hashes."""

    at: float  # seconds since the start of the window
    prompt_tokens: int
    completion_tokens: int
    prompt_chars: int
    prefix_hashes: tuple[int, ...]


@dataclass(frozen=True)
class Shape:
    """The properties a sample has to keep."""

    requests: int
    prompt_tokens: tuple[int, ...]  # sorted
    completion_tokens: tuple[int, ...]  # sorted
    # Share of prompt characters that repeat the start of an earlier prompt.
    shared_prefix: float
    # Coefficient of variation of the gaps between arrivals: 1 for random
    # arrivals, higher when requests come in bursts.
    burstiness: float
    rate_rps: float


def fetch(client: Client, replica: str, window: Window) -> list[Recorded]:
    """The window's successful generation requests on a replica, in arrival order."""
    result = client.query(
        "SELECT toUnixTimestamp64Micro(ts) AS at, prompt_tokens, completion_tokens, prompt_chars, prefix_hashes "
        "FROM requests "
        "WHERE replica = {replica:String} AND ts >= {since:DateTime64(6, 'UTC')} AND ts < {until:DateTime64(6, 'UTC')} "
        "AND status = 200 AND endpoint IN ('chat', 'completions') AND prompt_tokens > 0 AND completion_tokens > 0 "
        "ORDER BY ts",
        parameters={"replica": replica, "since": stamp(window.since), "until": stamp(window.until)},
    )
    start = window.since.timestamp()
    return [Recorded(at / 1e6 - start, p, c, chars, tuple(hashes)) for at, p, c, chars, hashes in result.result_rows]


def recorded_ttft_p95(client: Client, replica: str, window: Window) -> tuple[float, int]:
    """The p95 time to first token the gateway saw in the window, and how many streamed requests it is over."""
    result = client.query(
        "SELECT quantile(0.95)(ttft_ms), count() FROM requests "
        "WHERE replica = {replica:String} AND ts >= {since:DateTime64(6, 'UTC')} AND ts < {until:DateTime64(6, 'UTC')} "
        "AND status = 200 AND stream AND ttft_ms > 0",
        parameters={"replica": replica, "since": stamp(window.since), "until": stamp(window.until)},
    )
    p95, count = result.result_rows[0]
    return (float(p95) if count else 0.0), int(count)


def shape(requests: Sequence[Recorded], block_chars: int) -> Shape:
    seen: set[int] = set()
    shared = total = 0
    for r in requests:
        # hash i covers blocks 0..i, so the leading hashes already seen are
        # exactly the blocks this prompt shares with an earlier one.
        known = 0
        for h in r.prefix_hashes:
            if h not in seen:
                break
            known += 1
        shared += min(known * block_chars, r.prompt_chars)
        total += r.prompt_chars
        seen.update(r.prefix_hashes)

    gaps = [b.at - a.at for a, b in zip(requests, requests[1:])]
    mean_gap = statistics.fmean(gaps) if gaps else 0.0
    return Shape(
        requests=len(requests),
        prompt_tokens=tuple(sorted(r.prompt_tokens for r in requests)),
        completion_tokens=tuple(sorted(r.completion_tokens for r in requests)),
        shared_prefix=shared / total if total else 0.0,
        burstiness=statistics.pstdev(gaps) / mean_gap if mean_gap > 0 else 0.0,
        rate_rps=(len(requests) - 1) / (requests[-1].at - requests[0].at) if len(requests) > 1 and requests[-1].at > requests[0].at else 0.0,
    )


def distance(a: Shape, b: Shape) -> float:
    """How unlike two shapes are. 0 is identical; each of the four terms is at most about 1."""
    return (
        _ks(a.prompt_tokens, b.prompt_tokens)
        + _ks(a.completion_tokens, b.completion_tokens)
        + abs(a.shared_prefix - b.shared_prefix)
        + abs(a.burstiness - b.burstiness) / (1 + max(a.burstiness, b.burstiness))
    )


def _ks(a: Sequence[int], b: Sequence[int]) -> float:
    """Kolmogorov-Smirnov distance between two sorted samples: the largest gap between their CDFs."""
    if not a or not b:
        return 1.0
    i = j = 0
    worst = 0.0
    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        while i < len(a) and a[i] == x:
            i += 1
        while j < len(b) and b[j] == x:
            j += 1
        worst = max(worst, abs(i / len(a) - j / len(b)))
    return worst


@dataclass(frozen=True)
class Sample:
    requests: tuple[Recorded, ...]
    shape: Shape
    # The shape of the whole window, and how far the sample is from it.
    window_shape: Shape
    distance: float


def sample(recorded: Sequence[Recorded], size: int, block_chars: int) -> Sample:
    """The run of ``size`` consecutive requests that looks most like the whole window."""
    if not recorded:
        raise ValueError("no requests to sample")
    whole = shape(recorded, block_chars)
    if len(recorded) <= size:
        return Sample(tuple(recorded), whole, whole, 0.0)

    best: tuple[float, int, Shape] | None = None
    # Runs overlap by three quarters, so a good stretch is not missed by
    # falling across a boundary.
    step = max(1, size // 4)
    for start in list(range(0, len(recorded) - size, step)) + [len(recorded) - size]:
        candidate = shape(recorded[start : start + size], block_chars)
        d = distance(candidate, whole)
        if best is None or d < best[0]:
            best = (d, start, candidate)
    d, start, chosen = best
    return Sample(tuple(recorded[start : start + size]), chosen, whole, d)
