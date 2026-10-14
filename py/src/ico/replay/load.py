"""Replays the sample at a chosen arrival rate and says whether the latency limits held.

Arrivals are open-loop: a request is sent when its turn comes, whether or not
earlier ones have finished. That is what real clients do, and it is what makes
an overloaded engine show up as growing latency instead of a quietly lower rate.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from ico.replay.engine import Outcome
from ico.replay.prompts import Prompt

# Share of requests allowed over a p95 limit.
TAIL = 0.05


@dataclass(frozen=True)
class Limits:
    ttft_p95_ms: float
    itl_p95_ms: float
    max_error_rate: float = 0.01


@dataclass(frozen=True)
class StepResult:
    rate_rps: float
    requests: int
    errors: int
    # Infinite when too few requests finished to say: more than 5% failed or were cut off.
    ttft_p95_ms: float
    itl_p95_ms: float
    held: bool
    seconds: float
    # What explains a break: how many finished requests were over each limit, and the slowest of them.
    over_ttft: int = 0
    over_itl: int = 0
    slowest_ttft_ms: float = 0.0
    slowest_itl_ms: float = 0.0
    finished: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0


def percentile95(values: Sequence[float], out_of: int) -> float:
    """The 95th percentile by nearest rank among ``out_of`` requests.

    Requests with no value (failed, or cut off) count as slower than any
    measured one, so the result is infinite once more than 5% are missing.
    """
    rank = math.ceil((1 - TAIL) * out_of)
    ordered = sorted(values)
    return ordered[rank - 1] if 0 < rank <= len(ordered) else math.inf


def schedule(gaps: Sequence[float], recorded_rps: float, rate_rps: float, count: int) -> list[float]:
    """Send times, in seconds from the start of a step, for ``count`` requests at ``rate_rps``.

    The recorded gaps are all scaled by one factor, so the average rate
    becomes the one asked for while bursts stay bursts. ``gaps[i]`` is the
    recorded wait between request i and request i + 1; the list is walked in
    a loop when the step needs more requests than the sample has.
    """
    scale = recorded_rps / rate_rps if recorded_rps > 0 and gaps else 0.0
    times, at = [], 0.0
    for i in range(count):
        times.append(at)
        at += gaps[i % len(gaps)] * scale if scale else 1.0 / rate_rps
    return times


async def run_step(
    send: Callable[[Prompt], Awaitable[Outcome]],
    prompt_for: Callable[[int], Prompt],
    send_times: Sequence[float],
    rate_rps: float,
    limits: Limits,
    *,
    grace_s: float,
) -> StepResult:
    """Send one request at each of ``send_times`` and judge the limits over all of them.

    The step ends early once it can no longer pass: more than 5% of its
    requests are already over a limit, or more than the allowed share failed.
    Requests still running ``grace_s`` after the last one was sent count as
    over the limit.
    """
    count = len(send_times)
    allowed_slow = math.floor(TAIL * count)
    allowed_errors = math.floor(limits.max_error_rate * count)
    ttfts: list[float] = []
    itls: list[float] = []
    slow_ttft = slow_itl = errors = 0
    failed = asyncio.Event()

    async def one(index: int) -> None:
        nonlocal slow_ttft, slow_itl, errors
        outcome = await send(prompt_for(index))
        if not outcome.ok:
            errors += 1
            if errors > allowed_errors:
                failed.set()
            return
        ttfts.append(outcome.ttft_s * 1000)
        itls.append(outcome.itl_s * 1000)
        slow_ttft += outcome.ttft_s * 1000 > limits.ttft_p95_ms
        slow_itl += outcome.itl_s * 1000 > limits.itl_p95_ms
        if slow_ttft > allowed_slow or slow_itl > allowed_slow:
            failed.set()

    started = time.perf_counter()
    tasks: list[asyncio.Task] = []
    try:
        for index, at in enumerate(send_times):
            wait = started + at - time.perf_counter()
            if wait > 0:
                try:
                    await asyncio.wait_for(failed.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass
            if failed.is_set():
                break
            tasks.append(asyncio.create_task(one(index)))

        if not failed.is_set() and tasks:
            finishing = asyncio.gather(*tasks, return_exceptions=True)
            stop = asyncio.create_task(failed.wait())
            await asyncio.wait({finishing, stop}, timeout=grace_s, return_when=asyncio.FIRST_COMPLETED)
            stop.cancel()
    finally:
        # Whatever is still running is cut off; the engine drops a request
        # whose client has gone.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    ttft_p95 = percentile95(ttfts, count)
    itl_p95 = percentile95(itls, count)
    held = (
        not failed.is_set()
        and len(tasks) == count
        and errors <= allowed_errors
        and ttft_p95 <= limits.ttft_p95_ms
        and itl_p95 <= limits.itl_p95_ms
    )
    return StepResult(
        rate_rps=rate_rps,
        requests=count,
        errors=errors,
        ttft_p95_ms=ttft_p95,
        itl_p95_ms=itl_p95,
        held=held,
        seconds=time.perf_counter() - started,
        over_ttft=slow_ttft,
        over_itl=slow_itl,
        slowest_ttft_ms=max(ttfts, default=0.0),
        slowest_itl_ms=max(itls, default=0.0),
        finished=len(ttfts),
    )
