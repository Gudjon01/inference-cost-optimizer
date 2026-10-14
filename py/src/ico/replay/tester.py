"""Wires the sample, the prompts and the load generator into the one call a run makes per config."""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from typing import Callable, Sequence

from ico.replay.engine import Engine
from ico.replay.load import Limits, StepResult, run_step, schedule
from ico.replay.prompts import Synthesizer
from ico.replay.ramp import RampResult, RampSettings, Stage, ramp
from ico.replay.sweep import Candidate


def describe(step: StepResult) -> str:
    if step.held:
        return f"    {step.rate_rps:8.2f} requests/s  held   ttft p95 {step.ttft_p95_ms:,.0f} ms, itl p95 {step.itl_p95_ms:,.0f} ms"
    # A broken step is cut short, so it has no p95 to show. What it has is
    # the reason: how many requests were over which limit, and by how much.
    reasons = []
    if step.over_ttft:
        reasons.append(f"{step.over_ttft} of {step.requests} over the ttft limit, slowest {step.slowest_ttft_ms:,.0f} ms")
    if step.over_itl:
        reasons.append(f"{step.over_itl} of {step.requests} over the itl limit, slowest {step.slowest_itl_ms:,.0f} ms")
    if step.errors:
        reasons.append(f"{step.errors} failed")
    if not reasons:
        reasons.append(f"only {step.finished} of {step.requests} finished in time")
    return f"    {step.rate_rps:8.2f} requests/s  broke  " + "; ".join(reasons)


def make(
    engine_url: str,
    synth: Synthesizer,
    gaps: Sequence[float],
    recorded_rps: float,
    limits: Limits,
    full: RampSettings,
    say: Callable[[str], None],
) -> Callable[[Candidate, float, float, Stage], RampResult]:
    """A function that ramps the sample against whatever engine is up at ``engine_url``."""

    async def run(start_rps: float, allow_s: float, settings: RampSettings) -> RampResult:
        began = time.perf_counter()
        async with Engine(engine_url) as engine:
            # Asked again for every config: one that swaps the model serves it under another name.
            model = await engine.model()
            uses = itertools.count()

            async def step(rate: float, judge: Limits = limits) -> StepResult:
                use = next(uses)
                times = schedule(gaps, recorded_rps, rate, settings.requests_at(rate))
                result = await run_step(
                    lambda prompt: engine.complete(model, prompt),
                    lambda index: synth.prompt(index, use),
                    times,
                    rate,
                    judge,
                    grace_s=settings.grace_seconds,
                )
                await asyncio.sleep(settings.settle_seconds)
                return result

            async def judged(rate: float) -> StepResult:
                result = await step(rate)
                say(describe(result))
                return result

            # One pass that is not judged. A freshly started engine has an
            # empty prefix cache, which production never has: this puts the
            # shared prefixes back in it.
            start = max(start_rps, settings.lowest_rps)
            if settings.seconds_at(start) < allow_s:
                await step(start, Limits(math.inf, math.inf, max_error_rate=1.0))
            return await ramp(judged, start, settings, allow_s - (time.perf_counter() - began))

    def test(candidate: Candidate, start_rps: float, allow_s: float, stage: Stage) -> RampResult:
        return asyncio.run(run(start_rps, allow_s, full.for_stage(stage)))

    return test
