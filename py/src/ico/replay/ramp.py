"""Finds the highest arrival rate a config sustains within the latency limits.

The rate is raised step by step until a limit breaks, then the gap between
the last rate that held and the first that broke is halved a few times. What
comes out is one number per config, the rate it sustains, and from it the
cost of a million tokens on that config.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Awaitable, Callable

from ico.replay.load import StepResult


@dataclass(frozen=True)
class RampSettings:
    # Each step lasts about this long at the step's rate.
    step_seconds: float = 15.0
    # But never fewer requests than this: a p95 over a handful means nothing.
    min_requests: int = 40
    # How much the rate grows per step while everything still holds.
    growth: float = 2.0
    # How many times the gap between "held" and "broke" is halved. Four
    # halvings of a doubling pin the rate down to about 4%.
    refinements: int = 4
    # Steps in either direction before giving up on finding the edge.
    max_steps: int = 12
    # How long requests may keep running after the last one of a step was sent.
    grace_seconds: float = 20.0
    # Pause between steps, for the engine to finish dropping cut-off requests.
    settle_seconds: float = 2.0
    # No step may take longer than this many times step_seconds. It puts a
    # floor under the rates tried: below it a step would run too long to be
    # worth it, and a config that breaks there holds at no useful rate.
    longest_step: float = 8.0

    def requests_at(self, rate_rps: float) -> int:
        return max(self.min_requests, math.ceil(rate_rps * self.step_seconds))

    @property
    def lowest_rps(self) -> float:
        return self.min_requests / (self.longest_step * self.step_seconds)

    def seconds_at(self, rate_rps: float) -> float:
        """The longest a step at this rate can take, waiting and settling included."""
        return self.requests_at(rate_rps) / rate_rps + self.grace_seconds + self.settle_seconds

    def for_stage(self, stage: Stage) -> RampSettings:
        """These settings cut down to a stage of successive halving."""
        if stage.refinements is None:
            return self
        return replace(
            self,
            step_seconds=self.step_seconds * stage.step_scale,
            min_requests=max(1, round(self.min_requests * stage.requests_scale)),
            grace_seconds=self.grace_seconds * stage.step_scale,
            refinements=stage.refinements,
        )


@dataclass(frozen=True)
class Stage:
    """How thoroughly a config is tested.

    Successive halving gives every config a short run, the better half a
    longer one and the final two a full test. Shorter runs use shorter steps,
    fewer requests per step and fewer halvings of the gap at the edge: good
    enough to rank configs, not to quote a number.
    """

    number: int
    name: str
    step_scale: float
    requests_scale: float
    # None: as many as the settings say, which is a full test.
    refinements: int | None


SHORT = Stage(1, "short run", step_scale=1 / 3, requests_scale=0.5, refinements=1)
LONGER = Stage(2, "longer run", step_scale=1 / 2, requests_scale=0.75, refinements=2)
FULL = Stage(3, "full test", step_scale=1.0, requests_scale=1.0, refinements=None)
STAGES = {s.number: s for s in (SHORT, LONGER, FULL)}


@dataclass(frozen=True)
class RampResult:
    # The step at the highest rate that held, or None if no rate held.
    best: StepResult | None
    steps: tuple[StepResult, ...]
    seconds: float
    # The time allowance ended the search before it was done.
    out_of_time: bool = False


class _OutOfTime(Exception):
    pass


async def ramp(
    step: Callable[[float], Awaitable[StepResult]], start_rps: float, settings: RampSettings, allow_s: float = math.inf
) -> RampResult:
    """Run steps through ``step`` until the highest sustainable rate is pinned down.

    ``allow_s`` is how long the whole search may take. A step that could not
    finish within what is left is not started, and the search ends with
    whatever it has found.
    """
    began = time.perf_counter()
    steps: list[StepResult] = []

    async def at(rate: float) -> StepResult:
        if time.perf_counter() - began + settings.seconds_at(rate) > allow_s:
            raise _OutOfTime
        result = await step(rate)
        steps.append(result)
        return result

    held: StepResult | None = None
    broke: float | None = None
    out_of_time = False
    rate = max(start_rps, settings.lowest_rps)
    try:
        # Up, until something breaks.
        for _ in range(settings.max_steps):
            result = await at(rate)
            if not result.held:
                broke = rate
                break
            held = result
            rate *= settings.growth
        # The very first step broke: come down until something holds, or the
        # rate is so low that a step no longer fits in a reasonable time.
        while held is None and rate / settings.growth >= settings.lowest_rps:
            rate /= settings.growth
            result = await at(rate)
            if result.held:
                held = result
            else:
                broke = rate
        # Close in on the edge. The midpoint is geometric because the steps were.
        if held is not None and broke is not None:
            for _ in range(settings.refinements):
                result = await at(math.sqrt(held.rate_rps * broke))
                if result.held:
                    held = result
                else:
                    broke = result.rate_rps
    except _OutOfTime:
        out_of_time = True
    return RampResult(best=held, steps=tuple(steps), seconds=time.perf_counter() - began, out_of_time=out_of_time)
