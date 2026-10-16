"""One replay run, from draining the replica to putting it back.

The order of events, and the one rule that matters: whatever happens between
draining and the end, the engine goes back to its original flags and the
replica goes back into service. Every exit from the middle of a run passes
through the same restore, be it a config that crashes, a failed health check,
Ctrl-C, or a bug in this file.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from ico.replay import watchdog as watchdog_module
from ico.replay.gateway import GatewayAdmin
from ico.replay.launchers import Launcher, LauncherError
from ico.replay.load import StepResult
from ico.replay.ramp import FULL, LONGER, SHORT, RampResult, Stage
from ico.replay.sweep import Candidate, Flags, baseline

TESTED = "tested"
RULED_OUT = "ruled_out"  # by the roofline check, before any restart
FAILED_TO_START = "failed_to_start"
SKIPPED = "skipped"  # the time budget ran out first

FINISHED, ABORTED, FAILED, DECLINED = "finished", "aborted", "failed", "declined"


class ReplayError(RuntimeError):
    """The run cannot start, or cannot go on."""


class Reporter(Protocol):
    def report(self, state: str) -> None: ...

    def done(self) -> None: ...


@dataclass(frozen=True)
class ConfigResult:
    """What became of one config."""

    candidate: Candidate
    outcome: str
    note: str = ""
    # The step at the highest rate that held the limits; None if none did or none was run.
    best: StepResult | None = None
    tokens_s: float = 0.0
    cost_per_mtok: float | None = None
    is_baseline: bool = False
    seconds: float = 0.0
    # How thoroughly it was tested; see ramp.Stage. Only a full test can win.
    stage: int = FULL.number
    # How many judged steps the ramp took.
    steps: int = 0

    @property
    def can_win(self) -> bool:
        return self.best is not None and not self.candidate.quality_unverified and self.stage == FULL.number


# How much cheaper than the current config a config must measure to be
# recommended. One measurement of a config varies from run to run: the same
# config measured twice on an RTX 2060 held rates 8.5% apart. A config that
# measures less cheaper than this has not shown that it is.
MIN_SAVING = 0.10


@dataclass(frozen=True)
class RunResult:
    run_id: uuid.UUID
    status: str
    results: tuple[ConfigResult, ...]
    note: str = ""
    min_saving: float = MIN_SAVING

    @property
    def baseline(self) -> ConfigResult | None:
        return next((t for t in self.results if t.is_baseline), None)

    @property
    def final(self) -> tuple[ConfigResult, ...]:
        """One result per config: its furthest stage that was measured, else whatever stopped it."""
        latest: dict[str, ConfigResult] = {}
        for t in self.results:
            known = latest.get(t.candidate.id)
            if known is None or (t.outcome == TESTED, t.stage) >= (known.outcome == TESTED, known.stage):
                latest[t.candidate.id] = t
        return tuple(latest.values())

    @property
    def winner(self) -> ConfigResult | None:
        """The config to recommend: the cheapest that held the limits, if it beats the current one by min_saving.

        Otherwise the current config: nothing measured clearly cheaper. When
        the current config held the limits at no rate, any config that held
        them is better.
        """
        contenders = [t for t in self.results if t.can_win]
        if not contenders:
            return None
        cheapest = min(contenders, key=lambda t: (t.cost_per_mtok, not t.is_baseline))
        current = self.baseline
        if cheapest.is_baseline or current is None or not current.can_win:
            return cheapest
        if cheapest.cost_per_mtok <= current.cost_per_mtok * (1 - self.min_saving):
            return cheapest
        return current

    @property
    def close_calls(self) -> tuple[ConfigResult, ...]:
        """Configs that measured cheaper than the current one, but by less than min_saving."""
        current, winner = self.baseline, self.winner
        if current is None or not current.can_win or winner is not current:
            return ()
        return tuple(
            sorted(
                (t for t in self.results if t.can_win and not t.is_baseline and t.cost_per_mtok < current.cost_per_mtok),
                key=lambda t: t.cost_per_mtok,
            )
        )


@dataclass(frozen=True)
class Settings:
    replica: str
    budget_s: float
    # Dollars per hour for all of the replica's GPUs together.
    replica_price_per_hour: float
    # Prompt plus completion tokens of an average request in the sample.
    tokens_per_request: float
    # The arrival rate the first ramp starts from.
    start_rps: float
    # How far below the current config's rate a candidate's ramp starts.
    ramp_growth: float = 2.0
    single_replica: bool = False
    assume_yes: bool = False
    health_timeout_s: float = 900.0
    # How much cheaper a config must measure to be recommended; see MIN_SAVING.
    min_saving: float = MIN_SAVING
    idle_timeout_s: float = 300.0


@dataclass
class Replay:
    settings: Settings
    original: Flags
    admin: GatewayAdmin
    launcher: Launcher
    # Runs the ramp against whatever engine is up now, from a starting rate,
    # within a number of seconds and at a stage, and returns what it found.
    test: Callable[[Candidate, float, float, Stage], RampResult]
    healthy: Callable[[], bool]
    say: Callable[[str], None]
    confirm: Callable[[str], bool]
    watchdog: Reporter | None = None
    # How many halvings of the gap at the edge a full test makes; scales the time estimates.
    full_refinements: int = 4
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    run_id: uuid.UUID = field(default_factory=uuid.uuid4)

    def run(self, candidates: Sequence[Candidate], ruled_out: Sequence[ConfigResult] = ()) -> RunResult:
        s = self.settings
        deadline = self.clock() + s.budget_s
        try:
            self._check_replica()
            answer = self.admin.drain(s.replica)
            if answer.get("was_drained"):
                # Someone drained it between the check and now. It is theirs; leave it as found.
                raise ReplayError(f"replica {s.replica} was drained by someone else just now; another replay may be running")
        except BaseException:
            # Nothing of ours to put back, so the watchdog can stand down.
            if self.watchdog is not None:
                self.watchdog.done()
            raise
        self._report(watchdog_module.DRAINED)

        results: list[ConfigResult] = list(ruled_out)
        status, note = FAILED, ""
        # Whether the engine is, as far as we know, up on its original flags.
        self._on_original = True
        try:
            self.say(f"drained {s.replica}; waiting for its requests in flight to finish")
            self.admin.wait_idle(s.replica, s.idle_timeout_s, sleep=self.sleep)

            # Every config is measured the same way: on an engine freshly
            # started with it. That includes the current one. An engine that
            # has been up for days and one that just loaded are not the same
            # machine, and the comparison must not be between those two.
            status = FINISHED
            self._on_original = False
            current, load_s = self._load_current()
            self._on_original = True
            self._report(watchdog_module.RESTORED)
            self.say("measuring the current config")
            current = self._measure(current, s.start_rps, deadline - self.clock(), FULL, is_baseline=True)
            results.append(current)
            start_rps = current.best.rate_rps / s.ramp_growth if current.best else s.start_rps

            if candidates:
                plan = Plan.make(self, deadline, load_s, current, candidates)
                self.say(plan.describe())
                if not s.assume_yes and not self.confirm("Go ahead?"):
                    status = DECLINED
                else:
                    results += self._sweep(plan, deadline, load_s, start_rps)
        except KeyboardInterrupt:
            status, note = ABORTED, "interrupted"
            self.say("interrupted; putting the replica back")
        except Exception as e:  # noqa: BLE001 - anything at all must still lead to the restore below
            status, note = FAILED, f"{type(e).__name__}: {e}"
            self.say(f"the run failed ({note}); putting the replica back")
        finally:
            back = True if self._on_original else self._restore()
            if back:
                self._report(watchdog_module.RESTORED)
                self.admin.undrain(s.replica)
                self.say(f"{s.replica} is back in service on its original flags")
            else:
                note = (note + "; " if note else "") + "the original config could not be restored"
                status = FAILED
                self.say(
                    f"{s.replica} could NOT be restored and stays drained, so it gets no traffic. "
                    f"Start its engine with the original flags, then undrain it."
                )
            if self.watchdog is not None:
                self.watchdog.done()

        return RunResult(self.run_id, status, tuple(results), note, min_saving=self.settings.min_saving)

    def _check_replica(self) -> None:
        s = self.settings
        replicas = self.admin.replicas()
        target = next((r for r in replicas if r["name"] == s.replica), None)
        if target is None:
            raise ReplayError(f"the gateway has no replica named {s.replica}")
        if target["drained"]:
            raise ReplayError(
                f"replica {s.replica} is already drained. One replay runs per replica at a time; "
                "if no replay is running, undrain it first."
            )
        others = [r for r in replicas if r["name"] != s.replica and not r["drained"]]
        if not others and not s.single_replica:
            raise ReplayError(
                f"{s.replica} is the only replica taking traffic, so draining it means downtime for the whole run. "
                "Pass --single-replica to accept that."
            )

    def _load_current(self) -> tuple[Candidate, float]:
        """Restart the engine on its own flags. Returns the current config and how long a load takes."""
        current = baseline(self.original)
        self._report(watchdog_module.STOPPED)
        began = self.clock()
        try:
            self._stop()
            self.launcher.start(current.flags)
            self._wait_healthy(self.settings.health_timeout_s)
        except (LauncherError, ReplayError) as e:
            # Nothing to compare against, and no reason to believe a candidate
            # would start where the original does not.
            raise ReplayError(f"the engine did not come back up on its own flags ({e})") from None
        return current, self.clock() - began

    def _sweep(self, plan: Plan, deadline: float, load_s: float, start_rps: float) -> list[ConfigResult]:
        """Test the candidates as the plan says: each in full, or by successive halving."""
        if not plan.halving:
            return [self._try(c, FULL, start_rps, deadline, load_s, plan.estimates[FULL.number]) for c in plan.candidates]

        out: list[ConfigResult] = []
        contenders = list(plan.candidates)
        rates: dict[str, float] = {}
        for stage, keep in ((SHORT, None), (LONGER, plan.longer), (FULL, 2)):
            if keep is not None:
                if keep == 0:
                    continue
                # The best of the last stage go on; a config that changes
                # precision cannot win, so it does not take a place.
                measured = [t for t in out if t.stage == previous.number and t.best and not t.candidate.quality_unverified]
                ranked = sorted(measured, key=lambda t: t.cost_per_mtok)
                contenders = [t.candidate for t in ranked[:keep]]
                self.say(f"on to the {stage.name}: " + (", ".join(c.id for c in contenders) or "none"))
            for c in contenders:
                result = self._try(c, stage, rates.get(c.id, start_rps), deadline, load_s, plan.estimates[stage.number])
                out.append(result)
                if result.best:
                    rates[c.id] = result.best.rate_rps / self.settings.ramp_growth
            previous = stage
        return out

    def _try(self, candidate: Candidate, stage: Stage, start_rps: float, deadline: float, load_s: float, estimate_s: float) -> ConfigResult:
        """Load one config and test it at one stage, if the budget allows."""
        if not self._fits(deadline, load_s, estimate_s):
            return ConfigResult(candidate, SKIPPED, "the time budget ran out", stage=stage.number)
        self._report(watchdog_module.STOPPED)
        self._on_original = False
        try:
            self._stop()
            self.launcher.start(candidate.flags)
            # A config that takes three times as long as the original to load is not coming up.
            self._wait_healthy(min(self.settings.health_timeout_s, max(120.0, 3 * load_s)))
        except (LauncherError, ReplayError) as e:
            self.say(f"{candidate.id}: did not start ({e})")
            return ConfigResult(candidate, FAILED_TO_START, str(e), stage=stage.number)
        # What is left of the budget, less the load that puts the original back.
        return self._measure(candidate, start_rps, deadline - self.clock() - load_s, stage)

    def _measure(self, candidate: Candidate, start_rps: float, allow_s: float, stage: Stage, is_baseline: bool = False) -> ConfigResult:
        s = self.settings
        found = self.test(candidate, start_rps, allow_s, stage)
        common = dict(is_baseline=is_baseline, seconds=found.seconds, stage=stage.number, steps=len(found.steps))
        label = candidate.id if stage is FULL else f"{candidate.id} ({stage.name})"
        if found.best is None:
            note = "ran out of time before any rate held" if found.out_of_time else "held the limits at no rate tried"
            self.say(f"{label}: {note}")
            return ConfigResult(candidate, TESTED, note, **common)
        tokens_s = found.best.rate_rps * s.tokens_per_request * (1 - found.best.error_rate)
        cost = s.replica_price_per_hour / (tokens_s * 3600) * 1_000_000
        self.say(f"{label}: holds {found.best.rate_rps:.2f} requests/s, {tokens_s:,.0f} tokens/s, ${cost:.4f} per million tokens")
        return ConfigResult(candidate, TESTED, best=found.best, tokens_s=tokens_s, cost_per_mtok=cost, **common)

    def _stop(self) -> None:
        try:
            self.launcher.stop()
        except LauncherError:
            # A stop command may well fail when there is nothing left to stop,
            # say after a config crashed. That is only a problem if the engine
            # is still there.
            if self.healthy():
                raise

    def _wait_healthy(self, timeout_s: float) -> None:
        deadline = self.clock() + timeout_s
        while not self.healthy():
            if self.clock() > deadline:
                raise ReplayError(f"the engine did not answer its health check within {timeout_s:.0f}s")
            self.sleep(2.0)

    def _fits(self, deadline: float, load_s: float, test_s: float) -> bool:
        """Whether one more config can be loaded and tested with time left to load the original back."""
        return self.clock() + load_s + test_s + load_s <= deadline


    def _restore(self) -> bool:
        """Put the engine back on its original flags. False if it would not come back."""
        self.say("restoring the original config")
        try:
            self._stop()
            self.launcher.start(self.original)
            self._wait_healthy(self.settings.health_timeout_s)
            return True
        except (LauncherError, ReplayError) as e:
            self.say(f"restore failed: {e}")
            return False

    def _report(self, state: str) -> None:
        if self.watchdog is not None:
            self.watchdog.report(state)


@dataclass(frozen=True)
class Plan:
    """How the candidates will be tested, decided once the current config's full test shows what a test costs."""

    candidates: tuple[Candidate, ...]
    load_s: float
    # Expected seconds of one test, by stage number.
    estimates: dict[int, float]
    halving: bool
    # How many get a longer run under halving; 0 when the field is already down to the final two.
    longer: int
    # How many get a full test without halving.
    fitting: int
    full_minutes: float
    halving_minutes: float | None
    left_minutes: float

    @classmethod
    def make(cls, replay: Replay, deadline: float, load_s: float, current: ConfigResult, candidates: Sequence[Candidate]) -> Plan:
        estimates = {stage.number: _estimate(current, stage, replay) for stage in (SHORT, LONGER, FULL)}
        n = len(candidates)
        # Time for the candidates, with one more load at the end to put the original back.
        left = deadline - replay.clock() - load_s
        full = n * (load_s + estimates[FULL.number])
        longer = math.ceil(n / 2) if math.ceil(n / 2) > 2 else 0
        halving = None
        if n > 2:
            halving = (
                n * (load_s + estimates[SHORT.number])
                + longer * (load_s + estimates[LONGER.number])
                + 2 * (load_s + estimates[FULL.number])
            )
        use_halving = full > left and halving is not None and halving <= left
        fitting = n if use_halving else min(n, max(0, int(left // (load_s + estimates[FULL.number]))))
        return cls(
            tuple(candidates), load_s, estimates, use_halving, longer, fitting,
            full / 60, None if halving is None else halving / 60, left / 60,
        )  # fmt: skip

    def describe(self) -> str:
        n = len(self.candidates)
        head = f"Loading a config takes about {self.load_s:.0f}s and a full test about {self.estimates[FULL.number]:.0f}s."
        if self.halving:
            stages = f"a short run for each of the {n}"
            if self.longer:
                stages += f", a longer one for the best {self.longer}"
            return (
                f"{head} Full tests for all {n} configs would take {self.full_minutes:.0f} min, more than the "
                f"{self.left_minutes:.0f} left; successive halving takes about {self.halving_minutes:.0f}: "
                f"{stages}, and a full test for the final 2."
            )
        lines = [f"{head} Within the budget that leaves time for {self.fitting} of the {n} configs:"]
        lines += [f"  {'test' if i < self.fitting else 'skip'}  {c.id}" for i, c in enumerate(self.candidates)]
        return "\n".join(lines)


def _estimate(current: ConfigResult, stage: Stage, replay: Replay) -> float:
    """Seconds one test at this stage should take, scaled from the current config's full test.

    A stage changes how long each step is and how many halvings close in on
    the edge; the climb to the edge takes as many steps as before.
    """
    full_refinements = replay.full_refinements
    climb = max(1, current.steps - full_refinements)
    refinements = full_refinements if stage.refinements is None else stage.refinements
    return current.seconds * stage.step_scale * (climb + refinements) / (climb + full_refinements)
