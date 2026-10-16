"""A replay run from drain to restore, with the gateway, the launcher and the engine faked."""

import pytest

from ico.replay.launchers import Launcher, LauncherError
from ico.replay.load import StepResult
from ico.replay.ramp import FULL, LONGER, SHORT, RampResult
from ico.replay.run import ABORTED, DECLINED, FAILED, FAILED_TO_START, FINISHED, SKIPPED, TESTED, Replay, ReplayError, Settings
from ico.replay.sweep import Candidate

ORIGINAL = {"max-num-seqs": 64}


def candidate(seqs=None, **changes):
    changes = {"max-num-seqs": seqs, **changes} if seqs is not None else changes
    name = ",".join(f"{k}={v}" for k, v in sorted(changes.items()))
    return Candidate(id=name, flags={**ORIGINAL, **changes}, changes=changes)


class World:
    """The gateway, the launcher, the engine and the clock of a pretend install."""

    def __init__(self, *, replicas=("r1", "r2"), load_s=100.0, test_s=200.0) -> None:
        self.now = 0.0
        self.load_s, self.test_s = load_s, test_s
        self.drained = {name: False for name in replicas}
        self.engine_flags: dict | None = dict(ORIGINAL)  # None while the engine is down
        self.events: list[str] = []
        self.said: list[str] = []
        self.asked: list[str] = []
        self.answer = True
        self.broken_starts: set[int] = set()  # max-num-seqs values the engine cannot start with
        self.original_starts_once = False  # the original flags work for the first restart only
        self.stop_fails = False
        self.test_raises: dict[str, BaseException] = {}
        self.holds_nothing: set[str] = set()  # configs that break the limits at every rate
        self.start_rates: dict[str, float] = {}
        self.allowances: dict[str, float] = {}
        self.stages: list[tuple[str, int]] = []
        self.undrain_calls = 0

    # the gateway's admin API
    def replicas(self):
        return [{"name": n, "drained": d, "inflight": 0} for n, d in self.drained.items()]

    def drain(self, name):
        was = self.drained[name]
        self.drained[name] = True
        self.events.append("drain")
        return {"name": name, "drained": True, "was_drained": was}

    def undrain(self, name):
        self.drained[name] = False
        self.undrain_calls += 1
        self.events.append("undrain")
        return {"name": name, "drained": False}

    def wait_idle(self, name, timeout_s, *, sleep):
        self.events.append("idle")

    # the launcher
    def stop(self):
        self.events.append("stop")
        if self.stop_fails:
            raise LauncherError("nothing to stop")
        self.engine_flags = None

    def start(self, flags):
        self.events.append(f"start {flags['max-num-seqs']}")
        self.now += self.load_s
        broken = flags["max-num-seqs"] in self.broken_starts
        if self.original_starts_once and flags == ORIGINAL and self.events.count("start 64") > 1:
            broken = True
        self.engine_flags = None if broken else dict(flags)

    # the engine
    def healthy(self):
        return self.engine_flags is not None

    def test(self, c, start_rps, allow_s, stage):
        self.events.append(f"test {c.id}" if stage is FULL else f"{stage.name} {c.id}")
        self.stages.append((c.id, stage.number))
        self.start_rates[c.id] = start_rps
        self.allowances[c.id] = allow_s
        if c.id in self.test_raises:
            raise self.test_raises[c.id]
        assert self.engine_flags == c.flags, "the engine under test must be running the config being tested"
        # A full test climbs 4 steps and closes in with 4 more. A shorter
        # stage has shorter steps and fewer of the closing ones.
        refinements = 4 if stage.refinements is None else stage.refinements
        seconds = self.test_s * stage.step_scale * (4 + refinements) / 8
        self.now += seconds
        if c.id in self.holds_nothing:
            return RampResult(best=None, steps=(), seconds=seconds)
        # Pretend throughput grows with max-num-seqs up to 256, and FP8 doubles it.
        rate = min(self.engine_flags["max-num-seqs"], 256) / 16 * (2 if self.engine_flags.get("quantization") else 1)
        best = StepResult(rate_rps=rate, requests=40, errors=0, ttft_p95_ms=300.0, itl_p95_ms=20.0, held=True, seconds=10.0)
        return RampResult(best=best, steps=(best,) * (4 + refinements), seconds=seconds)

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def confirm(self, question):
        self.asked.append(question)
        return self.answer


class Reports:
    def __init__(self):
        self.states = []

    def report(self, state):
        self.states.append(state)

    def done(self):
        self.states.append("done")


def replay(world: World, **settings) -> Replay:
    launcher = type("L", (Launcher,), {"unattended": True, "stop": lambda self: world.stop(), "start": lambda self, f: world.start(f)})()
    defaults = dict(
        replica="r1", budget_s=3600.0, replica_price_per_hour=3.6, tokens_per_request=1000.0, start_rps=1.0, assume_yes=True, health_timeout_s=30.0
    )
    return Replay(
        settings=Settings(**{**defaults, **settings}),
        original=dict(ORIGINAL),
        admin=world,
        launcher=launcher,
        test=world.test,
        healthy=world.healthy,
        say=world.said.append,
        confirm=world.confirm,
        watchdog=Reports(),
        clock=world.clock,
        sleep=world.sleep,
    )


def test_a_run_measures_current_then_each_config_and_puts_the_original_back():
    world = World()
    r = replay(world)
    result = r.run([candidate(128), candidate(256)])

    assert world.events == [
        "drain", "idle",
        "stop", "start 64", "test current",   # the current config too is measured on a fresh start
        "stop", "start 128", "test max-num-seqs=128",
        "stop", "start 256", "test max-num-seqs=256",
        "stop", "start 64",                   # the original flags, whatever was tested last
        "undrain",
    ]  # fmt: skip
    assert result.status == FINISHED
    assert world.engine_flags == ORIGINAL and world.drained["r1"] is False
    # "restored" tells the watchdog the engine is on its original flags; "stopped" that it is not.
    assert r.watchdog.states == ["drained", "stopped", "restored", "stopped", "stopped", "restored", "done"]


def test_cost_per_million_tokens_comes_from_the_rate_that_held():
    world = World()
    result = replay(world).run([candidate(128)])
    current, faster = result.results
    # 64 seqs holds 4 requests/s of 1,000 tokens = 4,000 tokens/s. At $3.60 an
    # hour that is $0.001 a second, so a million tokens cost $0.25.
    assert (current.best.rate_rps, current.tokens_s, current.cost_per_mtok) == (4, 4000, pytest.approx(0.25))
    assert (faster.best.rate_rps, faster.cost_per_mtok) == (8, pytest.approx(0.125))
    assert result.baseline is current and result.winner is faster


def test_candidates_start_their_ramp_just_below_the_current_rate():
    world = World()
    replay(world, start_rps=0.5, ramp_growth=2.0).run([candidate(128)])
    assert world.start_rates == {"current": 0.5, "max-num-seqs=128": 2.0}  # half of the 4 the current config held


def test_the_current_config_wins_when_nothing_is_cheaper():
    world = World()
    result = replay(world).run([candidate(32), candidate(64, **{"enable-prefix-caching": False})])
    assert result.winner.is_baseline, "a tie goes to what already runs: no reason to change"


def test_a_config_only_a_little_cheaper_is_a_close_call_not_a_winner():
    from ico.replay.results import summary

    # 68 seqs hold 4.25 requests/s against 64's 4: 6% cheaper, less than one
    # measurement of a config varies by.
    result = replay(World()).run([candidate(68)])
    current, near = result.results
    assert result.winner is current and result.close_calls == (near,)
    text = summary(result)
    assert "No config was clearly cheaper than the current one" in text
    assert "max-num-seqs=68 measured 6% cheaper: less than the 10% one measurement can vary by" in text


def test_a_config_cheaper_by_the_margin_wins():
    # 72 seqs hold 4.5 requests/s: 11% cheaper than the current config.
    result = replay(World()).run([candidate(68), candidate(72)])
    assert result.winner.candidate.id == "max-num-seqs=72" and result.close_calls == ()


def test_the_margin_is_a_setting():
    result = replay(World(), min_saving=0.05).run([candidate(68)])
    assert result.winner.candidate.id == "max-num-seqs=68"


def test_a_config_that_changes_precision_is_measured_but_cannot_win():
    world = World()
    result = replay(world).run([candidate(128), candidate(128, quantization="fp8")])
    fp8 = result.results[-1]
    assert fp8.candidate.quality_unverified and fp8.cost_per_mtok < result.winner.cost_per_mtok
    assert result.winner.candidate.id == "max-num-seqs=128"


def test_a_config_that_does_not_start_is_noted_and_the_run_goes_on():
    world = World()
    world.broken_starts = {128}
    result = replay(world).run([candidate(128), candidate(256)])

    assert [(t.candidate.id, t.outcome) for t in result.results] == [
        ("current", TESTED),
        ("max-num-seqs=128", FAILED_TO_START),
        ("max-num-seqs=256", TESTED),
    ]
    assert "health check" in result.results[1].note
    assert result.status == FINISHED and world.engine_flags == ORIGINAL and not world.drained["r1"]


@pytest.mark.parametrize(
    ("error", "status"),
    [(RuntimeError("the load generator fell over"), FAILED), (KeyboardInterrupt(), ABORTED)],
)
def test_any_failure_mid_run_restores_the_original_and_undrains(error, status):
    world = World()
    world.test_raises = {"max-num-seqs=128": error}
    r = replay(world)
    result = r.run([candidate(128), candidate(256)])

    assert result.status == status
    assert world.events[-3:] == ["stop", "start 64", "undrain"]
    assert world.engine_flags == ORIGINAL and not world.drained["r1"]
    assert "test max-num-seqs=256" not in world.events, "nothing more is tested after a failure"
    assert r.watchdog.states[-2:] == ["restored", "done"]


def test_a_failure_while_the_engine_is_on_its_original_flags_only_undrains():
    world = World()
    world.test_raises = {"current": RuntimeError("no tokens came back")}
    result = replay(world).run([candidate(128)])
    assert result.status == FAILED
    assert world.events == ["drain", "idle", "stop", "start 64", "test current", "undrain"], "already on its original flags: no second restart"


def test_a_replica_that_cannot_be_restored_stays_drained():
    world = World()
    world.original_starts_once = True  # fine for the first restart, broken when it is time to restore
    r = replay(world)
    result = r.run([candidate(128)])

    assert result.status == FAILED and "could not be restored" in result.note
    assert world.drained["r1"] is True and world.undrain_calls == 0, "traffic must not be sent to an engine that is down"
    assert any("could NOT be restored and stays drained" in line for line in world.said)
    assert r.watchdog.states[-2:] == ["stopped", "done"], "the watchdog is never told the engine is back"
    assert [t.outcome for t in result.results] == [TESTED, TESTED], "what was measured before that is kept"


def test_an_engine_that_will_not_restart_on_its_own_flags_ends_the_run_at_once():
    world = World()
    world.broken_starts = {64}
    result = replay(world).run([candidate(128), candidate(256)])

    assert result.status == FAILED and "did not come back up on its own flags" in result.note
    assert "start 128" not in world.events, "no candidate is tried when the original itself does not start"
    assert world.drained["r1"] is True


def test_a_failing_stop_command_is_fine_once_the_engine_is_gone():
    world = World()
    world.broken_starts = {128}
    result = None

    original_stop = world.stop

    def stop():
        # After the crashed config there is nothing left for the stop command to stop.
        world.stop_fails = world.engine_flags is None
        original_stop()

    world.stop = stop
    result = replay(world).run([candidate(128), candidate(256)])
    assert [t.outcome for t in result.results] == [TESTED, FAILED_TO_START, TESTED]


def test_the_budget_decides_how_many_configs_are_tested():
    # Loading takes 100 s and testing 200 s, so every config, the current one
    # included, costs 300 s. Restoring the original: another 100 s at the end.
    world = World(load_s=100.0, test_s=200.0)
    result = replay(world, budget_s=1000.0).run([candidate(n) for n in (96, 128, 160, 192)])

    assert [t.outcome for t in result.results] == [TESTED, TESTED, TESTED, SKIPPED, SKIPPED]
    assert result.results[-1].note == "the time budget ran out"
    assert world.now <= 1000.0, "the whole run, restore included, fits the budget"
    plan = next(line for line in world.said if "Within the budget" in line)
    assert "time for 2 of the 4 configs" in plan
    assert plan.splitlines()[1:] == ["  test  max-num-seqs=96", "  test  max-num-seqs=128", "  skip  max-num-seqs=160", "  skip  max-num-seqs=192"]


def test_the_plan_waits_for_a_yes_and_a_no_puts_everything_back():
    world = World()
    world.answer = False
    result = replay(world, assume_yes=False).run([candidate(128), candidate(256)])

    assert world.asked == ["Go ahead?"]
    assert result.status == DECLINED
    # The plan comes after the current config was measured, so the engine is on its original flags.
    assert world.events == ["drain", "idle", "stop", "start 64", "test current", "undrain"]

    hands_off = World()
    replay(hands_off, assume_yes=True).run([candidate(128)])
    assert hands_off.asked == [], "--yes runs without anyone at the keyboard"


def test_it_refuses_to_start_on_an_unknown_or_already_drained_replica():
    world = World()
    with pytest.raises(ReplayError, match="no replica named r9"):
        replay(world, replica="r9").run([candidate(128)])

    world.drained["r1"] = True
    r = replay(world)
    with pytest.raises(ReplayError, match="already drained. One replay runs per replica at a time"):
        r.run([candidate(128)])
    assert world.events == [] and world.drained["r1"] is True, "a replica someone else drained is left exactly as found"
    assert r.watchdog.states == ["done"]


def test_draining_the_only_replica_needs_explicit_consent():
    alone = World(replicas=("r1",))
    with pytest.raises(ReplayError, match="only replica taking traffic.*--single-replica"):
        replay(alone).run([candidate(128)])
    assert alone.events == []

    # A second replica that is itself drained does not count as cover.
    covered_by_nothing = World()
    covered_by_nothing.drained["r2"] = True
    with pytest.raises(ReplayError, match="only replica taking traffic"):
        replay(covered_by_nothing).run([candidate(128)])

    assert replay(alone, single_replica=True).run([candidate(128)]).status == FINISHED


def test_losing_a_race_for_the_drain_leaves_the_replica_to_the_other_run():
    world = World()
    original_drain = world.drain

    def drain(name):
        world.drained[name] = True  # another run got there between our check and our drain
        return original_drain(name)

    world.drain = drain
    with pytest.raises(ReplayError, match="drained by someone else just now"):
        replay(world).run([candidate(128)])
    assert world.undrain_calls == 0 and world.drained["r1"] is True


def test_every_measurement_is_told_how_long_it_may_take():
    # A 1,000 s budget. Each measurement may use what is left after its own
    # load. A candidate leaves room for one more load, to put the original back.
    world = World(load_s=100.0, test_s=200.0)
    replay(world, budget_s=1000.0).run([candidate(128)])
    assert world.allowances == {"current": 1000.0 - 100.0, "max-num-seqs=128": 1000.0 - 300.0 - 100.0 - 100.0}


def test_a_current_config_that_holds_at_no_rate_does_not_end_the_search():
    from ico.replay import results

    world = World()
    world.holds_nothing = {"current"}
    result = replay(world, start_rps=3.0).run([candidate(128)])

    assert result.baseline.best is None and result.baseline.note == "held the limits at no rate tried"
    assert world.start_rates["max-num-seqs=128"] == 3.0, "with no current rate to go by, a candidate starts where the current config did"
    assert result.winner.candidate.id == "max-num-seqs=128"
    text = results.summary(result)
    assert "Cheapest config that holds the limits: max-num-seqs=128" in text and "below the current config" not in text
    assert "The current config held the limits at no rate tried, so there is no current cost to compare with." in text


def halving_world():
    # Loads are quick next to tests: 20 s against 300 s for a full test. Eight
    # candidates in full would need 8 x 320 s = 2,560 s; halving needs 1,830 s
    # (8 short runs of 62.5 s, 4 longer ones of 112.5 s, 2 full tests).
    world = World(load_s=20.0, test_s=300.0)
    budget = 320 + 1900 + 20  # the current config, the candidates, the restore
    return world, budget


def test_halving_is_chosen_when_full_tests_do_not_fit_but_halving_does():
    from ico.replay import results

    world, budget = halving_world()
    sizes = (32, 48, 64, 96, 128, 160, 192, 224)  # 2 to 14 requests/s
    r = replay(world, budget_s=budget)
    result = r.run([candidate(n) for n in sizes])

    plan = next(line for line in world.said if "successive halving" in line)
    assert "Full tests for all 8 configs would take 43 min, more than the 32 left; successive halving takes about 30" in plan
    assert "a short run for each of the 8, a longer one for the best 4, and a full test for the final 2" in plan

    by_stage = {}
    for config_id, stage in world.stages:
        by_stage.setdefault(stage, []).append(config_id)
    assert by_stage[SHORT.number] == [f"max-num-seqs={n}" for n in sizes]
    assert by_stage[LONGER.number] == ["max-num-seqs=224", "max-num-seqs=192", "max-num-seqs=160", "max-num-seqs=128"], "the best half, best first"
    assert by_stage[FULL.number] == ["current", "max-num-seqs=224", "max-num-seqs=192"], "the current config and the final two"
    assert world.now <= budget

    assert result.winner.candidate.id == "max-num-seqs=224" and result.winner.stage == FULL.number
    final = {t.candidate.id: t.stage for t in result.final}
    assert final["max-num-seqs=32"] == SHORT.number and final["max-num-seqs=160"] == LONGER.number and final["max-num-seqs=192"] == FULL.number
    text = results.summary(result)
    assert "out after a short run" in next(line for line in text.splitlines() if line.startswith("max-num-seqs=32 "))
    assert "out after a longer run" in next(line for line in text.splitlines() if line.startswith("max-num-seqs=160 "))
    assert len([line for line in text.splitlines() if line.startswith("max-num-seqs=224 ")]) == 1, "each config shows once"


def test_a_short_run_never_wins_however_good_it_looked():
    world, budget = halving_world()
    # fp8 doubles the rate, so it tops every stage it is in; it cannot win, so it does not take a place.
    configs = [candidate(n) for n in (32, 48, 64, 96, 128, 160, 192)] + [candidate(64, quantization="fp8")]
    result = replay(world, budget_s=budget).run(configs)
    assert ("max-num-seqs=64,quantization=fp8", LONGER.number) not in world.stages
    assert result.winner.candidate.id == "max-num-seqs=192"
    assert all(t.stage == FULL.number for t in result.results if t.can_win)


def test_full_tests_are_kept_when_they_fit():
    world = World(load_s=20.0, test_s=300.0)
    replay(world, budget_s=5000.0).run([candidate(n) for n in (32, 48, 64, 96)])
    assert {stage for _, stage in world.stages} == {FULL.number}
    assert not any("successive halving" in line for line in world.said)


def test_ruled_out_configs_are_carried_into_the_result_untested():
    from ico.replay.run import RULED_OUT, ConfigResult

    world = World()
    ruled_out = [ConfigResult(candidate(512), RULED_OUT, "the weights need 1.4 GiB and this memory fraction allows 1.2 GiB")]
    result = replay(world).run([candidate(128)], ruled_out)
    assert [t.outcome for t in result.results] == [RULED_OUT, TESTED, TESTED]
    assert "start 512" not in world.events
