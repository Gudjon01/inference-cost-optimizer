"""The pieces of the replay lab on their own: sweep files, launchers, sampling, prompts, load and ramp."""

import asyncio
import math
import sys

import pytest

from ico.config import LauncherConfig
from ico.replay import launchers, sweep, workload
from ico.replay.engine import Engine, Outcome
from ico.replay.load import Limits, StepResult, percentile95, run_step, schedule
from ico.replay.prompts import Prompt, Synthesizer
from ico.replay.ramp import RampSettings, ramp
from ico.replay.workload import Recorded
from tests.fake_engine import FakeEngine

ORIGINAL = {"max-num-seqs": 64, "gpu-memory-utilization": 0.6, "enable-prefix-caching": True}


# Sweep files ----------------------------------------------------------------


def write(tmp_path, text):
    path = tmp_path / "sweep.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_flags_render_as_a_command_line():
    flags = {"max-num-seqs": 128, "enable-prefix-caching": True, "enable-log-requests": False, "model": "org/name with space"}
    assert sweep.render(flags) == "--max-num-seqs 128 --enable-prefix-caching --no-enable-log-requests --model 'org/name with space'"
    assert sweep.render({}) == ""


def test_configs_are_the_original_with_some_flags_changed(tmp_path):
    path = write(tmp_path, "configs:\n  - max-num-seqs: 128\n  - gpu-memory-utilization: 0.7\n    enable-prefix-caching: null\n")
    first, second = sweep.load(path, ORIGINAL)
    assert first.id == "max-num-seqs=128"
    assert first.flags == {**ORIGINAL, "max-num-seqs": 128} and first.changes == {"max-num-seqs": 128}
    # null removes a flag the original has.
    assert second.id == "enable-prefix-caching=off,gpu-memory-utilization=0.7"
    assert second.flags == {"max-num-seqs": 64, "gpu-memory-utilization": 0.7}


def test_grid_expands_to_every_combination_and_drops_the_original(tmp_path):
    path = write(tmp_path, "grid:\n  max-num-seqs: [64, 128]\n  enable-prefix-caching: [true, false]\n")
    ids = [c.id for c in sweep.load(path, ORIGINAL)]
    # 2 x 2 combinations, minus the one that is what already runs.
    assert ids == ["enable-prefix-caching=False", "max-num-seqs=128", "enable-prefix-caching=False,max-num-seqs=128"]


def test_precision_changes_are_marked_quality_unverified(tmp_path):
    path = write(tmp_path, "configs:\n  - quantization: fp8\n  - max-num-seqs: 128\n  - model: org/model-awq\n")
    assert [c.quality_unverified for c in sweep.load(path, ORIGINAL)] == [True, False, True]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("configs:\n  - max-num-seqs: 64\n", "no config differs"),
        ("configs:\n  - just a string\n", r"configs\[0\]"),
        ("grid:\n  max-num-seqs: 128\n", "list of values"),
        ("candidates: []\n", "expected the keys"),
    ],
)
def test_bad_sweep_files_say_what_is_wrong(tmp_path, text, message):
    with pytest.raises(sweep.SweepError, match=message):
        sweep.load(write(tmp_path, text), ORIGINAL)


# Launchers ------------------------------------------------------------------

PY = f'"{sys.executable}"'


def test_command_launcher_fills_the_flags_slot_in_command_and_env(tmp_path):
    out = tmp_path / "started.txt"
    script = "import os, sys; open(sys.argv[1], 'w').write(' '.join(sys.argv[2:]) + '|' + os.environ['ENGINE_ARGS'])"
    cfg = LauncherConfig(
        kind="command",
        start=f'{PY} -c "{script}" "{out}" {{flags}}',
        stop=f"{PY} -c \"open(r'{tmp_path / 'stopped.txt'}', 'w')\"",
        env={"ENGINE_ARGS": "serve {flags}"},
    )
    launcher = launchers.build(cfg, "r1", ask=lambda _: None)
    assert launcher.unattended
    launcher.stop()
    launcher.start({"max-num-seqs": 128, "enable-prefix-caching": True})
    assert (tmp_path / "stopped.txt").exists()
    assert out.read_text() == "--max-num-seqs 128 --enable-prefix-caching|serve --max-num-seqs 128 --enable-prefix-caching"


def test_command_launcher_reports_a_failing_command_with_its_output():
    cfg = LauncherConfig(kind="command", start=f'{PY} -c "import sys; print(\'no such gpu\'); sys.exit(3)"', stop="exit 0")
    with pytest.raises(launchers.LauncherError, match="exited with 3:\nno such gpu"):
        launchers.build(cfg, "r1", ask=lambda _: None).start({})


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (LauncherConfig(kind="kubernetes"), "unknown kind 'kubernetes', expected one of manual, command, docker"),
        (LauncherConfig(kind="docker"), "needs the name of the container"),
        (LauncherConfig(kind="command", start="up"), "needs both a start and a stop command"),
    ],
)
def test_the_replay_command_checks_its_launcher(cfg, message):
    with pytest.raises(launchers.LauncherError, match=message):
        launchers.build(cfg, "r1", ask=lambda _: None)


def test_manual_launcher_asks_a_person_and_cannot_run_unattended():
    asked = []
    launcher = launchers.build(LauncherConfig(), "r2", ask=asked.append)
    assert not launcher.unattended
    launcher.stop()
    assert asked == [], "stopping is part of the one restart the person is asked for"
    launcher.start({"max-num-seqs": 128})
    assert "replica r2" in asked[0] and "--max-num-seqs 128" in asked[0]


# Sampling -------------------------------------------------------------------


def recorded(at, prompt=100, completion=20, hashes=(), chars=None):
    return Recorded(at, prompt, completion, chars if chars is not None else prompt * 4, tuple(hashes))


def test_shape_measures_lengths_sharing_and_burstiness():
    # Three prompts of 1,000 characters; the second and third repeat 512 of an earlier one's.
    requests = [
        recorded(0.0, 250, 10, hashes=(1, 2, 3)),
        recorded(1.0, 250, 30, hashes=(1, 2, 9)),
        recorded(2.0, 250, 20, hashes=(1, 2, 3)),
    ]
    s = workload.shape(requests, block_chars=256)
    assert s.prompt_tokens == (250, 250, 250) and s.completion_tokens == (10, 20, 30)
    # Request 2 shares 2 blocks, request 3 shares 3: 5 x 256 of 3,000 characters.
    assert s.shared_prefix == pytest.approx(5 * 256 / 3000)
    assert s.burstiness == 0, "evenly spaced arrivals are not bursty at all"
    assert s.rate_rps == 1.0

    bursty = workload.shape([recorded(t) for t in (0, 0.1, 0.2, 10, 10.1, 10.2)], 256)
    assert bursty.burstiness > 1.5


def test_sample_is_the_run_most_like_the_whole_window():
    # A window that is mostly short prompts, with one stretch of long ones.
    window = [recorded(i, prompt=100) for i in range(300)]
    window[100:140] = [recorded(100 + i, prompt=4000) for i in range(40)]
    chosen = workload.sample(window, size=60, block_chars=256)

    assert len(chosen.requests) == 60
    long_share = sum(r.prompt_tokens == 4000 for r in chosen.requests) / 60
    # The whole window is 13% long prompts. A run of all-short or mostly-long
    # prompts would misrepresent it.
    assert 0.05 <= long_share <= 0.25
    assert chosen.distance < 0.15
    # Consecutive, in order: gaps and shared prefixes only mean something between neighbours.
    times = [r.at for r in chosen.requests]
    assert times == sorted(times) and times[-1] - times[0] == 59


def test_a_small_window_is_used_whole():
    window = [recorded(i) for i in range(10)]
    chosen = workload.sample(window, size=60, block_chars=256)
    assert chosen.requests == tuple(window) and chosen.distance == 0


# Prompts --------------------------------------------------------------------

POOL = list(range(1000, 1400))


def test_prompts_have_exactly_the_recorded_token_counts():
    requests = [recorded(0, prompt=p, completion=c) for p, c in ((1, 1), (17, 5), (4096, 300))]
    synth = Synthesizer(POOL, requests, block_chars=256)
    for i, r in enumerate(requests):
        prompt = synth.prompt(i, use=0)
        assert len(prompt.tokens) == r.prompt_tokens and prompt.max_tokens == r.completion_tokens
        assert set(prompt.tokens) <= set(POOL), "only ordinary tokens of the model's tokenizer"


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def test_requests_with_matching_hashes_share_exactly_those_blocks():
    # 1,024 characters and 256 tokens each: a block of 256 characters is 64 tokens.
    a = recorded(0, prompt=256, hashes=(11, 22, 33), chars=1024)
    b = recorded(1, prompt=256, hashes=(11, 22, 99), chars=1024)
    c = recorded(2, prompt=256, hashes=(77,), chars=1024)
    synth = Synthesizer(POOL, [a, b, c], block_chars=256)
    pa, pb, pc = (synth.prompt(i, use=0).tokens for i in range(3))

    assert common_prefix(pa, pb) == 128, "two shared blocks of 64 tokens, and nothing more"
    assert common_prefix(pa, pc) < 4, "no shared hash, no shared prefix beyond chance"
    # a shares 0 blocks with what came before, b shares 2, c shares 0: 128 of 768 tokens.
    assert synth.shared_tokens() == pytest.approx(128 / 768)


def test_a_second_replay_keeps_the_shared_part_and_redraws_the_rest():
    r = recorded(0, prompt=256, hashes=(11, 22), chars=1024)
    synth = Synthesizer(POOL, [r], block_chars=256)
    first, again = synth.prompt(0, use=0).tokens, synth.prompt(0, use=1).tokens
    assert first[:128] == again[:128]
    assert first[128:] != again[128:], "otherwise the engine's prefix cache would serve the whole prompt"
    assert synth.prompt(0, use=0).tokens == first, "the same use is the same prompt, run after run"


def test_shared_blocks_never_push_a_prompt_past_its_length():
    # The hashes claim more shared text than this short prompt has tokens for.
    short = recorded(0, prompt=100, hashes=(11, 22, 33), chars=1024)
    long = recorded(1, prompt=300, hashes=(11, 22, 33), chars=1024)
    synth = Synthesizer(POOL, [long, short], block_chars=256)
    assert len(synth.prompt(1, use=0).tokens) == 100


# Load -----------------------------------------------------------------------


def test_percentile_counts_missing_requests_as_slow():
    assert percentile95([10.0] * 100, out_of=100) == 10
    assert percentile95(list(range(1, 101)), out_of=100) == 95
    assert percentile95([10.0] * 95, out_of=100) == 10, "5 of 100 may be missing"
    assert percentile95([10.0] * 94, out_of=100) == math.inf


def test_schedule_scales_recorded_gaps_to_the_asked_rate():
    # Recorded: a burst of three, a long pause, and so on. 4 requests per 4 seconds = 1 rps.
    gaps = [0.1, 0.1, 3.7, 0.1]
    times = schedule(gaps, recorded_rps=1.0, rate_rps=2.0, count=5)
    assert times == pytest.approx([0, 0.05, 0.10, 1.95, 2.0]), "twice the rate: every gap halved, the burst still a burst"
    assert schedule([], recorded_rps=0, rate_rps=4.0, count=3) == pytest.approx([0, 0.25, 0.5]), "no recorded gaps: evenly spaced"


LIMITS = Limits(ttft_p95_ms=100, itl_p95_ms=50)
ONE = Prompt((1, 2, 3), 4)


def step(send, count, **limits):
    times = [i * 0.001 for i in range(count)]
    return asyncio.run(run_step(send, lambda i: ONE, times, 10.0, Limits(**{**LIMITS.__dict__, **limits}), grace_s=0.5))


def test_step_holds_when_latencies_stay_under_the_limits():
    async def fast(_):
        return Outcome(ok=True, ttft_s=0.05, itl_s=0.02, prompt_tokens=3, completion_tokens=4)

    result = step(fast, 40)
    assert result.held and (result.ttft_p95_ms, result.itl_p95_ms, result.errors) == (50, 20, 0)


def test_step_breaks_when_more_than_five_percent_are_slow():
    def sender(slow_every):
        sent = 0

        async def send(_):
            nonlocal sent
            sent += 1
            return Outcome(ok=True, ttft_s=0.5 if sent % slow_every == 0 else 0.05, itl_s=0.02, prompt_tokens=3, completion_tokens=4)

        return send

    assert step(sender(slow_every=20), 40).held, "2 slow of 40 is exactly 5%: the p95 still holds"
    assert not step(sender(slow_every=10), 40).held, "4 slow of 40 is 10%"


def test_step_breaks_on_slow_tokens_and_on_errors():
    async def slow_tokens(_):
        return Outcome(ok=True, ttft_s=0.05, itl_s=0.2, prompt_tokens=3, completion_tokens=4)

    async def failing(_):
        return Outcome(ok=False)

    assert not step(slow_tokens, 40).held
    broken = step(failing, 40)
    assert not broken.held and broken.errors >= 1


def test_a_broken_step_says_what_broke_it():
    from ico.replay.tester import describe

    sent = 0

    async def bursts(_):
        nonlocal sent
        sent += 1
        return Outcome(ok=True, ttft_s=0.66 if sent % 5 == 0 else 0.07, itl_s=0.01, prompt_tokens=3, completion_tokens=4)

    result = step(bursts, 40)
    assert not result.held and result.over_ttft >= 3 and result.slowest_ttft_ms == 660
    text = describe(result)
    assert "broke" in text and f"{result.over_ttft} of 40 over the ttft limit, slowest 660 ms" in text
    assert "itl" not in text, "only the limit that broke is named"

    async def fast(_):
        return Outcome(ok=True, ttft_s=0.05, itl_s=0.02, prompt_tokens=3, completion_tokens=4)

    assert describe(step(fast, 40)).endswith("held   ttft p95 50 ms, itl p95 20 ms")


def test_requests_that_never_finish_break_the_step_and_are_cut_off():
    cut_off = 0

    async def hangs(_):
        nonlocal cut_off
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cut_off += 1
            raise

    result = step(hangs, 20)
    assert not result.held and result.ttft_p95_ms == math.inf
    assert cut_off == 20, "nothing may keep running against the engine after the step"
    assert result.seconds < 2, "the grace period, not the requests, decides how long the step waits"


# Ramp -----------------------------------------------------------------------


def capacity(limit_rps, log):
    """A step function for a config that holds up to ``limit_rps`` and breaks above it."""

    async def run(rate):
        log.append(rate)
        return StepResult(rate, 40, 0, 50.0, 20.0, held=rate <= limit_rps, seconds=0.0)

    return run


def test_ramp_doubles_until_a_break_then_closes_in():
    log = []
    result = asyncio.run(ramp(capacity(37.0, log), start_rps=4.0, settings=RampSettings(refinements=4)))
    assert log[:5] == [4, 8, 16, 32, 64], "up by doubling; 64 is the first to break"
    assert len(log) == 9, "then four halvings of the gap between 32 and 64"
    # Four halvings of a doubling leave a window of 2^(1/16), about 4%.
    assert 37.0 / 2 ** (1 / 16) <= result.best.rate_rps <= 37.0


def test_ramp_comes_down_when_the_first_step_already_breaks():
    log = []
    result = asyncio.run(ramp(capacity(3.0, log), start_rps=16.0, settings=RampSettings(refinements=2)))
    assert log[:4] == [16, 8, 4, 2], "down by halving until 2 holds"
    assert 2.0 <= result.best.rate_rps <= 3.0


def test_shorter_stages_cut_the_step_the_requests_and_the_closing_in():
    from ico.replay.ramp import FULL, LONGER, SHORT

    full = RampSettings(step_seconds=15, min_requests=40, refinements=4, grace_seconds=20)
    short, longer = full.for_stage(SHORT), full.for_stage(LONGER)
    assert (short.step_seconds, short.min_requests, short.refinements, short.grace_seconds) == (5, 20, 1, pytest.approx(20 / 3))
    assert (longer.step_seconds, longer.min_requests, longer.refinements) == (7.5, 30, 2)
    assert full.for_stage(FULL) is full


def test_ramp_stops_coming_down_where_a_step_would_take_too_long():
    # 40 requests per step and no step longer than 8 x 15 s: nothing below a third of a request per second.
    settings = RampSettings(step_seconds=15, min_requests=40, longest_step=8)
    assert settings.lowest_rps == pytest.approx(1 / 3)
    log = []
    result = asyncio.run(ramp(capacity(0.0, log), start_rps=8.0, settings=settings))
    assert result.best is None and not result.out_of_time
    assert log == [8, 4, 2, 1, 0.5], "0.25 would be below the floor, so the search ends: this config holds at no useful rate"


def test_ramp_never_starts_below_the_floor():
    log = []
    asyncio.run(ramp(capacity(100.0, log), start_rps=0.01, settings=RampSettings(max_steps=2)))
    assert log[0] == pytest.approx(1 / 3)


def test_ramp_keeps_to_its_time_allowance():
    # Every step at these settings may take up to 40/rate + 22 s. Pretend steps are instant:
    # what matters is that a step that could not finish in the time left is never started.
    settings = RampSettings()
    log = []
    plenty = asyncio.run(ramp(capacity(37.0, log), start_rps=4.0, settings=settings, allow_s=3600))
    assert not plenty.out_of_time and len(log) == 9

    log.clear()
    # At 4 requests/s a step can take 15 + 22 = 37 s. With 30 s nothing can be started at all.
    none = asyncio.run(ramp(capacity(37.0, log), start_rps=4.0, settings=settings, allow_s=30))
    assert none.out_of_time and none.best is None and log == []


def test_a_search_cut_short_by_time_keeps_what_it_found(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("ico.replay.ramp.time.perf_counter", lambda: clock[0])

    async def slow_step(rate):
        clock[0] += 40.0  # each step costs 40 s of the allowance
        return StepResult(rate, 40, 0, 50.0, 20.0, held=rate <= 37.0, seconds=40.0)

    result = asyncio.run(ramp(slow_step, start_rps=4.0, settings=RampSettings(), allow_s=130))
    # Steps at 4, 8 and 16 fit (120 s); the one at 32 could take until 157 s and is not started.
    assert result.out_of_time and [s.rate_rps for s in result.steps] == [4, 8, 16]
    assert result.best.rate_rps == 16, "a rate that held is still a result"


def measure_capacity(slots):
    """The sustained rate the whole stack finds for a fake engine: prompts, HTTP, streaming, limits, ramp."""
    fake = FakeEngine(slots=slots, ttft_s=0.03, token_s=0.02)
    settings = RampSettings(step_seconds=0.5, min_requests=12, refinements=3, grace_seconds=1.0, settle_seconds=0.0)
    limits = Limits(ttft_p95_ms=120, itl_p95_ms=200)

    async def go():
        async with Engine("http://engine", transport=fake.transport()) as engine:
            model = await engine.model()
            synth = Synthesizer(await engine.token_pool(model), [recorded(0, prompt=32, completion=4)], 256)

            async def one_step(rate):
                times = schedule([], 0, rate, settings.requests_at(rate))
                return await run_step(lambda p: engine.complete(model, p), lambda i: synth.prompt(i, 0), times, rate, limits, grace_s=1.0)

            return await ramp(one_step, start_rps=8.0, settings=settings)

    return asyncio.run(go()), fake


def test_the_ramp_finds_a_real_capacity_edge_end_to_end():
    two, fake = measure_capacity(slots=2)
    four, _ = measure_capacity(slots=4)
    # Each request holds a slot for about 0.1 s, so capacity is roughly ten
    # requests per second per slot. Timers are coarse, so the bounds are loose;
    # what must hold is that twice the slots sustain clearly more.
    assert two.best is not None and four.best is not None
    assert 6 <= two.best.rate_rps <= 30
    assert four.best.rate_rps >= 1.5 * two.best.rate_rps
    assert all(len(p) == 32 for p in fake.prompts), "every prompt reached the engine as 32 token ids"
